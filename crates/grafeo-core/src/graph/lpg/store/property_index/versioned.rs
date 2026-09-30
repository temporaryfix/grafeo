//! Persistent AVL roots for the value keys live at each committed epoch.
//!
//! Nodes use indices into one payload-owned arena. Readers retain the history
//! lock; no per-node Arc allocation or external pointer lifetime is needed.
//! Preparation copies affected paths only. Explicit GC compacts the shared DAG.

#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use std::cmp::Ordering;
use std::ops::Bound;

use grafeo_common::memory::AllocError;
use grafeo_common::types::EpochId;
use grafeo_common::utils::error::{Error, Result, TransactionError};

type Link = Option<usize>;

#[derive(Clone)]
struct Node<K> {
    key: K,
    left: Link,
    right: Link,
    height: u16,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Root {
    epoch: EpochId,
    node: Link,
}

pub(super) struct VersionedDirectory<K> {
    nodes: Vec<Node<K>>,
    roots: Vec<Root>,
    revision: u64,
}

pub(super) struct PreparedVersion<K> {
    nodes: Vec<Node<K>>,
    root: Root,
    previous: Option<Root>,
    arena_len: usize,
    revision: u64,
    next_revision: u64,
    changed: bool,
    installed: bool,
}

impl<K> Default for VersionedDirectory<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K> VersionedDirectory<K> {
    pub(super) const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            roots: Vec::new(),
            revision: 0,
        }
    }

    /// Invalidates any workspace that captured a replaced arena, even when
    /// its rebuilt indices and node count happen to match the old representation.
    pub(super) fn qualify_replacement(&mut self, previous: &Self) -> Result<()> {
        self.revision = previous
            .revision
            .checked_add(1)
            .ok_or(AllocError::InsufficientSpace)?;
        Ok(())
    }

    pub(super) fn latest_epoch(&self) -> Option<EpochId> {
        self.roots.last().map(|root| root.epoch)
    }
}

fn invalid(message: &str) -> Error {
    Error::Transaction(TransactionError::WriteConflict(message.into()))
}

impl<K: Ord + Clone> VersionedDirectory<K> {
    /// Builds detached path copies and reserves all final append capacity.
    /// `true` inserts a live key; `false` removes one. Unchanged subtrees share
    /// arena indices. A later update in the same epoch replaces that epoch's
    /// root, without changing any earlier retained root.
    pub(super) fn prepare(
        &mut self,
        epoch: EpochId,
        updates: &[(K, bool)],
    ) -> Result<PreparedVersion<K>> {
        if epoch == EpochId::PENDING || self.latest_epoch().is_some_and(|latest| epoch < latest) {
            return Err(invalid(
                "value-key root publication epoch is regressive or pending",
            ));
        }
        let next_revision = self
            .revision
            .checked_add(1)
            .ok_or(AllocError::InsufficientSpace)?;
        let previous = self.roots.last().copied();
        let initial = previous.and_then(|root| root.node);
        let mut builder = Builder {
            base: &self.nodes,
            nodes: Vec::new(),
        };
        builder
            .nodes
            .try_reserve_exact(updates.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        let mut root = initial;
        for (key, insert) in updates {
            root = if *insert {
                builder.insert(root, key)?
            } else {
                builder.remove(root, key)?
            };
        }
        let nodes = builder.nodes;
        let changed = root != initial;
        if changed {
            self.nodes
                .try_reserve(nodes.len())
                .map_err(|_| AllocError::OutOfMemory)?;
            self.roots
                .try_reserve(1)
                .map_err(|_| AllocError::OutOfMemory)?;
        }
        Ok(PreparedVersion {
            nodes,
            root: Root { epoch, node: root },
            previous,
            arena_len: self.nodes.len(),
            revision: self.revision,
            next_revision,
            changed,
            installed: false,
        })
    }

    /// Epoch lookup costs O(log E), followed by a bounded O(log D + R) AVL
    /// traversal over keys actually live at that epoch. Counts both boundaries.
    pub(super) fn visit_at(
        &self,
        epoch: EpochId,
        lower: Bound<&K>,
        upper: Bound<&K>,
        mut visitor: impl FnMut(&K),
    ) -> (usize, usize) {
        let empty = match (lower, upper) {
            (Bound::Included(a), Bound::Included(b)) => a > b,
            (Bound::Included(a) | Bound::Excluded(a), Bound::Included(b) | Bound::Excluded(b)) => {
                a >= b
            }
            _ => false,
        };
        if empty {
            return (0, 0);
        }
        let mut inspected = 0;
        let end = self.roots.partition_point(|root| {
            inspected += 1;
            root.epoch <= epoch
        });
        let root = end.checked_sub(1).and_then(|index| self.roots[index].node);
        let mut matched = 0;
        visit(
            &self.nodes,
            root,
            lower,
            upper,
            &mut visitor,
            &mut matched,
            &mut inspected,
        );
        (matched, inspected)
    }

    /// Explicit maintenance only. Keep the predecessor root covering floor and
    /// all later roots, then copy only reachable arena owners into reserved
    /// storage. Every allocation succeeds before live ownership is changed.
    pub(super) fn gc(&mut self, floor: EpochId) -> Result<()> {
        if floor == EpochId::PENDING {
            return Ok(());
        }
        let next_revision = self
            .revision
            .checked_add(1)
            .ok_or(AllocError::InsufficientSpace)?;
        let end = self.roots.partition_point(|root| root.epoch <= floor);
        let keep = end.saturating_sub(1);
        let mut marked = Vec::new();
        marked
            .try_reserve_exact(self.nodes.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        marked.resize(self.nodes.len(), false);
        for root in &self.roots[keep..] {
            mark(&self.nodes, root.node, &mut marked);
        }
        let mut mapping = Vec::new();
        mapping
            .try_reserve_exact(self.nodes.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        let mut count = 0;
        for live in &marked {
            mapping.push(if *live {
                let index = count;
                count += 1;
                Some(index)
            } else {
                None
            });
        }
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(count)
            .map_err(|_| AllocError::OutOfMemory)?;
        for (index, mut node) in self.nodes.drain(..).enumerate() {
            if marked[index] {
                node.left = node.left.and_then(|index| mapping[index]);
                node.right = node.right.and_then(|index| mapping[index]);
                nodes.push(node);
            }
        }
        self.nodes = nodes;
        self.roots.drain(..keep);
        for root in &mut self.roots {
            root.node = root.node.and_then(|index| mapping[index]);
        }
        self.revision = next_revision;
        Ok(())
    }
}

impl<K> PreparedVersion<K> {
    pub(super) fn validate(&self, target: &VersionedDirectory<K>) -> bool {
        !self.installed
            && target.revision == self.revision
            && target.nodes.len() == self.arena_len
            && target.roots.last().copied() == self.previous
            && (!self.changed
                || (target.nodes.capacity().saturating_sub(target.nodes.len()) >= self.nodes.len()
                    && target.roots.capacity() > target.roots.len()))
    }

    /// No allocation, key clone or owner retirement. The private Vec's buffer
    /// and any unused copies remain owned by this workspace through guard drain.
    pub(super) fn install(&mut self, target: &mut VersionedDirectory<K>) {
        if self.installed {
            return;
        }
        if self.changed {
            target.nodes.append(&mut self.nodes);
            if let Some(last) = target.roots.last_mut()
                && last.epoch == self.root.epoch
            {
                *last = self.root;
            } else {
                target.roots.push(self.root);
            }
            target.revision = self.next_revision;
        }
        self.installed = true;
    }
}

struct Builder<'a, K> {
    base: &'a [Node<K>],
    nodes: Vec<Node<K>>,
}

impl<K: Ord + Clone> Builder<'_, K> {
    fn node(&self, index: usize) -> &Node<K> {
        if index < self.base.len() {
            &self.base[index]
        } else {
            &self.nodes[index - self.base.len()]
        }
    }
    fn height(&self, link: Link) -> u16 {
        link.map_or(0, |index| self.node(index).height)
    }
    fn make(&mut self, key: K, left: Link, right: Link) -> Result<Link> {
        let height = 1 + self.height(left).max(self.height(right));
        self.nodes
            .try_reserve(1)
            .map_err(|_| AllocError::OutOfMemory)?;
        let index = self
            .base
            .len()
            .checked_add(self.nodes.len())
            .ok_or(AllocError::InsufficientSpace)?;
        self.nodes.push(Node {
            key,
            left,
            right,
            height,
        });
        Ok(Some(index))
    }
    fn balance(&mut self, key: K, left: Link, right: Link) -> Result<Link> {
        if self.height(left) > self.height(right) + 1
            && let Some(index) = left
        {
            let pivot = self.node(index).clone();
            if self.height(pivot.left) >= self.height(pivot.right) {
                let right = self.make(key, pivot.right, right)?;
                return self.make(pivot.key, pivot.left, right);
            }
            if let Some(middle) = pivot.right {
                let middle = self.node(middle).clone();
                let left = self.make(pivot.key, pivot.left, middle.left)?;
                let right = self.make(key, middle.right, right)?;
                return self.make(middle.key, left, right);
            }
        }
        if self.height(right) > self.height(left) + 1
            && let Some(index) = right
        {
            let pivot = self.node(index).clone();
            if self.height(pivot.right) >= self.height(pivot.left) {
                let left = self.make(key, left, pivot.left)?;
                return self.make(pivot.key, left, pivot.right);
            }
            if let Some(middle) = pivot.left {
                let middle = self.node(middle).clone();
                let left = self.make(key, left, middle.left)?;
                let right = self.make(pivot.key, middle.right, pivot.right)?;
                return self.make(middle.key, left, right);
            }
        }
        self.make(key, left, right)
    }
    fn insert(&mut self, root: Link, key: &K) -> Result<Link> {
        let Some(index) = root else {
            return self.make(key.clone(), None, None);
        };
        let node = self.node(index).clone();
        match key.cmp(&node.key) {
            Ordering::Equal => Ok(root),
            Ordering::Less => {
                let left = self.insert(node.left, key)?;
                if left == node.left {
                    Ok(root)
                } else {
                    self.balance(node.key, left, node.right)
                }
            }
            Ordering::Greater => {
                let right = self.insert(node.right, key)?;
                if right == node.right {
                    Ok(root)
                } else {
                    self.balance(node.key, node.left, right)
                }
            }
        }
    }
    fn remove(&mut self, root: Link, key: &K) -> Result<Link> {
        let Some(index) = root else {
            return Ok(None);
        };
        let node = self.node(index).clone();
        match key.cmp(&node.key) {
            Ordering::Less => {
                let left = self.remove(node.left, key)?;
                if left == node.left {
                    Ok(root)
                } else {
                    self.balance(node.key, left, node.right)
                }
            }
            Ordering::Greater => {
                let right = self.remove(node.right, key)?;
                if right == node.right {
                    Ok(root)
                } else {
                    self.balance(node.key, node.left, right)
                }
            }
            Ordering::Equal => match (node.left, node.right) {
                (None, right) => Ok(right),
                (left, None) => Ok(left),
                (left, Some(right)) => {
                    let (key, right) = self.detach_min(right)?;
                    self.balance(key, left, right)
                }
            },
        }
    }
    fn detach_min(&mut self, root: usize) -> Result<(K, Link)> {
        let node = self.node(root).clone();
        if let Some(left) = node.left {
            let (key, left) = self.detach_min(left)?;
            let root = self.balance(node.key, left, node.right)?;
            Ok((key, root))
        } else {
            Ok((node.key, node.right))
        }
    }
}

fn visit<K: Ord>(
    nodes: &[Node<K>],
    root: Link,
    lower: Bound<&K>,
    upper: Bound<&K>,
    visitor: &mut impl FnMut(&K),
    matched: &mut usize,
    inspected: &mut usize,
) {
    let Some(index) = root else {
        return;
    };
    *inspected += 1;
    let node = &nodes[index];
    let below = match lower {
        Bound::Included(key) => node.key < *key,
        Bound::Excluded(key) => node.key <= *key,
        Bound::Unbounded => false,
    };
    if below {
        visit(nodes, node.right, lower, upper, visitor, matched, inspected);
        return;
    }
    let above = match upper {
        Bound::Included(key) => node.key > *key,
        Bound::Excluded(key) => node.key >= *key,
        Bound::Unbounded => false,
    };
    if above {
        visit(nodes, node.left, lower, upper, visitor, matched, inspected);
        return;
    }
    visit(nodes, node.left, lower, upper, visitor, matched, inspected);
    *matched += 1;
    visitor(&node.key);
    visit(nodes, node.right, lower, upper, visitor, matched, inspected);
}

fn mark<K>(nodes: &[Node<K>], root: Link, marked: &mut [bool]) {
    if let Some(index) = root
        && !marked[index]
    {
        marked[index] = true;
        mark(nodes, nodes[index].left, marked);
        mark(nodes, nodes[index].right, marked);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn apply(directory: &mut VersionedDirectory<i32>, epoch: u64, updates: &[(i32, bool)]) {
        let mut prepared = directory.prepare(EpochId::new(epoch), updates).unwrap();
        assert!(prepared.validate(directory));
        prepared.install(directory);
    }

    fn values(directory: &VersionedDirectory<i32>, epoch: u64) -> BTreeSet<i32> {
        let mut result = BTreeSet::new();
        directory.visit_at(
            EpochId::new(epoch),
            Bound::Unbounded,
            Bound::Unbounded,
            |key| {
                result.insert(*key);
            },
        );
        result
    }

    fn check(
        directory: &VersionedDirectory<i32>,
        root: Link,
        lower: Option<i32>,
        upper: Option<i32>,
    ) -> u16 {
        let Some(index) = root else {
            return 0;
        };
        let node = &directory.nodes[index];
        assert!(lower.is_none_or(|key| key < node.key));
        assert!(upper.is_none_or(|key| key > node.key));
        let left = check(directory, node.left, lower, Some(node.key));
        let right = check(directory, node.right, Some(node.key), upper);
        assert!(left.abs_diff(right) <= 1);
        assert_eq!(node.height, 1 + left.max(right));
        node.height
    }

    #[test]
    fn versioned_roots_match_ordered_sets_across_insert_delete_and_same_epoch_updates() {
        let mut directory = VersionedDirectory::new();
        let mut expected = BTreeSet::new();
        let mut snapshots = Vec::new();
        for step in 0..512 {
            let key = (step * 137) % 256;
            let insert = step < 256;
            if insert {
                expected.insert(key);
            } else {
                expected.remove(&key);
            }
            let epoch = u64::try_from(step / 4 + 1).unwrap();
            apply(&mut directory, epoch, &[(key, insert)]);
            if step % 4 == 3 {
                snapshots.push((epoch, expected.clone()));
            }
            for root in &directory.roots {
                check(&directory, root.node, None, None);
            }
        }
        for (epoch, expected) in snapshots {
            assert_eq!(values(&directory, epoch), expected);
        }
    }

    #[test]
    fn versioned_roots_seek_expired_distinct_ranges_and_exact_reentry_gaps() {
        let mut directory = VersionedDirectory::new();
        let initial: Vec<_> = (1..=4096).map(|key| (key, true)).collect();
        apply(&mut directory, 1, &initial);
        let moved: Vec<_> = (1..=4096)
            .flat_map(|key| [(key, false), (key + 10_000, true)])
            .collect();
        apply(&mut directory, 2, &moved);
        apply(&mut directory, 3, &[(67, true)]);
        apply(&mut directory, 4, &[(67, false)]);
        for (epoch, expected) in [(2, vec![]), (3, vec![67]), (4, vec![])] {
            let mut found = Vec::new();
            let (_, work) = directory.visit_at(
                EpochId::new(epoch),
                Bound::Included(&1),
                Bound::Included(&4096),
                |key| found.push(*key),
            );
            assert_eq!(found, expected);
            assert!(work <= 64, "epoch {epoch} inspected {work} keys");
        }
        let mut old = Vec::new();
        directory.visit_at(
            EpochId::new(1),
            Bound::Included(&67),
            Bound::Included(&67),
            |key| old.push(*key),
        );
        assert_eq!(old, vec![67]);
    }

    #[test]
    fn versioned_preparation_aborts_and_final_publication_has_no_allocator_traffic() {
        let mut directory = VersionedDirectory::new();
        apply(&mut directory, 1, &[(1, true), (2, true), (3, true)]);
        let before = values(&directory, 1);
        let abandoned = directory
            .prepare(EpochId::new(2), &[(2, false), (4, true)])
            .unwrap();
        drop(abandoned);
        assert_eq!(values(&directory, 2), before);
        let mut prepared = directory
            .prepare(EpochId::new(2), &[(2, false), (4, true)])
            .unwrap();
        crate::allocation_test::start();
        let valid = prepared.validate(&directory);
        prepared.install(&mut directory);
        let stale = prepared.validate(&directory);
        prepared.install(&mut directory);
        let traffic = crate::allocation_test::stop();
        assert!(valid && !stale);
        assert_eq!(traffic, crate::allocation_test::Counts::default());
        assert_eq!(values(&directory, 1), BTreeSet::from([1, 2, 3]));
        assert_eq!(values(&directory, 2), BTreeSet::from([1, 3, 4]));
    }

    #[test]
    fn versioned_shared_root_gc_preserves_floor_and_rejects_stale_arena_bindings() {
        let mut directory = VersionedDirectory::new();
        apply(&mut directory, 1, &[(1, true), (2, true), (3, true)]);
        apply(&mut directory, 2, &[(1, false), (4, true)]);
        apply(&mut directory, 3, &[(2, false), (5, true)]);
        let two = values(&directory, 2);
        let three = values(&directory, 3);
        let pending = directory.prepare(EpochId::new(4), &[(9, true)]).unwrap();
        let allocated = directory.nodes.len();
        directory.gc(EpochId::new(2)).unwrap();
        assert!(directory.nodes.len() < allocated);
        crate::allocation_test::start();
        let valid = pending.validate(&directory);
        let traffic = crate::allocation_test::stop();
        assert!(!valid);
        assert_eq!(traffic, crate::allocation_test::Counts::default());
        assert_eq!(values(&directory, 2), two);
        assert_eq!(values(&directory, 3), three);
        assert_eq!(directory.roots.first().unwrap().epoch, EpochId::new(2));
    }

    #[test]
    fn versioned_path_copy_allocation_failure_is_fallible_and_logically_inert() {
        let mut directory = VersionedDirectory::<i32>::new();
        let (result, fired) = crate::allocation_test::with_failure(
            std::mem::size_of::<Node<i32>>(),
            std::mem::align_of::<Node<i32>>(),
            0,
            || directory.prepare(EpochId::new(1), &[(1, true)]),
        );
        assert!(fired && result.is_err());
        assert!(directory.nodes.is_empty() && directory.roots.is_empty());
    }

    fn history_reserve_failure_preserves_payload(fail_roots: bool) {
        use super::super::{
            PropertyIndexImage, PropertyIndexRows, key::PropertyOrderKey,
            maintenance::PreparedHistory,
        };
        use crate::graph::PropertyIndexPredicate;
        use grafeo_common::types::{HashableValue, NodeId, Value};

        fn fixture() -> PropertyIndexRows {
            // An odd root count separates the final root-buffer allocation
            // from the small power-of-two scratch buffers used by preparation.
            let mut rows = PropertyIndexRows::from_image(PropertyIndexImage {
                current: (1..=67)
                    .map(|id| (NodeId::new(id), Value::Int64(i64::try_from(id).unwrap())))
                    .collect(),
                history: (1..=67)
                    .map(|id| {
                        (
                            NodeId::new(id),
                            vec![(EpochId::new(id), Value::Int64(i64::try_from(id).unwrap()))],
                        )
                    })
                    .collect(),
                floor: EpochId::INITIAL,
            })
            .unwrap();
            let versions = &mut rows.history.get_mut().versions;
            versions.nodes.shrink_to_fit();
            versions.roots.shrink_to_fit();
            assert_eq!(versions.nodes.capacity(), versions.nodes.len());
            assert_eq!(versions.roots.capacity(), versions.roots.len());
            rows
        }

        fn snapshots(rows: &PropertyIndexRows) -> Vec<Vec<NodeId>> {
            [1, 32, 67, 68]
                .into_iter()
                .flat_map(|epoch| {
                    [(1, 1), (1000, 1000), (1, 1000)].map(|(min, max)| {
                        let min = Value::Int64(min);
                        let max = Value::Int64(max);
                        let (mut ids, _) = rows
                            .history
                            .read()
                            .candidates(
                                PropertyIndexPredicate::Range {
                                    min: Some(&min),
                                    max: Some(&max),
                                    min_inclusive: true,
                                    max_inclusive: true,
                                },
                                EpochId::new(epoch),
                            )
                            .unwrap();
                        ids.sort_unstable();
                        ids
                    })
                })
                .collect()
        }

        let changes = [(
            NodeId::new(1),
            Some(Value::Int64(1)),
            Some(Value::Int64(1000)),
        )];
        // Observe actual Vec growth on an identical detached fixture instead
        // of assuming a particular allocator's growth/layout policy.
        let mut probe = fixture();
        let probe_history = probe.history.get_mut();
        let prepared = PreparedHistory::prepare(probe_history, &changes, EpochId::new(68)).unwrap();
        let arena_capacity = probe_history.versions.nodes.capacity();
        let roots_capacity = probe_history.versions.roots.capacity();
        drop(prepared);
        let (size, align) = if fail_roots {
            (
                roots_capacity * std::mem::size_of::<Root>(),
                std::mem::align_of::<Root>(),
            )
        } else {
            (
                arena_capacity * std::mem::size_of::<Node<PropertyOrderKey>>(),
                std::mem::align_of::<Node<PropertyOrderKey>>(),
            )
        };

        let mut rows = fixture();
        let before_views = snapshots(&rows);
        let history = rows.history.get_mut();
        let before_members = history.rows.clone();
        let before_ordered = history.ordered.len();
        let before_nodes = history.versions.nodes.len();
        let before_roots = history.versions.roots.clone();
        let before_revision = history.versions.revision;
        let before_arena_capacity = history.versions.nodes.capacity();
        let before_roots_capacity = history.versions.roots.capacity();
        assert!(arena_capacity > before_arena_capacity);
        assert!(roots_capacity > before_roots_capacity);
        let (result, fired) = crate::allocation_test::with_failure(size, align, 0, || {
            PreparedHistory::prepare(history, &changes, EpochId::new(68))
        });
        assert!(fired && result.is_err());
        drop(result);
        // The successful earlier arena reserve proves root injection reached
        // the final reserve, after all detached historical work was prepared.
        assert_eq!(
            history.versions.nodes.capacity(),
            if fail_roots {
                arena_capacity
            } else {
                before_arena_capacity
            }
        );
        assert_eq!(history.versions.roots.capacity(), before_roots_capacity);
        assert_eq!(history.rows, before_members);
        assert_eq!(history.ordered.len(), before_ordered);
        assert_eq!(history.versions.nodes.len(), before_nodes);
        assert!(history.versions.roots == before_roots);
        assert_eq!(history.versions.revision, before_revision);
        assert_eq!(snapshots(&rows), before_views);
        assert_eq!(rows.len(), 67);
        for id in 1..=67 {
            let members = rows
                .get(&HashableValue::new(Value::Int64(
                    i64::try_from(id).unwrap(),
                )))
                .unwrap();
            assert_eq!(members.len(), 1);
            assert!(members.contains(&NodeId::new(id)));
        }
        assert!(!rows.contains_key(&HashableValue::new(Value::Int64(1000))));

        let history = rows.history.get_mut();
        let mut retry = PreparedHistory::prepare(history, &changes, EpochId::new(68)).unwrap();
        crate::allocation_test::start();
        let validation = retry.validate(history);
        retry.install(history);
        let traffic = crate::allocation_test::stop();
        assert!(validation.is_ok());
        assert_eq!(traffic, crate::allocation_test::Counts::default());
        let after_views = snapshots(&rows);
        assert_eq!(&after_views[..9], &before_views[..9]);
        assert!(after_views[9].is_empty());
        assert_eq!(after_views[10], vec![NodeId::new(1)]);
        assert_eq!(after_views[11], before_views[11]);
    }

    #[test]
    fn history_live_arena_reserve_failure_preserves_current_and_retained_views() {
        history_reserve_failure_preserves_payload(false);
    }

    #[test]
    fn history_epoch_roots_reserve_failure_preserves_current_and_retained_views() {
        history_reserve_failure_preserves_payload(true);
    }
}
