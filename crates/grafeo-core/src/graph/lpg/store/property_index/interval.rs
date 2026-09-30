//! Prepared interval stabbing with AVL height and subtree maximum-end pruning.

#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use super::Interval;
use grafeo_common::types::{EpochId, NodeId};
use std::cmp::Ordering;
type Key = (EpochId, NodeId);

use grafeo_common::memory::AllocError;
use grafeo_common::utils::error::Result;

/// A detached AVL node, allocated before entering final publication.
///
/// The private buffer always contains exactly one node. Keeping the Vec avoids
/// an infallible Box allocation or a possible allocation while shrinking a Vec
/// into a boxed slice. Tree links move this owner without moving its allocation.
#[cfg_attr(test, derive(Clone, Debug, PartialEq, Eq))]
pub(super) struct PreparedInterval {
    storage: Vec<Node>,
}

#[cfg_attr(test, derive(Clone, Debug, PartialEq, Eq))]
struct Node {
    key: Key,
    left: Option<PreparedInterval>,
    right: Option<PreparedInterval>,
    height: u16,
    end: Option<EpochId>,
    max_end: EpochId,
    open_count: usize,
}

impl PreparedInterval {
    /// Allocates one detached node. Allocation failure leaves no directory edit.
    pub(super) fn new(id: NodeId, interval: Interval) -> Result<Self> {
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(1)
            .map_err(|_| AllocError::OutOfMemory)?;
        storage.push(Node {
            key: (interval.from, id),
            left: None,
            right: None,
            height: 1,
            end: interval.to,
            max_end: interval.to.unwrap_or(EpochId::PENDING),
            open_count: usize::from(interval.to.is_none()),
        });
        Ok(Self { storage })
    }

    /// Ordinary infallible store mutation follows the allocator's standard
    /// failure behavior. Publication workspaces use the fallible constructor.
    pub(super) fn for_direct_mutation(id: NodeId, interval: Interval) -> Self {
        Self {
            storage: vec![Node {
                key: (interval.from, id),
                left: None,
                right: None,
                height: 1,
                end: interval.to,
                max_end: interval.to.unwrap_or(EpochId::PENDING),
                open_count: usize::from(interval.to.is_none()),
            }],
        }
    }

    /// Borrows the key without retiring its node allocation.
    pub(super) fn key(&self) -> &Key {
        &self.node().key
    }

    fn node(&self) -> &Node {
        // Construction fixes len=1; no operation resizes this private buffer.
        &self.storage[0]
    }

    fn node_mut(&mut self) -> &mut Node {
        &mut self.storage[0]
    }

    fn refresh_height(&mut self) {
        let node = self.node_mut();
        // An AVL's height is logarithmic in its node count, which is bounded by
        // addressable allocations. u16 cannot overflow for an allocated tree.
        node.height = 1 + height(&node.left).max(height(&node.right));
        node.open_count = usize::from(node.end.is_none())
            + node.left.as_ref().map_or(0, |root| root.node().open_count)
            + node.right.as_ref().map_or(0, |root| root.node().open_count);
        node.max_end = node
            .end
            .unwrap_or(EpochId::PENDING)
            .max(max_end(&node.left))
            .max(max_end(&node.right));
    }

    fn balance(&self) -> i32 {
        let node = self.node();
        i32::from(height(&node.left)) - i32::from(height(&node.right))
    }
}

/// Interval ownership and epoch selection with separately prepared storage.
/// Insert/remove/update/visit allocate no memory and never retire a node.
/// Duplicate and removed nodes return to their outer retirement owner.
/// Publication visitors must be allocation-free and non-panicking.
#[cfg_attr(test, derive(Clone, Debug, PartialEq, Eq))]
pub(super) struct IntervalDirectory {
    root: Option<PreparedInterval>,
    len: usize,
}

impl Default for IntervalDirectory {
    fn default() -> Self {
        Self::new()
    }
}

impl IntervalDirectory {
    pub(super) const fn new() -> Self {
        Self { root: None, len: 0 }
    }

    pub(super) fn open_count(&self) -> usize {
        self.root.as_ref().map_or(0, |root| root.node().open_count)
    }

    #[cfg(test)]
    pub(super) const fn len(&self) -> usize {
        self.len
    }
}

impl IntervalDirectory {
    /// Moves a detached node into the tree, or returns it unused on equality.
    pub(super) fn insert_prepared(&mut self, key: PreparedInterval) -> Option<PreparedInterval> {
        let (root, unused) = insert(self.root.take(), key);
        self.root = root;
        if unused.is_none() {
            self.len += 1;
        }
        unused
    }

    /// Detaches the matching node for reuse or retirement outside publication.
    pub(super) fn remove(&mut self, key: &Key) -> Option<PreparedInterval> {
        let (root, removed) = remove(self.root.take(), key);
        self.root = root;
        if removed.is_some() {
            self.len -= 1;
        }
        removed
    }

    pub(super) fn get(&self, key: &Key) -> Option<Interval> {
        let mut current = self.root.as_ref();
        while let Some(root) = current {
            let node = root.node();
            match key.cmp(&node.key) {
                Ordering::Less => current = node.left.as_ref(),
                Ordering::Greater => current = node.right.as_ref(),
                Ordering::Equal => {
                    return Some(Interval {
                        from: node.key.0,
                        to: node.end,
                    });
                }
            }
        }
        None
    }

    /// Changes one endpoint without moving or retiring its node allocation.
    pub(super) fn set_end(&mut self, key: &Key, end: Option<EpochId>) -> bool {
        update_end(self.root.as_mut(), key, end)
    }

    /// Visits only intervals containing epoch. Subtree maxima prune expired
    /// histories; the start-key order prunes future histories. With H retained
    /// intervals and K matches, conservative work is O((K+1) log(H+1)), using
    /// O(log(H+1)) call-stack space and no heap traversal stack. An expired
    /// bucket is rejected at its root. Every inspected node is counted.
    pub(super) fn visit_at(
        &self,
        epoch: EpochId,
        mut visitor: impl FnMut(NodeId),
    ) -> (usize, usize) {
        let mut matched = 0;
        let mut inspected = 0;
        visit_at(
            self.root.as_ref(),
            epoch,
            &mut visitor,
            &mut matched,
            &mut inspected,
        );
        (matched, inspected)
    }
}

fn height(root: &Option<PreparedInterval>) -> u16 {
    root.as_ref().map_or(0, |root| root.node().height)
}

fn rotate_left(mut root: PreparedInterval) -> PreparedInterval {
    let Some(mut pivot) = root.node_mut().right.take() else {
        return root;
    };
    root.node_mut().right = pivot.node_mut().left.take();
    root.refresh_height();
    pivot.node_mut().left = Some(root);
    pivot.refresh_height();
    pivot
}

fn rotate_right(mut root: PreparedInterval) -> PreparedInterval {
    let Some(mut pivot) = root.node_mut().left.take() else {
        return root;
    };
    root.node_mut().left = pivot.node_mut().right.take();
    root.refresh_height();
    pivot.node_mut().right = Some(root);
    pivot.refresh_height();
    pivot
}

fn rebalance(mut root: PreparedInterval) -> PreparedInterval {
    root.refresh_height();
    if root.balance() > 1 {
        if root
            .node()
            .left
            .as_ref()
            .is_some_and(|left| left.balance() < 0)
            && let Some(left) = root.node_mut().left.take()
        {
            root.node_mut().left = Some(rotate_left(left));
        }
        rotate_right(root)
    } else if root.balance() < -1 {
        if root
            .node()
            .right
            .as_ref()
            .is_some_and(|right| right.balance() > 0)
            && let Some(right) = root.node_mut().right.take()
        {
            root.node_mut().right = Some(rotate_right(right));
        }
        rotate_left(root)
    } else {
        root
    }
}

fn insert(
    root: Option<PreparedInterval>,
    key: PreparedInterval,
) -> (Option<PreparedInterval>, Option<PreparedInterval>) {
    let Some(mut root) = root else {
        return (Some(key), None);
    };
    let unused = match key.key().cmp(root.key()) {
        Ordering::Less => {
            let (left, unused) = insert(root.node_mut().left.take(), key);
            root.node_mut().left = left;
            unused
        }
        Ordering::Greater => {
            let (right, unused) = insert(root.node_mut().right.take(), key);
            root.node_mut().right = right;
            unused
        }
        Ordering::Equal => return (Some(root), Some(key)),
    };
    (Some(rebalance(root)), unused)
}

/// Returns the remaining subtree and its detached minimum allocation.
fn detach_min(mut root: PreparedInterval) -> (Option<PreparedInterval>, PreparedInterval) {
    let Some(left) = root.node_mut().left.take() else {
        let remainder = root.node_mut().right.take();
        root.refresh_height();
        return (remainder, root);
    };
    let (left, minimum) = detach_min(left);
    root.node_mut().left = left;
    (Some(rebalance(root)), minimum)
}

/// Replaces one removed AVL node; its child heights differ by at most one.
fn join(
    left: Option<PreparedInterval>,
    right: Option<PreparedInterval>,
) -> Option<PreparedInterval> {
    match (left, right) {
        (None, right) => right,
        (left, None) => left,
        (Some(left), Some(right)) => {
            let (right, mut successor) = detach_min(right);
            successor.node_mut().left = Some(left);
            successor.node_mut().right = right;
            Some(rebalance(successor))
        }
    }
}

fn remove(
    root: Option<PreparedInterval>,
    key: &Key,
) -> (Option<PreparedInterval>, Option<PreparedInterval>) {
    let Some(mut root) = root else {
        return (None, None);
    };
    let removed = match key.cmp(root.key()) {
        Ordering::Less => {
            let (left, removed) = remove(root.node_mut().left.take(), key);
            root.node_mut().left = left;
            removed
        }
        Ordering::Greater => {
            let (right, removed) = remove(root.node_mut().right.take(), key);
            root.node_mut().right = right;
            removed
        }
        Ordering::Equal => {
            let remainder = join(root.node_mut().left.take(), root.node_mut().right.take());
            root.refresh_height();
            return (remainder, Some(root));
        }
    };
    (Some(rebalance(root)), removed)
}

fn max_end(root: &Option<PreparedInterval>) -> EpochId {
    root.as_ref()
        .map_or(EpochId::INITIAL, |root| root.node().max_end)
}

fn update_end(root: Option<&mut PreparedInterval>, key: &Key, end: Option<EpochId>) -> bool {
    let Some(root) = root else {
        return false;
    };
    let changed = match key.cmp(root.key()) {
        Ordering::Less => update_end(root.node_mut().left.as_mut(), key, end),
        Ordering::Greater => update_end(root.node_mut().right.as_mut(), key, end),
        Ordering::Equal => {
            root.node_mut().end = end;
            true
        }
    };
    if changed {
        root.refresh_height();
    }
    changed
}

fn visit_at(
    root: Option<&PreparedInterval>,
    epoch: EpochId,
    visitor: &mut impl FnMut(NodeId),
    matched: &mut usize,
    inspected: &mut usize,
) {
    let Some(root) = root else {
        return;
    };
    *inspected += 1;
    let node = root.node();
    if node.max_end <= epoch {
        return;
    }
    visit_at(node.left.as_ref(), epoch, visitor, matched, inspected);
    if node.key.0 > epoch {
        return;
    }
    if node.end.is_none_or(|end| epoch < end) {
        *matched += 1;
        visitor(node.key.1);
    }
    visit_at(node.right.as_ref(), epoch, visitor, matched, inspected);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    fn interval(from: u64, to: Option<u64>) -> Interval {
        Interval {
            from: EpochId::new(from),
            to: to.map(EpochId::new),
        }
    }

    fn check(
        root: Option<&PreparedInterval>,
        low: Option<Key>,
        high: Option<Key>,
    ) -> (u16, EpochId, usize) {
        let Some(root) = root else {
            return (0, EpochId::INITIAL, 0);
        };
        let node = root.node();
        assert!(low.is_none_or(|key| key < node.key));
        assert!(high.is_none_or(|key| key > node.key));
        let (left, left_end, left_count) = check(node.left.as_ref(), low, Some(node.key));
        let (right, right_end, right_count) = check(node.right.as_ref(), Some(node.key), high);
        assert!(left.abs_diff(right) <= 1);
        assert_eq!(node.height, 1 + left.max(right));
        let maximum = node
            .end
            .unwrap_or(EpochId::PENDING)
            .max(left_end)
            .max(right_end);
        assert_eq!(node.max_end, maximum);
        assert_eq!(
            node.open_count,
            usize::from(node.end.is_none())
                + node.left.as_ref().map_or(0, |root| root.node().open_count)
                + node.right.as_ref().map_or(0, |root| root.node().open_count)
        );
        (node.height, maximum, left_count + right_count + 1)
    }

    #[test]
    fn interval_balancing_end_updates_and_removal_match_history_oracle() {
        for order in 0..3 {
            let mut directory = IntervalDirectory::new();
            let mut expected = BTreeMap::new();
            for step in 0..512 {
                let id = match order {
                    0 => step,
                    1 => 511 - step,
                    _ => (step * 137) % 512,
                };
                let value = interval(id % 64, (id % 7 != 0).then_some(id % 64 + id % 13));
                let id = NodeId::new(id);
                assert!(
                    directory
                        .insert_prepared(PreparedInterval::new(id, value).unwrap())
                        .is_none()
                );
                expected.insert((value.from, id), value);
                assert_eq!(
                    check(directory.root.as_ref(), None, None).2,
                    directory.len()
                );
            }
            let keys: Vec<_> = expected.keys().copied().collect();
            for (offset, key) in keys.iter().enumerate() {
                if offset % 3 == 0 {
                    let detached = directory.remove(key).unwrap();
                    assert_eq!(detached.key(), key);
                    expected.remove(key);
                } else if offset % 5 == 0 {
                    assert!(directory.set_end(key, None));
                    expected.get_mut(key).unwrap().to = None;
                }
                check(directory.root.as_ref(), None, None);
                assert_eq!(
                    directory.open_count(),
                    expected
                        .values()
                        .filter(|interval| interval.to.is_none())
                        .count()
                );
            }
            for epoch in 0..80 {
                let epoch = EpochId::new(epoch);
                let mut actual = BTreeSet::new();
                directory.visit_at(epoch, |id| {
                    actual.insert(id);
                });
                let oracle: BTreeSet<_> = expected
                    .iter()
                    .filter_map(|(key, value)| value.contains(epoch).then_some(key.1))
                    .collect();
                assert_eq!(actual, oracle);
            }
        }
    }

    #[test]
    fn interval_empty_and_selective_views_prune_historical_population() {
        let mut directory = IntervalDirectory::new();
        for id in 0..4096 {
            drop(directory.insert_prepared(
                PreparedInterval::new(NodeId::new(id), interval(1, Some(2))).unwrap(),
            ));
        }
        let mut found = Vec::new();
        assert_eq!(
            directory.visit_at(EpochId::new(3), |id| found.push(id)),
            (0, 1)
        );
        assert!(found.is_empty());
        let target = NodeId::new(2048);
        directory.set_end(&(EpochId::new(1), target), None);
        let (matches, work) = directory.visit_at(EpochId::new(3), |id| found.push(id));
        assert_eq!(matches, 1);
        assert_eq!(found, vec![target]);
        assert!(
            work <= 64,
            "selective interval visit inspected {work} nodes"
        );
        directory.set_end(&(EpochId::new(1), target), Some(EpochId::new(2)));
        for id in 4096..8192 {
            drop(directory.insert_prepared(
                PreparedInterval::new(NodeId::new(id), interval(100, Some(101))).unwrap(),
            ));
        }
        let (matches, work) = directory.visit_at(EpochId::new(50), |_| panic!("gap must be empty"));
        assert_eq!(matches, 0);
        assert!(
            work <= 64,
            "gap between old/future intervals inspected {work} nodes"
        );
        let (matches, _) = directory.visit_at(EpochId::new(1), |_| {});
        assert_eq!(matches, 4096);
    }

    #[test]
    fn interval_final_mutations_and_retirement_are_allocation_free() {
        let mut directory = IntervalDirectory::new();
        for id in 0..32 {
            drop(directory.insert_prepared(
                PreparedInterval::new(NodeId::new(id), interval(id, None)).unwrap(),
            ));
        }
        let prepared = PreparedInterval::new(NodeId::new(32), interval(32, None)).unwrap();
        let duplicate = PreparedInterval::new(NodeId::new(16), interval(16, None)).unwrap();
        crate::allocation_test::start();
        let unused = directory.insert_prepared(duplicate);
        let inserted = directory.insert_prepared(prepared);
        let changed = directory.set_end(&(EpochId::new(0), NodeId::new(0)), Some(EpochId::new(1)));
        let detached = directory.remove(&(EpochId::new(16), NodeId::new(16)));
        let work = directory.visit_at(EpochId::new(40), |_| {});
        let counts = crate::allocation_test::stop();
        assert_eq!(counts, crate::allocation_test::Counts::default());
        assert!(unused.is_some() && inserted.is_none() && changed && detached.is_some());
        assert_eq!(work.0, 31);
        let reused = directory.insert_prepared(detached.unwrap());
        assert!(reused.is_none());
        check(directory.root.as_ref(), None, None);
    }

    #[test]
    fn interval_node_preparation_reports_allocation_failure() {
        let (result, fired) = crate::allocation_test::with_failure(
            std::mem::size_of::<Node>(),
            std::mem::align_of::<Node>(),
            0,
            || PreparedInterval::new(NodeId::new(1), interval(1, None)),
        );
        assert!(fired);
        assert!(result.is_err());
    }
}
