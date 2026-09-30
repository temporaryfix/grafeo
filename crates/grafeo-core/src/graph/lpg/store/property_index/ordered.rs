//! Prepared, allocation-free ordered routing for property index keys.

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
#[cfg(test)]
use std::ops::Bound;

use grafeo_common::memory::AllocError;
use grafeo_common::utils::error::Result;

/// A detached AVL node, allocated before entering final publication.
///
/// The private buffer always contains exactly one node. Keeping the Vec avoids
/// an infallible Box allocation or a possible allocation while shrinking a Vec
/// into a boxed slice. Tree links move this owner without moving its allocation.
pub(super) struct PreparedKey<K> {
    storage: Vec<Node<K>>,
}

struct Node<K> {
    key: K,
    left: Option<PreparedKey<K>>,
    right: Option<PreparedKey<K>>,
    height: u16,
}

impl<K> PreparedKey<K> {
    /// Allocates one detached node. Allocation failure leaves no directory edit.
    pub(super) fn new(key: K) -> Result<Self> {
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(1)
            .map_err(|_| AllocError::OutOfMemory)?;
        storage.push(Node {
            key,
            left: None,
            right: None,
            height: 1,
        });
        Ok(Self { storage })
    }

    /// Ordinary infallible store mutation follows the allocator's standard
    /// failure behavior. Publication workspaces use the fallible constructor.
    pub(super) fn for_direct_mutation(key: K) -> Self {
        Self {
            storage: vec![Node {
                key,
                left: None,
                right: None,
                height: 1,
            }],
        }
    }

    /// Borrows the key without retiring its node allocation.
    pub(super) fn key(&self) -> &K {
        &self.node().key
    }

    fn node(&self) -> &Node<K> {
        // Construction fixes len=1; no operation resizes this private buffer.
        &self.storage[0]
    }

    fn node_mut(&mut self) -> &mut Node<K> {
        &mut self.storage[0]
    }

    fn refresh_height(&mut self) {
        let node = self.node_mut();
        // An AVL's height is logarithmic in its node count, which is bounded by
        // addressable allocations. u16 cannot overflow for an allocated tree.
        node.height = 1 + height(&node.left).max(height(&node.right));
    }

    fn balance(&self) -> i32 {
        let node = self.node();
        i32::from(height(&node.left)) - i32::from(height(&node.right))
    }
}

/// Ordered scalar-key routing with separately prepared storage.
///
/// Insert/remove/visit allocate no memory and never retire a key or node.
/// Duplicate and removed nodes are returned to their outer retirement owner.
/// Comparators and visitors are supplied by the caller; publication callers
/// must use allocation-free, non-panicking callbacks and a lawful total order.
pub(super) struct OrderedDirectory<K> {
    root: Option<PreparedKey<K>>,
    len: usize,
}

impl<K> Default for OrderedDirectory<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K> OrderedDirectory<K> {
    pub(super) const fn new() -> Self {
        Self { root: None, len: 0 }
    }

    #[cfg(test)]
    pub(super) const fn len(&self) -> usize {
        self.len
    }
}

impl<K: Ord> OrderedDirectory<K> {
    /// Moves a detached node into the tree, or returns it unused on equality.
    pub(super) fn insert_prepared(&mut self, key: PreparedKey<K>) -> Option<PreparedKey<K>> {
        let (root, unused) = insert(self.root.take(), key);
        self.root = root;
        if unused.is_none() {
            self.len += 1;
        }
        unused
    }

    /// Detaches the matching node for reuse or retirement outside publication.
    pub(super) fn remove(&mut self, key: &K) -> Option<PreparedKey<K>> {
        let (root, removed) = remove(self.root.take(), key);
        self.root = root;
        if removed.is_some() {
            self.len -= 1;
        }
        removed
    }

    pub(super) fn contains(&self, key: &K) -> bool {
        let mut current = self.root.as_ref();
        while let Some(root) = current {
            let node = root.node();
            match key.cmp(&node.key) {
                Ordering::Less => current = node.left.as_ref(),
                Ordering::Greater => current = node.right.as_ref(),
                Ordering::Equal => return true,
            }
        }
        false
    }

    /// Visits the bounded keys in order, stopping when the visitor returns false.
    ///
    /// Returns the number of keys delivered, including the stopping key. Empty
    /// or reversed bounds visit nothing. Traversal seeks in O(log len), then
    /// visits matching keys without a heap-allocated traversal stack.
    #[cfg(test)]
    pub(super) fn visit(
        &self,
        lower: Bound<&K>,
        upper: Bound<&K>,
        visitor: impl FnMut(&K) -> bool,
    ) -> usize {
        self.visit_with_work(lower, upper, visitor).0
    }

    #[cfg(test)]
    pub(super) fn visit_with_work(
        &self,
        lower: Bound<&K>,
        upper: Bound<&K>,
        mut visitor: impl FnMut(&K) -> bool,
    ) -> (usize, usize) {
        let empty = match (lower, upper) {
            (Bound::Included(low), Bound::Included(high)) => low > high,
            (
                Bound::Included(low) | Bound::Excluded(low),
                Bound::Included(high) | Bound::Excluded(high),
            ) => low >= high,
            _ => false,
        };
        if empty {
            return (0, 0);
        }
        let mut visited = 0;
        let mut inspected = 0;
        visit(
            self.root.as_ref(),
            lower,
            upper,
            &mut visitor,
            &mut visited,
            &mut inspected,
        );
        (visited, inspected)
    }
}

fn height<K>(root: &Option<PreparedKey<K>>) -> u16 {
    root.as_ref().map_or(0, |root| root.node().height)
}

fn rotate_left<K>(mut root: PreparedKey<K>) -> PreparedKey<K> {
    let Some(mut pivot) = root.node_mut().right.take() else {
        return root;
    };
    root.node_mut().right = pivot.node_mut().left.take();
    root.refresh_height();
    pivot.node_mut().left = Some(root);
    pivot.refresh_height();
    pivot
}

fn rotate_right<K>(mut root: PreparedKey<K>) -> PreparedKey<K> {
    let Some(mut pivot) = root.node_mut().left.take() else {
        return root;
    };
    root.node_mut().left = pivot.node_mut().right.take();
    root.refresh_height();
    pivot.node_mut().right = Some(root);
    pivot.refresh_height();
    pivot
}

fn rebalance<K>(mut root: PreparedKey<K>) -> PreparedKey<K> {
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

fn insert<K: Ord>(
    root: Option<PreparedKey<K>>,
    key: PreparedKey<K>,
) -> (Option<PreparedKey<K>>, Option<PreparedKey<K>>) {
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
fn detach_min<K>(mut root: PreparedKey<K>) -> (Option<PreparedKey<K>>, PreparedKey<K>) {
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
fn join<K>(left: Option<PreparedKey<K>>, right: Option<PreparedKey<K>>) -> Option<PreparedKey<K>> {
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

fn remove<K: Ord>(
    root: Option<PreparedKey<K>>,
    key: &K,
) -> (Option<PreparedKey<K>>, Option<PreparedKey<K>>) {
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

#[cfg(test)]
fn visit<K: Ord>(
    root: Option<&PreparedKey<K>>,
    lower: Bound<&K>,
    upper: Bound<&K>,
    visitor: &mut impl FnMut(&K) -> bool,
    visited: &mut usize,
    inspected: &mut usize,
) -> bool {
    let Some(root) = root else {
        return true;
    };
    *inspected += 1;
    let node = root.node();
    let below = match lower {
        Bound::Included(key) => node.key < *key,
        Bound::Excluded(key) => node.key <= *key,
        Bound::Unbounded => false,
    };
    if below {
        return visit(
            node.right.as_ref(),
            lower,
            upper,
            visitor,
            visited,
            inspected,
        );
    }
    let above = match upper {
        Bound::Included(key) => node.key > *key,
        Bound::Excluded(key) => node.key >= *key,
        Bound::Unbounded => false,
    };
    if above {
        return visit(
            node.left.as_ref(),
            lower,
            upper,
            visitor,
            visited,
            inspected,
        );
    }
    if !visit(
        node.left.as_ref(),
        lower,
        upper,
        visitor,
        visited,
        inspected,
    ) {
        return false;
    }
    *visited += 1;
    visitor(&node.key)
        && visit(
            node.right.as_ref(),
            lower,
            upper,
            visitor,
            visited,
            inspected,
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    fn prepared<K: Ord>(key: K) -> PreparedKey<K> {
        PreparedKey::new(key).expect("prepare key")
    }

    fn keys(directory: &OrderedDirectory<i32>) -> Vec<i32> {
        let mut keys = Vec::new();
        let visited = directory.visit(Bound::Unbounded, Bound::Unbounded, |key| {
            keys.push(*key);
            true
        });
        assert_eq!(visited, keys.len());
        keys
    }

    fn check_avl<K: Ord>(
        root: &Option<PreparedKey<K>>,
        min: Option<&K>,
        max: Option<&K>,
    ) -> (usize, u16) {
        let Some(root) = root else { return (0, 0) };
        let node = root.node();
        assert!(min.is_none_or(|min| min < &node.key));
        assert!(max.is_none_or(|max| &node.key < max));
        let (left_count, left_height) = check_avl(&node.left, min, Some(&node.key));
        let (right_count, right_height) = check_avl(&node.right, Some(&node.key), max);
        assert!(left_height.abs_diff(right_height) <= 1);
        assert_eq!(node.height, 1 + left_height.max(right_height));
        (1 + left_count + right_count, node.height)
    }

    #[test]
    fn ordered_bounds_and_early_stop_are_exact() {
        let mut directory = OrderedDirectory::new();
        for key in [5, 1, 7, 3, 9] {
            assert!(directory.insert_prepared(prepared(key)).is_none());
        }
        for (lower, upper, expected) in [
            (Bound::Unbounded, Bound::Unbounded, vec![1, 3, 5, 7, 9]),
            (Bound::Included(&3), Bound::Included(&7), vec![3, 5, 7]),
            (Bound::Excluded(&3), Bound::Excluded(&7), vec![5]),
            (Bound::Included(&5), Bound::Included(&5), vec![5]),
            (Bound::Excluded(&5), Bound::Included(&5), vec![]),
            (Bound::Included(&5), Bound::Excluded(&5), vec![]),
            (Bound::Included(&7), Bound::Included(&3), vec![]),
            (Bound::Unbounded, Bound::Excluded(&3), vec![1]),
            (Bound::Excluded(&7), Bound::Unbounded, vec![9]),
        ] {
            let mut actual = Vec::new();
            let count = directory.visit(lower, upper, |key| {
                actual.push(*key);
                true
            });
            assert_eq!(actual, expected);
            assert_eq!(count, expected.len());
        }
        let mut actual = Vec::new();
        let visited = directory.visit(Bound::Included(&3), Bound::Unbounded, |key| {
            actual.push(*key);
            actual.len() < 2
        });
        assert_eq!(actual, vec![3, 5]);
        assert_eq!(visited, 2, "the stopping key was delivered and counts");
        assert!(directory.contains(&7));
        assert!(!directory.contains(&6));
        assert_eq!(directory.len(), 5);
    }

    #[test]
    fn ordered_all_rotation_shapes_remain_balanced() {
        for order in [[3, 2, 1], [1, 2, 3], [3, 1, 2], [1, 3, 2]] {
            let mut directory = OrderedDirectory::default();
            for key in order {
                assert!(directory.insert_prepared(prepared(key)).is_none());
                assert_eq!(check_avl(&directory.root, None, None).0, directory.len());
            }
            assert_eq!(keys(&directory), vec![1, 2, 3]);
            assert_eq!(*directory.root.as_ref().unwrap().key(), 2);
        }
    }

    #[test]
    fn ordered_adversarial_insert_remove_matches_set() {
        for descending in [false, true] {
            let mut directory = OrderedDirectory::new();
            let mut expected = BTreeSet::new();
            for offset in 0..1024 {
                let key = if descending { 1023 - offset } else { offset };
                assert!(directory.insert_prepared(prepared(key)).is_none());
                expected.insert(key);
                assert_eq!(check_avl(&directory.root, None, None).0, expected.len());
            }
            // Alternating extremes, then interior keys, exercise deletion rotations.
            for offset in 0..1024 {
                let key = if offset % 2 == 0 {
                    offset / 2
                } else {
                    1023 - offset / 2
                };
                let removed = directory.remove(&key).expect("existing key");
                assert_eq!(*removed.key(), key);
                assert!(removed.node().left.is_none() && removed.node().right.is_none());
                assert_eq!(removed.node().height, 1);
                expected.remove(&key);
                assert_eq!(directory.len(), expected.len());
                assert_eq!(check_avl(&directory.root, None, None).0, expected.len());
                assert_eq!(
                    keys(&directory),
                    expected.iter().copied().collect::<Vec<_>>()
                );
            }
            assert!(directory.remove(&9999).is_none());
        }
    }

    #[test]
    fn ordered_detached_and_duplicate_nodes_can_be_reused() {
        let mut first = OrderedDirectory::new();
        for key in [4, 2, 6, 1, 3, 5, 7] {
            assert!(first.insert_prepared(prepared(key)).is_none());
        }
        let duplicate = first
            .insert_prepared(prepared(4))
            .expect("unused duplicate");
        assert_eq!(*duplicate.key(), 4);
        assert_eq!(first.len(), 7);
        let detached = first.remove(&4).expect("two-child removal");
        let mut second = OrderedDirectory::new();
        assert!(second.insert_prepared(detached).is_none());
        assert!(first.insert_prepared(duplicate).is_none());
        assert_eq!(keys(&first), vec![1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(keys(&second), vec![4]);
    }

    #[test]
    fn ordered_install_remove_and_visit_have_no_allocator_traffic() {
        let mut directory = OrderedDirectory::new();
        let mut pending: Vec<_> = [
            "middle", "alpha", "zulu", "bravo", "charlie", "xray", "yankee",
        ]
        .into_iter()
        .map(|key| prepared(key.to_owned()))
        .collect();
        let duplicate = prepared("middle".to_owned());
        let target = "middle".to_owned();
        let targets: Vec<_> = [
            "middle", "alpha", "zulu", "bravo", "charlie", "xray", "yankee",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        let mut retired = Vec::with_capacity(pending.len());
        crate::allocation_test::start();
        for key in pending.drain(..) {
            assert!(directory.insert_prepared(key).is_none());
        }
        let duplicate = directory.insert_prepared(duplicate);
        let detached = directory.remove(&target);
        assert!(directory.insert_prepared(detached.unwrap()).is_none());
        let visited = directory.visit(Bound::Unbounded, Bound::Unbounded, |_| true);
        for key in &targets {
            retired.push(directory.remove(key).expect("existing prepared target"));
        }
        let traffic = crate::allocation_test::stop();
        assert_eq!(traffic, crate::allocation_test::Counts::default());
        assert_eq!(visited, 7);
        assert_eq!(retired.len(), 7);
        assert!(duplicate.is_some());
        crate::allocation_test::start();
        drop(retired);
        drop(duplicate);
        let retirement = crate::allocation_test::stop();
        assert!(
            retirement.dealloc > 0,
            "outer retirement owns the allocations"
        );
    }

    #[test]
    fn ordered_node_allocation_failure_is_recoverable() {
        let (result, fired) = crate::allocation_test::with_failure(
            std::mem::size_of::<Node<i64>>(),
            std::mem::align_of::<Node<i64>>(),
            0,
            || PreparedKey::new(7i64),
        );
        assert!(fired, "the fallible node allocation must be intercepted");
        assert!(result.is_err());
        assert_eq!(*prepared(7i64).key(), 7);
    }

    struct CountedKey(i32, Arc<AtomicUsize>);
    impl PartialEq for CountedKey {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }
    impl Eq for CountedKey {}
    impl PartialOrd for CountedKey {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for CountedKey {
        fn cmp(&self, other: &Self) -> Ordering {
            self.1.fetch_add(1, AtomicOrdering::Relaxed);
            self.0.cmp(&other.0)
        }
    }

    #[test]
    fn ordered_narrow_range_seeks_instead_of_walking_keys() {
        let comparisons = Arc::new(AtomicUsize::new(0));
        let mut directory = OrderedDirectory::new();
        for key in 0..4096 {
            assert!(
                directory
                    .insert_prepared(prepared(CountedKey(key, Arc::clone(&comparisons))))
                    .is_none()
            );
        }
        let bound = CountedKey(2048, Arc::clone(&comparisons));
        comparisons.store(0, AtomicOrdering::Relaxed);
        let count = directory.visit(Bound::Included(&bound), Bound::Included(&bound), |key| {
            assert_eq!(key.0, 2048);
            true
        });
        assert_eq!(count, 1);
        assert!(
            comparisons.load(AtomicOrdering::Relaxed) < 128,
            "selective lookup must seek"
        );
    }
}
