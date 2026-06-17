//! Sharded read-registry (SIREAD locks) for incremental SSI.
//!
//! Maps each entity to the set of *active Serializable* transactions that have read
//! it, so a writer can find concurrent readers and record the read-write
//! antidependency. Sharded by entity hash to avoid a single global lock. A reverse
//! index (`by_tx`) makes GC of a finished reader O(its reads).

use grafeo_common::types::TransactionId;
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLock;

use super::EntityId;

const SHARDS: usize = 64; // power of two

/// Sharded registry of active Serializable readers (SIREAD locks).
pub struct ReadRegistry {
    shards: Vec<RwLock<FxHashMap<EntityId, FxHashSet<TransactionId>>>>,
    by_tx: RwLock<FxHashMap<TransactionId, Vec<EntityId>>>,
}

impl ReadRegistry {
    /// Creates a new, empty `ReadRegistry`.
    pub fn new() -> Self {
        let mut shards = Vec::with_capacity(SHARDS);
        for _ in 0..SHARDS {
            shards.push(RwLock::new(FxHashMap::default()));
        }
        Self {
            shards,
            by_tx: RwLock::new(FxHashMap::default()),
        }
    }

    fn shard_idx(entity: &EntityId) -> usize {
        // Truncating to u32 before widening to usize is intentional: the lower
        // 32 bits provide sufficient spread for SHARDS (≤64), and usize may be
        // narrower than u64 on 32-bit targets.
        #[allow(clippy::cast_possible_truncation)]
        let low = grafeo_common::utils::hash::hash_one(entity) as u32;
        (low as usize) & (SHARDS - 1)
    }

    fn shard(&self, entity: &EntityId) -> &RwLock<FxHashMap<EntityId, FxHashSet<TransactionId>>> {
        &self.shards[Self::shard_idx(entity)]
    }

    /// Register that `tx` (an active Serializable reader) read `entity`.
    pub fn record_reader(&self, entity: EntityId, tx: TransactionId) {
        let inserted = self
            .shard(&entity)
            .write()
            .entry(entity)
            .or_default()
            .insert(tx);
        if inserted {
            self.by_tx.write().entry(tx).or_default().push(entity);
        }
    }

    /// Active readers of `entity` (for write-time antidependency detection).
    pub fn readers_of(&self, entity: EntityId) -> Vec<TransactionId> {
        self.shard(&entity)
            .read()
            .get(&entity)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default()
    }

    /// GC every entry for a finished (committed/aborted) reader.
    pub fn remove_reader(&self, tx: TransactionId) {
        let entities = self.by_tx.write().remove(&tx).unwrap_or_default();
        for e in entities {
            let mut sh = self.shard(&e).write();
            if let Some(set) = sh.get_mut(&e) {
                set.remove(&tx);
                if set.is_empty() {
                    sh.remove(&e);
                }
            }
        }
    }
}

impl Default for ReadRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::{EdgeId, NodeId, TransactionId};

    use super::ReadRegistry;
    use crate::transaction::EntityId;

    fn txid(n: u64) -> TransactionId {
        TransactionId::new(n)
    }

    #[test]
    fn test_record_and_readers_of() {
        let reg = ReadRegistry::new();
        let t1 = txid(1);
        let t2 = txid(2);

        let node1 = EntityId::Node(NodeId::new(1));
        let edge9 = EntityId::Edge(EdgeId::new(9));
        let node7 = EntityId::Node(NodeId::new(7));

        reg.record_reader(node1, t1);
        reg.record_reader(node1, t2);
        reg.record_reader(edge9, t1);

        let mut r1 = reg.readers_of(node1);
        r1.sort();
        assert_eq!(r1, vec![t1, t2], "node1 should have readers t1 and t2");

        let mut r9 = reg.readers_of(edge9);
        r9.sort();
        assert_eq!(r9, vec![t1], "edge9 should have reader t1");

        let r7 = reg.readers_of(node7);
        assert!(r7.is_empty(), "node7 has no readers");
    }

    #[test]
    fn test_remove_reader_cleans_up() {
        let reg = ReadRegistry::new();
        let t1 = txid(1);
        let t2 = txid(2);

        let node1 = EntityId::Node(NodeId::new(1));
        let edge9 = EntityId::Edge(EdgeId::new(9));

        reg.record_reader(node1, t1);
        reg.record_reader(node1, t2);
        reg.record_reader(edge9, t1);

        reg.remove_reader(t1);

        let mut r1 = reg.readers_of(node1);
        r1.sort();
        assert_eq!(r1, vec![t2], "node1 should only have t2 after removing t1");

        let r9 = reg.readers_of(edge9);
        assert!(
            r9.is_empty(),
            "edge9 set should be empty (and entity entry dropped)"
        );

        // Verify the entity entry is truly gone from the shard (not just empty).
        // We can only observe this indirectly — readers_of returns [] which is
        // consistent with both "entry absent" and "entry present but empty",
        // but the implementation removes the entry when the set empties.
        // A double-remove should be a no-op (no panic).
        reg.remove_reader(t1);
    }

    #[test]
    fn test_remove_unknown_tx_is_noop() {
        let reg = ReadRegistry::new();
        // No prior records — must not panic.
        reg.remove_reader(txid(99));
    }

    #[test]
    fn test_idempotent_record() {
        let reg = ReadRegistry::new();
        let t1 = txid(1);
        let node1 = EntityId::Node(NodeId::new(1));

        reg.record_reader(node1, t1);
        reg.record_reader(node1, t1); // duplicate — must not double-insert

        let r = reg.readers_of(node1);
        assert_eq!(r.len(), 1, "duplicate record_reader must not double-insert");
    }
}
