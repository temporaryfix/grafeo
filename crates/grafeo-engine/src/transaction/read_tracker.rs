//! Bridge between read operators and the transaction manager's read tracking.

use std::sync::Arc;

use grafeo_common::types::{EdgeId, EdgeTypeId, LabelId, NodeId, TransactionId};
use grafeo_core::execution::operators::ReadTracker;

use super::{
    ConflictGranularity, IndexId, IsolationLevel, STRUCT_TAG, TransactionManager, prop_tag,
};

/// Implements [`ReadTracker`] by forwarding to [`TransactionManager::record_read`].
///
/// No-op unless the transaction is Serializable (one isolation check), so
/// SnapshotIsolation / ReadCommitted pay nothing even if a tracker is attached.
pub struct TransactionReadTracker {
    manager: Arc<TransactionManager>,
    granularity: ConflictGranularity,
}

impl TransactionReadTracker {
    /// Creates a new read tracker backed by the given transaction manager.
    pub fn new(manager: Arc<TransactionManager>) -> Self {
        Self {
            manager,
            granularity: ConflictGranularity::Entity,
        }
    }

    /// Creates a new read tracker with the specified conflict granularity.
    pub fn with_granularity(
        manager: Arc<TransactionManager>,
        granularity: ConflictGranularity,
    ) -> Self {
        Self {
            manager,
            granularity,
        }
    }
}

impl ReadTracker for TransactionReadTracker {
    fn record_node_read(&self, transaction_id: TransactionId, node_id: NodeId) {
        // A purely structural read (label-scan visit / existence / label check)
        // names no property. Under Property granularity we tag it with the
        // reserved `STRUCT_TAG` rather than dropping it: dropping would miss a
        // concurrent structural write (DELETE/label change, recorded as `None`)
        // — an unsound missed rw-antidependency. `Some(STRUCT_TAG)` is
        // `prop_compatible` with a `None` write (conflict detected = sound) yet
        // disjoint from any property tag (no false conflict = the knob holds).
        if self.manager.isolation_level(transaction_id) == Some(IsolationLevel::Serializable) {
            let tag = if self.granularity == ConflictGranularity::Property {
                Some(STRUCT_TAG)
            } else {
                None
            };
            let _ = self.manager.record_read(transaction_id, node_id, tag);
        }
    }

    fn record_edge_read(&self, transaction_id: TransactionId, edge_id: EdgeId) {
        // Symmetric with `record_node_read`: structural edge reads record the
        // reserved `STRUCT_TAG` under Property granularity (entity-level `None`
        // under Entity), never nothing.
        if self.manager.isolation_level(transaction_id) == Some(IsolationLevel::Serializable) {
            let tag = if self.granularity == ConflictGranularity::Property {
                Some(STRUCT_TAG)
            } else {
                None
            };
            let _ = self.manager.record_read(transaction_id, edge_id, tag);
        }
    }

    fn record_node_property_read(&self, transaction_id: TransactionId, node_id: NodeId, key: &str) {
        if self.manager.isolation_level(transaction_id) == Some(IsolationLevel::Serializable) {
            let tag = if self.granularity == ConflictGranularity::Property {
                Some(prop_tag(key))
            } else {
                None
            };
            let _ = self.manager.record_read(transaction_id, node_id, tag);
        }
    }

    fn record_edge_property_read(&self, transaction_id: TransactionId, edge_id: EdgeId, key: &str) {
        if self.manager.isolation_level(transaction_id) == Some(IsolationLevel::Serializable) {
            let tag = if self.granularity == ConflictGranularity::Property {
                Some(prop_tag(key))
            } else {
                None
            };
            let _ = self.manager.record_read(transaction_id, edge_id, tag);
        }
    }

    fn record_read_node_in_label(
        &self,
        transaction_id: TransactionId,
        node_id: NodeId,
        label_id: LabelId,
    ) {
        if self.manager.isolation_level(transaction_id) == Some(IsolationLevel::Serializable) {
            let tag = if self.granularity == ConflictGranularity::Property {
                Some(STRUCT_TAG)
            } else {
                None
            };
            let _ = self
                .manager
                .record_read_in_label(transaction_id, node_id, tag, label_id);
        }
    }

    fn record_read_edge_in_rel_type(
        &self,
        transaction_id: TransactionId,
        edge_id: EdgeId,
        rel_type: EdgeTypeId,
    ) {
        if self.manager.isolation_level(transaction_id) == Some(IsolationLevel::Serializable) {
            let tag = if self.granularity == ConflictGranularity::Property {
                Some(STRUCT_TAG)
            } else {
                None
            };
            let _ = self
                .manager
                .record_read_in_rel_type(transaction_id, edge_id, tag, rel_type);
        }
    }

    /// Records a Serializable text search as a coarse **index-level predicate read**.
    ///
    /// A text search reads a *predicate* ("docs matching these terms") over all
    /// documents in the `(label, property)` index. Any concurrent transaction that
    /// inserts or updates a matching document is a phantom; the coarse SSI fix is
    /// to record the whole index as read. A concurrent indexed SET then records the
    /// same index as written, and the existing rw-antidependency machinery aborts
    /// the conflicting pair (preventing the phantom).
    ///
    /// Only fires for Serializable; SI/ReadCommitted pay nothing.
    fn record_index_read(&self, transaction_id: TransactionId, index_key: &str) {
        if self.manager.isolation_level(transaction_id) == Some(IsolationLevel::Serializable) {
            // The `index_key` format is "label:property". Feed it directly to
            // `IndexId::for_text_index` by splitting on the first ':'.
            let idx = if let Some((label, property)) = index_key.split_once(':') {
                IndexId::for_text_index(label, property)
            } else {
                // Malformed key: fall back to hashing the whole string.
                IndexId::for_text_index(index_key, "")
            };
            let _ = self.manager.record_read(transaction_id, idx, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use grafeo_common::types::NodeId;
    use grafeo_core::execution::operators::ReadTracker;

    use super::TransactionReadTracker;
    use crate::transaction::{ConflictGranularity, EntityId, IsolationLevel, TransactionManager};

    #[test]
    fn test_serializable_tx_records_node_read() {
        let mgr = Arc::new(TransactionManager::new());
        let tx_s = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let tx_si = mgr.begin(); // SnapshotIsolation (default)

        let t = TransactionReadTracker::new(Arc::clone(&mgr));

        // Serializable tx: node read should be recorded.
        t.record_node_read(tx_s, NodeId::new(7));
        let rs = mgr.read_set(tx_s);
        assert!(
            rs.contains(&EntityId::Node(NodeId::new(7))),
            "expected NodeId(7) in read-set of serializable tx, got: {rs:?}"
        );

        // SnapshotIsolation tx: bridge must be a no-op.
        t.record_node_read(tx_si, NodeId::new(8));
        let rs_si = mgr.read_set(tx_si);
        assert!(
            rs_si.is_empty(),
            "expected empty read-set for SI tx, got: {rs_si:?}"
        );
    }

    #[test]
    fn test_property_granularity_records_prop_tag() {
        let mgr = Arc::new(TransactionManager::new());
        let tx = mgr.begin_with_isolation(IsolationLevel::Serializable);

        let t = TransactionReadTracker::with_granularity(
            Arc::clone(&mgr),
            ConflictGranularity::Property,
        );

        // Property-granularity read: should record entity + Some(tag), not None.
        t.record_node_property_read(tx, NodeId::new(1), "balance");

        // The read-set entry for NodeId(1) should have a Some tag.
        let rs = mgr.read_set_tagged(tx);
        let has_prop_tag = rs
            .iter()
            .any(|(e, tag)| *e == EntityId::Node(NodeId::new(1)) && tag.is_some());
        assert!(
            has_prop_tag,
            "expected a Some-tagged entry for NodeId(1) under Property granularity, got: {rs:?}"
        );
    }

    #[test]
    fn test_entity_granularity_records_none_tag() {
        let mgr = Arc::new(TransactionManager::new());
        let tx = mgr.begin_with_isolation(IsolationLevel::Serializable);

        let t =
            TransactionReadTracker::with_granularity(Arc::clone(&mgr), ConflictGranularity::Entity);

        // Entity-granularity property read: should record None tag (entity-level).
        t.record_node_property_read(tx, NodeId::new(2), "balance");

        let rs = mgr.read_set_tagged(tx);
        let has_none_tag = rs
            .iter()
            .any(|(e, tag)| *e == EntityId::Node(NodeId::new(2)) && tag.is_none());
        assert!(
            has_none_tag,
            "expected a None-tagged entry for NodeId(2) under Entity granularity, got: {rs:?}"
        );
    }

    /// Soundness regression (Part G hole): a *purely structural* read (no property
    /// accessed — e.g. a label-scan visit / existence check, which flows through
    /// [`ReadTracker::record_node_read`]) must still form rw-antidependencies
    /// against a concurrent **structural write** (a DELETE / label change, recorded
    /// as the `None` wildcard tag) under `Property` granularity.
    ///
    /// This is the classic write-skew dangerous structure, except both readers
    /// observe their entities **structurally** through the tracker (not via a
    /// property read). Under the original `Property` suppression the structural
    /// reads recorded NOTHING, so T2's read-set was empty, no rw-edge formed, and
    /// T2 committed → a missed rw-antidependency (unsound). With the reserved
    /// `STRUCT_TAG`, the structural read is `Some(STRUCT_TAG)`, which is
    /// `prop_compatible` with the delete's `None` write → the cycle closes and the
    /// second committer aborts (sound).
    ///
    /// Mirrors the all-`None` `test_ssi_concurrent_commit_race` in `manager.rs`,
    /// but routes the reads through the `Property`-granularity tracker to exercise
    /// the suppressed structural-read path specifically.
    #[test]
    fn structural_read_vs_structural_write_aborts_under_property_granularity() {
        let mgr = Arc::new(TransactionManager::new());

        let entity_a = NodeId::new(1);
        let entity_b = NodeId::new(2);

        let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

        // Property-granularity trackers: the structural reads below go through
        // `record_node_read` (the path that used to `return` early under Property).
        let t1 = TransactionReadTracker::with_granularity(
            Arc::clone(&mgr),
            ConflictGranularity::Property,
        );
        let t2 = TransactionReadTracker::with_granularity(
            Arc::clone(&mgr),
            ConflictGranularity::Property,
        );

        // Both txns read both entities *structurally* (existence/label, no property).
        t1.record_node_read(tx1, entity_a);
        t1.record_node_read(tx1, entity_b);
        t2.record_node_read(tx2, entity_a);
        t2.record_node_read(tx2, entity_b);

        // Each txn issues a *structural* write (a DELETE → entity-level `None`):
        // tx1 deletes A, tx2 deletes B. Disjoint entities → no write-write conflict.
        mgr.record_write(tx1, entity_a, None).unwrap();
        mgr.record_write(tx2, entity_b, None).unwrap();

        // tx1 commits first.
        mgr.commit(tx1).expect("tx1 (first committer) must succeed");

        // tx2 must abort: it read A structurally, tx1 deleted A (structural write),
        // and tx1 read B structurally while tx2 deletes B → the rw-cycle closes.
        let c2 = mgr.commit(tx2);
        assert!(
            matches!(
                c2,
                Err(grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::SerializationFailure(_)
                ))
            ),
            "tx2 must abort under Property granularity: a structural read vs a \
             structural (DELETE) write is a real rw-antidependency. Got: {c2:?}"
        );
    }
}
