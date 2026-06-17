//! Bridge between read operators and the transaction manager's read tracking.

use std::sync::Arc;

use grafeo_common::types::{EdgeId, NodeId, TransactionId};
use grafeo_core::execution::operators::ReadTracker;

use super::{ConflictGranularity, IsolationLevel, TransactionManager, prop_tag};

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
        // Under Property granularity, entity-level scan reads are suppressed:
        // only property-tagged reads from `record_node_property_read` are
        // tracked. Structural writes (None-tagged) will still conflict via
        // `prop_compatible(Some(tag), None) = true` when they do happen.
        if self.granularity == ConflictGranularity::Property {
            return;
        }
        if self.manager.isolation_level(transaction_id) == Some(IsolationLevel::Serializable) {
            let _ = self.manager.record_read(transaction_id, node_id, None);
        }
    }

    fn record_edge_read(&self, transaction_id: TransactionId, edge_id: EdgeId) {
        // Under Property granularity, entity-level scan reads are suppressed.
        if self.granularity == ConflictGranularity::Property {
            return;
        }
        if self.manager.isolation_level(transaction_id) == Some(IsolationLevel::Serializable) {
            let _ = self.manager.record_read(transaction_id, edge_id, None);
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
}
