//! Bridge between read operators and the transaction manager's read tracking.

use std::sync::Arc;

use grafeo_common::types::{EdgeId, NodeId, TransactionId};
use grafeo_core::execution::operators::ReadTracker;

use super::{IsolationLevel, TransactionManager};

/// Implements [`ReadTracker`] by forwarding to [`TransactionManager::record_read`].
///
/// No-op unless the transaction is Serializable (one isolation check), so
/// SnapshotIsolation / ReadCommitted pay nothing even if a tracker is attached.
pub struct TransactionReadTracker {
    manager: Arc<TransactionManager>,
}

impl TransactionReadTracker {
    /// Creates a new read tracker backed by the given transaction manager.
    pub fn new(manager: Arc<TransactionManager>) -> Self {
        Self { manager }
    }
}

impl ReadTracker for TransactionReadTracker {
    fn record_node_read(&self, transaction_id: TransactionId, node_id: NodeId) {
        if self.manager.isolation_level(transaction_id) == Some(IsolationLevel::Serializable) {
            let _ = self.manager.record_read(transaction_id, node_id);
        }
    }

    fn record_edge_read(&self, transaction_id: TransactionId, edge_id: EdgeId) {
        if self.manager.isolation_level(transaction_id) == Some(IsolationLevel::Serializable) {
            let _ = self.manager.record_read(transaction_id, edge_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use grafeo_common::types::NodeId;
    use grafeo_core::execution::operators::ReadTracker;

    use super::TransactionReadTracker;
    use crate::transaction::{EntityId, IsolationLevel, TransactionManager};

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
}
