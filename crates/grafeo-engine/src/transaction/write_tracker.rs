//! Bridge between mutation operators and the transaction manager's write tracking.

use std::sync::Arc;

use grafeo_common::types::{EdgeId, NodeId, TransactionId};
use grafeo_core::execution::operators::{OperatorError, WriteTracker};

use super::{ConflictGranularity, IndexId, TransactionManager, prop_tag};

/// Implements [`WriteTracker`] by forwarding to [`TransactionManager::record_write`].
///
/// Created by the planner when a transaction is active, and passed to each
/// mutation operator so it can record writes for conflict detection.
pub struct TransactionWriteTracker {
    manager: Arc<TransactionManager>,
    granularity: ConflictGranularity,
}

impl TransactionWriteTracker {
    /// Creates a new write tracker backed by the given transaction manager.
    pub fn new(manager: Arc<TransactionManager>) -> Self {
        Self {
            manager,
            granularity: ConflictGranularity::Entity,
        }
    }

    /// Creates a new write tracker with the specified conflict granularity.
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

impl WriteTracker for TransactionWriteTracker {
    fn record_node_write(
        &self,
        transaction_id: TransactionId,
        node_id: NodeId,
    ) -> Result<(), OperatorError> {
        self.manager
            .record_write(transaction_id, node_id, None)
            .map_err(|e| OperatorError::WriteConflict(e.to_string()))
    }

    fn record_edge_write(
        &self,
        transaction_id: TransactionId,
        edge_id: EdgeId,
    ) -> Result<(), OperatorError> {
        self.manager
            .record_write(transaction_id, edge_id, None)
            .map_err(|e| OperatorError::WriteConflict(e.to_string()))
    }

    fn record_node_property_write(
        &self,
        transaction_id: TransactionId,
        node_id: NodeId,
        key: &str,
    ) -> Result<(), OperatorError> {
        let tag = if self.granularity == ConflictGranularity::Property {
            Some(prop_tag(key))
        } else {
            None
        };
        self.manager
            .record_write(transaction_id, node_id, tag)
            .map_err(|e| OperatorError::WriteConflict(e.to_string()))
    }

    fn record_edge_property_write(
        &self,
        transaction_id: TransactionId,
        edge_id: EdgeId,
        key: &str,
    ) -> Result<(), OperatorError> {
        let tag = if self.granularity == ConflictGranularity::Property {
            Some(prop_tag(key))
        } else {
            None
        };
        self.manager
            .record_write(transaction_id, edge_id, tag)
            .map_err(|e| OperatorError::WriteConflict(e.to_string()))
    }

    /// Records that `transaction_id` wrote to the `(label, property)` text index
    /// identified by `index_key`. Coarse index-write recording for anti-phantom SSI.
    ///
    /// Intentionally infallible: the index-entity write is a coarse SSI signal,
    /// not a first-writer-wins entity lock (two concurrent indexed SETs to different
    /// *nodes* in the same index are not a W-W conflict on the index entity itself).
    fn record_index_write(&self, transaction_id: TransactionId, index_key: &str) {
        let idx = if let Some((label, property)) = index_key.split_once(':') {
            IndexId::for_text_index(label, property)
        } else {
            IndexId::for_text_index(index_key, "")
        };
        // Ignore W-W conflict result: the index is not a first-writer-wins entity.
        let _ = self.manager.record_write(transaction_id, idx, None);
    }
}
