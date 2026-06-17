//! Bridge between mutation operators and the transaction manager's write tracking.

use std::sync::Arc;

use grafeo_common::types::{EdgeId, NodeId, TransactionId};
use grafeo_core::execution::operators::{OperatorError, WriteTracker};

use super::{ConflictGranularity, TransactionManager, prop_tag};

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
}
