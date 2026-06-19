//! Bridge between mutation operators and the transaction manager's write tracking.

use std::sync::Arc;

use grafeo_common::types::{EdgeId, EdgeTypeId, LabelId, NodeId, TransactionId};
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

    /// Records a node write with coarse `Label(L)` fan-out for each label.
    ///
    /// Calls [`TransactionManager::record_node_write`] which records both
    /// `EntityId::Node(node_id)` and `EntityId::Label(L)` for every `L`
    /// in `labels`. Used by `CreateNodeOperator` (which knows its labels)
    /// and the store's `create_node_versioned` phantom-write path.
    fn record_node_write_with_labels(
        &self,
        transaction_id: TransactionId,
        node_id: NodeId,
        labels: &[LabelId],
    ) -> Result<(), OperatorError> {
        self.manager
            .record_node_write(transaction_id, node_id, labels, None)
            .map_err(|e| OperatorError::WriteConflict(e.to_string()))
    }

    /// Records an edge write with coarse `RelType(T)` fan-out.
    ///
    /// Calls [`TransactionManager::record_edge_write`] which records both
    /// `EntityId::Edge(edge_id)` and `EntityId::RelType(rel_type)`.
    /// Used by `CreateEdgeOperator` and the store's `create_edge_versioned`
    /// phantom-write path.
    fn record_edge_write_with_type(
        &self,
        transaction_id: TransactionId,
        edge_id: EdgeId,
        rel_type: EdgeTypeId,
    ) -> Result<(), OperatorError> {
        self.manager
            .record_edge_write(transaction_id, edge_id, rel_type, None)
            .map_err(|e| OperatorError::WriteConflict(e.to_string()))
    }

    /// Fans out only the coarse `Label(L)` phantom writes (no fine `Node` write).
    ///
    /// Calls [`TransactionManager::record_node_labels_write`]. Used by the store's
    /// property-write paths so an escalated `Label(L)` reader is caught without
    /// adding a `None`-tagged fine `Node` write that would defeat Property
    /// granularity (the fine write is recorded elsewhere under its property tag).
    fn record_node_labels_write(
        &self,
        transaction_id: TransactionId,
        labels: &[LabelId],
    ) -> Result<(), OperatorError> {
        self.manager
            .record_node_labels_write(transaction_id, labels)
            .map_err(|e| OperatorError::WriteConflict(e.to_string()))
    }

    /// Fans out only the coarse `RelType(T)` phantom write (no fine `Edge` write).
    ///
    /// Calls [`TransactionManager::record_edge_type_write`]. Edge mirror of
    /// [`record_node_labels_write`](Self::record_node_labels_write).
    fn record_edge_type_write(
        &self,
        transaction_id: TransactionId,
        rel_type: EdgeTypeId,
    ) -> Result<(), OperatorError> {
        self.manager
            .record_edge_type_write(transaction_id, rel_type)
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
