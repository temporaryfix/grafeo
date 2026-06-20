//! Physical operators that actually execute queries.
//!
//! These are the building blocks of query execution. The optimizer picks which
//! operators to use and how to wire them together.
//!
//! **Graph operators:**
//! - [`ScanOperator`] - Read nodes/edges from storage
//! - [`ExpandOperator`] - Traverse edges (the core of graph queries)
//! - [`VariableLengthExpandOperator`] - Paths of variable length
//! - [`ShortestPathOperator`] - Find shortest paths
//!
//! **Relational operators:**
//! - [`FilterOperator`] - Apply predicates
//! - [`ProjectOperator`] - Select/transform columns
//! - [`HashJoinOperator`] - Efficient equi-joins
//! - [`HashAggregateOperator`] - Group by with aggregation
//! - [`SortOperator`] - Order results
//! - [`LimitOperator`] - SKIP and LIMIT
//!
//! The [`push`] submodule has push-based variants for pipeline execution.

pub mod accumulator;
mod aggregate;
mod apply;
mod distinct;
mod expand;
mod factorized_aggregate;
mod factorized_expand;
mod factorized_filter;
mod filter;
mod horizontal_aggregate;
mod join;
mod leapfrog_join;
mod limit;
mod load_data;
mod map_collect;
mod merge;
mod mutation;
mod parameter_scan;
mod project;
pub mod push;
mod range_scan;
mod scan;
#[cfg(feature = "text-index")]
mod scan_text;
#[cfg(feature = "vector-index")]
mod scan_vector;
mod set_ops;
mod shortest_path;
pub mod single_row;
mod sort;
pub mod top_k;
mod union;
mod unwind;
pub mod value_utils;
mod variable_length_expand;
mod vector_join;

pub use accumulator::{AggregateExpr, AggregateFunction, HashableValue};
pub use aggregate::{HashAggregateOperator, SimpleAggregateOperator};
pub use apply::ApplyOperator;
pub use distinct::DistinctOperator;
pub use expand::ExpandOperator;
pub use factorized_aggregate::{
    FactorizedAggregate, FactorizedAggregateOperator, FactorizedOperator,
};
pub use factorized_expand::{
    ExpandStep, FactorizedExpandChain, FactorizedExpandOperator, FactorizedResult,
    LazyFactorizedChainOperator,
};
pub use factorized_filter::{
    AndPredicate, ColumnPredicate, CompareOp as FactorizedCompareOp, FactorizedFilterOperator,
    FactorizedPredicate, OrPredicate, PropertyPredicate,
};
pub use filter::{
    BinaryFilterOp, ExpressionPredicate, FilterExpression, FilterOperator, LazyValue,
    ListPredicateKind, Predicate, SessionContext, UnaryFilterOp,
};
pub use horizontal_aggregate::{EntityKind, HorizontalAggregateOperator};
pub use join::{
    EqualityCondition, HashJoinOperator, HashKey, JoinCondition, JoinType, NestedLoopJoinOperator,
};
pub use leapfrog_join::LeapfrogJoinOperator;
pub use limit::{LimitOperator, LimitSkipOperator, SkipOperator};
pub use load_data::{LoadDataFormat, LoadDataOperator};
pub use map_collect::MapCollectOperator;
pub use merge::{MergeConfig, MergeOperator, MergeRelationshipConfig, MergeRelationshipOperator};
pub use mutation::{
    AddLabelOperator, ConstraintValidator, CreateEdgeOperator, CreateNodeOperator,
    DeleteEdgeOperator, DeleteNodeOperator, PropertySource, RemoveLabelOperator,
    SetPropertyOperator,
};
pub use parameter_scan::{ParameterScanOperator, ParameterState};
pub use project::{ProjectExpr, ProjectOperator};
pub use push::{
    AggregatePushOperator, DistinctMaterializingOperator, DistinctPushOperator, FilterPushOperator,
    LimitPushOperator, ProjectPushOperator, SkipLimitPushOperator, SkipPushOperator,
    SortPushOperator,
};
#[cfg(feature = "spill")]
pub use push::{SpillableAggregatePushOperator, SpillableSortPushOperator};
pub use range_scan::RangeScanOperator;
pub use scan::ScanOperator;
#[cfg(feature = "text-index")]
pub use scan_text::TextScanOperator;
#[cfg(feature = "vector-index")]
pub use scan_vector::VectorScanOperator;
pub use set_ops::{ExceptOperator, IntersectOperator, OtherwiseOperator};
pub use shortest_path::ShortestPathOperator;
pub use single_row::{EmptyOperator, NodeListOperator, SingleRowOperator};
pub use sort::{NullOrder, SortDirection, SortKey, SortOperator};
pub use top_k::TopKOperator;
pub use union::UnionOperator;
pub use unwind::UnwindOperator;
pub use variable_length_expand::{PathMode as ExecutionPathMode, VariableLengthExpandOperator};
pub use vector_join::VectorJoinOperator;

use std::sync::Arc;

use grafeo_common::types::{EdgeId, EdgeTypeId, LabelId, NodeId, TransactionId};
use thiserror::Error;

use super::DataChunk;
use super::chunk_state::ChunkState;
use super::factorized_chunk::FactorizedChunk;

/// Trait for recording write operations during query execution.
///
/// This bridges `grafeo-core` mutation operators (which perform writes) with
/// `grafeo-engine`'s `TransactionManager` (which tracks write sets for conflict
/// detection). The trait lives in `grafeo-core` to avoid circular dependencies.
pub trait WriteTracker: Send + Sync {
    /// Records that a node was written (created, deleted, or modified).
    ///
    /// # Errors
    ///
    /// Returns `Err` if a write-write conflict is detected (first-writer-wins).
    fn record_node_write(
        &self,
        transaction_id: TransactionId,
        node_id: NodeId,
    ) -> Result<(), OperatorError>;

    /// Records that an edge was written (created, deleted, or modified).
    ///
    /// # Errors
    ///
    /// Returns `Err` if a write-write conflict is detected (first-writer-wins).
    fn record_edge_write(
        &self,
        transaction_id: TransactionId,
        edge_id: EdgeId,
    ) -> Result<(), OperatorError>;

    /// Records a write to property `key` of `node_id`. Default impl ignores the
    /// property and records an entity-level write — overridden by the engine bridge
    /// to record a property tag under property-level granularity.
    ///
    /// # Errors
    ///
    /// Returns `Err` on a write-write conflict (first-writer-wins), same as `record_node_write`.
    fn record_node_property_write(
        &self,
        transaction_id: TransactionId,
        node_id: NodeId,
        _key: &str,
    ) -> Result<(), OperatorError> {
        self.record_node_write(transaction_id, node_id)
    }

    /// Records a write to property `key` of `edge_id`. Default impl ignores the
    /// property and records an entity-level write — overridden by the engine bridge
    /// to record a property tag under property-level granularity.
    ///
    /// # Errors
    ///
    /// Returns `Err` on a write-write conflict (first-writer-wins), same as `record_edge_write`.
    fn record_edge_property_write(
        &self,
        transaction_id: TransactionId,
        edge_id: EdgeId,
        _key: &str,
    ) -> Result<(), OperatorError> {
        self.record_edge_write(transaction_id, edge_id)
    }

    /// Records a node write **with coarse label fan-out**.
    ///
    /// Records `EntityId::Node(node_id)` write **and** a coarse
    /// `EntityId::Label(L)` write for each `L` in `labels`.  The coarse
    /// label write is the phantom guard: a concurrent escalated
    /// `Label(L)` reader (e.g. from `MATCH (n:L)` scan escalation in
    /// GE3) will form an rw-antidependency with this write.
    ///
    /// Default implementation falls back to the entity-only
    /// `record_node_write` (no label fan-out) for backward compatibility
    /// with non-engine trackers.  The engine bridge overrides this to
    /// call [`TransactionManager::record_node_write`].
    ///
    /// # Errors
    ///
    /// Returns `Err` on any write-write conflict (first-writer-wins).
    fn record_node_write_with_labels(
        &self,
        transaction_id: TransactionId,
        node_id: NodeId,
        _labels: &[LabelId],
    ) -> Result<(), OperatorError> {
        self.record_node_write(transaction_id, node_id)
    }

    /// Records an edge write **with coarse rel-type fan-out**.
    ///
    /// Records `EntityId::Edge(edge_id)` write **and**
    /// `EntityId::RelType(rel_type)` write.  Symmetric with
    /// [`record_node_write_with_labels`](Self::record_node_write_with_labels):
    /// a concurrent escalated `RelType(T)` reader conflicts with any
    /// writer of an edge of that type.
    ///
    /// Default implementation falls back to the entity-only
    /// `record_edge_write`.  The engine bridge overrides this.
    ///
    /// # Errors
    ///
    /// Returns `Err` on any write-write conflict.
    fn record_edge_write_with_type(
        &self,
        transaction_id: TransactionId,
        edge_id: EdgeId,
        _rel_type: EdgeTypeId,
    ) -> Result<(), OperatorError> {
        self.record_edge_write(transaction_id, edge_id)
    }

    /// Records **only** the coarse `Label(L)` phantom write(s) for `labels`,
    /// WITHOUT recording the fine `Node` entity write.
    ///
    /// Used by the transactional property-write paths (`SET n.p` / `REMOVE n.p`):
    /// the fine `Node(n)` write was already recorded by the operator (under its
    /// property tag) or is completed at commit time, so re-recording it here as an
    /// entity-level (`None`) write would be a wildcard that defeats Property
    /// granularity. We only need the coarse `Label(L)` guard so an escalated
    /// `Label(L)` reader (whose fine `Node(n)` entry was dropped) still forms the
    /// rw-antidependency.
    ///
    /// `key` is the written property name; the engine bridge computes
    /// `Some(prop_tag(key))` and passes it as the coarse write's tag so that an
    /// escalated `(Label(L), Some(x))` reader keeps disjoint-property concurrency:
    /// an `x`-write conflicts, a `y`-write (with `y != x`) does not. A structural
    /// escalated reader `(Label(L), None)` still catches every coarse write via the
    /// `None`-wildcard in `prop_compatible`.
    ///
    /// Default implementation is a no-op: non-engine trackers (and SI/RC, where no
    /// tracker is registered) do not track coarse keys.
    ///
    /// # Errors
    ///
    /// Returns `Err` on any write-write conflict.
    fn record_node_labels_write(
        &self,
        _transaction_id: TransactionId,
        _labels: &[LabelId],
        _key: &str,
    ) -> Result<(), OperatorError> {
        Ok(())
    }

    /// Records **only** the coarse `RelType(T)` phantom write for an edge, WITHOUT
    /// recording the fine `Edge` entity write. Edge mirror of
    /// [`record_node_labels_write`](Self::record_node_labels_write); `key` carries
    /// the written property name for the same disjoint-property knob.
    ///
    /// Default implementation is a no-op.
    ///
    /// # Errors
    ///
    /// Returns `Err` on any write-write conflict.
    fn record_edge_type_write(
        &self,
        _transaction_id: TransactionId,
        _rel_type: EdgeTypeId,
        _key: &str,
    ) -> Result<(), OperatorError> {
        Ok(())
    }

    /// Records that `transaction_id` wrote to the `(label, property)` text index
    /// identified by `index_key` (format: `"label:property"`). Coarse index-write
    /// recording for anti-phantom SSI.
    ///
    /// Default impl is a no-op; overridden by the engine bridge
    /// (`TransactionWriteTracker`) to call
    /// `manager.record_write(tx, EntityId::Index(…), None)` so a concurrent
    /// Serializable text search can form the rw-antidependency edge.
    ///
    /// This is intentionally infallible: the index-entity write is a
    /// coarse SSI signal, not a first-writer-wins entity lock (two
    /// concurrent indexed SETs to different *nodes* in the same index
    /// are not a W-W conflict). The W-W check is for the node entity itself.
    fn record_index_write(&self, _transaction_id: TransactionId, _index_key: &str) {}
}

/// Type alias for a shared write tracker.
pub type SharedWriteTracker = Arc<dyn WriteTracker>;

/// Trait for recording read operations during query execution (Serializable only).
///
/// Bridges `grafeo-core` read operators with `grafeo-engine`'s `TransactionManager`
/// (which tracks read-sets for serializable conflict detection). Lives in `grafeo-core`
/// to avoid a circular dependency, mirroring [`WriteTracker`]. Only attached for
/// Serializable transactions, so SnapshotIsolation / ReadCommitted pay nothing.
pub trait ReadTracker: Send + Sync {
    /// Records that `transaction_id` read node `node_id` (at its snapshot).
    fn record_node_read(&self, transaction_id: TransactionId, node_id: NodeId);
    /// Records that `transaction_id` read edge `edge_id` (at its snapshot).
    fn record_edge_read(&self, transaction_id: TransactionId, edge_id: EdgeId);

    /// Records that `transaction_id` read property `_key` of node `node_id`.
    ///
    /// Default impl ignores the property and records an entity-level read — overridden
    /// by the engine bridge to record a property tag under property-level granularity.
    fn record_node_property_read(
        &self,
        transaction_id: TransactionId,
        node_id: NodeId,
        _key: &str,
    ) {
        self.record_node_read(transaction_id, node_id);
    }

    /// Records that `transaction_id` read property `_key` of edge `edge_id`.
    ///
    /// Default impl ignores the property and records an entity-level read — overridden
    /// by the engine bridge to record a property tag under property-level granularity.
    fn record_edge_property_read(
        &self,
        transaction_id: TransactionId,
        edge_id: EdgeId,
        _key: &str,
    ) {
        self.record_edge_read(transaction_id, edge_id);
    }

    /// Records that `transaction_id` read property `key` of node `node_id`, carrying
    /// the node's label set for escalation-aware routing.
    ///
    /// The engine bridge intersects `labels` with the labels the transaction has
    /// already scanned (present in the predicate escalation buckets), then routes
    /// the read under each matched label's predicate bucket so it can escalate to a
    /// coarse `(Label(L), Some(prop_tag))` key.  Non-matching labels are ignored;
    /// empty intersection falls back to a plain fine entity read.
    ///
    /// Default implementation ignores `labels` and delegates to
    /// [`record_node_property_read`](Self::record_node_property_read), so
    /// non-engine trackers remain correct.
    fn record_node_property_read_in_labels(
        &self,
        transaction_id: TransactionId,
        node_id: NodeId,
        key: &str,
        _labels: &[LabelId],
    ) {
        self.record_node_property_read(transaction_id, node_id, key);
    }

    /// Records that `transaction_id` read property `key` of edge `edge_id`, carrying
    /// the edge's relationship type for escalation-aware routing.
    ///
    /// The engine bridge checks whether the transaction has already scanned `rel`
    /// (present in the predicate escalation buckets) and, if so, routes the read
    /// under the `RelType(rel)` predicate bucket so it can escalate to a coarse
    /// `(RelType(T), Some(prop_tag))` key.  No match falls back to a plain fine read.
    ///
    /// Default implementation ignores `rel` and delegates to
    /// [`record_edge_property_read`](Self::record_edge_property_read), so
    /// non-engine trackers remain correct.
    fn record_edge_property_read_in_rel(
        &self,
        transaction_id: TransactionId,
        edge_id: EdgeId,
        key: &str,
        _rel: Option<EdgeTypeId>,
    ) {
        self.record_edge_property_read(transaction_id, edge_id, key);
    }

    /// Records that `transaction_id` executed a text search against the
    /// `(label, property)` text index identified by `index_key` (format:
    /// `"label:property"`). Coarse predicate-read recording for anti-phantom SSI.
    ///
    /// Default impl is a no-op; overridden by the engine bridge
    /// (`TransactionReadTracker`) to call
    /// `manager.record_read(tx, EntityId::Index(…), None)` for Serializable
    /// transactions, recording the read in the SSI read-registry so a concurrent
    /// indexed SET can form the rw-antidependency edge.
    fn record_index_read(&self, _transaction_id: TransactionId, _index_key: &str) {}

    /// Records that `transaction_id` read `node_id` as part of a label scan for
    /// `label_id` (e.g. `MATCH (n:L)`). Enables read-set escalation: the engine
    /// bridge accumulates fine `Node` reads under the `Label(L)` predicate bucket
    /// and promotes the bucket to a single coarse `EntityId::Label(L)` key once
    /// the escalation threshold is exceeded.
    ///
    /// Default implementation falls back to `record_node_read` (fine, no
    /// predicate), so non-engine trackers remain correct.
    fn record_read_node_in_label(
        &self,
        transaction_id: TransactionId,
        node_id: NodeId,
        _label_id: LabelId,
    ) {
        self.record_node_read(transaction_id, node_id);
    }

    /// Records that `transaction_id` read `edge_id` as part of a relationship-type
    /// scan for `rel_type` (e.g. `MATCH ()-[:T]->()`). Symmetric with
    /// [`record_read_node_in_label`](Self::record_read_node_in_label): enables
    /// escalation to a coarse `EntityId::RelType(T)` key.
    ///
    /// Default implementation falls back to `record_edge_read` (fine, no
    /// predicate), so non-engine trackers remain correct.
    fn record_read_edge_in_rel_type(
        &self,
        transaction_id: TransactionId,
        edge_id: EdgeId,
        _rel_type: EdgeTypeId,
    ) {
        self.record_edge_read(transaction_id, edge_id);
    }

    /// Structural node read carrying the node's intrinsic labels, so a
    /// materialization (e.g. `RETURN n`) of a node under an already-escalated
    /// label short-circuits instead of re-adding a fine entry.
    ///
    /// The engine bridge intersects `labels` with the label predicates the
    /// transaction has already escalated; a match routes through
    /// `record_read_in_predicate` (which short-circuits when the coarse
    /// `(Label(L), None)` key is already present) rather than re-inserting a
    /// fine `(Node(n), None)` entry.  Empty intersection falls back to a fine
    /// structural read via [`record_node_read`](Self::record_node_read).
    ///
    /// Default implementation ignores `labels` and delegates to
    /// [`record_node_read`](Self::record_node_read), so non-engine trackers
    /// remain correct.
    fn record_node_read_in_labels(
        &self,
        transaction_id: TransactionId,
        node_id: NodeId,
        _labels: &[LabelId],
    ) {
        self.record_node_read(transaction_id, node_id);
    }
}

/// Type alias for a shared read tracker.
pub type SharedReadTracker = Arc<dyn ReadTracker>;

/// Result of executing an operator.
pub type OperatorResult = Result<Option<DataChunk>, OperatorError>;

// ============================================================================
// Factorized Data Traits
// ============================================================================

/// Trait for data that can be in factorized or flat form.
///
/// This provides a common interface for operators that need to handle both
/// representations without caring which is used. Inspired by LadybugDB's
/// unified data model.
///
/// # Example
///
/// ```rust
/// use grafeo_core::execution::operators::FactorizedData;
///
/// fn process_data(data: &dyn FactorizedData) {
///     if data.is_factorized() {
///         // Handle factorized path
///         let chunk = data.as_factorized().unwrap();
///         // ... use factorized chunk directly
///     } else {
///         // Handle flat path
///         let chunk = data.flatten();
///         // ... process flat chunk
///     }
/// }
/// ```
pub trait FactorizedData: Send + Sync {
    /// Returns the chunk state (factorization status, cached data).
    fn chunk_state(&self) -> &ChunkState;

    /// Returns the logical row count (considering selection).
    fn logical_row_count(&self) -> usize;

    /// Returns the physical size (actual stored values).
    fn physical_size(&self) -> usize;

    /// Returns true if this data is factorized (multi-level).
    fn is_factorized(&self) -> bool;

    /// Flattens to a DataChunk (materializes if factorized).
    fn flatten(&self) -> DataChunk;

    /// Returns as FactorizedChunk if factorized, None if flat.
    fn as_factorized(&self) -> Option<&FactorizedChunk>;

    /// Returns as DataChunk if flat, None if factorized.
    fn as_flat(&self) -> Option<&DataChunk>;
}

/// Wrapper to treat a flat DataChunk as FactorizedData.
///
/// This enables uniform handling of flat and factorized data in operators.
pub struct FlatDataWrapper {
    chunk: DataChunk,
    state: ChunkState,
}

impl FlatDataWrapper {
    /// Creates a new wrapper around a flat DataChunk.
    #[must_use]
    pub fn new(chunk: DataChunk) -> Self {
        let state = ChunkState::flat(chunk.row_count());
        Self { chunk, state }
    }

    /// Returns the underlying DataChunk.
    #[must_use]
    pub fn into_inner(self) -> DataChunk {
        self.chunk
    }
}

impl FactorizedData for FlatDataWrapper {
    fn chunk_state(&self) -> &ChunkState {
        &self.state
    }

    fn logical_row_count(&self) -> usize {
        self.chunk.row_count()
    }

    fn physical_size(&self) -> usize {
        self.chunk.row_count() * self.chunk.column_count()
    }

    fn is_factorized(&self) -> bool {
        false
    }

    fn flatten(&self) -> DataChunk {
        self.chunk.clone()
    }

    fn as_factorized(&self) -> Option<&FactorizedChunk> {
        None
    }

    fn as_flat(&self) -> Option<&DataChunk> {
        Some(&self.chunk)
    }
}

/// Error during operator execution.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum OperatorError {
    /// Type mismatch during execution.
    #[error("type mismatch: expected {expected}, found {found}")]
    TypeMismatch {
        /// Expected type name.
        expected: String,
        /// Found type name.
        found: String,
    },
    /// Column not found.
    #[error("column not found: {0}")]
    ColumnNotFound(String),
    /// Execution error.
    #[error("execution error: {0}")]
    Execution(String),
    /// Schema constraint violation during a write operation.
    #[error("constraint violation: {0}")]
    ConstraintViolation(String),
    /// Write-write conflict detected (first-writer-wins).
    #[error("write conflict: {0}")]
    WriteConflict(String),
}

/// The core trait for pull-based operators.
///
/// Call [`next()`](Self::next) repeatedly until it returns `None`. Each call
/// returns a batch of rows (a DataChunk) or an error.
pub trait Operator: Send + Sync {
    /// Pulls the next batch of data. Returns `None` when exhausted.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operator encounters a runtime error.
    fn next(&mut self) -> OperatorResult;

    /// Resets to initial state so you can iterate again.
    fn reset(&mut self);

    /// Returns a name for debugging/explain output.
    fn name(&self) -> &'static str;

    /// Converts this boxed operator into `Box<dyn Any>` for type-based dispatch.
    ///
    /// Used by the pipeline converter to decompose pull-based operator trees
    /// into push-based pipelines via downcasting to concrete types.
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::vector::ValueVector;
    use grafeo_common::types::LogicalType;
    use std::sync::Mutex;

    // ── ReadTracker: object-safety + call recording ──────────────────────────

    /// Spy implementation backed by `Arc<Mutex<Vec>>` so the test can inspect
    /// recorded calls after the tracker has been type-erased to `dyn ReadTracker`.
    struct SpyReadTracker {
        node_reads: Arc<Mutex<Vec<(TransactionId, NodeId)>>>,
        edge_reads: Arc<Mutex<Vec<(TransactionId, EdgeId)>>>,
    }

    impl ReadTracker for SpyReadTracker {
        fn record_node_read(&self, transaction_id: TransactionId, node_id: NodeId) {
            self.node_reads
                .lock()
                .unwrap()
                .push((transaction_id, node_id));
        }

        fn record_edge_read(&self, transaction_id: TransactionId, edge_id: EdgeId) {
            self.edge_reads
                .lock()
                .unwrap()
                .push((transaction_id, edge_id));
        }
    }

    #[test]
    fn test_read_tracker_object_safe_and_records_calls() {
        // Shared state: the test keeps Arc clones; the spy also holds one each.
        let node_reads: Arc<Mutex<Vec<(TransactionId, NodeId)>>> = Arc::new(Mutex::new(Vec::new()));
        let edge_reads: Arc<Mutex<Vec<(TransactionId, EdgeId)>>> = Arc::new(Mutex::new(Vec::new()));

        let spy = SpyReadTracker {
            node_reads: Arc::clone(&node_reads),
            edge_reads: Arc::clone(&edge_reads),
        };

        // Verify object-safety: must be usable as `SharedReadTracker`.
        let t: SharedReadTracker = Arc::new(spy);

        let tx1 = TransactionId::new(1);
        let tx2 = TransactionId::new(2);
        let n42 = NodeId::new(42);
        let n99 = NodeId::new(99);
        let e7 = EdgeId::new(7);

        t.record_node_read(tx1, n42);
        t.record_node_read(tx2, n99);
        t.record_edge_read(tx1, e7);

        // Inspect via the test's own Arc clones — no downcast needed.
        let nr = node_reads.lock().unwrap();
        assert_eq!(nr.len(), 2);
        assert_eq!(nr[0], (tx1, n42));
        assert_eq!(nr[1], (tx2, n99));
        drop(nr);

        let er = edge_reads.lock().unwrap();
        assert_eq!(er.len(), 1);
        assert_eq!(er[0], (tx1, e7));
    }

    fn create_test_chunk() -> DataChunk {
        let mut col = ValueVector::with_type(LogicalType::Int64);
        col.push_int64(1);
        col.push_int64(2);
        col.push_int64(3);
        DataChunk::new(vec![col])
    }

    #[test]
    fn test_flat_data_wrapper_new() {
        let chunk = create_test_chunk();
        let wrapper = FlatDataWrapper::new(chunk);

        assert!(!wrapper.is_factorized());
        assert_eq!(wrapper.logical_row_count(), 3);
    }

    #[test]
    fn test_flat_data_wrapper_into_inner() {
        let chunk = create_test_chunk();
        let wrapper = FlatDataWrapper::new(chunk);

        let inner = wrapper.into_inner();
        assert_eq!(inner.row_count(), 3);
    }

    #[test]
    fn test_flat_data_wrapper_chunk_state() {
        let chunk = create_test_chunk();
        let wrapper = FlatDataWrapper::new(chunk);

        let state = wrapper.chunk_state();
        assert!(state.is_flat());
        assert_eq!(state.logical_row_count(), 3);
    }

    #[test]
    fn test_flat_data_wrapper_physical_size() {
        let mut col1 = ValueVector::with_type(LogicalType::Int64);
        col1.push_int64(1);
        col1.push_int64(2);

        let mut col2 = ValueVector::with_type(LogicalType::String);
        col2.push_string("a");
        col2.push_string("b");

        let chunk = DataChunk::new(vec![col1, col2]);
        let wrapper = FlatDataWrapper::new(chunk);

        // 2 rows * 2 columns = 4
        assert_eq!(wrapper.physical_size(), 4);
    }

    #[test]
    fn test_flat_data_wrapper_flatten() {
        let chunk = create_test_chunk();
        let wrapper = FlatDataWrapper::new(chunk);

        let flattened = wrapper.flatten();
        assert_eq!(flattened.row_count(), 3);
        assert_eq!(flattened.column(0).unwrap().get_int64(0), Some(1));
    }

    #[test]
    fn test_flat_data_wrapper_as_factorized() {
        let chunk = create_test_chunk();
        let wrapper = FlatDataWrapper::new(chunk);

        assert!(wrapper.as_factorized().is_none());
    }

    #[test]
    fn test_flat_data_wrapper_as_flat() {
        let chunk = create_test_chunk();
        let wrapper = FlatDataWrapper::new(chunk);

        let flat = wrapper.as_flat();
        assert!(flat.is_some());
        assert_eq!(flat.unwrap().row_count(), 3);
    }

    #[test]
    fn test_operator_error_type_mismatch() {
        let err = OperatorError::TypeMismatch {
            expected: "Int64".to_string(),
            found: "String".to_string(),
        };

        let msg = format!("{err}");
        assert!(msg.contains("type mismatch"));
        assert!(msg.contains("Int64"));
        assert!(msg.contains("String"));
    }

    #[test]
    fn test_operator_error_column_not_found() {
        let err = OperatorError::ColumnNotFound("missing_col".to_string());

        let msg = format!("{err}");
        assert!(msg.contains("column not found"));
        assert!(msg.contains("missing_col"));
    }

    #[test]
    fn test_operator_error_execution() {
        let err = OperatorError::Execution("something went wrong".to_string());

        let msg = format!("{err}");
        assert!(msg.contains("execution error"));
        assert!(msg.contains("something went wrong"));
    }

    #[test]
    fn test_operator_error_debug() {
        let err = OperatorError::TypeMismatch {
            expected: "Int64".to_string(),
            found: "String".to_string(),
        };

        let debug = format!("{err:?}");
        assert!(debug.contains("TypeMismatch"));
    }

    #[test]
    fn test_operator_error_clone() {
        let err1 = OperatorError::ColumnNotFound("col".to_string());
        let err2 = err1.clone();

        assert_eq!(format!("{err1}"), format!("{err2}"));
    }
}
