//! Physical operators that actually execute queries.
//!
//! These are the building blocks of query execution. The optimizer picks which
//! operators to use and how to wire them together.
//!
//! **Graph operators:**
//! - [`ScanOperator`] - Read nodes/edges from storage
//! - [`ExpandOperator`] - Traverse edges (the core of graph queries)
//! - [`VariableLengthExpandOperator`] - Variable-length and shortest path searches
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
pub(crate) mod distinct_state;
mod expand;
mod factorized_aggregate;
mod factorized_expand;
mod factorized_filter;
mod factorized_project;
mod filter;
mod flatten;
mod horizontal_aggregate;
mod join;
mod join_sip;
mod leapfrog_expand;
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
pub use distinct_state::{AccountedSemanticKey, encode_accounted_semantic_key};
pub use expand::ExpandOperator;
pub use factorized_aggregate::{
    FactorizedAggregate, FactorizedAggregateOperator, FactorizedOperator,
};
pub use factorized_expand::{
    ExpandStep, FactorizedExpandChain, FactorizedExpandOperator, FactorizedResult,
    LazyFactorizedChainOperator, SipTarget,
};
pub use factorized_filter::{
    AndPredicate, ColumnPredicate, CompareOp as FactorizedCompareOp, FactorizedFilterOperator,
    FactorizedPredicate, OrPredicate, PropertyPredicate,
};
pub use factorized_project::{FactorizedColumnSpec, FactorizedProjectOperator};
pub use filter::{
    BinaryFilterOp, ExpressionPredicate, FilterExpression, FilterOperator, LazyValue,
    ListPredicateKind, Predicate, PredicateAdapter, SessionContext, UnaryFilterOp,
};
pub use flatten::{FlattenMode, FlattenOperator, UnflattenOperator};
pub use horizontal_aggregate::{EntityKind, HorizontalAggregateOperator};
pub use join::{
    EqualityCondition, HashJoinOperator, HashKey, JoinCondition, JoinType, NestedLoopJoinOperator,
};
pub use join_sip::JoinSipExpandOperator;
pub use leapfrog_expand::{
    LeapfrogExpandOperator, LeapfrogExpandSpec, TriangleCountOperator, count_directed_triangles,
    count_leapfrog_triangles, count_nested_loop_triangles, gallop_seek, intersect_count,
    intersect_sorted,
};
pub use leapfrog_join::LeapfrogJoinOperator;
pub use limit::{DrainOperator, LimitOperator, LimitSkipOperator, SkipOperator};
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
pub use single_row::{EmptyOperator, NodeListOperator, SingleRowOperator};
pub use sort::{
    AccountedValueComparator, NullOrder, SemanticComparisonError, SortDirection, SortKey,
    SortOperator,
};
pub use top_k::TopKOperator;
pub use union::UnionOperator;
pub use unwind::UnwindOperator;
pub use variable_length_expand::{
    PathMode as ExecutionPathMode, PathSearch as ExecutionPathSearch, VariableLengthExpandOperator,
};
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
    /// call `TransactionManager::record_node_write`.
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

    /// Records the coarse logical predicate write for a node-label name.
    ///
    /// Structural creates, deletes, and label membership changes call this in
    /// addition to the numeric label fan-out. The name key protects qualified
    /// scans even when the label did not exist when the scan began.
    ///
    /// Default implementation is a no-op for non-engine trackers.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the engine transaction is no longer active.
    fn record_label_name_predicate_write(
        &self,
        _transaction_id: TransactionId,
        _label: &str,
    ) -> Result<(), OperatorError> {
        Ok(())
    }

    /// Records the coarse logical predicate write for a relationship-type name.
    /// Relationship creates and deletes call this in addition to the numeric
    /// relationship-type fan-out.
    ///
    /// Default implementation is a no-op for non-engine trackers.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the engine transaction is no longer active.
    fn record_rel_type_name_predicate_write(
        &self,
        _transaction_id: TransactionId,
        _rel_type: &str,
    ) -> Result<(), OperatorError> {
        Ok(())
    }

    /// Records the coarse native-LPG dataset predicate write.
    ///
    /// This explicit hook is for structural mutations that cannot use the
    /// normal node/edge fan-out path, such as a compact base-only tombstone.
    /// It must not be used for property-only writes.
    ///
    /// Default implementation is a no-op for non-engine trackers.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the engine transaction is no longer active.
    fn record_lpg_dataset_write(
        &self,
        _transaction_id: TransactionId,
    ) -> Result<(), OperatorError> {
        Ok(())
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

    /// Records a node-property predicate write, independently of index lifetime.
    /// This is a coarse SSI signal, not a first-writer-wins lock. Its typed
    /// identity is separate from text/vector logical index names.
    fn record_property_index_write(&self, _transaction_id: TransactionId, _property: &str) {}
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

    /// Records a complete label predicate by its logical name.
    ///
    /// Qualified scans call this before catalog lookup and enumeration, so an
    /// absent, empty, or small label set still receives a Serializable SIREAD.
    /// The numeric hook below remains independently useful for compact fine-read
    /// escalation once the name is interned.
    fn record_label_name_predicate_read(&self, _transaction_id: TransactionId, _label: &str) {}

    /// Records a complete relationship-type predicate by its logical name.
    /// This is the relationship mirror of
    /// [`record_label_name_predicate_read`](Self::record_label_name_predicate_read).
    fn record_rel_type_name_predicate_read(&self, _transaction_id: TransactionId, _rel_type: &str) {
    }

    /// Records the complete node-label predicate read by a label scan.
    ///
    /// Unlike [`record_read_node_in_label`](Self::record_read_node_in_label),
    /// this is a gap/range read and must be called once before enumeration,
    /// including when the scan returns no rows. The engine bridge records a
    /// coarse `EntityId::Label` SIREAD; non-engine trackers may ignore it.
    fn record_label_predicate_read(&self, _transaction_id: TransactionId, _label_id: LabelId) {}

    /// Records the complete relationship-type predicate read by a typed
    /// traversal, including an empty result.
    fn record_rel_type_predicate_read(
        &self,
        _transaction_id: TransactionId,
        _rel_type: EdgeTypeId,
    ) {
    }

    /// Records an unqualified structural scan of the native LPG dataset.
    ///
    /// This is the conservative anti-phantom key for unlabeled node scans and
    /// untyped edge traversals. Fine entity reads are still recorded separately.
    fn record_lpg_dataset_read(&self, _transaction_id: TransactionId) {}

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

    /// Records the property predicate of an indexed lookup, including an empty
    /// result. Property names have their own identity domain, so unrelated
    /// properties remain independent at property conflict granularity.
    fn record_property_index_read(&self, _transaction_id: TransactionId, _property: &str) {}

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

/// One object-safe step in pull-to-push pipeline decomposition.
///
/// Operators implemented outside `grafeo-core` can return [`Self::Unary`]
/// without teaching the core converter their concrete type or debug name.
/// A boundary remains a live pull operator and becomes the pipeline source.
#[non_exhaustive]
pub enum OperatorPipelineDecomposition {
    /// This operator is an explicit pull/source boundary.
    Boundary(Box<dyn Operator>),
    /// This unary pull operator supplied its child and equivalent push stage.
    Unary {
        /// The child to decompose next.
        child: Box<dyn Operator>,
        /// The push stage corresponding to the consumed pull wrapper.
        push: Box<dyn super::pipeline::PushOperator>,
    },
}

impl OperatorPipelineDecomposition {
    /// Creates an explicit pull/source boundary.
    #[must_use]
    pub fn boundary(operator: Box<dyn Operator>) -> Self {
        Self::Boundary(operator)
    }

    /// Creates a decomposed unary stage.
    #[must_use]
    pub fn unary(child: Box<dyn Operator>, push: Box<dyn super::pipeline::PushOperator>) -> Self {
        Self::Unary { child, push }
    }
}

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

/// Copy/clone-only classification for a pre-admitted move-only failure.
///
/// The classification preserves the public error category without exposing
/// or detaching the diagnostic payload from the accounting authority retained
/// by [`OperatorError::AccountedFailure`].
#[cfg(feature = "spill")]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AccountedFailureClassification {
    /// A consumer cannot accept the move-only accounted transport.
    UnsupportedAccountedTransport,
    /// The operation observed an incompatible value type.
    TypeMismatch,
    /// A referenced column does not exist.
    ColumnNotFound,
    /// A schema constraint rejected the operation.
    ConstraintViolation,
    /// First-writer-wins rejected a concurrent write.
    WriteConflict,
    /// A checked resident-memory transition failed.
    ResidentMemory(grafeo_common::memory::buffer::MemoryGrantError),
    /// A standard resident allocation was refused.
    ResidentAllocation,
    /// A pinned exact-vector allocation was refused.
    ResidentExactVectorAllocation(crate::execution::ResidentCapacityError),
    /// The query's spill-storage quota or backing device is full.
    StorageFull,
    /// A resident-container invariant failed.
    ResidentInvariant,
    /// Cooperative cancellation or deadline expiry won.
    QueryCancelled(crate::execution::QueryCancellationError),
    /// An execution, corruption, unsupported-input, or internal failure.
    Execution,
}

/// Error during operator execution.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum OperatorError {
    /// A pipeline stage cannot preserve a move-only accounted chunk envelope.
    #[error("accounted chunk transport is unsupported by {consumer}")]
    UnsupportedAccountedTransport {
        /// Static identity of the rejecting operator or sink.
        consumer: &'static str,
    },
    /// Cloneable ownership of a pre-admitted move-only execution failure.
    ///
    /// The wrapper deliberately exposes no `Error::source`; callers that own
    /// the concrete internal type inspect it through the accounted handle.
    #[cfg(feature = "spill")]
    #[error("{0}")]
    AccountedFailure(grafeo_common::memory::buffer::AccountedError),
    /// Pre-admitted move-only failure with an inspectable public category.
    ///
    /// This additive variant preserves the original tuple variant's source
    /// compatibility while qualified paths opt into structured classification.
    #[cfg(feature = "spill")]
    #[error("{authority}")]
    ClassifiedAccountedFailure {
        /// Public classification minted while the private primary was still
        /// inspectable; secondaries never overwrite it.
        classification: AccountedFailureClassification,
        /// Cloneable, non-detachable owner of the original diagnostic and all
        /// accounting authority needed to destroy it safely.
        authority: grafeo_common::memory::buffer::AccountedError,
    },
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
    /// A checked resident-memory grant transition failed during execution.
    #[error("resident-memory grant failed: {0}")]
    ResidentMemory(#[from] grafeo_common::memory::buffer::MemoryGrantError),
    /// An operator failure augmented with secondary operation context.
    #[error("{source}; {context}")]
    Context {
        /// Structured primary error, retained without reclassification.
        #[source]
        source: Box<Self>,
        /// Additional failure context.
        context: String,
    },
    /// The allocator rejected a fallible resident-container reservation.
    #[error("resident-memory allocation failed: {0}")]
    ResidentAllocation(String),
    /// The allocator rejected a named resident container without requiring
    /// another allocation to preserve the structured failure.
    #[error("resident-memory allocation failed for {container}: {source}")]
    ResidentContainerAllocation {
        /// Static identity of the container whose reservation failed.
        container: &'static str,
        /// Original allocator or address-space refusal.
        #[source]
        source: std::collections::TryReserveError,
    },
    /// The pinned exact vector allocator rejected a fallible reservation.
    #[cfg(feature = "spill")]
    #[error("resident exact-vector allocation failed: {0}")]
    ResidentExactVectorAllocation(#[source] crate::execution::ResidentCapacityError),
    /// The pinned native partition table rejected a fallible reservation.
    #[error("resident-memory allocation failed for native partition map: {source}")]
    ResidentNativeMapAllocation {
        /// Original table allocation or address-space refusal.
        #[source]
        source: crate::execution::NativeMapAllocationError,
    },
    /// Native partition allocation failed and its provisional grant could not
    /// be reconciled cleanly.
    #[error(
        "resident-memory allocation failed for native partition map: {source}; grant rollback also failed: {rollback}"
    )]
    ResidentNativeMapAllocationWithRollback {
        /// Original table allocation or address-space refusal.
        #[source]
        source: crate::execution::NativeMapAllocationError,
        /// Secondary accounting failure.
        rollback: grafeo_common::memory::buffer::MemoryGrantError,
    },
    /// A version-pinned resident container violated its allocation contract.
    #[error("resident-memory invariant failed for {container}: {message}")]
    ResidentContainerInvariant {
        /// Static container identity.
        container: &'static str,
        /// Allocation-free invariant detail.
        message: &'static str,
    },
    /// A resident-container invariant failed and its grant rollback also
    /// exposed an accounting failure.
    #[error(
        "resident-memory invariant failed for {container}: {message}; grant rollback also failed: {rollback}"
    )]
    ResidentContainerInvariantWithRollback {
        /// Static container identity.
        container: &'static str,
        /// Allocation-free invariant detail.
        message: &'static str,
        /// Secondary accounting failure.
        rollback: grafeo_common::memory::buffer::MemoryGrantError,
    },
    /// A query-scoped memory or disk storage boundary was exhausted.
    #[error("query storage limit reached: {0}")]
    StorageFull(String),
    /// Cooperative execution stopped by explicit cancellation or deadline.
    #[error(transparent)]
    QueryCancelled(#[from] crate::execution::QueryCancellationError),
}

impl OperatorError {
    #[cfg(feature = "spill")]
    pub(crate) fn from_spill_io_error(error: std::io::Error) -> Self {
        match error.kind() {
            std::io::ErrorKind::QuotaExceeded | std::io::ErrorKind::StorageFull => {
                Self::StorageFull(error.to_string())
            }
            _ => Self::Execution(error.to_string()),
        }
    }

    /// Recovers a clone of the complete downstream primary from an exact
    /// final-failure envelope when one is present.
    ///
    /// Recovery is intentionally deferred until a caller consumes the public
    /// error, so the allocation-free failure path does not clone heap-bearing
    /// strings or boxes. The accounted envelope remains attached as context
    /// until this conversion has copied its combined diagnostic.
    #[cfg(feature = "spill")]
    #[must_use]
    pub fn recover_accounted_primary(self) -> Self {
        let Self::ClassifiedAccountedFailure {
            classification,
            authority,
        } = self
        else {
            return self;
        };
        let Some(primary) =
            crate::execution::spill::recover_exact_final_operator_primary(&authority)
        else {
            return Self::ClassifiedAccountedFailure {
                classification,
                authority,
            };
        };
        let context = authority.to_string();
        drop(authority);
        primary.with_context(format!("exact accounted terminal envelope: {context}"))
    }

    /// Adds human-readable context without flattening the structured primary.
    #[must_use]
    pub fn with_context(self, context: impl Into<String>) -> Self {
        Self::Context {
            source: Box::new(self),
            context: context.into(),
        }
    }
}

/// The core trait for pull-based operators.
///
/// Call [`next()`](Self::next) repeatedly until it returns `None`. Each call
/// returns a batch of rows (a DataChunk) or an error.
/// With an explicit cooperative execution checkpoint, a successful call may
/// be followed immediately by cancellation. Externally visible effects must
/// remain rollbackable until the execution owner fences success.
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

    /// Converts this boxed operator into `Box<dyn Any>` for legacy type-based
    /// inspection outside pipeline conversion.
    ///
    /// Pipeline admission uses [`Self::decompose_pipeline_with_resources`] and does not infer
    /// concrete types from [`Self::name`] or downcast engine-local operators.
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send>;

    /// Installs one query's immutable resource context through this physical
    /// subtree.
    ///
    /// Unary wrappers override this method and forward the exact same context
    /// to their child. Blocking operators may retain a clone; clones preserve
    /// query identity, memory pool, spill manager, and cancellation token.
    ///
    /// # Errors
    ///
    /// Returns a structured grant or scoped-registration failure. The default
    /// is an explicit leaf/boundary and performs no work.
    fn install_resource_context(
        &mut self,
        _resources: &super::memory::QueryResourceContext,
    ) -> Result<(), super::memory::QueryResourceContextError> {
        Ok(())
    }

    /// Consumes one resource-qualified pull wrapper into a push stage, or
    /// returns an explicit pull/source boundary.
    ///
    /// External operators override this method when construction needs the
    /// query context. The default preserves unknown and engine-local operators
    /// as boundaries. It remains object-safe even for `dyn Operator`; no name
    /// matching or converter-side downcast is involved.
    ///
    /// Pipeline decomposition requires the query's resource context:
    ///
    /// ```compile_fail,E0599
    /// use grafeo_core::execution::operators::Operator;
    ///
    /// fn unqualified(operator: Box<dyn Operator>) {
    ///     let _ = operator.decompose_pipeline();
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a structured resource failure from push-stage construction.
    fn decompose_pipeline_with_resources(
        self: Box<Self>,
        _resources: &super::memory::QueryResourceContext,
    ) -> Result<OperatorPipelineDecomposition, super::memory::QueryResourceContextError>
    where
        Self: 'static,
    {
        Ok(OperatorPipelineDecomposition::Boundary(Box::new(
            OpaquePipelineBoundary { inner: self },
        )))
    }

    /// Live factorized producer, if this operator keeps multi-hop data unflat.
    ///
    /// The executor flattens once when collecting [`crate::execution::DataChunk`]
    /// rows. Default is `None` (flat `next()`).
    fn as_factorized_mut(&mut self) -> Option<&mut dyn FactorizedOperator> {
        None
    }
}

/// Sized erasure wrapper used by the default consuming hook.
///
/// `Box<Self>` cannot be coerced directly to `Box<dyn Operator>` in a default
/// method because `Self` may be unsized. Boxing that pointer inside this sized
/// delegating wrapper preserves the original operator and its dynamic vtable.
struct OpaquePipelineBoundary<T: Operator + ?Sized> {
    inner: Box<T>,
}

impl<T: Operator + ?Sized + 'static> Operator for OpaquePipelineBoundary<T> {
    fn next(&mut self) -> OperatorResult {
        self.inner.next()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self.inner.into_any()
    }

    fn install_resource_context(
        &mut self,
        resources: &super::memory::QueryResourceContext,
    ) -> Result<(), super::memory::QueryResourceContextError> {
        self.inner.install_resource_context(resources)
    }

    fn decompose_pipeline_with_resources(
        self: Box<Self>,
        _resources: &super::memory::QueryResourceContext,
    ) -> Result<OperatorPipelineDecomposition, super::memory::QueryResourceContextError> {
        Ok(OperatorPipelineDecomposition::Boundary(self))
    }

    fn as_factorized_mut(&mut self) -> Option<&mut dyn FactorizedOperator> {
        self.inner.as_factorized_mut()
    }
}

/// Takes a pull operator that is a factorized producer.
///
/// # Errors
///
/// Returns the original operator when it is not a factorized node.
pub fn take_factorized(
    op: Box<dyn Operator>,
) -> Result<Box<dyn FactorizedOperator>, Box<dyn Operator>> {
    match op.name() {
        "LazyFactorizedChain" | "FactorizedFilter" | "FactorizedProject" | "FactorizedExpand" => {}
        _ => return Err(op),
    }
    let name = op.name();
    let any = op.into_any();
    match name {
        "LazyFactorizedChain" => Ok(any
            .downcast::<LazyFactorizedChainOperator>()
            .unwrap_or_else(|_| unreachable!("name matched LazyFactorizedChain"))),
        "FactorizedFilter" => Ok(any
            .downcast::<FactorizedFilterOperator>()
            .unwrap_or_else(|_| unreachable!("name matched FactorizedFilter"))),
        "FactorizedProject" => Ok(any
            .downcast::<FactorizedProjectOperator>()
            .unwrap_or_else(|_| unreachable!("name matched FactorizedProject"))),
        "FactorizedExpand" => Ok(any
            .downcast::<FactorizedExpandOperator>()
            .unwrap_or_else(|_| unreachable!("name matched FactorizedExpand"))),
        _ => unreachable!("checked name above"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::vector::ValueVector;
    #[cfg(feature = "spill")]
    use grafeo_common::memory::buffer::{
        AccountedErrorPublisher, BufferManager, BufferManagerConfig, MemoryRegion,
    };
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

    #[cfg(feature = "spill")]
    #[test]
    fn reader_accounted_error_operator_clone_shares_owner_without_account_delta() {
        let required = AccountedErrorPublisher::<std::io::Error>::required_bytes();
        let mut config = BufferManagerConfig::with_budget(required);
        config.soft_limit_fraction = 1.0;
        config.evict_limit_fraction = 1.0;
        config.hard_limit_fraction = 1.0;
        let manager = BufferManager::new(config);
        let grant = manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .expect("dedicated zero grant");
        let shared = AccountedErrorPublisher::try_new(grant)
            .expect("exact control-block admission")
            .publish(std::io::Error::from(std::io::ErrorKind::PermissionDenied));
        let first = OperatorError::AccountedFailure(shared);
        let second = first.clone();

        let (
            OperatorError::AccountedFailure(first_owner),
            OperatorError::AccountedFailure(second_owner),
        ) = (&first, &second)
        else {
            panic!("both operator errors must retain the accounted owner");
        };
        assert!(first_owner.ptr_eq(second_owner));
        assert!(std::error::Error::source(&first).is_none());
        assert_eq!(
            first_owner.inspect::<std::io::Error, _>(std::io::Error::kind),
            Some(std::io::ErrorKind::PermissionDenied)
        );
        assert_eq!(manager.allocated(), required);

        drop(first);
        assert_eq!(manager.allocated(), required);
        drop(second);
        assert_eq!(manager.allocated(), 0);
    }
}
