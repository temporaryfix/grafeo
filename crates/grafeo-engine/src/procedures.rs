//! Built-in procedure registry for CALL statement execution.
//!
//! Provides the [`Procedure`] trait and [`BuiltinProcedures`] registry used by
//! all supported query languages (GQL, Cypher, SQL/PGQ) to dispatch
//! `CALL grafeo.<name>(...) [YIELD ...]` statements.
//!
//! # Unified dispatch
//!
//! Three kinds of procedures share the [`Procedure`] trait:
//!
//! - Graph algorithms (PageRank, BFS, Dijkstra, ...): adapted from
//!   `GraphAlgorithm` via `GraphAlgorithmProcedure`.
//! - Catalog introspection (`db.labels`, `db.relationshipTypes`,
//!   `db.propertyKeys`): implemented directly.
//! - Vector and text search (`grafeo.search.vector`, `grafeo.search.mmr`,
//!   `grafeo.search.text`): implemented directly, require
//!   [`ProcedureContext::lpg_store`].

use std::sync::Arc;

#[cfg(feature = "algos")]
use grafeo_adapters::plugins::algorithms::{
    ArticulationPointsAlgorithm, BellmanFordAlgorithm, BetweennessCentralityAlgorithm,
    BfsAlgorithm, BridgesAlgorithm, ClosenessCentralityAlgorithm, ClusteringCoefficientAlgorithm,
    ConnectedComponentsAlgorithm, DegreeCentralityAlgorithm, DfsAlgorithm, DijkstraAlgorithm,
    FloydWarshallAlgorithm, GraphAlgorithm, KCoreAlgorithm, KruskalAlgorithm,
    LabelPropagationAlgorithm, LouvainAlgorithm, MaxFlowAlgorithm, MinCostFlowAlgorithm,
    PageRankAlgorithm, PrimAlgorithm, SsspAlgorithm, StronglyConnectedComponentsAlgorithm,
    TopologicalSortAlgorithm,
};
use grafeo_adapters::plugins::{AlgorithmResult, ParameterDef, Parameters};
use grafeo_common::types::{EpochId, TransactionId, Value};
use grafeo_common::utils::error::Result;
#[cfg(not(feature = "gql"))]
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::LpgStore;
#[cfg(all(feature = "lpg", any(feature = "text-index", feature = "vector-index")))]
use grafeo_core::graph::lpg::encode_index_key;
use grafeo_core::graph::{Direction, GraphStoreSearch};
use hashbrown::HashMap;

use crate::query::plan::LogicalExpression;
#[cfg(not(feature = "gql"))]
use crate::query::plan::LogicalOperator;

/// The externally observable effect of a procedure invocation.
///
/// The default is deliberately fail-closed: a newly added procedure must opt
/// in to [`ReadOnly`](Self::ReadOnly) before the Session may execute it with
/// read authority or outside a write transaction.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ProcedureEffect {
    /// The procedure only observes graph or catalog state.
    ReadOnly,
    /// The procedure may mutate state, including through another procedure.
    MayWrite,
}

/// Unified interface for built-in procedures callable via `CALL`.
///
/// Subsumes graph algorithms (through `GraphAlgorithmProcedure`), catalog
/// introspection, and vector/text search procedures. The planner dispatches
/// every `CALL grafeo.<name>(...)` through this trait, so adding a new
/// procedure reduces to implementing [`Procedure`] and registering it in
/// [`BuiltinProcedures::new`].
pub trait Procedure: Send + Sync {
    /// Returns the procedure name (without the `grafeo.` prefix).
    fn name(&self) -> &str;

    /// Returns a short description for `grafeo.procedures()` listings.
    fn description(&self) -> &str;

    /// Returns parameter definitions (name, type, required, default).
    fn parameters(&self) -> &[ParameterDef];

    /// Returns the canonical output column names in order.
    fn output_columns(&self) -> Vec<String>;

    /// Classifies the procedure's externally visible effect.
    ///
    /// Implementations are write-capable until explicitly audited and marked
    /// read-only. This prevents a future builtin from silently bypassing
    /// Session authorization and transaction framing.
    fn effect(&self) -> ProcedureEffect {
        ProcedureEffect::MayWrite
    }

    /// Returns `true` if this procedure is safe to run under Serializable
    /// isolation.
    ///
    /// A procedure is Serializable-safe when its reads can be made
    /// snapshot-consistent and recorded into the SSI read-set so that
    /// concurrent conflicting writes will abort it correctly.
    ///
    /// - Graph algorithms (`GraphAlgorithmProcedure`): `true` — the executor
    ///   wraps the store in a [`SnapshotView`][grafeo_core::graph::SnapshotView]
    ///   so all reads are pinned to the transaction epoch and recorded.
    /// - Catalog introspection (`labels`, `relationshipTypes`, `propertyKeys`):
    ///   `true` — each procedure derives metadata from snapshot-visible
    ///   entities. The enclosing dataset scan and entity reads are recorded for
    ///   Serializable anti-phantom/conflict detection.
    /// - Vector/text search procedures (`search.vector`, `search.mmr`,
    ///   `search.text`): `true` — all three route through `*_visible` APIs
    ///   that record index reads for SSI conflict detection when
    ///   `snapshot_epoch`/`snapshot_tx` are set. The default `false` applies
    ///   only to hypothetical future procedures that have not yet been made
    ///   snapshot-aware.
    fn serializable_safe(&self) -> bool {
        false
    }

    /// Executes the procedure against the supplied context and parameters.
    ///
    /// # Errors
    ///
    /// Returns an error when required parameters are missing, types are
    /// wrong, or the backing operation (graph algorithm, search index, or
    /// catalog lookup) fails.
    fn execute(&self, ctx: &ProcedureContext<'_>, params: &Parameters) -> Result<AlgorithmResult>;
}

/// Runtime context passed to [`Procedure::execute`].
///
/// Graph algorithms and catalog introspection need only [`ProcedureContext::store`].
/// Vector and text search procedures additionally require
/// [`ProcedureContext::lpg_store`] to reach the HNSW and BM25 indexes owned
/// by the LPG store.
///
/// Under Serializable isolation `snapshot_epoch` and `snapshot_tx` are set so
/// that text search procedures can call `search_text_visible` (snapshot-pinned
/// + SSI read recording) instead of the committed-latest `index.read().search`.
pub struct ProcedureContext<'a> {
    /// Read-only graph store, sufficient for graph algorithms and catalog
    /// introspection (labels, edge types, property keys).
    pub store: &'a dyn GraphStoreSearch,

    /// Concrete LPG store reference when available, used by search procedures
    /// to reach vector and text indexes. `None` when the active backend is
    /// not an LPG store (e.g., pure RDF) or in contexts that do not need it.
    #[cfg(feature = "lpg")]
    pub lpg_store: Option<&'a LpgStore>,

    /// Snapshot epoch for Serializable transactions.  When `Some`, text search
    /// procedures must call `search_text_visible` rather than the raw index.
    pub snapshot_epoch: Option<EpochId>,

    /// Transaction ID paired with `snapshot_epoch` for SSI read recording.
    pub snapshot_tx: Option<TransactionId>,
}

impl<'a> ProcedureContext<'a> {
    /// Creates a context with only a graph store available.
    #[must_use]
    pub fn new(store: &'a dyn GraphStoreSearch) -> Self {
        Self {
            store,
            #[cfg(feature = "lpg")]
            lpg_store: None,
            snapshot_epoch: None,
            snapshot_tx: None,
        }
    }

    /// Creates a context with a graph store and LPG store available.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn with_lpg_store(store: &'a dyn GraphStoreSearch, lpg_store: &'a LpgStore) -> Self {
        Self {
            store,
            lpg_store: Some(lpg_store),
            snapshot_epoch: None,
            snapshot_tx: None,
        }
    }

    /// Creates a context with a graph store, LPG store, and Serializable
    /// snapshot context (epoch + transaction ID) for SSI-recording text search.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn with_lpg_store_and_snapshot(
        store: &'a dyn GraphStoreSearch,
        lpg_store: &'a LpgStore,
        epoch: EpochId,
        tx: TransactionId,
    ) -> Self {
        Self {
            store,
            lpg_store: Some(lpg_store),
            snapshot_epoch: Some(epoch),
            snapshot_tx: Some(tx),
        }
    }
}

/// Adapter that presents a `GraphAlgorithm` as a [`Procedure`].
///
/// Canonical output column names (e.g., `score` instead of `pagerank`) are
/// captured at construction time via [`canonical_output_columns`].
#[cfg(feature = "algos")]
pub struct GraphAlgorithmProcedure {
    inner: Arc<dyn GraphAlgorithm>,
    output_columns: Vec<String>,
}

#[cfg(feature = "algos")]
impl GraphAlgorithmProcedure {
    /// Wraps a graph algorithm as a procedure.
    pub fn new(algorithm: Arc<dyn GraphAlgorithm>) -> Self {
        let output_columns = canonical_output_columns(algorithm.as_ref());
        Self {
            inner: algorithm,
            output_columns,
        }
    }
}

#[cfg(feature = "algos")]
impl Procedure for GraphAlgorithmProcedure {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn parameters(&self) -> &[ParameterDef] {
        self.inner.parameters()
    }

    fn output_columns(&self) -> Vec<String> {
        self.output_columns.clone()
    }

    fn effect(&self) -> ProcedureEffect {
        ProcedureEffect::ReadOnly
    }

    /// Graph algorithms are Serializable-safe: the executor wraps the store in
    /// a `SnapshotView` so reads are pinned to the transaction epoch and
    /// recorded for SSI conflict detection.
    fn serializable_safe(&self) -> bool {
        true
    }

    fn execute(&self, ctx: &ProcedureContext<'_>, params: &Parameters) -> Result<AlgorithmResult> {
        self.inner.execute(ctx.store, params)
    }
}

// ---------------------------------------------------------------------------
// Catalog introspection procedures
// ---------------------------------------------------------------------------

/// `db.labels` / `grafeo.labels`: lists every label present in the graph.
struct LabelsProcedure;

impl Procedure for LabelsProcedure {
    fn name(&self) -> &str {
        "labels"
    }

    fn description(&self) -> &str {
        "Lists every node label present in the graph"
    }

    fn parameters(&self) -> &[ParameterDef] {
        &[]
    }

    fn output_columns(&self) -> Vec<String> {
        vec!["label".into()]
    }

    fn effect(&self) -> ProcedureEffect {
        ProcedureEffect::ReadOnly
    }

    /// Snapshot enumeration records the dataset scan and visible entity reads.
    fn serializable_safe(&self) -> bool {
        true
    }

    fn execute(&self, ctx: &ProcedureContext<'_>, _params: &Parameters) -> Result<AlgorithmResult> {
        let mut labels = std::collections::BTreeSet::new();
        for id in ctx.store.node_ids() {
            if let Some(node) = ctx.store.get_node(id) {
                labels.extend(node.labels.into_iter().map(|label| label.to_string()));
            }
        }
        let mut result = AlgorithmResult::new(vec!["label".into()]);
        for label in labels {
            result.rows.push(vec![Value::String(label.into())]);
        }
        Ok(result)
    }
}

/// `db.relationshipTypes` / `grafeo.relationshipTypes`: lists every edge type.
struct RelationshipTypesProcedure;

impl Procedure for RelationshipTypesProcedure {
    fn name(&self) -> &str {
        "relationshipTypes"
    }

    fn description(&self) -> &str {
        "Lists every edge type present in the graph"
    }

    fn parameters(&self) -> &[ParameterDef] {
        &[]
    }

    fn output_columns(&self) -> Vec<String> {
        vec!["relationshipType".into()]
    }

    fn effect(&self) -> ProcedureEffect {
        ProcedureEffect::ReadOnly
    }

    /// Snapshot enumeration records the dataset scan and visible entity reads.
    fn serializable_safe(&self) -> bool {
        true
    }

    fn execute(&self, ctx: &ProcedureContext<'_>, _params: &Parameters) -> Result<AlgorithmResult> {
        let mut edge_types = std::collections::BTreeSet::new();
        for edge in visible_edges(ctx.store) {
            edge_types.insert(edge.edge_type.to_string());
        }
        let mut result = AlgorithmResult::new(vec!["relationshipType".into()]);
        for t in edge_types {
            result.rows.push(vec![Value::String(t.into())]);
        }
        Ok(result)
    }
}

/// `db.propertyKeys` / `grafeo.propertyKeys`: lists every property key.
struct PropertyKeysProcedure;

impl Procedure for PropertyKeysProcedure {
    fn name(&self) -> &str {
        "propertyKeys"
    }

    fn description(&self) -> &str {
        "Lists every property key present in the graph"
    }

    fn parameters(&self) -> &[ParameterDef] {
        &[]
    }

    fn output_columns(&self) -> Vec<String> {
        vec!["propertyKey".into()]
    }

    fn effect(&self) -> ProcedureEffect {
        ProcedureEffect::ReadOnly
    }

    /// Snapshot enumeration records the dataset scan and visible entity reads.
    fn serializable_safe(&self) -> bool {
        true
    }

    fn execute(&self, ctx: &ProcedureContext<'_>, _params: &Parameters) -> Result<AlgorithmResult> {
        let mut keys = std::collections::BTreeSet::new();
        for id in ctx.store.node_ids() {
            if let Some(node) = ctx.store.get_node(id) {
                keys.extend(node.properties.iter().map(|(key, _)| key.to_string()));
            }
        }
        for edge in visible_edges(ctx.store) {
            keys.extend(edge.properties.iter().map(|(key, _)| key.to_string()));
        }
        let mut result = AlgorithmResult::new(vec!["propertyKey".into()]);
        for key in keys {
            result.rows.push(vec![Value::String(key.into())]);
        }
        Ok(result)
    }
}

/// Enumerates each snapshot-visible edge exactly once through the graph-store
/// interface. `SnapshotView::node_ids` records the complete dataset predicate;
/// `edges_from` and `get_edge` preserve exact MVCC visibility and SSI reads.
fn visible_edges(store: &dyn GraphStoreSearch) -> Vec<grafeo_core::graph::lpg::Edge> {
    let mut ids = std::collections::BTreeSet::new();
    for node in store.node_ids() {
        ids.extend(
            store
                .edges_from(node, Direction::Outgoing)
                .into_iter()
                .map(|(_, edge)| edge),
        );
    }
    ids.into_iter()
        .filter_map(|edge| store.get_edge(edge))
        .collect()
}

// ---------------------------------------------------------------------------
// Vector / text search procedures
// ---------------------------------------------------------------------------

#[cfg(all(feature = "lpg", feature = "vector-index"))]
fn require_lpg_store<'a>(ctx: &ProcedureContext<'a>, proc_name: &str) -> Result<&'a LpgStore> {
    ctx.lpg_store.ok_or_else(|| {
        grafeo_common::utils::error::Error::Internal(format!(
            "{proc_name} requires an LPG store. Ensure the session is backed by an LPG database \
             (not a pure RDF store or external custom store)."
        ))
    })
}

#[cfg(all(feature = "lpg", feature = "vector-index"))]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "vector indexes store f32; GQL list literals arrive as f64/i64 and must narrow"
)]
fn coerce_params_to_vector(params: &Parameters, key: &str) -> Result<Vec<f32>> {
    if let Some(list) = params.get_list(key) {
        let mut out = Vec::with_capacity(list.len());
        for v in list {
            match v {
                Value::Float64(f) => out.push(*f as f32),
                Value::Int64(i) => out.push(*i as f32),
                other => {
                    return Err(grafeo_common::utils::error::Error::Internal(format!(
                        "Expected numeric list for vector parameter '{key}', found {other:?}"
                    )));
                }
            }
        }
        return Ok(out);
    }
    Err(grafeo_common::utils::error::Error::Internal(format!(
        "Missing required vector parameter '{key}'"
    )))
}

/// Converts a `k` parameter (signed, user-supplied) to an unsigned limit.
/// Negative values clamp to 0 so they can't silently flip via cast.
#[cfg(all(feature = "lpg", any(feature = "vector-index", feature = "text-index")))]
fn k_limit(params: &Parameters, default: i64) -> usize {
    usize::try_from(params.get_int("k").unwrap_or(default).max(0)).unwrap_or(0)
}

/// Converts a NodeId into a `Value::Int64`.
/// Using `cast_signed` keeps the bit pattern; ids above `i64::MAX` would
/// become negative, but the ID space is bounded well below that in practice.
#[cfg(all(feature = "lpg", any(feature = "vector-index", feature = "text-index")))]
fn node_id_to_value(node_id: grafeo_common::types::NodeId) -> Value {
    Value::Int64(node_id.as_u64().cast_signed())
}

/// `grafeo.search.vector(label, property, query, k)`:
/// k-nearest-neighbors via the HNSW index.
#[cfg(all(feature = "lpg", feature = "vector-index"))]
struct SearchVectorProcedure {
    parameters: Vec<ParameterDef>,
}

#[cfg(all(feature = "lpg", feature = "vector-index"))]
impl SearchVectorProcedure {
    fn new() -> Self {
        use grafeo_adapters::plugins::ParameterType;
        Self {
            parameters: vec![
                ParameterDef {
                    name: "label".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Node label that was vector-indexed".into(),
                },
                ParameterDef {
                    name: "property".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Property name holding the embedding".into(),
                },
                ParameterDef {
                    name: "query".into(),
                    param_type: ParameterType::List,
                    required: true,
                    default: None,
                    description: "Query vector as a list of floats".into(),
                },
                ParameterDef {
                    name: "k".into(),
                    param_type: ParameterType::Integer,
                    required: false,
                    default: Some("10".into()),
                    description: "Number of nearest neighbors to return".into(),
                },
            ],
        }
    }
}

#[cfg(all(feature = "lpg", feature = "vector-index"))]
impl Procedure for SearchVectorProcedure {
    fn name(&self) -> &str {
        "search.vector"
    }

    fn description(&self) -> &str {
        "Approximate k-nearest neighbors via the HNSW vector index"
    }

    fn parameters(&self) -> &[ParameterDef] {
        &self.parameters
    }

    fn output_columns(&self) -> Vec<String> {
        vec!["node_id".into(), "distance".into()]
    }

    fn effect(&self) -> ProcedureEffect {
        ProcedureEffect::ReadOnly
    }

    /// `grafeo.search.vector` is Serializable-safe: the executor supplies
    /// `(epoch, tx)` via `ProcedureContext::snapshot_epoch` /
    /// `::snapshot_tx`, and `execute` routes through
    /// `LpgStore::search_vector_visible` which records the index read for
    /// SSI conflict detection.
    fn serializable_safe(&self) -> bool {
        true
    }

    fn execute(&self, ctx: &ProcedureContext<'_>, params: &Parameters) -> Result<AlgorithmResult> {
        use grafeo_core::index::vector::{PropertyVectorAccessor, VectorAccessorKind};

        let lpg = require_lpg_store(ctx, "CALL grafeo.search.vector")?;
        let label = params.get_string("label").ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "CALL grafeo.search.vector: missing required parameter 'label'".into(),
            )
        })?;
        let property = params.get_string("property").ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "CALL grafeo.search.vector: missing required parameter 'property'".into(),
            )
        })?;
        if lpg.get_vector_index(label, property).is_none() {
            return Err(grafeo_common::utils::error::Error::Internal(format!(
                "No vector index found for :{label}({property}). Call CREATE VECTOR INDEX first."
            )));
        }
        let query = coerce_params_to_vector(params, "query")?;
        let k = k_limit(params, 10);

        // Every planned CALL has a snapshot context. Route through the logical
        // store so a Layered backend scores both compact and overlay vectors;
        // the LPG owner alone may contain no vectors after compaction. The
        // visible search retains snapshot/own-write filtering, SSI index reads,
        // and its existing visibility-safe beam floor.
        if let (Some(epoch), Some(tx)) = (ctx.snapshot_epoch, ctx.snapshot_tx) {
            let results = ctx
                .store
                .vector_search_visible(label, property, &query, k, epoch, tx);

            let mut result = AlgorithmResult::new(vec!["node_id".into(), "distance".into()]);
            for (node_id, distance) in results {
                result
                    .rows
                    .push(vec![node_id_to_value(node_id), Value::Float64(distance)]);
            }
            return Ok(result);
        }

        // SI / RC: committed-latest, no SSI recording.
        let index = lpg.get_vector_index(label, property).ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(format!(
                "No vector index found for :{label}({property}). Call CREATE VECTOR INDEX first."
            ))
        })?;

        let accessor = VectorAccessorKind::Property(PropertyVectorAccessor::new(
            ctx.store as &dyn grafeo_core::graph::GraphStore,
            property,
        ));
        let results = index.search(&query, k, &accessor);

        let mut result = AlgorithmResult::new(vec!["node_id".into(), "distance".into()]);
        for (node_id, distance) in results {
            result.rows.push(vec![
                node_id_to_value(node_id),
                Value::Float64(f64::from(distance)),
            ]);
        }
        Ok(result)
    }
}

/// `grafeo.search.mmr(label, property, query, k, fetch_k, lambda)`:
/// Maximal-Marginal-Relevance re-ranking of HNSW candidates for diverse top-k.
#[cfg(all(feature = "lpg", feature = "vector-index"))]
struct SearchMmrProcedure {
    parameters: Vec<ParameterDef>,
}

#[cfg(all(feature = "lpg", feature = "vector-index"))]
impl SearchMmrProcedure {
    fn new() -> Self {
        use grafeo_adapters::plugins::ParameterType;
        Self {
            parameters: vec![
                ParameterDef {
                    name: "label".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Node label that was vector-indexed".into(),
                },
                ParameterDef {
                    name: "property".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Property name holding the embedding".into(),
                },
                ParameterDef {
                    name: "query".into(),
                    param_type: ParameterType::List,
                    required: true,
                    default: None,
                    description: "Query vector as a list of floats".into(),
                },
                ParameterDef {
                    name: "k".into(),
                    param_type: ParameterType::Integer,
                    required: false,
                    default: Some("10".into()),
                    description: "Number of diverse results to return".into(),
                },
                ParameterDef {
                    name: "fetch_k".into(),
                    param_type: ParameterType::Integer,
                    required: false,
                    default: None,
                    description: "Initial HNSW candidate count (default: 4*k)".into(),
                },
                ParameterDef {
                    name: "lambda".into(),
                    param_type: ParameterType::Float,
                    required: false,
                    default: Some("0.5".into()),
                    description: "Relevance vs diversity in [0, 1]".into(),
                },
            ],
        }
    }
}

#[cfg(all(feature = "lpg", feature = "vector-index"))]
impl Procedure for SearchMmrProcedure {
    fn name(&self) -> &str {
        "search.mmr"
    }

    fn description(&self) -> &str {
        "Maximal-Marginal-Relevance re-ranking for diverse nearest neighbors"
    }

    fn parameters(&self) -> &[ParameterDef] {
        &self.parameters
    }

    fn output_columns(&self) -> Vec<String> {
        vec!["node_id".into(), "distance".into()]
    }

    fn effect(&self) -> ProcedureEffect {
        ProcedureEffect::ReadOnly
    }

    /// `grafeo.search.mmr` is Serializable-safe: the executor supplies
    /// `(epoch, tx)` via `ProcedureContext::snapshot_epoch` /
    /// `::snapshot_tx`, and `execute` routes the initial candidate fetch
    /// through `LpgStore::search_vector_visible` which records the index
    /// read for SSI conflict detection and applies snapshot visibility.
    /// The MMR re-rank is deterministic post-processing and does not add
    /// additional reads beyond those already recorded by the candidate fetch.
    fn serializable_safe(&self) -> bool {
        true
    }

    fn execute(&self, ctx: &ProcedureContext<'_>, params: &Parameters) -> Result<AlgorithmResult> {
        use grafeo_core::index::vector::{
            PropertyVectorAccessor, VectorAccessor, VectorAccessorKind, mmr_select,
        };

        let lpg = require_lpg_store(ctx, "CALL grafeo.search.mmr")?;
        let label = params.get_string("label").ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "CALL grafeo.search.mmr: missing required parameter 'label'".into(),
            )
        })?;
        let property = params.get_string("property").ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "CALL grafeo.search.mmr: missing required parameter 'property'".into(),
            )
        })?;
        if lpg.get_vector_index(label, property).is_none() {
            return Err(grafeo_common::utils::error::Error::Internal(format!(
                "No vector index found for :{label}({property}). Call CREATE VECTOR INDEX first."
            )));
        }
        let query = coerce_params_to_vector(params, "query")?;
        let k = k_limit(params, 10);
        let fetch_k = params
            .get_int("fetch_k")
            .map_or(k.saturating_mul(4).max(k), |v| {
                usize::try_from(v.max(0)).unwrap_or(0)
            });
        #[allow(
            clippy::cast_possible_truncation,
            reason = "lambda is a unit-interval weighting factor; f32 precision is sufficient"
        )]
        let lambda = params.get_float("lambda").unwrap_or(0.5) as f32;

        // Under Serializable isolation `snapshot_epoch` and `snapshot_tx` are
        // set by the executor.  Route through `search_vector_visible` to merge
        // the per-tx write delta, pin candidates to the snapshot epoch, and
        // record the index read in the SSI read-set so concurrent indexed-SET
        // writes form an rw-antidependency edge.
        //
        // For the MMR re-rank we retrieve each candidate's as-of-epoch vector
        // via `read_node_property_visible` so the pairwise similarity in
        // `mmr_select` is also consistent with the snapshot (same epoch/tx).
        if let (Some(epoch), Some(tx)) = (ctx.snapshot_epoch, ctx.snapshot_tx) {
            let index_key = encode_index_key(label, property);
            let initial = lpg.search_vector_visible(&index_key, &query, fetch_k, epoch, tx);
            if initial.is_empty() {
                return Ok(AlgorithmResult::new(vec![
                    "node_id".into(),
                    "distance".into(),
                ]));
            }

            // Fetch as-of-epoch vectors for MMR pairwise comparison.
            let prop_key = grafeo_common::types::PropertyKey::new(property);
            let candidates: Vec<(grafeo_common::types::NodeId, f32, std::sync::Arc<[f32]>)> =
                initial
                    .into_iter()
                    .filter_map(|(id, dist)| {
                        let val = lpg.read_node_property_visible(id, &prop_key, epoch, Some(tx))?;
                        match val {
                            Value::Vector(v) => Some((id, dist, v)),
                            _ => None,
                        }
                    })
                    .collect();

            if candidates.is_empty() {
                return Ok(AlgorithmResult::new(vec![
                    "node_id".into(),
                    "distance".into(),
                ]));
            }

            let candidate_refs: Vec<(grafeo_common::types::NodeId, f32, &[f32])> = candidates
                .iter()
                .map(|(id, dist, vec)| (*id, *dist, vec.as_ref()))
                .collect();

            // Determine the distance metric from the committed index (fallback: Cosine).
            let metric = lpg
                .get_vector_index_by_key(&index_key)
                .as_ref()
                .map_or(grafeo_core::index::vector::DistanceMetric::Cosine, |idx| {
                    idx.config().metric
                });
            let selected = mmr_select(&query, &candidate_refs, k, lambda, metric);

            let mut result = AlgorithmResult::new(vec!["node_id".into(), "distance".into()]);
            for (node_id, distance) in selected {
                result.rows.push(vec![
                    node_id_to_value(node_id),
                    Value::Float64(f64::from(distance)),
                ]);
            }
            return Ok(result);
        }

        // SI / RC: committed-latest, no SSI recording.
        let index = lpg.get_vector_index(label, property).ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(format!(
                "No vector index found for :{label}({property}). Call CREATE VECTOR INDEX first."
            ))
        })?;

        let accessor = VectorAccessorKind::Property(PropertyVectorAccessor::new(
            ctx.store as &dyn grafeo_core::graph::GraphStore,
            property,
        ));
        let initial = index.search(&query, fetch_k, &accessor);
        if initial.is_empty() {
            return Ok(AlgorithmResult::new(vec![
                "node_id".into(),
                "distance".into(),
            ]));
        }

        let candidates: Vec<(grafeo_common::types::NodeId, f32, std::sync::Arc<[f32]>)> = initial
            .into_iter()
            .filter_map(|(id, dist)| accessor.get_vector(id).map(|v| (id, dist, v)))
            .collect();
        let candidate_refs: Vec<(grafeo_common::types::NodeId, f32, &[f32])> = candidates
            .iter()
            .map(|(id, dist, vec)| (*id, *dist, vec.as_ref()))
            .collect();

        let metric = index.config().metric;
        let selected = mmr_select(&query, &candidate_refs, k, lambda, metric);

        let mut result = AlgorithmResult::new(vec!["node_id".into(), "distance".into()]);
        for (node_id, distance) in selected {
            result.rows.push(vec![
                node_id_to_value(node_id),
                Value::Float64(f64::from(distance)),
            ]);
        }
        Ok(result)
    }
}

/// `grafeo.search.text(label, property, query, k)`:
/// BM25 full-text search over the inverted index.
#[cfg(all(feature = "lpg", feature = "text-index"))]
struct SearchTextProcedure {
    parameters: Vec<ParameterDef>,
}

#[cfg(all(feature = "lpg", feature = "text-index"))]
impl SearchTextProcedure {
    fn new() -> Self {
        use grafeo_adapters::plugins::ParameterType;
        Self {
            parameters: vec![
                ParameterDef {
                    name: "label".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Node label that was text-indexed".into(),
                },
                ParameterDef {
                    name: "property".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Property holding the indexed text".into(),
                },
                ParameterDef {
                    name: "query".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Text query for BM25 scoring".into(),
                },
                ParameterDef {
                    name: "k".into(),
                    param_type: ParameterType::Integer,
                    required: false,
                    default: Some("10".into()),
                    description: "Number of top results to return".into(),
                },
            ],
        }
    }
}

#[cfg(all(feature = "lpg", feature = "text-index"))]
impl Procedure for SearchTextProcedure {
    fn name(&self) -> &str {
        "search.text"
    }

    fn description(&self) -> &str {
        "BM25 full-text search over an inverted text index"
    }

    fn parameters(&self) -> &[ParameterDef] {
        &self.parameters
    }

    fn output_columns(&self) -> Vec<String> {
        vec!["node_id".into(), "score".into()]
    }

    fn effect(&self) -> ProcedureEffect {
        ProcedureEffect::ReadOnly
    }

    /// `grafeo.search.text` is Serializable-safe: the executor supplies
    /// `(epoch, tx)` via `ProcedureContext::snapshot_epoch` /
    /// `::snapshot_tx`, and `execute` routes through `search_text_visible`
    /// which records the index read for SSI conflict detection.
    fn serializable_safe(&self) -> bool {
        true
    }

    fn execute(&self, ctx: &ProcedureContext<'_>, params: &Parameters) -> Result<AlgorithmResult> {
        let lpg = ctx.lpg_store.ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "CALL grafeo.search.text requires an LPG store".into(),
            )
        })?;
        let label = params.get_string("label").ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "CALL grafeo.search.text: missing required parameter 'label'".into(),
            )
        })?;
        let property = params.get_string("property").ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "CALL grafeo.search.text: missing required parameter 'property'".into(),
            )
        })?;
        if lpg.get_text_index(label, property).is_none() {
            return Err(grafeo_common::utils::error::Error::Internal(format!(
                "No text index found for :{label}({property}). Call CREATE TEXT INDEX first."
            )));
        }
        let query = params.get_string("query").ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "CALL grafeo.search.text: missing required parameter 'query'".into(),
            )
        })?;
        let k = k_limit(params, 10);

        // Under Serializable isolation `snapshot_epoch` and `snapshot_tx` are
        // set by the executor.  Use `search_text_visible` to merge the per-tx
        // write delta, pin results to the snapshot epoch, and record the index
        // read in the SSI read-set so concurrent indexed-SET writes can form an
        // rw-antidependency edge.
        let results = if let (Some(epoch), Some(tx)) = (ctx.snapshot_epoch, ctx.snapshot_tx) {
            let index_key = encode_index_key(label, property);
            lpg.search_text_visible(&index_key, query, k, epoch, tx)?
        } else {
            // SI / RC: committed-latest, no SSI recording.
            let index = lpg.get_text_index(label, property).ok_or_else(|| {
                grafeo_common::utils::error::Error::Internal(format!(
                    "No text index found for :{label}({property}). Call CREATE TEXT INDEX first."
                ))
            })?;
            index.read().search(query, k)
        };

        let mut result = AlgorithmResult::new(vec!["node_id".into(), "score".into()]);
        for (node_id, score) in results {
            result
                .rows
                .push(vec![node_id_to_value(node_id), Value::Float64(score)]);
        }
        Ok(result)
    }
}

/// Registry of built-in procedures.
///
/// Stores procedures behind `Arc<dyn Procedure>` so dispatch goes through a
/// single path regardless of procedure kind (graph algorithm, catalog, or
/// search).
pub struct BuiltinProcedures {
    procedures: HashMap<String, Arc<dyn Procedure>>,
}

impl BuiltinProcedures {
    /// Creates a new registry with all built-in procedures registered.
    #[must_use]
    pub fn new() -> Self {
        let mut procedures: HashMap<String, Arc<dyn Procedure>> = HashMap::new();

        #[cfg(feature = "algos")]
        {
            // Graph algorithms, wrapped via the adapter.
            let mut register_algo = |algo: Arc<dyn GraphAlgorithm>| {
                let proc = Arc::new(GraphAlgorithmProcedure::new(algo));
                procedures.insert(proc.name().to_string(), proc);
            };

            // Centrality
            register_algo(Arc::new(PageRankAlgorithm));
            register_algo(Arc::new(BetweennessCentralityAlgorithm));
            register_algo(Arc::new(ClosenessCentralityAlgorithm));
            register_algo(Arc::new(DegreeCentralityAlgorithm));

            // Traversal
            register_algo(Arc::new(BfsAlgorithm));
            register_algo(Arc::new(DfsAlgorithm));

            // Components
            register_algo(Arc::new(ConnectedComponentsAlgorithm));
            register_algo(Arc::new(StronglyConnectedComponentsAlgorithm));
            register_algo(Arc::new(TopologicalSortAlgorithm));

            // Shortest Path
            register_algo(Arc::new(DijkstraAlgorithm));
            register_algo(Arc::new(SsspAlgorithm));
            register_algo(Arc::new(BellmanFordAlgorithm));
            register_algo(Arc::new(FloydWarshallAlgorithm));

            // Clustering
            register_algo(Arc::new(ClusteringCoefficientAlgorithm));

            // Community
            register_algo(Arc::new(LabelPropagationAlgorithm));
            register_algo(Arc::new(LouvainAlgorithm));

            // MST
            register_algo(Arc::new(KruskalAlgorithm));
            register_algo(Arc::new(PrimAlgorithm));

            // Flow
            register_algo(Arc::new(MaxFlowAlgorithm));
            register_algo(Arc::new(MinCostFlowAlgorithm));

            // Structure
            register_algo(Arc::new(ArticulationPointsAlgorithm));
            register_algo(Arc::new(BridgesAlgorithm));
            register_algo(Arc::new(KCoreAlgorithm));
        }

        // Catalog introspection
        let mut register = |proc: Arc<dyn Procedure>| {
            procedures.insert(proc.name().to_string(), proc);
        };
        register(Arc::new(LabelsProcedure));
        register(Arc::new(RelationshipTypesProcedure));
        register(Arc::new(PropertyKeysProcedure));

        // Search procedures (feature-gated).
        #[cfg(all(feature = "lpg", feature = "vector-index"))]
        {
            register(Arc::new(SearchVectorProcedure::new()));
            register(Arc::new(SearchMmrProcedure::new()));
        }
        #[cfg(all(feature = "lpg", feature = "text-index"))]
        {
            register(Arc::new(SearchTextProcedure::new()));
        }

        Self { procedures }
    }

    /// Resolves a dotted procedure name to its implementation.
    #[must_use]
    pub fn get(&self, name: &[String]) -> Option<Arc<dyn Procedure>> {
        let key = resolve_name(name);
        self.procedures.get(&key).cloned()
    }

    /// Returns metadata for every registered procedure, sorted by name.
    #[must_use]
    pub fn list(&self) -> Vec<ProcedureInfo> {
        let mut result: Vec<ProcedureInfo> = self
            .procedures
            .values()
            .map(|proc| ProcedureInfo {
                name: format!("grafeo.{}", proc.name()),
                description: proc.description().to_string(),
                parameters: proc.parameters().to_vec(),
                output_columns: proc.output_columns(),
            })
            .collect();
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }
}

impl Default for BuiltinProcedures {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns the process-wide immutable builtin procedure registry.
///
/// Resolution and effect analysis share this exact registry so a call cannot
/// be authorized as one implementation and planned as another.
#[must_use]
pub(crate) fn builtin_registry() -> &'static BuiltinProcedures {
    static PROCEDURES: std::sync::OnceLock<BuiltinProcedures> = std::sync::OnceLock::new();
    PROCEDURES.get_or_init(BuiltinProcedures::new)
}

/// Security-relevant effects of builtin calls in one logical plan.
///
/// This classifier deliberately lives outside the GQL-only catalog procedure
/// implementation: SQL/PGQ and any future query language can also produce a
/// `CALL`, and the fail-closed [`Procedure::effect`] default must apply in
/// every supported feature profile.
#[derive(Clone, Copy)]
#[cfg(not(feature = "gql"))]
pub(crate) struct BuiltinProcedureEffects {
    pub(crate) mutates: bool,
    #[cfg(any(
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    pub(crate) contains_call: bool,
    pub(crate) contains_mutating_call: bool,
}

/// Classifies every reachable builtin call without consulting the mutable
/// catalog. Unknown calls are semantic errors, not implicitly read-only.
#[cfg(not(feature = "gql"))]
pub(crate) fn analyze_builtin_procedure_effects(
    root: &LogicalOperator,
) -> Result<BuiltinProcedureEffects> {
    let contains_call = root.contains_procedure_call();
    if !contains_call {
        return Ok(BuiltinProcedureEffects {
            mutates: root.has_mutations(),
            #[cfg(any(
                feature = "cypher",
                feature = "gremlin",
                feature = "graphql",
                feature = "sql-pgq"
            ))]
            contains_call: false,
            contains_mutating_call: false,
        });
    }

    fn visit(operator: &LogicalOperator) -> Result<ProcedureEffect> {
        let mut effect = if operator.has_mutations() {
            ProcedureEffect::MayWrite
        } else {
            ProcedureEffect::ReadOnly
        };
        if let LogicalOperator::CallProcedure(call) = operator {
            effect = effect.max(builtin_call_effect(&call.name)?);
        }
        for child in operator.children() {
            effect = effect.max(visit(child)?);
        }
        Ok(effect)
    }

    let effect = visit(root)?;
    let contains_mutating_call =
        contains_call && calls_contain_effect(root, ProcedureEffect::MayWrite)?;
    Ok(BuiltinProcedureEffects {
        mutates: root.has_mutations() || effect == ProcedureEffect::MayWrite,
        #[cfg(any(
            feature = "cypher",
            feature = "gremlin",
            feature = "graphql",
            feature = "sql-pgq"
        ))]
        contains_call,
        contains_mutating_call,
    })
}

#[cfg(not(feature = "gql"))]
fn calls_contain_effect(root: &LogicalOperator, expected: ProcedureEffect) -> Result<bool> {
    if let LogicalOperator::CallProcedure(call) = root
        && builtin_call_effect(&call.name)? == expected
    {
        return Ok(true);
    }
    for child in root.children() {
        if calls_contain_effect(child, expected)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Resolves one builtin call for effect classification using the same registry
/// as physical planning.
#[cfg(not(feature = "gql"))]
pub(crate) fn builtin_call_effect(name: &[String]) -> Result<ProcedureEffect> {
    if matches!(name, [single] if single == "procedures")
        || matches!(name, [namespace, procedure]
            if namespace.eq_ignore_ascii_case("grafeo") && procedure == "procedures")
    {
        return Ok(ProcedureEffect::ReadOnly);
    }
    builtin_registry()
        .get(name)
        .map(|procedure| procedure.effect())
        .ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!(
                    "unknown procedure '{}'; use CALL grafeo.procedures() to list available procedures",
                    name.join(".")
                ),
            ))
        })
}

/// Metadata about a registered procedure.
pub struct ProcedureInfo {
    /// Qualified name (e.g., `"grafeo.pagerank"`).
    pub name: String,
    /// Description of what the procedure does.
    pub description: String,
    /// Parameter definitions.
    pub parameters: Vec<ParameterDef>,
    /// Output column names.
    pub output_columns: Vec<String>,
}

/// Resolves a dotted procedure name to its lookup key.
///
/// Strips a leading `"grafeo"` or `"db"` namespace segment if present, then
/// joins the remaining parts with `.`. This lets every form resolve to the
/// same registered key:
///
/// - `CALL pagerank()` / `CALL grafeo.pagerank()` → `"pagerank"`
/// - `CALL db.labels()` / `CALL grafeo.labels()` → `"labels"`
/// - `CALL grafeo.search.vector(...)` → `"search.vector"`
fn resolve_name(parts: &[String]) -> String {
    let slice = match parts {
        [ns, rest @ ..]
            if (ns.eq_ignore_ascii_case("grafeo") || ns.eq_ignore_ascii_case("db"))
                && !rest.is_empty() =>
        {
            rest
        }
        _ => parts,
    };
    slice.join(".")
}

/// Returns canonical output column names for the given graph algorithm.
///
/// These user-facing names (e.g., `"score"` instead of `"pagerank"`) must
/// match the column count produced by each algorithm's `execute()`.
#[must_use]
#[cfg(feature = "algos")]
pub fn canonical_output_columns(algo: &dyn GraphAlgorithm) -> Vec<String> {
    match algo.name() {
        "pagerank" => vec!["node_id".into(), "score".into()],
        "betweenness_centrality" | "closeness_centrality" => {
            vec!["node_id".into(), "centrality".into()]
        }
        "degree_centrality" => vec![
            "node_id".into(),
            "in_degree".into(),
            "out_degree".into(),
            "total_degree".into(),
        ],
        "bfs" | "dfs" => vec!["node_id".into(), "depth".into()],
        "connected_components" | "strongly_connected_components" => {
            vec!["node_id".into(), "component_id".into()]
        }
        "topological_sort" => vec!["node_id".into(), "order".into()],
        "dijkstra" | "sssp" => vec!["node_id".into(), "distance".into()],
        "bellman_ford" => vec![
            "node_id".into(),
            "distance".into(),
            "has_negative_cycle".into(),
        ],
        "floyd_warshall" => vec!["source".into(), "target".into(), "distance".into()],
        "clustering_coefficient" => vec![
            "node_id".into(),
            "coefficient".into(),
            "triangle_count".into(),
        ],
        "label_propagation" => vec!["node_id".into(), "community_id".into()],
        "louvain" => vec!["node_id".into(), "community_id".into(), "modularity".into()],
        "kruskal" | "prim" => vec!["source".into(), "target".into(), "weight".into()],
        "max_flow" => vec![
            "source".into(),
            "target".into(),
            "flow".into(),
            "max_flow".into(),
        ],
        "min_cost_max_flow" => vec![
            "source".into(),
            "target".into(),
            "flow".into(),
            "cost".into(),
            "max_flow".into(),
        ],
        "articulation_points" => vec!["node_id".into()],
        "bridges" => vec!["source".into(), "target".into()],
        "kcore" => vec!["node_id".into(), "core_number".into(), "max_core".into()],
        _ => vec!["node_id".into(), "value".into()],
    }
}

/// Converts logical expression arguments into [`Parameters`].
///
/// Supports two patterns:
/// 1. Map literal: `{damping: 0.85, iterations: 20}` → named parameters.
/// 2. Positional args: `(42, 'weight')` → mapped by index to `ParameterDef` names.
pub fn evaluate_arguments(args: &[LogicalExpression], param_defs: &[ParameterDef]) -> Parameters {
    let mut params = Parameters::new();

    if args.len() == 1
        && let LogicalExpression::Map(entries) = &args[0]
    {
        for (key, value_expr) in entries {
            set_param_from_expression(&mut params, key, value_expr);
        }
        return params;
    }

    for (i, arg) in args.iter().enumerate() {
        if let Some(def) = param_defs.get(i) {
            set_param_from_expression(&mut params, &def.name, arg);
        }
    }

    params
}

/// Sets a parameter from a `LogicalExpression` constant value.
fn set_param_from_expression(params: &mut Parameters, name: &str, expr: &LogicalExpression) {
    match expr {
        LogicalExpression::Literal(Value::Int64(v)) => params.set_int(name, *v),
        LogicalExpression::Literal(Value::Float64(v)) => params.set_float(name, *v),
        LogicalExpression::Literal(Value::String(v)) => {
            params.set_string(name, AsRef::<str>::as_ref(v));
        }
        LogicalExpression::Literal(Value::Bool(v)) => params.set_bool(name, *v),
        LogicalExpression::Literal(Value::List(items)) => {
            params.set_list(name, items.iter().cloned().collect());
        }
        LogicalExpression::Literal(Value::Vector(items)) => {
            params.set_list(
                name,
                items
                    .iter()
                    .map(|f| Value::Float64(f64::from(*f)))
                    .collect(),
            );
        }
        LogicalExpression::List(items) => {
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                if let LogicalExpression::Literal(v) = item {
                    values.push(v.clone());
                }
            }
            params.set_list(name, values);
        }
        _ => {}
    }
}

/// Builds a `grafeo.procedures()` result listing all registered procedures.
#[must_use]
pub fn procedures_result(registry: &BuiltinProcedures) -> AlgorithmResult {
    let procedures = registry.list();
    let mut result = AlgorithmResult::new(vec![
        "name".into(),
        "description".into(),
        "parameters".into(),
        "output_columns".into(),
    ]);
    for proc in procedures {
        let param_desc: String = proc
            .parameters
            .iter()
            .map(|p| {
                if p.required {
                    format!("{} ({:?})", p.name, p.param_type)
                } else if let Some(ref default) = p.default {
                    format!("{} ({:?}, default={})", p.name, p.param_type, default)
                } else {
                    format!("{} ({:?}, optional)", p.name, p.param_type)
                }
            })
            .collect::<Vec<_>>()
            .join(", ");

        let columns_desc = proc.output_columns.join(", ");

        result.add_row(vec![
            Value::from(proc.name.as_str()),
            Value::from(proc.description.as_str()),
            Value::from(param_desc.as_str()),
            Value::from(columns_desc.as_str()),
        ]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // serializable_safe classification
    // ---------------------------------------------------------------

    #[cfg(feature = "algos")]
    #[test]
    fn graph_algorithm_procedures_are_serializable_safe() {
        let registry = BuiltinProcedures::new();
        for name in ["pagerank", "connected_components", "dijkstra", "bfs"] {
            let proc = registry
                .get(&[name.to_string()])
                .unwrap_or_else(|| panic!("{name} must be registered"));
            assert!(
                proc.serializable_safe(),
                "{name}: GraphAlgorithmProcedure must be serializable_safe() == true"
            );
        }
    }

    #[test]
    fn introspection_procedures_are_serializable_safe() {
        let registry = BuiltinProcedures::new();
        for name in ["labels", "relationshipTypes", "propertyKeys"] {
            let proc = registry
                .get(&[name.to_string()])
                .unwrap_or_else(|| panic!("{name} must be registered"));
            assert!(
                proc.serializable_safe(),
                "{name}: introspection procedure must be serializable_safe() == true"
            );
        }
    }

    /// All built-in search procedures are Serializable-safe after VI7+MMR
    /// un-guarding.  Any future search procedure that is NOT yet snapshot-aware
    /// must document why and set `serializable_safe() == false` deliberately.
    #[cfg(all(feature = "lpg", feature = "vector-index", feature = "text-index"))]
    #[test]
    fn all_search_procedures_are_serializable_safe() {
        let registry = BuiltinProcedures::new();
        for name in ["search.vector", "search.mmr", "search.text"] {
            let proc = registry
                .get(&[name.to_string()])
                .unwrap_or_else(|| panic!("{name} must be registered"));
            assert!(
                proc.serializable_safe(),
                "{name}: search procedure must be serializable_safe() == true \
                 (VI7+MMR un-guarding: every search procedure routes through \
                 the snapshot-visible API when epoch+tx are set)"
            );
        }
    }

    #[test]
    fn default_serializable_safe_is_false() {
        // The trait default must be false; UnknownAlgo (defined below) doesn't
        // override it, so GraphAlgorithmProcedure wrapping it must still be true
        // (the wrapper overrides), but a bare struct relying on the default must
        // return false.
        struct BareProc;
        impl Procedure for BareProc {
            fn name(&self) -> &str {
                "bare"
            }
            fn description(&self) -> &str {
                "bare"
            }
            fn parameters(&self) -> &[ParameterDef] {
                &[]
            }
            fn output_columns(&self) -> Vec<String> {
                vec![]
            }
            fn execute(
                &self,
                _ctx: &ProcedureContext<'_>,
                _params: &Parameters,
            ) -> Result<AlgorithmResult> {
                Ok(AlgorithmResult::new(vec![]))
            }
        }
        assert!(
            !BareProc.serializable_safe(),
            "default serializable_safe() must be false"
        );
    }

    #[cfg(feature = "algos")]
    #[test]
    fn test_registry_has_all_algorithms() {
        let registry = BuiltinProcedures::new();
        let list = registry.list();
        assert!(
            list.len() >= 22,
            "Expected at least 22 procedures, got {}",
            list.len()
        );
    }

    #[cfg(feature = "algos")]
    #[test]
    fn test_resolve_with_namespace() {
        let registry = BuiltinProcedures::new();
        let name = vec!["grafeo".to_string(), "pagerank".to_string()];
        assert!(registry.get(&name).is_some());
    }

    #[cfg(feature = "algos")]
    #[test]
    fn test_resolve_without_namespace() {
        let registry = BuiltinProcedures::new();
        let name = vec!["pagerank".to_string()];
        assert!(registry.get(&name).is_some());
    }

    #[test]
    fn test_resolve_unknown() {
        let registry = BuiltinProcedures::new();
        let name = vec!["grafeo".to_string(), "nonexistent".to_string()];
        assert!(registry.get(&name).is_none());
    }

    #[test]
    fn test_evaluate_map_arguments() {
        let args = vec![LogicalExpression::Map(vec![
            (
                "damping".to_string(),
                LogicalExpression::Literal(Value::Float64(0.85)),
            ),
            (
                "max_iterations".to_string(),
                LogicalExpression::Literal(Value::Int64(20)),
            ),
        ])];
        let params = evaluate_arguments(&args, &[]);
        assert_eq!(params.get_float("damping"), Some(0.85));
        assert_eq!(params.get_int("max_iterations"), Some(20));
    }

    #[test]
    fn test_evaluate_empty_arguments() {
        let params = evaluate_arguments(&[], &[]);
        assert_eq!(params.get_float("damping"), None);
    }

    #[cfg(feature = "algos")]
    #[test]
    fn test_adapter_forwards_metadata() {
        let algo: Arc<dyn GraphAlgorithm> = Arc::new(PageRankAlgorithm);
        let proc = GraphAlgorithmProcedure::new(algo);
        assert_eq!(proc.name(), "pagerank");
        assert_eq!(proc.output_columns(), vec!["node_id", "score"]);
        assert!(!proc.parameters().is_empty());
    }

    // ---------------------------------------------------------------
    // resolve_name: every call site of CALL must dispatch through here,
    // so the namespace-stripping contract is load-bearing.
    // ---------------------------------------------------------------

    #[test]
    fn resolve_name_strips_db_prefix() {
        let registry = BuiltinProcedures::new();
        let name = vec!["db".to_string(), "labels".to_string()];
        assert!(
            registry.get(&name).is_some(),
            "db.labels must route to the labels procedure"
        );
    }

    #[test]
    fn resolve_name_is_case_insensitive_on_prefix() {
        // `CALL DB.Labels()` should route the same as `CALL db.labels()`.
        // (The leaf name itself is case-sensitive by design; we only
        // normalise the namespace token.)
        assert_eq!(
            resolve_name(&["DB".to_string(), "labels".to_string()]),
            "labels"
        );
        assert_eq!(
            resolve_name(&["Grafeo".to_string(), "pagerank".to_string()]),
            "pagerank"
        );
    }

    #[test]
    fn resolve_name_preserves_dotted_paths() {
        // Search procedures live under a dotted sub-namespace
        // (grafeo.search.vector). The resolver must NOT strip the
        // second segment just because the first is "grafeo".
        assert_eq!(
            resolve_name(&[
                "grafeo".to_string(),
                "search".to_string(),
                "vector".to_string()
            ]),
            "search.vector"
        );
    }

    #[test]
    fn resolve_name_leaves_bare_prefix_alone() {
        // `CALL grafeo()` has no child: do not strip, or the resolver
        // would produce the empty string and silently collide with any
        // procedure registered under "".
        assert_eq!(resolve_name(&["grafeo".to_string()]), "grafeo");
        assert_eq!(resolve_name(&["db".to_string()]), "db");
    }

    #[test]
    fn resolve_name_empty_input_returns_empty() {
        assert_eq!(resolve_name(&[]), "");
    }

    // ---------------------------------------------------------------
    // canonical_output_columns: the planner relies on every wrapped
    // algorithm reporting stable, user-facing column names. A rename
    // here silently breaks RETURN / YIELD lists.
    // ---------------------------------------------------------------

    #[cfg(feature = "algos")]
    #[test]
    fn canonical_output_columns_for_shortest_path_algorithms() {
        let dijkstra: Arc<dyn GraphAlgorithm> = Arc::new(DijkstraAlgorithm);
        assert_eq!(
            canonical_output_columns(dijkstra.as_ref()),
            vec!["node_id", "distance"]
        );

        let sssp: Arc<dyn GraphAlgorithm> = Arc::new(SsspAlgorithm);
        assert_eq!(
            canonical_output_columns(sssp.as_ref()),
            vec!["node_id", "distance"]
        );

        let bellman: Arc<dyn GraphAlgorithm> = Arc::new(BellmanFordAlgorithm);
        assert_eq!(
            canonical_output_columns(bellman.as_ref()),
            vec!["node_id", "distance", "has_negative_cycle"]
        );
    }

    #[cfg(feature = "algos")]
    #[test]
    fn canonical_output_columns_for_community_algorithms() {
        let louvain: Arc<dyn GraphAlgorithm> = Arc::new(LouvainAlgorithm);
        assert_eq!(
            canonical_output_columns(louvain.as_ref()),
            vec!["node_id", "community_id", "modularity"]
        );

        let lpa: Arc<dyn GraphAlgorithm> = Arc::new(LabelPropagationAlgorithm);
        assert_eq!(
            canonical_output_columns(lpa.as_ref()),
            vec!["node_id", "community_id"]
        );
    }

    #[cfg(feature = "algos")]
    #[test]
    fn canonical_output_columns_for_mst_algorithms() {
        let kruskal: Arc<dyn GraphAlgorithm> = Arc::new(KruskalAlgorithm);
        assert_eq!(
            canonical_output_columns(kruskal.as_ref()),
            vec!["source", "target", "weight"]
        );
        let prim: Arc<dyn GraphAlgorithm> = Arc::new(PrimAlgorithm);
        assert_eq!(
            canonical_output_columns(prim.as_ref()),
            vec!["source", "target", "weight"]
        );
    }

    #[cfg(feature = "algos")]
    #[test]
    fn canonical_output_columns_for_structural_algorithms() {
        let ap: Arc<dyn GraphAlgorithm> = Arc::new(ArticulationPointsAlgorithm);
        assert_eq!(canonical_output_columns(ap.as_ref()), vec!["node_id"]);

        let bridges: Arc<dyn GraphAlgorithm> = Arc::new(BridgesAlgorithm);
        assert_eq!(
            canonical_output_columns(bridges.as_ref()),
            vec!["source", "target"]
        );

        let kcore: Arc<dyn GraphAlgorithm> = Arc::new(KCoreAlgorithm);
        assert_eq!(
            canonical_output_columns(kcore.as_ref()),
            vec!["node_id", "core_number", "max_core"]
        );
    }

    /// A stand-in GraphAlgorithm whose name is not in the canonical list,
    /// used to verify the fallback branch.
    #[cfg(feature = "algos")]
    struct UnknownAlgo;
    #[cfg(feature = "algos")]
    impl GraphAlgorithm for UnknownAlgo {
        fn name(&self) -> &str {
            "totally_new_algorithm"
        }
        fn description(&self) -> &str {
            "unknown for test"
        }
        fn parameters(&self) -> &[ParameterDef] {
            &[]
        }
        fn execute(
            &self,
            _store: &dyn grafeo_core::graph::GraphStore,
            _params: &Parameters,
        ) -> Result<AlgorithmResult> {
            Ok(AlgorithmResult::new(vec!["node_id".into(), "value".into()]))
        }
    }

    #[cfg(feature = "algos")]
    #[test]
    fn canonical_output_columns_falls_back_to_node_id_value() {
        let unknown: Arc<dyn GraphAlgorithm> = Arc::new(UnknownAlgo);
        assert_eq!(
            canonical_output_columns(unknown.as_ref()),
            vec!["node_id", "value"],
            "unknown algorithms must fall back to (node_id, value); otherwise the \
             planner would emit an empty header row"
        );
    }

    // ---------------------------------------------------------------
    // evaluate_arguments: positional and Vector-literal paths were
    // previously exercised only via the Map case.
    // ---------------------------------------------------------------

    #[test]
    fn evaluate_positional_arguments_mapped_by_param_defs() {
        let defs = vec![
            ParameterDef {
                name: "damping".into(),
                param_type: grafeo_adapters::plugins::ParameterType::Float,
                required: true,
                default: None,
                description: String::new(),
            },
            ParameterDef {
                name: "iterations".into(),
                param_type: grafeo_adapters::plugins::ParameterType::Integer,
                required: true,
                default: None,
                description: String::new(),
            },
        ];
        let args = vec![
            LogicalExpression::Literal(Value::Float64(0.9)),
            LogicalExpression::Literal(Value::Int64(50)),
        ];
        let params = evaluate_arguments(&args, &defs);
        assert_eq!(params.get_float("damping"), Some(0.9));
        assert_eq!(params.get_int("iterations"), Some(50));
    }

    #[test]
    fn evaluate_arguments_drops_positional_past_last_param_def() {
        // Defensive: an extra positional arg past the end of `param_defs`
        // should be silently dropped (no panic), matching the existing
        // "extra args ignored" contract of Cypher/GQL procedure calls.
        let defs = vec![ParameterDef {
            name: "only".into(),
            param_type: grafeo_adapters::plugins::ParameterType::Integer,
            required: true,
            default: None,
            description: String::new(),
        }];
        let args = vec![
            LogicalExpression::Literal(Value::Int64(1)),
            LogicalExpression::Literal(Value::Int64(2)), // extra
        ];
        let params = evaluate_arguments(&args, &defs);
        assert_eq!(params.get_int("only"), Some(1));
    }

    #[test]
    fn evaluate_arguments_bool_and_string_literals() {
        let defs = vec![
            ParameterDef {
                name: "flag".into(),
                param_type: grafeo_adapters::plugins::ParameterType::Boolean,
                required: true,
                default: None,
                description: String::new(),
            },
            ParameterDef {
                name: "label".into(),
                param_type: grafeo_adapters::plugins::ParameterType::String,
                required: true,
                default: None,
                description: String::new(),
            },
        ];
        let args = vec![
            LogicalExpression::Literal(Value::Bool(true)),
            LogicalExpression::Literal(Value::String("Person".into())),
        ];
        let params = evaluate_arguments(&args, &defs);
        assert_eq!(params.get_bool("flag"), Some(true));
        assert_eq!(params.get_string("label"), Some("Person"));
    }

    #[test]
    fn evaluate_arguments_vector_literal_becomes_list_of_floats() {
        // Vector literals arrive from the parser as Value::Vector(Arc<[f32]>).
        // The planner must reflect them as a List<Float64> so that
        // `coerce_params_to_vector` can read them uniformly alongside
        // user-typed `[0.1, 0.2, ...]` lists.
        let defs = vec![ParameterDef {
            name: "query".into(),
            param_type: grafeo_adapters::plugins::ParameterType::List,
            required: true,
            default: None,
            description: String::new(),
        }];
        let vec_literal: Arc<[f32]> = Arc::from(vec![0.25_f32, 0.5_f32, 0.75_f32]);
        let args = vec![LogicalExpression::Literal(Value::Vector(vec_literal))];
        let params = evaluate_arguments(&args, &defs);

        let list = params.get_list("query").expect("query must be a list");
        assert_eq!(list.len(), 3);
        // Each element widens from f32 to Float64.
        assert_eq!(list[0], Value::Float64(0.25));
        assert_eq!(list[1], Value::Float64(0.5));
        assert_eq!(list[2], Value::Float64(0.75));
    }

    #[test]
    fn evaluate_arguments_logical_expression_list_flattens_literals() {
        // The parser may produce `LogicalExpression::List(vec![Literal(..), ..])`
        // rather than `Literal(Value::List(..))` depending on the call syntax.
        // Both paths must reach the same set_list call.
        let defs = vec![ParameterDef {
            name: "weights".into(),
            param_type: grafeo_adapters::plugins::ParameterType::List,
            required: true,
            default: None,
            description: String::new(),
        }];
        let args = vec![LogicalExpression::List(vec![
            LogicalExpression::Literal(Value::Int64(1)),
            LogicalExpression::Literal(Value::Int64(2)),
            LogicalExpression::Literal(Value::Int64(3)),
        ])];
        let params = evaluate_arguments(&args, &defs);
        let list = params.get_list("weights").unwrap();
        assert_eq!(list, &[Value::Int64(1), Value::Int64(2), Value::Int64(3)]);
    }

    #[test]
    fn evaluate_arguments_literal_list_value() {
        // Direct Value::List path (parser handed us a pre-wrapped list).
        let defs = vec![ParameterDef {
            name: "items".into(),
            param_type: grafeo_adapters::plugins::ParameterType::List,
            required: true,
            default: None,
            description: String::new(),
        }];
        let args = vec![LogicalExpression::Literal(Value::List(
            vec![Value::Int64(7), Value::Int64(8)].into(),
        ))];
        let params = evaluate_arguments(&args, &defs);
        let list = params.get_list("items").unwrap();
        assert_eq!(list, &[Value::Int64(7), Value::Int64(8)]);
    }

    // ---------------------------------------------------------------
    // procedures_result: the user-facing format of `CALL grafeo.procedures()`.
    // Changing the column order or the parameter-description shape is a
    // breaking change for scripts that parse this.
    // ---------------------------------------------------------------

    #[test]
    fn procedures_result_has_stable_column_shape() {
        let registry = BuiltinProcedures::new();
        let result = procedures_result(&registry);
        assert_eq!(
            result.columns,
            vec!["name", "description", "parameters", "output_columns"],
            "column shape is part of the public CALL grafeo.procedures() contract"
        );
        assert!(!result.rows.is_empty());
    }

    #[test]
    fn procedures_result_rows_are_sorted_by_name() {
        let registry = BuiltinProcedures::new();
        let result = procedures_result(&registry);
        let names: Vec<String> = result
            .rows
            .iter()
            .map(|row| match &row[0] {
                Value::String(s) => s.as_str().to_string(),
                other => panic!("name column must be String, got {other:?}"),
            })
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn procedures_result_names_are_fully_qualified() {
        let registry = BuiltinProcedures::new();
        let result = procedures_result(&registry);
        for row in &result.rows {
            let name = match &row[0] {
                Value::String(s) => s.as_str(),
                _ => panic!(),
            };
            assert!(
                name.starts_with("grafeo."),
                "every listed procedure name must be prefixed with 'grafeo.', got {name:?}"
            );
        }
    }

    #[cfg(feature = "algos")]
    #[test]
    fn procedures_result_parameter_description_distinguishes_required_and_default() {
        let registry = BuiltinProcedures::new();
        let result = procedures_result(&registry);
        // pagerank has required and defaulted params; find its row.
        let pagerank_row = result
            .rows
            .iter()
            .find(|row| matches!(&row[0], Value::String(s) if s.as_str() == "grafeo.pagerank"))
            .expect("pagerank must appear in the listing");
        let params_desc = match &pagerank_row[2] {
            Value::String(s) => s.as_str(),
            _ => panic!("parameters column must be String"),
        };
        // The exact param shape depends on PageRankAlgorithm's definition,
        // but the formatter contract is: defaulted params include `default=...`,
        // required params do not. At least one of each should appear in the
        // overall procedures listing.
        let all_desc: String = result
            .rows
            .iter()
            .map(|row| match &row[2] {
                Value::String(s) => s.as_str(),
                _ => "",
            })
            .collect::<Vec<_>>()
            .join(" | ");
        assert!(
            all_desc.contains("default="),
            "at least one procedure must report a parameter default, got: {all_desc}"
        );
        // Spot-check that pagerank's row has at least one parameter listed.
        assert!(
            !params_desc.is_empty(),
            "pagerank must expose at least one parameter in the description, got empty string"
        );
    }

    // ---------------------------------------------------------------
    // Feature-gated helpers (search procedures).
    // ---------------------------------------------------------------

    #[cfg(all(feature = "lpg", feature = "vector-index"))]
    #[test]
    fn coerce_params_to_vector_accepts_mixed_int_and_float() {
        let mut params = Parameters::new();
        params.set_list(
            "query",
            vec![Value::Float64(0.5), Value::Int64(1), Value::Float64(-0.25)],
        );
        let out = coerce_params_to_vector(&params, "query").unwrap();
        assert_eq!(out, vec![0.5_f32, 1.0_f32, -0.25_f32]);
    }

    #[cfg(all(feature = "lpg", feature = "vector-index"))]
    #[test]
    fn coerce_params_to_vector_errors_when_missing() {
        let params = Parameters::new();
        let err = coerce_params_to_vector(&params, "query").unwrap_err();
        assert!(
            err.to_string()
                .contains("Missing required vector parameter"),
            "error must name the parameter: {err}"
        );
    }

    #[cfg(all(feature = "lpg", feature = "vector-index"))]
    #[test]
    fn coerce_params_to_vector_errors_on_non_numeric_element() {
        let mut params = Parameters::new();
        params.set_list(
            "query",
            vec![Value::Float64(0.1), Value::String("nope".into())],
        );
        let err = coerce_params_to_vector(&params, "query").unwrap_err();
        assert!(
            err.to_string().contains("Expected numeric list"),
            "error must describe expected type: {err}"
        );
    }

    #[cfg(all(feature = "lpg", any(feature = "vector-index", feature = "text-index")))]
    #[test]
    fn k_limit_returns_default_when_missing() {
        let params = Parameters::new();
        assert_eq!(k_limit(&params, 7), 7);
    }

    #[cfg(all(feature = "lpg", any(feature = "vector-index", feature = "text-index")))]
    #[test]
    fn k_limit_clamps_negative_to_zero() {
        // A negative `k` must clamp to 0 (not flip to a huge usize via
        // bit-cast), so a caller that passes -1 gets "no results" instead
        // of scanning the entire index.
        let mut params = Parameters::new();
        params.set_int("k", -5);
        assert_eq!(k_limit(&params, 10), 0);
    }

    #[cfg(all(feature = "lpg", any(feature = "vector-index", feature = "text-index")))]
    #[test]
    fn k_limit_passes_through_positive_value() {
        let mut params = Parameters::new();
        params.set_int("k", 25);
        assert_eq!(k_limit(&params, 10), 25);
    }
}
