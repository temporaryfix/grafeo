//! Variable-length expand operator for multi-hop path traversal.

use super::filter::Predicate;
use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::QueryCancellationToken;
use crate::graph::Direction;
use crate::graph::GraphStoreSearch;
use grafeo_common::types::{EdgeId, EpochId, LogicalType, NodeId, TransactionId, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::sync::Arc;

/// Frontier pops between cooperative cancellation checks.
///
/// A pop is the unit of work that can blow up: one pop costs one adjacency
/// lookup. Polling every pop would make `check()` (an atomic load) a measurable
/// fraction of the loop, so amortise it over a batch.
const CANCELLATION_POLL_INTERVAL: u32 = 1024;

/// Path traversal mode controlling which paths are allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PathMode {
    /// Allows repeated nodes and edges (default).
    #[default]
    Walk,
    /// No repeated edges in a path.
    Trail,
    /// No repeated nodes except the start and end may be equal.
    Simple,
    /// No repeated nodes at all.
    Acyclic,
}

/// Which of the walks matching a pattern survive to the output.
///
/// [`PathMode`] decides whether a walk is *legal*; `PathSearch` decides how many
/// of the legal walks are kept. The two are independent: every search mode
/// applies the active [`PathMode`] gate first and only then its own budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PathSearch {
    /// Every legal walk within the hop bounds. Prunes nothing (default).
    #[default]
    All,
    /// Exactly one row per distinct target node.
    DistinctTargets,
    /// Only the shortest legal walks to each target.
    Shortest {
        /// How many walks (`groups: false`) or how many distinct lengths
        /// (`groups: true`) to keep per target. `k = 0` keeps none.
        k: u32,
        /// When true, keep *every* walk whose length is among the `k` smallest
        /// distinct in-bounds lengths, instead of the first `k` walks.
        groups: bool,
    },
}

/// Per-input-row admission and emission budget for one [`PathSearch`].
///
/// Two gates, deliberately separate:
///
/// * the **admission** gate decides whether a frontier entry is enqueued
///   ([`SearchState::admit_enqueue`]) or expanded out of
///   ([`SearchState::may_expand`]) — this is the pruning, and it is what makes
///   the search cheaper than full enumeration;
/// * the **emission** gate decides whether a dequeued in-bounds walk becomes an
///   output row ([`SearchState::admit_settle`]) — this is what makes the answer
///   *exact*, and it applies whether or not pruning is on.
///
/// Keeping them separate is what lets pruning be switched off for the
/// `PathMode`/bounds combinations where its derivation does not hold: the answer
/// stays correct, only the cost goes back up to today's enumeration cost.
///
/// Every budget counts only **emittable** settles, i.e. pops at a depth inside
/// `[min_hops, max_hops]`. Counting pops below `min_hops` would spend a target's
/// budget on walks that are never returned and silently drop in-bounds answers
/// (a 2-cycle `a -> b -> a` with `*2..3` and `SHORTEST 1` loses `b` at length 3
/// to `b` at length 1).
enum SearchState {
    /// No budget: every legal walk is enqueued, expanded and emitted.
    All,
    /// One row per target. Traversal is keyed on the node, or on
    /// `(node, depth.min(min_hops))` once `min_hops >= 2`, because a node first
    /// reached below `min_hops` must still be reachable again at an in-bounds
    /// depth while later cycle revisits add no new answer.
    DistinctTargets {
        /// Whether the traversal visited set is consulted at all.
        prune: bool,
        /// The depth cap used in the traversal key, if depth is required.
        depth_key: Option<u32>,
        /// Traversal keys already enqueued.
        visited: HashSet<(NodeId, u32)>,
        /// Targets already emitted, the `(input_idx, target)` dedup key.
        emitted: HashSet<NodeId>,
    },
    /// The `k` shortest walks per target, counted with multiplicity.
    ShortestCounted {
        /// Walks to keep per target.
        k: u32,
        /// Whether a settled-out target stops expanding.
        prune: bool,
        /// Emittable settles so far, per target.
        settled: HashMap<NodeId, u32>,
    },
    /// Every walk whose length is among the `k` smallest distinct lengths.
    ShortestGrouped {
        /// Distinct lengths to keep per target.
        k: usize,
        /// Whether a settled-out target stops expanding.
        prune: bool,
        /// Admitted depths so far, per target. At most `k` entries each.
        seen: HashMap<NodeId, Vec<u32>>,
    },
}

impl SearchState {
    /// Builds the budget for one input row.
    ///
    /// `prune` encodes the soundness side conditions from the design:
    ///
    /// * `DistinctTargets` pruning keeps the reachable-target set only under
    ///   `PathMode::Walk`, or when `min_hops <= 1` — where a target's shortest
    ///   walk is acyclic and therefore legal under every mode, so first-discovery
    ///   BFS never discards the only legal route to a target.
    /// * `Shortest` pruning rests on a prefix-counting argument that needs every
    ///   shorter walk to a predecessor to extend to a shorter walk to the target.
    ///   That holds unconditionally under `Walk`; under `Trail`, `Simple` and
    ///   `Acyclic` the extension can be illegal, and the argument survives only
    ///   at `k == 1` with `min_hops <= 1`, where the surviving walk is a shortest
    ///   walk and therefore acyclic.
    fn new(search: PathSearch, path_mode: PathMode, min_hops: u32) -> Self {
        match search {
            PathSearch::All => Self::All,
            PathSearch::DistinctTargets => Self::DistinctTargets {
                prune: path_mode == PathMode::Walk || min_hops <= 1,
                depth_key: (min_hops >= 2).then_some(min_hops),
                visited: HashSet::new(),
                emitted: HashSet::new(),
            },
            PathSearch::Shortest { k, groups } => {
                let prune = path_mode == PathMode::Walk || (k <= 1 && min_hops <= 1);
                if groups {
                    Self::ShortestGrouped {
                        k: k as usize,
                        prune,
                        seen: HashMap::new(),
                    }
                } else {
                    Self::ShortestCounted {
                        k,
                        prune,
                        settled: HashMap::new(),
                    }
                }
            }
        }
    }

    /// Whole-path eligibility depends on history, so a settled predecessor
    /// cannot dominate a later prefix. Keep only the output quotas.
    fn disable_pruning(&mut self) {
        match self {
            Self::DistinctTargets { prune, .. }
            | Self::ShortestCounted { prune, .. }
            | Self::ShortestGrouped { prune, .. } => *prune = false,
            Self::All => {}
        }
    }

    /// Decides whether a candidate frontier entry is enqueued.
    ///
    /// Only `DistinctTargets` prunes here: its answer depends on the target set,
    /// not on walk multiplicity, so duplicates can be dropped before they cost a
    /// queue slot. The `Shortest` budgets are settle-counted instead, because
    /// they must see every arriving walk in depth order to rank it.
    fn admit_enqueue(&mut self, target: NodeId, depth: u32) -> bool {
        match self {
            Self::DistinctTargets {
                prune: true,
                depth_key,
                visited,
                ..
            } => visited.insert((target, depth_key.map_or(0, |minimum| depth.min(minimum)))),
            _ => true,
        }
    }

    /// Decides whether an in-bounds walk of `depth` hops to `target` is emitted,
    /// charging it to the target's budget.
    ///
    /// Called only for depths inside `[min_hops, max_hops]`, in non-decreasing
    /// depth order (the frontier is FIFO over unit-length edges).
    fn admit_settle(&mut self, target: NodeId, depth: u32) -> bool {
        match self {
            Self::All => true,
            Self::DistinctTargets { emitted, .. } => emitted.insert(target),
            Self::ShortestCounted { k, settled, .. } => {
                let count = settled.entry(target).or_insert(0);
                if *count >= *k {
                    return false;
                }
                *count += 1;
                true
            }
            Self::ShortestGrouped { k, seen, .. } => {
                let depths = seen.entry(target).or_default();
                if depths.contains(&depth) {
                    return true;
                }
                if depths.len() >= *k {
                    return false;
                }
                depths.push(depth);
                true
            }
        }
    }

    /// Decides whether to expand out of a node that was just dequeued.
    ///
    /// `emittable` is whether its depth is inside the hop bounds, `settled`
    /// whether [`Self::admit_settle`] accepted it. A node below `min_hops` always
    /// expands: its pops are uncounted, so refusing to expand it would cut the
    /// search off before any answer is in range.
    fn may_expand(&self, emittable: bool, settled: bool) -> bool {
        match self {
            Self::ShortestCounted { prune: true, .. }
            | Self::ShortestGrouped { prune: true, .. } => !emittable || settled,
            _ => true,
        }
    }
}

/// Checks a cancellation token when one is attached.
fn poll_cancellation(cancellation: Option<&QueryCancellationToken>) -> Result<(), OperatorError> {
    if let Some(cancellation) = cancellation {
        cancellation.check()?;
    }
    Ok(())
}

/// An expand operator that handles variable-length path patterns like `*1..3`.
///
/// For each input row containing a source node, this operator produces
/// output rows for each neighbor reachable within the hop range.
#[allow(clippy::struct_excessive_bools)]
pub struct VariableLengthExpandOperator {
    /// The store to traverse.
    store: Arc<dyn GraphStoreSearch>,
    /// Input operator providing source nodes.
    input: Box<dyn Operator>,
    /// Index of the source node column in input.
    source_column: usize,
    /// Direction of edge traversal.
    direction: Direction,
    /// Edge type filter (empty = match all types, multiple = match any).
    edge_types: Vec<String>,
    /// Intrinsic per-edge admission, evaluated before traversal budgets.
    edge_predicate: Option<Box<dyn Predicate>>,
    /// Full candidate-path eligibility, evaluated before output quotas.
    path_predicate: Option<Box<dyn Predicate>>,
    /// Minimum number of hops.
    min_hops: u32,
    /// Maximum number of hops.
    max_hops: u32,
    /// Chunk capacity.
    chunk_capacity: usize,
    /// Transaction ID for MVCC visibility.
    transaction_id: Option<TransactionId>,
    /// Epoch for version visibility.
    viewing_epoch: Option<EpochId>,
    /// When true, skip versioned MVCC lookups (fast path for read-only queries).
    read_only: bool,
    /// Materialized input rows.
    input_rows: Option<Vec<InputRow>>,
    /// Current input row index.
    current_input_idx: usize,
    /// Output buffer for pending results.
    output_buffer: Vec<OutputRow>,
    /// Whether the operator is exhausted.
    exhausted: bool,
    /// Whether to output path length as an additional column.
    output_path_length: bool,
    /// Whether to output full path detail (node list and edge list).
    output_path_detail: bool,
    /// Path traversal mode (WALK, TRAIL, SIMPLE, ACYCLIC).
    path_mode: PathMode,
    /// Which of the legal walks survive (ALL, DISTINCT targets, SHORTEST k).
    path_search: PathSearch,
    /// Cooperative cancellation token, polled during traversal.
    cancellation: Option<QueryCancellationToken>,
}

/// A materialized input row.
struct InputRow {
    /// All column values from the input.
    columns: Vec<ColumnValue>,
    /// Actual source-chunk schema, shared only when predicate admission needs it.
    predicate_schema: Option<Arc<[LogicalType]>>,
    /// The source node ID for expansion.
    source_node: NodeId,
}

/// A column value that can be node ID, edge ID, or generic value.
#[derive(Clone)]
enum ColumnValue {
    NodeId(NodeId),
    EdgeId(EdgeId),
    Value(grafeo_common::types::Value),
}

/// A ready output row.
struct OutputRow {
    /// Index into input_rows for the source row.
    input_idx: usize,
    /// The final edge in the path (`None` for zero-length paths).
    edge_id: Option<EdgeId>,
    /// The target node.
    target_id: NodeId,
    /// The path length (number of edges/hops).
    path_length: u32,
    /// One materialized path shares its arrays between predicate evaluation,
    /// list columns and the emitted first-class path value.
    path: Option<Value>,
}

/// A shared-prefix path segment for efficient BFS path tracking.
///
/// Instead of cloning entire `Vec<NodeId>` / `Vec<EdgeId>` at each BFS expansion
/// step (O(depth) per clone), segments form an `Rc`-linked list that shares common
/// prefixes. Expansion costs O(1) (one `Rc::clone` + one allocation). Full paths
/// are only materialized when emitting output rows.
struct PathSegment {
    /// The node at this position in the path.
    node: NodeId,
    /// The edge taken to reach this node. `None` for the source/root node.
    edge: Option<EdgeId>,
    /// Parent segment, or `None` for the root.
    parent: Option<Rc<PathSegment>>,
}

impl PathSegment {
    /// Materializes the full node path from root to this segment.
    fn collect_nodes(&self, depth: u32) -> Vec<NodeId> {
        let mut nodes = Vec::with_capacity(depth as usize + 1);
        self.collect_nodes_into(&mut nodes);
        nodes
    }

    fn collect_nodes_into(&self, nodes: &mut Vec<NodeId>) {
        if let Some(parent) = &self.parent {
            parent.collect_nodes_into(nodes);
        }
        nodes.push(self.node);
    }

    /// Materializes the full edge path from root to this segment.
    fn collect_edges(&self, depth: u32) -> Vec<EdgeId> {
        let mut edges = Vec::with_capacity(depth as usize);
        self.collect_edges_into(&mut edges);
        edges
    }

    fn collect_edges_into(&self, edges: &mut Vec<EdgeId>) {
        if let Some(parent) = &self.parent {
            parent.collect_edges_into(edges);
        }
        if let Some(edge) = self.edge {
            edges.push(edge);
        }
    }

    /// Checks whether a node already appears in this path segment chain.
    fn contains_node(&self, target: NodeId) -> bool {
        if self.node == target {
            return true;
        }
        if let Some(parent) = &self.parent {
            return parent.contains_node(target);
        }
        false
    }

    /// Checks whether an edge already appears in this path segment chain.
    fn contains_edge(&self, target: EdgeId) -> bool {
        if self.edge == Some(target) {
            return true;
        }
        if let Some(parent) = &self.parent {
            return parent.contains_edge(target);
        }
        false
    }
}

impl VariableLengthExpandOperator {
    /// Creates a new variable-length expand operator.
    pub fn new(
        store: Arc<dyn GraphStoreSearch>,
        input: Box<dyn Operator>,
        source_column: usize,
        direction: Direction,
        edge_types: Vec<String>,
        min_hops: u32,
        max_hops: u32,
    ) -> Self {
        Self {
            store,
            input,
            source_column,
            direction,
            edge_types,
            edge_predicate: None,
            path_predicate: None,
            min_hops,
            max_hops: max_hops.max(min_hops), // Ensure max >= min
            chunk_capacity: 2048,
            transaction_id: None,
            viewing_epoch: None,
            read_only: false,
            input_rows: None,
            current_input_idx: 0,
            output_buffer: Vec::new(),
            exhausted: false,
            output_path_length: false,
            output_path_detail: false,
            path_mode: PathMode::Walk,
            path_search: PathSearch::All,
            cancellation: None,
        }
    }

    /// Sets the path traversal mode.
    pub fn with_path_mode(mut self, mode: PathMode) -> Self {
        self.path_mode = mode;
        self
    }

    /// Sets the path search mode (how many of the legal walks are kept).
    pub fn with_path_search(mut self, search: PathSearch) -> Self {
        self.path_search = search;
        self
    }

    /// Installs intrinsic edge admission before enqueue and shortest quotas.
    ///
    /// The predicate sees the unchanged input columns followed by one typed
    /// Edge column. It must depend only on that edge and fixed input bindings,
    /// not on traversal history or invocation order, to preserve BFS pruning.
    pub fn with_edge_predicate(mut self, predicate: Box<dyn Predicate>) -> Self {
        self.edge_predicate = Some(predicate);
        self
    }

    /// Filters a complete candidate before charging its output quota.
    /// Context: fixed input columns, Edge, Node, length, List(Node),
    /// List(Edge), and Value::Path. Rejected or settled prefixes may extend.
    /// The caller must select a terminating mode/bound for exhaustive history.
    #[must_use]
    pub fn with_path_predicate(mut self, predicate: Box<dyn Predicate>) -> Self {
        self.path_predicate = Some(predicate);
        self.output_path_length = true;
        self.output_path_detail = true;
        self
    }

    /// Attaches a cooperative cancellation token, polled during traversal.
    pub fn with_cancellation_token(mut self, token: QueryCancellationToken) -> Self {
        self.cancellation = Some(token);
        self
    }

    /// Enables path length output as an additional column.
    pub fn with_path_length_output(mut self) -> Self {
        self.output_path_length = true;
        self
    }

    /// Enables full path detail output (node list and edge list columns).
    pub fn with_path_detail_output(mut self) -> Self {
        self.output_path_detail = true;
        self
    }

    /// Sets the chunk capacity.
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity;
        self
    }

    /// Sets the transaction context for MVCC visibility.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Marks this expand as read-only, enabling fast-path lookups.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Materializes all input rows.
    fn materialize_input(&mut self) -> Result<(), OperatorError> {
        let mut rows = Vec::new();

        /// Minimum chunk size for locality sort to be worthwhile.
        const LOCALITY_SORT_THRESHOLD: usize = 1024;

        loop {
            poll_cancellation(self.cancellation.as_ref())?;
            let Some(mut chunk) = self.input.next()? else {
                break;
            };
            // Flatten to handle selection vectors
            chunk.flatten();
            // Sort by source node ID for cache locality during adjacency lookups
            if chunk.len() > LOCALITY_SORT_THRESHOLD {
                chunk = chunk.sort_by_column(self.source_column);
            }

            let predicate_schema: Option<Arc<[LogicalType]>> =
                (self.edge_predicate.is_some() || self.path_predicate.is_some()).then(|| {
                    chunk
                        .columns()
                        .iter()
                        .map(|column| column.data_type().clone())
                        .collect::<Vec<_>>()
                        .into()
                });

            for row_idx in 0..chunk.row_count() {
                // Extract the source node ID
                let col = chunk.column(self.source_column).ok_or_else(|| {
                    OperatorError::ColumnNotFound(format!(
                        "Column {} not found",
                        self.source_column
                    ))
                })?;

                let source_node = col.get_node_id(row_idx).ok_or_else(|| {
                    OperatorError::Execution("Expected node ID in source column".into())
                })?;

                // Materialize all columns
                let mut columns = Vec::with_capacity(chunk.column_count());
                for col_idx in 0..chunk.column_count() {
                    let col = chunk
                        .column(col_idx)
                        .expect("col_idx within column_count range");
                    let value = if let Some(node_id) = col.get_node_id(row_idx) {
                        ColumnValue::NodeId(node_id)
                    } else if let Some(edge_id) = col.get_edge_id(row_idx) {
                        ColumnValue::EdgeId(edge_id)
                    } else if let Some(val) = col.get_value(row_idx) {
                        ColumnValue::Value(val)
                    } else {
                        ColumnValue::Value(grafeo_common::types::Value::Null)
                    };
                    columns.push(value);
                }

                rows.push(InputRow {
                    columns,
                    predicate_schema: predicate_schema.clone(),
                    source_node,
                });
            }
        }

        self.input_rows = Some(rows);
        Ok(())
    }

    /// Creates a single reusable binding row, only for the predicate lane.
    fn predicate_context(
        &self,
        input_idx: usize,
        extra: &[LogicalType],
    ) -> Result<DataChunk, OperatorError> {
        let input = self
            .input_rows
            .as_ref()
            .and_then(|rows| rows.get(input_idx))
            .ok_or_else(|| {
                OperatorError::Execution("predicate admission lost its input row".to_string())
            })?;
        let schema = input.predicate_schema.as_ref().ok_or_else(|| {
            OperatorError::Execution("predicate admission lost its input schema".to_string())
        })?;
        let mut columns = Vec::with_capacity(input.columns.len() + extra.len());
        for (value, data_type) in input.columns.iter().zip(schema.iter()) {
            let mut column =
                crate::execution::vector::ValueVector::with_capacity(data_type.clone(), 1);
            match value {
                ColumnValue::NodeId(id) => column.push_node_id(*id),
                ColumnValue::EdgeId(id) => column.push_edge_id(*id),
                ColumnValue::Value(value) => column.push_value(value.clone()),
            }
            columns.push(column);
        }
        for data_type in extra {
            columns.push(crate::execution::vector::ValueVector::with_capacity(
                data_type.clone(),
                1,
            ));
        }
        let mut context = DataChunk::new(columns);
        context.set_count(1);
        Ok(context)
    }

    fn edge_predicate_context(&self, input_idx: usize) -> Result<Option<DataChunk>, OperatorError> {
        self.edge_predicate
            .as_ref()
            .map(|_| self.predicate_context(input_idx, &[LogicalType::Edge]))
            .transpose()
    }

    fn path_predicate_context(&self, input_idx: usize) -> Result<Option<DataChunk>, OperatorError> {
        self.path_predicate
            .as_ref()
            .map(|_| {
                self.predicate_context(
                    input_idx,
                    &[
                        LogicalType::Edge,
                        LogicalType::Node,
                        LogicalType::Int64,
                        LogicalType::List(Box::new(LogicalType::Node)),
                        LogicalType::List(Box::new(LogicalType::Edge)),
                        LogicalType::Any,
                    ],
                )
            })
            .transpose()
    }

    fn materialize_path(nodes: Vec<NodeId>, edges: Vec<EdgeId>) -> Result<Value, OperatorError> {
        let nodes = nodes
            .into_iter()
            .map(|id| {
                i64::try_from(id.0).map(Value::Int64).map_err(|_| {
                    OperatorError::Execution(format!("NodeId {} exceeds i64 range", id.0))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let edges = edges
            .into_iter()
            .map(|id| {
                i64::try_from(id.0).map(Value::Int64).map_err(|_| {
                    OperatorError::Execution(format!("EdgeId {} exceeds i64 range", id.0))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Value::Path {
            nodes: nodes.into(),
            edges: edges.into(),
        })
    }

    /// Shared output suffix writer; path arrays are assembled once per
    /// candidate and cloned only as Arc owners during predicate/output writes.
    fn append_output_suffix(&self, chunk: &mut DataChunk, row: &OutputRow, input_columns: usize) {
        if let Some(column) = chunk.column_mut(input_columns) {
            if let Some(edge) = row.edge_id {
                column.push_edge_id(edge);
            } else {
                column.push_value(Value::Null);
            }
        }
        if let Some(column) = chunk.column_mut(input_columns + 1) {
            column.push_node_id(row.target_id);
        }
        if self.output_path_length
            && let Some(column) = chunk.column_mut(input_columns + 2)
        {
            column.push_value(Value::Int64(i64::from(row.path_length)));
        }
        if self.output_path_detail
            && let Some(Value::Path { nodes, edges }) = &row.path
        {
            let base = input_columns + 2 + usize::from(self.output_path_length);
            if let Some(column) = chunk.column_mut(base) {
                column.push_value(Value::List(Arc::clone(nodes)));
            }
            if let Some(column) = chunk.column_mut(base + 1) {
                column.push_value(Value::List(Arc::clone(edges)));
            }
            if let Some(column) = chunk.column_mut(base + 2) {
                column.push_value(Value::Path {
                    nodes: Arc::clone(nodes),
                    edges: Arc::clone(edges),
                });
            }
        }
    }

    fn admits_path(
        &self,
        context: &mut Option<DataChunk>,
        row: &OutputRow,
    ) -> Result<bool, OperatorError> {
        let Some(predicate) = &self.path_predicate else {
            return Ok(true);
        };
        poll_cancellation(self.cancellation.as_ref())?;
        let context = context.as_mut().ok_or_else(|| {
            OperatorError::Execution("path admission lost its binding context".to_string())
        })?;
        let input_columns = context.column_count().checked_sub(6).ok_or_else(|| {
            OperatorError::Execution("path admission lost its output schema".to_string())
        })?;
        for index in input_columns..context.column_count() {
            if let Some(column) = context.column_mut(index) {
                column.clear();
            }
        }
        self.append_output_suffix(context, row, input_columns);
        let admitted = predicate.evaluate(context, 0)?;
        poll_cancellation(self.cancellation.as_ref())?;
        Ok(admitted)
    }

    fn admits_edge(
        &self,
        context: &mut Option<DataChunk>,
        edge_id: EdgeId,
    ) -> Result<bool, OperatorError> {
        let Some(predicate) = &self.edge_predicate else {
            return Ok(true);
        };
        poll_cancellation(self.cancellation.as_ref())?;
        let context = context.as_mut().ok_or_else(|| {
            OperatorError::Execution("edge admission lost its binding context".to_string())
        })?;
        let column_index = context.column_count() - 1;
        let candidate = context.column_mut(column_index).ok_or_else(|| {
            OperatorError::Execution("edge admission lost its candidate column".to_string())
        })?;
        candidate.clear();
        candidate.push_edge_id(edge_id);
        let admitted = predicate.evaluate(context, 0)?;
        poll_cancellation(self.cancellation.as_ref())?;
        Ok(admitted)
    }

    /// Gets edges from a node, respecting filters and visibility.
    fn get_edges(&self, node_id: NodeId) -> Vec<(NodeId, EdgeId)> {
        let mut out = Vec::new();
        super::factorized_expand::fill_neighbors_for_expand(
            self.store.as_ref(),
            node_id,
            self.direction,
            &self.edge_types,
            self.viewing_epoch,
            self.transaction_id,
            !self.read_only,
            None,
            &mut out,
        );
        out
    }

    /// Inputs may come from a bound row rather than a node scan, and internal
    /// path nodes never pass through the input scan. Record their visibility at
    /// the transaction snapshot without materializing labels or properties.
    /// Even a read-only statement needs these SSI reads when it has a transaction.
    fn node_is_visible_to_transaction(&self, node: NodeId) -> bool {
        match (self.viewing_epoch, self.transaction_id) {
            (Some(epoch), Some(transaction)) => {
                self.store
                    .is_node_visible_versioned(node, epoch, transaction)
            }
            _ => true,
        }
    }

    /// Checks whether a candidate expansion is allowed under the current path mode.
    fn is_expansion_allowed(
        &self,
        segment: &PathSegment,
        target: NodeId,
        edge_id: EdgeId,
        source_node: NodeId,
    ) -> bool {
        match self.path_mode {
            PathMode::Walk => true,
            PathMode::Trail => !segment.contains_edge(edge_id),
            PathMode::Simple => {
                // No repeated nodes except the start may equal the end
                target == source_node || !segment.contains_node(target)
            }
            PathMode::Acyclic => !segment.contains_node(target),
        }
    }

    /// Process one input row, generating all reachable outputs.
    ///
    /// The frontier is FIFO and every edge costs one hop, so pops arrive in
    /// non-decreasing depth order. That is what lets the [`SearchState`] budgets
    /// rank walks as they are settled instead of collecting and sorting them.
    fn process_input_row(
        &self,
        input_idx: usize,
        source_node: NodeId,
    ) -> Result<Vec<OutputRow>, OperatorError> {
        let mut results = Vec::new();
        let needs_tracking = self.output_path_detail || self.path_mode != PathMode::Walk;
        let mut state = SearchState::new(self.path_search, self.path_mode, self.min_hops);
        if self.path_predicate.is_some() {
            state.disable_pruning();
        }
        let mut pops_since_poll = 0_u32;

        // Poll before any work so an already-cancelled query yields no rows at
        // all rather than one row per input that happened to be cheap.
        poll_cancellation(self.cancellation.as_ref())?;
        if !self.node_is_visible_to_transaction(source_node) {
            return Ok(results);
        }

        let mut predicate_context = self.edge_predicate_context(input_idx)?;
        let mut path_context = self.path_predicate_context(input_idx)?;

        // Zero-length path: when min_hops is 0 the source node matches itself
        // with no edges traversed. Emit it before starting the BFS. It is a
        // settle like any other, so it charges the source's budget: under
        // SHORTEST the length-0 self pair is the shortest walk to the source.
        let mut root_may_expand = true;
        if self.min_hops == 0 {
            let row = OutputRow {
                input_idx,
                edge_id: None,
                target_id: source_node,
                path_length: 0,
                path: if self.output_path_detail {
                    Some(Self::materialize_path(vec![source_node], Vec::new())?)
                } else {
                    None
                },
            };
            let settled =
                self.admits_path(&mut path_context, &row)? && state.admit_settle(source_node, 0);
            if settled {
                results.push(row);
            }
            root_may_expand = state.may_expand(true, settled);
        }

        if !root_may_expand {
            return Ok(results);
        }

        if needs_tracking {
            // BFS with shared-prefix path tracking via Rc<PathSegment>.
            // Required for path detail output or non-Walk path modes.
            let mut frontier: VecDeque<(NodeId, u32, EdgeId, Rc<PathSegment>)> = VecDeque::new();

            let root = Rc::new(PathSegment {
                node: source_node,
                edge: None,
                parent: None,
            });

            for (target, edge_id) in self.get_edges(source_node) {
                if !self.admits_edge(&mut predicate_context, edge_id)? {
                    continue;
                }
                if !self.is_expansion_allowed(&root, target, edge_id, source_node) {
                    continue;
                }
                if !state.admit_enqueue(target, 1) {
                    continue;
                }
                let segment = Rc::new(PathSegment {
                    node: target,
                    edge: Some(edge_id),
                    parent: Some(Rc::clone(&root)),
                });
                frontier.push_back((target, 1, edge_id, segment));
            }

            while let Some((current_node, depth, edge_id, segment)) = frontier.pop_front() {
                pops_since_poll += 1;
                if pops_since_poll >= CANCELLATION_POLL_INTERVAL {
                    pops_since_poll = 0;
                    poll_cancellation(self.cancellation.as_ref())?;
                }

                if !self.node_is_visible_to_transaction(current_node) {
                    continue;
                }
                let emittable = depth >= self.min_hops && depth <= self.max_hops;
                // Without a whole-path predicate, keep the existing cheap
                // quota-first path and materialize only accepted output rows.
                let mut settled = emittable
                    && self.path_predicate.is_none()
                    && state.admit_settle(current_node, depth);
                if settled || (emittable && self.path_predicate.is_some()) {
                    let row = OutputRow {
                        input_idx,
                        edge_id: Some(edge_id),
                        target_id: current_node,
                        path_length: depth,
                        path: if self.output_path_detail {
                            Some(Self::materialize_path(
                                segment.collect_nodes(depth),
                                segment.collect_edges(depth),
                            )?)
                        } else {
                            None
                        },
                    };
                    if self.path_predicate.is_some() {
                        settled = self.admits_path(&mut path_context, &row)?
                            && state.admit_settle(current_node, depth);
                    }
                    if settled {
                        results.push(row);
                    }
                }

                let closed_simple_path =
                    self.path_mode == PathMode::Simple && current_node == source_node;
                if depth < self.max_hops
                    && !closed_simple_path
                    && state.may_expand(emittable, settled)
                {
                    for (target, next_edge_id) in self.get_edges(current_node) {
                        if !self.admits_edge(&mut predicate_context, next_edge_id)? {
                            continue;
                        }
                        if !self.is_expansion_allowed(&segment, target, next_edge_id, source_node) {
                            continue;
                        }
                        if !state.admit_enqueue(target, depth + 1) {
                            continue;
                        }
                        let new_segment = Rc::new(PathSegment {
                            node: target,
                            edge: Some(next_edge_id),
                            parent: Some(Rc::clone(&segment)),
                        });
                        frontier.push_back((target, depth + 1, next_edge_id, new_segment));
                    }
                }
            }
        } else {
            // BFS without path tracking (lightweight, Walk mode only)
            let mut frontier: VecDeque<(NodeId, u32, EdgeId)> = VecDeque::new();

            for (target, edge_id) in self.get_edges(source_node) {
                if !self.admits_edge(&mut predicate_context, edge_id)? {
                    continue;
                }
                if !state.admit_enqueue(target, 1) {
                    continue;
                }
                frontier.push_back((target, 1, edge_id));
            }

            while let Some((current_node, depth, edge_id)) = frontier.pop_front() {
                pops_since_poll += 1;
                if pops_since_poll >= CANCELLATION_POLL_INTERVAL {
                    pops_since_poll = 0;
                    poll_cancellation(self.cancellation.as_ref())?;
                }

                if !self.node_is_visible_to_transaction(current_node) {
                    continue;
                }
                let emittable = depth >= self.min_hops && depth <= self.max_hops;
                let settled = emittable && state.admit_settle(current_node, depth);
                if settled {
                    results.push(OutputRow {
                        input_idx,
                        edge_id: Some(edge_id),
                        target_id: current_node,
                        path_length: depth,
                        path: None,
                    });
                }

                if depth < self.max_hops && state.may_expand(emittable, settled) {
                    for (target, next_edge_id) in self.get_edges(current_node) {
                        if !self.admits_edge(&mut predicate_context, next_edge_id)? {
                            continue;
                        }
                        if !state.admit_enqueue(target, depth + 1) {
                            continue;
                        }
                        frontier.push_back((target, depth + 1, next_edge_id));
                    }
                }
            }
        }

        Ok(results)
    }

    /// Fill the output buffer with results from the next input row.
    fn fill_output_buffer(&mut self) -> Result<(), OperatorError> {
        let Some(input_rows) = &self.input_rows else {
            return Ok(());
        };

        while self.output_buffer.is_empty() && self.current_input_idx < input_rows.len() {
            let source_node = input_rows[self.current_input_idx].source_node;
            let results = self.process_input_row(self.current_input_idx, source_node)?;
            self.output_buffer.extend(results);
            self.current_input_idx += 1;
        }
        Ok(())
    }
}

impl Operator for VariableLengthExpandOperator {
    fn next(&mut self) -> OperatorResult {
        if self.exhausted {
            return Ok(None);
        }
        poll_cancellation(self.cancellation.as_ref())?;

        // Materialize input on first call
        if self.input_rows.is_none() {
            self.materialize_input()?;
            if self.input_rows.as_ref().map_or(true, Vec::is_empty) {
                self.exhausted = true;
                return Ok(None);
            }
        }

        // Fill output buffer if empty
        self.fill_output_buffer()?;

        if self.output_buffer.is_empty() {
            self.exhausted = true;
            return Ok(None);
        }

        let input_rows = self
            .input_rows
            .as_ref()
            .expect("input_rows is Some: populated during BFS");

        // Build output chunk from buffer
        let num_input_cols = input_rows.first().map_or(0, |r| r.columns.len());

        // Schema: [input_columns..., edge, target, (path_length)?, (path_nodes)?, (path_edges)?, (path)?]
        let extra_cols =
            2 + usize::from(self.output_path_length) + usize::from(self.output_path_detail) * 3;
        let mut schema: Vec<LogicalType> = Vec::with_capacity(num_input_cols + extra_cols);
        if let Some(first_row) = input_rows.first() {
            for col_val in &first_row.columns {
                let ty = match col_val {
                    ColumnValue::NodeId(_) => LogicalType::Node,
                    ColumnValue::EdgeId(_) => LogicalType::Edge,
                    ColumnValue::Value(_) => LogicalType::Any,
                };
                schema.push(ty);
            }
        }
        schema.push(LogicalType::Edge);
        schema.push(LogicalType::Node);
        if self.output_path_length {
            schema.push(LogicalType::Int64);
        }
        if self.output_path_detail {
            schema.push(LogicalType::Any); // path_nodes as Value::List
            schema.push(LogicalType::Any); // path_edges as Value::List
            schema.push(LogicalType::Any); // Value::Path (first-class path)
        }

        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_capacity);

        // Take up to chunk_capacity rows from buffer
        let take_count = self.output_buffer.len().min(self.chunk_capacity);
        let to_output: Vec<_> = self.output_buffer.drain(..take_count).collect();

        for out_row in &to_output {
            let input_row = &input_rows[out_row.input_idx];

            // Copy input columns
            for (col_idx, col_val) in input_row.columns.iter().enumerate() {
                if let Some(out_col) = chunk.column_mut(col_idx) {
                    match col_val {
                        ColumnValue::NodeId(id) => out_col.push_node_id(*id),
                        ColumnValue::EdgeId(id) => out_col.push_edge_id(*id),
                        ColumnValue::Value(v) => out_col.push_value(v.clone()),
                    }
                }
            }

            self.append_output_suffix(&mut chunk, out_row, num_input_cols);
        }

        chunk.set_count(to_output.len());
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.input.reset();
        self.input_rows = None;
        self.current_input_idx = 0;
        self.output_buffer.clear();
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "VariableLengthExpand"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn install_resource_context(
        &mut self,
        resources: &crate::execution::memory::QueryResourceContext,
    ) -> Result<(), crate::execution::memory::QueryResourceContextError> {
        self.cancellation = Some(resources.cancellation_token().clone());
        self.input.install_resource_context(resources)
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::execution::operators::ScanOperator;
    use crate::graph::lpg::LpgStore;

    #[test]
    fn test_variable_length_expand_chain() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create chain: a -> b -> c -> d
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        store.set_node_property(a, "name", "a".into());
        store.set_node_property(b, "name", "b".into());
        store.set_node_property(c, "name", "c".into());
        store.set_node_property(d, "name", "d".into());

        store.create_edge(a, b, "NEXT");
        store.create_edge(b, c, "NEXT");
        store.create_edge(c, d, "NEXT");

        // Create scan for all nodes
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));

        // Expand 1-3 hops from all nodes
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec!["NEXT".to_string()],
            1,
            3,
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // From 'a', we should reach b (1 hop), c (2 hops), d (3 hops)
        let a_targets: Vec<NodeId> = results
            .iter()
            .filter(|(s, _)| *s == a)
            .map(|(_, t)| *t)
            .collect();
        assert!(a_targets.contains(&b), "a should reach b");
        assert!(a_targets.contains(&c), "a should reach c");
        assert!(a_targets.contains(&d), "a should reach d");
        assert_eq!(a_targets.len(), 3, "a should reach exactly 3 nodes");
    }

    #[test]
    fn test_variable_length_expand_min_hops() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create chain: a -> b -> c
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);

        store.create_edge(a, b, "NEXT");
        store.create_edge(b, c, "NEXT");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));

        // Expand 2-3 hops only (skip 1 hop)
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec!["NEXT".to_string()],
            2, // min 2 hops
            3, // max 3 hops
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // From 'a', we should reach c (2 hops) but NOT b (1 hop)
        let a_targets: Vec<NodeId> = results
            .iter()
            .filter(|(s, _)| *s == a)
            .map(|(_, t)| *t)
            .collect();
        assert!(
            !a_targets.contains(&b),
            "a should NOT reach b with min_hops=2"
        );
        assert!(a_targets.contains(&c), "a should reach c");
    }

    #[test]
    fn test_variable_length_expand_diamond() {
        let store = Arc::new(LpgStore::new().unwrap());

        //     a
        //    / \
        //   b   c
        //    \ /
        //     d
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        store.create_edge(a, b, "EDGE");
        store.create_edge(a, c, "EDGE");
        store.create_edge(b, d, "EDGE");
        store.create_edge(c, d, "EDGE");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            2,
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // From 'a': b (1 hop), c (1 hop), d (2 hops via b), d (2 hops via c)
        let a_targets: Vec<NodeId> = results
            .iter()
            .filter(|(s, _)| *s == a)
            .map(|(_, t)| *t)
            .collect();
        assert!(a_targets.contains(&b));
        assert!(a_targets.contains(&c));
        assert!(a_targets.contains(&d));
        // d appears twice (two paths)
        assert_eq!(a_targets.iter().filter(|&&t| t == d).count(), 2);
    }

    #[test]
    fn test_variable_length_expand_no_matching_edges() {
        let store = Arc::new(LpgStore::new().unwrap());

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        // Filter for LIKES edges (which don't exist)
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec!["LIKES".to_string()],
            1,
            3,
        );

        let result = expand.next().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_variable_length_expand_single_hop() {
        let store = Arc::new(LpgStore::new().unwrap());

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "EDGE");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        // Exactly 1 hop
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            1,
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // Only a -> b (1 hop)
        let a_results: Vec<_> = results.iter().filter(|(s, _)| *s == a).collect();
        assert_eq!(a_results.len(), 1);
        assert_eq!(a_results[0].1, b);
    }

    #[test]
    fn test_variable_length_expand_with_path_length() {
        let store = Arc::new(LpgStore::new().unwrap());

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "EDGE");
        store.create_edge(b, c, "EDGE");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            2,
        )
        .with_path_length_output();

        let mut found_path_lengths = false;
        while let Ok(Some(chunk)) = expand.next() {
            // With path_length_output, there should be an extra column
            assert!(chunk.column_count() >= 4); // source, edge, target, path_length
            found_path_lengths = true;
        }
        assert!(found_path_lengths);
    }

    #[test]
    fn test_variable_length_expand_reset() {
        let store = Arc::new(LpgStore::new().unwrap());

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "EDGE");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            1,
        );

        // First pass
        let mut count1 = 0;
        while let Ok(Some(chunk)) = expand.next() {
            count1 += chunk.row_count();
        }

        expand.reset();

        // Second pass
        let mut count2 = 0;
        while let Ok(Some(chunk)) = expand.next() {
            count2 += chunk.row_count();
        }

        assert_eq!(count1, count2);
    }

    #[test]
    fn test_variable_length_expand_name() {
        let store = Arc::new(LpgStore::new().unwrap());
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        let expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            3,
        );
        assert_eq!(expand.name(), "VariableLengthExpand");
    }

    #[test]
    fn test_variable_length_expand_empty_input() {
        let store = Arc::new(LpgStore::new().unwrap());
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Nonexistent",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            3,
        );

        let result = expand.next().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_variable_length_expand_with_chunk_capacity() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create a star graph: center -> 5 outer nodes
        let center = store.create_node(&["Node"]);
        for _ in 0..5 {
            let outer = store.create_node(&["Node"]);
            store.create_edge(center, outer, "EDGE");
        }

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            1,
        )
        .with_chunk_capacity(2);

        let mut total = 0;
        let mut chunk_count = 0;
        while let Ok(Some(chunk)) = expand.next() {
            chunk_count += 1;
            total += chunk.row_count();
        }

        assert_eq!(total, 5);
        assert!(chunk_count >= 2);
    }

    #[test]
    fn test_trail_mode_no_repeated_edges() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create cycle: a -> b -> a (same edge types)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "EDGE");
        store.create_edge(b, a, "EDGE");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            4,
        )
        .with_path_mode(PathMode::Trail);

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // From 'a': Trail allows a->b (1 hop) and a->b->a (2 hops, different edges)
        // but NOT a->b->a->b (3 hops, would reuse the a->b edge)
        let a_results: Vec<_> = results.iter().filter(|(s, _)| *s == a).collect();
        assert_eq!(a_results.len(), 2, "Trail from a: a->b and a->b->a only");
    }

    #[test]
    fn test_acyclic_mode_no_repeated_nodes() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create cycle: a -> b -> a
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "EDGE");
        store.create_edge(b, a, "EDGE");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            4,
        )
        .with_path_mode(PathMode::Acyclic);

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // From 'a': Acyclic allows a->b only (cannot revisit a)
        let a_results: Vec<_> = results.iter().filter(|(s, _)| *s == a).collect();
        assert_eq!(a_results.len(), 1, "Acyclic from a: only a->b");
        assert_eq!(a_results[0].1, b);
    }

    #[test]
    fn test_variable_length_expand_into_any() {
        let store = Arc::new(LpgStore::new().unwrap());
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        let op = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            3,
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<VariableLengthExpandOperator>().is_ok());
    }

    // --- PathSegment collection tests ---

    #[test]
    fn test_path_segment_collect_nodes_single_hop() {
        // Root (Alix) -> target (Gus): one hop
        let root = Rc::new(PathSegment {
            node: NodeId(1),
            edge: None,
            parent: None,
        });
        let hop1 = Rc::new(PathSegment {
            node: NodeId(2),
            edge: Some(EdgeId(100)),
            parent: Some(Rc::clone(&root)),
        });

        let nodes = hop1.collect_nodes(1);
        assert_eq!(nodes, vec![NodeId(1), NodeId(2)]);
    }

    #[test]
    fn test_path_segment_collect_nodes_multi_hop() {
        // Chain: Alix(1) -> Gus(2) -> Vincent(3) -> Jules(4)
        let root = Rc::new(PathSegment {
            node: NodeId(1),
            edge: None,
            parent: None,
        });
        let hop1 = Rc::new(PathSegment {
            node: NodeId(2),
            edge: Some(EdgeId(100)),
            parent: Some(Rc::clone(&root)),
        });
        let hop2 = Rc::new(PathSegment {
            node: NodeId(3),
            edge: Some(EdgeId(101)),
            parent: Some(Rc::clone(&hop1)),
        });
        let hop3 = Rc::new(PathSegment {
            node: NodeId(4),
            edge: Some(EdgeId(102)),
            parent: Some(Rc::clone(&hop2)),
        });

        let nodes = hop3.collect_nodes(3);
        assert_eq!(nodes, vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)]);
    }

    #[test]
    fn test_path_segment_collect_edges_single_hop() {
        let root = Rc::new(PathSegment {
            node: NodeId(1),
            edge: None,
            parent: None,
        });
        let hop1 = Rc::new(PathSegment {
            node: NodeId(2),
            edge: Some(EdgeId(100)),
            parent: Some(Rc::clone(&root)),
        });

        let edges = hop1.collect_edges(1);
        assert_eq!(edges, vec![EdgeId(100)]);
    }

    #[test]
    fn test_path_segment_collect_edges_multi_hop() {
        // Chain: 3 edges connecting 4 nodes
        let root = Rc::new(PathSegment {
            node: NodeId(1),
            edge: None,
            parent: None,
        });
        let hop1 = Rc::new(PathSegment {
            node: NodeId(2),
            edge: Some(EdgeId(10)),
            parent: Some(Rc::clone(&root)),
        });
        let hop2 = Rc::new(PathSegment {
            node: NodeId(3),
            edge: Some(EdgeId(20)),
            parent: Some(Rc::clone(&hop1)),
        });
        let hop3 = Rc::new(PathSegment {
            node: NodeId(4),
            edge: Some(EdgeId(30)),
            parent: Some(Rc::clone(&hop2)),
        });

        let edges = hop3.collect_edges(3);
        assert_eq!(edges, vec![EdgeId(10), EdgeId(20), EdgeId(30)]);
    }

    #[test]
    fn test_path_segment_root_has_no_edges() {
        let root = PathSegment {
            node: NodeId(1),
            edge: None,
            parent: None,
        };

        let edges = root.collect_edges(0);
        assert!(edges.is_empty(), "Root segment should yield no edges");

        let nodes = root.collect_nodes(0);
        assert_eq!(
            nodes,
            vec![NodeId(1)],
            "Root should yield only its own node"
        );
    }

    #[test]
    fn test_path_segment_contains_node() {
        let root = Rc::new(PathSegment {
            node: NodeId(1),
            edge: None,
            parent: None,
        });
        let hop1 = Rc::new(PathSegment {
            node: NodeId(2),
            edge: Some(EdgeId(100)),
            parent: Some(Rc::clone(&root)),
        });

        assert!(hop1.contains_node(NodeId(1)), "Should find root node");
        assert!(hop1.contains_node(NodeId(2)), "Should find current node");
        assert!(
            !hop1.contains_node(NodeId(3)),
            "Should not find absent node"
        );
    }

    #[test]
    fn test_path_segment_contains_edge() {
        let root = Rc::new(PathSegment {
            node: NodeId(1),
            edge: None,
            parent: None,
        });
        let hop1 = Rc::new(PathSegment {
            node: NodeId(2),
            edge: Some(EdgeId(100)),
            parent: Some(Rc::clone(&root)),
        });

        assert!(hop1.contains_edge(EdgeId(100)), "Should find current edge");
        assert!(
            !hop1.contains_edge(EdgeId(999)),
            "Should not find absent edge"
        );
    }

    // --- Expansion validation tests (via operator integration) ---

    #[test]
    fn test_walk_mode_allows_everything() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create cycle: Alix -> Gus -> Alix
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(gus, alix, "KNOWS");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Person",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            3,
        )
        .with_path_mode(PathMode::Walk);

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // Walk mode should allow repeated nodes and edges
        // From Alix: Gus(1), Alix(2), Gus(3) = 3 results
        let alix_results: Vec<_> = results.iter().filter(|(s, _)| *s == alix).collect();
        assert_eq!(
            alix_results.len(),
            3,
            "Walk mode should allow all 3 hops from Alix in a cycle"
        );
    }

    #[test]
    fn test_simple_mode_rejects_repeated_node() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Triangle: Vincent -> Jules -> Mia -> Vincent
        let vincent = store.create_node(&["Person"]);
        let jules = store.create_node(&["Person"]);
        let mia = store.create_node(&["Person"]);
        store.create_edge(vincent, jules, "KNOWS");
        store.create_edge(jules, mia, "KNOWS");
        store.create_edge(mia, vincent, "KNOWS");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Person",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            5,
        )
        .with_path_mode(PathMode::Simple);

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // From Vincent: Jules(1), Mia(2), Vincent(3, allowed: start=end)
        // No further expansion because Vincent was already visited
        let vincent_results: Vec<_> = results.iter().filter(|(s, _)| *s == vincent).collect();
        assert_eq!(
            vincent_results.len(),
            3,
            "Simple: Vincent -> Jules, Mia, back to Vincent (start=end allowed)"
        );
    }

    #[test]
    fn test_simple_mode_does_not_extend_after_closed_path() {
        let store = Arc::new(LpgStore::new().unwrap());
        let source = store.create_node(&["N"]);
        let middle = store.create_node(&["N"]);
        let after_close = store.create_node(&["N"]);
        store.create_edge(source, middle, "E");
        store.create_edge(middle, source, "E");
        store.create_edge(source, after_close, "E");

        let rows = expand_rows(&store, "N", 1, 3, PathMode::Simple, PathSearch::All);
        let source_rows = from_source(&rows, source);
        assert!(source_rows.contains(&(middle.0, 1)));
        assert!(source_rows.contains(&(source.0, 2)));
        assert!(
            !source_rows.contains(&(after_close.0, 3)),
            "a Simple path may close at its source, but must not extend after closing"
        );
    }

    // --- Path detail output tests ---

    #[test]
    fn test_path_detail_output_node_and_edge_lists() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Chain: Alix -> Gus -> Vincent
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["Person"]);
        let e1 = store.create_edge(alix, gus, "KNOWS");
        store.create_edge(gus, vincent, "KNOWS");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Person",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec!["KNOWS".to_string()],
            1,
            2,
        )
        .with_path_detail_output();

        let mut found_path_nodes = false;
        let mut found_edge_with_correct_id = false;
        while let Ok(Some(chunk)) = expand.next() {
            // With path detail, extra columns: path_nodes (list), path_edges (list), path (Path)
            // Schema: [source_node, edge, target, path_nodes, path_edges, path]
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();

                // Check path nodes column (index 3) for any Alix-sourced path
                if src == alix
                    && let Some(col) = chunk.column(3)
                    && let Some(val) = col.get_value(i)
                    && let Some(list) = val.as_list()
                {
                    assert!(
                        list.len() >= 2,
                        "Path node list should have at least 2 entries"
                    );
                    found_path_nodes = true;
                }

                // Check path edges column (index 4) for the Alix->Gus single-hop path
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                if src == alix
                    && dst == gus
                    && let Some(col) = chunk.column(4)
                    && let Some(val) = col.get_value(i)
                    && let Some(list) = val.as_list()
                {
                    assert_eq!(list.len(), 1, "Single-hop path should have exactly 1 edge");
                    assert_eq!(list[0].as_int64(), Some(e1.0.cast_signed()));
                    found_edge_with_correct_id = true;
                }
            }
        }
        assert!(
            found_path_nodes,
            "Should have found path node lists in output"
        );
        assert!(
            found_edge_with_correct_id,
            "Should have found edge list with correct edge ID"
        );
    }

    // --- Edge type filtering tests ---

    #[test]
    fn test_edge_type_filter_case_insensitive() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Alix -[:KNOWS]-> Gus, Alix -[:LIKES]-> Vincent
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["Person"]);
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(alix, vincent, "LIKES");

        // Filter with lowercase "knows", should still match "KNOWS"
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Person",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec!["knows".to_string()],
            1,
            1,
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // From Alix, only Gus should be reached (KNOWS matches "knows")
        let alix_targets: Vec<NodeId> = results
            .iter()
            .filter(|(s, _)| *s == alix)
            .map(|(_, t)| *t)
            .collect();
        assert!(
            alix_targets.contains(&gus),
            "Case-insensitive match should find KNOWS edge"
        );
        assert!(
            !alix_targets.contains(&vincent),
            "LIKES edge should be filtered out"
        );
    }

    /// Runs the same intrinsic predicate through the core transition gate.
    fn expansion_with_intrinsic_edge_predicate(
        expand: VariableLengthExpandOperator,
        predicate: Box<dyn super::super::filter::Predicate>,
    ) -> Box<dyn Operator> {
        Box::new(expand.with_edge_predicate(predicate))
    }

    fn intrinsic_allowed_predicate(
        store: &Arc<LpgStore>,
        epoch: EpochId,
    ) -> Box<dyn super::super::filter::Predicate> {
        use super::super::filter::{BinaryFilterOp, ExpressionPredicate, FilterExpression};
        Box::new(
            ExpressionPredicate::new(
                FilterExpression::Binary {
                    left: Box::new(FilterExpression::Property {
                        variable: "e".to_string(),
                        property: "allowed".to_string(),
                    }),
                    op: BinaryFilterOp::Eq,
                    right: Box::new(FilterExpression::Literal(true.into())),
                },
                HashMap::from([("e".to_string(), 1)]),
                store.clone(),
            )
            .with_transaction_context(epoch, None),
        )
    }

    fn intrinsic_target_lengths(
        store: &Arc<LpgStore>,
        target: NodeId,
        search: PathSearch,
        detail: bool,
        epoch: EpochId,
    ) -> Vec<i64> {
        let input = Box::new(ScanOperator::with_label(store.clone(), "Start"));
        let mut expand = VariableLengthExpandOperator::new(
            store.clone(),
            input,
            0,
            Direction::Outgoing,
            vec![],
            1,
            4,
        )
        .with_path_search(search)
        .with_path_length_output()
        .with_transaction_context(epoch, None);
        if detail {
            expand = expand.with_path_detail_output();
        }
        let mut operator = expansion_with_intrinsic_edge_predicate(
            expand,
            intrinsic_allowed_predicate(store, epoch),
        );
        let mut lengths = Vec::new();
        while let Some(chunk) = operator.next().unwrap() {
            for row in chunk.selected_indices() {
                if chunk.column(2).unwrap().get_node_id(row) == Some(target) {
                    lengths.push(
                        chunk
                            .column(3)
                            .unwrap()
                            .get_value(row)
                            .unwrap()
                            .as_int64()
                            .unwrap(),
                    );
                }
            }
        }
        lengths.sort_unstable();
        lengths
    }

    #[test]
    fn intrinsic_edge_predicate_rejects_shortcut_before_shortest_quotas() {
        let store = Arc::new(LpgStore::new().unwrap());
        let source_node = store.create_node(&["Start"]);
        let target_node = store.create_node(&[]);
        let first_branch = store.create_node(&[]);
        let second_branch = store.create_node(&[]);
        let long_first = store.create_node(&[]);
        let long_second = store.create_node(&[]);
        for node in [
            source_node,
            target_node,
            first_branch,
            second_branch,
            long_first,
            long_second,
        ] {
            store.set_node_property(node, "allowed", false.into());
        }
        for (from, to, allowed) in [
            (source_node, target_node, false),
            (source_node, first_branch, true),
            (first_branch, target_node, true),
            (source_node, second_branch, true),
            (second_branch, target_node, true),
            (source_node, long_first, true),
            (long_first, long_second, true),
            (long_second, target_node, true),
        ] {
            let edge = store.create_edge(from, to, "R");
            store.set_edge_property(edge, "allowed", allowed.into());
        }
        for detail in [false, true] {
            for (search, expected) in [
                (
                    PathSearch::Shortest {
                        k: 1,
                        groups: false,
                    },
                    vec![2],
                ),
                (
                    PathSearch::Shortest {
                        k: 2,
                        groups: false,
                    },
                    vec![2, 2],
                ),
                (PathSearch::Shortest { k: 1, groups: true }, vec![2, 2]),
                (PathSearch::Shortest { k: 2, groups: true }, vec![2, 2, 3]),
            ] {
                assert_eq!(
                    intrinsic_target_lengths(
                        &store,
                        target_node,
                        search,
                        detail,
                        store.current_epoch()
                    ),
                    expected,
                    "search={search:?}, detail={detail}"
                );
            }
        }
    }

    #[test]
    fn intrinsic_edge_predicate_rejects_interior_edge_before_prefix_admission() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Start"]);
        let b = store.create_node(&[]);
        let c = store.create_node(&[]);
        let t = store.create_node(&[]);
        for (from, to, allowed) in [(a, b, false), (a, c, true), (c, b, true), (b, t, true)] {
            let edge = store.create_edge(from, to, "R");
            store.set_edge_property(edge, "allowed", allowed.into());
        }
        for detail in [false, true] {
            assert_eq!(
                intrinsic_target_lengths(
                    &store,
                    t,
                    PathSearch::Shortest {
                        k: 1,
                        groups: false
                    },
                    detail,
                    store.current_epoch()
                ),
                vec![3]
            );
        }
    }

    struct EdgePredicateInput {
        chunks: VecDeque<DataChunk>,
    }

    impl Operator for EdgePredicateInput {
        fn next(&mut self) -> OperatorResult {
            Ok(self.chunks.pop_front())
        }
        fn reset(&mut self) {}
        fn name(&self) -> &'static str {
            "EdgePredicateInput"
        }
        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[test]
    fn intrinsic_edge_predicate_preserves_correlated_values_and_each_input_schema() {
        use super::super::filter::{
            BinaryFilterOp, ExpressionPredicate, FilterExpression, Predicate,
        };
        use grafeo_common::types::Value;
        struct CheckSchema(ExpressionPredicate);
        impl Predicate for CheckSchema {
            fn evaluate(&self, chunk: &DataChunk, row: usize) -> Result<bool, OperatorError> {
                let typed = chunk.column(1).and_then(|column| column.get_value(row))
                    == Some(Value::Bool(true));
                let expected = if typed {
                    LogicalType::List(Box::new(LogicalType::Edge))
                } else {
                    LogicalType::Any
                };
                assert_eq!(chunk.column(2).unwrap().data_type(), &expected);
                assert_eq!(chunk.column(3).unwrap().data_type(), &LogicalType::Edge);
                assert!(chunk.column(3).unwrap().get_node_id(row).is_none());
                self.0.evaluate(chunk, row)
            }
        }
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&[]);
        let b = store.create_node(&[]);
        let t = store.create_node(&[]);
        for (from, to, allowed) in [(a, t, false), (a, b, true), (b, t, true)] {
            let edge = store.create_edge(from, to, "R");
            store.set_edge_property(edge, "allowed", allowed.into());
        }
        let chunks = [true, false, true]
            .into_iter()
            .map(|typed| {
                let schema = [
                    LogicalType::Node,
                    LogicalType::Bool,
                    if typed {
                        LogicalType::List(Box::new(LogicalType::Edge))
                    } else {
                        LogicalType::Any
                    },
                ];
                let mut chunk = DataChunk::with_capacity(&schema, 1);
                chunk.column_mut(0).unwrap().push_node_id(a);
                chunk.column_mut(1).unwrap().push_value(Value::Bool(typed));
                chunk
                    .column_mut(2)
                    .unwrap()
                    .push_value(Value::List(vec![Value::Int64(0)].into()));
                chunk.set_count(1);
                chunk
            })
            .collect();
        let predicate = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Property {
                    variable: "e".into(),
                    property: "allowed".into(),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Variable("need".into())),
            },
            HashMap::from([("e".into(), 3), ("need".into(), 1)]),
            store.clone(),
        );
        let mut expand = VariableLengthExpandOperator::new(
            store,
            Box::new(EdgePredicateInput { chunks }),
            0,
            Direction::Outgoing,
            vec![],
            1,
            3,
        )
        .with_path_search(PathSearch::Shortest {
            k: 1,
            groups: false,
        })
        .with_path_length_output()
        .with_edge_predicate(Box::new(CheckSchema(predicate)));
        let mut lengths = Vec::new();
        while let Some(chunk) = expand.next().unwrap() {
            for row in chunk.selected_indices() {
                if chunk.column(4).unwrap().get_node_id(row) == Some(t) {
                    lengths.push(chunk.column(5).unwrap().get_value(row).unwrap());
                }
            }
        }
        assert_eq!(
            lengths,
            vec![Value::Int64(2), Value::Int64(1), Value::Int64(2)]
        );
    }

    #[test]
    fn intrinsic_edge_predicate_reads_retained_epoch_and_own_transaction() {
        use super::super::filter::{BinaryFilterOp, ExpressionPredicate, FilterExpression};
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Start"]);
        let b = store.create_node(&[]);
        let t = store.create_node(&[]);
        let old = EpochId::new(1);
        let newer = EpochId::new(2);
        let tx = TransactionId::new(77);
        for (from, to, old_allowed) in [(a, t, false), (a, b, true), (b, t, true)] {
            let edge = store.create_edge(from, to, "R");
            store.set_edge_property_at_epoch(edge, "allowed", old_allowed.into(), old);
            store.set_edge_property_at_epoch(edge, "allowed", (!old_allowed).into(), newer);
            store.set_edge_property_buffered(edge, "allowed", old_allowed.into(), tx);
        }
        store.set_epoch(newer);
        for (epoch, transaction, length) in [(old, None, 2), (newer, None, 1), (newer, Some(tx), 2)]
        {
            for detail in [false, true] {
                let input = Box::new(ScanOperator::with_label(store.clone(), "Start"));
                let predicate = ExpressionPredicate::new(
                    FilterExpression::Binary {
                        left: Box::new(FilterExpression::Property {
                            variable: "e".into(),
                            property: "allowed".into(),
                        }),
                        op: BinaryFilterOp::Eq,
                        right: Box::new(FilterExpression::Literal(true.into())),
                    },
                    HashMap::from([("e".into(), 1)]),
                    store.clone(),
                )
                .with_transaction_context(epoch, transaction);
                let mut expand = VariableLengthExpandOperator::new(
                    store.clone(),
                    input,
                    0,
                    Direction::Outgoing,
                    vec![],
                    1,
                    3,
                )
                .with_path_search(PathSearch::Shortest {
                    k: 1,
                    groups: false,
                })
                .with_path_length_output()
                .with_transaction_context(epoch, transaction)
                .with_edge_predicate(Box::new(predicate));
                if detail {
                    expand = expand.with_path_detail_output();
                }
                let mut lengths = Vec::new();
                while let Some(chunk) = expand.next().unwrap() {
                    for row in chunk.selected_indices() {
                        if chunk.column(2).unwrap().get_node_id(row) == Some(t) {
                            lengths.push(
                                chunk
                                    .column(3)
                                    .unwrap()
                                    .get_value(row)
                                    .unwrap()
                                    .as_int64()
                                    .unwrap(),
                            );
                        }
                    }
                }
                assert_eq!(
                    lengths,
                    vec![length],
                    "epoch={epoch:?}, tx={transaction:?}, detail={detail}"
                );
            }
        }
    }

    #[test]
    fn intrinsic_edge_predicate_errors_and_cancellation_stop_before_output() {
        struct Refuse;
        impl Predicate for Refuse {
            fn evaluate(&self, _chunk: &DataChunk, _row: usize) -> Result<bool, OperatorError> {
                Err(OperatorError::Execution(
                    "edge admission read failed".into(),
                ))
            }
        }
        struct Cancel(crate::execution::QueryCancellationHandle);
        impl Predicate for Cancel {
            fn evaluate(&self, _chunk: &DataChunk, _row: usize) -> Result<bool, OperatorError> {
                self.0.cancel();
                Ok(false)
            }
        }
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Start"]);
        let b = store.create_node(&[]);
        store.create_edge(a, b, "R");
        for detail in [false, true] {
            let make = || {
                let input = Box::new(ScanOperator::with_label(store.clone(), "Start"));
                let expand = VariableLengthExpandOperator::new(
                    store.clone(),
                    input,
                    0,
                    Direction::Outgoing,
                    vec![],
                    1,
                    3,
                );
                if detail {
                    expand.with_path_detail_output()
                } else {
                    expand
                }
            };
            let mut error = make().with_edge_predicate(Box::new(Refuse));
            assert!(
                matches!(error.next(),Err(OperatorError::Execution(message)) if message == "edge admission read failed")
            );
            let control = crate::execution::QueryExecutionControl::new();
            let mut cancelled = make()
                .with_cancellation_token(control.token())
                .with_edge_predicate(Box::new(Cancel(control.cancellation_handle())));
            assert!(matches!(
                cancelled.next(),
                Err(OperatorError::QueryCancelled(_))
            ));
        }
    }

    fn expansion_with_full_path_predicate(
        expand: VariableLengthExpandOperator,
        predicate: Box<dyn Predicate>,
    ) -> Box<dyn Operator> {
        Box::new(expand.with_path_predicate(predicate))
    }

    fn full_path_test_predicate(
        store: &Arc<LpgStore>,
        expression: super::super::filter::FilterExpression,
    ) -> Box<dyn Predicate> {
        Box::new(super::super::filter::ExpressionPredicate::new(
            expression,
            HashMap::from([
                ("a".into(), 0),
                ("e".into(), 1),
                ("b".into(), 2),
                ("len".into(), 3),
                ("nodes".into(), 4),
                ("edges".into(), 5),
                ("p".into(), 6),
            ]),
            store.clone(),
        ))
    }

    fn full_path_target_lengths(mut operator: Box<dyn Operator>, target: NodeId) -> Vec<i64> {
        let mut lengths = Vec::new();
        while let Some(chunk) = operator.next().unwrap() {
            for row in chunk.selected_indices() {
                if chunk.column(2).unwrap().get_node_id(row) == Some(target) {
                    lengths.push(
                        chunk
                            .column(3)
                            .unwrap()
                            .get_value(row)
                            .unwrap()
                            .as_int64()
                            .unwrap(),
                    );
                }
            }
        }
        lengths
    }

    #[test]
    fn full_path_predicate_filters_before_counted_and_grouped_quotas() {
        use super::super::filter::{
            BinaryFilterOp as Op, FilterExpression as E, ListPredicateKind,
        };
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Start"]);
        let b = store.create_node(&[]);
        let c = store.create_node(&[]);
        let d = store.create_node(&[]);
        let target = store.create_node(&[]);
        let forbidden = store.create_edge(a, target, "R");
        store.set_edge_property(forbidden, "allowed", false.into());
        for (from, to) in [
            (a, b),
            (b, target),
            (a, c),
            (c, target),
            (b, d),
            (d, target),
        ] {
            let edge = store.create_edge(from, to, "R");
            store.set_edge_property(edge, "allowed", true.into());
        }
        for (search, expected) in [
            (
                PathSearch::Shortest {
                    k: 1,
                    groups: false,
                },
                vec![2],
            ),
            (
                PathSearch::Shortest {
                    k: 2,
                    groups: false,
                },
                vec![2, 2],
            ),
            (PathSearch::Shortest { k: 1, groups: true }, vec![2, 2]),
            (PathSearch::Shortest { k: 2, groups: true }, vec![2, 2, 3]),
        ] {
            let predicate = E::Binary {
                left: Box::new(E::Binary {
                    left: Box::new(E::Variable("len".into())),
                    op: Op::Ge,
                    right: Box::new(E::Literal(2.into())),
                }),
                op: Op::And,
                right: Box::new(E::ListPredicate {
                    kind: ListPredicateKind::All,
                    variable: "r".into(),
                    list_expr: Box::new(E::FunctionCall {
                        name: "relationships".into(),
                        args: vec![E::Variable("p".into())],
                    }),
                    predicate: Box::new(E::Property {
                        variable: "r".into(),
                        property: "allowed".into(),
                    }),
                }),
            };
            let expand = VariableLengthExpandOperator::new(
                store.clone(),
                Box::new(ScanOperator::with_label(store.clone(), "Start")),
                0,
                Direction::Outgoing,
                vec![],
                1,
                3,
            )
            .with_path_mode(PathMode::Trail)
            .with_path_search(search);
            assert_eq!(
                full_path_target_lengths(
                    expansion_with_full_path_predicate(
                        expand,
                        full_path_test_predicate(&store, predicate)
                    ),
                    target
                ),
                expected,
                "{search:?}"
            );
        }
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn full_path_predicate_later_settled_prefix_can_enable_eligible_suffix() {
        use super::super::filter::{BinaryFilterOp as Op, FilterExpression as E};
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Start"]);
        let via = store.create_node(&[]);
        let junction = store.create_node(&[]);
        let target = store.create_node(&[]);
        store.create_edge(a, junction, "R");
        let required = store.create_edge(a, via, "R");
        store.create_edge(via, junction, "R");
        store.create_edge(junction, target, "R");
        let predicate = E::Binary {
            left: Box::new(E::Binary {
                left: Box::new(E::Variable("len".into())),
                op: Op::Eq,
                right: Box::new(E::Literal(1.into())),
            }),
            op: Op::Or,
            right: Box::new(E::Binary {
                left: Box::new(E::Literal(Value::Int64(required.as_u64() as i64))),
                op: Op::In,
                right: Box::new(E::FunctionCall {
                    name: "edges".into(),
                    args: vec![E::Variable("p".into())],
                }),
            }),
        };
        for search in [
            PathSearch::DistinctTargets,
            PathSearch::Shortest {
                k: 1,
                groups: false,
            },
            PathSearch::Shortest { k: 1, groups: true },
        ] {
            let expand = VariableLengthExpandOperator::new(
                store.clone(),
                Box::new(ScanOperator::with_label(store.clone(), "Start")),
                0,
                Direction::Outgoing,
                vec![],
                1,
                3,
            )
            .with_path_mode(PathMode::Trail)
            .with_path_search(search);
            assert_eq!(
                full_path_target_lengths(
                    expansion_with_full_path_predicate(
                        expand,
                        full_path_test_predicate(&store, predicate.clone())
                    ),
                    target
                ),
                vec![3],
                "{search:?}"
            );
        }
    }

    #[test]
    fn full_path_predicate_zero_hop_and_unbounded_trail_cycle_terminate() {
        use super::super::filter::{BinaryFilterOp as Op, FilterExpression as E};
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Start"]);
        let mid = store.create_node(&[]);
        let target = store.create_node(&[]);
        store.create_edge(a, mid, "R");
        store.create_edge(mid, a, "R");
        store.create_edge(a, target, "R");
        for (minimum_length, selected_target, expected) in
            [(1, a, vec![2]), (3, target, vec![3]), (4, target, vec![])]
        {
            let predicate = E::Binary {
                left: Box::new(E::Variable("len".into())),
                op: Op::Ge,
                right: Box::new(E::Literal(Value::Int64(minimum_length))),
            };
            let expand = VariableLengthExpandOperator::new(
                store.clone(),
                Box::new(ScanOperator::with_label(store.clone(), "Start")),
                0,
                Direction::Outgoing,
                vec![],
                0,
                u32::MAX,
            )
            .with_path_mode(PathMode::Trail)
            .with_path_search(PathSearch::Shortest {
                k: 1,
                groups: false,
            });
            assert_eq!(
                full_path_target_lengths(
                    expansion_with_full_path_predicate(
                        expand,
                        full_path_test_predicate(&store, predicate)
                    ),
                    selected_target
                ),
                expected
            );
        }
    }

    #[test]
    fn full_path_predicate_preserves_mode_constraints_and_zero_hop_identity() {
        use super::super::filter::{BinaryFilterOp as Op, FilterExpression as E};
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Start"]);
        let via = store.create_node(&[]);
        let target = store.create_node(&[]);
        store.create_edge(a, via, "R");
        store.create_edge(via, a, "R");
        store.create_edge(a, target, "R");
        for (mode, selected, length, expected) in [
            (PathMode::Walk, via, 3, vec![3]),
            (PathMode::Trail, via, 3, vec![]),
            (PathMode::Trail, target, 3, vec![3]),
            (PathMode::Simple, a, 2, vec![2]),
            (PathMode::Simple, target, 3, vec![]),
            (PathMode::Acyclic, a, 2, vec![]),
        ] {
            let expand = VariableLengthExpandOperator::new(
                store.clone(),
                Box::new(ScanOperator::with_label(store.clone(), "Start")),
                0,
                Direction::Outgoing,
                vec![],
                0,
                3,
            )
            .with_path_mode(mode)
            .with_path_search(PathSearch::Shortest {
                k: 1,
                groups: false,
            })
            .with_path_predicate(full_path_test_predicate(
                &store,
                E::Binary {
                    left: Box::new(E::Variable("len".into())),
                    op: Op::Eq,
                    right: Box::new(E::Literal(Value::Int64(length))),
                },
            ));
            assert_eq!(
                full_path_target_lengths(Box::new(expand), selected),
                expected,
                "{mode:?}"
            );
        }
        struct Zero(NodeId);
        impl Predicate for Zero {
            fn evaluate(&self, chunk: &DataChunk, row: usize) -> Result<bool, OperatorError> {
                assert_eq!(chunk.column(1).unwrap().data_type(), &LogicalType::Edge);
                assert_eq!(chunk.column(1).unwrap().get_value(row), Some(Value::Null));
                assert_eq!(chunk.column(2).unwrap().get_node_id(row), Some(self.0));
                assert_eq!(
                    chunk.column(5).unwrap().get_value(row),
                    Some(Value::List(Vec::new().into()))
                );
                Ok(true)
            }
        }
        let expand = VariableLengthExpandOperator::new(
            store.clone(),
            Box::new(ScanOperator::with_label(store.clone(), "Start")),
            0,
            Direction::Outgoing,
            vec![],
            0,
            0,
        )
        .with_path_mode(PathMode::Trail)
        .with_path_predicate(Box::new(Zero(a)));
        assert_eq!(full_path_target_lengths(Box::new(expand), a), vec![0]);
    }

    #[test]
    fn full_path_predicate_reads_retained_epoch_and_own_transaction_and_resets() {
        use super::super::filter::{ExpressionPredicate, FilterExpression as E, ListPredicateKind};
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Start"]);
        let via = store.create_node(&[]);
        let target = store.create_node(&[]);
        let old = EpochId::new(1);
        let newer = EpochId::new(2);
        let tx = TransactionId::new(77);
        for (from, to, allowed) in [(a, target, false), (a, via, true), (via, target, true)] {
            let edge = store.create_edge(from, to, "R");
            store.set_edge_property_at_epoch(edge, "allowed", allowed.into(), old);
            store.set_edge_property_at_epoch(edge, "allowed", (!allowed).into(), newer);
            store.set_edge_property_buffered(edge, "allowed", allowed.into(), tx);
        }
        store.set_epoch(newer);
        for (epoch, transaction, expected) in
            [(old, None, 2), (newer, None, 1), (newer, Some(tx), 2)]
        {
            let predicate = ExpressionPredicate::new(
                E::ListPredicate {
                    kind: ListPredicateKind::All,
                    variable: "r".into(),
                    list_expr: Box::new(E::FunctionCall {
                        name: "relationships".into(),
                        args: vec![E::Variable("p".into())],
                    }),
                    predicate: Box::new(E::Property {
                        variable: "r".into(),
                        property: "allowed".into(),
                    }),
                },
                HashMap::from([("p".into(), 6)]),
                store.clone(),
            )
            .with_transaction_context(epoch, transaction);
            let mut expand = VariableLengthExpandOperator::new(
                store.clone(),
                Box::new(ScanOperator::with_label(store.clone(), "Start")),
                0,
                Direction::Outgoing,
                vec![],
                1,
                3,
            )
            .with_path_mode(PathMode::Trail)
            .with_path_search(PathSearch::Shortest {
                k: 1,
                groups: false,
            })
            .with_transaction_context(epoch, transaction)
            .with_path_predicate(Box::new(predicate));
            for _ in 0..2 {
                let mut lengths = Vec::new();
                while let Some(chunk) = expand.next().unwrap() {
                    for row in chunk.selected_indices() {
                        if chunk.column(2).unwrap().get_node_id(row) == Some(target) {
                            lengths.push(chunk.column(3).unwrap().get_value(row).unwrap());
                        }
                    }
                }
                assert_eq!(
                    lengths,
                    vec![Value::Int64(expected)],
                    "epoch={epoch:?}, tx={transaction:?}"
                );
                expand.reset();
            }
        }
    }

    #[test]
    fn full_path_predicate_context_keeps_correlated_schema_and_raw_identity() {
        use super::super::filter::{BinaryFilterOp, ExpressionPredicate, FilterExpression as E};
        struct CheckContext(ExpressionPredicate);
        impl Predicate for CheckContext {
            fn evaluate(&self, chunk: &DataChunk, row: usize) -> Result<bool, OperatorError> {
                let typed = chunk.column(1).unwrap().get_value(row) == Some(Value::Bool(true));
                assert_eq!(
                    chunk.column(2).unwrap().data_type(),
                    &if typed {
                        LogicalType::List(Box::new(LogicalType::Edge))
                    } else {
                        LogicalType::Any
                    }
                );
                for (offset, expected) in [
                    LogicalType::Edge,
                    LogicalType::Node,
                    LogicalType::Int64,
                    LogicalType::List(Box::new(LogicalType::Node)),
                    LogicalType::List(Box::new(LogicalType::Edge)),
                    LogicalType::Any,
                ]
                .iter()
                .enumerate()
                {
                    assert_eq!(chunk.column(3 + offset).unwrap().data_type(), expected);
                }
                assert!(chunk.column(3).unwrap().get_node_id(row).is_none());
                assert!(chunk.column(4).unwrap().get_edge_id(row).is_none());
                let Value::Path { nodes, edges } = chunk.column(8).unwrap().get_value(row).unwrap()
                else {
                    panic!("candidate must retain first-class path identity");
                };
                assert_eq!(
                    chunk.column(6).unwrap().get_value(row),
                    Some(Value::List(nodes))
                );
                assert_eq!(
                    chunk.column(7).unwrap().get_value(row),
                    Some(Value::List(edges))
                );
                self.0.evaluate(chunk, row)
            }
        }
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&[]);
        let via = store.create_node(&[]);
        let target = store.create_node(&[]);
        for (from, to, allowed) in [(a, target, false), (a, via, true), (via, target, true)] {
            let edge = store.create_edge(from, to, "R");
            store.set_edge_property(edge, "allowed", allowed.into());
        }
        let chunks = [true, false, true]
            .into_iter()
            .map(|typed| {
                let mut chunk = DataChunk::with_capacity(
                    &[
                        LogicalType::Node,
                        LogicalType::Bool,
                        if typed {
                            LogicalType::List(Box::new(LogicalType::Edge))
                        } else {
                            LogicalType::Any
                        },
                    ],
                    1,
                );
                chunk.column_mut(0).unwrap().push_node_id(a);
                chunk.column_mut(1).unwrap().push_value(Value::Bool(typed));
                chunk
                    .column_mut(2)
                    .unwrap()
                    .push_value(Value::List(vec![Value::Int64(0)].into()));
                chunk.set_count(1);
                chunk
            })
            .collect();
        let predicate = ExpressionPredicate::new(
            E::Binary {
                left: Box::new(E::Property {
                    variable: "e".into(),
                    property: "allowed".into(),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(E::Variable("need".into())),
            },
            HashMap::from([("e".into(), 3), ("need".into(), 1)]),
            store.clone(),
        );
        let mut expand = VariableLengthExpandOperator::new(
            store,
            Box::new(EdgePredicateInput { chunks }),
            0,
            Direction::Outgoing,
            vec![],
            1,
            3,
        )
        .with_path_mode(PathMode::Trail)
        .with_path_search(PathSearch::Shortest {
            k: 1,
            groups: false,
        })
        .with_path_predicate(Box::new(CheckContext(predicate)));
        let mut lengths = Vec::new();
        while let Some(chunk) = expand.next().unwrap() {
            for row in chunk.selected_indices() {
                if chunk.column(4).unwrap().get_node_id(row) == Some(target) {
                    lengths.push(chunk.column(5).unwrap().get_value(row).unwrap());
                }
            }
        }
        assert_eq!(
            lengths,
            vec![Value::Int64(2), Value::Int64(1), Value::Int64(2)]
        );
    }

    #[test]
    fn full_path_predicate_errors_and_cancellation_stop_before_output() {
        struct Refuse;
        impl Predicate for Refuse {
            fn evaluate(&self, _chunk: &DataChunk, _row: usize) -> Result<bool, OperatorError> {
                Err(OperatorError::Execution(
                    "path admission read failed".into(),
                ))
            }
        }
        struct Cancel(crate::execution::QueryCancellationHandle);
        impl Predicate for Cancel {
            fn evaluate(&self, _chunk: &DataChunk, _row: usize) -> Result<bool, OperatorError> {
                self.0.cancel();
                Ok(false)
            }
        }
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Start"]);
        let b = store.create_node(&[]);
        store.create_edge(a, b, "R");
        for minimum in [0, 1] {
            let make = || {
                VariableLengthExpandOperator::new(
                    store.clone(),
                    Box::new(ScanOperator::with_label(store.clone(), "Start")),
                    0,
                    Direction::Outgoing,
                    vec![],
                    minimum,
                    3,
                )
                .with_path_mode(PathMode::Trail)
            };
            let mut error = make().with_path_predicate(Box::new(Refuse));
            assert!(
                matches!(error.next(), Err(OperatorError::Execution(message)) if message == "path admission read failed")
            );
            let control = crate::execution::QueryExecutionControl::new();
            let mut cancelled = make()
                .with_cancellation_token(control.token())
                .with_path_predicate(Box::new(Cancel(control.cancellation_handle())));
            assert!(matches!(
                cancelled.next(),
                Err(OperatorError::QueryCancelled(_))
            ));
        }
    }

    /// Ported from the retired shortest-path operator: retain the visibility
    /// and SSI contract without depending on its bidirectional implementation.
    #[test]
    fn unified_shortest_snapshot_traversal_and_ssi_read_recording() {
        use crate::execution::operators::{ReadTracker, SharedReadTracker};
        use parking_lot::Mutex;
        struct Spy {
            nodes: Mutex<HashSet<NodeId>>,
            edges: Mutex<HashSet<EdgeId>>,
        }
        impl ReadTracker for Spy {
            fn record_node_read(&self, _tx: TransactionId, id: NodeId) {
                self.nodes.lock().insert(id);
            }
            fn record_edge_read(&self, _tx: TransactionId, id: EdgeId) {
                self.edges.lock().insert(id);
            }
        }
        let store = Arc::new(LpgStore::new().unwrap());
        let node_epoch = store.new_epoch();
        let a = store.create_node_versioned(&["Start"], node_epoch, TransactionId::SYSTEM);
        let b = store.create_node_versioned(&[], node_epoch, TransactionId::SYSTEM);
        let c = store.create_node_versioned(&[], node_epoch, TransactionId::SYSTEM);
        store.finalize_entities_by_id(TransactionId::SYSTEM, node_epoch, &[a, b, c], &[]);
        let edge_epoch = store.new_epoch();
        let ab = store.create_edge_versioned(a, b, "R", edge_epoch, TransactionId::SYSTEM);
        let bc = store.create_edge_versioned(b, c, "R", edge_epoch, TransactionId::SYSTEM);
        let snapshot = store.new_epoch();
        store.finalize_entities_by_id(TransactionId::SYSTEM, snapshot, &[], &[ab, bc]);
        let late_epoch = store.new_epoch();
        let late = store.create_edge_versioned(a, c, "R", late_epoch, TransactionId::SYSTEM);
        let latest = store.new_epoch();
        store.finalize_entities_by_id(TransactionId::SYSTEM, latest, &[], &[late]);
        for groups in [false, true] {
            for detail in [false, true] {
                let spy = Arc::new(Spy {
                    nodes: Mutex::new(HashSet::new()),
                    edges: Mutex::new(HashSet::new()),
                });
                let tx = TransactionId::new(77);
                store.register_read_tracker(tx, Arc::clone(&spy) as SharedReadTracker);
                let make = |epoch| {
                    let expand = VariableLengthExpandOperator::new(
                        store.clone(),
                        Box::new(ScanOperator::with_label(store.clone(), "Start")),
                        0,
                        Direction::Outgoing,
                        vec![],
                        1,
                        3,
                    )
                    .with_path_search(PathSearch::Shortest { k: 1, groups })
                    .with_path_length_output()
                    .with_transaction_context(epoch, Some(tx));
                    if detail {
                        expand.with_path_detail_output()
                    } else {
                        expand
                    }
                };
                assert_eq!(
                    full_path_target_lengths(Box::new(make(snapshot)), c),
                    vec![2]
                );
                assert_eq!(*spy.edges.lock(), HashSet::from([ab, bc]));
                let nodes = spy.nodes.lock();
                assert!(nodes.contains(&a));
                assert!(nodes.contains(&b));
                assert!(nodes.contains(&c));
                drop(nodes);
                assert_eq!(full_path_target_lengths(Box::new(make(latest)), c), vec![1]);
                assert!(spy.edges.lock().contains(&late));
                store.unregister_read_tracker(tx);
            }
        }
    }

    #[test]
    fn unified_shortest_zero_hop_validates_bound_source_snapshot_and_own_transaction() {
        use crate::execution::operators::{ReadTracker, SharedReadTracker};
        use parking_lot::Mutex;
        struct NodesRead(Mutex<HashSet<NodeId>>);
        impl ReadTracker for NodesRead {
            fn record_node_read(&self, _tx: TransactionId, node: NodeId) {
                self.0.lock().insert(node);
            }
            fn record_edge_read(&self, _tx: TransactionId, _edge: EdgeId) {}
        }
        let store = Arc::new(LpgStore::new().unwrap());
        let snapshot = store.new_epoch();
        let visible = store.create_node_versioned(&[], snapshot, TransactionId::SYSTEM);
        store.finalize_entities_by_id(TransactionId::SYSTEM, snapshot, &[visible], &[]);
        let later = store.new_epoch();
        let future = store.create_node_versioned(&[], later, TransactionId::SYSTEM);
        store.finalize_entities_by_id(TransactionId::SYSTEM, later, &[future], &[]);
        let owner = TransactionId::new(77);
        let other = TransactionId::new(78);
        let pending = store.create_node_versioned(&[], snapshot, owner);
        for read_only in [false, true] {
            for (source, tx, expected) in [
                (visible, owner, vec![0]),
                (NodeId::new(999), owner, vec![]),
                (future, owner, vec![]),
                (pending, owner, vec![0]),
                (pending, other, vec![]),
            ] {
                let spy = Arc::new(NodesRead(Mutex::new(HashSet::new())));
                store.register_read_tracker(tx, Arc::clone(&spy) as SharedReadTracker);
                // Deliberately bypass ScanOperator: already-bound source rows
                // must receive the same visibility/read contract, including zero hops.
                let mut input = DataChunk::with_capacity(&[LogicalType::Node], 1);
                input.column_mut(0).unwrap().push_node_id(source);
                input.set_count(1);
                let expand = VariableLengthExpandOperator::new(
                    store.clone(),
                    Box::new(EdgePredicateInput {
                        chunks: VecDeque::from([input]),
                    }),
                    0,
                    Direction::Outgoing,
                    vec![],
                    0,
                    0,
                )
                .with_path_search(PathSearch::Shortest {
                    k: 1,
                    groups: false,
                })
                .with_path_length_output()
                .with_transaction_context(snapshot, Some(tx))
                .with_read_only(read_only);
                assert_eq!(
                    full_path_target_lengths(Box::new(expand), source),
                    expected,
                    "source={source:?}, tx={tx:?}, read_only={read_only}"
                );
                assert_eq!(spy.0.lock().contains(&source), !expected.is_empty());
                store.unregister_read_tracker(tx);
            }
        }
    }

    /// Preserve long-chain result, relationship filtering and reset controls
    /// while exercising all directions through the one surviving executor.
    #[test]
    fn unified_shortest_long_chain_direction_type_bounds_and_chunked_reset() {
        let store = Arc::new(LpgStore::new().unwrap());
        let nodes: Vec<_> = (0..10)
            .map(|index| {
                store.create_node(match index {
                    0 => &["Left"],
                    9 => &["Right"],
                    _ => &[],
                })
            })
            .collect();
        for pair in nodes.windows(2) {
            store.create_edge(pair[0], pair[1], "NEXT");
        }
        store.create_edge(nodes[0], nodes[9], "OTHER");
        for groups in [false, true] {
            for (label, direction, edge_type, maximum, target, expected) in [
                ("Left", Direction::Outgoing, "next", 9, nodes[9], vec![9]),
                ("Right", Direction::Incoming, "NEXT", 9, nodes[0], vec![9]),
                ("Right", Direction::Both, "NEXT", 9, nodes[0], vec![9]),
                ("Left", Direction::Incoming, "NEXT", 9, nodes[9], vec![]),
                ("Left", Direction::Outgoing, "NEXT", 8, nodes[9], vec![]),
                ("Left", Direction::Outgoing, "OTHER", 9, nodes[9], vec![1]),
            ] {
                let mut expand = VariableLengthExpandOperator::new(
                    store.clone(),
                    Box::new(ScanOperator::with_label(store.clone(), label)),
                    0,
                    direction,
                    vec![edge_type.into()],
                    1,
                    maximum,
                )
                .with_path_search(PathSearch::Shortest { k: 1, groups })
                .with_path_length_output()
                .with_chunk_capacity(1);
                for _ in 0..2 {
                    let mut lengths = Vec::new();
                    while let Some(chunk) = expand.next().unwrap() {
                        assert_eq!(chunk.row_count(), 1);
                        if chunk.column(2).unwrap().get_node_id(0) == Some(target) {
                            lengths.push(
                                chunk
                                    .column(3)
                                    .unwrap()
                                    .get_value(0)
                                    .unwrap()
                                    .as_int64()
                                    .unwrap(),
                            );
                        }
                    }
                    assert_eq!(
                        lengths, expected,
                        "{direction:?}, type={edge_type}, groups={groups}"
                    );
                    expand.reset();
                }
            }
        }
    }

    // --- PathSearch: shared harness -----------------------------------------

    /// Runs an expand over every node carrying `label` and collects one
    /// `(source, target, length)` triple per output row.
    fn expand_rows(
        store: &Arc<LpgStore>,
        label: &str,
        min_hops: u32,
        max_hops: u32,
        mode: PathMode,
        search: PathSearch,
    ) -> Vec<(NodeId, NodeId, u32)> {
        expand_rows_with_detail(store, label, min_hops, max_hops, mode, search, false)
    }

    /// Same as [`expand_rows`], optionally forcing the path-tracking branch.
    fn expand_rows_with_detail(
        store: &Arc<LpgStore>,
        label: &str,
        min_hops: u32,
        max_hops: u32,
        mode: PathMode,
        search: PathSearch,
        path_detail: bool,
    ) -> Vec<(NodeId, NodeId, u32)> {
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(store) as Arc<dyn GraphStoreSearch>,
            label,
        ));
        let expand = VariableLengthExpandOperator::new(
            Arc::clone(store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            min_hops,
            max_hops,
        )
        .with_path_mode(mode)
        .with_path_search(search)
        .with_path_length_output();
        let mut expand = if path_detail {
            expand.with_path_detail_output()
        } else {
            expand
        };

        let mut rows = Vec::new();
        while let Some(chunk) = expand.next().expect("expand must not fail") {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                let len = chunk
                    .column(3)
                    .unwrap()
                    .get_value(i)
                    .unwrap()
                    .as_int64()
                    .unwrap();
                rows.push((src, dst, u32::try_from(len).expect("length fits u32")));
            }
        }
        rows
    }

    /// The `(target, length)` rows produced for one input row, sorted so tests
    /// can compare multisets rather than emission order.
    fn from_source(rows: &[(NodeId, NodeId, u32)], src: NodeId) -> Vec<(u64, u32)> {
        let mut out: Vec<(u64, u32)> = rows
            .iter()
            .filter(|(s, _, _)| *s == src)
            .map(|(_, t, l)| (t.0, *l))
            .collect();
        out.sort_unstable();
        out
    }

    /// `a -> b -> a`, the fixture that separates walk enumeration from pruning.
    fn two_cycle() -> (Arc<LpgStore>, NodeId, NodeId) {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        store.create_edge(a, b, "E");
        store.create_edge(b, a, "E");
        (store, a, b)
    }

    /// `a -> {b, c} -> d`: two distinct walks of equal minimal length to `d`.
    fn diamond() -> (Arc<LpgStore>, NodeId, NodeId, NodeId, NodeId) {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let c = store.create_node(&["N"]);
        let d = store.create_node(&["N"]);
        store.create_edge(a, b, "E");
        store.create_edge(a, c, "E");
        store.create_edge(b, d, "E");
        store.create_edge(c, d, "E");
        (store, a, b, c, d)
    }

    /// `a -> b -> c -> d`, the fixture with exactly one walk per target.
    fn chain4() -> (Arc<LpgStore>, Vec<NodeId>) {
        let store = Arc::new(LpgStore::new().unwrap());
        let nodes: Vec<NodeId> = (0..4).map(|_| store.create_node(&["N"])).collect();
        for pair in nodes.windows(2) {
            store.create_edge(pair[0], pair[1], "E");
        }
        (store, nodes)
    }

    // --- PathSearch: per-mode invariants ------------------------------------

    #[test]
    fn test_all_enumerates_every_walk_on_two_cycle() {
        let (store, a, b) = two_cycle();
        let rows = expand_rows(&store, "N", 1, 3, PathMode::Walk, PathSearch::All);
        assert_eq!(
            from_source(&rows, a),
            vec![(a.0, 2), (b.0, 1), (b.0, 3)],
            "All keeps one row per distinct walk"
        );
    }

    #[test]
    fn test_distinct_targets_emits_exactly_one_row_per_target() {
        let (store, a, b) = two_cycle();
        let rows = expand_rows(
            &store,
            "N",
            1,
            3,
            PathMode::Walk,
            PathSearch::DistinctTargets,
        );
        assert_eq!(
            from_source(&rows, a),
            vec![(a.0, 2), (b.0, 1)],
            "DistinctTargets dedups on (input row, target) and keeps the shortest"
        );
    }

    #[test]
    fn test_distinct_targets_min_hops_saturates_cycle_admission() {
        let (_, a, b) = two_cycle();
        let mut state = SearchState::new(PathSearch::DistinctTargets, PathMode::Walk, 2);

        // The first in-bounds prefix for each cycle node is sufficient. Once
        // that prefix is admitted, a later cycle revisit must not add work.
        assert!(state.admit_enqueue(a, 1));
        assert!(state.admit_enqueue(b, 2));
        assert!(state.admit_enqueue(a, 3));
        assert!(
            !state.admit_enqueue(b, 4),
            "cycle admission must saturate at the minimum hop depth"
        );
    }

    #[test]
    fn test_distinct_targets_two_cycle_completes_at_unbounded_max_hops() {
        let (store, a, b) = two_cycle();
        for path_detail in [false, true] {
            let rows = expand_rows_with_detail(
                &store,
                "N",
                2,
                u32::MAX,
                PathMode::Walk,
                PathSearch::DistinctTargets,
                path_detail,
            );
            assert_eq!(
                from_source(&rows, a),
                vec![(a.0, 2), (b.0, 3)],
                "two-cycle must terminate with or without path tracking"
            );
        }
    }

    #[test]
    fn test_distinct_targets_self_loop_completes_at_unbounded_max_hops() {
        let store = Arc::new(LpgStore::new().unwrap());
        let node = store.create_node(&["N"]);
        store.create_edge(node, node, "E");

        for path_detail in [false, true] {
            let rows = expand_rows_with_detail(
                &store,
                "N",
                2,
                u32::MAX,
                PathMode::Walk,
                PathSearch::DistinctTargets,
                path_detail,
            );
            assert_eq!(
                from_source(&rows, node),
                vec![(node.0, 2)],
                "self-loop must terminate with or without path tracking"
            );
        }
    }

    #[test]
    fn test_distinct_targets_keeps_depth_in_the_traversal_key_above_min_hops_one() {
        // `b` is reachable from `a` at depth 1 (out of bounds) and depth 3 (in
        // bounds). A node-only visited set would burn `b` on the depth-1 pop and
        // lose the only in-bounds row for it.
        let (store, a, b) = two_cycle();
        let rows = expand_rows(
            &store,
            "N",
            2,
            3,
            PathMode::Walk,
            PathSearch::DistinctTargets,
        );
        assert_eq!(
            from_source(&rows, a),
            vec![(a.0, 2), (b.0, 3)],
            "a node-only visited key would drop b entirely"
        );
    }

    /// A below-minimum arrival must not consume the one output slot for a target.
    #[test]
    fn test_distinct_targets_does_not_settle_before_min_hops() {
        let store = Arc::new(LpgStore::new().unwrap());
        let s = store.create_node(&["N"]);
        let t = store.create_node(&["N"]);
        let x = store.create_node(&["N"]);
        store.create_edge(s, t, "E");
        store.create_edge(s, x, "E");
        store.create_edge(x, t, "E");

        for path_detail in [false, true] {
            let rows = expand_rows_with_detail(
                &store,
                "N",
                2,
                2,
                PathMode::Walk,
                PathSearch::DistinctTargets,
                path_detail,
            );
            assert_eq!(
                from_source(&rows, s),
                vec![(t.0, 2)],
                "DistinctTargets must retain t's in-bounds arrival (path_detail={path_detail})"
            );
        }
    }

    /// The per-target shortest budget must start at the lower hop bound.
    #[test]
    fn test_shortest_does_not_count_before_min_hops() {
        let store = Arc::new(LpgStore::new().unwrap());
        let s = store.create_node(&["N"]);
        let t = store.create_node(&["N"]);
        let x = store.create_node(&["N"]);
        store.create_edge(s, t, "E");
        store.create_edge(s, x, "E");
        store.create_edge(x, t, "E");

        for path_detail in [false, true] {
            let rows = expand_rows_with_detail(
                &store,
                "N",
                2,
                2,
                PathMode::Walk,
                PathSearch::Shortest {
                    k: 1,
                    groups: false,
                },
                path_detail,
            );
            assert_eq!(
                from_source(&rows, s),
                vec![(t.0, 2)],
                "Shortest must retain t's in-bounds arrival (path_detail={path_detail})"
            );
        }
    }

    #[test]
    fn test_shortest_one_keeps_a_single_minimal_walk_per_target() {
        let (store, a, b, c, d) = diamond();
        let all = expand_rows(&store, "N", 1, 3, PathMode::Walk, PathSearch::All);
        assert_eq!(
            from_source(&all, a),
            vec![(b.0, 1), (c.0, 1), (d.0, 2), (d.0, 2)],
            "All sees both walks to d"
        );

        let shortest = expand_rows(
            &store,
            "N",
            1,
            3,
            PathMode::Walk,
            PathSearch::Shortest {
                k: 1,
                groups: false,
            },
        );
        assert_eq!(
            from_source(&shortest, a),
            vec![(b.0, 1), (c.0, 1), (d.0, 2)],
            "SHORTEST 1 keeps one walk per target, at the minimal length"
        );
    }

    #[test]
    fn test_all_shortest_keeps_every_minimal_walk_with_distinct_edge_lists() {
        let (store, a, _b, _c, d) = diamond();
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "N",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            3,
        )
        .with_path_search(PathSearch::Shortest { k: 1, groups: true })
        .with_path_length_output()
        .with_path_detail_output();

        // [source, edge, target, path_length, path_nodes, path_edges, path]
        let mut d_edge_lists: Vec<Vec<i64>> = Vec::new();
        let mut rows_from_a = 0;
        while let Some(chunk) = expand.next().expect("expand must not fail") {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                if src != a {
                    continue;
                }
                rows_from_a += 1;
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                if dst == d {
                    let val = chunk.column(5).unwrap().get_value(i).unwrap();
                    let list = val.as_list().expect("path edges is a list");
                    d_edge_lists.push(list.iter().map(|v| v.as_int64().unwrap()).collect());
                }
            }
        }

        assert_eq!(
            rows_from_a, 4,
            "ALL SHORTEST keeps b, c and both walks to d"
        );
        assert_eq!(d_edge_lists.len(), 2, "both minimal walks to d survive");
        assert_ne!(
            d_edge_lists[0], d_edge_lists[1],
            "the two rows for d must carry distinct _path_edges_"
        );
    }

    #[test]
    fn test_shortest_k_counted_keeps_the_k_smallest_lengths() {
        let (store, a, b) = two_cycle();
        let rows = expand_rows(
            &store,
            "N",
            1,
            5,
            PathMode::Walk,
            PathSearch::Shortest {
                k: 2,
                groups: false,
            },
        );
        // Walk lengths from a: b at 1, 3, 5 and a at 2, 4.
        assert_eq!(
            from_source(&rows, a),
            vec![(a.0, 2), (a.0, 4), (b.0, 1), (b.0, 3)],
            "SHORTEST 2 keeps the two smallest lengths per target"
        );
    }

    #[test]
    fn test_shortest_k_groups_keeps_every_walk_at_an_admitted_length() {
        // Two branches merge before the target; the left branch also has a shortcut.
        // Target walk lengths are 2 via the shortcut and 3 via either branch.
        let store = Arc::new(LpgStore::new().unwrap());
        let source_node = store.create_node(&["N"]);
        let left_branch = store.create_node(&["N"]);
        let right_branch = store.create_node(&["N"]);
        let merge_node = store.create_node(&["N"]);
        let target_node = store.create_node(&["N"]);
        store.create_edge(source_node, left_branch, "E");
        store.create_edge(source_node, right_branch, "E");
        store.create_edge(left_branch, merge_node, "E");
        store.create_edge(right_branch, merge_node, "E");
        store.create_edge(merge_node, target_node, "E");
        store.create_edge(left_branch, target_node, "E");

        let grouped = expand_rows(
            &store,
            "N",
            1,
            4,
            PathMode::Walk,
            PathSearch::Shortest { k: 2, groups: true },
        );
        assert_eq!(
            from_source(&grouped, source_node),
            vec![
                (left_branch.0, 1),
                (right_branch.0, 1),
                (merge_node.0, 2),
                (merge_node.0, 2),
                (target_node.0, 2),
                (target_node.0, 3),
                (target_node.0, 3)
            ],
            "GROUPS keeps every walk whose length is among the 2 smallest distinct lengths"
        );

        let counted = expand_rows(
            &store,
            "N",
            1,
            4,
            PathMode::Walk,
            PathSearch::Shortest {
                k: 2,
                groups: false,
            },
        );
        assert_eq!(
            from_source(&counted, source_node),
            vec![
                (left_branch.0, 1),
                (right_branch.0, 1),
                (merge_node.0, 2),
                (merge_node.0, 2),
                (target_node.0, 2),
                (target_node.0, 3)
            ],
            "without GROUPS only two walks per target survive"
        );
    }

    #[test]
    fn test_hop_bounds_are_honoured_in_every_search_mode() {
        let (store, nodes) = chain4();
        let (a, b, c, d) = (nodes[0], nodes[1], nodes[2], nodes[3]);
        for search in [
            PathSearch::All,
            PathSearch::DistinctTargets,
            PathSearch::Shortest {
                k: 1,
                groups: false,
            },
            PathSearch::Shortest { k: 3, groups: true },
        ] {
            let rows = expand_rows(&store, "N", 2, 3, PathMode::Walk, search);
            let from_a = from_source(&rows, a);
            assert_eq!(
                from_a,
                vec![(c.0, 2), (d.0, 3)],
                "{search:?} must respect *2..3"
            );
            assert!(
                !from_a.iter().any(|(t, _)| *t == b.0),
                "{search:?} must not emit the 1-hop neighbour"
            );
        }
    }

    #[test]
    fn test_source_is_its_own_target_only_through_the_zero_hop_row() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["N"]);
        for search in [
            PathSearch::All,
            PathSearch::DistinctTargets,
            PathSearch::Shortest {
                k: 1,
                groups: false,
            },
            PathSearch::Shortest { k: 2, groups: true },
        ] {
            let bounded = expand_rows(&store, "N", 1, 3, PathMode::Walk, search);
            assert!(
                from_source(&bounded, a).is_empty(),
                "{search:?} must not pair an edgeless source with itself"
            );
            let zero = expand_rows(&store, "N", 0, 3, PathMode::Walk, search);
            assert_eq!(
                from_source(&zero, a),
                vec![(a.0, 0)],
                "{search:?} keeps exactly the zero-hop self row"
            );
        }
    }

    #[test]
    fn test_source_reappears_once_through_an_in_bounds_cycle() {
        let (store, a, _b) = two_cycle();
        let rows = expand_rows(
            &store,
            "N",
            1,
            3,
            PathMode::Walk,
            PathSearch::Shortest {
                k: 1,
                groups: false,
            },
        );
        let self_rows: Vec<_> = from_source(&rows, a)
            .into_iter()
            .filter(|(t, _)| *t == a.0)
            .collect();
        assert_eq!(
            self_rows,
            vec![(a.0, 2)],
            "the source returns once, at the cycle length, not at length 0"
        );
    }

    #[test]
    fn test_unreachable_target_produces_no_row_in_any_search_mode() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let c = store.create_node(&["N"]);
        store.create_edge(a, b, "E");
        for search in [
            PathSearch::All,
            PathSearch::DistinctTargets,
            PathSearch::Shortest {
                k: 2,
                groups: false,
            },
            PathSearch::Shortest { k: 2, groups: true },
        ] {
            let rows = expand_rows(&store, "N", 1, 4, PathMode::Walk, search);
            assert_eq!(
                from_source(&rows, a),
                vec![(b.0, 1)],
                "{search:?} reaches only b from a"
            );
            assert!(
                from_source(&rows, c).is_empty(),
                "{search:?} emits nothing for an isolated source"
            );
        }
    }

    #[test]
    fn test_zero_hop_row_charges_the_shortest_budget() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["N"]);
        store.create_edge(a, a, "E");

        let all = expand_rows(&store, "N", 0, 2, PathMode::Walk, PathSearch::All);
        assert_eq!(
            from_source(&all, a),
            vec![(a.0, 0), (a.0, 1), (a.0, 2)],
            "All enumerates the self loop"
        );

        let shortest = expand_rows(
            &store,
            "N",
            0,
            2,
            PathMode::Walk,
            PathSearch::Shortest {
                k: 1,
                groups: false,
            },
        );
        assert_eq!(
            from_source(&shortest, a),
            vec![(a.0, 0)],
            "the zero-hop row is the shortest walk to the source and spends k=1"
        );

        let two = expand_rows(
            &store,
            "N",
            0,
            2,
            PathMode::Walk,
            PathSearch::Shortest {
                k: 2,
                groups: false,
            },
        );
        assert_eq!(from_source(&two, a), vec![(a.0, 0), (a.0, 1)]);

        let distinct = expand_rows(
            &store,
            "N",
            0,
            2,
            PathMode::Walk,
            PathSearch::DistinctTargets,
        );
        assert_eq!(
            from_source(&distinct, a),
            vec![(a.0, 0)],
            "the zero-hop row seeds the (input row, target) dedup key"
        );
    }

    #[test]
    fn test_search_mode_does_not_change_the_output_columns() {
        let (store, _nodes) = chain4();
        let mut widths = Vec::new();
        for search in [
            PathSearch::All,
            PathSearch::DistinctTargets,
            PathSearch::Shortest {
                k: 1,
                groups: false,
            },
            PathSearch::Shortest { k: 2, groups: true },
        ] {
            let scan = Box::new(ScanOperator::with_label(
                Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
                "N",
            ));
            let mut expand = VariableLengthExpandOperator::new(
                Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
                scan,
                0,
                Direction::Outgoing,
                vec![],
                1,
                3,
            )
            .with_path_search(search)
            .with_path_length_output()
            .with_path_detail_output();
            let chunk = expand
                .next()
                .expect("expand must not fail")
                .expect("chain yields rows");
            widths.push(chunk.column_count());
        }
        assert_eq!(
            widths,
            vec![7, 7, 7, 7],
            "every mode emits [source, edge, target, length, nodes, edges, path]"
        );
    }

    // --- Cancellation witness ------------------------------------------------

    #[test]
    fn test_pre_cancelled_expand_yields_an_error_and_no_rows() {
        use crate::execution::{QueryCancellationError, QueryExecutionControl};

        // 6-node complete digraph: `*1..8` is 5^8 walks if it ever starts.
        let store = Arc::new(LpgStore::new().unwrap());
        let nodes: Vec<NodeId> = (0..6).map(|_| store.create_node(&["N"])).collect();
        for &from in &nodes {
            for &to in &nodes {
                if from != to {
                    store.create_edge(from, to, "E");
                }
            }
        }

        let control = QueryExecutionControl::new();
        let handle = control.cancellation_handle();
        handle.cancel();

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "N",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            8,
        )
        .with_cancellation_token(control.token());

        let first = expand.next();
        assert!(
            matches!(
                first,
                Err(OperatorError::QueryCancelled(
                    QueryCancellationError::Cancelled
                ))
            ),
            "expected a cancellation error, got {first:?}"
        );

        let mut rows = 0;
        while let Ok(Some(chunk)) = expand.next() {
            rows += chunk.row_count();
        }
        assert_eq!(rows, 0, "a cancelled expand must emit no rows");
    }

    #[test]
    fn test_install_resource_context_propagates_pre_cancelled_token() {
        use crate::execution::{
            QueryCancellationError, QueryExecutionControl, QueryResourceContext,
        };
        use grafeo_common::memory::buffer::BufferManager;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountingInput {
            calls: Arc<AtomicUsize>,
        }

        impl Operator for CountingInput {
            fn next(&mut self) -> OperatorResult {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(None)
            }

            fn reset(&mut self) {}

            fn name(&self) -> &'static str {
                "CountingInput"
            }

            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }

        let store = Arc::new(LpgStore::new().unwrap());
        let source = store.create_node(&["N"]);
        let target = store.create_node(&["N"]);
        store.create_edge(source, target, "E");

        let control = QueryExecutionControl::new();
        control.cancellation_handle().cancel();
        let resources = QueryResourceContext::new_with_cancellation(
            BufferManager::with_budget(1_000_000),
            control.token(),
        )
        .unwrap();
        let input_calls = Arc::new(AtomicUsize::new(0));
        let input = Box::new(CountingInput {
            calls: Arc::clone(&input_calls),
        });
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            input,
            0,
            Direction::Outgoing,
            vec![],
            1,
            1,
        );
        expand.install_resource_context(&resources).unwrap();

        assert!(matches!(
            expand.next(),
            Err(OperatorError::QueryCancelled(
                QueryCancellationError::Cancelled
            ))
        ));
        assert_eq!(
            input_calls.load(Ordering::SeqCst),
            0,
            "pre-cancelled next must not drain its child input"
        );
    }

    // --- PathSearch: brute-force property test ------------------------------

    /// Node count for the random digraphs.
    const PROP_NODES: usize = 8;

    /// Every legal walk from `source` whose length lands inside the hop window,
    /// as `(target index, length)`.
    ///
    /// The oracle prunes nothing at all, so a disagreement with the operator is
    /// always an admission rule that drops or duplicates an answer.
    fn brute_force_walks(
        adj: &[Vec<(usize, usize)>],
        source: usize,
        mode: PathMode,
        min_hops: u32,
        max_hops: u32,
    ) -> Vec<(usize, u32)> {
        let mut out = Vec::new();
        if min_hops == 0 {
            out.push((source, 0));
        }
        let mut nodes = vec![source];
        let mut edges: Vec<usize> = Vec::new();
        walk_step(
            adj, mode, min_hops, max_hops, &mut nodes, &mut edges, &mut out,
        );
        out
    }

    /// One depth-first step of [`brute_force_walks`], mirroring
    /// `is_expansion_allowed` by hand so the two cannot share a bug.
    fn walk_step(
        adj: &[Vec<(usize, usize)>],
        mode: PathMode,
        min_hops: u32,
        max_hops: u32,
        nodes: &mut Vec<usize>,
        edges: &mut Vec<usize>,
        out: &mut Vec<(usize, u32)>,
    ) {
        let depth = u32::try_from(edges.len()).expect("depth fits u32");
        if depth >= max_hops {
            return;
        }
        let current = *nodes.last().expect("a walk always has a source");
        for &(target, edge) in &adj[current] {
            let legal = match mode {
                PathMode::Walk => true,
                PathMode::Trail => !edges.contains(&edge),
                PathMode::Simple => {
                    nodes.push(target);
                    let legal = simple_sequence_is_legal(nodes);
                    nodes.pop();
                    legal
                }
                PathMode::Acyclic => !nodes.contains(&target),
            };
            if !legal {
                continue;
            }
            nodes.push(target);
            edges.push(edge);
            if depth + 1 >= min_hops {
                out.push((target, depth + 1));
            }
            walk_step(adj, mode, min_hops, max_hops, nodes, edges, out);
            nodes.pop();
            edges.pop();
        }
    }

    /// Validates a complete Simple node sequence independently of the operator.
    /// Every repeated node is rejected except the first/last pair, which is the
    /// sole allowed equal-endpoint closure.
    fn simple_sequence_is_legal(nodes: &[usize]) -> bool {
        for (left, &node) in nodes.iter().enumerate() {
            for (right, &other) in nodes.iter().enumerate().skip(left + 1) {
                if node == other && !(left == 0 && right + 1 == nodes.len()) {
                    return false;
                }
            }
        }
        true
    }

    /// Applies one `PathSearch`'s emission rule to a brute-force walk set.
    fn expected_rows(walks: &[(usize, u32)], search: PathSearch) -> Vec<(usize, u32)> {
        let mut by_target: HashMap<usize, Vec<u32>> = HashMap::new();
        for &(target, length) in walks {
            by_target.entry(target).or_default().push(length);
        }
        let mut out = Vec::new();
        for (target, mut lengths) in by_target {
            lengths.sort_unstable();
            match search {
                PathSearch::All => out.extend(lengths.into_iter().map(|l| (target, l))),
                PathSearch::DistinctTargets => out.push((target, lengths[0])),
                PathSearch::Shortest { k, groups: false } => {
                    out.extend(lengths.into_iter().take(k as usize).map(|l| (target, l)));
                }
                PathSearch::Shortest { k, groups: true } => {
                    let mut distinct = lengths.clone();
                    distinct.dedup();
                    distinct.truncate(k as usize);
                    out.extend(
                        lengths
                            .into_iter()
                            .filter(|l| distinct.contains(l))
                            .map(|l| (target, l)),
                    );
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// Settles the derived `k`-bounded admission rules against brute force.
    ///
    /// Deterministically seeded, so a counterexample reproduces verbatim.
    #[test]
    fn test_path_search_modes_match_brute_force_enumeration() {
        use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

        let strategy = (
            proptest::collection::vec((0usize..PROP_NODES, 0usize..PROP_NODES), 0..=12),
            0u32..=2,
            0u32..=3,
            0usize..4,
            1u32..=3,
            proptest::bool::ANY,
        );
        let mut runner = TestRunner::new_with_rng(
            Config {
                cases: 96,
                failure_persistence: None,
                ..Config::default()
            },
            TestRng::deterministic_rng(RngAlgorithm::ChaCha),
        );

        runner
            .run(
                &strategy,
                |(edge_list, min_hops, span, mode_idx, k, groups)| {
                    let max_hops = min_hops + span;
                    let mode = [
                        PathMode::Walk,
                        PathMode::Trail,
                        PathMode::Simple,
                        PathMode::Acyclic,
                    ][mode_idx];

                    let store = Arc::new(LpgStore::new().unwrap());
                    let nodes: Vec<NodeId> =
                        (0..PROP_NODES).map(|_| store.create_node(&["N"])).collect();
                    let index: HashMap<NodeId, usize> =
                        nodes.iter().enumerate().map(|(i, n)| (*n, i)).collect();
                    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); PROP_NODES];
                    for (edge_idx, &(from, to)) in edge_list.iter().enumerate() {
                        store.create_edge(nodes[from], nodes[to], "E");
                        adj[from].push((to, edge_idx));
                    }

                    for search in [
                        PathSearch::All,
                        PathSearch::DistinctTargets,
                        PathSearch::Shortest { k, groups },
                    ] {
                        let rows = expand_rows(&store, "N", min_hops, max_hops, mode, search);
                        for (src_idx, src) in nodes.iter().enumerate() {
                            let mut actual: Vec<(usize, u32)> = rows
                                .iter()
                                .filter(|(s, _, _)| s == src)
                                .map(|(_, t, l)| (index[t], *l))
                                .collect();
                            actual.sort_unstable();
                            let walks = brute_force_walks(&adj, src_idx, mode, min_hops, max_hops);
                            let expected = expected_rows(&walks, search);
                            proptest::prop_assert_eq!(
                                actual,
                                expected,
                                "{:?} {:?} *{}..{} from node {} on {:?}",
                                search,
                                mode,
                                min_hops,
                                max_hops,
                                src_idx,
                                edge_list
                            );
                        }
                    }
                    Ok(())
                },
            )
            .expect("every PathSearch mode must agree with brute-force enumeration");
    }
}
