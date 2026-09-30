//! Project operator for selecting and transforming columns.

use super::filter::{ExpressionPredicate, FilterExpression, SessionContext};
use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::GraphStoreSearch;
use crate::graph::lpg::{Edge, Node};
use grafeo_common::types::{EpochId, LogicalType, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::FxHashMap;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// A projection expression.
#[non_exhaustive]
pub enum ProjectExpr {
    /// Reference to an input column.
    Column(usize),
    /// A constant value.
    Constant(Value),
    /// Property access on a node/edge column.
    PropertyAccess {
        /// The column containing the node or edge ID.
        column: usize,
        /// The property name to access.
        property: String,
    },
    /// Edge type accessor (for type(r) function).
    EdgeType {
        /// The column containing the edge ID.
        column: usize,
    },
    /// Full expression evaluation (for CASE WHEN, etc.).
    Expression {
        /// The filter expression to evaluate.
        expr: FilterExpression,
        /// Variable name to column index mapping.
        variable_columns: HashMap<String, usize>,
    },
    /// Resolve a node ID column to a full node map with metadata and properties.
    NodeResolve {
        /// The column containing the node ID.
        column: usize,
    },
    /// Resolve an edge ID column to a full edge map with metadata and properties.
    EdgeResolve {
        /// The column containing the edge ID.
        column: usize,
    },
    /// Returns the first non-null value from two columns (used for RIGHT/FULL join dedup).
    Coalesce {
        /// Primary column index.
        first: usize,
        /// Fallback column index.
        second: usize,
    },
}

/// A project operator that selects and transforms columns.
pub struct ProjectOperator {
    /// Child operator to read from.
    child: Box<dyn Operator>,
    /// Projection expressions.
    projections: Vec<ProjectExpr>,
    /// Output column types.
    output_types: Vec<LogicalType>,
    /// Optional store for property access.
    store: Option<Arc<dyn GraphStoreSearch>>,
    /// Transaction ID for MVCC-aware property lookups.
    transaction_id: Option<TransactionId>,
    /// Viewing epoch for MVCC-aware property lookups.
    viewing_epoch: Option<EpochId>,
    /// Session context for introspection functions in expression evaluation.
    session_context: SessionContext,
    /// Materialized output retained when one input chunk contains multiple
    /// runtime expression schemas.
    pending_output: Option<PendingProjectedOutput>,
}

struct PendingProjectedOutput {
    source: DataChunk,
    run_bounds: Vec<(usize, usize)>,
    certificate_words: usize,
    certificates: Vec<u64>,
    next_run: usize,
}

impl PendingProjectedOutput {
    fn next_chunk(&mut self) -> DataChunk {
        let (start, end) = self.run_bounds[self.next_run];
        let schema: Vec<LogicalType> = self
            .source
            .columns()
            .iter()
            .enumerate()
            .map(|(column_idx, column)| {
                let certificate = (self.certificates
                    [start * self.certificate_words + column_idx / 32]
                    >> ((column_idx % 32) * 2))
                    & 0b11;
                if matches!(column.data_type(), LogicalType::Any) {
                    match certificate {
                        1 => LogicalType::List(Box::new(LogicalType::Edge)),
                        2 => LogicalType::Edge,
                        3 => LogicalType::Node,
                        _ => column.data_type().clone(),
                    }
                } else {
                    column.data_type().clone()
                }
            })
            .collect();
        let mut output = DataChunk::with_capacity(&schema, end - start);
        for (column_idx, source) in self.source.columns().iter().enumerate() {
            if let Some(target) = output.column_mut(column_idx) {
                for row in start..end {
                    if let Some(value) = source.get_value(row) {
                        target.push_value(value);
                    }
                }
            }
        }
        output.set_count(end - start);
        self.next_run += 1;
        output
    }
}

impl ProjectOperator {
    /// Creates a new project operator.
    ///
    /// # Panics
    ///
    /// Panics if `projections` and `output_types` have different lengths.
    pub fn new(
        child: Box<dyn Operator>,
        projections: Vec<ProjectExpr>,
        output_types: Vec<LogicalType>,
    ) -> Self {
        assert_eq!(projections.len(), output_types.len());
        Self {
            child,
            projections,
            output_types,
            store: None,
            transaction_id: None,
            viewing_epoch: None,
            session_context: SessionContext::default(),
            pending_output: None,
        }
    }

    /// Creates a new project operator with store access for property lookups.
    ///
    /// # Panics
    ///
    /// Panics if `projections` and `output_types` have different lengths.
    pub fn with_store(
        child: Box<dyn Operator>,
        projections: Vec<ProjectExpr>,
        output_types: Vec<LogicalType>,
        store: Arc<dyn GraphStoreSearch>,
    ) -> Self {
        assert_eq!(projections.len(), output_types.len());
        Self {
            child,
            projections,
            output_types,
            store: Some(store),
            transaction_id: None,
            viewing_epoch: None,
            session_context: SessionContext::default(),
            pending_output: None,
        }
    }

    /// Sets the transaction context for MVCC-aware property lookups.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Sets the session context for introspection functions.
    pub fn with_session_context(mut self, context: SessionContext) -> Self {
        self.session_context = context;
        self
    }

    /// Decomposes this operator into its child and projections for push-based conversion.
    pub fn into_parts(self) -> (Box<dyn Operator>, Vec<ProjectExpr>, Vec<LogicalType>) {
        (self.child, self.projections, self.output_types)
    }

    /// Creates a project operator that selects specific columns.
    pub fn select_columns(
        child: Box<dyn Operator>,
        columns: Vec<usize>,
        types: Vec<LogicalType>,
    ) -> Self {
        let projections = columns.into_iter().map(ProjectExpr::Column).collect();
        Self::new(child, projections, types)
    }
}

impl Operator for ProjectOperator {
    fn next(&mut self) -> OperatorResult {
        if let Some(mut pending) = self.pending_output.take() {
            let output = pending.next_chunk();
            if pending.next_run < pending.run_bounds.len() {
                self.pending_output = Some(pending);
            }
            return Ok(Some(output));
        }

        // Get next chunk from child
        let Some(input) = self.child.next()? else {
            return Ok(None);
        };

        // A path edge list is represented as a generic Value list, but its
        // element type still carries useful provenance in the column schema.
        // Preserve that type when an alias/pass-through projection is
        // configured as Any so later list comprehensions can resolve edge
        // properties without relying on planner-global names.
        let output_types: Vec<LogicalType> = self
            .projections
            .iter()
            .enumerate()
            .map(|(output_idx, projection)| {
                let configured = &self.output_types[output_idx];
                if !matches!(configured, LogicalType::Any)
                    || !matches!(projection, ProjectExpr::Column(_))
                {
                    return configured.clone();
                }
                if let ProjectExpr::Column(input_idx) = projection
                    && let Some(data_type) =
                        input.column(*input_idx).map(|column| column.data_type())
                    && (matches!(data_type, LogicalType::Node | LogicalType::Edge)
                        || matches!(
                            data_type,
                            LogicalType::List(element) if element.as_ref() == &LogicalType::Edge
                        ))
                {
                    return data_type.clone();
                }
                configured.clone()
            })
            .collect();

        // Create output chunk
        let mut output = DataChunk::with_capacity(&output_types, input.row_count());
        let certificate_words = self.projections.len().div_ceil(32);
        let mut certified_edges: Option<Vec<u64>> = None;

        // Evaluate each projection
        for (i, proj) in self.projections.iter().enumerate() {
            match proj {
                ProjectExpr::Column(col_idx) => {
                    // Copy column from input to output
                    let input_col = input.column(*col_idx).ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("Column {col_idx}"))
                    })?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    if input.selection().is_none() && output_col.try_extend_from(input_col) {
                        continue;
                    }
                    for row in input.selected_indices() {
                        if let Some(value) = input_col.get_value(row) {
                            output_col.push_value(value);
                        }
                    }
                }
                ProjectExpr::Constant(value) => {
                    // Push constant for each row
                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");
                    for _ in input.selected_indices() {
                        output_col.push_value(value.clone());
                    }
                }
                ProjectExpr::PropertyAccess { column, property } => {
                    // Access property from node/edge in the specified column
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution("Store required for property access".to_string())
                    })?;

                    // Extract property for each row.
                    // For typed columns (VectorData::NodeId / EdgeId) there is
                    // no ambiguity. For Generic/Any columns (e.g. after a hash
                    // join), both get_node_id and get_edge_id can succeed on the
                    // same Int64 value, so we verify against the store to resolve
                    // the entity type.
                    let prop_key = PropertyKey::new(property);
                    let epoch = self.viewing_epoch;
                    let tx_id = self.transaction_id;
                    for row in input.selected_indices() {
                        let value = if let Some(node_id) = input_col.get_node_id(row) {
                            let snap_epoch = epoch.unwrap_or_else(|| store.current_epoch());
                            if let Some(prop) = store
                                .read_node_property_visible(node_id, &prop_key, snap_epoch, tx_id)
                            {
                                prop
                            } else if let Some(edge_id) = input_col.get_edge_id(row) {
                                // Node lookup returned no property: the ID may belong to an
                                // edge (common with Generic columns after joins).
                                store
                                    .read_edge_property_visible(
                                        edge_id, &prop_key, snap_epoch, tx_id,
                                    )
                                    .unwrap_or(Value::Null)
                            } else {
                                Value::Null
                            }
                        } else if let Some(edge_id) = input_col.get_edge_id(row) {
                            let snap_epoch = epoch.unwrap_or_else(|| store.current_epoch());
                            store
                                .read_edge_property_visible(edge_id, &prop_key, snap_epoch, tx_id)
                                .unwrap_or(Value::Null)
                        } else if let Some(Value::Map(map)) = input_col.get_value(row) {
                            map.get(&prop_key).cloned().unwrap_or(Value::Null)
                        } else {
                            Value::Null
                        };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::EdgeType { column } => {
                    // Get edge type string from an edge column
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution("Store required for edge type access".to_string())
                    })?;

                    let epoch = self.viewing_epoch;
                    let tx_id = self.transaction_id;
                    for row in input.selected_indices() {
                        let value = if let Some(edge_id) = input_col.get_edge_id(row) {
                            let etype = if let (Some(ep), Some(tx)) = (epoch, tx_id) {
                                store.edge_type_versioned(edge_id, ep, tx)
                            } else {
                                store.edge_type(edge_id)
                            };
                            etype.map_or(Value::Null, Value::String)
                        } else {
                            Value::Null
                        };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::Expression {
                    expr,
                    variable_columns,
                } => {
                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution(
                            "Store required for expression evaluation".to_string(),
                        )
                    })?;

                    // Use the ExpressionPredicate for expression evaluation
                    let mut evaluator = ExpressionPredicate::new(
                        expr.clone(),
                        variable_columns.clone(),
                        Arc::clone(store),
                    )
                    .with_session_context(self.session_context.clone());
                    if let (Some(ep), tx_id) = (self.viewing_epoch, self.transaction_id) {
                        evaluator = evaluator.with_transaction_context(ep, tx_id);
                    }

                    for (selected_row, row) in input.selected_indices().enumerate() {
                        let (value, provenance) = evaluator
                            .eval_at_with_type(&input, row)
                            .map_err(|error| super::OperatorError::Execution(error.to_string()))?;
                        if matches!(output_types[i], LogicalType::Any) {
                            let certificate: u64 = match provenance.as_ref() {
                                Some(LogicalType::List(element))
                                    if element.as_ref() == &LogicalType::Edge =>
                                {
                                    1
                                }
                                Some(LogicalType::Edge) => 2,
                                Some(LogicalType::Node) => 3,
                                _ => 0,
                            };
                            if certificate == 0 {
                                output_col.push_value(value.unwrap_or(Value::Null));
                                continue;
                            }
                            let certificates = certified_edges.get_or_insert_with(|| {
                                vec![0; input.row_count() * certificate_words]
                            });
                            let shift = (i % 32) * 2;
                            certificates[selected_row * certificate_words + i / 32] |=
                                certificate << shift;
                        }
                        output_col.push_value(value.unwrap_or(Value::Null));
                    }
                }
                ProjectExpr::NodeResolve { column } => {
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution("Store required for node resolution".to_string())
                    })?;

                    let epoch = self.viewing_epoch;
                    let tx_id = self.transaction_id;
                    for row in input.selected_indices() {
                        let value = if let Some(node_id) = input_col.get_node_id(row) {
                            let snap_epoch = epoch.unwrap_or_else(|| store.current_epoch());
                            // Resolve existence + labels from the committed version chain.
                            let node = if let (Some(ep), Some(tx)) = (epoch, tx_id) {
                                store.get_node_versioned(node_id, ep, tx)
                            } else if let Some(ep) = epoch {
                                store.get_node_at_epoch(node_id, ep)
                            } else {
                                store.get_node(node_id)
                            };
                            // Build properties and labels from their respective
                            // delta-aware accessors so that a writing transaction
                            // sees its own buffered writes (read-your-writes for
                            // RETURN n). When the delta is empty this returns exactly
                            // the committed map — behavior is preserved until Task 4
                            // buffers writes.
                            node.map_or(Value::Null, |n| {
                                let props =
                                    store.read_node_properties_visible(node_id, snap_epoch, tx_id);
                                let label_names =
                                    store.read_node_labels_visible(node_id, snap_epoch, tx_id);
                                node_to_map_with_properties(&n, props, label_names)
                            })
                        } else {
                            Value::Null
                        };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::EdgeResolve { column } => {
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution("Store required for edge resolution".to_string())
                    })?;

                    let epoch = self.viewing_epoch;
                    let tx_id = self.transaction_id;
                    for row in input.selected_indices() {
                        let value = if let Some(edge_id) = input_col.get_edge_id(row) {
                            let snap_epoch = epoch.unwrap_or_else(|| store.current_epoch());
                            // Resolve existence + type from the committed version chain.
                            let edge = if let (Some(ep), Some(tx)) = (epoch, tx_id) {
                                store.get_edge_versioned(edge_id, ep, tx)
                            } else if let Some(ep) = epoch {
                                store.get_edge_at_epoch(edge_id, ep)
                            } else {
                                store.get_edge(edge_id)
                            };
                            // Build properties from the delta-aware accessor.
                            edge.map_or(Value::Null, |e| {
                                let props =
                                    store.read_edge_properties_visible(edge_id, snap_epoch, tx_id);
                                edge_to_map_with_properties(&e, props)
                            })
                        } else {
                            Value::Null
                        };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::Coalesce { first, second } => {
                    let first_col = input
                        .column(*first)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {first}")))?;
                    let second_col = input
                        .column(*second)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {second}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    for row in input.selected_indices() {
                        let value = match first_col.get_value(row) {
                            Some(Value::Null) | None => {
                                second_col.get_value(row).unwrap_or(Value::Null)
                            }
                            Some(v) => v,
                        };
                        output_col.push_value(value);
                    }
                }
            }
        }

        output.set_count(input.row_count());

        let Some(certified_edges) = certified_edges else {
            return Ok(Some(output));
        };
        let row_count = input.row_count();
        let mut run_bounds = Vec::new();
        let mut run_start = 0;
        for row in 1..row_count {
            if certified_edges[row * certificate_words..(row + 1) * certificate_words]
                != certified_edges
                    [run_start * certificate_words..(run_start + 1) * certificate_words]
            {
                run_bounds.push((run_start, row));
                run_start = row;
            }
        }
        if row_count > 0 {
            run_bounds.push((run_start, row_count));
        }
        let mut pending = PendingProjectedOutput {
            source: output,
            run_bounds,
            certificate_words,
            certificates: certified_edges,
            next_run: 0,
        };
        let first = pending.next_chunk();
        if pending.next_run < pending.run_bounds.len() {
            self.pending_output = Some(pending);
        }
        Ok(Some(first))
    }

    fn reset(&mut self) {
        self.pending_output = None;
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        "Project"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn install_resource_context(
        &mut self,
        resources: &crate::execution::QueryResourceContext,
    ) -> Result<(), crate::execution::QueryResourceContextError> {
        self.child.install_resource_context(resources)
    }
}

/// Converts a [`Node`] to a `Value::Map` with metadata and properties.
///
/// Builds a `Value::Map` for a node using a supplied (snapshot-merged) property
/// map instead of the node's own committed `properties` field, and a supplied
/// snapshot-aware label name set instead of the node's committed `labels` field.
///
/// Callers must obtain `label_names` from `store.read_node_labels_visible` so
/// that a writing transaction's uncommitted label adds/removes are reflected here.
/// When the delta is empty (all tasks before Task 4) the set matches the
/// committed labels exactly — behavior is preserved.
fn node_to_map_with_properties(
    node: &Node,
    props: FxHashMap<PropertyKey, Value>,
    label_names: grafeo_common::utils::hash::FxHashSet<arcstr::ArcStr>,
) -> Value {
    let mut map = BTreeMap::new();
    // reason: entity IDs stored as i64, standard encoding
    #[allow(clippy::cast_possible_wrap)]
    let node_id_i64 = node.id.as_u64() as i64;
    map.insert(PropertyKey::new("_id"), Value::Int64(node_id_i64));
    // Use the snapshot-aware label set rather than node.labels so that
    // uncommitted label ops in the writing transaction are reflected.
    let labels: Vec<Value> = label_names.into_iter().map(Value::String).collect();
    map.insert(PropertyKey::new("_labels"), Value::List(labels.into()));
    for (key, value) in props {
        map.insert(key, value);
    }
    Value::Map(Arc::new(map))
}

/// Builds a `Value::Map` for an edge using a supplied (snapshot-merged) property
/// map instead of the edge's own committed `properties` field.
///
/// Edge twin of [`node_to_map_with_properties`].
fn edge_to_map_with_properties(edge: &Edge, props: FxHashMap<PropertyKey, Value>) -> Value {
    let mut map = BTreeMap::new();
    // reason: entity IDs stored as i64, standard encoding
    #[allow(clippy::cast_possible_wrap)]
    let edge_id_i64 = edge.id.as_u64() as i64;
    // reason: entity IDs stored as i64, standard encoding
    #[allow(clippy::cast_possible_wrap)]
    let src_id_i64 = edge.src.as_u64() as i64;
    // reason: entity IDs stored as i64, standard encoding
    #[allow(clippy::cast_possible_wrap)]
    let dst_id_i64 = edge.dst.as_u64() as i64;
    map.insert(PropertyKey::new("_id"), Value::Int64(edge_id_i64));
    map.insert(
        PropertyKey::new("_type"),
        Value::String(edge.edge_type.clone()),
    );
    map.insert(PropertyKey::new("_source"), Value::Int64(src_id_i64));
    map.insert(PropertyKey::new("_target"), Value::Int64(dst_id_i64));
    for (key, value) in props {
        map.insert(key, value);
    }
    Value::Map(Arc::new(map))
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use crate::execution::selection::SelectionVector;
    use crate::graph::lpg::LpgStore;
    use grafeo_common::types::Value;

    struct MockScanOperator {
        chunks: Vec<DataChunk>,
        position: usize,
    }

    impl Operator for MockScanOperator {
        fn next(&mut self) -> OperatorResult {
            if self.position < self.chunks.len() {
                let chunk = std::mem::replace(&mut self.chunks[self.position], DataChunk::empty());
                self.position += 1;
                Ok(Some(chunk))
            } else {
                Ok(None)
            }
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "MockScan"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    struct ResettableScanOperator {
        chunk: DataChunk,
        returned: bool,
    }

    impl Operator for ResettableScanOperator {
        fn next(&mut self) -> OperatorResult {
            if self.returned {
                Ok(None)
            } else {
                self.returned = true;
                Ok(Some(self.chunk.clone()))
            }
        }

        fn reset(&mut self) {
            self.returned = false;
        }

        fn name(&self) -> &'static str {
            "ResettableMockScan"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    fn edge_list(values: &[i64]) -> Value {
        Value::List(values.iter().copied().map(Value::Int64).collect())
    }

    fn mixed_case_projection() -> ProjectExpr {
        case_projection("flag", "edges", "ints", 0)
    }

    fn case_projection(flag: &str, edges: &str, ints: &str, input_start: usize) -> ProjectExpr {
        ProjectExpr::Expression {
            expr: FilterExpression::Case {
                operand: None,
                when_clauses: vec![(
                    FilterExpression::Variable(flag.into()),
                    FilterExpression::Variable(edges.into()),
                )],
                else_clause: Some(Box::new(FilterExpression::Variable(ints.into()))),
            },
            variable_columns: HashMap::from([
                (flag.into(), input_start),
                (edges.into(), input_start + 1),
                (ints.into(), input_start + 2),
            ]),
        }
    }

    fn scalar_case_projection(
        flag: &str,
        edge: &str,
        ints: &str,
        input_start: usize,
    ) -> ProjectExpr {
        ProjectExpr::Expression {
            expr: FilterExpression::Case {
                operand: None,
                when_clauses: vec![(
                    FilterExpression::Variable(flag.into()),
                    FilterExpression::Variable(edge.into()),
                )],
                else_clause: Some(Box::new(FilterExpression::Variable(ints.into()))),
            },
            variable_columns: HashMap::from([
                (flag.into(), input_start),
                (edge.into(), input_start + 1),
                (ints.into(), input_start + 2),
            ]),
        }
    }

    fn mixed_case_chunk(flags: &[bool]) -> DataChunk {
        let edge_type = LogicalType::List(Box::new(LogicalType::Edge));
        let int_list_type = LogicalType::List(Box::new(LogicalType::Int64));
        let mut builder = DataChunkBuilder::new(&[LogicalType::Bool, edge_type, int_list_type]);
        for (row, flag) in flags.iter().copied().enumerate() {
            builder.column_mut(0).unwrap().push_value(Value::Bool(flag));
            builder
                .column_mut(1)
                .unwrap()
                .push_value(edge_list(&[100 + i64::try_from(row).unwrap()]));
            builder
                .column_mut(2)
                .unwrap()
                .push_value(edge_list(&[200 + i64::try_from(row).unwrap()]));
            builder.advance_row();
        }
        builder.finish()
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn test_project_expression_certificates_cross_bitset_words() {
        let edge_type = LogicalType::List(Box::new(LogicalType::Edge));
        let int_list_type = LogicalType::List(Box::new(LogicalType::Int64));
        let mut input = DataChunkBuilder::new(&[
            LogicalType::Bool,
            edge_type.clone(),
            int_list_type.clone(),
            LogicalType::Bool,
            edge_type,
            int_list_type,
        ]);
        for (row, (first, second)) in [(true, true), (false, true), (true, false)]
            .into_iter()
            .enumerate()
        {
            input.column_mut(0).unwrap().push_value(Value::Bool(first));
            input
                .column_mut(1)
                .unwrap()
                .push_value(edge_list(&[100 + row as i64]));
            input
                .column_mut(2)
                .unwrap()
                .push_value(edge_list(&[200 + row as i64]));
            input.column_mut(3).unwrap().push_value(Value::Bool(second));
            input
                .column_mut(4)
                .unwrap()
                .push_value(edge_list(&[300 + row as i64]));
            input
                .column_mut(5)
                .unwrap()
                .push_value(edge_list(&[400 + row as i64]));
            input.advance_row();
        }

        let mut projections = vec![case_projection("flag0", "edges0", "ints0", 0)];
        projections
            .extend((1..65).map(|column| ProjectExpr::Constant(Value::Int64(column as i64))));
        projections.push(case_projection("flag65", "edges65", "ints65", 3));
        let output_types = projections
            .iter()
            .enumerate()
            .map(|(column, projection)| {
                if matches!(projection, ProjectExpr::Expression { .. }) {
                    LogicalType::Any
                } else {
                    debug_assert!(column > 0 && column < 65);
                    LogicalType::Int64
                }
            })
            .collect();
        let store = Arc::new(LpgStore::new().unwrap());
        let mut project = ProjectOperator::with_store(
            Box::new(MockScanOperator {
                chunks: vec![input.finish()],
                position: 0,
            }),
            projections,
            output_types,
            store,
        );

        let first = project.next().unwrap().unwrap();
        assert_eq!(first.row_count(), 1);
        assert_eq!(
            first.column(0).unwrap().data_type(),
            &LogicalType::List(Box::new(LogicalType::Edge))
        );
        assert_eq!(
            first.column(0).unwrap().get_value(0),
            Some(edge_list(&[100]))
        );
        assert_eq!(first.column(1).unwrap().get_value(0), Some(Value::Int64(1)));
        assert_eq!(
            first.column(64).unwrap().get_value(0),
            Some(Value::Int64(64))
        );
        assert_eq!(
            first.column(65).unwrap().data_type(),
            &LogicalType::List(Box::new(LogicalType::Edge))
        );
        assert_eq!(
            first.column(65).unwrap().get_value(0),
            Some(edge_list(&[300]))
        );

        let second = project.next().unwrap().unwrap();
        assert_eq!(second.row_count(), 1);
        assert_eq!(second.column(0).unwrap().data_type(), &LogicalType::Any);
        assert_eq!(
            second.column(0).unwrap().get_value(0),
            Some(edge_list(&[201]))
        );
        assert_eq!(
            second.column(64).unwrap().get_value(0),
            Some(Value::Int64(64))
        );
        assert_eq!(
            second.column(65).unwrap().data_type(),
            &LogicalType::List(Box::new(LogicalType::Edge))
        );
        assert_eq!(
            second.column(65).unwrap().get_value(0),
            Some(edge_list(&[301]))
        );

        let third = project.next().unwrap().unwrap();
        assert_eq!(third.row_count(), 1);
        assert_eq!(
            third.column(0).unwrap().data_type(),
            &LogicalType::List(Box::new(LogicalType::Edge))
        );
        assert_eq!(
            third.column(0).unwrap().get_value(0),
            Some(edge_list(&[102]))
        );
        assert_eq!(
            third.column(64).unwrap().get_value(0),
            Some(Value::Int64(64))
        );
        assert_eq!(third.column(65).unwrap().data_type(), &LogicalType::Any);
        assert_eq!(
            third.column(65).unwrap().get_value(0),
            Some(edge_list(&[402]))
        );
        assert!(project.next().unwrap().is_none());
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn test_project_expression_certificates_two_bit_cells_and_scalar_aliases() {
        let edge_list_type = LogicalType::List(Box::new(LogicalType::Edge));
        let int_list_type = LogicalType::List(Box::new(LogicalType::Int64));
        let mut input = DataChunkBuilder::new(&[
            LogicalType::Bool,
            edge_list_type,
            int_list_type,
            LogicalType::Bool,
            LogicalType::Edge,
            LogicalType::Int64,
        ]);
        for (row, (first, second)) in [(true, true), (false, true), (true, false), (true, false)]
            .into_iter()
            .enumerate()
        {
            input.column_mut(0).unwrap().push_value(Value::Bool(first));
            input
                .column_mut(1)
                .unwrap()
                .push_value(edge_list(&[100 + row as i64]));
            input
                .column_mut(2)
                .unwrap()
                .push_value(edge_list(&[200 + row as i64]));
            input.column_mut(3).unwrap().push_value(Value::Bool(second));
            input
                .column_mut(4)
                .unwrap()
                .push_value(Value::Int64(300 + row as i64));
            input
                .column_mut(5)
                .unwrap()
                .push_value(Value::Int64(400 + row as i64));
            input.advance_row();
        }
        let mut chunk = input.finish();
        chunk.set_selection(SelectionVector::from_predicate(4, |row| {
            row == 1 || row == 3
        }));

        let mut projections = vec![case_projection("flag0", "edges0", "ints0", 0)];
        projections
            .extend((1..32).map(|column| ProjectExpr::Constant(Value::Int64(column as i64))));
        projections.push(scalar_case_projection("flag32", "edge32", "ints32", 3));
        projections.push(ProjectExpr::Constant(Value::Int64(33)));
        let output_types = projections
            .iter()
            .map(|projection| {
                if matches!(projection, ProjectExpr::Expression { .. }) {
                    LogicalType::Any
                } else {
                    LogicalType::Int64
                }
            })
            .collect();
        let store = Arc::new(LpgStore::new().unwrap());
        let mut project = ProjectOperator::with_store(
            Box::new(ResettableScanOperator {
                chunk,
                returned: false,
            }),
            projections,
            output_types,
            store,
        );

        let first = project.next().unwrap().unwrap();
        assert_eq!(first.row_count(), 1);
        assert_eq!(first.column(0).unwrap().data_type(), &LogicalType::Any);
        assert_eq!(
            first.column(0).unwrap().get_value(0),
            Some(edge_list(&[201]))
        );
        assert_eq!(first.column(1).unwrap().get_value(0), Some(Value::Int64(1)));
        assert_eq!(
            first.column(31).unwrap().get_value(0),
            Some(Value::Int64(31))
        );
        assert_eq!(first.column(32).unwrap().data_type(), &LogicalType::Edge);
        assert_eq!(
            first.column(32).unwrap().get_value(0),
            Some(Value::Int64(301))
        );
        assert_eq!(
            first.column(33).unwrap().get_value(0),
            Some(Value::Int64(33))
        );

        project.reset();
        let replay = project.next().unwrap().unwrap();
        assert_eq!(replay.row_count(), 1);
        assert_eq!(replay.column(0).unwrap().data_type(), &LogicalType::Any);
        assert_eq!(
            replay.column(0).unwrap().get_value(0),
            Some(edge_list(&[201]))
        );
        assert_eq!(replay.column(32).unwrap().data_type(), &LogicalType::Edge);
        assert_eq!(
            replay.column(32).unwrap().get_value(0),
            Some(Value::Int64(301))
        );
        let replay_second = project.next().unwrap().unwrap();
        assert_eq!(replay_second.row_count(), 1);
        assert_eq!(
            replay_second.column(0).unwrap().data_type(),
            &LogicalType::List(Box::new(LogicalType::Edge))
        );
        assert_eq!(
            replay_second.column(0).unwrap().get_value(0),
            Some(edge_list(&[103]))
        );
        assert_eq!(
            replay_second.column(32).unwrap().data_type(),
            &LogicalType::Any
        );
        assert_eq!(
            replay_second.column(32).unwrap().get_value(0),
            Some(Value::Int64(403))
        );
        assert!(project.next().unwrap().is_none());
    }

    #[test]
    fn test_project_column_preserves_node_and_edge_alias_types_without_numeric_inference() {
        use grafeo_common::types::{EdgeId, NodeId};
        let node = NodeId::new(7);
        let edge = EdgeId::new(7);
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node, LogicalType::Edge]);
        builder.column_mut(0).unwrap().push_node_id(node);
        builder.column_mut(1).unwrap().push_edge_id(edge);
        builder.advance_row();
        let mut project = ProjectOperator::new(
            Box::new(MockScanOperator {
                chunks: vec![builder.finish()],
                position: 0,
            }),
            vec![ProjectExpr::Column(0), ProjectExpr::Column(1)],
            vec![LogicalType::Any, LogicalType::Any],
        );
        let result = project.next().unwrap().unwrap();
        assert_eq!(result.column(0).unwrap().data_type(), &LogicalType::Node);
        assert_eq!(result.column(1).unwrap().data_type(), &LogicalType::Edge);
        assert_eq!(result.column(0).unwrap().get_node_id(0), Some(node));
        assert_eq!(result.column(1).unwrap().get_edge_id(0), Some(edge));
    }

    #[test]
    fn test_project_expression_preserves_certified_edge_list_runs() {
        let mock_scan = MockScanOperator {
            chunks: vec![mixed_case_chunk(&[true, false, true])],
            position: 0,
        };
        let store = Arc::new(LpgStore::new().unwrap());
        let mut project = ProjectOperator::with_store(
            Box::new(mock_scan),
            vec![mixed_case_projection()],
            vec![LogicalType::Any],
            store,
        );

        let edge_type = LogicalType::List(Box::new(LogicalType::Edge));
        let first = project.next().unwrap().unwrap();
        assert_eq!(first.row_count(), 1);
        assert_eq!(first.column(0).unwrap().data_type(), &edge_type);
        assert_eq!(
            first.column(0).unwrap().get_value(0),
            Some(edge_list(&[100]))
        );

        let second = project.next().unwrap().unwrap();
        assert_eq!(second.row_count(), 1);
        assert_eq!(second.column(0).unwrap().data_type(), &LogicalType::Any);
        assert_eq!(
            second.column(0).unwrap().get_value(0),
            Some(edge_list(&[201]))
        );

        let third = project.next().unwrap().unwrap();
        assert_eq!(third.row_count(), 1);
        assert_eq!(third.column(0).unwrap().data_type(), &edge_type);
        assert_eq!(
            third.column(0).unwrap().get_value(0),
            Some(edge_list(&[102]))
        );
        assert!(project.next().unwrap().is_none());
    }

    #[test]
    fn test_project_expression_honors_sparse_selection_and_reset() {
        let mut chunk = mixed_case_chunk(&[true, false, false, true]);
        chunk.set_selection(SelectionVector::from_predicate(4, |row| {
            row == 1 || row == 3
        }));
        let store = Arc::new(LpgStore::new().unwrap());
        let mut project = ProjectOperator::with_store(
            Box::new(ResettableScanOperator {
                chunk,
                returned: false,
            }),
            vec![mixed_case_projection()],
            vec![LogicalType::Any],
            store,
        );

        let first = project.next().unwrap().unwrap();
        assert_eq!(first.row_count(), 1);
        assert_eq!(
            first.column(0).unwrap().get_value(0),
            Some(edge_list(&[201]))
        );
        assert_eq!(first.column(0).unwrap().data_type(), &LogicalType::Any);
        // Reset while the second homogeneous run is still pending; stale
        // output must not be emitted after the child is rewound.
        project.reset();
        let replay = project.next().unwrap().unwrap();
        assert_eq!(replay.row_count(), 1);
        assert_eq!(
            replay.column(0).unwrap().get_value(0),
            Some(edge_list(&[201]))
        );
        let replay_second = project.next().unwrap().unwrap();
        assert_eq!(replay_second.row_count(), 1);
        assert_eq!(
            replay_second.column(0).unwrap().get_value(0),
            Some(edge_list(&[103]))
        );
        assert_eq!(
            replay_second.column(0).unwrap().data_type(),
            &LogicalType::List(Box::new(LogicalType::Edge))
        );
        assert!(project.next().unwrap().is_none());
    }

    #[test]
    fn test_project_expression_certification_reaches_downstream_property_projection() {
        let store = Arc::new(LpgStore::new().unwrap());
        let source = store.create_node(&[]);
        let target = store.create_node(&[]);
        let edge = store.create_edge(source, target, "REL");
        store.set_edge_property(edge, "cost", Value::Int64(7));

        let mut input = mixed_case_chunk(&[true]);
        input.column_mut(1).unwrap().clear();
        input
            .column_mut(1)
            .unwrap()
            .push_value(edge_list(&[i64::try_from(edge.as_u64()).unwrap()]));
        let first = ProjectOperator::with_store(
            Box::new(MockScanOperator {
                chunks: vec![input],
                position: 0,
            }),
            vec![mixed_case_projection()],
            vec![LogicalType::Any],
            store.clone(),
        );
        let mut first = first;
        let edge_chunk = first.next().unwrap().unwrap();
        assert_eq!(
            edge_chunk.column(0).unwrap().data_type(),
            &LogicalType::List(Box::new(LogicalType::Edge))
        );

        let second_expr = FilterExpression::ListComprehension {
            variable: "e".into(),
            list_expr: Box::new(FilterExpression::Variable("es".into())),
            filter_expr: None,
            map_expr: Box::new(FilterExpression::Property {
                variable: "e".into(),
                property: "cost".into(),
            }),
        };
        let mut second = ProjectOperator::with_store(
            Box::new(MockScanOperator {
                chunks: vec![edge_chunk],
                position: 0,
            }),
            vec![ProjectExpr::Expression {
                expr: second_expr,
                variable_columns: HashMap::from([("es".into(), 0)]),
            }],
            vec![LogicalType::Any],
            store,
        );
        let result = second.next().unwrap().unwrap();
        assert_eq!(
            result.column(0).unwrap().get_value(0),
            Some(Value::List(vec![Value::Int64(7)].into()))
        );
    }

    #[test]
    fn test_project_select_columns() {
        // Create input with 3 columns: [int, string, int]
        let mut builder =
            DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String, LogicalType::Int64]);

        builder.column_mut(0).unwrap().push_int64(1);
        builder.column_mut(1).unwrap().push_string("hello");
        builder.column_mut(2).unwrap().push_int64(100);
        builder.advance_row();

        builder.column_mut(0).unwrap().push_int64(2);
        builder.column_mut(1).unwrap().push_string("world");
        builder.column_mut(2).unwrap().push_int64(200);
        builder.advance_row();

        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Project to select columns 2 and 0 (reordering)
        let mut project = ProjectOperator::select_columns(
            Box::new(mock_scan),
            vec![2, 0],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let result = project.next().unwrap().unwrap();

        assert_eq!(result.column_count(), 2);
        assert_eq!(result.row_count(), 2);

        // Check values are reordered
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(100));
        assert_eq!(result.column(1).unwrap().get_int64(0), Some(1));
    }

    #[test]
    fn test_project_constant() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(1);
        builder.advance_row();
        builder.column_mut(0).unwrap().push_int64(2);
        builder.advance_row();

        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Project with a constant
        let mut project = ProjectOperator::new(
            Box::new(mock_scan),
            vec![
                ProjectExpr::Column(0),
                ProjectExpr::Constant(Value::String("constant".into())),
            ],
            vec![LogicalType::Int64, LogicalType::String],
        );

        let result = project.next().unwrap().unwrap();

        assert_eq!(result.column_count(), 2);
        assert_eq!(result.column(1).unwrap().get_string(0), Some("constant"));
        assert_eq!(result.column(1).unwrap().get_string(1), Some("constant"));
    }

    #[test]
    fn test_project_empty_input() {
        let mock_scan = MockScanOperator {
            chunks: vec![],
            position: 0,
        };

        let mut project =
            ProjectOperator::select_columns(Box::new(mock_scan), vec![0], vec![LogicalType::Int64]);

        assert!(project.next().unwrap().is_none());
    }

    #[test]
    fn test_project_column_not_found() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(1);
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Reference column index 5 which doesn't exist
        let mut project = ProjectOperator::new(
            Box::new(mock_scan),
            vec![ProjectExpr::Column(5)],
            vec![LogicalType::Int64],
        );

        let result = project.next();
        assert!(result.is_err(), "Should fail with ColumnNotFound");
    }

    #[test]
    fn test_project_multiple_constants() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(1);
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut project = ProjectOperator::new(
            Box::new(mock_scan),
            vec![
                ProjectExpr::Constant(Value::Int64(42)),
                ProjectExpr::Constant(Value::String("fixed".into())),
                ProjectExpr::Constant(Value::Bool(true)),
            ],
            vec![LogicalType::Int64, LogicalType::String, LogicalType::Bool],
        );

        let result = project.next().unwrap().unwrap();
        assert_eq!(result.column_count(), 3);
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(42));
        assert_eq!(result.column(1).unwrap().get_string(0), Some("fixed"));
        assert_eq!(
            result.column(2).unwrap().get_value(0),
            Some(Value::Bool(true))
        );
    }

    #[test]
    fn test_project_identity() {
        // Select all columns in original order
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String]);
        builder.column_mut(0).unwrap().push_int64(10);
        builder.column_mut(1).unwrap().push_string("test");
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut project = ProjectOperator::select_columns(
            Box::new(mock_scan),
            vec![0, 1],
            vec![LogicalType::Int64, LogicalType::String],
        );

        let result = project.next().unwrap().unwrap();
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(10));
        assert_eq!(result.column(1).unwrap().get_string(0), Some("test"));
    }

    #[test]
    fn test_project_column_preserves_edge_list_type_when_output_is_any() {
        let edge_list_type = LogicalType::List(Box::new(LogicalType::Edge));
        let edge_list = Value::List(
            vec![Value::Int64(11), Value::Int64(12)]
                .into_iter()
                .collect::<Vec<_>>()
                .into(),
        );
        let mut builder = DataChunkBuilder::new(std::slice::from_ref(&edge_list_type));
        builder.column_mut(0).unwrap().push_value(edge_list.clone());
        builder.advance_row();

        let mock_scan = MockScanOperator {
            chunks: vec![builder.finish()],
            position: 0,
        };
        let mut project = ProjectOperator::new(
            Box::new(mock_scan),
            vec![ProjectExpr::Column(0)],
            vec![LogicalType::Any],
        );

        let result = project.next().unwrap().unwrap();
        assert_eq!(result.column(0).unwrap().data_type(), &edge_list_type);
        assert_eq!(result.column(0).unwrap().get_value(0), Some(edge_list));
    }

    #[test]
    fn test_project_column_does_not_infer_integer_list_as_edge_list() {
        let integer_list_type = LogicalType::List(Box::new(LogicalType::Int64));
        let integer_list = Value::List(
            vec![Value::Int64(11), Value::Int64(12)]
                .into_iter()
                .collect::<Vec<_>>()
                .into(),
        );
        let mut builder = DataChunkBuilder::new(std::slice::from_ref(&integer_list_type));
        builder
            .column_mut(0)
            .unwrap()
            .push_value(integer_list.clone());
        builder.advance_row();

        let mock_scan = MockScanOperator {
            chunks: vec![builder.finish()],
            position: 0,
        };
        let mut project = ProjectOperator::new(
            Box::new(mock_scan),
            vec![ProjectExpr::Column(0)],
            vec![LogicalType::Any],
        );

        let result = project.next().unwrap().unwrap();
        assert_eq!(result.column(0).unwrap().data_type(), &LogicalType::Any);
        assert_eq!(result.column(0).unwrap().get_value(0), Some(integer_list));
    }

    #[test]
    fn test_project_name() {
        let mock_scan = MockScanOperator {
            chunks: vec![],
            position: 0,
        };
        let project =
            ProjectOperator::select_columns(Box::new(mock_scan), vec![0], vec![LogicalType::Int64]);
        assert_eq!(project.name(), "Project");
    }

    #[test]
    // reason: test IDs are small sequential counters
    #[allow(clippy::cast_possible_wrap)]
    fn test_project_node_resolve() {
        // Create a store with a test node
        let store = LpgStore::new().unwrap();
        let node_id = store.create_node(&["Person"]);
        store.set_node_property(node_id, "name", Value::String("Alix".into()));
        store.set_node_property(node_id, "age", Value::Int64(30));

        // Create input chunk with a NodeId column
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        builder.column_mut(0).unwrap().push_node_id(node_id);
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut project = ProjectOperator::with_store(
            Box::new(mock_scan),
            vec![ProjectExpr::NodeResolve { column: 0 }],
            vec![LogicalType::Any],
            Arc::new(store),
        );

        let result = project.next().unwrap().unwrap();
        assert_eq!(result.column_count(), 1);

        let value = result.column(0).unwrap().get_value(0).unwrap();
        if let Value::Map(map) = value {
            assert_eq!(
                map.get(&PropertyKey::new("_id")),
                Some(&Value::Int64(node_id.as_u64() as i64))
            );
            assert!(map.get(&PropertyKey::new("_labels")).is_some());
            assert_eq!(
                map.get(&PropertyKey::new("name")),
                Some(&Value::String("Alix".into()))
            );
            assert_eq!(map.get(&PropertyKey::new("age")), Some(&Value::Int64(30)));
        } else {
            panic!("Expected Value::Map, got {:?}", value);
        }
    }

    #[test]
    // reason: test IDs are small sequential counters
    #[allow(clippy::cast_possible_wrap)]
    fn test_project_edge_resolve() {
        let store = LpgStore::new().unwrap();
        let src = store.create_node(&["Person"]);
        let dst = store.create_node(&["Company"]);
        let edge_id = store.create_edge(src, dst, "WORKS_AT");
        store.set_edge_property(edge_id, "since", Value::Int64(2020));

        // Create input chunk with an EdgeId column
        let mut builder = DataChunkBuilder::new(&[LogicalType::Edge]);
        builder.column_mut(0).unwrap().push_edge_id(edge_id);
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut project = ProjectOperator::with_store(
            Box::new(mock_scan),
            vec![ProjectExpr::EdgeResolve { column: 0 }],
            vec![LogicalType::Any],
            Arc::new(store),
        );

        let result = project.next().unwrap().unwrap();
        let value = result.column(0).unwrap().get_value(0).unwrap();
        if let Value::Map(map) = value {
            assert_eq!(
                map.get(&PropertyKey::new("_id")),
                Some(&Value::Int64(edge_id.as_u64() as i64))
            );
            assert_eq!(
                map.get(&PropertyKey::new("_type")),
                Some(&Value::String("WORKS_AT".into()))
            );
            assert_eq!(
                map.get(&PropertyKey::new("_source")),
                Some(&Value::Int64(src.as_u64() as i64))
            );
            assert_eq!(
                map.get(&PropertyKey::new("_target")),
                Some(&Value::Int64(dst.as_u64() as i64))
            );
            assert_eq!(
                map.get(&PropertyKey::new("since")),
                Some(&Value::Int64(2020))
            );
        } else {
            panic!("Expected Value::Map, got {:?}", value);
        }
    }

    #[test]
    fn test_project_resolve_missing_entity() {
        use grafeo_common::types::NodeId;

        let store = LpgStore::new().unwrap();

        // Create input chunk with a NodeId that doesn't exist in the store
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        builder
            .column_mut(0)
            .unwrap()
            .push_node_id(NodeId::new(999));
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut project = ProjectOperator::with_store(
            Box::new(mock_scan),
            vec![ProjectExpr::NodeResolve { column: 0 }],
            vec![LogicalType::Any],
            Arc::new(store),
        );

        let result = project.next().unwrap().unwrap();
        assert_eq!(result.column(0).unwrap().get_value(0), Some(Value::Null));
    }

    #[test]
    fn test_project_into_any() {
        let mock = MockScanOperator {
            chunks: vec![],
            position: 0,
        };
        let op = ProjectOperator::select_columns(Box::new(mock), vec![0], vec![LogicalType::Int64]);
        let any = Box::new(op).into_any();
        assert!(any.downcast::<ProjectOperator>().is_ok());
    }

    #[test]
    fn test_project_into_parts() {
        let mock = MockScanOperator {
            chunks: vec![],
            position: 0,
        };
        let op = ProjectOperator::new(
            Box::new(mock),
            vec![
                ProjectExpr::Column(0),
                ProjectExpr::Constant(Value::Int64(1)),
            ],
            vec![LogicalType::Int64, LogicalType::Int64],
        );
        let (child, projections, output_types) = op.into_parts();
        assert_eq!(projections.len(), 2);
        assert_eq!(output_types.len(), 2);
        // Verify child is still functional
        let mut child = child;
        assert!(child.next().unwrap().is_none());
    }
}
