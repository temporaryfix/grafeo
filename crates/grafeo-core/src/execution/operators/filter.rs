//! Filter operator for applying predicates.

use super::{Operator, OperatorPipelineDecomposition, OperatorResult};
use crate::execution::{ChunkZoneHints, DataChunk, SelectionVector, ValueVector};
use crate::graph::Direction;
use crate::graph::GraphStoreSearch;
use crate::graph::lpg::{Edge, Node};
use grafeo_common::types::{
    EdgeId, EpochId, HashableValue, LogicalType, NodeId, PropertyKey, TransactionId, Value,
};
use grafeo_common::utils::hash::FxHashMap;
#[cfg(feature = "regex")]
use regex::Regex;
#[cfg(all(feature = "regex-lite", not(feature = "regex")))]
use regex_lite::Regex;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Extracts a required integer field from a temporal constructor map.
///
/// Returns `Some(value)` when the key is present and holds a valid integer
/// (or a finite float that can be truncated to i64). Returns `None` when
/// the key is absent. Rejects non-finite floats (NaN, Infinity) by
/// returning `Some(None)` through `map_int_checked`, which lets callers
/// distinguish "missing" from "invalid".
fn map_int(m: &BTreeMap<PropertyKey, Value>, key: &str) -> Option<i64> {
    match map_int_checked(m, key) {
        MapIntResult::Absent => None,
        MapIntResult::Valid(v) => Some(v),
        // Invalid floats (NaN, Infinity): treat as missing so callers
        // that use map_int for required fields will fail with None.
        MapIntResult::Invalid => None,
    }
}

/// Result of extracting an integer field, distinguishing absent from invalid.
enum MapIntResult {
    /// Key was not present in the map.
    Absent,
    /// Key was present with a valid integer value.
    Valid(i64),
    /// Key was present but the value is not convertible (NaN, Infinity, wrong type).
    Invalid,
}

/// Extracts an integer field with three-way result: absent, valid, or invalid.
fn map_int_checked(m: &BTreeMap<PropertyKey, Value>, key: &str) -> MapIntResult {
    match m.get(&PropertyKey::from(key)) {
        None => MapIntResult::Absent,
        Some(Value::Int64(v)) => MapIntResult::Valid(*v),
        Some(Value::Float64(f)) => {
            let f = *f;
            if f.is_nan() || f.is_infinite() || f > i64::MAX as f64 || f < i64::MIN as f64 {
                MapIntResult::Invalid
            } else {
                // reason: intentional truncation of finite float to integer for temporal field extraction
                #[allow(clippy::cast_possible_truncation)]
                MapIntResult::Valid(f as i64)
            }
        }
        Some(_) => MapIntResult::Invalid,
    }
}

/// Extracts an optional integer field from a temporal constructor map,
/// returning a default value when the key is absent. Returns `None` when
/// the key is present but holds an invalid value (NaN, Infinity).
fn map_int_or(m: &BTreeMap<PropertyKey, Value>, key: &str, default: i64) -> Option<i64> {
    match map_int_checked(m, key) {
        MapIntResult::Absent => Some(default),
        MapIntResult::Valid(v) => Some(v),
        MapIntResult::Invalid => None,
    }
}

/// A predicate for filtering rows.
pub trait Predicate: Send + Sync {
    /// Evaluates the predicate for a single row.
    ///
    /// # Errors
    /// Returns a query execution error instead of treating a failed read as false.
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> Result<bool, super::OperatorError>;

    /// Returns `false` if zone map proves no rows in this chunk can match.
    ///
    /// This method enables chunk-level filtering optimization. When a chunk
    /// has zone map hints attached, the filter operator calls this method
    /// first. If it returns `false`, the entire chunk is skipped without
    /// evaluating any rows.
    ///
    /// The default implementation is conservative and returns `true` (might match).
    /// Predicates that support zone map checking should override this.
    fn might_match_chunk(&self, _hints: &ChunkZoneHints) -> bool {
        true
    }
}

/// Adapts a pull predicate to the push filter contract.
///
/// This remains public through `pipeline_convert` for compatibility, while
/// the operator-owned decomposition hook now constructs it directly.
pub struct PredicateAdapter(pub Box<dyn Predicate>);

impl super::push::FilterPredicate for PredicateAdapter {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> Result<bool, super::OperatorError> {
        self.0.evaluate(chunk, row)
    }
}

/// A comparison operator.
#[cfg(all(test, feature = "lpg"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompareOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
}

/// A simple comparison predicate.
#[cfg(all(test, feature = "lpg"))]
pub(crate) struct ComparisonPredicate {
    /// Column index to compare.
    column: usize,
    /// Comparison operator.
    op: CompareOp,
    /// Value to compare against.
    value: Value,
}

#[cfg(all(test, feature = "lpg"))]
impl ComparisonPredicate {
    /// Creates a new comparison predicate.
    pub(crate) fn new(column: usize, op: CompareOp, value: Value) -> Self {
        Self { column, op, value }
    }
}

#[cfg(all(test, feature = "lpg"))]
impl Predicate for ComparisonPredicate {
    fn evaluate(
        &self,
        chunk: &DataChunk,
        row: usize,
    ) -> Result<bool, crate::execution::operators::OperatorError> {
        Ok((|| {
            let Some(col) = chunk.column(self.column) else {
                return false;
            };

            let Some(cell_value) = col.get_value(row) else {
                return false;
            };

            match (&cell_value, &self.value) {
                (Value::Int64(a), Value::Int64(b)) => match self.op {
                    CompareOp::Eq => a == b,
                    CompareOp::Ne => a != b,
                    CompareOp::Lt => a < b,
                    CompareOp::Le => a <= b,
                    CompareOp::Gt => a > b,
                    CompareOp::Ge => a >= b,
                },
                (Value::Float64(a), Value::Float64(b)) => match self.op {
                    CompareOp::Eq => (a - b).abs() < f64::EPSILON,
                    CompareOp::Ne => (a - b).abs() >= f64::EPSILON,
                    CompareOp::Lt => a < b,
                    CompareOp::Le => a <= b,
                    CompareOp::Gt => a > b,
                    CompareOp::Ge => a >= b,
                },
                (Value::String(a), Value::String(b)) => match self.op {
                    CompareOp::Eq => a == b,
                    CompareOp::Ne => a != b,
                    CompareOp::Lt => a < b,
                    CompareOp::Le => a <= b,
                    CompareOp::Gt => a > b,
                    CompareOp::Ge => a >= b,
                },
                // Cross-type Int64/Float64 coercion
                (Value::Int64(a), Value::Float64(b)) => {
                    let a = *a as f64;
                    match self.op {
                        CompareOp::Eq => (a - b).abs() < f64::EPSILON,
                        CompareOp::Ne => (a - b).abs() >= f64::EPSILON,
                        CompareOp::Lt => a < *b,
                        CompareOp::Le => a <= *b,
                        CompareOp::Gt => a > *b,
                        CompareOp::Ge => a >= *b,
                    }
                }
                (Value::Float64(a), Value::Int64(b)) => {
                    let b = *b as f64;
                    match self.op {
                        CompareOp::Eq => (a - b).abs() < f64::EPSILON,
                        CompareOp::Ne => (a - b).abs() >= f64::EPSILON,
                        CompareOp::Lt => *a < b,
                        CompareOp::Le => *a <= b,
                        CompareOp::Gt => *a > b,
                        CompareOp::Ge => *a >= b,
                    }
                }
                (Value::Bool(a), Value::Bool(b)) => match self.op {
                    CompareOp::Eq => a == b,
                    CompareOp::Ne => a != b,
                    _ => false, // Ordering on booleans doesn't make sense
                },
                _ => false, // Type mismatch
            }
        })())
    }

    fn might_match_chunk(&self, hints: &ChunkZoneHints) -> bool {
        let Some(zone_map) = hints.column_hints.get(&self.column) else {
            return true; // No zone map for this column = conservative
        };

        match self.op {
            CompareOp::Eq => zone_map.might_contain_equal(&self.value),
            CompareOp::Ne => true, // Ne is always conservative (might have non-matching values)
            CompareOp::Lt => zone_map.might_contain_less_than(&self.value, false),
            CompareOp::Le => zone_map.might_contain_less_than(&self.value, true),
            CompareOp::Gt => zone_map.might_contain_greater_than(&self.value, false),
            CompareOp::Ge => zone_map.might_contain_greater_than(&self.value, true),
        }
    }
}

/// An expression-based predicate that evaluates logical expressions.
///
/// This predicate can evaluate complex expressions involving variables,
/// properties, and operators.
pub struct ExpressionPredicate {
    /// The expression to evaluate.
    expression: FilterExpression,
    /// Map from variable name to column index.
    variable_columns: HashMap<String, usize>,
    /// The graph store for property lookups.
    store: Arc<dyn GraphStoreSearch>,
    /// Transaction ID for MVCC-aware lookups.
    transaction_id: Option<TransactionId>,
    /// Viewing epoch for MVCC-aware lookups.
    viewing_epoch: Option<EpochId>,
    /// Session context for introspection functions (info, schema, current_schema, etc.).
    session_context: SessionContext,
}

/// A lazily-computed, cloneable value.
///
/// The factory runs at most once (via `OnceLock`). Cloning is cheap because
/// both the lock and the factory are behind `Arc`.
#[derive(Clone)]
pub struct LazyValue {
    cell: Arc<std::sync::OnceLock<Value>>,
    factory: Arc<dyn Fn() -> Value + Send + Sync>,
}

impl LazyValue {
    /// Creates a new lazy value with the given factory.
    pub fn new(factory: impl Fn() -> Value + Send + Sync + 'static) -> Self {
        Self {
            cell: Arc::new(std::sync::OnceLock::new()),
            factory: Arc::new(factory),
        }
    }

    /// Returns the value, computing it on first access.
    pub fn get(&self) -> &Value {
        self.cell.get_or_init(|| (self.factory)())
    }
}

impl std::fmt::Debug for LazyValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.cell.get() {
            Some(v) => write!(f, "LazyValue({v:?})"),
            None => write!(f, "LazyValue(<not yet computed>)"),
        }
    }
}

impl Default for LazyValue {
    fn default() -> Self {
        Self::new(|| Value::Null)
    }
}

/// Session-level context passed to the filter evaluator for introspection functions.
///
/// Lightweight strings (`current_schema`, `current_graph`) are stored directly.
/// Expensive introspection maps (`db_info`, `schema_info`) are lazily computed
/// on first access, so queries that never call `info()` or `schema()` pay zero cost.
#[derive(Debug, Clone, Default)]
pub struct SessionContext {
    /// Current session schema name (for `CURRENT_SCHEMA`).
    pub current_schema: Option<String>,
    /// Current session graph name (for `CURRENT_GRAPH`).
    pub current_graph: Option<String>,
    /// Lazily-computed `info()` result.
    pub db_info: LazyValue,
    /// Lazily-computed `schema()` result.
    pub schema_info: LazyValue,
}

/// A filter expression that can be evaluated.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum FilterExpression {
    /// A literal value.
    Literal(Value),
    /// A variable reference (column index).
    Variable(String),
    /// Property access on a variable.
    Property {
        /// The variable name.
        variable: String,
        /// The property name.
        property: String,
    },
    /// Binary operation.
    Binary {
        /// Left operand.
        left: Box<FilterExpression>,
        /// Operator.
        op: BinaryFilterOp,
        /// Right operand.
        right: Box<FilterExpression>,
    },
    /// Unary operation.
    Unary {
        /// Operator.
        op: UnaryFilterOp,
        /// Operand.
        operand: Box<FilterExpression>,
    },
    /// Function call.
    FunctionCall {
        /// Function name (e.g., "id", "labels", "type", "size", "coalesce", "exists").
        name: String,
        /// Arguments.
        args: Vec<FilterExpression>,
    },
    /// List literal.
    List(Vec<FilterExpression>),
    /// Map literal (e.g., {name: 'Alix', age: 30}).
    Map(Vec<(String, FilterExpression)>),
    /// Index access (e.g., `list[0]`).
    IndexAccess {
        /// The base expression.
        base: Box<FilterExpression>,
        /// The index expression.
        index: Box<FilterExpression>,
    },
    /// Slice access (e.g., list[1..3]).
    SliceAccess {
        /// The base expression.
        base: Box<FilterExpression>,
        /// Start index (None means from beginning).
        start: Option<Box<FilterExpression>>,
        /// End index (None means to end).
        end: Option<Box<FilterExpression>>,
    },
    /// CASE expression.
    Case {
        /// Test expression (for simple CASE).
        operand: Option<Box<FilterExpression>>,
        /// WHEN clauses (condition, result).
        when_clauses: Vec<(FilterExpression, FilterExpression)>,
        /// ELSE clause.
        else_clause: Option<Box<FilterExpression>>,
    },
    /// Entity ID access.
    Id(String),
    /// Node labels access.
    Labels(String),
    /// Edge type access.
    Type(String),
    /// List comprehension: [x IN list WHERE predicate | expression]
    ListComprehension {
        /// Variable name for each element.
        variable: String,
        /// The source list expression.
        list_expr: Box<FilterExpression>,
        /// Optional filter predicate.
        filter_expr: Option<Box<FilterExpression>>,
        /// The mapping expression for each element.
        map_expr: Box<FilterExpression>,
    },
    /// List predicate: all/any/none/single(x IN list WHERE pred).
    ListPredicate {
        /// The kind of list predicate.
        kind: ListPredicateKind,
        /// The iteration variable name.
        variable: String,
        /// The source list expression.
        list_expr: Box<FilterExpression>,
        /// The predicate to test for each element.
        predicate: Box<FilterExpression>,
    },
    /// EXISTS subquery: evaluates inner plan and returns true if results exist.
    ExistsSubquery {
        /// The start node variable from outer query.
        start_var: String,
        /// Direction of edge traversal.
        direction: Direction,
        /// Edge type filter (empty = match all types, multiple = match any).
        edge_types: Vec<String>,
        /// Optional end node labels filter.
        end_labels: Option<Vec<String>>,
        /// Minimum number of hops (for variable-length patterns).
        min_hops: Option<u32>,
        /// Maximum number of hops (for variable-length patterns).
        max_hops: Option<u32>,
    },
    /// COUNT subquery: counts matching edges from a node (fast path).
    CountSubquery {
        /// The start node variable from outer query.
        start_var: String,
        /// Direction of edge traversal.
        direction: Direction,
        /// Edge type filter (empty = match all types, multiple = match any).
        edge_types: Vec<String>,
        /// Optional end node labels filter.
        end_labels: Option<Vec<String>>,
    },
    /// reduce() accumulator: `reduce(acc = init, x IN list | expr)`.
    Reduce {
        /// Accumulator variable name.
        accumulator: String,
        /// Initial value for the accumulator.
        initial: Box<FilterExpression>,
        /// Iteration variable name.
        variable: String,
        /// List to iterate over.
        list: Box<FilterExpression>,
        /// Body expression (references both accumulator and variable).
        expression: Box<FilterExpression>,
    },
}

/// The kind of list predicate function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ListPredicateKind {
    /// all(x IN list WHERE pred): true if pred holds for every element.
    All,
    /// any(x IN list WHERE pred): true if pred holds for at least one element.
    Any,
    /// none(x IN list WHERE pred): true if pred holds for no element.
    None,
    /// single(x IN list WHERE pred): true if pred holds for exactly one element.
    Single,
}

/// Binary operators for filter expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BinaryFilterOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
    /// Logical AND.
    And,
    /// Logical OR.
    Or,
    /// Logical XOR.
    Xor,
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Modulo.
    Mod,
    /// String starts with.
    StartsWith,
    /// String ends with.
    EndsWith,
    /// String contains.
    Contains,
    /// List membership.
    In,
    /// Regex match (=~).
    Regex,
    /// Power/exponentiation (^).
    Pow,
    /// SQL LIKE pattern matching (% = any chars, _ = single char).
    Like,
    /// String concatenation (||).
    Concat,
}

/// Unary operators for filter expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnaryFilterOp {
    /// Logical NOT.
    Not,
    /// IS NULL.
    IsNull,
    /// IS NOT NULL.
    IsNotNull,
    /// Numeric negation.
    Neg,
}

impl ExpressionPredicate {
    /// Uses the expression evaluator's scalar rules for indexed residuals.
    #[must_use]
    pub fn matches_property_index_predicate(
        value: &Value,
        predicate: crate::graph::PropertyIndexPredicate<'_>,
    ) -> bool {
        ExpressionEvaluation::matches_property_index_predicate(value, predicate)
    }

    /// Creates a new expression predicate.
    pub fn new(
        expression: FilterExpression,
        variable_columns: HashMap<String, usize>,
        store: Arc<dyn GraphStoreSearch>,
    ) -> Self {
        Self {
            expression,
            variable_columns,
            store,
            transaction_id: None,
            viewing_epoch: None,
            session_context: SessionContext::default(),
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

    /// Resolves a node using transaction-aware access when available.
    fn resolve_node(&self, node_id: NodeId) -> Option<Node> {
        if let (Some(ep), Some(tx)) = (self.viewing_epoch, self.transaction_id) {
            self.store.get_node_versioned(node_id, ep, tx)
        } else if let Some(ep) = self.viewing_epoch {
            self.store.get_node_at_epoch(node_id, ep)
        } else {
            self.store.get_node(node_id)
        }
    }

    /// Returns true if an edge (and its other endpoint) matches the type and label filters.
    ///
    /// Used by both `ExistsSubquery` and `CountSubquery` fast-path evaluation.
    fn edge_matches(
        &self,
        other_node_id: NodeId,
        edge_id: EdgeId,
        edge_types: &[String],
        end_labels: &Option<Vec<String>>,
    ) -> bool {
        // MVCC: a candidate edge from the (non-versioned) adjacency index must still be
        // visible under the current snapshot — a writer's own PENDING-deleted edge must
        // not satisfy EXISTS/COUNT (read-your-writes); mirrors the expand operators.
        if let Some(epoch) = self.viewing_epoch {
            let visible = if let Some(tx) = self.transaction_id {
                self.store.is_edge_visible_versioned(edge_id, epoch, tx)
            } else {
                self.store.is_edge_visible_at_epoch(edge_id, epoch)
            };
            if !visible {
                return false;
            }
        }

        // Check edge type if specified
        if !edge_types.is_empty() {
            let type_ok = if let Some(actual_type) = self.store.edge_type(edge_id) {
                edge_types
                    .iter()
                    .any(|t| actual_type.as_str().eq_ignore_ascii_case(t.as_str()))
            } else {
                false
            };
            if !type_ok {
                return false;
            }
        }

        // Check end node labels if specified (e.g., (:Person)-[:KNOWS]->(n) requires
        // the other endpoint to have the Person label after direction flipping).
        if let Some(labels) = end_labels {
            // Resolve existence first, then check labels via the snapshot-aware
            // accessor so that uncommitted label ops in the writing transaction are
            // reflected here (behavior-preserving until Task 4 buffers writes).
            if self.resolve_node(other_node_id).is_some() {
                let snap_epoch = self
                    .viewing_epoch
                    .unwrap_or_else(|| self.store.current_epoch());
                let label_set = self.store.read_node_labels_visible(
                    other_node_id,
                    snap_epoch,
                    self.transaction_id,
                );
                labels
                    .iter()
                    .all(|l| label_set.iter().any(|s| s.as_str() == l.as_str()))
            } else {
                false
            }
        } else {
            true
        }
    }

    /// Resolves an edge using transaction-aware access when available.
    fn resolve_edge(&self, edge_id: grafeo_common::types::EdgeId) -> Option<Edge> {
        if let (Some(ep), Some(tx)) = (self.viewing_epoch, self.transaction_id) {
            self.store.get_edge_versioned(edge_id, ep, tx)
        } else if let Some(ep) = self.viewing_epoch {
            self.store.get_edge_at_epoch(edge_id, ep)
        } else {
            self.store.get_edge(edge_id)
        }
    }

    /// Snapshot-consistent whole property set for a node: the writing tx's buffered
    /// SET/REMOVE merged over the committed set (read-your-writes), else committed.
    fn visible_node_properties(&self, id: NodeId) -> FxHashMap<PropertyKey, Value> {
        let epoch = self
            .viewing_epoch
            .unwrap_or_else(|| self.store.current_epoch());
        self.store
            .read_node_properties_visible(id, epoch, self.transaction_id)
    }

    /// Snapshot-consistent whole property set for an edge: the writing tx's buffered
    /// SET/REMOVE merged over the committed set (read-your-writes), else committed.
    fn visible_edge_properties(&self, id: EdgeId) -> FxHashMap<PropertyKey, Value> {
        let epoch = self
            .viewing_epoch
            .unwrap_or_else(|| self.store.current_epoch());
        self.store
            .read_edge_properties_visible(id, epoch, self.transaction_id)
    }

    /// Evaluates the expression for a specific row in a chunk, returning the result value.
    /// This is useful for evaluating expressions in contexts like RETURN clauses.
    ///
    /// # Errors
    /// Returns any temporal Text read error rather than a missing value.
    pub fn eval_at(
        &self,
        chunk: &DataChunk,
        row: usize,
    ) -> grafeo_common::utils::error::Result<Option<Value>> {
        self.eval_with_provenance(chunk, row)
            .map(|(value, _)| value)
    }

    /// Evaluates once and returns certified runtime entity provenance, if any.
    /// Ordinary values, including integer IDs without a typed source, have no
    /// entity type. Consumers must not infer one from their representation.
    ///
    /// # Errors
    /// Returns the same evaluation errors as [`Self::eval_at`].
    pub fn eval_at_with_type(
        &self,
        chunk: &DataChunk,
        row: usize,
    ) -> grafeo_common::utils::error::Result<(Option<Value>, Option<LogicalType>)> {
        self.eval_with_provenance(chunk, row)
            .map(|(value, provenance)| (value, provenance.logical_type()))
    }

    fn eval_with_provenance(
        &self,
        chunk: &DataChunk,
        row: usize,
    ) -> grafeo_common::utils::error::Result<(Option<Value>, ValueProvenance)> {
        let error = std::cell::Cell::new(None);
        let evaluation = ExpressionEvaluation {
            predicate: self,
            error: &error,
            locals: None,
        };
        let (value, provenance) = evaluation.eval_typed(&self.expression, chunk, row);
        match error.into_inner() {
            Some(error) => Err(error),
            None => Ok((value, provenance)),
        }
    }
}

/// Only typed bindings and evaluated entity-producing expressions establish
/// provenance. Raw scalar results such as id(edge) remain ordinary values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ValueProvenance {
    #[default]
    Ordinary,
    Node,
    Edge,
    EdgeList,
}

impl ValueProvenance {
    fn from_type(data_type: &LogicalType) -> Self {
        match data_type {
            LogicalType::Node => Self::Node,
            LogicalType::Edge => Self::Edge,
            LogicalType::List(item) if item.as_ref() == &LogicalType::Edge => Self::EdgeList,
            _ => Self::Ordinary,
        }
    }

    fn logical_type(self) -> Option<LogicalType> {
        match self {
            Self::Ordinary => None,
            Self::Node => Some(LogicalType::Node),
            Self::Edge => Some(LogicalType::Edge),
            Self::EdgeList => Some(LogicalType::List(Box::new(LogicalType::Edge))),
        }
    }

    fn item(self) -> Self {
        if self == Self::EdgeList {
            Self::Edge
        } else {
            Self::Ordinary
        }
    }
}

/// A borrowed lexical frame lives only for its item's evaluation. A child
/// shadows matching outer names without cloning the row, AST, or name map.
struct LexicalBinding<'a> {
    parent: Option<&'a LexicalBinding<'a>>,
    name: &'a str,
    value: &'a Value,
    provenance: ValueProvenance,
    // reduce historically returns NULL for missing local map properties;
    // list comprehensions omit missing values instead.
    missing_map_property_is_null: bool,
}

enum ResolvedBinding<'a> {
    Column(&'a ValueVector),
    Local(&'a LexicalBinding<'a>),
}

impl ResolvedBinding<'_> {
    fn get_value(&self, row: usize) -> Option<Value> {
        match self {
            Self::Column(column) => column.get_value(row),
            Self::Local(binding) => Some(binding.value.clone()),
        }
    }

    fn provenance(&self) -> ValueProvenance {
        match self {
            Self::Column(column) => ValueProvenance::from_type(column.data_type()),
            Self::Local(binding) => binding.provenance,
        }
    }

    fn get_node_id(&self, row: usize) -> Option<NodeId> {
        match self {
            Self::Column(column) => column.get_node_id(row),
            Self::Local(binding) if binding.provenance == ValueProvenance::Node => {
                // reason: entity IDs use the established i64/u64 round-trip encoding
                #[allow(clippy::cast_sign_loss)]
                if let Value::Int64(id) = binding.value {
                    Some(NodeId(*id as u64))
                } else {
                    None
                }
            }
            Self::Local(_) => None,
        }
    }

    fn get_edge_id(&self, row: usize) -> Option<EdgeId> {
        match self {
            Self::Column(column) => column.get_edge_id(row),
            Self::Local(binding) if binding.provenance == ValueProvenance::Edge => {
                // reason: entity IDs use the established i64/u64 round-trip encoding
                #[allow(clippy::cast_sign_loss)]
                if let Value::Int64(id) = binding.value {
                    Some(EdgeId(*id as u64))
                } else {
                    None
                }
            }
            Self::Local(_) => None,
        }
    }
}

/// One borrowed evaluation shares its first-error slot with lexical children.
struct ExpressionEvaluation<'a> {
    predicate: &'a ExpressionPredicate,
    error: &'a std::cell::Cell<Option<grafeo_common::utils::error::Error>>,
    locals: Option<&'a LexicalBinding<'a>>,
}

impl std::ops::Deref for ExpressionEvaluation<'_> {
    type Target = ExpressionPredicate;

    fn deref(&self) -> &Self::Target {
        self.predicate
    }
}

impl ExpressionEvaluation<'_> {
    fn binding<'a>(&'a self, name: &str, chunk: &'a DataChunk) -> Option<ResolvedBinding<'a>> {
        let mut local = self.locals;
        while let Some(binding) = local {
            if binding.name == name {
                return Some(ResolvedBinding::Local(binding));
            }
            local = binding.parent;
        }
        let index = *self.variable_columns.get(name)?;
        Some(ResolvedBinding::Column(chunk.column(index)?))
    }

    fn eval_expr(&self, expr: &FilterExpression, chunk: &DataChunk, row: usize) -> Option<Value> {
        self.eval_typed(expr, chunk, row).0
    }

    fn eval_typed(
        &self,
        expr: &FilterExpression,
        chunk: &DataChunk,
        row: usize,
    ) -> (Option<Value>, ValueProvenance) {
        let mut provenance = ValueProvenance::Ordinary;
        let value = self.eval_expr_inner(expr, chunk, row, &mut provenance);
        if value.is_none() {
            provenance = ValueProvenance::Ordinary;
        }
        (value, provenance)
    }

    fn eval_expr_inner(
        &self,
        expr: &FilterExpression,
        chunk: &DataChunk,
        row: usize,
        provenance: &mut ValueProvenance,
    ) -> Option<Value> {
        match expr {
            FilterExpression::Literal(v) => Some(v.clone()),
            FilterExpression::Variable(name) => {
                let binding = self.binding(name, chunk)?;
                *provenance = binding.provenance();
                binding.get_value(row)
            }
            FilterExpression::Property { variable, property } => {
                let col = self.binding(variable, chunk)?;
                if let ResolvedBinding::Local(binding) = &col
                    && binding.missing_map_property_is_null
                    && let Value::Map(map) = binding.value
                {
                    return Some(
                        map.get(&PropertyKey::new(property))
                            .cloned()
                            .unwrap_or(Value::Null),
                    );
                }
                if let ResolvedBinding::Local(binding) = &col
                    && binding.provenance == ValueProvenance::Edge
                    && let Some(edge_id) = col.get_edge_id(row)
                {
                    return Some(
                        self.resolve_edge(edge_id)
                            .and_then(|edge| {
                                edge.properties.get(&PropertyKey::new(property)).cloned()
                            })
                            .unwrap_or(Value::Null),
                    );
                }
                let prop_key = grafeo_common::types::PropertyKey::new(property.as_str());
                let snap_epoch = self
                    .viewing_epoch
                    .unwrap_or_else(|| self.store.current_epoch());
                let tx = self.transaction_id;
                // Try as node first
                if let Some(node_id) = col.get_node_id(row)
                    && let result @ Some(_) = self
                        .store
                        .read_node_property_visible(node_id, &prop_key, snap_epoch, tx)
                {
                    return result;
                }
                // Try as edge if node lookup returned nothing
                if let Some(edge_id) = col.get_edge_id(row)
                    && let result @ Some(_) = self
                        .store
                        .read_edge_property_visible(edge_id, &prop_key, snap_epoch, tx)
                {
                    return result;
                }
                // Try as map value (e.g. from UNWIND with map elements)
                if let Some(Value::Map(map)) = col.get_value(row) {
                    return map.get(&prop_key).cloned();
                }
                None
            }
            FilterExpression::Binary { left, op, right } => {
                // For IN operator, right side is a list that we evaluate specially
                if *op == BinaryFilterOp::In {
                    let left_val = self.eval_expr(left, chunk, row)?;
                    return self.eval_in_operator(&left_val, right, chunk, row);
                }
                // For logical operators (AND/OR/XOR), treat missing values as
                // NULL so three-valued logic works correctly. Without this,
                // `NULL OR true` would short-circuit to None instead of true.
                if matches!(
                    op,
                    BinaryFilterOp::And | BinaryFilterOp::Or | BinaryFilterOp::Xor
                ) {
                    let left_val = self.eval_expr(left, chunk, row).unwrap_or(Value::Null);
                    let right_val = self.eval_expr(right, chunk, row).unwrap_or(Value::Null);
                    return self.eval_binary_op(&left_val, *op, &right_val);
                }
                let left_val = self.eval_expr(left, chunk, row)?;
                let right_val = self.eval_expr(right, chunk, row)?;
                self.eval_binary_op(&left_val, *op, &right_val)
            }
            FilterExpression::Unary { op, operand } => {
                let val = self.eval_expr(operand, chunk, row);
                self.eval_unary_op(*op, val)
            }
            FilterExpression::FunctionCall { name, args } => {
                self.eval_function(name, args, chunk, row, provenance)
            }
            FilterExpression::List(items) => {
                let mut values = Vec::new();
                let mut all_edges = true;
                for item in items {
                    let (value, kind) = self.eval_typed(item, chunk, row);
                    if let Some(value) = value {
                        all_edges &= kind == ValueProvenance::Edge;
                        values.push(value);
                    }
                }
                if all_edges && !values.is_empty() {
                    *provenance = ValueProvenance::EdgeList;
                }
                Some(Value::List(values.into()))
            }
            FilterExpression::Map(pairs) => {
                let mut map = BTreeMap::new();
                for (k, v) in pairs {
                    if let Some(val) = self.eval_expr(v, chunk, row) {
                        if k == "*" {
                            // AllProperties marker: flatten the inner map into the result
                            if let Value::Map(inner) = val {
                                map.extend(inner.iter().map(|(pk, pv)| (pk.clone(), pv.clone())));
                            }
                        } else {
                            map.insert(PropertyKey::new(k.as_str()), val);
                        }
                    }
                }
                Some(Value::Map(Arc::new(map)))
            }
            FilterExpression::IndexAccess { base, index } => {
                let (base_val, base_provenance) = self.eval_typed(base, chunk, row);
                let base_val = base_val?;
                let index_val = self.eval_expr(index, chunk, row)?;
                match (&base_val, &index_val) {
                    (Value::List(items), Value::Int64(i)) => {
                        // reason: list/string lengths fit i64; index values are user-provided
                        #[allow(
                            clippy::cast_possible_truncation,
                            clippy::cast_possible_wrap,
                            clippy::cast_sign_loss
                        )]
                        let idx = if *i < 0 {
                            // Negative indexing from end
                            let len = items.len() as i64;
                            (len + i) as usize
                        } else {
                            *i as usize
                        };
                        *provenance = base_provenance.item();
                        items.get(idx).cloned()
                    }
                    (Value::String(s), Value::Int64(i)) => {
                        // reason: list/string lengths fit i64; index values are user-provided
                        #[allow(
                            clippy::cast_possible_truncation,
                            clippy::cast_possible_wrap,
                            clippy::cast_sign_loss
                        )]
                        let idx = if *i < 0 {
                            let len = s.len() as i64;
                            (len + i) as usize
                        } else {
                            *i as usize
                        };
                        s.chars()
                            .nth(idx)
                            .map(|c| Value::String(c.to_string().into()))
                    }
                    (Value::Map(m), Value::String(key)) => {
                        let prop_key = PropertyKey::new(key.as_str());
                        m.get(&prop_key).cloned()
                    }
                    (_, Value::String(key)) => {
                        // Node/edge bracket access: n['name'] looks up a property
                        // via the store when the base variable refers to a node or edge.
                        if let FilterExpression::Variable(var) = base.as_ref()
                            && let Some(col) = self.binding(var, chunk)
                        {
                            let prop_key = grafeo_common::types::PropertyKey::new(key.as_str());
                            let snap_epoch = self
                                .viewing_epoch
                                .unwrap_or_else(|| self.store.current_epoch());
                            let tx = self.transaction_id;
                            if let Some(node_id) = col.get_node_id(row)
                                && let result @ Some(_) = self
                                    .store
                                    .read_node_property_visible(node_id, &prop_key, snap_epoch, tx)
                            {
                                return result;
                            }
                            if let Some(edge_id) = col.get_edge_id(row)
                                && let result @ Some(_) = self
                                    .store
                                    .read_edge_property_visible(edge_id, &prop_key, snap_epoch, tx)
                            {
                                return result;
                            }
                        }
                        None
                    }
                    _ => None,
                }
            }
            FilterExpression::SliceAccess { base, start, end } => {
                let (base_val, base_provenance) = self.eval_typed(base, chunk, row);
                let base_val = base_val?;
                let start_idx = start
                    .as_ref()
                    .and_then(|s| self.eval_expr(s, chunk, row))
                    .and_then(|v| {
                        if let Value::Int64(i) = v {
                            // reason: slice index from user query, non-negative for valid slices
                            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                            Some(i as usize)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);

                match &base_val {
                    Value::List(items) => {
                        *provenance = base_provenance;
                        let end_idx = end
                            .as_ref()
                            .and_then(|e| self.eval_expr(e, chunk, row))
                            .and_then(|v| {
                                if let Value::Int64(i) = v {
                                    // reason: slice end index from user query
                                    #[allow(
                                        clippy::cast_possible_truncation,
                                        clippy::cast_sign_loss
                                    )]
                                    Some(i as usize)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(items.len());
                        let sliced: Vec<Value> = items
                            .get(start_idx..end_idx.min(items.len()))
                            .unwrap_or(&[])
                            .to_vec();
                        Some(Value::List(sliced.into()))
                    }
                    Value::String(s) => {
                        let chars: Vec<char> = s.chars().collect();
                        let end_idx = end
                            .as_ref()
                            .and_then(|e| self.eval_expr(e, chunk, row))
                            .and_then(|v| {
                                if let Value::Int64(i) = v {
                                    // reason: slice end index from user query
                                    #[allow(
                                        clippy::cast_possible_truncation,
                                        clippy::cast_sign_loss
                                    )]
                                    Some(i as usize)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(chars.len());
                        let sliced: String = chars
                            .get(start_idx..end_idx.min(chars.len()))
                            .unwrap_or(&[])
                            .iter()
                            .collect();
                        Some(Value::String(sliced.into()))
                    }
                    _ => None,
                }
            }
            FilterExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => self.eval_case(
                operand.as_deref(),
                when_clauses,
                else_clause.as_deref(),
                chunk,
                row,
                provenance,
            ),
            FilterExpression::Id(variable) => {
                let col = self.binding(variable, chunk)?;
                // Try as node first, then as edge
                if let Some(node_id) = col.get_node_id(row) {
                    // reason: entity IDs stored as i64 values, standard encoding
                    #[allow(clippy::cast_possible_wrap)]
                    Some(Value::Int64(node_id.0 as i64))
                } else {
                    col.get_edge_id(row)
                        // reason: entity IDs stored as i64 values, standard encoding
                        .map(|edge_id| {
                            // reason: entity IDs are sequential counters, well within i64::MAX
                            #[allow(clippy::cast_possible_wrap)]
                            let val = Value::Int64(edge_id.0 as i64);
                            val
                        })
                }
            }
            FilterExpression::Labels(variable) => {
                let col = self.binding(variable, chunk)?;
                let node_id = col.get_node_id(row)?;
                // Guard: skip if node does not exist (preserves prior None semantics).
                self.resolve_node(node_id)?;
                // Route through the snapshot-aware accessor so that uncommitted
                // label ops in the writing transaction are reflected here.
                // (Behavior-preserving: delta is empty until Task 4 buffers writes.)
                let snap_epoch = self
                    .viewing_epoch
                    .unwrap_or_else(|| self.store.current_epoch());
                let label_set =
                    self.store
                        .read_node_labels_visible(node_id, snap_epoch, self.transaction_id);
                // Sort labels so sets with the same members always produce
                // the same list, regardless of internal storage order.
                let mut sorted: Vec<arcstr::ArcStr> = label_set.into_iter().collect();
                sorted.sort();
                let labels: Vec<Value> = sorted.into_iter().map(Value::String).collect();
                Some(Value::List(labels.into()))
            }
            FilterExpression::Type(variable) => {
                let col = self.binding(variable, chunk)?;
                let edge_id = col.get_edge_id(row)?;
                let edge = self.resolve_edge(edge_id)?;
                Some(Value::String(edge.edge_type.clone()))
            }
            FilterExpression::ListComprehension {
                variable,
                list_expr,
                filter_expr,
                map_expr,
            } => {
                let (list_val, list_provenance) = self.eval_typed(list_expr, chunk, row);
                let list_val = list_val?;
                let owned_items: Vec<Value>;
                let items: &[Value] = match &list_val {
                    Value::List(list) => list,
                    Value::Vector(vector) => {
                        owned_items = vector
                            .iter()
                            .map(|&value| Value::Float64(f64::from(value)))
                            .collect();
                        &owned_items
                    }
                    _ => return None,
                };
                let mut result = Vec::new();
                let mut all_edges = true;
                for item in items {
                    let binding = LexicalBinding {
                        parent: self.locals,
                        name: variable,
                        value: item,
                        provenance: list_provenance.item(),
                        missing_map_property_is_null: false,
                    };
                    let child = ExpressionEvaluation {
                        predicate: self.predicate,
                        error: self.error,
                        locals: Some(&binding),
                    };
                    let passes = filter_expr.as_ref().is_none_or(|filter| {
                        matches!(child.eval_expr(filter, chunk, row), Some(Value::Bool(true)))
                    });
                    if passes {
                        let (mapped, kind) = child.eval_typed(map_expr, chunk, row);
                        if let Some(mapped) = mapped {
                            all_edges &= kind == ValueProvenance::Edge;
                            result.push(mapped);
                        }
                    }
                }
                if all_edges && !result.is_empty() {
                    *provenance = ValueProvenance::EdgeList;
                }
                Some(Value::List(result.into()))
            }
            FilterExpression::ListPredicate {
                kind,
                variable,
                list_expr,
                predicate,
            } => {
                let (list_val, list_provenance) = self.eval_typed(list_expr, chunk, row);
                let list_val = list_val?;
                let owned_items: Vec<Value>;
                let items: &[Value] = match &list_val {
                    Value::List(list) => list,
                    Value::Vector(vector) => {
                        owned_items = vector
                            .iter()
                            .map(|&value| Value::Float64(f64::from(value)))
                            .collect();
                        &owned_items
                    }
                    _ => return None,
                };
                let mut match_count = 0usize;
                for item in items {
                    let binding = LexicalBinding {
                        parent: self.locals,
                        name: variable,
                        value: item,
                        provenance: list_provenance.item(),
                        missing_map_property_is_null: false,
                    };
                    let child = ExpressionEvaluation {
                        predicate: self.predicate,
                        error: self.error,
                        locals: Some(&binding),
                    };
                    if matches!(
                        child.eval_expr(predicate, chunk, row),
                        Some(Value::Bool(true))
                    ) {
                        match_count += 1;
                    }
                }
                Some(Value::Bool(match kind {
                    ListPredicateKind::All => match_count == items.len(),
                    ListPredicateKind::Any => match_count > 0,
                    ListPredicateKind::None => match_count == 0,
                    ListPredicateKind::Single => match_count == 1,
                }))
            }
            FilterExpression::ExistsSubquery {
                start_var,
                direction,
                edge_types,
                end_labels,
                // min_hops/max_hops are always None from the fast path
                // (extract_exists_pattern rejects multi-hop patterns).
                ..
            } => {
                // Get the start node ID from the current row
                let col = self.binding(start_var, chunk)?;
                let start_node_id = col.get_node_id(row)?;

                // Check if any matching edges exist
                let exists = self
                    .store
                    .edges_from(start_node_id, *direction)
                    .into_iter()
                    .any(|(other_node_id, edge_id)| {
                        self.edge_matches(other_node_id, edge_id, edge_types, end_labels)
                    });

                Some(Value::Bool(exists))
            }
            FilterExpression::CountSubquery {
                start_var,
                direction,
                edge_types,
                end_labels,
            } => {
                let col = self.binding(start_var, chunk)?;
                let start_node_id = col.get_node_id(row)?;

                let count = self
                    .store
                    .edges_from(start_node_id, *direction)
                    .into_iter()
                    .filter(|(other_node_id, edge_id)| {
                        self.edge_matches(*other_node_id, *edge_id, edge_types, end_labels)
                    })
                    .count();

                // reason: edge count from a single node fits i64
                #[allow(clippy::cast_possible_wrap)]
                Some(Value::Int64(count as i64))
            }
            FilterExpression::Reduce {
                accumulator,
                initial,
                variable,
                list,
                expression,
            } => {
                let (initial_value, mut acc_provenance) = self.eval_typed(initial, chunk, row);
                let mut acc = initial_value?;
                let (list_value, list_provenance) = self.eval_typed(list, chunk, row);
                let list_value = list_value?;
                let owned_items: Vec<Value>;
                let items: &[Value] = match &list_value {
                    Value::List(list) => list,
                    Value::Vector(vector) => {
                        owned_items = vector
                            .iter()
                            .map(|&value| Value::Float64(f64::from(value)))
                            .collect();
                        &owned_items
                    }
                    _ => return None,
                };
                for item in items {
                    let item_binding = LexicalBinding {
                        parent: self.locals,
                        name: variable,
                        value: item,
                        provenance: list_provenance.item(),
                        missing_map_property_is_null: true,
                    };
                    // The old reducer resolves the accumulator first if the
                    // two local names coincide; preserve that precedence.
                    let acc_binding = LexicalBinding {
                        parent: Some(&item_binding),
                        name: accumulator,
                        value: &acc,
                        provenance: acc_provenance,
                        missing_map_property_is_null: true,
                    };
                    let child = ExpressionEvaluation {
                        predicate: self.predicate,
                        error: self.error,
                        locals: Some(&acc_binding),
                    };
                    let (value, kind) = child.eval_typed(expression, chunk, row);
                    acc = value?;
                    acc_provenance = kind;
                }
                *provenance = acc_provenance;
                Some(acc)
            }
        }
    }

    /// Evaluates a binary operator with ISO three-valued logic.
    ///
    /// NULL propagation: `NULL = x`, `x = NULL`, `NULL <> x` all yield
    /// `Value::Null` (UNKNOWN), not `Value::Bool`. AND/OR/XOR follow the
    /// standard truth tables where FALSE AND UNKNOWN = FALSE, etc.
    ///
    /// For structural equality (DISTINCT, GROUP BY), use [`values_equal`]
    /// directly, which treats NULL == NULL as true.
    fn eval_binary_op(&self, left: &Value, op: BinaryFilterOp, right: &Value) -> Option<Value> {
        match op {
            // Three-valued logic for AND/OR/XOR (ISO/IEC 39075 Section 21)
            BinaryFilterOp::And => match (left.as_bool(), right.as_bool()) {
                (Some(false), _) | (_, Some(false)) => Some(Value::Bool(false)),
                (Some(true), Some(true)) => Some(Value::Bool(true)),
                _ => Some(Value::Null), // UNKNOWN
            },
            BinaryFilterOp::Or => match (left.as_bool(), right.as_bool()) {
                (Some(true), _) | (_, Some(true)) => Some(Value::Bool(true)),
                (Some(false), Some(false)) => Some(Value::Bool(false)),
                _ => Some(Value::Null), // UNKNOWN
            },
            BinaryFilterOp::Xor => match (left.as_bool(), right.as_bool()) {
                (Some(l), Some(r)) => Some(Value::Bool(l ^ r)),
                _ => Some(Value::Null), // UNKNOWN
            },
            // NULL = anything or anything = NULL is UNKNOWN (three-valued logic).
            // values_equal is preserved for structural equality (DISTINCT, GROUP BY).
            BinaryFilterOp::Eq => {
                if left.is_null() || right.is_null() {
                    Some(Value::Null)
                } else {
                    Some(Value::Bool(Self::values_equal(left, right)))
                }
            }
            BinaryFilterOp::Ne => {
                if left.is_null() || right.is_null() {
                    Some(Value::Null)
                } else {
                    Some(Value::Bool(!Self::values_equal(left, right)))
                }
            }
            BinaryFilterOp::Lt => Self::compare_values(left, right).map(|c| Value::Bool(c < 0)),
            BinaryFilterOp::Le => Self::compare_values(left, right).map(|c| Value::Bool(c <= 0)),
            BinaryFilterOp::Gt => Self::compare_values(left, right).map(|c| Value::Bool(c > 0)),
            BinaryFilterOp::Ge => Self::compare_values(left, right).map(|c| Value::Bool(c >= 0)),
            // Arithmetic operators
            BinaryFilterOp::Add => {
                // String concatenation: string + string, or string + other
                match (left, right) {
                    (Value::String(a), Value::String(b)) => {
                        let mut s = String::with_capacity(a.len() + b.len());
                        s.push_str(a);
                        s.push_str(b);
                        Some(Value::String(s.into()))
                    }
                    (Value::String(a), other) => {
                        let b = match other {
                            Value::Int64(i) => i.to_string(),
                            Value::Float64(f) => f.to_string(),
                            Value::Bool(b) => b.to_string(),
                            Value::Null => return Some(Value::Null),
                            _ => return None,
                        };
                        let mut s = String::with_capacity(a.len() + b.len());
                        s.push_str(a);
                        s.push_str(&b);
                        Some(Value::String(s.into()))
                    }
                    // Temporal addition
                    (Value::Date(d), Value::Duration(dur))
                    | (Value::Duration(dur), Value::Date(d)) => {
                        Some(Value::Date(d.add_duration(dur)))
                    }
                    (Value::Time(t), Value::Duration(dur))
                    | (Value::Duration(dur), Value::Time(t)) => {
                        Some(Value::Time(t.add_duration(dur)))
                    }
                    (Value::Timestamp(ts), Value::Duration(dur))
                    | (Value::Duration(dur), Value::Timestamp(ts)) => {
                        Some(Value::Timestamp(ts.add_duration(dur)))
                    }
                    (Value::Duration(a), Value::Duration(b)) => Some(Value::Duration(a.add(*b))),
                    (Value::List(a), Value::List(b)) => {
                        let mut combined = Vec::with_capacity(a.len() + b.len());
                        combined.extend_from_slice(a);
                        combined.extend_from_slice(b);
                        Some(Value::List(combined.into()))
                    }
                    _ => self.eval_arithmetic(left, right, i64::checked_add, |a, b| a + b),
                }
            }
            BinaryFilterOp::Sub => match (left, right) {
                // Temporal subtraction
                (Value::Date(a), Value::Duration(dur)) => Some(Value::Date(a.sub_duration(dur))),
                (Value::Time(a), Value::Duration(dur)) => {
                    Some(Value::Time(a.add_duration(&dur.neg())))
                }
                (Value::Timestamp(a), Value::Duration(dur)) => {
                    Some(Value::Timestamp(a.add_duration(&dur.neg())))
                }
                (Value::Date(a), Value::Date(b)) => {
                    let days = a.as_days() as i64 - b.as_days() as i64;
                    Some(Value::Duration(grafeo_common::types::Duration::from_days(
                        days,
                    )))
                }
                (Value::Time(a), Value::Time(b)) => {
                    // reason: time-of-day nanos (max ~86.4 trillion) fit i64
                    #[allow(clippy::cast_possible_wrap)]
                    let nanos = a.as_nanos() as i64 - b.as_nanos() as i64;
                    Some(Value::Duration(grafeo_common::types::Duration::from_nanos(
                        nanos,
                    )))
                }
                (Value::Timestamp(a), Value::Timestamp(b)) => {
                    let micros = a.duration_since(*b);
                    Some(Value::Duration(grafeo_common::types::Duration::from_nanos(
                        micros * 1000,
                    )))
                }
                (Value::Duration(a), Value::Duration(b)) => Some(Value::Duration(a.sub(*b))),
                _ => self.eval_arithmetic(left, right, i64::checked_sub, |a, b| a - b),
            },
            BinaryFilterOp::Mul => match (left, right) {
                (Value::Duration(d), Value::Int64(n)) | (Value::Int64(n), Value::Duration(d)) => {
                    Some(Value::Duration(d.mul(*n)))
                }
                _ => self.eval_arithmetic(left, right, i64::checked_mul, |a, b| a * b),
            },
            BinaryFilterOp::Div => match (left, right) {
                (Value::Duration(d), Value::Int64(n)) if *n != 0 => {
                    Some(Value::Duration(d.div(*n)))
                }
                _ => self.eval_arithmetic(left, right, i64::checked_div, |a, b| a / b),
            },
            BinaryFilterOp::Mod => self.eval_modulo(left, right),
            // String operators
            BinaryFilterOp::StartsWith => {
                let l = left.as_str()?;
                let r = right.as_str()?;
                Some(Value::Bool(l.starts_with(r)))
            }
            BinaryFilterOp::EndsWith => {
                let l = left.as_str()?;
                let r = right.as_str()?;
                Some(Value::Bool(l.ends_with(r)))
            }
            BinaryFilterOp::Contains => {
                let l = left.as_str()?;
                let r = right.as_str()?;
                Some(Value::Bool(l.contains(r)))
            }
            // IN is handled separately
            BinaryFilterOp::In => None,
            // Regex match (=~)
            BinaryFilterOp::Regex => {
                #[cfg(any(feature = "regex", feature = "regex-lite"))]
                match (left, right) {
                    (Value::String(s), Value::String(pattern)) => match Regex::new(pattern) {
                        Ok(re) => Some(Value::Bool(re.is_match(s))),
                        Err(_) => None,
                    },
                    _ => None,
                }
                #[cfg(not(any(feature = "regex", feature = "regex-lite")))]
                {
                    let _ = (left, right);
                    None
                }
            }
            // Power/exponentiation (^)
            BinaryFilterOp::Pow => {
                match (left, right) {
                    (Value::Int64(base), Value::Int64(exp)) => {
                        Some(Value::Float64((*base as f64).powf(*exp as f64)))
                    }
                    (Value::Float64(base), Value::Float64(exp)) => {
                        Some(Value::Float64(base.powf(*exp)))
                    }
                    (Value::Int64(base), Value::Float64(exp)) => {
                        Some(Value::Float64((*base as f64).powf(*exp)))
                    }
                    (Value::Float64(base), Value::Int64(exp)) => {
                        Some(Value::Float64(base.powf(*exp as f64)))
                    }
                    _ => None, // Type mismatch
                }
            }
            // SQL LIKE pattern matching
            BinaryFilterOp::Like => {
                #[cfg(any(feature = "regex", feature = "regex-lite"))]
                match (left, right) {
                    (Value::String(s), Value::String(pattern)) => {
                        let mut regex_pattern = String::with_capacity(pattern.len() + 4);
                        regex_pattern.push('^');
                        let mut chars = pattern.chars().peekable();
                        while let Some(ch) = chars.next() {
                            match ch {
                                '%' => regex_pattern.push_str(".*"),
                                '_' => regex_pattern.push('.'),
                                '\\' => {
                                    if let Some(next) = chars.next() {
                                        regex_escape_char(next, &mut regex_pattern);
                                    }
                                }
                                _ => regex_escape_char(ch, &mut regex_pattern),
                            }
                        }
                        regex_pattern.push('$');
                        match Regex::new(&regex_pattern) {
                            Ok(re) => Some(Value::Bool(re.is_match(s))),
                            Err(_) => None,
                        }
                    }
                    (Value::Null, _) | (_, Value::Null) => Some(Value::Null),
                    _ => None,
                }
                #[cfg(not(any(feature = "regex", feature = "regex-lite")))]
                {
                    let _ = (left, right);
                    None
                }
            }
            // String concatenation (||)
            BinaryFilterOp::Concat => match (left, right) {
                (Value::String(a), Value::String(b)) => {
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(a);
                    s.push_str(b);
                    Some(Value::String(s.into()))
                }
                (Value::String(a), other) => {
                    let b = value_to_string(other)?;
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(a);
                    s.push_str(&b);
                    Some(Value::String(s.into()))
                }
                (other, Value::String(b)) => {
                    let a = value_to_string(other)?;
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(&a);
                    s.push_str(b);
                    Some(Value::String(s.into()))
                }
                (Value::Null, _) | (_, Value::Null) => Some(Value::Null),
                _ => None,
            },
        }
    }

    fn eval_arithmetic<F1, F2>(
        &self,
        left: &Value,
        right: &Value,
        int_op: F1,
        float_op: F2,
    ) -> Option<Value>
    where
        F1: Fn(i64, i64) -> Option<i64>,
        F2: Fn(f64, f64) -> f64,
    {
        match (left, right) {
            (Value::Int64(a), Value::Int64(b)) => int_op(*a, *b).map(Value::Int64),
            (Value::Float64(a), Value::Float64(b)) => Some(Value::Float64(float_op(*a, *b))),
            (Value::Int64(a), Value::Float64(b)) => Some(Value::Float64(float_op(*a as f64, *b))),
            (Value::Float64(a), Value::Int64(b)) => Some(Value::Float64(float_op(*a, *b as f64))),
            _ => None,
        }
    }

    fn eval_modulo(&self, left: &Value, right: &Value) -> Option<Value> {
        match (left, right) {
            (Value::Int64(a), Value::Int64(b)) if *b != 0 => a.checked_rem(*b).map(Value::Int64),
            (Value::Float64(a), Value::Float64(b)) if *b != 0.0 => Some(Value::Float64(a % b)),
            (Value::Int64(a), Value::Float64(b)) if *b != 0.0 => {
                Some(Value::Float64(*a as f64 % b))
            }
            (Value::Float64(a), Value::Int64(b)) if *b != 0 => Some(Value::Float64(a % *b as f64)),
            _ => None,
        }
    }

    /// Evaluates `left IN right` with three-valued NULL semantics.
    ///
    /// - `NULL IN [...]` yields UNKNOWN.
    /// - If no element matches but the list contains NULLs, yields UNKNOWN.
    /// - Otherwise yields `true` on first match, `false` if none match.
    fn eval_in_operator(
        &self,
        left: &Value,
        right: &FilterExpression,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        let right_val = self.eval_expr(right, chunk, row)?;
        match right_val {
            Value::List(items) => {
                // Three-valued IN: NULL IN (...) is UNKNOWN
                if left.is_null() {
                    return Some(Value::Null);
                }
                let mut has_null = false;
                for item in items.iter() {
                    if item.is_null() {
                        has_null = true;
                    } else if Self::values_equal(left, item) {
                        return Some(Value::Bool(true));
                    }
                }
                if has_null {
                    Some(Value::Null) // no match but NULLs present: UNKNOWN
                } else {
                    Some(Value::Bool(false))
                }
            }
            _ => None,
        }
    }

    fn eval_function(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
        provenance: &mut ValueProvenance,
    ) -> Option<Value> {
        let name_lower = name.to_lowercase();
        let name = name_lower.as_str();
        self.eval_graph_element_fn(name, args, chunk, row)
            .or_else(|| self.eval_type_fn(name, args, chunk, row))
            .or_else(|| self.eval_collection_fn(name, args, chunk, row, provenance))
            .or_else(|| self.eval_string_fn(name, args, chunk, row))
            .or_else(|| self.eval_numeric_fn(name, args, chunk, row))
            .or_else(|| self.eval_temporal_fn(name, args, chunk, row))
            .or_else(|| self.eval_path_fn(name, args, chunk, row, provenance))
            .or_else(|| self.eval_vector_fn(name, args, chunk, row))
            .or_else(|| self.eval_text_fn(name, args, chunk, row))
            .or_else(|| self.eval_session_fn(name, args, chunk, row))
    }

    fn eval_graph_element_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "id" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col = self.binding(var, chunk)?;
                    if let Some(node_id) = col.get_node_id(row) {
                        // reason: entity IDs stored as i64, standard encoding
                        #[allow(clippy::cast_possible_wrap)]
                        return Some(Value::Int64(node_id.0 as i64));
                    } else if let Some(edge_id) = col.get_edge_id(row) {
                        // reason: entity IDs stored as i64, standard encoding
                        #[allow(clippy::cast_possible_wrap)]
                        return Some(Value::Int64(edge_id.0 as i64));
                    }
                }
                None
            }
            "element_id" | "elementid" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col = self.binding(var, chunk)?;
                    // Resolve ambiguity between node/edge by verifying against the
                    // store. VectorData::Generic stores raw Int64 values that both
                    // get_node_id and get_edge_id accept, so we must check which
                    // entity actually exists.
                    if let Some(edge_id) = col.get_edge_id(row)
                        && self.resolve_edge(edge_id).is_some()
                    {
                        return Some(Value::String(format!("e:{}", edge_id.0).into()));
                    }
                    if let Some(node_id) = col.get_node_id(row)
                        && self.resolve_node(node_id).is_some()
                    {
                        return Some(Value::String(format!("n:{}", node_id.0).into()));
                    }
                }
                None
            }
            "labels" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col = self.binding(var, chunk)?;
                    let node_id = col.get_node_id(row)?;
                    // Guard: skip if node does not exist.
                    self.resolve_node(node_id)?;
                    // Route through the snapshot-aware accessor so that uncommitted
                    // label ops in the writing transaction are reflected here.
                    let snap_epoch = self
                        .viewing_epoch
                        .unwrap_or_else(|| self.store.current_epoch());
                    let label_set = self.store.read_node_labels_visible(
                        node_id,
                        snap_epoch,
                        self.transaction_id,
                    );
                    let mut sorted: Vec<arcstr::ArcStr> = label_set.into_iter().collect();
                    sorted.sort();
                    let labels: Vec<Value> = sorted.into_iter().map(Value::String).collect();
                    return Some(Value::List(labels.into()));
                }
                None
            }
            "type" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col = self.binding(var, chunk)?;
                    let edge_id = col.get_edge_id(row)?;
                    let edge = self.resolve_edge(edge_id)?;
                    return Some(Value::String(edge.edge_type.clone()));
                }
                None
            }
            "startnode" | "start_node" => {
                // startNode(edge) - returns the source node ID
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col = self.binding(var, chunk)?;
                    let edge_id = col.get_edge_id(row)?;
                    let edge = self.resolve_edge(edge_id)?;
                    // reason: entity IDs stored as i64, standard encoding
                    #[allow(clippy::cast_possible_wrap)]
                    return Some(Value::Int64(edge.src.0 as i64));
                }
                None
            }
            "endnode" | "end_node" => {
                // endNode(edge) - returns the destination node ID
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col = self.binding(var, chunk)?;
                    let edge_id = col.get_edge_id(row)?;
                    let edge = self.resolve_edge(edge_id)?;
                    // reason: entity IDs stored as i64, standard encoding
                    #[allow(clippy::cast_possible_wrap)]
                    return Some(Value::Int64(edge.dst.0 as i64));
                }
                None
            }
            "property_exists" => {
                // property_exists(entity, key) - checks if a property key exists on an entity
                if args.len() != 2 {
                    return None;
                }
                let Value::String(key) = self.eval_expr(&args[1], chunk, row)? else {
                    return None;
                };
                // Try node first, then edge
                if let FilterExpression::Variable(var) = &args[0] {
                    let col = self.binding(var, chunk)?;
                    if let Some(nid) = col.get_node_id(row)
                        && self.resolve_node(nid).is_some()
                    {
                        let exists = self
                            .visible_node_properties(nid)
                            .keys()
                            .any(|k| k.as_str() == key.as_str());
                        return Some(Value::Bool(exists));
                    }
                    if let Some(eid) = col.get_edge_id(row)
                        && self.resolve_edge(eid).is_some()
                    {
                        let exists = self
                            .visible_edge_properties(eid)
                            .keys()
                            .any(|k| k.as_str() == key.as_str());
                        return Some(Value::Bool(exists));
                    }
                }
                Some(Value::Bool(false))
            }
            "haslabel" => {
                // hasLabel(node, label) - checks if a node has a specific label
                if args.len() != 2 {
                    return None;
                }
                // First arg is the node variable
                let node_id = if let FilterExpression::Variable(var) = &args[0] {
                    let col = self.binding(var, chunk)?;
                    col.get_node_id(row)?
                } else {
                    return None;
                };
                // Second arg is the label to check
                let Value::String(label) = self.eval_expr(&args[1], chunk, row)? else {
                    return None;
                };
                // Check if the node has this label via the snapshot-aware accessor
                // so uncommitted label ops in the writing transaction are reflected.
                self.resolve_node(node_id)?;
                let snap_epoch = self
                    .viewing_epoch
                    .unwrap_or_else(|| self.store.current_epoch());
                let label_set =
                    self.store
                        .read_node_labels_visible(node_id, snap_epoch, self.transaction_id);
                let has_label = label_set.iter().any(|l| l.as_str() == label.as_str());
                Some(Value::Bool(has_label))
            }
            "issource" => {
                // isSource(node, edge) - checks if node is the source of edge
                if args.len() != 2 {
                    return None;
                }
                let node_id = if let FilterExpression::Variable(var) = &args[0] {
                    let col = self.binding(var, chunk)?;
                    col.get_node_id(row)?
                } else {
                    return None;
                };
                let edge_id = if let FilterExpression::Variable(var) = &args[1] {
                    let col = self.binding(var, chunk)?;
                    col.get_edge_id(row)?
                } else {
                    return None;
                };
                let edge = self.resolve_edge(edge_id)?;
                Some(Value::Bool(edge.src == node_id))
            }
            "isdestination" => {
                // isDestination(node, edge) - checks if node is the destination of edge
                if args.len() != 2 {
                    return None;
                }
                let node_id = if let FilterExpression::Variable(var) = &args[0] {
                    let col = self.binding(var, chunk)?;
                    col.get_node_id(row)?
                } else {
                    return None;
                };
                let edge_id = if let FilterExpression::Variable(var) = &args[1] {
                    let col = self.binding(var, chunk)?;
                    col.get_edge_id(row)?
                } else {
                    return None;
                };
                let edge = self.resolve_edge(edge_id)?;
                Some(Value::Bool(edge.dst == node_id))
            }
            "isdirected" => {
                // isDirected(edge) - checks if an edge is directed (always true in LPG)
                if args.len() != 1 {
                    return None;
                }
                // In LPG, all edges are directed
                if let FilterExpression::Variable(var) = &args[0] {
                    let col = self.binding(var, chunk)?;
                    // If the column contains an edge ID, it's directed
                    if col.get_edge_id(row).is_some() {
                        return Some(Value::Bool(true));
                    }
                }
                Some(Value::Bool(false))
            }
            _ => None,
        }
    }

    fn eval_type_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "tostring" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let s = match &val {
                    Value::String(s) => s.to_string(),
                    Value::Int64(i) => i.to_string(),
                    Value::Float64(f) => f.to_string(),
                    Value::Bool(b) => b.to_string(),
                    Value::Null => return Some(Value::Null),
                    _ => format!("{val:?}"),
                };
                Some(Value::String(s.into()))
            }
            "tointeger" | "toint" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i)),
                    // reason: toInteger() intentionally truncates float to int per GQL spec
                    #[allow(clippy::cast_possible_truncation)]
                    Value::Float64(f) => Some(Value::Int64(f as i64)),
                    Value::Bool(b) => Some(Value::Int64(i64::from(b))),
                    Value::String(s) => s.parse::<i64>().ok().map(Value::Int64),
                    _ => None,
                }
            }
            "tofloat" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64(i as f64)),
                    Value::Float64(f) => Some(Value::Float64(f)),
                    Value::String(s) => s.parse::<f64>().ok().map(Value::Float64),
                    _ => None,
                }
            }
            "toboolean" | "tobool" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Bool(b) => Some(Value::Bool(b)),
                    Value::String(s) => match s.to_lowercase().as_str() {
                        "true" => Some(Value::Bool(true)),
                        "false" => Some(Value::Bool(false)),
                        _ => None,
                    },
                    _ => None,
                }
            }
            "tolist" => {
                // toList(value) - wraps a scalar in a single-element list, or returns list as-is
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::List(_) => Some(val),
                    Value::Null => Some(Value::Null),
                    other => Some(Value::List(vec![other].into())),
                }
            }
            "totypedlist" => {
                // toTypedList(value, element_type) - coerces value to a list with typed elements
                if args.len() != 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let Value::String(elem_type) = self.eval_expr(&args[1], chunk, row)? else {
                    return None;
                };
                if matches!(val, Value::Null) {
                    return Some(Value::Null);
                }
                // Wrap scalar in a list first
                let items = match val {
                    Value::List(items) => items.to_vec(),
                    other => vec![other],
                };
                // Coerce each element to the target type
                let coerced: Option<Vec<Value>> = items
                    .into_iter()
                    .map(|v| Self::coerce_to_type(v, &elem_type))
                    .collect();
                coerced.map(|v| Value::List(v.into()))
            }
            "istyped" => {
                // isTyped(value, type_name) - checks if a value has a specific GQL type
                if args.len() != 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let Value::String(type_name) = self.eval_expr(&args[1], chunk, row)? else {
                    return None;
                };
                let matches = match type_name.to_uppercase().as_str() {
                    "BOOLEAN" | "BOOL" => matches!(val, Value::Bool(_)),
                    "INTEGER" | "INT" | "INT64" => matches!(val, Value::Int64(_)),
                    "FLOAT" | "FLOAT64" | "DOUBLE" => matches!(val, Value::Float64(_)),
                    "STRING" => matches!(val, Value::String(_)),
                    "LIST" => matches!(val, Value::List(_)),
                    "MAP" | "RECORD" => matches!(val, Value::Map(_)),
                    "NULL" => matches!(val, Value::Null),
                    "DATE" => matches!(val, Value::Date(_)),
                    "TIME" => matches!(val, Value::Time(_)),
                    "DATETIME" | "TIMESTAMP" => matches!(val, Value::Timestamp(_)),
                    "DURATION" => matches!(val, Value::Duration(_)),
                    "PATH" => matches!(val, Value::Path { .. }),
                    "NODE" | "EDGE" | "GRAPH" => false,
                    s if s.starts_with("LIST<") && s.ends_with('>') => {
                        let elem_type = &s[5..s.len() - 1];
                        match &val {
                            Value::List(items) => {
                                items.iter().all(|v| Self::value_matches_type(v, elem_type))
                            }
                            _ => false,
                        }
                    }
                    _ => false,
                };
                Some(Value::Bool(matches))
            }
            _ => None,
        }
    }

    fn eval_collection_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
        provenance: &mut ValueProvenance,
    ) -> Option<Value> {
        match name {
            "size" | "length" | "cardinality" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    // reason: collection lengths fit i64 for practical sizes
                    #[allow(clippy::cast_possible_wrap)]
                    Value::List(items) => Some(Value::Int64(items.len() as i64)),
                    // reason: string lengths fit i64 for practical sizes
                    #[allow(clippy::cast_possible_wrap)]
                    Value::String(s) => Some(Value::Int64(s.len() as i64)),
                    // reason: path lengths fit i64 for practical sizes
                    #[allow(clippy::cast_possible_wrap)]
                    Value::Path { edges, .. } => Some(Value::Int64(edges.len() as i64)),
                    _ => None,
                }
            }
            "coalesce" => {
                for arg in args {
                    let (value, kind) = self.eval_typed(arg, chunk, row);
                    if let Some(value) = value
                        && !value.is_null()
                    {
                        *provenance = kind;
                        return Some(value);
                    }
                }
                Some(Value::Null)
            }
            "exists" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row);
                Some(Value::Bool(
                    val.is_some() && !matches!(val, Some(Value::Null)),
                ))
            }
            "keys" => {
                if args.len() != 1 {
                    return None;
                }
                // keys(n) on a node variable: get property keys from the store
                if let FilterExpression::Variable(var) = &args[0] {
                    let col = self.binding(var, chunk)?;
                    if let Some(node_id) = col.get_node_id(row) {
                        self.resolve_node(node_id)?;
                        let keys: Vec<Value> = self
                            .visible_node_properties(node_id)
                            .keys()
                            .map(|k| Value::String(k.as_str().into()))
                            .collect();
                        return Some(Value::List(keys.into()));
                    }
                }
                // keys(map) on a map value
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Map(map) => {
                        let keys: Vec<Value> = map
                            .keys()
                            .map(|k| Value::String(k.as_str().into()))
                            .collect();
                        Some(Value::List(keys.into()))
                    }
                    _ => None,
                }
            }
            "properties" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col = self.binding(var, chunk)?;
                    if let Some(node_id) = col.get_node_id(row) {
                        self.resolve_node(node_id)?;
                        let map: std::collections::BTreeMap<PropertyKey, Value> =
                            self.visible_node_properties(node_id).into_iter().collect();
                        return Some(Value::Map(Arc::new(map)));
                    } else if let Some(edge_id) = col.get_edge_id(row) {
                        self.resolve_edge(edge_id)?;
                        let map: std::collections::BTreeMap<PropertyKey, Value> =
                            self.visible_edge_properties(edge_id).into_iter().collect();
                        return Some(Value::Map(Arc::new(map)));
                    }
                }
                None
            }
            // property_values(n) returns all property values of a node or edge as a flat list.
            // Used by Gremlin values() with no keys.
            "property_values" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col = self.binding(var, chunk)?;
                    if let Some(node_id) = col.get_node_id(row) {
                        self.resolve_node(node_id)?;
                        let vals: Vec<Value> = self
                            .visible_node_properties(node_id)
                            .into_values()
                            .collect();
                        return Some(Value::List(vals.into()));
                    } else if let Some(edge_id) = col.get_edge_id(row) {
                        self.resolve_edge(edge_id)?;
                        let vals: Vec<Value> = self
                            .visible_edge_properties(edge_id)
                            .into_values()
                            .collect();
                        return Some(Value::List(vals.into()));
                    }
                }
                None
            }
            "head" => {
                // head(list) - returns the first element of a list
                if args.len() != 1 {
                    return None;
                }
                let (val, list_provenance) = self.eval_typed(&args[0], chunk, row);
                let val = val?;
                match val {
                    Value::List(items) => {
                        *provenance = list_provenance.item();
                        items.first().cloned()
                    }
                    _ => None,
                }
            }
            "tail" => {
                // tail(list) - returns all elements except the first
                if args.len() != 1 {
                    return None;
                }
                let (val, list_provenance) = self.eval_typed(&args[0], chunk, row);
                let val = val?;
                match val {
                    Value::List(items) => {
                        *provenance = list_provenance;
                        if items.is_empty() {
                            Some(Value::List(vec![].into()))
                        } else {
                            Some(Value::List(items[1..].to_vec().into()))
                        }
                    }
                    _ => None,
                }
            }
            "last" => {
                // last(list) - returns the last element of a list
                if args.len() != 1 {
                    return None;
                }
                let (val, list_provenance) = self.eval_typed(&args[0], chunk, row);
                let val = val?;
                match val {
                    Value::List(items) => {
                        *provenance = list_provenance.item();
                        items.last().cloned()
                    }
                    _ => None,
                }
            }
            "reverse" => {
                // reverse(list) - returns the list in reverse order
                if args.len() != 1 {
                    return None;
                }
                let (val, list_provenance) = self.eval_typed(&args[0], chunk, row);
                let val = val?;
                match val {
                    Value::List(items) => {
                        *provenance = list_provenance;
                        let reversed: Vec<Value> = items.iter().rev().cloned().collect();
                        Some(Value::List(reversed.into()))
                    }
                    Value::String(s) => {
                        let reversed: String = s.chars().rev().collect();
                        Some(Value::String(reversed.into()))
                    }
                    _ => None,
                }
            }
            // vector(list) - converts a list of numbers to a Vector
            "vector" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::List(items) => {
                        let floats: Vec<f32> = items
                            .iter()
                            .filter_map(|v| match v {
                                // reason: vector() intentionally converts f64 to f32 for storage
                                #[allow(clippy::cast_possible_truncation)]
                                Value::Float64(f) => Some(*f as f32),
                                Value::Int64(i) => Some(*i as f32),
                                _ => None,
                            })
                            .collect();
                        if floats.len() == items.len() {
                            Some(Value::Vector(floats.into()))
                        } else {
                            None
                        }
                    }
                    Value::Vector(v) => Some(Value::Vector(v)),
                    _ => None,
                }
            }
            "all_different" => {
                // ALL_DIFFERENT - ISO GQL predicate (G113)
                // Two calling conventions:
                //   all_different(list)          - check list elements are distinct
                //   ALL_DIFFERENT(var1, var2, ..) - check graph elements are distinct
                if args.is_empty() {
                    return Some(Value::Bool(true));
                }
                if args.len() == 1 {
                    // Single-argument: treat as list check
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    return match val {
                        Value::List(items) => {
                            let mut seen = std::collections::HashSet::new();
                            let all_diff = items.iter().all(|item| {
                                let key = format!("{item:?}");
                                seen.insert(key)
                            });
                            Some(Value::Bool(all_diff))
                        }
                        _ => Some(Value::Bool(true)),
                    };
                }
                // Multi-argument: compare element IDs
                let mut ids: Vec<u64> = Vec::with_capacity(args.len());
                for arg in args {
                    if let FilterExpression::Variable(var) = arg {
                        let col = self.binding(var, chunk)?;
                        if let Some(nid) = col.get_node_id(row) {
                            ids.push(nid.0);
                        } else {
                            let eid = col.get_edge_id(row)?;
                            ids.push(eid.0);
                        }
                    } else {
                        return None;
                    }
                }
                let length = ids.len();
                ids.sort_unstable();
                ids.dedup();
                Some(Value::Bool(ids.len() == length))
            }
            "same" => {
                // SAME - ISO GQL predicate (G114)
                // Two calling conventions:
                //   same(list)          - check list elements are equal
                //   SAME(var1, var2, ..) - check graph elements are identical
                if args.is_empty() {
                    return Some(Value::Bool(true));
                }
                if args.len() == 1 {
                    // Single-argument: treat as list check
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    return match val {
                        Value::List(items) => {
                            let all_same = if items.is_empty() {
                                true
                            } else {
                                items.iter().all(|item| item == &items[0])
                            };
                            Some(Value::Bool(all_same))
                        }
                        _ => Some(Value::Bool(true)),
                    };
                }
                // Multi-argument: compare element IDs
                let mut first_id: Option<u64> = None;
                for arg in args {
                    if let FilterExpression::Variable(var) = arg {
                        let col = self.binding(var, chunk)?;
                        let current_id = if let Some(nid) = col.get_node_id(row) {
                            nid.0
                        } else {
                            col.get_edge_id(row)?.0
                        };
                        match first_id {
                            None => first_id = Some(current_id),
                            Some(fid) if fid != current_id => return Some(Value::Bool(false)),
                            _ => {}
                        }
                    } else {
                        return None;
                    }
                }
                Some(Value::Bool(true))
            }
            "range" => {
                if args.len() < 2 || args.len() > 3 {
                    return None;
                }
                let start = self.eval_expr(&args[0], chunk, row)?;
                let stop = self.eval_expr(&args[1], chunk, row)?;
                let Value::Int64(start_val) = start else {
                    return None;
                };
                let Value::Int64(end_val) = stop else {
                    return None;
                };
                let step = if args.len() == 3 {
                    let s = self.eval_expr(&args[2], chunk, row)?;
                    let Value::Int64(sv) = s else {
                        return None;
                    };
                    if sv == 0 {
                        return None;
                    }
                    sv
                } else {
                    1
                };
                let mut result = Vec::new();
                let mut current = start_val;
                if step > 0 {
                    while current <= end_val {
                        result.push(Value::Int64(current));
                        current += step;
                    }
                } else {
                    while current >= end_val {
                        result.push(Value::Int64(current));
                        current += step;
                    }
                }
                Some(Value::List(result.into()))
            }
            "string_join" => {
                // string_join(list, separator) - join list elements with separator
                if args.len() != 2 {
                    return None;
                }
                let list_val = self.eval_expr(&args[0], chunk, row)?;
                let sep_val = self.eval_expr(&args[1], chunk, row)?;
                match (list_val, sep_val) {
                    (Value::List(items), Value::String(sep)) => {
                        let sep_str: &str = &sep;
                        let joined: String = items
                            .iter()
                            .filter_map(|v| match v {
                                Value::String(s) => Some(s.to_string()),
                                Value::Int64(i) => Some(i.to_string()),
                                Value::Float64(f) => Some(f.to_string()),
                                Value::Bool(b) => Some(b.to_string()),
                                Value::Null => None,
                                other => Some(format!("{other}")),
                            })
                            .collect::<Vec<String>>()
                            .join(sep_str);
                        Some(Value::String(joined.into()))
                    }
                    (Value::Null, _) | (_, Value::Null) => Some(Value::Null),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn eval_string_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "trim" => {
                if args.len() == 1 {
                    // Simple trim(string) - trim whitespace
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    return match val {
                        Value::String(s) => Some(Value::String(s.trim().to_string().into())),
                        _ => None,
                    };
                }
                if args.len() == 3 {
                    // Extended trim(string, chars, mode)
                    // mode: 0=both, 1=leading, 2=trailing
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    let chars_val = self.eval_expr(&args[1], chunk, row)?;
                    let mode_val = self.eval_expr(&args[2], chunk, row)?;
                    let Value::String(s) = val else { return None };
                    let Value::String(chars) = chars_val else {
                        return None;
                    };
                    let Value::Int64(mode) = mode_val else {
                        return None;
                    };
                    let char_set: Vec<char> = chars.chars().collect();
                    let result = match mode {
                        0 => s.trim_matches(|c| char_set.contains(&c)).to_string(),
                        1 => s.trim_start_matches(|c| char_set.contains(&c)).to_string(),
                        2 => s.trim_end_matches(|c| char_set.contains(&c)).to_string(),
                        _ => return None,
                    };
                    return Some(Value::String(result.into()));
                }
                None
            }
            "ltrim" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s.trim_start().to_string().into())),
                    _ => None,
                }
            }
            "rtrim" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s.trim_end().to_string().into())),
                    _ => None,
                }
            }
            "replace" => {
                if args.len() != 3 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let search = self.eval_expr(&args[1], chunk, row)?;
                let replacement = self.eval_expr(&args[2], chunk, row)?;
                match (&val, &search, &replacement) {
                    (Value::String(s), Value::String(from), Value::String(to)) => {
                        Some(Value::String(s.replace(from.as_str(), to.as_str()).into()))
                    }
                    _ => None,
                }
            }
            "substring" => {
                if args.len() < 2 || args.len() > 3 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let start = self.eval_expr(&args[1], chunk, row)?;
                let Value::String(s) = val else {
                    return None;
                };
                let Value::Int64(start_idx) = start else {
                    return None;
                };
                // reason: clamped to >= 0 by max(0), safe to cast to usize
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let start_idx = start_idx.max(0) as usize;
                if args.len() == 3 {
                    let length = self.eval_expr(&args[2], chunk, row)?;
                    let Value::Int64(len) = length else {
                        return None;
                    };
                    // reason: clamped to >= 0 by max(0), safe to cast to usize
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let len = len.max(0) as usize;
                    let chars: String = s.chars().skip(start_idx).take(len).collect();
                    Some(Value::String(chars.into()))
                } else {
                    let chars: String = s.chars().skip(start_idx).collect();
                    Some(Value::String(chars.into()))
                }
            }
            "split" => {
                if args.len() != 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let delim = self.eval_expr(&args[1], chunk, row)?;
                match (&val, &delim) {
                    (Value::String(s), Value::String(d)) => {
                        let parts: Vec<Value> = s
                            .split(d.as_str())
                            .map(|p| Value::String(p.to_string().into()))
                            .collect();
                        Some(Value::List(parts.into()))
                    }
                    _ => None,
                }
            }
            "toupper" | "upper" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s.to_uppercase().into())),
                    _ => None,
                }
            }
            "tolower" | "lower" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s.to_lowercase().into())),
                    _ => None,
                }
            }
            "char_length" | "charlength" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    // reason: char count fits i64 for practical string sizes
                    #[allow(clippy::cast_possible_wrap)]
                    Value::String(s) => Some(Value::Int64(s.chars().count() as i64)),
                    _ => None,
                }
            }
            "left" => {
                if args.len() != 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let len = self.eval_expr(&args[1], chunk, row)?;
                match (&val, &len) {
                    (Value::String(s), Value::Int64(n)) => {
                        // reason: clamped to >= 0 by max(0), safe to cast to usize
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let n = (*n).max(0) as usize;
                        let result: String = s.chars().take(n).collect();
                        Some(Value::String(result.into()))
                    }
                    _ => None,
                }
            }
            "right" => {
                if args.len() != 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let len = self.eval_expr(&args[1], chunk, row)?;
                match (&val, &len) {
                    (Value::String(s), Value::Int64(n)) => {
                        // reason: clamped to >= 0 by max(0), safe to cast to usize
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let n = (*n).max(0) as usize;
                        let char_count = s.chars().count();
                        let skip = char_count.saturating_sub(n);
                        let result: String = s.chars().skip(skip).collect();
                        Some(Value::String(result.into()))
                    }
                    _ => None,
                }
            }
            "octet_length" | "byte_length" => {
                // octet_length(string) - byte length
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    // reason: string byte length fits i64 for practical sizes
                    #[allow(clippy::cast_possible_wrap)]
                    Value::String(s) => Some(Value::Int64(s.len() as i64)),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "normalize" => {
                // normalize(string) - returns the string as-is (NFC normalization).
                // Rust strings are valid UTF-8; full NFC normalization requires the
                // unicode-normalization crate, which is deferred to Phase 2.
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s)),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "isnormalized" => {
                // IS [NFC|NFD|NFKC|NFKD] NORMALIZED - check Unicode normalization form.
                // Args: (string_value) or (string_value, form_name)
                // Default form is NFC when not specified.
                if args.is_empty() || args.len() > 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let form = if args.len() == 2 {
                    match self.eval_expr(&args[1], chunk, row)? {
                        Value::String(s) => s.to_uppercase(),
                        _ => return None,
                    }
                } else {
                    "NFC".to_string()
                };
                match val {
                    Value::String(ref s) => {
                        use unicode_normalization::UnicodeNormalization;
                        let normalized = match form.as_str() {
                            "NFC" => s.nfc().collect::<String>() == s.as_ref(),
                            "NFD" => s.nfd().collect::<String>() == s.as_ref(),
                            "NFKC" => s.nfkc().collect::<String>() == s.as_ref(),
                            "NFKD" => s.nfkd().collect::<String>() == s.as_ref(),
                            _ => return None,
                        };
                        Some(Value::Bool(normalized))
                    }
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn eval_numeric_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "abs" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i.abs())),
                    Value::Float64(f) => Some(Value::Float64(f.abs())),
                    _ => None,
                }
            }
            "ceil" | "ceiling" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i)),
                    Value::Float64(f) => Some(Value::Float64(f.ceil())),
                    _ => None,
                }
            }
            "floor" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i)),
                    Value::Float64(f) => Some(Value::Float64(f.floor())),
                    _ => None,
                }
            }
            "round" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i)),
                    Value::Float64(f) => Some(Value::Float64(f.round())),
                    _ => None,
                }
            }
            "sqrt" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).sqrt())),
                    Value::Float64(f) => Some(Value::Float64(f.sqrt())),
                    _ => None,
                }
            }
            "sign" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i.signum())),
                    Value::Float64(f) => {
                        if f > 0.0 {
                            Some(Value::Int64(1))
                        } else if f < 0.0 {
                            Some(Value::Int64(-1))
                        } else {
                            Some(Value::Int64(0))
                        }
                    }
                    _ => None,
                }
            }
            "power" | "pow" => {
                if args.len() != 2 {
                    return None;
                }
                let base_val = self.eval_expr(&args[0], chunk, row)?;
                let exp_val = self.eval_expr(&args[1], chunk, row)?;
                let base = match base_val {
                    Value::Int64(i) => i as f64,
                    Value::Float64(f) => f,
                    _ => return None,
                };
                let exponent = match exp_val {
                    Value::Int64(i) => i as f64,
                    Value::Float64(f) => f,
                    _ => return None,
                };
                Some(Value::Float64(base.powf(exponent)))
            }
            "rand" | "random" => {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                row.hash(&mut hasher);
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .ok()?
                    .as_nanos()
                    .hash(&mut hasher);
                let hash = hasher.finish();
                let random = (hash as f64) / (u64::MAX as f64);
                Some(Value::Float64(random))
            }
            "log" | "ln" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).ln())),
                    Value::Float64(f) => Some(Value::Float64(f.ln())),
                    _ => None,
                }
            }
            "log10" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).log10())),
                    Value::Float64(f) => Some(Value::Float64(f.log10())),
                    _ => None,
                }
            }
            "log2" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).log2())),
                    Value::Float64(f) => Some(Value::Float64(f.log2())),
                    _ => None,
                }
            }
            "exp" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).exp())),
                    Value::Float64(f) => Some(Value::Float64(f.exp())),
                    _ => None,
                }
            }
            "e" => Some(Value::Float64(std::f64::consts::E)),
            "pi" => Some(Value::Float64(std::f64::consts::PI)),
            "sin" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).sin())),
                    Value::Float64(f) => Some(Value::Float64(f.sin())),
                    _ => None,
                }
            }
            "cos" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).cos())),
                    Value::Float64(f) => Some(Value::Float64(f.cos())),
                    _ => None,
                }
            }
            "tan" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).tan())),
                    Value::Float64(f) => Some(Value::Float64(f.tan())),
                    _ => None,
                }
            }
            "asin" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).asin())),
                    Value::Float64(f) => Some(Value::Float64(f.asin())),
                    _ => None,
                }
            }
            "acos" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).acos())),
                    Value::Float64(f) => Some(Value::Float64(f.acos())),
                    _ => None,
                }
            }
            "atan" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).atan())),
                    Value::Float64(f) => Some(Value::Float64(f.atan())),
                    _ => None,
                }
            }
            "atan2" => {
                if args.len() != 2 {
                    return None;
                }
                let y_val = self.eval_expr(&args[0], chunk, row)?;
                let x_val = self.eval_expr(&args[1], chunk, row)?;
                let y = match y_val {
                    Value::Int64(i) => i as f64,
                    Value::Float64(f) => f,
                    _ => return None,
                };
                let x = match x_val {
                    Value::Int64(i) => i as f64,
                    Value::Float64(f) => f,
                    _ => return None,
                };
                Some(Value::Float64(y.atan2(x)))
            }
            "degrees" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).to_degrees())),
                    Value::Float64(f) => Some(Value::Float64(f.to_degrees())),
                    _ => None,
                }
            }
            "radians" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).to_radians())),
                    Value::Float64(f) => Some(Value::Float64(f.to_radians())),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn eval_temporal_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "date" | "todate" => {
                if args.is_empty() {
                    return Some(Value::Date(grafeo_common::types::Date::today()));
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => grafeo_common::types::Date::parse(&s).map(Value::Date),
                    Value::Timestamp(ts) => Some(Value::Date(ts.to_date())),
                    Value::Date(_) => Some(val),
                    Value::Map(m) => {
                        let year = i32::try_from(map_int(&m, "year")?).ok()?;
                        let month = u32::try_from(map_int_or(&m, "month", 1)?).ok()?;
                        let day = u32::try_from(map_int_or(&m, "day", 1)?).ok()?;
                        grafeo_common::types::Date::from_ymd(year, month, day).map(Value::Date)
                    }
                    _ => None,
                }
            }
            "time" | "totime" | "local_time" => {
                if args.is_empty() {
                    return Some(Value::Time(grafeo_common::types::Time::now()));
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => grafeo_common::types::Time::parse(&s).map(Value::Time),
                    Value::Timestamp(ts) => Some(Value::Time(ts.to_time())),
                    Value::Time(_) => Some(val),
                    Value::Map(m) => {
                        let hour = u32::try_from(map_int_or(&m, "hour", 0)?).ok()?;
                        let minute = u32::try_from(map_int_or(&m, "minute", 0)?).ok()?;
                        let second = u32::try_from(map_int_or(&m, "second", 0)?).ok()?;
                        let nanosecond = u32::try_from(map_int_or(&m, "nanosecond", 0)?).ok()?;
                        grafeo_common::types::Time::from_hms_nano(hour, minute, second, nanosecond)
                            .map(Value::Time)
                    }
                    _ => None,
                }
            }
            "datetime" | "localdatetime" | "local_datetime" | "todatetime" => {
                if args.is_empty() {
                    return Some(Value::Timestamp(grafeo_common::types::Timestamp::now()));
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => {
                        // Parse ISO datetime: try Date first, then full timestamp
                        if let Some(d) = grafeo_common::types::Date::parse(&s) {
                            return Some(Value::Timestamp(d.to_timestamp()));
                        }
                        // Try full ISO format: YYYY-MM-DDTHH:MM:SS[.fff][Z|+HH:MM]
                        if let Some(pos) = s.find('T') {
                            let date_part = &s[..pos];
                            let time_part = &s[pos + 1..];
                            if let (Some(d), Some(t)) = (
                                grafeo_common::types::Date::parse(date_part),
                                grafeo_common::types::Time::parse(time_part),
                            ) {
                                return Some(Value::Timestamp(
                                    grafeo_common::types::Timestamp::from_date_time(d, t),
                                ));
                            }
                        }
                        None
                    }
                    Value::Timestamp(_) => Some(val),
                    Value::Map(m) => {
                        let year = i32::try_from(map_int(&m, "year")?).ok()?;
                        let month = u32::try_from(map_int_or(&m, "month", 1)?).ok()?;
                        let day = u32::try_from(map_int_or(&m, "day", 1)?).ok()?;
                        let hour = u32::try_from(map_int_or(&m, "hour", 0)?).ok()?;
                        let minute = u32::try_from(map_int_or(&m, "minute", 0)?).ok()?;
                        let second = u32::try_from(map_int_or(&m, "second", 0)?).ok()?;
                        let nanosecond = u32::try_from(map_int_or(&m, "nanosecond", 0)?).ok()?;
                        let date = grafeo_common::types::Date::from_ymd(year, month, day)?;
                        let time = grafeo_common::types::Time::from_hms_nano(
                            hour, minute, second, nanosecond,
                        )?;
                        Some(Value::Timestamp(
                            grafeo_common::types::Timestamp::from_date_time(date, time),
                        ))
                    }
                    _ => None,
                }
            }
            "duration" | "toduration" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => {
                        grafeo_common::types::Duration::parse(&s).map(Value::Duration)
                    }
                    Value::Duration(_) => Some(val),
                    Value::Map(m) => {
                        let years = map_int_or(&m, "years", 0)?;
                        let months = map_int_or(&m, "months", 0)?;
                        let weeks = map_int_or(&m, "weeks", 0)?;
                        let days = map_int_or(&m, "days", 0)?;
                        let hours = map_int_or(&m, "hours", 0)?;
                        let minutes = map_int_or(&m, "minutes", 0)?;
                        let seconds = map_int_or(&m, "seconds", 0)?;
                        let nanoseconds = map_int_or(&m, "nanoseconds", 0)?;
                        let total_months = years * 12 + months;
                        let total_days = weeks * 7 + days;
                        let total_nanos = hours * 3_600_000_000_000
                            + minutes * 60_000_000_000
                            + seconds * 1_000_000_000
                            + nanoseconds;
                        Some(Value::Duration(grafeo_common::types::Duration::new(
                            total_months,
                            total_days,
                            total_nanos,
                        )))
                    }
                    _ => None,
                }
            }
            "tozoneddatetime" | "zoneddatetime" | "zoned_datetime" => {
                if args.is_empty() {
                    return Some(Value::ZonedDatetime(
                        grafeo_common::types::ZonedDatetime::from_timestamp_offset(
                            grafeo_common::types::Timestamp::now(),
                            0,
                        ),
                    ));
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => {
                        grafeo_common::types::ZonedDatetime::parse(&s).map(Value::ZonedDatetime)
                    }
                    Value::Timestamp(ts) => Some(Value::ZonedDatetime(
                        grafeo_common::types::ZonedDatetime::from_timestamp_offset(ts, 0),
                    )),
                    Value::ZonedDatetime(_) => Some(val),
                    _ => None,
                }
            }
            "tozonedtime" | "zonedtime" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::String(s) => {
                        let t = grafeo_common::types::Time::parse(&s)?;
                        if t.offset_seconds().is_some() {
                            Some(Value::Time(t))
                        } else {
                            None
                        }
                    }
                    Value::Time(t) if t.offset_seconds().is_some() => Some(val),
                    _ => None,
                }
            }
            "current_date" | "currentdate" => {
                Some(Value::Date(grafeo_common::types::Date::today()))
            }
            "current_time" | "currenttime" => Some(Value::Time(grafeo_common::types::Time::now())),
            "now" | "current_timestamp" | "currenttimestamp" => {
                Some(Value::Timestamp(grafeo_common::types::Timestamp::now()))
            }
            "timestamp" => Some(Value::Int64(
                grafeo_common::types::Timestamp::now().as_millis(),
            )),
            "year" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Date(d) => Some(Value::Int64(i64::from(d.year()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_date().year()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_date().year())))
                    }
                    _ => None,
                }
            }
            "month" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Date(d) => Some(Value::Int64(i64::from(d.month()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_date().month()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_date().month())))
                    }
                    _ => None,
                }
            }
            "day" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Date(d) => Some(Value::Int64(i64::from(d.day()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_date().day()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_date().day())))
                    }
                    _ => None,
                }
            }
            "hour" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Time(t) => Some(Value::Int64(i64::from(t.hour()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_time().hour()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_time().hour())))
                    }
                    _ => None,
                }
            }
            "minute" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Time(t) => Some(Value::Int64(i64::from(t.minute()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_time().minute()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_time().minute())))
                    }
                    _ => None,
                }
            }
            "second" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Time(t) => Some(Value::Int64(i64::from(t.second()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_time().second()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_time().second())))
                    }
                    _ => None,
                }
            }
            "date_trunc" | "truncate" => {
                if args.len() < 2 {
                    return None;
                }
                let unit = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_lowercase(),
                    _ => return None,
                };
                let val = self.eval_expr(&args[1], chunk, row)?;
                match val {
                    Value::Date(d) => Some(Value::Date(d.truncate(&unit)?)),
                    Value::Time(t) => Some(Value::Time(t.truncate(&unit)?)),
                    Value::Timestamp(ts) => Some(Value::Timestamp(ts.truncate(&unit)?)),
                    Value::ZonedDatetime(zdt) => Some(Value::ZonedDatetime(zdt.truncate(&unit)?)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn eval_path_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
        provenance: &mut ValueProvenance,
    ) -> Option<Value> {
        match name {
            "path" => {
                if args.len() == 2 {
                    // path(nodes_list, edges_list) - construct from component lists
                    let nodes_val = self.eval_expr(&args[0], chunk, row)?;
                    let edges_val = self.eval_expr(&args[1], chunk, row)?;
                    match (&nodes_val, &edges_val) {
                        (Value::Null, _) | (_, Value::Null) => Some(Value::Null),
                        (Value::List(nodes), Value::List(edges)) => {
                            if nodes.is_empty() || edges.len() != nodes.len() - 1 {
                                return None;
                            }
                            Some(Value::Path {
                                nodes: Arc::from(nodes.as_ref()),
                                edges: Arc::from(edges.as_ref()),
                            })
                        }
                        _ => None,
                    }
                } else {
                    // path(node1, edge1, node2, ...) - alternating nodes and edges
                    if args.is_empty() || args.len().is_multiple_of(2) {
                        return None;
                    }
                    let mut nodes = Vec::with_capacity(args.len() / 2 + 1);
                    let mut edges = Vec::with_capacity(args.len() / 2);
                    for (i, arg) in args.iter().enumerate() {
                        let val = self.eval_expr(arg, chunk, row)?;
                        if i % 2 == 0 {
                            nodes.push(val);
                        } else {
                            edges.push(val);
                        }
                    }
                    Some(Value::Path {
                        nodes: nodes.into(),
                        edges: edges.into(),
                    })
                }
            }
            "nodes" => {
                // nodes(path) - extracts nodes from a path value
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Path { nodes, .. } => {
                        // Resolve Int64 node IDs to property maps for property access
                        let snap_epoch = self
                            .viewing_epoch
                            .unwrap_or_else(|| self.store.current_epoch());
                        let resolved: Vec<Value> = nodes
                            .iter()
                            .map(|n| {
                                if let Value::Int64(id) = n {
                                    // reason: ID encoding: i64 <-> u64 round-trip
                                    #[allow(clippy::cast_sign_loss)]
                                    let node_id = NodeId(*id as u64);
                                    if let Some(node) = self.resolve_node(node_id) {
                                        let mut map = BTreeMap::new();
                                        // reason: entity IDs stored as i64, standard encoding
                                        #[allow(clippy::cast_possible_wrap)]
                                        map.insert(
                                            PropertyKey::new("_id"),
                                            Value::Int64(node.id.as_u64() as i64),
                                        );
                                        // Route through the snapshot-aware accessor so that
                                        // uncommitted label ops in the writing transaction
                                        // are reflected here.
                                        let label_set = self.store.read_node_labels_visible(
                                            node_id,
                                            snap_epoch,
                                            self.transaction_id,
                                        );
                                        let labels: Vec<Value> =
                                            label_set.into_iter().map(Value::String).collect();
                                        map.insert(
                                            PropertyKey::new("_labels"),
                                            Value::List(labels.into()),
                                        );
                                        for (key, value) in &node.properties {
                                            map.insert(key.clone(), value.clone());
                                        }
                                        Value::Map(Arc::new(map))
                                    } else {
                                        n.clone()
                                    }
                                } else {
                                    n.clone()
                                }
                            })
                            .collect();
                        Some(Value::List(resolved.into()))
                    }
                    Value::Map(map) => map.get(&PropertyKey::from("nodes")).cloned(),
                    Value::List(items) => {
                        // Legacy: alternating node, edge, node, edge, ...
                        let nodes: Vec<Value> = items.iter().step_by(2).cloned().collect();
                        Some(Value::List(nodes.into()))
                    }
                    _ => None,
                }
            }
            "edges" | "relationships" => {
                // edges(path) / relationships(path) - extracts edges from a path value
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let result = match val {
                    Value::Path { edges, .. } => Some(Value::List(edges)),
                    Value::Map(map) => map.get(&PropertyKey::from("edges")).cloned(),
                    Value::List(items) => {
                        // Legacy: alternating node, edge, node, edge, ...
                        let edges: Vec<Value> = items.iter().skip(1).step_by(2).cloned().collect();
                        Some(Value::List(edges.into()))
                    }
                    _ => None,
                };
                if matches!(result, Some(Value::List(_))) {
                    *provenance = ValueProvenance::EdgeList;
                }
                result
            }
            "isacyclic" => {
                // isAcyclic(path) - true if no node appears more than once
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Path { nodes, .. } => {
                        let mut seen = std::collections::HashSet::new();
                        let acyclic = nodes
                            .iter()
                            .all(|n| seen.insert(HashableValue::new(n.clone())));
                        Some(Value::Bool(acyclic))
                    }
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "issimple" => {
                // isSimple(path) - true if no node repeats except possibly first == last
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Path { nodes, .. } => {
                        if nodes.is_empty() {
                            return Some(Value::Bool(true));
                        }
                        let mut seen = std::collections::HashSet::new();
                        let simple = nodes.iter().enumerate().all(|(i, n)| {
                            let hv = HashableValue::new(n.clone());
                            if !seen.insert(hv) {
                                // Duplicate allowed only if last node == first node
                                i == nodes.len() - 1 && n == &nodes[0]
                            } else {
                                true
                            }
                        });
                        Some(Value::Bool(simple))
                    }
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "istrail" => {
                // isTrail(path) - true if no edge repeats
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Path { edges, .. } => {
                        let mut seen = std::collections::HashSet::new();
                        let trail = edges
                            .iter()
                            .all(|e| seen.insert(HashableValue::new(e.clone())));
                        Some(Value::Bool(trail))
                    }
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Coerces a `Value` to a borrowed or owned float slice for vector math.
    ///
    /// Handles both `Value::Vector` (native) and `Value::List` (GQL inline literal
    /// form `[0.9, 0.1, 0.0]` which the parser translates to a list of Float64/Int64).
    fn coerce_to_float_vec(val: &Value) -> Option<std::borrow::Cow<'_, [f32]>> {
        match val {
            Value::Vector(v) => Some(std::borrow::Cow::Borrowed(v.as_ref())),
            Value::List(list) => {
                let mut vec = Vec::with_capacity(list.len());
                for item in list.iter() {
                    match item {
                        // GQL numeric literals are Float64; f64→f32 precision loss is intentional
                        // since all HNSW indexes store f32 components.
                        #[allow(clippy::cast_possible_truncation)]
                        Value::Float64(f) => vec.push(*f as f32),
                        Value::Int64(i) => vec.push(*i as f32),
                        _ => return None,
                    }
                }
                Some(std::borrow::Cow::Owned(vec))
            }
            _ => None,
        }
    }

    fn eval_vector_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "cosine_similarity" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = Self::coerce_to_float_vec(&a_val)?;
                let b = Self::coerce_to_float_vec(&b_val)?;
                if a.len() != b.len() {
                    return None;
                }
                Some(Value::Float64(
                    crate::index::vector::cosine_similarity(&a, &b) as f64,
                ))
            }
            "cosine_distance" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = Self::coerce_to_float_vec(&a_val)?;
                let b = Self::coerce_to_float_vec(&b_val)?;
                if a.len() != b.len() {
                    return None;
                }
                // cosine_distance = 1 - cosine_similarity, range [0, 2]
                Some(Value::Float64(
                    1.0 - crate::index::vector::cosine_similarity(&a, &b) as f64,
                ))
            }
            "euclidean_distance" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = Self::coerce_to_float_vec(&a_val)?;
                let b = Self::coerce_to_float_vec(&b_val)?;
                if a.len() != b.len() {
                    return None;
                }
                Some(Value::Float64(
                    crate::index::vector::euclidean_distance(&a, &b) as f64,
                ))
            }
            "dot_product" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = Self::coerce_to_float_vec(&a_val)?;
                let b = Self::coerce_to_float_vec(&b_val)?;
                if a.len() != b.len() {
                    return None;
                }
                Some(Value::Float64(
                    crate::index::vector::dot_product(&a, &b) as f64
                ))
            }
            "manhattan_distance" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = Self::coerce_to_float_vec(&a_val)?;
                let b = Self::coerce_to_float_vec(&b_val)?;
                if a.len() != b.len() {
                    return None;
                }
                Some(Value::Float64(
                    crate::index::vector::manhattan_distance(&a, &b) as f64,
                ))
            }
            _ => None,
        }
    }

    #[cfg(feature = "text-index")]
    fn eval_text_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "text_score" | "text_match" => {
                if args.len() != 2 {
                    return None;
                }

                // First arg must be a property access (e.g., n.body)
                let FilterExpression::Property { variable, property } = &args[0] else {
                    return None;
                };

                // Get node_id from the chunk
                let col = self.binding(variable.as_str(), chunk)?;
                let node_id = col.get_node_id(row)?;

                // Second arg: query string
                let query_val = self.eval_expr(&args[1], chunk, row)?;
                let Value::String(query_str) = &query_val else {
                    return None;
                };

                // Guard: skip if node does not exist.
                self.resolve_node(node_id)?;
                // Route through the snapshot-aware accessor so that uncommitted
                // label ops in the writing transaction are reflected here.
                let snap_epoch = self
                    .viewing_epoch
                    .unwrap_or_else(|| self.store.current_epoch());
                let label_set =
                    self.store
                        .read_node_labels_visible(node_id, snap_epoch, self.transaction_id);
                // A viewing epoch is authoritative even without a transaction.
                // A valid transaction additionally enables overlays and SSI.
                let score = if let Some(epoch) = self.viewing_epoch {
                    let tx = self.transaction_id.unwrap_or(TransactionId::INVALID);
                    let mut score = None;
                    for label in &label_set {
                        match self
                            .store
                            .score_text_visible(node_id, label, property, query_str, epoch, tx)
                        {
                            Ok(Some(value)) => {
                                score = Some(value);
                                break;
                            }
                            Ok(None) => {}
                            Err(error) => {
                                self.error.set(Some(match self.error.take() {
                                    Some(first) => first,
                                    None => error,
                                }));
                                return None;
                            }
                        }
                    }
                    score?
                } else {
                    label_set.iter().find_map(|label| {
                        self.store.score_text(node_id, label, property, query_str)
                    })?
                };

                if name == "text_match" {
                    Some(Value::Bool(score > 0.0))
                } else {
                    Some(Value::Float64(score))
                }
            }
            _ => None,
        }
    }

    #[cfg(not(feature = "text-index"))]
    fn eval_text_fn(
        &self,
        _name: &str,
        _args: &[FilterExpression],
        _chunk: &DataChunk,
        _row: usize,
    ) -> Option<Value> {
        None
    }

    fn eval_session_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "session_user" => {
                // session_user() - returns the current session user
                // For embedded databases, returns a default user string
                Some(Value::String("default".into()))
            }
            // ISO/IEC 39075 Section 17.1 / Section 21: session schema/graph references
            "current_schema" => Some(self.session_context.current_schema.as_ref().map_or_else(
                || Value::String("default".into()),
                |s| Value::String(s.clone().into()),
            )),
            "current_graph" => Some(self.session_context.current_graph.as_ref().map_or_else(
                || Value::String("default".into()),
                |g| Value::String(g.clone().into()),
            )),
            "home_schema" | "home_graph" => {
                // Home schema/graph: not configurable yet, returns null
                Some(Value::Null)
            }
            // Grafeo extension: info() returns database metadata as a map
            "info" => Some(self.session_context.db_info.get().clone()),
            // Grafeo extension: schema() returns schema metadata as a map
            "schema" => Some(self.session_context.schema_info.get().clone()),
            "nullif" => {
                // NULLIF(expr1, expr2) - returns NULL if expr1 = expr2, else expr1
                if args.len() != 2 {
                    return None;
                }
                let val1 = self.eval_expr(&args[0], chunk, row)?;
                let val2 = self.eval_expr(&args[1], chunk, row)?;
                // Three-valued: NULLIF(NULL, x) = NULL; NULLIF(x, NULL) = x
                if val1.is_null() || val2.is_null() {
                    Some(val1)
                } else if Self::values_equal(&val1, &val2) {
                    Some(Value::Null)
                } else {
                    Some(val1)
                }
            }
            _ => None,
        }
    }

    fn eval_case(
        &self,
        operand: Option<&FilterExpression>,
        when_clauses: &[(FilterExpression, FilterExpression)],
        else_clause: Option<&FilterExpression>,
        chunk: &DataChunk,
        row: usize,
        provenance: &mut ValueProvenance,
    ) -> Option<Value> {
        if let Some(test_expr) = operand {
            // Simple CASE: CASE expr WHEN val1 THEN res1 ...
            // Use unwrap_or(Null) so a NULL test expression falls through to ELSE
            // rather than short-circuiting the entire CASE (NULL != anything).
            let test_val = self.eval_expr(test_expr, chunk, row).unwrap_or(Value::Null);
            for (when_expr, then_expr) in when_clauses {
                let when_val = self.eval_expr(when_expr, chunk, row).unwrap_or(Value::Null);
                // Three-valued logic: NULL never matches anything in simple CASE
                if !test_val.is_null()
                    && !when_val.is_null()
                    && Self::values_equal(&test_val, &when_val)
                {
                    return self.eval_expr_inner(then_expr, chunk, row, provenance);
                }
            }
        } else {
            // Searched CASE: CASE WHEN cond1 THEN res1 ...
            // Use unwrap_or(Null) so a NULL/UNKNOWN condition falls through
            // to the next WHEN or ELSE (three-valued logic: only TRUE matches).
            for (when_expr, then_expr) in when_clauses {
                let when_val = self.eval_expr(when_expr, chunk, row).unwrap_or(Value::Null);
                if when_val.as_bool() == Some(true) {
                    return self.eval_expr_inner(then_expr, chunk, row, provenance);
                }
            }
        }
        // No match - return ELSE or NULL
        if let Some(else_expr) = else_clause {
            self.eval_expr_inner(else_expr, chunk, row, provenance)
        } else {
            Some(Value::Null)
        }
    }

    fn eval_unary_op(&self, op: UnaryFilterOp, val: Option<Value>) -> Option<Value> {
        match op {
            UnaryFilterOp::Not => {
                let v = val?.as_bool()?;
                Some(Value::Bool(!v))
            }
            UnaryFilterOp::IsNull => Some(Value::Bool(
                val.is_none() || matches!(val, Some(Value::Null)),
            )),
            UnaryFilterOp::IsNotNull => Some(Value::Bool(
                val.is_some() && !matches!(val, Some(Value::Null)),
            )),
            UnaryFilterOp::Neg => match val? {
                Value::Int64(i) => i.checked_neg().map(Value::Int64),
                Value::Float64(f) => Some(Value::Float64(-f)),
                _ => None,
            },
        }
    }

    /// Structural equality for DISTINCT, GROUP BY, and list/map comparison.
    ///
    /// Treats NULL == NULL as `true` (grouping semantics). For SQL/GQL
    /// comparison operators, use [`eval_binary_op`] which returns UNKNOWN
    /// when either operand is NULL.
    fn values_equal(left: &Value, right: &Value) -> bool {
        match (left, right) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int64(a), Value::Int64(b)) => a == b,
            (Value::Float64(a), Value::Float64(b)) => (a - b).abs() < f64::EPSILON,
            (Value::Date(a), Value::Date(b)) => a == b,
            (Value::Time(a), Value::Time(b)) => a == b,
            (Value::Timestamp(a), Value::Timestamp(b)) => a == b,
            (Value::Duration(a), Value::Duration(b)) => a == b,
            (Value::ZonedDatetime(a), Value::ZonedDatetime(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Int64(a), Value::Float64(b)) | (Value::Float64(b), Value::Int64(a)) => {
                (*a as f64 - b).abs() < f64::EPSILON
            }
            // RDF stores numeric literals as strings; allow cross-type equality
            (Value::String(s), Value::Int64(i)) | (Value::Int64(i), Value::String(s)) => {
                s.parse::<i64>().is_ok_and(|n| n == *i)
            }
            (Value::String(s), Value::Float64(f)) | (Value::Float64(f), Value::String(s)) => {
                s.parse::<f64>().is_ok_and(|n| (n - f).abs() < f64::EPSILON)
            }
            (Value::List(a), Value::List(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| Self::values_equal(x, y))
            }
            (Value::Map(a), Value::Map(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|((k1, v1), (k2, v2))| k1 == k2 && Self::values_equal(v1, v2))
            }
            (
                Value::Path {
                    nodes: n1,
                    edges: e1,
                },
                Value::Path {
                    nodes: n2,
                    edges: e2,
                },
            ) => {
                n1.len() == n2.len()
                    && e1.len() == e2.len()
                    && n1
                        .iter()
                        .zip(n2.iter())
                        .all(|(a, b)| Self::values_equal(a, b))
                    && e1
                        .iter()
                        .zip(e2.iter())
                        .all(|(a, b)| Self::values_equal(a, b))
            }
            _ => false,
        }
    }

    /// Checks if a value matches a GQL type name (used by IS TYPED).
    fn value_matches_type(val: &Value, type_name: &str) -> bool {
        match type_name {
            "BOOLEAN" | "BOOL" => matches!(val, Value::Bool(_)),
            "INTEGER" | "INT" | "INT64" => matches!(val, Value::Int64(_)),
            "FLOAT" | "FLOAT64" | "DOUBLE" => matches!(val, Value::Float64(_)),
            "STRING" => matches!(val, Value::String(_)),
            "LIST" => matches!(val, Value::List(_)),
            "MAP" | "RECORD" => matches!(val, Value::Map(_)),
            "NULL" => matches!(val, Value::Null),
            "DATE" => matches!(val, Value::Date(_)),
            "TIME" => matches!(val, Value::Time(_)),
            "DATETIME" | "TIMESTAMP" => matches!(val, Value::Timestamp(_)),
            "DURATION" => matches!(val, Value::Duration(_)),
            "PATH" => matches!(val, Value::Path { .. }),
            "NODE" | "EDGE" | "GRAPH" => false, // element refs not stored as values
            _ => false,
        }
    }

    /// Coerces a value to a target GQL type. Returns None if coercion is impossible.
    fn coerce_to_type(val: Value, type_name: &str) -> Option<Value> {
        match type_name.to_uppercase().as_str() {
            "INTEGER" | "INT" | "INT64" => match val {
                Value::Int64(_) => Some(val),
                // reason: type coercion intentionally truncates float to int
                #[allow(clippy::cast_possible_truncation)]
                Value::Float64(f) => Some(Value::Int64(f as i64)),
                Value::String(ref s) => s.parse::<i64>().ok().map(Value::Int64),
                Value::Bool(b) => Some(Value::Int64(i64::from(b))),
                _ => None,
            },
            "FLOAT" | "FLOAT64" | "DOUBLE" => match val {
                Value::Float64(_) => Some(val),
                Value::Int64(i) => Some(Value::Float64(i as f64)),
                Value::String(ref s) => s.parse::<f64>().ok().map(Value::Float64),
                _ => None,
            },
            "STRING" => match val {
                Value::String(_) => Some(val),
                other => Some(Value::String(other.to_string().into())),
            },
            "BOOLEAN" | "BOOL" => match val {
                Value::Bool(_) => Some(val),
                Value::String(ref s) => match s.to_lowercase().as_str() {
                    "true" => Some(Value::Bool(true)),
                    "false" => Some(Value::Bool(false)),
                    _ => None,
                },
                _ => None,
            },
            _ => {
                // For unknown types, keep value as-is if it already matches
                if Self::value_matches_type(&val, &type_name.to_uppercase()) {
                    Some(val)
                } else {
                    None
                }
            }
        }
    }

    /// Applies the same scalar predicate used by expression evaluation to an
    /// indexed candidate. NULL never passes a WHERE predicate.
    pub fn matches_property_index_predicate(
        value: &Value,
        predicate: crate::graph::PropertyIndexPredicate<'_>,
    ) -> bool {
        use crate::graph::PropertyIndexPredicate;
        if value.is_null() {
            return false;
        }
        match predicate {
            PropertyIndexPredicate::Equal(expected) => {
                !expected.is_null() && Self::values_equal(value, expected)
            }
            PropertyIndexPredicate::In(values) => values
                .iter()
                .any(|expected| !expected.is_null() && Self::values_equal(value, expected)),
            PropertyIndexPredicate::Range {
                min,
                max,
                min_inclusive,
                max_inclusive,
            } => {
                min.is_none_or(|bound| {
                    Self::compare_values(value, bound)
                        .is_some_and(|order| order > 0 || (min_inclusive && order == 0))
                }) && max.is_none_or(|bound| {
                    Self::compare_values(value, bound)
                        .is_some_and(|order| order < 0 || (max_inclusive && order == 0))
                })
            }
        }
    }

    fn compare_values(left: &Value, right: &Value) -> Option<i32> {
        match (left, right) {
            (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b) as i32),
            (Value::Float64(a), Value::Float64(b)) => {
                if a < b {
                    Some(-1)
                } else if a > b {
                    Some(1)
                } else {
                    Some(0)
                }
            }
            (Value::String(a), Value::String(b)) => Some(a.cmp(b) as i32),
            (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b).map(|o| o as i32),
            (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)).map(|o| o as i32),
            // RDF stores numeric literals as strings; allow cross-type comparison
            (Value::String(s), Value::Int64(i)) => s
                .parse::<f64>()
                .ok()
                .and_then(|n| n.partial_cmp(&(*i as f64)).map(|o| o as i32)),
            (Value::Int64(i), Value::String(s)) => s
                .parse::<f64>()
                .ok()
                .and_then(|n| (*i as f64).partial_cmp(&n).map(|o| o as i32)),
            (Value::String(s), Value::Float64(f)) => s
                .parse::<f64>()
                .ok()
                .and_then(|n| n.partial_cmp(f).map(|o| o as i32)),
            (Value::Float64(f), Value::String(s)) => s
                .parse::<f64>()
                .ok()
                .and_then(|n| f.partial_cmp(&n).map(|o| o as i32)),
            // Temporal comparisons
            (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b) as i32),
            (Value::Date(a), Value::Date(b)) => Some(a.cmp(b) as i32),
            (Value::Time(a), Value::Time(b)) => Some(a.cmp(b) as i32),
            _ => None,
        }
    }
}

impl Predicate for ExpressionPredicate {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> Result<bool, super::OperatorError> {
        Ok(
            match self
                .eval_at(chunk, row)
                .map_err(|error| super::OperatorError::Execution(error.to_string()))?
            {
                Some(Value::Bool(b)) => b,
                _ => false,
            },
        )
    }
}

/// A filter operator that applies a predicate to filter rows.
pub struct FilterOperator {
    /// Child operator to read from.
    child: Box<dyn Operator>,
    /// Predicate to apply.
    predicate: Box<dyn Predicate>,
    /// The installed query owner must remain observable while all rows are rejected.
    cancellation: Option<crate::execution::QueryCancellationToken>,
    /// Text-index (label, property) pairs whose reads must be recorded for SSI
    /// anti-phantom detection.  Set by the planner via `with_text_index_reads`
    /// when the filter carries a `text_match`/`text_score` predicate that was
    /// NOT pushed down to a `TextScanOperator`.
    ///
    /// The recording fires exactly once on the **first execution poll** (or when
    /// the operator is decomposed for push-based execution), regardless of
    /// whether any rows are produced.  This is robust to physical-plan caching
    /// (the operator is freshly polled every execution) and to the 0-row case
    /// (where the per-row `eval_text_fn` path is never reached).
    #[cfg(feature = "text-index")]
    text_index_reads: Vec<(String, String)>,
    /// Guard: ensures the text-index reads are recorded exactly once.
    #[cfg(feature = "text-index")]
    text_reads_recorded: bool,
    /// Graph store — used only to call `score_text_visible` for index-read
    /// recording.  `None` when `text_index_reads` is empty (default).
    #[cfg(feature = "text-index")]
    text_store: Option<Arc<dyn GraphStoreSearch>>,
    /// Snapshot epoch for the recording call.  Populated by `with_text_index_reads`.
    #[cfg(feature = "text-index")]
    text_epoch: Option<EpochId>,
    /// Transaction ID for the recording call.  `None` → not Serializable → skip.
    #[cfg(feature = "text-index")]
    text_transaction_id: Option<TransactionId>,
}

impl FilterOperator {
    fn check_cancellation(&self) -> Result<(), super::OperatorError> {
        if let Some(token) = &self.cancellation {
            token.check()?;
        }
        Ok(())
    }

    /// Creates a new filter operator.
    pub fn new(child: Box<dyn Operator>, predicate: Box<dyn Predicate>) -> Self {
        Self {
            child,
            predicate,
            cancellation: None,
            #[cfg(feature = "text-index")]
            text_index_reads: Vec::new(),
            #[cfg(feature = "text-index")]
            text_reads_recorded: false,
            #[cfg(feature = "text-index")]
            text_store: None,
            #[cfg(feature = "text-index")]
            text_epoch: None,
            #[cfg(feature = "text-index")]
            text_transaction_id: None,
        }
    }

    /// Attaches text-index read pairs and the transaction context needed to
    /// record them at execution time.
    ///
    /// Called by the planner instead of the old plan-time side-effect call.
    /// The `(label, property)` pairs are derived from the complete predicate
    /// walk in `collect_text_predicate_pairs`.  Recording fires once on the
    /// first poll (see `record_text_index_reads_once`).
    #[cfg(feature = "text-index")]
    pub fn with_text_index_reads(
        mut self,
        pairs: Vec<(String, String)>,
        store: Arc<dyn GraphStoreSearch>,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.text_index_reads = pairs;
        self.text_store = Some(store);
        self.text_epoch = Some(epoch);
        self.text_transaction_id = transaction_id;
        self
    }

    /// Fires the text-index read recording exactly once.
    ///
    /// Recorded at execution time (first poll), not plan time, so it is robust
    /// to logical-plan caching and fires per-transaction with live context.
    ///
    /// Qualifies the retained epoch even without an active transaction. Uses `score_text_visible` with
    /// `NodeId::INVALID` as a sentinel — the implementation records the index
    /// read before attempting to look up the node, so the recording fires even
    /// for a non-existent node ID.
    #[cfg(feature = "text-index")]
    ///
    /// # Errors
    /// Propagates temporal index qualification failures.
    pub fn record_text_index_reads_once(&mut self) -> Result<(), super::OperatorError> {
        if self.text_reads_recorded {
            return Ok(());
        }

        let tx = self.text_transaction_id.unwrap_or(TransactionId::INVALID);
        let Some(epoch) = self.text_epoch else {
            return Ok(());
        };
        let Some(store) = &self.text_store else {
            return Ok(());
        };

        for (label, property) in &self.text_index_reads {
            store
                .score_text_visible(NodeId::INVALID, label, property, "", epoch, tx)
                .map_err(|error| super::OperatorError::Execution(error.to_string()))?;
        }
        self.text_reads_recorded = true;
        Ok(())
    }

    /// Decomposes this operator into its child and predicate for push-based
    /// conversion.
    ///
    /// Fires any pending text-index read recording before decomposition so
    /// that the recording still happens on the push-pipeline path (where
    /// `next()` is never called on this operator directly).
    ///
    /// # Errors
    /// Propagates pending temporal index qualification failures.
    pub fn into_parts(
        self,
    ) -> Result<(Box<dyn Operator>, Box<dyn Predicate>), super::OperatorError> {
        #[cfg(feature = "text-index")]
        let mut this = self;
        #[cfg(not(feature = "text-index"))]
        let this = self;
        // Fire text-index recording for the push-pipeline path.
        // On the pull path this fires in `next()` instead.
        #[cfg(feature = "text-index")]
        this.record_text_index_reads_once()?;
        Ok((this.child, this.predicate))
    }
}

impl Operator for FilterOperator {
    fn next(&mut self) -> OperatorResult {
        self.check_cancellation()?;
        // Recorded at execution time (first poll), not plan time, so it is
        // robust to logical-plan caching and fires per-transaction with live
        // context.
        #[cfg(feature = "text-index")]
        self.record_text_index_reads_once()?;

        loop {
            // Get next chunk from child
            self.check_cancellation()?;
            let chunk = self.child.next()?;
            self.check_cancellation()?;
            let Some(mut chunk) = chunk else {
                return Ok(None);
            };

            // Zone map check: skip entire chunk if no rows can match
            if let Some(hints) = chunk.zone_hints()
                && !self.predicate.might_match_chunk(hints)
            {
                continue; // Skip entire chunk - zone map proves no matches
            }

            // Apply predicate to create selection vector, respecting any
            // existing selection from child operators (stacked filters).
            let mut selection = SelectionVector::new_empty();
            for (position, row) in chunk.selected_indices().enumerate() {
                if position % 128 == 0 {
                    self.check_cancellation()?;
                }
                if self.predicate.evaluate(&chunk, row)? {
                    selection.push(row);
                }
            }

            self.check_cancellation()?;

            // If nothing passes, skip to next chunk
            if selection.is_empty() {
                continue;
            }

            chunk.set_selection(selection);
            return Ok(Some(chunk));
        }
    }

    fn reset(&mut self) {
        self.child.reset();
        // Reset the recording guard so a re-executed plan records again on the
        // next poll (e.g., correlated sub-plans that get reset per outer row).
        #[cfg(feature = "text-index")]
        {
            self.text_reads_recorded = false;
        }
    }

    fn name(&self) -> &'static str {
        "Filter"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn install_resource_context(
        &mut self,
        resources: &crate::execution::QueryResourceContext,
    ) -> Result<(), crate::execution::QueryResourceContextError> {
        self.child.install_resource_context(resources)?;
        self.cancellation = Some(resources.cancellation_token().clone());
        Ok(())
    }

    fn decompose_pipeline_with_resources(
        self: Box<Self>,
        _resources: &crate::execution::QueryResourceContext,
    ) -> Result<OperatorPipelineDecomposition, crate::execution::QueryResourceContextError> {
        // Decomposition cannot report a read failure. Keep a pending Text
        // admission on the existing pull boundary so next() can propagate it.
        #[cfg(feature = "text-index")]
        if !self.text_reads_recorded
            && self.text_epoch.is_some()
            && self.text_store.is_some()
            && !self.text_index_reads.is_empty()
        {
            return Ok(OperatorPipelineDecomposition::boundary(self));
        }
        let Self {
            child, predicate, ..
        } = *self;
        Ok(OperatorPipelineDecomposition::unary(
            child,
            Box::new(super::push::FilterPushOperator::new(Box::new(
                PredicateAdapter(predicate),
            ))),
        ))
    }
}

/// Escapes a character for use in a regex pattern.
#[cfg(any(feature = "regex", feature = "regex-lite"))]
fn regex_escape_char(ch: char, out: &mut String) {
    if ".+*?^${}()|[]\\".contains(ch) {
        out.push('\\');
    }
    out.push(ch);
}

/// Converts a `Value` to its string representation for concatenation.
fn value_to_string(val: &Value) -> Option<String> {
    match val {
        Value::Int64(i) => Some(i.to_string()),
        Value::Float64(f) => Some(f.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::String(s) => Some(s.to_string()),
        Value::Null => None,
        _ => Some(format!("{val}")),
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use grafeo_common::types::LogicalType;

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

    #[test]
    fn cancellation_after_rejected_chunk_stops_before_second_pull() {
        use crate::execution::{
            QueryCancellationToken, QueryExecutionControl, QueryResourceContext,
        };
        use grafeo_common::memory::buffer::BufferManager;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountedSource {
            pulls: Arc<AtomicUsize>,
            received: Arc<parking_lot::Mutex<Option<QueryCancellationToken>>>,
        }
        impl Operator for CountedSource {
            fn next(&mut self) -> OperatorResult {
                assert_eq!(
                    self.pulls.fetch_add(1, Ordering::SeqCst),
                    0,
                    "cancelled source must not be pulled twice"
                );
                let mut chunk = DataChunk::with_capacity(&[LogicalType::Int64], 1);
                chunk.column_mut(0).unwrap().push_int64(1);
                chunk.set_count(1);
                Ok(Some(chunk))
            }
            fn reset(&mut self) {}
            fn name(&self) -> &'static str {
                "CancellationSource"
            }
            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
            fn install_resource_context(
                &mut self,
                resources: &QueryResourceContext,
            ) -> Result<(), crate::execution::QueryResourceContextError> {
                *self.received.lock() = Some(resources.cancellation_token().clone());
                Ok(())
            }
        }
        struct CancelAndReject(crate::execution::QueryCancellationHandle);
        impl Predicate for CancelAndReject {
            fn evaluate(
                &self,
                _: &DataChunk,
                _: usize,
            ) -> Result<bool, super::super::OperatorError> {
                self.0.cancel();
                Ok(false)
            }
        }
        for aggregate in [false, true] {
            let control = QueryExecutionControl::new();
            let resources = QueryResourceContext::new_with_cancellation(
                BufferManager::with_budget(1 << 20),
                control.token(),
            )
            .unwrap();
            let pulls = Arc::new(AtomicUsize::new(0));
            let received = Arc::new(parking_lot::Mutex::new(None));
            let filter: Box<dyn Operator> = Box::new(FilterOperator::new(
                Box::new(CountedSource {
                    pulls: pulls.clone(),
                    received: received.clone(),
                }),
                Box::new(CancelAndReject(control.cancellation_handle())),
            ));
            let mut root = if aggregate {
                Box::new(super::super::SimpleAggregateOperator::new(
                    filter,
                    vec![super::super::AggregateExpr::count_star()],
                    vec![LogicalType::Int64],
                )) as Box<dyn Operator>
            } else {
                filter
            };
            root.install_resource_context(&resources).unwrap();
            let error = root.next().unwrap_err();
            assert!(matches!(
                error,
                super::super::OperatorError::QueryCancelled(
                    crate::execution::QueryCancellationError::Cancelled,
                )
            ));
            assert_eq!(pulls.load(Ordering::SeqCst), 1, "aggregate={aggregate}");
            assert!(matches!(
                received.lock().as_ref().unwrap().check(),
                Err(crate::execution::QueryCancellationError::Cancelled)
            ));
        }
    }

    #[test]
    fn test_filter_comparison() {
        // Create a chunk with values [10, 20, 30, 40, 50]
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for i in 1..=5 {
            builder.column_mut(0).unwrap().push_int64(i * 10);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Filter for values > 25
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(25));
        let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

        let result = filter.next().unwrap().unwrap();
        // Should have 30, 40, 50 (3 values)
        assert_eq!(result.row_count(), 3);
    }

    #[cfg(any(feature = "regex", feature = "regex-lite"))]
    #[test]
    fn test_regex_operator() -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::LpgStore;

        // Create a store and expression predicate to test regex
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let variable_columns = HashMap::new();

        // Create predicate to test "Smith" =~ ".*Smith$" (should match)
        let predicate = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String(
                    "John Smith".into(),
                ))),
                op: BinaryFilterOp::Regex,
                right: Box::new(FilterExpression::Literal(Value::String(".*Smith$".into()))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );

        // Create a minimal chunk for evaluation
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Should match
        assert!(predicate.evaluate(&chunk, 0)?);

        // Test non-matching pattern
        let predicate_no_match = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String("John Doe".into()))),
                op: BinaryFilterOp::Regex,
                right: Box::new(FilterExpression::Literal(Value::String(".*Smith$".into()))),
            },
            variable_columns,
            store,
        );

        // Should not match
        assert!(!predicate_no_match.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_pow_operator() -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let variable_columns = HashMap::new();

        // Create a minimal chunk for evaluation
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Create predicate to test 2^3 = 8.0
        let predicate = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(2))),
                    op: BinaryFilterOp::Pow,
                    right: Box::new(FilterExpression::Literal(Value::Int64(3))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Float64(8.0))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );

        // 2^3 should equal 8.0
        assert!(predicate.evaluate(&chunk, 0)?);

        // Test with floats: 2.5^2.0 = 6.25
        let predicate_float = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Float64(2.5))),
                    op: BinaryFilterOp::Pow,
                    right: Box::new(FilterExpression::Literal(Value::Float64(2.0))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Float64(6.25))),
            },
            variable_columns,
            store,
        );

        assert!(predicate_float.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_map_expression() -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let variable_columns = HashMap::new();

        // Create a minimal chunk for evaluation
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Create map {name: 'Alix', age: 30}
        let predicate = ExpressionPredicate::new(
            FilterExpression::Map(vec![
                (
                    "name".to_string(),
                    FilterExpression::Literal(Value::String("Alix".into())),
                ),
                (
                    "age".to_string(),
                    FilterExpression::Literal(Value::Int64(30)),
                ),
            ]),
            variable_columns,
            store,
        );

        // Evaluate the map expression
        let result = predicate.eval_at(&chunk, 0)?;
        assert!(result.is_some());

        if let Some(Value::Map(m)) = result {
            assert_eq!(
                m.get(&PropertyKey::new("name")),
                Some(&Value::String("Alix".into()))
            );
            assert_eq!(m.get(&PropertyKey::new("age")), Some(&Value::Int64(30)));
        } else {
            panic!("Expected Map value");
        }
        Ok(())
    }

    #[test]
    fn test_index_access_list() -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let variable_columns = HashMap::new();

        // Create a minimal chunk for evaluation
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test [1, 2, 3][1] = 2
        let predicate = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::IndexAccess {
                    base: Box::new(FilterExpression::List(vec![
                        FilterExpression::Literal(Value::Int64(1)),
                        FilterExpression::Literal(Value::Int64(2)),
                        FilterExpression::Literal(Value::Int64(3)),
                    ])),
                    index: Box::new(FilterExpression::Literal(Value::Int64(1))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(2))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );

        assert!(predicate.evaluate(&chunk, 0)?);

        // Test negative indexing: [1, 2, 3][-1] = 3
        let predicate_neg = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::IndexAccess {
                    base: Box::new(FilterExpression::List(vec![
                        FilterExpression::Literal(Value::Int64(1)),
                        FilterExpression::Literal(Value::Int64(2)),
                        FilterExpression::Literal(Value::Int64(3)),
                    ])),
                    index: Box::new(FilterExpression::Literal(Value::Int64(-1))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(3))),
            },
            variable_columns,
            store,
        );

        assert!(predicate_neg.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_slice_access() -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let variable_columns = HashMap::new();

        // Create a minimal chunk for evaluation
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test [1, 2, 3, 4, 5][1..3] should return [2, 3]
        let predicate = ExpressionPredicate::new(
            FilterExpression::SliceAccess {
                base: Box::new(FilterExpression::List(vec![
                    FilterExpression::Literal(Value::Int64(1)),
                    FilterExpression::Literal(Value::Int64(2)),
                    FilterExpression::Literal(Value::Int64(3)),
                    FilterExpression::Literal(Value::Int64(4)),
                    FilterExpression::Literal(Value::Int64(5)),
                ])),
                start: Some(Box::new(FilterExpression::Literal(Value::Int64(1)))),
                end: Some(Box::new(FilterExpression::Literal(Value::Int64(3)))),
            },
            variable_columns,
            store,
        );

        let result = predicate.eval_at(&chunk, 0)?;
        assert!(result.is_some());

        if let Some(Value::List(items)) = result {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0], Value::Int64(2));
            assert_eq!(items[1], Value::Int64(3));
        } else {
            panic!("Expected List value");
        }
        Ok(())
    }

    #[test]
    fn test_might_match_chunk_no_hints() {
        let predicate = ComparisonPredicate::new(0, CompareOp::Eq, Value::Int64(50));
        let hints = ChunkZoneHints::default();

        // With no zone map for the column, should return true (conservative)
        assert!(predicate.might_match_chunk(&hints));
    }

    #[test]
    fn test_might_match_chunk_equality_match() {
        let predicate = ComparisonPredicate::new(0, CompareOp::Eq, Value::Int64(50));

        let mut hints = ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
        );

        // 50 is within [10, 100], should return true
        assert!(predicate.might_match_chunk(&hints));
    }

    #[test]
    fn test_might_match_chunk_equality_no_match() {
        let predicate = ComparisonPredicate::new(0, CompareOp::Eq, Value::Int64(200));

        let mut hints = ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
        );

        // 200 is outside [10, 100], should return false
        assert!(!predicate.might_match_chunk(&hints));
    }

    #[test]
    fn test_might_match_chunk_greater_than_match() {
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(50));

        let mut hints = ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
        );

        // max=100 > 50, so some values might be > 50
        assert!(predicate.might_match_chunk(&hints));
    }

    #[test]
    fn test_might_match_chunk_greater_than_no_match() {
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(200));

        let mut hints = ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
        );

        // max=100 < 200, so no values can be > 200
        assert!(!predicate.might_match_chunk(&hints));
    }

    #[test]
    fn test_might_match_chunk_less_than_match() {
        let predicate = ComparisonPredicate::new(0, CompareOp::Lt, Value::Int64(50));

        let mut hints = ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
        );

        // min=10 < 50, so some values might be < 50
        assert!(predicate.might_match_chunk(&hints));
    }

    #[test]
    fn test_might_match_chunk_less_than_no_match() {
        let predicate = ComparisonPredicate::new(0, CompareOp::Lt, Value::Int64(5));

        let mut hints = ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
        );

        // min=10 > 5, so no values can be < 5
        assert!(!predicate.might_match_chunk(&hints));
    }

    #[test]
    fn test_might_match_chunk_not_equal_always_conservative() {
        let predicate = ComparisonPredicate::new(0, CompareOp::Ne, Value::Int64(50));

        let mut hints = ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(Value::Int64(50), Value::Int64(50), 0, 10),
        );

        // Even if min=max=50, Ne is conservative and returns true
        assert!(predicate.might_match_chunk(&hints));
    }

    #[test]
    fn test_comparison_string() -> Result<(), Box<dyn std::error::Error>> {
        let mut builder = DataChunkBuilder::new(&[LogicalType::String]);
        builder.column_mut(0).unwrap().push_string("banana");
        builder.advance_row();
        let chunk = builder.finish();

        // Test string equality
        let pred_eq = ComparisonPredicate::new(0, CompareOp::Eq, Value::String("banana".into()));
        assert!(pred_eq.evaluate(&chunk, 0)?);

        let pred_ne = ComparisonPredicate::new(0, CompareOp::Ne, Value::String("apple".into()));
        assert!(pred_ne.evaluate(&chunk, 0)?);

        // Test string ordering
        let pred_lt = ComparisonPredicate::new(0, CompareOp::Lt, Value::String("cherry".into()));
        assert!(pred_lt.evaluate(&chunk, 0)?); // "banana" < "cherry"

        let pred_gt = ComparisonPredicate::new(0, CompareOp::Gt, Value::String("apple".into()));
        assert!(pred_gt.evaluate(&chunk, 0)?); // "banana" > "apple"
        Ok(())
    }

    #[test]
    fn test_comparison_float64() -> Result<(), Box<dyn std::error::Error>> {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Float64]);
        builder
            .column_mut(0)
            .unwrap()
            .push_float64(std::f64::consts::PI);
        builder.advance_row();
        let chunk = builder.finish();

        // Test float equality (within epsilon)
        let pred_eq =
            ComparisonPredicate::new(0, CompareOp::Eq, Value::Float64(std::f64::consts::PI));
        assert!(pred_eq.evaluate(&chunk, 0)?);

        let pred_ne = ComparisonPredicate::new(0, CompareOp::Ne, Value::Float64(2.71));
        assert!(pred_ne.evaluate(&chunk, 0)?);

        let pred_lt = ComparisonPredicate::new(0, CompareOp::Lt, Value::Float64(4.0));
        assert!(pred_lt.evaluate(&chunk, 0)?);

        let pred_ge =
            ComparisonPredicate::new(0, CompareOp::Ge, Value::Float64(std::f64::consts::PI));
        assert!(pred_ge.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_comparison_bool() -> Result<(), Box<dyn std::error::Error>> {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Bool]);
        builder.column_mut(0).unwrap().push_bool(true);
        builder.advance_row();
        let chunk = builder.finish();

        let pred_eq = ComparisonPredicate::new(0, CompareOp::Eq, Value::Bool(true));
        assert!(pred_eq.evaluate(&chunk, 0)?);

        let pred_ne = ComparisonPredicate::new(0, CompareOp::Ne, Value::Bool(false));
        assert!(pred_ne.evaluate(&chunk, 0)?);

        // Ordering on booleans returns false
        let pred_lt = ComparisonPredicate::new(0, CompareOp::Lt, Value::Bool(false));
        assert!(!pred_lt.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_unary_operators() -> Result<(), Box<dyn std::error::Error>> {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test NOT
        let pred_not = ExpressionPredicate::new(
            FilterExpression::Unary {
                op: UnaryFilterOp::Not,
                operand: Box::new(FilterExpression::Literal(Value::Bool(false))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_not.evaluate(&chunk, 0)?);

        // Test IS NULL
        let pred_is_null = ExpressionPredicate::new(
            FilterExpression::Unary {
                op: UnaryFilterOp::IsNull,
                operand: Box::new(FilterExpression::Literal(Value::Null)),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_is_null.evaluate(&chunk, 0)?);

        // Test IS NOT NULL
        let pred_is_not_null = ExpressionPredicate::new(
            FilterExpression::Unary {
                op: UnaryFilterOp::IsNotNull,
                operand: Box::new(FilterExpression::Literal(Value::Int64(42))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_is_not_null.evaluate(&chunk, 0)?);

        // Test negation
        let pred_neg = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Unary {
                    op: UnaryFilterOp::Neg,
                    operand: Box::new(FilterExpression::Literal(Value::Int64(5))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(-5))),
            },
            variable_columns,
            store,
        );
        assert!(pred_neg.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_arithmetic_operators() -> Result<(), Box<dyn std::error::Error>> {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test Add: 2 + 3 = 5
        let pred_add = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(2))),
                    op: BinaryFilterOp::Add,
                    right: Box::new(FilterExpression::Literal(Value::Int64(3))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(5))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_add.evaluate(&chunk, 0)?);

        // Test Sub: 10 - 4 = 6
        let pred_sub = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(10))),
                    op: BinaryFilterOp::Sub,
                    right: Box::new(FilterExpression::Literal(Value::Int64(4))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(6))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_sub.evaluate(&chunk, 0)?);

        // Test Mul: 3 * 4 = 12
        let pred_mul = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(3))),
                    op: BinaryFilterOp::Mul,
                    right: Box::new(FilterExpression::Literal(Value::Int64(4))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(12))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_mul.evaluate(&chunk, 0)?);

        // Test Div: 20 / 4 = 5
        let pred_div = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(20))),
                    op: BinaryFilterOp::Div,
                    right: Box::new(FilterExpression::Literal(Value::Int64(4))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(5))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_div.evaluate(&chunk, 0)?);

        // Test Mod: 17 % 5 = 2
        let pred_mod = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(17))),
                    op: BinaryFilterOp::Mod,
                    right: Box::new(FilterExpression::Literal(Value::Int64(5))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(2))),
            },
            variable_columns,
            store,
        );
        assert!(pred_mod.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_string_operators() -> Result<(), Box<dyn std::error::Error>> {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test STARTS WITH
        let pred_starts = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String(
                    "hello world".into(),
                ))),
                op: BinaryFilterOp::StartsWith,
                right: Box::new(FilterExpression::Literal(Value::String("hello".into()))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_starts.evaluate(&chunk, 0)?);

        // Test ENDS WITH
        let pred_ends = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String(
                    "hello world".into(),
                ))),
                op: BinaryFilterOp::EndsWith,
                right: Box::new(FilterExpression::Literal(Value::String("world".into()))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_ends.evaluate(&chunk, 0)?);

        // Test CONTAINS
        let pred_contains = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String(
                    "hello world".into(),
                ))),
                op: BinaryFilterOp::Contains,
                right: Box::new(FilterExpression::Literal(Value::String("lo wo".into()))),
            },
            variable_columns,
            store,
        );
        assert!(pred_contains.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_in_operator() -> Result<(), Box<dyn std::error::Error>> {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test 3 IN [1, 2, 3, 4, 5]
        let pred_in = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Int64(3))),
                op: BinaryFilterOp::In,
                right: Box::new(FilterExpression::List(vec![
                    FilterExpression::Literal(Value::Int64(1)),
                    FilterExpression::Literal(Value::Int64(2)),
                    FilterExpression::Literal(Value::Int64(3)),
                    FilterExpression::Literal(Value::Int64(4)),
                    FilterExpression::Literal(Value::Int64(5)),
                ])),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_in.evaluate(&chunk, 0)?);

        // Test 10 NOT IN [1, 2, 3]
        let pred_not_in = ExpressionPredicate::new(
            FilterExpression::Unary {
                op: UnaryFilterOp::Not,
                operand: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(10))),
                    op: BinaryFilterOp::In,
                    right: Box::new(FilterExpression::List(vec![
                        FilterExpression::Literal(Value::Int64(1)),
                        FilterExpression::Literal(Value::Int64(2)),
                        FilterExpression::Literal(Value::Int64(3)),
                    ])),
                }),
            },
            variable_columns,
            store,
        );
        assert!(pred_not_in.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_logical_operators() -> Result<(), Box<dyn std::error::Error>> {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test AND: true AND true = true
        let pred_and = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Bool(true))),
                op: BinaryFilterOp::And,
                right: Box::new(FilterExpression::Literal(Value::Bool(true))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_and.evaluate(&chunk, 0)?);

        // Test OR: false OR true = true
        let pred_or = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Bool(false))),
                op: BinaryFilterOp::Or,
                right: Box::new(FilterExpression::Literal(Value::Bool(true))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_or.evaluate(&chunk, 0)?);

        // Test XOR: true XOR false = true
        let pred_xor = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Bool(true))),
                op: BinaryFilterOp::Xor,
                right: Box::new(FilterExpression::Literal(Value::Bool(false))),
            },
            variable_columns,
            store,
        );
        assert!(pred_xor.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_case_expression_simple() -> Result<(), Box<dyn std::error::Error>> {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test simple CASE: CASE 2 WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END = 'two'
        let pred_case = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Case {
                    operand: Some(Box::new(FilterExpression::Literal(Value::Int64(2)))),
                    when_clauses: vec![
                        (
                            FilterExpression::Literal(Value::Int64(1)),
                            FilterExpression::Literal(Value::String("one".into())),
                        ),
                        (
                            FilterExpression::Literal(Value::Int64(2)),
                            FilterExpression::Literal(Value::String("two".into())),
                        ),
                    ],
                    else_clause: Some(Box::new(FilterExpression::Literal(Value::String(
                        "other".into(),
                    )))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::String("two".into()))),
            },
            variable_columns,
            store,
        );
        assert!(pred_case.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_case_expression_searched() -> Result<(), Box<dyn std::error::Error>> {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test searched CASE: CASE WHEN 5 > 3 THEN 'yes' ELSE 'no' END = 'yes'
        let pred_case = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Case {
                    operand: None,
                    when_clauses: vec![(
                        FilterExpression::Binary {
                            left: Box::new(FilterExpression::Literal(Value::Int64(5))),
                            op: BinaryFilterOp::Gt,
                            right: Box::new(FilterExpression::Literal(Value::Int64(3))),
                        },
                        FilterExpression::Literal(Value::String("yes".into())),
                    )],
                    else_clause: Some(Box::new(FilterExpression::Literal(Value::String(
                        "no".into(),
                    )))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::String("yes".into()))),
            },
            variable_columns,
            store,
        );
        assert!(pred_case.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_list_functions() -> Result<(), Box<dyn std::error::Error>> {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test head([1, 2, 3]) = 1
        let pred_head = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::FunctionCall {
                    name: "head".to_string(),
                    args: vec![FilterExpression::List(vec![
                        FilterExpression::Literal(Value::Int64(1)),
                        FilterExpression::Literal(Value::Int64(2)),
                        FilterExpression::Literal(Value::Int64(3)),
                    ])],
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(1))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_head.evaluate(&chunk, 0)?);

        // Test last([1, 2, 3]) = 3
        let pred_last = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::FunctionCall {
                    name: "last".to_string(),
                    args: vec![FilterExpression::List(vec![
                        FilterExpression::Literal(Value::Int64(1)),
                        FilterExpression::Literal(Value::Int64(2)),
                        FilterExpression::Literal(Value::Int64(3)),
                    ])],
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(3))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_last.evaluate(&chunk, 0)?);

        // Test size([1, 2, 3]) = 3
        let pred_size = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::FunctionCall {
                    name: "size".to_string(),
                    args: vec![FilterExpression::List(vec![
                        FilterExpression::Literal(Value::Int64(1)),
                        FilterExpression::Literal(Value::Int64(2)),
                        FilterExpression::Literal(Value::Int64(3)),
                    ])],
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(3))),
            },
            variable_columns,
            store,
        );
        assert!(pred_size.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_type_conversion_functions() -> Result<(), Box<dyn std::error::Error>> {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test toInteger("42") = 42
        let pred_to_int = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::FunctionCall {
                    name: "toInteger".to_string(),
                    args: vec![FilterExpression::Literal(Value::String("42".into()))],
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(42))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_to_int.evaluate(&chunk, 0)?);

        // Test toFloat(42) = 42.0
        let pred_to_float = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::FunctionCall {
                    name: "toFloat".to_string(),
                    args: vec![FilterExpression::Literal(Value::Int64(42))],
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Float64(42.0))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_to_float.evaluate(&chunk, 0)?);

        // Test toBoolean("true") = true
        let pred_to_bool = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::FunctionCall {
                    name: "toBoolean".to_string(),
                    args: vec![FilterExpression::Literal(Value::String("true".into()))],
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Bool(true))),
            },
            variable_columns,
            store,
        );
        assert!(pred_to_bool.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_coalesce_function() -> Result<(), Box<dyn std::error::Error>> {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test coalesce(null, null, 'default') = 'default'
        let pred_coalesce = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::FunctionCall {
                    name: "coalesce".to_string(),
                    args: vec![
                        FilterExpression::Literal(Value::Null),
                        FilterExpression::Literal(Value::Null),
                        FilterExpression::Literal(Value::String("default".into())),
                    ],
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::String("default".into()))),
            },
            variable_columns,
            store,
        );
        assert!(pred_coalesce.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_filter_empty_result() {
        // Create a chunk with values that won't match the predicate
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for i in 1..=5 {
            builder.column_mut(0).unwrap().push_int64(i);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Filter for values > 100 (none will match)
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(100));
        let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

        // Should return None since nothing matches
        let result = filter.next().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_filter_operator_reset() {
        // Test that reset() calls child.reset()
        // Since MockScanOperator doesn't preserve chunks after reading,
        // we test that reset is called by checking position resets
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(50);
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let predicate = ComparisonPredicate::new(0, CompareOp::Eq, Value::Int64(50));
        let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

        // First iteration
        let result = filter.next().unwrap();
        assert!(result.is_some());
        let result = filter.next().unwrap();
        assert!(result.is_none());

        // Note: MockScanOperator replaces chunks with empty ones when read,
        // so reset doesn't restore the data. This test verifies reset() is called.
        filter.reset();
        // After reset, position is 0 but chunk is empty
        let result = filter.next().unwrap();
        // Empty chunk produces no matches, returns None
        assert!(result.is_none());
    }

    #[test]
    fn test_mixed_type_comparison_int_float() -> Result<(), Box<dyn std::error::Error>> {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test 5 == 5.0 (mixed int/float comparison)
        let pred_mixed = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Int64(5))),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Float64(5.0))),
            },
            variable_columns,
            store,
        );
        assert!(pred_mixed.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_zone_map_allows_matching_chunk() {
        // Test that a chunk with zone hints indicating potential matches is evaluated
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for i in 10..=20 {
            builder.column_mut(0).unwrap().push_int64(i);
            builder.advance_row();
        }
        let mut chunk = builder.finish();

        // Set zone hints: min=10, max=20
        let mut hints = crate::execution::chunk::ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(20), 0, 11),
        );
        chunk.set_zone_hints(hints);

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Filter for values > 15 (some will match)
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(15));
        let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

        // Should return matching rows
        let result = filter.next().unwrap();
        assert!(result.is_some());
        let chunk = result.unwrap();

        // Should have rows 16, 17, 18, 19, 20 (5 rows)
        assert_eq!(chunk.row_count(), 5);
    }

    #[test]
    fn test_filter_with_all_rows_matching() {
        // All values in chunk match the predicate
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for i in 100..=110 {
            builder.column_mut(0).unwrap().push_int64(i);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Filter for values > 50 (all will match)
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(50));
        let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

        let result = filter.next().unwrap();
        assert!(result.is_some());
        let chunk = result.unwrap();

        // All 11 rows should be returned
        assert_eq!(chunk.row_count(), 11);
    }

    #[test]
    fn test_filter_with_sparse_data() {
        // Test filtering with sparse matching data
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        // Create values where only some match: 1, 10, 2, 20, 3, 30
        for &v in &[1i64, 10, 2, 20, 3, 30] {
            builder.column_mut(0).unwrap().push_int64(v);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Filter for values > 5 (only 10, 20, 30 should match)
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(5));
        let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

        let result = filter.next().unwrap();
        assert!(result.is_some());
        let chunk = result.unwrap();

        // Only 10, 20, 30 should match (3 rows)
        assert_eq!(chunk.row_count(), 3);
    }

    #[test]
    fn test_predicate_on_wrong_column_returns_empty() {
        // When the predicate references a column index that's out of bounds
        // or the column type is incompatible
        let mut builder = DataChunkBuilder::new(&[LogicalType::String]);
        builder.column_mut(0).unwrap().push_string("hello");
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Predicate on column 5 (doesn't exist)
        let predicate = ComparisonPredicate::new(5, CompareOp::Eq, Value::Int64(42));
        let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

        // Should handle gracefully (either error or empty result)
        let result = filter.next();
        // The behavior depends on implementation - just verify no panic
        let _ = result;
    }

    #[test]
    fn test_expression_predicate_with_labels_function() -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::GraphStoreMut;

        // Test the labels() function in predicates
        let store: Arc<dyn GraphStoreMut> = Arc::new(crate::graph::lpg::LpgStore::new().unwrap());

        // Create a node with a label
        let node_id = store.create_node(&["Person", "Employee"]);

        // Build a chunk with the node
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        builder.column_mut(0).unwrap().push_node_id(node_id);
        builder.advance_row();
        let chunk = builder.finish();

        // Map column 0 to variable "n"
        let mut variable_columns = HashMap::new();
        variable_columns.insert("n".to_string(), 0);

        // Test: 'Person' IN labels(n)
        let pred = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String("Person".into()))),
                op: BinaryFilterOp::In,
                right: Box::new(FilterExpression::FunctionCall {
                    name: "labels".to_string(),
                    args: vec![FilterExpression::Variable("n".to_string())],
                }),
            },
            variable_columns,
            store.clone() as Arc<dyn GraphStoreSearch>,
        );

        assert!(pred.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_comparison_with_boundary_values() -> Result<(), Box<dyn std::error::Error>> {
        // Test comparisons at exact boundary values
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(i64::MAX);
        builder.advance_row();
        builder.column_mut(0).unwrap().push_int64(i64::MIN);
        builder.advance_row();
        builder.column_mut(0).unwrap().push_int64(0);
        builder.advance_row();
        let chunk = builder.finish();

        // Test >= 0
        let pred_ge = ComparisonPredicate::new(0, CompareOp::Ge, Value::Int64(0));
        assert!(pred_ge.evaluate(&chunk, 0)?); // i64::MAX >= 0
        assert!(!pred_ge.evaluate(&chunk, 1)?); // i64::MIN >= 0 is false
        assert!(pred_ge.evaluate(&chunk, 2)?); // 0 >= 0

        // Test <= 0
        let pred_le = ComparisonPredicate::new(0, CompareOp::Le, Value::Int64(0));
        assert!(!pred_le.evaluate(&chunk, 0)?); // i64::MAX <= 0 is false
        assert!(pred_le.evaluate(&chunk, 1)?); // i64::MIN <= 0
        assert!(pred_le.evaluate(&chunk, 2)?); // 0 <= 0
        Ok(())
    }

    // ── Cross-type equality (String ↔ numeric) ──────────────────────────

    /// Regression test: RDF stores numeric literals as strings, so filters
    /// like `FILTER(?age = 30)` compare `Value::String("30")` with
    /// `Value::Int64(30)`.  The `values_equal` path must coerce.
    #[test]
    fn test_cross_type_string_int_equality() -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let vc = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // String "42" == Int64(42)
        let pred = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String("42".into()))),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(42))),
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert!(pred.evaluate(&chunk, 0)?);

        // String "42" != Int64(99)
        let pred_ne = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String("42".into()))),
                op: BinaryFilterOp::Ne,
                right: Box::new(FilterExpression::Literal(Value::Int64(99))),
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert!(pred_ne.evaluate(&chunk, 0)?);

        // Non-numeric string should NOT equal any integer
        let pred_bad = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String("hello".into()))),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(42))),
            },
            vc,
            store,
        );
        assert!(!pred_bad.evaluate(&chunk, 0)?);
        Ok(())
    }

    /// String ↔ Float64 equality: "7.25" == Float64(7.25)
    #[test]
    fn test_cross_type_string_float_equality() -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let vc = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        let pred = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String("7.25".into()))),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Float64(7.25))),
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert!(pred.evaluate(&chunk, 0)?);

        // "7.25" != 2.5
        let pred_ne = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Float64(2.5))),
                op: BinaryFilterOp::Ne,
                right: Box::new(FilterExpression::Literal(Value::String("7.25".into()))),
            },
            vc,
            store,
        );
        assert!(pred_ne.evaluate(&chunk, 0)?);
        Ok(())
    }

    // ── Cross-type ordering (String ↔ numeric) ──────────────────────────

    /// Regression test: String-encoded numbers must support range comparisons
    /// so that `FILTER(?age > 25)` works when `?age` is stored as "30".
    #[test]
    fn test_cross_type_string_numeric_ordering() -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let vc = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // "30" > Int64(25)
        let pred_gt = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String("30".into()))),
                op: BinaryFilterOp::Gt,
                right: Box::new(FilterExpression::Literal(Value::Int64(25))),
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert!(pred_gt.evaluate(&chunk, 0)?);

        // Int64(10) < "20.5" (cross Float64 path)
        let pred_lt = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Int64(10))),
                op: BinaryFilterOp::Lt,
                right: Box::new(FilterExpression::Literal(Value::String("20.5".into()))),
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert!(pred_lt.evaluate(&chunk, 0)?);

        // "2.5" <= Float64(2.5)
        let pred_le = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String("2.5".into()))),
                op: BinaryFilterOp::Le,
                right: Box::new(FilterExpression::Literal(Value::Float64(2.5))),
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert!(pred_le.evaluate(&chunk, 0)?);

        // Float64(100.0) >= "99.9"
        let pred_ge = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Float64(100.0))),
                op: BinaryFilterOp::Ge,
                right: Box::new(FilterExpression::Literal(Value::String("99.9".into()))),
            },
            vc,
            store,
        );
        assert!(pred_ge.evaluate(&chunk, 0)?);
        Ok(())
    }

    // ── Stacked filter (selection vector preservation) ───────────────────

    /// Regression test: when two FilterOperators are stacked (child filter →
    /// parent filter), the parent must respect the child's selection vector
    /// instead of re-evaluating all physical rows.
    #[test]
    fn test_stacked_filters_respect_selection_vector() {
        // Chunk: ages = [20, 35, 45, 25, 50]
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for age in [20, 35, 45, 25, 50] {
            builder.column_mut(0).unwrap().push_int64(age);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // First filter: age > 25 → rows 1(35), 2(45), 4(50)
        let pred1 = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(25));
        let filter1 = FilterOperator::new(Box::new(scan), Box::new(pred1));

        // Second (stacked) filter: age < 50 → should intersect → rows 1(35), 2(45)
        let pred2 = ComparisonPredicate::new(0, CompareOp::Lt, Value::Int64(50));
        let mut filter2 = FilterOperator::new(Box::new(filter1), Box::new(pred2));

        let result = filter2.next().unwrap().unwrap();
        assert_eq!(
            result.row_count(),
            2,
            "stacked filter should yield 2 rows (35, 45)"
        );

        // Verify it's exhausted
        assert!(filter2.next().unwrap().is_none());
    }

    // === eval_binary_op: Arithmetic Tests ===

    /// Helper: creates an `ExpressionPredicate` wrapping a literal expression,
    /// evaluates it against an empty chunk, and returns the result `Value`.
    fn eval_literal_expr(
        expr: FilterExpression,
    ) -> grafeo_common::utils::error::Result<Option<Value>> {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let pred = ExpressionPredicate::new(expr, HashMap::new(), store);
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();
        pred.eval_at(&chunk, 0)
    }

    fn binary(left: Value, op: BinaryFilterOp, right: Value) -> FilterExpression {
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(left)),
            op,
            right: Box::new(FilterExpression::Literal(right)),
        }
    }

    fn unary(op: UnaryFilterOp, operand: FilterExpression) -> FilterExpression {
        FilterExpression::Unary {
            op,
            operand: Box::new(operand),
        }
    }

    #[test]
    fn test_eval_binary_addition_int() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Int64(10),
            BinaryFilterOp::Add,
            Value::Int64(20),
        ))?;
        assert_eq!(result, Some(Value::Int64(30)));
        Ok(())
    }

    #[test]
    fn test_eval_binary_subtraction_int() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Int64(50),
            BinaryFilterOp::Sub,
            Value::Int64(18),
        ))?;
        assert_eq!(result, Some(Value::Int64(32)));
        Ok(())
    }

    #[test]
    fn test_eval_binary_multiplication_int() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Int64(7),
            BinaryFilterOp::Mul,
            Value::Int64(6),
        ))?;
        assert_eq!(result, Some(Value::Int64(42)));
        Ok(())
    }

    #[test]
    fn test_eval_binary_division_int() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Int64(100),
            BinaryFilterOp::Div,
            Value::Int64(4),
        ))?;
        assert_eq!(result, Some(Value::Int64(25)));
        Ok(())
    }

    #[test]
    fn test_eval_binary_modulo_int() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Int64(17),
            BinaryFilterOp::Mod,
            Value::Int64(5),
        ))?;
        assert_eq!(result, Some(Value::Int64(2)));
        Ok(())
    }

    // === eval_binary_op: Comparisons ===

    #[test]
    fn test_eval_comparison_lt() -> Result<(), Box<dyn std::error::Error>> {
        let result =
            eval_literal_expr(binary(Value::Int64(3), BinaryFilterOp::Lt, Value::Int64(5)))?;
        assert_eq!(result, Some(Value::Bool(true)));

        let result =
            eval_literal_expr(binary(Value::Int64(5), BinaryFilterOp::Lt, Value::Int64(3)))?;
        assert_eq!(result, Some(Value::Bool(false)));
        Ok(())
    }

    #[test]
    fn test_eval_comparison_gt() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Int64(10),
            BinaryFilterOp::Gt,
            Value::Int64(5),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));
        Ok(())
    }

    #[test]
    fn test_eval_comparison_eq() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Int64(42),
            BinaryFilterOp::Eq,
            Value::Int64(42),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::Int64(42),
            BinaryFilterOp::Eq,
            Value::Int64(43),
        ))?;
        assert_eq!(result, Some(Value::Bool(false)));
        Ok(())
    }

    #[test]
    fn test_eval_comparison_ne() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::String("hello".into()),
            BinaryFilterOp::Ne,
            Value::String("world".into()),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::String("same".into()),
            BinaryFilterOp::Ne,
            Value::String("same".into()),
        ))?;
        assert_eq!(result, Some(Value::Bool(false)));
        Ok(())
    }

    #[test]
    fn test_eval_comparison_le_ge() -> Result<(), Box<dyn std::error::Error>> {
        // <=
        let result =
            eval_literal_expr(binary(Value::Int64(5), BinaryFilterOp::Le, Value::Int64(5)))?;
        assert_eq!(result, Some(Value::Bool(true)));

        let result =
            eval_literal_expr(binary(Value::Int64(6), BinaryFilterOp::Le, Value::Int64(5)))?;
        assert_eq!(result, Some(Value::Bool(false)));

        // >=
        let result =
            eval_literal_expr(binary(Value::Int64(5), BinaryFilterOp::Ge, Value::Int64(5)))?;
        assert_eq!(result, Some(Value::Bool(true)));

        let result =
            eval_literal_expr(binary(Value::Int64(4), BinaryFilterOp::Ge, Value::Int64(5)))?;
        assert_eq!(result, Some(Value::Bool(false)));
        Ok(())
    }

    // === eval_binary_op: Logical Operators ===

    #[test]
    fn test_eval_logical_and() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Bool(true),
            BinaryFilterOp::And,
            Value::Bool(true),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::Bool(true),
            BinaryFilterOp::And,
            Value::Bool(false),
        ))?;
        assert_eq!(result, Some(Value::Bool(false)));
        Ok(())
    }

    #[test]
    fn test_eval_logical_or() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Bool(false),
            BinaryFilterOp::Or,
            Value::Bool(true),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::Bool(false),
            BinaryFilterOp::Or,
            Value::Bool(false),
        ))?;
        assert_eq!(result, Some(Value::Bool(false)));
        Ok(())
    }

    #[test]
    fn test_eval_logical_xor() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Bool(true),
            BinaryFilterOp::Xor,
            Value::Bool(false),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::Bool(true),
            BinaryFilterOp::Xor,
            Value::Bool(true),
        ))?;
        assert_eq!(result, Some(Value::Bool(false)));
        Ok(())
    }

    // === Type Coercion: Int + Float Arithmetic ===

    #[test]
    fn test_eval_type_coercion_int_plus_float() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Int64(10),
            BinaryFilterOp::Add,
            Value::Float64(2.5),
        ))?;
        assert_eq!(result, Some(Value::Float64(12.5)));
        Ok(())
    }

    #[test]
    fn test_eval_type_coercion_float_minus_int() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Float64(10.0),
            BinaryFilterOp::Sub,
            Value::Int64(3),
        ))?;
        assert_eq!(result, Some(Value::Float64(7.0)));
        Ok(())
    }

    #[test]
    fn test_eval_type_coercion_int_mul_float() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Int64(4),
            BinaryFilterOp::Mul,
            Value::Float64(2.5),
        ))?;
        assert_eq!(result, Some(Value::Float64(10.0)));
        Ok(())
    }

    #[test]
    fn test_eval_type_coercion_int_eq_float() -> Result<(), Box<dyn std::error::Error>> {
        // Int 42 should equal Float 42.0
        let result = eval_literal_expr(binary(
            Value::Int64(42),
            BinaryFilterOp::Eq,
            Value::Float64(42.0),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));
        Ok(())
    }

    #[test]
    fn test_eval_type_coercion_int_lt_float() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Int64(3),
            BinaryFilterOp::Lt,
            Value::Float64(3.5),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));
        Ok(())
    }

    // === String Comparison ===

    #[test]
    fn test_eval_string_comparison() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::String("apple".into()),
            BinaryFilterOp::Lt,
            Value::String("banana".into()),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::String("zebra".into()),
            BinaryFilterOp::Gt,
            Value::String("apple".into()),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));
        Ok(())
    }

    #[test]
    fn test_eval_string_concatenation() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::String("Hello".into()),
            BinaryFilterOp::Add,
            Value::String(" World".into()),
        ))?;
        assert_eq!(result, Some(Value::String("Hello World".into())));
        Ok(())
    }

    // === IS NULL / IS NOT NULL ===

    #[test]
    fn test_eval_is_null() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(unary(
            UnaryFilterOp::IsNull,
            FilterExpression::Literal(Value::Null),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(unary(
            UnaryFilterOp::IsNull,
            FilterExpression::Literal(Value::Int64(42)),
        ))?;
        assert_eq!(result, Some(Value::Bool(false)));
        Ok(())
    }

    #[test]
    fn test_eval_is_not_null() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(unary(
            UnaryFilterOp::IsNotNull,
            FilterExpression::Literal(Value::Int64(42)),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(unary(
            UnaryFilterOp::IsNotNull,
            FilterExpression::Literal(Value::Null),
        ))?;
        assert_eq!(result, Some(Value::Bool(false)));
        Ok(())
    }

    #[test]
    fn test_eval_is_null_on_missing_variable() -> Result<(), Box<dyn std::error::Error>> {
        // Accessing a non-existent variable should produce None,
        // which IS NULL treats as true
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let expr = FilterExpression::Unary {
            op: UnaryFilterOp::IsNull,
            operand: Box::new(FilterExpression::Variable("missing_var".to_string())),
        };
        let pred = ExpressionPredicate::new(expr, HashMap::new(), store);
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();
        let result = pred.eval_at(&chunk, 0)?;
        assert_eq!(result, Some(Value::Bool(true)));
        Ok(())
    }

    // === STARTS WITH / ENDS WITH / CONTAINS ===

    #[test]
    fn test_eval_starts_with() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::StartsWith,
            Value::String("hello".into()),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::StartsWith,
            Value::String("world".into()),
        ))?;
        assert_eq!(result, Some(Value::Bool(false)));
        Ok(())
    }

    #[test]
    fn test_eval_ends_with() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::EndsWith,
            Value::String("world".into()),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::EndsWith,
            Value::String("hello".into()),
        ))?;
        assert_eq!(result, Some(Value::Bool(false)));
        Ok(())
    }

    #[test]
    fn test_eval_contains() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::Contains,
            Value::String("lo wo".into()),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::Contains,
            Value::String("xyz".into()),
        ))?;
        assert_eq!(result, Some(Value::Bool(false)));
        Ok(())
    }

    // === List Operations: IN Operator ===

    #[test]
    fn test_eval_in_operator() -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // 2 IN [1, 2, 3] should be true
        let expr = FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::Int64(2))),
            op: BinaryFilterOp::In,
            right: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::Int64(3)),
            ])),
        };
        let pred = ExpressionPredicate::new(expr, HashMap::new(), Arc::clone(&store));
        let result = pred.eval_at(&chunk, 0)?;
        assert_eq!(result, Some(Value::Bool(true)));

        // 5 IN [1, 2, 3] should be false
        let expr = FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::Int64(5))),
            op: BinaryFilterOp::In,
            right: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::Int64(3)),
            ])),
        };
        let pred = ExpressionPredicate::new(expr, HashMap::new(), Arc::clone(&store));
        let result = pred.eval_at(&chunk, 0)?;
        assert_eq!(result, Some(Value::Bool(false)));
        Ok(())
    }

    #[test]
    fn test_eval_in_operator_strings() -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // "banana" IN ["apple", "banana", "cherry"]
        let expr = FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::String("banana".into()))),
            op: BinaryFilterOp::In,
            right: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::String("apple".into())),
                FilterExpression::Literal(Value::String("banana".into())),
                FilterExpression::Literal(Value::String("cherry".into())),
            ])),
        };
        let pred = ExpressionPredicate::new(expr, HashMap::new(), store);
        let result = pred.eval_at(&chunk, 0)?;
        assert_eq!(result, Some(Value::Bool(true)));
        Ok(())
    }

    // === List Index Access ===

    #[test]
    fn test_eval_list_index_access() -> Result<(), Box<dyn std::error::Error>> {
        // [10, 20, 30][2] = 30
        let result = eval_literal_expr(FilterExpression::IndexAccess {
            base: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(10)),
                FilterExpression::Literal(Value::Int64(20)),
                FilterExpression::Literal(Value::Int64(30)),
            ])),
            index: Box::new(FilterExpression::Literal(Value::Int64(2))),
        })?;
        assert_eq!(result, Some(Value::Int64(30)));
        Ok(())
    }

    #[test]
    fn test_eval_list_negative_index() -> Result<(), Box<dyn std::error::Error>> {
        // [10, 20, 30][-2] = 20
        let result = eval_literal_expr(FilterExpression::IndexAccess {
            base: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(10)),
                FilterExpression::Literal(Value::Int64(20)),
                FilterExpression::Literal(Value::Int64(30)),
            ])),
            index: Box::new(FilterExpression::Literal(Value::Int64(-2))),
        })?;
        assert_eq!(result, Some(Value::Int64(20)));
        Ok(())
    }

    // === CASE / NULLIF Pattern ===

    #[test]
    fn test_eval_case_simple() -> Result<(), Box<dyn std::error::Error>> {
        // CASE WHEN true THEN 'yes' ELSE 'no' END
        let result = eval_literal_expr(FilterExpression::Case {
            operand: None,
            when_clauses: vec![(
                FilterExpression::Literal(Value::Bool(true)),
                FilterExpression::Literal(Value::String("yes".into())),
            )],
            else_clause: Some(Box::new(FilterExpression::Literal(Value::String(
                "no".into(),
            )))),
        })?;
        assert_eq!(result, Some(Value::String("yes".into())));
        Ok(())
    }

    #[test]
    fn test_eval_case_falls_to_else() -> Result<(), Box<dyn std::error::Error>> {
        // CASE WHEN false THEN 'yes' ELSE 'no' END
        let result = eval_literal_expr(FilterExpression::Case {
            operand: None,
            when_clauses: vec![(
                FilterExpression::Literal(Value::Bool(false)),
                FilterExpression::Literal(Value::String("yes".into())),
            )],
            else_clause: Some(Box::new(FilterExpression::Literal(Value::String(
                "no".into(),
            )))),
        })?;
        assert_eq!(result, Some(Value::String("no".into())));
        Ok(())
    }

    #[test]
    fn test_eval_case_no_else_returns_null() -> Result<(), Box<dyn std::error::Error>> {
        // CASE WHEN false THEN 'yes' END (no ELSE, so NULL)
        let result = eval_literal_expr(FilterExpression::Case {
            operand: None,
            when_clauses: vec![(
                FilterExpression::Literal(Value::Bool(false)),
                FilterExpression::Literal(Value::String("yes".into())),
            )],
            else_clause: None,
        })?;
        assert_eq!(result, Some(Value::Null));
        Ok(())
    }

    #[test]
    fn test_eval_nullif_via_case() -> Result<(), Box<dyn std::error::Error>> {
        // NULLIF(a, b) is equivalent to: CASE WHEN a = b THEN NULL ELSE a END
        // Test NULLIF(5, 5) => NULL
        let result = eval_literal_expr(FilterExpression::Case {
            operand: None,
            when_clauses: vec![(
                FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(5))),
                    op: BinaryFilterOp::Eq,
                    right: Box::new(FilterExpression::Literal(Value::Int64(5))),
                },
                FilterExpression::Literal(Value::Null),
            )],
            else_clause: Some(Box::new(FilterExpression::Literal(Value::Int64(5)))),
        })?;
        assert_eq!(result, Some(Value::Null));

        // NULLIF(5, 3) => 5
        let result = eval_literal_expr(FilterExpression::Case {
            operand: None,
            when_clauses: vec![(
                FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(5))),
                    op: BinaryFilterOp::Eq,
                    right: Box::new(FilterExpression::Literal(Value::Int64(3))),
                },
                FilterExpression::Literal(Value::Null),
            )],
            else_clause: Some(Box::new(FilterExpression::Literal(Value::Int64(5)))),
        })?;
        assert_eq!(result, Some(Value::Int64(5)));
        Ok(())
    }

    #[test]
    fn test_eval_simple_case_with_operand() -> Result<(), Box<dyn std::error::Error>> {
        // CASE 2 WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END
        let result = eval_literal_expr(FilterExpression::Case {
            operand: Some(Box::new(FilterExpression::Literal(Value::Int64(2)))),
            when_clauses: vec![
                (
                    FilterExpression::Literal(Value::Int64(1)),
                    FilterExpression::Literal(Value::String("one".into())),
                ),
                (
                    FilterExpression::Literal(Value::Int64(2)),
                    FilterExpression::Literal(Value::String("two".into())),
                ),
            ],
            else_clause: Some(Box::new(FilterExpression::Literal(Value::String(
                "other".into(),
            )))),
        })?;
        assert_eq!(result, Some(Value::String("two".into())));
        Ok(())
    }

    // === Unary Operators ===

    #[test]
    fn test_eval_unary_not() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(unary(
            UnaryFilterOp::Not,
            FilterExpression::Literal(Value::Bool(true)),
        ))?;
        assert_eq!(result, Some(Value::Bool(false)));

        let result = eval_literal_expr(unary(
            UnaryFilterOp::Not,
            FilterExpression::Literal(Value::Bool(false)),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));
        Ok(())
    }

    #[test]
    fn test_eval_unary_neg() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(unary(
            UnaryFilterOp::Neg,
            FilterExpression::Literal(Value::Int64(42)),
        ))?;
        assert_eq!(result, Some(Value::Int64(-42)));

        let result = eval_literal_expr(unary(
            UnaryFilterOp::Neg,
            FilterExpression::Literal(Value::Float64(7.25)),
        ))?;
        assert_eq!(result, Some(Value::Float64(-7.25)));
        Ok(())
    }

    // === Reduce Expression Evaluation ===

    #[test]
    fn test_eval_reduce_sum() -> Result<(), Box<dyn std::error::Error>> {
        // reduce(acc = 0, x IN [1, 2, 3] | acc + x) = 6
        let result = eval_literal_expr(FilterExpression::Reduce {
            accumulator: "acc".to_string(),
            initial: Box::new(FilterExpression::Literal(Value::Int64(0))),
            variable: "x".to_string(),
            list: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::Int64(3)),
            ])),
            expression: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable("acc".to_string())),
                op: BinaryFilterOp::Add,
                right: Box::new(FilterExpression::Variable("x".to_string())),
            }),
        })?;
        assert_eq!(result, Some(Value::Int64(6)));
        Ok(())
    }

    #[test]
    fn test_eval_reduce_product() -> Result<(), Box<dyn std::error::Error>> {
        // reduce(acc = 1, x IN [2, 3, 4] | acc * x) = 24
        let result = eval_literal_expr(FilterExpression::Reduce {
            accumulator: "acc".to_string(),
            initial: Box::new(FilterExpression::Literal(Value::Int64(1))),
            variable: "x".to_string(),
            list: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::Int64(3)),
                FilterExpression::Literal(Value::Int64(4)),
            ])),
            expression: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable("acc".to_string())),
                op: BinaryFilterOp::Mul,
                right: Box::new(FilterExpression::Variable("x".to_string())),
            }),
        })?;
        assert_eq!(result, Some(Value::Int64(24)));
        Ok(())
    }

    // === List Comprehension ===

    #[test]
    fn test_eval_list_comprehension_with_filter() -> Result<(), Box<dyn std::error::Error>> {
        // [x IN [1, 2, 3, 4, 5] WHERE x > 2 | x * 10]
        // Should produce [30, 40, 50]
        let result = eval_literal_expr(FilterExpression::ListComprehension {
            variable: "x".to_string(),
            list_expr: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::Int64(3)),
                FilterExpression::Literal(Value::Int64(4)),
                FilterExpression::Literal(Value::Int64(5)),
            ])),
            filter_expr: Some(Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable("x".to_string())),
                op: BinaryFilterOp::Gt,
                right: Box::new(FilterExpression::Literal(Value::Int64(2))),
            })),
            map_expr: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable("x".to_string())),
                op: BinaryFilterOp::Mul,
                right: Box::new(FilterExpression::Literal(Value::Int64(10))),
            }),
        })?;

        if let Some(Value::List(items)) = result {
            assert_eq!(items.len(), 3);
            assert_eq!(items[0], Value::Int64(30));
            assert_eq!(items[1], Value::Int64(40));
            assert_eq!(items[2], Value::Int64(50));
        } else {
            panic!("Expected List, got {:?}", result);
        }
        Ok(())
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn path_edges_unwind_preserves_id_type_and_endpoints() {
        use crate::execution::operators::unwind::UnwindOperator;
        use crate::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        let source = store.create_node(&[]);
        let target = store.create_node(&[]);
        let edge = store.create_edge(source, target, "REL");
        for name in ["edges", "relationships"] {
            let predicate = ExpressionPredicate::new(
                FilterExpression::FunctionCall {
                    name: name.into(),
                    args: vec![FilterExpression::Literal(Value::Path {
                        nodes: Arc::from([
                            Value::Int64(source.as_u64() as i64),
                            Value::Int64(target.as_u64() as i64),
                        ]),
                        edges: Arc::from([Value::Int64(edge.as_u64() as i64)]),
                    })],
                },
                HashMap::new(),
                store.clone(),
            );
            let values = predicate.eval_at(&DataChunk::empty(), 0).unwrap().unwrap();
            let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
            builder.column_mut(0).unwrap().push_value(values);
            builder.advance_row();
            let mut unwind = UnwindOperator::new(
                Box::new(MockScanOperator {
                    chunks: vec![builder.finish()],
                    position: 0,
                }),
                0,
                "e".into(),
                vec![LogicalType::Any],
                false,
                false,
            );
            let chunk = unwind.next().unwrap().unwrap();
            for (function, expected) in [
                ("id", Value::Int64(edge.as_u64() as i64)),
                ("type", Value::String("REL".into())),
                ("startNode", Value::Int64(source.as_u64() as i64)),
                ("endNode", Value::Int64(target.as_u64() as i64)),
            ] {
                let predicate = ExpressionPredicate::new(
                    FilterExpression::FunctionCall {
                        name: function.into(),
                        args: vec![FilterExpression::Variable("e".into())],
                    },
                    HashMap::from([("e".into(), 0)]),
                    store.clone(),
                );
                assert_eq!(
                    predicate.eval_at(&chunk, 0).unwrap(),
                    Some(expected),
                    "{name}: {function}"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn path_edges_parallel_identity_survives_user_id_property() {
        use crate::execution::operators::accumulator::AggregateExpr;
        use crate::execution::operators::aggregate::SimpleAggregateOperator;
        use crate::execution::operators::unwind::UnwindOperator;
        use crate::graph::lpg::LpgStore;
        let store = Arc::new(LpgStore::new().unwrap());
        let source = store.create_node(&[]);
        let target = store.create_node(&[]);
        let edges = [
            store.create_edge(source, target, "REL"),
            store.create_edge(source, target, "REL"),
        ];
        for edge in edges {
            store.set_edge_property(edge, "_id", Value::Int64(0));
        }
        let predicate = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "relationships".into(),
                args: vec![FilterExpression::Literal(Value::Path {
                    nodes: Arc::from([
                        Value::Int64(source.as_u64() as i64),
                        Value::Int64(target.as_u64() as i64),
                        Value::Int64(source.as_u64() as i64),
                    ]),
                    edges: edges.map(|edge| Value::Int64(edge.as_u64() as i64)).into(),
                })],
            },
            HashMap::new(),
            store,
        );
        let values = predicate.eval_at(&DataChunk::empty(), 0).unwrap().unwrap();
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        builder.column_mut(0).unwrap().push_value(values);
        builder.advance_row();
        let unwind = UnwindOperator::new(
            Box::new(MockScanOperator {
                chunks: vec![builder.finish()],
                position: 0,
            }),
            0,
            "e".into(),
            vec![LogicalType::Any],
            false,
            false,
        );
        let mut aggregate = SimpleAggregateOperator::new(
            Box::new(unwind),
            vec![AggregateExpr::count(0).with_distinct()],
            vec![LogicalType::Int64],
        );
        let result = aggregate.next().unwrap().unwrap();
        assert_eq!(
            result.column(0).unwrap().get_value(0),
            Some(Value::Int64(2))
        );
    }

    fn nested_path_edge_call(name: &str, args: Vec<FilterExpression>) -> FilterExpression {
        FilterExpression::FunctionCall {
            name: name.into(),
            args,
        }
    }

    fn nested_path_edge_map(source: FilterExpression, body: FilterExpression) -> FilterExpression {
        FilterExpression::ListComprehension {
            variable: "e".into(),
            list_expr: Box::new(source),
            filter_expr: None,
            map_expr: Box::new(body),
        }
    }

    fn nested_path_edge_property(name: &str) -> FilterExpression {
        FilterExpression::Property {
            variable: "e".into(),
            property: name.into(),
        }
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn nested_path_edge_functions_preserve_identity_and_null_cardinality() {
        use crate::graph::lpg::LpgStore;
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&[]);
        let b = store.create_node(&[]);
        let edge = store.create_edge(a, b, "REL");
        store.set_edge_property(edge, "cost", Value::Int64(7));
        store.set_edge_property(edge, "_id", Value::Int64(-10));
        store.set_node_property(a, "cost", Value::Int64(700));
        let source = nested_path_edge_call(
            "edges",
            vec![FilterExpression::Literal(Value::Path {
                nodes: Arc::from([
                    Value::Int64(a.as_u64() as i64),
                    Value::Int64(b.as_u64() as i64),
                ]),
                edges: Arc::from([Value::Int64(edge.as_u64() as i64)]),
            })],
        );
        let var = || FilterExpression::Variable("e".into());
        let body = FilterExpression::List(vec![
            var(),
            nested_path_edge_call("id", vec![var()]),
            nested_path_edge_call("type", vec![var()]),
            nested_path_edge_call("startNode", vec![var()]),
            nested_path_edge_call("endNode", vec![var()]),
            nested_path_edge_call(
                "coalesce",
                vec![
                    nested_path_edge_property("cost"),
                    FilterExpression::Literal(Value::Int64(0)),
                ],
            ),
            nested_path_edge_call(
                "coalesce",
                vec![
                    nested_path_edge_property("missing"),
                    FilterExpression::Literal(Value::Int64(0)),
                ],
            ),
            nested_path_edge_property("_id"),
            nested_path_edge_property("missing"),
        ]);
        let predicate = ExpressionPredicate::new(
            nested_path_edge_map(source.clone(), body),
            HashMap::new(),
            store.clone(),
        );
        assert_eq!(
            predicate.eval_at(&DataChunk::empty(), 0).unwrap(),
            Some(Value::List(Arc::from([Value::List(Arc::from([
                Value::Int64(edge.as_u64() as i64),
                Value::Int64(edge.as_u64() as i64),
                Value::String("REL".into()),
                Value::Int64(a.as_u64() as i64),
                Value::Int64(b.as_u64() as i64),
                Value::Int64(7),
                Value::Int64(0),
                Value::Int64(-10),
                Value::Null,
            ]))])))
        );
        let predicate = ExpressionPredicate::new(
            FilterExpression::ListPredicate {
                kind: ListPredicateKind::All,
                variable: "e".into(),
                list_expr: Box::new(source),
                predicate: Box::new(FilterExpression::Binary {
                    left: Box::new(nested_path_edge_call(
                        "coalesce",
                        vec![
                            nested_path_edge_property("cost"),
                            FilterExpression::Literal(Value::Int64(0)),
                        ],
                    )),
                    op: BinaryFilterOp::Gt,
                    right: Box::new(FilterExpression::Literal(Value::Int64(0))),
                }),
            },
            HashMap::new(),
            store,
        );
        assert_eq!(
            predicate.eval_at(&DataChunk::empty(), 0).unwrap(),
            Some(Value::Bool(true))
        );
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn nested_path_edge_lexical_outer_and_shadowed_bindings() {
        use crate::graph::lpg::LpgStore;
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&[]);
        let edge = store.create_edge(a, a, "REL");
        store.set_edge_property(edge, "cost", Value::Int64(7));
        let typed = LogicalType::List(Box::new(LogicalType::Edge));
        let mut builder = DataChunkBuilder::new(&[typed, LogicalType::Int64]);
        builder
            .column_mut(0)
            .unwrap()
            .push_value(Value::List(Arc::from([Value::Int64(edge.as_u64() as i64)])));
        builder.column_mut(1).unwrap().push_value(Value::Int64(5));
        builder.advance_row();
        let source = FilterExpression::Variable("rels".into());
        let inner = FilterExpression::ListComprehension {
            variable: "x".into(),
            list_expr: Box::new(FilterExpression::List(vec![FilterExpression::Literal(
                Value::Int64(2),
            )])),
            filter_expr: None,
            map_expr: Box::new(FilterExpression::Binary {
                left: Box::new(nested_path_edge_property("cost")),
                op: BinaryFilterOp::Add,
                right: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Variable("x".into())),
                    op: BinaryFilterOp::Add,
                    right: Box::new(FilterExpression::Variable("offset".into())),
                }),
            }),
        };
        // The nested scalar named e must hide, then restore, the outer edge.
        let shadow = nested_path_edge_map(
            FilterExpression::List(vec![FilterExpression::Literal(Value::Int64(3))]),
            FilterExpression::Variable("e".into()),
        );
        let body = FilterExpression::List(vec![inner, shadow, nested_path_edge_property("cost")]);
        let predicate = ExpressionPredicate::new(
            nested_path_edge_map(source, body),
            HashMap::from([("rels".into(), 0), ("offset".into(), 1)]),
            store,
        );
        assert_eq!(
            predicate.eval_at(&builder.finish(), 0).unwrap(),
            Some(Value::List(Arc::from([Value::List(Arc::from([
                Value::List(Arc::from([Value::Int64(14)])),
                Value::List(Arc::from([Value::Int64(3)])),
                Value::Int64(7),
            ]))])))
        );
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn nested_path_edge_evaluated_sources_keep_only_selected_provenance() {
        use crate::graph::lpg::LpgStore;
        let store = Arc::new(LpgStore::new().unwrap());
        let node = store.create_node(&[]);
        let edge = store.create_edge(node, node, "REL");
        store.set_edge_property(edge, "cost", Value::Int64(7));
        store.set_node_property(node, "cost", Value::Int64(700));
        let list = Value::List(Arc::from([Value::Int64(edge.as_u64() as i64)]));
        for (first, second, first_type, second_type, expected) in [
            (
                Value::Null,
                list.clone(),
                LogicalType::Any,
                LogicalType::List(Box::new(LogicalType::Edge)),
                7,
            ),
            (
                list.clone(),
                list.clone(),
                LogicalType::Any,
                LogicalType::List(Box::new(LogicalType::Edge)),
                -1,
            ),
            (
                list.clone(),
                list.clone(),
                LogicalType::List(Box::new(LogicalType::Edge)),
                LogicalType::Any,
                7,
            ),
            (
                Value::Null,
                list,
                LogicalType::Any,
                LogicalType::List(Box::new(LogicalType::Int64)),
                -1,
            ),
        ] {
            let mut builder = DataChunkBuilder::new(&[first_type, second_type]);
            builder.column_mut(0).unwrap().push_value(first);
            builder.column_mut(1).unwrap().push_value(second);
            builder.advance_row();
            let source = FilterExpression::SliceAccess {
                base: Box::new(nested_path_edge_call(
                    "coalesce",
                    vec![
                        FilterExpression::Variable("first".into()),
                        FilterExpression::Variable("second".into()),
                    ],
                )),
                start: Some(Box::new(FilterExpression::Literal(Value::Int64(0)))),
                end: Some(Box::new(FilterExpression::Literal(Value::Int64(1)))),
            };
            let body = nested_path_edge_call(
                "coalesce",
                vec![
                    nested_path_edge_property("cost"),
                    FilterExpression::Literal(Value::Int64(-1)),
                ],
            );
            let predicate = ExpressionPredicate::new(
                nested_path_edge_map(source, body),
                HashMap::from([("first".into(), 0), ("second".into(), 1)]),
                store.clone(),
            );
            assert_eq!(
                predicate.eval_at(&builder.finish(), 0).unwrap(),
                Some(Value::List(Arc::from([Value::Int64(expected)])))
            );
        }
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn nested_path_edge_functions_keep_projected_snapshot_transaction_and_session() {
        use crate::graph::lpg::LpgStore;
        use crate::graph::projection::{GraphProjection, ProjectionSpec};
        let store = Arc::new(LpgStore::new().unwrap());
        let old = EpochId::new(1);
        let new = EpochId::new(5);
        let node = store.create_node_versioned(&[], old, TransactionId::SYSTEM);
        let edge = store.create_edge_versioned(node, node, "REL", old, TransactionId::SYSTEM);
        store.set_edge_property_at_epoch(edge, "cost", Value::Int64(10), old);
        store.set_edge_property_at_epoch(edge, "cost", Value::Int64(50), new);
        store.set_epoch(new);
        let tx = TransactionId::new(42);
        store.set_edge_property_buffered(edge, "cost", Value::Int64(99), tx);
        for (spec, epoch, transaction, expected) in [
            (ProjectionSpec::default(), old, None, 10),
            (ProjectionSpec::default(), new, None, 50),
            (ProjectionSpec::default(), old, Some(tx), 99),
            (
                ProjectionSpec::default().with_edge_types(["HIDDEN"]),
                old,
                None,
                -1,
            ),
        ] {
            let projection: Arc<dyn GraphStoreSearch> =
                Arc::new(GraphProjection::new(store.clone(), spec));
            let source = nested_path_edge_call(
                "relationships",
                vec![FilterExpression::Literal(Value::Path {
                    nodes: Arc::from([
                        Value::Int64(node.as_u64() as i64),
                        Value::Int64(node.as_u64() as i64),
                    ]),
                    edges: Arc::from([Value::Int64(edge.as_u64() as i64)]),
                })],
            );
            let expression = nested_path_edge_map(
                source,
                FilterExpression::List(vec![
                    nested_path_edge_call(
                        "coalesce",
                        vec![
                            nested_path_edge_property("cost"),
                            FilterExpression::Literal(Value::Int64(-1)),
                        ],
                    ),
                    nested_path_edge_call("current_schema", vec![]),
                ]),
            );
            let predicate = ExpressionPredicate::new(expression, HashMap::new(), projection)
                .with_transaction_context(epoch, transaction)
                .with_session_context(SessionContext {
                    current_schema: Some("private".into()),
                    ..SessionContext::default()
                });
            assert_eq!(
                predicate.eval_at(&DataChunk::empty(), 0).unwrap(),
                Some(Value::List(Arc::from([Value::List(Arc::from([
                    Value::Int64(expected),
                    Value::String("private".into()),
                ]))])))
            );
        }
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn nested_path_edge_selected_case_type_and_plain_id_results() {
        use crate::graph::lpg::LpgStore;
        let store = Arc::new(LpgStore::new().unwrap());
        let node = store.create_node(&[]);
        let edge = store.create_edge(node, node, "REL");
        let list = Value::List(Arc::from([Value::Int64(edge.as_u64() as i64)]));
        let edge_list = LogicalType::List(Box::new(LogicalType::Edge));
        let mut builder = DataChunkBuilder::new(&[edge_list.clone(), LogicalType::Any]);
        builder.column_mut(0).unwrap().push_value(list.clone());
        builder.column_mut(1).unwrap().push_value(list.clone());
        builder.advance_row();
        let chunk = builder.finish();
        let columns = HashMap::from([("typed".into(), 0), ("plain".into(), 1)]);
        for select_typed in [true, false] {
            let expression = FilterExpression::Case {
                operand: None,
                when_clauses: vec![(
                    FilterExpression::Literal(Value::Bool(select_typed)),
                    FilterExpression::Variable("typed".into()),
                )],
                else_clause: Some(Box::new(FilterExpression::Variable("plain".into()))),
            };
            let predicate = ExpressionPredicate::new(expression, columns.clone(), store.clone());
            assert_eq!(
                predicate.eval_at_with_type(&chunk, 0).unwrap(),
                (Some(list.clone()), select_typed.then(|| edge_list.clone()))
            );
        }
        for (body, expected_type) in [
            (FilterExpression::Variable("e".into()), Some(edge_list)),
            (
                nested_path_edge_call("id", vec![FilterExpression::Variable("e".into())]),
                None,
            ),
        ] {
            let predicate = ExpressionPredicate::new(
                nested_path_edge_map(FilterExpression::Variable("typed".into()), body),
                columns.clone(),
                store.clone(),
            );
            assert_eq!(
                predicate.eval_at_with_type(&chunk, 0).unwrap(),
                (Some(list.clone()), expected_type)
            );
        }
        let predicate = ExpressionPredicate::new(
            nested_path_edge_map(
                FilterExpression::Variable("plain".into()),
                nested_path_edge_call(
                    "coalesce",
                    vec![
                        nested_path_edge_call("id", vec![FilterExpression::Variable("e".into())]),
                        FilterExpression::Literal(Value::Int64(-1)),
                    ],
                ),
            ),
            columns,
            store,
        );
        assert_eq!(
            predicate.eval_at_with_type(&chunk, 0).unwrap(),
            (Some(Value::List(Arc::from([Value::Int64(-1)]))), None)
        );
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn nested_path_edge_collection_wrappers_preserve_provenance_and_properties() {
        use crate::graph::lpg::LpgStore;
        let store = Arc::new(LpgStore::new().unwrap());
        let node = store.create_node(&[]);
        let target = store.create_node(&[]);
        let first = store.create_edge(node, target, "REL");
        let second = store.create_edge(node, target, "REL");
        store.set_edge_property(first, "cost", Value::Int64(7));
        store.set_edge_property(second, "cost", Value::Int64(8));
        let edge_list_type = LogicalType::List(Box::new(LogicalType::Edge));
        let list = Value::List(Arc::from([
            Value::Int64(first.as_u64() as i64),
            Value::Int64(second.as_u64() as i64),
        ]));
        let mut builder = DataChunkBuilder::new(std::slice::from_ref(&edge_list_type));
        builder.column_mut(0).unwrap().push_value(list.clone());
        builder.advance_row();
        let chunk = builder.finish();
        let columns = HashMap::from([("rels".into(), 0)]);

        for (name, expected_type, expected_value) in [
            (
                "head",
                Some(LogicalType::Edge),
                Value::Int64(first.as_u64() as i64),
            ),
            (
                "last",
                Some(LogicalType::Edge),
                Value::Int64(second.as_u64() as i64),
            ),
        ] {
            let predicate = ExpressionPredicate::new(
                nested_path_edge_call(name, vec![FilterExpression::Variable("rels".into())]),
                columns.clone(),
                store.clone(),
            );
            assert_eq!(
                predicate.eval_at_with_type(&chunk, 0).unwrap(),
                (Some(expected_value), expected_type)
            );
        }

        for (name, expected) in [("tail", vec![8]), ("reverse", vec![8, 7])] {
            let source =
                nested_path_edge_call(name, vec![FilterExpression::Variable("rels".into())]);
            let predicate = ExpressionPredicate::new(
                nested_path_edge_map(source, nested_path_edge_property("cost")),
                columns.clone(),
                store.clone(),
            );
            assert_eq!(
                predicate.eval_at(&chunk, 0).unwrap(),
                Some(Value::List(
                    expected
                        .into_iter()
                        .map(Value::Int64)
                        .collect::<Vec<_>>()
                        .into(),
                ))
            );
        }

        let empty = FilterExpression::List(Vec::new());
        for name in ["head", "last"] {
            let predicate = ExpressionPredicate::new(
                nested_path_edge_call(name, vec![empty.clone()]),
                HashMap::new(),
                store.clone(),
            );
            assert_eq!(
                predicate.eval_at_with_type(&DataChunk::empty(), 0).unwrap(),
                (None, None)
            );
        }
        for name in ["tail", "reverse"] {
            let predicate = ExpressionPredicate::new(
                nested_path_edge_call(name, vec![empty.clone()]),
                HashMap::new(),
                store.clone(),
            );
            assert_eq!(
                predicate.eval_at_with_type(&DataChunk::empty(), 0).unwrap(),
                (Some(Value::List(Arc::from([]))), None)
            );
        }
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn nested_path_edge_reduce_functions_share_accumulator_item_and_outer_scope() {
        use crate::graph::lpg::LpgStore;
        let store = Arc::new(LpgStore::new().unwrap());
        let node = store.create_node(&[]);
        let edge = store.create_edge(node, node, "REL");
        store.set_edge_property(edge, "cost", Value::Int64(7));
        let source = nested_path_edge_call(
            "edges",
            vec![FilterExpression::Literal(Value::Path {
                nodes: Arc::from([
                    Value::Int64(node.as_u64() as i64),
                    Value::Int64(node.as_u64() as i64),
                ]),
                edges: Arc::from([Value::Int64(edge.as_u64() as i64)]),
            })],
        );
        let add = |left, right| FilterExpression::Binary {
            left: Box::new(left),
            op: BinaryFilterOp::Add,
            right: Box::new(right),
        };
        let var = |name: &str| FilterExpression::Variable(name.into());
        let coalesce = |value| {
            nested_path_edge_call(
                "coalesce",
                vec![value, FilterExpression::Literal(Value::Int64(0))],
            )
        };
        let numbers = || {
            FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(2)),
            ])
        };
        let reduce = |variable: &str, list, expression| FilterExpression::Reduce {
            accumulator: "total".into(),
            initial: Box::new(FilterExpression::Literal(Value::Int64(0))),
            variable: variable.into(),
            list: Box::new(list),
            expression: Box::new(expression),
        };
        let correlated = reduce(
            "x",
            numbers(),
            add(
                add(coalesce(var("total")), nested_path_edge_property("cost")),
                var("x"),
            ),
        );
        let edge_item = reduce(
            "x",
            FilterExpression::List(vec![var("e")]),
            add(
                var("total"),
                coalesce(FilterExpression::Property {
                    variable: "x".into(),
                    property: "cost".into(),
                }),
            ),
        );
        let shadow = reduce("e", numbers(), add(var("total"), coalesce(var("e"))));
        // Existing reduce map-property semantics retain NULL rather than omit it.
        let missing = FilterExpression::Reduce {
            accumulator: "total".into(),
            initial: Box::new(FilterExpression::Map(vec![])),
            variable: "x".into(),
            list: Box::new(FilterExpression::List(vec![FilterExpression::Literal(
                Value::Int64(1),
            )])),
            expression: Box::new(FilterExpression::Property {
                variable: "total".into(),
                property: "absent".into(),
            }),
        };
        let expression = nested_path_edge_map(
            source,
            FilterExpression::List(vec![
                correlated,
                edge_item,
                shadow,
                missing,
                nested_path_edge_property("cost"),
            ]),
        );
        let predicate = ExpressionPredicate::new(expression, HashMap::new(), store.clone());
        assert_eq!(
            predicate.eval_at(&DataChunk::empty(), 0).unwrap(),
            Some(Value::List(Arc::from([Value::List(Arc::from([
                Value::Int64(17),
                Value::Int64(7),
                Value::Int64(3),
                Value::Null,
                Value::Int64(7),
            ]))])))
        );
        let same_name = reduce(
            "total",
            numbers(),
            add(
                coalesce(var("total")),
                FilterExpression::Literal(Value::Int64(1)),
            ),
        );
        let predicate = ExpressionPredicate::new(same_name, HashMap::new(), store.clone());
        assert_eq!(
            predicate.eval_at(&DataChunk::empty(), 0).unwrap(),
            Some(Value::Int64(2)),
            "accumulator wins when local names coincide"
        );
        // Duplicate names are accepted by the public parsers/binder. The old
        // mini-dispatcher inconsistently selected the item for x.v but the
        // accumulator for x['v']. Both now obey one lexical accumulator binding;
        // this intentionally corrects the observable old [2, 1] result.
        let map = |value| {
            FilterExpression::Map(vec![(
                "v".into(),
                FilterExpression::Literal(Value::Int64(value)),
            )])
        };
        let duplicate_map = FilterExpression::Reduce {
            accumulator: "x".into(),
            initial: Box::new(map(1)),
            variable: "x".into(),
            list: Box::new(FilterExpression::List(vec![map(2)])),
            expression: Box::new(FilterExpression::List(vec![
                FilterExpression::Property {
                    variable: "x".into(),
                    property: "v".into(),
                },
                FilterExpression::IndexAccess {
                    base: Box::new(var("x")),
                    index: Box::new(FilterExpression::Literal(Value::String("v".into()))),
                },
            ])),
        };
        let predicate = ExpressionPredicate::new(duplicate_map, HashMap::new(), store);
        assert_eq!(
            predicate.eval_at(&DataChunk::empty(), 0).unwrap(),
            Some(Value::List(Arc::from([Value::Int64(1), Value::Int64(1)])))
        );
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn typed_path_edge_alias_uses_snapshot_transaction_and_scope() {
        use crate::graph::lpg::LpgStore;
        let store = Arc::new(LpgStore::new().unwrap());
        let old = EpochId::new(1);
        let current = EpochId::new(5);
        let source = store.create_node_versioned(&[], old, TransactionId::SYSTEM);
        let target = store.create_node_versioned(&[], old, TransactionId::SYSTEM);
        let edge = store.create_edge_versioned(source, target, "REL", old, TransactionId::SYSTEM);
        store.set_edge_property_at_epoch(edge, "tag", Value::Int64(10), old);
        store.set_edge_property_at_epoch(edge, "tag", Value::Int64(50), current);
        store.set_epoch(current);
        let tx = TransactionId::new(42);
        store.set_edge_property_buffered(edge, "tag", Value::Int64(99), tx);
        let list = Value::List(Arc::from([Value::Int64(edge.as_u64() as i64)]));
        let typed = LogicalType::List(Box::new(LogicalType::Edge));
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any, typed.clone()]);
        builder.column_mut(0).unwrap().push_value(Value::Null);
        builder.column_mut(1).unwrap().push_value(list.clone());
        builder.advance_row();
        let chunk = builder.finish();
        assert_eq!(chunk.column(1).unwrap().data_type(), &typed);
        for alias in ["rels", "renamed"] {
            // The iterator deliberately shadows its outer source alias.
            let expression = FilterExpression::ListComprehension {
                variable: alias.into(),
                list_expr: Box::new(FilterExpression::Variable(alias.into())),
                filter_expr: None,
                map_expr: Box::new(FilterExpression::List(vec![
                    FilterExpression::Variable(alias.into()),
                    FilterExpression::Property {
                        variable: alias.into(),
                        property: "tag".into(),
                    },
                ])),
            };
            for (transaction, expected) in [(None, 10), (Some(tx), 99)] {
                let predicate = ExpressionPredicate::new(
                    expression.clone(),
                    HashMap::from([(alias.into(), 1)]),
                    store.clone(),
                )
                .with_transaction_context(old, transaction);
                assert_eq!(
                    predicate.eval_at(&chunk, 0).unwrap(),
                    Some(Value::List(Arc::from([Value::List(Arc::from([
                        Value::Int64(edge.as_u64() as i64),
                        Value::Int64(expected)
                    ]))])))
                );
                // Provenance belongs to the actual chunk column, not a cached
                // alias-name decision from an earlier evaluation.
                let mut untyped = DataChunkBuilder::new(&[LogicalType::Any, LogicalType::Any]);
                untyped.column_mut(0).unwrap().push_value(Value::Null);
                untyped.column_mut(1).unwrap().push_value(list.clone());
                untyped.advance_row();
                assert_eq!(
                    predicate.eval_at(&untyped.finish(), 0).unwrap(),
                    Some(Value::List(Arc::from([Value::List(Arc::from([
                        Value::Int64(edge.as_u64() as i64)
                    ]))])))
                );
            }
        }
        let predicate = ExpressionPredicate::new(
            FilterExpression::ListPredicate {
                kind: ListPredicateKind::All,
                variable: "e".into(),
                list_expr: Box::new(FilterExpression::Variable("rels".into())),
                predicate: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Property {
                        variable: "e".into(),
                        property: "tag".into(),
                    }),
                    op: BinaryFilterOp::Eq,
                    right: Box::new(FilterExpression::Literal(Value::Int64(10))),
                }),
            },
            HashMap::from([("rels".into(), 1)]),
            store,
        )
        .with_transaction_context(old, None);
        assert_eq!(
            predicate.eval_at(&chunk, 0).unwrap(),
            Some(Value::Bool(true))
        );
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn path_edge_alias_requires_exact_edge_list_type() {
        use crate::graph::lpg::LpgStore;
        let store = Arc::new(LpgStore::new().unwrap());
        let node = store.create_node(&[]);
        let edge = store.create_edge(node, node, "REL");
        assert_eq!(
            node.as_u64(),
            edge.as_u64(),
            "fixture requires colliding entity IDs"
        );
        store.set_node_property(node, "tag", Value::String("node".into()));
        store.set_edge_property(edge, "tag", Value::String("edge".into()));
        let predicate = ExpressionPredicate::new(
            FilterExpression::ListComprehension {
                variable: "e".into(),
                list_expr: Box::new(FilterExpression::Variable("values".into())),
                filter_expr: None,
                map_expr: Box::new(FilterExpression::Property {
                    variable: "e".into(),
                    property: "tag".into(),
                }),
            },
            HashMap::from([("values".into(), 0)]),
            store,
        );
        for data_type in [
            LogicalType::Any,
            LogicalType::List(Box::new(LogicalType::Int64)),
            LogicalType::List(Box::new(LogicalType::Any)),
            LogicalType::List(Box::new(LogicalType::Node)),
        ] {
            let mut builder = DataChunkBuilder::new(std::slice::from_ref(&data_type));
            builder
                .column_mut(0)
                .unwrap()
                .push_value(Value::List(Arc::from([Value::Int64(edge.as_u64() as i64)])));
            builder.advance_row();
            assert_eq!(
                predicate.eval_at(&builder.finish(), 0).unwrap(),
                Some(Value::List(Arc::from([]))),
                "{data_type:?} must not infer edge identity from integers"
            );
        }
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn path_edge_binding_preserves_identity_and_user_properties() {
        use crate::graph::lpg::LpgStore;
        let store = Arc::new(LpgStore::new().unwrap());
        let source = store.create_node(&[]);
        let target = store.create_node(&[]);
        let _ = store.create_edge(source, target, "OTHER");
        let edge = store.create_edge(source, target, "REL");
        store.set_edge_property(edge, "_id", Value::Int64(-7));
        store.set_edge_property(edge, "tag", Value::String("edge".into()));
        for node in [source, target] {
            store.set_node_property(node, "tag", Value::String("node".into()));
        }
        let edge_value = Value::Int64(edge.as_u64() as i64);
        let edge_list = Value::List(Arc::from([edge_value.clone()]));
        for path in [
            Value::Path {
                nodes: Arc::from([
                    Value::Int64(source.as_u64() as i64),
                    Value::Int64(target.as_u64() as i64),
                ]),
                edges: Arc::from([edge_value.clone()]),
            },
            Value::Map(Arc::new(BTreeMap::from([(
                PropertyKey::new("edges"),
                edge_list,
            )]))),
            Value::List(Arc::from([
                Value::Int64(source.as_u64() as i64),
                edge_value.clone(),
                Value::Int64(target.as_u64() as i64),
            ])),
        ] {
            let list_expr = FilterExpression::FunctionCall {
                name: "relationships".into(),
                args: vec![FilterExpression::Literal(path)],
            };
            let property = |name: &str| FilterExpression::Property {
                variable: "e".into(),
                property: name.into(),
            };
            let predicate = ExpressionPredicate::new(
                FilterExpression::ListComprehension {
                    variable: "e".into(),
                    list_expr: Box::new(list_expr.clone()),
                    filter_expr: Some(Box::new(FilterExpression::Binary {
                        left: Box::new(property("tag")),
                        op: BinaryFilterOp::Eq,
                        right: Box::new(FilterExpression::Literal(Value::String("edge".into()))),
                    })),
                    map_expr: Box::new(FilterExpression::List(vec![
                        FilterExpression::Variable("e".into()),
                        property("_id"),
                        property("missing"),
                    ])),
                },
                HashMap::new(),
                store.clone(),
            );
            assert_eq!(
                predicate.eval_at(&DataChunk::empty(), 0).unwrap(),
                Some(Value::List(Arc::from([Value::List(Arc::from([
                    edge_value.clone(),
                    Value::Int64(-7),
                    Value::Null
                ]))])))
            );
            let predicate = ExpressionPredicate::new(
                FilterExpression::ListPredicate {
                    kind: ListPredicateKind::All,
                    variable: "e".into(),
                    list_expr: Box::new(list_expr),
                    predicate: Box::new(FilterExpression::Binary {
                        left: Box::new(property("tag")),
                        op: BinaryFilterOp::Eq,
                        right: Box::new(FilterExpression::Literal(Value::String("edge".into()))),
                    }),
                },
                HashMap::new(),
                store.clone(),
            );
            assert_eq!(
                predicate.eval_at(&DataChunk::empty(), 0).unwrap(),
                Some(Value::Bool(true))
            );
        }
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn path_edge_properties_preserve_projected_snapshot_and_transaction() {
        use crate::graph::lpg::LpgStore;
        use crate::graph::projection::{GraphProjection, ProjectionSpec};
        let store = Arc::new(LpgStore::new().unwrap());
        let old = EpochId::new(1);
        let newer = EpochId::new(5);
        let source = store.create_node_versioned(&[], old, TransactionId::SYSTEM);
        let target = store.create_node_versioned(&[], old, TransactionId::SYSTEM);
        let edge = store.create_edge_versioned(source, target, "REL", old, TransactionId::SYSTEM);
        store.set_edge_property_at_epoch(edge, "tag", Value::Int64(10), old);
        store.set_edge_property_at_epoch(edge, "tag", Value::Int64(50), newer);
        // Finish the replay fixture by publishing its frontier. Otherwise both
        // supplied epochs are future requests relative to INITIAL, rather than
        // an old snapshot followed by a newer committed value.
        store.set_epoch(newer);
        assert_eq!(
            store
                .get_edge_at_epoch(edge, old)
                .unwrap()
                .properties
                .get(&PropertyKey::new("tag")),
            Some(&Value::Int64(10))
        );
        let projection: Arc<dyn GraphStoreSearch> = Arc::new(GraphProjection::new(
            store.clone(),
            ProjectionSpec::default(),
        ));
        assert_eq!(
            projection
                .get_edge_at_epoch(edge, old)
                .unwrap()
                .properties
                .get(&PropertyKey::new("tag")),
            Some(&Value::Int64(10))
        );
        let expression = FilterExpression::ListComprehension {
            variable: "e".into(),
            list_expr: Box::new(FilterExpression::FunctionCall {
                name: "edges".into(),
                args: vec![FilterExpression::Literal(Value::Path {
                    nodes: Arc::from([
                        Value::Int64(source.as_u64() as i64),
                        Value::Int64(target.as_u64() as i64),
                    ]),
                    edges: Arc::from([Value::Int64(edge.as_u64() as i64)]),
                })],
            }),
            filter_expr: None,
            map_expr: Box::new(FilterExpression::Property {
                variable: "e".into(),
                property: "tag".into(),
            }),
        };
        let tx = TransactionId::new(42);
        store.set_edge_property_buffered(edge, "tag", Value::Int64(99), tx);
        for (epoch, transaction, expected) in
            [(old, None, 10), (newer, None, 50), (old, Some(tx), 99)]
        {
            let predicate =
                ExpressionPredicate::new(expression.clone(), HashMap::new(), projection.clone())
                    .with_transaction_context(epoch, transaction);
            assert_eq!(
                predicate.eval_at(&DataChunk::empty(), 0).unwrap(),
                Some(Value::List(Arc::from([Value::Int64(expected)]))),
                "epoch={epoch:?}, transaction={transaction:?}"
            );
        }
        store.remove_edge_property_buffered(edge, "tag", tx);
        let predicate = ExpressionPredicate::new(expression.clone(), HashMap::new(), projection)
            .with_transaction_context(old, Some(tx));
        assert_eq!(
            predicate.eval_at(&DataChunk::empty(), 0).unwrap(),
            Some(Value::List(Arc::from([Value::Null])))
        );
        let hidden: Arc<dyn GraphStoreSearch> = Arc::new(GraphProjection::new(
            store,
            ProjectionSpec::default().with_edge_types(["HIDDEN"]),
        ));
        let predicate = ExpressionPredicate::new(expression, HashMap::new(), hidden)
            .with_transaction_context(old, None);
        assert_eq!(
            predicate.eval_at(&DataChunk::empty(), 0).unwrap(),
            Some(Value::List(Arc::from([Value::Null]))),
            "property binding must not bypass the projection's edge visibility"
        );
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn test_edges_path_list_comprehension_reads_edge_properties() {
        use crate::graph::lpg::LpgStore;
        use grafeo_common::types::LogicalType;

        let store = Arc::new(LpgStore::new().unwrap());
        let source = store.create_node(&[]);
        let target = store.create_node(&[]);
        let edge = store.create_edge(source, target, "REL");
        store.set_edge_property(edge, "tag", Value::String("s-t".into()));

        #[allow(clippy::cast_possible_wrap)]
        let path = Value::Path {
            nodes: Arc::from([
                Value::Int64(source.as_u64() as i64),
                Value::Int64(target.as_u64() as i64),
            ]),
            edges: Arc::from([Value::Int64(edge.as_u64() as i64)]),
        };
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        builder.column_mut(0).unwrap().push_value(path);
        builder.advance_row();
        let chunk = builder.finish();

        let edge_list = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "edges".to_string(),
                args: vec![FilterExpression::Variable("p".to_string())],
            },
            HashMap::from([("p".to_string(), 0)]),
            store.clone(),
        );
        let Some(Value::List(edges)) = edge_list.eval_at(&chunk, 0).unwrap() else {
            panic!("edges(path) should return a list");
        };
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0], Value::Int64(edge.as_u64() as i64));

        let predicate = ExpressionPredicate::new(
            FilterExpression::ListComprehension {
                variable: "e".to_string(),
                list_expr: Box::new(FilterExpression::FunctionCall {
                    name: "edges".to_string(),
                    args: vec![FilterExpression::Variable("p".to_string())],
                }),
                filter_expr: None,
                map_expr: Box::new(FilterExpression::Property {
                    variable: "e".to_string(),
                    property: "tag".to_string(),
                }),
            },
            HashMap::from([("p".to_string(), 0)]),
            store,
        );

        assert_eq!(
            predicate.eval_at(&chunk, 0).unwrap(),
            Some(Value::List(Arc::from([Value::String("s-t".into())])))
        );
    }

    // === List Predicate (any/all/none/single) ===

    #[test]
    fn test_eval_list_predicate_any() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(FilterExpression::ListPredicate {
            kind: ListPredicateKind::Any,
            variable: "x".to_string(),
            list_expr: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(5)),
                FilterExpression::Literal(Value::Int64(3)),
            ])),
            predicate: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable("x".to_string())),
                op: BinaryFilterOp::Gt,
                right: Box::new(FilterExpression::Literal(Value::Int64(4))),
            }),
        })?;
        assert_eq!(result, Some(Value::Bool(true)));
        Ok(())
    }

    #[test]
    fn test_eval_list_predicate_all() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(FilterExpression::ListPredicate {
            kind: ListPredicateKind::All,
            variable: "x".to_string(),
            list_expr: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(10)),
                FilterExpression::Literal(Value::Int64(20)),
                FilterExpression::Literal(Value::Int64(30)),
            ])),
            predicate: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable("x".to_string())),
                op: BinaryFilterOp::Gt,
                right: Box::new(FilterExpression::Literal(Value::Int64(5))),
            }),
        })?;
        assert_eq!(result, Some(Value::Bool(true)));
        Ok(())
    }

    #[test]
    fn test_eval_list_predicate_none() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(FilterExpression::ListPredicate {
            kind: ListPredicateKind::None,
            variable: "x".to_string(),
            list_expr: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::Int64(3)),
            ])),
            predicate: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable("x".to_string())),
                op: BinaryFilterOp::Gt,
                right: Box::new(FilterExpression::Literal(Value::Int64(10))),
            }),
        })?;
        assert_eq!(result, Some(Value::Bool(true)));
        Ok(())
    }

    #[test]
    fn test_eval_list_predicate_single() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(FilterExpression::ListPredicate {
            kind: ListPredicateKind::Single,
            variable: "x".to_string(),
            list_expr: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(5)),
                FilterExpression::Literal(Value::Int64(3)),
            ])),
            predicate: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable("x".to_string())),
                op: BinaryFilterOp::Gt,
                right: Box::new(FilterExpression::Literal(Value::Int64(4))),
            }),
        })?;
        // Only x=5 satisfies x > 4, so exactly one
        assert_eq!(result, Some(Value::Bool(true)));
        Ok(())
    }

    // === Map key access via index ===

    #[test]
    fn test_eval_map_key_access() -> Result<(), Box<dyn std::error::Error>> {
        // {name: 'Alix'}['name'] = 'Alix'
        let result = eval_literal_expr(FilterExpression::IndexAccess {
            base: Box::new(FilterExpression::Map(vec![(
                "name".to_string(),
                FilterExpression::Literal(Value::String("Alix".into())),
            )])),
            index: Box::new(FilterExpression::Literal(Value::String("name".into()))),
        })?;
        assert_eq!(result, Some(Value::String("Alix".into())));
        Ok(())
    }

    // === LIKE operator tests (require regex for pattern conversion) ===

    #[cfg(any(feature = "regex", feature = "regex-lite"))]
    #[test]
    fn test_eval_like_wildcard() -> Result<(), Box<dyn std::error::Error>> {
        // 'hello world' LIKE 'hello%'
        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::Like,
            Value::String("hello%".into()),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));

        // 'hello world' LIKE '%world'
        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::Like,
            Value::String("%world".into()),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));

        // 'hello world' LIKE '%llo%'
        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::Like,
            Value::String("%llo%".into()),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));

        // 'hello' LIKE 'world%'
        let result = eval_literal_expr(binary(
            Value::String("hello".into()),
            BinaryFilterOp::Like,
            Value::String("world%".into()),
        ))?;
        assert_eq!(result, Some(Value::Bool(false)));
        Ok(())
    }

    #[cfg(any(feature = "regex", feature = "regex-lite"))]
    #[test]
    fn test_eval_like_single_char() -> Result<(), Box<dyn std::error::Error>> {
        // 'cat' LIKE 'c_t'
        let result = eval_literal_expr(binary(
            Value::String("cat".into()),
            BinaryFilterOp::Like,
            Value::String("c_t".into()),
        ))?;
        assert_eq!(result, Some(Value::Bool(true)));

        // 'cart' LIKE 'c_t'
        let result = eval_literal_expr(binary(
            Value::String("cart".into()),
            BinaryFilterOp::Like,
            Value::String("c_t".into()),
        ))?;
        assert_eq!(result, Some(Value::Bool(false)));
        Ok(())
    }

    #[cfg(any(feature = "regex", feature = "regex-lite"))]
    #[test]
    fn test_eval_like_null() -> Result<(), Box<dyn std::error::Error>> {
        // NULL LIKE '%' -> NULL
        let result = eval_literal_expr(binary(
            Value::Null,
            BinaryFilterOp::Like,
            Value::String("%".into()),
        ))?;
        assert_eq!(result, Some(Value::Null));
        Ok(())
    }

    // === Concat operator (||) tests ===

    #[test]
    fn test_eval_concat_strings() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::String("hello".into()),
            BinaryFilterOp::Concat,
            Value::String(" world".into()),
        ))?;
        assert_eq!(result, Some(Value::String("hello world".into())));
        Ok(())
    }

    #[test]
    fn test_eval_concat_string_with_int() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::String("count: ".into()),
            BinaryFilterOp::Concat,
            Value::Int64(42),
        ))?;
        assert_eq!(result, Some(Value::String("count: 42".into())));
        Ok(())
    }

    #[test]
    fn test_eval_concat_int_with_string() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Int64(42),
            BinaryFilterOp::Concat,
            Value::String(" items".into()),
        ))?;
        assert_eq!(result, Some(Value::String("42 items".into())));
        Ok(())
    }

    #[test]
    fn test_eval_concat_null() -> Result<(), Box<dyn std::error::Error>> {
        // Null || Null -> Null (hits the null arm)
        let result = eval_literal_expr(binary(Value::Null, BinaryFilterOp::Concat, Value::Null))?;
        assert_eq!(result, Some(Value::Null));
        Ok(())
    }

    // === Modulo operator tests ===

    #[test]
    fn test_eval_modulo_float() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Float64(10.5),
            BinaryFilterOp::Mod,
            Value::Float64(3.0),
        ))?;
        if let Some(Value::Float64(v)) = result {
            assert!((v - 1.5).abs() < 0.001);
        } else {
            panic!("Expected Float64");
        }
        Ok(())
    }

    #[test]
    fn test_eval_modulo_mixed() -> Result<(), Box<dyn std::error::Error>> {
        // int % float
        let result = eval_literal_expr(binary(
            Value::Int64(10),
            BinaryFilterOp::Mod,
            Value::Float64(3.0),
        ))?;
        if let Some(Value::Float64(v)) = result {
            assert!((v - 1.0).abs() < 0.001);
        } else {
            panic!("Expected Float64");
        }

        // float % int
        let result = eval_literal_expr(binary(
            Value::Float64(10.0),
            BinaryFilterOp::Mod,
            Value::Int64(3),
        ))?;
        if let Some(Value::Float64(v)) = result {
            assert!((v - 1.0).abs() < 0.001);
        } else {
            panic!("Expected Float64");
        }
        Ok(())
    }

    #[test]
    fn test_eval_modulo_by_zero() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::Int64(10),
            BinaryFilterOp::Mod,
            Value::Int64(0),
        ))?;
        assert_eq!(result, None);

        let result = eval_literal_expr(binary(
            Value::Float64(10.0),
            BinaryFilterOp::Mod,
            Value::Float64(0.0),
        ))?;
        assert_eq!(result, None);
        Ok(())
    }

    // === String addition with type coercion ===

    #[test]
    fn test_eval_string_add_int() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::String("val:".into()),
            BinaryFilterOp::Add,
            Value::Int64(42),
        ))?;
        assert_eq!(result, Some(Value::String("val:42".into())));
        Ok(())
    }

    #[test]
    fn test_eval_string_add_bool() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::String("is:".into()),
            BinaryFilterOp::Add,
            Value::Bool(true),
        ))?;
        assert_eq!(result, Some(Value::String("is:true".into())));
        Ok(())
    }

    #[test]
    fn test_eval_string_add_null() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(binary(
            Value::String("val:".into()),
            BinaryFilterOp::Add,
            Value::Null,
        ))?;
        assert_eq!(result, Some(Value::Null));
        Ok(())
    }

    // === Slice access tests ===

    #[test]
    fn test_eval_string_slice() -> Result<(), Box<dyn std::error::Error>> {
        // "hello"[1..3] = "el"
        let result = eval_literal_expr(FilterExpression::SliceAccess {
            base: Box::new(FilterExpression::Literal(Value::String("hello".into()))),
            start: Some(Box::new(FilterExpression::Literal(Value::Int64(1)))),
            end: Some(Box::new(FilterExpression::Literal(Value::Int64(3)))),
        })?;
        assert_eq!(result, Some(Value::String("el".into())));
        Ok(())
    }

    #[test]
    fn test_eval_string_index_access() -> Result<(), Box<dyn std::error::Error>> {
        // "hello"[1] = "e"
        let result = eval_literal_expr(FilterExpression::IndexAccess {
            base: Box::new(FilterExpression::Literal(Value::String("hello".into()))),
            index: Box::new(FilterExpression::Literal(Value::Int64(1))),
        })?;
        assert_eq!(result, Some(Value::String("e".into())));
        Ok(())
    }

    #[test]
    fn test_eval_string_negative_index() -> Result<(), Box<dyn std::error::Error>> {
        // "hello"[-1] = "o"
        let result = eval_literal_expr(FilterExpression::IndexAccess {
            base: Box::new(FilterExpression::Literal(Value::String("hello".into()))),
            index: Box::new(FilterExpression::Literal(Value::Int64(-1))),
        })?;
        assert_eq!(result, Some(Value::String("o".into())));
        Ok(())
    }

    // === Function tests for uncovered branches ===

    #[test]
    fn test_eval_tostring_types() -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::LpgStore;
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let vc = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Bool -> String
        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toString".to_string(),
                args: vec![FilterExpression::Literal(Value::Bool(true))],
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert_eq!(pred.eval_at(&chunk, 0)?, Some(Value::String("true".into())));

        // Float -> String
        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toString".to_string(),
                args: vec![FilterExpression::Literal(Value::Float64(2.72))],
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert_eq!(pred.eval_at(&chunk, 0)?, Some(Value::String("2.72".into())));

        // Null -> Null
        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toString".to_string(),
                args: vec![FilterExpression::Literal(Value::Null)],
            },
            vc,
            store,
        );
        assert_eq!(pred.eval_at(&chunk, 0)?, Some(Value::Null));
        Ok(())
    }

    #[test]
    fn test_eval_toboolean() -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::LpgStore;
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let vc = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toBoolean".to_string(),
                args: vec![FilterExpression::Literal(Value::String("true".into()))],
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert_eq!(pred.eval_at(&chunk, 0)?, Some(Value::Bool(true)));

        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toBoolean".to_string(),
                args: vec![FilterExpression::Literal(Value::String("false".into()))],
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert_eq!(pred.eval_at(&chunk, 0)?, Some(Value::Bool(false)));

        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toBoolean".to_string(),
                args: vec![FilterExpression::Literal(Value::Bool(true))],
            },
            vc,
            store,
        );
        assert_eq!(pred.eval_at(&chunk, 0)?, Some(Value::Bool(true)));
        Ok(())
    }

    #[test]
    fn test_eval_tofloat() -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::LpgStore;
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let vc = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toFloat".to_string(),
                args: vec![FilterExpression::Literal(Value::String("2.72".into()))],
            },
            vc.clone(),
            Arc::clone(&store),
        );
        if let Some(Value::Float64(v)) = pred.eval_at(&chunk, 0)? {
            assert!((v - 2.72).abs() < 0.001);
        } else {
            panic!("Expected Float64");
        }

        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toFloat".to_string(),
                args: vec![FilterExpression::Literal(Value::Int64(42))],
            },
            vc,
            store,
        );
        assert_eq!(pred.eval_at(&chunk, 0)?, Some(Value::Float64(42.0)));
        Ok(())
    }

    #[test]
    fn test_eval_tointeger_from_float() -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::LpgStore;
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let vc = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toInteger".to_string(),
                args: vec![FilterExpression::Literal(Value::Float64(3.7))],
            },
            vc,
            store,
        );
        assert_eq!(pred.eval_at(&chunk, 0)?, Some(Value::Int64(3)));
        Ok(())
    }

    #[test]
    fn test_eval_reverse_list() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(FilterExpression::FunctionCall {
            name: "reverse".to_string(),
            args: vec![FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::Int64(3)),
            ])],
        })?;
        assert_eq!(
            result,
            Some(Value::List(
                vec![Value::Int64(3), Value::Int64(2), Value::Int64(1)].into()
            ))
        );
        Ok(())
    }

    #[test]
    fn test_eval_reverse_string() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(FilterExpression::FunctionCall {
            name: "reverse".to_string(),
            args: vec![FilterExpression::Literal(Value::String("abc".into()))],
        })?;
        assert_eq!(result, Some(Value::String("cba".into())));
        Ok(())
    }

    #[test]
    fn test_eval_exists_function() -> Result<(), Box<dyn std::error::Error>> {
        let result = eval_literal_expr(FilterExpression::FunctionCall {
            name: "exists".to_string(),
            args: vec![FilterExpression::Literal(Value::Int64(42))],
        })?;
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(FilterExpression::FunctionCall {
            name: "exists".to_string(),
            args: vec![FilterExpression::Literal(Value::Null)],
        })?;
        assert_eq!(result, Some(Value::Bool(false)));
        Ok(())
    }

    #[test]
    fn test_filter_into_any() {
        let mock = MockScanOperator {
            chunks: vec![],
            position: 0,
        };
        let predicate = ComparisonPredicate::new(0, CompareOp::Eq, Value::Int64(1));
        let op = FilterOperator::new(Box::new(mock), Box::new(predicate));
        let any = Box::new(op).into_any();
        assert!(any.downcast::<FilterOperator>().is_ok());
    }

    #[test]
    fn test_filter_into_parts() -> Result<(), Box<dyn std::error::Error>> {
        let mock = MockScanOperator {
            chunks: vec![],
            position: 0,
        };
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(5));
        let op = FilterOperator::new(Box::new(mock), Box::new(predicate));
        let (mut child, _predicate) = op.into_parts()?;
        assert!(child.next().unwrap().is_none());
        Ok(())
    }
}

#[cfg(all(test, feature = "text-index", feature = "lpg"))]
mod text_fn_tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use crate::graph::GraphStoreSearch;
    use crate::graph::lpg::LpgStore;
    use crate::index::text::{BM25Config, InvertedIndex};
    use grafeo_common::types::LogicalType;
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn setup_store_with_text_index() -> (
        Arc<LpgStore>,
        grafeo_common::types::NodeId,
        grafeo_common::types::NodeId,
    ) {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create nodes with text properties
        let n1 = store.create_node(&["Article"]);
        store.set_node_property(
            n1,
            "body",
            Value::String("rust graph database engine".into()),
        );
        let n2 = store.create_node(&["Article"]);
        store.set_node_property(n2, "body", Value::String("python web framework".into()));

        // Create and populate text index
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(n1, "rust graph database engine");
        index.insert(n2, "python web framework");
        store.add_text_index("Article", "body", Arc::new(RwLock::new(index)));

        (store, n1, n2)
    }

    #[test]
    fn test_text_score_function() -> Result<(), Box<dyn std::error::Error>> {
        let (store, n1, n2) = setup_store_with_text_index();

        // Build a chunk with two rows: n1 in row 0, n2 in row 1
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        builder.column_mut(0).unwrap().push_node_id(n1);
        builder.advance_row();
        builder.column_mut(0).unwrap().push_node_id(n2);
        builder.advance_row();
        let chunk = builder.finish();

        let mut variable_columns = HashMap::new();
        variable_columns.insert("doc".to_string(), 0);

        // text_score(doc.body, "rust database") > 0.0
        let predicate = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::FunctionCall {
                    name: "text_score".to_string(),
                    args: vec![
                        FilterExpression::Property {
                            variable: "doc".to_string(),
                            property: "body".to_string(),
                        },
                        FilterExpression::Literal(Value::String("rust database".into())),
                    ],
                }),
                op: BinaryFilterOp::Gt,
                right: Box::new(FilterExpression::Literal(Value::Float64(0.0))),
            },
            variable_columns,
            store as Arc<dyn GraphStoreSearch>,
        );

        // n1 matches "rust database" — should pass
        assert!(
            predicate.evaluate(&chunk, 0)?,
            "n1 should score > 0 for 'rust database'"
        );
        // n2 does not match — should fail
        assert!(
            !predicate.evaluate(&chunk, 1)?,
            "n2 should score 0 for 'rust database'"
        );
        Ok(())
    }

    #[test]
    fn test_text_match_function() -> Result<(), Box<dyn std::error::Error>> {
        let (store, n1, n2) = setup_store_with_text_index();

        // Build a chunk with two rows
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        builder.column_mut(0).unwrap().push_node_id(n1);
        builder.advance_row();
        builder.column_mut(0).unwrap().push_node_id(n2);
        builder.advance_row();
        let chunk = builder.finish();

        let mut variable_columns = HashMap::new();
        variable_columns.insert("doc".to_string(), 0);

        // text_match(doc.body, "rust") = true/false
        let predicate = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "text_match".to_string(),
                args: vec![
                    FilterExpression::Property {
                        variable: "doc".to_string(),
                        property: "body".to_string(),
                    },
                    FilterExpression::Literal(Value::String("rust".into())),
                ],
            },
            variable_columns,
            store as Arc<dyn GraphStoreSearch>,
        );

        // n1 contains "rust" — text_match should return Bool(true) → evaluates to true
        assert!(predicate.evaluate(&chunk, 0)?, "n1 should match 'rust'");
        // n2 does not contain "rust" — text_match should return Bool(false)
        assert!(
            !predicate.evaluate(&chunk, 1)?,
            "n2 should not match 'rust'"
        );
        Ok(())
    }

    #[test]
    fn test_text_score_wrong_arg_count_returns_none() -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(LpgStore::new().unwrap());
        let builder = DataChunkBuilder::new(&[LogicalType::Node]);
        let chunk = builder.finish();
        let variable_columns = HashMap::new();

        // text_score with wrong number of args should return None (evaluate = false)
        let predicate = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "text_score".to_string(),
                args: vec![FilterExpression::Literal(Value::String("only_one".into()))],
            },
            variable_columns,
            store as Arc<dyn GraphStoreSearch>,
        );
        assert!(!predicate.evaluate(&chunk, 0)?);
        Ok(())
    }

    #[test]
    fn test_text_score_no_index_returns_none() -> Result<(), Box<dyn std::error::Error>> {
        // Node has a label but no text index for that label+property
        let store = Arc::new(LpgStore::new().unwrap());
        let n1 = store.create_node(&["Article"]);
        store.set_node_property(n1, "body", Value::String("rust graph database".into()));
        // No text index added

        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        builder.column_mut(0).unwrap().push_node_id(n1);
        builder.advance_row();
        let chunk = builder.finish();

        let mut variable_columns = HashMap::new();
        variable_columns.insert("doc".to_string(), 0);

        let predicate = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "text_score".to_string(),
                args: vec![
                    FilterExpression::Property {
                        variable: "doc".to_string(),
                        property: "body".to_string(),
                    },
                    FilterExpression::Literal(Value::String("rust".into())),
                ],
            },
            variable_columns,
            store as Arc<dyn GraphStoreSearch>,
        );
        // No index → score_text returns None → eval returns None → evaluate returns false
        assert!(!predicate.evaluate(&chunk, 0)?);
        Ok(())
    }
}
