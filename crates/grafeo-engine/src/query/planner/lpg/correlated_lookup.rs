//! Correlated property lookup through retained index probes or query-owned buckets.
//!
//! Without an index, a snapshot-aware ScanOperator supplies candidates once. Buckets
//! prune same-type mismatches; other type cohorts remain candidates so the
//! original FilterOperator retains numeric/string coercion and NULL semantics.
//! This private blocking operator requires owner-installed resources on each
//! execution after reset. Admission excludes Join/Apply/mutation rescan shapes;
//! admitted input NodeScans keep the lookup on their once-drained left side.

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::memory::buffer::{MemoryGrant, MemoryGrantError};
use grafeo_common::types::{
    EpochId, HashableValue, LogicalType, NodeId, PropertyKey, TransactionId, Value,
};
use grafeo_common::utils::hash::FxHashMap;
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::chunk::DataChunkBuilder;
use grafeo_core::execution::memory::{QueryResourceContext, QueryResourceContextError};
use grafeo_core::execution::operators::{
    ExpressionPredicate, FilterOperator, Operator, OperatorError, OperatorResult,
};
use grafeo_core::graph::{GraphStoreSearch, PropertyIndexPredicate, PropertyIndexRequest};

use super::{BinaryOp, FilterOp, LogicalExpression, LogicalOperator, NodeScanOp, Planner, Result};

#[derive(Debug, Default)]
pub(super) struct LookupPlanEffects {
    admitted_shape: bool,
    all_properties: bool,
    properties: Vec<String>,
    labels: Vec<String>,
    creations: Vec<Vec<String>>,
}

impl LookupPlanEffects {
    // Different variable names do not prove distinct nodes: effects cover aliases.
    fn allows(&self, label: Option<&str>, property: &str) -> bool {
        self.admitted_shape
            && !self.all_properties
            && !self.properties.iter().any(|written| written == property)
            && label.is_none_or(|label| !self.labels.iter().any(|changed| changed == label))
            && !self.creations.iter().any(|labels| {
                label.is_none_or(|label| labels.iter().any(|created| created == label))
            })
    }
    fn record_properties(&mut self, properties: &[(String, LogicalExpression)]) {
        for (property, _) in properties {
            self.all_properties |= property == "*";
            self.properties.push(property.clone());
        }
    }
}

impl Planner {
    pub(super) fn lookup_plan_effects(op: &LogicalOperator) -> LookupPlanEffects {
        match op {
            LogicalOperator::Empty => LookupPlanEffects {
                admitted_shape: true,
                ..LookupPlanEffects::default()
            },
            LogicalOperator::NodeScan(op) => op.input.as_deref().map_or_else(
                || Self::lookup_plan_effects(&LogicalOperator::Empty),
                Self::lookup_plan_effects,
            ),
            LogicalOperator::Filter(op) => Self::lookup_plan_effects(&op.input),
            LogicalOperator::Unwind(op) => Self::lookup_plan_effects(&op.input),
            LogicalOperator::Return(op) => Self::lookup_plan_effects(&op.input),
            LogicalOperator::Project(op) => Self::lookup_plan_effects(&op.input),
            LogicalOperator::Sort(op) => Self::lookup_plan_effects(&op.input),
            LogicalOperator::Limit(op) => Self::lookup_plan_effects(&op.input),
            LogicalOperator::Skip(op) => Self::lookup_plan_effects(&op.input),
            LogicalOperator::Distinct(op) => Self::lookup_plan_effects(&op.input),
            LogicalOperator::Aggregate(op) => Self::lookup_plan_effects(&op.input),
            LogicalOperator::CreateEdge(op) => Self::lookup_plan_effects(&op.input),
            LogicalOperator::DeleteEdge(op) => Self::lookup_plan_effects(&op.input),
            LogicalOperator::MergeRelationship(op) => Self::lookup_plan_effects(&op.input),
            LogicalOperator::CreateNode(op) => {
                let mut effects = op.input.as_deref().map_or_else(
                    || Self::lookup_plan_effects(&LogicalOperator::Empty),
                    Self::lookup_plan_effects,
                );
                let placeholder = op.labels.is_empty()
                    && op.properties.is_empty()
                    && op
                        .input
                        .as_deref()
                        .is_some_and(|input| defines_variable(input, &op.variable));
                if !placeholder {
                    effects.creations.push(op.labels.clone());
                }
                effects
            }
            LogicalOperator::SetProperty(op) => {
                let mut effects = Self::lookup_plan_effects(&op.input);
                if !binds_edge_variable(&op.input, &op.variable) {
                    effects.all_properties |= op.replace;
                    effects.record_properties(&op.properties);
                }
                effects
            }
            LogicalOperator::AddLabel(op) => {
                let mut effects = Self::lookup_plan_effects(&op.input);
                effects.labels.extend(op.labels.iter().cloned());
                effects
            }
            LogicalOperator::RemoveLabel(op) => {
                let mut effects = Self::lookup_plan_effects(&op.input);
                effects.labels.extend(op.labels.iter().cloned());
                effects
            }
            LogicalOperator::Merge(op) => {
                let mut effects = Self::lookup_plan_effects(&op.input);
                effects.creations.push(op.labels.clone());
                // ON CREATE writes affect only new nodes, covered by membership.
                effects.record_properties(&op.on_match);
                effects
            }
            // GQL uses DeleteNode for DELETE regardless of entity kind;
            // plan_delete_node resolves the same proven edge binding later.
            LogicalOperator::DeleteNode(op) => {
                if binds_edge_variable(&op.input, &op.variable) {
                    Self::lookup_plan_effects(&op.input)
                } else {
                    LookupPlanEffects::default()
                }
            }
            LogicalOperator::InsertTriple(_)
            | LogicalOperator::DeleteTriple(_)
            | LogicalOperator::Modify(_)
            | LogicalOperator::ClearGraph(_)
            | LogicalOperator::CreateGraph(_)
            | LogicalOperator::DropGraph(_)
            | LogicalOperator::LoadGraph(_)
            | LogicalOperator::CopyGraph(_)
            | LogicalOperator::MoveGraph(_)
            | LogicalOperator::AddGraph(_)
            | LogicalOperator::CreatePropertyGraph(_)
            | LogicalOperator::CallProcedure(_) => LookupPlanEffects::default(),
            // Multi-input, lateral and rescan shapes. The lookup is a blocking
            // operator whose left side is drained exactly once, so it must not
            // sit under an operator that re-reads its child.
            LogicalOperator::Join(_)
            | LogicalOperator::LeftJoin(_)
            | LogicalOperator::AntiJoin(_)
            | LogicalOperator::Union(_)
            | LogicalOperator::Except(_)
            | LogicalOperator::Intersect(_)
            | LogicalOperator::Otherwise(_)
            | LogicalOperator::Apply(_)
            | LogicalOperator::ParameterScan(_)
            | LogicalOperator::MultiWayJoin(_) => LookupPlanEffects::default(),
            // Read-only shapes that have not been reviewed for admission. They
            // stay excluded exactly as before; the listing keeps that a
            // deliberate decision instead of a wildcard's side effect.
            LogicalOperator::EdgeScan(_)
            | LogicalOperator::Expand(_)
            | LogicalOperator::Bind(_)
            | LogicalOperator::MapCollect(_)
            | LogicalOperator::HorizontalAggregate(_)
            | LogicalOperator::PropertyPath(_)
            | LogicalOperator::TripleScan(_)
            | LogicalOperator::Construct(_)
            | LogicalOperator::VectorScan(_)
            | LogicalOperator::VectorJoin(_)
            | LogicalOperator::TextScan(_)
            | LogicalOperator::LoadData(_) => LookupPlanEffects::default(),
        }
    }

    #[cfg(test)]
    fn lookup_plan_is_stable(op: &LogicalOperator) -> bool {
        let effects = Self::lookup_plan_effects(op);
        effects.admitted_shape
            && !effects.all_properties
            && effects.properties.is_empty()
            && effects.labels.is_empty()
            && effects.creations.is_empty()
    }

    pub(super) fn try_plan_correlated_property_lookup(
        &self,
        filter: &FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        if !plain_predicate(&filter.predicate) {
            return Ok(None);
        }
        let LogicalOperator::NodeScan(scan) = filter.input.as_ref() else {
            return Ok(None);
        };
        let Some(input) = scan.input.as_deref() else {
            return Ok(None);
        };
        if defines_variable(input, &scan.variable) {
            return Ok(None);
        }
        let Some((property, key_expression)) =
            lookup_condition(&filter.predicate, &scan.variable, input)
        else {
            return Ok(None);
        };
        if !self
            .snapshot_lookup_effects
            .borrow()
            .allows(scan.label.as_deref(), property)
        {
            self.record_decline(super::Decline::CorrelatedAdmission);
            return Ok(None);
        }
        let mode = if self.store.has_property_index(property) {
            LookupMode::Index
        } else {
            LookupMode::Scan
        };

        let (child, mut columns) = self.plan_operator(input)?;
        let variables: HashMap<_, _> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        let key = ExpressionPredicate::new(
            self.convert_expression(key_expression)?,
            variables,
            Arc::clone(&self.store),
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id)
        .with_session_context(self.session_context.clone());
        // Keep the actual NodeScan profile entry, MVCC enumeration and SSI
        // predicate recording; do not substitute committed-latest registry IDs.
        let leaf = LogicalOperator::NodeScan(NodeScanOp {
            variable: scan.variable.clone(),
            label: scan.label.clone(),
            input: None,
        });
        let (candidates, _) = self.plan_operator(&leaf)?;
        let mut schema = self.derive_schema_from_columns(&columns);
        schema.push(LogicalType::Node);
        columns.push(scan.variable.clone());
        self.record_access_path(|| {
            if mode.is_indexed() {
                super::AccessPath::CorrelatedIndex
            } else {
                super::AccessPath::CorrelatedScan
            }
        });
        let lookup = CorrelatedPropertyLookup::new(
            child,
            candidates,
            key,
            CorrelatedPropertyLookupContext {
                property: property.into(),
                store: Arc::clone(&self.store),
                epoch: self.viewing_epoch,
                transaction_id: self.transaction_id,
                schema,
                scan_label: scan.label.clone(),
                mode,
            },
        );
        let variables = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        let residual = ExpressionPredicate::new(
            self.convert_expression(&filter.predicate)?,
            variables,
            Arc::clone(&self.store),
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id)
        .with_session_context(self.session_context.clone());
        Ok(Some((
            Box::new(FilterOperator::new(Box::new(lookup), Box::new(residual))),
            columns,
        )))
    }
}

// Only reference-producing scalar expressions are hoisted; no functions,
// subqueries or temporal Text reads move across the existing filter boundary.
fn plain_predicate(expr: &LogicalExpression) -> bool {
    match expr {
        LogicalExpression::Literal(_)
        | LogicalExpression::Variable(_)
        | LogicalExpression::Property { .. } => true,
        LogicalExpression::Binary { left, right, .. } => {
            plain_predicate(left) && plain_predicate(right)
        }
        _ => false,
    }
}

fn defines_variable(op: &LogicalOperator, variable: &str) -> bool {
    match op {
        LogicalOperator::Unwind(op) => {
            op.variable == variable || defines_variable(&op.input, variable)
        }
        LogicalOperator::NodeScan(op) => {
            op.variable == variable
                || op
                    .input
                    .as_deref()
                    .is_some_and(|input| defines_variable(input, variable))
        }
        LogicalOperator::Filter(op) => defines_variable(&op.input, variable),
        LogicalOperator::Sort(op) => defines_variable(&op.input, variable),
        LogicalOperator::Limit(op) => defines_variable(&op.input, variable),
        LogicalOperator::Skip(op) => defines_variable(&op.input, variable),
        LogicalOperator::Distinct(op) => defines_variable(&op.input, variable),
        LogicalOperator::CreateNode(op) => {
            op.variable == variable
                || op
                    .input
                    .as_deref()
                    .is_some_and(|input| defines_variable(input, variable))
        }
        LogicalOperator::CreateEdge(op) => {
            op.variable.as_deref() == Some(variable) || defines_variable(&op.input, variable)
        }
        LogicalOperator::Merge(op) => {
            op.variable == variable || defines_variable(&op.input, variable)
        }
        LogicalOperator::MergeRelationship(op) => {
            op.variable == variable || defines_variable(&op.input, variable)
        }
        LogicalOperator::SetProperty(op) => defines_variable(&op.input, variable),
        LogicalOperator::AddLabel(op) => defines_variable(&op.input, variable),
        LogicalOperator::RemoveLabel(op) => defines_variable(&op.input, variable),
        LogicalOperator::DeleteEdge(op) => defines_variable(&op.input, variable),
        LogicalOperator::DeleteNode(op) => defines_variable(&op.input, variable),
        LogicalOperator::Project(op) => {
            op.projections.iter().any(|projection| {
                super::output_column_name(projection.alias.as_deref(), &projection.expression)
                    == variable
            }) || (op.pass_through_input && defines_variable(&op.input, variable))
        }
        LogicalOperator::Return(op) => op.items.iter().any(|item| {
            super::output_column_name(item.alias.as_deref(), &item.expression) == variable
        }),
        // Exhaustive: new shapes need an explicit binding-scope decision.
        LogicalOperator::Empty
        | LogicalOperator::EdgeScan(_)
        | LogicalOperator::Expand(_)
        | LogicalOperator::Join(_)
        | LogicalOperator::Aggregate(_)
        | LogicalOperator::TripleScan(_)
        | LogicalOperator::Union(_)
        | LogicalOperator::LeftJoin(_)
        | LogicalOperator::AntiJoin(_)
        | LogicalOperator::Construct(_)
        | LogicalOperator::Bind(_)
        | LogicalOperator::MapCollect(_)
        | LogicalOperator::InsertTriple(_)
        | LogicalOperator::DeleteTriple(_)
        | LogicalOperator::Modify(_)
        | LogicalOperator::ClearGraph(_)
        | LogicalOperator::CreateGraph(_)
        | LogicalOperator::DropGraph(_)
        | LogicalOperator::LoadGraph(_)
        | LogicalOperator::CopyGraph(_)
        | LogicalOperator::MoveGraph(_)
        | LogicalOperator::AddGraph(_)
        | LogicalOperator::PropertyPath(_)
        | LogicalOperator::HorizontalAggregate(_)
        | LogicalOperator::VectorScan(_)
        | LogicalOperator::VectorJoin(_)
        | LogicalOperator::TextScan(_)
        | LogicalOperator::Except(_)
        | LogicalOperator::Intersect(_)
        | LogicalOperator::Otherwise(_)
        | LogicalOperator::Apply(_)
        | LogicalOperator::ParameterScan(_)
        | LogicalOperator::CreatePropertyGraph(_)
        | LogicalOperator::MultiWayJoin(_)
        | LogicalOperator::CallProcedure(_)
        | LogicalOperator::LoadData(_) => false,
    }
}

/// Whether `variable` is bound to an **edge** by `op` or one of its inputs.
///
/// `SetPropertyOp::is_edge` cannot answer this at admission time: only some
/// Cypher spellings populate it, GQL hardcodes `false`
/// (`translators/gql/mod.rs:487,497`), and the planner reconstructs the truth
/// from `Planner::edge_columns` much later (`planner/lpg/mutation.rs:1369`).
///
/// Only the binders that actually insert into `edge_columns` count as edge
/// bindings here, so this answer cannot disagree with the operator the planner
/// finally builds. Anything unrecognised answers `false`, so its writes are
/// conservatively checked as possible node-property effects.
fn binds_edge_variable(op: &LogicalOperator, variable: &str) -> bool {
    match op {
        // `plan_create_edge` registers a named created edge in `edge_columns`.
        LogicalOperator::CreateEdge(op) => {
            op.variable.as_deref() == Some(variable) || binds_edge_variable(&op.input, variable)
        }
        // `register_edge_column` does the same for an expand's edge variable.
        LogicalOperator::Expand(op) => {
            op.edge_variable.as_deref() == Some(variable)
                || binds_edge_variable(&op.input, variable)
        }
        LogicalOperator::MergeRelationship(op) => {
            op.variable == variable || binds_edge_variable(&op.input, variable)
        }
        LogicalOperator::Merge(op) => {
            op.variable != variable && binds_edge_variable(&op.input, variable)
        }
        // Passthrough shapes: the binding, if any, is below.
        LogicalOperator::Filter(op) => binds_edge_variable(&op.input, variable),
        LogicalOperator::Project(op) => {
            if let Some(projection) = op.projections.iter().rev().find(|projection| {
                super::output_column_name(projection.alias.as_deref(), &projection.expression) == variable
            }) {
                matches!(&projection.expression, LogicalExpression::Variable(source) if binds_edge_variable(&op.input, source))
            } else { op.pass_through_input && binds_edge_variable(&op.input, variable) }
        },
        LogicalOperator::Return(op) => op.items.iter().rev().find(|item| {
            super::output_column_name(item.alias.as_deref(), &item.expression) == variable
        }).is_some_and(|item| matches!(&item.expression, LogicalExpression::Variable(source) if binds_edge_variable(&op.input, source))),
        LogicalOperator::Sort(op) => binds_edge_variable(&op.input, variable),
        LogicalOperator::Limit(op) => binds_edge_variable(&op.input, variable),
        LogicalOperator::Skip(op) => binds_edge_variable(&op.input, variable),
        LogicalOperator::Distinct(op) => binds_edge_variable(&op.input, variable),
        LogicalOperator::Unwind(op) => op.variable != variable && binds_edge_variable(&op.input, variable),
        LogicalOperator::SetProperty(op) => binds_edge_variable(&op.input, variable),
        LogicalOperator::AddLabel(op) => binds_edge_variable(&op.input, variable),
        LogicalOperator::RemoveLabel(op) => binds_edge_variable(&op.input, variable),
        LogicalOperator::DeleteEdge(op) => binds_edge_variable(&op.input, variable),
        LogicalOperator::DeleteNode(op) => binds_edge_variable(&op.input, variable),
        LogicalOperator::CreateNode(op) => op.variable != variable && op
            .input
            .as_deref()
            .is_some_and(|input| binds_edge_variable(input, variable)),
        LogicalOperator::NodeScan(op) => op.variable != variable && op
            .input
            .as_deref()
            .is_some_and(|input| binds_edge_variable(input, variable)),
        _ => false,
    }
}

fn bound_reference(expr: &LogicalExpression, input: &LogicalOperator, scan: &str) -> bool {
    match expr {
        LogicalExpression::Variable(variable) | LogicalExpression::Property { variable, .. } => {
            variable != scan && defines_variable(input, variable)
        }
        _ => false,
    }
}

fn lookup_condition<'a>(
    expr: &'a LogicalExpression,
    variable: &str,
    input: &LogicalOperator,
) -> Option<(&'a str, &'a LogicalExpression)> {
    match expr {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => lookup_condition(left, variable, input)
            .or_else(|| lookup_condition(right, variable, input)),
        LogicalExpression::Binary {
            left,
            op: BinaryOp::Eq,
            right,
        } => {
            for (property_side, bound_side) in [
                (left.as_ref(), right.as_ref()),
                (right.as_ref(), left.as_ref()),
            ] {
                if let LogicalExpression::Property {
                    variable: owner,
                    property,
                } = property_side
                    && owner == variable
                    && bound_reference(bound_side, input, variable)
                {
                    return Some((property.as_str(), bound_side));
                }
            }
            None
        }
        _ => None,
    }
}

fn exact_cohort(value: &Value) -> Option<usize> {
    match value {
        Value::Int64(_) => Some(0),
        Value::String(_) => Some(1),
        Value::Bool(_) => Some(2),
        _ => None,
    }
}

// One query account covers all retained lookup containers. Growth admission
// includes old and new allocations; conservative geometric growth receipts
// remain charged until the scratch is discarded (no undercharged realloc peak).
#[derive(Default)]
struct LookupMemory {
    resources: Option<QueryResourceContext>,
    grant: Option<MemoryGrant>,
}

impl LookupMemory {
    fn check_cancelled(&self) -> std::result::Result<(), OperatorError> {
        self.resources
            .as_ref()
            .ok_or(OperatorError::ResidentContainerInvariant {
                container: "correlated property lookup",
                message: "query resources were not installed",
            })?
            .check_cancelled()?;
        Ok(())
    }

    fn grow(&mut self, bytes: usize) -> std::result::Result<(), OperatorError> {
        self.check_cancelled()?;
        if let Some(grant) = &mut self.grant {
            let total =
                grant
                    .size()
                    .checked_add(bytes)
                    .ok_or(MemoryGrantError::ArithmeticOverflow {
                        current_bytes: grant.size(),
                        additional_bytes: bytes,
                    })?;
            grant.try_resize(total)?;
        } else {
            let resources =
                self.resources
                    .as_ref()
                    .ok_or(OperatorError::ResidentContainerInvariant {
                        container: "correlated property lookup",
                        message: "query resources were not installed",
                    })?;
            self.grant = Some(resources.try_allocate(bytes).map_err(|error| match error {
                QueryResourceContextError::Memory(source) => OperatorError::ResidentMemory(source),
                error => OperatorError::Execution(error.to_string()),
            })?);
        }
        Ok(())
    }
}

fn lookup_capacity_error() -> OperatorError {
    OperatorError::ResidentContainerInvariant {
        container: "correlated property lookup",
        message: "scratch capacity overflow",
    }
}

fn reserve_lookup<T>(
    values: &mut Vec<T>,
    required: usize,
    memory: &mut LookupMemory,
) -> std::result::Result<(), OperatorError> {
    if required <= values.capacity() {
        return Ok(());
    }
    let target = values
        .capacity()
        .checked_mul(2)
        .ok_or_else(lookup_capacity_error)?
        .max(4)
        .max(required);
    let bytes = target
        .checked_mul(size_of::<T>())
        .ok_or_else(lookup_capacity_error)?;
    memory.grow(bytes)?;
    values
        .try_reserve_exact(target - values.len())
        .map_err(|source| OperatorError::ResidentContainerAllocation {
            container: "correlated lookup vector",
            source,
        })?;
    if values.capacity() > target {
        return Err(OperatorError::ResidentContainerInvariant {
            container: "correlated lookup vector",
            message: "allocator exceeded admitted capacity",
        });
    }
    Ok(())
}

#[derive(Default)]
struct CandidateLookup {
    nodes: Vec<NodeId>,
    exact: FxHashMap<HashableValue, Vec<usize>>,
    cohorts: [Vec<usize>; 4],
}

impl CandidateLookup {
    fn insert(
        &mut self,
        id: NodeId,
        value: Value,
        memory: &mut LookupMemory,
    ) -> std::result::Result<(), OperatorError> {
        if value.is_null() {
            return Ok(());
        }
        let position = self.nodes.len();
        reserve_lookup(
            &mut self.nodes,
            position.checked_add(1).ok_or_else(lookup_capacity_error)?,
            memory,
        )?;
        self.nodes.push(id);
        match exact_cohort(&value) {
            Some(cohort) => {
                let required = self.cohorts[cohort]
                    .len()
                    .checked_add(1)
                    .ok_or_else(lookup_capacity_error)?;
                reserve_lookup(&mut self.cohorts[cohort], required, memory)?;
                self.cohorts[cohort].push(position);
                let key = HashableValue::new(value.clone());
                if !self.exact.contains_key(&key) {
                    // Hashbrown 0.17.1 + allocator-api2 Global: <=4 power-of-two
                    // buckets per required entry, one control byte per bucket,
                    // and at most 16 trailing/padding bytes each. This ceiling
                    // also covers small tables; allocation_size verifies it.
                    if self.exact.len() == self.exact.capacity() {
                        let required = self
                            .exact
                            .len()
                            .checked_add(1)
                            .ok_or_else(lookup_capacity_error)?;
                        let buckets = required
                            .checked_next_power_of_two()
                            .and_then(|n| n.checked_mul(4))
                            .ok_or_else(lookup_capacity_error)?;
                        let bytes = buckets
                            .checked_mul(size_of::<(HashableValue, Vec<usize>)>() + 1)
                            .and_then(|n| n.checked_add(32))
                            .ok_or_else(lookup_capacity_error)?;
                        memory.grow(bytes)?;
                        self.exact.try_reserve(1).map_err(|error| {
                            OperatorError::ResidentAllocation(error.to_string())
                        })?;
                        if self.exact.allocation_size() > bytes {
                            return Err(OperatorError::ResidentContainerInvariant {
                                container: "correlated lookup map",
                                message: "allocator exceeded admitted capacity",
                            });
                        }
                    }
                    if let Value::String(text) = &value {
                        memory.grow(
                            text.len()
                                .checked_add(2 * size_of::<usize>())
                                .ok_or_else(lookup_capacity_error)?,
                        )?;
                    }
                }
                let bucket = self.exact.entry(key).or_default();
                let required = bucket
                    .len()
                    .checked_add(1)
                    .ok_or_else(lookup_capacity_error)?;
                reserve_lookup(bucket, required, memory)?;
                bucket.push(position);
            }
            None => {
                let required = self.cohorts[3]
                    .len()
                    .checked_add(1)
                    .ok_or_else(lookup_capacity_error)?;
                reserve_lookup(&mut self.cohorts[3], required, memory)?;
                self.cohorts[3].push(position);
            }
        }
        Ok(())
    }

    fn candidates(
        &self,
        value: &Value,
        out: &mut Vec<usize>,
        memory: &mut LookupMemory,
    ) -> std::result::Result<(), OperatorError> {
        out.clear();
        if value.is_null() {
            return Ok(());
        }
        if let Some(cohort) = exact_cohort(value) {
            // Float epsilon equality and numeric-string coercion are not
            // hash equality. Retain every other cohort for full residual
            // evaluation rather than silently dropping a coercible match.
            let mut slices = self.cohorts.each_ref().map(Vec::as_slice);
            slices[cohort] = self
                .exact
                .get(&HashableValue::new(value.clone()))
                .map_or(&[], Vec::as_slice);
            let required = slices
                .iter()
                .try_fold(0_usize, |total, slice| total.checked_add(slice.len()))
                .ok_or_else(lookup_capacity_error)?;
            reserve_lookup(out, required, memory)?;
            // Merge the already scan-ordered cohorts without an unbounded
            // sort; cancellation is checked at most 256 candidates apart.
            loop {
                if out.len().is_multiple_of(256) {
                    memory.check_cancelled()?;
                }
                let next = slices
                    .iter()
                    .enumerate()
                    .filter_map(|(i, values)| values.first().map(|position| (i, *position)))
                    .min_by_key(|(_, position)| *position);
                let Some((index, position)) = next else {
                    break;
                };
                out.push(position);
                if let Some((_, tail)) = slices[index].split_first() {
                    slices[index] = tail;
                }
            }
        } else {
            reserve_lookup(out, self.nodes.len(), memory)?;
            for start in (0..self.nodes.len()).step_by(256) {
                memory.check_cancelled()?;
                out.extend(start..start.saturating_add(256).min(self.nodes.len()));
            }
        }
        Ok(())
    }
}

struct CorrelatedPropertyLookupContext {
    property: PropertyKey,
    store: Arc<dyn GraphStoreSearch>,
    epoch: EpochId,
    transaction_id: Option<TransactionId>,
    schema: Vec<LogicalType>,
    scan_label: Option<String>,
    mode: LookupMode,
}

#[derive(Clone, Copy)]
enum LookupMode {
    Scan,
    Index,
}

impl LookupMode {
    fn is_indexed(self) -> bool {
        matches!(self, Self::Index)
    }
}

#[derive(Clone, Copy, Default)]
enum LookupPreparation {
    #[default]
    Unprepared,
    Prepared,
}

#[derive(Clone, Copy, Default)]
enum LookupExhaustion {
    #[default]
    Active,
    Exhausted,
}

#[derive(Clone, Copy, Default)]
enum LookupMatchSource {
    #[default]
    Scan,
    Index,
}

#[derive(Clone, Copy, Default)]
enum LookupReadRegistration {
    #[default]
    Pending,
    Recorded,
}

impl LookupPreparation {
    fn is_prepared(self) -> bool {
        matches!(self, Self::Prepared)
    }
}

impl LookupExhaustion {
    fn is_exhausted(self) -> bool {
        matches!(self, Self::Exhausted)
    }
}

impl LookupMatchSource {
    fn is_index(self) -> bool {
        matches!(self, Self::Index)
    }
}

impl LookupReadRegistration {
    fn is_recorded(self) -> bool {
        matches!(self, Self::Recorded)
    }
}

struct CorrelatedPropertyLookup {
    child: Box<dyn Operator>,
    candidates: Box<dyn Operator>,
    key: ExpressionPredicate,
    property: PropertyKey,
    store: Arc<dyn GraphStoreSearch>,
    epoch: EpochId,
    transaction_id: Option<TransactionId>,
    schema: Vec<LogicalType>,
    scan_label: Option<String>,
    mode: LookupMode,
    lookup: CandidateLookup,
    preparation: LookupPreparation,
    exhaustion: LookupExhaustion,
    input: Option<DataChunk>,
    selected: Vec<usize>,
    next_row: usize,
    active_row: usize,
    matches: Vec<usize>,
    next_match: usize,
    indexed_matches: Vec<NodeId>,
    match_source: LookupMatchSource,
    read_registration: LookupReadRegistration,
    // Declared after every scratch container so normal Drop retires allocations
    // before their real query/global grant. reset/error use the same ordering.
    memory: LookupMemory,
    #[cfg(test)]
    work: [usize; 3], // preparation rows, key evaluations, emitted candidate pairs
}

impl CorrelatedPropertyLookup {
    fn new(
        child: Box<dyn Operator>,
        candidates: Box<dyn Operator>,
        key: ExpressionPredicate,
        context: CorrelatedPropertyLookupContext,
    ) -> Self {
        Self {
            child,
            candidates,
            key,
            property: context.property,
            store: context.store,
            epoch: context.epoch,
            transaction_id: context.transaction_id,
            schema: context.schema,
            scan_label: context.scan_label,
            mode: context.mode,
            lookup: CandidateLookup::default(),
            preparation: LookupPreparation::default(),
            exhaustion: LookupExhaustion::default(),
            input: None,
            selected: Vec::new(),
            next_row: 0,
            active_row: 0,
            matches: Vec::new(),
            next_match: 0,
            indexed_matches: Vec::new(),
            match_source: LookupMatchSource::default(),
            read_registration: LookupReadRegistration::default(),
            memory: LookupMemory::default(),
            #[cfg(test)]
            work: [0; 3],
        }
    }

    fn prepare(&mut self) -> std::result::Result<(), OperatorError> {
        self.memory.check_cancelled()?;
        if self.preparation.is_prepared() {
            return Ok(());
        }
        while let Some(chunk) = self.candidates.next()? {
            self.memory.check_cancelled()?;
            let nodes = chunk
                .column(0)
                .ok_or_else(|| OperatorError::ColumnNotFound("lookup node".into()))?;
            for (ordinal, row) in chunk.selected_indices().enumerate() {
                if ordinal.is_multiple_of(256) {
                    self.memory.check_cancelled()?;
                }
                let id = nodes.get_node_id(row).ok_or_else(|| {
                    OperatorError::Execution("lookup candidate is not a node".into())
                })?;
                #[cfg(test)]
                {
                    self.work[0] += 1;
                }
                if let Some(value) = self.store.read_node_property_visible(
                    id,
                    &self.property,
                    self.epoch,
                    self.transaction_id,
                ) {
                    self.lookup.insert(id, value, &mut self.memory)?;
                }
            }
        }
        self.preparation = LookupPreparation::Prepared;
        Ok(())
    }

    fn advance_input(&mut self) -> std::result::Result<bool, OperatorError> {
        loop {
            self.memory.check_cancelled()?;
            if let Some(row) = self.selected.get(self.next_row).copied() {
                self.next_row += 1;
                self.active_row = row;
                let input = self
                    .input
                    .as_ref()
                    .ok_or_else(|| OperatorError::Execution("lookup input row is absent".into()))?;
                let value = self
                    .key
                    .eval_at(input, row)
                    .map_err(|error| OperatorError::Execution(error.to_string()))?;
                #[cfg(test)]
                {
                    self.work[1] += 1;
                }
                self.matches.clear();
                self.indexed_matches.clear();
                self.match_source = LookupMatchSource::Scan;
                if let Some(value) = value {
                    if self.mode.is_indexed() {
                        self.memory.check_cancelled()?;
                        if !self.read_registration.is_recorded() {
                            if let Some(tx) = self.transaction_id {
                                if let Some(label) = &self.scan_label {
                                    self.store.record_label_predicate_read(tx, label);
                                } else {
                                    self.store.record_lpg_dataset_read(tx);
                                }
                            }
                            self.read_registration = LookupReadRegistration::Recorded;
                        }
                        let indexed = self
                            .store
                            .lookup_nodes_indexed(PropertyIndexRequest {
                                property: self.property.as_str(),
                                predicate: PropertyIndexPredicate::Equal(&value),
                                epoch: self.epoch,
                                transaction_id: self.transaction_id,
                            })
                            .map_err(|error| OperatorError::Execution(error.to_string()))?;
                        if let Some(nodes) = indexed {
                            reserve_lookup(
                                &mut self.indexed_matches,
                                nodes.len(),
                                &mut self.memory,
                            )?;
                            for (ordinal, id) in nodes.into_iter().enumerate() {
                                if ordinal.is_multiple_of(256) {
                                    self.memory.check_cancelled()?;
                                }
                                if self.scan_label.as_deref().is_none_or(|label| {
                                    self.store.node_has_label_at_epoch(
                                        id,
                                        label,
                                        self.epoch,
                                        self.transaction_id.unwrap_or(TransactionId::SYSTEM),
                                    )
                                }) {
                                    self.indexed_matches.push(id);
                                }
                            }
                            self.match_source = LookupMatchSource::Index;
                        } else {
                            // A registered index may decline an unsupported
                            // runtime type. Materialize the established
                            // snapshot-aware fallback only in that case.
                            self.prepare()?;
                            self.lookup
                                .candidates(&value, &mut self.matches, &mut self.memory)?;
                        }
                    } else {
                        self.lookup
                            .candidates(&value, &mut self.matches, &mut self.memory)?;
                    }
                }
                self.next_match = 0;
                if (self.match_source.is_index() && !self.indexed_matches.is_empty())
                    || (!self.match_source.is_index() && !self.matches.is_empty())
                {
                    return Ok(true);
                }
                continue;
            }
            self.input = self.child.next()?;
            self.selected.clear();
            self.next_row = 0;
            let Some(input) = &self.input else {
                return Ok(false);
            };
            reserve_lookup(&mut self.selected, input.row_count(), &mut self.memory)?;
            self.selected.extend(input.selected_indices());
        }
    }
    fn next_inner(&mut self) -> OperatorResult {
        self.memory.check_cancelled()?;
        if !self.mode.is_indexed() {
            self.prepare()?;
        }
        let mut out = DataChunkBuilder::with_capacity(&self.schema, 2048);
        while out.row_count() < 2048 {
            if out.row_count().is_multiple_of(256) {
                self.memory.check_cancelled()?;
            }
            let active_matches = if self.match_source.is_index() {
                self.indexed_matches.len()
            } else {
                self.matches.len()
            };
            if self.next_match >= active_matches && !self.advance_input()? {
                break;
            }
            let id = if self.match_source.is_index() {
                self.indexed_matches
                    .get(self.next_match)
                    .copied()
                    .ok_or_else(|| OperatorError::Execution("indexed match is absent".into()))?
            } else {
                let position = self
                    .matches
                    .get(self.next_match)
                    .copied()
                    .ok_or_else(|| OperatorError::Execution("lookup match is absent".into()))?;
                self.lookup.nodes.get(position).copied().ok_or_else(|| {
                    OperatorError::Execution("lookup node position is absent".into())
                })?
            };
            self.next_match += 1;
            let input = self
                .input
                .as_ref()
                .ok_or_else(|| OperatorError::Execution("lookup input is absent".into()))?;
            for column in 0..input.column_count() {
                let value = input
                    .column(column)
                    .and_then(|values| values.get_value(self.active_row))
                    .ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("lookup input column {column}"))
                    })?;
                out.column_mut(column)
                    .ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("lookup output column {column}"))
                    })?
                    .push_value(value);
            }
            out.column_mut(input.column_count())
                .ok_or_else(|| OperatorError::ColumnNotFound("lookup output node".into()))?
                .push_node_id(id);
            out.advance_row();
            #[cfg(test)]
            {
                self.work[2] += 1;
            }
        }
        if out.row_count() == 0 {
            Ok(None)
        } else {
            Ok(Some(out.finish()))
        }
    }

    fn clear_scratch(&mut self) {
        self.lookup = CandidateLookup::default();
        self.input = None;
        self.selected = Vec::new();
        self.matches = Vec::new();
        self.indexed_matches = Vec::new();
        self.match_source = LookupMatchSource::Scan;
        self.read_registration = LookupReadRegistration::Pending;
        self.memory.grant = None;
        // A cached plan must not keep a completed query pool active (or retain
        // its spill manager). Every new execution installs its exact context.
        self.memory.resources = None;
        self.preparation = LookupPreparation::Unprepared;
        self.next_row = 0;
        self.next_match = 0;
    }
}

impl Operator for CorrelatedPropertyLookup {
    fn next(&mut self) -> OperatorResult {
        if self.exhaustion.is_exhausted() {
            return Ok(None);
        }
        let result = self.next_inner();
        if !matches!(&result, Ok(Some(_))) {
            self.clear_scratch();
            self.exhaustion = LookupExhaustion::Exhausted;
        }
        result
    }

    fn reset(&mut self) {
        self.child.reset();
        self.candidates.reset();
        self.clear_scratch();
        self.exhaustion = LookupExhaustion::Active;
        #[cfg(test)]
        {
            self.work = [0; 3];
        }
    }

    fn name(&self) -> &'static str {
        "CorrelatedPropertyLookup"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> std::result::Result<(), QueryResourceContextError> {
        if self
            .memory
            .resources
            .as_ref()
            .is_some_and(|previous| previous.query_id() != resources.query_id())
        {
            self.reset();
        }
        self.child.install_resource_context(resources)?;
        self.candidates.install_resource_context(resources)?;
        // Reinstallation rebinds only after releasing this execution's scratch.
        // Child operators receive the same single context, never a fresh pool.
        self.memory.resources = Some(resources.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "lpg")]
    use grafeo_common::memory::buffer::BufferManager;
    #[cfg(feature = "lpg")]
    use grafeo_core::execution::operators::{FilterExpression, ScanOperator};
    #[cfg(feature = "lpg")]
    use grafeo_core::execution::{QueryCancellationHandle, QueryExecutionControl};
    #[cfg(feature = "lpg")]
    use grafeo_core::graph::lpg::LpgStore;

    #[cfg(feature = "lpg")]
    struct Rows {
        values: Vec<Value>,
        position: usize,
        cancel: Option<QueryCancellationHandle>,
    }

    #[cfg(feature = "lpg")]
    impl Operator for Rows {
        fn next(&mut self) -> OperatorResult {
            if let Some(cancel) = &self.cancel {
                let _ = cancel.try_cancel();
            }
            if self.position == self.values.len() {
                return Ok(None);
            }
            let mut out = DataChunkBuilder::with_capacity(&[LogicalType::Any], 2048);
            while self.position < self.values.len() && out.row_count() < 2048 {
                let value = self
                    .values
                    .get(self.position)
                    .cloned()
                    .ok_or_else(|| OperatorError::Execution("test row missing".into()))?;
                out.column_mut(0)
                    .ok_or_else(|| OperatorError::ColumnNotFound("test value".into()))?
                    .push_value(value);
                out.advance_row();
                self.position += 1;
            }
            Ok(Some(out.finish()))
        }
        fn reset(&mut self) {
            self.position = 0;
        }
        fn name(&self) -> &'static str {
            "Rows"
        }
        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[cfg(feature = "lpg")]
    fn lookup_fixture(
        cancel: Option<QueryCancellationHandle>,
    ) -> std::result::Result<CorrelatedPropertyLookup, Box<dyn std::error::Error>> {
        let store = Arc::new(LpgStore::new()?);
        for id in 0_i64..32 {
            store.create_node_with_props(&["Node"], [("lookup", Value::Int64(id))]);
        }
        let reader: Arc<dyn GraphStoreSearch> = store;
        let epoch = reader.current_epoch();
        let candidates = Box::new(
            ScanOperator::with_label(Arc::clone(&reader), "Node")
                .with_transaction_context(epoch, None),
        );
        let key = ExpressionPredicate::new(
            FilterExpression::Variable("e".into()),
            HashMap::from([("e".into(), 0)]),
            Arc::clone(&reader),
        );
        let rows = Box::new(Rows {
            values: (0_i64..5000).map(|id| Value::Int64(id % 32)).collect(),
            position: 0,
            cancel,
        });
        Ok(CorrelatedPropertyLookup::new(
            rows,
            candidates,
            key,
            CorrelatedPropertyLookupContext {
                property: "lookup".into(),
                store: reader,
                epoch,
                transaction_id: None,
                schema: vec![LogicalType::Any, LogicalType::Node],
                scan_label: Some("Node".into()),
                mode: LookupMode::Scan,
            },
        ))
    }

    #[cfg(feature = "lpg")]
    fn list_value_lookup(
        indexed: bool,
    ) -> std::result::Result<Vec<NodeId>, Box<dyn std::error::Error>> {
        let store = Arc::new(LpgStore::new()?);
        let scalar = store.create_node_with_props(&["Node"], [("lookup", Value::Int64(7))]);
        let list = store.create_node_with_props(
            &["Node"],
            [("lookup", Value::List(vec![Value::Int64(7)].into()))],
        );
        if indexed {
            store.create_property_index("lookup");
            assert!(store.has_property_index("lookup"));
        }
        let reader: Arc<dyn GraphStoreSearch> = store;
        let epoch = reader.current_epoch();
        let candidates = Box::new(
            ScanOperator::with_label(Arc::clone(&reader), "Node")
                .with_transaction_context(epoch, None),
        );
        let key = ExpressionPredicate::new(
            FilterExpression::Variable("e".into()),
            HashMap::from([("e".into(), 0)]),
            Arc::clone(&reader),
        );
        let rows = Box::new(Rows {
            values: vec![Value::List(vec![Value::Int64(7)].into())],
            position: 0,
            cancel: None,
        });
        let mut operator = CorrelatedPropertyLookup::new(
            rows,
            candidates,
            key,
            CorrelatedPropertyLookupContext {
                property: "lookup".into(),
                store: Arc::clone(&reader),
                epoch,
                transaction_id: None,
                schema: vec![LogicalType::Any, LogicalType::Node],
                scan_label: Some("Node".into()),
                mode: if indexed {
                    LookupMode::Index
                } else {
                    LookupMode::Scan
                },
            },
        );
        let resources = QueryResourceContext::new(BufferManager::with_budget(1_000_000))?;
        operator.install_resource_context(&resources)?;
        let mut ids = Vec::new();
        while let Some(chunk) = operator.next()? {
            let nodes = chunk
                .column(1)
                .ok_or_else(|| OperatorError::ColumnNotFound("lookup node".into()))?;
            for row in chunk.selected_indices() {
                ids.push(
                    nodes
                        .get_node_id(row)
                        .ok_or_else(|| OperatorError::Execution("lookup node missing".into()))?,
                );
            }
        }
        assert!(ids.contains(&scalar));
        assert!(ids.contains(&list));
        Ok(ids)
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn correlated_lookup_prepares_candidates_once_and_rebuilds_on_reset()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut operator = lookup_fixture(None)?;
        let resources = QueryResourceContext::new(BufferManager::with_budget(1_000_000))?;
        for pass in 0..2 {
            operator.install_resource_context(&resources)?;
            let mut rows = 0;
            while let Some(chunk) = operator.next()? {
                rows += chunk.row_count();
            }
            assert_eq!(rows, 5000);
            assert_eq!(
                operator.work,
                [32, 5000, 5000],
                "pass {pass}: no input × candidate materialization"
            );
            assert_eq!(
                resources.query_stats().allocated_bytes,
                0,
                "EOF releases cached-plan scratch"
            );
            assert!(operator.next()?.is_none());
            assert!(operator.memory.resources.is_none());
            assert_eq!(
                operator.work,
                [32, 5000, 5000],
                "EOF remains terminal without a second scan"
            );
            operator.reset();
            assert_eq!(resources.query_stats().allocated_bytes, 0);
        }
        let next_resources = QueryResourceContext::new(BufferManager::with_budget(1_000_000))?;
        operator.install_resource_context(&next_resources)?;
        assert!(operator.next()?.is_some());
        assert!(next_resources.query_stats().allocated_bytes > 0);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        operator.reset();
        assert_eq!(next_resources.query_stats().allocated_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn indexed_correlated_equality_keeps_list_value_as_equality()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let generic = list_value_lookup(false)?;
        let indexed = list_value_lookup(true)?;
        assert_eq!(
            indexed, generic,
            "indexed Eq must use generic equality candidates"
        );
        Ok(())
    }

    #[cfg(all(feature = "lpg", any(feature = "gql", feature = "cypher")))]
    fn assert_endpoint_lookup_plan(
        mut logical: crate::query::plan::LogicalPlan,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::query::{binder::Binder, optimizer::Optimizer, processor::substitute_params};
        use grafeo_core::graph::GraphStoreMut;
        let store = Arc::new(LpgStore::new()?);
        for id in 0_i64..32 {
            store.create_node_with_props(&["Node"], [("id", Value::Int64(id))]);
        }
        let rows = (0_i64..128)
            .map(|i| {
                Value::Map(
                    std::collections::BTreeMap::from([
                        (PropertyKey::new("s"), Value::Int64(i % 32)),
                        (PropertyKey::new("t"), Value::Int64((i + 1) % 32)),
                        (
                            PropertyKey::new("p"),
                            Value::Map(
                                std::collections::BTreeMap::from([(
                                    PropertyKey::new("w"),
                                    Value::Int64(i),
                                )])
                                .into(),
                            ),
                        ),
                    ])
                    .into(),
                )
            })
            .collect::<Vec<_>>();
        substitute_params(
            &mut logical,
            &HashMap::from([("es".into(), Value::List(rows.into()))]),
        )?;
        Binder::new().bind(&logical)?;
        let logical = Optimizer::from_graph_store(store.as_ref()).optimize(logical)?;
        let reader: Arc<dyn GraphStoreSearch> = store.clone();
        let mut planner = Planner::new(reader);
        planner.write_store = Some(store.clone() as Arc<dyn GraphStoreMut>);
        let (mut physical, entries) = planner.plan_profiled(&logical)?;
        let names = entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names.iter().filter(|name| **name == "Scan").count(),
            2,
            "{names:?}"
        );
        assert!(
            !names.contains(&"NestedLoopJoin"),
            "the actual frontend lowering must not retain the Cartesian scans: {names:?}"
        );
        let resources = QueryResourceContext::new(BufferManager::with_budget(4_000_000))?;
        physical.operator.install_resource_context(&resources)?;
        while physical.operator.next()?.is_some() {}
        assert_eq!(
            store.edge_count(),
            128,
            "the profiled endpoint query must actually create every edge"
        );
        for entry in entries.iter().filter(|entry| entry.name == "Scan") {
            assert_eq!(
                entry.stats.lock().rows_out,
                32,
                "each endpoint scans once, not 128 × 32 pairs"
            );
        }
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(feature = "gql")]
    fn default_gql_endpoint_query_lowers_to_two_snapshot_lookup_scans()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        assert_endpoint_lookup_plan(crate::query::translators::gql::translate(
            "UNWIND $es AS e MATCH (s:Node {id:e.s}), (t:Node {id:e.t}) CREATE (s)-[:TYPE]->(t)",
        )?)
    }

    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(feature = "cypher")]
    fn cypher_endpoint_query_lowers_to_two_snapshot_lookup_scans()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        assert_endpoint_lookup_plan(crate::query::translators::cypher::translate(
            "UNWIND $es AS e MATCH (s:Node {id:e.s}), (t:Node {id:e.t}) CREATE (s)-[:TYPE]->(t)",
        )?)
    }

    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(feature = "gql")]
    fn nested_unwind_forwards_resources_to_earlier_lookup()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        assert_endpoint_lookup_plan(crate::query::translators::gql::translate(
            "UNWIND $es AS e MATCH (s:Node {id:e.s}) UNWIND [e] AS again MATCH (t:Node {id:again.t}) CREATE (s)-[:TYPE]->(t)",
        )?)
    }

    // A trailing clause after CREATE must not disable the endpoint lookups for
    // the whole statement. Both surfaces are asserted: only Cypher populates
    // `SetPropertyOp::is_edge`, GQL hardcodes it false, so admission must
    // derive edge-ness from the logical tree instead of the flag.
    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(feature = "gql")]
    fn gql_trailing_edge_set_map_keeps_endpoint_lookup_plan()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        assert_endpoint_lookup_plan(crate::query::translators::gql::translate(
            "UNWIND $es AS e MATCH (s:Node {id:e.s}), (t:Node {id:e.t}) CREATE (s)-[r:TYPE]->(t) SET r = e.p",
        )?)
    }

    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(feature = "cypher")]
    fn cypher_trailing_edge_set_map_keeps_endpoint_lookup_plan()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        assert_endpoint_lookup_plan(crate::query::translators::cypher::translate(
            "UNWIND $es AS e MATCH (s:Node {id:e.s}), (t:Node {id:e.t}) CREATE (s)-[r:TYPE]->(t) SET r = e.p",
        )?)
    }

    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(feature = "gql")]
    fn gql_trailing_edge_set_property_keeps_endpoint_lookup_plan()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        assert_endpoint_lookup_plan(crate::query::translators::gql::translate(
            "UNWIND $es AS e MATCH (s:Node {id:e.s}), (t:Node {id:e.t}) CREATE (s)-[r:TYPE]->(t) SET r.k = e.s",
        )?)
    }

    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(feature = "cypher")]
    fn cypher_trailing_edge_set_property_keeps_endpoint_lookup_plan()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        assert_endpoint_lookup_plan(crate::query::translators::cypher::translate(
            "UNWIND $es AS e MATCH (s:Node {id:e.s}), (t:Node {id:e.t}) CREATE (s)-[r:TYPE]->(t) SET r.k = e.s",
        )?)
    }

    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(feature = "gql")]
    fn gql_trailing_aggregate_keeps_endpoint_lookup_plan()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        assert_endpoint_lookup_plan(crate::query::translators::gql::translate(
            "UNWIND $es AS e MATCH (s:Node {id:e.s}), (t:Node {id:e.t}) CREATE (s)-[r:TYPE]->(t) RETURN count(r)",
        )?)
    }

    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(feature = "cypher")]
    fn cypher_trailing_aggregate_keeps_endpoint_lookup_plan()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        assert_endpoint_lookup_plan(crate::query::translators::cypher::translate(
            "UNWIND $es AS e MATCH (s:Node {id:e.s}), (t:Node {id:e.t}) CREATE (s)-[r:TYPE]->(t) RETURN count(r)",
        )?)
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn correlated_lookup_budget_denial_and_cancellation_release_residency()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        // The first node vector fits; the map admission then fails, exercising
        // retirement after partial preparation rather than just empty denial.
        let mut denied = lookup_fixture(None)?;
        let resources = QueryResourceContext::new(BufferManager::with_budget(128))?;
        denied.install_resource_context(&resources)?;
        assert!(matches!(
            denied.next(),
            Err(OperatorError::ResidentMemory(_))
        ));
        assert!(denied.work[0] > 0);
        assert!(denied.lookup.nodes.is_empty());
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(resources.buffer_stats().total_allocated, 0);

        let control = QueryExecutionControl::new();
        let resources = QueryResourceContext::new_with_cancellation(
            BufferManager::with_budget(1_000_000),
            control.token(),
        )?;
        let mut cancelled = lookup_fixture(Some(control.cancellation_handle()))?;
        cancelled.install_resource_context(&resources)?;
        assert!(matches!(
            cancelled.next(),
            Err(OperatorError::QueryCancelled(_))
        ));
        assert_eq!(
            cancelled.work,
            [32, 0, 0],
            "cancel during input acquisition after lookup preparation"
        );
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(resources.buffer_stats().total_allocated, 0);
        assert!(cancelled.lookup.nodes.is_empty());

        let resources = QueryResourceContext::new(BufferManager::with_budget(1_000_000))?;
        let mut abandoned = lookup_fixture(None)?;
        abandoned.install_resource_context(&resources)?;
        assert!(abandoned.next()?.is_some());
        assert!(resources.query_stats().allocated_bytes > 0);
        drop(abandoned);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        Ok(())
    }

    #[test]
    fn lookup_effects_track_keys_labels_and_aliases() {
        use crate::query::plan::{
            AddLabelOp, CreateNodeOp, DeleteEdgeOp, DeleteNodeOp, MergeOp, SetPropertyOp,
        };
        assert!(!LookupPlanEffects::default().allows(Some("Target"), "id"));
        let write = |variable: &str, property: &str, replace| {
            LogicalOperator::SetProperty(SetPropertyOp {
                variable: variable.into(),
                properties: vec![(property.into(), LogicalExpression::Literal(Value::Int64(1)))],
                replace,
                is_edge: false,
                input: Box::new(LogicalOperator::Empty),
            })
        };
        assert!(
            Planner::lookup_plan_effects(&write("other", "note", false))
                .allows(Some("Target"), "id")
        );
        // Either name may identify the same node at runtime; naming is no alias proof.
        for variable in ["target", "alias"] {
            assert!(
                !Planner::lookup_plan_effects(&write(variable, "id", false))
                    .allows(Some("Target"), "id")
            );
        }
        assert!(
            !Planner::lookup_plan_effects(&write("other", "*", false)).allows(Some("Target"), "id")
        );
        assert!(
            !Planner::lookup_plan_effects(&write("other", "note", true))
                .allows(Some("Target"), "id")
        );
        let labels = Planner::lookup_plan_effects(&LogicalOperator::AddLabel(AddLabelOp {
            variable: "alias".into(),
            labels: vec!["Other".into()],
            input: Box::new(LogicalOperator::Empty),
        }));
        assert!(labels.allows(Some("Target"), "id"));
        assert!(labels.allows(None, "id"));
        assert!(!labels.allows(Some("Other"), "id"));
        let creation = Planner::lookup_plan_effects(&LogicalOperator::CreateNode(CreateNodeOp {
            variable: "fresh".into(),
            labels: vec!["Other".into()],
            properties: vec![("id".into(), LogicalExpression::Literal(Value::Int64(1)))],
            input: None,
        }));
        assert!(creation.allows(Some("Target"), "id"));
        assert!(!creation.allows(Some("Other"), "id"));
        assert!(!creation.allows(None, "id"));
        assert!(
            Planner::lookup_plan_effects(&LogicalOperator::DeleteEdge(DeleteEdgeOp {
                variable: "r".into(),
                input: Box::new(LogicalOperator::Empty),
            }))
            .allows(Some("Target"), "id")
        );
        assert!(
            !Planner::lookup_plan_effects(&LogicalOperator::DeleteNode(DeleteNodeOp {
                variable: "other".into(),
                detach: false,
                input: Box::new(LogicalOperator::Empty),
            }))
            .allows(Some("Target"), "id")
        );
        let mut merge = MergeOp {
            variable: "m".into(),
            labels: vec!["Other".into()],
            match_properties: Vec::new(),
            on_create: vec![("id".into(), LogicalExpression::Literal(Value::Int64(1)))],
            on_match: Vec::new(),
            input: Box::new(LogicalOperator::Empty),
        };
        assert!(
            Planner::lookup_plan_effects(&LogicalOperator::Merge(merge.clone()))
                .allows(Some("Target"), "id")
        );
        merge.on_match = merge.on_create.clone();
        assert!(
            !Planner::lookup_plan_effects(&LogicalOperator::Merge(merge))
                .allows(Some("Target"), "id")
        );
    }

    #[test]
    fn lookup_binding_scope_preserves_mutations_and_project_passthrough() {
        use crate::query::plan::{
            CreateEdgeOp, DeleteNodeOp, MergeOp, ProjectOp, Projection, SetPropertyOp,
        };
        let input = LogicalOperator::CreateEdge(CreateEdgeOp {
            variable: Some("r".into()),
            from_variable: "n".into(),
            to_variable: "n".into(),
            edge_type: "R".into(),
            properties: Vec::new(),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".into(),
                label: None,
                input: None,
            })),
        });
        assert!(defines_variable(&input, "n"));
        let delete = |variable: &str| {
            LogicalOperator::DeleteNode(DeleteNodeOp {
                variable: variable.into(),
                detach: false,
                input: Box::new(input.clone()),
            })
        };
        assert!(Planner::lookup_plan_effects(&delete("r")).allows(Some("Target"), "id"));
        assert!(defines_variable(&delete("r"), "n"));
        assert!(!Planner::lookup_plan_effects(&delete("n")).allows(Some("Target"), "id"));
        assert!(!Planner::lookup_plan_effects(&delete("alias")).allows(Some("Target"), "id"));
        let mut merge = MergeOp {
            variable: "other".into(),
            labels: vec!["Other".into()],
            match_properties: Vec::new(),
            on_create: Vec::new(),
            on_match: Vec::new(),
            input: Box::new(input.clone()),
        };
        assert!(binds_edge_variable(
            &LogicalOperator::Merge(merge.clone()),
            "r"
        ));
        merge.variable = "r".into();
        assert!(!binds_edge_variable(&LogicalOperator::Merge(merge), "r"));
        let mut project = ProjectOp {
            projections: Vec::new(),
            pass_through_input: true,
            input: Box::new(input),
        };
        assert!(defines_variable(
            &LogicalOperator::Project(project.clone()),
            "n"
        ));
        assert!(binds_edge_variable(
            &LogicalOperator::Project(project.clone()),
            "r"
        ));
        project.projections.push(Projection {
            expression: LogicalExpression::Variable("n".into()),
            alias: Some("r".into()),
        });
        let shadowed = LogicalOperator::Project(project.clone());
        assert!(!binds_edge_variable(&shadowed, "r"));
        let update = LogicalOperator::SetProperty(SetPropertyOp {
            variable: "r".into(),
            properties: vec![("id".into(), LogicalExpression::Literal(Value::Int64(1)))],
            replace: false,
            is_edge: false,
            input: Box::new(shadowed),
        });
        assert!(defines_variable(&update, "n"));
        assert!(!Planner::lookup_plan_effects(&update).allows(None, "id"));
        project.pass_through_input = false;
        assert!(!defines_variable(&LogicalOperator::Project(project), "n"));
    }

    #[test]
    fn lookup_plan_admission_excludes_node_property_mutation() {
        use crate::query::plan::{AggregateOp, CreateEdgeOp, DeleteNodeOp, SetPropertyOp};
        let update = SetPropertyOp {
            variable: "n".into(),
            replace: false,
            is_edge: false,
            properties: vec![("lookup".into(), LogicalExpression::Literal(Value::Int64(2)))],
            input: Box::new(LogicalOperator::Empty),
        };
        assert!(
            !Planner::lookup_plan_is_stable(&LogicalOperator::SetProperty(update.clone())),
            "a node SET can change the property the lookup key filters on"
        );

        // The flag is never the deciding input: a node target claiming
        // `is_edge: true` must still be refused, and an edge target spelled
        // `is_edge: false` (every GQL statement) must still be admitted.
        let mut lying_flag = update;
        lying_flag.is_edge = true;
        assert!(!Planner::lookup_plan_is_stable(
            &LogicalOperator::SetProperty(lying_flag)
        ));

        let create_edge = |input: LogicalOperator| {
            LogicalOperator::CreateEdge(CreateEdgeOp {
                variable: Some("r".into()),
                from_variable: "s".into(),
                to_variable: "t".into(),
                edge_type: "TYPE".into(),
                properties: Vec::new(),
                input: Box::new(input),
            })
        };
        let edge_set = |variable: &str| {
            LogicalOperator::SetProperty(SetPropertyOp {
                variable: variable.into(),
                replace: true,
                is_edge: false,
                properties: vec![("k".into(), LogicalExpression::Literal(Value::Int64(2)))],
                input: Box::new(create_edge(LogicalOperator::Empty)),
            })
        };
        assert!(
            Planner::lookup_plan_is_stable(&edge_set("r")),
            "an edge property write cannot disturb a node-keyed lookup"
        );
        assert!(
            !Planner::lookup_plan_is_stable(&edge_set("other")),
            "a variable the tree does not bind as an edge is not an edge"
        );

        let aggregate = |input: LogicalOperator| {
            LogicalOperator::Aggregate(AggregateOp {
                group_by: Vec::new(),
                aggregates: Vec::new(),
                input: Box::new(input),
                having: None,
            })
        };
        assert!(
            Planner::lookup_plan_is_stable(&aggregate(create_edge(LogicalOperator::Empty))),
            "counting the created edges is a pure read"
        );
        assert!(
            !Planner::lookup_plan_is_stable(&aggregate(LogicalOperator::DeleteNode(
                DeleteNodeOp {
                    variable: "n".into(),
                    detach: true,
                    input: Box::new(LogicalOperator::Empty),
                }
            ))),
            "an aggregate does not launder an unstable input"
        );
    }

    #[test]
    fn lookup_plan_admits_only_bound_empty_create_node_references() {
        use crate::query::plan::CreateNodeOp;
        let bound = CreateNodeOp {
            variable: "n".into(),
            labels: Vec::new(),
            properties: Vec::new(),
            input: Some(Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".into(),
                label: Some("Node".into()),
                input: None,
            }))),
        };
        assert!(Planner::lookup_plan_is_stable(
            &LogicalOperator::CreateNode(bound.clone())
        ));
        let mut creates_node = bound.clone();
        creates_node.variable = "fresh".into();
        assert!(!Planner::lookup_plan_is_stable(
            &LogicalOperator::CreateNode(creates_node)
        ));
        let mut declares_label = bound.clone();
        declares_label.labels.push("Other".into());
        assert!(!Planner::lookup_plan_is_stable(
            &LogicalOperator::CreateNode(declares_label)
        ));
        let mut declares_property = bound;
        declares_property
            .properties
            .push(("id".into(), LogicalExpression::Literal(Value::Int64(1))));
        assert!(!Planner::lookup_plan_is_stable(
            &LogicalOperator::CreateNode(declares_property)
        ));
    }
}
