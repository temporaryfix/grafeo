//! Statement depth limit.
//!
//! Binding, parameter substitution, optimization and planning all recurse
//! once per operator and once per nested expression. A long flat statement
//! builds a deep plan without deep syntax: `CREATE` with many comma-separated
//! patterns chains one operator per pattern, and `a OR b OR …` nests one
//! expression per term. On a 2 MiB thread about 700 levels overflowed the
//! stack and aborted the process. The check here walks at most
//! [`MAX_PLAN_DEPTH`] levels and rejects anything deeper with a query error
//! before any of those passes runs.

use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};

use super::plan::{LogicalExpression, LogicalOperator, LogicalPlan, MapProjectionEntry};

/// Deepest operator-and-expression path a statement may have.
///
/// A path counts every operator and every nested expression on it, because
/// planning converts an operator's expressions while its inputs are still on
/// the stack. Measured on a 2 MiB thread in the test profile, LPG chains ran at
/// 650 levels and overflowed at 800, and SPARQL plans 516 levels deep ran.
/// `statement_depth` tests run statements at this limit on a 2 MiB thread, so a
/// change that grows planner frames fails there instead of in production. Long
/// batches belong in `UNWIND` over a list parameter.
pub const MAX_PLAN_DEPTH: usize = 640;

/// Rejects a plan whose operator-and-expression depth exceeds
/// [`MAX_PLAN_DEPTH`].
///
/// # Errors
///
/// Returns a semantic query error naming the limit. The walk recurses at most
/// `MAX_PLAN_DEPTH` levels, so the check itself cannot overflow.
pub fn check_plan_depth(plan: &LogicalPlan) -> Result<()> {
    check_operator(&plan.root, 1)
}

/// Returns `plan` if it is within [`MAX_PLAN_DEPTH`], for translators to call
/// on the plan they built.
///
/// A rejected plan is taken apart without recursion: dropping it normally
/// recurses once per operator, and a plan tens of thousands of operators deep
/// overflowed the stack while being dropped after its rejection.
///
/// # Errors
///
/// Returns the depth-limit query error.
pub fn admit(plan: LogicalPlan) -> Result<LogicalPlan> {
    match check_plan_depth(&plan) {
        Ok(()) => Ok(plan),
        Err(error) => {
            dismantle(plan.root);
            Err(error)
        }
    }
}

/// Drops an operator tree one operator at a time.
fn dismantle(root: LogicalOperator) {
    let mut pending = vec![root];
    while let Some(op) = pending.pop() {
        let _shallow = op.map_children(|child| {
            pending.push(child);
            LogicalOperator::Empty
        });
    }
}

/// Counts how deeply a translator is nested while it converts syntax, so a
/// long flat chain such as `a OR b OR …` is rejected before the conversion
/// itself runs out of stack.
#[derive(Debug, Default)]
pub struct TranslationDepth(std::cell::Cell<usize>);

/// One level of [`TranslationDepth`], released when dropped.
pub struct TranslationLevel<'a>(&'a std::cell::Cell<usize>);

impl TranslationDepth {
    /// Enters one more level of nesting.
    ///
    /// # Errors
    ///
    /// Returns the depth-limit query error beyond [`MAX_PLAN_DEPTH`] levels.
    pub fn enter(&self) -> Result<TranslationLevel<'_>> {
        let depth = self.0.get() + 1;
        if depth > MAX_PLAN_DEPTH {
            return Err(too_deep());
        }
        self.0.set(depth);
        Ok(TranslationLevel(&self.0))
    }
}

impl Drop for TranslationLevel<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}

pub(crate) fn too_deep() -> Error {
    Error::Query(QueryError::new(
        QueryErrorKind::Semantic,
        format!(
            "Statement nests more than {MAX_PLAN_DEPTH} operators and expressions; \
             split it, or pass long batches as a list parameter to UNWIND"
        ),
    ))
}

fn check_operator(op: &LogicalOperator, depth: usize) -> Result<()> {
    if depth > MAX_PLAN_DEPTH {
        return Err(too_deep());
    }
    for expression in operator_expressions(op) {
        check_expression(expression, depth + 1)?;
    }
    for child in op.children() {
        check_operator(child, depth + 1)?;
    }
    Ok(())
}

fn check_expression(expression: &LogicalExpression, depth: usize) -> Result<()> {
    if depth > MAX_PLAN_DEPTH {
        return Err(too_deep());
    }
    let next = depth + 1;
    match expression {
        LogicalExpression::Literal(_)
        | LogicalExpression::Variable(_)
        | LogicalExpression::Property { .. }
        | LogicalExpression::Parameter(_)
        | LogicalExpression::Labels(_)
        | LogicalExpression::Type(_)
        | LogicalExpression::Id(_) => Ok(()),
        LogicalExpression::Binary { left, right, .. } => {
            check_expression(left, next)?;
            check_expression(right, next)
        }
        LogicalExpression::Unary { operand, .. } => check_expression(operand, next),
        LogicalExpression::FunctionCall { args, .. } => {
            args.iter().try_for_each(|arg| check_expression(arg, next))
        }
        LogicalExpression::List(items) => items
            .iter()
            .try_for_each(|item| check_expression(item, next)),
        LogicalExpression::Map(pairs) => pairs
            .iter()
            .try_for_each(|(_, value)| check_expression(value, next)),
        LogicalExpression::IndexAccess { base, index } => {
            check_expression(base, next)?;
            check_expression(index, next)
        }
        LogicalExpression::SliceAccess { base, start, end } => {
            check_expression(base, next)?;
            start
                .iter()
                .chain(end.iter())
                .try_for_each(|bound| check_expression(bound, next))
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            operand
                .iter()
                .chain(else_clause.iter())
                .try_for_each(|expression| check_expression(expression, next))?;
            when_clauses.iter().try_for_each(|(condition, result)| {
                check_expression(condition, next)?;
                check_expression(result, next)
            })
        }
        LogicalExpression::ListComprehension {
            list_expr,
            filter_expr,
            map_expr,
            ..
        } => {
            check_expression(list_expr, next)?;
            filter_expr
                .iter()
                .try_for_each(|filter| check_expression(filter, next))?;
            check_expression(map_expr, next)
        }
        LogicalExpression::ListPredicate {
            list_expr,
            predicate,
            ..
        } => {
            check_expression(list_expr, next)?;
            check_expression(predicate, next)
        }
        LogicalExpression::ExistsSubquery(subplan)
        | LogicalExpression::CountSubquery(subplan)
        | LogicalExpression::ValueSubquery(subplan) => check_operator(subplan, next),
        LogicalExpression::MapProjection { entries, .. } => {
            entries.iter().try_for_each(|entry| match entry {
                MapProjectionEntry::LiteralEntry(_, value) => check_expression(value, next),
                MapProjectionEntry::PropertySelector(_) | MapProjectionEntry::AllProperties => {
                    Ok(())
                }
            })
        }
        LogicalExpression::Reduce {
            initial,
            list,
            expression,
            ..
        } => {
            check_expression(initial, next)?;
            check_expression(list, next)?;
            check_expression(expression, next)
        }
        LogicalExpression::PatternComprehension {
            subplan,
            projection,
        } => {
            check_operator(subplan, next)?;
            check_expression(projection, next)
        }
    }
}

/// The expressions an operator evaluates itself; its inputs come from
/// [`LogicalOperator::children`].
fn operator_expressions(op: &LogicalOperator) -> Vec<&LogicalExpression> {
    match op {
        LogicalOperator::Filter(filter) => vec![&filter.predicate],
        LogicalOperator::Return(ret) => ret.items.iter().map(|item| &item.expression).collect(),
        LogicalOperator::Project(project) => project
            .projections
            .iter()
            .map(|projection| &projection.expression)
            .collect(),
        LogicalOperator::Expand(expand) => expand
            .edge_predicate
            .iter()
            .chain(expand.path_predicate.iter())
            .collect(),
        LogicalOperator::Join(join) => join
            .conditions
            .iter()
            .flat_map(|condition| [&condition.left, &condition.right])
            .collect(),
        LogicalOperator::LeftJoin(join) => join
            .compatibility_conditions
            .iter()
            .flat_map(|condition| [&condition.left, &condition.right])
            .chain(join.condition.iter())
            .collect(),
        LogicalOperator::AntiJoin(anti) => anti
            .compatibility_conditions
            .iter()
            .flat_map(|condition| [&condition.left, &condition.right])
            .collect(),
        LogicalOperator::MultiWayJoin(join) => join
            .conditions
            .iter()
            .flat_map(|condition| [&condition.left, &condition.right])
            .collect(),
        LogicalOperator::Aggregate(aggregate) => aggregate
            .group_by
            .iter()
            .chain(aggregate.aggregates.iter().flat_map(|function| {
                function
                    .expression
                    .iter()
                    .chain(function.expression2.iter())
                    .chain(function.distinct_key.iter())
            }))
            .collect(),
        LogicalOperator::Sort(sort) => sort.keys.iter().map(|key| &key.expression).collect(),
        LogicalOperator::CreateNode(create) => {
            create.properties.iter().map(|(_, value)| value).collect()
        }
        LogicalOperator::CreateEdge(create) => {
            create.properties.iter().map(|(_, value)| value).collect()
        }
        LogicalOperator::SetProperty(set) => {
            set.properties.iter().map(|(_, value)| value).collect()
        }
        LogicalOperator::Bind(bind) => vec![&bind.expression],
        LogicalOperator::Unwind(unwind) => vec![&unwind.expression],
        LogicalOperator::Merge(merge) => merge
            .match_properties
            .iter()
            .chain(merge.on_create.iter())
            .chain(merge.on_match.iter())
            .map(|(_, value)| value)
            .collect(),
        LogicalOperator::MergeRelationship(merge) => merge
            .match_properties
            .iter()
            .chain(merge.on_create.iter())
            .chain(merge.on_match.iter())
            .map(|(_, value)| value)
            .collect(),
        LogicalOperator::VectorScan(scan) => vec![&scan.query_vector],
        LogicalOperator::VectorJoin(join) => vec![&join.query_vector],
        LogicalOperator::TextScan(scan) => vec![&scan.query],
        LogicalOperator::CallProcedure(call) => call.arguments.iter().collect(),
        LogicalOperator::NodeScan(_)
        | LogicalOperator::EdgeScan(_)
        | LogicalOperator::TripleScan(_)
        | LogicalOperator::Limit(_)
        | LogicalOperator::Skip(_)
        | LogicalOperator::Distinct(_)
        | LogicalOperator::DeleteNode(_)
        | LogicalOperator::DeleteEdge(_)
        | LogicalOperator::Union(_)
        | LogicalOperator::MapCollect(_)
        | LogicalOperator::AddLabel(_)
        | LogicalOperator::RemoveLabel(_)
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
        | LogicalOperator::Empty
        | LogicalOperator::Except(_)
        | LogicalOperator::Intersect(_)
        | LogicalOperator::Otherwise(_)
        | LogicalOperator::Apply(_)
        | LogicalOperator::ParameterScan(_)
        | LogicalOperator::CreatePropertyGraph(_)
        | LogicalOperator::LoadData(_)
        | LogicalOperator::Construct(_) => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::DistinctOp;

    fn distinct_chain(depth: usize) -> LogicalPlan {
        let mut root = LogicalOperator::Empty;
        for _ in 1..depth {
            root = LogicalOperator::Distinct(DistinctOp {
                input: Box::new(root),
                columns: None,
            });
        }
        LogicalPlan::new(root)
    }

    #[test]
    fn the_limit_is_exact() {
        assert!(check_plan_depth(&distinct_chain(MAX_PLAN_DEPTH)).is_ok());
        assert!(check_plan_depth(&distinct_chain(MAX_PLAN_DEPTH + 1)).is_err());
    }

    #[test]
    fn expressions_count_towards_the_depth() {
        let mut predicate = LogicalExpression::Literal(true.into());
        for _ in 0..MAX_PLAN_DEPTH {
            predicate = LogicalExpression::Unary {
                op: crate::query::plan::UnaryOp::Not,
                operand: Box::new(predicate),
            };
        }
        let plan = LogicalPlan::new(LogicalOperator::Filter(crate::query::plan::FilterOp {
            predicate,
            input: Box::new(LogicalOperator::Empty),
            pushdown_hint: None,
        }));
        assert!(check_plan_depth(&plan).is_err());
    }

    #[test]
    fn a_rejected_plan_is_dropped_without_recursion() {
        // Dropping this chain recursively overflows a 2 MiB thread.
        std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(|| assert!(admit(distinct_chain(200_000)).is_err()))
            .unwrap()
            .join()
            .expect("admit must not overflow while dropping a rejected plan");
    }
}
