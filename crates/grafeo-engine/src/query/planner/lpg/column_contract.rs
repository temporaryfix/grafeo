//! Debug-only physical output obligations, respecting logical scope boundaries.
//!
//! Required names are a subset: physical plans may carry additional internal
//! columns. Scope replacement is explicit, so an unprojected path must exist at
//! its Expand boundary but need not survive RETURN or aggregation.

use super::{
    Error, LogicalExpression, LogicalOperator, Planner, Result, expression_to_string,
    output_column_name,
};
use crate::query::binder::path_binding_names;
use crate::query::plan::JoinType;

impl Planner {
    pub(super) fn validate_output_bindings(
        &self,
        op: &LogicalOperator,
        columns: &[String],
    ) -> Result<()> {
        let parameters = self
            .correlated_param_state
            .borrow()
            .as_ref()
            .map(|state| state.columns.clone());
        let required = required_bindings(op, parameters.as_deref())?;
        let missing: Vec<_> = required
            .iter()
            .filter(|name| !columns.contains(name))
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        Err(Error::Internal(format!(
            "Physical output binding contract violated for {:?}: missing {missing:?}; columns {columns:?}",
            std::mem::discriminant(op),
        )))
    }
}

fn required_bindings(op: &LogicalOperator, parameters: Option<&[String]>) -> Result<Vec<String>> {
    let child = |op: &LogicalOperator| required_bindings(op, parameters);
    let optional = |op: Option<&LogicalOperator>| op.map_or_else(|| Ok(Vec::new()), child);
    let append = |mut names: Vec<String>, name: &str| {
        names.push(name.to_owned());
        names
    };
    let union = |mut left: Vec<String>, right: Vec<String>| {
        left.extend(right);
        left
    };
    Ok(match op {
        LogicalOperator::Empty => Vec::new(),
        LogicalOperator::NodeScan(op) => append(optional(op.input.as_deref())?, &op.variable),
        LogicalOperator::EdgeScan(op) => append(optional(op.input.as_deref())?, &op.variable),
        LogicalOperator::Expand(op) => {
            let mut names = append(child(&op.input)?, &op.to_variable);
            names.extend(op.edge_variable.iter().cloned());
            if let Some(alias) = &op.path_alias {
                names.extend(path_binding_names(alias));
            }
            names
        }
        LogicalOperator::Filter(op) => child(&op.input)?,
        LogicalOperator::Sort(op) => child(&op.input)?,
        LogicalOperator::Limit(op) => child(&op.input)?,
        LogicalOperator::Skip(op) => {
            // The shared FINISH sentinel drains its child and deliberately
            // exposes no columns; ordinary SKIP preserves the child schema.
            if op.count.value() == usize::MAX {
                Vec::new()
            } else {
                child(&op.input)?
            }
        }
        LogicalOperator::Distinct(op) => child(&op.input)?,
        LogicalOperator::DeleteNode(op) => child(&op.input)?,
        LogicalOperator::DeleteEdge(op) => child(&op.input)?,
        LogicalOperator::SetProperty(op) => child(&op.input)?,
        LogicalOperator::AddLabel(op) => child(&op.input)?,
        LogicalOperator::RemoveLabel(op) => child(&op.input)?,
        LogicalOperator::CreateNode(op) => append(optional(op.input.as_deref())?, &op.variable),
        LogicalOperator::CreateEdge(op) => {
            let mut names = child(&op.input)?;
            names.extend(op.variable.iter().cloned());
            names
        }
        LogicalOperator::Merge(op) => append(child(&op.input)?, &op.variable),
        LogicalOperator::MergeRelationship(op) => append(child(&op.input)?, &op.variable),
        LogicalOperator::Unwind(op) => {
            let mut names = append(child(&op.input)?, &op.variable);
            names.extend(op.ordinality_var.iter().cloned());
            names.extend(op.offset_var.iter().cloned());
            names
        }
        LogicalOperator::Project(op) => {
            let mut names = if op.pass_through_input {
                child(&op.input)?
            } else {
                Vec::new()
            };
            names.extend(
                op.projections
                    .iter()
                    .map(|item| output_column_name(item.alias.as_deref(), &item.expression)),
            );
            names
        }
        LogicalOperator::Return(op) => {
            if op.items.len() == 1
                && matches!(&op.items[0].expression, LogicalExpression::Variable(name) if name == "*")
            {
                child(&op.input)?
                    .into_iter()
                    .filter(|name| !name.starts_with('_'))
                    .collect()
            } else {
                op.items
                    .iter()
                    .map(|item| output_column_name(item.alias.as_deref(), &item.expression))
                    .collect()
            }
        }
        LogicalOperator::Aggregate(op) => {
            let mut names: Vec<_> = op.group_by.iter().map(expression_to_string).collect();
            names.extend(op.aggregates.iter().map(|agg| {
                agg.alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}(...)", agg.function).to_lowercase())
            }));
            names
        }
        LogicalOperator::HorizontalAggregate(op) => append(child(&op.input)?, &op.alias),
        LogicalOperator::MapCollect(op) => vec![op.alias.clone()],
        LogicalOperator::Join(op) => {
            let left = child(&op.left)?;
            if matches!(op.join_type, JoinType::Semi | JoinType::Anti) {
                left
            } else {
                union(left, child(&op.right)?)
            }
        }
        LogicalOperator::LeftJoin(op) => union(child(&op.left)?, child(&op.right)?),
        LogicalOperator::AntiJoin(op) => child(&op.left)?,
        LogicalOperator::Except(op) => child(&op.left)?,
        LogicalOperator::Intersect(op) => child(&op.left)?,
        LogicalOperator::Otherwise(op) => child(&op.left)?,
        LogicalOperator::Union(op) => optional(op.inputs.first())?,
        LogicalOperator::MultiWayJoin(op) => {
            let mut names = Vec::new();
            for input in &op.inputs {
                names.extend(child(input)?);
            }
            names
        }
        LogicalOperator::Apply(op) => {
            let outer = child(&op.input)?;
            let imported = if op.shared_variables.as_slice() == ["*"] {
                outer.clone()
            } else {
                op.shared_variables.clone()
            };
            // The live planner state has already been restored at this boundary.
            // Reconstruct this Apply's imports for its own nested scope only.
            let inner = required_bindings(&op.subplan, Some(&imported))?;
            union(outer, inner)
        }
        LogicalOperator::ParameterScan(op) => {
            if let Some(parameters) = parameters {
                parameters.to_vec()
            } else if op.columns.iter().any(|name| name == "*") {
                return Err(Error::Internal(
                    "ParameterScan wildcard has no binding context".into(),
                ));
            } else {
                op.columns.clone()
            }
        }
        LogicalOperator::VectorScan(op) => append(optional(op.input.as_deref())?, &op.variable),
        LogicalOperator::TextScan(op) => {
            let mut names = vec![op.variable.clone()];
            names.extend(op.score_column.iter().cloned());
            names
        }
        LogicalOperator::CallProcedure(op) => {
            // Without YIELD the binder declares no variable names. Runtime
            // procedure result metadata may still add physical columns.
            op.yield_items
                .iter()
                .flatten()
                .map(|item| item.alias.as_ref().unwrap_or(&item.field_name).clone())
                .collect()
        }
        LogicalOperator::LoadData(op) => vec![op.variable.clone()],
        // These have no successful LPG lowering; never silently exempt them
        // if a future planner starts producing physical output for them.
        LogicalOperator::TripleScan(_)
        | LogicalOperator::Construct(_)
        | LogicalOperator::Bind(_)
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
        | LogicalOperator::VectorJoin(_)
        | LogicalOperator::CreatePropertyGraph(_) => {
            return Err(Error::Internal(format!(
                "No LPG output binding contract for {:?}",
                std::mem::discriminant(op)
            )));
        }
    })
}

#[cfg(test)]
mod tests;
