//! Aggregate and factorized aggregate planning.

use super::{
    AggregateOp, Arc, Direction, Error, ExpandDirection, ExpandOp, ExpandStep, ExpressionPredicate,
    FactorizedAggregate, FactorizedAggregateOperator, FilterExpression, FilterOperator,
    GraphStoreSearch, HashAggregateOperator, HashMap, JoinOp, JoinType,
    LazyFactorizedChainOperator, LogicalAggregateFunction, LogicalExpression, LogicalOperator,
    LogicalType, NodeScanOp, Operator, PhysicalAggregateExpr, ProjectExpr, ProjectOperator, Result,
    SimpleAggregateOperator, TriangleCountOperator, convert_aggregate_function,
    expression_to_string, resolved_column_name,
};

impl super::Planner {
    /// Plans an AGGREGATE operator.
    pub(super) fn plan_aggregate(
        &self,
        agg: &AggregateOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Check if we can use factorized aggregation for speedup
        // Conditions:
        // 1. Factorized execution is enabled
        // 2. Input is an expand chain (multi-hop)
        // 3. No GROUP BY
        // 4. All aggregates are simple (COUNT, SUM, AVG, MIN, MAX)
        if self.is_triangle_count_agg(agg)
            && self.triangle_count_binding_is_nonnull(agg, &agg.input)
            && let Some((scan, types, dest_label)) = self.triangle_count_scan(&agg.input)
        {
            return self.plan_triangle_count(scan, types, dest_label, agg);
        }
        // A triangle close is not a 3-hop walk. Factorized COUNT would
        // rebind the start variable and over-count (8×2³ = 64 on the
        // circulant pin that has 0 directed cycles).
        let triangle_shape = self.is_directed_triangle_chain(&agg.input)
            || matches!(&*agg.input, LogicalOperator::Join(j) if Self::triangle_join_hops(j).is_some());
        if self.factorized_execution
            && !triangle_shape
            && agg.group_by.is_empty()
            && Self::count_expand_chain(&agg.input).0 >= 2
            && Self::qualified_expand_chain(&agg.input).is_some()
            && self.is_simple_aggregate(agg)
            && let Ok((op, cols)) = self.plan_factorized_aggregate(agg)
        {
            return Ok((op, cols));
        }
        // Fall through to regular aggregate if factorized planning fails

        let (mut input_op, input_columns) =
            self.plan_operator_preserving_expand_constraints(&agg.input)?;

        // Build variable to column index mapping
        let mut variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Collect all extra projections (property access and complex expressions)
        // in a single ordered list so that column index assignment matches the
        // order they are added to the ProjectOperator.
        enum ExtraProjection {
            Property { variable: String, property: String },
            Expression { filter_expr: FilterExpression },
        }
        let mut extra_projections: Vec<ExtraProjection> = Vec::new();
        let mut next_col_idx = input_columns.len();

        // Check group-by expressions for properties and complex expressions
        // (Labels, Type, FunctionCall, IndexAccess, etc.)
        for expr in &agg.group_by {
            match expr {
                LogicalExpression::Property { variable, property } => {
                    let col_name = resolved_column_name(expr);
                    if !variable_columns.contains_key(&col_name) {
                        extra_projections.push(ExtraProjection::Property {
                            variable: variable.clone(),
                            property: property.clone(),
                        });
                        variable_columns.insert(col_name, next_col_idx);
                        next_col_idx += 1;
                    }
                }
                LogicalExpression::Variable(_) => {
                    // Already in variable_columns, nothing to project
                }
                _ => {
                    // Complex expression (Labels, Type, FunctionCall, IndexAccess,
                    // CASE, Binary, etc.): project as computed column
                    let col_name = resolved_column_name(expr);
                    if !variable_columns.contains_key(&col_name) {
                        let filter_expr = self.convert_expression(expr)?;
                        extra_projections.push(ExtraProjection::Expression { filter_expr });
                        variable_columns.insert(col_name, next_col_idx);
                        next_col_idx += 1;
                    }
                }
            }
        }

        // Check aggregate expressions for properties and complex expressions
        // (both arguments and an optional independent DISTINCT key)
        for agg_expr in &agg.aggregates {
            for expr_opt in [
                &agg_expr.expression,
                &agg_expr.expression2,
                &agg_expr.distinct_key,
            ] {
                let Some(expr) = expr_opt else { continue };
                match expr {
                    LogicalExpression::Property { variable, property } => {
                        let col_name = resolved_column_name(expr);
                        if !variable_columns.contains_key(&col_name) {
                            extra_projections.push(ExtraProjection::Property {
                                variable: variable.clone(),
                                property: property.clone(),
                            });
                            variable_columns.insert(col_name, next_col_idx);
                            next_col_idx += 1;
                        }
                    }
                    LogicalExpression::Variable(_) => {
                        // Already in variable_columns, nothing to project
                    }
                    _ => {
                        // Complex expression (CASE, Binary, etc.): project as computed column
                        let col_name = resolved_column_name(expr);
                        if !variable_columns.contains_key(&col_name) {
                            let filter_expr = self.convert_expression(expr)?;
                            extra_projections.push(ExtraProjection::Expression { filter_expr });
                            variable_columns.insert(col_name, next_col_idx);
                            next_col_idx += 1;
                        }
                    }
                }
            }
        }

        // If we have extra projections, add a projection to materialize them
        if !extra_projections.is_empty() {
            let mut projections = Vec::new();
            let mut output_types = Vec::new();

            // First, pass through all existing columns (use Node type to preserve node IDs
            // for subsequent property access - nodes need VectorData::NodeId for get_node_id())
            for (i, _) in input_columns.iter().enumerate() {
                projections.push(ProjectExpr::Column(i));
                output_types.push(LogicalType::Node);
            }

            // Add extra projections in the same order as index assignment
            for proj in &extra_projections {
                match proj {
                    ExtraProjection::Property {
                        variable, property, ..
                    } => {
                        let source_col = *variable_columns.get(variable).ok_or_else(|| {
                            Error::Internal(format!(
                                "Variable '{}' not found for property projection",
                                variable
                            ))
                        })?;
                        projections.push(ProjectExpr::PropertyAccess {
                            column: source_col,
                            property: property.clone(),
                        });
                        output_types.push(LogicalType::Any);
                    }
                    ExtraProjection::Expression { filter_expr, .. } => {
                        projections.push(ProjectExpr::Expression {
                            expr: filter_expr.clone(),
                            variable_columns: variable_columns.clone(),
                        });
                        output_types.push(LogicalType::Any);
                    }
                }
            }

            input_op = Box::new(
                ProjectOperator::with_store(
                    input_op,
                    projections,
                    output_types,
                    Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone()),
            );
        }

        // Convert group-by expressions to column indices
        let group_columns: Vec<usize> = agg
            .group_by
            .iter()
            .map(|expr| self.resolve_expression_to_column_with_properties(expr, &variable_columns))
            .collect::<Result<Vec<_>>>()?;

        // Convert aggregate expressions to physical form
        let physical_aggregates: Vec<PhysicalAggregateExpr> = agg
            .aggregates
            .iter()
            .map(|agg_expr| {
                let column = agg_expr
                    .expression
                    .as_ref()
                    .map(|e| {
                        self.resolve_expression_to_column_with_properties(e, &variable_columns)
                    })
                    .transpose()?;

                let column2 = agg_expr
                    .expression2
                    .as_ref()
                    .map(|e| {
                        self.resolve_expression_to_column_with_properties(e, &variable_columns)
                    })
                    .transpose()?;

                let distinct_key_column = agg_expr
                    .distinct_key
                    .as_ref()
                    .map(|e| {
                        self.resolve_expression_to_column_with_properties(e, &variable_columns)
                    })
                    .transpose()?;

                Ok(PhysicalAggregateExpr {
                    function: convert_aggregate_function(agg_expr.function),
                    column,
                    column2,
                    distinct_key_column,
                    distinct: agg_expr.distinct,
                    alias: agg_expr.alias.clone(),
                    percentile: agg_expr.percentile,
                    separator: agg_expr.separator.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Build output schema and column names
        let mut output_schema = Vec::new();
        let mut output_columns = Vec::new();

        // Add group-by columns
        for expr in &agg.group_by {
            output_schema.push(LogicalType::Any); // Group-by values can be any type
            output_columns.push(expression_to_string(expr));
        }

        // Add aggregate result columns
        for agg_expr in &agg.aggregates {
            let result_type = match agg_expr.function {
                LogicalAggregateFunction::Count | LogicalAggregateFunction::CountNonNull => {
                    LogicalType::Int64
                }
                LogicalAggregateFunction::Sum => LogicalType::Any,
                LogicalAggregateFunction::Avg => LogicalType::Float64,
                LogicalAggregateFunction::Min | LogicalAggregateFunction::Max => {
                    // MIN/MAX preserve input type: the result can be any type
                    // (Int64, Float64, String, Date, etc.), so use Any/Generic
                    // to avoid type mismatch when pushing the finalized value.
                    LogicalType::Any
                }
                LogicalAggregateFunction::Collect => LogicalType::Any, // List type (using Any since List is a complex type)
                LogicalAggregateFunction::GroupConcat => LogicalType::String,
                LogicalAggregateFunction::Sample => LogicalType::Any,
                // Statistical functions return Float64
                LogicalAggregateFunction::StdDev
                | LogicalAggregateFunction::StdDevPop
                | LogicalAggregateFunction::Variance
                | LogicalAggregateFunction::VariancePop
                | LogicalAggregateFunction::PercentileDisc
                | LogicalAggregateFunction::PercentileCont
                | LogicalAggregateFunction::CovarSamp
                | LogicalAggregateFunction::CovarPop
                | LogicalAggregateFunction::Corr
                | LogicalAggregateFunction::RegrSlope
                | LogicalAggregateFunction::RegrIntercept
                | LogicalAggregateFunction::RegrR2
                | LogicalAggregateFunction::RegrSxx
                | LogicalAggregateFunction::RegrSyy
                | LogicalAggregateFunction::RegrSxy
                | LogicalAggregateFunction::RegrAvgx
                | LogicalAggregateFunction::RegrAvgy => LogicalType::Float64,
                // REGR_COUNT returns Int64
                LogicalAggregateFunction::RegrCount => LogicalType::Int64,
            };
            output_schema.push(result_type);
            output_columns.push(
                agg_expr
                    .alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}(...)", agg_expr.function).to_lowercase()),
            );
        }

        // Register all aggregate output columns as scalar (group-by values and
        // aggregate results are materialized scalar values, not entity references)
        for col in &output_columns {
            self.scalar_columns.borrow_mut().insert(col.clone());
        }

        // Choose operator based on whether there are group-by columns
        let mut operator: Box<dyn Operator> = if group_columns.is_empty() {
            Box::new(SimpleAggregateOperator::new(
                input_op,
                physical_aggregates,
                output_schema,
            ))
        } else {
            Box::new(HashAggregateOperator::new(
                input_op,
                group_columns,
                physical_aggregates,
                output_schema,
            ))
        };

        // Apply HAVING clause filter if present
        if let Some(having_expr) = &agg.having {
            // Build variable to column mapping for the aggregate output
            let having_var_columns: HashMap<String, usize> = output_columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();

            let filter_expr = self.convert_expression(having_expr)?;
            let predicate = ExpressionPredicate::new(
                filter_expr,
                having_var_columns,
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone());
            operator = Box::new(FilterOperator::new(operator, Box::new(predicate)));
        }

        Ok((operator, output_columns))
    }

    /// COUNT(*) / COUNT(var) with no GROUP BY / HAVING / DISTINCT.
    /// On a triangle every binding is non-null, so COUNT(a) = COUNT(*).
    fn is_triangle_count_agg(&self, agg: &AggregateOp) -> bool {
        agg.group_by.is_empty()
            && agg.having.is_none()
            && agg.aggregates.len() == 1
            && !agg.aggregates[0].distinct
            && agg.aggregates[0].distinct_key.is_none()
            && matches!(
                agg.aggregates[0].function,
                LogicalAggregateFunction::Count | LogicalAggregateFunction::CountNonNull
            )
            && matches!(
                &agg.aggregates[0].expression,
                None | Some(LogicalExpression::Variable(_))
            )
    }

    /// Start scan + shared edge types + optional dest label for COUNT.
    fn triangle_count_scan(
        &self,
        op: &LogicalOperator,
    ) -> Option<(LogicalOperator, Vec<String>, Option<String>)> {
        let hops = if self.is_directed_triangle_chain(op)
            && matches!(Self::qualified_expand_chain(op), Some((3, _)))
        {
            let expands = Self::collect_expand_chain(op);
            [expands[0], expands[1], expands[2]]
        } else if let LogicalOperator::Join(join) = op {
            Self::triangle_join_hops(join)?
        } else {
            return None;
        };
        if hops
            .iter()
            .any(|h| h.min_hops != 1 || h.max_hops != Some(1))
        {
            return None;
        }
        if hops
            .iter()
            .any(|h| !matches!(h.direction, ExpandDirection::Outgoing))
        {
            return None;
        }
        let types = hops[0].edge_types.clone();
        if hops.iter().any(|h| h.edge_types != types) {
            return None;
        }
        Some(((*hops[0].input).clone(), types, None))
    }

    /// A COUNT(variable) shortcut is valid only when the variable is produced
    /// by one of the fixed-width triangle hops. Other variables may be null or
    /// absent from the input and must use ordinary aggregate semantics.
    fn triangle_count_binding_is_nonnull(
        &self,
        agg: &AggregateOp,
        input: &LogicalOperator,
    ) -> bool {
        let Some(LogicalExpression::Variable(variable)) = &agg.aggregates[0].expression else {
            return true;
        };
        let hops: Vec<&ExpandOp> = if self.is_directed_triangle_chain(input)
            && matches!(Self::qualified_expand_chain(input), Some((3, _)))
        {
            Self::collect_expand_chain(input)
        } else if let LogicalOperator::Join(join) = input {
            match Self::triangle_join_hops(join) {
                Some(hops) => hops.into_iter().collect(),
                None => return false,
            }
        } else {
            return false;
        };
        hops.iter().any(|hop| {
            hop.from_variable == *variable
                || hop.to_variable == *variable
                || hop.edge_variable.as_deref() == Some(variable.as_str())
        })
    }

    fn triangle_join_hops(join: &JoinOp) -> Option<[&ExpandOp; 3]> {
        if !matches!(join.join_type, JoinType::Inner) {
            return None;
        }
        let left_shape = Self::qualified_expand_chain(&join.left);
        let right_shape = Self::qualified_expand_chain(&join.right);
        let (two_logical, close_logical) = match (left_shape, right_shape) {
            (Some((2, _)), Some((1, close_scan))) if close_scan.label.is_none() => {
                (&*join.left, &*join.right)
            }
            (Some((1, close_scan)), Some((2, _))) if close_scan.label.is_none() => {
                (&*join.right, &*join.left)
            }
            _ => return None,
        };
        let two = Self::collect_expand_chain(two_logical);
        let close = Self::collect_expand_chain(close_logical)[0];
        if two[1].from_variable != two[0].to_variable {
            return None;
        }
        if close.from_variable != two[1].to_variable || close.to_variable != two[0].from_variable {
            return None;
        }
        if !Self::has_exact_same_name_value_conditions(
            join,
            [&two[0].from_variable, &two[1].to_variable],
        ) {
            return None;
        }
        Some([two[0], two[1], close])
    }

    fn plan_triangle_count(
        &self,
        scan: LogicalOperator,
        types: Vec<String>,
        dest_label: Option<String>,
        agg: &AggregateOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let count_all = match &scan {
            LogicalOperator::NodeScan(NodeScanOp {
                input: None,
                label: None,
                ..
            }) => {
                // The native count-all call reads the current store directly;
                // it is equivalent to this scan only at the current epoch,
                // without a transaction overlay, and with no label filter.
                self.transaction_id.is_none() && self.viewing_epoch == self.store.current_epoch()
            }
            _ => false,
        };
        let (input_op, _) = self.plan_operator(&scan)?;
        let mut op = TriangleCountOperator::new(
            Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            input_op,
            types,
        )
        .with_dest_label(dest_label)
        .with_count_all(count_all)
        .with_read_only(self.read_only);
        if let Some(transaction_id) = self.transaction_id {
            op = op.with_transaction_context(self.viewing_epoch, Some(transaction_id));
        } else {
            op = op.with_transaction_context(self.viewing_epoch, None);
        }
        let name = agg.aggregates[0]
            .alias
            .clone()
            .unwrap_or_else(|| format!("{:?}(...)", agg.aggregates[0].function).to_lowercase());
        self.scalar_columns.borrow_mut().insert(name.clone());
        Ok((Box::new(op), vec![name]))
    }

    /// Checks if an aggregate is simple enough for factorized execution.
    ///
    /// Simple aggregates:
    /// - COUNT(*) or COUNT(variable)
    /// - SUM, AVG, MIN, MAX on variables (not properties for now)
    pub(super) fn is_simple_aggregate(&self, agg: &AggregateOp) -> bool {
        agg.aggregates.iter().all(|agg_expr| {
            if agg_expr.distinct || agg_expr.distinct_key.is_some() {
                return false;
            }
            match agg_expr.function {
                LogicalAggregateFunction::Count | LogicalAggregateFunction::CountNonNull => {
                    // COUNT(*) is always OK, COUNT(var) is OK
                    agg_expr.expression.is_none()
                        || matches!(&agg_expr.expression, Some(LogicalExpression::Variable(_)))
                }
                LogicalAggregateFunction::Sum
                | LogicalAggregateFunction::Avg
                | LogicalAggregateFunction::Min
                | LogicalAggregateFunction::Max => {
                    // For now, only support when expression is a variable
                    // (property access would require flattening first)
                    matches!(&agg_expr.expression, Some(LogicalExpression::Variable(_)))
                }
                // Other aggregates (Collect, StdDev, Percentile) not supported in factorized form
                _ => false,
            }
        })
    }

    /// Plans a factorized aggregate that operates directly on factorized data.
    ///
    /// This avoids the O(n²) cost of flattening before aggregation.
    pub(super) fn plan_factorized_aggregate(
        &self,
        agg: &AggregateOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Build the expand chain - this returns a LazyFactorizedChainOperator
        let expands = Self::collect_expand_chain(&agg.input);
        if expands.is_empty() {
            return Err(Error::Internal(
                "Expected expand chain for factorized aggregate".to_string(),
            ));
        }

        // Get the base operator (before first expand)
        let first_expand = expands[0];
        let (base_op, base_columns) = self.plan_operator(&first_expand.input)?;

        let mut columns = base_columns.clone();
        let mut steps = Vec::new();
        let mut is_first = true;
        let last_idx = expands.len().saturating_sub(1);

        for (i, expand) in expands.iter().enumerate() {
            // Find source column for this expand
            let source_column = if is_first {
                base_columns
                    .iter()
                    .position(|c| c == &expand.from_variable)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "Source variable '{}' not found in base columns",
                            expand.from_variable
                        ))
                    })?
            } else {
                1 // Target from previous level
            };

            let direction = match expand.direction {
                ExpandDirection::Outgoing => Direction::Outgoing,
                ExpandDirection::Incoming => Direction::Incoming,
                ExpandDirection::Both => Direction::Both,
            };

            let need_edge = i != last_idx || expand.edge_variable.is_some();
            steps.push(ExpandStep {
                source_column,
                direction,
                edge_types: expand.edge_types.clone(),
                sip: None,
                need_edge,
            });

            if need_edge {
                let edge_col_name = self.register_edge_column(&expand.edge_variable);
                columns.push(edge_col_name);
            }
            columns.push(expand.to_variable.clone());

            is_first = false;
        }

        // Create the lazy factorized chain operator
        let mut lazy_op = LazyFactorizedChainOperator::new(
            Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            base_op,
            steps,
        )
        .with_read_only(self.read_only);

        if let Some(transaction_id) = self.transaction_id {
            lazy_op = lazy_op.with_transaction_context(self.viewing_epoch, Some(transaction_id));
        } else {
            lazy_op = lazy_op.with_transaction_context(self.viewing_epoch, None);
        }

        // Convert logical aggregates to factorized aggregates
        let factorized_aggs: Vec<FactorizedAggregate> = agg
            .aggregates
            .iter()
            .map(|agg_expr| {
                match agg_expr.function {
                    LogicalAggregateFunction::Count | LogicalAggregateFunction::CountNonNull => {
                        // COUNT(hop-target) is COUNT(*) on this chain: every
                        // produced neighbor is non-null. Avoid CountColumn's
                        // multiplicity walk.
                        FactorizedAggregate::count()
                    }
                    LogicalAggregateFunction::Sum => {
                        // SUM on deepest level target
                        FactorizedAggregate::sum(1)
                    }
                    LogicalAggregateFunction::Avg => FactorizedAggregate::avg(1),
                    LogicalAggregateFunction::Min => FactorizedAggregate::min(1),
                    LogicalAggregateFunction::Max => FactorizedAggregate::max(1),
                    _ => {
                        // Shouldn't reach here due to is_simple_aggregate check
                        FactorizedAggregate::count()
                    }
                }
            })
            .collect();

        // Build output column names
        let output_columns: Vec<String> = agg
            .aggregates
            .iter()
            .map(|agg_expr| {
                agg_expr
                    .alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}(...)", agg_expr.function).to_lowercase())
            })
            .collect();

        // Register output columns as scalar (aggregate results are materialized
        // scalar values, not entity references). Without this, a post-Return
        // projection would treat them as node IDs and attempt NodeResolve, which
        // corrupts the result on 3+ hop queries.
        for col in &output_columns {
            self.scalar_columns.borrow_mut().insert(col.clone());
        }

        // Create the factorized aggregate operator
        let factorized_agg_op = FactorizedAggregateOperator::new(lazy_op, factorized_aggs);

        Ok((Box::new(factorized_agg_op), output_columns))
    }

    /// Resolves a logical expression to a column index, using projected property columns.
    ///
    /// This is used for aggregations where properties have been projected into their own columns.
    pub(super) fn resolve_expression_to_column_with_properties(
        &self,
        expr: &LogicalExpression,
        variable_columns: &HashMap<String, usize>,
    ) -> Result<usize> {
        crate::query::planner::common::resolve_expression_to_column(expr, variable_columns, "")
    }
}

#[cfg(all(test, feature = "lpg"))]
mod shortcut_qualification_tests {
    use super::super::{
        AggregateOp, ExpandDirection, ExpandOp, FilterOp, GraphStoreSearch, JoinOp, JoinType,
        LogicalAggregateFunction, LogicalExpression, LogicalOperator, LogicalPlan, NodeScanOp,
        PathMode, Planner,
    };
    use crate::query::plan::{AggregateExpr, JoinCondition, JoinKeySemantics};
    use grafeo_common::types::Value;
    use grafeo_core::graph::lpg::LpgStore;
    use std::sync::Arc;

    fn scan(variable: &str, label: Option<&str>) -> LogicalOperator {
        LogicalOperator::NodeScan(NodeScanOp {
            variable: variable.to_string(),
            label: label.map(str::to_string),
            input: None,
        })
    }

    fn expand(from: &str, to: &str, input: LogicalOperator) -> LogicalOperator {
        LogicalOperator::Expand(ExpandOp {
            from_variable: from.to_string(),
            to_variable: to.to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["R".to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(input),
            path_alias: None,
            path_mode: PathMode::Walk,
            edge_predicate: None,
            path_predicate: None,
            path_search: crate::query::plan::PathSearch::All,
        })
    }

    fn triangle_chain() -> LogicalOperator {
        expand(
            "c",
            "a",
            expand("b", "c", expand("a", "b", scan("a", Some("V")))),
        )
    }

    fn has_label(variable: &str, label: &str, input: LogicalOperator) -> LogicalOperator {
        LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::FunctionCall {
                name: "hasLabel".to_string(),
                args: vec![
                    LogicalExpression::Variable(variable.to_string()),
                    LogicalExpression::Literal(Value::String(label.into())),
                ],
                distinct: false,
            },
            input: Box::new(input),
            pushdown_hint: None,
        })
    }

    fn count_star(input: LogicalOperator) -> LogicalOperator {
        count_expression(input, None)
    }

    fn count_variable(input: LogicalOperator, variable: &str) -> LogicalOperator {
        count_expression(
            input,
            Some(LogicalExpression::Variable(variable.to_string())),
        )
    }

    fn count_expression(
        input: LogicalOperator,
        expression: Option<LogicalExpression>,
    ) -> LogicalOperator {
        LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![],
            aggregates: vec![AggregateExpr {
                function: LogicalAggregateFunction::Count,
                expression,
                expression2: None,
                distinct_key: None,
                distinct: false,
                alias: Some("n".to_string()),
                percentile: None,
                separator: None,
            }],
            input: Box::new(input),
            having: None,
        })
    }

    #[test]
    fn triangle_count_default_column_matches_general_aggregation() {
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let fallback = expand(
            "c",
            "a",
            expand(
                "b",
                "c",
                has_label("b", "V", expand("a", "b", scan("a", Some("V")))),
            ),
        );
        for input in [triangle_chain(), fallback] {
            let mut aggregate = count_star(input);
            let LogicalOperator::Aggregate(op) = &mut aggregate else {
                unreachable!()
            };
            op.aggregates[0].alias = None;
            let physical = Planner::new(Arc::clone(&store))
                .plan(&LogicalPlan::new(aggregate))
                .expect("optimized and general aggregates must satisfy the same output contract");
            assert_eq!(physical.columns(), &["count(...)".to_string()]);
        }
    }

    #[test]
    fn triangle_count_does_not_convert_count_of_unrelated_variable() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["V"]);
        let b = store.create_node(&["V"]);
        let c = store.create_node(&["V"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, c, "R");
        store.create_edge(c, a, "R");

        // `x` is not a binding produced by the triangle. The specialized
        // COUNT(*) kernel must not silently turn COUNT(x) into COUNT(*).
        let result = Planner::new(store as Arc<dyn GraphStoreSearch>)
            .plan(&LogicalPlan::new(count_variable(triangle_chain(), "x")));
        assert!(
            result.is_err(),
            "unbound COUNT variable must stay semantic error"
        );
    }

    #[test]
    fn triangle_count_historical_scan_does_not_use_current_count_fast_path() {
        use crate::transaction::TransactionManager;
        use grafeo_common::types::{EpochId, TransactionId};
        use grafeo_core::graph::GraphStoreMut;

        let store = Arc::new(LpgStore::new().unwrap());
        let old_first = store.create_node(&["V"]);
        let old_second = store.create_node(&["V"]);
        let old_third = store.create_node(&["V"]);
        store.create_edge(old_first, old_second, "R");
        store.create_edge(old_second, old_third, "R");
        store.create_edge(old_third, old_first, "R");
        let old_epoch = store.current_epoch();

        let future = EpochId::new(old_epoch.as_u64() + 10);
        let future_first = store.create_node_versioned(&["V"], future, TransactionId::SYSTEM);
        let future_second = store.create_node_versioned(&["V"], future, TransactionId::SYSTEM);
        let future_third = store.create_node_versioned(&["V"], future, TransactionId::SYSTEM);
        store.create_edge_versioned(
            future_first,
            future_second,
            "R",
            future,
            TransactionId::SYSTEM,
        );
        store.create_edge_versioned(
            future_second,
            future_third,
            "R",
            future,
            TransactionId::SYSTEM,
        );
        store.create_edge_versioned(
            future_third,
            future_first,
            "R",
            future,
            TransactionId::SYSTEM,
        );
        store.sync_epoch(future);

        let manager = Arc::new(TransactionManager::new());
        let planner = Planner::with_context(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            Some(Arc::clone(&store) as Arc<dyn GraphStoreMut>),
            manager,
            None,
            old_epoch,
        );
        let mut planned = planner
            .plan(&LogicalPlan::new(count_star(triangle_chain())))
            .unwrap();
        let row = planned.operator.next().unwrap().unwrap();
        assert_eq!(row.column(0).unwrap().get_int64(0), Some(3));
    }

    fn execute_count(planner: Planner, input: LogicalOperator) -> (String, i64) {
        let mut planned = planner.plan(&LogicalPlan::new(count_star(input))).unwrap();
        let name = planned.operator.name().to_string();
        let chunk = planned
            .operator
            .next()
            .unwrap()
            .expect("COUNT must produce one row");
        (name, chunk.column(0).unwrap().get_int64(0).unwrap())
    }

    #[test]
    fn factorized_aggregate_does_not_discard_intermediate_haslabel() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["A"]);
        let keep_b = store.create_node(&["Keep"]);
        let drop_b = store.create_node(&["Drop"]);
        let keep_c = store.create_node(&["C"]);
        let drop_c = store.create_node(&["C"]);
        store.create_edge(a, keep_b, "R");
        store.create_edge(a, drop_b, "R");
        store.create_edge(keep_b, keep_c, "R");
        store.create_edge(drop_b, drop_c, "R");

        let chain = expand(
            "b",
            "c",
            has_label("b", "Keep", expand("a", "b", scan("a", Some("A")))),
        );
        let (name, count) = execute_count(Planner::new(store as Arc<dyn GraphStoreSearch>), chain);

        assert_ne!(name, "FactorizedAggregate");
        assert_eq!(count, 1);
    }

    #[test]
    fn aggregate_fallback_preserves_haslabel_at_any_chain_depth() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["A"]);
        let keep_b = store.create_node(&["Keep"]);
        let drop_b = store.create_node(&["Drop"]);
        let keep_c = store.create_node(&["C"]);
        let drop_c = store.create_node(&["C"]);
        let keep_d = store.create_node(&["D"]);
        let drop_d = store.create_node(&["D"]);
        let keep_e = store.create_node(&["E"]);
        let drop_e = store.create_node(&["E"]);
        store.create_edge(a, keep_b, "R");
        store.create_edge(a, drop_b, "R");
        store.create_edge(keep_b, keep_c, "R");
        store.create_edge(drop_b, drop_c, "R");
        store.create_edge(keep_c, keep_d, "R");
        store.create_edge(drop_c, drop_d, "R");
        store.create_edge(keep_d, keep_e, "R");
        store.create_edge(drop_d, drop_e, "R");

        let chain = expand(
            "d",
            "e",
            expand(
                "c",
                "d",
                expand(
                    "b",
                    "c",
                    has_label("b", "Keep", expand("a", "b", scan("a", Some("A")))),
                ),
            ),
        );
        let (_, flat_count) = execute_count(
            Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>)
                .with_factorized_execution(false),
            chain.clone(),
        );
        let (name, count) = execute_count(Planner::new(store as Arc<dyn GraphStoreSearch>), chain);

        assert_ne!(name, "FactorizedAggregate");
        assert_eq!(flat_count, 1, "the unfused plan is the semantic oracle");
        assert_eq!(count, flat_count);
    }

    #[test]
    fn factorized_aggregate_does_not_discard_root_haslabel() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);
        let keep = store.create_node(&["Keep"]);
        let drop = store.create_node(&["Drop"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, keep, "R");
        store.create_edge(b, drop, "R");

        let chain = has_label(
            "c",
            "Keep",
            expand("b", "c", expand("a", "b", scan("a", Some("A")))),
        );
        let (name, count) = execute_count(Planner::new(store as Arc<dyn GraphStoreSearch>), chain);

        assert_ne!(name, "FactorizedAggregate");
        assert_eq!(count, 1);
    }

    #[test]
    fn factorized_aggregate_preserves_base_node_scan_label() {
        let store = Arc::new(LpgStore::new().unwrap());
        let keep_a = store.create_node(&["A"]);
        let drop_a = store.create_node(&["Drop"]);
        let keep_b = store.create_node(&["B"]);
        let drop_b = store.create_node(&["B"]);
        let keep_c = store.create_node(&["C"]);
        let drop_c = store.create_node(&["C"]);
        store.create_edge(keep_a, keep_b, "R");
        store.create_edge(drop_a, drop_b, "R");
        store.create_edge(keep_b, keep_c, "R");
        store.create_edge(drop_b, drop_c, "R");

        let chain = expand("b", "c", expand("a", "b", scan("a", Some("A"))));
        let (name, count) = execute_count(Planner::new(store as Arc<dyn GraphStoreSearch>), chain);

        assert_eq!(name, "FactorizedAggregate");
        assert_eq!(count, 1);
    }

    #[test]
    fn triangle_count_does_not_discard_close_scan_label() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["V"]);
        let b = store.create_node(&["V"]);
        let c = store.create_node(&["V"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, c, "R");
        store.create_edge(c, a, "R");

        let logical = LogicalOperator::Join(JoinOp {
            left: Box::new(expand("b", "c", expand("a", "b", scan("a", None)))),
            right: Box::new(expand("c", "a", scan("c", Some("Required")))),
            join_type: JoinType::Inner,
            conditions: vec![
                JoinCondition {
                    left: LogicalExpression::Variable("a".to_string()),
                    right: LogicalExpression::Variable("a".to_string()),
                    semantics: JoinKeySemantics::Value,
                },
                JoinCondition {
                    left: LogicalExpression::Variable("c".to_string()),
                    right: LogicalExpression::Variable("c".to_string()),
                    semantics: JoinKeySemantics::Value,
                },
            ],
        });
        let (name, count) =
            execute_count(Planner::new(store as Arc<dyn GraphStoreSearch>), logical);

        assert_ne!(name, "TriangleCount");
        assert_eq!(count, 0);
    }

    #[test]
    fn triangle_count_does_not_misapply_variable_specific_haslabel() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Start"]);
        let b = store.create_node(&["V"]);
        let c = store.create_node(&["Required"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, c, "R");
        store.create_edge(c, a, "R");

        let chain = expand(
            "c",
            "a",
            expand(
                "b",
                "c",
                has_label("a", "Required", expand("a", "b", scan("a", Some("Start")))),
            ),
        );
        let (_, flat_count) = execute_count(
            Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>)
                .with_factorized_execution(false),
            chain.clone(),
        );
        let (name, count) = execute_count(Planner::new(store as Arc<dyn GraphStoreSearch>), chain);

        assert_ne!(name, "TriangleCount");
        assert_eq!(flat_count, 0, "the unfused plan is the semantic oracle");
        assert_eq!(count, flat_count);
    }
}
