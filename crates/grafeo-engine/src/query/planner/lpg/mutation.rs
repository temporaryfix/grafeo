//! Mutation planning (CREATE, DELETE, SET, MERGE, CALL, labels).

use super::{
    AddLabelOp, AddLabelOperator, AntiJoinOp, AntiJoinSemantics, Arc, CreateEdgeOp,
    CreateEdgeOperator, CreateNodeOp, CreateNodeOperator, DeleteEdgeOp, DeleteEdgeOperator,
    DeleteNodeOp, DeleteNodeOperator, Error, ExpressionPredicate, FilterOperator, HashMap,
    JoinKeySemantics, LeftJoinOp, LogicalExpression, LogicalOperator, LogicalType, MergeConfig,
    MergeOp, MergeOperator, MergeRelationshipConfig, MergeRelationshipOp,
    MergeRelationshipOperator, Operator, ProjectExpr, ProjectOperator, PropertySource,
    RemoveLabelOp, RemoveLabelOperator, Result, SetPropertyOp, SetPropertyOperator, UnaryOp,
    UnwindOp, UnwindOperator, Value,
};
#[cfg(any(feature = "lpg", feature = "algos"))]
use super::{CallProcedureOp, StaticResultOperator, TransactionId};

impl super::Planner {
    /// Plans property inputs for a CREATE operator.
    ///
    /// Simple property sources stay attached to the mutation operator. Runtime
    /// expressions are evaluated in a projection over the original input
    /// scope, then passed to the mutation as ordinary columns. The projection
    /// columns are private to the physical CREATE and are removed by the
    /// caller before its output is exposed downstream.
    fn plan_create_property_sources(
        &self,
        input: Option<Box<dyn Operator>>,
        input_columns: &[String],
        create_properties: &[(String, LogicalExpression)],
    ) -> Result<(
        Option<Box<dyn Operator>>,
        Vec<(String, PropertySource)>,
        Vec<LogicalType>,
    )> {
        let mut properties = Vec::with_capacity(create_properties.len());
        let mut projection_exprs: Vec<ProjectExpr> =
            (0..input_columns.len()).map(ProjectExpr::Column).collect();
        let mut projection_schema = self.derive_schema_from_columns(input_columns);
        let mut needs_projection = false;

        for (name, expr) in create_properties {
            let source = match self.expression_to_property_source(expr, input_columns) {
                Ok(source) => source,
                Err(_) => {
                    if let Some(value) = Self::try_fold_expression(expr) {
                        PropertySource::Constant(value)
                    } else {
                        let Some(_) = input.as_ref() else {
                            return Err(Error::Internal(format!(
                                "Cannot resolve CREATE expression for property '{name}': \
                                 variable not in scope or unsupported expression"
                            )));
                        };
                        let filter_expr = self.convert_expression(expr)?;
                        let col_idx = projection_schema.len();
                        let variable_columns: HashMap<String, usize> = input_columns
                            .iter()
                            .enumerate()
                            .map(|(index, column)| (column.clone(), index))
                            .collect();
                        projection_exprs.push(ProjectExpr::Expression {
                            expr: filter_expr,
                            variable_columns,
                        });
                        // Scratch values have no binding name or entity type.
                        projection_schema.push(LogicalType::Any);
                        needs_projection = true;
                        PropertySource::Column(col_idx)
                    }
                }
            };
            properties.push((name.clone(), source));
        }

        let input = if needs_projection {
            let input = input.ok_or_else(|| {
                Error::Internal("CREATE expression projection has no input".to_string())
            })?;
            Some(Box::new(
                ProjectOperator::with_store(
                    input,
                    projection_exprs,
                    projection_schema.clone(),
                    Arc::clone(&self.store),
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone()),
            ) as Box<dyn Operator>)
        } else {
            input
        };

        Ok((input, properties, projection_schema))
    }

    /// Removes the private expression columns added around a CREATE operator.
    fn hide_create_projection_columns(
        &self,
        operator: Box<dyn Operator>,
        input_column_count: usize,
        created_column: Option<usize>,
        output_schema: Vec<LogicalType>,
        projected: bool,
    ) -> Box<dyn Operator> {
        if !projected {
            return operator;
        }

        let mut projections: Vec<ProjectExpr> =
            (0..input_column_count).map(ProjectExpr::Column).collect();
        if let Some(created_column) = created_column {
            projections.push(ProjectExpr::Column(created_column));
        }
        Box::new(ProjectOperator::new(operator, projections, output_schema))
    }

    /// Plans a CREATE NODE operator.
    pub(super) fn plan_create_node(
        &self,
        create: &CreateNodeOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan input if present
        let (input_op, input_columns) = if let Some(ref input) = create.input {
            let (op, cols) = self.plan_operator(input)?;
            (Some(op), cols)
        } else {
            (None, vec![])
        };

        // If the variable already exists in input columns and no labels/properties
        // are specified, this is a reference to an existing node (e.g., from MATCH).
        // Skip creating a new node and just pass through.
        if input_columns.contains(&create.variable)
            && create.labels.is_empty()
            && create.properties.is_empty()
            && let Some(op) = input_op
        {
            return Ok((op, input_columns));
        }

        let (input_op, properties, mut output_schema) =
            self.plan_create_property_sources(input_op, &input_columns, &create.properties)?;
        let projected = output_schema.len() != input_columns.len();
        let output_column = output_schema.len();

        // Input pass-through columns use generic types (Any); the new node column
        // gets Node for compact VectorData::NodeId storage.
        output_schema.push(LogicalType::Node);

        let mut op = CreateNodeOperator::new(
            self.write_store()?,
            input_op,
            create.labels.clone(),
            properties,
            output_schema,
            output_column,
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id);

        if let Some(ref tracker) = self.write_tracker {
            op = op.with_write_tracker(Arc::clone(tracker));
        }
        if let Some(ref validator) = self.validator {
            op = op.with_validator(Arc::clone(validator));
        }

        let mut output_columns = input_columns.clone();
        output_columns.push(create.variable.clone());
        let mut public_schema = self.derive_schema_from_columns(&input_columns);
        public_schema.push(LogicalType::Node);
        let operator = self.hide_create_projection_columns(
            Box::new(op),
            input_columns.len(),
            Some(output_column),
            public_schema,
            projected,
        );
        Ok((operator, output_columns))
    }

    /// Plans a CREATE EDGE operator.
    pub(super) fn plan_create_edge(
        &self,
        create: &CreateEdgeOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, input_columns) = self.plan_operator(&create.input)?;

        // Find source and target columns
        let from_column = input_columns
            .iter()
            .position(|c| c == &create.from_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Source variable '{}' not found",
                    create.from_variable
                ))
            })?;

        let to_column = input_columns
            .iter()
            .position(|c| c == &create.to_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Target variable '{}' not found",
                    create.to_variable
                ))
            })?;

        let (input_op, properties, mut output_schema) =
            self.plan_create_property_sources(Some(input_op), &input_columns, &create.properties)?;
        let projected = output_schema.len() != input_columns.len();
        let property_column_count = output_schema.len();
        let output_column = create.variable.as_ref().map(|variable| {
            self.edge_columns.borrow_mut().insert(variable.clone());
            property_column_count
        });
        if create.variable.is_some() {
            output_schema.push(LogicalType::Edge);
        }

        let mut operator = CreateEdgeOperator::new(
            self.write_store()?,
            input_op.ok_or_else(|| Error::Internal("CREATE EDGE has no input operator".into()))?,
            from_column,
            to_column,
            create.edge_type.clone(),
            output_schema,
        )
        .with_properties(properties)
        .with_transaction_context(self.viewing_epoch, self.transaction_id);

        if let Some(ref tracker) = self.write_tracker {
            operator = operator.with_write_tracker(Arc::clone(tracker));
        }
        if let Some(col) = output_column {
            operator = operator.with_output_column(col);
        }
        if let Some(ref validator) = self.validator {
            operator = operator.with_validator(Arc::clone(validator));
        }

        let mut output_columns = input_columns.clone();
        let mut public_schema = self.derive_schema_from_columns(&input_columns);
        if let Some(variable) = &create.variable {
            output_columns.push(variable.clone());
            public_schema.push(LogicalType::Edge);
        }
        let operator = self.hide_create_projection_columns(
            Box::new(operator),
            input_columns.len(),
            output_column,
            public_schema,
            projected,
        );

        Ok((operator, output_columns))
    }

    /// Plans a DELETE NODE operator.
    ///
    /// If the variable is tracked as an edge (via `edge_columns`), this
    /// automatically delegates to [`DeleteEdgeOperator`] instead.
    pub(super) fn plan_delete_node(
        &self,
        delete: &DeleteNodeOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&delete.input)?;

        let col_idx = columns
            .iter()
            .position(|c| c == &delete.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for delete",
                    delete.variable
                ))
            })?;

        // Preserve input columns so downstream RETURN/aggregate can reference
        // the deleted variable (e.g., DETACH DELETE n RETURN count(n)).
        let output_schema = self.derive_schema_from_columns(&columns);
        let output_columns = columns.clone();

        // Auto-detect edge variables and use the correct operator
        let is_edge = self.edge_columns.borrow().contains(&delete.variable);

        if is_edge {
            let mut op =
                DeleteEdgeOperator::new(self.write_store()?, input_op, col_idx, output_schema)
                    .with_transaction_context(self.viewing_epoch, self.transaction_id);
            if let Some(ref tracker) = self.write_tracker {
                op = op.with_write_tracker(Arc::clone(tracker));
            }
            Ok((Box::new(op), output_columns))
        } else {
            let mut op = DeleteNodeOperator::new(
                self.write_store()?,
                input_op,
                col_idx,
                output_schema,
                delete.detach,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id);
            if let Some(ref tracker) = self.write_tracker {
                op = op.with_write_tracker(Arc::clone(tracker));
            }
            if let Some(ref validator) = self.validator {
                op = op.with_validator(Arc::clone(validator));
            }
            Ok((Box::new(op), output_columns))
        }
    }

    /// Plans a DELETE EDGE operator.
    pub(super) fn plan_delete_edge(
        &self,
        delete: &DeleteEdgeOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&delete.input)?;

        let edge_column = columns
            .iter()
            .position(|c| c == &delete.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for delete",
                    delete.variable
                ))
            })?;

        // Preserve input columns so downstream clauses can reference the
        // deleted variable (same pass-through pattern as delete_node).
        let output_schema = self.derive_schema_from_columns(&columns);
        let output_columns = columns.clone();

        let mut op =
            DeleteEdgeOperator::new(self.write_store()?, input_op, edge_column, output_schema)
                .with_transaction_context(self.viewing_epoch, self.transaction_id);
        if let Some(ref tracker) = self.write_tracker {
            op = op.with_write_tracker(Arc::clone(tracker));
        }
        Ok((Box::new(op), output_columns))
    }

    /// Plans a LEFT JOIN operator (for OPTIONAL MATCH).
    pub(super) fn plan_left_join(
        &self,
        left_join: &LeftJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        if left_join
            .compatibility_conditions
            .iter()
            .any(|condition| condition.semantics != JoinKeySemantics::Value)
        {
            return Err(Error::InvalidValue(
                "RDF term-identity left-join metadata cannot be planned by the LPG engine"
                    .to_string(),
            ));
        }
        // Handle Empty left input (OPTIONAL MATCH as first clause):
        // substitute a SingleRowOperator so the left side produces one row.
        let (left_op, left_columns): (Box<dyn Operator>, Vec<String>) =
            if matches!(left_join.left.as_ref(), LogicalOperator::Empty) {
                let single_row: Box<dyn Operator> = Box::new(
                    grafeo_core::execution::operators::single_row::SingleRowOperator::new(),
                );
                (single_row, Vec::new())
            } else {
                self.plan_operator(&left_join.left)?
            };
        let (right_op, right_columns) = self.plan_operator(&left_join.right)?;
        let left_types = self.derive_schema_from_columns(&left_columns);
        let right_types = self.derive_schema_from_columns(&right_columns);
        let explicit_keys = if left_join.compatibility_conditions.is_empty() {
            None
        } else {
            Some(
                left_join
                    .compatibility_conditions
                    .iter()
                    .map(|condition| {
                        Ok((
                            self.expression_to_column(&condition.left, &left_columns)?,
                            self.expression_to_column(&condition.right, &right_columns)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .unzip::<_, _, Vec<_>, Vec<_>>(),
            )
        };
        let (join_op, join_columns, _join_types) =
            if let Some((left_keys, right_keys)) = explicit_keys {
                super::common::build_left_join_with_keys(
                    left_op,
                    right_op,
                    &left_columns,
                    &right_columns,
                    &left_types,
                    &right_types,
                    left_keys,
                    right_keys,
                )
            } else {
                super::common::build_left_join(
                    left_op,
                    right_op,
                    &left_columns,
                    &right_columns,
                    &left_types,
                    &right_types,
                )
            };

        // If the LeftJoin carries a cross-side condition (null-safe predicate),
        // apply it as a Filter above the join. The condition already incorporates
        // IS NULL guards so NULL-padded rows from unmatched optional sides pass through.
        if let Some(condition) = &left_join.condition {
            let filter_expr = self.convert_expression(condition)?;
            let variable_columns: HashMap<String, usize> = join_columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();
            let predicate =
                ExpressionPredicate::new(filter_expr, variable_columns, Arc::clone(&self.store))
                    .with_transaction_context(self.viewing_epoch, self.transaction_id)
                    .with_session_context(self.session_context.clone());
            let filter_op: Box<dyn Operator> =
                Box::new(FilterOperator::new(join_op, Box::new(predicate)));
            return Ok((filter_op, join_columns));
        }

        Ok((join_op, join_columns))
    }

    /// Plans an ANTI JOIN operator (for WHERE NOT EXISTS patterns).
    pub(super) fn plan_anti_join(
        &self,
        anti_join: &AntiJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        if anti_join
            .compatibility_conditions
            .iter()
            .any(|condition| condition.semantics != JoinKeySemantics::Value)
        {
            return Err(Error::InvalidValue(
                "RDF term-identity anti-join metadata cannot be planned by the LPG engine"
                    .to_string(),
            ));
        }
        let (left_op, left_columns) = self.plan_operator(&anti_join.left)?;
        let (right_op, right_columns) = self.plan_operator(&anti_join.right)?;
        let schema = self.derive_schema_from_columns(&left_columns);
        if anti_join.compatibility_conditions.is_empty() {
            if anti_join.semantics == AntiJoinSemantics::NotExists {
                return Ok(super::common::build_anti_join_with_keys(
                    left_op,
                    right_op,
                    left_columns,
                    schema,
                    Vec::new(),
                    Vec::new(),
                    false,
                ));
            }
            return Ok(super::common::build_anti_join(
                left_op,
                right_op,
                left_columns,
                &right_columns,
                schema,
            ));
        }
        let (left_keys, right_keys) = anti_join
            .compatibility_conditions
            .iter()
            .map(|condition| {
                Ok((
                    self.expression_to_column(&condition.left, &left_columns)?,
                    self.expression_to_column(&condition.right, &right_columns)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip();
        Ok(super::common::build_anti_join_with_keys(
            left_op,
            right_op,
            left_columns,
            schema,
            left_keys,
            right_keys,
            anti_join.semantics == AntiJoinSemantics::Minus,
        ))
    }

    /// Plans an unwind operator.
    pub(super) fn plan_unwind(
        &self,
        unwind: &UnwindOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan the input operator first
        // Handle Empty specially - use a single-row operator
        let input_was_empty = matches!(&*unwind.input, LogicalOperator::Empty);
        let (input_op, input_columns): (Box<dyn Operator>, Vec<String>) = if input_was_empty {
            // For UNWIND without prior MATCH, create a single-row input
            // We need an operator that produces one row with the list to unwind
            // For now, use EmptyScan which produces no rows - we'll handle the literal
            // list in the unwind operator itself
            let literal_list = self.convert_expression(&unwind.expression)?;

            // Create a project operator that produces a single row with the list
            let single_row_op: Box<dyn Operator> =
                Box::new(grafeo_core::execution::operators::single_row::SingleRowOperator::new());
            let project_op: Box<dyn Operator> = Box::new(
                ProjectOperator::with_store(
                    single_row_op,
                    vec![ProjectExpr::Expression {
                        expr: literal_list,
                        variable_columns: HashMap::new(),
                    }],
                    vec![LogicalType::Any],
                    Arc::clone(&self.store),
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone()),
            );

            // The logical tree still contains Unwind(Empty), so under
            // PROFILE, build_profile_tree will walk into Empty and expect
            // an entry. plan_operator(&Empty) returns Err and is bypassed
            // here, so push a synthetic entry attributed to Empty.
            if self.records_entries() {
                let (entry, _stats) = crate::query::profile::ProfileEntry::new(
                    "Empty",
                    LogicalOperator::Empty.display_label(),
                );
                self.profile_entries.borrow_mut().push(entry);
            }
            (project_op, vec!["__list__".to_string()])
        } else {
            self.plan_operator(&unwind.input)?
        };

        // The UNWIND expression should be a list - we need to find/evaluate it
        // Handle variable references, property access, and literal lists

        // Find if the expression references an existing column that is itself a list
        let list_col_idx = match &unwind.expression {
            LogicalExpression::Variable(var) => input_columns.iter().position(|c| c == var),
            LogicalExpression::FunctionCall { .. } if input_was_empty => Some(0),
            LogicalExpression::List(_) | LogicalExpression::Literal(_) => {
                // Literal list expression - needs to be added as a column
                None
            }
            _ => None,
        };

        // When the expression needs runtime evaluation (property access, literal list, etc.),
        // wrap input in a ProjectOperator that computes the list as an extra column.
        let (final_input_op, final_input_columns, col_idx) = if let Some(idx) = list_col_idx {
            (input_op, input_columns, idx)
        } else if matches!(
            &unwind.expression,
            LogicalExpression::List(_)
                | LogicalExpression::Literal(Value::List(_))
                | LogicalExpression::Literal(Value::Vector(_))
                | LogicalExpression::Property { .. }
                | LogicalExpression::FunctionCall { .. }
        ) {
            // Wrap input in a ProjectOperator that adds the list as an extra column
            let literal_list = self.convert_expression(&unwind.expression)?;
            let mut proj_exprs: Vec<ProjectExpr> =
                (0..input_columns.len()).map(ProjectExpr::Column).collect();
            let var_cols: HashMap<String, usize> = input_columns
                .iter()
                .enumerate()
                .map(|(i, c)| (c.clone(), i))
                .collect();
            proj_exprs.push(ProjectExpr::Expression {
                expr: literal_list,
                variable_columns: var_cols,
            });
            let mut proj_schema = self.derive_schema_from_columns(&input_columns);
            proj_schema.push(LogicalType::Any);
            let project_op: Box<dyn Operator> = Box::new(
                ProjectOperator::with_store(
                    input_op,
                    proj_exprs,
                    proj_schema,
                    Arc::clone(&self.store),
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone()),
            );
            let list_col = input_columns.len();
            let mut cols = input_columns;
            cols.push("__unwind_list__".to_string());
            (project_op, cols, list_col)
        } else {
            // Fallback: assume column 0 contains the list
            (input_op, input_columns, 0)
        };

        // Build output columns: all input columns plus the new variable
        let mut columns = final_input_columns.clone();
        columns.push(unwind.variable.clone());

        // Mark the UNWIND variable as scalar (not a node/edge ID) so that
        // plan_return uses LogicalType::Any instead of Node for it.
        self.scalar_columns
            .borrow_mut()
            .insert(unwind.variable.clone());

        // Build output schema
        let mut output_schema = self.derive_schema_from_columns(&final_input_columns);
        output_schema.push(LogicalType::Any); // The unwound element type is dynamic

        // Add ORDINALITY column (1-based index) if requested
        let emit_ordinality = unwind.ordinality_var.is_some();
        if let Some(ref ord_var) = unwind.ordinality_var {
            columns.push(ord_var.clone());
            output_schema.push(LogicalType::Int64);
            self.scalar_columns.borrow_mut().insert(ord_var.clone());
        }

        // Add OFFSET column (0-based index) if requested
        let emit_offset = unwind.offset_var.is_some();
        if let Some(ref off_var) = unwind.offset_var {
            columns.push(off_var.clone());
            output_schema.push(LogicalType::Int64);
            self.scalar_columns.borrow_mut().insert(off_var.clone());
        }

        let operator: Box<dyn Operator> = Box::new(UnwindOperator::new(
            final_input_op,
            col_idx,
            unwind.variable.clone(),
            output_schema,
            emit_ordinality,
            emit_offset,
        ));

        Ok((operator, columns))
    }

    /// Plans a MERGE operator.
    pub(super) fn plan_merge(&self, merge: &MergeOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan the input operator if present (skip if Empty)
        let (input_op, mut columns) = if matches!(merge.input.as_ref(), LogicalOperator::Empty) {
            (None, Vec::new())
        } else {
            let (op, cols) = self.plan_operator(&merge.input)?;
            (Some(op), cols)
        };

        // Match properties cannot reference the MERGE variable (ISO §15.5).
        let match_properties: Vec<(String, PropertySource)> = merge
            .match_properties
            .iter()
            .map(|(name, expr)| {
                let source = self
                    .expression_to_property_source(expr, &columns)
                    .unwrap_or_else(|_| {
                        Self::try_fold_expression(expr).map_or(
                            PropertySource::Constant(Value::Null),
                            PropertySource::Constant,
                        )
                    });
                (name.clone(), source)
            })
            .collect();

        // ON CREATE / ON MATCH expressions are evaluated against an augmented row
        // that includes the merged node. Build the action-scope columns now so
        // `coalesce(n.x, 0)` and similar expressions can resolve `n`.
        let mut action_scope_columns = columns.clone();
        action_scope_columns.push(merge.variable.clone());

        let on_create_properties: Vec<(String, PropertySource)> = merge
            .on_create
            .iter()
            .map(|(name, expr)| {
                let source = self.merge_action_property_source(expr, &action_scope_columns)?;
                Ok::<_, Error>((name.clone(), source))
            })
            .collect::<Result<Vec<_>>>()?;

        let on_match_properties: Vec<(String, PropertySource)> = merge
            .on_match
            .iter()
            .map(|(name, expr)| {
                let source = self.merge_action_property_source(expr, &action_scope_columns)?;
                Ok::<_, Error>((name.clone(), source))
            })
            .collect::<Result<Vec<_>>>()?;

        // Detect if the merge variable is already bound from the input.
        // If so, record its column index for NULL-reference checking at runtime.
        let bound_variable_column = columns.iter().position(|c| c == &merge.variable);

        // Column index for the merged node ID in the output
        let output_column = columns.len();
        columns.push(merge.variable.clone());

        // Build output schema: type-aware pass-through for input columns,
        // Node for the newly-added merge variable column.
        let input_cols = &columns[..output_column];
        let mut output_schema = self.derive_schema_from_columns(input_cols);
        output_schema.push(LogicalType::Node);

        let mut merge_op = MergeOperator::new(
            self.write_store()?,
            input_op,
            MergeConfig {
                variable: merge.variable.clone(),
                labels: merge.labels.clone(),
                match_properties,
                on_create_properties,
                on_match_properties,
                output_schema,
                output_column,
                bound_variable_column,
            },
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id)
        .with_search_store(Arc::clone(&self.store))
        .with_session_context(self.session_context.clone());

        if let Some(ref validator) = self.validator {
            merge_op = merge_op.with_validator(Arc::clone(validator));
        }

        let operator: Box<dyn Operator> = Box::new(merge_op);

        Ok((operator, columns))
    }

    /// Plans a MERGE RELATIONSHIP operator.
    pub(super) fn plan_merge_relationship(
        &self,
        merge_rel: &MergeRelationshipOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, mut columns) = self.plan_operator(&merge_rel.input)?;

        // Find source and target node columns
        let source_column = columns
            .iter()
            .position(|c| c == &merge_rel.source_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Source variable '{}' not found for MERGE relationship",
                    merge_rel.source_variable
                ))
            })?;

        let target_column = columns
            .iter()
            .position(|c| c == &merge_rel.target_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Target variable '{}' not found for MERGE relationship",
                    merge_rel.target_variable
                ))
            })?;

        // Convert match properties to PropertySource (supports variables from input)
        let match_properties: Vec<(String, PropertySource)> = merge_rel
            .match_properties
            .iter()
            .map(|(name, expr)| {
                let source = self
                    .expression_to_property_source(expr, &columns)
                    .unwrap_or_else(|_| {
                        Self::try_fold_expression(expr).map_or(
                            PropertySource::Constant(Value::Null),
                            PropertySource::Constant,
                        )
                    });
                (name.clone(), source)
            })
            .collect();

        // ON CREATE / ON MATCH SET on a MERGE relationship may reference the
        // edge variable itself: build an augmented scope that includes it.
        let mut action_scope_columns = columns.clone();
        action_scope_columns.push(merge_rel.variable.clone());

        let on_create_properties: Vec<(String, PropertySource)> = merge_rel
            .on_create
            .iter()
            .map(|(name, expr)| {
                let source = self.merge_action_property_source(expr, &action_scope_columns)?;
                Ok::<_, Error>((name.clone(), source))
            })
            .collect::<Result<Vec<_>>>()?;

        let on_match_properties: Vec<(String, PropertySource)> = merge_rel
            .on_match
            .iter()
            .map(|(name, expr)| {
                let source = self.merge_action_property_source(expr, &action_scope_columns)?;
                Ok::<_, Error>((name.clone(), source))
            })
            .collect::<Result<Vec<_>>>()?;

        // Add the edge variable to output columns and track it as an edge
        let edge_output_column = columns.len();
        columns.push(merge_rel.variable.clone());
        self.edge_columns
            .borrow_mut()
            .insert(merge_rel.variable.clone());

        // Build output schema: type-aware pass-through for input columns,
        // Edge for the newly-added merge relationship column.
        let input_cols = &columns[..edge_output_column];
        let mut output_schema = self.derive_schema_from_columns(input_cols);
        output_schema.push(LogicalType::Edge);

        let config = MergeRelationshipConfig {
            source_column,
            target_column,
            source_variable: merge_rel.source_variable.clone(),
            target_variable: merge_rel.target_variable.clone(),
            edge_type: merge_rel.edge_type.clone(),
            match_properties,
            on_create_properties,
            on_match_properties,
            output_schema,
            edge_output_column,
        };

        let mut merge_rel_op =
            MergeRelationshipOperator::new(self.write_store()?, input_op, config)
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_search_store(Arc::clone(&self.store))
                .with_session_context(self.session_context.clone());

        if let Some(ref validator) = self.validator {
            merge_rel_op = merge_rel_op.with_validator(Arc::clone(validator));
        }

        let operator: Box<dyn Operator> = Box::new(merge_rel_op);

        Ok((operator, columns))
    }

    /// Plans a CALL procedure operator.
    #[cfg(any(feature = "lpg", feature = "algos"))]
    pub(super) fn plan_call_procedure(
        &self,
        call: &CallProcedureOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use crate::procedures::{self, builtin_registry};

        #[cfg(feature = "gql")]
        {
            use crate::query::procedure_effect::{ResolvedProcedure, analyze_procedure_effects};

            // Session supplies the immutable snapshot used for authorization.
            // Direct lower-level Planner callers still get the same canonical
            // resolution by taking a one-call snapshot here.
            let fallback;
            let procedures = if let Some(procedures) = &self.resolved_procedures {
                procedures
            } else if let Some(catalog) = &self.catalog {
                let root = crate::query::plan::LogicalOperator::CallProcedure(call.clone());
                fallback = analyze_procedure_effects(&root, catalog)?.procedures;
                &fallback
            } else {
                fallback = crate::query::procedure_effect::ResolvedProcedureCatalog::empty();
                &fallback
            };

            match procedures.resolve(call)? {
                ResolvedProcedure::Listing => {
                    let result = procedures::procedures_result(builtin_registry());
                    self.plan_static_result(result, &call.yield_items)
                }
                ResolvedProcedure::Catalog(procedure) => self.plan_user_procedure(call, &procedure),
                ResolvedProcedure::Builtin(procedure) => {
                    self.plan_builtin_procedure(call, procedure)
                }
            }
        }

        #[cfg(not(feature = "gql"))]
        {
            let resolved_name = call.name.join(".");
            if matches!(&call.name[..], [single] if single == "procedures")
                || matches!(&call.name[..], [namespace, procedure]
                    if namespace.eq_ignore_ascii_case("grafeo") && procedure == "procedures")
            {
                let result = procedures::procedures_result(builtin_registry());
                return self.plan_static_result(result, &call.yield_items);
            }
            let procedure = builtin_registry()
                .get(&call.name)
                .ok_or_else(|| Error::Internal(format!("Unknown procedure: '{resolved_name}'")))?;
            self.plan_builtin_procedure(call, procedure)
        }
    }

    #[cfg(any(feature = "lpg", feature = "algos"))]
    fn plan_builtin_procedure(
        &self,
        call: &CallProcedureOp,
        procedure: Arc<dyn crate::procedures::Procedure>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use crate::procedures::{self, ProcedureEffect};

        if procedure.effect() == ProcedureEffect::MayWrite {
            return Err(Error::Query(grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Unsupported,
                format!(
                    "write-capable builtin procedure '{}' is disabled until builtin mutation context carries Session WAL, CDC, validation, and conflict tracking",
                    procedure.name()
                ),
            )));
        }

        // Per-procedure Serializable guard: procedures that cannot record reads
        // for SSI (e.g. vector/text index searches) are rejected; procedures that
        // are snapshot-safe (graph algorithms, catalog introspection) are allowed.
        if self.is_serializable() && !procedure.serializable_safe() {
            return Err(Error::Internal(format!(
                "Serializable isolation is not yet supported with procedure '{}'; use SnapshotIsolation",
                procedure.name()
            )));
        }

        // Evaluate arguments to Parameters
        let params = procedures::evaluate_arguments(&call.arguments, procedure.parameters());

        // Canonical column names for this procedure (user-facing names)
        let canonical_columns = procedure.output_columns();

        // Determine output columns from YIELD or procedure defaults
        let yield_columns = call.yield_items.as_ref().map(|items| {
            items
                .iter()
                .map(|item| (item.field_name.clone(), item.alias.clone()))
                .collect::<Vec<_>>()
        });

        let output_columns = if let Some(yield_cols) = &yield_columns {
            yield_cols
                .iter()
                .map(|(name, alias)| alias.clone().unwrap_or_else(|| name.clone()))
                .collect()
        } else {
            canonical_columns.clone()
        };

        let mut op = crate::query::executor::procedure_call::ProcedureCallOperator::new(
            Arc::clone(&self.store),
            procedure,
            params,
            yield_columns,
            canonical_columns,
        );
        // Every CALL gets the statement's exact MVCC cut. Active transactions
        // keep their identity for read-your-writes and SSI; autocommit reads
        // use INVALID, which cannot see a transaction overlay and has no read
        // tracker. Lower-level callers receive the same logical cut; retaining
        // that cut across concurrent GC remains their responsibility.
        op = op.with_snapshot_context(
            self.viewing_epoch,
            self.transaction_id.unwrap_or(TransactionId::INVALID),
        );
        #[cfg(feature = "lpg")]
        if let Some(lpg_store) = self.lpg_store.as_ref() {
            op = op.with_lpg_store(Arc::clone(lpg_store));
        }
        let operator: Box<dyn Operator> = Box::new(op);

        // Procedure outputs are scalar values, not node/edge IDs
        for col in &output_columns {
            self.scalar_columns.borrow_mut().insert(col.clone());
        }

        Ok((operator, output_columns))
    }

    /// Plans a static result set (e.g., from `grafeo.procedures()`).
    #[cfg(any(feature = "lpg", feature = "algos"))]
    pub(super) fn plan_static_result(
        &self,
        result: grafeo_adapters::plugins::AlgorithmResult,
        yield_items: &Option<Vec<crate::query::plan::ProcedureYield>>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Determine output columns and column indices
        let (output_columns, column_indices) = if let Some(items) = yield_items {
            let mut cols = Vec::new();
            let mut indices = Vec::new();
            for item in items {
                let idx = result
                    .columns
                    .iter()
                    .position(|c| c == &item.field_name)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "YIELD column '{}' not found (available: {})",
                            item.field_name,
                            result.columns.join(", ")
                        ))
                    })?;
                indices.push(idx);
                cols.push(
                    item.alias
                        .clone()
                        .unwrap_or_else(|| item.field_name.clone()),
                );
            }
            (cols, indices)
        } else {
            let indices: Vec<usize> = (0..result.columns.len()).collect();
            (result.columns.clone(), indices)
        };

        let operator = Box::new(StaticResultOperator {
            rows: result.rows,
            column_indices,
            row_index: 0,
        });

        // Static result outputs are scalar values, not node/edge IDs
        for col in &output_columns {
            self.scalar_columns.borrow_mut().insert(col.clone());
        }

        Ok((operator, output_columns))
    }

    /// Plans a user-defined procedure call.
    #[cfg(all(any(feature = "lpg", feature = "algos"), feature = "gql"))]
    fn plan_user_procedure(
        &self,
        call: &CallProcedureOp,
        procedure: &crate::query::procedure_effect::ResolvedCatalogProcedure,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use grafeo_common::utils::error::{QueryError, QueryErrorKind, TransactionError};

        use crate::procedures::ProcedureEffect;
        use crate::query::binder::Binder;
        use crate::query::executor::procedure_output::{
            EagerProcedureBoundaryOperator, ProcedureOutputContractOperator,
            procedure_logical_type, procedure_value_matches,
        };
        use crate::query::optimizer::Optimizer;
        use crate::query::processor::{QueryParams, substitute_params};

        let proc_def = &procedure.definition;

        // Validate argument count
        if call.arguments.len() != proc_def.params.len() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!(
                    "procedure '{}' expects {} arguments, got {}",
                    proc_def.name,
                    proc_def.params.len(),
                    call.arguments.len()
                ),
            )));
        }

        // Evaluate and validate arguments as typed values. Runtime-dependent
        // expressions are not supported at this root CALL boundary and fail as
        // a user-facing semantic error rather than an internal engine error.
        let mut arg_values = Vec::with_capacity(call.arguments.len());
        for (((param_name, _), expected), arg) in proc_def
            .params
            .iter()
            .zip(&procedure.parameter_types)
            .zip(&call.arguments)
        {
            let val = crate::query::planner::eval_constant_expression(arg).map_err(|_| {
                Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!(
                        "procedure '{}' argument '{param_name}' must be a constant value",
                        proc_def.name
                    ),
                ))
            })?;
            if !procedure_value_matches(expected, &val) {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!(
                        "procedure '{}' argument '{param_name}' expects {expected}, found {}",
                        proc_def.name,
                        val.type_name()
                    ),
                )));
            }
            arg_values.push(val);
        }

        // Build a typed parameter map. Substitution happens in the parsed
        // logical tree; procedure arguments are never executable source text.
        let mut param_map = QueryParams::new();
        for (param, value) in proc_def.params.iter().zip(arg_values) {
            param_map.insert(param.0.clone(), value);
        }

        // A write-capable body is valid only inside the transaction that
        // Session opened after transitive authorization. This also makes the
        // lower-level QueryProcessor/Planner APIs fail closed when they do not
        // own transaction framing.
        if procedure.effect == ProcedureEffect::MayWrite
            && (!self.procedure_write_authority || self.transaction_id.is_none())
        {
            return Err(Error::Transaction(TransactionError::InvalidState(format!(
                "write-capable procedure '{}' requires a Session-owned transaction",
                proc_def.name
            ))));
        }

        let _frame = self.enter_procedure_frame(&proc_def.name)?;
        let mut logical_body = procedure.logical_body.clone();
        substitute_params(&mut logical_body, &param_map)?;

        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_body)?;
        let optimizer = Optimizer::from_graph_store(self.store.as_ref());
        let optimized_body = optimizer.optimize(logical_body)?;
        let child = self.fork_for_procedure();
        let (body_operator, body_columns) = child.plan_operator(&optimized_body.root)?;

        // Determine output columns
        let return_columns: Vec<String> = proc_def.returns.iter().map(|r| r.0.clone()).collect();

        let output_columns = if let Some(yield_items) = &call.yield_items {
            yield_items
                .iter()
                .map(|item| {
                    item.alias
                        .clone()
                        .unwrap_or_else(|| item.field_name.clone())
                })
                .collect()
        } else {
            return_columns.clone()
        };

        if body_columns.len() != return_columns.len() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!(
                    "procedure '{}' declares {} output columns, but its body returns {} ([{}])",
                    proc_def.name,
                    return_columns.len(),
                    body_columns.len(),
                    body_columns.join(", ")
                ),
            )));
        }
        let source_columns = if let Some(yield_items) = &call.yield_items {
            yield_items
                .iter()
                .map(|item| item.field_name.clone())
                .collect::<Vec<_>>()
        } else {
            return_columns.clone()
        };
        let contract_columns = proc_def
            .returns
            .iter()
            .zip(&procedure.return_types)
            .map(|((name, _), data_type)| (name.clone(), data_type.clone()))
            .collect();
        let mut body_operator: Box<dyn Operator> = Box::new(ProcedureOutputContractOperator::new(
            body_operator,
            proc_def.name.clone(),
            contract_columns,
        ));
        if procedure.effect == ProcedureEffect::MayWrite {
            body_operator = Box::new(EagerProcedureBoundaryOperator::new(
                body_operator,
                proc_def.name.clone(),
            ));
        }

        let mut seen = std::collections::HashSet::new();
        let mut projections = Vec::with_capacity(source_columns.len());
        let mut output_types = Vec::with_capacity(source_columns.len());
        for source in &source_columns {
            if !seen.insert(source) {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!(
                        "procedure '{}' requests duplicate output column '{source}'",
                        proc_def.name
                    ),
                )));
            }
            // RETURNS defines the public procedure schema. Body columns map to
            // that schema positionally, so a body may use local aliases while
            // YIELD consistently addresses the declared names.
            let index = return_columns
                .iter()
                .position(|column| column == source)
                .ok_or_else(|| {
                    Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!(
                            "procedure '{}' has no declared output '{source}' (available: [{}])",
                            proc_def.name,
                            return_columns.join(", ")
                        ),
                    ))
                })?;
            projections.push(ProjectExpr::Column(index));
            output_types.push(procedure_logical_type(&procedure.return_types[index]));
        }
        let operator: Box<dyn Operator> = Box::new(ProjectOperator::new(
            body_operator,
            projections,
            output_types,
        ));

        // Procedure outputs are scalar values, not node/edge IDs
        for col in &output_columns {
            self.scalar_columns.borrow_mut().insert(col.clone());
        }

        Ok((operator, output_columns))
    }

    /// Plans an ADD LABEL operator.
    pub(super) fn plan_add_label(
        &self,
        add_label: &AddLabelOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&add_label.input)?;

        // Find the node column
        let node_column = columns
            .iter()
            .position(|c| c == &add_label.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for ADD LABEL",
                    add_label.variable
                ))
            })?;

        // Preserve input columns (like SetPropertyOperator) and append update count
        let mut output_schema = self.derive_schema_from_columns(&columns);
        output_schema.push(LogicalType::Int64);
        let mut output_columns = columns.clone();
        output_columns.push("labels_added".to_string());

        let mut op = AddLabelOperator::new(
            self.write_store()?,
            input_op,
            node_column,
            add_label.labels.clone(),
            output_schema,
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id);
        if let Some(ref tracker) = self.write_tracker {
            op = op.with_write_tracker(Arc::clone(tracker));
        }
        if let Some(ref validator) = self.validator {
            op = op.with_validator(Arc::clone(validator));
        }

        Ok((Box::new(op), output_columns))
    }

    /// Plans a REMOVE LABEL operator.
    pub(super) fn plan_remove_label(
        &self,
        remove_label: &RemoveLabelOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&remove_label.input)?;

        // Find the node column
        let node_column = columns
            .iter()
            .position(|c| c == &remove_label.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for REMOVE LABEL",
                    remove_label.variable
                ))
            })?;

        // Preserve input columns (like SetPropertyOperator) and append update count
        let mut output_schema = self.derive_schema_from_columns(&columns);
        output_schema.push(LogicalType::Int64);
        let mut output_columns = columns.clone();
        output_columns.push("labels_removed".to_string());

        let mut op = RemoveLabelOperator::new(
            self.write_store()?,
            input_op,
            node_column,
            remove_label.labels.clone(),
            output_schema,
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id);
        if let Some(ref tracker) = self.write_tracker {
            op = op.with_write_tracker(Arc::clone(tracker));
        }
        if let Some(ref validator) = self.validator {
            op = op.with_validator(Arc::clone(validator));
        }

        Ok((Box::new(op), output_columns))
    }

    /// Plans a SET PROPERTY operator.
    pub(super) fn plan_set_property(
        &self,
        set_prop: &SetPropertyOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&set_prop.input)?;

        // Find the entity column (node or edge variable)
        let entity_column = columns
            .iter()
            .position(|c| c == &set_prop.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for SET",
                    set_prop.variable
                ))
            })?;

        // Convert properties to PropertySource (supports constants, variables, and
        // complex expressions like `c.value + 1`). Expressions that cannot be resolved
        // to a simple PropertySource are pre-computed via a projection operator.
        let mut properties: Vec<(String, PropertySource)> = Vec::new();
        let mut projection_exprs: Vec<ProjectExpr> = Vec::new();
        let mut projection_columns: Vec<String> = columns.clone();

        // Start with pass-through for all existing columns
        for i in 0..columns.len() {
            projection_exprs.push(ProjectExpr::Column(i));
        }

        let mut needs_projection = false;

        for (name, expr) in &set_prop.properties {
            let source = match self.expression_to_property_source(expr, &columns) {
                Ok(s) => s,
                Err(_) => {
                    // Fallback: try constant folding for complex expressions
                    // (e.g., vector([1,2,3]), date('2024-01-01')).
                    if let Some(v) = Self::try_fold_expression(expr) {
                        PropertySource::Constant(v)
                    } else {
                        // Complex runtime expression (e.g., c.value + 1): add a
                        // projection column that evaluates it, then SET from that column.
                        match self.convert_expression(expr) {
                            Ok(filter_expr) => {
                                let col_idx = projection_columns.len();
                                let col_name = format!("__set_expr_{name}");
                                let variable_columns: HashMap<String, usize> = columns
                                    .iter()
                                    .enumerate()
                                    .map(|(i, c)| (c.clone(), i))
                                    .collect();
                                projection_exprs.push(ProjectExpr::Expression {
                                    expr: filter_expr,
                                    variable_columns,
                                });
                                projection_columns.push(col_name);
                                needs_projection = true;
                                PropertySource::Column(col_idx)
                            }
                            Err(_) => {
                                return Err(Error::Internal(format!(
                                    "Cannot resolve SET expression for property '{name}': \
                                     variable not in scope or unsupported expression"
                                )));
                            }
                        }
                    }
                }
            };
            properties.push((name.clone(), source));
        }

        // If any SET expression needed runtime evaluation, wrap input in a projection.
        let actual_input: Box<dyn Operator> = if needs_projection {
            let proj_schema = self.derive_schema_from_columns(&projection_columns);
            Box::new(
                ProjectOperator::with_store(
                    input_op,
                    projection_exprs,
                    proj_schema,
                    Arc::clone(&self.store),
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone()),
            )
        } else {
            input_op
        };

        // Output schema: type-aware pass-through for input columns.
        let output_schema = self.derive_schema_from_columns(&columns);
        let output_columns = columns.clone();

        // Determine if this is a node or edge using tracked edge columns
        let is_edge = set_prop.is_edge || self.edge_columns.borrow().contains(&set_prop.variable);
        let operator: Box<dyn Operator> = if is_edge {
            let mut op = SetPropertyOperator::new_for_edge(
                self.write_store()?,
                actual_input,
                entity_column,
                properties,
                output_schema,
            )
            .with_replace(set_prop.replace)
            .with_transaction_context(self.viewing_epoch, self.transaction_id);
            if let Some(ref tracker) = self.write_tracker {
                op = op.with_write_tracker(Arc::clone(tracker));
            }
            if let Some(ref validator) = self.validator {
                op = op.with_validator(Arc::clone(validator));
            }
            Box::new(op)
        } else {
            let mut op = SetPropertyOperator::new_for_node(
                self.write_store()?,
                actual_input,
                entity_column,
                properties,
                output_schema,
            )
            .with_replace(set_prop.replace)
            .with_transaction_context(self.viewing_epoch, self.transaction_id);
            if let Some(ref tracker) = self.write_tracker {
                op = op.with_write_tracker(Arc::clone(tracker));
            }
            if let Some(ref validator) = self.validator {
                op = op.with_validator(Arc::clone(validator));
            }
            Box::new(op)
        };

        Ok((operator, output_columns))
    }

    /// Lowers an ON CREATE / ON MATCH SET expression for a MERGE clause.
    ///
    /// Resolution order:
    /// 1. Simple lowering against the augmented action scope (input columns +
    ///    the MERGE variable). Catches literals, plain variable refs, and
    ///    direct property access.
    /// 2. Constant folding for plan-time-evaluable expressions like
    ///    `vector([1,2,3])` or `date('2024-01-01')`.
    /// 3. Fall back to a runtime [`PropertySource::Expression`] carrying the
    ///    converted [`FilterExpression`] and the variable-column map. The
    ///    operator builds an augmented row containing the merged node/edge
    ///    and evaluates the expression via [`ExpressionPredicate`].
    pub(super) fn merge_action_property_source(
        &self,
        expr: &LogicalExpression,
        action_scope_columns: &[String],
    ) -> Result<PropertySource> {
        if let Ok(source) = self.expression_to_property_source(expr, action_scope_columns) {
            return Ok(source);
        }
        if let Some(value) = Self::try_fold_expression(expr) {
            return Ok(PropertySource::Constant(value));
        }
        let filter_expr = self.convert_expression(expr)?;
        // When the merge variable is already bound from input, it appears
        // twice in `action_scope_columns`. Collecting via `HashMap::insert`
        // keeps the LAST occurrence, which is the appended column matching
        // the operator's augmented row position. Do not switch to a
        // first-wins collector here.
        let variable_columns: HashMap<String, usize> = action_scope_columns
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();
        Ok(PropertySource::Expression {
            expr: Box::new(filter_expr),
            variable_columns,
        })
    }

    /// Converts a logical expression to a PropertySource.
    ///
    /// Variable resolution uses `rposition` rather than `position` so that
    /// when `columns` legitimately contains a duplicated variable name,
    /// references resolve to the most recently added (innermost / latest)
    /// column. The MERGE planner relies on this for ON CREATE / ON MATCH
    /// expressions whose action scope appends the merge variable on top of
    /// an input that may already bind it: the appended column is the one
    /// the operator's augmented row populates with the merged node/edge id,
    /// and resolving to the earlier bound copy would read the pre-merge
    /// (potentially stale) value instead. For unique columns the two are
    /// equivalent.
    pub(super) fn expression_to_property_source(
        &self,
        expr: &LogicalExpression,
        columns: &[String],
    ) -> Result<PropertySource> {
        match expr {
            LogicalExpression::Literal(value) => Ok(PropertySource::Constant(value.clone())),
            LogicalExpression::Variable(name) => {
                let col_idx = columns.iter().rposition(|c| c == name).ok_or_else(|| {
                    Error::Internal(format!("Variable '{}' not found for property source", name))
                })?;
                Ok(PropertySource::Column(col_idx))
            }
            LogicalExpression::Property { variable, property } => {
                let col_idx = columns.iter().rposition(|c| c == variable).ok_or_else(|| {
                    Error::Internal(format!(
                        "Variable '{}' not found for property access '{}.{}'",
                        variable, variable, property
                    ))
                })?;
                Ok(PropertySource::PropertyAccess {
                    column: col_idx,
                    property: property.clone(),
                })
            }
            LogicalExpression::Parameter(name) => {
                // Parameters should be resolved before planning
                // For now, treat as a placeholder
                Ok(PropertySource::Constant(
                    grafeo_common::types::Value::String(format!("${}", name).into()),
                ))
            }
            _ => {
                if let Some(value) = Self::try_fold_expression(expr) {
                    Ok(PropertySource::Constant(value))
                } else {
                    Err(Error::Internal(format!(
                        "Unsupported expression type for property source: {:?}",
                        expr
                    )))
                }
            }
        }
    }

    /// Tries to evaluate a constant expression at plan time.
    ///
    /// Recursively folds literals, unary operators, lists, and known function calls
    /// (like `vector()`) into concrete values. Returns `None` if the expression
    /// contains non-constant parts (variables, property accesses, etc.).
    pub(super) fn try_fold_expression(expr: &LogicalExpression) -> Option<Value> {
        match expr {
            LogicalExpression::Literal(v) => Some(v.clone()),
            LogicalExpression::List(items) => {
                let values: Option<Vec<Value>> =
                    items.iter().map(Self::try_fold_expression).collect();
                Some(Value::List(values?.into()))
            }
            LogicalExpression::FunctionCall { name, args, .. } => {
                match name.to_lowercase().as_str() {
                    "vector" => {
                        if args.len() != 1 {
                            return None;
                        }
                        let val = Self::try_fold_expression(&args[0])?;
                        match val {
                            Value::List(items) => {
                                // reason: intentional lossy f64/i64 to f32 for vector elements
                                #[allow(clippy::cast_possible_truncation)]
                                let floats: Vec<f32> = items
                                    .iter()
                                    .filter_map(|v| match v {
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
                            // Already a vector (from all-numeric list folding)
                            Value::Vector(v) => Some(Value::Vector(v)),
                            _ => None,
                        }
                    }
                    "timestamp" => {
                        if !args.is_empty() {
                            return None;
                        }
                        Some(Value::Int64(
                            grafeo_common::types::Timestamp::now().as_millis(),
                        ))
                    }
                    "now" | "current_timestamp" | "currenttimestamp" => {
                        if !args.is_empty() {
                            return None;
                        }
                        Some(Value::Timestamp(grafeo_common::types::Timestamp::now()))
                    }
                    "date" | "todate" | "current_date" | "currentdate" => {
                        if args.is_empty() {
                            return Some(Value::Date(grafeo_common::types::Date::today()));
                        }
                        if args.len() != 1 {
                            return None;
                        }
                        let val = Self::try_fold_expression(&args[0])?;
                        match val {
                            Value::String(s) => {
                                grafeo_common::types::Date::parse(&s).map(Value::Date)
                            }
                            _ => None,
                        }
                    }
                    "time" | "totime" | "local_time" | "current_time" | "currenttime" => {
                        if args.is_empty() {
                            return Some(Value::Time(grafeo_common::types::Time::now()));
                        }
                        if args.len() != 1 {
                            return None;
                        }
                        let val = Self::try_fold_expression(&args[0])?;
                        match val {
                            Value::String(s) => {
                                grafeo_common::types::Time::parse(&s).map(Value::Time)
                            }
                            _ => None,
                        }
                    }
                    "datetime" | "localdatetime" | "local_datetime" | "todatetime" => {
                        if args.is_empty() {
                            return Some(Value::Timestamp(grafeo_common::types::Timestamp::now()));
                        }
                        if args.len() != 1 {
                            return None;
                        }
                        let val = Self::try_fold_expression(&args[0])?;
                        match val {
                            Value::String(s) => {
                                if let Some(d) = grafeo_common::types::Date::parse(&s) {
                                    return Some(Value::Timestamp(d.to_timestamp()));
                                }
                                if let Some(pos) = s.find('T') {
                                    let (date_part, time_part) = (&s[..pos], &s[pos + 1..]);
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
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            LogicalExpression::Map(entries) => {
                let folded: Option<Vec<(String, Value)>> = entries
                    .iter()
                    .map(|(k, v)| Self::try_fold_expression(v).map(|val| (k.clone(), val)))
                    .collect();
                let folded = folded?;
                let map: std::collections::BTreeMap<grafeo_common::types::PropertyKey, Value> =
                    folded
                        .into_iter()
                        .map(|(k, v)| (grafeo_common::types::PropertyKey::from(k), v))
                        .collect();
                Some(Value::Map(std::sync::Arc::new(map)))
            }
            LogicalExpression::Unary { op, operand } => {
                let value = Self::try_fold_expression(operand)?;
                match op {
                    UnaryOp::Neg => match value {
                        Value::Int64(n) => Some(Value::Int64(-n)),
                        Value::Float64(f) => Some(Value::Float64(-f)),
                        _ => None,
                    },
                    UnaryOp::Not => match value {
                        Value::Bool(b) => Some(Value::Bool(!b)),
                        _ => None,
                    },
                    UnaryOp::IsNull | UnaryOp::IsNotNull => None,
                }
            }
            _ => None,
        }
    }
}
