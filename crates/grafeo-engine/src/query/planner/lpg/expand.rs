//! Relationship expansion and factorized chain planning.

use std::collections::{HashMap, HashSet};

use super::{
    Arc, Direction, Error, ExecutionPathMode, ExpandDirection, ExpandOp, ExpandOperator,
    ExpandStep, ExpressionPredicate, FilterOperator, GraphStoreSearch, LazyFactorizedChainOperator,
    LeapfrogExpandOperator, LeapfrogExpandSpec, LogicalOperator, Operator, PathMode, Result,
    VariableLengthExpandOperator,
};
use crate::query::plan::PathSearch;
use grafeo_core::execution::operators::ExecutionPathSearch;

impl super::Planner {
    /// Plans an expand operator.
    pub(super) fn plan_expand(
        &self,
        expand: &ExpandOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        if expand.path_search != PathSearch::All
            && expand.edge_predicate.as_ref().is_some_and(|predicate| {
                !crate::query::optimizer::Optimizer::intrinsic_edge_predicate_is_stable(predicate)
            })
        {
            return Err(Error::Query(grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Unsupported,
                "pruned path searches require supported deterministic intrinsic edge predicates",
            )));
        }
        // Plan the input operator first
        let (input_op, input_columns) = self.plan_operator(&expand.input)?;

        // Find the source column index
        let source_column = input_columns
            .iter()
            .position(|c| c == &expand.from_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Source variable '{}' not found in input columns",
                    expand.from_variable
                ))
            })?;
        let close_to = input_columns.iter().position(|c| c == &expand.to_variable);

        // Convert expand direction
        let direction = match expand.direction {
            ExpandDirection::Outgoing => Direction::Outgoing,
            ExpandDirection::Incoming => Direction::Incoming,
            ExpandDirection::Both => Direction::Both,
        };

        // Check if this is a variable-length path
        let is_variable_length =
            expand.min_hops != 1 || expand.max_hops.is_none() || expand.max_hops != Some(1);

        // Unified traversal owns multi-hop paths, named path detail and explicit
        // search budgets, including anonymous single-hop searches.
        let needs_path_details = expand.path_alias.is_some();

        let operator: Box<dyn Operator> = if is_variable_length
            || needs_path_details
            || expand.path_search != PathSearch::All
            || expand.edge_predicate.is_some()
            || expand.path_predicate.is_some()
        {
            // Use VariableLengthExpandOperator for multi-hop paths or named paths
            let min_hops = if is_variable_length {
                expand.min_hops
            } else {
                1
            };
            // An unbounded `*` needs *some* cap for an unpruned walk enumeration,
            // or a cycle diverges; `min_hops + 100` is that cap. A pruned search
            // terminates on its own budget, so it must not inherit the cap: the
            // operator it replaces had none, and 100 would silently drop the
            // paths beyond hop 101 that work today.
            let max_hops = if is_variable_length {
                expand.max_hops.unwrap_or(match expand.path_search {
                    PathSearch::All => expand.min_hops.saturating_add(100),
                    PathSearch::DistinctTargets | PathSearch::Shortest { .. } => u32::MAX,
                })
            } else {
                1
            };
            let exec_path_mode = match expand.path_mode {
                PathMode::Walk => ExecutionPathMode::Walk,
                PathMode::Trail => ExecutionPathMode::Trail,
                PathMode::Simple => ExecutionPathMode::Simple,
                PathMode::Acyclic => ExecutionPathMode::Acyclic,
            };
            let exec_path_search = match expand.path_search {
                PathSearch::All => ExecutionPathSearch::All,
                PathSearch::DistinctTargets => ExecutionPathSearch::DistinctTargets,
                PathSearch::Shortest { k, groups } => ExecutionPathSearch::Shortest { k, groups },
            };

            let mut expand_op = VariableLengthExpandOperator::new(
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
                input_op,
                source_column,
                direction,
                expand.edge_types.clone(),
                min_hops,
                max_hops,
            )
            .with_path_mode(exec_path_mode)
            .with_path_search(exec_path_search)
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_read_only(self.read_only);

            if let Some(predicate) = &expand.edge_predicate {
                let mut variable_columns: HashMap<String, usize> = input_columns
                    .iter()
                    .enumerate()
                    .map(|(index, name)| (name.clone(), index))
                    .collect();
                if let Some(edge_variable) = &expand.edge_variable {
                    variable_columns.insert(edge_variable.clone(), input_columns.len());
                }
                if expand.path_search != PathSearch::All {
                    // The binder may have seen sibling join bindings which are
                    // absent from this physical input. A certified predicate
                    // has no local scopes, so every reference must be present.
                    let mut referenced = HashSet::new();
                    crate::query::optimizer::Optimizer::collect_variables(
                        predicate,
                        &mut referenced,
                    );
                    if let Some(missing) = referenced
                        .iter()
                        .find(|name| !variable_columns.contains_key(*name))
                    {
                        return Err(Error::Query(grafeo_common::utils::error::QueryError::new(
                            grafeo_common::utils::error::QueryErrorKind::Unsupported,
                            format!(
                                "intrinsic edge predicate binding '{missing}' is absent from the path input"
                            ),
                        )));
                    }
                }
                let predicate = ExpressionPredicate::new(
                    self.convert_expression(predicate)?,
                    variable_columns,
                    Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone());
                expand_op = expand_op.with_edge_predicate(Box::new(predicate));
            }

            if let Some(predicate) = &expand.path_predicate {
                let path_alias = expand.path_alias.as_ref().ok_or_else(|| {
                    Error::Internal("candidate path predicate requires a path alias".into())
                })?;
                let mut variable_columns: HashMap<String, usize> = input_columns
                    .iter()
                    .enumerate()
                    .map(|(index, name)| (name.clone(), index))
                    .collect();
                let base = input_columns.len();
                if let Some(edge_variable) = &expand.edge_variable {
                    // Cypher variable-length relationship bindings are lists;
                    // the first appended column remains the final typed edge.
                    variable_columns.insert(
                        edge_variable.clone(),
                        if is_variable_length { base + 4 } else { base },
                    );
                }
                if close_to.is_none() {
                    variable_columns.insert(expand.to_variable.clone(), base + 1);
                }
                for (offset, name) in [
                    format!("_path_length_{path_alias}"),
                    format!("_path_nodes_{path_alias}"),
                    format!("_path_edges_{path_alias}"),
                    path_alias.clone(),
                ]
                .into_iter()
                .enumerate()
                {
                    variable_columns.insert(name, base + 2 + offset);
                }
                let mut referenced = HashSet::new();
                crate::query::optimizer::Optimizer::collect_variables(predicate, &mut referenced);
                if let Some(missing) = referenced
                    .iter()
                    .find(|name| !variable_columns.contains_key(*name))
                {
                    return Err(Error::Query(grafeo_common::utils::error::QueryError::new(
                        grafeo_common::utils::error::QueryErrorKind::Unsupported,
                        format!(
                            "path predicate binding '{missing}' is absent from the candidate input"
                        ),
                    )));
                }
                let predicate = ExpressionPredicate::new(
                    self.convert_expression(predicate)?,
                    variable_columns,
                    Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone());
                expand_op = expand_op.with_path_predicate(Box::new(predicate));
            }

            // If a path alias is set, enable path length and detail output
            if needs_path_details {
                expand_op = expand_op
                    .with_path_length_output()
                    .with_path_detail_output();
            }

            Box::new(expand_op)
        } else {
            // Use simple ExpandOperator for single-hop paths without named paths
            let expand_op = ExpandOperator::new(
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
                input_op,
                source_column,
                direction,
                expand.edge_types.clone(),
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_read_only(self.read_only);
            Box::new(expand_op)
        };

        // Build output columns: [input_columns..., edge, target, (path_length)?]
        // Preserve all input columns and add edge + target to match ExpandOperator output
        let mut columns = input_columns;

        // Generate edge column name and register for EdgeResolve in RETURN
        let edge_col_name = self.register_edge_column(&expand.edge_variable);
        if is_variable_length {
            self.group_list_variables
                .borrow_mut()
                .insert(edge_col_name.clone());
        }
        columns.push(edge_col_name);

        let target_column = if close_to.is_some() {
            format!("_close_{}", expand.to_variable)
        } else {
            expand.to_variable.clone()
        };
        columns.push(target_column.clone());

        // If a path alias is set, add columns for path length, nodes, edges, and the path itself
        if let Some(ref path_alias) = expand.path_alias {
            let length_col = format!("_path_length_{}", path_alias);
            let nodes_col = format!("_path_nodes_{}", path_alias);
            let edges_col = format!("_path_edges_{}", path_alias);
            // Mark as scalar so plan_return uses Column pass-through, not NodeResolve
            self.scalar_columns.borrow_mut().insert(length_col.clone());
            self.scalar_columns.borrow_mut().insert(nodes_col.clone());
            self.scalar_columns.borrow_mut().insert(edges_col.clone());
            // The path alias itself is also a scalar column containing Value::Path
            self.scalar_columns.borrow_mut().insert(path_alias.clone());
            columns.push(length_col);
            columns.push(nodes_col);
            columns.push(edges_col);
            columns.push(path_alias.clone());
        }

        if close_to.is_some() {
            // Dest must equal the already-bound node (cycle close), not rebind.
            let dest_name = target_column;
            let pred = crate::query::plan::LogicalExpression::Binary {
                left: Box::new(crate::query::plan::LogicalExpression::FunctionCall {
                    name: "id".into(),
                    args: vec![crate::query::plan::LogicalExpression::Variable(
                        dest_name.clone(),
                    )],
                    distinct: false,
                }),
                op: crate::query::plan::BinaryOp::Eq,
                right: Box::new(crate::query::plan::LogicalExpression::FunctionCall {
                    name: "id".into(),
                    args: vec![crate::query::plan::LogicalExpression::Variable(
                        expand.to_variable.clone(),
                    )],
                    distinct: false,
                }),
            };
            let filter_expr = self.convert_expression(&pred)?;
            let variable_columns: HashMap<String, usize> = columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();
            let predicate = ExpressionPredicate::new(
                filter_expr,
                variable_columns,
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone());
            let operator = Box::new(FilterOperator::new(operator, Box::new(predicate)));
            return Ok((operator, columns));
        }

        Ok((operator, columns))
    }

    /// Plans a chain of consecutive expand operations using factorized execution.
    ///
    /// This avoids the Cartesian product explosion that occurs with separate expands.
    /// For a 2-hop query with degree d, this uses O(d) memory instead of O(d^2).
    ///
    /// The chain is executed lazily at query time, not during planning. This ensures
    /// that any filters applied above the expand chain are properly respected.
    pub(super) fn plan_expand_chain(
        &self,
        op: &LogicalOperator,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (lazy_op, columns) = self.plan_expand_chain_lazy(op)?;
        Ok((Box::new(lazy_op), columns))
    }

    /// Same as [`Self::plan_expand_chain`] but keeps the concrete operator
    /// so a factorized filter can wrap it without flattening.
    pub(super) fn plan_expand_chain_lazy(
        &self,
        op: &LogicalOperator,
    ) -> Result<(LazyFactorizedChainOperator, Vec<String>)> {
        let expands = Self::collect_expand_chain(op);
        if expands.is_empty() {
            return Err(Error::Internal("Empty expand chain".to_string()));
        }

        // Get the base operator (before first expand)
        let first_expand = expands[0];
        let (base_op, base_columns) = self.plan_operator(&first_expand.input)?;

        let mut columns = base_columns.clone();
        let mut steps = Vec::new();

        // Track the level-local source column for each expand
        // For the first expand, it's the column in the input (base_columns)
        // For subsequent expands, the target from the previous level is always at index 1
        // (each level adds [edge, target], so target is at index 1)
        let mut is_first = true;
        let last_idx = expands.len().saturating_sub(1);

        for (i, expand) in expands.iter().enumerate() {
            // Find source column for this expand
            let source_column = if is_first {
                // For first expand, find in base columns
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
                // For subsequent expands, the target from the previous level is at index 1
                // (each level adds [edge, target], so target is the second column)
                1
            };

            // Convert direction
            let direction = match expand.direction {
                ExpandDirection::Outgoing => Direction::Outgoing,
                ExpandDirection::Incoming => Direction::Incoming,
                ExpandDirection::Both => Direction::Both,
            };

            // Anonymous last hop: dest nodes only (RETURN c does not need EdgeIds).
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

        // Create lazy operator that executes at query time, not planning time
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

        Ok((lazy_op, columns))
    }

    /// True when `op` is `(a)-[]->(b)-[]->(c)-[]->(a)` (three single-hop expands
    /// that close a directed cycle on the start variable).
    pub(super) fn is_directed_triangle_chain(&self, op: &LogicalOperator) -> bool {
        let expands = Self::collect_expand_chain(op);
        if expands.len() != 3 {
            return false;
        }
        let (ab, bc, ca) = (expands[0], expands[1], expands[2]);
        ca.to_variable == ab.from_variable
            && bc.from_variable == ab.to_variable
            && ca.from_variable == bc.to_variable
    }

    /// `CostModel::factorized_benefit` for a consecutive expand chain.
    pub(super) fn factorized_benefit_for_chain(&self, op: &LogicalOperator) -> f64 {
        let expands = Self::collect_expand_chain(op);
        if expands.is_empty() {
            return 1.0;
        }
        let stats = self.store.statistics();
        let hops = expands.len();
        let fanout = expands
            .iter()
            .map(|e| self.estimate_expand_degree(&stats, e))
            .sum::<f64>()
            / hops as f64;
        crate::query::optimizer::CostModel::new().factorized_benefit(fanout, hops)
    }

    /// First hop as a normal expand, closing hop as [`LeapfrogExpandOperator`].
    pub(super) fn plan_leapfrog_triangle(
        &self,
        op: &LogicalOperator,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let expands = Self::collect_expand_chain(op);
        if expands.len() != 3 {
            return Err(Error::Internal(
                "Leapfrog triangle requires a 3-hop expand chain".to_string(),
            ));
        }
        let (ab, bc, ca) = (expands[0], expands[1], expands[2]);
        let (first_op, mut columns) = self.plan_expand(ab)?;
        let a_col = columns
            .iter()
            .position(|c| c == &ab.from_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Triangle start '{}' missing after first expand",
                    ab.from_variable
                ))
            })?;
        let b_col = columns
            .iter()
            .position(|c| c == &ab.to_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Triangle mid '{}' missing after first expand",
                    ab.to_variable
                ))
            })?;

        let left_dir = match bc.direction {
            ExpandDirection::Outgoing => Direction::Outgoing,
            ExpandDirection::Incoming => Direction::Incoming,
            ExpandDirection::Both => Direction::Both,
        };
        let right_dir = match ca.direction {
            ExpandDirection::Outgoing => Direction::Incoming,
            ExpandDirection::Incoming => Direction::Outgoing,
            ExpandDirection::Both => Direction::Both,
        };

        let mut leap = LeapfrogExpandOperator::new(
            Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            first_op,
            LeapfrogExpandSpec {
                left_column: b_col,
                right_column: a_col,
                left_direction: left_dir,
                right_direction: right_dir,
                left_edge_types: bc.edge_types.clone(),
                right_edge_types: ca.edge_types.clone(),
            },
        )
        .with_read_only(self.read_only);
        if let Some(transaction_id) = self.transaction_id {
            leap = leap.with_transaction_context(self.viewing_epoch, Some(transaction_id));
        } else {
            leap = leap.with_transaction_context(self.viewing_epoch, None);
        }

        columns.push(self.register_edge_column(&bc.edge_variable));
        columns.push(bc.to_variable.clone());
        columns.push(self.register_edge_column(&ca.edge_variable));

        let mut operator: Box<dyn Operator> = Box::new(leap);
        for pred in Self::collect_haslabel_filters(op) {
            let filter_expr = self.convert_expression(pred)?;
            let variable_columns: HashMap<String, usize> = columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();
            let predicate = ExpressionPredicate::new(
                filter_expr,
                variable_columns,
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone());
            operator = Box::new(FilterOperator::new(operator, Box::new(predicate)));
        }

        Ok((operator, columns))
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::super::{
        ExpandDirection, ExpandOp, GraphStoreSearch, LogicalOperator, LogicalPlan, NodeScanOp,
        PathMode, Planner,
    };
    use crate::query::plan::PathSearch;
    use grafeo_common::memory::buffer::BufferManager;
    use grafeo_common::types::NodeId;
    use grafeo_core::execution::QueryExecutionControl;
    use grafeo_core::execution::memory::QueryResourceContext;
    use grafeo_core::execution::operators::OperatorError;
    use grafeo_core::execution::pipeline_convert::convert_to_pipeline_with_resources;
    use grafeo_core::graph::lpg::LpgStore;
    use std::sync::Arc;

    fn expand_from_source(path_search: PathSearch, max_hops: Option<u32>) -> LogicalPlan {
        LogicalPlan::new(LogicalOperator::Expand(ExpandOp {
            from_variable: "s".to_string(),
            to_variable: "t".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["R".to_string()],
            min_hops: 1,
            max_hops,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "s".to_string(),
                label: Some("Source".to_string()),
                input: None,
            })),
            path_alias: None,
            path_mode: PathMode::Walk,
            path_search,
            edge_predicate: None,
            path_predicate: None,
        }))
    }

    fn run_targets(store: Arc<LpgStore>, plan: LogicalPlan) -> Vec<NodeId> {
        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>);
        let planned = planner.plan(&plan).unwrap();
        let target_column = planned
            .columns
            .iter()
            .position(|column| column == "t")
            .expect("target column");
        let mut operator = planned.operator;
        let mut targets = Vec::new();
        while let Some(chunk) = operator.next().unwrap() {
            for row in 0..chunk.row_count() {
                targets.push(
                    chunk
                        .column(target_column)
                        .unwrap()
                        .get_node_id(row)
                        .expect("node target"),
                );
            }
        }
        targets
    }

    fn branching_chain(store: &LpgStore, hops: u32) -> NodeId {
        assert!(hops >= 2);
        let source = store.create_node(&["Source"]);
        let mut left = store.create_node(&["V"]);
        let mut right = store.create_node(&["V"]);
        store.create_edge(source, left, "R");
        store.create_edge(source, right, "R");
        for _ in 2..hops {
            let next_left = store.create_node(&["V"]);
            let next_right = store.create_node(&["V"]);
            store.create_edge(left, next_left, "R");
            store.create_edge(right, next_right, "R");
            left = next_left;
            right = next_right;
        }
        let target = store.create_node(&["Target"]);
        store.create_edge(left, target, "R");
        store.create_edge(right, target, "R");
        target
    }

    #[test]
    fn shortest_intrinsic_predicate_rejects_missing_physical_binding() {
        use crate::query::plan::{BinaryOp, LogicalExpression};
        use grafeo_common::utils::error::{Error, QueryErrorKind};
        let store = Arc::new(LpgStore::new().unwrap());
        for search in [
            PathSearch::Shortest {
                k: 1,
                groups: false,
            },
            PathSearch::DistinctTargets,
        ] {
            let mut plan = expand_from_source(search, Some(2));
            let LogicalOperator::Expand(expand) = &mut plan.root else {
                panic!("expand fixture")
            };
            expand.edge_variable = Some("r".into());
            expand.edge_predicate = Some(LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: "r".into(),
                    property: "cost".into(),
                }),
                op: BinaryOp::Le,
                right: Box::new(LogicalExpression::Property {
                    variable: "sibling".into(),
                    property: "budget".into(),
                }),
            });
            let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>);
            let Err(error) = planner.plan(&plan) else {
                panic!("missing sibling must fail closed")
            };
            assert!(
                matches!(&error, Error::Query(error) if error.kind == QueryErrorKind::Unsupported)
            );
            assert!(error.to_string().contains("sibling"));
        }
    }

    #[test]
    fn bound_target_shortest_keeps_all_named_path_columns() {
        let store = Arc::new(LpgStore::new().unwrap());
        let source = store.create_node(&["Source"]);
        let target = store.create_node(&["Target"]);
        store.create_edge(source, target, "R");
        let mut plan = expand_from_source(
            PathSearch::Shortest {
                k: 1,
                groups: false,
            },
            Some(2),
        );
        let LogicalOperator::Expand(expand) = &mut plan.root else {
            panic!("expand fixture")
        };
        expand.path_alias = Some("p".into());
        let LogicalOperator::NodeScan(source_scan) = expand.input.as_mut() else {
            panic!("source fixture")
        };
        source_scan.input = Some(Box::new(LogicalOperator::NodeScan(NodeScanOp {
            variable: "t".into(),
            label: Some("Target".into()),
            input: None,
        })));
        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>);
        let mut planned = planner.plan(&plan).unwrap();
        for name in ["_path_length_p", "_path_nodes_p", "_path_edges_p", "p"] {
            assert!(
                planned.columns.iter().any(|column| column == name),
                "missing {name}"
            );
        }
        let alias = planned
            .columns
            .iter()
            .position(|column| column == "p")
            .unwrap();
        let chunk = planned.operator.next().unwrap().unwrap();
        assert_eq!(chunk.len(), 1);
        assert!(matches!(
            chunk.column(alias).unwrap().get_value(0),
            Some(grafeo_common::types::Value::Path { .. })
        ));
        assert!(planned.operator.next().unwrap().is_none());
    }

    #[test]
    fn planner_maps_pruned_searches_to_unbounded_variable_expands() {
        let store = Arc::new(LpgStore::new().unwrap());
        let target = branching_chain(&store, 149);

        for search in [
            PathSearch::DistinctTargets,
            PathSearch::Shortest {
                k: 1,
                groups: false,
            },
        ] {
            let targets = run_targets(Arc::clone(&store), expand_from_source(search, None));
            assert_eq!(
                targets
                    .iter()
                    .filter(|candidate| **candidate == target)
                    .count(),
                1,
                "{search:?} must retain the target reached only at hop 149"
            );
        }
    }

    #[test]
    fn planner_default_all_keeps_the_existing_unbounded_cap() {
        let store = Arc::new(LpgStore::new().unwrap());
        let source = store.create_node(&["Source"]);
        let mut chain = vec![source];
        let mut previous = source;
        for _ in 0..102 {
            let next = store.create_node(&["V"]);
            store.create_edge(previous, next, "R");
            chain.push(next);
            previous = next;
        }

        let targets = run_targets(
            Arc::clone(&store),
            expand_from_source(PathSearch::All, None),
        );
        assert!(targets.contains(&chain[101]));
        assert!(!targets.contains(&chain[102]));
        assert_eq!(targets.len(), 101);
    }

    #[test]
    fn planner_registers_named_path_detail_columns() {
        let store = Arc::new(LpgStore::new().unwrap());
        let source = store.create_node(&["Source"]);
        let target = store.create_node(&["Target"]);
        store.create_edge(source, target, "R");
        let mut plan = expand_from_source(PathSearch::All, Some(1));
        if let LogicalOperator::Expand(expand) = &mut plan.root {
            expand.path_alias = Some("p".to_string());
        }

        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>);
        let planned = planner.plan(&plan).unwrap();
        assert!(
            planned
                .columns
                .iter()
                .any(|column| column == "_path_length_p")
        );
        assert!(
            planned
                .columns
                .iter()
                .any(|column| column == "_path_nodes_p")
        );
        assert!(
            planned
                .columns
                .iter()
                .any(|column| column == "_path_edges_p")
        );
        assert!(planned.columns.iter().any(|column| column == "p"));
        let mut operator = planned.operator;
        let chunk = operator.next().unwrap().expect("one path");
        assert_eq!(chunk.row_count(), 1);
        assert_eq!(chunk.column_count(), 7);
        assert_eq!(chunk.column(2).unwrap().get_node_id(0), Some(target));
    }

    #[test]
    fn planner_maps_query_resource_cancellation_to_expand() {
        let store = Arc::new(LpgStore::new().unwrap());
        let source_node = store.create_node(&["Source"]);
        let target = store.create_node(&["Target"]);
        store.create_edge(source_node, target, "R");

        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>);
        let planned = planner
            .plan(&expand_from_source(PathSearch::DistinctTargets, None))
            .unwrap();
        let control = QueryExecutionControl::new();
        let resources = QueryResourceContext::new_with_cancellation(
            BufferManager::with_budget(1 << 20),
            control.token(),
        )
        .unwrap();
        let (mut source, push_ops) =
            convert_to_pipeline_with_resources(planned.into_operator(), &resources).unwrap();
        assert!(push_ops.is_empty(), "plain expand remains a pull boundary");

        control.cancellation_handle().cancel();
        assert!(matches!(
            source.next(),
            Err(OperatorError::QueryCancelled(_))
        ));
    }

    #[test]
    fn planner_applies_search_modes_to_explicit_single_hop_expands() {
        let store = Arc::new(LpgStore::new().unwrap());
        let source = store.create_node(&["Source"]);
        let target = store.create_node(&["Target"]);
        store.create_edge(source, target, "R");
        store.create_edge(source, target, "R");

        let all_targets = run_targets(
            Arc::clone(&store),
            expand_from_source(PathSearch::All, Some(1)),
        );
        assert_eq!(
            all_targets
                .iter()
                .filter(|candidate| **candidate == target)
                .count(),
            2,
            "ALL retains parallel single-hop walks"
        );

        for search in [
            PathSearch::DistinctTargets,
            PathSearch::Shortest {
                k: 1,
                groups: false,
            },
        ] {
            let targets = run_targets(Arc::clone(&store), expand_from_source(search, Some(1)));
            assert_eq!(
                targets
                    .iter()
                    .filter(|candidate| **candidate == target)
                    .count(),
                1,
                "{search:?} must collapse parallel single-hop walks"
            );
        }
    }

    #[test]
    fn planner_fusion_preserves_non_all_search_cardinality() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["V"]);
        let b = store.create_node(&["V"]);
        let c = store.create_node(&["V"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, c, "R");
        store.create_edge(c, a, "R");
        store.create_edge(c, a, "R");

        let search = PathSearch::Shortest {
            k: 2,
            groups: false,
        };
        let logical = {
            let mut logical = triangle_logical();
            if let LogicalOperator::Expand(expand) = &mut logical {
                // Named path detail forces the unfused plan through the same
                // variable-length operator used for this explicit search.
                expand.path_alias = Some("p".to_string());
                expand.path_search = search;
            }
            logical
        };

        let count_rows = |planner: Planner, logical: LogicalOperator| {
            let mut operator = planner.plan(&LogicalPlan::new(logical)).unwrap().operator;
            let mut rows = 0;
            while let Some(chunk) = operator.next().unwrap() {
                rows += chunk.row_count();
            }
            rows
        };
        let fused_rows = count_rows(
            Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>),
            logical.clone(),
        );
        let unfused_rows = count_rows(
            Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>)
                .with_factorized_execution(false),
            logical,
        );

        // Three starting vertices each have two legal shortest walks through
        // the parallel closing edges. Fusion must preserve all six rows.
        assert_eq!(unfused_rows, 6);
        assert_eq!(fused_rows, 6);
    }

    #[test]
    fn planner_generic_all_fusion_preserves_parallel_edge_rows() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["V"]);
        let b = store.create_node(&["V"]);
        let c = store.create_node(&["V"]);
        let ab = store.create_edge(a, b, "R");
        let bc = store.create_edge(b, c, "R");
        let ca0 = store.create_edge(c, a, "R");
        let ca1 = store.create_edge(c, a, "R");

        let logical = {
            let mut logical = triangle_logical();
            if let LogicalOperator::Expand(expand) = &mut logical {
                // Keep this an ALL path while making the outer hop ineligible
                // for the specialized triangle shortcut. The two inner ALL
                // hops therefore exercise generic prefix fusion directly.
                expand.edge_predicate = Some(crate::query::plan::LogicalExpression::Literal(
                    grafeo_common::types::Value::Bool(true),
                ));
            }
            logical
        };

        let collect_rows = |planner: Planner, logical: LogicalOperator| {
            let mut planned = planner.plan(&LogicalPlan::new(logical)).unwrap();
            let a_col = planned.columns.iter().position(|name| name == "a").unwrap();
            let b_col = planned.columns.iter().position(|name| name == "b").unwrap();
            let c_col = planned.columns.iter().position(|name| name == "c").unwrap();
            let closing_edge_col = planned
                .columns
                .iter()
                .rposition(|name| name.starts_with("_anon_edge_"))
                .unwrap();
            let mut rows = Vec::new();
            while let Some(chunk) = planned.operator.next().unwrap() {
                for row in chunk.selected_indices() {
                    rows.push((
                        chunk.column(a_col).unwrap().get_node_id(row).unwrap(),
                        chunk.column(b_col).unwrap().get_node_id(row).unwrap(),
                        chunk.column(c_col).unwrap().get_node_id(row).unwrap(),
                        chunk
                            .column(closing_edge_col)
                            .unwrap()
                            .get_edge_id(row)
                            .unwrap(),
                    ));
                }
            }
            rows.sort_unstable();
            rows
        };

        let mut expected = vec![
            (a, b, c, ca0),
            (a, b, c, ca1),
            (b, c, a, ab),
            (b, c, a, ab),
            (c, a, b, bc),
            (c, a, b, bc),
        ];
        expected.sort_unstable();

        let fused = collect_rows(
            Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>),
            logical.clone(),
        );
        let unfused = collect_rows(
            Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>)
                .with_factorized_execution(false),
            logical,
        );

        assert_eq!(unfused, expected);
        assert_eq!(fused, expected);
    }

    fn triangle_logical() -> LogicalOperator {
        LogicalOperator::Expand(ExpandOp {
            from_variable: "c".to_string(),
            to_variable: "a".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["R".to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::Expand(ExpandOp {
                from_variable: "b".to_string(),
                to_variable: "c".to_string(),
                edge_variable: None,
                direction: ExpandDirection::Outgoing,
                edge_types: vec!["R".to_string()],
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(LogicalOperator::Expand(ExpandOp {
                    from_variable: "a".to_string(),
                    to_variable: "b".to_string(),
                    edge_variable: None,
                    direction: ExpandDirection::Outgoing,
                    edge_types: vec!["R".to_string()],
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "a".to_string(),
                        label: Some("V".to_string()),
                        input: None,
                    })),
                    path_alias: None,
                    path_mode: PathMode::Walk,
                    edge_predicate: None,
                    path_predicate: None,
                    path_search: crate::query::plan::PathSearch::All,
                })),
                path_alias: None,
                path_mode: PathMode::Walk,
                edge_predicate: None,
                path_predicate: None,
                path_search: crate::query::plan::PathSearch::All,
            })),
            path_alias: None,
            path_mode: PathMode::Walk,
            edge_predicate: None,
            path_predicate: None,
            path_search: crate::query::plan::PathSearch::All,
        })
    }

    #[test]
    fn detects_directed_triangle_chain() {
        let store = Arc::new(LpgStore::new().unwrap());
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let op = triangle_logical();
        assert!(planner.is_directed_triangle_chain(&op));
    }

    #[test]
    fn plans_leapfrog_expand_for_triangle() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["V"]);
        let b = store.create_node(&["V"]);
        let c = store.create_node(&["V"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, c, "R");
        store.create_edge(c, a, "R");
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let (mut op, cols) = planner.plan_leapfrog_triangle(&triangle_logical()).unwrap();
        assert_eq!(op.name(), "LeapfrogExpand");
        assert!(cols.contains(&"a".to_string()));
        assert!(cols.contains(&"b".to_string()));
        assert!(cols.contains(&"c".to_string()));
        let mut rows = 0usize;
        while let Some(chunk) = op.next().unwrap() {
            rows += chunk.row_count();
        }
        // One directed cycle, three starting vertices.
        assert_eq!(rows, 3);

        let planned = planner.plan(&LogicalPlan::new(triangle_logical())).unwrap();
        assert_eq!(planned.operator.name(), "LeapfrogExpand");
    }

    fn hop2_then_close_join() -> LogicalOperator {
        use super::super::{JoinOp, JoinType, LogicalExpression};
        use crate::query::plan::JoinCondition;
        let hop2 = LogicalOperator::Expand(ExpandOp {
            from_variable: "b".to_string(),
            to_variable: "c".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["R".to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::Expand(ExpandOp {
                from_variable: "a".to_string(),
                to_variable: "b".to_string(),
                edge_variable: None,
                direction: ExpandDirection::Outgoing,
                edge_types: vec!["R".to_string()],
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: Some("V".to_string()),
                    input: None,
                })),
                path_alias: None,
                path_mode: PathMode::Walk,
                edge_predicate: None,
                path_predicate: None,
                path_search: crate::query::plan::PathSearch::All,
            })),
            path_alias: None,
            path_mode: PathMode::Walk,
            edge_predicate: None,
            path_predicate: None,
            path_search: crate::query::plan::PathSearch::All,
        });
        let close = LogicalOperator::Expand(ExpandOp {
            from_variable: "c".to_string(),
            to_variable: "a".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["R".to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "c".to_string(),
                label: None,
                input: None,
            })),
            path_alias: None,
            path_mode: PathMode::Walk,
            edge_predicate: None,
            path_predicate: None,
            path_search: crate::query::plan::PathSearch::All,
        });
        LogicalOperator::Join(JoinOp {
            left: Box::new(hop2),
            right: Box::new(close),
            join_type: JoinType::Inner,
            conditions: vec![
                JoinCondition {
                    left: LogicalExpression::Variable("a".to_string()),
                    right: LogicalExpression::Variable("a".to_string()),
                    semantics: crate::query::plan::JoinKeySemantics::Value,
                },
                JoinCondition {
                    left: LogicalExpression::Variable("c".to_string()),
                    right: LogicalExpression::Variable("c".to_string()),
                    semantics: crate::query::plan::JoinKeySemantics::Value,
                },
            ],
        })
    }

    #[test]
    fn qualified_triangle_join_preserves_non_all_expand_modes() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["V"]);
        let b = store.create_node(&["V"]);
        let c = store.create_node(&["V"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, c, "R");
        store.create_edge(c, a, "R");

        for modify_close in [true, false] {
            let mut logical = hop2_then_close_join();
            if let LogicalOperator::Join(join) = &mut logical {
                if modify_close && let LogicalOperator::Expand(close) = &mut *join.right {
                    close.path_search = PathSearch::DistinctTargets;
                }
                if !modify_close
                    && let LogicalOperator::Expand(hop2) = &mut *join.left
                    && let LogicalOperator::Expand(inner) = &mut *hop2.input
                {
                    inner.path_search = PathSearch::Shortest {
                        k: 1,
                        groups: false,
                    };
                }
            }

            let count_rows = |planner: Planner| {
                let mut operator = planner
                    .plan(&LogicalPlan::new(logical.clone()))
                    .unwrap()
                    .operator;
                let mut rows = 0;
                while let Some(chunk) = operator.next().unwrap() {
                    rows += chunk.row_count();
                }
                rows
            };
            let unfused_rows = count_rows(
                Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>)
                    .with_factorized_execution(false),
            );
            let fused_rows =
                count_rows(Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>));

            assert_eq!(unfused_rows, 3);
            assert_eq!(fused_rows, 3);
        }
    }

    #[test]
    fn comma_join_triangle_uses_leapfrog() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["V"]);
        let b = store.create_node(&["V"]);
        let c = store.create_node(&["V"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, c, "R");
        store.create_edge(c, a, "R");
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let planned = planner
            .plan(&LogicalPlan::new(hop2_then_close_join()))
            .unwrap();
        assert_eq!(planned.operator.name(), "LeapfrogExpand");
        let mut rows = 0usize;
        let mut op = planned.operator;
        while let Some(chunk) = op.next().unwrap() {
            rows += chunk.row_count();
        }
        assert_eq!(rows, 3);
    }

    fn count_star_over(input: LogicalOperator) -> LogicalOperator {
        use super::super::{AggregateOp, LogicalAggregateFunction};
        use crate::query::plan::AggregateExpr;
        LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![],
            aggregates: vec![AggregateExpr {
                function: LogicalAggregateFunction::Count,
                expression: None,
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
    fn comma_join_triangle_count_uses_kernel() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["V"]);
        let b = store.create_node(&["V"]);
        let c = store.create_node(&["V"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, c, "R");
        store.create_edge(c, a, "R");
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let mut planned = planner
            .plan(&LogicalPlan::new(count_star_over(hop2_then_close_join())))
            .unwrap();
        assert_eq!(planned.operator.name(), "TriangleCount");
        let out = planned.operator.next().unwrap().expect("one COUNT row");
        assert_eq!(out.row_count(), 1);
        assert_eq!(out.column(0).unwrap().get_int64(0), Some(3));
    }

    #[cfg(feature = "gql")]
    #[test]
    fn fair_card_gql_triangle_count_uses_kernel() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Person"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        store.create_edge(c, a, "KNOWS");
        let logical = crate::query::translators::gql::translate(
            "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c), (c)-[:KNOWS]->(a) RETURN count(*) AS n",
        )
        .unwrap();
        fn unwrap_agg(op: &LogicalOperator) -> &LogicalOperator {
            match op {
                LogicalOperator::Return(ret) => unwrap_agg(&ret.input),
                other => other,
            }
        }
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let planned = planner
            .plan(&LogicalPlan::new(unwrap_agg(&logical.root).clone()))
            .unwrap();
        assert_eq!(
            planned.operator.name(),
            "TriangleCount",
            "fair-card COUNT(*) must skip Leapfrog materialize"
        );
    }

    #[test]
    fn linear_triangle_count_uses_kernel() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["V"]);
        let b = store.create_node(&["V"]);
        let c = store.create_node(&["V"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, c, "R");
        store.create_edge(c, a, "R");
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let planned = planner
            .plan(&LogicalPlan::new(count_star_over(triangle_logical())))
            .unwrap();
        assert_eq!(planned.operator.name(), "TriangleCount");
    }

    #[test]
    fn labeled_triangle_chain_still_detects() {
        use crate::query::plan::FilterOp;
        let store = Arc::new(LpgStore::new().unwrap());
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let with_filter = LogicalOperator::Expand(ExpandOp {
            from_variable: "c".to_string(),
            to_variable: "a".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["R".to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: super::super::LogicalExpression::FunctionCall {
                    name: "hasLabel".into(),
                    args: vec![
                        super::super::LogicalExpression::Variable("b".into()),
                        super::super::LogicalExpression::Literal(
                            grafeo_common::types::Value::from("V"),
                        ),
                    ],
                    distinct: false,
                },
                input: Box::new(LogicalOperator::Expand(ExpandOp {
                    from_variable: "b".to_string(),
                    to_variable: "c".to_string(),
                    edge_variable: None,
                    direction: ExpandDirection::Outgoing,
                    edge_types: vec!["R".to_string()],
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(LogicalOperator::Expand(ExpandOp {
                        from_variable: "a".to_string(),
                        to_variable: "b".to_string(),
                        edge_variable: None,
                        direction: ExpandDirection::Outgoing,
                        edge_types: vec!["R".to_string()],
                        min_hops: 1,
                        max_hops: Some(1),
                        input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                            variable: "a".to_string(),
                            label: Some("V".to_string()),
                            input: None,
                        })),
                        path_alias: None,
                        path_mode: PathMode::Walk,
                        edge_predicate: None,
                        path_predicate: None,
                        path_search: crate::query::plan::PathSearch::All,
                    })),
                    path_alias: None,
                    path_mode: PathMode::Walk,
                    edge_predicate: None,
                    path_predicate: None,
                    path_search: crate::query::plan::PathSearch::All,
                })),
                pushdown_hint: None,
            })),
            path_alias: None,
            path_mode: PathMode::Walk,
            edge_predicate: None,
            path_predicate: None,
            path_search: crate::query::plan::PathSearch::All,
        });
        assert!(planner.is_directed_triangle_chain(&with_filter));
    }
}
