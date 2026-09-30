//! Join, union, and distinct planning.

use super::{
    ApplyOp, ApplyOperator, Arc, DistinctOp, Error, ExceptOp, ExpressionPredicate, FilterOperator,
    GraphStoreSearch, HashJoinOperator, IntersectOp, JoinKeySemantics, JoinOp,
    JoinSipExpandOperator, JoinType, LeapfrogJoinOperator, LogicalExpression, LogicalOperator,
    LogicalType, MultiWayJoinOp, NodeScanOp, Operator, OtherwiseOp, PathMode, PhysicalJoinType,
    ProjectExpr, ProjectOperator, Result, UnionOp, common,
};

impl super::Planner {
    /// Plans a JOIN operator.
    ///
    /// When join conditions reference shared variables, deduplicates the output
    /// columns by projecting out the right-side copies (whose values are equal
    /// to the left-side copies due to the join condition).
    pub(super) fn plan_join(&self, join: &JoinOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        if join
            .conditions
            .iter()
            .any(|condition| condition.semantics != JoinKeySemantics::Value)
        {
            return Err(Error::InvalidValue(
                "RDF term-identity join metadata cannot be planned by the LPG engine".to_string(),
            ));
        }
        if self.factorized_execution
            && let Some(planned) = self.try_plan_triangle_join(join)?
        {
            return Ok(planned);
        }
        if self.factorized_execution
            && let Some(planned) = self.try_plan_join_sip(join)?
        {
            return Ok(planned);
        }

        let (left_op, left_columns) =
            self.plan_operator_preserving_expand_constraints(&join.left)?;
        let (right_op, right_columns) =
            self.plan_operator_preserving_expand_constraints(&join.right)?;

        // Full column list before deduplication (HashJoin produces all columns)
        let mut all_columns = left_columns.clone();
        all_columns.extend(right_columns.clone());

        // Convert join type
        let physical_join_type = match join.join_type {
            JoinType::Inner => PhysicalJoinType::Inner,
            JoinType::Left => PhysicalJoinType::Left,
            JoinType::Right => PhysicalJoinType::Right,
            JoinType::Full => PhysicalJoinType::Full,
            JoinType::Cross => PhysicalJoinType::Cross,
            JoinType::Semi => PhysicalJoinType::Semi,
            JoinType::Anti => PhysicalJoinType::Anti,
        };

        // Build key columns from join conditions
        let (probe_keys, build_keys): (Vec<usize>, Vec<usize>) = if join.conditions.is_empty() {
            // Cross join - no keys
            (vec![], vec![])
        } else {
            join.conditions
                .iter()
                .map(|cond| {
                    Ok((
                        self.expression_to_column(&cond.left, &left_columns)?,
                        self.expression_to_column(&cond.right, &right_columns)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip()
        };

        let output_schema = self.derive_schema_from_columns(&all_columns);

        let join_op: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left_op,
            right_op,
            probe_keys,
            build_keys,
            physical_join_type,
            output_schema,
        ));

        // Deduplicate shared variable columns: right-side columns that also
        // appear on the left are redundant (the join guarantees equality).
        // Skip deduplication for CROSS joins (no equality guarantee).
        // For RIGHT/FULL joins, use COALESCE to prefer the non-NULL side.
        if matches!(join.join_type, JoinType::Cross) || join.conditions.is_empty() {
            return Ok((join_op, all_columns));
        }
        let needs_coalesce = matches!(join.join_type, JoinType::Right | JoinType::Full);

        let left_count = left_columns.len();
        let mut proj_exprs: Vec<ProjectExpr> = Vec::new();
        let mut deduped_columns: Vec<String> = Vec::new();
        let mut has_duplicates = false;

        for (li, col_name) in left_columns.iter().enumerate() {
            if needs_coalesce {
                if let Some(ri) = right_columns.iter().position(|c| c == col_name) {
                    proj_exprs.push(ProjectExpr::Coalesce {
                        first: li,
                        second: left_count + ri,
                    });
                    has_duplicates = true;
                } else {
                    proj_exprs.push(ProjectExpr::Column(li));
                }
            } else {
                proj_exprs.push(ProjectExpr::Column(li));
            }
            deduped_columns.push(col_name.clone());
        }

        for (ri, col_name) in right_columns.iter().enumerate() {
            if !deduped_columns.contains(col_name) {
                proj_exprs.push(ProjectExpr::Column(left_count + ri));
                deduped_columns.push(col_name.clone());
            } else {
                has_duplicates = true;
            }
        }

        if !has_duplicates {
            return Ok((join_op, deduped_columns));
        }

        let proj_schema = self.derive_schema_from_columns(&deduped_columns);
        let operator: Box<dyn Operator> =
            Box::new(ProjectOperator::new(join_op, proj_exprs, proj_schema));

        Ok((operator, deduped_columns))
    }

    /// Plans a multi-way leapfrog join (WCOJ) operator.
    ///
    /// Materializes each input into a sorted trie and uses `LeapfrogJoinOperator`
    /// for worst-case optimal intersection.
    pub(super) fn plan_multi_way_join(
        &self,
        mwj: &MultiWayJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        if mwj.inputs.len() < 3 {
            return Err(Error::InvalidValue(
                "LPG MultiWayJoin requires at least three inputs".to_string(),
            ));
        }

        let shared_variables = mwj
            .shared_variables
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        if shared_variables.is_empty() || shared_variables.len() != mwj.shared_variables.len() {
            return Err(Error::InvalidValue(
                "LPG MultiWayJoin requires non-empty, unique shared variables".to_string(),
            ));
        }

        let mut condition_variables = std::collections::HashSet::new();
        for condition in &mwj.conditions {
            if condition.semantics != JoinKeySemantics::Value {
                return Err(Error::InvalidValue(
                    "RDF term-identity join metadata cannot be planned by the LPG engine"
                        .to_string(),
                ));
            }
            let variable = match (&condition.left, &condition.right) {
                (LogicalExpression::Variable(left), LogicalExpression::Variable(right))
                    if left == right =>
                {
                    left
                }
                _ => {
                    return Err(Error::InvalidValue(
                        "LPG MultiWayJoin requires same-named variable equality conditions"
                            .to_string(),
                    ));
                }
            };
            if !condition_variables.insert(variable.clone()) {
                return Err(Error::InvalidValue(
                    "LPG MultiWayJoin requires one unique condition per shared variable"
                        .to_string(),
                ));
            }
        }
        if condition_variables != shared_variables {
            return Err(Error::InvalidValue(
                "LPG MultiWayJoin conditions must exactly declare every shared variable"
                    .to_string(),
            ));
        }

        for input in &mwj.inputs {
            for shared_variable in &mwj.shared_variables {
                match Self::multiway_key_type(input, shared_variable) {
                    Ok(Some(LogicalType::Node | LogicalType::Edge | LogicalType::Int64))
                    | Ok(None) => {}
                    Ok(Some(_)) | Err(()) => {
                        return Err(Error::InvalidValue(format!(
                            "LPG MultiWayJoin key '{shared_variable}' must have physical type Node, Edge, or Int64"
                        )));
                    }
                }
            }
        }

        // Plan each input, collecting operators and their column lists
        let mut input_ops: Vec<Box<dyn Operator>> = Vec::with_capacity(mwj.inputs.len());
        let mut input_columns: Vec<Vec<String>> = Vec::with_capacity(mwj.inputs.len());

        for input in &mwj.inputs {
            let (op, cols) = self.plan_operator(input)?;
            input_ops.push(op);
            input_columns.push(cols);
        }

        let mut public_occurrences = std::collections::HashMap::<&str, usize>::new();
        for columns in &input_columns {
            let mut input_names = std::collections::HashSet::new();
            for column in columns {
                if !input_names.insert(column.as_str()) {
                    return Err(Error::InvalidValue(format!(
                        "LPG MultiWayJoin input has duplicate public column '{column}'"
                    )));
                }
            }
            for column in input_names {
                *public_occurrences.entry(column).or_default() += 1;
            }
        }
        let actual_overlaps = public_occurrences
            .into_iter()
            .filter_map(|(column, count)| (count >= 2).then_some(column))
            .collect::<std::collections::HashSet<_>>();
        let declared_overlaps = shared_variables
            .iter()
            .map(String::as_str)
            .collect::<std::collections::HashSet<_>>();
        if actual_overlaps != declared_overlaps {
            return Err(Error::InvalidValue(
                "LPG MultiWayJoin declared shared variables must exactly match actual public column overlap"
                    .to_string(),
            ));
        }

        let mut participation = std::collections::HashMap::<&str, usize>::new();
        for cols in &input_columns {
            let mut input_participates = false;
            for shared_variable in &mwj.shared_variables {
                let occurrences = cols
                    .iter()
                    .filter(|column| *column == shared_variable)
                    .count();
                if occurrences > 1 {
                    return Err(Error::InvalidValue(format!(
                        "LPG MultiWayJoin key '{shared_variable}' must occur exactly once per input"
                    )));
                }
                if occurrences == 1 {
                    input_participates = true;
                    *participation.entry(shared_variable).or_default() += 1;
                }
            }
            if !input_participates {
                return Err(Error::InvalidValue(
                    "Every LPG MultiWayJoin input must carry at least one shared variable"
                        .to_string(),
                ));
            }
        }
        for shared_variable in &mwj.shared_variables {
            if participation
                .get(shared_variable.as_str())
                .copied()
                .unwrap_or(0)
                < 2
            {
                return Err(Error::InvalidValue(format!(
                    "LPG MultiWayJoin key '{shared_variable}' must be shared by at least two inputs"
                )));
            }
        }

        // Build the global-variable alignment:
        // shared_var_cols[input_idx][global_var_idx] = Some(col_in_input) | None
        // This ensures every input's trie is built in the same global variable order,
        // enabling the recursive multi-level leapfrog to correctly intersect ragged
        // variable sets (e.g. triangle R1(a,b), R2(b,c), R3(c,a)).
        let mut shared_var_cols: Vec<Vec<Option<usize>>> = Vec::with_capacity(mwj.inputs.len());
        for cols in &input_columns {
            let alignment: Vec<Option<usize>> = mwj
                .shared_variables
                .iter()
                .map(|shared_var| cols.iter().position(|c| c == shared_var))
                .collect();
            shared_var_cols.push(alignment);
        }

        // Build combined output columns: shared variables first (deduplicated),
        // then remaining columns from each input
        let mut output_columns: Vec<String> = mwj.shared_variables.clone();
        let mut output_column_mapping: Vec<(usize, usize)> = Vec::new();

        // Map shared variables from the first input that has them
        for shared_var in &mwj.shared_variables {
            let mut found = false;
            for (input_idx, cols) in input_columns.iter().enumerate() {
                if let Some(col_idx) = cols.iter().position(|c| c == shared_var) {
                    output_column_mapping.push((input_idx, col_idx));
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(Error::Internal(format!(
                    "Shared variable '{}' not found in any input",
                    shared_var
                )));
            }
        }

        // Add non-shared columns from each input
        for (input_idx, cols) in input_columns.iter().enumerate() {
            for (col_idx, col_name) in cols.iter().enumerate() {
                if !mwj.shared_variables.contains(col_name) {
                    output_columns.push(col_name.clone());
                    output_column_mapping.push((input_idx, col_idx));
                }
            }
        }

        let output_schema = self.derive_schema_from_columns(&output_columns);

        let operator: Box<dyn Operator> = Box::new(LeapfrogJoinOperator::new(
            input_ops,
            shared_var_cols,
            output_schema,
            output_column_mapping,
        ));

        Ok((operator, output_columns))
    }

    /// `expand-chain ⋈ scan` on the last hop variable → collect scan ids, SIP.
    /// `(a)-[]->(b)-[]->(c) ⋈ (c)-[]->(a)` → LeapfrogExpand close.
    fn try_plan_triangle_join(
        &self,
        join: &JoinOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        if !matches!(join.join_type, JoinType::Inner) {
            return Ok(None);
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
            _ => return Ok(None),
        };
        let two = Self::collect_expand_chain(two_logical);
        let close = Self::collect_expand_chain(close_logical)[0];
        if two[1].from_variable != two[0].to_variable {
            return Ok(None);
        }
        if close.from_variable != two[1].to_variable || close.to_variable != two[0].from_variable {
            return Ok(None);
        }
        if !Self::has_exact_same_name_value_conditions(
            join,
            [&two[0].from_variable, &two[1].to_variable],
        ) {
            return Ok(None);
        }
        let two_hop = LogicalOperator::Expand(two[1].clone());
        let mut closer = close.clone();
        closer.input = Box::new(two_hop);
        Ok(Some(self.plan_leapfrog_triangle(
            &LogicalOperator::Expand(closer),
        )?))
    }

    fn try_plan_join_sip(&self, join: &JoinOp) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        if !matches!(join.join_type, JoinType::Inner | JoinType::Semi) {
            return Ok(None);
        }
        let [cond] = join.conditions.as_slice() else {
            return Ok(None);
        };
        let (LogicalExpression::Variable(lv), LogicalExpression::Variable(rv)) =
            (&cond.left, &cond.right)
        else {
            return Ok(None);
        };
        if lv != rv {
            return Ok(None);
        }

        let left_chain = Self::qualified_expand_chain(&join.left);
        let right_chain = Self::qualified_expand_chain(&join.right);
        let (chain_log, scan_log, chain_is_left) = match (left_chain, right_chain) {
            (Some((hops, _)), None) if hops >= 1 => (&join.left, &join.right, true),
            (None, Some((hops, _))) if hops >= 1 => (&join.right, &join.left, false),
            _ => return Ok(None),
        };
        if matches!(join.join_type, JoinType::Semi) && !chain_is_left {
            return Ok(None);
        }

        let Some(scan) = Self::leaf_node_scan(scan_log) else {
            return Ok(None);
        };
        let expands = Self::collect_expand_chain(chain_log);
        let Some(last) = expands.last() else {
            return Ok(None);
        };
        let last_to = last.to_variable.as_str();
        let scan_var = scan.variable.as_str();
        if last_to != lv || scan_var != rv {
            return Ok(None);
        }

        let hop = expands.len() - 1;
        let (build, build_cols) = self.plan_operator(scan_log)?;
        let build_col = build_cols
            .iter()
            .position(|c| c == scan_var)
            .ok_or_else(|| Error::Internal(format!("Variable '{scan_var}' not found")))?;
        let (lazy, cols) = self.plan_expand_chain_lazy(chain_log)?;
        Ok(Some((
            Box::new(JoinSipExpandOperator::new(build, build_col, lazy, hop)),
            cols,
        )))
    }

    pub(super) fn has_exact_same_name_value_conditions<'a>(
        join: &JoinOp,
        expected: impl IntoIterator<Item = &'a String>,
    ) -> bool {
        let expected = expected
            .into_iter()
            .map(String::as_str)
            .collect::<std::collections::HashSet<_>>();
        if join.conditions.len() != expected.len() {
            return false;
        }
        let actual = join
            .conditions
            .iter()
            .filter_map(|condition| match (&condition.left, &condition.right) {
                (LogicalExpression::Variable(left), LogicalExpression::Variable(right))
                    if condition.semantics == JoinKeySemantics::Value && left == right =>
                {
                    Some(left.as_str())
                }
                _ => None,
            })
            .collect::<std::collections::HashSet<_>>();
        actual == expected && actual.len() == join.conditions.len()
    }

    fn leaf_node_scan(op: &LogicalOperator) -> Option<&NodeScanOp> {
        match op {
            LogicalOperator::NodeScan(scan) if scan.input.is_none() => Some(scan),
            _ => None,
        }
    }

    /// A shortcut may rebuild this chain only when there are no wrappers to
    /// drop and the base scan remains an explicit child of the rebuilt plan.
    pub(super) fn qualified_expand_chain(op: &LogicalOperator) -> Option<(usize, &NodeScanOp)> {
        let mut current = op;
        let mut hops = 0;
        loop {
            match current {
                LogicalOperator::Expand(expand)
                    if expand.min_hops == 1
                        && expand.max_hops == Some(1)
                        && expand.path_alias.is_none()
                        && expand.edge_predicate.is_none()
                        && expand.path_predicate.is_none()
                        && expand.path_mode == PathMode::Walk
                        && expand.path_search == crate::query::plan::PathSearch::All =>
                {
                    hops += 1;
                    current = &expand.input;
                }
                LogicalOperator::NodeScan(scan) if hops > 0 && scan.input.is_none() => {
                    return Some((hops, scan));
                }
                _ => return None,
            }
        }
    }

    /// General factorized planning skips `HasLabel` wrappers while collecting
    /// an expand spine. Remove each such wrapper from the fallback logical
    /// shape before planning, then install it exactly once against the complete
    /// output schema. PROFILE already disables fusion and keeps the original
    /// logical ownership, so it bypasses this rewrite.
    pub(super) fn plan_operator_preserving_expand_constraints(
        &self,
        op: &LogicalOperator,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        if !self.factorized_execution || self.profiling.get() {
            return self.plan_operator(op);
        }

        let mut predicates = Vec::new();
        let stripped = Self::strip_linear_haslabel_predicates(op, &mut predicates);
        if predicates.is_empty() {
            return self.plan_operator(op);
        }
        let (mut operator, columns) = self.plan_operator(&stripped)?;

        let variable_columns = columns
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index))
            .collect::<std::collections::HashMap<_, _>>();
        for predicate in &predicates {
            let filter_expr = self.convert_expression(predicate)?;
            let physical_predicate = ExpressionPredicate::new(
                filter_expr,
                variable_columns.clone(),
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone());
            operator = Box::new(FilterOperator::new(operator, Box::new(physical_predicate)));
        }
        Ok((operator, columns))
    }

    fn strip_linear_haslabel_predicates(
        op: &LogicalOperator,
        predicates: &mut Vec<LogicalExpression>,
    ) -> LogicalOperator {
        match op {
            LogicalOperator::Expand(expand) => {
                let mut stripped = expand.clone();
                stripped.input = Box::new(Self::strip_linear_haslabel_predicates(
                    &expand.input,
                    predicates,
                ));
                LogicalOperator::Expand(stripped)
            }
            LogicalOperator::Filter(filter) => {
                if Self::is_haslabel_only_filter(&filter.predicate) {
                    predicates.push(filter.predicate.clone());
                    Self::strip_linear_haslabel_predicates(&filter.input, predicates)
                } else {
                    let mut stripped = filter.clone();
                    stripped.input = Box::new(Self::strip_linear_haslabel_predicates(
                        &filter.input,
                        predicates,
                    ));
                    LogicalOperator::Filter(stripped)
                }
            }
            LogicalOperator::NodeScan(scan) => {
                let mut stripped = scan.clone();
                stripped.input = scan.input.as_ref().map(|input| {
                    Box::new(Self::strip_linear_haslabel_predicates(input, predicates))
                });
                LogicalOperator::NodeScan(stripped)
            }
            _ => op.clone(),
        }
    }

    fn multiway_key_type(
        op: &LogicalOperator,
        variable: &str,
    ) -> std::result::Result<Option<LogicalType>, ()> {
        match op {
            LogicalOperator::NodeScan(scan) => {
                if scan.variable == variable {
                    Ok(Some(LogicalType::Node))
                } else if let Some(input) = &scan.input {
                    Self::multiway_key_type(input, variable)
                } else {
                    Ok(None)
                }
            }
            LogicalOperator::EdgeScan(scan) => {
                if scan.variable == variable {
                    Ok(Some(LogicalType::Edge))
                } else if let Some(input) = &scan.input {
                    Self::multiway_key_type(input, variable)
                } else {
                    Ok(None)
                }
            }
            LogicalOperator::Expand(expand) => {
                if expand.to_variable == variable || expand.from_variable == variable {
                    Ok(Some(LogicalType::Node))
                } else if expand.edge_variable.as_deref() == Some(variable) {
                    Ok(Some(LogicalType::Edge))
                } else {
                    Self::multiway_key_type(&expand.input, variable)
                }
            }
            LogicalOperator::Filter(filter) => Self::multiway_key_type(&filter.input, variable),
            LogicalOperator::Project(project) => {
                let projection = project.projections.iter().find(|projection| {
                    super::output_column_name(projection.alias.as_deref(), &projection.expression)
                        == variable
                });
                if let Some(projection) = projection {
                    Self::multiway_expression_type(&projection.expression, &project.input).map(Some)
                } else if project.pass_through_input {
                    Self::multiway_key_type(&project.input, variable)
                } else {
                    Ok(None)
                }
            }
            LogicalOperator::Empty => Ok(None),
            _ => Err(()),
        }
    }

    fn multiway_expression_type(
        expression: &LogicalExpression,
        input: &LogicalOperator,
    ) -> std::result::Result<LogicalType, ()> {
        match expression {
            LogicalExpression::Literal(value) => Ok(super::value_to_logical_type(value)),
            LogicalExpression::Variable(variable) => {
                Self::multiway_key_type(input, variable)?.ok_or(())
            }
            _ => Err(()),
        }
    }

    /// Extracts a column index from an expression.
    pub(super) fn expression_to_column(
        &self,
        expr: &LogicalExpression,
        columns: &[String],
    ) -> Result<usize> {
        match expr {
            LogicalExpression::Variable(name) => columns
                .iter()
                .position(|c| c == name)
                .ok_or_else(|| Error::Internal(format!("Variable '{}' not found", name))),
            _ => Err(Error::Internal(
                "Only variables supported in join conditions".to_string(),
            )),
        }
    }

    /// Plans a UNION operator.
    pub(super) fn plan_union(&self, union: &UnionOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let mut inputs = Vec::with_capacity(union.inputs.len());
        let mut columns = Vec::new();

        for (i, input) in union.inputs.iter().enumerate() {
            let (op, cols) = self.plan_operator(input)?;
            if i == 0 {
                columns = cols;
            }
            inputs.push(op);
        }

        let schema = self.derive_schema_from_columns(&columns);
        common::build_union(inputs, columns, schema)
    }

    /// Plans a DISTINCT operator.
    pub(super) fn plan_distinct(
        &self,
        distinct: &DistinctOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&distinct.input)?;
        let schema = self.derive_schema_from_columns(&columns);
        Ok(common::build_distinct(
            input_op,
            columns,
            distinct.columns.as_deref(),
            schema,
        ))
    }

    /// Plans an EXCEPT operator.
    pub(super) fn plan_except(
        &self,
        except: &ExceptOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, columns) = self.plan_operator(&except.left)?;
        let (right_op, _) = self.plan_operator(&except.right)?;
        let schema = self.derive_schema_from_columns(&columns);
        Ok(common::build_except(
            left_op, right_op, columns, except.all, schema,
        ))
    }

    /// Plans an INTERSECT operator.
    pub(super) fn plan_intersect(
        &self,
        intersect: &IntersectOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, columns) = self.plan_operator(&intersect.left)?;
        let (right_op, _) = self.plan_operator(&intersect.right)?;
        let schema = self.derive_schema_from_columns(&columns);
        Ok(common::build_intersect(
            left_op,
            right_op,
            columns,
            intersect.all,
            schema,
        ))
    }

    /// Plans an OTHERWISE operator.
    pub(super) fn plan_otherwise(
        &self,
        otherwise: &OtherwiseOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, columns) = self.plan_operator(&otherwise.left)?;
        let (right_op, _) = self.plan_operator(&otherwise.right)?;
        Ok(common::build_otherwise(left_op, right_op, columns))
    }

    /// Plans an APPLY (lateral join) operator.
    ///
    /// When `shared_variables` is non-empty, creates a correlated Apply that
    /// injects outer row values into the inner plan via [`ParameterState`].
    pub(super) fn plan_apply(&self, apply: &ApplyOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (outer_op, outer_columns) = self.plan_operator(&apply.input)?;

        if apply.shared_variables.is_empty() {
            // Uncorrelated Apply
            let (inner_op, inner_columns) = self.plan_operator(&apply.subplan)?;
            // Inner subquery RETURN materializes values (PropertyAccess, NodeResolve,
            // aggregates, etc.), so all its output columns are scalar.
            for col in &inner_columns {
                self.scalar_columns.borrow_mut().insert(col.clone());
            }
            let inner_col_count = inner_columns.len();
            let mut columns = outer_columns;
            columns.extend(inner_columns);
            let mut op = ApplyOperator::new(outer_op, inner_op);
            if apply.optional {
                op = op.with_optional(inner_col_count);
            }
            return Ok((Box::new(op), columns));
        }

        // Expand wildcard: WITH * imports all outer-scope variables
        let shared_vars = if apply.shared_variables.len() == 1 && apply.shared_variables[0] == "*" {
            outer_columns.clone()
        } else {
            apply.shared_variables.clone()
        };

        // Correlated Apply: create shared ParameterState
        let param_state = std::sync::Arc::new(
            grafeo_core::execution::operators::ParameterState::new(shared_vars.clone()),
        );

        // Find column indices for the shared variables in outer columns
        let param_col_indices: Vec<usize> = shared_vars
            .iter()
            .map(|var| outer_columns.iter().position(|c| c == var).unwrap_or(0))
            .collect();

        // Set the parameter state so the inner plan's ParameterScan can find it
        *self.correlated_param_state.borrow_mut() = Some(std::sync::Arc::clone(&param_state));

        let (inner_op, inner_columns) = self.plan_operator(&apply.subplan)?;

        // Clear the parameter state after planning the inner operator
        *self.correlated_param_state.borrow_mut() = None;

        // Inner subquery RETURN materializes values, so register as scalar
        // to prevent the outer RETURN from misinterpreting them as node IDs.
        for col in &inner_columns {
            self.scalar_columns.borrow_mut().insert(col.clone());
        }

        // Build correlated Apply
        let mut columns = outer_columns;
        let inner_col_count = inner_columns.len();
        columns.extend(inner_columns);
        let mut op =
            ApplyOperator::new_correlated(outer_op, inner_op, param_state, param_col_indices);
        if apply.optional {
            op = op.with_optional(inner_col_count);
        }
        Ok((Box::new(op), columns))
    }
}

#[cfg(all(test, feature = "lpg"))]
mod join_sip_plan_tests {
    use super::super::{
        AggregateOp, BinaryOp, ExpandDirection, ExpandOp, FilterOp, GraphStoreSearch, JoinOp,
        JoinType, LogicalAggregateFunction, LogicalExpression, LogicalOperator, LogicalPlan,
        MultiWayJoinOp, NodeScanOp, PathMode, Planner,
    };
    use crate::query::plan::{
        AggregateExpr, JoinCondition, JoinKeySemantics, ProjectOp, Projection,
    };
    use grafeo_common::types::Value;
    use grafeo_core::execution::operators::{FilterOperator, HashAggregateOperator};
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

    fn property_equals(
        variable: &str,
        property: &str,
        value: Value,
        input: LogicalOperator,
    ) -> LogicalOperator {
        LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: variable.to_string(),
                    property: property.to_string(),
                }),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Literal(value)),
            },
            input: Box::new(input),
            pushdown_hint: None,
        })
    }

    fn value_condition(left: &str, right: &str) -> JoinCondition {
        JoinCondition {
            left: LogicalExpression::Variable(left.to_string()),
            right: LogicalExpression::Variable(right.to_string()),
            semantics: JoinKeySemantics::Value,
        }
    }

    fn triangle_join(two_hop: LogicalOperator, close: LogicalOperator) -> LogicalOperator {
        LogicalOperator::Join(JoinOp {
            left: Box::new(two_hop),
            right: Box::new(close),
            join_type: JoinType::Inner,
            conditions: vec![value_condition("a", "a"), value_condition("c", "c")],
        })
    }

    fn execute_row_count(
        planner: Planner,
        logical: LogicalOperator,
    ) -> (String, Vec<String>, usize) {
        let mut planned = planner.plan(&LogicalPlan::new(logical)).unwrap();
        let name = planned.operator.name().to_string();
        let columns = planned.columns().to_vec();
        let mut rows = 0;
        while let Some(chunk) = planned.operator.next().unwrap() {
            rows += chunk.row_count();
        }
        (name, columns, rows)
    }

    fn grouped_count(input: LogicalOperator) -> LogicalOperator {
        LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![LogicalExpression::Variable("a".to_string())],
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
    fn plans_join_sip_expand() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);
        let c = store.create_node(&["C"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, c, "R");
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let chain = LogicalOperator::Expand(ExpandOp {
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
                    label: Some("A".to_string()),
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
        let scan_c = LogicalOperator::NodeScan(NodeScanOp {
            variable: "c".to_string(),
            label: Some("C".to_string()),
            input: None,
        });
        let join = LogicalOperator::Join(JoinOp {
            left: Box::new(chain),
            right: Box::new(scan_c),
            join_type: JoinType::Inner,
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("c".to_string()),
                right: LogicalExpression::Variable("c".to_string()),
                semantics: crate::query::plan::JoinKeySemantics::Value,
            }],
        });
        let mut planned = planner.plan(&LogicalPlan::new(join)).unwrap();
        assert_eq!(planned.operator.name(), "JoinSipExpand");
        let mut rows = 0;
        while let Some(chunk) = planned.operator.next().unwrap() {
            rows += chunk.row_count();
        }
        assert_eq!(rows, 1, "a constrained NodeScan remains a valid SIP seed");
    }

    #[test]
    fn rejects_rdf_identity_metadata_in_lpg_planner() {
        let store = Arc::new(LpgStore::new().unwrap());
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let scan = |variable: &str| {
            LogicalOperator::NodeScan(NodeScanOp {
                variable: variable.to_string(),
                label: None,
                input: None,
            })
        };
        let join = LogicalOperator::Join(JoinOp {
            left: Box::new(scan("a")),
            right: Box::new(scan("b")),
            join_type: JoinType::Inner,
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("a".to_string()),
                right: LogicalExpression::Variable("b".to_string()),
                semantics: crate::query::plan::JoinKeySemantics::RdfTermIdentity,
            }],
        });

        let Err(error) = planner.plan(&LogicalPlan::new(join)) else {
            panic!("LPG planner unexpectedly accepted RDF term-identity metadata");
        };
        assert!(
            error
                .to_string()
                .contains("cannot be planned by the LPG engine"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn triangle_shortcut_does_not_discard_close_scan_label() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["V"]);
        let b = store.create_node(&["V"]);
        let c = store.create_node(&["V"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, c, "R");
        store.create_edge(c, a, "R");

        let logical = triangle_join(
            expand("b", "c", expand("a", "b", scan("a", None))),
            expand("c", "a", scan("c", Some("Required"))),
        );
        let (name, _, rows) =
            execute_row_count(Planner::new(store as Arc<dyn GraphStoreSearch>), logical);

        assert_ne!(name, "LeapfrogExpand");
        assert_eq!(rows, 0);
    }

    #[test]
    fn triangle_shortcut_does_not_discard_close_scan_input() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["V"]);
        let b = store.create_node(&["V"]);
        let c = store.create_node(&["V"]);
        store.create_node(&["Seed"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, c, "R");
        store.create_edge(c, a, "R");

        let close_scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: "c".to_string(),
            label: None,
            input: Some(Box::new(scan("seed", Some("Seed")))),
        });
        let logical = triangle_join(
            expand("b", "c", expand("a", "b", scan("a", Some("V")))),
            expand("c", "a", close_scan),
        );
        let (_, _, flat_rows) = execute_row_count(
            Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>)
                .with_factorized_execution(false),
            logical.clone(),
        );
        let (name, columns, rows) =
            execute_row_count(Planner::new(store as Arc<dyn GraphStoreSearch>), logical);

        assert_ne!(name, "LeapfrogExpand");
        assert!(columns.contains(&"seed".to_string()));
        assert_eq!(flat_rows, 3, "the unfused plan is the semantic oracle");
        assert_eq!(rows, flat_rows);
    }

    #[test]
    fn triangle_shortcut_does_not_discard_chain_root_filter() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["V"]);
        let b = store.create_node(&["V"]);
        let c = store.create_node(&["V"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, c, "R");
        store.create_edge(c, a, "R");

        let two_hop = has_label(
            "a",
            "Required",
            expand("b", "c", expand("a", "b", scan("a", None))),
        );
        let logical = triangle_join(two_hop, expand("c", "a", scan("c", None)));
        let (name, _, rows) =
            execute_row_count(Planner::new(store as Arc<dyn GraphStoreSearch>), logical);

        assert_ne!(name, "LeapfrogExpand");
        assert_eq!(rows, 0);
    }

    #[test]
    fn sip_requires_same_named_owned_join_key() {
        let store = Arc::new(LpgStore::new().unwrap());
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let logical = LogicalOperator::Join(JoinOp {
            left: Box::new(expand("b", "c", expand("a", "b", scan("a", None)))),
            right: Box::new(scan("z", None)),
            join_type: JoinType::Inner,
            conditions: vec![value_condition("c", "z")],
        });

        let planned = planner.plan(&LogicalPlan::new(logical)).unwrap();
        assert_ne!(planned.operator.name(), "JoinSipExpand");
        assert!(planned.columns().contains(&"z".to_string()));
    }

    #[test]
    fn sip_does_not_discard_chain_side_haslabel_filter() {
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
        let logical = LogicalOperator::Join(JoinOp {
            left: Box::new(chain),
            right: Box::new(scan("c", Some("C"))),
            join_type: JoinType::Inner,
            conditions: vec![value_condition("c", "c")],
        });
        let (name, _, rows) =
            execute_row_count(Planner::new(store as Arc<dyn GraphStoreSearch>), logical);

        assert_ne!(name, "JoinSipExpand");
        assert_eq!(rows, 1);
    }

    #[test]
    fn sip_falls_back_for_close_side_filter_and_preserves_base_scan_label() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);
        let keep = store.create_node(&["Keep"]);
        let drop = store.create_node(&["Drop"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, keep, "R");
        store.create_edge(b, drop, "R");
        store.set_node_property(keep, "selected", Value::Bool(true));
        store.set_node_property(drop, "selected", Value::Bool(false));

        let logical = LogicalOperator::Join(JoinOp {
            left: Box::new(expand("b", "c", expand("a", "b", scan("a", Some("A"))))),
            right: Box::new(property_equals(
                "c",
                "selected",
                Value::Bool(true),
                scan("c", None),
            )),
            join_type: JoinType::Inner,
            conditions: vec![value_condition("c", "c")],
        });
        let (_, _, flat_rows) = execute_row_count(
            Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>)
                .with_factorized_execution(false),
            logical.clone(),
        );
        let (name, _, rows) =
            execute_row_count(Planner::new(store as Arc<dyn GraphStoreSearch>), logical);

        assert_eq!(flat_rows, 1, "the unfused plan is the semantic oracle");
        assert_eq!(
            rows, flat_rows,
            "fallback must preserve the build-side filter"
        );
        assert_ne!(name, "JoinSipExpand");
    }

    #[test]
    fn aggregate_fallback_physically_owns_haslabel_exactly_once() {
        let store = Arc::new(LpgStore::new().unwrap());
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let chain = has_label(
            "c",
            "Keep",
            expand("b", "c", expand("a", "b", scan("a", Some("A")))),
        );
        let planned = planner
            .plan(&LogicalPlan::new(grouped_count(chain)))
            .unwrap();

        let aggregate = planned
            .operator
            .into_any()
            .downcast::<HashAggregateOperator>()
            .expect("grouped COUNT must use HashAggregate");
        let (child, _, _) = (*aggregate).into_parts();
        let filter = child
            .into_any()
            .downcast::<FilterOperator>()
            .expect("fallback must retain one physical HasLabel filter");
        let (filter_child, _) = (*filter).into_parts().expect("qualified filter admission");
        assert_ne!(
            filter_child.name(),
            "Filter",
            "the same HasLabel predicate must not be installed twice"
        );
    }

    #[test]
    fn aggregate_fallback_profiles_one_haslabel_owner() {
        let store = Arc::new(LpgStore::new().unwrap());
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let logical = LogicalPlan::new(grouped_count(has_label(
            "c",
            "Keep",
            expand("b", "c", expand("a", "b", scan("a", Some("A")))),
        )));

        let (_, entries) = planner.plan_profiled(&logical).unwrap();
        assert_eq!(
            entries
                .iter()
                .filter(|entry| entry.name == "Filter")
                .count(),
            1,
            "PROFILE must attribute the label predicate to one physical owner"
        );
        let mut entries = entries.into_iter();
        crate::query::profile::build_profile_tree(&logical.root, &mut entries);
        assert!(entries.next().is_none());
    }

    #[test]
    fn unresolved_join_condition_is_an_error_not_a_cross_join() {
        let store = Arc::new(LpgStore::new().unwrap());
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let logical = LogicalOperator::Join(JoinOp {
            left: Box::new(scan("a", None)),
            right: Box::new(scan("b", None)),
            join_type: JoinType::Inner,
            conditions: vec![value_condition("missing", "b")],
        });

        let error = planner
            .plan(&LogicalPlan::new(logical))
            .err()
            .expect("an unresolved declared condition must fail planning");
        assert!(error.to_string().contains("Variable 'missing' not found"));
    }

    fn projected_constant(variable: &str, value: Value) -> LogicalOperator {
        projected_constants(vec![(variable, value)])
    }

    fn projected_constants(values: Vec<(&str, Value)>) -> LogicalOperator {
        LogicalOperator::Project(ProjectOp {
            projections: values
                .into_iter()
                .map(|(variable, value)| Projection {
                    expression: LogicalExpression::Literal(value),
                    alias: Some(variable.to_string()),
                })
                .collect(),
            input: Box::new(LogicalOperator::Empty),
            pass_through_input: false,
        })
    }

    fn direct_multiway(
        inputs: Vec<LogicalOperator>,
        conditions: Vec<JoinCondition>,
        shared_variables: Vec<&str>,
    ) -> LogicalOperator {
        LogicalOperator::MultiWayJoin(MultiWayJoinOp {
            inputs,
            conditions,
            shared_variables: shared_variables.into_iter().map(str::to_string).collect(),
        })
    }

    #[test]
    fn multiway_rejects_duplicate_shared_keys() {
        let store = Arc::new(LpgStore::new().unwrap());
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let logical = direct_multiway(
            vec![scan("k", None), scan("k", None), scan("k", None)],
            vec![value_condition("k", "k")],
            vec!["k", "k"],
        );

        let error = planner
            .plan(&LogicalPlan::new(logical))
            .err()
            .expect("duplicate shared variables must fail planning");
        assert!(error.to_string().contains("unique shared variables"));
    }

    #[test]
    fn multiway_rejects_duplicate_declared_conditions() {
        let store = Arc::new(LpgStore::new().unwrap());
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let logical = direct_multiway(
            vec![scan("k", None), scan("k", None), scan("k", None)],
            vec![value_condition("k", "k"), value_condition("k", "k")],
            vec!["k"],
        );

        let error = planner
            .plan(&LogicalPlan::new(logical))
            .err()
            .expect("duplicate conditions must fail planning");
        assert!(error.to_string().contains("one unique condition"));
    }

    #[test]
    fn multiway_rejects_duplicate_key_columns_within_an_input() {
        let store = Arc::new(LpgStore::new().unwrap());
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let duplicate = LogicalOperator::Project(ProjectOp {
            projections: vec![
                Projection {
                    expression: LogicalExpression::Variable("k".to_string()),
                    alias: Some("k".to_string()),
                },
                Projection {
                    expression: LogicalExpression::Variable("k".to_string()),
                    alias: Some("k".to_string()),
                },
            ],
            input: Box::new(scan("k", None)),
            pass_through_input: false,
        });
        let logical = direct_multiway(
            vec![duplicate, scan("k", None), scan("k", None)],
            vec![value_condition("k", "k")],
            vec!["k"],
        );

        let error = planner
            .plan(&LogicalPlan::new(logical))
            .err()
            .expect("duplicate key columns must fail planning");
        assert!(error.to_string().contains("duplicate public column 'k'"));
    }

    #[test]
    fn multiway_rejects_undeclared_public_column_overlap() {
        let store = Arc::new(LpgStore::new().unwrap());
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let logical = direct_multiway(
            vec![
                projected_constants(vec![("k", Value::Int64(7)), ("extra", Value::Int64(1))]),
                projected_constants(vec![("k", Value::Int64(7)), ("extra", Value::Int64(2))]),
                projected_constant("k", Value::Int64(7)),
            ],
            vec![value_condition("k", "k")],
            vec!["k"],
        );

        let error = planner
            .plan(&LogicalPlan::new(logical))
            .err()
            .expect("undeclared public-column overlap must fail planning");
        assert!(error.to_string().contains("actual public column overlap"));
    }

    #[test]
    fn multiway_rejects_duplicate_non_key_public_column_within_input() {
        let store = Arc::new(LpgStore::new().unwrap());
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let logical = direct_multiway(
            vec![
                projected_constants(vec![
                    ("k", Value::Int64(7)),
                    ("extra", Value::Int64(1)),
                    ("extra", Value::Int64(2)),
                ]),
                projected_constant("k", Value::Int64(7)),
                projected_constant("k", Value::Int64(7)),
            ],
            vec![value_condition("k", "k")],
            vec!["k"],
        );

        let error = planner
            .plan(&LogicalPlan::new(logical))
            .err()
            .expect("duplicate public output names must fail planning");
        assert!(
            error
                .to_string()
                .contains("duplicate public column 'extra'")
        );
    }

    #[test]
    fn multiway_rejects_unsupported_string_key_before_execution() {
        let store = Arc::new(LpgStore::new().unwrap());
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let logical = direct_multiway(
            vec![
                projected_constant("k", Value::String("same".into())),
                projected_constant("k", Value::String("same".into())),
                projected_constant("k", Value::String("same".into())),
            ],
            vec![value_condition("k", "k")],
            vec!["k"],
        );

        let error = planner
            .plan(&LogicalPlan::new(logical))
            .err()
            .expect("unsupported physical key types must fail planning");
        assert!(error.to_string().contains("Node, Edge, or Int64"));
    }

    #[test]
    fn multiway_accepts_supported_int64_key() {
        let store = Arc::new(LpgStore::new().unwrap());
        let planner = Planner::new(store as Arc<dyn GraphStoreSearch>);
        let logical = direct_multiway(
            vec![
                projected_constant("k", Value::Int64(7)),
                projected_constant("k", Value::Int64(7)),
                projected_constant("k", Value::Int64(7)),
            ],
            vec![value_condition("k", "k")],
            vec!["k"],
        );

        let (name, columns, rows) = execute_row_count(planner, logical);
        assert_eq!(name, "LeapfrogJoin");
        assert_eq!(columns, vec!["k"]);
        assert_eq!(rows, 1);
    }
}
