//! Makes your queries faster without changing their meaning.
//!
//! The optimizer transforms logical plans to run more efficiently:
//!
//! | Optimization | What it does |
//! | ------------ | ------------ |
//! | Filter Pushdown | Moves `WHERE` clauses closer to scans - filter early, process less |
//! | Join Reordering | Picks the best order to join tables using the DPccp algorithm |
//! | Predicate Simplification | Folds constants like `1 + 1` into `2` |
//!
//! The optimizer uses [`CostModel`] and [`CardinalityEstimator`] to predict
//! how expensive different plans are, then picks the cheapest.

pub mod cardinality;
pub mod cost;
pub mod join_order;

pub use cardinality::{
    CardinalityEstimator, ColumnStats, EstimationLog, SelectivityConfig, TableStats,
};
pub use cost::{Cost, CostModel};
pub use join_order::{BitSet, DPccp, JoinGraph, JoinGraphBuilder, JoinPlan};

use crate::query::plan::{
    BinaryOp, FilterOp, JoinCondition, JoinKeySemantics, LogicalExpression, LogicalOperator,
    LogicalPlan, MultiWayJoinOp, PathMode, PathSearch,
};
use crate::query::planner::common::output_column_name;
use grafeo_common::grafeo_debug_span;
use grafeo_common::utils::error::Result;
use std::collections::{HashMap, HashSet};

/// Information about a join condition for join reordering.
#[derive(Debug, Clone)]
struct JoinInfo {
    left_relation: usize,
    right_relation: usize,
    left_var: String,
    right_var: String,
    left_expr: LogicalExpression,
    right_expr: LogicalExpression,
    semantics: JoinKeySemantics,
}

/// A column required by the query, used for projection pushdown.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RequiredColumn {
    /// A variable (node, edge, or path binding)
    Variable(String),
    /// A specific property of a variable
    Property(String, String),
}

/// How a candidate expand output can affect a distinct consumer.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DistinctPathLineage {
    /// A value invariant for one input row (including source-side columns).
    Stable,
    /// A deterministic value derived from the expanded target.
    Endpoint,
    /// A value that can distinguish walks, edges, or paths.
    Tainted,
}

/// Synthetic key for an anonymous edge/path column emitted by an Expand.
/// It cannot be addressed by a query expression, but it still participates in
/// whole-row DISTINCT and therefore must remain visible to the safety check.
const ANONYMOUS_PATH_COLUMN: &str = "\0grafeo::anonymous_path_column";

/// Transforms logical plans for faster execution.
///
/// Create with [`new()`](Self::new), then call [`optimize()`](Self::optimize).
/// Use the builder methods to enable/disable specific optimizations.
pub struct Optimizer {
    /// Whether to enable filter pushdown.
    enable_filter_pushdown: bool,
    /// Whether to enable join reordering.
    enable_join_reorder: bool,
    /// Whether to enable projection pushdown.
    enable_projection_pushdown: bool,
    /// Cost model for estimation.
    cost_model: CostModel,
    /// Cardinality estimator.
    card_estimator: CardinalityEstimator,
}

impl Optimizer {
    /// Creates a new optimizer with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            enable_filter_pushdown: true,
            enable_join_reorder: true,
            enable_projection_pushdown: true,
            cost_model: CostModel::new(),
            card_estimator: CardinalityEstimator::new(),
        }
    }

    /// Creates an optimizer with cardinality estimates from the store's statistics.
    ///
    /// Pre-populates the cardinality estimator with per-label row counts and
    /// edge type fanout. Feeds per-edge-type degree stats, label cardinalities,
    /// and graph totals into the cost model for accurate estimation.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn from_store(store: &grafeo_core::graph::lpg::LpgStore) -> Self {
        store.ensure_statistics_fresh();
        let stats = store.statistics();
        Self::from_statistics(&stats)
    }

    /// Creates an optimizer from any GraphStore implementation.
    ///
    /// Unlike [`from_store`](Self::from_store), this does not call
    /// `ensure_statistics_fresh()` since external stores manage their own
    /// statistics. The store's [`statistics()`](grafeo_core::graph::GraphStore::statistics) method
    /// is called directly.
    #[must_use]
    pub fn from_graph_store(store: &dyn grafeo_core::graph::GraphStore) -> Self {
        let stats = store.statistics();
        Self::from_statistics(&stats)
    }

    /// Creates an optimizer from RDF statistics.
    ///
    /// Uses triple pattern cardinality estimates for cost-based optimization
    /// of SPARQL queries. Maps total triples to graph totals for the cost model.
    #[cfg(feature = "triple-store")]
    #[must_use]
    pub fn from_rdf_statistics(rdf_stats: grafeo_core::statistics::RdfStatistics) -> Self {
        let total = rdf_stats.total_triples;
        let estimator = CardinalityEstimator::from_rdf_statistics(rdf_stats);
        Self {
            enable_filter_pushdown: true,
            enable_join_reorder: true,
            enable_projection_pushdown: true,
            cost_model: CostModel::new().with_graph_totals(total, total),
            card_estimator: estimator,
        }
    }

    /// Creates an optimizer from a Statistics snapshot.
    ///
    /// Extracts label cardinalities, edge type degrees, and graph totals
    /// for both the cardinality estimator and cost model.
    #[must_use]
    fn from_statistics(stats: &grafeo_core::statistics::Statistics) -> Self {
        let estimator = CardinalityEstimator::from_statistics(stats);

        let avg_fanout = if stats.total_nodes > 0 {
            (stats.total_edges as f64 / stats.total_nodes as f64).max(1.0)
        } else {
            10.0
        };

        let edge_type_degrees: std::collections::HashMap<String, (f64, f64)> = stats
            .edge_types
            .iter()
            .map(|(name, et)| (name.clone(), (et.avg_out_degree, et.avg_in_degree)))
            .collect();

        let label_cardinalities: std::collections::HashMap<String, u64> = stats
            .labels
            .iter()
            .map(|(name, ls)| (name.clone(), ls.node_count))
            .collect();

        Self {
            enable_filter_pushdown: true,
            enable_join_reorder: true,
            enable_projection_pushdown: true,
            cost_model: CostModel::new()
                .with_avg_fanout(avg_fanout)
                .with_edge_type_degrees(edge_type_degrees)
                .with_label_cardinalities(label_cardinalities)
                .with_graph_totals(stats.total_nodes, stats.total_edges),
            card_estimator: estimator,
        }
    }

    /// Enables or disables filter pushdown.
    pub fn with_filter_pushdown(mut self, enabled: bool) -> Self {
        self.enable_filter_pushdown = enabled;
        self
    }

    /// Enables or disables join reordering.
    pub fn with_join_reorder(mut self, enabled: bool) -> Self {
        self.enable_join_reorder = enabled;
        self
    }

    /// Enables or disables projection pushdown.
    pub fn with_projection_pushdown(mut self, enabled: bool) -> Self {
        self.enable_projection_pushdown = enabled;
        self
    }

    /// Sets the cost model.
    pub fn with_cost_model(mut self, cost_model: CostModel) -> Self {
        self.cost_model = cost_model;
        self
    }

    /// Sets the cardinality estimator.
    pub fn with_cardinality_estimator(mut self, estimator: CardinalityEstimator) -> Self {
        self.card_estimator = estimator;
        self
    }

    /// Sets the selectivity configuration for the cardinality estimator.
    pub fn with_selectivity_config(mut self, config: SelectivityConfig) -> Self {
        self.card_estimator = CardinalityEstimator::with_selectivity_config(config);
        self
    }

    /// Returns a reference to the cost model.
    pub fn cost_model(&self) -> &CostModel {
        &self.cost_model
    }

    /// Returns a reference to the cardinality estimator.
    pub fn cardinality_estimator(&self) -> &CardinalityEstimator {
        &self.card_estimator
    }

    /// Estimates the total cost of a plan by recursively costing the entire tree.
    ///
    /// Walks every operator in the plan, computing cardinality at each level
    /// and summing the per-operator costs. Uses actual child cardinalities
    /// for join cost estimation rather than approximations.
    pub fn estimate_cost(&self, plan: &LogicalPlan) -> Cost {
        self.cost_model
            .estimate_tree(&plan.root, &self.card_estimator)
    }

    /// Estimates the cardinality of a plan.
    pub fn estimate_cardinality(&self, plan: &LogicalPlan) -> f64 {
        self.card_estimator.estimate(&plan.root)
    }

    /// Optimizes a logical plan.
    ///
    /// # Errors
    ///
    /// Returns an error if optimization fails.
    pub fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        let _span = grafeo_debug_span!("grafeo::query::optimize");
        // Every rule below recurses per operator and expression, and planning
        // follows: bound the depth for paths that neither bind nor substitute.
        super::plan_depth::check_plan_depth(&plan)?;
        let mut root = plan.root;

        // Apply optimization rules
        if self.enable_filter_pushdown {
            // Propagate filters across LeftJoin shared variables BEFORE
            // pushdown, so OPTIONAL MATCH right-side subtrees pick up
            // the same WHERE constraints the main MATCH applies.
            // Otherwise pushdown rewrites the left subtree and the
            // chance is gone.
            root = self.propagate_join_predicates(root);
            root = self.push_filters_down(root);
        }

        if self.enable_join_reorder {
            root = self.reorder_joins(root);
        }

        root = self.rewrite_distinct_path_search(root);

        if self.enable_projection_pushdown {
            root = self.push_projections_down(root);
        }

        Ok(LogicalPlan {
            root,
            explain: plan.explain,
            profile: plan.profile,
            default_params: plan.default_params,
        })
    }

    /// Pushes a target-only DISTINCT consumer into a variable-length expand.
    ///
    /// A variable-length WALK normally enumerates walks, which is the wrong
    /// work shape for consumers that only retain one value per endpoint. This
    /// pass is deliberately conservative: it recognizes only a complete
    /// consumer slice whose expressions are known deterministic and whose
    /// retained columns cannot distinguish parallel edges or paths.
    fn rewrite_distinct_path_search(&self, op: LogicalOperator) -> LogicalOperator {
        let op = op.map_children(|child| self.rewrite_distinct_path_search(child));
        match op {
            LogicalOperator::Return(mut ret) if ret.distinct => {
                if self.return_distinct_path_is_safe(&ret) {
                    ret.input = Box::new(Self::mark_distinct_expand(*ret.input));
                }
                LogicalOperator::Return(ret)
            }
            LogicalOperator::Distinct(mut distinct)
                if self.distinct_operator_path_is_safe(&distinct) =>
            {
                distinct.input = Box::new(Self::mark_distinct_expand(*distinct.input));
                LogicalOperator::Distinct(distinct)
            }
            LogicalOperator::Aggregate(mut aggregate)
                if self.distinct_aggregate_path_is_safe(&aggregate) =>
            {
                aggregate.input = Box::new(Self::mark_distinct_expand(*aggregate.input));
                LogicalOperator::Aggregate(aggregate)
            }
            other => other,
        }
    }

    /// Marks the single candidate found by the matching consumer analysis.
    fn mark_distinct_expand(op: LogicalOperator) -> LogicalOperator {
        match op {
            LogicalOperator::Expand(mut expand) => {
                expand.path_search = PathSearch::DistinctTargets;
                LogicalOperator::Expand(expand)
            }
            LogicalOperator::Project(mut project) => {
                project.input = Box::new(Self::mark_distinct_expand(*project.input));
                LogicalOperator::Project(project)
            }
            LogicalOperator::Filter(mut filter) => {
                filter.input = Box::new(Self::mark_distinct_expand(*filter.input));
                LogicalOperator::Filter(filter)
            }
            LogicalOperator::Return(mut ret) => {
                ret.input = Box::new(Self::mark_distinct_expand(*ret.input));
                LogicalOperator::Return(ret)
            }
            other => other,
        }
    }

    fn return_distinct_path_is_safe(&self, ret: &crate::query::plan::ReturnOp) -> bool {
        let Some(lineage) = self.distinct_path_lineage(&ret.input) else {
            return false;
        };
        ret.items
            .iter()
            .all(|item| Self::lineage_is_safe(Self::expression_lineage(&item.expression, &lineage)))
    }

    fn distinct_operator_path_is_safe(&self, distinct: &crate::query::plan::DistinctOp) -> bool {
        let Some(lineage) = self.distinct_path_lineage(&distinct.input) else {
            return false;
        };
        match &distinct.columns {
            Some(columns) => {
                // A keyed Distinct still carries every physical input column
                // downstream. Retained edge/path values therefore cannot be
                // hidden merely because they are absent from the key list.
                lineage
                    .values()
                    .all(|value| Self::lineage_is_safe(Some(*value)))
                    && columns.iter().all(|column| {
                        lineage
                            .get(column)
                            .is_some_and(|value| Self::lineage_is_safe(Some(*value)))
                    })
            }
            None => lineage
                .values()
                .all(|lineage| Self::lineage_is_safe(Some(*lineage))),
        }
    }

    fn distinct_aggregate_path_is_safe(&self, aggregate: &crate::query::plan::AggregateOp) -> bool {
        let Some(lineage) = self.distinct_path_lineage(&aggregate.input) else {
            return false;
        };
        if aggregate.aggregates.is_empty()
            || aggregate
                .group_by
                .iter()
                .any(|expr| !Self::lineage_is_safe(Self::expression_lineage(expr, &lineage)))
        {
            return false;
        }
        aggregate.aggregates.iter().all(|agg| {
            let is_binary = matches!(
                agg.function,
                crate::query::plan::AggregateFunction::CovarSamp
                    | crate::query::plan::AggregateFunction::CovarPop
                    | crate::query::plan::AggregateFunction::Corr
                    | crate::query::plan::AggregateFunction::RegrSlope
                    | crate::query::plan::AggregateFunction::RegrIntercept
                    | crate::query::plan::AggregateFunction::RegrR2
                    | crate::query::plan::AggregateFunction::RegrCount
                    | crate::query::plan::AggregateFunction::RegrSxx
                    | crate::query::plan::AggregateFunction::RegrSyy
                    | crate::query::plan::AggregateFunction::RegrSxy
                    | crate::query::plan::AggregateFunction::RegrAvgx
                    | crate::query::plan::AggregateFunction::RegrAvgy
            );
            let arguments_are_safe = agg.expression.as_ref().is_some_and(|expr| {
                Self::lineage_is_safe(Self::expression_lineage(expr, &lineage))
            }) && match (&agg.expression2, is_binary) {
                (Some(expr), true) => {
                    Self::lineage_is_safe(Self::expression_lineage(expr, &lineage))
                }
                (None, false) => true,
                _ => false,
            };
            agg.distinct
                && agg.distinct_key.as_ref().is_none_or(|expr| {
                    Self::lineage_is_safe(Self::expression_lineage(expr, &lineage))
                })
                && matches!(
                    agg.function,
                    crate::query::plan::AggregateFunction::Count
                        | crate::query::plan::AggregateFunction::CountNonNull
                        | crate::query::plan::AggregateFunction::Sum
                        | crate::query::plan::AggregateFunction::Avg
                        | crate::query::plan::AggregateFunction::Min
                        | crate::query::plan::AggregateFunction::Max
                        | crate::query::plan::AggregateFunction::Collect
                        | crate::query::plan::AggregateFunction::StdDev
                        | crate::query::plan::AggregateFunction::StdDevPop
                        | crate::query::plan::AggregateFunction::Variance
                        | crate::query::plan::AggregateFunction::VariancePop
                        | crate::query::plan::AggregateFunction::PercentileDisc
                        | crate::query::plan::AggregateFunction::PercentileCont
                        | crate::query::plan::AggregateFunction::GroupConcat
                        | crate::query::plan::AggregateFunction::Sample
                        | crate::query::plan::AggregateFunction::CovarSamp
                        | crate::query::plan::AggregateFunction::CovarPop
                        | crate::query::plan::AggregateFunction::Corr
                        | crate::query::plan::AggregateFunction::RegrSlope
                        | crate::query::plan::AggregateFunction::RegrIntercept
                        | crate::query::plan::AggregateFunction::RegrR2
                        | crate::query::plan::AggregateFunction::RegrCount
                        | crate::query::plan::AggregateFunction::RegrSxx
                        | crate::query::plan::AggregateFunction::RegrSyy
                        | crate::query::plan::AggregateFunction::RegrSxy
                        | crate::query::plan::AggregateFunction::RegrAvgx
                        | crate::query::plan::AggregateFunction::RegrAvgy
                )
                && arguments_are_safe
        })
    }

    /// Computes visible-column lineage from an expand through only the wrappers
    /// whose row mapping remains inspectable by this pass.
    fn distinct_path_lineage(
        &self,
        op: &LogicalOperator,
    ) -> Option<HashMap<String, DistinctPathLineage>> {
        match op {
            LogicalOperator::Expand(expand) => self.expand_path_lineage(expand),
            LogicalOperator::Project(project) => {
                let lineage = self.distinct_path_lineage(&project.input)?;
                self.project_path_lineage(lineage, &project.projections, project.pass_through_input)
            }
            LogicalOperator::Filter(filter) => {
                let lineage = self.distinct_path_lineage(&filter.input)?;
                Self::lineage_is_safe(Self::expression_lineage(&filter.predicate, &lineage))
                    .then_some(lineage)
            }
            LogicalOperator::Return(ret) if !ret.distinct => {
                let lineage = self.distinct_path_lineage(&ret.input)?;
                self.return_path_lineage(lineage, &ret.items)
            }
            _ => None,
        }
    }

    /// Certifies intrinsic edge predicates using the existing deterministic
    /// expression allowlist. The binder must separately establish that every
    /// referenced name belongs to the candidate edge or fixed input bindings.
    /// Unsupported expression forms and unknown/volatile functions are not
    /// certified; ordinary unpruned traversal may still evaluate them.
    pub(crate) fn intrinsic_edge_predicate_is_stable(predicate: &LogicalExpression) -> bool {
        let mut variables = HashSet::new();
        Self::collect_variables(predicate, &mut variables);
        let lineage = variables
            .into_iter()
            .map(|name| (name, DistinctPathLineage::Stable))
            .collect();
        matches!(
            Self::expression_lineage(predicate, &lineage),
            Some(DistinctPathLineage::Stable)
        )
    }

    fn expand_path_lineage(
        &self,
        expand: &crate::query::plan::ExpandOp,
    ) -> Option<HashMap<String, DistinctPathLineage>> {
        let variable_length = expand.min_hops != 1 || expand.max_hops != Some(1);
        if !variable_length
            || expand.path_search != PathSearch::All
            || (expand.path_mode != PathMode::Walk && expand.min_hops > 1)
            || expand.from_variable == expand.to_variable
            // A full-path predicate can reject one walk while accepting
            // another walk to the same endpoint.  Target deduplication would
            // therefore move the predicate after the quota and change rows.
            || expand.path_predicate.is_some()
            || expand
                .edge_predicate
                .as_ref()
                .is_some_and(|predicate| !Self::intrinsic_edge_predicate_is_stable(predicate))
        {
            return None;
        }

        let mut lineage = self
            .collect_output_variables(&expand.input)
            .into_iter()
            .map(|name| (name, DistinctPathLineage::Stable))
            .collect::<HashMap<_, _>>();
        lineage.insert(expand.from_variable.clone(), DistinctPathLineage::Stable);
        lineage.insert(expand.to_variable.clone(), DistinctPathLineage::Endpoint);
        if let Some(edge) = &expand.edge_variable {
            Self::merge_path_lineage(&mut lineage, edge.clone(), DistinctPathLineage::Tainted);
        } else {
            lineage.insert(
                ANONYMOUS_PATH_COLUMN.to_string(),
                DistinctPathLineage::Tainted,
            );
        }
        if let Some(path) = &expand.path_alias {
            Self::merge_path_lineage(&mut lineage, path.clone(), DistinctPathLineage::Tainted);
            for column in [
                format!("_path_length_{path}"),
                format!("_path_nodes_{path}"),
                format!("_path_edges_{path}"),
            ] {
                Self::merge_path_lineage(&mut lineage, column, DistinctPathLineage::Tainted);
            }
        }
        Some(lineage)
    }

    fn merge_path_lineage(
        lineage: &mut HashMap<String, DistinctPathLineage>,
        name: String,
        value: DistinctPathLineage,
    ) {
        lineage
            .entry(name)
            .and_modify(|existing| *existing = Self::combine_path_lineage(*existing, value))
            .or_insert(value);
    }

    fn project_path_lineage(
        &self,
        input: HashMap<String, DistinctPathLineage>,
        projections: &[crate::query::plan::Projection],
        pass_through: bool,
    ) -> Option<HashMap<String, DistinctPathLineage>> {
        let mut output = if pass_through {
            input.clone()
        } else {
            HashMap::new()
        };
        for projection in projections {
            let value = Self::expression_lineage(&projection.expression, &input)?;
            // Every projection expression is evaluated, even when a later
            // wrapper drops its output column. A path-dependent expression
            // can therefore not be hidden from this admission check.
            if !Self::lineage_is_safe(Some(value)) {
                return None;
            }
            let name = output_column_name(projection.alias.as_deref(), &projection.expression);
            Self::merge_path_lineage(&mut output, name, value);
        }
        Some(output)
    }

    fn return_path_lineage(
        &self,
        input: HashMap<String, DistinctPathLineage>,
        items: &[crate::query::plan::ReturnItem],
    ) -> Option<HashMap<String, DistinctPathLineage>> {
        let mut output = HashMap::new();
        for item in items {
            let value = Self::expression_lineage(&item.expression, &input)?;
            // Return evaluates all items before a later DISTINCT can select a
            // subset, so retain the same invariance requirement as Project.
            if !Self::lineage_is_safe(Some(value)) {
                return None;
            }
            let name = output_column_name(item.alias.as_deref(), &item.expression);
            Self::merge_path_lineage(&mut output, name, value);
        }
        Some(output)
    }

    fn lineage_is_safe(lineage: Option<DistinctPathLineage>) -> bool {
        matches!(
            lineage,
            Some(DistinctPathLineage::Stable | DistinctPathLineage::Endpoint)
        )
    }

    fn combine_path_lineage(
        left: DistinctPathLineage,
        right: DistinctPathLineage,
    ) -> DistinctPathLineage {
        if left == DistinctPathLineage::Tainted || right == DistinctPathLineage::Tainted {
            DistinctPathLineage::Tainted
        } else if left == DistinctPathLineage::Endpoint || right == DistinctPathLineage::Endpoint {
            DistinctPathLineage::Endpoint
        } else {
            DistinctPathLineage::Stable
        }
    }

    fn expression_lineage(
        expr: &LogicalExpression,
        lineage: &HashMap<String, DistinctPathLineage>,
    ) -> Option<DistinctPathLineage> {
        let combine = |left: Option<DistinctPathLineage>, right: Option<DistinctPathLineage>| {
            Some(Self::combine_path_lineage(left?, right?))
        };
        match expr {
            LogicalExpression::Literal(_) | LogicalExpression::Parameter(_) => {
                Some(DistinctPathLineage::Stable)
            }
            LogicalExpression::Variable(name) => lineage.get(name).copied(),
            LogicalExpression::Property { variable, .. } => lineage.get(variable).copied(),
            LogicalExpression::Binary { left, right, .. } => combine(
                Self::expression_lineage(left, lineage),
                Self::expression_lineage(right, lineage),
            ),
            LogicalExpression::Unary { operand, .. } => Self::expression_lineage(operand, lineage),
            LogicalExpression::FunctionCall {
                name,
                args,
                distinct,
            } => {
                let name = name.to_ascii_lowercase();
                let pure = matches!(
                    name.as_str(),
                    "abs"
                        | "ceil"
                        | "coalesce"
                        | "elementid"
                        | "floor"
                        | "haslabel"
                        | "hasproperty"
                        | "id"
                        | "labels"
                        | "length"
                        | "lower"
                        | "ltrim"
                        | "round"
                        | "rtrim"
                        | "size"
                        | "tofloat"
                        | "tointeger"
                        | "tolower"
                        | "tostring"
                        | "toupper"
                        | "trim"
                        | "type"
                );
                if *distinct || !pure {
                    return None;
                }
                let mut result = DistinctPathLineage::Stable;
                for arg in args {
                    result =
                        Self::combine_path_lineage(result, Self::expression_lineage(arg, lineage)?);
                }
                Some(result)
            }
            LogicalExpression::List(items) => {
                let mut result = DistinctPathLineage::Stable;
                for item in items {
                    result = Self::combine_path_lineage(
                        result,
                        Self::expression_lineage(item, lineage)?,
                    );
                }
                Some(result)
            }
            LogicalExpression::Map(pairs) => {
                let mut result = DistinctPathLineage::Stable;
                for (_, value) in pairs {
                    result = Self::combine_path_lineage(
                        result,
                        Self::expression_lineage(value, lineage)?,
                    );
                }
                Some(result)
            }
            LogicalExpression::IndexAccess { base, index } => combine(
                Self::expression_lineage(base, lineage),
                Self::expression_lineage(index, lineage),
            ),
            LogicalExpression::SliceAccess { base, start, end } => {
                let mut result = Self::expression_lineage(base, lineage)?;
                if let Some(start) = start {
                    result = Self::combine_path_lineage(
                        result,
                        Self::expression_lineage(start, lineage)?,
                    );
                }
                if let Some(end) = end {
                    result =
                        Self::combine_path_lineage(result, Self::expression_lineage(end, lineage)?);
                }
                Some(result)
            }
            LogicalExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                let mut result = DistinctPathLineage::Stable;
                if let Some(operand) = operand {
                    result = Self::combine_path_lineage(
                        result,
                        Self::expression_lineage(operand, lineage)?,
                    );
                }
                for (when, then) in when_clauses {
                    result = Self::combine_path_lineage(
                        result,
                        Self::expression_lineage(when, lineage)?,
                    );
                    result = Self::combine_path_lineage(
                        result,
                        Self::expression_lineage(then, lineage)?,
                    );
                }
                if let Some(else_clause) = else_clause {
                    result = Self::combine_path_lineage(
                        result,
                        Self::expression_lineage(else_clause, lineage)?,
                    );
                }
                Some(result)
            }
            LogicalExpression::Labels(variable)
            | LogicalExpression::Type(variable)
            | LogicalExpression::Id(variable) => lineage.get(variable).copied(),
            // These forms introduce local scopes or execute subplans. Keep the
            // rule conservative until their lineage contracts are explicit.
            _ => None,
        }
    }

    /// Pushes projections down the operator tree to eliminate unused columns early.
    ///
    /// This optimization:
    /// 1. Collects required variables/properties from the root
    /// 2. Propagates requirements down through the tree
    /// 3. Inserts projections to eliminate unneeded columns before expensive operations
    fn push_projections_down(&self, op: LogicalOperator) -> LogicalOperator {
        // Collect required columns from the top of the plan
        let required = self.collect_required_columns(&op);

        // Push projections down
        self.push_projections_recursive(op, &required)
    }

    /// Collects all variables and properties required by an operator and its ancestors.
    fn collect_required_columns(&self, op: &LogicalOperator) -> HashSet<RequiredColumn> {
        let mut required = HashSet::new();
        Self::collect_required_recursive(op, &mut required);
        required
    }

    /// Recursively collects required columns.
    fn collect_required_recursive(op: &LogicalOperator, required: &mut HashSet<RequiredColumn>) {
        match op {
            LogicalOperator::Return(ret) => {
                for item in &ret.items {
                    Self::collect_from_expression(&item.expression, required);
                }
                Self::collect_required_recursive(&ret.input, required);
            }
            LogicalOperator::Project(proj) => {
                for p in &proj.projections {
                    Self::collect_from_expression(&p.expression, required);
                }
                Self::collect_required_recursive(&proj.input, required);
            }
            LogicalOperator::Filter(filter) => {
                Self::collect_from_expression(&filter.predicate, required);
                Self::collect_required_recursive(&filter.input, required);
            }
            LogicalOperator::Sort(sort) => {
                for key in &sort.keys {
                    Self::collect_from_expression(&key.expression, required);
                }
                Self::collect_required_recursive(&sort.input, required);
            }
            LogicalOperator::Aggregate(agg) => {
                for expr in &agg.group_by {
                    Self::collect_from_expression(expr, required);
                }
                for agg_expr in &agg.aggregates {
                    if let Some(ref expr) = agg_expr.expression {
                        Self::collect_from_expression(expr, required);
                    }
                    if let Some(ref expr) = agg_expr.expression2 {
                        Self::collect_from_expression(expr, required);
                    }
                    if let Some(ref expr) = agg_expr.distinct_key {
                        Self::collect_from_expression(expr, required);
                    }
                }
                if let Some(ref having) = agg.having {
                    Self::collect_from_expression(having, required);
                }
                Self::collect_required_recursive(&agg.input, required);
            }
            LogicalOperator::Join(join) => {
                for cond in &join.conditions {
                    Self::collect_from_expression(&cond.left, required);
                    Self::collect_from_expression(&cond.right, required);
                }
                Self::collect_required_recursive(&join.left, required);
                Self::collect_required_recursive(&join.right, required);
            }
            LogicalOperator::Expand(expand) => {
                // The source and target variables are needed
                required.insert(RequiredColumn::Variable(expand.from_variable.clone()));
                required.insert(RequiredColumn::Variable(expand.to_variable.clone()));
                if let Some(ref edge_var) = expand.edge_variable {
                    required.insert(RequiredColumn::Variable(edge_var.clone()));
                }
                if let Some(predicate) = &expand.edge_predicate {
                    Self::collect_from_expression(predicate, required);
                }
                if let Some(predicate) = &expand.path_predicate {
                    Self::collect_from_expression(predicate, required);
                    let mut free_variables = HashSet::new();
                    Self::collect_variables(predicate, &mut free_variables);
                    required.extend(free_variables.into_iter().map(RequiredColumn::Variable));
                }
                Self::collect_required_recursive(&expand.input, required);
            }
            LogicalOperator::Limit(limit) => {
                Self::collect_required_recursive(&limit.input, required);
            }
            LogicalOperator::Skip(skip) => {
                Self::collect_required_recursive(&skip.input, required);
            }
            LogicalOperator::Distinct(distinct) => {
                Self::collect_required_recursive(&distinct.input, required);
            }
            LogicalOperator::NodeScan(scan) => {
                required.insert(RequiredColumn::Variable(scan.variable.clone()));
            }
            LogicalOperator::EdgeScan(scan) => {
                required.insert(RequiredColumn::Variable(scan.variable.clone()));
            }
            LogicalOperator::MultiWayJoin(mwj) => {
                for cond in &mwj.conditions {
                    Self::collect_from_expression(&cond.left, required);
                    Self::collect_from_expression(&cond.right, required);
                }
                for input in &mwj.inputs {
                    Self::collect_required_recursive(input, required);
                }
            }
            _ => {}
        }
    }

    /// Collects required columns from an expression.
    fn collect_from_expression(expr: &LogicalExpression, required: &mut HashSet<RequiredColumn>) {
        match expr {
            LogicalExpression::Variable(var) => {
                required.insert(RequiredColumn::Variable(var.clone()));
            }
            LogicalExpression::Property { variable, property } => {
                required.insert(RequiredColumn::Property(variable.clone(), property.clone()));
                required.insert(RequiredColumn::Variable(variable.clone()));
            }
            LogicalExpression::Binary { left, right, .. } => {
                Self::collect_from_expression(left, required);
                Self::collect_from_expression(right, required);
            }
            LogicalExpression::Unary { operand, .. } => {
                Self::collect_from_expression(operand, required);
            }
            LogicalExpression::FunctionCall { args, .. } => {
                for arg in args {
                    Self::collect_from_expression(arg, required);
                }
            }
            LogicalExpression::List(items) => {
                for item in items {
                    Self::collect_from_expression(item, required);
                }
            }
            LogicalExpression::Map(pairs) => {
                for (_, value) in pairs {
                    Self::collect_from_expression(value, required);
                }
            }
            LogicalExpression::IndexAccess { base, index } => {
                Self::collect_from_expression(base, required);
                Self::collect_from_expression(index, required);
            }
            LogicalExpression::SliceAccess { base, start, end } => {
                Self::collect_from_expression(base, required);
                if let Some(s) = start {
                    Self::collect_from_expression(s, required);
                }
                if let Some(e) = end {
                    Self::collect_from_expression(e, required);
                }
            }
            LogicalExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                if let Some(op) = operand {
                    Self::collect_from_expression(op, required);
                }
                for (cond, result) in when_clauses {
                    Self::collect_from_expression(cond, required);
                    Self::collect_from_expression(result, required);
                }
                if let Some(else_expr) = else_clause {
                    Self::collect_from_expression(else_expr, required);
                }
            }
            LogicalExpression::Labels(var)
            | LogicalExpression::Type(var)
            | LogicalExpression::Id(var) => {
                required.insert(RequiredColumn::Variable(var.clone()));
            }
            LogicalExpression::ListComprehension {
                list_expr,
                filter_expr,
                map_expr,
                ..
            } => {
                Self::collect_from_expression(list_expr, required);
                if let Some(filter) = filter_expr {
                    Self::collect_from_expression(filter, required);
                }
                Self::collect_from_expression(map_expr, required);
            }
            _ => {}
        }
    }

    /// Recursively pushes projections down, adding them before expensive operations.
    fn push_projections_recursive(
        &self,
        op: LogicalOperator,
        required: &HashSet<RequiredColumn>,
    ) -> LogicalOperator {
        match op {
            LogicalOperator::Return(mut ret) => {
                ret.input = Box::new(self.push_projections_recursive(*ret.input, required));
                LogicalOperator::Return(ret)
            }
            LogicalOperator::Project(mut proj) => {
                proj.input = Box::new(self.push_projections_recursive(*proj.input, required));
                LogicalOperator::Project(proj)
            }
            LogicalOperator::Filter(mut filter) => {
                filter.input = Box::new(self.push_projections_recursive(*filter.input, required));
                LogicalOperator::Filter(filter)
            }
            LogicalOperator::Sort(mut sort) => {
                // Sort is expensive - consider adding a projection before it
                // to reduce tuple width
                sort.input = Box::new(self.push_projections_recursive(*sort.input, required));
                LogicalOperator::Sort(sort)
            }
            LogicalOperator::Aggregate(mut agg) => {
                agg.input = Box::new(self.push_projections_recursive(*agg.input, required));
                LogicalOperator::Aggregate(agg)
            }
            LogicalOperator::Join(mut join) => {
                // Joins are expensive - the required columns help determine
                // what to project on each side
                let left_vars = self.collect_output_variables(&join.left);
                let right_vars = self.collect_output_variables(&join.right);

                // Filter required columns to each side
                let left_required: HashSet<_> = required
                    .iter()
                    .filter(|c| match c {
                        RequiredColumn::Variable(v) => left_vars.contains(v),
                        RequiredColumn::Property(v, _) => left_vars.contains(v),
                    })
                    .cloned()
                    .collect();

                let right_required: HashSet<_> = required
                    .iter()
                    .filter(|c| match c {
                        RequiredColumn::Variable(v) => right_vars.contains(v),
                        RequiredColumn::Property(v, _) => right_vars.contains(v),
                    })
                    .cloned()
                    .collect();

                join.left = Box::new(self.push_projections_recursive(*join.left, &left_required));
                join.right =
                    Box::new(self.push_projections_recursive(*join.right, &right_required));
                LogicalOperator::Join(join)
            }
            LogicalOperator::Expand(mut expand) => {
                expand.input = Box::new(self.push_projections_recursive(*expand.input, required));
                LogicalOperator::Expand(expand)
            }
            LogicalOperator::Limit(mut limit) => {
                limit.input = Box::new(self.push_projections_recursive(*limit.input, required));
                LogicalOperator::Limit(limit)
            }
            LogicalOperator::Skip(mut skip) => {
                skip.input = Box::new(self.push_projections_recursive(*skip.input, required));
                LogicalOperator::Skip(skip)
            }
            LogicalOperator::Distinct(mut distinct) => {
                distinct.input =
                    Box::new(self.push_projections_recursive(*distinct.input, required));
                LogicalOperator::Distinct(distinct)
            }
            LogicalOperator::MapCollect(mut mc) => {
                mc.input = Box::new(self.push_projections_recursive(*mc.input, required));
                LogicalOperator::MapCollect(mc)
            }
            LogicalOperator::MultiWayJoin(mut mwj) => {
                mwj.inputs = mwj
                    .inputs
                    .into_iter()
                    .map(|input| self.push_projections_recursive(input, required))
                    .collect();
                LogicalOperator::MultiWayJoin(mwj)
            }
            other => other,
        }
    }

    /// Reorders joins in the operator tree using the DPccp algorithm.
    ///
    /// This optimization finds the optimal join order by:
    /// 1. Extracting all base relations (scans) and join conditions
    /// 2. Building a join graph
    /// 3. Using dynamic programming to find the cheapest join order
    fn reorder_joins(&self, op: LogicalOperator) -> LogicalOperator {
        // First, recursively optimize children
        let op = self.reorder_joins_recursive(op);

        // Then, if this is a join tree, try to optimize it
        if let Some((relations, conditions)) = self.extract_join_tree(&op)
            && relations.len() >= 2
            && let Some(optimized) = self.optimize_join_order(&relations, &conditions)
        {
            return optimized;
        }

        op
    }

    /// Recursively applies join reordering to child operators.
    fn reorder_joins_recursive(&self, op: LogicalOperator) -> LogicalOperator {
        match op {
            LogicalOperator::Return(mut ret) => {
                ret.input = Box::new(self.reorder_joins(*ret.input));
                LogicalOperator::Return(ret)
            }
            LogicalOperator::Project(mut proj) => {
                proj.input = Box::new(self.reorder_joins(*proj.input));
                LogicalOperator::Project(proj)
            }
            LogicalOperator::Filter(mut filter) => {
                filter.input = Box::new(self.reorder_joins(*filter.input));
                LogicalOperator::Filter(filter)
            }
            LogicalOperator::Limit(mut limit) => {
                limit.input = Box::new(self.reorder_joins(*limit.input));
                LogicalOperator::Limit(limit)
            }
            LogicalOperator::Skip(mut skip) => {
                skip.input = Box::new(self.reorder_joins(*skip.input));
                LogicalOperator::Skip(skip)
            }
            LogicalOperator::Sort(mut sort) => {
                sort.input = Box::new(self.reorder_joins(*sort.input));
                LogicalOperator::Sort(sort)
            }
            LogicalOperator::Distinct(mut distinct) => {
                distinct.input = Box::new(self.reorder_joins(*distinct.input));
                LogicalOperator::Distinct(distinct)
            }
            LogicalOperator::Aggregate(mut agg) => {
                agg.input = Box::new(self.reorder_joins(*agg.input));
                LogicalOperator::Aggregate(agg)
            }
            LogicalOperator::Expand(mut expand) => {
                expand.input = Box::new(self.reorder_joins(*expand.input));
                LogicalOperator::Expand(expand)
            }
            LogicalOperator::MapCollect(mut mc) => {
                mc.input = Box::new(self.reorder_joins(*mc.input));
                LogicalOperator::MapCollect(mc)
            }
            LogicalOperator::MultiWayJoin(mut mwj) => {
                mwj.inputs = mwj
                    .inputs
                    .into_iter()
                    .map(|input| self.reorder_joins(input))
                    .collect();
                LogicalOperator::MultiWayJoin(mwj)
            }
            // Join operators are handled by the parent reorder_joins call
            other => other,
        }
    }

    /// Extracts base relations and join conditions from a join tree.
    ///
    /// Returns None if the operator is not a join tree.
    fn extract_join_tree(
        &self,
        op: &LogicalOperator,
    ) -> Option<(Vec<(String, LogicalOperator)>, Vec<JoinInfo>)> {
        let mut relations = Vec::new();
        let mut join_conditions = Vec::new();

        if !self.collect_join_tree(op, &mut relations, &mut join_conditions) {
            return None;
        }

        if relations.len() < 2 {
            return None;
        }

        Some((relations, join_conditions))
    }

    /// Recursively collects base relations and join conditions.
    ///
    /// Returns true if this subtree is part of a join tree.
    fn collect_join_tree(
        &self,
        op: &LogicalOperator,
        relations: &mut Vec<(String, LogicalOperator)>,
        conditions: &mut Vec<JoinInfo>,
    ) -> bool {
        match op {
            LogicalOperator::Join(join) => {
                if !matches!(join.join_type, crate::query::plan::JoinType::Inner) {
                    return false;
                }
                // Record the concrete relation ranges owned by each side.
                // Expression variable names are not relation identities in
                // SPARQL: the same shared name deliberately appears on both
                // sides of a join.
                let left_start = relations.len();
                let left_ok = self.collect_join_tree(&join.left, relations, conditions);
                let left_end = relations.len();
                let right_start = left_end;
                let right_ok = self.collect_join_tree(&join.right, relations, conditions);
                let right_end = relations.len();

                // Add conditions from this join
                for cond in &join.conditions {
                    let (Some(left_var), Some(right_var)) = (
                        self.extract_variable_from_expr(&cond.left),
                        self.extract_variable_from_expr(&cond.right),
                    ) else {
                        return false;
                    };
                    let find_relation = |start: usize, end: usize, variable: &str| {
                        relations[start..end]
                            .iter()
                            .position(|(_, relation)| {
                                self.collect_output_variables(relation).contains(variable)
                            })
                            .map(|index| start + index)
                    };
                    let oriented = find_relation(left_start, left_end, &left_var)
                        .zip(find_relation(right_start, right_end, &right_var))
                        .map(|(left_relation, right_relation)| {
                            (
                                left_relation,
                                right_relation,
                                left_var.clone(),
                                right_var.clone(),
                                cond.left.clone(),
                                cond.right.clone(),
                            )
                        })
                        .or_else(|| {
                            find_relation(left_start, left_end, &right_var)
                                .zip(find_relation(right_start, right_end, &left_var))
                                .map(|(left_relation, right_relation)| {
                                    (
                                        left_relation,
                                        right_relation,
                                        right_var.clone(),
                                        left_var.clone(),
                                        cond.right.clone(),
                                        cond.left.clone(),
                                    )
                                })
                        });
                    let Some((
                        left_relation,
                        right_relation,
                        left_var,
                        right_var,
                        left_expr,
                        right_expr,
                    )) = oriented
                    else {
                        return false;
                    };
                    conditions.push(JoinInfo {
                        left_relation,
                        right_relation,
                        left_var,
                        right_var,
                        left_expr,
                        right_expr,
                        semantics: cond.semantics,
                    });
                }

                left_ok && right_ok
            }
            LogicalOperator::NodeScan(scan) => {
                relations.push((scan.variable.clone(), op.clone()));
                true
            }
            LogicalOperator::EdgeScan(scan) => {
                relations.push((scan.variable.clone(), op.clone()));
                true
            }
            LogicalOperator::Filter(filter) => {
                // A filter wrapping a single base relation rides along with that
                // relation through reordering: record the whole Filter(relation)
                // as the relation entry so its predicate is never lost. If the
                // filter sits above a join (spans relations), decline to flatten
                // (return false) so reordering is skipped and the original,
                // predicate-bearing plan is kept intact.
                match filter.input.as_ref() {
                    LogicalOperator::NodeScan(scan) => {
                        relations.push((scan.variable.clone(), op.clone()));
                        true
                    }
                    LogicalOperator::EdgeScan(scan) => {
                        relations.push((scan.variable.clone(), op.clone()));
                        true
                    }
                    LogicalOperator::Expand(expand) => {
                        relations.push((expand.to_variable.clone(), op.clone()));
                        true
                    }
                    #[cfg(feature = "triple-store")]
                    LogicalOperator::TripleScan(scan) => {
                        let name = scan
                            .subject
                            .as_variable()
                            .or_else(|| scan.predicate.as_variable())
                            .or_else(|| scan.object.as_variable())
                            .map_or_else(|| format!("__tp_{}", relations.len()), String::from);
                        relations.push((name, op.clone()));
                        true
                    }
                    _ => false,
                }
            }
            LogicalOperator::Expand(expand) => {
                // Expand is a special case - it's like a join with the adjacency
                // For now, treat the whole Expand subtree as a single relation
                relations.push((expand.to_variable.clone(), op.clone()));
                true
            }
            #[cfg(feature = "triple-store")]
            LogicalOperator::TripleScan(scan) => {
                // Use the first variable found as the relation name.
                // For all-constant patterns, generate a unique fallback to
                // avoid HashMap key collisions in `add_relation`.
                let name = scan
                    .subject
                    .as_variable()
                    .or_else(|| scan.predicate.as_variable())
                    .or_else(|| scan.object.as_variable())
                    .map_or_else(|| format!("__tp_{}", relations.len()), String::from);
                relations.push((name, op.clone()));
                true
            }
            _ => false,
        }
    }

    /// Extracts the primary variable from an expression.
    fn extract_variable_from_expr(&self, expr: &LogicalExpression) -> Option<String> {
        match expr {
            LogicalExpression::Variable(v) => Some(v.clone()),
            LogicalExpression::Property { variable, .. } => Some(variable.clone()),
            _ => None,
        }
    }

    /// Optimizes the join order using DPccp, or produces a multi-way
    /// leapfrog join for cyclic patterns when the cost model prefers it.
    fn optimize_join_order(
        &self,
        relations: &[(String, LogicalOperator)],
        conditions: &[JoinInfo],
    ) -> Option<LogicalOperator> {
        use join_order::{DPccp, JoinGraphBuilder};

        // Build the join graph
        let mut builder = JoinGraphBuilder::new();

        for (var, relation) in relations {
            builder.add_relation(var, relation.clone());
        }

        for cond in conditions {
            builder.add_join_condition_between_nodes(
                cond.left_relation,
                cond.right_relation,
                cond.left_expr.clone(),
                cond.right_expr.clone(),
                cond.semantics,
            );
        }

        let graph = builder.build();

        // For cyclic graphs with 3+ relations, use leapfrog (WCOJ) join.
        // Cyclic joins (e.g. triangle patterns) benefit from worst-case optimal
        // multi-way intersection rather than binary hash join cascades that can
        // produce intermediate blowup.
        let multi_way_conditions_are_owned_equivalences = conditions.first().is_some_and(|first| {
            let mut names = std::collections::HashSet::new();
            first.semantics != JoinKeySemantics::SparqlCompatibility
                && conditions.iter().all(|condition| {
                    condition.semantics == first.semantics
                        && matches!(
                            (&condition.left_expr, &condition.right_expr),
                            (
                                LogicalExpression::Variable(left),
                                LogicalExpression::Variable(right)
                            ) if left == right && names.insert(left.as_str())
                        )
                })
        });
        if graph.is_cyclic() && relations.len() >= 3 && multi_way_conditions_are_owned_equivalences
        {
            // Collect shared variables (variables appearing in 2+ conditions)
            let mut var_counts: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();
            for cond in conditions {
                *var_counts.entry(&cond.left_var).or_default() += 1;
                *var_counts.entry(&cond.right_var).or_default() += 1;
            }
            let shared_variables: Vec<String> = var_counts
                .into_iter()
                .filter(|(_, count)| *count >= 2)
                .map(|(var, _)| var.to_string())
                .collect();

            let join_conditions: Vec<JoinCondition> = conditions
                .iter()
                .map(|c| JoinCondition {
                    left: c.left_expr.clone(),
                    right: c.right_expr.clone(),
                    semantics: c.semantics,
                })
                .collect();

            return Some(LogicalOperator::MultiWayJoin(MultiWayJoinOp {
                inputs: relations.iter().map(|(_, rel)| rel.clone()).collect(),
                conditions: join_conditions,
                shared_variables,
            }));
        }

        // Fall through to DPccp for binary join ordering
        let mut dpccp = DPccp::new(&graph, &self.cost_model, &self.card_estimator);
        let plan = dpccp.optimize()?;

        Some(plan.operator)
    }

    /// Propagates filter predicates across LeftJoin boundaries when the
    /// predicate references variables bound on both sides of the join.
    ///
    /// OPTIONAL MATCH compiles to `LeftJoin(left, right)` where the right
    /// subtree binds the same variable as the left (typically via its own
    /// NodeScan), so the right side independently scans the full table even
    /// when the left side is bound to a tiny set. After regular pushdown
    /// rewrites `Filter(P, MainMatch)` into something like
    /// `Expand(NodeList(P))`, the original filter is gone and we can no
    /// longer detect the constraint to mirror to the right. This pass runs
    /// BEFORE pushdown: for every `LeftJoin(Filter(P, X), Y)` where P's
    /// variables are bound in both X and Y, wrap Y with the same Filter.
    ///
    /// Safe by LeftJoin semantics: matched pairs satisfy left.v = right.v
    /// for shared variable v, so a right row that fails P also fails P
    /// when joined to its matching left row, which already failed P. And
    /// OPTIONAL left rows with no right match are unaffected, since we
    /// only constrain the right side's enumeration.
    fn propagate_join_predicates(&self, op: LogicalOperator) -> LogicalOperator {
        // Recurse into children first (post-order). After children are
        // processed, propagate at this level if it's a LeftJoin.
        let op = op.map_children(|child| self.propagate_join_predicates(child));

        let LogicalOperator::LeftJoin(mut left_join) = op else {
            return op;
        };

        // Variables bound on both sides act as the implicit join keys
        // (typically a single shared variable like `c`). Predicates that
        // reference only these variables are safe to mirror to the right.
        let left_vars = self.collect_output_variables(&left_join.left);
        let right_vars = self.collect_output_variables(&left_join.right);
        let compatibility_vars =
            Self::sparql_compatibility_variables(&left_join.compatibility_conditions);
        let shared_vars: HashSet<String> = left_vars
            .intersection(&right_vars)
            .filter(|variable| !compatibility_vars.contains(*variable))
            .cloned()
            .collect();

        if shared_vars.is_empty() {
            return LogicalOperator::LeftJoin(left_join);
        }

        let mut shared_filters = Vec::new();
        self.collect_shared_var_filters(&left_join.left, &shared_vars, &mut shared_filters);
        for predicate in shared_filters {
            left_join.right = Box::new(LogicalOperator::Filter(FilterOp {
                predicate,
                pushdown_hint: None,
                input: left_join.right,
            }));
        }

        LogicalOperator::LeftJoin(left_join)
    }

    /// Variables joined with SPARQL solution-mapping compatibility cannot
    /// participate in equality-based predicate mirroring. Either side may be
    /// unbound (a wildcard), and the output may be coalesced from the other
    /// side, so a predicate evaluated before the join is not equivalent to the
    /// same predicate evaluated after it.
    fn sparql_compatibility_variables(conditions: &[JoinCondition]) -> HashSet<String> {
        let mut variables = HashSet::new();
        for condition in conditions {
            if condition.semantics == JoinKeySemantics::SparqlCompatibility {
                Self::collect_variables(&condition.left, &mut variables);
                Self::collect_variables(&condition.right, &mut variables);
            }
        }
        variables
    }

    /// Walks a (sub)tree collecting filter predicates that reference only
    /// the given shared variables. These are safe to mirror across a
    /// LeftJoin to the right subtree.
    fn collect_shared_var_filters(
        &self,
        op: &LogicalOperator,
        shared_vars: &HashSet<String>,
        out: &mut Vec<LogicalExpression>,
    ) {
        match op {
            LogicalOperator::Filter(f) => {
                let predicate_vars = self.extract_variables(&f.predicate);
                if !predicate_vars.is_empty()
                    && predicate_vars.iter().all(|v| shared_vars.contains(v))
                {
                    out.push(f.predicate.clone());
                }
                self.collect_shared_var_filters(&f.input, shared_vars, out);
            }
            // Don't descend through operators that change row semantics
            // (Aggregate, Limit, Skip, Sort, Distinct): predicates above
            // them aren't safe to extract because they apply to a
            // post-aggregation/limit world. Project, Return, and Expand
            // are row-preserving so we can keep walking.
            LogicalOperator::Project(p) => {
                self.collect_shared_var_filters(&p.input, shared_vars, out);
            }
            LogicalOperator::Return(r) => {
                self.collect_shared_var_filters(&r.input, shared_vars, out);
            }
            LogicalOperator::Expand(e) => {
                self.collect_shared_var_filters(&e.input, shared_vars, out);
            }
            // For nested LeftJoins (chained OPTIONAL MATCH), the LEFT side
            // is the row-driving subtree: its surviving rows feed the outer
            // join. Filters anywhere in that left subtree are implicitly
            // applied to every output row, so they're safe to mirror across
            // the outer join. The right side is optional and may contribute
            // NULL-padded rows that don't satisfy those filters, so we
            // deliberately don't walk into it.
            LogicalOperator::LeftJoin(j) => {
                self.collect_shared_var_filters(&j.left, shared_vars, out);
            }
            LogicalOperator::Join(j) => {
                self.collect_shared_var_filters(&j.left, shared_vars, out);
                self.collect_shared_var_filters(&j.right, shared_vars, out);
            }
            // Stop at Apply, Aggregate, Limit, Skip, etc.: too risky to
            // pull predicates across those boundaries.
            _ => {}
        }
    }

    /// Pushes filters down the operator tree.
    ///
    /// This optimization moves filter predicates as close to the data source
    /// as possible to reduce the amount of data processed by upper operators.
    fn push_filters_down(&self, op: LogicalOperator) -> LogicalOperator {
        match op {
            // For Filter operators, split the top-level AND chain into individual
            // conjuncts and push each independently. A conjunct that can anchor on
            // one relation no longer rides above a cartesian product just because
            // it shares a Filter with a conjunct on another relation. Conjuncts
            // that cannot be pushed re-stack as filters via try_push_filter_into.
            LogicalOperator::Filter(filter) => {
                let mut current = self.push_filters_down(*filter.input);
                for conjunct in split_conjuncts(filter.predicate) {
                    current = self.try_push_filter_into(conjunct, current);
                }
                current
            }
            // Recursively optimize children for other operators
            LogicalOperator::Return(mut ret) => {
                ret.input = Box::new(self.push_filters_down(*ret.input));
                LogicalOperator::Return(ret)
            }
            LogicalOperator::Project(mut proj) => {
                proj.input = Box::new(self.push_filters_down(*proj.input));
                LogicalOperator::Project(proj)
            }
            LogicalOperator::Limit(mut limit) => {
                limit.input = Box::new(self.push_filters_down(*limit.input));
                LogicalOperator::Limit(limit)
            }
            LogicalOperator::Skip(mut skip) => {
                skip.input = Box::new(self.push_filters_down(*skip.input));
                LogicalOperator::Skip(skip)
            }
            LogicalOperator::Sort(mut sort) => {
                sort.input = Box::new(self.push_filters_down(*sort.input));
                LogicalOperator::Sort(sort)
            }
            LogicalOperator::Distinct(mut distinct) => {
                distinct.input = Box::new(self.push_filters_down(*distinct.input));
                LogicalOperator::Distinct(distinct)
            }
            LogicalOperator::Expand(mut expand) => {
                expand.input = Box::new(self.push_filters_down(*expand.input));
                LogicalOperator::Expand(expand)
            }
            LogicalOperator::Join(mut join) => {
                join.left = Box::new(self.push_filters_down(*join.left));
                join.right = Box::new(self.push_filters_down(*join.right));
                LogicalOperator::Join(join)
            }
            LogicalOperator::LeftJoin(mut left_join) => {
                left_join.left = Box::new(self.push_filters_down(*left_join.left));
                left_join.right = Box::new(self.push_filters_down(*left_join.right));
                LogicalOperator::LeftJoin(left_join)
            }
            LogicalOperator::AntiJoin(mut anti_join) => {
                anti_join.left = Box::new(self.push_filters_down(*anti_join.left));
                anti_join.right = Box::new(self.push_filters_down(*anti_join.right));
                LogicalOperator::AntiJoin(anti_join)
            }
            LogicalOperator::Apply(mut apply) => {
                apply.input = Box::new(self.push_filters_down(*apply.input));
                apply.subplan = Box::new(self.push_filters_down(*apply.subplan));
                LogicalOperator::Apply(apply)
            }
            LogicalOperator::Union(mut union) => {
                union.inputs = union
                    .inputs
                    .into_iter()
                    .map(|input| self.push_filters_down(input))
                    .collect();
                LogicalOperator::Union(union)
            }
            LogicalOperator::Unwind(mut unwind) => {
                unwind.input = Box::new(self.push_filters_down(*unwind.input));
                LogicalOperator::Unwind(unwind)
            }
            LogicalOperator::Aggregate(mut agg) => {
                agg.input = Box::new(self.push_filters_down(*agg.input));
                LogicalOperator::Aggregate(agg)
            }
            LogicalOperator::MapCollect(mut mc) => {
                mc.input = Box::new(self.push_filters_down(*mc.input));
                LogicalOperator::MapCollect(mc)
            }
            LogicalOperator::MultiWayJoin(mut mwj) => {
                mwj.inputs = mwj
                    .inputs
                    .into_iter()
                    .map(|input| self.push_filters_down(input))
                    .collect();
                LogicalOperator::MultiWayJoin(mwj)
            }
            // Leaf operators and unsupported operators are returned as-is
            other => other,
        }
    }

    /// Tries to push a filter predicate into the given operator.
    ///
    /// Returns either the predicate pushed into the operator, or a new
    /// Filter operator on top if the predicate cannot be pushed further.
    fn try_push_filter_into(
        &self,
        predicate: LogicalExpression,
        op: LogicalOperator,
    ) -> LogicalOperator {
        match op {
            // Can push through Project if predicate doesn't depend on computed columns
            LogicalOperator::Project(mut proj) => {
                let predicate_vars = self.extract_variables(&predicate);
                let computed_vars = self.extract_projection_aliases(&proj.projections);

                // If predicate doesn't use any computed columns, push through
                if predicate_vars.is_disjoint(&computed_vars) {
                    proj.input = Box::new(self.try_push_filter_into(predicate, *proj.input));
                    LogicalOperator::Project(proj)
                } else {
                    // Can't push through, keep filter on top
                    LogicalOperator::Filter(FilterOp {
                        predicate,
                        pushdown_hint: None,
                        input: Box::new(LogicalOperator::Project(proj)),
                    })
                }
            }

            // Can push through Return (which is like a projection)
            LogicalOperator::Return(mut ret) => {
                ret.input = Box::new(self.try_push_filter_into(predicate, *ret.input));
                LogicalOperator::Return(ret)
            }

            // Can push through Expand if predicate doesn't use variables introduced by this expand
            LogicalOperator::Expand(mut expand) => {
                let predicate_vars = self.extract_variables(&predicate);

                // Variables introduced by this expand are:
                // - The target variable (to_variable)
                // - The edge variable (if any)
                // - The path alias and its physical auxiliary columns (if any)
                let path_bindings = expand
                    .path_alias
                    .as_deref()
                    .map(crate::query::binder::path_binding_names);
                let mut introduced_vars = vec![&expand.to_variable];
                if let Some(ref edge_var) = expand.edge_variable {
                    introduced_vars.push(edge_var);
                }
                if let Some(ref names) = path_bindings {
                    introduced_vars.extend(names);
                }

                // Check if predicate uses any variables introduced by this expand
                let uses_introduced_vars =
                    predicate_vars.iter().any(|v| introduced_vars.contains(&v));

                if !uses_introduced_vars {
                    // Predicate doesn't use vars from this expand, so push through
                    expand.input = Box::new(self.try_push_filter_into(predicate, *expand.input));
                    LogicalOperator::Expand(expand)
                } else {
                    // Keep filter after expand
                    LogicalOperator::Filter(FilterOp {
                        predicate,
                        pushdown_hint: None,
                        input: Box::new(LogicalOperator::Expand(expand)),
                    })
                }
            }

            // Can push through Join to left/right side based on variables used
            LogicalOperator::Join(mut join) => {
                let predicate_vars = self.extract_variables(&predicate);
                let compatibility_vars = Self::sparql_compatibility_variables(&join.conditions);
                if predicate_vars
                    .iter()
                    .any(|variable| compatibility_vars.contains(variable))
                {
                    return LogicalOperator::Filter(FilterOp {
                        predicate,
                        pushdown_hint: None,
                        input: Box::new(LogicalOperator::Join(join)),
                    });
                }
                let left_vars = self.collect_output_variables(&join.left);
                let right_vars = self.collect_output_variables(&join.right);

                let uses_left = predicate_vars.iter().any(|v| left_vars.contains(v));
                let uses_right = predicate_vars.iter().any(|v| right_vars.contains(v));

                if uses_left && !uses_right {
                    // Push to left side
                    join.left = Box::new(self.try_push_filter_into(predicate, *join.left));
                    LogicalOperator::Join(join)
                } else if uses_right && !uses_left {
                    // Push to right side
                    join.right = Box::new(self.try_push_filter_into(predicate, *join.right));
                    LogicalOperator::Join(join)
                } else {
                    // Uses both sides - keep above join
                    LogicalOperator::Filter(FilterOp {
                        predicate,
                        pushdown_hint: None,
                        input: Box::new(LogicalOperator::Join(join)),
                    })
                }
            }

            // LeftJoin pushdown is semantics-preserving only on the LEFT
            // side: anything that filters out a left row also filters out
            // every (left, NULL) pair the join would have emitted. Pushing
            // to the right side is unsafe because OPTIONAL MATCH must keep
            // left rows that have no right match.
            LogicalOperator::LeftJoin(mut left_join) => {
                let predicate_vars = self.extract_variables(&predicate);
                let left_vars = self.collect_output_variables(&left_join.left);
                let right_vars = self.collect_output_variables(&left_join.right);
                let compatibility_vars =
                    Self::sparql_compatibility_variables(&left_join.compatibility_conditions);

                let uses_left = predicate_vars.iter().any(|v| left_vars.contains(v));
                let uses_right = predicate_vars.iter().any(|v| right_vars.contains(v));
                let uses_compatibility_key = predicate_vars
                    .iter()
                    .any(|variable| compatibility_vars.contains(variable));

                if uses_compatibility_key {
                    return LogicalOperator::Filter(FilterOp {
                        predicate,
                        pushdown_hint: None,
                        input: Box::new(LogicalOperator::LeftJoin(left_join)),
                    });
                }

                if uses_left && !uses_right {
                    left_join.left =
                        Box::new(self.try_push_filter_into(predicate, *left_join.left));
                    LogicalOperator::LeftJoin(left_join)
                } else if uses_left
                    && uses_right
                    && predicate_vars
                        .iter()
                        .all(|v| left_vars.contains(v) && right_vars.contains(v))
                {
                    // Predicate references only variables that are bound on
                    // both sides (i.e. join-key variables). The OPTIONAL
                    // MATCH compiles to a LeftJoin where the right subtree
                    // independently re-binds the shared variable (typically
                    // via its own NodeScan), so the right side can balloon
                    // to the full table even when the left side is bound
                    // to a tiny set. Duplicating the predicate to both
                    // sides is safe: matched pairs satisfy left.x == right.x,
                    // so a right row that fails the predicate either has
                    // no left match or pairs with a left row that also
                    // fails. Unmatched (OPTIONAL) left rows are unaffected.
                    // This is what collapses hydrate's three independent
                    // 30k-row scans into 6-id index lookups.
                    left_join.left =
                        Box::new(self.try_push_filter_into(predicate.clone(), *left_join.left));
                    left_join.right =
                        Box::new(self.try_push_filter_into(predicate, *left_join.right));
                    LogicalOperator::LeftJoin(left_join)
                } else {
                    LogicalOperator::Filter(FilterOp {
                        predicate,
                        pushdown_hint: None,
                        input: Box::new(LogicalOperator::LeftJoin(left_join)),
                    })
                }
            }

            // Apply (correlated subquery): the input is the outer plan, the
            // subplan re-evaluates per outer row. Predicates that only use
            // outer-side variables can safely push into the input.
            LogicalOperator::Apply(mut apply) => {
                let predicate_vars = self.extract_variables(&predicate);
                let input_vars = self.collect_output_variables(&apply.input);
                let subplan_vars = self.collect_output_variables(&apply.subplan);

                let uses_input = predicate_vars.iter().any(|v| input_vars.contains(v));
                let uses_subplan = predicate_vars.iter().any(|v| subplan_vars.contains(v));

                if uses_input && !uses_subplan {
                    apply.input = Box::new(self.try_push_filter_into(predicate, *apply.input));
                    LogicalOperator::Apply(apply)
                } else {
                    LogicalOperator::Filter(FilterOp {
                        predicate,
                        pushdown_hint: None,
                        input: Box::new(LogicalOperator::Apply(apply)),
                    })
                }
            }

            // Cannot push through Aggregate (predicate refers to aggregated values)
            LogicalOperator::Aggregate(agg) => LogicalOperator::Filter(FilterOp {
                predicate,
                pushdown_hint: None,
                input: Box::new(LogicalOperator::Aggregate(agg)),
            }),

            // For NodeScan, we've reached the bottom - keep filter on top
            LogicalOperator::NodeScan(scan) => LogicalOperator::Filter(FilterOp {
                predicate,
                pushdown_hint: None,
                input: Box::new(LogicalOperator::NodeScan(scan)),
            }),

            // Filters commute (boolean conjunction is associative/commutative),
            // so we can push the outer predicate past an existing inner
            // Filter and continue trying to push further down. Without this
            // case, a pattern like
            // `Filter(r.id IN [...], Filter(hasLabel(d), Expand))` (emitted
            // by Cypher when the WHERE clause is conjoined with automatic
            // label filters) gets stuck at the top, missing the chance to
            // anchor the filter on r's NodeScan and trigger the property-
            // index fast path.
            //
            // Only commute when the outer predicate references variables
            // that are bound at or below the inner filter's input. If the
            // inner filter is a final scope-narrowing step that introduces
            // synthetic columns the outer predicate depends on (e.g. path
            // variables produced by a variable-length expand wrapped in a
            // label filter), pushing past it would evaluate the outer
            // predicate against rows where those columns aren't bound and
            // every row would be filtered out.
            LogicalOperator::Filter(inner_filter) => {
                let predicate_vars = self.extract_variables(&predicate);
                let inner_input_vars = self.collect_output_variables(&inner_filter.input);
                let safe_to_commute = predicate_vars.iter().all(|v| inner_input_vars.contains(v));
                if safe_to_commute {
                    let mut inner_filter = inner_filter;
                    inner_filter.input =
                        Box::new(self.try_push_filter_into(predicate, *inner_filter.input));
                    LogicalOperator::Filter(inner_filter)
                } else {
                    LogicalOperator::Filter(FilterOp {
                        predicate,
                        pushdown_hint: None,
                        input: Box::new(LogicalOperator::Filter(inner_filter)),
                    })
                }
            }

            // For other operators, keep filter on top
            other => LogicalOperator::Filter(FilterOp {
                predicate,
                pushdown_hint: None,
                input: Box::new(other),
            }),
        }
    }

    // NOTE: Filter-into-TripleScan pushdown is intentionally not implemented.
    // Binding a variable position in the scan removes it from output columns,
    // which breaks downstream operators (SELECT, ORDER BY) that reference it.
    // A correct implementation needs a separate `pushed_bindings` field on
    // TripleScanOp so the variable stays in output while the scan is constrained.

    /// Collects all output variable names from an operator.
    fn collect_output_variables(&self, op: &LogicalOperator) -> HashSet<String> {
        let mut vars = HashSet::new();
        Self::collect_output_variables_recursive(op, &mut vars);
        vars
    }

    /// Recursively collects output variables from an operator.
    fn collect_output_variables_recursive(op: &LogicalOperator, vars: &mut HashSet<String>) {
        match op {
            LogicalOperator::NodeScan(scan) => {
                vars.insert(scan.variable.clone());
            }
            LogicalOperator::EdgeScan(scan) => {
                vars.insert(scan.variable.clone());
            }
            LogicalOperator::Expand(expand) => {
                vars.insert(expand.to_variable.clone());
                if let Some(edge_var) = &expand.edge_variable {
                    vars.insert(edge_var.clone());
                }
                if let Some(path_alias) = &expand.path_alias {
                    vars.insert(path_alias.clone());
                    vars.insert(format!("_path_length_{path_alias}"));
                    vars.insert(format!("_path_nodes_{path_alias}"));
                    vars.insert(format!("_path_edges_{path_alias}"));
                }
                Self::collect_output_variables_recursive(&expand.input, vars);
            }
            LogicalOperator::Filter(filter) => {
                Self::collect_output_variables_recursive(&filter.input, vars);
            }
            LogicalOperator::Project(proj) => {
                for p in &proj.projections {
                    if let Some(alias) = &p.alias {
                        vars.insert(alias.clone());
                    } else if let LogicalExpression::Variable(variable) = &p.expression {
                        vars.insert(variable.clone());
                    }
                }
                if proj.pass_through_input {
                    Self::collect_output_variables_recursive(&proj.input, vars);
                }
            }
            LogicalOperator::Join(join) => {
                Self::collect_output_variables_recursive(&join.left, vars);
                Self::collect_output_variables_recursive(&join.right, vars);
            }
            // A LeftJoin outputs every variable from its (required) left input
            // plus the (optional) variables introduced on the right — those are
            // real output columns even when NULL-padded. Without this arm,
            // chained OPTIONAL MATCH (nested LeftJoins) reported an EMPTY var
            // set for every outer join, so `propagate_join_predicates` found no
            // shared join key and only mirrored the anchor filter to the
            // innermost optional. The 2nd+ OPTIONAL MATCH then re-scanned the
            // whole graph instead of reusing the indexed anchor.
            LogicalOperator::LeftJoin(join) => {
                Self::collect_output_variables_recursive(&join.left, vars);
                Self::collect_output_variables_recursive(&join.right, vars);
            }
            // Apply (lateral join) outputs the outer input's variables plus
            // whatever the per-row subplan binds; mirror the same shape so
            // correlated / OPTIONAL CALL patterns expose their join keys too.
            LogicalOperator::Apply(apply) => {
                Self::collect_output_variables_recursive(&apply.input, vars);
                Self::collect_output_variables_recursive(&apply.subplan, vars);
            }
            // Unwind preserves the input row's variables and binds one new
            // element variable (plus optional ordinality / offset variables).
            LogicalOperator::Unwind(unwind) => {
                Self::collect_output_variables_recursive(&unwind.input, vars);
                vars.insert(unwind.variable.clone());
                if let Some(ord) = &unwind.ordinality_var {
                    vars.insert(ord.clone());
                }
                if let Some(off) = &unwind.offset_var {
                    vars.insert(off.clone());
                }
            }
            // An AntiJoin (MINUS) emits only its left input's columns; the right
            // side is used purely to exclude and contributes no output columns.
            LogicalOperator::AntiJoin(join) => {
                Self::collect_output_variables_recursive(&join.left, vars);
            }
            LogicalOperator::Aggregate(agg) => {
                for expr in &agg.group_by {
                    Self::collect_variables(expr, vars);
                }
                for agg_expr in &agg.aggregates {
                    if let Some(alias) = &agg_expr.alias {
                        vars.insert(alias.clone());
                    }
                }
            }
            LogicalOperator::Return(ret) => {
                Self::collect_output_variables_recursive(&ret.input, vars);
            }
            LogicalOperator::Limit(limit) => {
                Self::collect_output_variables_recursive(&limit.input, vars);
            }
            LogicalOperator::Skip(skip) => {
                Self::collect_output_variables_recursive(&skip.input, vars);
            }
            LogicalOperator::Sort(sort) => {
                Self::collect_output_variables_recursive(&sort.input, vars);
            }
            LogicalOperator::Distinct(distinct) => {
                Self::collect_output_variables_recursive(&distinct.input, vars);
            }
            #[cfg(feature = "triple-store")]
            LogicalOperator::TripleScan(scan) => {
                if let Some(v) = scan.subject.as_variable() {
                    vars.insert(v.to_string());
                }
                if let Some(v) = scan.predicate.as_variable() {
                    vars.insert(v.to_string());
                }
                if let Some(v) = scan.object.as_variable() {
                    vars.insert(v.to_string());
                }
                if let Some(ref g) = scan.graph
                    && let Some(v) = g.as_variable()
                {
                    vars.insert(v.to_string());
                }
            }
            _ => {}
        }
    }

    /// Extracts all variable names referenced in an expression.
    fn extract_variables(&self, expr: &LogicalExpression) -> HashSet<String> {
        let mut vars = HashSet::new();
        Self::collect_variables(expr, &mut vars);
        vars
    }

    /// Recursively collects free variable names from an expression.
    pub(crate) fn collect_variables(expr: &LogicalExpression, vars: &mut HashSet<String>) {
        Self::collect_variables_scoped(expr, &HashSet::new(), vars);
    }

    /// Recurses through expression-local scopes without treating a shadowed
    /// name in a nested source expression as the local body binding.
    fn collect_variables_scoped(
        expr: &LogicalExpression,
        bound: &HashSet<String>,
        vars: &mut HashSet<String>,
    ) {
        match expr {
            LogicalExpression::Variable(name) => {
                if !bound.contains(name) {
                    vars.insert(name.clone());
                }
            }
            LogicalExpression::Property { variable, .. } => {
                if !bound.contains(variable) {
                    vars.insert(variable.clone());
                }
            }
            LogicalExpression::Binary { left, right, .. } => {
                Self::collect_variables_scoped(left, bound, vars);
                Self::collect_variables_scoped(right, bound, vars);
            }
            LogicalExpression::Unary { operand, .. } => {
                Self::collect_variables_scoped(operand, bound, vars);
            }
            LogicalExpression::FunctionCall { args, .. } => {
                for arg in args {
                    Self::collect_variables_scoped(arg, bound, vars);
                }
            }
            LogicalExpression::List(items) => {
                for item in items {
                    Self::collect_variables_scoped(item, bound, vars);
                }
            }
            LogicalExpression::Map(pairs) => {
                for (_, value) in pairs {
                    Self::collect_variables_scoped(value, bound, vars);
                }
            }
            LogicalExpression::IndexAccess { base, index } => {
                Self::collect_variables_scoped(base, bound, vars);
                Self::collect_variables_scoped(index, bound, vars);
            }
            LogicalExpression::SliceAccess { base, start, end } => {
                Self::collect_variables_scoped(base, bound, vars);
                if let Some(s) = start {
                    Self::collect_variables_scoped(s, bound, vars);
                }
                if let Some(e) = end {
                    Self::collect_variables_scoped(e, bound, vars);
                }
            }
            LogicalExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                if let Some(op) = operand {
                    Self::collect_variables_scoped(op, bound, vars);
                }
                for (cond, result) in when_clauses {
                    Self::collect_variables_scoped(cond, bound, vars);
                    Self::collect_variables_scoped(result, bound, vars);
                }
                if let Some(else_expr) = else_clause {
                    Self::collect_variables_scoped(else_expr, bound, vars);
                }
            }
            LogicalExpression::Labels(var)
            | LogicalExpression::Type(var)
            | LogicalExpression::Id(var) => {
                if !bound.contains(var) {
                    vars.insert(var.clone());
                }
            }
            LogicalExpression::Literal(_) | LogicalExpression::Parameter(_) => {}
            LogicalExpression::ListComprehension {
                variable,
                list_expr,
                filter_expr,
                map_expr,
            } => {
                // The source list is evaluated in the enclosing scope.
                Self::collect_variables_scoped(list_expr, bound, vars);
                let mut local_bound = bound.clone();
                local_bound.insert(variable.clone());
                if let Some(filter) = filter_expr {
                    Self::collect_variables_scoped(filter, &local_bound, vars);
                }
                Self::collect_variables_scoped(map_expr, &local_bound, vars);
            }
            LogicalExpression::ListPredicate {
                variable,
                list_expr,
                predicate,
                ..
            } => {
                Self::collect_variables_scoped(list_expr, bound, vars);
                let mut local_bound = bound.clone();
                local_bound.insert(variable.clone());
                Self::collect_variables_scoped(predicate, &local_bound, vars);
            }
            LogicalExpression::ExistsSubquery(_)
            | LogicalExpression::CountSubquery(_)
            | LogicalExpression::ValueSubquery(_) => {
                // Subqueries have their own variable scope.
            }
            LogicalExpression::PatternComprehension { projection, .. } => {
                Self::collect_variables_scoped(projection, bound, vars);
            }
            LogicalExpression::MapProjection { base, entries } => {
                if !bound.contains(base) {
                    vars.insert(base.clone());
                }
                for entry in entries {
                    if let crate::query::plan::MapProjectionEntry::LiteralEntry(_, expr) = entry {
                        Self::collect_variables_scoped(expr, bound, vars);
                    }
                }
            }
            LogicalExpression::Reduce {
                accumulator,
                variable,
                initial,
                list,
                expression,
            } => {
                Self::collect_variables_scoped(initial, bound, vars);
                Self::collect_variables_scoped(list, bound, vars);
                let mut local_bound = bound.clone();
                local_bound.insert(accumulator.clone());
                local_bound.insert(variable.clone());
                Self::collect_variables_scoped(expression, &local_bound, vars);
            }
        }
    }

    /// Extracts aliases from projection expressions.
    fn extract_projection_aliases(
        &self,
        projections: &[crate::query::plan::Projection],
    ) -> HashSet<String> {
        projections.iter().filter_map(|p| p.alias.clone()).collect()
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Splits a top-level conjunctive (AND-chain) predicate into individual conjuncts.
fn split_conjuncts(expr: LogicalExpression) -> Vec<LogicalExpression> {
    fn go(expr: LogicalExpression, out: &mut Vec<LogicalExpression>) {
        if let LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } = expr
        {
            go(*left, out);
            go(*right, out);
        } else {
            out.push(expr);
        }
    }
    let mut out = Vec::new();
    go(expr, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{
        AggregateExpr, AggregateFunction, AggregateOp, BinaryOp, BindOp, DistinctOp,
        ExpandDirection, ExpandOp, JoinOp, JoinType, LeftJoinOp, LimitOp, NodeScanOp, PathMode,
        ProjectOp, Projection, ReturnItem, ReturnOp, SkipOp, SortKey, SortOp, SortOrder, UnaryOp,
    };
    use grafeo_common::types::Value;

    fn variable_expand(
        path_mode: PathMode,
        path_search: PathSearch,
        edge_variable: Option<&str>,
        path_alias: Option<&str>,
        min_hops: u32,
        max_hops: Option<u32>,
    ) -> LogicalOperator {
        LogicalOperator::Expand(ExpandOp {
            from_variable: "s".to_string(),
            to_variable: "t".to_string(),
            edge_variable: edge_variable.map(str::to_string),
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["R".to_string()],
            min_hops,
            max_hops,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "s".to_string(),
                label: None,
                input: None,
            })),
            path_alias: path_alias.map(str::to_string),
            path_mode,
            path_search,
            edge_predicate: None,
            path_predicate: None,
        })
    }

    fn optimized_expand(plan: LogicalPlan) -> ExpandOp {
        let optimized = Optimizer::new().optimize(plan).unwrap();
        fn peel(op: LogicalOperator) -> ExpandOp {
            match op {
                LogicalOperator::Expand(expand) => expand,
                LogicalOperator::Return(ret) => peel(*ret.input),
                LogicalOperator::Aggregate(aggregate) => peel(*aggregate.input),
                LogicalOperator::Distinct(distinct) => peel(*distinct.input),
                LogicalOperator::Project(project) => peel(*project.input),
                LogicalOperator::Filter(filter) => peel(*filter.input),
                _ => panic!("expected a variable-length Expand in the consumer slice"),
            }
        }
        peel(optimized.root)
    }

    fn distinct_return(input: LogicalOperator, expression: &str) -> LogicalPlan {
        LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable(expression.to_string()),
                alias: None,
            }],
            distinct: true,
            input: Box::new(input),
        }))
    }

    fn count_return(input: LogicalOperator, distinct: bool) -> LogicalPlan {
        LogicalPlan::new(LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![],
            aggregates: vec![AggregateExpr {
                function: AggregateFunction::Count,
                expression: Some(LogicalExpression::Variable("t".to_string())),
                expression2: None,
                distinct_key: None,
                distinct,
                alias: Some("n".to_string()),
                percentile: None,
                separator: None,
            }],
            input: Box::new(input),
            having: None,
        }))
    }

    #[test]
    fn optimizer_selects_distinct_targets_for_return_distinct_target() {
        let expand = variable_expand(PathMode::Walk, PathSearch::All, None, None, 1, Some(3));
        let optimized = optimized_expand(distinct_return(expand, "t"));
        assert_eq!(optimized.path_search, PathSearch::DistinctTargets);
    }

    #[test]
    fn optimizer_selects_distinct_targets_for_count_distinct_target() {
        let expand = variable_expand(PathMode::Walk, PathSearch::All, None, None, 1, Some(3));
        let optimized = optimized_expand(count_return(expand, true));
        assert_eq!(optimized.path_search, PathSearch::DistinctTargets);
    }

    #[test]
    fn optimizer_keeps_all_when_distinct_requires_edge_path_or_unsafe_mode() {
        let edge_expand =
            variable_expand(PathMode::Walk, PathSearch::All, Some("r"), None, 1, Some(3));
        assert_eq!(
            optimized_expand(distinct_return(edge_expand, "r")).path_search,
            PathSearch::All
        );

        let path_expand =
            variable_expand(PathMode::Walk, PathSearch::All, None, Some("p"), 1, Some(3));
        assert_eq!(
            optimized_expand(distinct_return(path_expand, "p")).path_search,
            PathSearch::All
        );

        let unsafe_expand =
            variable_expand(PathMode::Simple, PathSearch::All, None, None, 2, Some(4));
        assert_eq!(
            optimized_expand(distinct_return(unsafe_expand, "t")).path_search,
            PathSearch::All
        );

        let count_all_expand =
            variable_expand(PathMode::Walk, PathSearch::All, None, None, 1, Some(3));
        assert_eq!(
            optimized_expand(count_return(count_all_expand, false)).path_search,
            PathSearch::All
        );
    }

    #[test]
    fn optimizer_keeps_all_for_whole_row_distinct_with_anonymous_edge() {
        let expand = variable_expand(PathMode::Walk, PathSearch::All, None, None, 1, Some(3));
        let plan = LogicalPlan::new(LogicalOperator::Distinct(DistinctOp {
            input: Box::new(expand),
            columns: None,
        }));
        assert_eq!(optimized_expand(plan).path_search, PathSearch::All);
    }

    #[test]
    fn optimizer_keeps_all_for_keyed_distinct_with_retained_edge_and_path() {
        let edge_expand =
            variable_expand(PathMode::Walk, PathSearch::All, Some("r"), None, 1, Some(3));
        let edge_plan = LogicalPlan::new(LogicalOperator::Distinct(DistinctOp {
            input: Box::new(edge_expand),
            columns: Some(vec!["t".to_string()]),
        }));
        assert_eq!(optimized_expand(edge_plan).path_search, PathSearch::All);

        let path_expand =
            variable_expand(PathMode::Walk, PathSearch::All, None, Some("p"), 1, Some(3));
        let path_plan = LogicalPlan::new(LogicalOperator::Distinct(DistinctOp {
            input: Box::new(path_expand),
            columns: Some(vec!["t".to_string()]),
        }));
        assert_eq!(optimized_expand(path_plan).path_search, PathSearch::All);
    }

    #[test]
    fn optimizer_keeps_all_when_duplicate_output_names_hide_path_columns() {
        let expand = variable_expand(PathMode::Walk, PathSearch::All, Some("r"), None, 1, Some(3));
        let project = LogicalOperator::Project(ProjectOp {
            projections: vec![
                Projection {
                    expression: LogicalExpression::Variable("r".to_string()),
                    alias: Some("x".to_string()),
                },
                Projection {
                    expression: LogicalExpression::Variable("t".to_string()),
                    alias: Some("x".to_string()),
                },
            ],
            input: Box::new(expand),
            pass_through_input: false,
        });
        let whole_row = LogicalPlan::new(LogicalOperator::Distinct(DistinctOp {
            input: Box::new(project),
            columns: None,
        }));
        assert_eq!(optimized_expand(whole_row).path_search, PathSearch::All);

        let expand = variable_expand(PathMode::Walk, PathSearch::All, Some("r"), None, 1, Some(3));
        let project = LogicalOperator::Project(ProjectOp {
            projections: vec![Projection {
                expression: LogicalExpression::Variable("t".to_string()),
                alias: Some("r".to_string()),
            }],
            input: Box::new(expand),
            pass_through_input: true,
        });
        let shadowed_pass_through = LogicalPlan::new(LogicalOperator::Distinct(DistinctOp {
            input: Box::new(project),
            columns: None,
        }));
        assert_eq!(
            optimized_expand(shadowed_pass_through).path_search,
            PathSearch::All
        );
    }

    #[test]
    fn optimizer_uses_planner_name_for_unaliased_projected_property() {
        let expand = variable_expand(PathMode::Walk, PathSearch::All, None, None, 1, Some(3));
        let project = LogicalOperator::Project(ProjectOp {
            projections: vec![Projection {
                expression: LogicalExpression::Property {
                    variable: "t".to_string(),
                    property: "id".to_string(),
                },
                alias: None,
            }],
            input: Box::new(expand),
            pass_through_input: false,
        });
        let plan = LogicalPlan::new(LogicalOperator::Distinct(DistinctOp {
            input: Box::new(project),
            columns: Some(vec!["t.id".to_string()]),
        }));
        assert_eq!(
            optimized_expand(plan).path_search,
            PathSearch::DistinctTargets
        );
    }

    #[test]
    fn optimizer_accepts_alias_and_deterministic_endpoint_filter() {
        let expand = variable_expand(PathMode::Walk, PathSearch::All, None, None, 1, Some(3));
        let project = LogicalOperator::Project(ProjectOp {
            projections: vec![Projection {
                expression: LogicalExpression::Variable("t".to_string()),
                alias: Some("target".to_string()),
            }],
            input: Box::new(expand),
            pass_through_input: false,
        });
        let filtered = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Property {
                variable: "target".to_string(),
                property: "kind".to_string(),
            },
            input: Box::new(project),
            pushdown_hint: None,
        });
        let plan = distinct_return(filtered, "target");
        assert_eq!(
            optimized_expand(plan).path_search,
            PathSearch::DistinctTargets
        );

        let hidden_edge_and_path = variable_expand(
            PathMode::Walk,
            PathSearch::All,
            Some("r"),
            Some("p"),
            1,
            Some(3),
        );
        assert_eq!(
            optimized_expand(distinct_return(hidden_edge_and_path, "t")).path_search,
            PathSearch::DistinctTargets
        );

        let path_sensitive =
            variable_expand(PathMode::Walk, PathSearch::All, Some("r"), None, 1, Some(3));
        let path_sensitive = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Property {
                variable: "r".to_string(),
                property: "weight".to_string(),
            },
            input: Box::new(path_sensitive),
            pushdown_hint: None,
        });
        assert_eq!(
            optimized_expand(distinct_return(path_sensitive, "t")).path_search,
            PathSearch::All
        );
    }

    #[test]
    fn optimizer_rejects_evaluated_path_values_hidden_by_later_distinct() {
        let expand = variable_expand(PathMode::Walk, PathSearch::All, Some("r"), None, 1, Some(3));
        let project = LogicalOperator::Project(ProjectOp {
            projections: vec![
                Projection {
                    expression: LogicalExpression::Variable("r".to_string()),
                    alias: Some("hidden_edge".to_string()),
                },
                Projection {
                    expression: LogicalExpression::Variable("t".to_string()),
                    alias: Some("target".to_string()),
                },
            ],
            input: Box::new(expand),
            pass_through_input: false,
        });
        assert_eq!(
            optimized_expand(distinct_return(project, "target")).path_search,
            PathSearch::All
        );

        let expand = variable_expand(PathMode::Walk, PathSearch::All, Some("r"), None, 1, Some(3));
        let wrapper = LogicalOperator::Return(ReturnOp {
            items: vec![
                ReturnItem {
                    expression: LogicalExpression::Variable("r".to_string()),
                    alias: Some("hidden_edge".to_string()),
                },
                ReturnItem {
                    expression: LogicalExpression::Variable("t".to_string()),
                    alias: Some("target".to_string()),
                },
            ],
            distinct: false,
            input: Box::new(expand),
        });
        assert_eq!(
            optimized_expand(distinct_return(wrapper, "target")).path_search,
            PathSearch::All
        );
    }

    #[test]
    fn optimizer_keeps_upstream_path_columns_stable_per_input_row() {
        let upstream = LogicalOperator::Expand(ExpandOp {
            from_variable: "s".to_string(),
            to_variable: "u".to_string(),
            edge_variable: Some("upstream_edge".to_string()),
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["R".to_string()],
            min_hops: 1,
            max_hops: Some(3),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "s".to_string(),
                label: None,
                input: None,
            })),
            path_alias: Some("upstream_path".to_string()),
            path_mode: PathMode::Walk,
            edge_predicate: None,
            path_predicate: None,
            path_search: PathSearch::All,
        });
        let candidate = LogicalOperator::Expand(ExpandOp {
            from_variable: "u".to_string(),
            to_variable: "t".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["R".to_string()],
            min_hops: 1,
            max_hops: Some(3),
            input: Box::new(upstream),
            path_alias: None,
            path_mode: PathMode::Walk,
            edge_predicate: None,
            path_predicate: None,
            path_search: PathSearch::All,
        });
        assert_eq!(
            optimized_expand(distinct_return(candidate, "t")).path_search,
            PathSearch::DistinctTargets
        );
    }

    #[test]
    fn optimizer_accepts_safe_distinct_key_for_endpoint_value() {
        let expand = variable_expand(PathMode::Walk, PathSearch::All, None, None, 1, Some(3));
        let plan = LogicalPlan::new(LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![],
            aggregates: vec![AggregateExpr {
                function: AggregateFunction::Count,
                expression: Some(LogicalExpression::Variable("t".to_string())),
                expression2: None,
                distinct_key: Some(LogicalExpression::Variable("t".to_string())),
                distinct: true,
                alias: Some("n".to_string()),
                percentile: None,
                separator: None,
            }],
            input: Box::new(expand),
            having: None,
        }));
        assert_eq!(
            optimized_expand(plan).path_search,
            PathSearch::DistinctTargets
        );
    }

    fn intrinsic_predicate_wrappers(expression: LogicalExpression) -> Vec<LogicalExpression> {
        vec![
            expression.clone(),
            LogicalExpression::FunctionCall {
                name: "coalesce".into(),
                args: vec![expression.clone(), LogicalExpression::Literal(false.into())],
                distinct: false,
            },
            LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Literal(true.into())),
                op: BinaryOp::And,
                right: Box::new(expression.clone()),
            },
            LogicalExpression::Case {
                operand: None,
                when_clauses: vec![(LogicalExpression::Literal(true.into()), expression.clone())],
                else_clause: Some(Box::new(LogicalExpression::Literal(false.into()))),
            },
            LogicalExpression::IndexAccess {
                base: Box::new(LogicalExpression::List(vec![expression.clone()])),
                index: Box::new(LogicalExpression::Literal(0.into())),
            },
            LogicalExpression::Map(vec![("nested".into(), expression)]),
        ]
    }

    #[test]
    fn intrinsic_edge_stability_reuses_allowlist_through_nested_wrappers() {
        let stable = LogicalExpression::Binary {
            left: Box::new(LogicalExpression::Property {
                variable: "edge".into(),
                property: "cost".into(),
            }),
            op: BinaryOp::Lt,
            right: Box::new(LogicalExpression::FunctionCall {
                name: "ABS".into(),
                args: vec![LogicalExpression::Variable("input_threshold".into())],
                distinct: false,
            }),
        };
        for wrapped in intrinsic_predicate_wrappers(stable) {
            assert!(
                Optimizer::intrinsic_edge_predicate_is_stable(&wrapped),
                "{wrapped:?}"
            );
        }
        for name in ["rand", "RaNdOm", "unknown_udf"] {
            let volatile = LogicalExpression::FunctionCall {
                name: name.into(),
                args: vec![],
                distinct: false,
            };
            for wrapped in intrinsic_predicate_wrappers(volatile) {
                assert!(
                    !Optimizer::intrinsic_edge_predicate_is_stable(&wrapped),
                    "{wrapped:?}"
                );
            }
        }
        let local_scope = LogicalExpression::ListComprehension {
            variable: "item".into(),
            list_expr: Box::new(LogicalExpression::List(vec![LogicalExpression::Literal(
                1.into(),
            )])),
            filter_expr: None,
            map_expr: Box::new(LogicalExpression::Variable("item".into())),
        };
        assert!(
            !Optimizer::intrinsic_edge_predicate_is_stable(&local_scope),
            "uncertified local scope must not become implicitly pure"
        );
    }

    #[test]
    fn optimizer_preserves_all_for_unstable_intrinsic_edge_predicates() {
        for stable in [false, true] {
            let predicate = if stable {
                LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "r".into(),
                        property: "allowed".into(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(LogicalExpression::Literal(true.into())),
                }
            } else {
                LogicalExpression::FunctionCall {
                    name: "coalesce".into(),
                    args: vec![LogicalExpression::FunctionCall {
                        name: "random".into(),
                        args: vec![],
                        distinct: false,
                    }],
                    distinct: false,
                }
            };
            let mut expand =
                variable_expand(PathMode::Walk, PathSearch::All, Some("r"), None, 1, Some(3));
            if let LogicalOperator::Expand(expand) = &mut expand {
                expand.edge_predicate = Some(predicate);
            }
            let expected = if stable {
                PathSearch::DistinctTargets
            } else {
                PathSearch::All
            };
            for plan in [
                distinct_return(expand.clone(), "t"),
                count_return(expand.clone(), true),
            ] {
                let optimized = optimized_expand(plan);
                assert_eq!(optimized.path_search, expected);
                assert!(optimized.edge_predicate.is_some());
            }
            let ordinary = optimized_expand(LogicalPlan::new(expand));
            assert_eq!(ordinary.path_search, PathSearch::All);
            assert!(ordinary.edge_predicate.is_some());
        }
    }

    #[test]
    fn optimizer_preserves_all_when_expand_has_full_path_predicate() {
        let mut expand =
            variable_expand(PathMode::Walk, PathSearch::All, None, Some("p"), 1, Some(3));
        if let LogicalOperator::Expand(expand) = &mut expand {
            expand.path_predicate = Some(LogicalExpression::FunctionCall {
                name: "length".into(),
                args: vec![LogicalExpression::Variable("p".into())],
                distinct: false,
            });
        }

        let optimized = optimized_expand(LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("t".into()),
                alias: None,
            }],
            distinct: true,
            input: Box::new(expand),
        })));
        assert_eq!(optimized.path_search, PathSearch::All);
        assert!(optimized.path_predicate.is_some());
    }

    #[test]
    fn collect_variables_excludes_local_predicate_binders() {
        let optimizer = Optimizer::new();
        let comprehension = LogicalExpression::ListComprehension {
            variable: "item".into(),
            list_expr: Box::new(LogicalExpression::Variable("items".into())),
            filter_expr: Some(Box::new(LogicalExpression::Variable("item".into()))),
            map_expr: Box::new(LogicalExpression::Variable("item".into())),
        };
        let vars = optimizer.extract_variables(&comprehension);
        assert_eq!(vars, HashSet::from(["items".to_string()]));

        let reduce = LogicalExpression::Reduce {
            accumulator: "acc".into(),
            initial: Box::new(LogicalExpression::Variable("seed".into())),
            variable: "item".into(),
            list: Box::new(LogicalExpression::Variable("items".into())),
            expression: Box::new(LogicalExpression::Variable("acc".into())),
        };
        let vars = optimizer.extract_variables(&reduce);
        assert_eq!(
            vars,
            HashSet::from(["seed".to_string(), "items".to_string()])
        );

        // The source expression is evaluated before the local binder.  The
        // same name must therefore remain a free dependency even when a
        // nested comprehension shadows it in its own body.
        let nested_shadow = LogicalExpression::ListComprehension {
            variable: "x".into(),
            list_expr: Box::new(LogicalExpression::Variable("x".into())),
            filter_expr: None,
            map_expr: Box::new(LogicalExpression::ListComprehension {
                variable: "x".into(),
                list_expr: Box::new(LogicalExpression::Variable("x".into())),
                filter_expr: None,
                map_expr: Box::new(LogicalExpression::Variable("x".into())),
            }),
        };
        let vars = optimizer.extract_variables(&nested_shadow);
        assert_eq!(vars, HashSet::from(["x".to_string()]));
    }

    #[test]
    fn path_predicate_required_columns_include_outer_list_predicate_inputs() {
        let optimizer = Optimizer::new();
        let mut expand =
            variable_expand(PathMode::Walk, PathSearch::All, None, Some("p"), 1, Some(3));
        if let LogicalOperator::Expand(expand) = &mut expand {
            expand.path_predicate = Some(LogicalExpression::ListPredicate {
                kind: crate::query::plan::ListPredicateKind::All,
                variable: "edge".into(),
                list_expr: Box::new(LogicalExpression::Variable("edges".into())),
                predicate: Box::new(LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "edge".into(),
                        property: "ok".into(),
                    }),
                    op: BinaryOp::And,
                    right: Box::new(LogicalExpression::Property {
                        variable: "fixed".into(),
                        property: "enabled".into(),
                    }),
                }),
            });
        }
        let required = optimizer.collect_required_columns(&expand);
        assert!(required.contains(&RequiredColumn::Variable("edges".into())));
        assert!(required.contains(&RequiredColumn::Variable("fixed".into())));
        assert!(!required.contains(&RequiredColumn::Variable("edge".into())));
    }

    #[test]
    fn optimizer_rejects_volatile_but_accepts_statistical_distinct_consumers() {
        let volatile = LogicalExpression::FunctionCall {
            name: "rand".to_string(),
            args: vec![],
            distinct: false,
        };
        let expand = variable_expand(PathMode::Walk, PathSearch::All, None, None, 1, Some(3));
        let volatile_plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: volatile,
                alias: None,
            }],
            distinct: true,
            input: Box::new(expand),
        }));
        assert_eq!(optimized_expand(volatile_plan).path_search, PathSearch::All);

        let expand = variable_expand(PathMode::Walk, PathSearch::All, None, None, 1, Some(3));
        let statistical_plan = LogicalPlan::new(LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![],
            aggregates: vec![AggregateExpr {
                function: AggregateFunction::StdDev,
                expression: Some(LogicalExpression::Variable("t".to_string())),
                expression2: None,
                distinct_key: None,
                distinct: true,
                alias: Some("spread".to_string()),
                percentile: None,
                separator: None,
            }],
            input: Box::new(expand),
            having: None,
        }));
        assert_eq!(
            optimized_expand(statistical_plan).path_search,
            PathSearch::DistinctTargets
        );
    }

    #[test]
    fn optimizer_selects_distinct_targets_for_endpoint_statistical_aggregates() {
        let unary_functions = [
            AggregateFunction::StdDev,
            AggregateFunction::StdDevPop,
            AggregateFunction::Variance,
            AggregateFunction::VariancePop,
            AggregateFunction::PercentileDisc,
            AggregateFunction::PercentileCont,
            AggregateFunction::Sample,
        ];
        for function in unary_functions {
            let expand = variable_expand(PathMode::Walk, PathSearch::All, None, None, 1, Some(3));
            let plan = LogicalPlan::new(LogicalOperator::Aggregate(AggregateOp {
                group_by: vec![],
                aggregates: vec![AggregateExpr {
                    function,
                    expression: Some(LogicalExpression::Variable("t".to_string())),
                    expression2: None,
                    distinct_key: Some(LogicalExpression::Variable("t".to_string())),
                    distinct: true,
                    alias: Some("value".to_string()),
                    percentile: Some(0.5),
                    separator: None,
                }],
                input: Box::new(expand),
                having: None,
            }));
            assert_eq!(
                optimized_expand(plan).path_search,
                PathSearch::DistinctTargets,
                "endpoint-only {:?} aggregate should admit target deduplication",
                function
            );
        }

        let binary_functions = [
            AggregateFunction::CovarSamp,
            AggregateFunction::CovarPop,
            AggregateFunction::Corr,
            AggregateFunction::RegrSlope,
            AggregateFunction::RegrIntercept,
            AggregateFunction::RegrR2,
            AggregateFunction::RegrCount,
            AggregateFunction::RegrSxx,
            AggregateFunction::RegrSyy,
            AggregateFunction::RegrSxy,
            AggregateFunction::RegrAvgx,
            AggregateFunction::RegrAvgy,
        ];
        for function in binary_functions {
            let expand = variable_expand(PathMode::Walk, PathSearch::All, None, None, 1, Some(3));
            let plan = LogicalPlan::new(LogicalOperator::Aggregate(AggregateOp {
                group_by: vec![],
                aggregates: vec![AggregateExpr {
                    function,
                    expression: Some(LogicalExpression::Variable("t".to_string())),
                    expression2: Some(LogicalExpression::Variable("t".to_string())),
                    distinct_key: Some(LogicalExpression::Variable("t".to_string())),
                    distinct: true,
                    alias: Some("value".to_string()),
                    percentile: None,
                    separator: None,
                }],
                input: Box::new(expand),
                having: None,
            }));
            assert_eq!(
                optimized_expand(plan).path_search,
                PathSearch::DistinctTargets,
                "endpoint-only {:?} aggregate should admit target deduplication",
                function
            );
        }
    }

    #[test]
    fn optimizer_keeps_all_when_distinct_aggregate_second_operand_is_tainted() {
        let expand = variable_expand(PathMode::Walk, PathSearch::All, Some("r"), None, 1, Some(3));
        let plan = LogicalPlan::new(LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![],
            aggregates: vec![AggregateExpr {
                function: AggregateFunction::Corr,
                expression: Some(LogicalExpression::Variable("t".to_string())),
                expression2: Some(LogicalExpression::Variable("r".to_string())),
                distinct_key: Some(LogicalExpression::Variable("t".to_string())),
                distinct: true,
                alias: Some("correlation".to_string()),
                percentile: None,
                separator: None,
            }],
            input: Box::new(expand),
            having: None,
        }));
        assert_eq!(optimized_expand(plan).path_search, PathSearch::All);
    }

    #[test]
    fn test_reorder_preserves_filter_on_relation() {
        // Join(Filter(a.age > 30, NodeScan a:Person), NodeScan b:Person) ON a.id = b.id.
        // Join reorder must NOT drop the a.age predicate when flattening the tree.
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("a".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Join(JoinOp {
                left: Box::new(LogicalOperator::Filter(FilterOp {
                    predicate: LogicalExpression::Binary {
                        left: Box::new(LogicalExpression::Property {
                            variable: "a".to_string(),
                            property: "age".to_string(),
                        }),
                        op: BinaryOp::Gt,
                        right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
                    },
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "a".to_string(),
                        label: Some("Person".to_string()),
                        input: None,
                    })),
                    pushdown_hint: None,
                })),
                right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "b".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
                join_type: JoinType::Inner,
                conditions: vec![JoinCondition {
                    left: LogicalExpression::Property {
                        variable: "a".to_string(),
                        property: "id".to_string(),
                    },
                    right: LogicalExpression::Property {
                        variable: "b".to_string(),
                        property: "id".to_string(),
                    },
                    semantics: JoinKeySemantics::Value,
                }],
            })),
        }));

        let optimized = Optimizer::new().optimize(plan).unwrap();
        let tree = optimized.root.explain_tree();
        assert!(
            tree.contains("age"),
            "join reorder dropped the a.age filter predicate; plan was:\n{tree}"
        );
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn test_rdf_join_reorder_preserves_filter_wrapped_triple_scan() {
        use crate::query::plan::{TripleComponent, TripleScanOp};

        let scan = |subject: &str, predicate: &str, object: &str| {
            LogicalOperator::TripleScan(TripleScanOp {
                subject: TripleComponent::Variable(subject.to_string()),
                predicate: TripleComponent::Iri(predicate.to_string()),
                object: TripleComponent::Variable(object.to_string()),
                graph: None,
                input: None,
                dataset: None,
            })
        };
        let filtered = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Variable("label".to_string())),
                op: BinaryOp::Ne,
                right: Box::new(LogicalExpression::Literal(Value::String("reject".into()))),
            },
            input: Box::new(scan("s", "urn:label", "label")),
            pushdown_hint: None,
        });
        let plan = LogicalPlan::new(LogicalOperator::Join(JoinOp {
            left: Box::new(filtered),
            right: Box::new(scan("s", "urn:type", "kind")),
            join_type: JoinType::Inner,
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("s".to_string()),
                right: LogicalExpression::Variable("s".to_string()),
                semantics: JoinKeySemantics::RdfTermIdentity,
            }],
        }));

        let optimized = Optimizer::new().optimize(plan).unwrap();
        let tree = optimized.root.explain_tree();
        assert!(
            tree.contains("Filter") && tree.contains("reject"),
            "RDF join reorder dropped the predicate-bearing Filter relation:\n{tree}"
        );
    }

    #[test]
    fn join_reorder_preserves_rdf_identity_key_semantics() {
        let plan = LogicalPlan::new(LogicalOperator::Join(JoinOp {
            left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "a".to_string(),
                label: None,
                input: None,
            })),
            right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "b".to_string(),
                label: None,
                input: None,
            })),
            join_type: JoinType::Inner,
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("a".to_string()),
                right: LogicalExpression::Variable("b".to_string()),
                semantics: JoinKeySemantics::RdfTermIdentity,
            }],
        }));

        let optimized = Optimizer::new().optimize(plan).unwrap();
        let LogicalOperator::Join(join) = optimized.root else {
            panic!("expected binary join after two-relation DPccp optimization");
        };
        assert_eq!(join.conditions.len(), 1);
        assert_eq!(
            join.conditions[0].semantics,
            JoinKeySemantics::RdfTermIdentity
        );
        let LogicalOperator::NodeScan(left_scan) = join.left.as_ref() else {
            panic!("expected a left node scan");
        };
        let LogicalOperator::NodeScan(right_scan) = join.right.as_ref() else {
            panic!("expected a right node scan");
        };
        assert!(matches!(
            &join.conditions[0].left,
            LogicalExpression::Variable(variable) if variable == &left_scan.variable
        ));
        assert!(matches!(
            &join.conditions[0].right,
            LogicalExpression::Variable(variable) if variable == &right_scan.variable
        ));
    }

    #[test]
    fn test_optimizer_filter_pushdown_simple() {
        // Query: MATCH (n:Person) WHERE n.age > 30 RETURN n
        // Before: Return -> Filter -> NodeScan
        // After:  Return -> Filter -> NodeScan (filter stays at bottom)

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "age".to_string(),
                    }),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
                },
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
                pushdown_hint: None,
            })),
        }));

        let optimizer = Optimizer::new();
        let optimized = optimizer.optimize(plan).unwrap();

        // The structure should remain similar (filter stays near scan)
        if let LogicalOperator::Return(ret) = &optimized.root
            && let LogicalOperator::Filter(filter) = ret.input.as_ref()
            && let LogicalOperator::NodeScan(scan) = filter.input.as_ref()
        {
            assert_eq!(scan.variable, "n");
            return;
        }
        panic!("Expected Return -> Filter -> NodeScan structure");
    }

    #[test]
    fn test_optimizer_filter_pushdown_through_expand() {
        // Query: MATCH (a:Person)-[:KNOWS]->(b) WHERE a.age > 30 RETURN b
        // The filter on 'a' should be pushed before the expand

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("b".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "a".to_string(),
                        property: "age".to_string(),
                    }),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
                },
                pushdown_hint: None,
                input: Box::new(LogicalOperator::Expand(ExpandOp {
                    from_variable: "a".to_string(),
                    to_variable: "b".to_string(),
                    edge_variable: None,
                    direction: ExpandDirection::Outgoing,
                    edge_types: vec!["KNOWS".to_string()],
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "a".to_string(),
                        label: Some("Person".to_string()),
                        input: None,
                    })),
                    path_alias: None,
                    path_mode: PathMode::Walk,
                    edge_predicate: None,
                    path_predicate: None,
                    path_search: crate::query::plan::PathSearch::All,
                })),
            })),
        }));

        let optimizer = Optimizer::new();
        let optimized = optimizer.optimize(plan).unwrap();

        // Filter on 'a' should be pushed before the expand
        // Expected: Return -> Expand -> Filter -> NodeScan
        if let LogicalOperator::Return(ret) = &optimized.root
            && let LogicalOperator::Expand(expand) = ret.input.as_ref()
            && let LogicalOperator::Filter(filter) = expand.input.as_ref()
            && let LogicalOperator::NodeScan(scan) = filter.input.as_ref()
        {
            assert_eq!(scan.variable, "a");
            assert_eq!(expand.from_variable, "a");
            assert_eq!(expand.to_variable, "b");
            return;
        }
        panic!("Expected Return -> Expand -> Filter -> NodeScan structure");
    }

    #[test]
    fn test_optimizer_filter_not_pushed_through_expand_for_target_var() {
        // Query: MATCH (a:Person)-[:KNOWS]->(b) WHERE b.age > 30 RETURN a
        // The filter on 'b' should NOT be pushed before the expand

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("a".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "b".to_string(),
                        property: "age".to_string(),
                    }),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
                },
                pushdown_hint: None,
                input: Box::new(LogicalOperator::Expand(ExpandOp {
                    from_variable: "a".to_string(),
                    to_variable: "b".to_string(),
                    edge_variable: None,
                    direction: ExpandDirection::Outgoing,
                    edge_types: vec!["KNOWS".to_string()],
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "a".to_string(),
                        label: Some("Person".to_string()),
                        input: None,
                    })),
                    path_alias: None,
                    path_mode: PathMode::Walk,
                    edge_predicate: None,
                    path_predicate: None,
                    path_search: crate::query::plan::PathSearch::All,
                })),
            })),
        }));

        let optimizer = Optimizer::new();
        let optimized = optimizer.optimize(plan).unwrap();

        // Filter on 'b' should stay after the expand
        // Expected: Return -> Filter -> Expand -> NodeScan
        if let LogicalOperator::Return(ret) = &optimized.root
            && let LogicalOperator::Filter(filter) = ret.input.as_ref()
        {
            // Check that the filter is on 'b'
            if let LogicalExpression::Binary { left, .. } = &filter.predicate
                && let LogicalExpression::Property { variable, .. } = left.as_ref()
            {
                assert_eq!(variable, "b");
            }

            if let LogicalOperator::Expand(expand) = filter.input.as_ref()
                && let LogicalOperator::NodeScan(_) = expand.input.as_ref()
            {
                return;
            }
        }
        panic!("Expected Return -> Filter -> Expand -> NodeScan structure");
    }

    #[test]
    fn test_optimizer_extract_variables() {
        let optimizer = Optimizer::new();

        let expr = LogicalExpression::Binary {
            left: Box::new(LogicalExpression::Property {
                variable: "n".to_string(),
                property: "age".to_string(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
        };

        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("n"));
    }

    // Additional tests for optimizer configuration

    #[test]
    fn test_optimizer_default() {
        let optimizer = Optimizer::default();
        // Should be able to optimize an empty plan
        let plan = LogicalPlan::new(LogicalOperator::Empty);
        let result = optimizer.optimize(plan);
        assert!(result.is_ok());
    }

    #[test]
    fn test_optimizer_with_filter_pushdown_disabled() {
        let optimizer = Optimizer::new().with_filter_pushdown(false);

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Literal(Value::Bool(true)),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
                pushdown_hint: None,
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();
        // Structure should be unchanged
        if let LogicalOperator::Return(ret) = &optimized.root
            && let LogicalOperator::Filter(_) = ret.input.as_ref()
        {
            return;
        }
        panic!("Expected unchanged structure");
    }

    #[test]
    fn test_optimizer_with_join_reorder_disabled() {
        let optimizer = Optimizer::new().with_join_reorder(false);
        assert!(
            optimizer
                .optimize(LogicalPlan::new(LogicalOperator::Empty))
                .is_ok()
        );
    }

    #[test]
    fn test_optimizer_with_cost_model() {
        let cost_model = CostModel::new();
        let optimizer = Optimizer::new().with_cost_model(cost_model);
        assert!(
            optimizer
                .cost_model()
                .estimate(&LogicalOperator::Empty, 0.0)
                .total()
                < 0.001
        );
    }

    #[test]
    fn test_optimizer_with_cardinality_estimator() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Test", TableStats::new(500));
        let optimizer = Optimizer::new().with_cardinality_estimator(estimator);

        let scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: "n".to_string(),
            label: Some("Test".to_string()),
            input: None,
        });
        let plan = LogicalPlan::new(scan);

        let cardinality = optimizer.estimate_cardinality(&plan);
        assert!((cardinality - 500.0).abs() < 0.001);
    }

    #[test]
    fn test_optimizer_estimate_cost() {
        let optimizer = Optimizer::new();
        let plan = LogicalPlan::new(LogicalOperator::NodeScan(NodeScanOp {
            variable: "n".to_string(),
            label: None,
            input: None,
        }));

        let cost = optimizer.estimate_cost(&plan);
        assert!(cost.total() > 0.0);
    }

    // Filter pushdown through various operators

    #[test]
    fn test_filter_pushdown_through_project() {
        let optimizer = Optimizer::new();

        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "age".to_string(),
                }),
                op: BinaryOp::Gt,
                right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
            },
            pushdown_hint: None,
            input: Box::new(LogicalOperator::Project(ProjectOp {
                projections: vec![Projection {
                    expression: LogicalExpression::Variable("n".to_string()),
                    alias: None,
                }],
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
                pass_through_input: false,
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter should be pushed through Project
        if let LogicalOperator::Project(proj) = &optimized.root
            && let LogicalOperator::Filter(_) = proj.input.as_ref()
        {
            return;
        }
        panic!("Expected Project -> Filter structure");
    }

    #[test]
    fn test_filter_not_pushed_through_project_with_alias() {
        let optimizer = Optimizer::new();

        // Filter on computed column 'x' should not be pushed through project that creates 'x'
        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Variable("x".to_string())),
                op: BinaryOp::Gt,
                right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
            },
            pushdown_hint: None,
            input: Box::new(LogicalOperator::Project(ProjectOp {
                projections: vec![Projection {
                    expression: LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "age".to_string(),
                    },
                    alias: Some("x".to_string()),
                }],
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
                pass_through_input: false,
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter should stay above Project
        if let LogicalOperator::Filter(filter) = &optimized.root
            && let LogicalOperator::Project(_) = filter.input.as_ref()
        {
            return;
        }
        panic!("Expected Filter -> Project structure");
    }

    #[test]
    fn test_filter_pushdown_through_limit() {
        let optimizer = Optimizer::new();

        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Literal(Value::Bool(true)),
            pushdown_hint: None,
            input: Box::new(LogicalOperator::Limit(LimitOp {
                count: 10.into(),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter stays above Limit (cannot be pushed through)
        if let LogicalOperator::Filter(filter) = &optimized.root
            && let LogicalOperator::Limit(_) = filter.input.as_ref()
        {
            return;
        }
        panic!("Expected Filter -> Limit structure");
    }

    #[test]
    fn test_filter_pushdown_through_sort() {
        let optimizer = Optimizer::new();

        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Literal(Value::Bool(true)),
            pushdown_hint: None,
            input: Box::new(LogicalOperator::Sort(SortOp {
                keys: vec![SortKey {
                    expression: LogicalExpression::Variable("n".to_string()),
                    order: SortOrder::Ascending,
                    nulls: None,
                }],
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter stays above Sort
        if let LogicalOperator::Filter(filter) = &optimized.root
            && let LogicalOperator::Sort(_) = filter.input.as_ref()
        {
            return;
        }
        panic!("Expected Filter -> Sort structure");
    }

    #[test]
    fn test_filter_pushdown_through_distinct() {
        let optimizer = Optimizer::new();

        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Literal(Value::Bool(true)),
            pushdown_hint: None,
            input: Box::new(LogicalOperator::Distinct(DistinctOp {
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
                columns: None,
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter stays above Distinct
        if let LogicalOperator::Filter(filter) = &optimized.root
            && let LogicalOperator::Distinct(_) = filter.input.as_ref()
        {
            return;
        }
        panic!("Expected Filter -> Distinct structure");
    }

    #[test]
    fn test_filter_not_pushed_through_aggregate() {
        let optimizer = Optimizer::new();

        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Variable("cnt".to_string())),
                op: BinaryOp::Gt,
                right: Box::new(LogicalExpression::Literal(Value::Int64(10))),
            },
            pushdown_hint: None,
            input: Box::new(LogicalOperator::Aggregate(AggregateOp {
                group_by: vec![],
                aggregates: vec![AggregateExpr {
                    function: AggregateFunction::Count,
                    expression: None,
                    expression2: None,
                    distinct_key: None,
                    distinct: false,
                    alias: Some("cnt".to_string()),
                    percentile: None,
                    separator: None,
                }],
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
                having: None,
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter should stay above Aggregate
        if let LogicalOperator::Filter(filter) = &optimized.root
            && let LogicalOperator::Aggregate(_) = filter.input.as_ref()
        {
            return;
        }
        panic!("Expected Filter -> Aggregate structure");
    }

    #[test]
    fn test_filter_pushdown_to_left_join_side() {
        let optimizer = Optimizer::new();

        // Filter on left variable should be pushed to left side
        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: "a".to_string(),
                    property: "age".to_string(),
                }),
                op: BinaryOp::Gt,
                right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
            },
            pushdown_hint: None,
            input: Box::new(LogicalOperator::Join(JoinOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
                right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "b".to_string(),
                    label: Some("Company".to_string()),
                    input: None,
                })),
                join_type: JoinType::Inner,
                conditions: vec![],
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter should be pushed to left side of join
        if let LogicalOperator::Join(join) = &optimized.root
            && let LogicalOperator::Filter(_) = join.left.as_ref()
        {
            return;
        }
        panic!("Expected Join with Filter on left side");
    }

    #[test]
    fn test_filter_pushdown_to_right_join_side() {
        let optimizer = Optimizer::new();

        // Filter on right variable should be pushed to right side
        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: "b".to_string(),
                    property: "name".to_string(),
                }),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Literal(Value::String("Acme".into()))),
            },
            pushdown_hint: None,
            input: Box::new(LogicalOperator::Join(JoinOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
                right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "b".to_string(),
                    label: Some("Company".to_string()),
                    input: None,
                })),
                join_type: JoinType::Inner,
                conditions: vec![],
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter should be pushed to right side of join
        if let LogicalOperator::Join(join) = &optimized.root
            && let LogicalOperator::Filter(_) = join.right.as_ref()
        {
            return;
        }
        panic!("Expected Join with Filter on right side");
    }

    #[test]
    fn test_filter_not_pushed_when_uses_both_join_sides() {
        let optimizer = Optimizer::new();

        // Filter using both variables should stay above join
        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: "a".to_string(),
                    property: "id".to_string(),
                }),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Property {
                    variable: "b".to_string(),
                    property: "a_id".to_string(),
                }),
            },
            pushdown_hint: None,
            input: Box::new(LogicalOperator::Join(JoinOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "b".to_string(),
                    label: None,
                    input: None,
                })),
                join_type: JoinType::Inner,
                conditions: vec![],
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter should stay above join
        if let LogicalOperator::Filter(filter) = &optimized.root
            && let LogicalOperator::Join(_) = filter.input.as_ref()
        {
            return;
        }
        panic!("Expected Filter -> Join structure");
    }

    /// Regression: chained OPTIONAL MATCH (nested LeftJoins) must mirror the
    /// shared-anchor filter to EVERY optional right-side, not just the
    /// innermost one. Before the `collect_output_variables` LeftJoin arm was
    /// added, outer LeftJoins reported an empty var set, so the 2nd+ optional
    /// re-scanned the whole graph instead of reusing the constrained anchor.
    #[test]
    fn test_chained_optional_match_propagates_anchor_to_all_sides() {
        // MATCH (o:Order) WHERE o.business_id = 'b1'
        // OPTIONAL MATCH (o)-[:R1]->(a)
        // OPTIONAL MATCH (o)-[:R2]->(c)
        let bid_filter = LogicalExpression::Binary {
            left: Box::new(LogicalExpression::Property {
                variable: "o".to_string(),
                property: "business_id".to_string(),
            }),
            op: BinaryOp::Eq,
            right: Box::new(LogicalExpression::Literal(Value::String("b1".into()))),
        };
        let optional_expand = |to: &str, etype: &str| {
            LogicalOperator::Expand(ExpandOp {
                from_variable: "o".to_string(),
                to_variable: to.to_string(),
                edge_variable: None,
                direction: ExpandDirection::Outgoing,
                edge_types: vec![etype.to_string()],
                min_hops: 1,
                max_hops: Some(1),
                // Optional right-sides re-derive `o` from a bare scan — exactly
                // the shape that re-scanned the whole graph before the fix.
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "o".to_string(),
                    label: None,
                    input: None,
                })),
                path_alias: None,
                path_mode: PathMode::Walk,
                edge_predicate: None,
                path_predicate: None,
                path_search: crate::query::plan::PathSearch::All,
            })
        };
        let inner = LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "o".to_string(),
                label: Some("Order".to_string()),
                input: None,
            })),
            right: Box::new(optional_expand("a", "R1")),
            condition: None,
            compatibility_conditions: Vec::new(),
        });
        let outer = LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(inner),
            right: Box::new(optional_expand("c", "R2")),
            condition: None,
            compatibility_conditions: Vec::new(),
        });
        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: bid_filter.clone(),
            pushdown_hint: None,
            input: Box::new(outer),
        }));

        let optimized = Optimizer::new().optimize(plan).unwrap();

        // The anchor must be this exact equality, rather than any predicate
        // that happens to mention `o.business_id`.
        fn is_exact_anchor(expr: &LogicalExpression) -> bool {
            match expr {
                LogicalExpression::Binary {
                    left,
                    op: BinaryOp::Eq,
                    right,
                } => {
                    matches!(
                        (left.as_ref(), right.as_ref()),
                        (
                            LogicalExpression::Property { variable, property },
                            LogicalExpression::Literal(Value::String(value)),
                        ) if variable == "o" && property == "business_id" && value.as_str() == "b1"
                    )
                }
                _ => false,
            }
        }
        fn rhs_has_exact_anchor(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Filter(f) => {
                    is_exact_anchor(&f.predicate) || rhs_has_exact_anchor(&f.input)
                }
                LogicalOperator::Expand(e) => rhs_has_exact_anchor(&e.input),
                LogicalOperator::Project(p) => rhs_has_exact_anchor(&p.input),
                LogicalOperator::Return(r) => rhs_has_exact_anchor(&r.input),
                LogicalOperator::NodeScan(s) => {
                    s.input.as_deref().is_some_and(rhs_has_exact_anchor)
                }
                _ => false,
            }
        }

        // Count and validate this one expected nested-OPTIONAL shape.  A
        // wildcard leaf would make an accidental zero-join tree pass.
        fn has_exact_two_left_joins(op: &LogicalOperator) -> bool {
            fn visit(op: &LogicalOperator, joins: &mut usize) -> bool {
                match op {
                    LogicalOperator::LeftJoin(j) => {
                        *joins += 1;
                        rhs_has_exact_anchor(&j.right)
                            && visit(&j.left, joins)
                            && visit(&j.right, joins)
                    }
                    LogicalOperator::Filter(f) => visit(&f.input, joins),
                    LogicalOperator::Expand(e) => visit(&e.input, joins),
                    LogicalOperator::Project(p) => visit(&p.input, joins),
                    LogicalOperator::Return(r) => visit(&r.input, joins),
                    LogicalOperator::Join(j) => visit(&j.left, joins) && visit(&j.right, joins),
                    LogicalOperator::NodeScan(s) => {
                        s.input.as_deref().is_none_or(|input| visit(input, joins))
                    }
                    _ => true,
                }
            }
            let mut joins = 0;
            visit(op, &mut joins) && joins == 2
        }

        let anchored_right = LogicalOperator::Filter(FilterOp {
            predicate: bid_filter.clone(),
            pushdown_hint: None,
            input: Box::new(optional_expand("a", "R1")),
        });
        let one_missing_anchor = LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(LogicalOperator::LeftJoin(LeftJoinOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "o".to_string(),
                    label: Some("Order".to_string()),
                    input: None,
                })),
                right: Box::new(anchored_right),
                condition: None,
                compatibility_conditions: Vec::new(),
            })),
            right: Box::new(optional_expand("c", "R2")),
            condition: None,
            compatibility_conditions: Vec::new(),
        });

        // Keep the negative controls explicit: each must fail the strict
        // contract for a distinct reason.
        let wrong_anchor = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: "o".to_string(),
                    property: "business_id".to_string(),
                }),
                op: BinaryOp::Ne,
                right: Box::new(LogicalExpression::Literal(Value::String("b1".into()))),
            },
            pushdown_hint: None,
            input: Box::new(optional_expand("a", "R1")),
        });
        let wrong_comparison_plan = LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(LogicalOperator::LeftJoin(LeftJoinOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "o".to_string(),
                    label: Some("Order".to_string()),
                    input: None,
                })),
                right: Box::new(wrong_anchor),
                condition: None,
                compatibility_conditions: Vec::new(),
            })),
            right: Box::new(optional_expand("c", "R2")),
            condition: None,
            compatibility_conditions: Vec::new(),
        });

        assert!(
            !has_exact_two_left_joins(&LogicalOperator::NodeScan(NodeScanOp {
                variable: "o".to_string(),
                label: Some("Order".to_string()),
                input: None,
            })),
            "anchor oracle must reject a plan with zero optional joins"
        );
        assert!(
            !has_exact_two_left_joins(&wrong_comparison_plan),
            "anchor oracle must reject a wrong comparison operator"
        );

        assert!(
            !has_exact_two_left_joins(&one_missing_anchor),
            "anchor oracle must reject a two-optional plan missing one RHS anchor"
        );

        assert!(
            has_exact_two_left_joins(&optimized.root),
            "anchor filter o.business_id must be mirrored into every OPTIONAL MATCH \
             right-side; an outer optional was left unconstrained (full re-scan): {:#?}",
            optimized.root
        );
    }

    #[test]
    fn sparql_compatibility_keys_block_left_join_filter_mirroring() {
        fn scan(variable: &str) -> LogicalOperator {
            LogicalOperator::NodeScan(NodeScanOp {
                variable: variable.to_string(),
                label: None,
                input: None,
            })
        }

        fn unbound_x() -> LogicalExpression {
            LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(LogicalExpression::FunctionCall {
                    name: "BOUND".to_string(),
                    args: vec![LogicalExpression::Variable("x".to_string())],
                    distinct: false,
                }),
            }
        }

        fn compatibility() -> Vec<JoinCondition> {
            vec![JoinCondition {
                left: LogicalExpression::Variable("x".to_string()),
                right: LogicalExpression::Variable("x".to_string()),
                semantics: JoinKeySemantics::SparqlCompatibility,
            }]
        }

        fn contains_filter(operator: &LogicalOperator) -> bool {
            matches!(operator, LogicalOperator::Filter(_))
                || operator.children().into_iter().any(contains_filter)
        }

        let optimizer = Optimizer::new()
            .with_join_reorder(false)
            .with_projection_pushdown(false);

        let propagated = optimizer
            .optimize(LogicalPlan::new(LogicalOperator::LeftJoin(LeftJoinOp {
                left: Box::new(LogicalOperator::Filter(FilterOp {
                    predicate: unbound_x(),
                    pushdown_hint: None,
                    input: Box::new(scan("x")),
                })),
                right: Box::new(scan("x")),
                condition: None,
                compatibility_conditions: compatibility(),
            })))
            .unwrap();
        let LogicalOperator::LeftJoin(propagated) = propagated.root else {
            panic!("expected compatibility LeftJoin");
        };
        assert!(contains_filter(&propagated.left));
        assert!(
            !contains_filter(&propagated.right),
            "an unbound-wildcard key cannot mirror a left predicate into the right side"
        );

        let pushed = optimizer
            .optimize(LogicalPlan::new(LogicalOperator::Filter(FilterOp {
                predicate: unbound_x(),
                pushdown_hint: None,
                input: Box::new(LogicalOperator::LeftJoin(LeftJoinOp {
                    left: Box::new(scan("x")),
                    right: Box::new(scan("x")),
                    condition: None,
                    compatibility_conditions: compatibility(),
                })),
            })))
            .unwrap();
        assert!(
            matches!(pushed.root, LogicalOperator::Filter(ref filter)
                if matches!(filter.input.as_ref(), LogicalOperator::LeftJoin(_))),
            "a predicate on a wildcard/coalescing key must remain above the LeftJoin: {:#?}",
            pushed.root
        );

        let right_bind = || {
            LogicalOperator::Bind(BindOp {
                expression: LogicalExpression::Literal(Value::String("urn:a".into())),
                variable: "x".to_string(),
                input: Box::new(LogicalOperator::Empty),
            })
        };
        for input in [
            LogicalOperator::Join(JoinOp {
                left: Box::new(scan("x")),
                right: Box::new(right_bind()),
                join_type: JoinType::Inner,
                conditions: compatibility(),
            }),
            LogicalOperator::LeftJoin(LeftJoinOp {
                left: Box::new(scan("x")),
                right: Box::new(right_bind()),
                condition: None,
                compatibility_conditions: compatibility(),
            }),
        ] {
            let optimized = optimizer
                .optimize(LogicalPlan::new(LogicalOperator::Filter(FilterOp {
                    predicate: LogicalExpression::Binary {
                        left: Box::new(LogicalExpression::Variable("x".to_string())),
                        op: BinaryOp::Eq,
                        right: Box::new(LogicalExpression::Literal(Value::String("urn:a".into()))),
                    },
                    pushdown_hint: None,
                    input: Box::new(input),
                })))
                .unwrap();
            assert!(
                matches!(optimized.root, LogicalOperator::Filter(_)),
                "compatibility metadata is authoritative even when output discovery misses a Bind: {:#?}",
                optimized.root
            );
        }
    }

    #[test]
    fn project_output_collection_respects_lexical_boundaries() {
        let optimizer = Optimizer::new();
        let closed = LogicalOperator::Project(ProjectOp {
            projections: vec![Projection {
                expression: LogicalExpression::Variable("hidden".to_string()),
                alias: Some("exported".to_string()),
            }],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "hidden".to_string(),
                label: None,
                input: None,
            })),
            pass_through_input: false,
        });
        assert_eq!(
            optimizer.collect_output_variables(&closed),
            HashSet::from(["exported".to_string()]),
            "a closed projection must not expose its input's local names"
        );

        let bare = LogicalOperator::Project(ProjectOp {
            projections: vec![Projection {
                expression: LogicalExpression::Variable("visible".to_string()),
                alias: None,
            }],
            input: Box::new(LogicalOperator::Empty),
            pass_through_input: false,
        });
        assert_eq!(
            optimizer.collect_output_variables(&bare),
            HashSet::from(["visible".to_string()]),
            "an unaliased projected variable remains an output"
        );
    }

    // Variable extraction tests

    #[test]
    fn test_extract_variables_from_variable() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::Variable("x".to_string());
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_extract_variables_from_unary() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::Unary {
            op: UnaryOp::Not,
            operand: Box::new(LogicalExpression::Variable("x".to_string())),
        };
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_extract_variables_from_function_call() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::FunctionCall {
            name: "length".to_string(),
            args: vec![
                LogicalExpression::Variable("a".to_string()),
                LogicalExpression::Variable("b".to_string()),
            ],
            distinct: false,
        };
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
    }

    #[test]
    fn test_extract_variables_from_list() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::List(vec![
            LogicalExpression::Variable("a".to_string()),
            LogicalExpression::Literal(Value::Int64(1)),
            LogicalExpression::Variable("b".to_string()),
        ]);
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
    }

    #[test]
    fn test_extract_variables_from_map() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::Map(vec![
            (
                "key1".to_string(),
                LogicalExpression::Variable("a".to_string()),
            ),
            (
                "key2".to_string(),
                LogicalExpression::Variable("b".to_string()),
            ),
        ]);
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
    }

    #[test]
    fn test_extract_variables_from_index_access() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::IndexAccess {
            base: Box::new(LogicalExpression::Variable("list".to_string())),
            index: Box::new(LogicalExpression::Variable("idx".to_string())),
        };
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("list"));
        assert!(vars.contains("idx"));
    }

    #[test]
    fn test_extract_variables_from_slice_access() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::SliceAccess {
            base: Box::new(LogicalExpression::Variable("list".to_string())),
            start: Some(Box::new(LogicalExpression::Variable("s".to_string()))),
            end: Some(Box::new(LogicalExpression::Variable("e".to_string()))),
        };
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 3);
        assert!(vars.contains("list"));
        assert!(vars.contains("s"));
        assert!(vars.contains("e"));
    }

    #[test]
    fn test_extract_variables_from_case() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::Case {
            operand: Some(Box::new(LogicalExpression::Variable("x".to_string()))),
            when_clauses: vec![(
                LogicalExpression::Literal(Value::Int64(1)),
                LogicalExpression::Variable("a".to_string()),
            )],
            else_clause: Some(Box::new(LogicalExpression::Variable("b".to_string()))),
        };
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 3);
        assert!(vars.contains("x"));
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
    }

    #[test]
    fn test_extract_variables_from_labels() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::Labels("n".to_string());
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("n"));
    }

    #[test]
    fn test_extract_variables_from_type() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::Type("e".to_string());
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("e"));
    }

    #[test]
    fn test_extract_variables_from_id() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::Id("n".to_string());
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("n"));
    }

    #[test]
    fn test_extract_variables_from_list_comprehension() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::ListComprehension {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpression::Variable("items".to_string())),
            filter_expr: Some(Box::new(LogicalExpression::Variable("pred".to_string()))),
            map_expr: Box::new(LogicalExpression::Variable("result".to_string())),
        };
        let vars = optimizer.extract_variables(&expr);
        assert!(vars.contains("items"));
        assert!(vars.contains("pred"));
        assert!(vars.contains("result"));
    }

    #[test]
    fn test_extract_variables_from_literal_and_parameter() {
        let optimizer = Optimizer::new();

        let literal = LogicalExpression::Literal(Value::Int64(42));
        assert!(optimizer.extract_variables(&literal).is_empty());

        let param = LogicalExpression::Parameter("p".to_string());
        assert!(optimizer.extract_variables(&param).is_empty());
    }

    // Recursive filter pushdown tests

    #[test]
    fn test_recursive_filter_pushdown_through_skip() {
        let optimizer = Optimizer::new();

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Literal(Value::Bool(true)),
                pushdown_hint: None,
                input: Box::new(LogicalOperator::Skip(SkipOp {
                    count: 5.into(),
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "n".to_string(),
                        label: None,
                        input: None,
                    })),
                })),
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Verify optimization succeeded
        assert!(matches!(&optimized.root, LogicalOperator::Return(_)));
    }

    #[test]
    fn test_nested_filter_pushdown() {
        let optimizer = Optimizer::new();

        // Multiple nested filters
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "x".to_string(),
                    }),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(1))),
                },
                pushdown_hint: None,
                input: Box::new(LogicalOperator::Filter(FilterOp {
                    predicate: LogicalExpression::Binary {
                        left: Box::new(LogicalExpression::Property {
                            variable: "n".to_string(),
                            property: "y".to_string(),
                        }),
                        op: BinaryOp::Lt,
                        right: Box::new(LogicalExpression::Literal(Value::Int64(10))),
                    },
                    pushdown_hint: None,
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "n".to_string(),
                        label: None,
                        input: None,
                    })),
                })),
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();
        assert!(matches!(&optimized.root, LogicalOperator::Return(_)));
    }

    /// Returns the binary join count and the sorted variable-equality pairs of
    /// their conditions, or `None` if the tree holds a `MultiWayJoin`, a
    /// non-inner `Join` or a non-variable condition. The walk goes through
    /// `children()`, which matches every operator, so no subtree is skipped.
    fn binary_join_condition_pairs(
        root: &LogicalOperator,
    ) -> Option<(usize, Vec<(String, String)>)> {
        let mut joins = 0;
        let mut pairs = Vec::new();
        let mut stack = vec![root];
        while let Some(op) = stack.pop() {
            match op {
                LogicalOperator::MultiWayJoin(_) => return None,
                LogicalOperator::Join(j) => {
                    if j.join_type != JoinType::Inner {
                        return None;
                    }
                    joins += 1;
                    for condition in &j.conditions {
                        let (LogicalExpression::Variable(l), LogicalExpression::Variable(r)) =
                            (&condition.left, &condition.right)
                        else {
                            return None;
                        };
                        let (l, r) = if l <= r { (l, r) } else { (r, l) };
                        pairs.push((l.clone(), r.clone()));
                    }
                }
                _ => {}
            }
            stack.extend(op.children());
        }
        pairs.sort();
        Some((joins, pairs))
    }

    /// Exactly `joins` inner binary joins carrying exactly `expected`
    /// variable equalities, and no `MultiWayJoin` anywhere.
    fn is_binary_join_plan(
        root: &LogicalOperator,
        joins: usize,
        expected: &[(&str, &str)],
    ) -> bool {
        let mut expected: Vec<(String, String)> = expected
            .iter()
            .map(|(l, r)| {
                let (l, r) = if l <= r { (l, r) } else { (r, l) };
                ((*l).to_string(), (*r).to_string())
            })
            .collect();
        expected.sort();
        binary_join_condition_pairs(root) == Some((joins, expected))
    }

    #[test]
    fn test_cross_name_cyclic_join_stays_binary() {
        use crate::query::plan::JoinCondition;

        // Triangle pattern: a ⋈ b ⋈ c ⋈ a (cyclic)
        let scan_a = LogicalOperator::NodeScan(NodeScanOp {
            variable: "a".to_string(),
            label: Some("Person".to_string()),
            input: None,
        });
        let scan_b = LogicalOperator::NodeScan(NodeScanOp {
            variable: "b".to_string(),
            label: Some("Person".to_string()),
            input: None,
        });
        let scan_c = LogicalOperator::NodeScan(NodeScanOp {
            variable: "c".to_string(),
            label: Some("Person".to_string()),
            input: None,
        });

        // Build: Join(Join(a, b, a=b), c, b=c) with extra condition c=a
        let join_ab = LogicalOperator::Join(JoinOp {
            left: Box::new(scan_a),
            right: Box::new(scan_b),
            join_type: JoinType::Inner,
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("a".to_string()),
                right: LogicalExpression::Variable("b".to_string()),
                semantics: JoinKeySemantics::Value,
            }],
        });

        let join_abc = LogicalOperator::Join(JoinOp {
            left: Box::new(join_ab),
            right: Box::new(scan_c),
            join_type: JoinType::Inner,
            conditions: vec![
                JoinCondition {
                    left: LogicalExpression::Variable("b".to_string()),
                    right: LogicalExpression::Variable("c".to_string()),
                    semantics: JoinKeySemantics::Value,
                },
                JoinCondition {
                    left: LogicalExpression::Variable("c".to_string()),
                    right: LogicalExpression::Variable("a".to_string()),
                    semantics: JoinKeySemantics::Value,
                },
            ],
        });

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("a".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(join_abc),
        }));

        let mut optimizer = Optimizer::new();
        optimizer
            .card_estimator
            .add_table_stats("Person", cardinality::TableStats::new(1000));

        let optimized = optimizer.optimize(plan).unwrap();

        let triangle = [("a", "b"), ("b", "c"), ("c", "a")];
        assert!(
            is_binary_join_plan(&optimized.root, 2, &triangle),
            "cross-name equality conditions must remain on binary joins that execute them: {:#?}",
            optimized.root
        );

        // Controls, each derived from the accepted plan so that exactly one
        // property differs: a MultiWayJoin replacing a scan under the outer
        // join, a third binary join, and a dropped closing condition.
        fn outer_join(plan: &mut LogicalOperator) -> &mut JoinOp {
            let LogicalOperator::Return(ret) = plan else {
                panic!("optimized plan must keep its Return root");
            };
            let LogicalOperator::Join(outer) = ret.input.as_mut() else {
                panic!("Return must sit directly on the outer join");
            };
            outer
        }

        let mut multi_way_leaf = optimized.root.clone();
        *outer_join(&mut multi_way_leaf).right =
            LogicalOperator::MultiWayJoin(crate::query::plan::MultiWayJoinOp {
                inputs: vec![],
                conditions: vec![],
                shared_variables: vec![],
            });
        assert!(!is_binary_join_plan(&multi_way_leaf, 2, &triangle));

        let mut extra_join = optimized.root.clone();
        let outer = outer_join(&mut extra_join);
        let right = std::mem::replace(outer.right.as_mut(), LogicalOperator::Empty);
        *outer.right = LogicalOperator::Join(JoinOp {
            left: Box::new(right),
            right: Box::new(LogicalOperator::Empty),
            join_type: JoinType::Inner,
            conditions: vec![],
        });
        assert!(!is_binary_join_plan(&extra_join, 2, &triangle));

        let mut dropped = optimized.root;
        outer_join(&mut dropped).conditions.pop();
        assert!(!is_binary_join_plan(&dropped, 2, &triangle));
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn test_multi_way_requires_homogeneous_owned_equivalence_conditions() {
        use crate::query::plan::{TripleComponent, TripleScanOp};

        fn scan(subject: &str, predicate: &str, object: &str) -> LogicalOperator {
            LogicalOperator::TripleScan(TripleScanOp {
                subject: TripleComponent::Variable(subject.to_string()),
                predicate: TripleComponent::Iri(predicate.to_string()),
                object: TripleComponent::Variable(object.to_string()),
                graph: None,
                input: None,
                dataset: None,
            })
        }

        fn plan(semantics: [JoinKeySemantics; 3]) -> LogicalPlan {
            let first = LogicalOperator::Join(JoinOp {
                left: Box::new(scan("a", "urn:p", "b")),
                right: Box::new(scan("b", "urn:q", "c")),
                join_type: JoinType::Inner,
                conditions: vec![JoinCondition {
                    left: LogicalExpression::Variable("b".to_string()),
                    right: LogicalExpression::Variable("b".to_string()),
                    semantics: semantics[0],
                }],
            });
            LogicalPlan::new(LogicalOperator::Join(JoinOp {
                left: Box::new(first),
                right: Box::new(scan("c", "urn:r", "a")),
                join_type: JoinType::Inner,
                conditions: vec![
                    JoinCondition {
                        left: LogicalExpression::Variable("c".to_string()),
                        right: LogicalExpression::Variable("c".to_string()),
                        semantics: semantics[1],
                    },
                    JoinCondition {
                        left: LogicalExpression::Variable("a".to_string()),
                        right: LogicalExpression::Variable("a".to_string()),
                        semantics: semantics[2],
                    },
                ],
            }))
        }

        fn is_multi_way(operator: &LogicalOperator) -> bool {
            matches!(operator, LogicalOperator::MultiWayJoin(_))
                || operator.children().into_iter().any(is_multi_way)
        }

        let optimizer = Optimizer::new();
        let identity = optimizer
            .optimize(plan([JoinKeySemantics::RdfTermIdentity; 3]))
            .unwrap();
        assert!(
            is_multi_way(&identity.root),
            "homogeneous same-name identity edges retain MultiWay metadata"
        );

        for semantics in [
            [JoinKeySemantics::SparqlCompatibility; 3],
            [
                JoinKeySemantics::RdfTermIdentity,
                JoinKeySemantics::Value,
                JoinKeySemantics::RdfTermIdentity,
            ],
        ] {
            let optimized = optimizer.optimize(plan(semantics)).unwrap();
            assert!(
                !is_multi_way(&optimized.root),
                "compatibility or mixed semantics require ownership-preserving binary joins"
            );
        }
    }

    #[test]
    fn test_acyclic_join_uses_binary_joins() {
        use crate::query::plan::JoinCondition;

        // Chain: a ⋈ b ⋈ c (acyclic)
        let scan_a = LogicalOperator::NodeScan(NodeScanOp {
            variable: "a".to_string(),
            label: Some("Person".to_string()),
            input: None,
        });
        let scan_b = LogicalOperator::NodeScan(NodeScanOp {
            variable: "b".to_string(),
            label: Some("Person".to_string()),
            input: None,
        });
        let scan_c = LogicalOperator::NodeScan(NodeScanOp {
            variable: "c".to_string(),
            label: Some("Company".to_string()),
            input: None,
        });

        let join_ab = LogicalOperator::Join(JoinOp {
            left: Box::new(scan_a),
            right: Box::new(scan_b),
            join_type: JoinType::Inner,
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("a".to_string()),
                right: LogicalExpression::Variable("b".to_string()),
                semantics: JoinKeySemantics::Value,
            }],
        });

        let join_abc = LogicalOperator::Join(JoinOp {
            left: Box::new(join_ab),
            right: Box::new(scan_c),
            join_type: JoinType::Inner,
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("b".to_string()),
                right: LogicalExpression::Variable("c".to_string()),
                semantics: JoinKeySemantics::Value,
            }],
        });

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("a".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(join_abc),
        }));

        let mut optimizer = Optimizer::new();
        optimizer
            .card_estimator
            .add_table_stats("Person", cardinality::TableStats::new(1000));
        optimizer
            .card_estimator
            .add_table_stats("Company", cardinality::TableStats::new(100));

        let optimized = optimizer.optimize(plan).unwrap();

        let chain = [("a", "b"), ("b", "c")];
        assert!(
            is_binary_join_plan(&optimized.root, 2, &chain),
            "Acyclic join should use two binary joins, not MultiWayJoin: {:#?}",
            optimized.root
        );

        // Control: a MultiWayJoin below an operator the old walker skipped.
        let sort_nested = LogicalOperator::Return(ReturnOp {
            items: vec![],
            distinct: false,
            input: Box::new(LogicalOperator::Sort(SortOp {
                keys: vec![],
                input: Box::new(LogicalOperator::MultiWayJoin(
                    crate::query::plan::MultiWayJoinOp {
                        inputs: vec![],
                        conditions: vec![],
                        shared_variables: vec![],
                    },
                )),
            })),
        });
        assert!(!is_binary_join_plan(&sort_nested, 0, &[]));
    }
}
