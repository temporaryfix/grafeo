//! SPARQL to LogicalPlan translator.
//!
//! Translates SPARQL 1.1 AST to the common logical plan representation.

use super::common::{wrap_distinct, wrap_filter, wrap_limit, wrap_skip, wrap_sort};
use crate::query::plan::{
    AddGraphOp, AggregateExpr, AggregateFunction, AggregateOp, AntiJoinOp, AntiJoinSemantics,
    BinaryOp, BindOp, ClearGraphOp, ConstructOp, CopyGraphOp, CreateGraphOp, DatasetRestriction,
    DeleteTripleOp, DropGraphOp, InsertTripleOp, JoinCondition, JoinKeySemantics, JoinOp, JoinType,
    LeftJoinOp, LoadGraphOp, LogicalExpression, LogicalOperator, LogicalPlan, ModifyOp,
    MoveGraphOp, PathStep, ProjectOp, Projection, PropertyPathOp, RDF_DISTINCT_TERM_OR_VALUE_KEY,
    RDF_EXPLICIT_EMPTY_DEFAULT_DATASET, RDF_EXPLICIT_EMPTY_NAMED_DATASET, RDF_IS_BLANK, RDF_IS_IRI,
    RDF_IS_LITERAL, RDF_IS_NUMERIC, RDF_NUMERIC_VALUE, RDF_SAME_TERM, RDF_SEALED_MODIFY_COLUMN,
    RDF_TAG_BLANK_TERM, RDF_TAG_BOUND_TERM, RDF_TAG_EXACT, RDF_TAG_IRI_TERM,
    RDF_TAG_LANG_LITERAL_TERM, RDF_TAG_LITERAL_TERM, RDF_TAG_TYPED_LITERAL_TERM, RDF_TAG_VALUE,
    RDF_TERM_EQUAL, RDF_TERM_IDENTITY_KEY, RDF_TERM_IN, RDF_TERM_OR_NATIVE_EXACT,
    RDF_TERM_OR_NATIVE_VALUE, RDF_TERM_OR_NATIVE_VISIBLE, SortKey, SortOrder, TripleComponent,
    TripleScanOp, TripleTemplate, UnaryOp, UnionOp, is_rdf_internal_term_column,
    rdf_exact_term_column, rdf_graph_variable_template, rdf_group_key_column,
    rdf_identity_key_column, rdf_tagged_term_column,
};
use crate::query::planner::rdf::rdf_numeric_literal_is_valid;
use grafeo_adapters::query::sparql::{self, ast};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use grafeo_core::graph::rdf::{Literal, Term};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};

/// Global counter for generating unique query IDs (blank node scoping).
static QUERY_ID_COUNTER: AtomicU32 = AtomicU32::new(0);

#[cfg(test)]
std::thread_local! {
    static ORDINARY_EXACT_PATTERN_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ORDINARY_EXACT_PATTERN_CONSUMES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ORDINARY_EXACT_EXPRESSION_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ORDINARY_EXACT_EXPRESSION_ANALYSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ORDINARY_EXACT_FALLBACK_ANALYSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ORDINARY_EXACT_DEMAND_DELTA_WORK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ORDINARY_EXACT_SEED_COPY_WORK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ORDINARY_EXACT_STRUCTURAL_INSERT_WORK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ORDINARY_EXACT_CORRELATION_WORK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Translates a SPARQL query string to a logical plan.
///
/// # Errors
///
/// Returns an error if the query cannot be parsed or translated.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    crate::query::plan_depth::admit(translate_unchecked(query)?)
}

fn translate_unchecked(query: &str) -> Result<LogicalPlan> {
    // Check for EXPLAIN [ANALYZE] prefix (case-insensitive, non-standard extension).
    // EXPLAIN: show physical plan without executing.
    // EXPLAIN ANALYZE: execute with profiling, show actual vs estimated stats.
    let trimmed = query.trim_start();
    let (explain, profile, actual_query) = if trimmed
        .get(..7)
        .is_some_and(|s| s.eq_ignore_ascii_case("EXPLAIN"))
        && trimmed
            .as_bytes()
            .get(7)
            .is_some_and(u8::is_ascii_whitespace)
    {
        let rest = trimmed[7..].trim_start();
        if rest
            .get(..7)
            .is_some_and(|s| s.eq_ignore_ascii_case("ANALYZE"))
            && rest.as_bytes().get(7).is_some_and(u8::is_ascii_whitespace)
        {
            // EXPLAIN ANALYZE: execute with profiling
            (false, true, rest[7..].trim_start())
        } else {
            // EXPLAIN: show plan only
            (true, false, rest)
        }
    } else {
        (false, false, query)
    };

    let sparql_query = sparql::parse(actual_query)?;
    let mut translator = SparqlTranslator::new();
    let mut plan = translator.translate_query(&sparql_query)?;
    plan.explain = explain;
    plan.profile = profile;
    Ok(plan)
}

/// Translator from SPARQL AST to LogicalPlan.
struct SparqlTranslator {
    /// Prefix mappings for IRI resolution.
    prefixes: HashMap<String, String>,
    /// Base IRI for relative IRI resolution.
    base: Option<String>,
    /// Counter for generating anonymous variables.
    anon_counter: u32,
    /// Stack of active graph contexts (pushed/popped around GRAPH patterns).
    graph_context_stack: Vec<TripleComponent>,
    /// Unique ID for this query (used for blank node scoping).
    query_id: u32,
    /// Dataset restriction from FROM / FROM NAMED clauses, if any.
    dataset: Option<DatasetRestriction>,
    /// Whether VALUES/BIND must carry sealed RDF term identity for an update template.
    exact_mutation_bindings: bool,
    /// Variables whose RDF identity can flow into the active update template.
    exact_mutation_variables: Option<HashSet<String>>,
    /// Variables whose producers control mutation solution selection.
    /// Kept separate from exact identity requirements so ordinary boolean
    /// extension results remain usable while nested RDF predicates are traced.
    mutation_control_variables: Option<HashSet<String>>,
    /// Expression nesting while translating; bounded by
    /// [`MAX_PLAN_DEPTH`](crate::query::plan_depth::MAX_PLAN_DEPTH).
    expression_depth: usize,
}

/// One SELECT's ordered set-function registry. Borrowed AST and annotation
/// references keep physical preparation tied to the original expression tree
/// even after consumer aggregate leaves are rewritten to canonical variables.
#[derive(Debug, Default)]
struct AggregateHoist<'a> {
    entries: Vec<AggregateHoistEntry<'a>>,
}

#[allow(dead_code)]
#[derive(Debug)]
struct AggregateHoistEntry<'a> {
    aggregate: &'a ast::AggregateExpression,
    canonical_column: String,
    canonical_is_direct_projection: bool,
    occurrences: Vec<AggregateHoistOccurrence<'a>>,
}

#[allow(dead_code)]
#[derive(Debug)]
struct AggregateHoistOccurrence<'a> {
    location: AggregateHoistLocation,
    direct_alias: Option<&'a str>,
    demand: AggregateHoistResultDemand,
    annotation: Option<&'a OrdinaryAggregateExactAnnotations>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AggregateHoistLocation {
    Projection { index: usize, direct: bool },
    Having,
    OrderBy { index: usize },
}

/// Records the aggregate result state demanded by one original consumer.
/// Direct roots retain analyzed exact demand; nested/modifier consumers use
/// the conservative full RDF-or-native envelope that Task 3 will prepare.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct AggregateHoistResultDemand {
    exact: bool,
    full_rdf_or_native: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct AggregateHoistMutationDemand {
    exact: bool,
    control: bool,
}

impl AggregateHoistResultDemand {
    const FULL: Self = Self {
        exact: true,
        full_rdf_or_native: true,
    };

    const fn direct(exact: bool) -> Self {
        Self {
            exact,
            full_rdf_or_native: false,
        }
    }

    const fn union(self, other: Self) -> Self {
        Self {
            exact: self.exact || other.exact,
            full_rdf_or_native: self.full_rdf_or_native || other.full_rdf_or_native,
        }
    }
}

#[allow(dead_code)]
impl AggregateHoistEntry<'_> {
    fn result_demand(&self) -> AggregateHoistResultDemand {
        self.occurrences.iter().fold(
            AggregateHoistResultDemand::default(),
            |demand, occurrence| demand.union(occurrence.demand),
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExactBoundTermKind {
    Iri,
    Blank,
    Literal,
}

#[derive(Debug)]
struct OrdinarySelectExactAnnotations {
    variables: OrdinaryVariableTable,
    exports: Vec<String>,
    projection: Vec<OrdinaryProjectionExactAnnotation>,
    group_by: Option<Vec<OrdinaryGroupExactAnnotation>>,
    having: Option<OrdinaryExpressionExactAnnotations>,
    order_by: Option<Vec<OrdinaryExpressionExactAnnotations>>,
    where_clause: OrdinaryPatternExactAnnotations,
}

#[derive(Debug)]
struct OrdinaryProjectionExactAnnotation {
    exact: bool,
    rdf_or_native: bool,
    expression: OrdinaryExpressionExactAnnotations,
}

#[derive(Debug)]
enum OrdinaryGroupExactAnnotation {
    Variable,
    Expression {
        exact: bool,
        expression: OrdinaryExpressionExactAnnotations,
    },
    BuiltInCall(OrdinaryExpressionExactAnnotations),
}

#[derive(Debug)]
enum OrdinaryExpressionExactAnnotations {
    Variable,
    Iri,
    Literal,
    Binary(Box<Self>, Box<Self>),
    Unary(Box<Self>),
    FunctionCall(Vec<Self>),
    Bound,
    Conditional {
        condition: Box<Self>,
        then_expression: Box<Self>,
        else_expression: Box<Self>,
    },
    Coalesce(Vec<Self>),
    Exists {
        pattern: Box<OrdinaryPatternExactAnnotations>,
        correlated: Vec<usize>,
    },
    NotExists {
        pattern: Box<OrdinaryPatternExactAnnotations>,
        correlated: Vec<usize>,
    },
    In {
        expression: Box<Self>,
        list: Vec<Self>,
    },
    NotIn {
        expression: Box<Self>,
        list: Vec<Self>,
    },
    Aggregate(OrdinaryAggregateExactAnnotations),
    Bracketed(Box<Self>),
}

#[derive(Debug)]
enum OrdinaryAggregateExactAnnotations {
    Count(Option<Box<OrdinaryExpressionExactAnnotations>>),
    Sum(Box<OrdinaryExpressionExactAnnotations>),
    Average(Box<OrdinaryExpressionExactAnnotations>),
    Minimum(Box<OrdinaryExpressionExactAnnotations>),
    Maximum(Box<OrdinaryExpressionExactAnnotations>),
    Sample(Box<OrdinaryExpressionExactAnnotations>),
    GroupConcat(Box<OrdinaryExpressionExactAnnotations>),
}

#[derive(Debug)]
enum OrdinaryPatternExactAnnotations {
    Basic {
        shared: Vec<usize>,
    },
    Group(Vec<Self>),
    Optional {
        pattern: Box<Self>,
        correlated: Vec<usize>,
    },
    Union(Vec<OrdinaryUnionExactAnnotation>),
    NamedGraph {
        pattern: Box<Self>,
        shared_graph: Option<usize>,
    },
    Minus {
        pattern: Box<Self>,
        correlated: Vec<usize>,
    },
    Filter(OrdinaryExpressionExactAnnotations),
    Bind {
        variable: usize,
        exact: bool,
        rdf_or_native: bool,
        expression: OrdinaryExpressionExactAnnotations,
    },
    InlineData {
        shared: Vec<usize>,
    },
    SubSelect {
        select: Box<OrdinarySelectExactAnnotations>,
        shared: Vec<usize>,
    },
    Service,
}

#[derive(Debug)]
struct OrdinaryUnionExactAnnotation {
    pattern: OrdinaryPatternExactAnnotations,
    correlated: Vec<usize>,
}

impl OrdinaryPatternExactAnnotations {
    fn kind(&self) -> &'static str {
        match self {
            Self::Basic { .. } => "Basic",
            Self::Group(_) => "Group",
            Self::Optional { .. } => "Optional",
            Self::Union(_) => "Union",
            Self::NamedGraph { .. } => "NamedGraph",
            Self::Minus { .. } => "Minus",
            Self::Filter(_) => "Filter",
            Self::Bind { .. } => "Bind",
            Self::InlineData { .. } => "InlineData",
            Self::SubSelect { .. } => "SubSelect",
            Self::Service => "Service",
        }
    }
}

#[derive(Debug, Default)]
struct OrdinaryVariableTable {
    by_name: HashMap<String, usize>,
    names: Vec<String>,
}

impl OrdinaryVariableTable {
    fn intern(&mut self, name: &str) -> usize {
        if let Some(id) = self.by_name.get(name) {
            return *id;
        }
        let id = self.names.len();
        self.names.push(name.to_string());
        self.by_name.insert(name.to_string(), id);
        id
    }

    fn get(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }
}

#[derive(Default)]
struct OrdinaryMembershipState {
    present: Vec<bool>,
    bound_since: Vec<usize>,
    undo: Vec<(usize, bool, usize)>,
}

impl OrdinaryMembershipState {
    fn ensure(&mut self, id: usize) {
        if self.present.len() <= id {
            self.present.resize(id + 1, false);
            self.bound_since.resize(id + 1, usize::MAX);
        }
    }

    fn mark(&self) -> usize {
        self.undo.len()
    }

    fn contains(&self, id: usize) -> bool {
        self.present.get(id).copied().unwrap_or(false)
    }

    fn was_present_at(&self, mark: usize, id: usize) -> bool {
        self.contains(id) && self.bound_since[id] < mark
    }

    fn insert(&mut self, id: usize) -> bool {
        self.ensure(id);
        if self.present[id] {
            return false;
        }
        let log_index = self.undo.len();
        self.undo.push((id, self.present[id], self.bound_since[id]));
        self.present[id] = true;
        self.bound_since[id] = log_index;
        true
    }

    fn remove(&mut self, id: usize) -> bool {
        self.ensure(id);
        if !self.present[id] {
            return false;
        }
        self.undo.push((id, self.present[id], self.bound_since[id]));
        self.present[id] = false;
        self.bound_since[id] = usize::MAX;
        true
    }

    fn rollback(&mut self, mark: usize) -> Result<()> {
        // Validate the whole undo range before changing membership, so an
        // inconsistent annotation state cannot leave a partially undone scope.
        for (id, _, _) in self.undo.iter().skip(mark) {
            if self.present.get(*id).is_none() {
                return Err(Error::Internal(
                    "ordinary exact-demand membership slot is missing".to_string(),
                ));
            }
            if self.bound_since.get(*id).is_none() {
                return Err(Error::Internal(
                    "ordinary exact-demand binding epoch is missing".to_string(),
                ));
            }
        }
        while self.undo.len() > mark {
            let (id, previous, previous_since) = self.undo.pop().ok_or_else(|| {
                Error::Internal("ordinary exact-demand undo entry is missing".to_string())
            })?;
            let present = self.present.get_mut(id).ok_or_else(|| {
                Error::Internal("ordinary exact-demand membership slot is missing".to_string())
            })?;
            let bound_since = self.bound_since.get_mut(id).ok_or_else(|| {
                Error::Internal("ordinary exact-demand binding epoch is missing".to_string())
            })?;
            *present = previous;
            *bound_since = previous_since;
        }
        Ok(())
    }

    fn changed_ids_since(&self, mark: usize) -> Vec<usize> {
        let mut changed = Vec::new();
        let mut seen = HashSet::new();
        for (id, _, _) in &self.undo[mark..] {
            #[cfg(test)]
            ORDINARY_EXACT_DEMAND_DELTA_WORK.with(|work| work.set(work.get() + 1));
            if seen.insert(*id) {
                changed.push(*id);
            }
        }
        changed
    }
}

impl SparqlTranslator {
    fn new() -> Self {
        Self {
            prefixes: HashMap::new(),
            base: None,
            anon_counter: 0,
            graph_context_stack: Vec::new(),
            query_id: QUERY_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            dataset: None,
            exact_mutation_bindings: false,
            exact_mutation_variables: None,
            mutation_control_variables: None,
            expression_depth: 0,
        }
    }

    fn translate_query(&mut self, query: &ast::Query) -> Result<LogicalPlan> {
        // Process prologue
        if let Some(base) = &query.base {
            self.base = Some(base.as_str().to_string());
        }

        for prefix in &query.prefixes {
            self.prefixes
                .insert(prefix.prefix.clone(), prefix.namespace.as_str().to_string());
        }

        // Translate query form
        match &query.query_form {
            ast::QueryForm::Select(select) => self.translate_select(select),
            ast::QueryForm::Ask(ask) => self.translate_ask(ask),
            ast::QueryForm::Construct(construct) => self.translate_construct(construct),
            ast::QueryForm::Describe(describe) => self.translate_describe(describe),
            ast::QueryForm::Update(update) => self.translate_update(update),
        }
    }

    fn translate_select(&mut self, select: &ast::SelectQuery) -> Result<LogicalPlan> {
        Self::validate_select_assignment_targets(select)?;
        Self::validate_supported_aggregate_placements(select)?;
        if self.exact_mutation_bindings {
            self.translate_select_with_annotations(select, None)
        } else {
            let (annotations, _) =
                Self::ordinary_select_exact_annotations(select, &HashSet::new())?;
            self.translate_select_with_annotations(select, Some(&annotations))
        }
    }

    fn translate_select_with_annotations(
        &mut self,
        select: &ast::SelectQuery,
        ordinary_annotations: Option<&OrdinarySelectExactAnnotations>,
    ) -> Result<LogicalPlan> {
        // Subqueries enter this annotated path directly, so keep the placement
        // guard here as well as at the public SELECT entry point.
        Self::validate_select_assignment_targets(select)?;
        Self::validate_supported_aggregate_placements(select)?;
        if let Some(annotations) = ordinary_annotations {
            Self::validate_ordinary_select_annotations(select, annotations, "SELECT")?;
        }
        let aggregate_hoist = self.collect_aggregate_hoist(select, ordinary_annotations)?;
        // A subselect inherits its enclosing active dataset. An explicit FROM
        // clause overrides it only for this SELECT, and every exit restores the
        // enclosing scope so later sibling patterns see the same dataset.
        let previous_dataset = self.dataset.clone();
        if let Some(dataset) = self.translate_dataset_clause(&select.dataset) {
            self.dataset = Some(dataset);
        }

        // A SELECT introduces a lexical variable scope. Derive the exact RDF
        // requirements that cross its projection boundary, then add selectors
        // local to this SELECT. Hidden same-spelled variables in nested scopes
        // must never inherit the parent's requirements.
        let previous_exact_variables = if self.exact_mutation_bindings {
            let previous = self.exact_mutation_variables.take();
            let previous_control = self.mutation_control_variables.take();
            let (local_exact, local_control) = Self::select_exact_dependencies(
                select,
                previous.as_ref(),
                previous_control.as_ref(),
            );
            self.exact_mutation_variables = Some(local_exact);
            self.mutation_control_variables = Some(local_control);
            // Preserve both lexical parent scopes in one restoration value.
            let previous = (previous, previous_control);
            Some(previous)
        } else {
            None
        };
        let translated = (|| {
            // Start with the WHERE clause pattern
            let mut plan = self.translate_graph_pattern_with_annotations(
                &select.where_clause,
                ordinary_annotations.map(|annotations| &annotations.where_clause),
            )?;

            // Any legal set function in this SELECT scope creates an implicit
            // global group, including modifier-only occurrences.
            let is_grouped = Self::select_is_grouped(select);
            let mut grouped_expression_sources = Vec::new();

            // Apply GROUP BY if present, OR create aggregate for implicit aggregation
            if is_grouped {
                let mut aggregates = self.extract_aggregates_for_select(&aggregate_hoist)?;
                let (
                    prepared_plan,
                    prepared_aggregates,
                    literal_aggregate_aliases,
                    tagged_aggregate_aliases,
                    rdf_or_native_aggregate_aliases,
                ) = self.prepare_exact_aggregates(plan, select, &aggregate_hoist, aggregates)?;
                plan = prepared_plan;
                aggregates = prepared_aggregates;

                // Get explicit GROUP BY expressions, or empty vec for whole-dataset aggregation
                let group_by_exprs = if let Some(group_by) = &select.solution_modifiers.group_by {
                    let mut expressions = Vec::with_capacity(group_by.len());
                    for (index, group) in group_by.iter().enumerate() {
                        let annotation = ordinary_annotations
                            .and_then(|annotations| annotations.group_by.as_ref())
                            .and_then(|annotations| annotations.get(index));
                        let translated = match group {
                            ast::GroupCondition::Variable(variable) => {
                                grouped_expression_sources.push((
                                    ast::Expression::Variable(variable.clone()),
                                    variable.clone(),
                                ));
                                self.translate_group_condition_with_annotations(group, annotation)?
                            }
                            ast::GroupCondition::Expression { expression, alias } => {
                                let expression_annotation = match annotation {
                                    Some(OrdinaryGroupExactAnnotation::Expression {
                                        expression,
                                        ..
                                    }) => Some(expression),
                                    Some(_) => {
                                        return Err(Error::Internal(
                                            "ordinary exact-demand GROUP BY annotation mismatch"
                                                .to_string(),
                                        ));
                                    }
                                    None => None,
                                };
                                if alias.is_none()
                                    && let Some(variable) =
                                        Self::unwrapped_variable_expression(expression)
                                {
                                    grouped_expression_sources
                                        .push((expression.clone(), variable.to_string()));
                                    expressions
                                        .push(LogicalExpression::Variable(variable.to_string()));
                                    continue;
                                }
                                let projection_match = Self::projection_alias_for_expression(
                                    &select.projection,
                                    expression,
                                );
                                let source = if let Some(alias) = alias {
                                    alias.clone()
                                } else if let Some((_, alias)) = projection_match {
                                    alias.to_string()
                                } else {
                                    rdf_tagged_term_column(&format!("group-key:{index}"))
                                };
                                plan = self.bind_group_key_expression(
                                    plan,
                                    expression,
                                    &source,
                                    expression_annotation,
                                )?;
                                grouped_expression_sources
                                    .push((expression.clone(), source.clone()));
                                LogicalExpression::Variable(source)
                            }
                            ast::GroupCondition::BuiltInCall(expression) => {
                                let expression_annotation = match annotation {
                                    Some(OrdinaryGroupExactAnnotation::BuiltInCall(annotation)) => {
                                        Some(annotation)
                                    }
                                    Some(_) => {
                                        return Err(Error::Internal(
                                            "ordinary exact-demand GROUP BY annotation mismatch"
                                                .to_string(),
                                        ));
                                    }
                                    None => None,
                                };
                                let projection_match = Self::projection_alias_for_expression(
                                    &select.projection,
                                    expression,
                                );
                                let source = projection_match.map_or_else(
                                    || rdf_tagged_term_column(&format!("group-key:{index}")),
                                    |(_, alias)| alias.to_string(),
                                );
                                plan = self.bind_group_key_expression(
                                    plan,
                                    expression,
                                    &source,
                                    expression_annotation,
                                )?;
                                grouped_expression_sources
                                    .push((expression.clone(), source.clone()));
                                LogicalExpression::Variable(source)
                            }
                        };
                        expressions.push(translated);
                    }
                    expressions
                } else {
                    // No GROUP BY means aggregate over entire dataset (empty group_by)
                    Vec::new()
                };

                // Translate HAVING: rewrite aggregate calls as variable references
                // to the computed aggregate column aliases
                let having_expr = if let Some(having) = &select.solution_modifiers.having {
                    let provenance =
                        Self::rewrite_grouped_expressions(having, &grouped_expression_sources);
                    Self::validate_grouped_user_scope(
                        &provenance,
                        &grouped_expression_sources
                            .iter()
                            .map(|(_, source)| source.clone())
                            .collect(),
                        "HAVING",
                    )?;
                    let rewritten = Self::rewrite_registered_aggregates(having, &aggregate_hoist);
                    let rewritten =
                        Self::rewrite_grouped_expressions(&rewritten, &grouped_expression_sources);
                    Self::validate_grouped_modifier_scope(
                        &rewritten,
                        &grouped_expression_sources,
                        &select.projection,
                        &aggregate_hoist,
                        false,
                        "HAVING",
                    )?;
                    if self.exact_mutation_bindings {
                        // Replace aggregate AST nodes before exact translation.
                        // Translating the entire HAVING ordinarily would also
                        // weaken unrelated RDF equality beside an aggregate.
                        Some(self.translate_expression(&rewritten)?)
                    } else {
                        Some(
                            self.translate_expression_with_annotations(
                                &rewritten,
                                ordinary_annotations
                                    .and_then(|annotations| annotations.having.as_ref()),
                            )?,
                        )
                    }
                } else {
                    None
                };

                // Tagged term aggregates are unpacked immediately after the
                // aggregate operator. Defer HAVING until then so it sees the
                // ordinary visible alias, not the internal [value, exact] pair.
                let defer_having = self.exact_mutation_bindings
                    || !literal_aggregate_aliases.is_empty()
                    || !tagged_aggregate_aliases.is_empty()
                    || !rdf_or_native_aggregate_aliases.is_empty();

                plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: group_by_exprs,
                    aggregates,
                    input: Box::new(plan),
                    having: if defer_having {
                        None
                    } else {
                        having_expr.clone()
                    },
                });
                for alias in literal_aggregate_aliases {
                    plan = Self::attach_literal_aggregate_identity(plan, &alias);
                }
                for (alias, tagged_column) in tagged_aggregate_aliases {
                    plan = Self::attach_tagged_aggregate_identity(plan, &alias, &tagged_column);
                }
                for (alias, value_column) in rdf_or_native_aggregate_aliases {
                    plan =
                        Self::attach_rdf_or_native_aggregate_identity(plan, &alias, &value_column);
                }
                if defer_having && let Some(having) = having_expr {
                    plan = wrap_filter(plan, having);
                }
            }

            // SPARQL extends computed SELECT aliases before ORDER BY, but does
            // not project away the rest of the solution until after ordering.
            // Materialize aliases once here so ORDER BY can see both aliases
            // and non-projected in-scope variables.
            let mut order_alias_dependencies = HashSet::new();
            if let Some(order_by) = &select.solution_modifiers.order_by {
                for condition in order_by {
                    Self::collect_expression_variables(
                        &condition.expression,
                        &mut order_alias_dependencies,
                    );
                }
            }
            let (extended_plan, materialized_projection_aliases) = if is_grouped {
                self.materialize_grouped_select_projection_aliases(
                    plan,
                    &select.projection,
                    ordinary_annotations.map(|annotations| annotations.projection.as_slice()),
                    &grouped_expression_sources,
                    &aggregate_hoist,
                    matches!(select.modifier, ast::SelectModifier::Distinct),
                )?
            } else {
                self.materialize_select_projection_aliases(
                    plan,
                    &select.projection,
                    ordinary_annotations.map(|annotations| annotations.projection.as_slice()),
                    &order_alias_dependencies,
                    matches!(select.modifier, ast::SelectModifier::Distinct),
                )?
            };
            plan = extended_plan;

            // Apply ORDER BY after SELECT-expression extension and before the
            // final projection, matching the SPARQL algebra translation.
            if let Some(order_by) = &select.solution_modifiers.order_by {
                let keys = order_by
                    .iter()
                    .enumerate()
                    .map(|(index, oc)| {
                        let annotation = ordinary_annotations
                            .and_then(|annotations| annotations.order_by.as_ref())
                            .and_then(|annotations| annotations.get(index));
                        let mut rewritten =
                            Self::rewrite_registered_aggregates(&oc.expression, &aggregate_hoist);
                        if is_grouped {
                            rewritten = Self::rewrite_grouped_expressions(
                                &rewritten,
                                &grouped_expression_sources,
                            );
                            Self::validate_grouped_modifier_scope(
                                &rewritten,
                                &grouped_expression_sources,
                                &select.projection,
                                &aggregate_hoist,
                                true,
                                "ORDER BY",
                            )?;
                        }
                        // Use one hidden tagged RDF term as the sort key. This
                        // preserves blank/IRI/literal category and exact numeric
                        // datatype while the public value remains unchanged.
                        // Non-RDF extension values retain native ordering.
                        let expression = self.rdf_term_or_native_expression_with_annotations(
                            &rewritten, annotation,
                        )?;
                        Ok(SortKey {
                            expression,
                            order: match oc.direction {
                                ast::SortDirection::Ascending => SortOrder::Ascending,
                                ast::SortDirection::Descending => SortOrder::Descending,
                            },
                            nulls: None,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                plan = wrap_sort(plan, keys);
            }

            plan = self.translate_select_projection(
                plan,
                &select.projection,
                ordinary_annotations.map(|annotations| annotations.projection.as_slice()),
                &materialized_projection_aliases,
                is_grouped.then_some(grouped_expression_sources.as_slice()),
            )?;

            // Apply DISTINCT (after projection, before OFFSET/LIMIT).
            // REDUCED is a no-op: the spec allows returning all rows unchanged.
            if matches!(select.modifier, ast::SelectModifier::Distinct) {
                plan = wrap_distinct(plan);
            }

            // Apply OFFSET
            if let Some(offset) = select.solution_modifiers.offset {
                // reason: SPARQL u64 offset fits usize on 64-bit targets
                #[allow(clippy::cast_possible_truncation)]
                let skip_n = offset as usize;
                plan = wrap_skip(plan, skip_n);
            }

            // Apply LIMIT
            if let Some(limit) = select.solution_modifiers.limit {
                // reason: SPARQL u64 limit fits usize on 64-bit targets
                #[allow(clippy::cast_possible_truncation)]
                let limit_n = limit as usize;
                plan = wrap_limit(plan, limit_n);
            }

            Ok(LogicalPlan::new(plan))
        })();
        self.dataset = previous_dataset;
        if let Some((previous_exact, previous_control)) = previous_exact_variables {
            self.exact_mutation_variables = previous_exact;
            self.mutation_control_variables = previous_control;
        }
        translated
    }

    fn seal_modify_bindings(input: LogicalOperator) -> LogicalOperator {
        LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::Literal(Value::Bool(true)),
            variable: RDF_SEALED_MODIFY_COLUMN.to_string(),
            input: Box::new(input),
        })
    }

    fn translate_ask(&mut self, ask: &ast::AskQuery) -> Result<LogicalPlan> {
        // Apply dataset restriction from FROM / FROM NAMED clauses
        self.dataset = self.translate_dataset_clause(&ask.dataset);

        // ASK returns true if the pattern has any matches
        let plan = self.translate_graph_pattern(&ask.where_clause)?;

        // Clear dataset after translating the WHERE clause
        self.dataset = None;

        // Limit to 1 result for efficiency
        let plan = wrap_limit(plan, 1);

        Ok(LogicalPlan::new(plan))
    }

    fn translate_construct(&mut self, construct: &ast::ConstructQuery) -> Result<LogicalPlan> {
        // Apply dataset restriction from FROM / FROM NAMED clauses
        self.dataset = self.translate_dataset_clause(&construct.dataset);

        // Evaluate the WHERE pattern to produce variable bindings
        let mut plan = self.translate_graph_pattern(&construct.where_clause)?;

        // Clear dataset after translating the WHERE clause
        self.dataset = None;

        // Apply solution modifiers to the WHERE output
        if let Some(limit) = construct.solution_modifiers.limit {
            // reason: SPARQL u64 limit fits usize on 64-bit targets
            #[allow(clippy::cast_possible_truncation)]
            let limit_n = limit as usize;
            plan = wrap_limit(plan, limit_n);
        }

        // Translate template triples: substitute variable bindings from WHERE.
        // Use translate_data_term for subject/object so that blank nodes become
        // TripleComponent::BlankNode (constants in output) rather than variables
        // that would fail to bind against the WHERE clause results.
        let mut templates = Vec::new();
        for tp in &construct.template {
            let subject = self.translate_data_term(&tp.subject)?;
            // CONSTRUCT templates use simple predicates (IRIs/variables), not paths
            let predicate = match &tp.predicate {
                ast::PropertyPath::Predicate(iri) => TripleComponent::Iri(self.resolve_iri(iri)),
                ast::PropertyPath::Variable(name) => TripleComponent::Variable(name.clone()),
                ast::PropertyPath::RdfType => TripleComponent::Iri(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                _ => continue, // Skip complex property paths in templates
            };
            let object = self.translate_data_term(&tp.object)?;
            templates.push(TripleTemplate {
                subject,
                predicate,
                object,
                graph: None,
            });
        }

        Ok(LogicalPlan::new(LogicalOperator::Construct(ConstructOp {
            templates,
            input: Box::new(plan),
        })))
    }

    fn translate_describe(&mut self, describe: &ast::DescribeQuery) -> Result<LogicalPlan> {
        // Apply dataset restriction from FROM / FROM NAMED clauses
        self.dataset = self.translate_dataset_clause(&describe.dataset);

        // DESCRIBE implements Concise Bounded Description (CBD):
        // Return all triples where the described resource is the subject.
        let pred_var = format!("__describe_p{}", self.next_anon());
        let obj_var = format!("__describe_o{}", self.next_anon());

        let mut cbd_scans: Vec<LogicalOperator> = Vec::new();
        for resource in &describe.resources {
            let subject = match resource {
                ast::VariableOrIri::Iri(iri) => TripleComponent::Iri(self.resolve_iri(iri)),
                ast::VariableOrIri::Variable(name) => TripleComponent::Variable(name.clone()),
            };
            cbd_scans.push(self.make_triple_scan(
                subject,
                TripleComponent::Variable(pred_var.clone()),
                TripleComponent::Variable(obj_var.clone()),
                None,
            ));
        }

        let cbd_plan = if cbd_scans.len() == 1 {
            cbd_scans
                .pop()
                .ok_or_else(|| Error::Internal("DESCRIBE resource scan is missing".to_string()))?
        } else {
            LogicalOperator::Union(UnionOp { inputs: cbd_scans })
        };

        let plan = if let Some(where_clause) = &describe.where_clause {
            let where_plan = self.translate_graph_pattern(where_clause)?;
            self.join_patterns(where_plan, cbd_plan)
        } else {
            cbd_plan
        };

        // Clear dataset after translating the WHERE clause
        self.dataset = None;

        Ok(LogicalPlan::new(plan))
    }

    // ==================== SPARQL Update Translation ====================

    fn translate_update(&mut self, update: &ast::UpdateOperation) -> Result<LogicalPlan> {
        match update {
            ast::UpdateOperation::InsertData { data } => self.translate_insert_data(data),
            ast::UpdateOperation::DeleteData { data } => self.translate_delete_data(data),
            ast::UpdateOperation::DeleteWhere { pattern } => self.translate_delete_where(pattern),
            ast::UpdateOperation::Modify {
                with_graph,
                delete_template,
                insert_template,
                using_clauses,
                where_clause,
            } => self.translate_modify(
                with_graph,
                delete_template,
                insert_template,
                using_clauses,
                where_clause,
            ),
            ast::UpdateOperation::Load {
                silent,
                source,
                destination,
            } => self.translate_load(*silent, source, destination.as_ref()),
            ast::UpdateOperation::Clear { silent, target } => self.translate_clear(*silent, target),
            ast::UpdateOperation::Drop { silent, target } => self.translate_drop(*silent, target),
            ast::UpdateOperation::Create { silent, graph } => self.translate_create(*silent, graph),
            ast::UpdateOperation::Copy {
                silent,
                source,
                destination,
            } => self.translate_copy(*silent, source, destination),
            ast::UpdateOperation::Move {
                silent,
                source,
                destination,
            } => self.translate_move(*silent, source, destination),
            ast::UpdateOperation::Add {
                silent,
                source,
                destination,
            } => self.translate_add(*silent, source, destination),
        }
    }

    fn translate_insert_data(&mut self, data: &[ast::QuadPattern]) -> Result<LogicalPlan> {
        // Build a sequence of InsertTriple operators
        let mut ops = Vec::new();
        for quad in data {
            let subject = self.translate_data_term(&quad.triple.subject)?;
            let predicate = self.translate_property_path(&quad.triple.predicate)?;
            let object = self.translate_data_term(&quad.triple.object)?;
            let graph = quad.graph.as_ref().map(|g| self.resolve_variable_or_iri(g));

            ops.push(LogicalOperator::InsertTriple(InsertTripleOp {
                subject,
                predicate,
                object,
                graph,
                input: None,
            }));
        }

        // Combine all inserts into a sequence using Union
        if ops.is_empty() {
            Ok(LogicalPlan::new(LogicalOperator::Empty))
        } else if ops.len() == 1 {
            Ok(LogicalPlan::new(ops.pop().ok_or_else(|| {
                Error::Internal("INSERT DATA operation is missing".to_string())
            })?))
        } else {
            Ok(LogicalPlan::new(LogicalOperator::Union(UnionOp {
                inputs: ops,
            })))
        }
    }

    /// Translates a triple term in a data context (INSERT DATA / DELETE DATA).
    ///
    /// Blank nodes become `TripleComponent::BlankNode` instead of variables,
    /// because data operations have no WHERE clause to bind variables against.
    fn translate_data_term(&mut self, term: &ast::TripleTerm) -> Result<TripleComponent> {
        match term {
            ast::TripleTerm::BlankNode(bnode) => {
                let label = match bnode {
                    ast::BlankNode::Labeled(label) => {
                        format!("q{}_{label}", self.query_id)
                    }
                    ast::BlankNode::Anonymous(_) => {
                        let anon = self.next_anon();
                        format!("q{}_anon{anon}", self.query_id)
                    }
                };
                Ok(TripleComponent::BlankNode(label))
            }
            // Non-blank-node terms use the standard translation
            other => self.translate_triple_term(other),
        }
    }

    fn translate_delete_data(&mut self, data: &[ast::QuadPattern]) -> Result<LogicalPlan> {
        // Build a sequence of DeleteTriple operators
        let mut ops = Vec::new();
        for quad in data {
            let subject = self.translate_triple_term(&quad.triple.subject)?;
            let predicate = self.translate_property_path(&quad.triple.predicate)?;
            let object = self.translate_triple_term(&quad.triple.object)?;
            let graph = quad.graph.as_ref().map(|g| self.resolve_variable_or_iri(g));

            ops.push(LogicalOperator::DeleteTriple(DeleteTripleOp {
                subject,
                predicate,
                object,
                graph,
                input: None,
            }));
        }

        if ops.is_empty() {
            Ok(LogicalPlan::new(LogicalOperator::Empty))
        } else if ops.len() == 1 {
            Ok(LogicalPlan::new(ops.pop().ok_or_else(|| {
                Error::Internal("DELETE DATA operation is missing".to_string())
            })?))
        } else {
            Ok(LogicalPlan::new(LogicalOperator::Union(UnionOp {
                inputs: ops,
            })))
        }
    }

    fn translate_delete_where(&mut self, pattern: &ast::GraphPattern) -> Result<LogicalPlan> {
        // DELETE WHERE is a DELETE template whose bindings come from the same
        // quad pattern. Preserve GRAPH constants and variables: flattening to
        // bare triples silently redirected named-graph deletes to default.
        let quads = Self::extract_delete_where_quads(pattern, None)?;
        let exact_variables = Self::mutation_template_variables(&quads);
        let match_plan = self.translate_mutation_graph_pattern(pattern, exact_variables)?;
        let mut delete_templates = Vec::with_capacity(quads.len());
        for quad in quads {
            delete_templates.push(TripleTemplate {
                subject: self.translate_triple_term(&quad.triple.subject)?,
                predicate: self.translate_property_path(&quad.triple.predicate)?,
                object: self.translate_triple_term(&quad.triple.object)?,
                graph: quad
                    .graph
                    .as_ref()
                    .map(|graph| self.resolve_mutation_graph_template(graph)),
            });
        }

        // One Modify operator materializes the WHERE solutions once before any
        // deletion. This is required when a DELETE WHERE contains multiple
        // quad templates: later templates must not re-read the first one's
        // post-image.
        Ok(LogicalPlan::new(LogicalOperator::Modify(ModifyOp {
            delete_templates,
            insert_templates: Vec::new(),
            where_clause: Box::new(match_plan),
            graph: None,
        })))
    }

    fn extract_delete_where_quads(
        pattern: &ast::GraphPattern,
        graph: Option<&ast::VariableOrIri>,
    ) -> Result<Vec<ast::QuadPattern>> {
        match pattern {
            ast::GraphPattern::Basic(triples) => Ok(triples
                .iter()
                .cloned()
                .map(|triple| ast::QuadPattern {
                    graph: graph.cloned(),
                    triple,
                })
                .collect()),
            ast::GraphPattern::Group(patterns) => {
                let mut quads = Vec::new();
                for pattern in patterns {
                    quads.extend(Self::extract_delete_where_quads(pattern, graph)?);
                }
                Ok(quads)
            }
            ast::GraphPattern::NamedGraph {
                graph: named_graph,
                pattern,
            } => Self::extract_delete_where_quads(pattern, Some(named_graph)),
            _ => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "DELETE WHERE accepts only basic and GRAPH quad patterns",
            ))),
        }
    }

    fn translate_modify(
        &mut self,
        with_graph: &Option<ast::Iri>,
        delete_template: &Option<Vec<ast::QuadPattern>>,
        insert_template: &Option<Vec<ast::QuadPattern>>,
        using_clauses: &[ast::UsingClause],
        where_clause: &ast::GraphPattern,
    ) -> Result<LogicalPlan> {
        let mut exact_variables = HashSet::new();
        if let Some(quads) = delete_template {
            exact_variables.extend(Self::mutation_template_variables(quads));
        }
        if let Some(quads) = insert_template {
            exact_variables.extend(Self::mutation_template_variables(quads));
        }
        let default_graph = with_graph.as_ref().map(|g| self.resolve_iri(g));
        let where_dataset = if using_clauses.is_empty() {
            default_graph.as_ref().map(|graph| DatasetRestriction {
                default_graphs: vec![graph.clone()],
                named_graphs: vec![RDF_EXPLICIT_EMPTY_NAMED_DATASET.to_string()],
            })
        } else {
            let mut default_graphs = Vec::new();
            let mut named_graphs = Vec::new();
            for clause in using_clauses {
                match clause {
                    ast::UsingClause::Default(iri) => default_graphs.push(self.resolve_iri(iri)),
                    ast::UsingClause::Named(iri) => named_graphs.push(self.resolve_iri(iri)),
                }
            }
            if named_graphs.is_empty() {
                named_graphs.push(RDF_EXPLICIT_EMPTY_NAMED_DATASET.to_string());
            }
            if default_graphs.is_empty() {
                default_graphs.push(RDF_EXPLICIT_EMPTY_DEFAULT_DATASET.to_string());
            }
            Some(DatasetRestriction {
                default_graphs,
                named_graphs,
            })
        };
        let previous_dataset = std::mem::replace(&mut self.dataset, where_dataset);
        // The WHERE clause is evaluated once against the USING dataset, or
        // against WITH as its default graph when no USING clause is present.
        let where_plan = self.translate_mutation_graph_pattern(where_clause, exact_variables);
        self.dataset = previous_dataset;
        let where_plan = where_plan?;

        let default_graph_template = default_graph.clone();

        // Build DELETE templates
        let mut delete_templates = Vec::new();
        if let Some(delete_quads) = delete_template {
            for quad in delete_quads {
                let subject = self.translate_triple_term(&quad.triple.subject)?;
                let predicate = self.translate_property_path(&quad.triple.predicate)?;
                let object = self.translate_triple_term(&quad.triple.object)?;
                let graph = quad
                    .graph
                    .as_ref()
                    .map(|graph| self.resolve_mutation_graph_template(graph))
                    .or_else(|| default_graph_template.clone());

                delete_templates.push(TripleTemplate {
                    subject,
                    predicate,
                    object,
                    graph,
                });
            }
        }

        // Build INSERT templates
        let mut insert_templates = Vec::new();
        if let Some(insert_quads) = insert_template {
            for quad in insert_quads {
                let subject = self.translate_insert_template_term(&quad.triple.subject)?;
                let predicate = self.translate_property_path(&quad.triple.predicate)?;
                let object = self.translate_insert_template_term(&quad.triple.object)?;
                let graph = quad
                    .graph
                    .as_ref()
                    .map(|graph| self.resolve_mutation_graph_template(graph))
                    .or_else(|| default_graph_template.clone());

                insert_templates.push(TripleTemplate {
                    subject,
                    predicate,
                    object,
                    graph,
                });
            }
        }

        // Use ModifyOp which handles SPARQL MODIFY semantics correctly:
        // 1. Evaluate WHERE once
        // 2. Apply DELETE templates
        // 3. Apply INSERT templates (using same bindings)
        Ok(LogicalPlan::new(LogicalOperator::Modify(ModifyOp {
            delete_templates,
            insert_templates,
            where_clause: Box::new(where_plan),
            graph: default_graph,
        })))
    }

    /// Blank nodes in an INSERT template are constructors, not WHERE
    /// variables. The query-scoped label is further scoped per solution by
    /// the MODIFY executor, preserving sharing within one solution while
    /// making different solutions fresh.
    fn translate_insert_template_term(
        &mut self,
        term: &ast::TripleTerm,
    ) -> Result<TripleComponent> {
        match term {
            ast::TripleTerm::BlankNode(ast::BlankNode::Labeled(label)) => Ok(
                TripleComponent::BlankNode(format!("q{}_{label}", self.query_id)),
            ),
            ast::TripleTerm::BlankNode(ast::BlankNode::Anonymous(_)) => {
                let anon = self.next_anon();
                Ok(TripleComponent::BlankNode(format!(
                    "q{}_anon{anon}",
                    self.query_id
                )))
            }
            _ => self.translate_triple_term(term),
        }
    }

    fn translate_load(
        &mut self,
        silent: bool,
        source: &ast::Iri,
        destination: Option<&ast::Iri>,
    ) -> Result<LogicalPlan> {
        Ok(LogicalPlan::new(LogicalOperator::LoadGraph(LoadGraphOp {
            source: self.resolve_iri(source),
            destination: destination.map(|d| self.resolve_iri(d)),
            silent,
        })))
    }

    fn translate_clear(&mut self, silent: bool, target: &ast::GraphTarget) -> Result<LogicalPlan> {
        let graph = self.translate_graph_target(target);
        Ok(LogicalPlan::new(LogicalOperator::ClearGraph(
            ClearGraphOp { graph, silent },
        )))
    }

    fn translate_drop(&mut self, silent: bool, target: &ast::GraphTarget) -> Result<LogicalPlan> {
        let graph = self.translate_graph_target(target);
        Ok(LogicalPlan::new(LogicalOperator::DropGraph(DropGraphOp {
            graph,
            silent,
        })))
    }

    fn translate_create(&mut self, silent: bool, graph: &ast::Iri) -> Result<LogicalPlan> {
        Ok(LogicalPlan::new(LogicalOperator::CreateGraph(
            CreateGraphOp {
                graph: self.resolve_iri(graph),
                silent,
            },
        )))
    }

    fn translate_copy(
        &mut self,
        silent: bool,
        source: &ast::GraphTarget,
        destination: &ast::GraphTarget,
    ) -> Result<LogicalPlan> {
        Ok(LogicalPlan::new(LogicalOperator::CopyGraph(CopyGraphOp {
            source: self.translate_graph_target(source),
            destination: self.translate_graph_target(destination),
            silent,
        })))
    }

    fn translate_move(
        &mut self,
        silent: bool,
        source: &ast::GraphTarget,
        destination: &ast::GraphTarget,
    ) -> Result<LogicalPlan> {
        Ok(LogicalPlan::new(LogicalOperator::MoveGraph(MoveGraphOp {
            source: self.translate_graph_target(source),
            destination: self.translate_graph_target(destination),
            silent,
        })))
    }

    fn translate_add(
        &mut self,
        silent: bool,
        source: &ast::GraphTarget,
        destination: &ast::GraphTarget,
    ) -> Result<LogicalPlan> {
        Ok(LogicalPlan::new(LogicalOperator::AddGraph(AddGraphOp {
            source: self.translate_graph_target(source),
            destination: self.translate_graph_target(destination),
            silent,
        })))
    }

    fn translate_graph_target(&self, target: &ast::GraphTarget) -> Option<String> {
        match target {
            ast::GraphTarget::Default => None,
            ast::GraphTarget::Named(iri) => Some(self.resolve_iri(iri)),
            ast::GraphTarget::NamedAll => Some("\u{1}NAMED".to_string()),
            ast::GraphTarget::All => Some(String::new()), // Empty string represents "all"
        }
    }

    fn resolve_variable_or_iri(&self, var_or_iri: &ast::VariableOrIri) -> String {
        match var_or_iri {
            ast::VariableOrIri::Variable(name) => format!("?{}", name),
            ast::VariableOrIri::Iri(iri) => self.resolve_iri(iri),
        }
    }

    fn resolve_mutation_graph_template(&self, graph: &ast::VariableOrIri) -> String {
        match graph {
            ast::VariableOrIri::Variable(name) => rdf_graph_variable_template(name),
            ast::VariableOrIri::Iri(iri) => self.resolve_iri(iri),
        }
    }

    fn translate_projection(
        &mut self,
        projection: &ast::Projection,
        ordinary_annotations: Option<&[OrdinaryProjectionExactAnnotation]>,
        materialized_aliases: &HashSet<String>,
    ) -> Result<Vec<Projection>> {
        match projection {
            ast::Projection::Wildcard => Ok(Vec::new()), // Empty means select all
            ast::Projection::Variables(vars) => vars
                .iter()
                .enumerate()
                .map(|(index, pv)| {
                    let projection_annotation =
                        ordinary_annotations.and_then(|annotations| annotations.get(index));
                    let annotation = projection_annotation.map(|annotation| &annotation.expression);
                    let materialized_alias = pv
                        .alias
                        .as_ref()
                        .filter(|alias| materialized_aliases.contains(*alias));
                    Ok(Projection {
                        // Computed projections are extended before ORDER BY.
                        // Project the stored alias so volatile expressions are
                        // evaluated exactly once and non-selected sort inputs
                        // remain in scope until ordering is complete.
                        expression: if let Some(alias) = materialized_alias {
                            LogicalExpression::Variable(alias.clone())
                        } else if self.exact_mutation_bindings {
                            self.translate_ordinary_expression_with_annotations(
                                &pv.expression,
                                annotation,
                            )?
                        } else {
                            self.translate_expression_with_annotations(&pv.expression, annotation)?
                        },
                        alias: pv.alias.clone(),
                    })
                })
                .collect(),
        }
    }

    fn materialize_select_projection_aliases(
        &mut self,
        mut input: LogicalOperator,
        projection: &ast::Projection,
        ordinary_exact: Option<&[OrdinaryProjectionExactAnnotation]>,
        required_before_projection: &HashSet<String>,
        distinct: bool,
    ) -> Result<(LogicalOperator, HashSet<String>)> {
        let expected_projection_count = match projection {
            ast::Projection::Wildcard => 0,
            ast::Projection::Variables(variables) => variables.len(),
        };
        if let Some(ordinary_exact) = ordinary_exact
            && ordinary_exact.len() != expected_projection_count
        {
            return Err(Error::Internal(format!(
                "ordinary exact-demand projection annotation mismatch: expected {expected_projection_count}, got {}",
                ordinary_exact.len()
            )));
        }

        let ast::Projection::Variables(variables) = projection else {
            return Ok((input, HashSet::new()));
        };
        let mut materialized = HashSet::new();
        for (index, projected) in variables.iter().enumerate() {
            let Some(alias) = &projected.alias else {
                continue;
            };
            let projection_annotation =
                ordinary_exact.and_then(|annotations| annotations.get(index));
            let requires_preprojection_exact =
                !matches!(projected.expression, ast::Expression::Variable(_))
                    && (self.exact_mutation_bindings
                        || projection_annotation.is_some_and(|annotation| annotation.exact));
            let requires_distinct_identity =
                distinct && !matches!(projected.expression, ast::Expression::Variable(_));
            let requires_rdf_or_native_identity =
                projection_annotation.is_some_and(|annotation| annotation.rdf_or_native);
            if !requires_preprojection_exact
                && !requires_distinct_identity
                && !requires_rdf_or_native_identity
                && !required_before_projection.contains(alias)
            {
                continue;
            }
            input = if requires_distinct_identity || requires_rdf_or_native_identity {
                self.bind_group_key_expression(
                    input,
                    &projected.expression,
                    alias,
                    projection_annotation.map(|annotation| &annotation.expression),
                )?
            } else {
                self.bind_select_projection_alias(
                    input,
                    &projected.expression,
                    alias,
                    projection_annotation,
                )?
            };
            materialized.insert(alias.clone());
        }
        Ok((input, materialized))
    }

    fn materialize_grouped_select_projection_aliases(
        &mut self,
        mut input: LogicalOperator,
        projection: &ast::Projection,
        ordinary_exact: Option<&[OrdinaryProjectionExactAnnotation]>,
        grouped_expression_sources: &[(ast::Expression, String)],
        aggregate_hoist: &AggregateHoist<'_>,
        distinct: bool,
    ) -> Result<(LogicalOperator, HashSet<String>)> {
        let expected_projection_count = match projection {
            ast::Projection::Wildcard => 0,
            ast::Projection::Variables(variables) => variables.len(),
        };
        if let Some(ordinary_exact) = ordinary_exact
            && ordinary_exact.len() != expected_projection_count
        {
            return Err(Error::Internal(format!(
                "ordinary exact-demand projection annotation mismatch: expected {expected_projection_count}, got {}",
                ordinary_exact.len()
            )));
        }

        let ast::Projection::Variables(projected_variables) = projection else {
            return Ok((input, HashSet::new()));
        };
        let mut materialized = HashSet::new();
        let mut available_user_names = grouped_expression_sources
            .iter()
            .map(|(_, source)| source.clone())
            .collect::<HashSet<_>>();
        let mut available_post_group_names = grouped_expression_sources
            .iter()
            .map(|(_, source)| source.clone())
            .collect::<HashSet<_>>();
        available_post_group_names.extend(
            aggregate_hoist
                .entries
                .iter()
                .map(|entry| entry.canonical_column.clone()),
        );

        for (index, projected) in projected_variables.iter().enumerate() {
            let provenance = Self::rewrite_grouped_expressions(
                &projected.expression,
                grouped_expression_sources,
            );
            Self::validate_grouped_user_scope(
                &provenance,
                &available_user_names,
                "SELECT expression",
            )?;
            let rewritten =
                Self::rewrite_registered_aggregates(&projected.expression, aggregate_hoist);
            let rewritten =
                Self::rewrite_grouped_expressions(&rewritten, grouped_expression_sources);
            let mut dependencies = HashSet::new();
            Self::collect_expression_variables(&rewritten, &mut dependencies);
            let mut missing = dependencies
                .iter()
                .filter(|variable| !available_post_group_names.contains(*variable))
                .cloned()
                .collect::<Vec<_>>();
            missing.sort();

            let Some(alias) = &projected.alias else {
                if let Some(variable) = missing.first() {
                    return Err(Self::ungrouped_projection_error(variable));
                }
                continue;
            };
            let projection_annotation =
                ordinary_exact.and_then(|annotations| annotations.get(index));

            if let Some(variable) = missing.first() {
                return Err(Self::ungrouped_projection_error(variable));
            }
            if matches!(&rewritten, ast::Expression::Variable(source) if source == alias) {
                materialized.insert(alias.clone());
                available_user_names.insert(alias.clone());
                available_post_group_names.insert(alias.clone());
                continue;
            }
            if let ast::Expression::Aggregate(aggregate) = &projected.expression {
                let entry = aggregate_hoist
                    .entries
                    .iter()
                    .find(|entry| entry.aggregate == aggregate)
                    .ok_or_else(|| {
                        Error::Internal(
                            "direct SELECT aggregate is missing from the hoist registry"
                                .to_string(),
                        )
                    })?;
                let mutation_demand = self.aggregate_hoist_mutation_demand(entry);
                let result_demand = entry.result_demand();
                input = Self::bind_aggregate_alias(
                    input,
                    &entry.canonical_column,
                    alias,
                    result_demand.exact
                        || result_demand.full_rdf_or_native
                        || mutation_demand.exact
                        || mutation_demand.control,
                );
            } else if distinct
                || projection_annotation.is_some_and(|annotation| annotation.rdf_or_native)
            {
                input = self.bind_group_key_expression(
                    input,
                    &rewritten,
                    alias,
                    projection_annotation.map(|annotation| &annotation.expression),
                )?;
            } else {
                input = self.bind_select_projection_alias(
                    input,
                    &rewritten,
                    alias,
                    projection_annotation,
                )?;
            }
            materialized.insert(alias.clone());
            available_user_names.insert(alias.clone());
            available_post_group_names.insert(alias.clone());
        }

        Ok((input, materialized))
    }

    fn bind_aggregate_alias(
        input: LogicalOperator,
        source: &str,
        alias: &str,
        copy_identity: bool,
    ) -> LogicalOperator {
        let visible = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::Variable(source.to_string()),
            variable: alias.to_string(),
            input: Box::new(input),
        });
        if !copy_identity {
            return visible;
        }
        let exact = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::Variable(rdf_exact_term_column(source)),
            variable: rdf_exact_term_column(alias),
            input: Box::new(visible),
        });
        let identity = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::Variable(rdf_identity_key_column(source)),
            variable: rdf_identity_key_column(alias),
            input: Box::new(exact),
        });
        LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::Variable(rdf_group_key_column(source)),
            variable: rdf_group_key_column(alias),
            input: Box::new(identity),
        })
    }

    fn bind_select_projection_alias(
        &mut self,
        input: LogicalOperator,
        expression: &ast::Expression,
        alias: &str,
        projection_annotation: Option<&OrdinaryProjectionExactAnnotation>,
    ) -> Result<LogicalOperator> {
        if self.exact_mutation_bindings {
            self.translate_pattern_bind(input, expression, alias)
        } else {
            self.translate_ordinary_pattern_bind_with_annotations(
                input,
                expression,
                alias,
                projection_annotation.map(|annotation| &annotation.expression),
                projection_annotation.is_some_and(|annotation| annotation.exact),
                projection_annotation.is_some_and(|annotation| annotation.rdf_or_native),
            )
        }
    }

    /// Evaluates one GROUP BY expression exactly once and derives three
    /// representations from that stored result: its public value, optional
    /// lossless RDF term, and an RDF/native-discriminated grouping identity.
    fn bind_group_key_expression(
        &mut self,
        input: LogicalOperator,
        expression: &ast::Expression,
        source: &str,
        annotation: Option<&OrdinaryExpressionExactAnnotations>,
    ) -> Result<LogicalOperator> {
        let evaluated = rdf_tagged_term_column(&format!("group-evaluated:{}", self.next_anon()));
        let evaluated_input = LogicalOperator::Bind(BindOp {
            expression: self
                .rdf_term_or_native_expression_with_annotations(expression, annotation)?,
            variable: evaluated.clone(),
            input: Box::new(input),
        });
        let visible = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_TERM_OR_NATIVE_VISIBLE.to_string(),
                args: vec![LogicalExpression::Variable(evaluated.clone())],
                distinct: false,
            },
            variable: source.to_string(),
            input: Box::new(evaluated_input),
        });
        let exact = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_TERM_OR_NATIVE_EXACT.to_string(),
                args: vec![LogicalExpression::Variable(evaluated.clone())],
                distinct: false,
            },
            variable: rdf_exact_term_column(source),
            input: Box::new(visible),
        });
        let identity = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_TERM_IDENTITY_KEY.to_string(),
                args: vec![LogicalExpression::Variable(evaluated.clone())],
                distinct: false,
            },
            variable: rdf_identity_key_column(source),
            input: Box::new(exact),
        });
        Ok(LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_DISTINCT_TERM_OR_VALUE_KEY.to_string(),
                args: vec![LogicalExpression::Variable(evaluated)],
                distinct: false,
            },
            variable: rdf_group_key_column(source),
            input: Box::new(identity),
        }))
    }

    fn unwrapped_variable_expression(expression: &ast::Expression) -> Option<&str> {
        match expression {
            ast::Expression::Variable(variable) => Some(variable),
            ast::Expression::Bracketed(inner) => Self::unwrapped_variable_expression(inner),
            _ => None,
        }
    }

    /// SPARQL parentheses are transparent for expression identity. Preserve
    /// the original AST for diagnostics and evaluation, but ignore redundant
    /// `Bracketed` nodes while matching SELECT/HAVING/ORDER expressions to a
    /// value already evaluated by GROUP BY.
    fn group_expressions_equivalent(
        mut left: &ast::Expression,
        mut right: &ast::Expression,
    ) -> bool {
        use ast::Expression;

        while let Expression::Bracketed(inner) = left {
            left = inner;
        }
        while let Expression::Bracketed(inner) = right {
            right = inner;
        }

        match (left, right) {
            (
                Expression::Binary {
                    left: left_left,
                    operator: left_operator,
                    right: left_right,
                },
                Expression::Binary {
                    left: right_left,
                    operator: right_operator,
                    right: right_right,
                },
            ) => {
                left_operator == right_operator
                    && Self::group_expressions_equivalent(left_left, right_left)
                    && Self::group_expressions_equivalent(left_right, right_right)
            }
            (
                Expression::Unary {
                    operator: left_operator,
                    operand: left_operand,
                },
                Expression::Unary {
                    operator: right_operator,
                    operand: right_operand,
                },
            ) => {
                left_operator == right_operator
                    && Self::group_expressions_equivalent(left_operand, right_operand)
            }
            (
                Expression::FunctionCall {
                    function: left_function,
                    arguments: left_arguments,
                },
                Expression::FunctionCall {
                    function: right_function,
                    arguments: right_arguments,
                },
            ) => {
                left_function == right_function
                    && left_arguments.len() == right_arguments.len()
                    && left_arguments
                        .iter()
                        .zip(right_arguments)
                        .all(|(left, right)| Self::group_expressions_equivalent(left, right))
            }
            (
                Expression::Conditional {
                    condition: left_condition,
                    then_expression: left_then,
                    else_expression: left_else,
                },
                Expression::Conditional {
                    condition: right_condition,
                    then_expression: right_then,
                    else_expression: right_else,
                },
            ) => {
                Self::group_expressions_equivalent(left_condition, right_condition)
                    && Self::group_expressions_equivalent(left_then, right_then)
                    && Self::group_expressions_equivalent(left_else, right_else)
            }
            (Expression::Coalesce(left), Expression::Coalesce(right)) => {
                left.len() == right.len()
                    && left
                        .iter()
                        .zip(right)
                        .all(|(left, right)| Self::group_expressions_equivalent(left, right))
            }
            (
                Expression::In {
                    expression: left_expression,
                    list: left_list,
                },
                Expression::In {
                    expression: right_expression,
                    list: right_list,
                },
            )
            | (
                Expression::NotIn {
                    expression: left_expression,
                    list: left_list,
                },
                Expression::NotIn {
                    expression: right_expression,
                    list: right_list,
                },
            ) => {
                Self::group_expressions_equivalent(left_expression, right_expression)
                    && left_list.len() == right_list.len()
                    && left_list
                        .iter()
                        .zip(right_list)
                        .all(|(left, right)| Self::group_expressions_equivalent(left, right))
            }
            _ => left == right,
        }
    }

    fn projection_alias_for_expression<'a>(
        projection: &'a ast::Projection,
        expression: &ast::Expression,
    ) -> Option<(usize, &'a str)> {
        let ast::Projection::Variables(projected) = projection else {
            return None;
        };
        projected.iter().enumerate().find_map(|(index, projected)| {
            (!Self::contains_aggregate(&projected.expression)
                && Self::group_expressions_equivalent(&projected.expression, expression))
            .then(|| projected.alias.as_deref().map(|alias| (index, alias)))
            .flatten()
        })
    }

    fn ungrouped_projection_error(variable: &str) -> Error {
        Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            format!("SELECT expression references ?{variable}, which is not grouped or aggregated"),
        ))
    }

    fn validate_grouped_modifier_scope(
        expression: &ast::Expression,
        grouped_expression_sources: &[(ast::Expression, String)],
        projection: &ast::Projection,
        aggregate_hoist: &AggregateHoist<'_>,
        include_select_aliases: bool,
        clause: &str,
    ) -> Result<()> {
        let mut allowed = grouped_expression_sources
            .iter()
            .map(|(_, source)| source.clone())
            .collect::<HashSet<_>>();
        if let ast::Projection::Variables(projected) = projection {
            for projected in projected {
                let Some(alias) = &projected.alias else {
                    continue;
                };
                if include_select_aliases {
                    allowed.insert(alias.clone());
                }
            }
        }
        allowed.extend(
            aggregate_hoist
                .entries
                .iter()
                .map(|entry| entry.canonical_column.clone()),
        );

        let mut referenced = HashSet::new();
        Self::collect_expression_variables(expression, &mut referenced);
        let mut missing = referenced.difference(&allowed).cloned().collect::<Vec<_>>();
        missing.sort();
        if let Some(variable) = missing.first() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!("{clause} references ?{variable}, which is not grouped or aggregated"),
            )));
        }
        Ok(())
    }

    fn validate_grouped_user_scope(
        expression: &ast::Expression,
        allowed: &HashSet<String>,
        clause: &str,
    ) -> Result<()> {
        let mut referenced = HashSet::new();
        Self::collect_non_aggregate_expression_variables(expression, &mut referenced);
        let mut missing = referenced.difference(allowed).cloned().collect::<Vec<_>>();
        missing.sort();
        if let Some(variable) = missing.first() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!("{clause} references ?{variable}, which is not grouped or aggregated"),
            )));
        }
        Ok(())
    }

    fn translate_select_projection(
        &mut self,
        input: LogicalOperator,
        projection: &ast::Projection,
        ordinary_exact: Option<&[OrdinaryProjectionExactAnnotation]>,
        materialized_aliases: &HashSet<String>,
        grouped_expression_sources: Option<&[(ast::Expression, String)]>,
    ) -> Result<LogicalOperator> {
        let mut projections =
            self.translate_projection(projection, ordinary_exact, materialized_aliases)?;
        let expected_projection_count = match projection {
            ast::Projection::Wildcard => 0,
            ast::Projection::Variables(variables) => variables.len(),
        };
        if let Some(ordinary_exact) = ordinary_exact
            && ordinary_exact.len() != expected_projection_count
        {
            return Err(Error::Internal(format!(
                "ordinary exact-demand projection annotation mismatch: expected {expected_projection_count}, got {}",
                ordinary_exact.len()
            )));
        }
        if projections.is_empty() {
            let Some(grouped_expression_sources) = grouped_expression_sources else {
                return Ok(input);
            };
            // A grouped wildcard projects named group outputs only. Build a
            // real lexical boundary even when there are no such names: hidden
            // anonymous group keys must not affect an outer DISTINCT or join.
            let mut seen = HashSet::new();
            projections.extend(
                grouped_expression_sources
                    .iter()
                    .map(|(_, source)| source)
                    .filter(|source| {
                        !is_rdf_internal_term_column(source) && seen.insert((*source).clone())
                    })
                    .map(|source| Projection {
                        expression: LogicalExpression::Variable(source.clone()),
                        alias: None,
                    }),
            );
        }

        // Computed aliases were materialized before ORDER BY. Project those
        // stored values and any demanded companions now that sort-only input
        // variables may safely leave scope. Plain variables are renamed with
        // their available exact companions by the RDF planner.
        if let ast::Projection::Variables(variables) = projection {
            let mut companion_projections = Vec::new();
            for (index, (logical, projected)) in projections.iter_mut().zip(variables).enumerate() {
                let projection_demand = ordinary_exact.and_then(|demands| demands.get(index));
                let exact_is_demanded =
                    projection_demand.is_some_and(|annotation| annotation.exact);
                let rdf_or_native_is_demanded =
                    projection_demand.is_some_and(|annotation| annotation.rdf_or_native);
                if let ast::Expression::Variable(source) = &projected.expression {
                    let output = projected.alias.as_deref().unwrap_or(source);
                    let companion_source = projected
                        .alias
                        .as_deref()
                        .filter(|alias| materialized_aliases.contains(*alias))
                        .unwrap_or(source);
                    if exact_is_demanded {
                        companion_projections.push(Projection {
                            expression: LogicalExpression::Variable(rdf_exact_term_column(
                                companion_source,
                            )),
                            alias: Some(rdf_exact_term_column(output)),
                        });
                        companion_projections.push(Projection {
                            expression: LogicalExpression::Variable(rdf_identity_key_column(
                                companion_source,
                            )),
                            alias: Some(rdf_identity_key_column(output)),
                        });
                    }
                    if rdf_or_native_is_demanded {
                        companion_projections.push(Projection {
                            expression: LogicalExpression::Variable(rdf_group_key_column(
                                companion_source,
                            )),
                            alias: Some(rdf_group_key_column(output)),
                        });
                    }
                    continue;
                }
                let Some(alias) = &projected.alias else {
                    continue;
                };
                if !materialized_aliases.contains(alias) {
                    continue;
                }
                logical.expression = LogicalExpression::Variable(alias.clone());
                logical.alias = None;
                if exact_is_demanded {
                    companion_projections.push(Projection {
                        expression: LogicalExpression::Variable(rdf_exact_term_column(alias)),
                        alias: Some(rdf_exact_term_column(alias)),
                    });
                    companion_projections.push(Projection {
                        expression: LogicalExpression::Variable(rdf_identity_key_column(alias)),
                        alias: Some(rdf_identity_key_column(alias)),
                    });
                }
                if rdf_or_native_is_demanded {
                    companion_projections.push(Projection {
                        expression: LogicalExpression::Variable(rdf_group_key_column(alias)),
                        alias: Some(rdf_group_key_column(alias)),
                    });
                }
            }
            projections.extend(companion_projections);
        }

        Ok(LogicalOperator::Project(ProjectOp {
            projections,
            input: Box::new(input),
            pass_through_input: false,
        }))
    }

    fn translate_pattern_bind(
        &mut self,
        input: LogicalOperator,
        expression: &ast::Expression,
        variable: &str,
    ) -> Result<LogicalOperator> {
        self.translate_pattern_bind_with_annotations(
            input, expression, variable, None, false, false,
        )
    }

    fn translate_pattern_bind_with_annotations(
        &mut self,
        input: LogicalOperator,
        expression: &ast::Expression,
        variable: &str,
        annotation: Option<&OrdinaryExpressionExactAnnotations>,
        ordinary_exact_required: bool,
        rdf_or_native_required: bool,
    ) -> Result<LogicalOperator> {
        if !self.exact_mutation_bindings {
            return self.translate_ordinary_pattern_bind_with_annotations(
                input,
                expression,
                variable,
                annotation,
                ordinary_exact_required,
                rdf_or_native_required,
            );
        }

        if !self.mutation_requires_exact(variable) {
            if self.mutation_controls_selection(variable) {
                return Ok(LogicalOperator::Bind(BindOp {
                    expression: self
                        .translate_expression_with_annotations(expression, annotation)?,
                    variable: variable.to_string(),
                    input: Box::new(input),
                }));
            }
            return self.translate_ordinary_pattern_bind_with_annotations(
                input,
                expression,
                variable,
                annotation,
                ordinary_exact_required,
                rdf_or_native_required,
            );
        }

        // Evaluate the RDF expression exactly once into a sealed tagged term.
        // Visible, lossless N-Triples, and canonical identity columns project that
        // stored result, so volatile functions and mixed-kind IF/COALESCE
        // expressions cannot disagree about the term that was bound.
        let tagged_expression =
            self.tagged_bind_expression_with_annotations(expression, annotation)?;
        Ok(Self::bind_tagged_expression(
            input,
            tagged_expression,
            variable,
        ))
    }

    fn translate_ordinary_pattern_bind_with_annotations(
        &mut self,
        input: LogicalOperator,
        expression: &ast::Expression,
        variable: &str,
        annotation: Option<&OrdinaryExpressionExactAnnotations>,
        exact_required: bool,
        rdf_or_native_required: bool,
    ) -> Result<LogicalOperator> {
        if rdf_or_native_required {
            return self.bind_group_key_expression(input, expression, variable, annotation);
        }
        // Constant RDF terms already carry their source lexical identity. Keep
        // that identity even when no downstream consumer was known at the
        // binding site: DISTINCT and later subselect consumers may demand it,
        // and deriving a typed literal back from its native host value would
        // collapse e.g. "01"^^xsd:integer into "1"^^xsd:integer.
        if exact_required
            || matches!(
                expression,
                ast::Expression::Iri(_) | ast::Expression::Literal(_)
            )
        {
            let tagged_expression =
                self.tagged_bind_expression_with_annotations(expression, annotation)?;
            return Ok(Self::bind_tagged_expression(
                input,
                tagged_expression,
                variable,
            ));
        }
        let visible =
            self.translate_ordinary_expression_with_annotations(expression, annotation)?;
        let Some(kind) = self.exact_bound_term_kind(expression) else {
            return Ok(LogicalOperator::Bind(BindOp {
                expression: visible,
                variable: variable.to_string(),
                input: Box::new(input),
            }));
        };
        let tagger = match kind {
            ExactBoundTermKind::Iri => RDF_TAG_IRI_TERM,
            ExactBoundTermKind::Blank => RDF_TAG_BLANK_TERM,
            ExactBoundTermKind::Literal => RDF_TAG_LITERAL_TERM,
        };
        Ok(Self::bind_tagged_expression(
            input,
            LogicalExpression::FunctionCall {
                name: tagger.to_string(),
                args: vec![visible],
                distinct: false,
            },
            variable,
        ))
    }

    fn bind_tagged_expression(
        input: LogicalOperator,
        tagged_expression: LogicalExpression,
        variable: &str,
    ) -> LogicalOperator {
        let tagged_variable = rdf_tagged_term_column(variable);
        let tagged = LogicalOperator::Bind(BindOp {
            expression: tagged_expression,
            variable: tagged_variable.clone(),
            input: Box::new(input),
        });
        let visible = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_TAG_VALUE.to_string(),
                args: vec![LogicalExpression::Variable(tagged_variable.clone())],
                distinct: false,
            },
            variable: variable.to_string(),
            input: Box::new(tagged),
        });
        let exact = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_TAG_EXACT.to_string(),
                args: vec![LogicalExpression::Variable(tagged_variable.clone())],
                distinct: false,
            },
            variable: rdf_exact_term_column(variable),
            input: Box::new(visible),
        });
        LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_TERM_IDENTITY_KEY.to_string(),
                args: vec![LogicalExpression::Variable(tagged_variable)],
                distinct: false,
            },
            variable: rdf_identity_key_column(variable),
            input: Box::new(exact),
        })
    }

    /// Translates a helper expression without mutation-only RDF identity
    /// rewrites. This keeps the established extension-expression surface for
    /// aliases that do not feed a template or a term-sensitive selector.
    fn translate_ordinary_expression_with_annotations(
        &mut self,
        expression: &ast::Expression,
        annotation: Option<&OrdinaryExpressionExactAnnotations>,
    ) -> Result<LogicalExpression> {
        let previous = self.exact_mutation_bindings;
        self.exact_mutation_bindings = false;
        let translated = self.translate_expression_with_annotations(expression, annotation);
        self.exact_mutation_bindings = previous;
        translated
    }

    fn tagged_bind_expression_with_annotations(
        &mut self,
        expression: &ast::Expression,
        annotation: Option<&OrdinaryExpressionExactAnnotations>,
    ) -> Result<LogicalExpression> {
        match expression {
            ast::Expression::Iri(iri) => Ok(Self::tag_known_exact_term(
                self.translate_expression_with_annotations(expression, annotation)?,
                format!("<{}>", self.resolve_iri(iri)),
            )),
            ast::Expression::Literal(literal) => Ok(Self::tag_known_exact_term(
                self.translate_expression_with_annotations(expression, annotation)?,
                self.literal_as_exact_term(literal),
            )),
            ast::Expression::Variable(source) => Ok(LogicalExpression::FunctionCall {
                name: RDF_TAG_BOUND_TERM.to_string(),
                args: vec![
                    LogicalExpression::Variable(source.clone()),
                    LogicalExpression::Variable(rdf_exact_term_column(source)),
                ],
                distinct: false,
            }),
            ast::Expression::Bracketed(inner) => {
                let inner_annotation = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Bracketed(inner)) => {
                        Some(inner.as_ref())
                    }
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand expression annotation mismatch at tagged Bracketed"
                                .to_string(),
                        ));
                    }
                    None => None,
                };
                self.tagged_bind_expression_with_annotations(inner, inner_annotation)
            }
            ast::Expression::FunctionCall {
                function,
                arguments,
            } => {
                let argument_annotations = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::FunctionCall(annotations))
                        if annotations.len() == arguments.len() =>
                    {
                        Some(annotations.as_slice())
                    }
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand expression annotation mismatch at tagged FunctionCall"
                                .to_string(),
                        ));
                    }
                    None => None,
                };
                let name = self.translate_function_name(function).to_ascii_uppercase();
                if name == "IF" && arguments.len() == 3 {
                    return Ok(LogicalExpression::FunctionCall {
                        name,
                        args: vec![
                            self.translate_expression_with_annotations(
                                &arguments[0],
                                argument_annotations.map(|annotations| &annotations[0]),
                            )?,
                            self.tagged_bind_expression_with_annotations(
                                &arguments[1],
                                argument_annotations.map(|annotations| &annotations[1]),
                            )?,
                            self.tagged_bind_expression_with_annotations(
                                &arguments[2],
                                argument_annotations.map(|annotations| &annotations[2]),
                            )?,
                        ],
                        distinct: false,
                    });
                }
                if name == "COALESCE" {
                    return Ok(LogicalExpression::FunctionCall {
                        name,
                        args: arguments
                            .iter()
                            .enumerate()
                            .map(|(index, argument)| {
                                self.tagged_bind_expression_with_annotations(
                                    argument,
                                    argument_annotations.map(|annotations| &annotations[index]),
                                )
                            })
                            .collect::<Result<Vec<_>>>()?,
                        distinct: false,
                    });
                }
                if matches!(name.as_str(), "STRLANG" | "STRDT") && arguments.len() != 2 {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("{name} requires exactly two arguments"),
                    )));
                }
                if name == "STRLANG" {
                    return Ok(LogicalExpression::FunctionCall {
                        name: RDF_TAG_LANG_LITERAL_TERM.to_string(),
                        args: arguments
                            .iter()
                            .enumerate()
                            .map(|(index, argument)| {
                                self.translate_expression_with_annotations(
                                    argument,
                                    argument_annotations.map(|annotations| &annotations[index]),
                                )
                            })
                            .collect::<Result<Vec<_>>>()?,
                        distinct: false,
                    });
                }
                if name == "STRDT" {
                    return Ok(LogicalExpression::FunctionCall {
                        name: RDF_TAG_TYPED_LITERAL_TERM.to_string(),
                        args: vec![
                            self.translate_expression_with_annotations(
                                &arguments[0],
                                argument_annotations.map(|annotations| &annotations[0]),
                            )?,
                            self.tagged_bind_expression_with_annotations(
                                &arguments[1],
                                argument_annotations.map(|annotations| &annotations[1]),
                            )?,
                        ],
                        distinct: false,
                    });
                }

                let Some(kind) = Self::function_term_kind(function, &name) else {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("cannot preserve RDF term identity for function {name}"),
                    )));
                };
                let tagger = match kind {
                    ExactBoundTermKind::Iri => RDF_TAG_IRI_TERM,
                    ExactBoundTermKind::Blank => RDF_TAG_BLANK_TERM,
                    ExactBoundTermKind::Literal => RDF_TAG_LITERAL_TERM,
                };
                Ok(LogicalExpression::FunctionCall {
                    name: tagger.to_string(),
                    args: vec![self.translate_expression_with_annotations(expression, annotation)?],
                    distinct: false,
                })
            }
            ast::Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                let (condition_annotation, then_annotation, else_annotation) = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Conditional {
                        condition,
                        then_expression,
                        else_expression,
                    }) => (
                        Some(condition.as_ref()),
                        Some(then_expression.as_ref()),
                        Some(else_expression.as_ref()),
                    ),
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand expression annotation mismatch at tagged Conditional"
                                .to_string(),
                        ));
                    }
                    None => (None, None, None),
                };
                Ok(LogicalExpression::FunctionCall {
                    name: "IF".to_string(),
                    args: vec![
                        self.translate_expression_with_annotations(
                            condition,
                            condition_annotation,
                        )?,
                        self.tagged_bind_expression_with_annotations(
                            then_expression,
                            then_annotation,
                        )?,
                        self.tagged_bind_expression_with_annotations(
                            else_expression,
                            else_annotation,
                        )?,
                    ],
                    distinct: false,
                })
            }
            ast::Expression::Coalesce(expressions) => Ok(LogicalExpression::FunctionCall {
                name: "COALESCE".to_string(),
                args: {
                    let annotations = match annotation {
                        Some(OrdinaryExpressionExactAnnotations::Coalesce(annotations))
                            if annotations.len() == expressions.len() =>
                        {
                            Some(annotations.as_slice())
                        }
                        Some(_) => {
                            return Err(Error::Internal(
                                "ordinary exact-demand expression annotation mismatch at tagged Coalesce"
                                    .to_string(),
                            ));
                        }
                        None => None,
                    };
                    expressions
                        .iter()
                        .enumerate()
                        .map(|(index, expression)| {
                            self.tagged_bind_expression_with_annotations(
                                expression,
                                annotations.map(|annotations| &annotations[index]),
                            )
                        })
                        .collect::<Result<Vec<_>>>()?
                },
                distinct: false,
            }),
            _ => {
                let Some(kind) = self.exact_bound_term_kind(expression) else {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "cannot preserve RDF term identity for this binding expression",
                    )));
                };
                let function = match kind {
                    ExactBoundTermKind::Iri => RDF_TAG_IRI_TERM,
                    ExactBoundTermKind::Blank => RDF_TAG_BLANK_TERM,
                    ExactBoundTermKind::Literal => RDF_TAG_LITERAL_TERM,
                };
                Ok(LogicalExpression::FunctionCall {
                    name: function.to_string(),
                    args: vec![self.translate_expression_with_annotations(expression, annotation)?],
                    distinct: false,
                })
            }
        }
    }

    /// Preserves the exact RDF kind of a STRDT datatype expression.
    fn translate_strdt_datatype_argument_with_annotations(
        &mut self,
        expression: &ast::Expression,
        annotation: Option<&OrdinaryExpressionExactAnnotations>,
    ) -> Result<LogicalExpression> {
        self.tagged_bind_expression_with_annotations(expression, annotation)
    }

    fn tag_known_exact_term(visible: LogicalExpression, exact: String) -> LogicalExpression {
        LogicalExpression::FunctionCall {
            name: RDF_TAG_BOUND_TERM.to_string(),
            args: vec![
                visible,
                LogicalExpression::Literal(Value::String(exact.into())),
            ],
            distinct: false,
        }
    }

    fn function_term_kind(
        function: &ast::FunctionName,
        normalized_name: &str,
    ) -> Option<ExactBoundTermKind> {
        use ast::{BuiltInFunction as BuiltIn, FunctionName};

        match function {
            FunctionName::BuiltIn(BuiltIn::Iri | BuiltIn::Uuid | BuiltIn::Datatype) => {
                Some(ExactBoundTermKind::Iri)
            }
            FunctionName::BuiltIn(BuiltIn::Bnode) => Some(ExactBoundTermKind::Blank),
            FunctionName::BuiltIn(BuiltIn::Vector) => None,
            FunctionName::BuiltIn(_) => Some(ExactBoundTermKind::Literal),
            FunctionName::Custom(_) => match normalized_name {
                "IRI" | "URI" | "UUID" | "DATATYPE" => Some(ExactBoundTermKind::Iri),
                "BNODE" => Some(ExactBoundTermKind::Blank),
                "STR" | "LANG" | "LANGMATCHES" | "STRLEN" | "SUBSTR" | "UCASE" | "LCASE"
                | "STRSTARTS" | "STRENDS" | "CONTAINS" | "STRBEFORE" | "STRAFTER"
                | "ENCODE_FOR_URI" | "CONCAT" | "REPLACE" | "REGEX" | "ABS" | "ROUND" | "CEIL"
                | "FLOOR" | "RAND" | "NOW" | "YEAR" | "MONTH" | "DAY" | "HOURS" | "MINUTES"
                | "SECONDS" | "TIMEZONE" | "TZ" | "MD5" | "SHA1" | "SHA256" | "SHA384"
                | "SHA512" | "STRUUID" | "STRLANG" | "STRDT" | "SAMETERM" | "ISIRI" | "ISURI"
                | "ISBLANK" | "ISLITERAL" | "ISNUMERIC" => Some(ExactBoundTermKind::Literal),
                _ => None,
            },
        }
    }

    fn exact_bound_term_kind(&self, expression: &ast::Expression) -> Option<ExactBoundTermKind> {
        use ast::{BuiltInFunction as BuiltIn, Expression, FunctionName};

        match expression {
            Expression::Iri(_) => Some(ExactBoundTermKind::Iri),
            Expression::Literal(_)
            | Expression::Binary { .. }
            | Expression::Unary { .. }
            | Expression::Bound(_)
            | Expression::Exists(_)
            | Expression::NotExists(_)
            | Expression::In { .. }
            | Expression::NotIn { .. } => Some(ExactBoundTermKind::Literal),
            Expression::FunctionCall {
                function: FunctionName::BuiltIn(BuiltIn::Iri | BuiltIn::Uuid | BuiltIn::Datatype),
                ..
            } => Some(ExactBoundTermKind::Iri),
            Expression::FunctionCall {
                function: FunctionName::BuiltIn(BuiltIn::Bnode),
                ..
            } => Some(ExactBoundTermKind::Blank),
            Expression::FunctionCall {
                function: FunctionName::BuiltIn(BuiltIn::Vector),
                ..
            } => None,
            Expression::FunctionCall {
                function: FunctionName::BuiltIn(_),
                ..
            } => Some(ExactBoundTermKind::Literal),
            Expression::FunctionCall {
                function: FunctionName::Custom(iri),
                ..
            } => match self.resolve_iri(iri).to_ascii_uppercase().as_str() {
                "IRI" | "URI" | "UUID" | "DATATYPE" => Some(ExactBoundTermKind::Iri),
                "BNODE" => Some(ExactBoundTermKind::Blank),
                _ => None,
            },
            Expression::Conditional {
                then_expression,
                else_expression,
                ..
            } => {
                let then_kind = self.exact_bound_term_kind(then_expression)?;
                (self.exact_bound_term_kind(else_expression)? == then_kind).then_some(then_kind)
            }
            Expression::Coalesce(expressions) => {
                let mut expressions = expressions.iter();
                let first = self.exact_bound_term_kind(expressions.next()?)?;
                expressions
                    .all(|expression| self.exact_bound_term_kind(expression) == Some(first))
                    .then_some(first)
            }
            Expression::Bracketed(inner) => self.exact_bound_term_kind(inner),
            Expression::Variable(_) | Expression::Aggregate(_) => None,
        }
    }

    fn can_tag_order_expression(&self, expression: &ast::Expression) -> bool {
        use ast::Expression;

        match expression {
            Expression::Variable(_) => true,
            Expression::Bracketed(inner) => self.can_tag_order_expression(inner),
            Expression::Conditional {
                then_expression,
                else_expression,
                ..
            } => {
                self.can_tag_order_expression(then_expression)
                    && self.can_tag_order_expression(else_expression)
            }
            Expression::Coalesce(expressions) => expressions
                .iter()
                .all(|expression| self.can_tag_order_expression(expression)),
            _ => self.exact_bound_term_kind(expression).is_some(),
        }
    }

    /// Lowers a selector value without imposing one representation on the
    /// whole expression. Each IF/COALESCE branch independently retains an
    /// exact RDF term when possible or its native extension value otherwise.
    fn rdf_term_or_native_expression_with_annotations(
        &mut self,
        expression: &ast::Expression,
        annotation: Option<&OrdinaryExpressionExactAnnotations>,
    ) -> Result<LogicalExpression> {
        use ast::Expression;

        match expression {
            Expression::Bracketed(inner) => {
                let inner_annotation = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Bracketed(inner)) => {
                        Some(inner.as_ref())
                    }
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand expression annotation mismatch at RDF-or-native Bracketed"
                                .to_string(),
                        ));
                    }
                    None => None,
                };
                self.rdf_term_or_native_expression_with_annotations(inner, inner_annotation)
            }
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                let (condition_annotation, then_annotation, else_annotation) = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Conditional {
                        condition,
                        then_expression,
                        else_expression,
                    }) => (
                        Some(condition.as_ref()),
                        Some(then_expression.as_ref()),
                        Some(else_expression.as_ref()),
                    ),
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand expression annotation mismatch at RDF-or-native Conditional"
                                .to_string(),
                        ));
                    }
                    None => (None, None, None),
                };
                Ok(LogicalExpression::FunctionCall {
                    name: "IF".to_string(),
                    args: vec![
                        self.translate_expression_with_annotations(
                            condition,
                            condition_annotation,
                        )?,
                        self.rdf_term_or_native_expression_with_annotations(
                            then_expression,
                            then_annotation,
                        )?,
                        self.rdf_term_or_native_expression_with_annotations(
                            else_expression,
                            else_annotation,
                        )?,
                    ],
                    distinct: false,
                })
            }
            Expression::Coalesce(expressions) => {
                let annotations = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Coalesce(annotations))
                        if annotations.len() == expressions.len() =>
                    {
                        Some(annotations.as_slice())
                    }
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand expression annotation mismatch at RDF-or-native Coalesce"
                                .to_string(),
                        ));
                    }
                    None => None,
                };
                Ok(LogicalExpression::FunctionCall {
                    name: "COALESCE".to_string(),
                    args: expressions
                        .iter()
                        .enumerate()
                        .map(|(index, expression)| {
                            self.rdf_term_or_native_expression_with_annotations(
                                expression,
                                annotations.map(|annotations| &annotations[index]),
                            )
                        })
                        .collect::<Result<Vec<_>>>()?,
                    distinct: false,
                })
            }
            _ if self.can_tag_order_expression(expression) => Ok(LogicalExpression::FunctionCall {
                name: RDF_TERM_OR_NATIVE_VALUE.to_string(),
                args: vec![self.tagged_bind_expression_with_annotations(expression, annotation)?],
                distinct: false,
            }),
            _ => self.translate_ordinary_expression_with_annotations(expression, annotation),
        }
    }

    fn literal_as_exact_term(&self, literal: &ast::Literal) -> String {
        let term = if let Some(language) = &literal.language {
            Term::lang_literal(literal.value.clone(), language.clone())
        } else if let Some(datatype) = &literal.datatype {
            Term::typed_literal(literal.value.clone(), self.resolve_iri(datatype))
        } else {
            Term::literal(literal.value.clone())
        };
        term.to_ntriples()
    }

    fn translate_mutation_graph_pattern(
        &mut self,
        pattern: &ast::GraphPattern,
        exact_variables: HashSet<String>,
    ) -> Result<LogicalOperator> {
        let previous = self.exact_mutation_bindings;
        let previous_variables = self.exact_mutation_variables.replace(exact_variables);
        let previous_control = self.mutation_control_variables.replace(HashSet::new());
        self.exact_mutation_bindings = true;
        let translated = self.translate_graph_pattern(pattern);
        self.exact_mutation_bindings = previous;
        self.exact_mutation_variables = previous_variables;
        self.mutation_control_variables = previous_control;
        translated.map(Self::seal_modify_bindings)
    }

    fn mutation_requires_exact(&self, variable: &str) -> bool {
        self.exact_mutation_bindings
            && self
                .exact_mutation_variables
                .as_ref()
                .is_some_and(|variables| variables.contains(variable))
    }

    fn mutation_controls_selection(&self, variable: &str) -> bool {
        self.exact_mutation_bindings
            && self
                .mutation_control_variables
                .as_ref()
                .is_some_and(|variables| variables.contains(variable))
    }

    fn mutation_template_variables(quads: &[ast::QuadPattern]) -> HashSet<String> {
        let mut variables = HashSet::new();
        for quad in quads {
            for term in [&quad.triple.subject, &quad.triple.object] {
                if let ast::TripleTerm::Variable(variable) = term {
                    variables.insert(variable.clone());
                }
            }
            if let ast::PropertyPath::Variable(variable) = &quad.triple.predicate {
                variables.insert(variable.clone());
            }
            if let Some(ast::VariableOrIri::Variable(variable)) = &quad.graph {
                variables.insert(variable.clone());
            }
        }
        variables
    }

    fn ordinary_select_exact_annotations(
        select: &ast::SelectQuery,
        downstream: &HashSet<String>,
    ) -> Result<(OrdinarySelectExactAnnotations, HashSet<String>)> {
        let mut annotation = Self::build_ordinary_select_annotations(select)?;
        Self::analyze_ordinary_select(select, &mut annotation, downstream)?;
        Self::mark_rdf_or_native_select_demands(select, &mut annotation, &HashSet::new())?;
        let exports = annotation.exports.iter().cloned().collect();
        Ok((annotation, exports))
    }

    fn build_ordinary_select_annotations(
        select: &ast::SelectQuery,
    ) -> Result<OrdinarySelectExactAnnotations> {
        let mut variables = OrdinaryVariableTable::default();
        let mut bound = OrdinaryMembershipState::default();
        let (where_clause, _) = Self::build_ordinary_pattern_annotations(
            &select.where_clause,
            &mut variables,
            &mut bound,
        )?;

        let mut projection = Vec::new();
        let mut exports = Vec::new();
        match &select.projection {
            ast::Projection::Wildcard => {
                if Self::select_is_grouped(select) {
                    let mut grouped_outputs = HashSet::new();
                    if let Some(group_by) = &select.solution_modifiers.group_by {
                        Self::collect_group_output_variables(group_by, &mut grouped_outputs);
                    }
                    exports.extend(grouped_outputs);
                    exports.sort();
                } else {
                    for (id, name) in variables.names.iter().enumerate() {
                        if bound.contains(id) {
                            exports.push(name.clone());
                        }
                    }
                }
            }
            ast::Projection::Variables(projected) => {
                for item in projected {
                    let (expression, _) = Self::build_ordinary_expression_annotations(
                        &item.expression,
                        &mut variables,
                        &mut bound,
                    )?;
                    projection.push(OrdinaryProjectionExactAnnotation {
                        exact: false,
                        rdf_or_native: false,
                        expression,
                    });
                    if let Some(output) = item.alias.as_deref().or({
                        if let ast::Expression::Variable(variable) = &item.expression {
                            Some(variable.as_str())
                        } else {
                            None
                        }
                    }) {
                        exports.push(output.to_string());
                        bound.insert(variables.intern(output));
                    }
                }
            }
        }

        let group_by = select
            .solution_modifiers
            .group_by
            .as_ref()
            .map(|conditions| {
                conditions
                    .iter()
                    .map(|condition| match condition {
                        ast::GroupCondition::Variable(variable) => {
                            variables.intern(variable);
                            Ok(OrdinaryGroupExactAnnotation::Variable)
                        }
                        ast::GroupCondition::Expression { expression, alias } => {
                            let (annotation, _) = Self::build_ordinary_expression_annotations(
                                expression,
                                &mut variables,
                                &mut bound,
                            )?;
                            if let Some(alias) = alias {
                                bound.insert(variables.intern(alias));
                            }
                            Ok(OrdinaryGroupExactAnnotation::Expression {
                                exact: false,
                                expression: annotation,
                            })
                        }
                        ast::GroupCondition::BuiltInCall(expression) => {
                            let (annotation, _) = Self::build_ordinary_expression_annotations(
                                expression,
                                &mut variables,
                                &mut bound,
                            )?;
                            Ok(OrdinaryGroupExactAnnotation::BuiltInCall(annotation))
                        }
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let having = select
            .solution_modifiers
            .having
            .as_ref()
            .map(|expression| {
                Self::build_ordinary_expression_annotations(expression, &mut variables, &mut bound)
                    .map(|(annotation, _)| annotation)
            })
            .transpose()?;
        let order_by = select
            .solution_modifiers
            .order_by
            .as_ref()
            .map(|conditions| {
                conditions
                    .iter()
                    .map(|condition| {
                        Self::build_ordinary_expression_annotations(
                            &condition.expression,
                            &mut variables,
                            &mut bound,
                        )
                        .map(|(annotation, _)| annotation)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;

        Ok(OrdinarySelectExactAnnotations {
            variables,
            exports,
            projection,
            group_by,
            having,
            order_by,
            where_clause,
        })
    }

    fn sorted_ids(ids: HashSet<usize>) -> Vec<usize> {
        let mut ids = ids.into_iter().collect::<Vec<_>>();
        ids.sort_unstable();
        ids
    }

    fn observe_bound_variable(
        name: &str,
        variables: &mut OrdinaryVariableTable,
        bound: &mut OrdinaryMembershipState,
        entry: usize,
        correlated: &mut HashSet<usize>,
    ) -> usize {
        let id = variables.intern(name);
        bound.ensure(id);
        if bound.was_present_at(entry, id) {
            correlated.insert(id);
        }
        if bound.insert(id) {
            #[cfg(test)]
            ORDINARY_EXACT_STRUCTURAL_INSERT_WORK.with(|work| work.set(work.get() + 1));
        }
        id
    }

    fn build_ordinary_pattern_annotations(
        pattern: &ast::GraphPattern,
        variables: &mut OrdinaryVariableTable,
        bound: &mut OrdinaryMembershipState,
    ) -> Result<(OrdinaryPatternExactAnnotations, Vec<usize>)> {
        #[cfg(test)]
        ORDINARY_EXACT_PATTERN_VISITS.with(|visits| visits.set(visits.get() + 1));
        let entry = bound.mark();
        match pattern {
            ast::GraphPattern::Basic(_) => {
                let mut names = HashSet::new();
                Self::collect_pattern_output_variables(pattern, &mut names);
                let mut shared = HashSet::new();
                for name in names {
                    Self::observe_bound_variable(&name, variables, bound, entry, &mut shared);
                }
                let shared = Self::sorted_ids(shared);
                Ok((
                    OrdinaryPatternExactAnnotations::Basic {
                        shared: shared.clone(),
                    },
                    shared,
                ))
            }
            ast::GraphPattern::InlineData(data) => {
                let mut shared = HashSet::new();
                for (index, name) in data.variables.iter().enumerate() {
                    let id = variables.intern(name);
                    bound.ensure(id);
                    let may_bind = data
                        .values
                        .iter()
                        .any(|row| row.get(index).is_some_and(Option::is_some));
                    if may_bind && bound.was_present_at(entry, id) {
                        shared.insert(id);
                    }
                    if may_bind && bound.insert(id) {
                        #[cfg(test)]
                        ORDINARY_EXACT_STRUCTURAL_INSERT_WORK.with(|work| work.set(work.get() + 1));
                    }
                }
                let shared = Self::sorted_ids(shared);
                Ok((
                    OrdinaryPatternExactAnnotations::InlineData {
                        shared: shared.clone(),
                    },
                    shared,
                ))
            }
            ast::GraphPattern::Bind {
                expression,
                variable,
            } => {
                let (expression_annotation, correlated) =
                    Self::build_ordinary_expression_annotations(expression, variables, bound)?;
                let variable = variables.intern(variable);
                bound.insert(variable);
                Ok((
                    OrdinaryPatternExactAnnotations::Bind {
                        variable,
                        exact: false,
                        rdf_or_native: false,
                        expression: expression_annotation,
                    },
                    correlated,
                ))
            }
            ast::GraphPattern::Filter(expression) => {
                let (annotation, correlated) =
                    Self::build_ordinary_expression_annotations(expression, variables, bound)?;
                Ok((
                    OrdinaryPatternExactAnnotations::Filter(annotation),
                    correlated,
                ))
            }
            ast::GraphPattern::Group(patterns) => {
                let mut children = std::iter::repeat_with(|| None)
                    .take(patterns.len())
                    .collect::<Vec<_>>();
                let mut correlated = HashSet::new();
                for (index, child) in patterns.iter().enumerate() {
                    if !matches!(child, ast::GraphPattern::Filter(_)) {
                        let (annotation, child_correlated) =
                            Self::build_ordinary_pattern_annotations(child, variables, bound)?;
                        for id in child_correlated {
                            if bound.was_present_at(entry, id) {
                                correlated.insert(id);
                            }
                        }
                        children[index] = Some(annotation);
                    }
                }
                // FILTER has group scope. Build it only after the group's
                // structural bindings are known, then translate from this owned node.
                for (index, child) in patterns.iter().enumerate() {
                    if matches!(child, ast::GraphPattern::Filter(_)) {
                        let (annotation, child_correlated) =
                            Self::build_ordinary_pattern_annotations(child, variables, bound)?;
                        for id in child_correlated {
                            if bound.was_present_at(entry, id) {
                                correlated.insert(id);
                            }
                        }
                        children[index] = Some(annotation);
                    }
                }
                let correlated = Self::sorted_ids(correlated);
                Ok((
                    OrdinaryPatternExactAnnotations::Group(
                        children
                            .into_iter()
                            .map(|child| {
                                child.ok_or_else(|| {
                                    Error::Internal(
                                        "ordinary exact-demand group annotation is missing"
                                            .to_string(),
                                    )
                                })
                            })
                            .collect::<Result<Vec<_>>>()?,
                    ),
                    correlated,
                ))
            }
            ast::GraphPattern::Union(alternatives) => {
                let mut children = Vec::with_capacity(alternatives.len());
                let mut union_correlated = HashSet::new();
                let mut additions = HashSet::new();
                for alternative in alternatives {
                    bound.rollback(entry)?;
                    let (annotation, correlated) =
                        Self::build_ordinary_pattern_annotations(alternative, variables, bound)?;
                    for id in bound.changed_ids_since(entry) {
                        if bound.contains(id) {
                            additions.insert(id);
                        }
                    }
                    for id in &correlated {
                        union_correlated.insert(*id);
                    }
                    children.push(OrdinaryUnionExactAnnotation {
                        pattern: annotation,
                        correlated,
                    });
                }
                bound.rollback(entry)?;
                for id in additions {
                    bound.insert(id);
                }
                let union_correlated = Self::sorted_ids(union_correlated);
                Ok((
                    OrdinaryPatternExactAnnotations::Union(children),
                    union_correlated,
                ))
            }
            ast::GraphPattern::Optional(child) => {
                let (annotation, correlated) =
                    Self::build_ordinary_pattern_annotations(child, variables, bound)?;
                Ok((
                    OrdinaryPatternExactAnnotations::Optional {
                        pattern: Box::new(annotation),
                        correlated: correlated.clone(),
                    },
                    correlated,
                ))
            }
            ast::GraphPattern::Minus(child) => {
                let (annotation, correlated) =
                    Self::build_ordinary_pattern_annotations(child, variables, bound)?;
                bound.rollback(entry)?;
                Ok((
                    OrdinaryPatternExactAnnotations::Minus {
                        pattern: Box::new(annotation),
                        correlated: correlated.clone(),
                    },
                    correlated,
                ))
            }
            ast::GraphPattern::NamedGraph { graph, pattern } => {
                let mut correlated = HashSet::new();
                let shared_graph = if let ast::VariableOrIri::Variable(name) = graph {
                    let id = Self::observe_bound_variable(
                        name,
                        variables,
                        bound,
                        entry,
                        &mut correlated,
                    );
                    correlated.contains(&id).then_some(id)
                } else {
                    None
                };
                let (annotation, child_correlated) =
                    Self::build_ordinary_pattern_annotations(pattern, variables, bound)?;
                for id in child_correlated {
                    if bound.was_present_at(entry, id) {
                        correlated.insert(id);
                    }
                }
                let correlated = Self::sorted_ids(correlated);
                Ok((
                    OrdinaryPatternExactAnnotations::NamedGraph {
                        pattern: Box::new(annotation),
                        shared_graph,
                    },
                    correlated,
                ))
            }
            ast::GraphPattern::SubSelect(select) => {
                let annotation = Self::build_ordinary_select_annotations(select)?;
                let mut shared = HashSet::new();
                for output in &annotation.exports {
                    Self::observe_bound_variable(output, variables, bound, entry, &mut shared);
                }
                let shared = Self::sorted_ids(shared);
                Ok((
                    OrdinaryPatternExactAnnotations::SubSelect {
                        select: Box::new(annotation),
                        shared: shared.clone(),
                    },
                    shared,
                ))
            }
            ast::GraphPattern::Service { .. } => {
                Ok((OrdinaryPatternExactAnnotations::Service, Vec::new()))
            }
        }
    }

    fn build_ordinary_expression_annotations(
        expression: &ast::Expression,
        variables: &mut OrdinaryVariableTable,
        bound: &mut OrdinaryMembershipState,
    ) -> Result<(OrdinaryExpressionExactAnnotations, Vec<usize>)> {
        let mut correlated = OrdinaryMembershipState::default();
        let annotation = Self::build_ordinary_expression_annotation_into(
            expression,
            variables,
            bound,
            &mut correlated,
        )?;
        let mut correlated = correlated
            .undo
            .into_iter()
            .map(|(id, _, _)| id)
            .collect::<Vec<_>>();
        correlated.sort_unstable();
        Ok((annotation, correlated))
    }

    fn build_ordinary_expression_annotation_into(
        expression: &ast::Expression,
        variables: &mut OrdinaryVariableTable,
        bound: &mut OrdinaryMembershipState,
        correlated: &mut OrdinaryMembershipState,
    ) -> Result<OrdinaryExpressionExactAnnotations> {
        #[cfg(test)]
        ORDINARY_EXACT_EXPRESSION_BUILDS.with(|count| count.set(count.get() + 1));
        use ast::Expression;
        let annotation = match expression {
            Expression::Variable(name) => {
                let id = variables.intern(name);
                bound.ensure(id);
                #[cfg(test)]
                ORDINARY_EXACT_CORRELATION_WORK.with(|work| work.set(work.get() + 1));
                if bound.contains(id) {
                    correlated.insert(id);
                }
                OrdinaryExpressionExactAnnotations::Variable
            }
            Expression::Iri(_) => OrdinaryExpressionExactAnnotations::Iri,
            Expression::Literal(_) => OrdinaryExpressionExactAnnotations::Literal,
            Expression::Bound(name) => {
                let id = variables.intern(name);
                bound.ensure(id);
                #[cfg(test)]
                ORDINARY_EXACT_CORRELATION_WORK.with(|work| work.set(work.get() + 1));
                if bound.contains(id) {
                    correlated.insert(id);
                }
                OrdinaryExpressionExactAnnotations::Bound
            }
            Expression::Binary { left, right, .. } => {
                let left = Self::build_ordinary_expression_annotation_into(
                    left, variables, bound, correlated,
                )?;
                let right = Self::build_ordinary_expression_annotation_into(
                    right, variables, bound, correlated,
                )?;
                OrdinaryExpressionExactAnnotations::Binary(Box::new(left), Box::new(right))
            }
            Expression::Unary { operand, .. } => {
                let operand = Self::build_ordinary_expression_annotation_into(
                    operand, variables, bound, correlated,
                )?;
                OrdinaryExpressionExactAnnotations::Unary(Box::new(operand))
            }
            Expression::FunctionCall { arguments, .. } => {
                let mut annotations = Vec::with_capacity(arguments.len());
                for argument in arguments {
                    let annotation = Self::build_ordinary_expression_annotation_into(
                        argument, variables, bound, correlated,
                    )?;
                    annotations.push(annotation);
                }
                OrdinaryExpressionExactAnnotations::FunctionCall(annotations)
            }
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                let condition = Self::build_ordinary_expression_annotation_into(
                    condition, variables, bound, correlated,
                )?;
                let then_expression = Self::build_ordinary_expression_annotation_into(
                    then_expression,
                    variables,
                    bound,
                    correlated,
                )?;
                let else_expression = Self::build_ordinary_expression_annotation_into(
                    else_expression,
                    variables,
                    bound,
                    correlated,
                )?;
                OrdinaryExpressionExactAnnotations::Conditional {
                    condition: Box::new(condition),
                    then_expression: Box::new(then_expression),
                    else_expression: Box::new(else_expression),
                }
            }
            Expression::Coalesce(expressions) => {
                let mut annotations = Vec::with_capacity(expressions.len());
                for expression in expressions {
                    let annotation = Self::build_ordinary_expression_annotation_into(
                        expression, variables, bound, correlated,
                    )?;
                    annotations.push(annotation);
                }
                OrdinaryExpressionExactAnnotations::Coalesce(annotations)
            }
            Expression::Exists(pattern) | Expression::NotExists(pattern) => {
                let entry = bound.mark();
                let (pattern, inner_correlated) =
                    Self::build_ordinary_pattern_annotations(pattern, variables, bound)?;
                bound.rollback(entry)?;
                for id in inner_correlated.iter().copied() {
                    #[cfg(test)]
                    ORDINARY_EXACT_CORRELATION_WORK.with(|work| work.set(work.get() + 1));
                    correlated.insert(id);
                }
                if matches!(expression, Expression::Exists(_)) {
                    OrdinaryExpressionExactAnnotations::Exists {
                        pattern: Box::new(pattern),
                        correlated: inner_correlated,
                    }
                } else {
                    OrdinaryExpressionExactAnnotations::NotExists {
                        pattern: Box::new(pattern),
                        correlated: inner_correlated,
                    }
                }
            }
            Expression::In { expression, list } => {
                let head = Self::build_ordinary_expression_annotation_into(
                    expression, variables, bound, correlated,
                )?;
                let mut annotations = Vec::with_capacity(list.len());
                for item in list {
                    let annotation = Self::build_ordinary_expression_annotation_into(
                        item, variables, bound, correlated,
                    )?;
                    annotations.push(annotation);
                }
                OrdinaryExpressionExactAnnotations::In {
                    expression: Box::new(head),
                    list: annotations,
                }
            }
            Expression::NotIn { expression, list } => {
                let head = Self::build_ordinary_expression_annotation_into(
                    expression, variables, bound, correlated,
                )?;
                let mut annotations = Vec::with_capacity(list.len());
                for item in list {
                    let annotation = Self::build_ordinary_expression_annotation_into(
                        item, variables, bound, correlated,
                    )?;
                    annotations.push(annotation);
                }
                OrdinaryExpressionExactAnnotations::NotIn {
                    expression: Box::new(head),
                    list: annotations,
                }
            }
            Expression::Aggregate(aggregate) => {
                let annotation = match aggregate {
                    ast::AggregateExpression::Count { expression, .. } => {
                        let expression = expression
                            .as_ref()
                            .map(|expression| {
                                Self::build_ordinary_expression_annotation_into(
                                    expression, variables, bound, correlated,
                                )
                                .map(Box::new)
                            })
                            .transpose()?;
                        OrdinaryAggregateExactAnnotations::Count(expression)
                    }
                    ast::AggregateExpression::Sum { expression, .. }
                    | ast::AggregateExpression::Average { expression, .. }
                    | ast::AggregateExpression::Minimum { expression }
                    | ast::AggregateExpression::Maximum { expression }
                    | ast::AggregateExpression::Sample { expression }
                    | ast::AggregateExpression::GroupConcat { expression, .. } => {
                        let inner = Self::build_ordinary_expression_annotation_into(
                            expression, variables, bound, correlated,
                        )?;
                        match aggregate {
                            ast::AggregateExpression::Sum { .. } => {
                                OrdinaryAggregateExactAnnotations::Sum(Box::new(inner))
                            }
                            ast::AggregateExpression::Average { .. } => {
                                OrdinaryAggregateExactAnnotations::Average(Box::new(inner))
                            }
                            ast::AggregateExpression::Minimum { .. } => {
                                OrdinaryAggregateExactAnnotations::Minimum(Box::new(inner))
                            }
                            ast::AggregateExpression::Maximum { .. } => {
                                OrdinaryAggregateExactAnnotations::Maximum(Box::new(inner))
                            }
                            ast::AggregateExpression::Sample { .. } => {
                                OrdinaryAggregateExactAnnotations::Sample(Box::new(inner))
                            }
                            ast::AggregateExpression::GroupConcat { .. } => {
                                OrdinaryAggregateExactAnnotations::GroupConcat(Box::new(inner))
                            }
                            ast::AggregateExpression::Count { .. } => {
                                return Err(Error::Internal(
                                    "ordinary exact-demand aggregate annotation mismatch"
                                        .to_string(),
                                ));
                            }
                        }
                    }
                };
                OrdinaryExpressionExactAnnotations::Aggregate(annotation)
            }
            Expression::Bracketed(inner) => {
                let inner = Self::build_ordinary_expression_annotation_into(
                    inner, variables, bound, correlated,
                )?;
                OrdinaryExpressionExactAnnotations::Bracketed(Box::new(inner))
            }
        };
        Ok(annotation)
    }

    fn add_ordinary_identity_dependencies(
        expression: &ast::Expression,
        variables: &mut OrdinaryVariableTable,
        demand: &mut OrdinaryMembershipState,
    ) {
        let mut names = HashSet::new();
        Self::collect_identity_dependencies(expression, &mut names);
        for name in names {
            demand.insert(variables.intern(&name));
        }
    }

    fn collect_ordinary_exact_consumer_names(
        expression: &ast::Expression,
        required: &mut HashSet<String>,
    ) {
        use ast::{BuiltInFunction, Expression, FunctionName};
        match expression {
            Expression::FunctionCall {
                function,
                arguments,
            } => {
                match function {
                    FunctionName::BuiltIn(BuiltInFunction::StrDt) => {
                        if let Some(datatype) = arguments.get(1) {
                            Self::collect_identity_dependencies(datatype, required);
                        }
                    }
                    FunctionName::BuiltIn(BuiltInFunction::SameTerm) => {
                        for argument in arguments {
                            Self::collect_identity_dependencies(argument, required);
                        }
                    }
                    _ => {}
                }
                for argument in arguments {
                    Self::collect_ordinary_exact_consumer_names(argument, required);
                }
            }
            Expression::Binary { left, right, .. } => {
                Self::collect_ordinary_exact_consumer_names(left, required);
                Self::collect_ordinary_exact_consumer_names(right, required);
            }
            Expression::Unary { operand, .. } | Expression::Bracketed(operand) => {
                Self::collect_ordinary_exact_consumer_names(operand, required);
            }
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                Self::collect_ordinary_exact_consumer_names(condition, required);
                Self::collect_ordinary_exact_consumer_names(then_expression, required);
                Self::collect_ordinary_exact_consumer_names(else_expression, required);
            }
            Expression::Coalesce(expressions) => {
                for expression in expressions {
                    Self::collect_ordinary_exact_consumer_names(expression, required);
                }
            }
            Expression::In { expression, list } | Expression::NotIn { expression, list } => {
                Self::collect_ordinary_exact_consumer_names(expression, required);
                for expression in list {
                    Self::collect_ordinary_exact_consumer_names(expression, required);
                }
            }
            Expression::Aggregate(aggregate) => match aggregate {
                ast::AggregateExpression::Count { expression, .. } => {
                    if let Some(expression) = expression {
                        Self::collect_ordinary_exact_consumer_names(expression, required);
                    }
                }
                ast::AggregateExpression::Sum { expression, .. }
                | ast::AggregateExpression::Average { expression, .. }
                | ast::AggregateExpression::Minimum { expression }
                | ast::AggregateExpression::Maximum { expression }
                | ast::AggregateExpression::Sample { expression }
                | ast::AggregateExpression::GroupConcat { expression, .. } => {
                    Self::collect_ordinary_exact_consumer_names(expression, required);
                }
            },
            Expression::Exists(pattern) | Expression::NotExists(pattern) => {
                Self::collect_ordinary_pattern_exact_consumer_names(pattern, required);
            }
            Expression::Variable(_)
            | Expression::Iri(_)
            | Expression::Literal(_)
            | Expression::Bound(_) => {}
        }
    }

    fn collect_ordinary_pattern_exact_consumer_names(
        pattern: &ast::GraphPattern,
        required: &mut HashSet<String>,
    ) {
        match pattern {
            ast::GraphPattern::Group(patterns) | ast::GraphPattern::Union(patterns) => {
                for pattern in patterns {
                    Self::collect_ordinary_pattern_exact_consumer_names(pattern, required);
                }
            }
            ast::GraphPattern::Optional(pattern)
            | ast::GraphPattern::Minus(pattern)
            | ast::GraphPattern::NamedGraph { pattern, .. }
            | ast::GraphPattern::Service { pattern, .. } => {
                Self::collect_ordinary_pattern_exact_consumer_names(pattern, required);
            }
            ast::GraphPattern::Filter(expression) | ast::GraphPattern::Bind { expression, .. } => {
                Self::collect_ordinary_exact_consumer_names(expression, required);
            }
            // A SPARQL subselect is a lexical boundary. Its local names must
            // never be mapped onto same-spelled aliases in the outer SELECT.
            ast::GraphPattern::SubSelect(_)
            | ast::GraphPattern::Basic(_)
            | ast::GraphPattern::InlineData(_) => {}
        }
    }

    fn analyze_ordinary_select(
        select: &ast::SelectQuery,
        annotation: &mut OrdinarySelectExactAnnotations,
        downstream: &HashSet<String>,
    ) -> Result<()> {
        let mut demand = OrdinaryMembershipState::default();
        let mut modifier_consumers = HashSet::new();
        if let Some(group_by) = &select.solution_modifiers.group_by {
            for condition in group_by {
                match condition {
                    // Direct variables remain native-capable. A separate
                    // RDF-or-native projection demand carries an existing
                    // discriminated key across subselect boundaries.
                    ast::GroupCondition::Variable(_) => {}
                    ast::GroupCondition::Expression { expression, .. }
                    | ast::GroupCondition::BuiltInCall(expression) => {
                        Self::collect_identity_dependencies(expression, &mut modifier_consumers);
                        Self::collect_ordinary_exact_consumer_names(
                            expression,
                            &mut modifier_consumers,
                        );
                    }
                }
            }
        }
        if let Some(having) = &select.solution_modifiers.having {
            let rewritten = Self::rewrite_having_aggregates(having, &select.projection);
            Self::collect_ordinary_exact_consumer_names(&rewritten, &mut modifier_consumers);
        }
        if let Some(order_by) = &select.solution_modifiers.order_by {
            for condition in order_by {
                let rewritten =
                    Self::rewrite_having_aggregates(&condition.expression, &select.projection);
                Self::collect_ordinary_exact_consumer_names(&rewritten, &mut modifier_consumers);
            }
        }

        let mut output_indices = HashMap::new();
        if let ast::Projection::Variables(projected) = &select.projection {
            for (index, item) in projected.iter().enumerate() {
                if let Some(output) = item.alias.as_deref().or({
                    if let ast::Expression::Variable(variable) = &item.expression {
                        Some(variable.as_str())
                    } else {
                        None
                    }
                }) {
                    output_indices.insert(output.to_string(), index);
                }
            }
        }
        let mut demanded_outputs = downstream.clone();
        demanded_outputs.extend(
            modifier_consumers
                .iter()
                .filter(|name| output_indices.contains_key(*name))
                .cloned(),
        );
        for name in &modifier_consumers {
            if !output_indices.contains_key(name) {
                demand.insert(annotation.variables.intern(name));
            }
        }

        match &select.projection {
            ast::Projection::Wildcard => {
                for name in downstream {
                    demand.insert(annotation.variables.intern(name));
                }
            }
            ast::Projection::Variables(projected) => {
                if projected.len() != annotation.projection.len() {
                    return Err(Error::Internal(format!(
                        "ordinary exact-demand projection annotation mismatch: expected {}, got {}",
                        projected.len(),
                        annotation.projection.len()
                    )));
                }
                for (item, item_annotation) in
                    projected.iter().zip(annotation.projection.iter_mut())
                {
                    Self::analyze_ordinary_expression(
                        &item.expression,
                        &mut item_annotation.expression,
                        &mut annotation.variables,
                        &mut demand,
                    )?;
                    let output = item.alias.as_deref().or({
                        if let ast::Expression::Variable(variable) = &item.expression {
                            Some(variable.as_str())
                        } else {
                            None
                        }
                    });
                    if output.is_some_and(|name| demanded_outputs.contains(name)) {
                        item_annotation.exact = true;
                        Self::add_ordinary_identity_dependencies(
                            &item.expression,
                            &mut annotation.variables,
                            &mut demand,
                        );
                    }
                }
                // A computed projection alias belongs after WHERE. Modifier
                // demand for that alias is mapped through the definition above,
                // not leaked to a same-spelled hidden input binding.
                for item in projected {
                    if !matches!(item.expression, ast::Expression::Variable(_))
                        && let Some(alias) = &item.alias
                        && let Some(id) = annotation.variables.get(alias)
                    {
                        demand.remove(id);
                    }
                }
            }
        }

        if let (Some(group_by), Some(group_annotations)) = (
            &select.solution_modifiers.group_by,
            annotation.group_by.as_mut(),
        ) {
            for (condition, condition_annotation) in group_by.iter().zip(group_annotations) {
                match (condition, condition_annotation) {
                    (
                        ast::GroupCondition::Expression { expression, alias },
                        OrdinaryGroupExactAnnotation::Expression {
                            exact,
                            expression: expression_annotation,
                        },
                    ) => {
                        if let Some(alias) = alias {
                            let alias_id = annotation.variables.intern(alias);
                            if demand.contains(alias_id) {
                                *exact = true;
                                Self::add_ordinary_identity_dependencies(
                                    expression,
                                    &mut annotation.variables,
                                    &mut demand,
                                );
                            }
                            demand.remove(alias_id);
                        }
                        Self::analyze_ordinary_expression(
                            expression,
                            expression_annotation,
                            &mut annotation.variables,
                            &mut demand,
                        )?;
                    }
                    (
                        ast::GroupCondition::BuiltInCall(expression),
                        OrdinaryGroupExactAnnotation::BuiltInCall(expression_annotation),
                    ) => Self::analyze_ordinary_expression(
                        expression,
                        expression_annotation,
                        &mut annotation.variables,
                        &mut demand,
                    )?,
                    (ast::GroupCondition::Variable(_), OrdinaryGroupExactAnnotation::Variable) => {}
                    _ => {
                        return Err(Error::Internal(
                            "ordinary exact-demand GROUP BY annotation mismatch".to_string(),
                        ));
                    }
                }
            }
        }
        if let (Some(having), Some(having_annotation)) = (
            &select.solution_modifiers.having,
            annotation.having.as_mut(),
        ) {
            Self::analyze_ordinary_expression(
                having,
                having_annotation,
                &mut annotation.variables,
                &mut demand,
            )?;
        }
        if let (Some(order_by), Some(order_annotations)) = (
            &select.solution_modifiers.order_by,
            annotation.order_by.as_mut(),
        ) {
            for (condition, expression_annotation) in order_by.iter().zip(order_annotations) {
                Self::analyze_ordinary_expression(
                    &condition.expression,
                    expression_annotation,
                    &mut annotation.variables,
                    &mut demand,
                )?;
            }
        }

        Self::analyze_ordinary_pattern(
            &select.where_clause,
            &mut annotation.where_clause,
            &mut annotation.variables,
            &mut demand,
        )
    }

    /// Marks projection outputs whose RDF-or-native discriminator must cross
    /// a SELECT boundary for grouping, ordering, or DISTINCT. This is separate
    /// from exact RDF-term demand so native extension values remain legal.
    fn mark_rdf_or_native_select_demands(
        select: &ast::SelectQuery,
        annotation: &mut OrdinarySelectExactAnnotations,
        downstream: &HashSet<String>,
    ) -> Result<()> {
        let mut consumers = downstream.clone();
        for condition in select.solution_modifiers.group_by.iter().flatten() {
            match condition {
                ast::GroupCondition::Variable(variable) => {
                    consumers.insert(variable.clone());
                }
                ast::GroupCondition::Expression { expression, .. }
                | ast::GroupCondition::BuiltInCall(expression) => {
                    Self::collect_identity_dependencies(expression, &mut consumers);
                }
            }
        }
        for condition in select.solution_modifiers.order_by.iter().flatten() {
            Self::collect_identity_dependencies(&condition.expression, &mut consumers);
        }

        let mut outputs = HashMap::new();
        let mut materialized_outputs = HashSet::new();
        match &select.projection {
            ast::Projection::Wildcard => {
                if matches!(select.modifier, ast::SelectModifier::Distinct) {
                    consumers.extend(annotation.exports.iter().cloned());
                }
            }
            ast::Projection::Variables(projected) => {
                if projected.len() != annotation.projection.len() {
                    return Err(Error::Internal(format!(
                        "RDF-or-native projection annotation mismatch: expected {}, got {}",
                        projected.len(),
                        annotation.projection.len()
                    )));
                }
                for (index, item) in projected.iter().enumerate() {
                    if let Some(alias) = &item.alias {
                        materialized_outputs.insert(alias.clone());
                    }
                    if let Some(output) = item.alias.as_deref().or({
                        if let ast::Expression::Variable(variable) = &item.expression {
                            Some(variable.as_str())
                        } else {
                            None
                        }
                    }) {
                        outputs.insert(output.to_string(), index);
                    }
                }
                if matches!(select.modifier, ast::SelectModifier::Distinct) {
                    consumers.extend(outputs.keys().cloned());
                }
            }
        }

        if let ast::Projection::Variables(projected) = &select.projection {
            // Projection aliases are sequential: a later demanded alias can
            // depend on an earlier alias, so walk backwards and propagate the
            // demand transitively to its materialized source.
            for (index, item) in projected.iter().enumerate().rev() {
                let output = item.alias.as_deref().or({
                    if let ast::Expression::Variable(variable) = &item.expression {
                        Some(variable.as_str())
                    } else {
                        None
                    }
                });
                if output.is_some_and(|output| consumers.contains(output)) {
                    if item.alias.is_some() {
                        annotation.projection[index].rdf_or_native = true;
                    }
                    Self::collect_identity_dependencies(&item.expression, &mut consumers);
                }
            }
        }
        let pattern_demands = consumers
            .into_iter()
            .filter(|name| !materialized_outputs.contains(name))
            .collect::<HashSet<_>>();
        let mut demand = OrdinaryMembershipState::default();
        for name in pattern_demands {
            demand.insert(annotation.variables.intern(&name));
        }

        Self::mark_rdf_or_native_pattern_demands(
            &select.where_clause,
            &mut annotation.where_clause,
            &mut annotation.variables,
            &mut demand,
        )
    }

    fn mark_rdf_or_native_pattern_demands(
        pattern: &ast::GraphPattern,
        annotation: &mut OrdinaryPatternExactAnnotations,
        variables: &mut OrdinaryVariableTable,
        demand: &mut OrdinaryMembershipState,
    ) -> Result<()> {
        match (pattern, annotation) {
            (
                ast::GraphPattern::Group(patterns),
                OrdinaryPatternExactAnnotations::Group(annotations),
            ) if patterns.len() == annotations.len() => {
                for (pattern, annotation) in patterns.iter().zip(annotations).rev() {
                    Self::mark_rdf_or_native_pattern_demands(
                        pattern, annotation, variables, demand,
                    )?;
                }
                Ok(())
            }
            (
                ast::GraphPattern::Union(patterns),
                OrdinaryPatternExactAnnotations::Union(annotations),
            ) if patterns.len() == annotations.len() => {
                let mark = demand.mark();
                let mut keep = HashSet::new();
                for (pattern, annotation) in patterns.iter().zip(annotations) {
                    demand.rollback(mark)?;
                    Self::mark_rdf_or_native_pattern_demands(
                        pattern,
                        &mut annotation.pattern,
                        variables,
                        demand,
                    )?;
                    for id in demand.changed_ids_since(mark) {
                        if demand.contains(id) && annotation.correlated.binary_search(&id).is_ok() {
                            keep.insert(id);
                        }
                    }
                }
                demand.rollback(mark)?;
                for id in keep {
                    demand.insert(id);
                }
                Ok(())
            }
            (
                ast::GraphPattern::Optional(pattern),
                OrdinaryPatternExactAnnotations::Optional {
                    pattern: annotation,
                    correlated,
                },
            ) => {
                let mark = demand.mark();
                Self::mark_rdf_or_native_pattern_demands(pattern, annotation, variables, demand)?;
                let keep = demand
                    .changed_ids_since(mark)
                    .into_iter()
                    .filter(|id| demand.contains(*id) && correlated.binary_search(id).is_ok())
                    .collect::<Vec<_>>();
                demand.rollback(mark)?;
                for id in keep {
                    demand.insert(id);
                }
                Ok(())
            }
            (
                ast::GraphPattern::Minus(pattern),
                OrdinaryPatternExactAnnotations::Minus {
                    pattern: annotation,
                    correlated,
                },
            ) => {
                // MINUS does not export its RHS. Only correlated compatibility
                // keys may create upstream RDF-or-native demand there.
                let mut inner_demand = OrdinaryMembershipState::default();
                for id in correlated.iter().copied() {
                    inner_demand.insert(id);
                }
                Self::mark_rdf_or_native_pattern_demands(
                    pattern,
                    annotation,
                    variables,
                    &mut inner_demand,
                )?;
                for id in correlated.iter().copied() {
                    if inner_demand.contains(id) {
                        demand.insert(id);
                    }
                }
                Ok(())
            }
            (
                ast::GraphPattern::NamedGraph { pattern, .. },
                OrdinaryPatternExactAnnotations::NamedGraph {
                    pattern: annotation,
                    shared_graph,
                },
            ) => {
                if let Some(id) = shared_graph {
                    demand.insert(*id);
                }
                Self::mark_rdf_or_native_pattern_demands(pattern, annotation, variables, demand)
            }
            (
                ast::GraphPattern::Bind { expression, .. },
                OrdinaryPatternExactAnnotations::Bind {
                    variable,
                    rdf_or_native,
                    ..
                },
            ) => {
                *rdf_or_native = demand.remove(*variable);
                if *rdf_or_native {
                    Self::add_ordinary_identity_dependencies(expression, variables, demand);
                }
                Ok(())
            }
            (
                ast::GraphPattern::SubSelect(select),
                OrdinaryPatternExactAnnotations::SubSelect {
                    select: annotation, ..
                },
            ) => {
                let inner_downstream = annotation
                    .exports
                    .iter()
                    .filter(|name| variables.get(name).is_some_and(|id| demand.contains(id)))
                    .cloned()
                    .collect();
                Self::mark_rdf_or_native_select_demands(select, annotation, &inner_downstream)
            }
            (ast::GraphPattern::Basic(_), OrdinaryPatternExactAnnotations::Basic { .. })
            | (
                ast::GraphPattern::InlineData(_),
                OrdinaryPatternExactAnnotations::InlineData { .. },
            )
            | (ast::GraphPattern::Filter(_), OrdinaryPatternExactAnnotations::Filter(_))
            | (ast::GraphPattern::Service { .. }, OrdinaryPatternExactAnnotations::Service) => {
                Ok(())
            }
            _ => Err(Error::Internal(
                "RDF-or-native graph-pattern annotation mismatch".to_string(),
            )),
        }
    }

    fn analyze_ordinary_pattern(
        pattern: &ast::GraphPattern,
        annotation: &mut OrdinaryPatternExactAnnotations,
        variables: &mut OrdinaryVariableTable,
        demand: &mut OrdinaryMembershipState,
    ) -> Result<()> {
        #[cfg(test)]
        ORDINARY_EXACT_PATTERN_CONSUMES.with(|count| count.set(count.get() + 1));
        let annotation_kind = annotation.kind();
        match (pattern, annotation) {
            (ast::GraphPattern::Basic(_), OrdinaryPatternExactAnnotations::Basic { shared })
            | (
                ast::GraphPattern::InlineData(_),
                OrdinaryPatternExactAnnotations::InlineData { shared, .. },
            ) => {
                for id in shared {
                    demand.insert(*id);
                }
                Ok(())
            }
            (
                ast::GraphPattern::Group(patterns),
                OrdinaryPatternExactAnnotations::Group(children),
            ) if patterns.len() == children.len() => {
                for (pattern, child) in patterns.iter().zip(children.iter_mut()) {
                    if let (
                        ast::GraphPattern::Filter(expression),
                        OrdinaryPatternExactAnnotations::Filter(expression_annotation),
                    ) = (pattern, child)
                    {
                        #[cfg(test)]
                        ORDINARY_EXACT_PATTERN_CONSUMES.with(|count| count.set(count.get() + 1));
                        Self::analyze_ordinary_expression(
                            expression,
                            expression_annotation,
                            variables,
                            demand,
                        )?;
                    }
                }
                for (pattern, child) in patterns.iter().zip(children.iter_mut()).rev() {
                    if !matches!(pattern, ast::GraphPattern::Filter(_)) {
                        Self::analyze_ordinary_pattern(pattern, child, variables, demand)?;
                    }
                }
                Ok(())
            }
            (
                ast::GraphPattern::Optional(inner),
                OrdinaryPatternExactAnnotations::Optional {
                    pattern: inner_annotation,
                    correlated,
                },
            ) => {
                let mark = demand.mark();
                Self::analyze_ordinary_pattern(inner, inner_annotation, variables, demand)?;
                let keep = demand
                    .changed_ids_since(mark)
                    .into_iter()
                    .filter(|id| demand.contains(*id) && correlated.binary_search(id).is_ok())
                    .collect::<Vec<_>>();
                demand.rollback(mark)?;
                for id in keep {
                    demand.insert(id);
                }
                Ok(())
            }
            (
                ast::GraphPattern::Union(patterns),
                OrdinaryPatternExactAnnotations::Union(children),
            ) if patterns.len() == children.len() => {
                let mark = demand.mark();
                let mut keep = HashSet::new();
                for (pattern, child) in patterns.iter().zip(children.iter_mut()) {
                    demand.rollback(mark)?;
                    Self::analyze_ordinary_pattern(pattern, &mut child.pattern, variables, demand)?;
                    for id in demand.changed_ids_since(mark) {
                        if demand.contains(id) && child.correlated.binary_search(&id).is_ok() {
                            keep.insert(id);
                        }
                    }
                }
                demand.rollback(mark)?;
                for id in keep {
                    demand.insert(id);
                }
                Ok(())
            }
            (
                ast::GraphPattern::Minus(inner),
                OrdinaryPatternExactAnnotations::Minus {
                    pattern: inner_annotation,
                    correlated,
                },
            ) => {
                let mut inner_demand = OrdinaryMembershipState::default();
                for id in correlated.iter() {
                    inner_demand.insert(*id);
                }
                Self::analyze_ordinary_pattern(
                    inner,
                    inner_annotation,
                    variables,
                    &mut inner_demand,
                )?;
                for id in correlated.iter().copied() {
                    if inner_demand.contains(id) {
                        demand.insert(id);
                    }
                }
                Ok(())
            }
            (
                ast::GraphPattern::NamedGraph { pattern: inner, .. },
                OrdinaryPatternExactAnnotations::NamedGraph {
                    pattern: inner_annotation,
                    shared_graph,
                },
            ) => {
                if let Some(id) = shared_graph {
                    demand.insert(*id);
                }
                Self::analyze_ordinary_pattern(inner, inner_annotation, variables, demand)
            }
            (
                ast::GraphPattern::Filter(expression),
                OrdinaryPatternExactAnnotations::Filter(expression_annotation),
            ) => Self::analyze_ordinary_expression(
                expression,
                expression_annotation,
                variables,
                demand,
            ),
            (
                ast::GraphPattern::Bind { expression, .. },
                OrdinaryPatternExactAnnotations::Bind {
                    variable,
                    exact,
                    expression: expression_annotation,
                    ..
                },
            ) => {
                *exact = demand.remove(*variable);
                if *exact {
                    Self::add_ordinary_identity_dependencies(expression, variables, demand);
                }
                Self::analyze_ordinary_expression(
                    expression,
                    expression_annotation,
                    variables,
                    demand,
                )
            }
            (
                ast::GraphPattern::SubSelect(select),
                OrdinaryPatternExactAnnotations::SubSelect {
                    select: select_annotation,
                    shared,
                },
            ) => {
                for id in shared {
                    demand.insert(*id);
                }
                let downstream = select_annotation
                    .exports
                    .iter()
                    .filter(|name| variables.get(name).is_some_and(|id| demand.contains(id)))
                    .cloned()
                    .collect();
                Self::analyze_ordinary_select(select, select_annotation, &downstream)
            }
            (ast::GraphPattern::Service { .. }, OrdinaryPatternExactAnnotations::Service) => Ok(()),
            _ => Err(Error::Internal(format!(
                "ordinary exact-demand pattern annotation mismatch: expected {}, got {}",
                Self::ordinary_pattern_kind(pattern),
                annotation_kind
            ))),
        }
    }

    fn analyze_ordinary_expression(
        expression: &ast::Expression,
        annotation: &mut OrdinaryExpressionExactAnnotations,
        variables: &mut OrdinaryVariableTable,
        demand: &mut OrdinaryMembershipState,
    ) -> Result<()> {
        #[cfg(test)]
        ORDINARY_EXACT_EXPRESSION_ANALYSES.with(|count| count.set(count.get() + 1));
        use ast::{BuiltInFunction, Expression, FunctionName};
        match (expression, annotation) {
            (Expression::Variable(_), OrdinaryExpressionExactAnnotations::Variable)
            | (Expression::Iri(_), OrdinaryExpressionExactAnnotations::Iri)
            | (Expression::Literal(_), OrdinaryExpressionExactAnnotations::Literal)
            | (Expression::Bound(_), OrdinaryExpressionExactAnnotations::Bound) => Ok(()),
            (
                Expression::Binary { left, right, .. },
                OrdinaryExpressionExactAnnotations::Binary(left_annotation, right_annotation),
            ) => {
                Self::analyze_ordinary_expression(left, left_annotation, variables, demand)?;
                Self::analyze_ordinary_expression(right, right_annotation, variables, demand)
            }
            (
                Expression::Unary { operand, .. },
                OrdinaryExpressionExactAnnotations::Unary(operand_annotation),
            )
            | (
                Expression::Bracketed(operand),
                OrdinaryExpressionExactAnnotations::Bracketed(operand_annotation),
            ) => Self::analyze_ordinary_expression(operand, operand_annotation, variables, demand),
            (
                Expression::FunctionCall {
                    function,
                    arguments,
                },
                OrdinaryExpressionExactAnnotations::FunctionCall(argument_annotations),
            ) if arguments.len() == argument_annotations.len() => {
                match function {
                    FunctionName::BuiltIn(BuiltInFunction::StrDt) => {
                        if let Some(datatype) = arguments.get(1) {
                            Self::add_ordinary_identity_dependencies(datatype, variables, demand);
                        }
                    }
                    FunctionName::BuiltIn(BuiltInFunction::SameTerm) => {
                        for argument in arguments {
                            Self::add_ordinary_identity_dependencies(argument, variables, demand);
                        }
                    }
                    _ => {}
                }
                for (argument, argument_annotation) in
                    arguments.iter().zip(argument_annotations.iter_mut())
                {
                    Self::analyze_ordinary_expression(
                        argument,
                        argument_annotation,
                        variables,
                        demand,
                    )?;
                }
                Ok(())
            }
            (
                Expression::Conditional {
                    condition,
                    then_expression,
                    else_expression,
                },
                OrdinaryExpressionExactAnnotations::Conditional {
                    condition: condition_annotation,
                    then_expression: then_annotation,
                    else_expression: else_annotation,
                },
            ) => {
                Self::analyze_ordinary_expression(
                    condition,
                    condition_annotation,
                    variables,
                    demand,
                )?;
                Self::analyze_ordinary_expression(
                    then_expression,
                    then_annotation,
                    variables,
                    demand,
                )?;
                Self::analyze_ordinary_expression(
                    else_expression,
                    else_annotation,
                    variables,
                    demand,
                )
            }
            (
                Expression::Coalesce(expressions),
                OrdinaryExpressionExactAnnotations::Coalesce(annotations),
            ) if expressions.len() == annotations.len() => {
                for (expression, annotation) in expressions.iter().zip(annotations.iter_mut()) {
                    Self::analyze_ordinary_expression(expression, annotation, variables, demand)?;
                }
                Ok(())
            }
            (
                Expression::Exists(pattern),
                OrdinaryExpressionExactAnnotations::Exists {
                    pattern: pattern_annotation,
                    correlated,
                },
            )
            | (
                Expression::NotExists(pattern),
                OrdinaryExpressionExactAnnotations::NotExists {
                    pattern: pattern_annotation,
                    correlated,
                },
            ) => {
                let mut inner_demand = OrdinaryMembershipState::default();
                for id in correlated.iter() {
                    inner_demand.insert(*id);
                }
                Self::analyze_ordinary_pattern(
                    pattern,
                    pattern_annotation,
                    variables,
                    &mut inner_demand,
                )?;
                for id in correlated.iter() {
                    if inner_demand.contains(*id) {
                        demand.insert(*id);
                    }
                }
                Ok(())
            }
            (
                Expression::In { expression, list },
                OrdinaryExpressionExactAnnotations::In {
                    expression: expression_annotation,
                    list: list_annotations,
                },
            )
            | (
                Expression::NotIn { expression, list },
                OrdinaryExpressionExactAnnotations::NotIn {
                    expression: expression_annotation,
                    list: list_annotations,
                },
            ) if list.len() == list_annotations.len() => {
                Self::analyze_ordinary_expression(
                    expression,
                    expression_annotation,
                    variables,
                    demand,
                )?;
                for (item, item_annotation) in list.iter().zip(list_annotations.iter_mut()) {
                    Self::analyze_ordinary_expression(item, item_annotation, variables, demand)?;
                }
                Ok(())
            }
            (
                Expression::Aggregate(aggregate),
                OrdinaryExpressionExactAnnotations::Aggregate(annotation),
            ) => Self::analyze_ordinary_aggregate(aggregate, annotation, variables, demand),
            _ => Err(Error::Internal(
                "ordinary exact-demand expression annotation mismatch".to_string(),
            )),
        }
    }

    fn analyze_ordinary_aggregate(
        aggregate: &ast::AggregateExpression,
        annotation: &mut OrdinaryAggregateExactAnnotations,
        variables: &mut OrdinaryVariableTable,
        demand: &mut OrdinaryMembershipState,
    ) -> Result<()> {
        // These set functions depend on the exact RDF operand even when no
        // downstream sameTerm/DATATYPE consumer requests the result identity:
        // DISTINCT keys are term-based, numeric promotion uses the datatype,
        // and GROUP_CONCAT must consume RDF lexical forms. Propagate that
        // requirement through scans, VALUES, BIND, and subselect boundaries.
        match aggregate {
            ast::AggregateExpression::Count {
                distinct: true,
                expression: None,
            } => {}
            ast::AggregateExpression::Count {
                distinct: true,
                expression: Some(expression),
            }
            | ast::AggregateExpression::Sum { expression, .. }
            | ast::AggregateExpression::Average { expression, .. }
            | ast::AggregateExpression::GroupConcat { expression, .. } => {
                Self::add_ordinary_identity_dependencies(expression, variables, demand);
            }
            _ => {}
        }
        match (aggregate, annotation) {
            (
                ast::AggregateExpression::Count { expression, .. },
                OrdinaryAggregateExactAnnotations::Count(annotation),
            ) => match (expression, annotation) {
                (Some(expression), Some(annotation)) => {
                    Self::analyze_ordinary_expression(expression, annotation, variables, demand)
                }
                (None, None) => Ok(()),
                _ => Err(Error::Internal(
                    "ordinary exact-demand aggregate annotation mismatch".to_string(),
                )),
            },
            (
                ast::AggregateExpression::Sum { expression, .. },
                OrdinaryAggregateExactAnnotations::Sum(annotation),
            )
            | (
                ast::AggregateExpression::Average { expression, .. },
                OrdinaryAggregateExactAnnotations::Average(annotation),
            )
            | (
                ast::AggregateExpression::Minimum { expression },
                OrdinaryAggregateExactAnnotations::Minimum(annotation),
            )
            | (
                ast::AggregateExpression::Maximum { expression },
                OrdinaryAggregateExactAnnotations::Maximum(annotation),
            )
            | (
                ast::AggregateExpression::Sample { expression },
                OrdinaryAggregateExactAnnotations::Sample(annotation),
            )
            | (
                ast::AggregateExpression::GroupConcat { expression, .. },
                OrdinaryAggregateExactAnnotations::GroupConcat(annotation),
            ) => Self::analyze_ordinary_expression(expression, annotation, variables, demand),
            _ => Err(Error::Internal(
                "ordinary exact-demand aggregate annotation mismatch".to_string(),
            )),
        }
    }

    fn ordinary_pattern_kind(pattern: &ast::GraphPattern) -> &'static str {
        match pattern {
            ast::GraphPattern::Basic(_) => "Basic",
            ast::GraphPattern::Group(_) => "Group",
            ast::GraphPattern::Optional(_) => "Optional",
            ast::GraphPattern::Union(_) => "Union",
            ast::GraphPattern::NamedGraph { .. } => "NamedGraph",
            ast::GraphPattern::Minus(_) => "Minus",
            ast::GraphPattern::Filter(_) => "Filter",
            ast::GraphPattern::Bind { .. } => "Bind",
            ast::GraphPattern::InlineData(_) => "InlineData",
            ast::GraphPattern::SubSelect(_) => "SubSelect",
            ast::GraphPattern::Service { .. } => "Service",
        }
    }

    fn validate_ordinary_select_annotations(
        select: &ast::SelectQuery,
        annotation: &OrdinarySelectExactAnnotations,
        path: &str,
    ) -> Result<()> {
        let projected = match &select.projection {
            ast::Projection::Wildcard => &[][..],
            ast::Projection::Variables(projected) => projected.as_slice(),
        };
        if projected.len() != annotation.projection.len() {
            return Err(Error::Internal(format!(
                "ordinary exact-demand annotation mismatch at {path}.projection: expected {} nodes, got {}",
                projected.len(),
                annotation.projection.len()
            )));
        }
        for (index, (projected, projected_annotation)) in projected
            .iter()
            .zip(annotation.projection.iter())
            .enumerate()
        {
            Self::validate_ordinary_expression_annotations(
                &projected.expression,
                &projected_annotation.expression,
                &format!("{path}.projection[{index}]"),
            )?;
        }
        match (&select.solution_modifiers.group_by, &annotation.group_by) {
            (None, None) => {}
            (Some(group_by), Some(group_annotations))
                if group_by.len() == group_annotations.len() =>
            {
                for (index, (condition, condition_annotation)) in
                    group_by.iter().zip(group_annotations).enumerate()
                {
                    match (condition, condition_annotation) {
                        (
                            ast::GroupCondition::Variable(_),
                            OrdinaryGroupExactAnnotation::Variable,
                        ) => {}
                        (
                            ast::GroupCondition::Expression { expression, .. },
                            OrdinaryGroupExactAnnotation::Expression {
                                expression: expression_annotation,
                                ..
                            },
                        )
                        | (
                            ast::GroupCondition::BuiltInCall(expression),
                            OrdinaryGroupExactAnnotation::BuiltInCall(expression_annotation),
                        ) => Self::validate_ordinary_expression_annotations(
                            expression,
                            expression_annotation,
                            &format!("{path}.group_by[{index}]"),
                        )?,
                        _ => {
                            return Err(Error::Internal(format!(
                                "ordinary exact-demand annotation mismatch at {path}.group_by[{index}]"
                            )));
                        }
                    }
                }
            }
            _ => {
                return Err(Error::Internal(format!(
                    "ordinary exact-demand annotation mismatch at {path}.group_by"
                )));
            }
        }
        match (&select.solution_modifiers.having, &annotation.having) {
            (None, None) => {}
            (Some(expression), Some(expression_annotation)) => {
                Self::validate_ordinary_expression_annotations(
                    expression,
                    expression_annotation,
                    &format!("{path}.having"),
                )?;
            }
            _ => {
                return Err(Error::Internal(format!(
                    "ordinary exact-demand annotation mismatch at {path}.having"
                )));
            }
        }
        match (&select.solution_modifiers.order_by, &annotation.order_by) {
            (None, None) => {}
            (Some(order_by), Some(order_annotations))
                if order_by.len() == order_annotations.len() =>
            {
                for (index, (condition, expression_annotation)) in
                    order_by.iter().zip(order_annotations).enumerate()
                {
                    Self::validate_ordinary_expression_annotations(
                        &condition.expression,
                        expression_annotation,
                        &format!("{path}.order_by[{index}]"),
                    )?;
                }
            }
            _ => {
                return Err(Error::Internal(format!(
                    "ordinary exact-demand annotation mismatch at {path}.order_by"
                )));
            }
        }
        Self::validate_ordinary_pattern_annotations(
            &select.where_clause,
            &annotation.where_clause,
            &format!("{path}.where"),
        )
    }

    fn validate_ordinary_pattern_annotations(
        pattern: &ast::GraphPattern,
        annotation: &OrdinaryPatternExactAnnotations,
        path: &str,
    ) -> Result<()> {
        match (pattern, annotation) {
            (ast::GraphPattern::Basic(_), OrdinaryPatternExactAnnotations::Basic { .. })
            | (
                ast::GraphPattern::InlineData(_),
                OrdinaryPatternExactAnnotations::InlineData { .. },
            )
            | (ast::GraphPattern::Service { .. }, OrdinaryPatternExactAnnotations::Service) => {
                Ok(())
            }
            (
                ast::GraphPattern::Group(patterns),
                OrdinaryPatternExactAnnotations::Group(children),
            ) if patterns.len() == children.len() => {
                for (index, (pattern, child)) in patterns.iter().zip(children).enumerate() {
                    Self::validate_ordinary_pattern_annotations(
                        pattern,
                        child,
                        &format!("{path}[{index}]"),
                    )?;
                }
                Ok(())
            }
            (
                ast::GraphPattern::Union(patterns),
                OrdinaryPatternExactAnnotations::Union(children),
            ) if patterns.len() == children.len() => {
                for (index, (pattern, child)) in patterns.iter().zip(children).enumerate() {
                    Self::validate_ordinary_pattern_annotations(
                        pattern,
                        &child.pattern,
                        &format!("{path}.union[{index}]"),
                    )?;
                }
                Ok(())
            }
            (
                ast::GraphPattern::Optional(inner),
                OrdinaryPatternExactAnnotations::Optional {
                    pattern: inner_annotation,
                    ..
                },
            )
            | (
                ast::GraphPattern::Minus(inner),
                OrdinaryPatternExactAnnotations::Minus {
                    pattern: inner_annotation,
                    ..
                },
            ) => Self::validate_ordinary_pattern_annotations(inner, inner_annotation, path),
            (
                ast::GraphPattern::NamedGraph { pattern: inner, .. },
                OrdinaryPatternExactAnnotations::NamedGraph {
                    pattern: inner_annotation,
                    ..
                },
            ) => Self::validate_ordinary_pattern_annotations(inner, inner_annotation, path),
            (
                ast::GraphPattern::Filter(expression),
                OrdinaryPatternExactAnnotations::Filter(expression_annotation),
            ) => Self::validate_ordinary_expression_annotations(
                expression,
                expression_annotation,
                &format!("{path}.filter"),
            ),
            (
                ast::GraphPattern::Bind { expression, .. },
                OrdinaryPatternExactAnnotations::Bind {
                    expression: expression_annotation,
                    ..
                },
            ) => Self::validate_ordinary_expression_annotations(
                expression,
                expression_annotation,
                &format!("{path}.bind"),
            ),
            (
                ast::GraphPattern::SubSelect(select),
                OrdinaryPatternExactAnnotations::SubSelect {
                    select: select_annotation,
                    ..
                },
            ) => Self::validate_ordinary_select_annotations(
                select,
                select_annotation,
                &format!("{path}.subselect"),
            ),
            _ => Err(Error::Internal(format!(
                "ordinary exact-demand annotation mismatch at {path}: expected {}, got {}",
                Self::ordinary_pattern_kind(pattern),
                annotation.kind()
            ))),
        }
    }

    fn validate_ordinary_expression_annotations(
        expression: &ast::Expression,
        annotation: &OrdinaryExpressionExactAnnotations,
        path: &str,
    ) -> Result<()> {
        use ast::Expression;
        match (expression, annotation) {
            (Expression::Variable(_), OrdinaryExpressionExactAnnotations::Variable)
            | (Expression::Iri(_), OrdinaryExpressionExactAnnotations::Iri)
            | (Expression::Literal(_), OrdinaryExpressionExactAnnotations::Literal)
            | (Expression::Bound(_), OrdinaryExpressionExactAnnotations::Bound) => Ok(()),
            (
                Expression::Binary { left, right, .. },
                OrdinaryExpressionExactAnnotations::Binary(left_annotation, right_annotation),
            ) => {
                Self::validate_ordinary_expression_annotations(
                    left,
                    left_annotation,
                    &format!("{path}.left"),
                )?;
                Self::validate_ordinary_expression_annotations(
                    right,
                    right_annotation,
                    &format!("{path}.right"),
                )
            }
            (
                Expression::Unary { operand, .. },
                OrdinaryExpressionExactAnnotations::Unary(operand_annotation),
            )
            | (
                Expression::Bracketed(operand),
                OrdinaryExpressionExactAnnotations::Bracketed(operand_annotation),
            ) => Self::validate_ordinary_expression_annotations(
                operand,
                operand_annotation,
                &format!("{path}.operand"),
            ),
            (
                Expression::FunctionCall { arguments, .. },
                OrdinaryExpressionExactAnnotations::FunctionCall(argument_annotations),
            ) if arguments.len() == argument_annotations.len() => {
                for (index, (argument, argument_annotation)) in
                    arguments.iter().zip(argument_annotations).enumerate()
                {
                    Self::validate_ordinary_expression_annotations(
                        argument,
                        argument_annotation,
                        &format!("{path}.argument[{index}]"),
                    )?;
                }
                Ok(())
            }
            (
                Expression::Conditional {
                    condition,
                    then_expression,
                    else_expression,
                },
                OrdinaryExpressionExactAnnotations::Conditional {
                    condition: condition_annotation,
                    then_expression: then_annotation,
                    else_expression: else_annotation,
                },
            ) => {
                Self::validate_ordinary_expression_annotations(
                    condition,
                    condition_annotation,
                    &format!("{path}.condition"),
                )?;
                Self::validate_ordinary_expression_annotations(
                    then_expression,
                    then_annotation,
                    &format!("{path}.then"),
                )?;
                Self::validate_ordinary_expression_annotations(
                    else_expression,
                    else_annotation,
                    &format!("{path}.else"),
                )
            }
            (
                Expression::Coalesce(expressions),
                OrdinaryExpressionExactAnnotations::Coalesce(annotations),
            ) if expressions.len() == annotations.len() => {
                for (index, (expression, annotation)) in
                    expressions.iter().zip(annotations).enumerate()
                {
                    Self::validate_ordinary_expression_annotations(
                        expression,
                        annotation,
                        &format!("{path}.coalesce[{index}]"),
                    )?;
                }
                Ok(())
            }
            (
                Expression::Exists(pattern),
                OrdinaryExpressionExactAnnotations::Exists {
                    pattern: pattern_annotation,
                    ..
                },
            )
            | (
                Expression::NotExists(pattern),
                OrdinaryExpressionExactAnnotations::NotExists {
                    pattern: pattern_annotation,
                    ..
                },
            ) => Self::validate_ordinary_pattern_annotations(
                pattern,
                pattern_annotation,
                &format!("{path}.exists"),
            ),
            (
                Expression::In { expression, list },
                OrdinaryExpressionExactAnnotations::In {
                    expression: expression_annotation,
                    list: list_annotations,
                },
            )
            | (
                Expression::NotIn { expression, list },
                OrdinaryExpressionExactAnnotations::NotIn {
                    expression: expression_annotation,
                    list: list_annotations,
                },
            ) if list.len() == list_annotations.len() => {
                Self::validate_ordinary_expression_annotations(
                    expression,
                    expression_annotation,
                    &format!("{path}.head"),
                )?;
                for (index, (item, item_annotation)) in
                    list.iter().zip(list_annotations).enumerate()
                {
                    Self::validate_ordinary_expression_annotations(
                        item,
                        item_annotation,
                        &format!("{path}.list[{index}]"),
                    )?;
                }
                Ok(())
            }
            (
                Expression::Aggregate(aggregate),
                OrdinaryExpressionExactAnnotations::Aggregate(aggregate_annotation),
            ) => {
                Self::validate_ordinary_aggregate_annotations(aggregate, aggregate_annotation, path)
            }
            _ => Err(Error::Internal(format!(
                "ordinary exact-demand expression annotation mismatch at {path}"
            ))),
        }
    }

    fn validate_ordinary_aggregate_annotations(
        aggregate: &ast::AggregateExpression,
        annotation: &OrdinaryAggregateExactAnnotations,
        path: &str,
    ) -> Result<()> {
        match (aggregate, annotation) {
            (
                ast::AggregateExpression::Count { expression, .. },
                OrdinaryAggregateExactAnnotations::Count(expression_annotation),
            ) => match (expression, expression_annotation) {
                (None, None) => Ok(()),
                (Some(expression), Some(expression_annotation)) => {
                    Self::validate_ordinary_expression_annotations(
                        expression,
                        expression_annotation,
                        &format!("{path}.count"),
                    )
                }
                _ => Err(Error::Internal(format!(
                    "ordinary exact-demand aggregate annotation mismatch at {path}"
                ))),
            },
            (
                ast::AggregateExpression::Sum { expression, .. },
                OrdinaryAggregateExactAnnotations::Sum(expression_annotation),
            )
            | (
                ast::AggregateExpression::Average { expression, .. },
                OrdinaryAggregateExactAnnotations::Average(expression_annotation),
            )
            | (
                ast::AggregateExpression::Minimum { expression },
                OrdinaryAggregateExactAnnotations::Minimum(expression_annotation),
            )
            | (
                ast::AggregateExpression::Maximum { expression },
                OrdinaryAggregateExactAnnotations::Maximum(expression_annotation),
            )
            | (
                ast::AggregateExpression::Sample { expression },
                OrdinaryAggregateExactAnnotations::Sample(expression_annotation),
            )
            | (
                ast::AggregateExpression::GroupConcat { expression, .. },
                OrdinaryAggregateExactAnnotations::GroupConcat(expression_annotation),
            ) => Self::validate_ordinary_expression_annotations(
                expression,
                expression_annotation,
                path,
            ),
            _ => Err(Error::Internal(format!(
                "ordinary exact-demand aggregate annotation mismatch at {path}"
            ))),
        }
    }

    fn select_exact_dependencies(
        select: &ast::SelectQuery,
        parent_exact: Option<&HashSet<String>>,
        parent_control: Option<&HashSet<String>>,
    ) -> (HashSet<String>, HashSet<String>) {
        let mut required = HashSet::new();
        let mut control = HashSet::new();
        let mut available = HashSet::new();
        Self::collect_pattern_output_variables(&select.where_clause, &mut available);
        match &select.projection {
            ast::Projection::Wildcard => {
                if let Some(parent) = parent_exact {
                    required.extend(parent.iter().cloned());
                }
                if let Some(parent) = parent_control {
                    control.extend(parent.iter().cloned());
                }
            }
            ast::Projection::Variables(projections) => {
                for projection in projections {
                    let output = projection.alias.as_deref().or({
                        if let ast::Expression::Variable(variable) = &projection.expression {
                            Some(variable.as_str())
                        } else {
                            None
                        }
                    });
                    if let Some(output) = output {
                        if parent_exact.is_some_and(|set| set.contains(output)) {
                            required.insert(output.to_string());
                        }
                        if parent_control.is_some_and(|set| set.contains(output)) {
                            control.insert(output.to_string());
                        }
                    }
                }
            }
        }

        if let ast::Projection::Variables(projections) = &select.projection {
            for projection in projections {
                let ast::Expression::Aggregate(aggregate) = &projection.expression else {
                    continue;
                };
                match aggregate {
                    ast::AggregateExpression::Count {
                        distinct: true,
                        expression: None,
                    } => {}
                    ast::AggregateExpression::Count {
                        distinct: true,
                        expression: Some(expression),
                    }
                    | ast::AggregateExpression::Sum { expression, .. }
                    | ast::AggregateExpression::Average { expression, .. }
                    | ast::AggregateExpression::GroupConcat { expression, .. } => {
                        Self::collect_identity_dependencies(expression, &mut required);
                    }
                    _ => {}
                }
            }
        }

        if let Some(group_by) = &select.solution_modifiers.group_by {
            for condition in group_by {
                match condition {
                    ast::GroupCondition::Expression { expression, .. }
                    | ast::GroupCondition::BuiltInCall(expression) => {
                        // SPARQL grouping compares RDF term identity. Constructors
                        // trace only the inputs needed to reconstruct their result,
                        // while native extension values retain their own identity.
                        Self::collect_identity_dependencies(expression, &mut required);
                        Self::collect_selector_dependencies(
                            expression,
                            &mut required,
                            &mut control,
                            &available,
                        );
                    }
                    ast::GroupCondition::Variable(variable) => {
                        required.insert(variable.clone());
                        control.insert(variable.clone());
                    }
                }
            }
        }
        if let Some(having) = &select.solution_modifiers.having {
            Self::collect_selector_dependencies(having, &mut required, &mut control, &available);
            Self::collect_aggregate_alias_dependencies(having, &select.projection, &mut required);
        }
        if let Some(order_by) = &select.solution_modifiers.order_by {
            let order_controls_selection = select.solution_modifiers.limit.is_some()
                || select.solution_modifiers.offset.is_some();
            for condition in order_by {
                if order_controls_selection {
                    Self::collect_selector_dependencies(
                        &condition.expression,
                        &mut required,
                        &mut control,
                        &available,
                    );
                }
                let rewritten =
                    Self::rewrite_having_aggregates(&condition.expression, &select.projection);
                Self::collect_identity_dependencies(&rewritten, &mut required);
                Self::collect_aggregate_alias_dependencies(
                    &condition.expression,
                    &select.projection,
                    &mut required,
                );
            }
        }

        loop {
            let previous_len = required.len() + control.len();
            if let ast::Projection::Variables(projections) = &select.projection {
                for projection in projections {
                    let output = projection.alias.as_deref().or({
                        if let ast::Expression::Variable(variable) = &projection.expression {
                            Some(variable.as_str())
                        } else {
                            None
                        }
                    });
                    if output.is_some_and(|output| required.contains(output)) {
                        Self::collect_identity_dependencies(&projection.expression, &mut required);
                        Self::collect_term_semantic_dependencies(
                            &projection.expression,
                            &mut required,
                        );
                        Self::collect_identity_control_dependencies(
                            &projection.expression,
                            &mut required,
                            &mut control,
                            &available,
                        );
                    }
                    if output.is_some_and(|output| control.contains(output)) {
                        Self::collect_selector_dependencies(
                            &projection.expression,
                            &mut required,
                            &mut control,
                            &available,
                        );
                    }
                }
            }
            if required.len() + control.len() == previous_len {
                break;
            }
        }
        (required, control)
    }

    fn aggregate_projection_alias<'a>(
        aggregate: &ast::AggregateExpression,
        projection: &'a ast::Projection,
    ) -> Option<&'a str> {
        let ast::Projection::Variables(projections) = projection else {
            return None;
        };
        projections.iter().find_map(|projected| {
            matches!(
                &projected.expression,
                ast::Expression::Aggregate(candidate) if candidate == aggregate
            )
            .then(|| projected.alias.as_deref())
            .flatten()
        })
    }

    fn collect_aggregate_alias_dependencies(
        expression: &ast::Expression,
        projection: &ast::Projection,
        required: &mut HashSet<String>,
    ) {
        use ast::Expression;

        match expression {
            Expression::Aggregate(aggregate) => {
                if let Some(alias) = Self::aggregate_projection_alias(aggregate, projection) {
                    required.insert(alias.to_string());
                }
            }
            Expression::Binary { left, right, .. } => {
                Self::collect_aggregate_alias_dependencies(left, projection, required);
                Self::collect_aggregate_alias_dependencies(right, projection, required);
            }
            Expression::Unary { operand, .. } | Expression::Bracketed(operand) => {
                Self::collect_aggregate_alias_dependencies(operand, projection, required);
            }
            Expression::FunctionCall { arguments, .. } | Expression::Coalesce(arguments) => {
                for argument in arguments {
                    Self::collect_aggregate_alias_dependencies(argument, projection, required);
                }
            }
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                Self::collect_aggregate_alias_dependencies(condition, projection, required);
                Self::collect_aggregate_alias_dependencies(then_expression, projection, required);
                Self::collect_aggregate_alias_dependencies(else_expression, projection, required);
            }
            Expression::In { expression, list } | Expression::NotIn { expression, list } => {
                Self::collect_aggregate_alias_dependencies(expression, projection, required);
                for item in list {
                    Self::collect_aggregate_alias_dependencies(item, projection, required);
                }
            }
            Expression::Variable(_)
            | Expression::Iri(_)
            | Expression::Literal(_)
            | Expression::Bound(_)
            | Expression::Exists(_)
            | Expression::NotExists(_) => {}
        }
    }

    /// Expands only the current graph-pattern scope. Nested UNION branches,
    /// OPTIONALs, and subselects derive their own child scopes when translated,
    /// preventing a selector in one branch from rejecting an unrelated helper
    /// in a sibling branch.
    fn expand_pattern_scope_dependencies(
        pattern: &ast::GraphPattern,
        required: &mut HashSet<String>,
        control: &mut HashSet<String>,
    ) {
        loop {
            let previous_len = required.len() + control.len();
            Self::collect_pattern_scope_dependencies(pattern, required, control);
            if required.len() + control.len() == previous_len {
                break;
            }
        }
    }

    fn collect_pattern_scope_dependencies(
        pattern: &ast::GraphPattern,
        required: &mut HashSet<String>,
        control: &mut HashSet<String>,
    ) {
        match pattern {
            ast::GraphPattern::Bind {
                expression,
                variable,
            } if required.contains(variable) => {
                Self::collect_identity_dependencies(expression, required);
                Self::collect_term_semantic_dependencies(expression, required);
                let mut available = HashSet::new();
                available.extend(required.iter().cloned());
                available.extend(control.iter().cloned());
                Self::collect_identity_control_dependencies(
                    expression, required, control, &available,
                );
            }
            ast::GraphPattern::Filter(expression) => {
                let mut available = HashSet::new();
                available.extend(required.iter().cloned());
                available.extend(control.iter().cloned());
                Self::collect_selector_dependencies(expression, required, control, &available);
            }
            ast::GraphPattern::Group(patterns) => {
                let mut available = HashSet::new();
                for pattern in patterns {
                    Self::collect_pattern_output_variables(pattern, &mut available);
                }

                let mut bound = HashSet::new();
                for pattern in patterns {
                    match pattern {
                        ast::GraphPattern::Bind {
                            expression,
                            variable,
                        } => {
                            if required.contains(variable) {
                                Self::collect_identity_dependencies(expression, required);
                                Self::collect_term_semantic_dependencies(expression, required);
                                Self::collect_identity_control_dependencies(
                                    expression, required, control, &available,
                                );
                            }
                            if control.contains(variable) {
                                Self::collect_selector_dependencies(
                                    expression, required, control, &available,
                                );
                            }
                            // BIND extends a solution mapping; it is not a
                            // compatibility join against a sibling binding.
                            bound.insert(variable.clone());
                            continue;
                        }
                        ast::GraphPattern::Filter(expression) => {
                            Self::collect_selector_dependencies(
                                expression, required, control, &available,
                            );
                        }
                        ast::GraphPattern::Minus(inner) => {
                            let mut joined = HashSet::new();
                            Self::collect_pattern_output_variables(inner, &mut joined);
                            required.extend(bound.intersection(&joined).cloned());
                            continue;
                        }
                        ast::GraphPattern::InlineData(data) => {
                            let mut joined = HashSet::new();
                            for (index, variable) in data.variables.iter().enumerate() {
                                if data
                                    .values
                                    .iter()
                                    .any(|row| row.get(index).is_some_and(Option::is_some))
                                {
                                    joined.insert(variable.clone());
                                }
                            }
                            required.extend(bound.intersection(&joined).cloned());
                            bound.extend(joined);
                            continue;
                        }
                        _ => {}
                    }
                    let mut joined = HashSet::new();
                    Self::collect_pattern_output_variables(pattern, &mut joined);
                    required.extend(bound.intersection(&joined).cloned());
                    bound.extend(joined);
                }
            }
            _ => {}
        }
    }

    fn collect_pattern_output_variables(
        pattern: &ast::GraphPattern,
        variables: &mut HashSet<String>,
    ) {
        match pattern {
            ast::GraphPattern::Basic(triples) => {
                for triple in triples {
                    for term in [&triple.subject, &triple.object] {
                        if let ast::TripleTerm::Variable(variable) = term {
                            variables.insert(variable.clone());
                        }
                    }
                    if let ast::PropertyPath::Variable(variable) = &triple.predicate {
                        variables.insert(variable.clone());
                    }
                }
            }
            ast::GraphPattern::Bind { variable, .. } => {
                variables.insert(variable.clone());
            }
            ast::GraphPattern::InlineData(data) => {
                variables.extend(data.variables.iter().cloned());
            }
            ast::GraphPattern::Group(patterns) | ast::GraphPattern::Union(patterns) => {
                for pattern in patterns {
                    Self::collect_pattern_output_variables(pattern, variables);
                }
            }
            ast::GraphPattern::Optional(pattern) | ast::GraphPattern::Service { pattern, .. } => {
                Self::collect_pattern_output_variables(pattern, variables);
            }
            // MINUS can inspect shared variables but never exposes new ones.
            ast::GraphPattern::Minus(_) => {}
            ast::GraphPattern::NamedGraph { graph, pattern } => {
                if let ast::VariableOrIri::Variable(variable) = graph {
                    variables.insert(variable.clone());
                }
                Self::collect_pattern_output_variables(pattern, variables);
            }
            ast::GraphPattern::SubSelect(select) => {
                Self::collect_subselect_output_variables(select, variables);
            }
            _ => {}
        }
    }

    fn collect_subselect_output_variables(
        select: &ast::SelectQuery,
        variables: &mut HashSet<String>,
    ) {
        match &select.projection {
            ast::Projection::Wildcard => {
                if Self::select_is_grouped(select) {
                    // Aggregation removes every pre-group binding except named
                    // group outputs. A wildcard must not make the WHERE
                    // variables appear correlated or joinable outside the
                    // subselect when they are absent from its physical schema.
                    if let Some(group_by) = &select.solution_modifiers.group_by {
                        Self::collect_group_output_variables(group_by, variables);
                    }
                } else {
                    Self::collect_pattern_output_variables(&select.where_clause, variables);
                }
            }
            ast::Projection::Variables(projections) => {
                for projection in projections {
                    if let Some(alias) = &projection.alias {
                        variables.insert(alias.clone());
                    } else if let ast::Expression::Variable(variable) = &projection.expression {
                        variables.insert(variable.clone());
                    }
                }
            }
        }
    }

    fn collect_group_output_variables(
        group_by: &[ast::GroupCondition],
        variables: &mut HashSet<String>,
    ) {
        for condition in group_by {
            match condition {
                ast::GroupCondition::Variable(variable) => {
                    variables.insert(variable.clone());
                }
                ast::GroupCondition::Expression {
                    alias: Some(alias), ..
                } => {
                    variables.insert(alias.clone());
                }
                ast::GroupCondition::Expression {
                    expression,
                    alias: None,
                } => {
                    if let Some(variable) = Self::unwrapped_variable_expression(expression) {
                        variables.insert(variable.to_string());
                    }
                }
                ast::GroupCondition::BuiltInCall(_) => {}
            }
        }
    }

    fn collect_expression_variables(expression: &ast::Expression, variables: &mut HashSet<String>) {
        use ast::Expression;

        match expression {
            Expression::Variable(variable) | Expression::Bound(variable) => {
                variables.insert(variable.clone());
            }
            Expression::Binary { left, right, .. } => {
                Self::collect_expression_variables(left, variables);
                Self::collect_expression_variables(right, variables);
            }
            Expression::Unary { operand, .. } | Expression::Bracketed(operand) => {
                Self::collect_expression_variables(operand, variables);
            }
            Expression::FunctionCall { arguments, .. } | Expression::Coalesce(arguments) => {
                for argument in arguments {
                    Self::collect_expression_variables(argument, variables);
                }
            }
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                Self::collect_expression_variables(condition, variables);
                Self::collect_expression_variables(then_expression, variables);
                Self::collect_expression_variables(else_expression, variables);
            }
            Expression::In { expression, list } | Expression::NotIn { expression, list } => {
                Self::collect_expression_variables(expression, variables);
                for item in list {
                    Self::collect_expression_variables(item, variables);
                }
            }
            Expression::Aggregate(aggregate) => match aggregate {
                ast::AggregateExpression::Count { expression, .. } => {
                    if let Some(expression) = expression {
                        Self::collect_expression_variables(expression, variables);
                    }
                }
                ast::AggregateExpression::Sum { expression, .. }
                | ast::AggregateExpression::Average { expression, .. }
                | ast::AggregateExpression::Minimum { expression }
                | ast::AggregateExpression::Maximum { expression }
                | ast::AggregateExpression::Sample { expression }
                | ast::AggregateExpression::GroupConcat { expression, .. } => {
                    Self::collect_expression_variables(expression, variables);
                }
            },
            Expression::Iri(_)
            | Expression::Literal(_)
            | Expression::Exists(_)
            | Expression::NotExists(_) => {}
        }
    }

    /// Collects user-visible dependencies evaluated outside aggregate
    /// operands in the current SELECT scope. Aggregate inputs belong to the
    /// pre-group solution and EXISTS owns a separate graph-pattern scope.
    fn collect_non_aggregate_expression_variables(
        expression: &ast::Expression,
        variables: &mut HashSet<String>,
    ) {
        use ast::Expression;

        match expression {
            Expression::Variable(variable) | Expression::Bound(variable) => {
                variables.insert(variable.clone());
            }
            Expression::Binary { left, right, .. } => {
                Self::collect_non_aggregate_expression_variables(left, variables);
                Self::collect_non_aggregate_expression_variables(right, variables);
            }
            Expression::Unary { operand, .. } | Expression::Bracketed(operand) => {
                Self::collect_non_aggregate_expression_variables(operand, variables);
            }
            Expression::FunctionCall { arguments, .. } | Expression::Coalesce(arguments) => {
                for argument in arguments {
                    Self::collect_non_aggregate_expression_variables(argument, variables);
                }
            }
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                Self::collect_non_aggregate_expression_variables(condition, variables);
                Self::collect_non_aggregate_expression_variables(then_expression, variables);
                Self::collect_non_aggregate_expression_variables(else_expression, variables);
            }
            Expression::In { expression, list } | Expression::NotIn { expression, list } => {
                Self::collect_non_aggregate_expression_variables(expression, variables);
                for item in list {
                    Self::collect_non_aggregate_expression_variables(item, variables);
                }
            }
            Expression::Aggregate(_)
            | Expression::Iri(_)
            | Expression::Literal(_)
            | Expression::Exists(_)
            | Expression::NotExists(_) => {}
        }
    }

    fn collect_selector_dependencies(
        expression: &ast::Expression,
        required: &mut HashSet<String>,
        control: &mut HashSet<String>,
        available: &HashSet<String>,
    ) {
        Self::collect_term_semantic_dependencies(expression, required);
        Self::collect_expression_variables(expression, control);
        Self::collect_exists_dependencies(expression, required, control, available);
    }

    fn collect_exists_dependencies(
        expression: &ast::Expression,
        required: &mut HashSet<String>,
        control: &mut HashSet<String>,
        available: &HashSet<String>,
    ) {
        use ast::Expression;

        match expression {
            Expression::Exists(pattern) | Expression::NotExists(pattern) => {
                Self::collect_correlated_pattern_dependencies(
                    pattern, available, required, control,
                );
            }
            Expression::Binary { left, right, .. } => {
                Self::collect_exists_dependencies(left, required, control, available);
                Self::collect_exists_dependencies(right, required, control, available);
            }
            Expression::Unary { operand, .. } | Expression::Bracketed(operand) => {
                Self::collect_exists_dependencies(operand, required, control, available);
            }
            Expression::FunctionCall { arguments, .. } | Expression::Coalesce(arguments) => {
                for argument in arguments {
                    Self::collect_exists_dependencies(argument, required, control, available);
                }
            }
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                Self::collect_exists_dependencies(condition, required, control, available);
                Self::collect_exists_dependencies(then_expression, required, control, available);
                Self::collect_exists_dependencies(else_expression, required, control, available);
            }
            Expression::In { expression, list } | Expression::NotIn { expression, list } => {
                Self::collect_exists_dependencies(expression, required, control, available);
                for item in list {
                    Self::collect_exists_dependencies(item, required, control, available);
                }
            }
            Expression::Aggregate(aggregate) => match aggregate {
                ast::AggregateExpression::Count { expression, .. } => {
                    if let Some(expression) = expression {
                        Self::collect_exists_dependencies(expression, required, control, available);
                    }
                }
                ast::AggregateExpression::Sum { expression, .. }
                | ast::AggregateExpression::Average { expression, .. }
                | ast::AggregateExpression::Minimum { expression }
                | ast::AggregateExpression::Maximum { expression }
                | ast::AggregateExpression::Sample { expression }
                | ast::AggregateExpression::GroupConcat { expression, .. } => {
                    Self::collect_exists_dependencies(expression, required, control, available);
                }
            },
            Expression::Variable(_)
            | Expression::Iri(_)
            | Expression::Literal(_)
            | Expression::Bound(_) => {}
        }
    }

    fn collect_correlated_pattern_dependencies(
        pattern: &ast::GraphPattern,
        available: &HashSet<String>,
        required: &mut HashSet<String>,
        control: &mut HashSet<String>,
    ) {
        Self::collect_correlated_pattern_dependencies_inner(
            pattern,
            available,
            &HashSet::new(),
            &HashSet::new(),
            required,
            control,
        );
    }

    fn collect_correlated_pattern_dependencies_inner(
        pattern: &ast::GraphPattern,
        available: &HashSet<String>,
        inherited_required: &HashSet<String>,
        inherited_control: &HashSet<String>,
        required: &mut HashSet<String>,
        control: &mut HashSet<String>,
    ) {
        let mut local_required = inherited_required.clone();
        let mut local_control = inherited_control.clone();
        let mut compatibility = HashSet::new();
        Self::collect_pattern_compatibility_variables(pattern, &mut compatibility);
        local_required.extend(compatibility.intersection(available).cloned());
        Self::expand_pattern_scope_dependencies(pattern, &mut local_required, &mut local_control);
        required.extend(local_required.intersection(available).cloned());
        control.extend(local_control.intersection(available).cloned());

        match pattern {
            ast::GraphPattern::Group(patterns) | ast::GraphPattern::Union(patterns) => {
                for child in patterns {
                    Self::collect_correlated_pattern_dependencies_inner(
                        child,
                        available,
                        &local_required,
                        &local_control,
                        required,
                        control,
                    );
                }
            }
            ast::GraphPattern::Optional(child)
            | ast::GraphPattern::Minus(child)
            | ast::GraphPattern::NamedGraph { pattern: child, .. }
            | ast::GraphPattern::Service { pattern: child, .. } => {
                Self::collect_correlated_pattern_dependencies_inner(
                    child,
                    available,
                    &local_required,
                    &local_control,
                    required,
                    control,
                );
            }
            // A subselect is a lexical boundary and derives its own projected
            // requirements during translation.
            ast::GraphPattern::SubSelect(_)
            | ast::GraphPattern::Basic(_)
            | ast::GraphPattern::Filter(_)
            | ast::GraphPattern::Bind { .. }
            | ast::GraphPattern::InlineData(_) => {}
        }
    }

    /// Variables that participate in solution-mapping compatibility inside a
    /// graph pattern. Unlike ordinary output collection, BIND is excluded: it
    /// extends a mapping and is not an RDF-term join against an outer value.
    fn collect_pattern_compatibility_variables(
        pattern: &ast::GraphPattern,
        variables: &mut HashSet<String>,
    ) {
        match pattern {
            ast::GraphPattern::Basic(_) => {
                Self::collect_pattern_output_variables(pattern, variables);
            }
            ast::GraphPattern::InlineData(data) => {
                for (index, variable) in data.variables.iter().enumerate() {
                    if data
                        .values
                        .iter()
                        .any(|row| row.get(index).is_some_and(Option::is_some))
                    {
                        variables.insert(variable.clone());
                    }
                }
            }
            ast::GraphPattern::Group(patterns) | ast::GraphPattern::Union(patterns) => {
                for child in patterns {
                    Self::collect_pattern_compatibility_variables(child, variables);
                }
            }
            ast::GraphPattern::Optional(child)
            | ast::GraphPattern::Minus(child)
            | ast::GraphPattern::Service { pattern: child, .. } => {
                Self::collect_pattern_compatibility_variables(child, variables);
            }
            ast::GraphPattern::NamedGraph { graph, pattern } => {
                if let ast::VariableOrIri::Variable(variable) = graph {
                    variables.insert(variable.clone());
                }
                Self::collect_pattern_compatibility_variables(pattern, variables);
            }
            ast::GraphPattern::SubSelect(select) => {
                Self::collect_subselect_output_variables(select, variables);
            }
            ast::GraphPattern::Filter(_) | ast::GraphPattern::Bind { .. } => {}
        }
    }

    /// Collects variables whose exact RDF identity is consumed by a
    /// term-sensitive expression. Other arithmetic and helper expressions use
    /// visible values and do not force an opaque extension into the RDF term
    /// model.
    fn collect_term_semantic_dependencies(
        expression: &ast::Expression,
        variables: &mut HashSet<String>,
    ) {
        use ast::{BinaryOperator, BuiltInFunction, Expression, FunctionName};

        match expression {
            Expression::Binary {
                left,
                operator,
                right,
            } => {
                if matches!(operator, BinaryOperator::Equal | BinaryOperator::NotEqual) {
                    Self::collect_identity_dependencies(left, variables);
                    Self::collect_identity_dependencies(right, variables);
                }
                Self::collect_term_semantic_dependencies(left, variables);
                Self::collect_term_semantic_dependencies(right, variables);
            }
            Expression::FunctionCall {
                function,
                arguments,
            } => {
                let term_sensitive = matches!(
                    function,
                    FunctionName::BuiltIn(
                        BuiltInFunction::IsIri
                            | BuiltInFunction::IsBlank
                            | BuiltInFunction::IsLiteral
                            | BuiltInFunction::IsNumeric
                            | BuiltInFunction::SameTerm
                    )
                );
                if term_sensitive {
                    for argument in arguments {
                        Self::collect_identity_dependencies(argument, variables);
                    }
                }
                for argument in arguments {
                    Self::collect_term_semantic_dependencies(argument, variables);
                }
            }
            Expression::Unary { operand, .. } | Expression::Bracketed(operand) => {
                Self::collect_term_semantic_dependencies(operand, variables);
            }
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                Self::collect_term_semantic_dependencies(condition, variables);
                Self::collect_term_semantic_dependencies(then_expression, variables);
                Self::collect_term_semantic_dependencies(else_expression, variables);
            }
            Expression::Coalesce(expressions) => {
                for expression in expressions {
                    Self::collect_term_semantic_dependencies(expression, variables);
                }
            }
            Expression::In { expression, list } | Expression::NotIn { expression, list } => {
                Self::collect_identity_dependencies(expression, variables);
                Self::collect_term_semantic_dependencies(expression, variables);
                for expression in list {
                    Self::collect_identity_dependencies(expression, variables);
                    Self::collect_term_semantic_dependencies(expression, variables);
                }
            }
            Expression::Aggregate(aggregate) => match aggregate {
                ast::AggregateExpression::Count {
                    distinct,
                    expression,
                } => {
                    if let Some(expression) = expression {
                        if *distinct {
                            Self::collect_identity_dependencies(expression, variables);
                        }
                        Self::collect_term_semantic_dependencies(expression, variables);
                    }
                }
                ast::AggregateExpression::Sum {
                    distinct,
                    expression,
                }
                | ast::AggregateExpression::Average {
                    distinct,
                    expression,
                }
                | ast::AggregateExpression::GroupConcat {
                    distinct,
                    expression,
                    ..
                } => {
                    if *distinct {
                        Self::collect_identity_dependencies(expression, variables);
                    }
                    Self::collect_term_semantic_dependencies(expression, variables);
                }
                ast::AggregateExpression::Minimum { expression }
                | ast::AggregateExpression::Maximum { expression }
                | ast::AggregateExpression::Sample { expression } => {
                    Self::collect_term_semantic_dependencies(expression, variables);
                }
            },
            Expression::Variable(_)
            | Expression::Iri(_)
            | Expression::Literal(_)
            | Expression::Bound(_)
            | Expression::Exists(_)
            | Expression::NotExists(_) => {}
        }
    }

    /// Propagates only RDF-identity dependencies. Constructors, arithmetic,
    /// predicates, and selector conditions consume visible values but create
    /// a new RDF literal/IRI; requiring their inputs' identities would reject
    /// harmless helper expressions without improving template exactness.
    fn collect_identity_control_dependencies(
        expression: &ast::Expression,
        required: &mut HashSet<String>,
        control: &mut HashSet<String>,
        available: &HashSet<String>,
    ) {
        match expression {
            ast::Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                Self::collect_selector_dependencies(condition, required, control, available);
                Self::collect_identity_control_dependencies(
                    then_expression,
                    required,
                    control,
                    available,
                );
                Self::collect_identity_control_dependencies(
                    else_expression,
                    required,
                    control,
                    available,
                );
            }
            ast::Expression::Bracketed(inner) => {
                Self::collect_identity_control_dependencies(inner, required, control, available);
            }
            ast::Expression::Coalesce(arguments) => {
                for argument in arguments {
                    Self::collect_identity_control_dependencies(
                        argument, required, control, available,
                    );
                }
            }
            ast::Expression::Aggregate(aggregate) => match aggregate {
                ast::AggregateExpression::Minimum { expression }
                | ast::AggregateExpression::Maximum { expression }
                | ast::AggregateExpression::Sample { expression } => {
                    Self::collect_identity_control_dependencies(
                        expression, required, control, available,
                    );
                }
                ast::AggregateExpression::Count { .. }
                | ast::AggregateExpression::Sum { .. }
                | ast::AggregateExpression::Average { .. }
                | ast::AggregateExpression::GroupConcat { .. } => {}
            },
            ast::Expression::Variable(_)
            | ast::Expression::Iri(_)
            | ast::Expression::Literal(_)
            | ast::Expression::Binary { .. }
            | ast::Expression::Unary { .. }
            | ast::Expression::FunctionCall { .. }
            | ast::Expression::Bound(_)
            | ast::Expression::Exists(_)
            | ast::Expression::NotExists(_)
            | ast::Expression::In { .. }
            | ast::Expression::NotIn { .. } => {}
        }
    }

    fn collect_identity_dependencies(
        expression: &ast::Expression,
        variables: &mut HashSet<String>,
    ) {
        match expression {
            ast::Expression::Variable(variable) => {
                variables.insert(variable.clone());
            }
            ast::Expression::Bracketed(inner) => {
                Self::collect_identity_dependencies(inner, variables);
            }
            ast::Expression::Coalesce(arguments) => {
                for argument in arguments {
                    Self::collect_identity_dependencies(argument, variables);
                }
            }
            ast::Expression::Conditional {
                then_expression,
                else_expression,
                ..
            } => {
                Self::collect_identity_dependencies(then_expression, variables);
                Self::collect_identity_dependencies(else_expression, variables);
            }
            ast::Expression::Aggregate(aggregate) => match aggregate {
                ast::AggregateExpression::Minimum { expression }
                | ast::AggregateExpression::Maximum { expression }
                | ast::AggregateExpression::Sample { expression } => {
                    Self::collect_identity_dependencies(expression, variables);
                }
                ast::AggregateExpression::Count { .. }
                | ast::AggregateExpression::Sum { .. }
                | ast::AggregateExpression::Average { .. }
                | ast::AggregateExpression::GroupConcat { .. } => {}
            },
            ast::Expression::Iri(_)
            | ast::Expression::Literal(_)
            | ast::Expression::Binary { .. }
            | ast::Expression::Unary { .. }
            | ast::Expression::FunctionCall { .. }
            | ast::Expression::Bound(_)
            | ast::Expression::Exists(_)
            | ast::Expression::NotExists(_)
            | ast::Expression::In { .. }
            | ast::Expression::NotIn { .. } => {}
        }
    }

    fn ordinary_annotation_mismatch(
        expected: &str,
        annotation: &OrdinaryPatternExactAnnotations,
    ) -> Error {
        Error::Internal(format!(
            "ordinary exact-demand pattern annotation mismatch: expected {expected}, got {}",
            annotation.kind()
        ))
    }

    fn expect_ordinary_annotation(
        annotation: Option<&OrdinaryPatternExactAnnotations>,
        expected: &str,
    ) -> Result<()> {
        if let Some(annotation) = annotation
            && annotation.kind() != expected
        {
            return Err(Self::ordinary_annotation_mismatch(expected, annotation));
        }
        Ok(())
    }

    fn translate_graph_pattern(&mut self, pattern: &ast::GraphPattern) -> Result<LogicalOperator> {
        if !self.exact_mutation_bindings {
            #[cfg(test)]
            ORDINARY_EXACT_FALLBACK_ANALYSES.with(|count| count.set(count.get() + 1));
            let mut variables = OrdinaryVariableTable::default();
            let mut bound = OrdinaryMembershipState::default();
            let (mut annotation, _) =
                Self::build_ordinary_pattern_annotations(pattern, &mut variables, &mut bound)?;
            let mut demand = OrdinaryMembershipState::default();
            Self::analyze_ordinary_pattern(pattern, &mut annotation, &mut variables, &mut demand)?;
            return self.translate_graph_pattern_with_annotations(pattern, Some(&annotation));
        }

        self.translate_graph_pattern_with_annotations(pattern, None)
    }

    fn extract_group_exists_filter<'a>(
        expression: &'a ast::Expression,
        annotation: Option<&'a OrdinaryExpressionExactAnnotations>,
        negated: bool,
    ) -> Result<
        Option<(
            bool,
            &'a ast::GraphPattern,
            Option<&'a OrdinaryPatternExactAnnotations>,
        )>,
    > {
        match expression {
            ast::Expression::Exists(pattern) => {
                let annotation = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Exists { pattern, .. }) => {
                        Some(pattern.as_ref())
                    }
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand EXISTS annotation mismatch".to_string(),
                        ));
                    }
                    None => None,
                };
                Ok(Some((negated, pattern.as_ref(), annotation)))
            }
            ast::Expression::NotExists(pattern) => {
                let annotation = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::NotExists { pattern, .. }) => {
                        Some(pattern.as_ref())
                    }
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand NOT EXISTS annotation mismatch".to_string(),
                        ));
                    }
                    None => None,
                };
                Ok(Some((!negated, pattern.as_ref(), annotation)))
            }
            ast::Expression::Unary {
                operator: ast::UnaryOperator::Not,
                operand,
            } => {
                let annotation = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Unary(annotation)) => {
                        Some(annotation.as_ref())
                    }
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand unary EXISTS annotation mismatch".to_string(),
                        ));
                    }
                    None => None,
                };
                Self::extract_group_exists_filter(operand, annotation, !negated)
            }
            ast::Expression::Bracketed(inner) => {
                let annotation = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Bracketed(annotation)) => {
                        Some(annotation.as_ref())
                    }
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand bracketed EXISTS annotation mismatch"
                                .to_string(),
                        ));
                    }
                    None => None,
                };
                Self::extract_group_exists_filter(inner, annotation, negated)
            }
            _ => Ok(None),
        }
    }

    fn translate_graph_pattern_with_annotations(
        &mut self,
        pattern: &ast::GraphPattern,
        ordinary_annotation: Option<&OrdinaryPatternExactAnnotations>,
    ) -> Result<LogicalOperator> {
        if !self.exact_mutation_bindings {
            return self.translate_graph_pattern_inner(pattern, ordinary_annotation);
        }

        let previous_exact = self.exact_mutation_variables.take();
        let previous_control = self.mutation_control_variables.take();
        let mut local_exact = previous_exact.clone().unwrap_or_default();
        let mut local_control = previous_control.clone().unwrap_or_default();
        Self::expand_pattern_scope_dependencies(pattern, &mut local_exact, &mut local_control);
        self.exact_mutation_variables = Some(local_exact);
        self.mutation_control_variables = Some(local_control);

        let translated = self.translate_graph_pattern_inner(pattern, None);
        self.exact_mutation_variables = previous_exact;
        self.mutation_control_variables = previous_control;
        translated
    }

    fn translate_graph_pattern_inner(
        &mut self,
        pattern: &ast::GraphPattern,
        ordinary_annotation: Option<&OrdinaryPatternExactAnnotations>,
    ) -> Result<LogicalOperator> {
        match pattern {
            ast::GraphPattern::Basic(triples) => {
                Self::expect_ordinary_annotation(ordinary_annotation, "Basic")?;
                self.translate_basic_pattern(triples)
            }

            ast::GraphPattern::Group(patterns) => {
                // As for basic patterns: each element joins onto the whole
                // left side, so a longer group cannot fit the depth limit.
                if patterns.len() > crate::query::plan_depth::MAX_PLAN_DEPTH {
                    return Err(crate::query::plan_depth::too_deep());
                }
                let child_annotations = match ordinary_annotation {
                    Some(OrdinaryPatternExactAnnotations::Group(children))
                        if children.len() == patterns.len() =>
                    {
                        Some(children.as_slice())
                    }
                    Some(OrdinaryPatternExactAnnotations::Group(children)) => {
                        return Err(Error::Internal(format!(
                            "ordinary exact-demand group annotation mismatch: expected {} children, got {}",
                            patterns.len(),
                            children.len()
                        )));
                    }
                    Some(annotation) => {
                        return Err(Self::ordinary_annotation_mismatch("Group", annotation));
                    }
                    None => None,
                };

                // Process patterns in document order so that BIND, OPTIONAL,
                // MINUS, etc. see the variables introduced by preceding
                // patterns. FILTER and FILTER NOT EXISTS/EXISTS scope over
                // the entire group (SPARQL spec), so they are collected and
                // applied last.
                let mut filter_exprs: Vec<(
                    &ast::Expression,
                    Option<&OrdinaryExpressionExactAnnotations>,
                )> = Vec::new();
                let mut not_exists_patterns: Vec<(
                    &ast::GraphPattern,
                    Option<&OrdinaryPatternExactAnnotations>,
                )> = Vec::new();
                let mut exists_patterns: Vec<(
                    &ast::GraphPattern,
                    Option<&OrdinaryPatternExactAnnotations>,
                )> = Vec::new();

                let mut plan = LogicalOperator::Empty;

                for (index, p) in patterns.iter().enumerate() {
                    let child_annotation = child_annotations.map(|children| &children[index]);
                    match p {
                        ast::GraphPattern::Filter(expr) => {
                            let expression_annotation = match child_annotation {
                                Some(OrdinaryPatternExactAnnotations::Filter(annotation)) => {
                                    Some(annotation)
                                }
                                Some(annotation) => {
                                    return Err(Self::ordinary_annotation_mismatch(
                                        "Filter", annotation,
                                    ));
                                }
                                None => None,
                            };
                            // Collect whole FILTER EXISTS/NOT EXISTS for
                            // row-aware semi/anti lowering, including parser
                            // bracket/unary wrappers.
                            if let Some((negated, inner, annotation)) =
                                Self::extract_group_exists_filter(
                                    expr,
                                    expression_annotation,
                                    false,
                                )?
                            {
                                if negated {
                                    not_exists_patterns.push((inner, annotation));
                                } else {
                                    exists_patterns.push((inner, annotation));
                                }
                            } else {
                                filter_exprs.push((expr, expression_annotation));
                            }
                        }
                        ast::GraphPattern::Bind {
                            expression,
                            variable,
                        } => {
                            let (expression_annotation, exact_required, rdf_or_native_required) =
                                match child_annotation {
                                    Some(OrdinaryPatternExactAnnotations::Bind {
                                        expression,
                                        exact,
                                        rdf_or_native,
                                        ..
                                    }) => (Some(expression), *exact, *rdf_or_native),
                                    Some(annotation) => {
                                        return Err(Self::ordinary_annotation_mismatch(
                                            "Bind", annotation,
                                        ));
                                    }
                                    None => (None, false, false),
                                };
                            plan = self.translate_pattern_bind_with_annotations(
                                plan,
                                expression,
                                variable,
                                expression_annotation,
                                exact_required,
                                rdf_or_native_required,
                            )?;
                        }
                        ast::GraphPattern::Optional(inner) => {
                            let inner_annotation = match child_annotation {
                                Some(OrdinaryPatternExactAnnotations::Optional {
                                    pattern, ..
                                }) => Some(pattern.as_ref()),
                                Some(annotation) => {
                                    return Err(Self::ordinary_annotation_mismatch(
                                        "Optional", annotation,
                                    ));
                                }
                                None => None,
                            };
                            let inner_plan = self.translate_graph_pattern_with_annotations(
                                inner,
                                inner_annotation,
                            )?;
                            // `Empty` is the unit solution (one empty mapping),
                            // not an empty result. A leading OPTIONAL must still
                            // be a left join so an empty RHS preserves that row.
                            plan = self.left_join_patterns(plan, inner_plan);
                        }
                        ast::GraphPattern::Minus(inner) => {
                            let inner_annotation = match child_annotation {
                                Some(OrdinaryPatternExactAnnotations::Minus {
                                    pattern, ..
                                }) => Some(pattern.as_ref()),
                                Some(annotation) => {
                                    return Err(Self::ordinary_annotation_mismatch(
                                        "Minus", annotation,
                                    ));
                                }
                                None => None,
                            };
                            let inner_plan = self.translate_graph_pattern_with_annotations(
                                inner,
                                inner_annotation,
                            )?;
                            if !matches!(plan, LogicalOperator::Empty) {
                                plan = self.anti_join_patterns(
                                    plan,
                                    inner_plan,
                                    AntiJoinSemantics::Minus,
                                );
                            }
                        }
                        _ => {
                            let p_plan =
                                self.translate_graph_pattern_with_annotations(p, child_annotation)?;
                            plan = self.join_patterns(plan, p_plan);
                        }
                    }
                }

                // Apply FILTER NOT EXISTS as anti joins
                for (inner, inner_annotation) in not_exists_patterns {
                    let inner_plan =
                        self.translate_graph_pattern_with_annotations(inner, inner_annotation)?;
                    plan = self.anti_join_patterns(plan, inner_plan, AntiJoinSemantics::NotExists);
                }

                // 4c. Apply FILTER EXISTS as true semi joins: matching inner
                // multiplicity must never duplicate an outer solution.
                for (inner, inner_annotation) in exists_patterns {
                    let inner_plan =
                        self.translate_graph_pattern_with_annotations(inner, inner_annotation)?;
                    plan = self.semi_join_patterns(plan, inner_plan);
                }

                // 5. Apply FILTER expressions last (they scope over entire group)
                if !filter_exprs.is_empty() {
                    let predicates: Vec<LogicalExpression> = filter_exprs
                        .into_iter()
                        .map(|(expression, annotation)| {
                            self.translate_expression_with_annotations(expression, annotation)
                        })
                        .collect::<Result<Vec<_>>>()?;

                    // Combine all predicates with AND
                    let combined = predicates
                        .into_iter()
                        .reduce(|acc, pred| LogicalExpression::Binary {
                            left: Box::new(acc),
                            op: BinaryOp::And,
                            right: Box::new(pred),
                        })
                        .ok_or_else(|| {
                            Error::Internal("SPARQL group filter predicate is missing".to_string())
                        })?;

                    plan = wrap_filter(plan, combined);
                }

                Ok(plan)
            }

            ast::GraphPattern::Optional(inner) => {
                // Standalone OPTIONAL - handled in Group translation, but support direct call
                let inner_annotation = match ordinary_annotation {
                    Some(OrdinaryPatternExactAnnotations::Optional { pattern, .. }) => {
                        Some(pattern.as_ref())
                    }
                    Some(annotation) => {
                        return Err(Self::ordinary_annotation_mismatch("Optional", annotation));
                    }
                    None => None,
                };
                let inner =
                    self.translate_graph_pattern_with_annotations(inner, inner_annotation)?;
                Ok(self.left_join_patterns(LogicalOperator::Empty, inner))
            }

            ast::GraphPattern::Union(alternatives) => {
                let child_annotations = match ordinary_annotation {
                    Some(OrdinaryPatternExactAnnotations::Union(children))
                        if children.len() == alternatives.len() =>
                    {
                        Some(children.as_slice())
                    }
                    Some(OrdinaryPatternExactAnnotations::Union(children)) => {
                        return Err(Error::Internal(format!(
                            "ordinary exact-demand union annotation mismatch: expected {} children, got {}",
                            alternatives.len(),
                            children.len()
                        )));
                    }
                    Some(annotation) => {
                        return Err(Self::ordinary_annotation_mismatch("Union", annotation));
                    }
                    None => None,
                };
                let inputs = alternatives
                    .iter()
                    .enumerate()
                    .map(|(index, pattern)| {
                        self.translate_graph_pattern_with_annotations(
                            pattern,
                            child_annotations.map(|children| &children[index].pattern),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;

                Ok(LogicalOperator::Union(UnionOp { inputs }))
            }

            ast::GraphPattern::Minus(inner) => {
                // Standalone MINUS - handled in Group translation, but support direct call
                let inner_annotation = match ordinary_annotation {
                    Some(OrdinaryPatternExactAnnotations::Minus { pattern, .. }) => {
                        Some(pattern.as_ref())
                    }
                    Some(annotation) => {
                        return Err(Self::ordinary_annotation_mismatch("Minus", annotation));
                    }
                    None => None,
                };
                let inner =
                    self.translate_graph_pattern_with_annotations(inner, inner_annotation)?;
                Ok(
                    self.anti_join_patterns(
                        LogicalOperator::Empty,
                        inner,
                        AntiJoinSemantics::Minus,
                    ),
                )
            }

            ast::GraphPattern::Filter(expr) => {
                let expression_annotation = match ordinary_annotation {
                    Some(OrdinaryPatternExactAnnotations::Filter(annotation)) => Some(annotation),
                    Some(annotation) => {
                        return Err(Self::ordinary_annotation_mismatch("Filter", annotation));
                    }
                    None => None,
                };
                if let Some((negated, inner, inner_annotation)) =
                    Self::extract_group_exists_filter(expr, expression_annotation, false)?
                {
                    let inner_plan =
                        self.translate_graph_pattern_with_annotations(inner, inner_annotation)?;
                    return Ok(if negated {
                        self.anti_join_patterns(
                            LogicalOperator::Empty,
                            inner_plan,
                            AntiJoinSemantics::NotExists,
                        )
                    } else {
                        self.semi_join_patterns(LogicalOperator::Empty, inner_plan)
                    });
                }
                // Standalone FILTER - handled in Group translation, but support direct call
                // This can happen when Filter is the top-level pattern
                let predicate =
                    self.translate_expression_with_annotations(expr, expression_annotation)?;
                Ok(wrap_filter(LogicalOperator::Empty, predicate))
            }

            ast::GraphPattern::Bind {
                expression,
                variable,
            } => {
                let (expression_annotation, exact_required, rdf_or_native_required) =
                    match ordinary_annotation {
                        Some(OrdinaryPatternExactAnnotations::Bind {
                            expression,
                            exact,
                            rdf_or_native,
                            ..
                        }) => (Some(expression), *exact, *rdf_or_native),
                        Some(annotation) => {
                            return Err(Self::ordinary_annotation_mismatch("Bind", annotation));
                        }
                        None => (None, false, false),
                    };
                // Standalone BIND - handled in Group translation, but support direct call
                self.translate_pattern_bind_with_annotations(
                    LogicalOperator::Empty,
                    expression,
                    variable,
                    expression_annotation,
                    exact_required,
                    rdf_or_native_required,
                )
            }

            ast::GraphPattern::NamedGraph { graph, pattern } => {
                let inner_annotation = match ordinary_annotation {
                    Some(OrdinaryPatternExactAnnotations::NamedGraph { pattern, .. }) => {
                        Some(pattern.as_ref())
                    }
                    Some(annotation) => {
                        return Err(Self::ordinary_annotation_mismatch("NamedGraph", annotation));
                    }
                    None => None,
                };
                let graph_component = match graph {
                    ast::VariableOrIri::Variable(name) => TripleComponent::Variable(name.clone()),
                    ast::VariableOrIri::Iri(iri) => TripleComponent::Iri(self.resolve_iri(iri)),
                };
                self.graph_context_stack.push(graph_component);
                let plan = self.translate_graph_pattern_with_annotations(pattern, inner_annotation);
                self.graph_context_stack.pop();
                plan
            }

            ast::GraphPattern::SubSelect(subquery) => {
                let select_annotation = match ordinary_annotation {
                    Some(OrdinaryPatternExactAnnotations::SubSelect { select, .. }) => {
                        Some(select.as_ref())
                    }
                    Some(annotation) => {
                        return Err(Self::ordinary_annotation_mismatch("SubSelect", annotation));
                    }
                    None => None,
                };
                let plan = self.translate_select_with_annotations(subquery, select_annotation)?;
                Ok(plan.root)
            }

            ast::GraphPattern::Service { .. } => {
                Self::expect_ordinary_annotation(ordinary_annotation, "Service")?;
                Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "SPARQL SERVICE (federated queries) is not yet supported",
                )))
            }

            ast::GraphPattern::InlineData(data) => {
                Self::expect_ordinary_annotation(ordinary_annotation, "InlineData")?;
                // VALUES clause: each row becomes a chain of BIND operators
                // starting from Empty, and all rows are combined with UNION.
                if data.values.is_empty() {
                    // LogicalOperator::Empty is the unit solution in the RDF
                    // planner. An empty VALUES table instead has zero rows.
                    return Ok(wrap_filter(
                        LogicalOperator::Empty,
                        LogicalExpression::Literal(Value::Bool(false)),
                    ));
                }
                let mut branches = Vec::new();
                for row in &data.values {
                    let mut plan = LogicalOperator::Empty;
                    for (var, val) in data.variables.iter().zip(row.iter()) {
                        if let Some(dv) = val {
                            let expression = match dv {
                                ast::DataValue::Iri(iri) => ast::Expression::Iri(iri.clone()),
                                ast::DataValue::Literal(literal) => {
                                    ast::Expression::Literal(literal.clone())
                                }
                            };
                            plan = self.translate_pattern_bind(plan, &expression, var)?;
                        } else {
                            plan = LogicalOperator::Bind(BindOp {
                                expression: LogicalExpression::Literal(Value::Null),
                                variable: rdf_tagged_term_column(var),
                                input: Box::new(plan),
                            });
                            plan = LogicalOperator::Bind(BindOp {
                                expression: LogicalExpression::Literal(Value::Null),
                                variable: var.clone(),
                                input: Box::new(plan),
                            });
                            plan = LogicalOperator::Bind(BindOp {
                                expression: LogicalExpression::Literal(Value::Null),
                                variable: rdf_exact_term_column(var),
                                input: Box::new(plan),
                            });
                            plan = LogicalOperator::Bind(BindOp {
                                expression: LogicalExpression::Literal(Value::Null),
                                variable: rdf_identity_key_column(var),
                                input: Box::new(plan),
                            });
                        }
                    }
                    branches.push(plan);
                }
                if branches.len() == 1 {
                    branches.pop().ok_or_else(|| {
                        Error::Internal("SPARQL VALUES branch is missing".to_string())
                    })
                } else {
                    Ok(LogicalOperator::Union(UnionOp { inputs: branches }))
                }
            }
        }
    }

    fn translate_basic_pattern(
        &mut self,
        triples: &[ast::TriplePattern],
    ) -> Result<LogicalOperator> {
        if triples.is_empty() {
            return Ok(LogicalOperator::Empty);
        }
        // Each triple adds a join level, and joining inspects the whole left
        // side: a group this long would exceed the statement depth limit, and
        // building it first recursed deep enough to overflow the stack.
        if triples.len() > crate::query::plan_depth::MAX_PLAN_DEPTH {
            return Err(crate::query::plan_depth::too_deep());
        }

        let mut plan = LogicalOperator::Empty;

        for triple in triples {
            let triple_scan = self.translate_triple_pattern(triple)?;
            plan = self.join_patterns(plan, triple_scan);
        }

        Ok(plan)
    }

    fn translate_triple_pattern(&mut self, triple: &ast::TriplePattern) -> Result<LogicalOperator> {
        let mut normalized = triple.clone();
        let mut first_occurrence = HashMap::<String, String>::new();
        let mut outputs = Vec::new();
        let mut repeated = Vec::new();
        let mut next_anon = self.anon_counter;
        let query_id = self.query_id;
        let graph_index = self.graph_context_stack.len().checked_sub(1);
        let mut normalized_graph = graph_index
            .and_then(|index| self.graph_context_stack.get(index))
            .cloned();
        {
            let mut normalize_variable = |variable: &mut String| {
                if let Some(first) = first_occurrence.get(variable) {
                    let replacement = format!("\0grafeo:rdf-repeat:{query_id}:{next_anon}");
                    next_anon += 1;
                    repeated.push((first.clone(), replacement.clone()));
                    *variable = replacement;
                } else {
                    first_occurrence.insert(variable.clone(), variable.clone());
                    outputs.push(variable.clone());
                }
            };

            if let ast::TripleTerm::Variable(variable) = &mut normalized.subject {
                normalize_variable(variable);
            }
            if let ast::PropertyPath::Variable(variable) = &mut normalized.predicate {
                normalize_variable(variable);
            }
            if let ast::TripleTerm::Variable(variable) = &mut normalized.object {
                normalize_variable(variable);
            }
            if let Some(TripleComponent::Variable(variable)) = &mut normalized_graph {
                normalize_variable(variable);
            }
        }
        self.anon_counter = next_anon;

        if repeated.is_empty() {
            return self.translate_triple_pattern_distinct(&normalized);
        }

        let original_graph = graph_index.zip(normalized_graph).map(|(index, graph)| {
            let original = std::mem::replace(&mut self.graph_context_stack[index], graph);
            (index, original)
        });
        let translated = self.translate_triple_pattern_distinct(&normalized);
        if let Some((index, original)) = original_graph {
            self.graph_context_stack[index] = original;
        }
        let mut plan = translated?;

        let tagged = |variable: &str| LogicalExpression::FunctionCall {
            name: RDF_TAG_BOUND_TERM.to_string(),
            args: vec![
                LogicalExpression::Variable(variable.to_string()),
                LogicalExpression::Variable(rdf_exact_term_column(variable)),
            ],
            distinct: false,
        };
        let predicate = repeated
            .into_iter()
            .map(|(first, repeated)| LogicalExpression::FunctionCall {
                name: RDF_SAME_TERM.to_string(),
                args: vec![tagged(&first), tagged(&repeated)],
                distinct: false,
            })
            .reduce(|left, right| LogicalExpression::Binary {
                left: Box::new(left),
                op: BinaryOp::And,
                right: Box::new(right),
            })
            .ok_or_else(|| {
                Error::Internal("SPARQL repeated-variable predicate is missing".to_string())
            })?;
        plan = wrap_filter(plan, predicate);

        Ok(LogicalOperator::Project(ProjectOp {
            projections: outputs
                .into_iter()
                .map(|variable| Projection {
                    expression: LogicalExpression::Variable(variable),
                    alias: None,
                })
                .collect(),
            input: Box::new(plan),
            pass_through_input: false,
        }))
    }

    fn translate_triple_pattern_distinct(
        &mut self,
        triple: &ast::TriplePattern,
    ) -> Result<LogicalOperator> {
        // Handle Sequence property paths: expand into chained triple patterns
        // e.g. ?person foaf:knows/foaf:name ?name  becomes:
        //   ?person foaf:knows ?_anon0 . ?_anon0 foaf:name ?name
        if let ast::PropertyPath::Sequence(paths) = &triple.predicate {
            let subject = self.translate_triple_term(&triple.subject)?;
            let object = self.translate_triple_term(&triple.object)?;
            let graph = self.graph_context_stack.last().cloned();

            let mut current_subject = subject;
            let mut plan = LogicalOperator::Empty;

            for (i, path) in paths.iter().enumerate() {
                let next_object = if i == paths.len() - 1 {
                    object.clone()
                } else {
                    TripleComponent::Variable(format!("_:seq{}", self.next_anon()))
                };

                let step = if self.is_simple_path(path) {
                    let pred = self.translate_property_path(path)?;
                    self.make_triple_scan(
                        current_subject.clone(),
                        pred,
                        next_object.clone(),
                        graph.clone(),
                    )
                } else {
                    // Complex path (ZeroOrMore, OneOrMore, etc.): recurse
                    let sub_triple = ast::TriplePattern {
                        subject: self.triple_component_to_term(&current_subject),
                        predicate: path.clone(),
                        object: self.triple_component_to_term(&next_object),
                    };
                    self.translate_triple_pattern(&sub_triple)?
                };

                plan = self.join_patterns(plan, step);
                current_subject = next_object;
            }

            return Ok(plan);
        }

        // Handle Alternative property paths: translate as Union of triple scans
        if let ast::PropertyPath::Alternative(alternatives) = &triple.predicate {
            let subject = self.translate_triple_term(&triple.subject)?;
            let object = self.translate_triple_term(&triple.object)?;
            let graph = self.graph_context_stack.last().cloned();

            let mut branches = Vec::new();
            for alt_path in alternatives {
                let pred = self.translate_property_path(alt_path)?;
                branches.push(self.make_triple_scan(
                    subject.clone(),
                    pred,
                    object.clone(),
                    graph.clone(),
                ));
            }

            return Ok(LogicalOperator::Union(UnionOp { inputs: branches }));
        }

        // Handle OneOrMore (path+): bounded expansion
        if let ast::PropertyPath::OneOrMore(inner) = &triple.predicate {
            return self.translate_one_or_more_path(triple, inner);
        }

        // Handle ZeroOrMore (path*): bounded expansion
        if let ast::PropertyPath::ZeroOrMore(inner) = &triple.predicate {
            return self.translate_zero_or_more_path(triple, inner);
        }

        // Handle Inverse (^path): swap subject and object, translate inner path
        if let ast::PropertyPath::Inverse(inner) = &triple.predicate {
            let swapped = ast::TriplePattern {
                subject: triple.object.clone(),
                predicate: *inner.clone(),
                object: triple.subject.clone(),
            };
            return self.translate_triple_pattern(&swapped);
        }

        // Handle ZeroOrOne (path?): union of reflexive 0-hop and 1-hop
        if let ast::PropertyPath::ZeroOrOne(inner) = &triple.predicate {
            return self.translate_zero_or_one_path(triple, inner);
        }

        // Handle Negation: !(iri1|^iri2) scans all triples and filters out excluded predicates
        if let ast::PropertyPath::Negation(negated_iris) = &triple.predicate {
            return self.translate_negated_property_set(triple, negated_iris);
        }

        let subject = self.translate_triple_term(&triple.subject)?;
        let predicate = self.translate_property_path(&triple.predicate)?;
        let object = self.translate_triple_term(&triple.object)?;
        Ok(self.make_triple_scan(
            subject,
            predicate,
            object,
            self.graph_context_stack.last().cloned(),
        ))
    }

    fn translate_triple_term(&mut self, term: &ast::TripleTerm) -> Result<TripleComponent> {
        match term {
            ast::TripleTerm::Variable(name) => Ok(TripleComponent::Variable(name.clone())),
            ast::TripleTerm::Iri(iri) => Ok(TripleComponent::Iri(self.resolve_iri(iri))),
            ast::TripleTerm::Literal(lit) => {
                if let Some(lang) = &lit.language {
                    Ok(TripleComponent::LangLiteral {
                        value: lit.value.clone(),
                        lang: lang.clone(),
                    })
                } else {
                    Ok(TripleComponent::Literal(self.pattern_literal_to_value(lit)))
                }
            }
            ast::TripleTerm::BlankNode(bnode) => {
                // Treat blank nodes as variables, scoped by query_id
                match bnode {
                    ast::BlankNode::Labeled(label) => Ok(TripleComponent::Variable(format!(
                        "_:q{}_{label}",
                        self.query_id
                    ))),
                    ast::BlankNode::Anonymous(_) => {
                        let anon = self.next_anon();
                        let var = format!("_:q{}_anon{anon}", self.query_id);
                        Ok(TripleComponent::Variable(var))
                    }
                }
            }
        }
    }

    /// Returns true if the property path is a simple predicate (IRI, variable, or rdf:type).
    fn is_simple_path(&self, path: &ast::PropertyPath) -> bool {
        matches!(
            path,
            ast::PropertyPath::Predicate(_)
                | ast::PropertyPath::Variable(_)
                | ast::PropertyPath::RdfType
        )
    }

    /// Converts a `TripleComponent` back to an AST `TripleTerm` for recursive translation.
    fn triple_component_to_term(&self, component: &TripleComponent) -> ast::TripleTerm {
        match component {
            TripleComponent::Variable(name) => ast::TripleTerm::Variable(name.clone()),
            TripleComponent::Iri(iri) => ast::TripleTerm::Iri(ast::Iri(iri.clone())),
            TripleComponent::Literal(val) => ast::TripleTerm::Literal(ast::Literal {
                value: val.to_string(),
                datatype: None,
                language: None,
            }),
            TripleComponent::LangLiteral { value, lang } => {
                ast::TripleTerm::Literal(ast::Literal {
                    value: value.clone(),
                    datatype: None,
                    language: Some(lang.clone()),
                })
            }
            TripleComponent::BlankNode(label) => {
                ast::TripleTerm::BlankNode(ast::BlankNode::Labeled(label.clone()))
            }
        }
    }

    fn translate_property_path(&mut self, path: &ast::PropertyPath) -> Result<TripleComponent> {
        match path {
            ast::PropertyPath::Predicate(iri) => Ok(TripleComponent::Iri(self.resolve_iri(iri))),
            ast::PropertyPath::Variable(name) => Ok(TripleComponent::Variable(name.clone())),
            ast::PropertyPath::RdfType => Ok(TripleComponent::Iri(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            )),
            // Complex property paths are not fully supported yet
            _ => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Complex property paths not yet supported",
            ))),
        }
    }

    fn translate_expression(&mut self, expr: &ast::Expression) -> Result<LogicalExpression> {
        self.translate_expression_with_annotations(expr, None)
    }

    fn translate_expression_with_annotations(
        &mut self,
        expr: &ast::Expression,
        annotation: Option<&OrdinaryExpressionExactAnnotations>,
    ) -> Result<LogicalExpression> {
        if self.expression_depth >= crate::query::plan_depth::MAX_PLAN_DEPTH {
            return Err(crate::query::plan_depth::too_deep());
        }
        self.expression_depth += 1;
        let result = self.translate_expression_node(expr, annotation);
        self.expression_depth -= 1;
        result
    }

    fn translate_expression_node(
        &mut self,
        expr: &ast::Expression,
        annotation: Option<&OrdinaryExpressionExactAnnotations>,
    ) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),

            ast::Expression::Iri(iri) => Ok(LogicalExpression::Literal(Value::String(
                self.resolve_iri(iri).into(),
            ))),

            ast::Expression::Literal(lit) => {
                Ok(LogicalExpression::Literal(self.literal_to_value(lit)))
            }

            ast::Expression::Binary {
                left,
                operator,
                right,
            } => {
                let (left_annotation, right_annotation) = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Binary(left, right)) => {
                        (Some(left.as_ref()), Some(right.as_ref()))
                    }
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand expression annotation mismatch at Binary"
                                .to_string(),
                        ));
                    }
                    None => (None, None),
                };
                if self.exact_mutation_bindings
                    && matches!(
                        operator,
                        ast::BinaryOperator::Equal | ast::BinaryOperator::NotEqual
                    )
                {
                    let left =
                        self.tagged_bind_expression_with_annotations(left, left_annotation)?;
                    let right =
                        self.tagged_bind_expression_with_annotations(right, right_annotation)?;
                    let equality = LogicalExpression::FunctionCall {
                        name: RDF_TERM_EQUAL.to_string(),
                        args: vec![left, right],
                        distinct: false,
                    };
                    return if *operator == ast::BinaryOperator::NotEqual {
                        Ok(LogicalExpression::Unary {
                            op: UnaryOp::Not,
                            operand: Box::new(equality),
                        })
                    } else {
                        Ok(equality)
                    };
                }

                // Detect language-tagged literal comparisons: ?var = "value"@lang
                // or "value"@lang = ?var. Rewrite to check both lexical value and
                // language tag so that "Barcelona"@es does not match "Barcelona"@ca.
                if matches!(
                    operator,
                    ast::BinaryOperator::Equal | ast::BinaryOperator::NotEqual
                ) && let Some(expanded) = self.try_expand_lang_comparison(
                    left,
                    left_annotation,
                    *operator,
                    right,
                    right_annotation,
                )? {
                    return Ok(expanded);
                }

                let left = self.translate_expression_with_annotations(left, left_annotation)?;
                let right = self.translate_expression_with_annotations(right, right_annotation)?;
                let op = self.translate_binary_op(*operator);
                Ok(LogicalExpression::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                })
            }

            ast::Expression::Unary { operator, operand } => {
                let operand_annotation = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Unary(operand)) => {
                        Some(operand.as_ref())
                    }
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand expression annotation mismatch at Unary"
                                .to_string(),
                        ));
                    }
                    None => None,
                };
                let operand =
                    self.translate_expression_with_annotations(operand, operand_annotation)?;
                match operator {
                    ast::UnaryOperator::Plus => Ok(LogicalExpression::FunctionCall {
                        name: RDF_NUMERIC_VALUE.to_string(),
                        args: vec![operand],
                        distinct: false,
                    }),
                    ast::UnaryOperator::Not => Ok(LogicalExpression::Unary {
                        op: UnaryOp::Not,
                        operand: Box::new(operand),
                    }),
                    ast::UnaryOperator::Minus => Ok(LogicalExpression::Unary {
                        op: UnaryOp::Neg,
                        operand: Box::new(operand),
                    }),
                }
            }

            ast::Expression::FunctionCall {
                function,
                arguments,
            } => {
                let argument_annotations = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::FunctionCall(annotations))
                        if annotations.len() == arguments.len() =>
                    {
                        Some(annotations.as_slice())
                    }
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand expression annotation mismatch at FunctionCall"
                                .to_string(),
                        ));
                    }
                    None => None,
                };
                let name = self.translate_function_name(function);
                let datetime_component = matches!(
                    name.as_str(),
                    "YEAR" | "MONTH" | "DAY" | "HOURS" | "MINUTES" | "SECONDS" | "TIMEZONE" | "TZ"
                );
                if datetime_component && arguments.len() != 1 {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("{name} requires exactly 1 argument"),
                    )));
                }
                if matches!(
                    function,
                    ast::FunctionName::BuiltIn(ast::BuiltInFunction::IsNumeric)
                ) && arguments.len() != 1
                {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "ISNUMERIC requires exactly 1 argument",
                    )));
                }
                if self.exact_mutation_bindings
                    || matches!(
                        function,
                        ast::FunctionName::BuiltIn(ast::BuiltInFunction::SameTerm)
                    )
                {
                    use ast::{BuiltInFunction as BuiltIn, FunctionName};

                    let upper = name.to_ascii_uppercase();
                    let exact_function = match (self.exact_mutation_bindings, function) {
                        (_, FunctionName::BuiltIn(BuiltIn::SameTerm)) => Some((RDF_SAME_TERM, 2)),
                        (true, FunctionName::BuiltIn(BuiltIn::IsIri)) => Some((RDF_IS_IRI, 1)),
                        (true, FunctionName::BuiltIn(BuiltIn::IsBlank)) => Some((RDF_IS_BLANK, 1)),
                        (true, FunctionName::BuiltIn(BuiltIn::IsLiteral)) => {
                            Some((RDF_IS_LITERAL, 1))
                        }
                        (true, FunctionName::BuiltIn(BuiltIn::IsNumeric)) => {
                            Some((RDF_IS_NUMERIC, 1))
                        }
                        _ => None,
                    };
                    if let Some((internal, arity)) = exact_function {
                        if arguments.len() != arity {
                            return Err(Error::Query(QueryError::new(
                                QueryErrorKind::Semantic,
                                format!(
                                    "{upper} requires exactly {arity} argument{}",
                                    if arity == 1 { "" } else { "s" }
                                ),
                            )));
                        }
                        let args = arguments
                            .iter()
                            .enumerate()
                            .map(|(index, argument)| {
                                self.tagged_bind_expression_with_annotations(
                                    argument,
                                    argument_annotations.map(|annotations| &annotations[index]),
                                )
                            })
                            .collect::<Result<Vec<_>>>()?;
                        return Ok(LogicalExpression::FunctionCall {
                            name: internal.to_string(),
                            args,
                            distinct: false,
                        });
                    }
                }
                let args = arguments
                    .iter()
                    .enumerate()
                    .map(|(index, argument)| {
                        let annotation =
                            argument_annotations.map(|annotations| &annotations[index]);
                        if datetime_component {
                            self.tagged_bind_expression_with_annotations(argument, annotation)
                        } else if name == "STRDT" && index == 1 {
                            self.translate_strdt_datatype_argument_with_annotations(
                                argument, annotation,
                            )
                        } else {
                            self.translate_expression_with_annotations(argument, annotation)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::FunctionCall {
                    name,
                    args,
                    distinct: false,
                })
            }

            ast::Expression::Bound(var) => {
                // BOUND(?x) checks if variable is bound
                Ok(LogicalExpression::FunctionCall {
                    name: "BOUND".to_string(),
                    args: vec![LogicalExpression::Variable(var.clone())],
                    distinct: false,
                })
            }

            ast::Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                let (condition_annotation, then_annotation, else_annotation) = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Conditional {
                        condition,
                        then_expression,
                        else_expression,
                    }) => (
                        Some(condition.as_ref()),
                        Some(then_expression.as_ref()),
                        Some(else_expression.as_ref()),
                    ),
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand expression annotation mismatch at Conditional"
                                .to_string(),
                        ));
                    }
                    None => (None, None, None),
                };
                let cond =
                    self.translate_expression_with_annotations(condition, condition_annotation)?;
                let then_expr =
                    self.translate_expression_with_annotations(then_expression, then_annotation)?;
                let else_expr =
                    self.translate_expression_with_annotations(else_expression, else_annotation)?;
                Ok(LogicalExpression::Case {
                    operand: None,
                    when_clauses: vec![(cond, then_expr)],
                    else_clause: Some(Box::new(else_expr)),
                })
            }

            ast::Expression::Coalesce(exprs) => {
                let annotations = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Coalesce(annotations))
                        if annotations.len() == exprs.len() =>
                    {
                        Some(annotations.as_slice())
                    }
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand expression annotation mismatch at Coalesce"
                                .to_string(),
                        ));
                    }
                    None => None,
                };
                let args = exprs
                    .iter()
                    .enumerate()
                    .map(|(index, expression)| {
                        self.translate_expression_with_annotations(
                            expression,
                            annotations.map(|annotations| &annotations[index]),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::FunctionCall {
                    name: "COALESCE".to_string(),
                    args,
                    distinct: false,
                })
            }

            ast::Expression::Exists(_) | ast::Expression::NotExists(_) => {
                Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "compound or modifier EXISTS/NOT EXISTS is not yet supported by RDF execution",
                )))
            }

            ast::Expression::In { expression, list } => {
                let (expression_annotation, list_annotations) = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::In {
                        expression: annotated_expression,
                        list: annotated_list,
                    }) if annotated_list.len() == list.len() => (
                        Some(annotated_expression.as_ref()),
                        Some(annotated_list.as_slice()),
                    ),
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand expression annotation mismatch at In"
                                .to_string(),
                        ));
                    }
                    None => (None, None),
                };
                if self.exact_mutation_bindings {
                    let mut args = Vec::with_capacity(list.len() + 1);
                    args.push(self.tagged_bind_expression_with_annotations(
                        expression,
                        expression_annotation,
                    )?);
                    args.extend(
                        list.iter()
                            .enumerate()
                            .map(|(index, item)| {
                                self.tagged_bind_expression_with_annotations(
                                    item,
                                    list_annotations.map(|annotations| &annotations[index]),
                                )
                            })
                            .collect::<Result<Vec<_>>>()?,
                    );
                    return Ok(LogicalExpression::FunctionCall {
                        name: RDF_TERM_IN.to_string(),
                        args,
                        distinct: false,
                    });
                }
                let expr =
                    self.translate_expression_with_annotations(expression, expression_annotation)?;
                let items = list
                    .iter()
                    .enumerate()
                    .map(|(index, item)| {
                        self.translate_expression_with_annotations(
                            item,
                            list_annotations.map(|annotations| &annotations[index]),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::In,
                    right: Box::new(LogicalExpression::List(items)),
                })
            }

            ast::Expression::NotIn { expression, list } => {
                let (expression_annotation, list_annotations) = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::NotIn {
                        expression: annotated_expression,
                        list: annotated_list,
                    }) if annotated_list.len() == list.len() => (
                        Some(annotated_expression.as_ref()),
                        Some(annotated_list.as_slice()),
                    ),
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand expression annotation mismatch at NotIn"
                                .to_string(),
                        ));
                    }
                    None => (None, None),
                };
                if self.exact_mutation_bindings {
                    let mut args = Vec::with_capacity(list.len() + 1);
                    args.push(self.tagged_bind_expression_with_annotations(
                        expression,
                        expression_annotation,
                    )?);
                    args.extend(
                        list.iter()
                            .enumerate()
                            .map(|(index, item)| {
                                self.tagged_bind_expression_with_annotations(
                                    item,
                                    list_annotations.map(|annotations| &annotations[index]),
                                )
                            })
                            .collect::<Result<Vec<_>>>()?,
                    );
                    return Ok(LogicalExpression::Unary {
                        op: UnaryOp::Not,
                        operand: Box::new(LogicalExpression::FunctionCall {
                            name: RDF_TERM_IN.to_string(),
                            args,
                            distinct: false,
                        }),
                    });
                }
                let expr =
                    self.translate_expression_with_annotations(expression, expression_annotation)?;
                let items = list
                    .iter()
                    .enumerate()
                    .map(|(index, item)| {
                        self.translate_expression_with_annotations(
                            item,
                            list_annotations.map(|annotations| &annotations[index]),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(LogicalExpression::Binary {
                        left: Box::new(expr),
                        op: BinaryOp::In,
                        right: Box::new(LogicalExpression::List(items)),
                    }),
                })
            }

            ast::Expression::Aggregate(agg) => {
                let aggregate_annotation = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Aggregate(annotation)) => {
                        Some(annotation)
                    }
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand expression annotation mismatch at Aggregate"
                                .to_string(),
                        ));
                    }
                    None => None,
                };
                self.translate_aggregate_expression_with_annotations(agg, aggregate_annotation)
            }

            ast::Expression::Bracketed(inner) => {
                let inner_annotation = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Bracketed(inner)) => {
                        Some(inner.as_ref())
                    }
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand expression annotation mismatch at Bracketed"
                                .to_string(),
                        ));
                    }
                    None => None,
                };
                self.translate_expression_with_annotations(inner, inner_annotation)
            }
        }
    }

    fn translate_aggregate_expression_with_annotations(
        &mut self,
        agg: &ast::AggregateExpression,
        annotation: Option<&OrdinaryAggregateExactAnnotations>,
    ) -> Result<LogicalExpression> {
        let (func_name, distinct) = match agg {
            ast::AggregateExpression::Count { distinct, .. } => ("COUNT", *distinct),
            ast::AggregateExpression::Sum { distinct, .. } => ("SUM", *distinct),
            ast::AggregateExpression::Average { distinct, .. } => ("AVG", *distinct),
            ast::AggregateExpression::Minimum { .. } => ("MIN", false),
            ast::AggregateExpression::Maximum { .. } => ("MAX", false),
            ast::AggregateExpression::Sample { .. } => ("SAMPLE", false),
            ast::AggregateExpression::GroupConcat { distinct, .. } => ("GROUP_CONCAT", *distinct),
        };

        let args = match agg {
            ast::AggregateExpression::Count { expression, .. } => {
                if let Some(expr) = expression {
                    let expression_annotation = match annotation {
                        Some(OrdinaryAggregateExactAnnotations::Count(Some(annotation))) => {
                            Some(annotation.as_ref())
                        }
                        Some(_) => {
                            return Err(Error::Internal(
                                "ordinary exact-demand aggregate annotation mismatch at Count"
                                    .to_string(),
                            ));
                        }
                        None => None,
                    };
                    vec![self.translate_expression_with_annotations(expr, expression_annotation)?]
                } else {
                    vec![]
                }
            }
            ast::AggregateExpression::Sum { expression, .. }
            | ast::AggregateExpression::Average { expression, .. }
            | ast::AggregateExpression::Minimum { expression, .. }
            | ast::AggregateExpression::Maximum { expression, .. }
            | ast::AggregateExpression::Sample { expression, .. }
            | ast::AggregateExpression::GroupConcat { expression, .. } => {
                let expression_annotation = match annotation {
                    Some(OrdinaryAggregateExactAnnotations::Sum(annotation))
                    | Some(OrdinaryAggregateExactAnnotations::Average(annotation))
                    | Some(OrdinaryAggregateExactAnnotations::Minimum(annotation))
                    | Some(OrdinaryAggregateExactAnnotations::Maximum(annotation))
                    | Some(OrdinaryAggregateExactAnnotations::Sample(annotation))
                    | Some(OrdinaryAggregateExactAnnotations::GroupConcat(annotation)) => {
                        Some(annotation.as_ref())
                    }
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand aggregate annotation mismatch".to_string(),
                        ));
                    }
                    None => None,
                };
                vec![self.translate_expression_with_annotations(expression, expression_annotation)?]
            }
        };

        Ok(LogicalExpression::FunctionCall {
            name: func_name.to_string(),
            args,
            distinct,
        })
    }

    fn translate_binary_op(&self, op: ast::BinaryOperator) -> BinaryOp {
        match op {
            ast::BinaryOperator::Or => BinaryOp::Or,
            ast::BinaryOperator::And => BinaryOp::And,
            ast::BinaryOperator::Equal => BinaryOp::Eq,
            ast::BinaryOperator::NotEqual => BinaryOp::Ne,
            ast::BinaryOperator::LessThan => BinaryOp::Lt,
            ast::BinaryOperator::LessOrEqual => BinaryOp::Le,
            ast::BinaryOperator::GreaterThan => BinaryOp::Gt,
            ast::BinaryOperator::GreaterOrEqual => BinaryOp::Ge,
            ast::BinaryOperator::Add => BinaryOp::Add,
            ast::BinaryOperator::Subtract => BinaryOp::Sub,
            ast::BinaryOperator::Multiply => BinaryOp::Mul,
            ast::BinaryOperator::Divide => BinaryOp::Div,
        }
    }

    fn translate_function_name(&self, func: &ast::FunctionName) -> String {
        match func {
            ast::FunctionName::BuiltIn(builtin) => format!("{:?}", builtin).to_uppercase(),
            ast::FunctionName::Custom(iri) => self.resolve_iri(iri),
        }
    }

    fn translate_group_condition_with_annotations(
        &mut self,
        cond: &ast::GroupCondition,
        annotation: Option<&OrdinaryGroupExactAnnotation>,
    ) -> Result<LogicalExpression> {
        match cond {
            ast::GroupCondition::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),
            ast::GroupCondition::Expression { expression, alias } => {
                let expression_annotation = match annotation {
                    Some(OrdinaryGroupExactAnnotation::Expression {
                        expression: annotation,
                        ..
                    }) => Some(annotation),
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand GROUP BY annotation mismatch".to_string(),
                        ));
                    }
                    None => None,
                };
                if let Some(alias) = alias {
                    Ok(LogicalExpression::Variable(alias.clone()))
                } else {
                    self.translate_expression_with_annotations(expression, expression_annotation)
                }
            }
            ast::GroupCondition::BuiltInCall(expression) => {
                let expression_annotation = match annotation {
                    Some(OrdinaryGroupExactAnnotation::BuiltInCall(annotation)) => Some(annotation),
                    Some(_) => {
                        return Err(Error::Internal(
                            "ordinary exact-demand GROUP BY annotation mismatch".to_string(),
                        ));
                    }
                    None => None,
                };
                self.translate_expression_with_annotations(expression, expression_annotation)
            }
        }
    }

    fn extract_aggregates_for_select(
        &mut self,
        aggregate_hoist: &AggregateHoist<'_>,
    ) -> Result<Vec<AggregateExpr>> {
        aggregate_hoist
            .entries
            .iter()
            .map(|entry| {
                let annotation = Self::aggregate_hoist_representative_annotation(entry)?;
                // Validate the retained annotation against the original
                // aggregate before any consumer rewrites Aggregate -> Variable.
                Self::ordinary_aggregate_operand_annotation(entry.aggregate, annotation)?;
                let mutation_demand = self.aggregate_hoist_mutation_demand(entry);
                self.extract_aggregate(
                    entry.aggregate,
                    &entry.canonical_column,
                    annotation,
                    entry.result_demand(),
                    mutation_demand,
                )
            })
            .collect()
    }

    fn aggregate_hoist_mutation_demand(
        &self,
        entry: &AggregateHoistEntry<'_>,
    ) -> AggregateHoistMutationDemand {
        entry.occurrences.iter().fold(
            AggregateHoistMutationDemand::default(),
            |demand, occurrence| {
                let Some(alias) = occurrence.direct_alias else {
                    return demand;
                };
                AggregateHoistMutationDemand {
                    exact: demand.exact || self.mutation_requires_exact(alias),
                    control: demand.control || self.mutation_controls_selection(alias),
                }
            },
        )
    }

    fn aggregate_hoist_representative_annotation<'a>(
        entry: &'a AggregateHoistEntry<'a>,
    ) -> Result<Option<&'a OrdinaryAggregateExactAnnotations>> {
        entry
            .occurrences
            .first()
            .map(|occurrence| occurrence.annotation)
            .ok_or_else(|| {
                Error::Internal(
                    "aggregate-hoist registry entry has no representative occurrence".to_string(),
                )
            })
    }

    fn ordinary_aggregate_operand_annotation<'a>(
        aggregate: &ast::AggregateExpression,
        annotation: Option<&'a OrdinaryAggregateExactAnnotations>,
    ) -> Result<Option<&'a OrdinaryExpressionExactAnnotations>> {
        let Some(annotation) = annotation else {
            return Ok(None);
        };
        match (aggregate, annotation) {
            (
                ast::AggregateExpression::Count {
                    expression: Some(_),
                    ..
                },
                OrdinaryAggregateExactAnnotations::Count(Some(annotation)),
            ) => Ok(Some(annotation.as_ref())),
            (
                ast::AggregateExpression::Count {
                    expression: None, ..
                },
                OrdinaryAggregateExactAnnotations::Count(None),
            ) => Ok(None),
            (
                ast::AggregateExpression::Sum { .. },
                OrdinaryAggregateExactAnnotations::Sum(annotation),
            )
            | (
                ast::AggregateExpression::Average { .. },
                OrdinaryAggregateExactAnnotations::Average(annotation),
            )
            | (
                ast::AggregateExpression::Minimum { .. },
                OrdinaryAggregateExactAnnotations::Minimum(annotation),
            )
            | (
                ast::AggregateExpression::Maximum { .. },
                OrdinaryAggregateExactAnnotations::Maximum(annotation),
            )
            | (
                ast::AggregateExpression::Sample { .. },
                OrdinaryAggregateExactAnnotations::Sample(annotation),
            )
            | (
                ast::AggregateExpression::GroupConcat { .. },
                OrdinaryAggregateExactAnnotations::GroupConcat(annotation),
            ) => Ok(Some(annotation.as_ref())),
            _ => Err(Error::Internal(
                "ordinary exact-demand aggregate annotation mismatch".to_string(),
            )),
        }
    }

    /// Prepares aggregate inputs whose RDF identity affects either their
    /// output or DISTINCT membership. The tagged value is materialized once,
    /// then its visible value and canonical identity are consumed in lockstep.
    fn prepare_exact_aggregates(
        &mut self,
        mut input: LogicalOperator,
        select: &ast::SelectQuery,
        aggregate_hoist: &AggregateHoist<'_>,
        mut aggregates: Vec<AggregateExpr>,
    ) -> Result<(
        LogicalOperator,
        Vec<AggregateExpr>,
        Vec<String>,
        Vec<(String, String)>,
        Vec<(String, String)>,
    )> {
        if aggregate_hoist.entries.len() != aggregates.len() {
            return Err(Error::Internal(format!(
                "aggregate-hoist registry/physical length mismatch: registry has {}, physical lowering has {}",
                aggregate_hoist.entries.len(),
                aggregates.len()
            )));
        }

        let mut tagged_aliases = Vec::new();
        let mut rdf_or_native_aliases = Vec::new();
        let mut literal_aliases = Vec::new();
        for (aggregate_index, (entry, physical)) in aggregate_hoist
            .entries
            .iter()
            .zip(aggregates.iter_mut())
            .enumerate()
        {
            let aggregate = entry.aggregate;
            if physical.alias.as_deref() != Some(entry.canonical_column.as_str()) {
                return Err(Error::Internal(format!(
                    "aggregate-hoist registry/physical alias mismatch at index {aggregate_index}: expected {}, got {:?}",
                    entry.canonical_column, physical.alias
                )));
            }
            let annotation = Self::aggregate_hoist_representative_annotation(entry)?;
            let operand_annotation =
                Self::ordinary_aggregate_operand_annotation(aggregate, annotation)?;
            let result_demand = entry.result_demand();
            let mutation_demand = self.aggregate_hoist_mutation_demand(entry);
            let alias = physical.alias.clone();
            let exact_output_required = result_demand.exact || mutation_demand.exact;
            let controls_selection = mutation_demand.control;
            let output_identity_required =
                exact_output_required || controls_selection || result_demand.full_rdf_or_native;
            let distinct_identity_required = physical.distinct && physical.expression.is_some();
            let sample_distinct_output_required =
                matches!(aggregate, ast::AggregateExpression::Sample { .. })
                    && matches!(select.modifier, ast::SelectModifier::Distinct);
            let semantic_input_required = matches!(
                aggregate,
                ast::AggregateExpression::Sum { .. }
                    | ast::AggregateExpression::Average { .. }
                    | ast::AggregateExpression::Minimum { .. }
                    | ast::AggregateExpression::Maximum { .. }
                    | ast::AggregateExpression::GroupConcat { .. }
            );
            if !output_identity_required
                && !distinct_identity_required
                && !sample_distinct_output_required
                && !semantic_input_required
            {
                continue;
            }
            let helper_name = alias
                .clone()
                .unwrap_or_else(|| format!("anonymous-{}", aggregate_index + 1));

            match aggregate {
                ast::AggregateExpression::Sample { expression } => {
                    if !output_identity_required && !sample_distinct_output_required {
                        continue;
                    }
                    let value_expression = if matches!(&**expression, ast::Expression::Variable(_))
                    {
                        LogicalExpression::FunctionCall {
                            name: RDF_TERM_OR_NATIVE_VALUE.to_string(),
                            args: vec![self.tagged_bind_expression_with_annotations(
                                expression,
                                operand_annotation,
                            )?],
                            distinct: false,
                        }
                    } else {
                        self.rdf_term_or_native_expression_with_annotations(
                            expression,
                            operand_annotation,
                        )?
                    };
                    let value_column =
                        rdf_tagged_term_column(&format!("aggregate-input:{helper_name}"));
                    input = LogicalOperator::Bind(BindOp {
                        expression: value_expression,
                        variable: value_column.clone(),
                        input: Box::new(input),
                    });
                    let value_output =
                        rdf_tagged_term_column(&format!("aggregate-output:{helper_name}"));
                    physical.expression = Some(LogicalExpression::Variable(value_column));
                    physical.alias = Some(value_output.clone());
                    if let Some(alias) = alias {
                        rdf_or_native_aliases.push((alias, value_output));
                    }
                }
                ast::AggregateExpression::Minimum { expression }
                | ast::AggregateExpression::Maximum { expression } => {
                    let tagged_expression = match self
                        .tagged_bind_expression_with_annotations(expression, operand_annotation)
                    {
                        Ok(tagged) => tagged,
                        Err(_) if !exact_output_required => continue,
                        Err(error) => return Err(error),
                    };
                    let tagged_column =
                        rdf_tagged_term_column(&format!("aggregate-input:{helper_name}"));
                    input = LogicalOperator::Bind(BindOp {
                        expression: tagged_expression,
                        variable: tagged_column.clone(),
                        input: Box::new(input),
                    });
                    let tagged_output =
                        rdf_tagged_term_column(&format!("aggregate-output:{helper_name}"));
                    physical.expression = Some(LogicalExpression::Variable(tagged_column));
                    physical.alias = Some(tagged_output.clone());
                    if let Some(alias) = alias {
                        tagged_aliases.push((alias, tagged_output));
                    }
                }
                ast::AggregateExpression::Count {
                    distinct: true,
                    expression: Some(expression),
                } => {
                    // COUNT(DISTINCT expr) accepts both RDF terms and native
                    // extension values. Evaluate the operand once, retaining
                    // canonical RDF identity where a companion term exists and
                    // falling back row-wise to the native value otherwise.
                    let value_expression = self.rdf_term_or_native_expression_with_annotations(
                        expression,
                        operand_annotation,
                    )?;
                    let value_column =
                        rdf_tagged_term_column(&format!("aggregate-distinct-input:{helper_name}"));
                    input = LogicalOperator::Bind(BindOp {
                        expression: value_expression,
                        variable: value_column.clone(),
                        input: Box::new(input),
                    });
                    let key_column =
                        rdf_tagged_term_column(&format!("aggregate-distinct-key:{helper_name}"));
                    input = LogicalOperator::Bind(BindOp {
                        expression: LogicalExpression::FunctionCall {
                            name: RDF_DISTINCT_TERM_OR_VALUE_KEY.to_string(),
                            args: vec![LogicalExpression::Variable(value_column.clone())],
                            distinct: false,
                        },
                        variable: key_column.clone(),
                        input: Box::new(input),
                    });
                    physical.expression = Some(LogicalExpression::Variable(value_column));
                    physical.distinct_key = Some(LogicalExpression::Variable(key_column));
                    if output_identity_required && let Some(alias) = alias {
                        literal_aliases.push(alias);
                    }
                }
                ast::AggregateExpression::Sum {
                    distinct: true,
                    expression,
                }
                | ast::AggregateExpression::Average {
                    distinct: true,
                    expression,
                }
                | ast::AggregateExpression::GroupConcat {
                    distinct: true,
                    expression,
                    ..
                } => {
                    let tagged_expression = self
                        .tagged_bind_expression_with_annotations(expression, operand_annotation)?;
                    let tagged_column =
                        rdf_tagged_term_column(&format!("aggregate-distinct-input:{helper_name}"));
                    input = LogicalOperator::Bind(BindOp {
                        expression: tagged_expression,
                        variable: tagged_column.clone(),
                        input: Box::new(input),
                    });
                    let key_column =
                        rdf_tagged_term_column(&format!("aggregate-distinct-key:{helper_name}"));
                    input = LogicalOperator::Bind(BindOp {
                        expression: LogicalExpression::FunctionCall {
                            name: RDF_TERM_IDENTITY_KEY.to_string(),
                            args: vec![LogicalExpression::Variable(tagged_column.clone())],
                            distinct: false,
                        },
                        variable: key_column.clone(),
                        input: Box::new(input),
                    });
                    physical.expression = Some(LogicalExpression::Variable(tagged_column));
                    physical.distinct_key = Some(LogicalExpression::Variable(key_column));
                    if output_identity_required && let Some(alias) = alias {
                        literal_aliases.push(alias);
                    }
                }
                ast::AggregateExpression::Sum { expression, .. }
                | ast::AggregateExpression::Average { expression, .. }
                | ast::AggregateExpression::GroupConcat { expression, .. } => {
                    let tagged_expression = self
                        .tagged_bind_expression_with_annotations(expression, operand_annotation)?;
                    let tagged_column =
                        rdf_tagged_term_column(&format!("aggregate-input:{helper_name}"));
                    input = LogicalOperator::Bind(BindOp {
                        expression: tagged_expression,
                        variable: tagged_column.clone(),
                        input: Box::new(input),
                    });
                    physical.expression = Some(LogicalExpression::Variable(tagged_column));
                    if output_identity_required && let Some(alias) = alias {
                        literal_aliases.push(alias);
                    }
                }
                ast::AggregateExpression::Count { .. } => {
                    if output_identity_required && let Some(alias) = alias {
                        literal_aliases.push(alias);
                    }
                }
            }
        }
        Ok((
            input,
            aggregates,
            literal_aliases,
            tagged_aliases,
            rdf_or_native_aliases,
        ))
    }

    fn attach_literal_aggregate_identity(input: LogicalOperator, alias: &str) -> LogicalOperator {
        let tagged_column = rdf_tagged_term_column(&format!("aggregate-output:{alias}"));
        let tagged = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_TAG_LITERAL_TERM.to_string(),
                args: vec![LogicalExpression::Variable(alias.to_string())],
                distinct: false,
            },
            variable: tagged_column.clone(),
            input: Box::new(input),
        });
        let exact = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_TAG_EXACT.to_string(),
                args: vec![LogicalExpression::Variable(tagged_column.clone())],
                distinct: false,
            },
            variable: rdf_exact_term_column(alias),
            input: Box::new(tagged),
        });
        let identity = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_TERM_IDENTITY_KEY.to_string(),
                args: vec![LogicalExpression::Variable(tagged_column.clone())],
                distinct: false,
            },
            variable: rdf_identity_key_column(alias),
            input: Box::new(exact),
        });
        LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_DISTINCT_TERM_OR_VALUE_KEY.to_string(),
                args: vec![LogicalExpression::Variable(tagged_column)],
                distinct: false,
            },
            variable: rdf_group_key_column(alias),
            input: Box::new(identity),
        })
    }

    fn attach_tagged_aggregate_identity(
        input: LogicalOperator,
        alias: &str,
        tagged_column: &str,
    ) -> LogicalOperator {
        let visible = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_TAG_VALUE.to_string(),
                args: vec![LogicalExpression::Variable(tagged_column.to_string())],
                distinct: false,
            },
            variable: alias.to_string(),
            input: Box::new(input),
        });
        let exact = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_TAG_EXACT.to_string(),
                args: vec![LogicalExpression::Variable(tagged_column.to_string())],
                distinct: false,
            },
            variable: rdf_exact_term_column(alias),
            input: Box::new(visible),
        });
        let identity = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_TERM_IDENTITY_KEY.to_string(),
                args: vec![LogicalExpression::Variable(tagged_column.to_string())],
                distinct: false,
            },
            variable: rdf_identity_key_column(alias),
            input: Box::new(exact),
        });
        LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_DISTINCT_TERM_OR_VALUE_KEY.to_string(),
                args: vec![LogicalExpression::Variable(tagged_column.to_string())],
                distinct: false,
            },
            variable: rdf_group_key_column(alias),
            input: Box::new(identity),
        })
    }

    /// Unpacks a SAMPLE result that may be either a sealed RDF term or a
    /// native extension value, retaining one discriminated helper for later
    /// DISTINCT/GROUP/compatibility consumers.
    fn attach_rdf_or_native_aggregate_identity(
        input: LogicalOperator,
        alias: &str,
        value_column: &str,
    ) -> LogicalOperator {
        let visible = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_TERM_OR_NATIVE_VISIBLE.to_string(),
                args: vec![LogicalExpression::Variable(value_column.to_string())],
                distinct: false,
            },
            variable: alias.to_string(),
            input: Box::new(input),
        });
        let exact = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_TERM_OR_NATIVE_EXACT.to_string(),
                args: vec![LogicalExpression::Variable(value_column.to_string())],
                distinct: false,
            },
            variable: rdf_exact_term_column(alias),
            input: Box::new(visible),
        });
        let identity = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_TERM_IDENTITY_KEY.to_string(),
                args: vec![LogicalExpression::Variable(value_column.to_string())],
                distinct: false,
            },
            variable: rdf_identity_key_column(alias),
            input: Box::new(exact),
        });
        LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::FunctionCall {
                name: RDF_DISTINCT_TERM_OR_VALUE_KEY.to_string(),
                args: vec![LogicalExpression::Variable(value_column.to_string())],
                distinct: false,
            },
            variable: rdf_group_key_column(alias),
            input: Box::new(identity),
        })
    }

    fn collect_aggregate_hoist<'a>(
        &mut self,
        select: &'a ast::SelectQuery,
        annotations: Option<&'a OrdinarySelectExactAnnotations>,
    ) -> Result<AggregateHoist<'a>> {
        let mut hoist = AggregateHoist::default();

        // First reserve every direct projected alias. A structurally identical
        // aggregate nested in an earlier scalar expression must still use the
        // first direct public alias as its canonical column.
        if let ast::Projection::Variables(projected) = &select.projection {
            for (index, projection) in projected.iter().enumerate() {
                let ast::Expression::Aggregate(aggregate) = &projection.expression else {
                    continue;
                };
                let projection_annotation =
                    annotations.and_then(|annotations| annotations.projection.get(index));
                let aggregate_annotation =
                    match projection_annotation.map(|annotation| &annotation.expression) {
                        Some(OrdinaryExpressionExactAnnotations::Aggregate(annotation)) => {
                            Some(annotation)
                        }
                        Some(_) => {
                            return Err(Error::Internal(
                                "aggregate-hoist projection annotation mismatch".to_string(),
                            ));
                        }
                        None => None,
                    };
                Self::validate_aggregate_operand(aggregate, "SELECT")?;
                self.register_aggregate_hoist(
                    &mut hoist,
                    aggregate,
                    projection.alias.as_deref(),
                    AggregateHoistOccurrence {
                        location: AggregateHoistLocation::Projection {
                            index,
                            direct: true,
                        },
                        direct_alias: projection.alias.as_deref(),
                        demand: projection_annotation.map_or_else(
                            || AggregateHoistResultDemand::direct(false),
                            |annotation| {
                                if annotation.rdf_or_native {
                                    AggregateHoistResultDemand::FULL
                                } else {
                                    AggregateHoistResultDemand::direct(annotation.exact)
                                }
                            },
                        ),
                        annotation: aggregate_annotation,
                    },
                );
            }

            // Direct roots were already registered above. Walk only scalar
            // projections here so nested occurrences follow direct aliases.
            for (index, projection) in projected.iter().enumerate() {
                if matches!(projection.expression, ast::Expression::Aggregate(_)) {
                    continue;
                }
                let projection_annotation =
                    annotations.and_then(|annotations| annotations.projection.get(index));
                self.collect_aggregate_hoist_expression(
                    &mut hoist,
                    &projection.expression,
                    projection_annotation.map(|annotation| &annotation.expression),
                    AggregateHoistLocation::Projection {
                        index,
                        direct: false,
                    },
                    AggregateHoistResultDemand::FULL,
                    "SELECT",
                )?;
            }
        }

        if let Some(having) = &select.solution_modifiers.having {
            self.collect_aggregate_hoist_expression(
                &mut hoist,
                having,
                annotations.and_then(|annotations| annotations.having.as_ref()),
                AggregateHoistLocation::Having,
                AggregateHoistResultDemand::FULL,
                "HAVING",
            )?;
        }
        if let Some(order_by) = &select.solution_modifiers.order_by {
            for (index, condition) in order_by.iter().enumerate() {
                self.collect_aggregate_hoist_expression(
                    &mut hoist,
                    &condition.expression,
                    annotations
                        .and_then(|annotations| annotations.order_by.as_ref())
                        .and_then(|annotations| annotations.get(index)),
                    AggregateHoistLocation::OrderBy { index },
                    AggregateHoistResultDemand::FULL,
                    "ORDER BY",
                )?;
            }
        }

        Ok(hoist)
    }

    fn register_aggregate_hoist<'a>(
        &mut self,
        hoist: &mut AggregateHoist<'a>,
        aggregate: &'a ast::AggregateExpression,
        direct_alias: Option<&str>,
        occurrence: AggregateHoistOccurrence<'a>,
    ) {
        if let Some(entry) = hoist
            .entries
            .iter_mut()
            .find(|entry| entry.aggregate == aggregate)
        {
            entry.occurrences.push(occurrence);
            return;
        }

        let canonical_is_direct_projection = direct_alias.is_some();
        let canonical_column = direct_alias.map_or_else(
            || rdf_tagged_term_column(&format!("aggregate-hoist:{}", self.next_anon())),
            str::to_string,
        );
        hoist.entries.push(AggregateHoistEntry {
            aggregate,
            canonical_column,
            canonical_is_direct_projection,
            occurrences: vec![occurrence],
        });
    }

    fn collect_aggregate_hoist_expression<'a>(
        &mut self,
        hoist: &mut AggregateHoist<'a>,
        expression: &'a ast::Expression,
        annotation: Option<&'a OrdinaryExpressionExactAnnotations>,
        location: AggregateHoistLocation,
        demand: AggregateHoistResultDemand,
        clause: &str,
    ) -> Result<()> {
        use ast::Expression;

        let mismatch = || {
            Error::Internal(format!(
                "aggregate-hoist expression annotation mismatch in {clause}"
            ))
        };
        match expression {
            Expression::Aggregate(aggregate) => {
                Self::validate_aggregate_operand(aggregate, clause)?;
                let aggregate_annotation = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Aggregate(annotation)) => {
                        Some(annotation)
                    }
                    Some(_) => return Err(mismatch()),
                    None => None,
                };
                self.register_aggregate_hoist(
                    hoist,
                    aggregate,
                    None,
                    AggregateHoistOccurrence {
                        location,
                        direct_alias: None,
                        demand,
                        annotation: aggregate_annotation,
                    },
                );
            }
            Expression::Binary { left, right, .. } => {
                let (left_annotation, right_annotation) = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Binary(left, right)) => {
                        (Some(left.as_ref()), Some(right.as_ref()))
                    }
                    Some(_) => return Err(mismatch()),
                    None => (None, None),
                };
                self.collect_aggregate_hoist_expression(
                    hoist,
                    left,
                    left_annotation,
                    location,
                    demand,
                    clause,
                )?;
                self.collect_aggregate_hoist_expression(
                    hoist,
                    right,
                    right_annotation,
                    location,
                    demand,
                    clause,
                )?;
            }
            Expression::Unary { operand, .. } => {
                let operand_annotation = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Unary(annotation)) => {
                        Some(annotation.as_ref())
                    }
                    Some(_) => return Err(mismatch()),
                    None => None,
                };
                self.collect_aggregate_hoist_expression(
                    hoist,
                    operand,
                    operand_annotation,
                    location,
                    demand,
                    clause,
                )?;
            }
            Expression::FunctionCall { arguments, .. } => {
                let annotations = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::FunctionCall(annotations))
                        if annotations.len() == arguments.len() =>
                    {
                        Some(annotations.as_slice())
                    }
                    Some(_) => return Err(mismatch()),
                    None => None,
                };
                for (index, argument) in arguments.iter().enumerate() {
                    self.collect_aggregate_hoist_expression(
                        hoist,
                        argument,
                        annotations.and_then(|annotations| annotations.get(index)),
                        location,
                        demand,
                        clause,
                    )?;
                }
            }
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                let (condition_annotation, then_annotation, else_annotation) = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Conditional {
                        condition,
                        then_expression,
                        else_expression,
                    }) => (
                        Some(condition.as_ref()),
                        Some(then_expression.as_ref()),
                        Some(else_expression.as_ref()),
                    ),
                    Some(_) => return Err(mismatch()),
                    None => (None, None, None),
                };
                for (expression, annotation) in [
                    (condition.as_ref(), condition_annotation),
                    (then_expression.as_ref(), then_annotation),
                    (else_expression.as_ref(), else_annotation),
                ] {
                    self.collect_aggregate_hoist_expression(
                        hoist, expression, annotation, location, demand, clause,
                    )?;
                }
            }
            Expression::Coalesce(expressions) => {
                let annotations = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Coalesce(annotations))
                        if annotations.len() == expressions.len() =>
                    {
                        Some(annotations.as_slice())
                    }
                    Some(_) => return Err(mismatch()),
                    None => None,
                };
                for (index, expression) in expressions.iter().enumerate() {
                    self.collect_aggregate_hoist_expression(
                        hoist,
                        expression,
                        annotations.and_then(|annotations| annotations.get(index)),
                        location,
                        demand,
                        clause,
                    )?;
                }
            }
            // EXISTS owns a graph-pattern scope. Its subselects validate and
            // collect their own aggregate registries during their translation.
            Expression::Exists(_) | Expression::NotExists(_) => {}
            Expression::In { expression, list } => {
                let (expression_annotation, list_annotations) = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::In {
                        expression: annotation,
                        list: annotations,
                    }) if annotations.len() == list.len() => {
                        (Some(annotation.as_ref()), Some(annotations.as_slice()))
                    }
                    Some(_) => return Err(mismatch()),
                    None => (None, None),
                };
                self.collect_aggregate_hoist_expression(
                    hoist,
                    expression,
                    expression_annotation,
                    location,
                    demand,
                    clause,
                )?;
                for (index, item) in list.iter().enumerate() {
                    self.collect_aggregate_hoist_expression(
                        hoist,
                        item,
                        list_annotations.and_then(|annotations| annotations.get(index)),
                        location,
                        demand,
                        clause,
                    )?;
                }
            }
            Expression::NotIn { expression, list } => {
                let (expression_annotation, list_annotations) = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::NotIn {
                        expression: annotation,
                        list: annotations,
                    }) if annotations.len() == list.len() => {
                        (Some(annotation.as_ref()), Some(annotations.as_slice()))
                    }
                    Some(_) => return Err(mismatch()),
                    None => (None, None),
                };
                self.collect_aggregate_hoist_expression(
                    hoist,
                    expression,
                    expression_annotation,
                    location,
                    demand,
                    clause,
                )?;
                for (index, item) in list.iter().enumerate() {
                    self.collect_aggregate_hoist_expression(
                        hoist,
                        item,
                        list_annotations.and_then(|annotations| annotations.get(index)),
                        location,
                        demand,
                        clause,
                    )?;
                }
            }
            Expression::Bracketed(inner) => {
                let inner_annotation = match annotation {
                    Some(OrdinaryExpressionExactAnnotations::Bracketed(annotation)) => {
                        Some(annotation.as_ref())
                    }
                    Some(_) => return Err(mismatch()),
                    None => None,
                };
                self.collect_aggregate_hoist_expression(
                    hoist,
                    inner,
                    inner_annotation,
                    location,
                    demand,
                    clause,
                )?;
            }
            Expression::Variable(_)
            | Expression::Iri(_)
            | Expression::Literal(_)
            | Expression::Bound(_) => {}
        }
        Ok(())
    }

    /// Recursively checks if an expression contains any aggregate function.
    fn contains_aggregate(expr: &ast::Expression) -> bool {
        match expr {
            ast::Expression::Aggregate(_) => true,
            ast::Expression::Binary { left, right, .. } => {
                Self::contains_aggregate(left) || Self::contains_aggregate(right)
            }
            ast::Expression::Unary { operand, .. } => Self::contains_aggregate(operand),
            ast::Expression::FunctionCall { arguments, .. } => {
                arguments.iter().any(Self::contains_aggregate)
            }
            ast::Expression::Bracketed(inner) => Self::contains_aggregate(inner),
            ast::Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                Self::contains_aggregate(condition)
                    || Self::contains_aggregate(then_expression)
                    || Self::contains_aggregate(else_expression)
            }
            ast::Expression::Coalesce(exprs) => exprs.iter().any(Self::contains_aggregate),
            ast::Expression::In { expression, list }
            | ast::Expression::NotIn { expression, list } => {
                Self::contains_aggregate(expression) || list.iter().any(Self::contains_aggregate)
            }
            ast::Expression::Exists(pattern) | ast::Expression::NotExists(pattern) => {
                Self::graph_pattern_contains_aggregate(pattern)
            }
            ast::Expression::Variable(_)
            | ast::Expression::Iri(_)
            | ast::Expression::Literal(_)
            | ast::Expression::Bound(_) => false,
        }
    }

    fn graph_pattern_contains_aggregate(pattern: &ast::GraphPattern) -> bool {
        match pattern {
            ast::GraphPattern::Group(patterns) | ast::GraphPattern::Union(patterns) => {
                patterns.iter().any(Self::graph_pattern_contains_aggregate)
            }
            ast::GraphPattern::Optional(pattern)
            | ast::GraphPattern::Minus(pattern)
            | ast::GraphPattern::NamedGraph { pattern, .. }
            | ast::GraphPattern::Service { pattern, .. } => {
                Self::graph_pattern_contains_aggregate(pattern)
            }
            ast::GraphPattern::Filter(expression) | ast::GraphPattern::Bind { expression, .. } => {
                Self::contains_aggregate(expression)
            }
            // A subselect owns an independent aggregate scope and validates
            // itself when its translation begins.
            ast::GraphPattern::SubSelect(_)
            | ast::GraphPattern::Basic(_)
            | ast::GraphPattern::InlineData(_) => false,
        }
    }

    fn aggregate_operand_contains_aggregate(aggregate: &ast::AggregateExpression) -> bool {
        match aggregate {
            ast::AggregateExpression::Count { expression, .. } => expression
                .as_deref()
                .is_some_and(Self::contains_aggregate_in_current_scope),
            ast::AggregateExpression::Sum { expression, .. }
            | ast::AggregateExpression::Average { expression, .. }
            | ast::AggregateExpression::Minimum { expression }
            | ast::AggregateExpression::Maximum { expression }
            | ast::AggregateExpression::Sample { expression }
            | ast::AggregateExpression::GroupConcat { expression, .. } => {
                Self::contains_aggregate_in_current_scope(expression)
            }
        }
    }

    fn contains_aggregate_in_current_scope(expression: &ast::Expression) -> bool {
        use ast::Expression;

        match expression {
            Expression::Aggregate(_) => true,
            Expression::Binary { left, right, .. } => {
                Self::contains_aggregate_in_current_scope(left)
                    || Self::contains_aggregate_in_current_scope(right)
            }
            Expression::Unary { operand, .. } | Expression::Bracketed(operand) => {
                Self::contains_aggregate_in_current_scope(operand)
            }
            Expression::FunctionCall { arguments, .. } | Expression::Coalesce(arguments) => {
                arguments
                    .iter()
                    .any(Self::contains_aggregate_in_current_scope)
            }
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                Self::contains_aggregate_in_current_scope(condition)
                    || Self::contains_aggregate_in_current_scope(then_expression)
                    || Self::contains_aggregate_in_current_scope(else_expression)
            }
            Expression::In { expression, list } | Expression::NotIn { expression, list } => {
                Self::contains_aggregate_in_current_scope(expression)
                    || list.iter().any(Self::contains_aggregate_in_current_scope)
            }
            // EXISTS owns an independent graph-pattern/subselect scope.
            Expression::Exists(_) | Expression::NotExists(_) => false,
            Expression::Variable(_)
            | Expression::Iri(_)
            | Expression::Literal(_)
            | Expression::Bound(_) => false,
        }
    }

    /// Whether this SELECT creates a group solution sequence in its own
    /// lexical scope. WHERE/EXISTS subqueries own separate aggregate scopes.
    fn select_is_grouped(select: &ast::SelectQuery) -> bool {
        select.solution_modifiers.group_by.is_some()
            || matches!(&select.projection, ast::Projection::Variables(projected)
            if projected.iter().any(|item| {
                Self::contains_aggregate_in_current_scope(&item.expression)
            }))
            || select
                .solution_modifiers
                .having
                .as_ref()
                .is_some_and(Self::contains_aggregate_in_current_scope)
            || select
                .solution_modifiers
                .order_by
                .iter()
                .flatten()
                .any(|condition| Self::contains_aggregate_in_current_scope(&condition.expression))
    }

    fn validate_aggregate_operand(
        aggregate: &ast::AggregateExpression,
        clause: &str,
    ) -> Result<()> {
        if Self::aggregate_operand_contains_aggregate(aggregate) {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!("{clause} contains a nested aggregate in a set-function operand"),
            )));
        }
        Ok(())
    }

    fn validate_aggregate_nesting(expression: &ast::Expression, clause: &str) -> Result<()> {
        use ast::Expression;

        match expression {
            Expression::Aggregate(aggregate) => Self::validate_aggregate_operand(aggregate, clause),
            Expression::Binary { left, right, .. } => {
                Self::validate_aggregate_nesting(left, clause)?;
                Self::validate_aggregate_nesting(right, clause)
            }
            Expression::Unary { operand, .. } | Expression::Bracketed(operand) => {
                Self::validate_aggregate_nesting(operand, clause)
            }
            Expression::FunctionCall { arguments, .. } | Expression::Coalesce(arguments) => {
                for argument in arguments {
                    Self::validate_aggregate_nesting(argument, clause)?;
                }
                Ok(())
            }
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                Self::validate_aggregate_nesting(condition, clause)?;
                Self::validate_aggregate_nesting(then_expression, clause)?;
                Self::validate_aggregate_nesting(else_expression, clause)
            }
            Expression::In { expression, list } | Expression::NotIn { expression, list } => {
                Self::validate_aggregate_nesting(expression, clause)?;
                for item in list {
                    Self::validate_aggregate_nesting(item, clause)?;
                }
                Ok(())
            }
            Expression::Exists(_) | Expression::NotExists(_) => Ok(()),
            Expression::Variable(_)
            | Expression::Iri(_)
            | Expression::Literal(_)
            | Expression::Bound(_) => Ok(()),
        }
    }

    fn row_scope_aggregate_clause(pattern: &ast::GraphPattern) -> Option<&'static str> {
        match pattern {
            ast::GraphPattern::Group(patterns) | ast::GraphPattern::Union(patterns) => {
                patterns.iter().find_map(Self::row_scope_aggregate_clause)
            }
            ast::GraphPattern::Optional(pattern)
            | ast::GraphPattern::Minus(pattern)
            | ast::GraphPattern::NamedGraph { pattern, .. }
            | ast::GraphPattern::Service { pattern, .. } => {
                Self::row_scope_aggregate_clause(pattern)
            }
            ast::GraphPattern::Filter(expression) => {
                if Self::contains_aggregate_in_current_scope(expression) {
                    Some("FILTER")
                } else {
                    Self::nested_row_scope_aggregate_clause(expression)
                }
            }
            ast::GraphPattern::Bind { expression, .. } => {
                if Self::contains_aggregate_in_current_scope(expression) {
                    Some("BIND")
                } else {
                    Self::nested_row_scope_aggregate_clause(expression)
                }
            }
            ast::GraphPattern::SubSelect(_)
            | ast::GraphPattern::Basic(_)
            | ast::GraphPattern::InlineData(_) => None,
        }
    }

    /// Finds illegal row-scope aggregates inside EXISTS/NOT EXISTS graph
    /// patterns while leaving actual subselects to their own SELECT validator.
    /// Unlike aggregate collection, legality validation must enter EXISTS.
    fn nested_row_scope_aggregate_clause(expression: &ast::Expression) -> Option<&'static str> {
        use ast::Expression;

        match expression {
            Expression::Exists(pattern) | Expression::NotExists(pattern) => {
                Self::row_scope_aggregate_clause(pattern)
            }
            Expression::Binary { left, right, .. } => Self::nested_row_scope_aggregate_clause(left)
                .or_else(|| Self::nested_row_scope_aggregate_clause(right)),
            Expression::Unary { operand, .. } | Expression::Bracketed(operand) => {
                Self::nested_row_scope_aggregate_clause(operand)
            }
            Expression::FunctionCall { arguments, .. } | Expression::Coalesce(arguments) => {
                arguments
                    .iter()
                    .find_map(Self::nested_row_scope_aggregate_clause)
            }
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => Self::nested_row_scope_aggregate_clause(condition)
                .or_else(|| Self::nested_row_scope_aggregate_clause(then_expression))
                .or_else(|| Self::nested_row_scope_aggregate_clause(else_expression)),
            Expression::In { expression, list } | Expression::NotIn { expression, list } => {
                Self::nested_row_scope_aggregate_clause(expression).or_else(|| {
                    list.iter()
                        .find_map(Self::nested_row_scope_aggregate_clause)
                })
            }
            Expression::Aggregate(aggregate) => match aggregate {
                ast::AggregateExpression::Count { expression, .. } => expression
                    .as_deref()
                    .and_then(Self::nested_row_scope_aggregate_clause),
                ast::AggregateExpression::Sum { expression, .. }
                | ast::AggregateExpression::Average { expression, .. }
                | ast::AggregateExpression::Minimum { expression }
                | ast::AggregateExpression::Maximum { expression }
                | ast::AggregateExpression::Sample { expression }
                | ast::AggregateExpression::GroupConcat { expression, .. } => {
                    Self::nested_row_scope_aggregate_clause(expression)
                }
            },
            Expression::Variable(_)
            | Expression::Iri(_)
            | Expression::Literal(_)
            | Expression::Bound(_) => None,
        }
    }

    fn validate_nested_row_scope_aggregates(expression: &ast::Expression) -> Result<()> {
        if let Some(clause) = Self::nested_row_scope_aggregate_clause(expression) {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!("aggregates are not allowed in {clause}"),
            )));
        }
        Ok(())
    }

    /// Enforces SPARQL's assignment-target rule before aliases become physical
    /// columns, preventing duplicate names from being resolved by HashMap order.
    fn validate_select_assignment_targets(select: &ast::SelectQuery) -> Result<()> {
        let mut occupied = HashSet::new();
        Self::collect_pattern_output_variables(&select.where_clause, &mut occupied);

        let claim = |occupied: &mut HashSet<String>, alias: &str, clause: &str| {
            if occupied.insert(alias.to_string()) {
                Ok(())
            } else {
                Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!("{clause} assignment target ?{alias} is already in scope"),
                )))
            }
        };

        if let Some(group_by) = &select.solution_modifiers.group_by {
            for condition in group_by {
                if let ast::GroupCondition::Expression {
                    alias: Some(alias), ..
                } = condition
                {
                    claim(&mut occupied, alias, "GROUP BY")?;
                }
            }
        }
        if let ast::Projection::Variables(projected) = &select.projection {
            for projection in projected {
                if let Some(alias) = &projection.alias {
                    claim(&mut occupied, alias, "SELECT")?;
                }
            }
        }
        Ok(())
    }

    /// Enforces permanent set-function placement rules. SELECT, HAVING, and
    /// ORDER BY are legal consumers in this scope; row/group expressions and
    /// nested set-function operands are not.
    fn validate_supported_aggregate_placements(select: &ast::SelectQuery) -> Result<()> {
        if let Some(clause) = Self::row_scope_aggregate_clause(&select.where_clause) {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!("aggregates are not allowed in {clause}"),
            )));
        }

        if select
            .solution_modifiers
            .group_by
            .iter()
            .flatten()
            .any(|condition| match condition {
                ast::GroupCondition::Variable(_) => false,
                ast::GroupCondition::Expression { expression, .. }
                | ast::GroupCondition::BuiltInCall(expression) => {
                    Self::contains_aggregate_in_current_scope(expression)
                }
            })
        {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "aggregates are not allowed in GROUP BY",
            )));
        }
        for condition in select.solution_modifiers.group_by.iter().flatten() {
            if let ast::GroupCondition::Expression { expression, .. }
            | ast::GroupCondition::BuiltInCall(expression) = condition
            {
                Self::validate_nested_row_scope_aggregates(expression)?;
            }
        }

        if let ast::Projection::Variables(projected) = &select.projection {
            for projection in projected {
                Self::validate_nested_row_scope_aggregates(&projection.expression)?;
                Self::validate_aggregate_nesting(&projection.expression, "SELECT")?;
            }
        }
        if let Some(having) = &select.solution_modifiers.having {
            Self::validate_nested_row_scope_aggregates(having)?;
            Self::validate_aggregate_nesting(having, "HAVING")?;
        }
        if let Some(order_by) = &select.solution_modifiers.order_by {
            for condition in order_by {
                Self::validate_nested_row_scope_aggregates(&condition.expression)?;
                Self::validate_aggregate_nesting(&condition.expression, "ORDER BY")?;
            }
        }
        Ok(())
    }

    fn extract_aggregate(
        &mut self,
        aggregate: &ast::AggregateExpression,
        alias: &str,
        annotation: Option<&OrdinaryAggregateExactAnnotations>,
        result_demand: AggregateHoistResultDemand,
        mutation_demand: AggregateHoistMutationDemand,
    ) -> Result<AggregateExpr> {
        let (func, expr_inner, distinct, separator) = match aggregate {
            ast::AggregateExpression::Count {
                distinct,
                expression,
            } => {
                // COUNT(?expr) uses CountNonNull to skip NULLs;
                // COUNT(*) (no expression) uses Count to count all rows.
                let func = if expression.is_some() {
                    AggregateFunction::CountNonNull
                } else {
                    AggregateFunction::Count
                };
                (
                    func,
                    expression.as_ref().map(|e| e.as_ref()),
                    *distinct,
                    None,
                )
            }
            ast::AggregateExpression::Sum {
                distinct,
                expression,
            } => (
                AggregateFunction::Sum,
                Some(expression.as_ref()),
                *distinct,
                None,
            ),
            ast::AggregateExpression::Average {
                distinct,
                expression,
            } => (
                AggregateFunction::Avg,
                Some(expression.as_ref()),
                *distinct,
                None,
            ),
            ast::AggregateExpression::Minimum { expression } => (
                AggregateFunction::Min,
                Some(expression.as_ref()),
                false,
                None,
            ),
            ast::AggregateExpression::Maximum { expression } => (
                AggregateFunction::Max,
                Some(expression.as_ref()),
                false,
                None,
            ),
            ast::AggregateExpression::Sample { expression } => (
                AggregateFunction::Sample,
                Some(expression.as_ref()),
                false,
                None,
            ),
            ast::AggregateExpression::GroupConcat {
                distinct,
                expression,
                separator,
            } => (
                AggregateFunction::GroupConcat,
                Some(expression.as_ref()),
                *distinct,
                separator.clone(),
            ),
        };

        let exact_result_required =
            result_demand.exact || mutation_demand.exact || mutation_demand.control;
        let expression = if let Some(e) = expr_inner {
            let operand_annotation =
                Self::ordinary_aggregate_operand_annotation(aggregate, annotation)?;
            Some(if self.exact_mutation_bindings && !exact_result_required {
                self.translate_ordinary_expression_with_annotations(e, operand_annotation)?
            } else {
                self.translate_expression_with_annotations(e, operand_annotation)?
            })
        } else {
            None
        };

        Ok(AggregateExpr {
            function: func,
            expression,
            expression2: None,
            distinct_key: None,
            distinct,
            alias: Some(alias.to_string()),
            percentile: None, // SPARQL doesn't support percentile functions
            separator,
        })
    }

    fn rewrite_having_aggregates(
        expression: &ast::Expression,
        projection: &ast::Projection,
    ) -> ast::Expression {
        use ast::Expression;

        match expression {
            Expression::Aggregate(aggregate) => {
                Self::aggregate_projection_alias(aggregate, projection).map_or_else(
                    || expression.clone(),
                    |alias| Expression::Variable(alias.to_string()),
                )
            }
            Expression::Binary {
                left,
                operator,
                right,
            } => Expression::Binary {
                left: Box::new(Self::rewrite_having_aggregates(left, projection)),
                operator: *operator,
                right: Box::new(Self::rewrite_having_aggregates(right, projection)),
            },
            Expression::Unary { operator, operand } => Expression::Unary {
                operator: *operator,
                operand: Box::new(Self::rewrite_having_aggregates(operand, projection)),
            },
            Expression::FunctionCall {
                function,
                arguments,
            } => Expression::FunctionCall {
                function: function.clone(),
                arguments: arguments
                    .iter()
                    .map(|argument| Self::rewrite_having_aggregates(argument, projection))
                    .collect(),
            },
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => Expression::Conditional {
                condition: Box::new(Self::rewrite_having_aggregates(condition, projection)),
                then_expression: Box::new(Self::rewrite_having_aggregates(
                    then_expression,
                    projection,
                )),
                else_expression: Box::new(Self::rewrite_having_aggregates(
                    else_expression,
                    projection,
                )),
            },
            Expression::Coalesce(expressions) => Expression::Coalesce(
                expressions
                    .iter()
                    .map(|expression| Self::rewrite_having_aggregates(expression, projection))
                    .collect(),
            ),
            Expression::In { expression, list } => Expression::In {
                expression: Box::new(Self::rewrite_having_aggregates(expression, projection)),
                list: list
                    .iter()
                    .map(|item| Self::rewrite_having_aggregates(item, projection))
                    .collect(),
            },
            Expression::NotIn { expression, list } => Expression::NotIn {
                expression: Box::new(Self::rewrite_having_aggregates(expression, projection)),
                list: list
                    .iter()
                    .map(|item| Self::rewrite_having_aggregates(item, projection))
                    .collect(),
            },
            Expression::Bracketed(inner) => {
                Expression::Bracketed(Box::new(Self::rewrite_having_aggregates(inner, projection)))
            }
            Expression::Variable(_)
            | Expression::Iri(_)
            | Expression::Literal(_)
            | Expression::Bound(_)
            | Expression::Exists(_)
            | Expression::NotExists(_) => expression.clone(),
        }
    }

    fn rewrite_registered_aggregates(
        expression: &ast::Expression,
        aggregate_hoist: &AggregateHoist<'_>,
    ) -> ast::Expression {
        use ast::Expression;

        match expression {
            Expression::Aggregate(aggregate) => aggregate_hoist
                .entries
                .iter()
                .find(|entry| entry.aggregate == aggregate)
                .map_or_else(
                    || expression.clone(),
                    |entry| Expression::Variable(entry.canonical_column.clone()),
                ),
            Expression::Binary {
                left,
                operator,
                right,
            } => Expression::Binary {
                left: Box::new(Self::rewrite_registered_aggregates(left, aggregate_hoist)),
                operator: *operator,
                right: Box::new(Self::rewrite_registered_aggregates(right, aggregate_hoist)),
            },
            Expression::Unary { operator, operand } => Expression::Unary {
                operator: *operator,
                operand: Box::new(Self::rewrite_registered_aggregates(
                    operand,
                    aggregate_hoist,
                )),
            },
            Expression::FunctionCall {
                function,
                arguments,
            } => Expression::FunctionCall {
                function: function.clone(),
                arguments: arguments
                    .iter()
                    .map(|argument| Self::rewrite_registered_aggregates(argument, aggregate_hoist))
                    .collect(),
            },
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => Expression::Conditional {
                condition: Box::new(Self::rewrite_registered_aggregates(
                    condition,
                    aggregate_hoist,
                )),
                then_expression: Box::new(Self::rewrite_registered_aggregates(
                    then_expression,
                    aggregate_hoist,
                )),
                else_expression: Box::new(Self::rewrite_registered_aggregates(
                    else_expression,
                    aggregate_hoist,
                )),
            },
            Expression::Coalesce(expressions) => Expression::Coalesce(
                expressions
                    .iter()
                    .map(|expression| {
                        Self::rewrite_registered_aggregates(expression, aggregate_hoist)
                    })
                    .collect(),
            ),
            Expression::In { expression, list } => Expression::In {
                expression: Box::new(Self::rewrite_registered_aggregates(
                    expression,
                    aggregate_hoist,
                )),
                list: list
                    .iter()
                    .map(|item| Self::rewrite_registered_aggregates(item, aggregate_hoist))
                    .collect(),
            },
            Expression::NotIn { expression, list } => Expression::NotIn {
                expression: Box::new(Self::rewrite_registered_aggregates(
                    expression,
                    aggregate_hoist,
                )),
                list: list
                    .iter()
                    .map(|item| Self::rewrite_registered_aggregates(item, aggregate_hoist))
                    .collect(),
            },
            Expression::Bracketed(inner) => Expression::Bracketed(Box::new(
                Self::rewrite_registered_aggregates(inner, aggregate_hoist),
            )),
            Expression::Variable(_)
            | Expression::Iri(_)
            | Expression::Literal(_)
            | Expression::Bound(_)
            | Expression::Exists(_)
            | Expression::NotExists(_) => expression.clone(),
        }
    }

    /// Replaces every structurally identical GROUP BY expression with the
    /// column that was evaluated before aggregation. This is both a scope rule
    /// and an evaluation rule: volatile or erroring group expressions must not
    /// be run again while projecting, filtering, or ordering grouped rows.
    fn rewrite_grouped_expressions(
        expression: &ast::Expression,
        grouped_expression_sources: &[(ast::Expression, String)],
    ) -> ast::Expression {
        use ast::Expression;

        if let Some(source) = grouped_expression_sources
            .iter()
            .find_map(|(grouped, source)| {
                Self::group_expressions_equivalent(grouped, expression).then_some(source)
            })
        {
            return Expression::Variable(source.clone());
        }

        match expression {
            Expression::Binary {
                left,
                operator,
                right,
            } => Expression::Binary {
                left: Box::new(Self::rewrite_grouped_expressions(
                    left,
                    grouped_expression_sources,
                )),
                operator: *operator,
                right: Box::new(Self::rewrite_grouped_expressions(
                    right,
                    grouped_expression_sources,
                )),
            },
            Expression::Unary { operator, operand } => Expression::Unary {
                operator: *operator,
                operand: Box::new(Self::rewrite_grouped_expressions(
                    operand,
                    grouped_expression_sources,
                )),
            },
            Expression::FunctionCall {
                function,
                arguments,
            } => Expression::FunctionCall {
                function: function.clone(),
                arguments: arguments
                    .iter()
                    .map(|argument| {
                        Self::rewrite_grouped_expressions(argument, grouped_expression_sources)
                    })
                    .collect(),
            },
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => Expression::Conditional {
                condition: Box::new(Self::rewrite_grouped_expressions(
                    condition,
                    grouped_expression_sources,
                )),
                then_expression: Box::new(Self::rewrite_grouped_expressions(
                    then_expression,
                    grouped_expression_sources,
                )),
                else_expression: Box::new(Self::rewrite_grouped_expressions(
                    else_expression,
                    grouped_expression_sources,
                )),
            },
            Expression::Coalesce(expressions) => Expression::Coalesce(
                expressions
                    .iter()
                    .map(|expression| {
                        Self::rewrite_grouped_expressions(expression, grouped_expression_sources)
                    })
                    .collect(),
            ),
            Expression::In { expression, list } => Expression::In {
                expression: Box::new(Self::rewrite_grouped_expressions(
                    expression,
                    grouped_expression_sources,
                )),
                list: list
                    .iter()
                    .map(|item| Self::rewrite_grouped_expressions(item, grouped_expression_sources))
                    .collect(),
            },
            Expression::NotIn { expression, list } => Expression::NotIn {
                expression: Box::new(Self::rewrite_grouped_expressions(
                    expression,
                    grouped_expression_sources,
                )),
                list: list
                    .iter()
                    .map(|item| Self::rewrite_grouped_expressions(item, grouped_expression_sources))
                    .collect(),
            },
            Expression::Bracketed(inner) => Expression::Bracketed(Box::new(
                Self::rewrite_grouped_expressions(inner, grouped_expression_sources),
            )),
            Expression::Variable(_)
            | Expression::Iri(_)
            | Expression::Literal(_)
            | Expression::Bound(_)
            | Expression::Exists(_)
            | Expression::NotExists(_)
            | Expression::Aggregate(_) => expression.clone(),
        }
    }

    fn join_patterns(&self, left: LogicalOperator, right: LogicalOperator) -> LogicalOperator {
        if matches!(left, LogicalOperator::Empty) {
            return right;
        }
        if matches!(right, LogicalOperator::Empty) {
            return left;
        }

        // Plain variable expressions remain optimizer edges. Typed semantics
        // separately tell the RDF planner when the canonical exact companion
        // is the physical equality key.
        let conditions = Self::operator_compatibility_conditions(&left, &right);

        LogicalOperator::Join(JoinOp {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
            conditions,
        })
    }

    fn left_join_patterns(&self, left: LogicalOperator, right: LogicalOperator) -> LogicalOperator {
        let compatibility_conditions = Self::operator_compatibility_conditions(&left, &right);
        LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(left),
            right: Box::new(right),
            condition: None,
            compatibility_conditions,
        })
    }

    fn semi_join_patterns(&self, left: LogicalOperator, right: LogicalOperator) -> LogicalOperator {
        let conditions = Self::operator_compatibility_conditions(&left, &right);
        LogicalOperator::Join(JoinOp {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Semi,
            conditions,
        })
    }

    fn anti_join_patterns(
        &self,
        left: LogicalOperator,
        right: LogicalOperator,
        semantics: AntiJoinSemantics,
    ) -> LogicalOperator {
        let compatibility_conditions = Self::operator_compatibility_conditions(&left, &right);
        LogicalOperator::AntiJoin(AntiJoinOp {
            left: Box::new(left),
            right: Box::new(right),
            compatibility_conditions,
            semantics,
        })
    }

    fn operator_compatibility_conditions(
        left: &LogicalOperator,
        right: &LogicalOperator,
    ) -> Vec<JoinCondition> {
        let left_vars = Self::collect_operator_variables(left);
        let right_vars = Self::collect_operator_variables(right);
        let right_vars_set = right_vars.iter().cloned().collect::<HashSet<_>>();
        let left_must_bind = Self::collect_operator_must_bound_variables(left);
        let right_must_bind = Self::collect_operator_must_bound_variables(right);

        left_vars
            .into_iter()
            .filter(|variable| {
                !is_rdf_internal_term_column(variable) && right_vars_set.contains(variable)
            })
            .map(|variable| JoinCondition {
                left: LogicalExpression::Variable(variable.clone()),
                right: LogicalExpression::Variable(variable.clone()),
                semantics: if left_must_bind.contains(&variable)
                    && right_must_bind.contains(&variable)
                {
                    JoinKeySemantics::RdfTermIdentity
                } else {
                    JoinKeySemantics::SparqlCompatibility
                },
            })
            .collect()
    }

    /// Collects the actual output schema of an operator subtree.
    ///
    /// Projection is a lexical boundary: only projected names (plus explicit
    /// pass-through) survive. Cardinality-only wrappers preserve their input
    /// shape, UNION takes the schema union, and anti joins return only the left
    /// schema. This avoids phantom compatibility keys from hidden subselect
    /// variables while retaining aliases through DISTINCT/SORT/LIMIT.
    fn collect_operator_variables(op: &LogicalOperator) -> Vec<String> {
        fn component_variable(component: &TripleComponent) -> Option<String> {
            match component {
                TripleComponent::Variable(variable) => Some(variable.clone()),
                _ => None,
            }
        }

        fn visit(op: &LogicalOperator) -> HashSet<String> {
            match op {
                LogicalOperator::TripleScan(scan) => {
                    let mut variables = scan.input.as_deref().map_or_else(HashSet::new, visit);
                    variables.extend(
                        [
                            Some(&scan.subject),
                            Some(&scan.predicate),
                            Some(&scan.object),
                            scan.graph.as_ref(),
                        ]
                        .into_iter()
                        .flatten()
                        .filter_map(component_variable),
                    );
                    variables
                }
                LogicalOperator::PropertyPath(path) => [&path.subject, &path.object]
                    .into_iter()
                    .filter_map(component_variable)
                    .collect(),
                LogicalOperator::Join(join) => match join.join_type {
                    JoinType::Semi | JoinType::Anti => visit(&join.left),
                    JoinType::Right => visit(&join.right),
                    JoinType::Inner | JoinType::Cross | JoinType::Left | JoinType::Full => {
                        let mut variables = visit(&join.left);
                        variables.extend(visit(&join.right));
                        variables
                    }
                },
                LogicalOperator::LeftJoin(join) => {
                    let mut variables = visit(&join.left);
                    variables.extend(visit(&join.right));
                    variables
                }
                LogicalOperator::AntiJoin(join) => visit(&join.left),
                LogicalOperator::Union(union) => {
                    let mut variables = HashSet::new();
                    for input in &union.inputs {
                        variables.extend(visit(input));
                    }
                    variables
                }
                LogicalOperator::Filter(filter) => visit(&filter.input),
                LogicalOperator::Bind(bind) => {
                    let mut variables = visit(&bind.input);
                    variables.insert(bind.variable.clone());
                    variables
                }
                LogicalOperator::Project(project) => {
                    let input = visit(&project.input);
                    let mut variables = if project.pass_through_input {
                        input
                    } else {
                        HashSet::new()
                    };
                    for projection in &project.projections {
                        if let Some(alias) = &projection.alias {
                            variables.insert(alias.clone());
                        } else if let LogicalExpression::Variable(variable) = &projection.expression
                        {
                            variables.insert(variable.clone());
                        }
                    }
                    variables
                }
                LogicalOperator::Distinct(distinct) => visit(&distinct.input),
                LogicalOperator::Sort(sort) => visit(&sort.input),
                LogicalOperator::Limit(limit) => visit(&limit.input),
                LogicalOperator::Skip(skip) => visit(&skip.input),
                LogicalOperator::Aggregate(aggregate) => {
                    let mut variables = HashSet::new();
                    for expression in &aggregate.group_by {
                        if let LogicalExpression::Variable(variable) = expression {
                            variables.insert(variable.clone());
                        }
                    }
                    for aggregate in &aggregate.aggregates {
                        if let Some(alias) = &aggregate.alias {
                            variables.insert(alias.clone());
                        }
                    }
                    variables
                }
                LogicalOperator::Return(ret) => ret
                    .items
                    .iter()
                    .filter_map(|item| {
                        item.alias.clone().or_else(|| match &item.expression {
                            LogicalExpression::Variable(variable) => Some(variable.clone()),
                            _ => None,
                        })
                    })
                    .collect(),
                LogicalOperator::Unwind(unwind) => {
                    let mut variables = visit(&unwind.input);
                    variables.insert(unwind.variable.clone());
                    variables.extend(unwind.ordinality_var.iter().cloned());
                    variables.extend(unwind.offset_var.iter().cloned());
                    variables
                }
                _ => HashSet::new(),
            }
        }

        let mut variables = visit(op).into_iter().collect::<Vec<_>>();
        variables.sort_unstable();
        variables
    }

    /// Returns only variables whose binding is guaranteed on every solution.
    /// This is deliberately stricter than output-shape discovery: UNION takes
    /// an intersection, OPTIONAL/MINUS retain only the left facts, and BIND
    /// does not add a fact because expression errors leave it unbound.
    fn collect_operator_must_bound_variables(op: &LogicalOperator) -> HashSet<String> {
        fn expression_is_total(
            expression: &LogicalExpression,
            must_bound: &HashSet<String>,
        ) -> bool {
            match expression {
                LogicalExpression::Literal(value) => !matches!(value, Value::Null),
                LogicalExpression::Variable(variable) => must_bound.contains(variable),
                LogicalExpression::FunctionCall { name, args, .. }
                    if matches!(
                        name.as_str(),
                        RDF_TAG_BOUND_TERM
                            | RDF_TAG_VALUE
                            | RDF_TAG_EXACT
                            | RDF_TAG_IRI_TERM
                            | RDF_TAG_BLANK_TERM
                            | RDF_TAG_LITERAL_TERM
                    ) =>
                {
                    !args.is_empty()
                        && args
                            .iter()
                            .all(|argument| expression_is_total(argument, must_bound))
                }
                LogicalExpression::FunctionCall { name, args, .. } if name == "COALESCE" => args
                    .iter()
                    .any(|argument| expression_is_total(argument, must_bound)),
                _ => false,
            }
        }

        fn visit(op: &LogicalOperator) -> HashSet<String> {
            match op {
                LogicalOperator::TripleScan(scan) => {
                    let mut variables = scan.input.as_deref().map_or_else(HashSet::new, visit);
                    for component in [
                        Some(&scan.subject),
                        Some(&scan.predicate),
                        Some(&scan.object),
                        scan.graph.as_ref(),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        if let TripleComponent::Variable(variable) = component {
                            variables.insert(variable.clone());
                        }
                    }
                    variables
                }
                LogicalOperator::PropertyPath(path) => [&path.subject, &path.object]
                    .into_iter()
                    .filter_map(|component| match component {
                        TripleComponent::Variable(variable) => Some(variable.clone()),
                        _ => None,
                    })
                    .collect(),
                LogicalOperator::Join(join) => match join.join_type {
                    JoinType::Inner | JoinType::Cross => {
                        let mut variables = visit(&join.left);
                        variables.extend(visit(&join.right));
                        variables
                    }
                    JoinType::Left | JoinType::Semi | JoinType::Anti => visit(&join.left),
                    JoinType::Right => visit(&join.right),
                    JoinType::Full => HashSet::new(),
                },
                LogicalOperator::LeftJoin(join) => visit(&join.left),
                LogicalOperator::AntiJoin(join) => visit(&join.left),
                LogicalOperator::Union(union) => {
                    let mut inputs = union.inputs.iter();
                    let Some(first) = inputs.next() else {
                        return HashSet::new();
                    };
                    let mut variables = visit(first);
                    for input in inputs {
                        let branch = visit(input);
                        variables.retain(|variable| branch.contains(variable));
                    }
                    variables
                }
                LogicalOperator::Filter(filter) => visit(&filter.input),
                LogicalOperator::Bind(bind) => {
                    let mut variables = visit(&bind.input);
                    if expression_is_total(&bind.expression, &variables) {
                        variables.insert(bind.variable.clone());
                    }
                    variables
                }
                LogicalOperator::Project(project) => {
                    let input = visit(&project.input);
                    let mut variables = if project.pass_through_input {
                        input.clone()
                    } else {
                        HashSet::new()
                    };
                    for projection in &project.projections {
                        if let LogicalExpression::Variable(source) = &projection.expression
                            && input.contains(source)
                        {
                            variables
                                .insert(projection.alias.clone().unwrap_or_else(|| source.clone()));
                        }
                    }
                    variables
                }
                LogicalOperator::Distinct(distinct) => visit(&distinct.input),
                LogicalOperator::Sort(sort) => visit(&sort.input),
                LogicalOperator::Limit(limit) => visit(&limit.input),
                LogicalOperator::Skip(skip) => visit(&skip.input),
                // This RDF aggregate intentionally emits one synthetic,
                // unbound group on empty input (the qualified G0/W3C behavior
                // covered by integration tests). Without a separate proof of
                // non-emptiness, no group key is therefore must-bound.
                LogicalOperator::Aggregate(_) => HashSet::new(),
                LogicalOperator::Return(ret) => {
                    let input = visit(&ret.input);
                    ret.items
                        .iter()
                        .filter_map(|item| match &item.expression {
                            LogicalExpression::Variable(source) if input.contains(source) => {
                                Some(item.alias.clone().unwrap_or_else(|| source.clone()))
                            }
                            _ => None,
                        })
                        .collect()
                }
                LogicalOperator::Unwind(unwind) => {
                    let mut variables = visit(&unwind.input);
                    variables.insert(unwind.variable.clone());
                    variables.extend(unwind.ordinality_var.iter().cloned());
                    variables.extend(unwind.offset_var.iter().cloned());
                    variables
                }
                _ => HashSet::new(),
            }
        }

        visit(op)
    }

    fn resolve_iri(&self, iri: &ast::Iri) -> String {
        let iri_str = iri.as_str();

        // Check if it's a prefixed name
        if let Some(colon_pos) = iri_str.find(':') {
            let prefix = &iri_str[..colon_pos];
            let local = &iri_str[colon_pos + 1..];

            if let Some(namespace) = self.prefixes.get(prefix) {
                return format!("{}{}", namespace, local);
            }
        }

        // Return as-is if no prefix match or already a full IRI
        iri_str.to_string()
    }

    /// Rewrites `?var = "value"@lang` to `?var = "value" AND LANG(?var) = "lang"`,
    /// and `?var != "value"@lang` to `?var != "value" OR LANG(?var) != "lang"`.
    ///
    /// Returns `None` when neither side is a language-tagged literal (the caller
    /// falls through to the normal binary translation path).
    fn try_expand_lang_comparison(
        &mut self,
        left: &ast::Expression,
        left_annotation: Option<&OrdinaryExpressionExactAnnotations>,
        operator: ast::BinaryOperator,
        right: &ast::Expression,
        right_annotation: Option<&OrdinaryExpressionExactAnnotations>,
    ) -> Result<Option<LogicalExpression>> {
        // Restrict the two-use expansion to a variable operand. Re-evaluating a
        // volatile or otherwise complex operand for value and LANG would change
        // SPARQL expression semantics.
        let (var_expr, var_annotation, lang_lit) = match (left, right) {
            (ast::Expression::Variable(_), ast::Expression::Literal(lit)) => {
                (left, left_annotation, lit)
            }
            (ast::Expression::Literal(lit), ast::Expression::Variable(_)) => {
                (right, right_annotation, lit)
            }
            _ => return Ok(None),
        };

        let Some(lang_tag) = lang_lit.language.as_ref() else {
            return Ok(None);
        };
        let translated_var =
            self.translate_expression_with_annotations(var_expr, var_annotation)?;
        let value_literal =
            LogicalExpression::Literal(Value::String(lang_lit.value.clone().into()));
        let lang_literal = LogicalExpression::Literal(Value::String(lang_tag.clone().into()));

        // Build LANG(?var)
        let lang_call = LogicalExpression::FunctionCall {
            name: "LANG".to_string(),
            args: vec![translated_var.clone()],
            distinct: false,
        };

        let (value_op, lang_op, combine_op) = if operator == ast::BinaryOperator::Equal {
            (BinaryOp::Eq, BinaryOp::Eq, BinaryOp::And)
        } else {
            // NotEqual: either value differs OR lang tag differs
            (BinaryOp::Ne, BinaryOp::Ne, BinaryOp::Or)
        };

        let value_cmp = LogicalExpression::Binary {
            left: Box::new(translated_var),
            op: value_op,
            right: Box::new(value_literal),
        };
        let lang_cmp = LogicalExpression::Binary {
            left: Box::new(lang_call),
            op: lang_op,
            right: Box::new(lang_literal),
        };

        Ok(Some(LogicalExpression::Binary {
            left: Box::new(value_cmp),
            op: combine_op,
            right: Box::new(lang_cmp),
        }))
    }

    /// Graph patterns and templates keep lexical form plus datatype.
    /// Numeric and temporal coercion is only for expression evaluation.
    fn pattern_literal_to_value(&self, lit: &ast::Literal) -> Value {
        if let Some(datatype) = &lit.datatype {
            Value::RdfLiteral {
                lexical: lit.value.clone().into(),
                language: None,
                datatype: Some(self.resolve_iri(datatype).into()),
            }
        } else {
            Value::String(lit.value.clone().into())
        }
    }

    fn literal_to_value(&self, lit: &ast::Literal) -> Value {
        if let Some(language) = &lit.language {
            return Value::RdfLiteral {
                lexical: lit.value.clone().into(),
                language: Some(language.to_ascii_lowercase().into()),
                datatype: None,
            };
        }

        // Check for typed literals
        if let Some(datatype) = &lit.datatype {
            let dt = self.resolve_iri(datatype);
            if dt == Literal::XSD_STRING {
                return Value::String(lit.value.clone().into());
            }
            let numeric_literal = Literal::typed(lit.value.as_str(), dt.as_str());
            match dt.as_str() {
                "http://www.w3.org/2001/XMLSchema#integer"
                | "http://www.w3.org/2001/XMLSchema#int"
                | "http://www.w3.org/2001/XMLSchema#long" => {
                    if rdf_numeric_literal_is_valid(&numeric_literal)
                        && let Ok(n) = lit.value.parse::<i64>()
                    {
                        return Value::Int64(n);
                    }
                }
                "http://www.w3.org/2001/XMLSchema#double" => {
                    if rdf_numeric_literal_is_valid(&numeric_literal)
                        && let Ok(n) = lit.value.parse::<f64>()
                    {
                        return Value::Float64(n);
                    }
                }
                // Decimal must not round through binary floating point, and
                // xsd:float must retain its distinct f32 promotion kind. Keep
                // both as typed RDF literals for the RDF numeric evaluator.
                "http://www.w3.org/2001/XMLSchema#decimal"
                | "http://www.w3.org/2001/XMLSchema#float" => {}
                "http://www.w3.org/2001/XMLSchema#boolean" => match lit.value.as_str() {
                    "true" | "1" => return Value::Bool(true),
                    "false" | "0" => return Value::Bool(false),
                    _ => {}
                },
                "http://www.w3.org/2001/XMLSchema#date" => {
                    if let Some(d) = grafeo_common::types::Date::parse(&lit.value) {
                        return Value::Date(d);
                    }
                }
                "http://www.w3.org/2001/XMLSchema#time" => {
                    if let Some(t) = grafeo_common::types::Time::parse(&lit.value) {
                        return Value::Time(t);
                    }
                }
                "http://www.w3.org/2001/XMLSchema#duration"
                | "http://www.w3.org/2001/XMLSchema#dayTimeDuration"
                | "http://www.w3.org/2001/XMLSchema#yearMonthDuration" => {
                    if let Some(d) = grafeo_common::types::Duration::parse(&lit.value) {
                        return Value::Duration(d);
                    }
                }
                "http://www.w3.org/2001/XMLSchema#dateTime" => {
                    // Prefer ZonedDatetime when the value has an explicit offset,
                    // so that local date/time and timezone are preserved for
                    // YEAR/MONTH/DAY/HOURS/MINUTES/SECONDS/TIMEZONE/TZ functions.
                    if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(&lit.value) {
                        return Value::ZonedDatetime(zdt);
                    }
                    // Fall back to Timestamp for values without offset
                    if let Some(pos) = lit.value.find('T')
                        && let (Some(d), Some(t)) = (
                            grafeo_common::types::Date::parse(&lit.value[..pos]),
                            grafeo_common::types::Time::parse(&lit.value[pos + 1..]),
                        )
                    {
                        return Value::Timestamp(grafeo_common::types::Timestamp::from_date_time(
                            d, t,
                        ));
                    }
                }
                _ => {}
            }
        }

        if let Some(datatype) = &lit.datatype {
            return Value::RdfLiteral {
                lexical: lit.value.clone().into(),
                language: None,
                datatype: Some(self.resolve_iri(datatype).into()),
            };
        }

        Value::String(lit.value.clone().into())
    }

    /// Translates a negated property set `!(iri1|^iri2)`.
    ///
    /// For forward IRIs: scans `?s ?p ?o` and filters out excluded predicates.
    /// For inverse IRIs: scans `?o ?p ?s` (swapped) and filters out excluded predicates.
    /// Mixed sets produce a `Union` of forward and inverse branches.
    fn translate_negated_property_set(
        &mut self,
        triple: &ast::TriplePattern,
        negated_iris: &[ast::NegatedIri],
    ) -> Result<LogicalOperator> {
        let subject = self.translate_triple_term(&triple.subject)?;
        let object = self.translate_triple_term(&triple.object)?;
        let graph = self.graph_context_stack.last().cloned();

        let forward_iris: Vec<&ast::Iri> = negated_iris
            .iter()
            .filter(|ni| !ni.inverse)
            .map(|ni| &ni.iri)
            .collect();
        let inverse_iris: Vec<&ast::Iri> = negated_iris
            .iter()
            .filter(|ni| ni.inverse)
            .map(|ni| &ni.iri)
            .collect();

        let has_forward = !forward_iris.is_empty() || inverse_iris.is_empty();
        let has_inverse = !inverse_iris.is_empty();

        let build_branch = |translator: &mut Self,
                            subj: TripleComponent,
                            obj: TripleComponent,
                            excluded: &[&ast::Iri]|
         -> Result<LogicalOperator> {
            let pred_var = format!("_:neg_pred{}", translator.next_anon());
            let scan = translator.make_triple_scan(
                subj,
                TripleComponent::Variable(pred_var.clone()),
                obj,
                graph.clone(),
            );

            if excluded.is_empty() {
                return Ok(scan);
            }

            // Build filter: _:neg_pred != iri1 AND _:neg_pred != iri2 AND ...
            let conditions: Vec<LogicalExpression> = excluded
                .iter()
                .map(|iri| LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Variable(pred_var.clone())),
                    op: BinaryOp::Ne,
                    right: Box::new(LogicalExpression::Literal(Value::String(
                        translator.resolve_iri(iri).into(),
                    ))),
                })
                .collect();

            let predicate = conditions
                .into_iter()
                .reduce(|left, right| LogicalExpression::Binary {
                    left: Box::new(left),
                    op: BinaryOp::And,
                    right: Box::new(right),
                })
                .ok_or_else(|| {
                    Error::Internal("SPARQL excluded-predicate filter is missing".to_string())
                })?;

            Ok(wrap_filter(scan, predicate))
        };

        if has_forward && has_inverse {
            // Union of forward scan (excluding forward IRIs) and
            // inverse scan with swapped s/o (excluding inverse IRIs)
            let forward_branch =
                build_branch(self, subject.clone(), object.clone(), &forward_iris)?;
            let inverse_branch = build_branch(self, object, subject, &inverse_iris)?;
            Ok(LogicalOperator::Union(UnionOp {
                inputs: vec![forward_branch, inverse_branch],
            }))
        } else if has_inverse {
            // Only inverse exclusions: scan with swapped subject/object
            build_branch(self, object, subject, &inverse_iris)
        } else {
            // Only forward exclusions (most common case)
            build_branch(self, subject, object, &forward_iris)
        }
    }

    /// Simple IRI, sequence, alternative, and inverse for native `*` / `+`.
    fn to_path_step(&self, path: &ast::PropertyPath) -> Option<PathStep> {
        match path {
            ast::PropertyPath::Predicate(iri) => Some(PathStep::Iri {
                iri: self.resolve_iri(iri),
                inverse: false,
            }),
            ast::PropertyPath::RdfType => Some(PathStep::Iri {
                iri: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                inverse: false,
            }),
            ast::PropertyPath::Inverse(inner) => Some(self.to_path_step(inner)?.inverted()),
            ast::PropertyPath::Sequence(steps) => {
                let converted: Option<Vec<_>> =
                    steps.iter().map(|s| self.to_path_step(s)).collect();
                Some(PathStep::Sequence(converted?))
            }
            ast::PropertyPath::Alternative(steps) => {
                let converted: Option<Vec<_>> =
                    steps.iter().map(|s| self.to_path_step(s)).collect();
                Some(PathStep::Alternative(converted?))
            }
            _ => None,
        }
    }

    fn graph_iri_from_context(&self) -> Option<String> {
        match self.graph_context_stack.last() {
            Some(TripleComponent::Iri(iri)) => Some(iri.clone()),
            Some(TripleComponent::Literal(Value::String(s))) => Some(s.to_string()),
            _ => None,
        }
    }

    fn translate_native_property_path(
        &mut self,
        triple: &ast::TriplePattern,
        inner_path: &ast::PropertyPath,
        min_hops: u32,
    ) -> Option<LogicalOperator> {
        let path = self.to_path_step(inner_path)?;
        let subject = self.translate_triple_term(&triple.subject).ok()?;
        let object = self.translate_triple_term(&triple.object).ok()?;
        Some(LogicalOperator::PropertyPath(PropertyPathOp {
            subject,
            path,
            object,
            min_hops,
            graph: self.graph_iri_from_context(),
        }))
    }

    /// Translates a `OneOrMore` property path (`path+`) using native reachability
    /// for simple IRIs, else bounded union expansion.
    fn translate_one_or_more_path(
        &mut self,
        triple: &ast::TriplePattern,
        inner_path: &ast::PropertyPath,
    ) -> Result<LogicalOperator> {
        if let Some(op) = self.translate_native_property_path(triple, inner_path, 1) {
            return Ok(op);
        }
        const MAX_DEPTH: usize = 50;

        let subject = self.translate_triple_term(&triple.subject)?;
        let object = self.translate_triple_term(&triple.object)?;
        let graph = self.graph_context_stack.last().cloned();

        let mut branches = Vec::new();

        for depth in 1..=MAX_DEPTH {
            let branch =
                self.translate_fixed_depth_path(inner_path, &subject, &object, &graph, depth)?;
            branches.push(branch);
        }

        let union = LogicalOperator::Union(UnionOp { inputs: branches });

        // Wrap in Distinct to deduplicate across depths
        Ok(wrap_distinct(union))
    }

    /// Translates a `ZeroOrMore` property path (`path*`) using bounded expansion.
    ///
    /// Includes reflexive (0-hop) matches for every node that participates as
    /// subject or object of the predicate, plus the same 1..`MAX_DEPTH` expansion
    /// used by `OneOrMore`.
    fn translate_zero_or_more_path(
        &mut self,
        triple: &ast::TriplePattern,
        inner_path: &ast::PropertyPath,
    ) -> Result<LogicalOperator> {
        if let Some(op) = self.translate_native_property_path(triple, inner_path, 0) {
            return Ok(op);
        }
        const MAX_DEPTH: usize = 50;

        let subject = self.translate_triple_term(&triple.subject)?;
        let object = self.translate_triple_term(&triple.object)?;
        let graph = self.graph_context_stack.last().cloned();

        let mut branches = Vec::new();

        // 0-hop reflexive branches
        self.add_reflexive_branches(&subject, &object, inner_path, &graph, &mut branches)?;

        // 1+ hops: same as OneOrMore (wrapped in projection to match reflexive column count)
        for depth in 1..=MAX_DEPTH {
            let branch =
                self.translate_fixed_depth_path(inner_path, &subject, &object, &graph, depth)?;
            branches.push(self.project_path_endpoints(&subject, &object, branch));
        }

        let union = LogicalOperator::Union(UnionOp { inputs: branches });
        Ok(wrap_distinct(union))
    }

    /// Translates a `ZeroOrOne` property path (`path?`).
    ///
    /// Produces a union of 0-hop reflexive matches and exactly 1-hop matches,
    /// then deduplicates. Same structure as `translate_zero_or_more_path` but
    /// bounded to depth 0..1 instead of 0..MAX_DEPTH.
    fn translate_zero_or_one_path(
        &mut self,
        triple: &ast::TriplePattern,
        inner_path: &ast::PropertyPath,
    ) -> Result<LogicalOperator> {
        let subject = self.translate_triple_term(&triple.subject)?;
        let object = self.translate_triple_term(&triple.object)?;
        let graph = self.graph_context_stack.last().cloned();

        let mut branches = Vec::new();

        // 0-hop reflexive branches
        self.add_reflexive_branches(&subject, &object, inner_path, &graph, &mut branches)?;

        // 1-hop: exactly one traversal of the predicate (wrapped to match reflexive column count)
        let one_hop = self.translate_fixed_depth_path(inner_path, &subject, &object, &graph, 1)?;
        branches.push(self.project_path_endpoints(&subject, &object, one_hop));

        let union = LogicalOperator::Union(UnionOp { inputs: branches });
        Ok(wrap_distinct(union))
    }

    /// Adds 0-hop reflexive branches for `ZeroOrMore` and `ZeroOrOne` paths.
    fn add_reflexive_branches(
        &mut self,
        subject: &TripleComponent,
        object: &TripleComponent,
        inner_path: &ast::PropertyPath,
        graph: &Option<TripleComponent>,
        branches: &mut Vec<LogicalOperator>,
    ) -> Result<()> {
        if matches!(subject, TripleComponent::Variable(_)) {
            let fresh_obj = TripleComponent::Variable(format!("_:refl{}", self.next_anon()));
            let pred = self.translate_property_path(inner_path)?;
            let subj_scan = self.make_triple_scan(subject.clone(), pred, fresh_obj, graph.clone());
            let subj_reflexive = self.project_reflexive(subject, object, subj_scan)?;
            branches.push(subj_reflexive);

            let fresh_subj = TripleComponent::Variable(format!("_:refl{}", self.next_anon()));
            let pred2 = self.translate_property_path(inner_path)?;
            let obj_scan = self.make_triple_scan(fresh_subj, pred2, object.clone(), graph.clone());
            let obj_reflexive = self.project_reflexive_from_object(subject, object, obj_scan)?;
            branches.push(obj_reflexive);
        } else if let TripleComponent::Variable(obj_var) = object {
            let subj_expr = self.triple_component_to_expression(subject);
            let reflexive = LogicalOperator::Bind(BindOp {
                expression: subj_expr,
                variable: obj_var.clone(),
                input: Box::new(LogicalOperator::Empty),
            });
            branches.push(reflexive);
        }
        Ok(())
    }

    /// Wraps a depth-branch operator in a projection that outputs only the
    /// subject and object variables, stripping companion columns so that all
    /// union branches have consistent column counts for DISTINCT.
    fn project_path_endpoints(
        &self,
        subject: &TripleComponent,
        object: &TripleComponent,
        input: LogicalOperator,
    ) -> LogicalOperator {
        let mut projections = Vec::new();
        if let TripleComponent::Variable(s) = subject {
            projections.push(Projection {
                expression: LogicalExpression::Variable(s.clone()),
                alias: Some(s.clone()),
            });
        }
        if let TripleComponent::Variable(o) = object {
            projections.push(Projection {
                expression: LogicalExpression::Variable(o.clone()),
                alias: Some(o.clone()),
            });
        }
        if projections.is_empty() {
            return input;
        }
        LogicalOperator::Project(ProjectOp {
            projections,
            input: Box::new(input),
            pass_through_input: false,
        })
    }

    /// Translates a property path at a fixed depth (number of hops).
    ///
    /// Depth 1 produces a single `TripleScan`. Depth N chains N scans with
    /// freshly generated intermediate variables joined together.
    fn translate_fixed_depth_path(
        &mut self,
        path: &ast::PropertyPath,
        subject: &TripleComponent,
        object: &TripleComponent,
        graph: &Option<TripleComponent>,
        depth: usize,
    ) -> Result<LogicalOperator> {
        if depth == 1 {
            let predicate = self.translate_property_path(path)?;
            return Ok(self.make_triple_scan(
                subject.clone(),
                predicate,
                object.clone(),
                graph.clone(),
            ));
        }

        // Multiple hops: chain triple scans with intermediate variables
        let mut current_subject = subject.clone();
        let mut plan = LogicalOperator::Empty;
        let mut first = true;

        for i in 0..depth {
            let next_object = if i == depth - 1 {
                object.clone()
            } else {
                TripleComponent::Variable(format!("_:path{}", self.next_anon()))
            };

            let predicate = self.translate_property_path(path)?;
            let scan = self.make_triple_scan(
                current_subject,
                predicate,
                next_object.clone(),
                graph.clone(),
            );

            if first {
                plan = scan;
                first = false;
            } else {
                plan = self.join_patterns(plan, scan);
            }

            current_subject = next_object;
        }

        Ok(plan)
    }

    /// Projects a scan so that the subject value appears as both the subject
    /// and object output variables, producing reflexive (0-hop) rows.
    fn project_reflexive(
        &self,
        subject: &TripleComponent,
        object: &TripleComponent,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let subj_expr = self.triple_component_to_expression(subject);
        let obj_var = match object {
            TripleComponent::Variable(v) => v.clone(),
            _ => return Ok(input),
        };
        let mut projections = Vec::new();
        // Keep subject variable in output if it is a variable
        if let TripleComponent::Variable(s_var) = subject {
            projections.push(Projection {
                expression: LogicalExpression::Variable(s_var.clone()),
                alias: Some(s_var.clone()),
            });
        }
        projections.push(Projection {
            expression: subj_expr,
            alias: Some(obj_var),
        });
        Ok(LogicalOperator::Project(ProjectOp {
            projections,
            input: Box::new(input),
            pass_through_input: false,
        }))
    }

    /// Projects a scan so that the object value appears as both the subject
    /// and object output variables, producing reflexive (0-hop) rows.
    fn project_reflexive_from_object(
        &self,
        subject: &TripleComponent,
        object: &TripleComponent,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let obj_expr = self.triple_component_to_expression(object);
        let subj_var = match subject {
            TripleComponent::Variable(v) => v.clone(),
            _ => return Ok(input),
        };
        let mut projections = vec![Projection {
            expression: obj_expr,
            alias: Some(subj_var),
        }];
        // Keep object variable in output if it is a variable
        if let TripleComponent::Variable(o_var) = object {
            projections.push(Projection {
                expression: LogicalExpression::Variable(o_var.clone()),
                alias: Some(o_var.clone()),
            });
        }
        Ok(LogicalOperator::Project(ProjectOp {
            projections,
            input: Box::new(input),
            pass_through_input: false,
        }))
    }

    /// Converts a `TripleComponent` to a `LogicalExpression` for use in projections.
    fn triple_component_to_expression(&self, component: &TripleComponent) -> LogicalExpression {
        match component {
            TripleComponent::Variable(name) => LogicalExpression::Variable(name.clone()),
            TripleComponent::Iri(iri) => {
                LogicalExpression::Literal(Value::String(iri.clone().into()))
            }
            TripleComponent::Literal(val) => LogicalExpression::Literal(val.clone()),
            TripleComponent::LangLiteral { value, .. } => {
                LogicalExpression::Literal(Value::String(value.clone().into()))
            }
            TripleComponent::BlankNode(label) => {
                LogicalExpression::Literal(Value::String(format!("_:{label}").into()))
            }
        }
    }

    /// Converts a parsed `DatasetClause` into a `DatasetRestriction` for the logical plan.
    fn translate_dataset_clause(
        &self,
        dataset: &Option<ast::DatasetClause>,
    ) -> Option<DatasetRestriction> {
        let clause = dataset.as_ref()?;

        let mut default_graphs: Vec<String> = clause
            .default_graphs
            .iter()
            .map(|iri| self.resolve_iri(iri))
            .collect();
        let mut named_graphs: Vec<String> = clause
            .named_graphs
            .iter()
            .map(|iri| self.resolve_iri(iri))
            .collect();

        if named_graphs.is_empty() && !default_graphs.is_empty() {
            named_graphs.push(RDF_EXPLICIT_EMPTY_NAMED_DATASET.to_string());
        }
        if default_graphs.is_empty() && !named_graphs.is_empty() {
            default_graphs.push(RDF_EXPLICIT_EMPTY_DEFAULT_DATASET.to_string());
        }

        // Only create a restriction if at least one FROM or FROM NAMED is present
        if default_graphs.is_empty() && named_graphs.is_empty() {
            return None;
        }

        Some(DatasetRestriction {
            default_graphs,
            named_graphs,
        })
    }

    /// Creates a `TripleScanOp` with the current graph context and dataset restriction.
    fn make_triple_scan(
        &self,
        subject: TripleComponent,
        predicate: TripleComponent,
        object: TripleComponent,
        graph: Option<TripleComponent>,
    ) -> LogicalOperator {
        LogicalOperator::TripleScan(TripleScanOp {
            subject,
            predicate,
            object,
            graph,
            input: None,
            dataset: self.dataset.clone(),
        })
    }

    fn next_anon(&mut self) -> u32 {
        let n = self.anon_counter;
        self.anon_counter += 1;
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{LimitOp, SkipOp, SortOp};

    #[test]
    fn totality_scope_rollback_preserves_bindings_and_rejects_corrupt_state_atomically() {
        let mut state = OrdinaryMembershipState::default();
        assert!(state.insert(0));
        let mark = state.mark();
        assert!(state.insert(1));
        assert!(state.remove(0));
        state.rollback(mark).unwrap();
        assert!(state.contains(0));
        assert!(!state.contains(1));
        assert!(state.was_present_at(mark, 0));
        assert_eq!(state.mark(), mark);
        state.rollback(mark).unwrap();

        assert!(state.insert(1));
        assert!(state.insert(2));
        // The corrupt older entry must be found before either valid newer
        // entry changes membership or is removed from the undo log.
        state.undo.insert(mark, (usize::MAX, true, 0));
        let before = (
            state.present.clone(),
            state.bound_since.clone(),
            state.undo.clone(),
        );
        let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| state.rollback(mark)))
            .expect("invalid membership must return an error")
            .unwrap_err();
        assert!(matches!(error, Error::Internal(message)
            if message == "ordinary exact-demand membership slot is missing"));
        assert_eq!((state.present, state.bound_since, state.undo), before);

        let mut state = OrdinaryMembershipState::default();
        state.insert(0);
        state.bound_since.clear();
        let error = state.rollback(0).unwrap_err();
        assert!(matches!(error, Error::Internal(message)
            if message == "ordinary exact-demand binding epoch is missing"));
        assert!(state.contains(0));
        assert_eq!(state.undo, vec![(0, false, usize::MAX)]);
    }

    #[test]
    fn totality_data_and_describe_preserve_branch_cardinality() {
        for verb in ["INSERT DATA", "DELETE DATA"] {
            for count in 0..=2 {
                let triples = "<urn:s> <urn:p> <urn:o> . ".repeat(count);
                let plan = translate(&format!("{verb} {{ {triples} }}")).unwrap();
                let is_operation = |op: &LogicalOperator| match verb {
                    "INSERT DATA" => matches!(op, LogicalOperator::InsertTriple(_)),
                    _ => matches!(op, LogicalOperator::DeleteTriple(_)),
                };
                match count {
                    0 => assert!(matches!(plan.root, LogicalOperator::Empty)),
                    1 => assert!(is_operation(&plan.root)),
                    _ => {
                        let LogicalOperator::Union(union) = plan.root else {
                            panic!("expected one branch per data operation");
                        };
                        assert_eq!(union.inputs.len(), count);
                        assert!(union.inputs.iter().all(is_operation));
                    }
                }
            }
        }
        let single = translate("DESCRIBE <urn:first>").unwrap();
        assert!(matches!(single.root, LogicalOperator::TripleScan(_)));
        let multiple = translate("DESCRIBE <urn:first> <urn:second>").unwrap();
        let LogicalOperator::Union(union) = multiple.root else {
            panic!("expected a CBD scan for each described resource");
        };
        assert_eq!(union.inputs.len(), 2);
        assert!(
            union
                .inputs
                .iter()
                .all(|op| matches!(op, LogicalOperator::TripleScan(_)))
        );
    }

    #[test]
    fn totality_values_preserve_zero_one_and_multiple_rows() {
        for count in 0..=2 {
            let pattern = ast::GraphPattern::InlineData(ast::InlineDataClause {
                variables: vec!["x".to_string()],
                values: vec![vec![Some(ast::DataValue::Literal(ast::Literal::string("x")))]; count],
            });
            let plan = SparqlTranslator::new()
                .translate_graph_pattern(&pattern)
                .unwrap();
            match count {
                0 => assert!(matches!(plan, LogicalOperator::Filter(filter)
                    if matches!(filter.predicate, LogicalExpression::Literal(Value::Bool(false)))
                    && matches!(*filter.input, LogicalOperator::Empty))),
                1 => assert!(matches!(plan, LogicalOperator::Bind(_))),
                _ => assert!(matches!(plan, LogicalOperator::Union(union)
                    if union.inputs.len() == count)),
            }
        }
    }

    #[test]
    fn totality_unary_translation_preserves_numeric_coercion_and_operators() {
        for operator in [
            ast::UnaryOperator::Plus,
            ast::UnaryOperator::Minus,
            ast::UnaryOperator::Not,
        ] {
            let expression = ast::Expression::Unary {
                operator,
                operand: Box::new(ast::Expression::Variable("x".to_string())),
            };
            let translated = SparqlTranslator::new()
                .translate_expression(&expression)
                .unwrap();
            match operator {
                ast::UnaryOperator::Plus => assert!(matches!(translated,
                    LogicalExpression::FunctionCall { name, args, distinct: false }
                    if name == RDF_NUMERIC_VALUE
                    && matches!(args.as_slice(), [LogicalExpression::Variable(name)] if name == "x"))),
                ast::UnaryOperator::Minus => assert!(matches!(translated,
                    LogicalExpression::Unary { op: UnaryOp::Neg, operand }
                    if matches!(operand.as_ref(), LogicalExpression::Variable(name) if name == "x"))),
                ast::UnaryOperator::Not => assert!(matches!(translated,
                    LogicalExpression::Unary { op: UnaryOp::Not, operand }
                    if matches!(operand.as_ref(), LogicalExpression::Variable(name) if name == "x"))),
            }
        }
    }

    #[test]
    fn totality_language_comparison_preserves_both_orientations_and_plain_literals() {
        let variable = ast::Expression::Variable("x".to_string());
        for (literal, tagged) in [
            (ast::Literal::with_language("word", "en"), true),
            (ast::Literal::string("word"), false),
        ] {
            let literal = ast::Expression::Literal(literal);
            for (left, right) in [(&variable, &literal), (&literal, &variable)] {
                let translated = SparqlTranslator::new()
                    .try_expand_lang_comparison(left, None, ast::BinaryOperator::Equal, right, None)
                    .unwrap();
                if tagged {
                    assert!(matches!(
                        translated,
                        Some(LogicalExpression::Binary {
                            op: BinaryOp::And,
                            ..
                        })
                    ));
                } else {
                    assert!(translated.is_none());
                }
            }
        }
    }

    fn find_bind_expression<'a>(
        op: &'a LogicalOperator,
        variable: &str,
    ) -> Option<&'a LogicalExpression> {
        if let LogicalOperator::Bind(bind) = op
            && bind.variable == variable
        {
            return Some(&bind.expression);
        }
        op.children()
            .into_iter()
            .find_map(|child| find_bind_expression(child, variable))
    }

    fn contains_bind(op: &LogicalOperator, variable: &str) -> bool {
        find_bind_expression(op, variable).is_some()
    }

    fn collect_physical_aggregate_aliases(op: &LogicalOperator) -> Vec<String> {
        let mut aliases = Vec::new();
        fn visit(op: &LogicalOperator, aliases: &mut Vec<String>) {
            if let LogicalOperator::Aggregate(aggregate) = op {
                aliases.extend(
                    aggregate
                        .aggregates
                        .iter()
                        .filter_map(|expression| expression.alias.clone()),
                );
            }
            for child in op.children() {
                visit(child, aliases);
            }
        }
        visit(op, &mut aliases);
        aliases
    }

    fn physical_aggregate_count(op: &LogicalOperator) -> usize {
        let mut count = 0;
        fn visit(op: &LogicalOperator, count: &mut usize) {
            if let LogicalOperator::Aggregate(aggregate) = op {
                *count += aggregate.aggregates.len();
            }
            for child in op.children() {
                visit(child, count);
            }
        }
        visit(op, &mut count);
        count
    }

    fn collect_hidden_aggregate_aliases(op: &LogicalOperator) -> Vec<String> {
        collect_physical_aggregate_aliases(op)
            .into_iter()
            .filter(|alias| is_rdf_internal_term_column(alias))
            .collect()
    }

    fn parse_select(query: &str) -> ast::SelectQuery {
        let query = sparql::parse(query).expect("SELECT parses");
        let ast::QueryForm::Select(select) = query.query_form else {
            panic!("expected SELECT query form");
        };
        select
    }

    fn aggregate_kind(aggregate: &ast::AggregateExpression) -> &'static str {
        match aggregate {
            ast::AggregateExpression::Count { .. } => "COUNT",
            ast::AggregateExpression::Sum { .. } => "SUM",
            ast::AggregateExpression::Average { .. } => "AVG",
            ast::AggregateExpression::Minimum { .. } => "MIN",
            ast::AggregateExpression::Maximum { .. } => "MAX",
            ast::AggregateExpression::Sample { .. } => "SAMPLE",
            ast::AggregateExpression::GroupConcat { .. } => "GROUP_CONCAT",
        }
    }

    fn collect_projection_directness(op: &LogicalOperator, variable: &str, direct: &mut Vec<bool>) {
        match op {
            LogicalOperator::Bind(bind) => {
                collect_projection_directness(&bind.input, variable, direct);
            }
            LogicalOperator::Project(project) => {
                if project.projections.iter().any(|projection| {
                    projection.alias.as_deref() == Some(variable)
                        || matches!(
                            &projection.expression,
                            LogicalExpression::Variable(projected) if projected == variable
                        )
                }) {
                    direct.push(matches!(*project.input, LogicalOperator::Empty));
                }
                collect_projection_directness(&project.input, variable, direct);
            }
            LogicalOperator::Union(union) => {
                for input in &union.inputs {
                    collect_projection_directness(input, variable, direct);
                }
            }
            LogicalOperator::Join(join) => {
                collect_projection_directness(&join.left, variable, direct);
                collect_projection_directness(&join.right, variable, direct);
            }
            _ => {}
        }
    }

    #[test]
    fn grouped_wildcard_subselect_exports_only_group_outputs() {
        let query = sparql::parse(
            "SELECT * WHERE { \
                 { SELECT * WHERE { VALUES (?group ?hidden) { (1 2) } } GROUP BY ?group } \
             }",
        )
        .expect("grouped wildcard subselect parses");
        let ast::QueryForm::Select(select) = query.query_form else {
            panic!("expected outer SELECT");
        };
        let mut variables = HashSet::new();
        SparqlTranslator::collect_pattern_output_variables(&select.where_clause, &mut variables);

        assert_eq!(variables, HashSet::from(["group".to_string()]));

        let grouped =
            sparql::parse("SELECT * WHERE { VALUES (?group ?hidden) { (1 2) } } GROUP BY ?group")
                .expect("grouped wildcard SELECT parses");
        let ast::QueryForm::Select(grouped) = grouped.query_form else {
            panic!("expected grouped SELECT");
        };
        let (_, exports) = SparqlTranslator::ordinary_select_exact_annotations(
            &grouped,
            &HashSet::from(["hidden".to_string()]),
        )
        .expect("exact-demand analysis respects the grouping boundary");
        assert_eq!(exports, HashSet::from(["group".to_string()]));
    }

    // === Basic SELECT Tests ===

    #[test]
    fn test_translate_simple_select() {
        let query = "SELECT ?x WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_with_prefix() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?name
            WHERE { ?x foaf:name ?name }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_wildcard() {
        let query = "SELECT * WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn ordinary_computed_projection_materializes_exact_term_only_when_consumed() {
        let unused = translate(
            r#"SELECT (IRI("urn:unused") AS ?computed)
               WHERE {}"#,
        )
        .unwrap();
        let LogicalOperator::Project(unused_project) = unused.root else {
            panic!("expected top-level PROJECT");
        };
        assert!(
            matches!(*unused_project.input, LogicalOperator::Empty),
            "unconsumed computed IRI must stay a direct projection: {:?}",
            unused_project.input
        );

        let consumed = translate(
            r#"SELECT ?value
               WHERE {
                   { SELECT (IRI("urn:datatype") AS ?datatype) WHERE {} }
                   BIND(STRDT("lexical", ?datatype) AS ?value)
               }"#,
        )
        .unwrap();

        assert!(
            contains_bind(&consumed.root, &rdf_tagged_term_column("datatype")),
            "subselect alias consumed by outer STRDT must retain its sealed tagged value"
        );
        assert!(
            contains_bind(&consumed.root, &rdf_exact_term_column("datatype")),
            "subselect alias consumed by outer STRDT must retain its exact companion"
        );
    }

    #[test]
    fn ordinary_exact_projection_demand_is_branch_local_across_union() {
        let plan = translate(
            r#"SELECT ?value WHERE {
                   {
                       { SELECT (IRI("urn:used") AS ?datatype) WHERE {} }
                       BIND(STRDT("x", ?datatype) AS ?value)
                   }
                   UNION {
                       { SELECT (IRI("urn:unused") AS ?datatype) WHERE {} }
                       BIND("plain" AS ?marker)
                   }
               }"#,
        )
        .unwrap();

        let mut direct = Vec::new();
        collect_projection_directness(&plan.root, "datatype", &mut direct);
        assert_eq!(
            direct,
            [false, true],
            "only the UNION branch whose alias feeds STRDT may materialize exact machinery"
        );
    }

    #[test]
    fn ordinary_exact_projection_demand_respects_subselect_exports() {
        let direct = translate(
            r#"SELECT ?value WHERE {
                   { SELECT (IRI("urn:direct") AS ?direct_dt) WHERE {} }
                   BIND(STRDT("x", ?direct_dt) AS ?value)
               }"#,
        )
        .unwrap();
        assert!(contains_bind(
            &direct.root,
            &rdf_tagged_term_column("direct_dt")
        ));

        let renamed = translate(
            r#"SELECT ?value WHERE {
                   {
                       SELECT (?source AS ?renamed_dt) WHERE {
                           { SELECT (IRI("urn:renamed") AS ?source) WHERE {} }
                       }
                   }
                   BIND(STRDT("x", ?renamed_dt) AS ?value)
               }"#,
        )
        .unwrap();
        assert!(contains_bind(
            &renamed.root,
            &rdf_tagged_term_column("source")
        ));

        let wildcard = translate(
            r#"SELECT ?value WHERE {
                   {
                       SELECT * WHERE {
                           { SELECT (IRI("urn:wildcard") AS ?wild_dt) WHERE {} }
                       }
                   }
                   BIND(STRDT("x", ?wild_dt) AS ?value)
               }"#,
        )
        .unwrap();
        assert!(contains_bind(
            &wildcard.root,
            &rdf_tagged_term_column("wild_dt")
        ));

        let hidden = translate(
            r#"SELECT ?value WHERE {
                   BIND(<urn:outer> AS ?hidden_dt)
                   {
                       SELECT ?other WHERE {
                           { SELECT (IRI("urn:hidden") AS ?hidden_dt) WHERE {} }
                           BIND("other" AS ?other)
                       }
                   }
                   BIND(STRDT("x", ?hidden_dt) AS ?value)
               }"#,
        )
        .unwrap();
        let mut hidden_direct = Vec::new();
        collect_projection_directness(&hidden.root, "hidden_dt", &mut hidden_direct);
        assert_eq!(
            hidden_direct,
            [true],
            "a non-exported same-spelled alias must not inherit outer demand"
        );
    }

    #[test]
    fn rdf_or_native_pattern_demand_respects_branch_and_minus_boundaries() {
        fn collect_bind_demands(
            annotation: &OrdinaryPatternExactAnnotations,
            variables: &OrdinaryVariableTable,
            demands: &mut Vec<(String, bool)>,
        ) {
            match annotation {
                OrdinaryPatternExactAnnotations::Bind {
                    variable,
                    rdf_or_native,
                    ..
                } => demands.push((variables.names[*variable].clone(), *rdf_or_native)),
                OrdinaryPatternExactAnnotations::Group(children) => {
                    for child in children {
                        collect_bind_demands(child, variables, demands);
                    }
                }
                OrdinaryPatternExactAnnotations::Optional { pattern, .. }
                | OrdinaryPatternExactAnnotations::Minus { pattern, .. }
                | OrdinaryPatternExactAnnotations::NamedGraph { pattern, .. } => {
                    collect_bind_demands(pattern, variables, demands);
                }
                OrdinaryPatternExactAnnotations::Union(children) => {
                    for child in children {
                        collect_bind_demands(&child.pattern, variables, demands);
                    }
                }
                OrdinaryPatternExactAnnotations::Basic { .. }
                | OrdinaryPatternExactAnnotations::Filter(_)
                | OrdinaryPatternExactAnnotations::InlineData { .. }
                | OrdinaryPatternExactAnnotations::SubSelect { .. }
                | OrdinaryPatternExactAnnotations::Service => {}
            }
        }

        let select = parse_select(
            r#"SELECT ?kept WHERE {
                   BIND(<urn:kept> AS ?kept)
                   OPTIONAL { BIND(?kept AS ?optional_copy) }
                   {
                     BIND(?kept AS ?union_copy)
                   } UNION {
                     BIND(?kept AS ?union_copy)
                   }
                   MINUS { BIND("rhs" AS ?kept) }
               }
               ORDER BY ?optional_copy ?union_copy ?kept"#,
        );
        let (annotation, _) =
            SparqlTranslator::ordinary_select_exact_annotations(&select, &HashSet::new()).unwrap();
        let mut demands = Vec::new();
        collect_bind_demands(
            &annotation.where_clause,
            &annotation.variables,
            &mut demands,
        );

        assert_eq!(
            demands
                .iter()
                .filter(|(name, demanded)| name == "kept" && *demanded)
                .count(),
            1,
            "the outer producer receives branch correlation demand"
        );
        assert_eq!(
            demands
                .iter()
                .filter(|(name, demanded)| name == "optional_copy" && *demanded)
                .count(),
            1
        );
        assert_eq!(
            demands
                .iter()
                .filter(|(name, demanded)| name == "union_copy" && *demanded)
                .count(),
            2
        );
        assert_eq!(
            demands
                .iter()
                .filter(|(name, demanded)| name == "kept" && !*demanded)
                .count(),
            1,
            "MINUS RHS outputs must not inherit outer output demand"
        );
    }

    #[test]
    fn ordinary_exact_demand_analysis_visits_a_copy_chain_linearly() {
        const COPY_COUNT: usize = 96;
        let mut query = String::from("SELECT ?value WHERE {\n  BIND(IRI(\"urn:seed\") AS ?v96)\n");
        for source in (1..=COPY_COUNT).rev() {
            use std::fmt::Write;
            writeln!(query, "  BIND(?v{source} AS ?v{})", source - 1).unwrap();
        }
        query.push_str("  BIND(STRDT(\"x\", ?v0) AS ?value)\n}");

        ORDINARY_EXACT_PATTERN_VISITS.with(|visits| visits.set(0));
        translate(&query).unwrap();
        let visits = ORDINARY_EXACT_PATTERN_VISITS.with(std::cell::Cell::get);
        assert!(
            visits <= (COPY_COUNT + 4) * 2,
            "a single reverse analysis should visit each pattern once; observed {visits} visits"
        );
    }

    #[test]
    fn ordinary_expression_correlations_accumulate_in_linear_work() {
        const WIDTH: usize = 96;
        let variables = (0..WIDTH)
            .map(|index| format!("?v{index}"))
            .collect::<Vec<_>>();
        let values = (0..WIDTH)
            .map(|index| index.to_string())
            .collect::<Vec<_>>();
        let binary = variables.join(" + ");
        let coalesce = variables.join(", ");
        let in_list = variables
            .iter()
            .skip(1)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let query = format!(
            "SELECT ?sum ?first ?inside WHERE {{ VALUES ({}) {{ ({}) }} BIND(({binary}) AS ?sum) BIND(COALESCE({coalesce}) AS ?first) BIND(?v0 IN ({in_list}) AS ?inside) }}",
            variables.join(" "),
            values.join(" "),
        );

        ORDINARY_EXACT_CORRELATION_WORK.with(|work| work.set(0));
        translate(&query).unwrap();
        let work = ORDINARY_EXACT_CORRELATION_WORK.with(std::cell::Cell::get);
        assert!(
            work <= WIDTH * 4 + 8,
            "binary/COALESCE/IN correlation work must scale with variable occurrences, observed {work} for width {WIDTH}"
        );
    }

    #[test]
    fn ordinary_expression_correlations_are_sorted_once_at_branch_boundaries() {
        for branch in [
            r#"OPTIONAL {
                   BIND(CONCAT(STR(STRDT("x", ?second)), STR(STRDT("y", ?first))) AS ?out)
               }"#,
            r#"{
                   BIND(CONCAT(STR(STRDT("x", ?second)), STR(STRDT("y", ?first))) AS ?out)
               } UNION {
                   BIND("other" AS ?out)
               }"#,
        ] {
            let query = format!(
                "SELECT ?out WHERE {{ VALUES (?first ?second) {{ (<urn:first> <urn:second>) }} {branch} }}"
            );
            let plan = translate(&query).unwrap();
            for variable in ["first", "second"] {
                assert!(
                    contains_bind(&plan.root, &rdf_tagged_term_column(variable)),
                    "reverse reference order lost correlated exact demand for ?{variable}: {query}"
                );
            }
        }
    }

    #[test]
    fn standalone_optional_preserves_the_unit_solution() {
        let query = sparql::parse(
            r#"SELECT * WHERE {
                   OPTIONAL { FILTER(false) }
               }"#,
        )
        .unwrap();
        let ast::QueryForm::Select(select) = &query.query_form else {
            panic!("expected SELECT query");
        };
        let optional = &select.where_clause;
        assert!(matches!(optional, ast::GraphPattern::Optional(_)));

        let mut translator = SparqlTranslator::new();
        let translated = translator.translate_graph_pattern(optional).unwrap();
        assert!(
            matches!(translated, LogicalOperator::LeftJoin(ref join)
                if matches!(join.left.as_ref(), LogicalOperator::Empty)),
            "standalone OPTIONAL must lower as LeftJoin(unit, RHS): {translated:#?}"
        );
    }

    #[test]
    fn standalone_minus_preserves_the_disjoint_unit_solution() {
        let query = sparql::parse(
            r#"SELECT * WHERE {
                   MINUS { VALUES ?right { 1 } }
               }"#,
        )
        .unwrap();
        let ast::QueryForm::Select(select) = &query.query_form else {
            panic!("expected SELECT query");
        };
        let minus = &select.where_clause;
        assert!(matches!(minus, ast::GraphPattern::Minus(_)));

        let mut translator = SparqlTranslator::new();
        let translated = translator.translate_graph_pattern(minus).unwrap();
        assert!(
            matches!(translated, LogicalOperator::AntiJoin(ref join)
                if join.semantics == AntiJoinSemantics::Minus
                    && matches!(join.left.as_ref(), LogicalOperator::Empty)),
            "standalone MINUS must lower as AntiJoin(unit, RHS, Minus): {translated:#?}"
        );
    }

    #[test]
    fn ordinary_exact_annotations_reject_ast_shape_mismatches() {
        let query = sparql::parse(
            r#"SELECT ?value WHERE {
                   BIND(IRI("urn:datatype") AS ?datatype)
                   BIND(STRDT("x", ?datatype) AS ?value)
               }"#,
        )
        .unwrap();
        let ast::QueryForm::Select(select) = &query.query_form else {
            panic!("expected SELECT query");
        };
        let mut mismatched = SparqlTranslator::build_ordinary_select_annotations(select).unwrap();
        mismatched.where_clause = OrdinaryPatternExactAnnotations::Basic { shared: Vec::new() };

        let mut translator = SparqlTranslator::new();
        translator.dataset = Some(DatasetRestriction {
            default_graphs: vec!["urn:preserved".to_string()],
            named_graphs: Vec::new(),
        });
        let error = translator
            .translate_select_with_annotations(select, Some(&mismatched))
            .unwrap_err();
        let Error::Internal(message) = error else {
            panic!("expected internal annotation mismatch, got {error:?}");
        };
        assert!(
            message.contains("ordinary exact-demand annotation mismatch at SELECT.where")
                && message.contains("expected Group, got Basic"),
            "unexpected mismatch error: {message}"
        );
        assert_eq!(
            translator
                .dataset
                .as_ref()
                .map(|dataset| dataset.default_graphs.as_slice()),
            Some(["urn:preserved".to_string()].as_slice()),
            "annotation preflight must fail before translation state changes"
        );
    }

    #[test]
    fn ordinary_exact_nested_exists_annotations_are_single_pass() {
        ORDINARY_EXACT_PATTERN_VISITS.with(|count| count.set(0));
        ORDINARY_EXACT_PATTERN_CONSUMES.with(|count| count.set(0));
        ORDINARY_EXACT_EXPRESSION_BUILDS.with(|count| count.set(0));
        ORDINARY_EXACT_EXPRESSION_ANALYSES.with(|count| count.set(0));
        ORDINARY_EXACT_FALLBACK_ANALYSES.with(|count| count.set(0));

        translate(
            r#"SELECT ?value WHERE {
                   BIND(IRI("urn:dt") AS ?datatype)
                   FILTER EXISTS {
                       FILTER NOT EXISTS { <urn:s> <urn:p> ?datatype }
                   }
                   BIND(STRDT("x", ?datatype) AS ?value)
               }"#,
        )
        .unwrap();

        let pattern_builds = ORDINARY_EXACT_PATTERN_VISITS.with(std::cell::Cell::get);
        let pattern_analyses = ORDINARY_EXACT_PATTERN_CONSUMES.with(std::cell::Cell::get);
        let expression_builds = ORDINARY_EXACT_EXPRESSION_BUILDS.with(std::cell::Cell::get);
        let expression_analyses = ORDINARY_EXACT_EXPRESSION_ANALYSES.with(std::cell::Cell::get);
        assert_eq!(pattern_builds, pattern_analyses);
        assert_eq!(expression_builds, expression_analyses);
        assert_eq!(
            ORDINARY_EXACT_FALLBACK_ANALYSES.with(std::cell::Cell::get),
            0,
            "owned EXISTS annotations must eliminate translation-time reanalysis"
        );
    }

    #[test]
    fn ordinary_exact_branch_deltas_do_not_copy_large_seeds() {
        const WIDTH: usize = 64;
        let inputs = (0..WIDTH)
            .map(|index| format!("?v{index}"))
            .collect::<Vec<_>>()
            .join(" ");
        let projection = (0..WIDTH)
            .map(|index| format!("?out{index}"))
            .collect::<Vec<_>>()
            .join(" ");
        let values = (0..WIDTH)
            .map(|index| format!("<urn:v{index}>"))
            .collect::<Vec<_>>()
            .join(" ");
        let mut alternatives = Vec::new();
        for branch in 0..WIDTH {
            alternatives.push(format!("{{ BIND(\"{branch}\" AS ?branch) }}"));
        }
        let consumers = (0..WIDTH)
            .map(|index| format!("BIND(STRDT(\"x\", ?v{index}) AS ?out{index})"))
            .collect::<Vec<_>>()
            .join(" ");
        let query = format!(
            "SELECT {projection} WHERE {{ VALUES ({inputs}) {{ ({values}) }} {} {consumers} }}",
            alternatives.join(" UNION "),
        );

        ORDINARY_EXACT_SEED_COPY_WORK.with(|work| work.set(0));
        ORDINARY_EXACT_DEMAND_DELTA_WORK.with(|work| work.set(0));
        translate(&query).unwrap();
        assert_eq!(
            ORDINARY_EXACT_SEED_COPY_WORK.with(std::cell::Cell::get),
            0,
            "UNION branches must share a rollback mark, never clone the downstream seed"
        );
        assert!(
            ORDINARY_EXACT_DEMAND_DELTA_WORK.with(std::cell::Cell::get) <= WIDTH * 8,
            "delta normalization must scale with actual branch changes"
        );
    }

    #[test]
    fn ordinary_exact_structural_shape_does_not_remerge_nested_groups() {
        const DEPTH: usize = 96;
        let mut query = String::from("SELECT ?value WHERE {");
        query.push_str(&"{".repeat(DEPTH));
        query.push_str("BIND(IRI(\"urn:value\") AS ?value)");
        query.push_str(&"}".repeat(DEPTH));
        query.push('}');

        ORDINARY_EXACT_STRUCTURAL_INSERT_WORK.with(|work| work.set(0));
        translate(&query).unwrap();
        let work = ORDINARY_EXACT_STRUCTURAL_INSERT_WORK.with(std::cell::Cell::get);
        assert!(
            work <= 2,
            "nested single-child groups must not repeatedly merge output sets; observed {work} insertions"
        );
    }

    #[test]
    fn ordinary_exact_deep_optional_uses_rollback_without_seed_copies() {
        const DEPTH: usize = 64;
        let mut query = String::from("SELECT ?out WHERE { VALUES ?datatype { <urn:dt> }");
        for _ in 0..DEPTH {
            query.push_str(" OPTIONAL {");
        }
        query.push_str(" BIND(\"marker\" AS ?marker)");
        query.push_str(&"}".repeat(DEPTH));
        query.push_str(" BIND(STRDT(\"x\", ?datatype) AS ?out) }");

        ORDINARY_EXACT_SEED_COPY_WORK.with(|work| work.set(0));
        ORDINARY_EXACT_DEMAND_DELTA_WORK.with(|work| work.set(0));
        translate(&query).unwrap();
        assert_eq!(ORDINARY_EXACT_SEED_COPY_WORK.with(std::cell::Cell::get), 0);
        assert!(
            ORDINARY_EXACT_DEMAND_DELTA_WORK.with(std::cell::Cell::get) <= DEPTH * 2,
            "unchanged OPTIONAL seeds must not be normalized by demand cardinality"
        );
    }

    #[test]
    fn ordinary_exact_modifier_alias_plan_has_unique_exact_projection() {
        let plan = translate(
            r#"SELECT (IRI(?lex) AS ?dt) WHERE {
                   VALUES ?lex { "urn:b" "urn:a" }
               }
               ORDER BY (COALESCE(STR(DATATYPE(STRDT("x", ?dt))), "zzz"))
               LIMIT 1"#,
        )
        .unwrap();
        let mut names = Vec::new();
        fn collect_projected_names(operator: &LogicalOperator, names: &mut Vec<String>) {
            match operator {
                LogicalOperator::Project(project) => {
                    for projection in &project.projections {
                        if let Some(alias) = &projection.alias {
                            names.push(alias.clone());
                        } else if let LogicalExpression::Variable(name) = &projection.expression {
                            names.push(name.clone());
                        }
                    }
                    collect_projected_names(&project.input, names);
                }
                LogicalOperator::Limit(limit) => collect_projected_names(&limit.input, names),
                LogicalOperator::Sort(sort) => collect_projected_names(&sort.input, names),
                _ => {}
            }
        }
        collect_projected_names(&plan.root, &mut names);
        let exact = rdf_exact_term_column("dt");
        assert_eq!(
            names.iter().filter(|name| **name == exact).count(),
            1,
            "logical projection must name the demanded exact companion exactly once"
        );
    }

    #[test]
    fn rdf_join_identity_metadata_preserves_variable_optimizer_edges() {
        fn conditions(operator: &LogicalOperator) -> Vec<&JoinCondition> {
            let mut found = Vec::new();
            fn visit<'a>(operator: &'a LogicalOperator, found: &mut Vec<&'a JoinCondition>) {
                match operator {
                    LogicalOperator::Join(join) => {
                        found.extend(join.conditions.iter());
                        visit(&join.left, found);
                        visit(&join.right, found);
                    }
                    LogicalOperator::Project(project) => visit(&project.input, found),
                    LogicalOperator::Filter(filter) => visit(&filter.input, found),
                    LogicalOperator::Distinct(distinct) => visit(&distinct.input, found),
                    LogicalOperator::Sort(sort) => visit(&sort.input, found),
                    LogicalOperator::Limit(limit) => visit(&limit.input, found),
                    LogicalOperator::Skip(skip) => visit(&skip.input, found),
                    _ => {}
                }
            }
            visit(operator, &mut found);
            found
        }

        let must_bound =
            translate("SELECT ?shared WHERE { ?left <urn:p> ?shared . ?right <urn:q> ?shared }")
                .unwrap();
        let identity = conditions(&must_bound.root).into_iter().find(|condition| {
            condition.semantics == JoinKeySemantics::RdfTermIdentity
                && matches!(&condition.left, LogicalExpression::Variable(name) if name == "shared")
                && matches!(&condition.right, LogicalExpression::Variable(name) if name == "shared")
        });
        assert!(
            identity.is_some(),
            "must-bound RDF joins must retain a plain variable optimizer edge with typed identity semantics"
        );

        let maybe_bound = translate(
            r#"SELECT ?shared WHERE {
                   ?left <urn:p> ?shared
                   { ?right <urn:q> ?shared } UNION { ?other <urn:r> ?unrelated }
               }"#,
        )
        .unwrap();
        assert!(conditions(&maybe_bound.root).into_iter().any(|condition| {
            matches!(&condition.left, LogicalExpression::Variable(name) if name == "shared")
                && matches!(&condition.right, LogicalExpression::Variable(name) if name == "shared")
                && condition.semantics == JoinKeySemantics::SparqlCompatibility
        }));

        for query in [
            "SELECT ?shared WHERE { BIND(<urn:x> AS ?shared) ?right <urn:q> ?shared }",
            "SELECT ?shared WHERE { <urn:start> <urn:p>+ ?shared . ?right <urn:q> ?shared }",
        ] {
            let plan = translate(query).unwrap();
            assert!(conditions(&plan.root).into_iter().any(|condition| {
                matches!(&condition.left, LogicalExpression::Variable(name) if name == "shared")
                    && matches!(&condition.right, LogicalExpression::Variable(name) if name == "shared")
                    && condition.semantics == JoinKeySemantics::RdfTermIdentity
            }), "rigorously bound producer lost RDF identity metadata: {query}");
        }

        let errorable_constructor = translate(
            r#"SELECT ?shared WHERE {
                   VALUES ?lex { "urn:x" }
                   BIND(IRI(?lex) AS ?shared)
                   ?right <urn:q> ?shared
               }"#,
        )
        .unwrap();
        assert!(conditions(&errorable_constructor.root).into_iter().any(|condition| {
            matches!(&condition.left, LogicalExpression::Variable(name) if name == "shared")
                && matches!(&condition.right, LogicalExpression::Variable(name) if name == "shared")
                && condition.semantics == JoinKeySemantics::SparqlCompatibility
        }), "errorable constructors must not be promoted to must-bound identity keys");
    }

    #[test]
    fn test_translate_select_distinct() {
        let query = "SELECT DISTINCT ?x WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_distinct(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Distinct(_) => true,
                LogicalOperator::Project(p) => find_distinct(&p.input),
                _ => false,
            }
        }
        assert!(find_distinct(&plan.root));
    }

    // === Filter Tests ===

    #[test]
    fn test_translate_select_with_filter() {
        let query = "SELECT ?x WHERE { ?x ?y ?z FILTER(?z > 10) }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_filter_equality() {
        let query = r#"SELECT ?x WHERE { ?x ?y ?z FILTER(?z = "test") }"#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_filter_and() {
        let query = "SELECT ?x WHERE { ?x ?y ?z FILTER(?z > 10 && ?z < 100) }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_filter_or() {
        let query = r#"SELECT ?x WHERE { ?x ?y ?z FILTER(?z = 1 || ?z = 2) }"#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_filter_bound() {
        let query = "SELECT ?x WHERE { ?x ?y ?z FILTER(BOUND(?z)) }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === ASK Query Tests ===

    #[test]
    fn test_translate_ask() {
        let query = "ASK { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // ASK should have a Limit(1)
        fn find_limit(op: &LogicalOperator) -> Option<&LimitOp> {
            match op {
                LogicalOperator::Limit(l) => Some(l),
                _ => None,
            }
        }
        let limit = find_limit(&plan.root).expect("Expected Limit");
        assert_eq!(limit.count, 1);
    }

    // === Solution Modifiers Tests ===

    #[test]
    fn test_translate_select_with_limit() {
        let query = "SELECT ?x WHERE { ?x ?y ?z } LIMIT 10";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_with_offset() {
        let query = "SELECT ?x WHERE { ?x ?y ?z } OFFSET 5";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_skip(op: &LogicalOperator) -> Option<&SkipOp> {
            match op {
                LogicalOperator::Skip(s) => Some(s),
                LogicalOperator::Project(p) => find_skip(&p.input),
                _ => None,
            }
        }
        let skip = find_skip(&plan.root).expect("Expected Skip");
        assert_eq!(skip.count, 5);
    }

    #[test]
    fn test_translate_select_with_order_by() {
        let query = "SELECT ?x WHERE { ?x ?y ?z } ORDER BY ?z";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_sort(op: &LogicalOperator) -> Option<&SortOp> {
            match op {
                LogicalOperator::Sort(s) => Some(s),
                LogicalOperator::Project(p) => find_sort(&p.input),
                _ => None,
            }
        }
        assert!(find_sort(&plan.root).is_some());
    }

    #[test]
    fn test_translate_select_with_order_by_desc() {
        let query = "SELECT ?x WHERE { ?x ?y ?z } ORDER BY DESC(?z)";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_sort(op: &LogicalOperator) -> Option<&SortOp> {
            match op {
                LogicalOperator::Sort(s) => Some(s),
                LogicalOperator::Project(p) => find_sort(&p.input),
                _ => None,
            }
        }
        let sort = find_sort(&plan.root).expect("Expected Sort");
        assert_eq!(sort.keys[0].order, SortOrder::Descending);
    }

    // === Graph Pattern Tests ===

    #[test]
    fn test_translate_union() {
        let query = "SELECT ?x WHERE { { ?x ?y ?z } UNION { ?x ?a ?b } }";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_union(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Union(_) => true,
                LogicalOperator::Project(p) => find_union(&p.input),
                _ => false,
            }
        }
        assert!(find_union(&plan.root));
    }

    #[test]
    fn test_translate_optional() {
        let query = "SELECT ?x ?name WHERE { ?x ?y ?z OPTIONAL { ?x ?p ?name } }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_bind() {
        let query = "SELECT ?x ?doubled WHERE { ?x ?y ?z BIND(?z * 2 AS ?doubled) }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === Aggregate Tests ===

    #[test]
    fn test_translate_count() {
        let query = "SELECT (COUNT(?x) AS ?cnt) WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_group_by() {
        let query = "SELECT ?y (COUNT(?x) AS ?cnt) WHERE { ?x ?y ?z } GROUP BY ?y";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_aggregate(op: &LogicalOperator) -> Option<&AggregateOp> {
            match op {
                LogicalOperator::Aggregate(a) => Some(a),
                LogicalOperator::Project(p) => find_aggregate(&p.input),
                _ => None,
            }
        }
        let agg = find_aggregate(&plan.root).expect("Expected Aggregate");
        assert!(!agg.group_by.is_empty());
    }

    #[test]
    fn aggregate_hoist_physical_duplicate_direct_aliases_compute_once() {
        let plan = translate(
            r#"SELECT (SUM(?value) AS ?first) (SUM(?value) AS ?second)
               WHERE { VALUES ?value { 1 2 } }
               HAVING (SUM(?value) > 0)"#,
        )
        .unwrap();

        assert_eq!(
            physical_aggregate_count(&plan.root),
            1,
            "structurally identical direct aggregates must have one physical slot"
        );
        assert_eq!(
            collect_physical_aggregate_aliases(&plan.root),
            ["first"],
            "the first direct projected alias is the canonical aggregate output"
        );
    }

    #[test]
    fn aggregate_hoist_physical_select_having_and_order_reuse_one_slot() {
        let plan = translate(
            r#"SELECT ?group (SUM(?value) AS ?total)
               WHERE { VALUES (?group ?value) { ("a" 1) ("a" 3) ("b" 5) } }
               GROUP BY ?group
               HAVING (SUM(?value) >= 4)
               ORDER BY DESC(SUM(?value))"#,
        )
        .unwrap();

        assert_eq!(physical_aggregate_count(&plan.root), 1);
        assert_eq!(collect_physical_aggregate_aliases(&plan.root), ["total"]);
        assert!(collect_hidden_aggregate_aliases(&plan.root).is_empty());
    }

    #[test]
    fn aggregate_hoist_physical_volatile_select_having_and_order_reuse_one_slot() {
        fn expression_function_count(expression: &LogicalExpression, target: &str) -> usize {
            match expression {
                LogicalExpression::FunctionCall { name, args, .. } => {
                    usize::from(name == target)
                        + args
                            .iter()
                            .map(|arg| expression_function_count(arg, target))
                            .sum::<usize>()
                }
                LogicalExpression::Binary { left, right, .. } => {
                    expression_function_count(left, target)
                        + expression_function_count(right, target)
                }
                LogicalExpression::Unary { operand, .. } => {
                    expression_function_count(operand, target)
                }
                LogicalExpression::List(items) => items
                    .iter()
                    .map(|item| expression_function_count(item, target))
                    .sum(),
                LogicalExpression::Map(entries) => entries
                    .iter()
                    .map(|(_, value)| expression_function_count(value, target))
                    .sum(),
                LogicalExpression::IndexAccess { base, index } => {
                    expression_function_count(base, target)
                        + expression_function_count(index, target)
                }
                LogicalExpression::SliceAccess { base, start, end } => {
                    expression_function_count(base, target)
                        + start
                            .as_deref()
                            .map_or(0, |start| expression_function_count(start, target))
                        + end
                            .as_deref()
                            .map_or(0, |end| expression_function_count(end, target))
                }
                LogicalExpression::Case {
                    operand,
                    when_clauses,
                    else_clause,
                } => {
                    operand
                        .as_deref()
                        .map_or(0, |operand| expression_function_count(operand, target))
                        + when_clauses
                            .iter()
                            .map(|(when, then)| {
                                expression_function_count(when, target)
                                    + expression_function_count(then, target)
                            })
                            .sum::<usize>()
                        + else_clause
                            .as_deref()
                            .map_or(0, |otherwise| expression_function_count(otherwise, target))
                }
                _ => 0,
            }
        }

        fn operator_function_count(operator: &LogicalOperator, target: &str) -> usize {
            let local = match operator {
                LogicalOperator::Bind(bind) => expression_function_count(&bind.expression, target),
                LogicalOperator::Filter(filter) => {
                    expression_function_count(&filter.predicate, target)
                }
                LogicalOperator::Project(project) => project
                    .projections
                    .iter()
                    .map(|projection| expression_function_count(&projection.expression, target))
                    .sum(),
                LogicalOperator::Aggregate(aggregate) => {
                    aggregate
                        .group_by
                        .iter()
                        .map(|expression| expression_function_count(expression, target))
                        .sum::<usize>()
                        + aggregate
                            .aggregates
                            .iter()
                            .map(|aggregate| {
                                aggregate.expression.as_ref().map_or(0, |expression| {
                                    expression_function_count(expression, target)
                                }) + aggregate.expression2.as_ref().map_or(0, |expression| {
                                    expression_function_count(expression, target)
                                }) + aggregate.distinct_key.as_ref().map_or(0, |expression| {
                                    expression_function_count(expression, target)
                                })
                            })
                            .sum::<usize>()
                        + aggregate
                            .having
                            .as_ref()
                            .map_or(0, |having| expression_function_count(having, target))
                }
                LogicalOperator::Sort(sort) => sort
                    .keys
                    .iter()
                    .map(|key| expression_function_count(&key.expression, target))
                    .sum(),
                _ => 0,
            };
            local
                + operator
                    .children()
                    .into_iter()
                    .map(|child| operator_function_count(child, target))
                    .sum::<usize>()
        }

        let inspection_sentinel = LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![LogicalExpression::FunctionCall {
                name: "UUID".to_string(),
                args: Vec::new(),
                distinct: false,
            }],
            aggregates: Vec::new(),
            having: Some(LogicalExpression::FunctionCall {
                name: "UUID".to_string(),
                args: Vec::new(),
                distinct: false,
            }),
            input: Box::new(LogicalOperator::Empty),
        });
        assert_eq!(
            operator_function_count(&inspection_sentinel, "UUID"),
            2,
            "plan inspection must include Aggregate GROUP BY and inline HAVING expressions"
        );

        let plan = translate(
            r#"SELECT (SAMPLE(UUID()) AS ?sample)
                      (IRI(STR(SAMPLE(UUID()))) AS ?copy)
                      (sameTerm(
                        SAMPLE(UUID()),
                        IRI(STR(SAMPLE(UUID())))
                      ) AS ?same)
               WHERE { VALUES ?row { 1 } }
               HAVING (sameTerm(
                 SAMPLE(UUID()),
                 IRI(STR(SAMPLE(UUID())))
               ))
               ORDER BY (SAMPLE(UUID()))"#,
        )
        .expect("volatile aggregate consumers translate");

        assert_eq!(
            physical_aggregate_count(&plan.root),
            1,
            "one structurally repeated SAMPLE has one physical aggregate slot"
        );
        let sealed_sample = rdf_tagged_term_column("aggregate-output:sample");
        assert_eq!(
            collect_physical_aggregate_aliases(&plan.root),
            std::slice::from_ref(&sealed_sample),
            "the one physical slot is the sealed exact SAMPLE result"
        );
        assert!(
            matches!(
                find_bind_expression(&plan.root, "sample"),
                Some(LogicalExpression::FunctionCall { name, args, distinct })
                    if name == RDF_TERM_OR_NATIVE_VISIBLE
                        && !distinct
                        && matches!(
                            args.as_slice(),
                            [LogicalExpression::Variable(source)] if source == &sealed_sample
                        )
            ),
            "the public alias must unwrap the one sealed physical SAMPLE slot"
        );
        assert_eq!(
            operator_function_count(&plan.root, "UUID"),
            1,
            "the volatile aggregate operand is evaluated once per input mapping"
        );
    }

    #[test]
    fn aggregate_hoist_physical_nested_selects_use_distinct_hidden_slots() {
        let plan = translate(
            r#"SELECT (1 AS ?one)
               WHERE {
                 VALUES ?value { <urn:outer> }
                 {
                   SELECT *
                   WHERE { VALUES ?value { <urn:inner> } }
                   HAVING (COUNT(*) > 0)
                 }
               }
               HAVING (COUNT(*) > 0)"#,
        )
        .expect("nested SELECTs allocate independent aggregate registries");

        assert_eq!(physical_aggregate_count(&plan.root), 2);
        let hidden = collect_hidden_aggregate_aliases(&plan.root);
        assert_eq!(hidden.len(), 2);
        assert_ne!(
            hidden[0], hidden[1],
            "inner and outer SELECT registries must allocate fresh hidden columns"
        );
    }

    #[test]
    fn aggregate_hoist_physical_modifier_only_aggregates_are_sealed_and_distinct() {
        let plan = translate(
            r#"SELECT ?group
               WHERE { VALUES (?group ?value) { ("a" 1) ("a" 3) ("b" 5) } }
               GROUP BY ?group
               HAVING (COUNT(*) > 0)
               ORDER BY DESC(SUM(?value))"#,
        )
        .unwrap();

        assert_eq!(physical_aggregate_count(&plan.root), 2);
        let aliases = collect_physical_aggregate_aliases(&plan.root);
        let hidden = collect_hidden_aggregate_aliases(&plan.root);
        assert_eq!(hidden.len(), 2, "both modifier-only slots must be sealed");
        assert_eq!(aliases, hidden, "no public alias may expose a hidden slot");
        assert_ne!(
            hidden[0], hidden[1],
            "distinct aggregates need distinct slots"
        );
    }

    #[test]
    fn aggregate_hoist_physical_registry_length_mismatch_is_internal_error() {
        let select = parse_select(
            r#"SELECT (SUM(?value) AS ?sum)
               WHERE { VALUES ?value { 1 2 } }"#,
        );
        let (annotations, _) =
            SparqlTranslator::ordinary_select_exact_annotations(&select, &HashSet::new())
                .expect("original SELECT annotations build and analyze");
        let mut translator = SparqlTranslator::new();
        let hoist = translator
            .collect_aggregate_hoist(&select, Some(&annotations))
            .expect("aggregate registry collects");
        let mut aggregates = translator
            .extract_aggregates_for_select(&hoist)
            .expect("one physical aggregate extracts");
        aggregates.clear();

        let error = translator
            .prepare_exact_aggregates(LogicalOperator::Empty, &select, &hoist, aggregates)
            .expect_err("registry/physical cardinality mismatch must fail closed");
        assert!(
            matches!(error, Error::Internal(ref message) if message.contains("aggregate-hoist registry/physical length mismatch")),
            "unexpected mismatch error: {error:?}"
        );
    }

    #[test]
    fn aggregate_hoist_physical_hidden_result_has_full_identity_envelope() {
        let plan = translate(
            r#"SELECT (1 AS ?one)
               WHERE { VALUES ?value { 1 2 } }
               HAVING (SUM(?value) > 0)"#,
        )
        .expect("modifier-only aggregate lowers through a sealed result");

        let hidden = collect_hidden_aggregate_aliases(&plan.root);
        assert_eq!(hidden.len(), 1);
        let alias = &hidden[0];
        assert!(contains_bind(&plan.root, &rdf_exact_term_column(alias)));
        assert!(contains_bind(&plan.root, &rdf_identity_key_column(alias)));
        assert!(
            contains_bind(&plan.root, &rdf_group_key_column(alias)),
            "FULL aggregate demand must retain an RDF-or-native group key"
        );
    }

    #[test]
    fn aggregate_hoist_physical_noncanonical_mutation_alias_retains_identity() {
        let plan = translate(
            r#"INSERT { <urn:result> <urn:value> ?used }
               WHERE {
                 {
                   SELECT (SAMPLE(?value) AS ?canonical)
                          (SAMPLE(?value) AS ?used)
                   WHERE { VALUES ?value { <urn:x> } }
                 }
               }"#,
        )
        .expect("an update can consume the later duplicate aggregate alias");

        assert_eq!(
            physical_aggregate_count(&plan.root),
            1,
            "duplicate aliases must still share one physical aggregate"
        );
        assert!(
            contains_bind(&plan.root, &rdf_exact_term_column("canonical")),
            "the canonical slot must retain exact state demanded only by ?used"
        );
        assert!(contains_bind(
            &plan.root,
            &rdf_identity_key_column("canonical")
        ));
        assert!(contains_bind(
            &plan.root,
            &rdf_group_key_column("canonical")
        ));
        assert!(
            contains_bind(&plan.root, &rdf_exact_term_column("used")),
            "the later alias must copy the canonical exact term"
        );
        assert!(
            contains_bind(&plan.root, &rdf_identity_key_column("used")),
            "the later alias must copy the canonical identity key"
        );
        assert!(
            contains_bind(&plan.root, &rdf_group_key_column("used")),
            "the later alias must copy the canonical RDF-or-native group key"
        );
        for (source, copied) in [
            (
                rdf_exact_term_column("canonical"),
                rdf_exact_term_column("used"),
            ),
            (
                rdf_identity_key_column("canonical"),
                rdf_identity_key_column("used"),
            ),
            (
                rdf_group_key_column("canonical"),
                rdf_group_key_column("used"),
            ),
        ] {
            assert!(
                matches!(
                    find_bind_expression(&plan.root, &copied),
                    Some(LogicalExpression::Variable(actual)) if actual == &source
                ),
                "{copied} must copy {source} without rebuilding identity from the visible value"
            );
        }
    }

    #[test]
    fn aggregate_hoist_registry_prefers_direct_alias_and_has_stable_clause_order() {
        let select = parse_select(
            r#"SELECT ((SUM(?value) + COUNT(*)) AS ?nested)
                      (SUM(?value) AS ?direct_sum)
                      (AVG(?value) AS ?direct_average)
               WHERE { VALUES ?value { 1 2 } }
               HAVING (MAX(?value) > 0)
               ORDER BY (MIN(?value))"#,
        );
        let (annotations, _) =
            SparqlTranslator::ordinary_select_exact_annotations(&select, &HashSet::new())
                .expect("original SELECT annotations build and analyze");
        SparqlTranslator::validate_ordinary_select_annotations(&select, &annotations, "SELECT")
            .expect("original SELECT annotations validate");
        let mut translator = SparqlTranslator::new();
        let hoist = translator
            .collect_aggregate_hoist(&select, Some(&annotations))
            .expect("legal aggregates collect");

        assert_eq!(
            hoist
                .entries
                .iter()
                .map(|entry| aggregate_kind(entry.aggregate))
                .collect::<Vec<_>>(),
            ["SUM", "AVG", "COUNT", "MAX", "MIN"]
        );
        assert_eq!(hoist.entries[0].canonical_column, "direct_sum");
        assert!(hoist.entries[0].canonical_is_direct_projection);
        assert_eq!(hoist.entries[0].occurrences.len(), 2);
        assert!(
            hoist.entries[0]
                .occurrences
                .iter()
                .all(|occurrence| occurrence.annotation.is_some())
        );
        assert_eq!(
            hoist.entries[0].occurrences[0].location,
            AggregateHoistLocation::Projection {
                index: 1,
                direct: true,
            }
        );
        assert_eq!(
            hoist.entries[0].occurrences[1].location,
            AggregateHoistLocation::Projection {
                index: 0,
                direct: false,
            }
        );
        assert!(!hoist.entries[2].canonical_is_direct_projection);
        assert!(is_rdf_internal_term_column(
            &hoist.entries[2].canonical_column
        ));
    }

    #[test]
    fn aggregate_hoist_registry_deduplicates_full_ast_only() {
        let select = parse_select(
            r#"SELECT (SUM(?value) AS ?first)
                      (SUM(?value) AS ?second)
                      (SUM(DISTINCT ?value) AS ?distinct_sum)
                      (GROUP_CONCAT(?value; SEPARATOR=",") AS ?comma)
                      (GROUP_CONCAT(?value; SEPARATOR=";") AS ?semicolon)
               WHERE { VALUES ?value { 1 2 } }"#,
        );
        let (annotations, _) =
            SparqlTranslator::ordinary_select_exact_annotations(&select, &HashSet::new())
                .expect("original SELECT annotations build and analyze");
        let mut translator = SparqlTranslator::new();
        let hoist = translator
            .collect_aggregate_hoist(&select, Some(&annotations))
            .expect("legal aggregates collect");

        assert_eq!(hoist.entries.len(), 4);
        assert_eq!(hoist.entries[0].canonical_column, "first");
        assert_eq!(hoist.entries[0].occurrences.len(), 2);
        assert_eq!(
            hoist
                .entries
                .iter()
                .map(|entry| aggregate_kind(entry.aggregate))
                .collect::<Vec<_>>(),
            ["SUM", "SUM", "GROUP_CONCAT", "GROUP_CONCAT"]
        );
        assert_ne!(hoist.entries[0].aggregate, hoist.entries[1].aggregate);
        assert_ne!(hoist.entries[2].aggregate, hoist.entries[3].aggregate);
    }

    #[test]
    fn aggregate_hoist_registry_unions_nested_identity_result_demand() {
        let select = parse_select(
            r#"SELECT (SAMPLE(?value) AS ?direct)
                      (sameTerm(SAMPLE(?value), <urn:x>) AS ?same)
                      (STRDT("lexical", SAMPLE(?value)) AS ?typed)
               WHERE { VALUES ?value { <urn:x> } }"#,
        );
        let (annotations, _) =
            SparqlTranslator::ordinary_select_exact_annotations(&select, &HashSet::new())
                .expect("original SELECT annotations build and analyze");
        let mut translator = SparqlTranslator::new();
        let hoist = translator
            .collect_aggregate_hoist(&select, Some(&annotations))
            .expect("identity-sensitive SAMPLE occurrences collect");

        assert_eq!(hoist.entries.len(), 1);
        let entry = &hoist.entries[0];
        assert_eq!(entry.canonical_column, "direct");
        assert_eq!(entry.occurrences.len(), 3);
        assert!(entry.occurrences[0].annotation.is_some());
        assert!(
            !entry.occurrences[0].demand.full_rdf_or_native,
            "the direct root retains only its analyzed exact demand"
        );
        assert!(
            entry.occurrences[1..]
                .iter()
                .all(|occurrence| occurrence.demand == AggregateHoistResultDemand::FULL)
        );
        assert_eq!(entry.result_demand(), AggregateHoistResultDemand::FULL);
    }

    #[test]
    fn aggregate_hoist_registry_allocates_fresh_sealed_columns() {
        let first = parse_select(
            r#"SELECT ((COUNT(*) + SUM(?value)) AS ?value)
               WHERE { VALUES ?value { 1 2 } }
               HAVING (AVG(?value) > 0)"#,
        );
        let second = parse_select(
            r#"SELECT ((COUNT(*) + 1) AS ?value)
               WHERE { VALUES ?value { 1 2 } }"#,
        );
        let mut translator = SparqlTranslator::new();
        let first_columns = {
            let hoist = translator
                .collect_aggregate_hoist(&first, None)
                .expect("first SELECT aggregates collect");
            hoist
                .entries
                .iter()
                .map(|entry| entry.canonical_column.clone())
                .collect::<Vec<_>>()
        };
        let second_column = translator
            .collect_aggregate_hoist(&second, None)
            .expect("second SELECT aggregates collect")
            .entries[0]
            .canonical_column
            .clone();

        assert_eq!(first_columns.len(), 3);
        assert!(
            first_columns
                .iter()
                .all(|column| is_rdf_internal_term_column(column))
        );
        assert_eq!(first_columns.iter().collect::<HashSet<_>>().len(), 3);
        assert!(is_rdf_internal_term_column(&second_column));
        assert!(!first_columns.contains(&second_column));
    }

    #[test]
    fn aggregate_hoist_registry_does_not_cross_exists_or_subselect_scope() {
        let select = parse_select(
            r#"SELECT (IF(EXISTS {
                         { SELECT (SUM(?inner) AS ?inner_sum)
                           WHERE { VALUES ?inner { 1 2 } } }
                       }, COUNT(*), 0) AS ?value)
               WHERE {}"#,
        );
        SparqlTranslator::validate_supported_aggregate_placements(&select)
            .expect("the inner subselect owns its legal aggregate scope");
        let (annotations, _) =
            SparqlTranslator::ordinary_select_exact_annotations(&select, &HashSet::new())
                .expect("original SELECT annotations build and analyze");
        let mut translator = SparqlTranslator::new();
        let hoist = translator
            .collect_aggregate_hoist(&select, Some(&annotations))
            .expect("outer SELECT aggregates collect");

        assert_eq!(hoist.entries.len(), 1);
        assert_eq!(aggregate_kind(hoist.entries[0].aggregate), "COUNT");
        assert_eq!(
            hoist.entries[0].occurrences[0].location,
            AggregateHoistLocation::Projection {
                index: 0,
                direct: false,
            }
        );
    }

    // === Expression Tests ===

    #[test]
    fn test_translate_arithmetic_expression() {
        let query = "SELECT (?x + ?y AS ?sum) WHERE { ?x ?p ?y }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_string_function() {
        let query = r#"SELECT ?x WHERE { ?x ?y ?z FILTER(CONTAINS(?z, "test")) }"#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === CONSTRUCT and DESCRIBE Tests ===

    #[test]
    fn test_translate_construct() {
        let query = "CONSTRUCT { ?x ?y ?z } WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_describe() {
        let query = "DESCRIBE ?x WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === Multiple Triple Patterns ===

    #[test]
    fn test_translate_multiple_triples() {
        let query = "SELECT ?x ?name ?age WHERE { ?x ?y ?name . ?x ?z ?age }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === Literal Types ===

    #[test]
    fn test_translate_literal_types() {
        let query = r#"SELECT ?x WHERE { ?x ?y 42 . ?x ?z "hello" . ?x ?w true }"#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === Helper Function Tests ===

    #[test]
    fn test_translator_new() {
        let translator = SparqlTranslator::new();
        assert!(translator.prefixes.is_empty());
        assert!(translator.base.is_none());
        assert_eq!(translator.anon_counter, 0);
    }

    #[test]
    fn test_translator_next_anon() {
        let mut translator = SparqlTranslator::new();
        assert_eq!(translator.next_anon(), 0);
        assert_eq!(translator.next_anon(), 1);
        assert_eq!(translator.next_anon(), 2);
    }

    // === SPARQL Update Tests ===

    #[test]
    fn test_translate_insert_data() {
        let query = r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "value" }"#;
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn has_insert_triple(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::InsertTriple(_) => true,
                LogicalOperator::Union(u) => u.inputs.iter().any(has_insert_triple),
                _ => false,
            }
        }
        assert!(has_insert_triple(&plan.root));
    }

    #[test]
    fn test_translate_delete_data() {
        let query = r#"DELETE DATA { <http://ex.org/s> <http://ex.org/p> "value" }"#;
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn has_delete_triple(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::DeleteTriple(_) => true,
                LogicalOperator::Union(u) => u.inputs.iter().any(has_delete_triple),
                _ => false,
            }
        }
        assert!(has_delete_triple(&plan.root));
    }

    #[test]
    fn test_translate_delete_where() {
        let query = r#"DELETE WHERE { ?s <http://ex.org/p> ?o }"#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_modify_delete_insert() {
        let query = r#"
            DELETE { ?s <http://ex.org/old> ?o }
            INSERT { ?s <http://ex.org/new> ?o }
            WHERE { ?s <http://ex.org/old> ?o }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn mutation_bind_iri_carries_sealed_exact_term_expression() {
        let plan = translate(
            r#"INSERT { GRAPH ?graph { <http://ex.org/s> <http://ex.org/p> "v" } }
               WHERE { BIND (IRI("http://ex.org/g") AS ?graph) }"#,
        )
        .unwrap();
        let LogicalOperator::Modify(modify) = plan.root else {
            panic!("expected MODIFY plan");
        };
        let LogicalOperator::Bind(seal) = *modify.where_clause else {
            panic!("expected sealed MODIFY marker");
        };
        assert_eq!(seal.variable, RDF_SEALED_MODIFY_COLUMN);
        let LogicalOperator::Bind(identity) = *seal.input else {
            panic!("expected canonical identity-key BIND");
        };
        assert_eq!(identity.variable, rdf_identity_key_column("graph"));
        assert!(
            matches!(
                identity.expression,
                LogicalExpression::FunctionCall { ref name, .. } if name == RDF_TERM_IDENTITY_KEY
            ),
            "unexpected identity expression: {:?}",
            identity.expression
        );
        let LogicalOperator::Bind(exact) = *identity.input else {
            panic!("expected lossless exact-term BIND");
        };
        assert_eq!(exact.variable, rdf_exact_term_column("graph"));
        assert!(
            matches!(
                exact.expression,
                LogicalExpression::FunctionCall { ref name, .. } if name == RDF_TAG_EXACT
            ),
            "unexpected exact expression: {:?}",
            exact.expression
        );
        let LogicalOperator::Bind(visible) = *exact.input else {
            panic!("expected visible BIND");
        };
        assert_eq!(visible.variable, "graph");
        assert!(matches!(
            visible.expression,
            LogicalExpression::FunctionCall { ref name, .. } if name == RDF_TAG_VALUE
        ));
        let LogicalOperator::Bind(tagged) = *visible.input else {
            panic!("expected tagged RDF-term BIND");
        };
        assert_eq!(tagged.variable, rdf_tagged_term_column("graph"));
        assert!(matches!(
            tagged.expression,
            LogicalExpression::FunctionCall { ref name, .. } if name == RDF_TAG_IRI_TERM
        ));
    }

    #[test]
    fn test_translate_clear_graph() {
        let query = "CLEAR DEFAULT";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::ClearGraph(_)));
    }

    #[test]
    fn test_translate_drop_graph() {
        let query = "DROP GRAPH <http://example.org/graph>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::DropGraph(_)));
    }

    #[test]
    fn test_translate_create_graph() {
        let query = "CREATE GRAPH <http://example.org/newgraph>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::CreateGraph(_)));
    }

    #[test]
    fn test_translate_copy_graph() {
        let query = "COPY DEFAULT TO <http://example.org/backup>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::CopyGraph(_)));
    }

    #[test]
    fn test_translate_move_graph() {
        let query = "MOVE <http://example.org/old> TO <http://example.org/new>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::MoveGraph(_)));
    }

    #[test]
    fn test_translate_add_graph() {
        let query = "ADD <http://example.org/source> TO <http://example.org/dest>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::AddGraph(_)));
    }

    #[test]
    fn test_translate_load_graph() {
        let query = "LOAD <http://example.org/data.ttl>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::LoadGraph(_)));
    }

    #[test]
    fn test_translate_load_into_graph() {
        let query = "LOAD <http://example.org/data.ttl> INTO GRAPH <http://example.org/target>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::LoadGraph(load) = &plan.root {
            assert!(load.destination.is_some());
        } else {
            panic!("Expected LoadGraph operator");
        }
    }

    #[test]
    fn test_translate_silent_operations() {
        let query = "DROP SILENT GRAPH <http://example.org/graph>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::DropGraph(drop) = &plan.root {
            assert!(drop.silent);
        } else {
            panic!("Expected DropGraph operator");
        }
    }

    // === BIND Expression Tests ===

    #[test]
    fn test_translate_bind_with_concat() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?name ?label
            WHERE {
                ?person foaf:name ?name .
                BIND (CONCAT(?name, " test") AS ?label)
            }
        "#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "BIND translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        fn find_bind(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Bind(_) => true,
                LogicalOperator::Project(p) => find_bind(&p.input),
                LogicalOperator::Filter(f) => find_bind(&f.input),
                _ => false,
            }
        }
        assert!(find_bind(&plan.root), "Expected Bind operator in plan");
    }

    // === VALUES Inline Data Tests ===

    #[test]
    fn test_translate_values_inline_data() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?name
            WHERE {
                VALUES ?person { <http://ex.org/alix> }
                ?person foaf:name ?name .
            }
        "#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "VALUES translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // VALUES with a single value produces a Bind chain joined with the
        // triple pattern. Walk the tree and verify we find at least one Bind.
        fn find_bind(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Bind(_) => true,
                LogicalOperator::Project(p) => find_bind(&p.input),
                LogicalOperator::Filter(f) => find_bind(&f.input),
                LogicalOperator::Join(j) => find_bind(&j.left) || find_bind(&j.right),
                LogicalOperator::Union(u) => u.inputs.iter().any(find_bind),
                _ => false,
            }
        }
        assert!(find_bind(&plan.root), "Expected Bind from VALUES clause");
    }

    #[test]
    fn values_undef_keeps_one_left_plan_and_uses_compatibility_metadata() {
        fn expression_calls(expression: &LogicalExpression, target: &str) -> bool {
            match expression {
                LogicalExpression::FunctionCall { name, args, .. } => {
                    name == target || args.iter().any(|arg| expression_calls(arg, target))
                }
                LogicalExpression::Binary { left, right, .. } => {
                    expression_calls(left, target) || expression_calls(right, target)
                }
                LogicalExpression::Unary { operand, .. } => expression_calls(operand, target),
                LogicalExpression::List(items) => {
                    items.iter().any(|item| expression_calls(item, target))
                }
                LogicalExpression::Map(entries) => entries
                    .iter()
                    .any(|(_, value)| expression_calls(value, target)),
                LogicalExpression::IndexAccess { base, index } => {
                    expression_calls(base, target) || expression_calls(index, target)
                }
                LogicalExpression::SliceAccess { base, start, end } => {
                    expression_calls(base, target)
                        || start
                            .as_deref()
                            .is_some_and(|start| expression_calls(start, target))
                        || end
                            .as_deref()
                            .is_some_and(|end| expression_calls(end, target))
                }
                LogicalExpression::Case {
                    operand,
                    when_clauses,
                    else_clause,
                } => {
                    operand
                        .as_deref()
                        .is_some_and(|operand| expression_calls(operand, target))
                        || when_clauses.iter().any(|(when, then)| {
                            expression_calls(when, target) || expression_calls(then, target)
                        })
                        || else_clause
                            .as_deref()
                            .is_some_and(|otherwise| expression_calls(otherwise, target))
                }
                _ => false,
            }
        }

        fn count_function(operator: &LogicalOperator, target: &str) -> usize {
            let local = match operator {
                LogicalOperator::Bind(bind) => {
                    usize::from(expression_calls(&bind.expression, target))
                }
                _ => 0,
            };
            local
                + operator
                    .children()
                    .into_iter()
                    .map(|child| count_function(child, target))
                    .sum::<usize>()
        }

        fn has_compatibility_join(operator: &LogicalOperator) -> bool {
            match operator {
                LogicalOperator::Join(join) => {
                    join.conditions.iter().any(|condition| {
                        condition.semantics == JoinKeySemantics::SparqlCompatibility
                    }) || has_compatibility_join(&join.left)
                        || has_compatibility_join(&join.right)
                }
                LogicalOperator::Project(project) => has_compatibility_join(&project.input),
                LogicalOperator::Filter(filter) => has_compatibility_join(&filter.input),
                LogicalOperator::Union(union) => union.inputs.iter().any(has_compatibility_join),
                _ => false,
            }
        }

        let plan = translate(
            r#"SELECT ?token WHERE {
                   BIND(UUID() AS ?token)
                   VALUES ?token { UNDEF UNDEF }
               }"#,
        )
        .unwrap();

        assert_eq!(
            count_function(&plan.root, "UUID"),
            1,
            "the volatile left mapping must not be cloned per VALUES row"
        );
        assert!(
            has_compatibility_join(&plan.root),
            "UNDEF must lower through explicit SPARQL compatibility semantics"
        );
    }

    #[test]
    fn unit_solution_not_exists_lowers_to_an_anti_join() {
        let plan = translate(
            r#"SELECT (1 AS ?kept) WHERE {
                   FILTER NOT EXISTS { VALUES ?right { 1 } }
               }"#,
        )
        .unwrap();
        fn contains_not_exists(operator: &LogicalOperator) -> bool {
            matches!(
                operator,
                LogicalOperator::AntiJoin(AntiJoinOp {
                    semantics: AntiJoinSemantics::NotExists,
                    ..
                })
            ) || operator.children().into_iter().any(contains_not_exists)
        }
        assert!(
            contains_not_exists(&plan.root),
            "unit solution lost NOT EXISTS lowering: {:#?}",
            plan.root
        );
    }

    // === OneOrMore Property Path Tests ===

    #[test]
    fn test_translate_one_or_more_property_path() {
        let query = "SELECT ?s ?o WHERE { ?s <http://ex.org/p>+ ?o }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "OneOrMore path translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        fn find_property_path(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::PropertyPath(p) => {
                    p.min_hops == 1 && matches!(p.path, PathStep::Iri { inverse: false, .. })
                }
                LogicalOperator::Project(p) => find_property_path(&p.input),
                LogicalOperator::Distinct(d) => find_property_path(&d.input),
                _ => false,
            }
        }
        assert!(
            find_property_path(&plan.root),
            "Expected native PropertyPath for simple IRI path+"
        );
    }

    // === ZeroOrMore Property Path Tests ===

    #[test]
    fn test_translate_zero_or_more_property_path() {
        let query = "SELECT ?s ?o WHERE { ?s <http://ex.org/p>* ?o }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "ZeroOrMore path translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        fn find_property_path(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::PropertyPath(p) => {
                    p.min_hops == 0 && matches!(p.path, PathStep::Iri { inverse: false, .. })
                }
                LogicalOperator::Project(p) => find_property_path(&p.input),
                LogicalOperator::Distinct(d) => find_property_path(&d.input),
                _ => false,
            }
        }
        assert!(
            find_property_path(&plan.root),
            "Expected native PropertyPath for simple IRI path*"
        );
    }

    // === Sequence Property Path Tests ===

    #[test]
    fn test_translate_sequence_property_path() {
        let query = r#"
            SELECT ?name
            WHERE {
                ?person <http://ex.org/knows>/<http://ex.org/name> ?name
            }
        "#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Sequence path translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // Sequence path expands into joined triple scans. Count TripleScan operators.
        fn count_triple_scans(op: &LogicalOperator) -> usize {
            match op {
                LogicalOperator::TripleScan(_) => 1,
                LogicalOperator::Project(p) => count_triple_scans(&p.input),
                LogicalOperator::Filter(f) => count_triple_scans(&f.input),
                LogicalOperator::Join(j) => {
                    count_triple_scans(&j.left) + count_triple_scans(&j.right)
                }
                _ => 0,
            }
        }
        let scan_count = count_triple_scans(&plan.root);
        assert!(
            scan_count >= 2,
            "Sequence path should produce at least 2 joined TripleScans, got {}",
            scan_count
        );
    }

    // === Alternative Property Path Tests ===

    #[test]
    fn test_translate_alternative_property_path() {
        let query = "SELECT ?v WHERE { ?s <http://a>|<http://b> ?v }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Alternative path translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // Alternative path produces a Union of TripleScans
        fn find_union(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Union(_) => true,
                LogicalOperator::Project(p) => find_union(&p.input),
                _ => false,
            }
        }
        assert!(
            find_union(&plan.root),
            "Expected Union for alternative property path"
        );

        fn count_union_branches(op: &LogicalOperator) -> Option<usize> {
            match op {
                LogicalOperator::Union(u) => Some(u.inputs.len()),
                LogicalOperator::Project(p) => count_union_branches(&p.input),
                _ => None,
            }
        }
        let branch_count =
            count_union_branches(&plan.root).expect("Expected Union in plan for alternative path");
        assert_eq!(
            branch_count, 2,
            "Alternative path with 2 predicates should have 2 Union branches"
        );
    }

    // === Inverse Property Path Tests ===

    #[test]
    fn test_translate_inverse_property_path() {
        let query = "SELECT ?s WHERE { ?o ^<http://ex.org/knows> ?s }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Inverse path translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // Inverse path swaps subject and object, so we should get a TripleScan
        // where the predicate is the inner IRI.
        fn find_triple_scan(op: &LogicalOperator) -> Option<&TripleScanOp> {
            match op {
                LogicalOperator::TripleScan(ts) => Some(ts),
                LogicalOperator::Project(p) => find_triple_scan(&p.input),
                LogicalOperator::Filter(f) => find_triple_scan(&f.input),
                LogicalOperator::Join(j) => {
                    find_triple_scan(&j.left).or_else(|| find_triple_scan(&j.right))
                }
                _ => None,
            }
        }
        let scan = find_triple_scan(&plan.root).expect("Expected TripleScan for inverse path");
        // The predicate should be the IRI (not the inverse wrapper)
        assert!(
            matches!(&scan.predicate, TripleComponent::Iri(_)),
            "Expected IRI predicate in TripleScan after inverse, got {:?}",
            scan.predicate
        );
    }

    // === translate() returns Ok for various path types ===

    #[test]
    fn test_translate_ok_for_named_iri_path() {
        let query = "SELECT ?s ?o WHERE { ?s <http://ex.org/rel> ?o }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Named IRI path should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_translate_ok_for_zero_or_one_path() {
        let query = "SELECT ?s ?o WHERE { ?s <http://ex.org/rel>? ?o }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "ZeroOrOne path should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_translate_ok_for_rdf_type_shorthand() {
        let query = "SELECT ?s WHERE { ?s a <http://ex.org/Person> }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "'a' (rdf:type) shorthand should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_translate_ok_for_negated_property_set() {
        let query = "SELECT ?s ?o WHERE { ?s !<http://ex.org/skip> ?o }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Negated property set should translate: {:?}",
            result.err()
        );
    }

    // === Basic SELECT verification ===

    #[test]
    fn test_translate_basic_select_structure() {
        let query = "SELECT ?x ?y WHERE { ?x <http://ex.org/p> ?y }";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Top-level should be a Project
        fn find_project(op: &LogicalOperator) -> Option<&ProjectOp> {
            match op {
                LogicalOperator::Project(p) => Some(p),
                _ => None,
            }
        }
        let project = find_project(&plan.root).expect("Expected Project operator at top level");
        assert_eq!(
            project.projections.len(),
            2,
            "SELECT ?x ?y should produce 2 projections"
        );
    }

    // === OPTIONAL pattern ===

    #[test]
    fn test_translate_optional_produces_left_join() {
        let query = "SELECT ?x ?name WHERE { ?x <http://ex.org/type> ?t OPTIONAL { ?x <http://ex.org/name> ?name } }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "OPTIONAL should translate: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        fn find_left_join(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::LeftJoin(_) => true,
                LogicalOperator::Project(p) => find_left_join(&p.input),
                LogicalOperator::Filter(f) => find_left_join(&f.input),
                _ => false,
            }
        }
        assert!(
            find_left_join(&plan.root),
            "OPTIONAL should produce a LeftJoin operator in the plan"
        );
    }

    // === SERVICE clause tests (federated queries not supported) ===

    #[test]
    fn test_service_clause_returns_explicit_error() {
        let result =
            translate("SELECT ?x WHERE { SERVICE <http://example.org/sparql> { ?x ?p ?o } }");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("SERVICE"),
            "Error should mention SERVICE, got: {err}"
        );
    }

    #[test]
    fn test_service_silent_clause_also_errors() {
        let result = translate(
            "SELECT ?x WHERE { SERVICE SILENT <http://example.org/sparql> { ?x ?p ?o } }",
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("SERVICE"),
            "SILENT SERVICE should also error, got: {err}"
        );
    }

    #[test]
    fn test_service_clause_with_local_patterns_still_errors() {
        // Ensure SERVICE is not silently executed locally
        let result =
            translate("SELECT ?x ?y WHERE { ?x ?p ?y . SERVICE <http://remote/> { ?y ?q ?z } }");
        assert!(result.is_err());
    }

    #[test]
    fn test_query_without_service_still_works() {
        // Regression: ensure normal queries still translate fine
        let result = translate("SELECT ?x ?y WHERE { ?x ?p ?y }");
        assert!(result.is_ok());
    }

    // === SAMPLE aggregate tests (3G verification) ===

    #[test]
    fn test_sample_aggregate_translates() {
        let result = translate(
            "SELECT (SAMPLE(?name) AS ?sampleName) WHERE { ?x <http://example.org/name> ?name }",
        );
        assert!(
            result.is_ok(),
            "SAMPLE aggregate should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_group_concat_with_separator_translates() {
        let result = translate(
            r#"SELECT (GROUP_CONCAT(?name; separator=", ") AS ?names) WHERE { ?x <http://example.org/name> ?name }"#,
        );
        assert!(
            result.is_ok(),
            "GROUP_CONCAT with separator should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_group_concat_without_separator_translates() {
        let result = translate(
            "SELECT (GROUP_CONCAT(?name) AS ?names) WHERE { ?x <http://example.org/name> ?name }",
        );
        assert!(
            result.is_ok(),
            "GROUP_CONCAT without separator should translate: {:?}",
            result.err()
        );
    }

    // === Property path depth tests (3D verification) ===

    #[test]
    fn test_property_path_plus_translates_beyond_10() {
        let result = translate("SELECT ?x ?y WHERE { ?x <http://example.org/knows>+ ?y }");
        assert!(
            result.is_ok(),
            "Property path + should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_property_path_star_translates() {
        let result = translate("SELECT ?x ?y WHERE { ?x <http://example.org/knows>* ?y }");
        assert!(
            result.is_ok(),
            "Property path * should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_property_path_optional_translates() {
        let result = translate("SELECT ?x ?y WHERE { ?x <http://example.org/knows>? ?y }");
        assert!(
            result.is_ok(),
            "Property path ? should translate: {:?}",
            result.err()
        );
    }
}
