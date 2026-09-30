//! RDF Query Planner.
//!
//! Converts logical plans with RDF operators (TripleScan, etc.) to physical
//! operators that execute against an RDF store.
//!
//! This planner follows the same push-based, vectorized execution model as
//! the LPG planner for consistent performance characteristics.

mod aggregate;
mod numeric;
mod sort;

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use grafeo_common::types::{
    HashableValue, INTERNAL_RDF_TAGGED_TERM_MARKER, LogicalType, TransactionId, ValidTimeInterval,
    Value,
};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::execution::operators::{
    BinaryFilterOp, FilterExpression, FilterOperator, Operator, OperatorError, Predicate,
    SingleRowOperator, UnaryFilterOp,
};
use grafeo_core::execution::{
    DataChunk, QueryResourceContext, QueryResourceContextError, ValueVector,
};
use grafeo_core::graph::rdf::{Literal, RdfStore, Term, Triple, TriplePattern};

use crate::query::plan::{
    AddGraphOp, AggregateFunction as LogicalAggregateFunction, AggregateOp, AntiJoinOp,
    AntiJoinSemantics, BindOp, ClearGraphOp, ConstructOp, CopyGraphOp, CreateGraphOp,
    DatasetRestriction, DeleteTripleOp, DistinctOp, DropGraphOp, FilterOp, InsertTripleOp,
    JoinCondition, JoinKeySemantics, JoinType, LeftJoinOp, LimitOp, LoadGraphOp, LogicalExpression,
    LogicalOperator, LogicalPlan, ModifyOp, MoveGraphOp, PathStep, PropertyPathOp,
    RDF_DISTINCT_TERM_OR_VALUE_KEY, RDF_EXACT_TERM_COLUMN_PREFIX,
    RDF_EXPLICIT_EMPTY_DEFAULT_DATASET, RDF_EXPLICIT_EMPTY_NAMED_DATASET,
    RDF_GROUP_KEY_COLUMN_PREFIX, RDF_IDENTITY_KEY_COLUMN_PREFIX, RDF_IDENTITY_OR_NATIVE_KEY,
    RDF_IS_BLANK, RDF_IS_IRI, RDF_IS_LITERAL, RDF_IS_NUMERIC, RDF_NUMERIC_VALUE, RDF_SAME_TERM,
    RDF_SEALED_MODIFY_COLUMN, RDF_TAG_BLANK_TERM, RDF_TAG_BOUND_TERM, RDF_TAG_EXACT,
    RDF_TAG_IRI_TERM, RDF_TAG_LANG_LITERAL_TERM, RDF_TAG_LITERAL_TERM, RDF_TAG_TYPED_LITERAL_TERM,
    RDF_TAG_VALUE, RDF_TERM_EQUAL, RDF_TERM_IDENTITY_KEY, RDF_TERM_IN, RDF_TERM_OR_NATIVE_EXACT,
    RDF_TERM_OR_NATIVE_VALUE, RDF_TERM_OR_NATIVE_VISIBLE, SkipOp, SortOp, TripleComponent,
    TripleScanOp, TripleTemplate, UnaryOp, is_rdf_internal_term_column, rdf_exact_term_column,
    rdf_graph_variable_from_template, rdf_group_key_column, rdf_identity_key_column,
};
use crate::query::planner::{PhysicalPlan, convert_aggregate_function, convert_filter_expression};

use self::aggregate::{RdfAggregateOperator, RdfRowIdentityColumn};
use self::numeric::RdfNumeric;
use self::sort::RdfSortOperator;

#[cfg(feature = "regex")]
use regex::Regex;
#[cfg(all(feature = "regex-lite", not(feature = "regex")))]
use regex_lite::Regex;

/// Default chunk size for morsel-driven execution.
const DEFAULT_CHUNK_SIZE: usize = 1024;

/// Native multi-pattern Ring qualification gate.
///
/// Selection remains conditional on the per-plan admission proof in
/// `try_leapfrog_ring`: fresh default-graph scans, exhaustive same-name typed
/// RDF-identity metadata, no transactional overlay, and no unsupported wrapper.
#[cfg(feature = "ring-index")]
fn rdf_native_ring_multi_pattern_is_qualified() -> bool {
    true
}

/// Returns whether a LIMIT can reach a native Ring join through row-preserving
/// logical wrappers only. Blocking, filtering, deduplicating, and offsetting
/// operators deliberately prevent this output cap from crossing their boundary.
#[cfg(feature = "ring-index")]
fn rdf_native_ring_limit_passthrough(input: &LogicalOperator) -> bool {
    match input {
        LogicalOperator::MultiWayJoin(_) => true,
        LogicalOperator::Project(project) => rdf_native_ring_limit_passthrough(&project.input),
        LogicalOperator::Return(ret) if !ret.distinct => {
            rdf_native_ring_limit_passthrough(&ret.input)
        }
        _ => false,
    }
}

#[cfg(test)]
thread_local! {
    static RDF_VOLATILE_EVALUATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Logs an RDF WAL record. Fail-closed: a log error is an operator error.
#[cfg(feature = "wal")]
fn log_rdf_wal(
    wal: &Option<Arc<RdfWal>>,
    record: &grafeo_storage::wal::WalRecord,
) -> std::result::Result<(), OperatorError> {
    if let Some(wal) = wal {
        wal.log(record)
            .map_err(|err| OperatorError::Execution(format!("RDF WAL log failed: {err}")))?;
    }
    Ok(())
}

/// Requires a transaction before a WAL-backed graph operation can change state.
#[cfg(feature = "wal")]
fn require_graph_wal_transaction(
    wal: &Option<Arc<RdfWal>>,
    transaction_id: Option<TransactionId>,
) -> std::result::Result<(), OperatorError> {
    if wal.is_some() && transaction_id.is_none() {
        return Err(OperatorError::Execution(
            "WAL-backed RDF graph mutation requires an active transaction".to_string(),
        ));
    }
    Ok(())
}

/// Builds the stable WAL shape for one RDF insert.
#[cfg(feature = "wal")]
fn rdf_insert_wal_record(
    triple: &Triple,
    graph: Option<&str>,
    graph_incarnation: grafeo_common::types::GraphIncarnationId,
    transaction_id: TransactionId,
    valid_time: Option<ValidTimeInterval>,
) -> grafeo_storage::wal::WalRecord {
    let (valid_from_tai_ns, valid_to_tai_ns) = valid_time.map_or((None, None), |valid| {
        (Some(valid.from().as_i128()), Some(valid.to().as_i128()))
    });
    grafeo_storage::wal::WalRecord::InsertRdfQuadV3 {
        subject: term_to_wal(triple.subject()),
        predicate: term_to_wal(triple.predicate()),
        object: term_to_wal(triple.object()),
        graph: graph.map(str::to_string),
        graph_incarnation,
        valid_from_tai_ns,
        valid_to_tai_ns,
        transaction_id,
    }
}

/// Builds the exact, graph-incarnation-qualified WAL shape for one RDF delete.
#[cfg(feature = "wal")]
fn rdf_delete_wal_record(
    triple: &Triple,
    graph: Option<&str>,
    graph_incarnation: grafeo_common::types::GraphIncarnationId,
    transaction_id: TransactionId,
) -> grafeo_storage::wal::WalRecord {
    grafeo_storage::wal::WalRecord::DeleteRdfQuadV3 {
        subject: term_to_wal(triple.subject()),
        predicate: term_to_wal(triple.predicate()),
        object: term_to_wal(triple.object()),
        graph: graph.map(str::to_string),
        graph_incarnation,
        transaction_id,
    }
}

#[cfg(feature = "wal")]
fn ensure_rdf_graph_high_water(
    wal: &Option<Arc<RdfWal>>,
    store: &RdfStore,
    graph: Option<&str>,
) -> std::result::Result<(), OperatorError> {
    if graph.is_some()
        && let Some(wal) = wal
    {
        wal.ensure_graph_high_water(store)?;
    }
    Ok(())
}

#[cfg(feature = "wal")]
fn active_graph_incarnation(
    store: &RdfStore,
    graph: Option<&str>,
    transaction_id: Option<TransactionId>,
) -> std::result::Result<grafeo_common::types::GraphIncarnationId, OperatorError> {
    match graph {
        None => Ok(grafeo_common::types::GraphIncarnationId::DEFAULT_GRAPH),
        Some(name) => store
            .graph_in_transaction(name, transaction_id)
            .map(|graph| graph.graph_incarnation())
            .ok_or_else(|| {
                OperatorError::Execution(format!(
                    "RDF graph <{name}> disappeared before WAL framing"
                ))
            }),
    }
}

/// Resolves an RDF deletion target without creating a named-graph lifetime.
fn rdf_delete_target(
    store: &Arc<RdfStore>,
    graph: Option<&str>,
    transaction_id: Option<TransactionId>,
) -> Option<Arc<RdfStore>> {
    match graph {
        Some(name) => transaction_id.map_or_else(
            || store.graph(name),
            |transaction_id| store.graph_for_mutation_in_tx(name, transaction_id),
        ),
        None => Some(Arc::clone(store)),
    }
}

/// Resolves aliases to the exact visible statement retained by the graph.
fn rdf_visible_representative(
    target: &RdfStore,
    transaction_id: Option<TransactionId>,
    triple: &Triple,
) -> Option<Arc<Triple>> {
    target
        .find_with_pending(
            &TriplePattern {
                subject: Some(triple.subject().clone()),
                predicate: Some(triple.predicate().clone()),
                object: Some(triple.object().clone()),
            },
            transaction_id,
        )
        .into_iter()
        .next()
}

fn rdf_triple_visible(
    store: &Arc<RdfStore>,
    graph: Option<&str>,
    transaction_id: Option<TransactionId>,
    triple: &Triple,
) -> bool {
    rdf_delete_target(store, graph, transaction_id)
        .is_some_and(|target| rdf_visible_representative(&target, transaction_id, triple).is_some())
}

/// Ensures that COPY/MOVE/ADD has the destination graph required by SPARQL
/// Update, even when the source is empty and no triple insertion would create
/// it as a side effect.
fn ensure_graph_operation_destination(
    store: &RdfStore,
    destination: Option<&str>,
    transaction_id: Option<TransactionId>,
) -> std::result::Result<bool, OperatorError> {
    let Some(name) = destination else {
        return Ok(false);
    };
    if store.graph_in_transaction(name, transaction_id).is_some() {
        return Ok(false);
    }
    store
        .graph_or_create_in_tx(name, transaction_id)
        .map_err(|error| {
            OperatorError::Execution(format!(
                "failed to create RDF destination graph <{name}>: {error}"
            ))
        })?;
    if store.graph_in_transaction(name, transaction_id).is_none() {
        return Err(OperatorError::Execution(format!(
            "RDF destination graph <{name}> was not published in the transaction"
        )));
    }
    Ok(true)
}

/// Frames a destination graph created by COPY/MOVE/ADD even when there are no
/// inserted triples from which replay could otherwise infer its lifecycle.
#[cfg(feature = "wal")]
fn log_graph_operation_destination_create(
    wal: &Option<Arc<RdfWal>>,
    store: &RdfStore,
    destination: Option<&str>,
    transaction_id: Option<TransactionId>,
    created: bool,
) -> std::result::Result<(), OperatorError> {
    let (true, Some(name)) = (created, destination) else {
        return Ok(());
    };
    ensure_rdf_graph_high_water(wal, store, destination)?;
    if let Some(tid) = transaction_id {
        let incarnation = active_graph_incarnation(store, destination, Some(tid))?;
        log_rdf_wal(
            wal,
            &grafeo_storage::wal::WalRecord::CreateNamedRdfGraphV2 {
                name: name.to_string(),
                incarnation,
                transaction_id: tid,
            },
        )
    } else {
        Ok(())
    }
}

/// Logs dest-graph triples after COPY/MOVE/ADD so replay rebuilds the graph.
#[cfg(feature = "wal")]
fn log_tagged_triples(
    wal: &Option<Arc<RdfWal>>,
    store: &RdfStore,
    graph: Option<&str>,
    graph_incarnation: grafeo_common::types::GraphIncarnationId,
    deleted: &[Triple],
    inserted: &[(Triple, Option<ValidTimeInterval>)],
    tid: TransactionId,
) -> std::result::Result<(), OperatorError> {
    ensure_rdf_graph_high_water(wal, store, graph)?;
    for t in deleted {
        log_rdf_wal(
            wal,
            &rdf_delete_wal_record(t, graph, graph_incarnation, tid),
        )?;
    }
    for (triple, valid_time) in inserted {
        log_rdf_wal(
            wal,
            &rdf_insert_wal_record(triple, graph, graph_incarnation, tid, *valid_time),
        )?;
    }
    Ok(())
}

/// Converts a Term to its N-Triples string for WAL serialization.
#[cfg(feature = "wal")]
fn term_to_wal(term: &Term) -> String {
    term.to_string()
}

/// Records a triple insertion to the CDC log if one is configured.
#[cfg(feature = "cdc")]
fn record_cdc_triple_insert(
    cdc_log: &Option<Arc<RdfCdcSink>>,
    subject: &Term,
    predicate: &Term,
    object: &Term,
    graph: Option<&str>,
    incarnation: grafeo_common::types::GraphIncarnationId,
) {
    if let Some(sink) = cdc_log {
        sink.record(
            crate::cdc::ChangeKind::Create,
            subject,
            predicate,
            object,
            (graph, incarnation),
        );
    }
}

/// Records a triple deletion to the CDC log if one is configured.
#[cfg(feature = "cdc")]
fn record_cdc_triple_delete(
    cdc_log: &Option<Arc<RdfCdcSink>>,
    subject: &Term,
    predicate: &Term,
    object: &Term,
    graph: Option<&str>,
    incarnation: grafeo_common::types::GraphIncarnationId,
) {
    if let Some(sink) = cdc_log {
        sink.record(
            crate::cdc::ChangeKind::Delete,
            subject,
            predicate,
            object,
            (graph, incarnation),
        );
    }
}

/// RDF CDC destination owned by the Session's transaction accumulator.
///
/// Mutation operators stage pending events; commit publishes them with the
/// data's epoch, while rollback discards them.
#[cfg(feature = "cdc")]
struct RdfCdcSink {
    log: Arc<crate::cdc::CdcLog>,
    pending_events: Arc<crate::cdc::TransactionChangeAccumulator>,
}

#[cfg(feature = "cdc")]
impl RdfCdcSink {
    fn record(
        &self,
        kind: crate::cdc::ChangeKind,
        subject: &Term,
        predicate: &Term,
        object: &Term,
        coordinate: (Option<&str>, grafeo_common::types::GraphIncarnationId),
    ) {
        let (graph, incarnation) = coordinate;
        let mut event = self.log.triple_event(
            kind,
            &subject.to_string(),
            &predicate.to_string(),
            &object.to_string(),
            graph,
            grafeo_common::types::EpochId::PENDING,
        );
        event.graph_incarnation = Some(incarnation);
        self.pending_events.stage(event);
    }
}

/// Fail-closed WAL sink used by RDF physical mutation operators.
///
/// [`grafeo_storage::wal::TypedWal`] is itself sticky-poisoned. Keeping the
/// database-wide poison flag here additionally makes the failure visible to
/// every Session immediately, including when an explicit transaction catches
/// the operator error before attempting commit.
#[cfg(feature = "wal")]
struct RdfWal {
    inner: Arc<grafeo_storage::wal::LpgWal>,
    durability_poisoned: Option<Arc<std::sync::atomic::AtomicBool>>,
    logged_graph_high_water: parking_lot::Mutex<u64>,
}

#[cfg(feature = "wal")]
impl RdfWal {
    fn new(
        inner: Arc<grafeo_storage::wal::LpgWal>,
        durability_poisoned: Option<Arc<std::sync::atomic::AtomicBool>>,
    ) -> Self {
        Self {
            inner,
            durability_poisoned,
            logged_graph_high_water: parking_lot::Mutex::new(0),
        }
    }

    fn log(
        &self,
        record: &grafeo_storage::wal::WalRecord,
    ) -> grafeo_common::utils::error::Result<()> {
        let result = self.inner.log(record);
        if result.is_err()
            && let Some(poisoned) = &self.durability_poisoned
        {
            poisoned.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        result
    }

    fn ensure_graph_high_water(&self, store: &RdfStore) -> std::result::Result<(), OperatorError> {
        let next = store.next_graph_incarnation();
        let mut logged = self.logged_graph_high_water.lock();
        if next.as_u64() <= *logged {
            return Ok(());
        }
        self.log(
            &grafeo_storage::wal::WalRecord::RdfGraphIncarnationHighWaterMeta {
                store_id: store.store_id(),
                next_incarnation: next,
            },
        )
        .map_err(|error| {
            OperatorError::Execution(format!(
                "RDF graph-incarnation WAL metadata failed: {error}"
            ))
        })?;
        *logged = next.as_u64();
        Ok(())
    }
}

/// Groups the variable-substitution operands for pattern-based mutation operators.
///
/// Used to keep `RdfInsertPatternOperator::new` and `RdfDeletePatternOperator::new`
/// within the 7-argument clippy limit.
struct TripleOperands {
    subject: TripleComponent,
    predicate: TripleComponent,
    object: TripleComponent,
    column_map: HashMap<String, usize>,
    graph: Option<String>,
    transaction_id: Option<TransactionId>,
    valid_time: Option<ValidTimeInterval>,
}

/// Converts logical plans with RDF operators to physical operators.
///
/// This planner produces push-based operators that process data in chunks
/// (morsels) for cache efficiency and parallelism compatibility.
#[cfg_attr(
    feature = "cdc",
    doc = r#"
CDC mutations stage through the Session-owned transaction accumulator.
The direct log/vector constructor is unavailable.

```compile_fail,E0599
use std::sync::Arc;
use grafeo_common::types::EpochId;
use grafeo_core::graph::rdf::RdfStore;
use grafeo_engine::query::planner::rdf::RdfPlanner;

let planner = RdfPlanner::new(Arc::new(RdfStore::new()));
let _ = planner.with_cdc_log(None, None, EpochId::INITIAL);
```
"#
)]
pub struct RdfPlanner {
    /// The RDF store to query.
    store: Arc<RdfStore>,
    /// Chunk size for vectorized execution.
    chunk_size: usize,
    /// Optional transaction ID for transactional operations.
    transaction_id: Option<TransactionId>,
    /// Session-scoped application valid-time captured at plan construction.
    valid_time: Option<ValidTimeInterval>,
    /// When true, each physical operator is wrapped in `ProfiledOperator`.
    profiling: std::cell::Cell<bool>,
    /// Profile entries collected during planning (post-order).
    profile_entries: std::cell::RefCell<Vec<crate::query::profile::ProfileEntry>>,
    /// Optional WAL for logging RDF mutations.
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
    /// Optional CDC log for recording RDF triple mutations.
    #[cfg(feature = "cdc")]
    cdc_log: Option<Arc<RdfCdcSink>>,
    /// Whether the query uses LANG()/LANGMATCHES()/DATATYPE() functions.
    /// When false, companion columns are not emitted, saving ~66% scan overhead.
    needs_companion_columns: std::cell::Cell<bool>,
    /// Whether scans must carry lossless hidden N-Triples terms into an exact
    /// RDF consumer. Visible `Value::String` columns cannot distinguish an
    /// arbitrary-scheme IRI from a simple literal.
    needs_exact_term_columns: std::cell::Cell<bool>,
    /// Whether scans must carry canonical RDF identity keys for typed joins.
    /// These keys are deliberately separate from lossless reconstruction data.
    needs_identity_key_columns: std::cell::Cell<bool>,
    /// Term dictionary for dictionary-encoded triple scans. When present,
    /// `plan_triple_scan()` emits Int64 term IDs instead of strings, and a
    /// `DictResolveOperator` at the result boundary converts them back.
    dictionary: Option<Arc<grafeo_core::graph::rdf::TermDictionary>>,
    /// Column names that carry dictionary-encoded Int64 term IDs.
    /// Populated by `plan_triple_scan()`, consumed by `plan()` for resolution.
    encoded_columns: std::cell::RefCell<std::collections::HashSet<String>>,
    /// Per-planner native Ring policy. Production defaults to enabled; unit
    /// differential tests disable it without process-global state.
    #[cfg(feature = "ring-index")]
    native_ring_enabled: bool,
    /// Caller-proven output cap visible only while planning a row-preserving
    /// LIMIT-to-native-Ring path.
    #[cfg(feature = "ring-index")]
    native_ring_output_cap: std::cell::Cell<Option<usize>>,
}

impl RdfPlanner {
    /// Creates a new RDF planner with the given store.
    #[must_use]
    pub fn new(store: Arc<RdfStore>) -> Self {
        Self {
            store,
            chunk_size: DEFAULT_CHUNK_SIZE,
            transaction_id: None,
            valid_time: None,
            profiling: std::cell::Cell::new(false),
            profile_entries: std::cell::RefCell::new(Vec::new()),
            needs_companion_columns: std::cell::Cell::new(false),
            needs_exact_term_columns: std::cell::Cell::new(false),
            needs_identity_key_columns: std::cell::Cell::new(false),
            dictionary: None,
            encoded_columns: std::cell::RefCell::new(std::collections::HashSet::new()),
            #[cfg(feature = "ring-index")]
            native_ring_enabled: true,
            #[cfg(feature = "ring-index")]
            native_ring_output_cap: std::cell::Cell::new(None),
            #[cfg(feature = "wal")]
            wal: None,
            #[cfg(feature = "cdc")]
            cdc_log: None,
        }
    }

    /// Sets the chunk size for vectorized execution.
    #[must_use]
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    #[cfg(all(test, feature = "ring-index"))]
    fn with_native_ring_enabled(mut self, enabled: bool) -> Self {
        self.native_ring_enabled = enabled;
        self
    }

    /// Sets the transaction ID for transactional operations.
    #[must_use]
    pub fn with_transaction_id(mut self, transaction_id: Option<TransactionId>) -> Self {
        self.transaction_id = transaction_id;
        self
    }

    /// Sets the application valid-time inherited by every SPARQL insert in the
    /// planned statement.
    #[must_use]
    pub fn with_valid_time(mut self, valid_time: Option<ValidTimeInterval>) -> Self {
        self.valid_time = valid_time;
        self
    }

    /// Sets the WAL for logging RDF mutations.
    #[cfg(feature = "wal")]
    #[must_use]
    pub fn with_wal(mut self, wal: Option<Arc<grafeo_storage::wal::LpgWal>>) -> Self {
        self.wal = wal.map(|inner| Arc::new(RdfWal::new(inner, None)));
        self
    }

    /// Sets the WAL and shared database poison flag at a Session boundary.
    #[cfg(all(feature = "wal", any(feature = "sparql", feature = "graphql")))]
    #[must_use]
    pub(crate) fn with_wal_poison(
        mut self,
        wal: Option<Arc<grafeo_storage::wal::LpgWal>>,
        durability_poisoned: Arc<std::sync::atomic::AtomicBool>,
    ) -> Self {
        self.wal =
            wal.map(|inner| Arc::new(RdfWal::new(inner, Some(Arc::clone(&durability_poisoned)))));
        self
    }

    /// Uses the session-owned transaction accumulator for RDF CDC staging.
    #[cfg(feature = "cdc")]
    #[must_use]
    pub(crate) fn with_cdc_accumulator(
        mut self,
        cdc_log: Option<Arc<crate::cdc::CdcLog>>,
        pending_events: Option<Arc<crate::cdc::TransactionChangeAccumulator>>,
    ) -> Self {
        self.cdc_log =
            cdc_log
                .zip(self.transaction_id.and(pending_events))
                .map(|(log, pending_events)| {
                    Arc::new(RdfCdcSink {
                        log,
                        pending_events,
                    })
                });
        self
    }

    /// Plans a logical plan into a physical operator tree.
    ///
    /// # Errors
    ///
    /// Returns an error if planning fails.
    pub fn plan(&self, logical_plan: &LogicalPlan) -> Result<PhysicalPlan> {
        validate_rdf_exists_placement(&logical_plan.root)?;
        validate_rdf_repeated_scan_variables(&logical_plan.root)?;
        // Pre-analyze: only emit companion columns if the query uses LANG/DATATYPE
        self.needs_companion_columns
            .set(uses_lang_or_datatype(&logical_plan.root));
        self.needs_exact_term_columns
            .set(needs_exact_rdf_term_columns(&logical_plan.root));
        self.needs_identity_key_columns
            .set(needs_identity_rdf_term_columns(&logical_plan.root));

        let (mut operator, columns, _types) = self.plan_operator(&logical_plan.root)?;

        // Resolve dictionary-encoded columns back to strings at the result boundary.
        if let Some(ref dict) = self.dictionary {
            let encoded = self.encoded_columns.borrow();
            if !encoded.is_empty() {
                let encoded_indices: Vec<usize> = columns
                    .iter()
                    .enumerate()
                    .filter(|(_, name)| encoded.contains(*name))
                    .map(|(i, _)| i)
                    .collect();
                if !encoded_indices.is_empty() {
                    operator = Box::new(DictResolveOperator::new(
                        operator,
                        Arc::clone(dict),
                        encoded_indices,
                    ));
                }
            }
        }

        // Strip internal companion columns (__lang_<var>, __datatype_<var>)
        // from the output. They are used by LANG()/LANGMATCHES()/DATATYPE()
        // during evaluation but should never appear in query results.
        let (operator, columns) = strip_internal_columns(operator, columns);
        Ok(PhysicalPlan {
            operator,
            columns,
            adaptive_context: None,
        })
    }

    /// Plans a logical plan with profiling: each physical operator is wrapped
    /// in [`ProfiledOperator`](grafeo_core::execution::ProfiledOperator) to
    /// collect row counts and timing.
    ///
    /// # Errors
    ///
    /// Returns an error if the logical plan contains unsupported SPARQL operators
    /// or invalid expressions.
    pub fn plan_profiled(
        &self,
        logical_plan: &LogicalPlan,
    ) -> Result<(PhysicalPlan, Vec<crate::query::profile::ProfileEntry>)> {
        validate_rdf_exists_placement(&logical_plan.root)?;
        validate_rdf_repeated_scan_variables(&logical_plan.root)?;
        self.profiling.set(true);
        self.profile_entries.borrow_mut().clear();
        self.needs_companion_columns
            .set(uses_lang_or_datatype(&logical_plan.root));
        self.needs_exact_term_columns
            .set(needs_exact_rdf_term_columns(&logical_plan.root));
        self.needs_identity_key_columns
            .set(needs_identity_rdf_term_columns(&logical_plan.root));

        let result = self.plan_operator(&logical_plan.root);

        self.profiling.set(false);
        let (operator, columns, _types) = result?;
        let (operator, columns) = strip_internal_columns(operator, columns);
        let entries = self.profile_entries.borrow_mut().drain(..).collect();

        Ok((
            PhysicalPlan {
                operator,
                columns,
                adaptive_context: None,
            },
            entries,
        ))
    }

    /// If profiling is enabled, wraps a planned result in `ProfiledOperator`
    /// and records a [`ProfileEntry`](crate::query::profile::ProfileEntry).
    fn maybe_profile(
        &self,
        result: Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)>,
        op: &LogicalOperator,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        if self.profiling.get() {
            let (physical, columns, types) = result?;
            let (entry, stats) =
                crate::query::profile::ProfileEntry::new(physical.name(), op.display_label());
            let profiled = grafeo_core::execution::ProfiledOperator::new(physical, stats);
            self.profile_entries.borrow_mut().push(entry);
            Ok((Box::new(profiled), columns, types))
        } else {
            result
        }
    }

    /// Plans a single logical operator.
    fn plan_operator(
        &self,
        op: &LogicalOperator,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let result = match op {
            LogicalOperator::TripleScan(scan) => self.plan_triple_scan(scan),
            LogicalOperator::PropertyPath(path) => self.plan_property_path(path),
            LogicalOperator::Filter(filter) => self.plan_filter(filter),
            LogicalOperator::Project(project) => self.plan_project(project),
            LogicalOperator::Limit(limit) => self.plan_limit(limit),
            LogicalOperator::Skip(skip) => self.plan_skip(skip),
            LogicalOperator::Sort(sort) => self.plan_sort(sort),
            LogicalOperator::Aggregate(agg) => self.plan_aggregate(agg),
            LogicalOperator::Return(ret) => self.plan_return(ret),
            LogicalOperator::Join(join) => self.plan_join(join),
            LogicalOperator::LeftJoin(join) => self.plan_left_join(join),
            LogicalOperator::AntiJoin(join) => self.plan_anti_join(join),
            LogicalOperator::Union(union) => self.plan_union(union),
            LogicalOperator::Distinct(distinct) => self.plan_distinct(distinct),
            LogicalOperator::InsertTriple(insert) => self.plan_insert_triple(insert),
            LogicalOperator::DeleteTriple(delete) => self.plan_delete_triple(delete),
            LogicalOperator::Modify(modify) => self.plan_modify(modify),
            LogicalOperator::ClearGraph(clear) => self.plan_clear_graph(clear),
            LogicalOperator::CreateGraph(create) => self.plan_create_graph(create),
            LogicalOperator::DropGraph(drop_op) => self.plan_drop_graph(drop_op),
            LogicalOperator::CopyGraph(copy) => self.plan_copy_graph(copy),
            LogicalOperator::MoveGraph(move_op) => self.plan_move_graph(move_op),
            LogicalOperator::AddGraph(add) => self.plan_add_graph(add),
            LogicalOperator::LoadGraph(load) => self.plan_load_graph(load),
            LogicalOperator::Bind(bind) => self.plan_bind(bind),
            LogicalOperator::Construct(construct) => self.plan_construct(construct),
            LogicalOperator::MultiWayJoin(mwj) => self.plan_multi_way_join(mwj),
            LogicalOperator::Empty => {
                let op: Box<dyn Operator> = Box::new(SingleRowOperator::new());
                Ok((op, vec![], vec![]))
            }
            _ => Err(Error::Internal(format!(
                "Unsupported RDF operator: {:?}",
                std::mem::discriminant(op)
            ))),
        };
        self.maybe_profile(result, op)
    }

    /// Plans a triple scan operator.
    ///
    /// Creates a lazy scanning operator that reads triples in chunks
    /// for cache-efficient, vectorized processing.
    fn plan_triple_scan(
        &self,
        scan: &TripleScanOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        // Build the triple pattern for querying the store
        let pattern = self.build_triple_pattern(scan);

        // Determine which columns are variables (and thus in output)
        let mut columns = Vec::new();
        let mut output_mask = [false, false, false, false]; // s, p, o, g
        let emit_exact_term_columns = self.needs_exact_term_columns.get();
        let emit_identity_key_columns = self.needs_identity_key_columns.get();

        if let TripleComponent::Variable(name) = &scan.subject {
            columns.push(name.clone());
            if emit_exact_term_columns {
                columns.push(rdf_exact_term_column(name));
            }
            if emit_identity_key_columns {
                columns.push(rdf_identity_key_column(name));
            }
            output_mask[0] = true;
        }
        if let TripleComponent::Variable(name) = &scan.predicate {
            columns.push(name.clone());
            if emit_exact_term_columns {
                columns.push(rdf_exact_term_column(name));
            }
            if emit_identity_key_columns {
                columns.push(rdf_identity_key_column(name));
            }
            output_mask[1] = true;
        }
        // Track whether the object is a variable (for language-tag companion column)
        let mut object_var_name: Option<String> = None;
        if let TripleComponent::Variable(name) = &scan.object {
            columns.push(name.clone());
            if emit_exact_term_columns {
                columns.push(rdf_exact_term_column(name));
            }
            if emit_identity_key_columns {
                columns.push(rdf_identity_key_column(name));
            }
            output_mask[2] = true;
            object_var_name = Some(name.clone());
        }

        // When the object is a variable, add a hidden companion column for
        // language tags so that LANG() and LANGMATCHES() can access them.
        // This must be added BEFORE the graph column to match the DataChunk
        // layout (the lang column is emitted right after the object column).
        let emit_companion_columns = object_var_name.is_some();
        let emit_datatype_column = emit_companion_columns && self.needs_companion_columns.get();
        if let Some(ref obj_name) = object_var_name {
            columns.push(format!("__lang_{obj_name}"));
            if emit_datatype_column {
                columns.push(format!("__datatype_{obj_name}"));
            }
        }

        if let Some(TripleComponent::Variable(name)) = &scan.graph {
            columns.push(name.clone());
            if emit_exact_term_columns {
                columns.push(rdf_exact_term_column(name));
            }
            if emit_identity_key_columns {
                columns.push(rdf_identity_key_column(name));
            }
            output_mask[3] = true;
        }

        // Resolve graph context
        let (graph_iri, scan_all_graphs) = match &scan.graph {
            Some(TripleComponent::Iri(iri)) => (Some(iri.clone()), false),
            Some(TripleComponent::Literal(Value::String(iri))) => (Some(iri.to_string()), false),
            Some(TripleComponent::Variable(_)) => (None, true),
            _ => (None, false),
        };

        // Create the lazy scanning operator
        let scan_op = RdfTripleScanOperator::new(
            Arc::clone(&self.store),
            pattern,
            RdfTripleScanOutput {
                mask: output_mask,
                companion_columns: emit_companion_columns,
                datatype_column: emit_datatype_column,
                term_companions: RdfTermCompanionOutput {
                    lossless: emit_exact_term_columns,
                    identity: emit_identity_key_columns,
                },
            },
            self.chunk_size,
            GraphContext {
                graph: graph_iri,
                scan_all_graphs,
                dataset: scan.dataset.clone(),
            },
            self.transaction_id,
        );

        // Dictionary encoding is available but not yet automatically enabled for
        // all queries. The infrastructure (TermDictionary, DictResolveOperator) is
        // in place for use by the Ring Index planner (Phase 4) and WCOJ joins.
        // Automatic activation requires detecting whether downstream operators
        // (FILTER, BIND, ORDER BY) inspect string content, which is deferred.
        let _ = &self.dictionary; // suppress unused warning
        let _ = &self.encoded_columns;

        let types: Vec<LogicalType> = columns
            .iter()
            .map(|name| {
                if object_var_name.as_ref() == Some(name) {
                    LogicalType::Any
                } else {
                    LogicalType::String
                }
            })
            .collect();
        Ok((Box::new(scan_op), columns, types))
    }

    fn plan_property_path(
        &self,
        path: &PropertyPathOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let mut columns = Vec::new();
        let mut types = Vec::new();
        let emit_exact_term_columns = self.needs_exact_term_columns.get();
        let emit_identity_key_columns = self.needs_identity_key_columns.get();
        if let TripleComponent::Variable(name) = &path.subject {
            columns.push(name.clone());
            types.push(LogicalType::String);
            if emit_exact_term_columns {
                columns.push(rdf_exact_term_column(name));
                types.push(LogicalType::String);
            }
            if emit_identity_key_columns {
                columns.push(rdf_identity_key_column(name));
                types.push(LogicalType::String);
            }
        }
        if let TripleComponent::Variable(name) = &path.object {
            columns.push(name.clone());
            types.push(LogicalType::String);
            if emit_exact_term_columns {
                columns.push(rdf_exact_term_column(name));
                types.push(LogicalType::String);
            }
            if emit_identity_key_columns {
                columns.push(rdf_identity_key_column(name));
                types.push(LogicalType::String);
            }
        }
        let operator = Box::new(RdfPropertyPathOperator::new(
            Arc::clone(&self.store),
            path.clone(),
            columns.clone(),
            self.chunk_size,
            self.transaction_id,
            emit_exact_term_columns,
            emit_identity_key_columns,
        ));
        Ok((operator, columns, types))
    }

    /// Builds a TriplePattern from a TripleScanOp.
    fn build_triple_pattern(&self, scan: &TripleScanOp) -> TriplePattern {
        TriplePattern {
            subject: component_to_term(&scan.subject),
            predicate: component_to_term(&scan.predicate),
            object: component_to_term(&scan.object),
        }
    }

    /// Plans a RETURN clause.
    fn plan_return(
        &self,
        ret: &crate::query::plan::ReturnOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let (input_op, input_columns, input_types) = self.plan_operator(&ret.input)?;

        if self.needs_exact_term_columns.get()
            || self.needs_identity_key_columns.get()
            || input_columns
                .iter()
                .any(|column| column.starts_with(RDF_GROUP_KEY_COLUMN_PREFIX))
        {
            let renames = input_columns
                .iter()
                .filter(|name| {
                    !is_rdf_internal_term_column(name)
                        && !name.starts_with("__lang_")
                        && !name.starts_with("__datatype_")
                })
                .zip(&ret.items)
                .map(|(column, item)| {
                    (
                        column.clone(),
                        output_column_name(item.alias.as_deref(), &item.expression),
                    )
                })
                .collect::<Vec<_>>();
            let mut columns = input_columns;
            for column in &mut columns {
                for (source, output) in &renames {
                    if column == source {
                        column.clone_from(output);
                        break;
                    }
                    if *column == rdf_exact_term_column(source) {
                        *column = rdf_exact_term_column(output);
                        break;
                    }
                    if *column == rdf_identity_key_column(source) {
                        *column = rdf_identity_key_column(output);
                        break;
                    }
                    if *column == rdf_group_key_column(source) {
                        *column = rdf_group_key_column(output);
                        break;
                    }
                }
            }
            return Ok((input_op, columns, input_types));
        }

        // Extract output column names
        let columns: Vec<String> = ret
            .items
            .iter()
            .map(|item| output_column_name(item.alias.as_deref(), &item.expression))
            .collect();

        Ok((input_op, columns, input_types))
    }

    /// Plans a filter operator.
    ///
    /// Handles EXISTS/NOT EXISTS patterns by transforming them into semi-joins/anti-joins.
    fn plan_filter(
        &self,
        filter: &FilterOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        // Check for EXISTS/NOT EXISTS patterns and transform to semi/anti joins
        if let Some((subquery, is_negated)) = self.extract_exists_pattern(&filter.predicate) {
            return self.plan_exists_as_join(&filter.input, subquery, is_negated);
        }

        let (input_op, columns, types) = self.plan_operator(&filter.input)?;

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Convert logical expression to filter expression
        let filter_expr = convert_filter_expression(&filter.predicate)?;

        // Create RDF-specific predicate (doesn't need LpgStore)
        let predicate = RdfExpressionPredicate::new(filter_expr, variable_columns);

        let operator = Box::new(FilterOperator::new(input_op, Box::new(predicate)));
        Ok((operator, columns, types))
    }

    /// Extracts an EXISTS or NOT EXISTS pattern from a filter predicate.
    /// Returns the subquery operator and whether it's negated (NOT EXISTS).
    fn extract_exists_pattern<'a>(
        &self,
        predicate: &'a LogicalExpression,
    ) -> Option<(&'a LogicalOperator, bool)> {
        use crate::query::plan::UnaryOp;

        match predicate {
            // EXISTS { pattern }
            LogicalExpression::ExistsSubquery(subquery) => Some((subquery.as_ref(), false)),
            // NOT EXISTS { pattern }
            LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand,
            } => {
                if let LogicalExpression::ExistsSubquery(subquery) = operand.as_ref() {
                    Some((subquery.as_ref(), true))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Plans an EXISTS/NOT EXISTS pattern as a semi-join or anti-join.
    fn plan_exists_as_join(
        &self,
        input: &LogicalOperator,
        subquery: &LogicalOperator,
        is_negated: bool,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;

        let (left_op, left_columns, left_types) = self.plan_operator(input)?;
        let (right_op, right_columns, _right_types) = self.plan_operator(subquery)?;

        // Use Anti for NOT EXISTS, Semi for EXISTS
        let (result_op, result_columns) = if is_negated {
            common::build_anti_join(
                left_op,
                right_op,
                left_columns,
                &right_columns,
                left_types.clone(),
            )
        } else {
            common::build_semi_join(
                left_op,
                right_op,
                left_columns,
                &right_columns,
                left_types.clone(),
            )
        };

        Ok((result_op, result_columns, left_types))
    }

    /// Plans a DISTINCT operator.
    fn plan_distinct(
        &self,
        distinct: &DistinctOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;
        let (mut input_op, mut columns, mut types) = self.plan_operator(&distinct.input)?;
        let target_columns = distinct.columns.clone().unwrap_or_else(|| {
            columns
                .iter()
                .filter(|column| !is_rdf_internal_physical_column(column))
                .cloned()
                .collect()
        });
        let variable_columns = columns
            .iter()
            .enumerate()
            .map(|(index, column)| (column.clone(), index))
            .collect::<HashMap<_, _>>();
        let mut distinct_keys = Vec::with_capacity(target_columns.len());
        let mut key_projections = Vec::new();
        let mut key_replacements = HashMap::new();

        for column in target_columns {
            let group_key = rdf_group_key_column(&column);
            let identity = rdf_identity_key_column(&column);
            let expression = LogicalExpression::FunctionCall {
                name: RDF_IDENTITY_OR_NATIVE_KEY.to_string(),
                args: vec![
                    LogicalExpression::Variable(column),
                    variable_columns.get(&group_key).map_or_else(
                        || LogicalExpression::Literal(Value::Null),
                        |_| LogicalExpression::Variable(group_key.clone()),
                    ),
                    variable_columns.get(&identity).map_or_else(
                        || LogicalExpression::Literal(Value::Null),
                        |_| LogicalExpression::Variable(identity),
                    ),
                ],
                distinct: false,
            };
            let expression = convert_filter_expression(&expression)?;
            if let Some(&group_key_index) = variable_columns.get(&group_key) {
                key_replacements.insert(group_key_index, expression);
            } else {
                key_projections.push((expression, group_key.clone()));
            }
            distinct_keys.push(group_key);
        }

        if !key_projections.is_empty() || !key_replacements.is_empty() {
            let mut projections = (0..columns.len())
                .map(|index| {
                    key_replacements
                        .remove(&index)
                        .map_or(RdfProjectExpr::Column(index), |expr| {
                            RdfProjectExpr::Expression {
                                expr,
                                variable_columns: variable_columns.clone(),
                            }
                        })
                })
                .collect::<Vec<_>>();
            for (expression, column) in key_projections {
                projections.push(RdfProjectExpr::Expression {
                    expr: expression,
                    variable_columns: variable_columns.clone(),
                });
                columns.push(column);
                types.push(LogicalType::Any);
            }
            input_op = Box::new(RdfProjectOperator::new(
                input_op,
                projections,
                types.clone(),
            ));
        }
        let (op, cols) = common::build_distinct(
            input_op,
            columns,
            (!distinct_keys.is_empty()).then_some(distinct_keys.as_slice()),
            types.clone(),
        );
        Ok((op, cols, types))
    }

    /// Plans a LIMIT operator.
    fn plan_limit(
        &self,
        limit: &LimitOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;
        #[cfg(feature = "ring-index")]
        let previous_ring_cap = self
            .native_ring_output_cap
            .replace(rdf_native_ring_limit_passthrough(&limit.input).then(|| limit.count.value()));
        let planned_input = self.plan_operator(&limit.input);
        #[cfg(feature = "ring-index")]
        self.native_ring_output_cap.set(previous_ring_cap);
        let (input_op, columns, types) = planned_input?;
        let (op, cols) = common::build_limit(input_op, columns, limit.count.value(), types.clone());
        Ok((op, cols, types))
    }

    /// Plans a SKIP operator.
    fn plan_skip(
        &self,
        skip: &SkipOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;
        let (input_op, columns, types) = self.plan_operator(&skip.input)?;
        let (op, cols) = common::build_skip(input_op, columns, skip.count.value(), types.clone());
        Ok((op, cols, types))
    }

    /// Plans a SORT operator.
    fn plan_sort(
        &self,
        sort: &SortOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::plan::SortOrder;
        use grafeo_core::execution::operators::{
            FilterExpression, NullOrder, SortDirection, SortKey,
        };

        let (mut input_op, columns, types) = self.plan_operator(&sort.input)?;

        let mut variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Pre-project complex sort key expressions
        let mut expression_projections: Vec<(FilterExpression, String)> = Vec::new();
        let mut next_col_idx = columns.len();
        for key in &sort.keys {
            match &key.expression {
                LogicalExpression::Variable(_) => {}
                _ => {
                    let col_name = resolved_column_name(&key.expression);
                    if !variable_columns.contains_key(&col_name) {
                        let filter_expr = convert_filter_expression(&key.expression)?;
                        expression_projections.push((filter_expr, col_name.clone()));
                        variable_columns.insert(col_name, next_col_idx);
                        next_col_idx += 1;
                    }
                }
            }
        }

        if !expression_projections.is_empty() {
            let mut projections: Vec<RdfProjectExpr> =
                (0..columns.len()).map(RdfProjectExpr::Column).collect();
            let mut output_types: Vec<LogicalType> = types.clone();

            for (filter_expr, _col_name) in &expression_projections {
                projections.push(RdfProjectExpr::Expression {
                    expr: filter_expr.clone(),
                    variable_columns: variable_columns.clone(),
                });
                output_types.push(LogicalType::Any); // computed expressions may produce non-string types
            }

            input_op = Box::new(RdfProjectOperator::new(
                input_op,
                projections,
                output_types.clone(),
            ));
        }

        let physical_keys: Vec<SortKey> = sort
            .keys
            .iter()
            .map(|key| {
                let col_idx = resolve_expression(&key.expression, &variable_columns)?;
                Ok(SortKey {
                    column: col_idx,
                    direction: match key.order {
                        SortOrder::Ascending => SortDirection::Ascending,
                        SortOrder::Descending => SortDirection::Descending,
                    },
                    // SPARQL fixes unbound/error results at the low end. DESC
                    // reverses this comparison and therefore places them last.
                    null_order: NullOrder::NullsFirst,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let operator = Box::new(RdfSortOperator::new(input_op, physical_keys, types.clone()));
        Ok((operator, columns, types))
    }

    /// Plans a PROJECT operator.
    ///
    /// Projects only the requested columns from the input.
    fn plan_project(
        &self,
        project: &crate::query::plan::ProjectOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let (input_op, input_columns, input_types) = self.plan_operator(&project.input)?;

        // Build mapping from variable name to column index
        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        let mut projections = Vec::new();
        let mut output_columns = Vec::new();
        let mut output_types = Vec::new();
        let explicitly_projected_outputs = project
            .projections
            .iter()
            .filter_map(|projection| {
                projection.alias.clone().or_else(|| {
                    if let LogicalExpression::Variable(name) = &projection.expression {
                        Some(name.clone())
                    } else {
                        None
                    }
                })
            })
            .collect::<HashSet<_>>();

        for proj in &project.projections {
            match &proj.expression {
                LogicalExpression::Variable(name) => {
                    if let Some(&col_idx) = variable_columns.get(name) {
                        let output_name = proj.alias.clone().unwrap_or_else(|| name.clone());
                        projections.push(RdfProjectExpr::Column(col_idx));
                        output_columns.push(output_name.clone());
                        output_types.push(input_types[col_idx].clone());
                        if self.needs_exact_term_columns.get()
                            && !explicitly_projected_outputs
                                .contains(&rdf_exact_term_column(&output_name))
                            && let Some(&exact_idx) =
                                variable_columns.get(&rdf_exact_term_column(name))
                        {
                            projections.push(RdfProjectExpr::Column(exact_idx));
                            output_columns.push(rdf_exact_term_column(&output_name));
                            output_types.push(input_types[exact_idx].clone());
                        }
                        if self.needs_identity_key_columns.get()
                            && !explicitly_projected_outputs
                                .contains(&rdf_identity_key_column(&output_name))
                            && let Some(&identity_idx) =
                                variable_columns.get(&rdf_identity_key_column(name))
                        {
                            projections.push(RdfProjectExpr::Column(identity_idx));
                            output_columns.push(rdf_identity_key_column(&output_name));
                            output_types.push(input_types[identity_idx].clone());
                        }
                        if !explicitly_projected_outputs
                            .contains(&rdf_group_key_column(&output_name))
                            && let Some(&group_key_idx) =
                                variable_columns.get(&rdf_group_key_column(name))
                        {
                            projections.push(RdfProjectExpr::Column(group_key_idx));
                            output_columns.push(rdf_group_key_column(&output_name));
                            output_types.push(input_types[group_key_idx].clone());
                        }
                    } else if (name.starts_with(RDF_EXACT_TERM_COLUMN_PREFIX)
                        && !self.needs_exact_term_columns.get())
                        || (name.starts_with(RDF_IDENTITY_KEY_COLUMN_PREFIX)
                            && !self.needs_identity_key_columns.get())
                    {
                        // The logical translator owns both RDF companion
                        // projections. Physical planning prunes the one whose
                        // independently analysed demand is absent rather than
                        // forcing lossless reconstruction state into an
                        // identity-only relational query (or vice versa).
                        continue;
                    } else {
                        return Err(Error::Internal(format!(
                            "Variable '{}' not found in input columns",
                            name
                        )));
                    }
                }
                LogicalExpression::Literal(value) => {
                    projections.push(RdfProjectExpr::Constant(value.clone()));
                    output_columns.push(proj.alias.clone().unwrap_or_else(|| format!("{value}")));
                    output_types.push(LogicalType::Any);
                }
                expr => {
                    // Convert complex expressions (function calls, arithmetic, etc.)
                    // to physical filter expressions and evaluate them in the projection.
                    let filter_expr = convert_filter_expression(expr)?;
                    projections.push(RdfProjectExpr::Expression {
                        expr: filter_expr,
                        variable_columns: variable_columns.clone(),
                    });
                    output_columns.push(proj.alias.clone().unwrap_or_else(|| format!("{expr:?}")));
                    output_types.push(LogicalType::Any);
                }
            }
        }

        // Pass-through projects add bindings without replacing the input.
        // A replacing project with zero outputs is a real zero-column lexical
        // boundary and must retain row cardinality without leaking columns.
        if projections.is_empty() && project.pass_through_input {
            return Ok((input_op, input_columns, input_types));
        }

        // Use RdfProjectOperator which delegates expression evaluation to
        // RdfExpressionPredicate, giving access to SPARQL functions (STRLEN,
        // UCASE, LCASE, etc.) that the generic ProjectOperator does not know.
        let operator: Box<dyn Operator> = Box::new(RdfProjectOperator::new(
            input_op,
            projections,
            output_types.clone(),
        ));
        Ok((operator, output_columns, output_types))
    }

    /// Plans a BIND operator.
    ///
    /// BIND adds a computed column to each row by evaluating an expression.
    /// For example: `BIND (CONCAT(?name, " (age ", STR(?age), ")") AS ?label)`
    fn plan_bind(
        &self,
        bind: &BindOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let (input_op, input_columns, mut input_types) = self.plan_operator(&bind.input)?;

        let copied_exact_column = if self.needs_exact_term_columns.get()
            && let LogicalExpression::Variable(source) = &bind.expression
        {
            input_columns
                .iter()
                .position(|column| column == &rdf_exact_term_column(source))
        } else {
            None
        };
        let copied_identity_column = if self.needs_identity_key_columns.get()
            && let LogicalExpression::Variable(source) = &bind.expression
        {
            input_columns
                .iter()
                .position(|column| column == &rdf_identity_key_column(source))
        } else {
            None
        };
        let copied_group_key_column = if let LogicalExpression::Variable(source) = &bind.expression
        {
            input_columns
                .iter()
                .position(|column| column == &rdf_group_key_column(source))
        } else {
            None
        };

        if let LogicalExpression::FunctionCall { name, args, .. } = &bind.expression
            && name == RDF_TAG_BOUND_TERM
            && let Some(LogicalExpression::Variable(visible_source)) = args.first()
            && let Some(LogicalExpression::Variable(exact_source)) = args.get(1)
            && input_columns.iter().any(|column| column == visible_source)
            && !input_columns.iter().any(|column| column == exact_source)
        {
            return Err(Error::Query(
                grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Semantic,
                    "RDF identity-preserving binding has no lossless source term identity",
                )
                .with_hint(
                    "Use an RDF-producing expression with lossless term identity before this binding"
                        .to_string(),
                ),
            ));
        }

        // Build variable-to-column mapping for expression evaluation
        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Convert the BIND expression to a FilterExpression
        let filter_expr = convert_filter_expression(&bind.expression)?;

        // Build output columns: all input columns + the new BIND variable
        let mut output_columns = input_columns;
        output_columns.push(bind.variable.clone());
        input_types.push(LogicalType::Any);

        let mut operator: Box<dyn Operator> = Box::new(RdfBindOperator::new(
            input_op,
            filter_expr,
            variable_columns,
        ));
        if copied_exact_column.is_some()
            || copied_identity_column.is_some()
            || copied_group_key_column.is_some()
        {
            let mut projections = (0..output_columns.len())
                .map(RdfProjectExpr::Column)
                .collect::<Vec<_>>();
            if let Some(exact_column) = copied_exact_column {
                projections.push(RdfProjectExpr::Column(exact_column));
                output_columns.push(rdf_exact_term_column(&bind.variable));
                input_types.push(input_types[exact_column].clone());
            }
            if let Some(identity_column) = copied_identity_column {
                projections.push(RdfProjectExpr::Column(identity_column));
                output_columns.push(rdf_identity_key_column(&bind.variable));
                input_types.push(input_types[identity_column].clone());
            }
            if let Some(group_key_column) = copied_group_key_column {
                projections.push(RdfProjectExpr::Column(group_key_column));
                output_columns.push(rdf_group_key_column(&bind.variable));
                input_types.push(input_types[group_key_column].clone());
            }
            operator = Box::new(RdfProjectOperator::new(
                operator,
                projections,
                input_types.clone(),
            ));
        }
        Ok((operator, output_columns, input_types))
    }

    /// Plans a CONSTRUCT operator.
    ///
    /// Evaluates the WHERE clause, then for each row substitutes variable
    /// bindings into the template to produce (subject, predicate, object) rows.
    fn plan_construct(
        &self,
        construct: &ConstructOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let (input_op, input_columns, _input_types) = self.plan_operator(&construct.input)?;

        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        let operator = Box::new(ConstructOperator::new(
            input_op,
            construct.templates.clone(),
            variable_columns,
        ));

        let columns = vec![
            "subject".to_string(),
            "predicate".to_string(),
            "object".to_string(),
        ];
        let types = vec![LogicalType::String; 3];
        Ok((operator, columns, types))
    }

    /// Plans an AGGREGATE operator.
    fn plan_aggregate(
        &self,
        agg: &AggregateOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        // COUNT(*) fast-path: when the aggregate is a single COUNT(*) with no
        // GROUP BY, no DISTINCT, no HAVING, and the input is a fully-unbound
        // TripleScan, short-circuit to store.len() in O(1).
        if let Some(result) = self.try_count_fast_path(agg) {
            return Ok(result);
        }

        use grafeo_core::execution::operators::AggregateExpr as PhysicalAggregateExpr;

        let (mut input_op, input_columns, input_types) = self.plan_operator(&agg.input)?;
        let mut current_types = input_types;

        let mut variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Pre-project complex expressions in group-by keys and aggregate arguments
        let mut expression_projections: Vec<(FilterExpression, String)> = Vec::new();
        let mut replacement_projections: HashMap<usize, FilterExpression> = HashMap::new();
        let mut next_col_idx = input_columns.len();

        // Group-by expressions (Labels, Type, FunctionCall, etc.)
        for expr in &agg.group_by {
            match expr {
                LogicalExpression::Variable(_) => {}
                _ => {
                    let col_name = resolved_column_name(expr);
                    if !variable_columns.contains_key(&col_name) {
                        let filter_expr = convert_filter_expression(expr)?;
                        expression_projections.push((filter_expr, col_name.clone()));
                        variable_columns.insert(col_name, next_col_idx);
                        next_col_idx += 1;
                    }
                }
            }
        }

        // Every variable group key gets one rowwise-normalized compositional
        // RDF-or-native discriminator. A sparse helper (for example after
        // UNION padding) is preserved only on rows where it is non-NULL;
        // otherwise canonical RDF identity or native typed identity supplies
        // the key. Expression group keys may already have a helper from the
        // logical translator, but still require this rowwise normalization.
        for expr in &agg.group_by {
            let LogicalExpression::Variable(variable) = expr else {
                continue;
            };
            let group_key = rdf_group_key_column(variable);
            let identity = rdf_identity_key_column(variable);
            let filter_expr = convert_filter_expression(&LogicalExpression::FunctionCall {
                name: RDF_IDENTITY_OR_NATIVE_KEY.to_string(),
                args: vec![
                    LogicalExpression::Variable(variable.clone()),
                    variable_columns.get(&group_key).map_or_else(
                        || LogicalExpression::Literal(Value::Null),
                        |_| LogicalExpression::Variable(group_key.clone()),
                    ),
                    variable_columns.get(&identity).map_or_else(
                        || LogicalExpression::Literal(Value::Null),
                        |_| LogicalExpression::Variable(identity),
                    ),
                ],
                distinct: false,
            })?;
            if let Some(&group_key_index) = variable_columns.get(&group_key) {
                replacement_projections.insert(group_key_index, filter_expr);
            } else {
                expression_projections.push((filter_expr, group_key.clone()));
                variable_columns.insert(group_key, next_col_idx);
                next_col_idx += 1;
            }
        }

        // Aggregate argument expressions
        for agg_expr in &agg.aggregates {
            for expr_opt in [
                &agg_expr.expression,
                &agg_expr.expression2,
                &agg_expr.distinct_key,
            ] {
                let Some(expr) = expr_opt else { continue };
                match expr {
                    LogicalExpression::Variable(_) => {}
                    _ => {
                        let col_name = resolved_column_name(expr);
                        if !variable_columns.contains_key(&col_name) {
                            let filter_expr = convert_filter_expression(expr)?;
                            expression_projections.push((filter_expr, col_name.clone()));
                            variable_columns.insert(col_name, next_col_idx);
                            next_col_idx += 1;
                        }
                    }
                }
            }
        }

        if !expression_projections.is_empty() || !replacement_projections.is_empty() {
            let mut projections: Vec<RdfProjectExpr> = (0..input_columns.len())
                .map(|index| {
                    replacement_projections.remove(&index).map_or(
                        RdfProjectExpr::Column(index),
                        |expr| RdfProjectExpr::Expression {
                            expr,
                            variable_columns: variable_columns.clone(),
                        },
                    )
                })
                .collect();
            let mut output_types = current_types.clone();

            for (filter_expr, _col_name) in &expression_projections {
                projections.push(RdfProjectExpr::Expression {
                    expr: filter_expr.clone(),
                    variable_columns: variable_columns.clone(),
                });
                output_types.push(LogicalType::Any); // computed expressions may produce non-string types
            }

            input_op = Box::new(RdfProjectOperator::new(
                input_op,
                projections,
                output_types.clone(),
            ));
            current_types = output_types;
        }

        enum GroupOutput {
            Ordinary {
                name: String,
                group_result: usize,
            },
            CanonicalRdf {
                name: String,
                group_result: usize,
                visible_source: usize,
                exact_source: Option<usize>,
                visible_result: usize,
                exact_result: Option<usize>,
            },
            DiscriminatedRdfOrNative {
                name: String,
                group_result: usize,
                visible_source: usize,
                exact_source: Option<usize>,
                identity_source: Option<usize>,
                visible_result: usize,
                exact_result: Option<usize>,
                identity_result: Option<usize>,
            },
        }

        let mut group_columns = Vec::new();
        let mut group_outputs = Vec::new();
        for expression in &agg.group_by {
            let name = expression_to_string(expression);
            if let LogicalExpression::Variable(variable) = expression
                && let Some(&group_key_source) =
                    variable_columns.get(&rdf_group_key_column(variable))
            {
                let group_result = group_columns.len();
                group_columns.push(group_key_source);
                group_outputs.push(GroupOutput::DiscriminatedRdfOrNative {
                    name,
                    group_result,
                    visible_source: resolve_expression(expression, &variable_columns)?,
                    exact_source: self
                        .needs_exact_term_columns
                        .get()
                        .then(|| {
                            variable_columns
                                .get(&rdf_exact_term_column(variable))
                                .copied()
                        })
                        .flatten(),
                    identity_source: self
                        .needs_identity_key_columns
                        .get()
                        .then(|| {
                            variable_columns
                                .get(&rdf_identity_key_column(variable))
                                .copied()
                        })
                        .flatten(),
                    visible_result: usize::MAX,
                    exact_result: None,
                    identity_result: None,
                });
            } else if let LogicalExpression::Variable(variable) = expression
                && let Some(&identity_source) =
                    variable_columns.get(&rdf_identity_key_column(variable))
            {
                let group_result = group_columns.len();
                group_columns.push(identity_source);
                group_outputs.push(GroupOutput::CanonicalRdf {
                    name,
                    group_result,
                    visible_source: resolve_expression(expression, &variable_columns)?,
                    exact_source: self
                        .needs_exact_term_columns
                        .get()
                        .then(|| {
                            variable_columns
                                .get(&rdf_exact_term_column(variable))
                                .copied()
                        })
                        .flatten(),
                    visible_result: usize::MAX,
                    exact_result: None,
                });
            } else {
                let group_result = group_columns.len();
                group_columns.push(resolve_expression(expression, &variable_columns)?);
                group_outputs.push(GroupOutput::Ordinary { name, group_result });
            }
        }

        let mut physical_aggregates: Vec<PhysicalAggregateExpr> = agg
            .aggregates
            .iter()
            .map(|agg_expr| {
                let column = agg_expr
                    .expression
                    .as_ref()
                    .map(|e| resolve_expression(e, &variable_columns))
                    .transpose()?;

                let column2 = agg_expr
                    .expression2
                    .as_ref()
                    .map(|e| resolve_expression(e, &variable_columns))
                    .transpose()?;

                let distinct_key_column = agg_expr
                    .distinct_key
                    .as_ref()
                    .map(|e| resolve_expression(e, &variable_columns))
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

        let group_count = group_columns.len();
        let user_aggregate_count = physical_aggregates.len();
        let row_identity_columns = if physical_aggregates.iter().any(|aggregate| {
            aggregate.function == grafeo_core::execution::operators::AggregateFunction::Count
                && aggregate.distinct
                && aggregate.column.is_none()
        }) {
            Some(
                input_columns
                    .iter()
                    .filter(|name| !is_rdf_internal_physical_column(name))
                    .map(|name| {
                        let visible = variable_columns.get(name).copied().ok_or_else(|| {
                            Error::Internal(format!(
                                "COUNT(DISTINCT *) input column ?{name} disappeared"
                            ))
                        })?;
                        Ok(RdfRowIdentityColumn {
                            visible,
                            canonical_rdf: variable_columns
                                .get(&rdf_identity_key_column(name))
                                .copied(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        for output in &mut group_outputs {
            match output {
                GroupOutput::CanonicalRdf {
                    visible_source,
                    exact_source,
                    visible_result,
                    exact_result,
                    ..
                } => {
                    *visible_result = group_count + physical_aggregates.len();
                    physical_aggregates.push(PhysicalAggregateExpr::min(*visible_source));
                    if let Some(exact_source) = exact_source {
                        *exact_result = Some(group_count + physical_aggregates.len());
                        physical_aggregates.push(PhysicalAggregateExpr::min(*exact_source));
                    }
                }
                GroupOutput::DiscriminatedRdfOrNative {
                    visible_source,
                    exact_source,
                    identity_source,
                    visible_result,
                    exact_result,
                    identity_result,
                    ..
                } => {
                    *visible_result = group_count + physical_aggregates.len();
                    physical_aggregates.push(PhysicalAggregateExpr::min(*visible_source));
                    if let Some(exact_source) = exact_source {
                        *exact_result = Some(group_count + physical_aggregates.len());
                        physical_aggregates.push(PhysicalAggregateExpr::min(*exact_source));
                    }
                    if let Some(identity_source) = identity_source {
                        *identity_result = Some(group_count + physical_aggregates.len());
                        physical_aggregates.push(PhysicalAggregateExpr::min(*identity_source));
                    }
                }
                GroupOutput::Ordinary { .. } => {}
            }
        }

        // Preserve every group key's proven input type. RDF object and computed
        // expression columns are already `Any`, while IRI and canonical identity
        // columns remain `String`; widening all of them would discard useful
        // schema information without making execution safer.
        let mut aggregate_schema = group_columns
            .iter()
            .map(|&column| current_types[column].clone())
            .collect::<Vec<_>>();
        let mut user_aggregate_columns = Vec::new();
        let mut user_aggregate_types = Vec::new();
        for agg_expr in &agg.aggregates {
            // Only claim a concrete type when the SPARQL set function guarantees
            // it independently of its input term. Numeric promotion and selector
            // functions can produce several physical value kinds and must remain
            // `Any` so the vector does not coerce a valid result into a default.
            let result_type = match agg_expr.function {
                LogicalAggregateFunction::Count | LogicalAggregateFunction::CountNonNull => {
                    LogicalType::Int64
                }
                LogicalAggregateFunction::GroupConcat => LogicalType::String,
                _ => LogicalType::Any,
            };
            aggregate_schema.push(result_type.clone());
            user_aggregate_types.push(result_type);
            user_aggregate_columns.push(
                agg_expr
                    .alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}(...)", agg_expr.function).to_lowercase()),
            );
        }

        for output in &group_outputs {
            match output {
                GroupOutput::CanonicalRdf {
                    visible_source,
                    exact_result,
                    ..
                } => {
                    aggregate_schema.push(current_types[*visible_source].clone());
                    if exact_result.is_some() {
                        aggregate_schema.push(LogicalType::String);
                    }
                }
                GroupOutput::DiscriminatedRdfOrNative {
                    visible_source,
                    exact_result,
                    identity_result,
                    ..
                } => {
                    aggregate_schema.push(current_types[*visible_source].clone());
                    if exact_result.is_some() {
                        aggregate_schema.push(LogicalType::String);
                    }
                    if identity_result.is_some() {
                        aggregate_schema.push(LogicalType::String);
                    }
                }
                GroupOutput::Ordinary { .. } => {}
            }
        }

        let agg_schema = aggregate_schema.clone();
        let mut operator: Box<dyn Operator> = Box::new(RdfAggregateOperator::new(
            input_op,
            group_columns,
            physical_aggregates,
            agg_schema,
            row_identity_columns,
        ));

        let mut output_columns = Vec::new();
        let mut output_schema = Vec::new();
        if group_outputs.iter().any(|output| {
            matches!(
                output,
                GroupOutput::CanonicalRdf { .. } | GroupOutput::DiscriminatedRdfOrNative { .. }
            )
        }) {
            let mut projections = Vec::new();
            for output in &group_outputs {
                match output {
                    GroupOutput::Ordinary { name, group_result } => {
                        projections.push(RdfProjectExpr::Column(*group_result));
                        output_columns.push(name.clone());
                        output_schema.push(aggregate_schema[*group_result].clone());
                    }
                    GroupOutput::CanonicalRdf {
                        name,
                        group_result,
                        visible_result,
                        exact_result,
                        ..
                    } => {
                        projections.push(RdfProjectExpr::Column(*visible_result));
                        output_columns.push(name.clone());
                        output_schema.push(aggregate_schema[*visible_result].clone());
                        if let Some(exact_result) = exact_result {
                            projections.push(RdfProjectExpr::Column(*exact_result));
                            output_columns.push(rdf_exact_term_column(name));
                            output_schema.push(aggregate_schema[*exact_result].clone());
                        }
                        projections.push(RdfProjectExpr::Column(*group_result));
                        output_columns.push(rdf_identity_key_column(name));
                        output_schema.push(aggregate_schema[*group_result].clone());
                    }
                    GroupOutput::DiscriminatedRdfOrNative {
                        name,
                        group_result,
                        visible_result,
                        exact_result,
                        identity_result,
                        ..
                    } => {
                        projections.push(RdfProjectExpr::Column(*visible_result));
                        output_columns.push(name.clone());
                        output_schema.push(aggregate_schema[*visible_result].clone());
                        if let Some(exact_result) = exact_result {
                            projections.push(RdfProjectExpr::Column(*exact_result));
                            output_columns.push(rdf_exact_term_column(name));
                            output_schema.push(aggregate_schema[*exact_result].clone());
                        }
                        if let Some(identity_result) = identity_result {
                            projections.push(RdfProjectExpr::Column(*identity_result));
                            output_columns.push(rdf_identity_key_column(name));
                            output_schema.push(aggregate_schema[*identity_result].clone());
                        }
                        projections.push(RdfProjectExpr::Column(*group_result));
                        output_columns.push(rdf_group_key_column(name));
                        output_schema.push(aggregate_schema[*group_result].clone());
                    }
                }
            }
            for index in 0..user_aggregate_count {
                projections.push(RdfProjectExpr::Column(group_count + index));
                output_columns.push(user_aggregate_columns[index].clone());
                output_schema.push(user_aggregate_types[index].clone());
            }
            operator = Box::new(RdfProjectOperator::new(
                operator,
                projections,
                output_schema.clone(),
            ));
        } else {
            output_columns.extend(group_outputs.iter().map(|output| match output {
                GroupOutput::Ordinary { name, .. }
                | GroupOutput::CanonicalRdf { name, .. }
                | GroupOutput::DiscriminatedRdfOrNative { name, .. } => name.clone(),
            }));
            output_columns.extend(user_aggregate_columns);
            output_schema = aggregate_schema;
        }

        // Apply HAVING clause filter if present
        if let Some(having_expr) = &agg.having {
            let having_var_columns: HashMap<String, usize> = output_columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();

            let filter_expr = convert_filter_expression(having_expr)?;
            let predicate = RdfExpressionPredicate::new(filter_expr, having_var_columns);
            operator = Box::new(FilterOperator::new(operator, Box::new(predicate)));
        }

        Ok((operator, output_columns, output_schema))
    }

    /// Detects COUNT(*) over a TripleScan and returns the count directly when possible.
    ///
    /// Supported patterns (no GROUP BY, no DISTINCT, no HAVING):
    /// - `COUNT(*) WHERE { ?s ?p ?o }`: fully unbound, returns `store.len()`
    /// - `COUNT(*) WHERE { ?s <pred> ?o }`: predicate-bound, returns per-predicate count
    /// - `COUNT(*) WHERE { GRAPH <g> { ?s ?p ?o } }`: per-graph count
    fn try_count_fast_path(
        &self,
        agg: &AggregateOp,
    ) -> Option<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        // Must be: no GROUP BY, no HAVING, exactly one aggregate
        if !agg.group_by.is_empty() || agg.having.is_some() || agg.aggregates.len() != 1 {
            return None;
        }

        let agg_expr = &agg.aggregates[0];

        // Must be COUNT(*): Count function, no expression (not COUNT(?x)), no DISTINCT
        if agg_expr.function != LogicalAggregateFunction::Count
            || agg_expr.expression.is_some()
            || agg_expr.distinct
            || agg_expr.distinct_key.is_some()
        {
            return None;
        }

        // Input must be a simple TripleScan (no chained input)
        let scan = Self::find_simple_triple_scan(&agg.input)?;

        // No dataset restriction (FROM / FROM NAMED)
        if scan.dataset.is_some() {
            return None;
        }

        let count = self.count_for_scan(scan)?;

        let alias = agg_expr
            .alias
            .clone()
            .unwrap_or_else(|| "count(*)".to_string());

        Some(Self::make_constant_int64(count, alias))
    }

    /// Resolves the count for a simple triple scan pattern, or returns `None`
    /// if the pattern is too complex for a fast-path.
    fn count_for_scan(&self, scan: &TripleScanOp) -> Option<i64> {
        // Transactional counts must include pending inserts/deletes and may
        // address a detached named graph. Fall through to the normal scan.
        if self.transaction_id.is_some() {
            return None;
        }
        let s_var = scan.subject.as_variable().is_some();
        let p_var = scan.predicate.as_variable().is_some();
        let o_var = scan.object.as_variable().is_some();

        // Per-graph count: GRAPH <iri> { ?s ?p ?o }
        if let Some(graph) = &scan.graph {
            if s_var
                && p_var
                && o_var
                && let TripleComponent::Iri(graph_iri) = graph
            {
                let ng = self.store.graph(graph_iri)?;
                // reason: triple count will not exceed i64::MAX
                #[allow(clippy::cast_possible_wrap)]
                return Some(ng.len() as i64);
            }
            return None;
        }

        // Fully unbound: ?s ?p ?o
        if s_var && p_var && o_var {
            // reason: triple count will not exceed i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            return Some(self.store.len() as i64);
        }

        // Ring Index: use ring.count() for any partially-bound pattern (O(log sigma))
        #[cfg(feature = "ring-index")]
        if let Some(ring) = self.store.ring() {
            let pattern = self.build_triple_pattern(scan);
            // reason: triple count will not exceed i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            return Some(ring.count(&pattern) as i64);
        }

        // Predicate-bound: ?s <pred> ?o
        if s_var && !p_var && o_var {
            if let TripleComponent::Iri(pred_iri) = &scan.predicate {
                let stats = self.store.get_or_collect_statistics();
                if let Some(pred_stats) = stats.get_predicate(pred_iri) {
                    // reason: predicate triple count will not exceed i64::MAX
                    #[allow(clippy::cast_possible_wrap)]
                    return Some(pred_stats.triple_count as i64);
                }
            }
            return None;
        }

        // Not a fast-path pattern
        None
    }

    /// Creates a constant Int64 single-row result.
    fn make_constant_int64(
        value: i64,
        column_name: String,
    ) -> (Box<dyn Operator>, Vec<String>, Vec<LogicalType>) {
        let mut column = ValueVector::with_capacity(LogicalType::Int64, 1);
        column.push_value(Value::Int64(value));
        let chunk = DataChunk::new(vec![column]);
        let operator = Box::new(ConstantOperator::new(chunk));
        (operator, vec![column_name], vec![LogicalType::Int64])
    }

    /// Walks through the input operator to find a simple TripleScan (no chained input).
    ///
    /// Sees through Project operators.
    fn find_simple_triple_scan(op: &LogicalOperator) -> Option<&TripleScanOp> {
        match op {
            LogicalOperator::TripleScan(scan) if scan.input.is_none() => Some(scan),
            LogicalOperator::Project(proj) => Self::find_simple_triple_scan(&proj.input),
            _ => None,
        }
    }

    /// Plans a JOIN operator using HashJoin.
    ///
    /// For SPARQL, we join on shared variables (equi-join). When no shared
    /// variables exist, falls back to cross join.
    fn plan_join(
        &self,
        join: &crate::query::plan::JoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;
        let (left_op, left_columns, left_types) = self.plan_operator(&join.left)?;
        let (right_op, right_columns, right_types) = self.plan_operator(&join.right)?;
        validate_rdf_binary_join_metadata(&left_columns, &right_columns, &join.conditions, "Join")?;

        // Estimate cardinalities for build-side selection
        let cardinalities = estimate_operator_cardinality(&join.left, &self.store)
            .zip(estimate_operator_cardinality(&join.right, &self.store));

        if has_sparql_compatibility(&join.conditions) {
            let mode = if join.join_type == JoinType::Semi {
                RdfCompatibilityMode::Semi
            } else if matches!(join.join_type, JoinType::Inner | JoinType::Cross) {
                RdfCompatibilityMode::Inner
            } else {
                return Err(Error::Internal(format!(
                    "RDF compatibility JoinOp does not support {:?} physical semantics",
                    join.join_type
                )));
            };
            return build_rdf_compatibility_join(
                PlannedRdfRelation::new(left_op, left_columns, left_types),
                PlannedRdfRelation::new(right_op, right_columns, right_types),
                &join.conditions,
                mode,
            );
        }

        let explicit_keys = resolve_rdf_join_keys(&join.conditions, &left_columns, &right_columns)?;
        if join.join_type == JoinType::Semi {
            let (operator, columns) = if let Some((left_keys, right_keys)) = explicit_keys {
                common::build_semi_join_with_keys(
                    left_op,
                    right_op,
                    left_columns,
                    left_types.clone(),
                    left_keys,
                    right_keys,
                )
            } else {
                common::build_semi_join(
                    left_op,
                    right_op,
                    left_columns,
                    &right_columns,
                    left_types.clone(),
                )
            };
            return Ok((operator, columns, left_types));
        }
        if !matches!(join.join_type, JoinType::Inner | JoinType::Cross) {
            return Err(Error::Internal(format!(
                "RDF JoinOp does not support {:?} physical semantics",
                join.join_type
            )));
        }

        if let Some((left_keys, right_keys)) = explicit_keys {
            return Ok(common::build_inner_join_with_keys(
                left_op,
                right_op,
                &left_columns,
                &right_columns,
                &left_types,
                &right_types,
                left_keys,
                right_keys,
                cardinalities,
            ));
        }

        Ok(common::build_inner_join(
            left_op,
            right_op,
            &left_columns,
            &right_columns,
            &left_types,
            &right_types,
            cardinalities,
        ))
    }

    /// Plans a LEFT JOIN operator (for SPARQL OPTIONAL) using HashJoin.
    fn plan_left_join(
        &self,
        join: &LeftJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;
        let (left_op, left_columns, left_types) = self.plan_operator(&join.left)?;
        let (right_op, right_columns, right_types) = self.plan_operator(&join.right)?;
        validate_rdf_binary_join_metadata(
            &left_columns,
            &right_columns,
            &join.compatibility_conditions,
            "LeftJoin",
        )?;
        if has_sparql_compatibility(&join.compatibility_conditions) {
            return build_rdf_compatibility_join(
                PlannedRdfRelation::new(left_op, left_columns, left_types),
                PlannedRdfRelation::new(right_op, right_columns, right_types),
                &join.compatibility_conditions,
                RdfCompatibilityMode::Left,
            );
        }
        if let Some((left_keys, right_keys)) = resolve_rdf_join_keys(
            &join.compatibility_conditions,
            &left_columns,
            &right_columns,
        )? {
            return Ok(common::build_left_join_with_keys(
                left_op,
                right_op,
                &left_columns,
                &right_columns,
                &left_types,
                &right_types,
                left_keys,
                right_keys,
            ));
        }

        Ok(common::build_left_join(
            left_op,
            right_op,
            &left_columns,
            &right_columns,
            &left_types,
            &right_types,
        ))
    }

    /// Plans an ANTI JOIN operator (for SPARQL MINUS).
    fn plan_anti_join(
        &self,
        join: &AntiJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;
        let (left_op, left_columns, left_types) = self.plan_operator(&join.left)?;
        let (right_op, right_columns, right_types) = self.plan_operator(&join.right)?;
        validate_rdf_binary_join_metadata(
            &left_columns,
            &right_columns,
            &join.compatibility_conditions,
            "AntiJoin",
        )?;
        if has_sparql_compatibility(&join.compatibility_conditions) {
            return build_rdf_compatibility_join(
                PlannedRdfRelation::new(left_op, left_columns, left_types),
                PlannedRdfRelation::new(right_op, right_columns, right_types),
                &join.compatibility_conditions,
                RdfCompatibilityMode::Anti {
                    require_bound_overlap: join.semantics == AntiJoinSemantics::Minus,
                },
            );
        }
        let explicit_keys = resolve_rdf_join_keys(
            &join.compatibility_conditions,
            &left_columns,
            &right_columns,
        )?;
        let (op, cols) = match explicit_keys {
            Some((left_keys, right_keys)) => common::build_anti_join_with_keys(
                left_op,
                right_op,
                left_columns,
                left_types.clone(),
                left_keys,
                right_keys,
                join.semantics == AntiJoinSemantics::Minus,
            ),
            None if join.semantics == AntiJoinSemantics::NotExists => {
                // With no correlated variables, every left mapping satisfies
                // NOT EXISTS iff the right side is empty. Hash the empty tuple
                // so any right row rejects every left row.
                common::build_anti_join_with_keys(
                    left_op,
                    right_op,
                    left_columns,
                    left_types.clone(),
                    Vec::new(),
                    Vec::new(),
                    false,
                )
            }
            None => common::build_anti_join(
                left_op,
                right_op,
                left_columns,
                &right_columns,
                left_types.clone(),
            ),
        };
        Ok((op, cols, left_types))
    }

    /// Plans a multi-way join. When the Ring Index is available and all inputs
    /// are TripleScans, uses LeapfrogRing (WCOJ) for worst-case optimal joins.
    /// Otherwise falls back to cascading pairwise hash joins.
    fn plan_multi_way_join(
        &self,
        mwj: &crate::query::plan::MultiWayJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;

        if mwj.inputs.is_empty() {
            return Err(Error::Internal(
                "MultiWayJoin requires at least one input".to_string(),
            ));
        }
        validate_rdf_multiway_metadata(mwj)?;
        let input_order = rdf_multiway_join_order(mwj, &self.store);

        // Try Ring-backed LeapfrogRing (WCOJ) when all inputs are TripleScans.
        // Skip when LANG/DATATYPE companions are consumed. Native output
        // otherwise mirrors scan-visible columns and demanded exact/identity
        // companions, but does not yet implement full datatype sidecars.
        #[cfg(feature = "ring-index")]
        if self.native_ring_enabled
            && rdf_native_ring_multi_pattern_is_qualified()
            && !self.needs_companion_columns.get()
            && let Some(result) = self.try_leapfrog_ring(mwj, &input_order)
        {
            return result;
        }

        // Plan all inputs and estimate cardinalities
        let mut planned: Vec<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>, f64)> = Vec::new();
        for input_index in input_order {
            let input = &mwj.inputs[input_index];
            let (op, cols, types) = self.plan_operator(input)?;
            let card = estimate_operator_cardinality(input, &self.store).unwrap_or(1000.0);
            planned.push((op, cols, types, card));
        }
        let declared = mwj
            .shared_variables
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let mut occurrences = HashMap::<&str, usize>::new();
        for (_, columns, _, _) in &planned {
            let mut input_names = HashSet::new();
            for column in columns {
                if !is_rdf_internal_physical_column(column) && input_names.insert(column.as_str()) {
                    *occurrences.entry(column.as_str()).or_default() += 1;
                }
            }
        }
        let overlaps = occurrences
            .into_iter()
            .filter_map(|(variable, count)| (count >= 2).then_some(variable))
            .collect::<HashSet<_>>();
        if overlaps != declared {
            return Err(Error::InvalidValue(
                "RDF MultiWayJoin declared shared variables must exactly match overlapping public input columns"
                    .to_string(),
            ));
        }
        for variable in &mwj.shared_variables {
            let occurrences = planned
                .iter()
                .filter(|(_, columns, _, _)| columns.iter().any(|column| column == variable))
                .count();
            if occurrences < 2 {
                return Err(Error::InvalidValue(format!(
                    "RDF MultiWayJoin shared variable ?{variable} must occur in at least two inputs"
                )));
            }
        }

        // Fold left-to-right with pairwise hash joins
        let (mut current_op, mut current_cols, mut current_types, mut current_card) =
            planned.remove(0);
        for (right_op, right_cols, right_types, right_card) in planned {
            let cardinalities = Some((current_card, right_card));
            let (joined_op, joined_cols, joined_types) = if let Some((left_keys, right_keys)) =
                resolve_rdf_multiway_join_keys(&mwj.conditions, &current_cols, &right_cols)?
            {
                common::build_inner_join_with_keys(
                    current_op,
                    right_op,
                    &current_cols,
                    &right_cols,
                    &current_types,
                    &right_types,
                    left_keys,
                    right_keys,
                    cardinalities,
                )
            } else {
                common::build_inner_join_with_keys(
                    current_op,
                    right_op,
                    &current_cols,
                    &right_cols,
                    &current_types,
                    &right_types,
                    Vec::new(),
                    Vec::new(),
                    cardinalities,
                )
            };
            // Rough estimate for cascaded join output
            current_card = (current_card * right_card * 0.1).max(1.0);
            current_op = joined_op;
            current_cols = joined_cols;
            current_types = joined_types;
        }

        Ok((current_op, current_cols, current_types))
    }

    /// Attempts to plan a multi-way join using LeapfrogRing (WCOJ).
    ///
    /// Returns `Some(Ok(...))` when the Ring is available and all inputs are
    /// simple TripleScans, `None` to fall back to cascading hash joins.
    #[cfg(feature = "ring-index")]
    fn try_leapfrog_ring(
        &self,
        mwj: &crate::query::plan::MultiWayJoinOp,
        input_order: &[usize],
    ) -> Option<Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)>> {
        use grafeo_core::index::ring::AnnotatedPattern;

        // Native LFTJ is admitted only for a fresh default-graph snapshot and
        // an exhaustive homogeneous same-name RDF-identity equivalence shape.
        // Every other shape retains the typed relational implementation.
        if mwj.inputs.len() < 3
            || self.transaction_id.is_some()
            || mwj.conditions.is_empty()
            || !mwj.conditions.iter().all(|condition| {
                condition.semantics == JoinKeySemantics::RdfTermIdentity
                    && matches!(
                        (&condition.left, &condition.right),
                        (
                            LogicalExpression::Variable(left),
                            LogicalExpression::Variable(right)
                        ) if left == right
                    )
            })
        {
            return None;
        }

        // All inputs must be simple TripleScans (no chained input, no graph context)
        let mut annotated = Vec::new();
        let mut all_vars: Vec<String> = Vec::new();
        let mut output_owners = Vec::new();
        let mut occurrences = HashMap::<String, usize>::new();
        for &input_index in input_order {
            let input = &mwj.inputs[input_index];
            let LogicalOperator::TripleScan(scan) = input else {
                return None;
            };
            if scan.input.is_some() || scan.graph.is_some() || scan.dataset.is_some() {
                return None;
            }
            let input_variables = [
                scan.subject.as_variable(),
                scan.predicate.as_variable(),
                scan.object.as_variable(),
            ]
            .into_iter()
            .flatten()
            .collect::<HashSet<_>>();
            for variable in input_variables {
                *occurrences.entry(variable.to_string()).or_default() += 1;
            }
            let pattern = self.build_triple_pattern(scan);
            let ap = AnnotatedPattern {
                pattern,
                subject_var: scan.subject.as_variable().map(str::to_string),
                predicate_var: scan.predicate.as_variable().map(str::to_string),
                object_var: scan.object.as_variable().map(str::to_string),
            };
            // Collect output variables in order
            for (component, variable) in [
                (0_u8, &ap.subject_var),
                (1_u8, &ap.predicate_var),
                (2_u8, &ap.object_var),
            ] {
                if let Some(variable) = variable
                    && !all_vars.contains(variable)
                {
                    all_vars.push(variable.clone());
                    output_owners.push((annotated.len(), component));
                }
            }
            annotated.push(ap);
        }
        let actual_overlap = occurrences
            .into_iter()
            .filter_map(|(variable, count)| (count >= 2).then_some(variable))
            .collect::<HashSet<_>>();
        let declared_overlap = mwj.shared_variables.iter().cloned().collect::<HashSet<_>>();
        let condition_variables = mwj
            .conditions
            .iter()
            .filter_map(|condition| match (&condition.left, &condition.right) {
                (LogicalExpression::Variable(left), LogicalExpression::Variable(right))
                    if left == right =>
                {
                    Some(left.clone())
                }
                _ => None,
            })
            .collect::<HashSet<_>>();
        if actual_overlap != declared_overlap || condition_variables != declared_overlap {
            return None;
        }

        // A stale or absent derived Ring is never authoritative. Falling back
        // observes the current store/transaction snapshot instead.
        let ring = self.store.ring()?;

        // Ring performs native term-identity intersection internally. Preserve
        // reconstruction terms and comparison keys as separate companions.
        let emit_exact_term_columns = self.needs_exact_term_columns.get();
        let emit_identity_key_columns = self.needs_identity_key_columns.get();
        let mut columns = Vec::new();
        let mut types = Vec::new();
        for (variable, (_, component)) in all_vars.iter().zip(&output_owners) {
            columns.push(variable.clone());
            types.push(if *component == 2 {
                LogicalType::Any
            } else {
                LogicalType::String
            });
            if emit_exact_term_columns {
                columns.push(rdf_exact_term_column(variable));
                types.push(LogicalType::String);
            }
            if emit_identity_key_columns {
                columns.push(rdf_identity_key_column(variable));
                types.push(LogicalType::String);
            }
            if *component == 2 {
                columns.push(format!("__lang_{variable}"));
                types.push(LogicalType::String);
            }
        }

        let operator = Box::new(RdfLeapfrogOperator::new(
            ring,
            annotated,
            RdfLeapfrogConfig {
                output_variables: all_vars,
                output_owners,
                output_types: types.clone(),
                emit_exact_term_columns,
                emit_identity_key_columns,
                chunk_size: self.chunk_size,
                output_cap: self.native_ring_output_cap.get(),
            },
        ));
        if self.profiling.get() {
            let mut entries = self.profile_entries.borrow_mut();
            for input in &mwj.inputs {
                let label = format!(
                    "{} [fused; stats unavailable, time in parent]",
                    input.display_label()
                );
                let (entry, _stats) =
                    crate::query::profile::ProfileEntry::new("RdfRingTrieInput", label);
                entries.push(entry);
            }
        }
        Some(Ok((operator, columns, types)))
    }

    /// Plans a UNION operator.
    fn plan_union(
        &self,
        union: &crate::query::plan::UnionOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        if union.inputs.is_empty() {
            return Err(Error::Internal("Empty UNION".to_string()));
        }

        let mut planned = Vec::with_capacity(union.inputs.len());
        let mut columns = Vec::new();
        let mut types = Vec::new();

        for input in &union.inputs {
            let (op, cols, tys) = self.plan_operator(input)?;
            for (column, ty) in cols.iter().zip(&tys) {
                if let Some(index) = columns.iter().position(|existing| existing == column) {
                    if types[index] != *ty {
                        types[index] = LogicalType::Any;
                    }
                } else {
                    columns.push(column.clone());
                    types.push(ty.clone());
                }
            }
            planned.push((op, cols));
        }

        let mut operators = Vec::with_capacity(planned.len());
        for (operator, branch_columns) in planned {
            if branch_columns == columns || columns.is_empty() {
                operators.push(operator);
                continue;
            }
            let branch_map: HashMap<&str, usize> = branch_columns
                .iter()
                .enumerate()
                .map(|(index, column)| (column.as_str(), index))
                .collect();
            let projections = columns
                .iter()
                .map(|column| {
                    branch_map.get(column.as_str()).map_or_else(
                        || RdfProjectExpr::Constant(Value::Null),
                        |index| RdfProjectExpr::Column(*index),
                    )
                })
                .collect();
            operators.push(Box::new(RdfProjectOperator::new(
                operator,
                projections,
                types.clone(),
            )) as Box<dyn Operator>);
        }

        if operators.len() == 1 {
            return Ok((
                operators.pop().ok_or_else(|| {
                    Error::Internal("RDF UNION lost its sole planned branch".to_string())
                })?,
                columns,
                types,
            ));
        }

        // Create a chain operator that executes all operators in sequence
        let operator = Box::new(RdfUnionOperator::new(operators));
        Ok((operator, columns, types))
    }

    /// Plans an INSERT TRIPLE operator.
    fn plan_insert_triple(
        &self,
        insert: &InsertTripleOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        // Check if this is a pattern-based insert (has variables in the template).
        // Blank nodes are concrete values, not variables, so they don't trigger
        // the pattern-based path.
        let has_variables = matches!(&insert.subject, TripleComponent::Variable(_))
            || matches!(&insert.predicate, TripleComponent::Variable(_))
            || matches!(&insert.object, TripleComponent::Variable(_));

        if has_variables {
            // Pattern-based insertion: need to query first, then insert each match
            if let Some(ref input) = insert.input {
                let (input_op, input_columns, _input_types) = self.plan_operator(input)?;

                // Build column index map for variable substitution
                let column_map: HashMap<String, usize> = input_columns
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.clone(), i))
                    .collect();

                let operator = Box::new(RdfInsertPatternOperator::new(
                    Arc::clone(&self.store),
                    input_op,
                    TripleOperands {
                        subject: insert.subject.clone(),
                        predicate: insert.predicate.clone(),
                        object: insert.object.clone(),
                        column_map,
                        graph: insert.graph.clone(),
                        transaction_id: self.transaction_id,
                        valid_time: self.valid_time,
                    },
                    #[cfg(feature = "wal")]
                    self.wal.clone(),
                    #[cfg(feature = "cdc")]
                    self.cdc_log.clone(),
                ));

                return Ok((operator, Vec::new(), Vec::new()));
            }
        }

        // Direct insertion with concrete terms
        let subject = self.component_to_term(&insert.subject)?;
        let predicate = self.component_to_term(&insert.predicate)?;
        let object = self.component_to_term(&insert.object)?;

        let triple = Triple::new(subject, predicate, object);
        let operator = Box::new(RdfInsertTripleOperator::new(
            Arc::clone(&self.store),
            triple,
            insert.graph.clone(),
            self.transaction_id,
            self.valid_time,
            #[cfg(feature = "wal")]
            self.wal.clone(),
            #[cfg(feature = "cdc")]
            self.cdc_log.clone(),
        ));

        // Insert operations don't output columns
        Ok((operator, Vec::new(), Vec::new()))
    }

    /// Converts a TripleComponent to an RDF Term.
    fn component_to_term(&self, component: &TripleComponent) -> Result<Term> {
        match component {
            TripleComponent::Iri(iri) => Ok(Term::Iri(iri.clone().into())),
            TripleComponent::Literal(value) => Ok(value_as_rdf_term(value)),
            TripleComponent::LangLiteral { value, lang } => {
                Ok(Term::lang_literal(value.clone(), lang.clone()))
            }
            TripleComponent::BlankNode(label) => Ok(Term::blank(label.clone())),
            TripleComponent::Variable(name) => {
                // Variables in INSERT DATA should have been bound
                Err(Error::Internal(format!(
                    "Unbound variable '{}' in INSERT DATA",
                    name
                )))
            }
        }
    }

    /// Plans a DELETE TRIPLE operator.
    fn plan_delete_triple(
        &self,
        delete: &DeleteTripleOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        // Check if this is a pattern-based delete (has variables in the template)
        let has_variables = matches!(&delete.subject, TripleComponent::Variable(_))
            || matches!(&delete.predicate, TripleComponent::Variable(_))
            || matches!(&delete.object, TripleComponent::Variable(_));

        if has_variables {
            // Pattern-based deletion: need to query first, then delete each match
            if let Some(ref input) = delete.input {
                let (input_op, input_columns, _input_types) = self.plan_operator(input)?;

                // Build column index map for variable substitution
                let column_map: HashMap<String, usize> = input_columns
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.clone(), i))
                    .collect();

                let operator = Box::new(RdfDeletePatternOperator::new(
                    Arc::clone(&self.store),
                    input_op,
                    TripleOperands {
                        subject: delete.subject.clone(),
                        predicate: delete.predicate.clone(),
                        object: delete.object.clone(),
                        column_map,
                        graph: delete.graph.clone(),
                        transaction_id: self.transaction_id,
                        valid_time: None,
                    },
                    #[cfg(feature = "wal")]
                    self.wal.clone(),
                    #[cfg(feature = "cdc")]
                    self.cdc_log.clone(),
                ));

                return Ok((operator, Vec::new(), Vec::new()));
            }
        }

        // Direct deletion with concrete terms
        let subject = self.component_to_term(&delete.subject)?;
        let predicate = self.component_to_term(&delete.predicate)?;
        let object = self.component_to_term(&delete.object)?;

        let triple = Triple::new(subject, predicate, object);
        let operator = Box::new(RdfDeleteTripleOperator::new(
            Arc::clone(&self.store),
            triple,
            delete.graph.clone(),
            self.transaction_id,
            #[cfg(feature = "wal")]
            self.wal.clone(),
            #[cfg(feature = "cdc")]
            self.cdc_log.clone(),
        ));

        Ok((operator, Vec::new(), Vec::new()))
    }

    /// Plans a CLEAR GRAPH operator.
    fn plan_clear_graph(
        &self,
        clear: &ClearGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let operator = Box::new(RdfClearGraphOperator::new(
            Arc::clone(&self.store),
            clear.graph.clone(),
            clear.silent,
            self.transaction_id,
            #[cfg(feature = "wal")]
            self.wal.clone(),
        ));
        Ok((operator, Vec::new(), Vec::new()))
    }

    /// Plans a CREATE GRAPH operator.
    fn plan_create_graph(
        &self,
        create: &CreateGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let operator = Box::new(RdfCreateGraphOperator::new(
            Arc::clone(&self.store),
            create.graph.clone(),
            create.silent,
            self.transaction_id,
            #[cfg(feature = "wal")]
            self.wal.clone(),
        ));
        Ok((operator, Vec::new(), Vec::new()))
    }

    /// Plans a DROP GRAPH operator.
    fn plan_drop_graph(
        &self,
        drop_op: &DropGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let operator = Box::new(RdfDropGraphOperator::new(
            Arc::clone(&self.store),
            drop_op.graph.clone(),
            drop_op.silent,
            self.transaction_id,
            #[cfg(feature = "wal")]
            self.wal.clone(),
        ));
        Ok((operator, Vec::new(), Vec::new()))
    }

    /// Plans a COPY graph operator.
    fn plan_copy_graph(
        &self,
        copy: &CopyGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let operator = Box::new(RdfCopyGraphOperator::new(
            Arc::clone(&self.store),
            copy.source.clone(),
            copy.destination.clone(),
            copy.silent,
            self.transaction_id,
            #[cfg(feature = "wal")]
            self.wal.clone(),
        ));
        Ok((operator, Vec::new(), Vec::new()))
    }

    /// Plans a MOVE graph operator.
    fn plan_move_graph(
        &self,
        move_op: &MoveGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let operator = Box::new(RdfMoveGraphOperator::new(
            Arc::clone(&self.store),
            move_op.source.clone(),
            move_op.destination.clone(),
            move_op.silent,
            self.transaction_id,
            #[cfg(feature = "wal")]
            self.wal.clone(),
        ));
        Ok((operator, Vec::new(), Vec::new()))
    }

    /// Plans an ADD graph operator.
    fn plan_add_graph(
        &self,
        add: &AddGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let operator = Box::new(RdfAddGraphOperator::new(
            Arc::clone(&self.store),
            add.source.clone(),
            add.destination.clone(),
            add.silent,
            self.transaction_id,
            #[cfg(feature = "wal")]
            self.wal.clone(),
        ));
        Ok((operator, Vec::new(), Vec::new()))
    }

    /// SPARQL LOAD is not executable in the embedded engine.
    fn plan_load_graph(
        &self,
        load: &LoadGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        if load.silent {
            let op: Box<dyn Operator> = Box::new(SingleRowOperator::new());
            return Ok((op, Vec::new(), Vec::new()));
        }
        Err(Error::Query(
            grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Semantic,
                format!("SPARQL LOAD <{}> is not supported", load.source),
            )
            .with_hint(
                "Ingest with INSERT DATA or GrafeoDB::batch_insert_rdf; LOAD from URLs is not executable".to_string(),
            ),
        ))
    }

    /// Plans a SPARQL MODIFY operator (DELETE/INSERT WHERE).
    ///
    /// Per SPARQL 1.1 spec:
    /// 1. Evaluate WHERE clause once to get bindings
    /// 2. Apply DELETE templates using those bindings
    /// 3. Apply INSERT templates using the SAME bindings
    fn plan_modify(
        &self,
        modify: &ModifyOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        // Plan the WHERE clause
        let (where_op, where_columns, _where_types) = self.plan_operator(&modify.where_clause)?;

        // Build column index map for variable substitution
        let column_map: HashMap<String, usize> = where_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        let sealed_identity = column_map.contains_key(RDF_SEALED_MODIFY_COLUMN);

        // Fail before constructing an executable mutation when a bound
        // template variable has no lossless RDF identity. Variables absent
        // from the WHERE schema remain legitimately unbound and simply omit
        // their template triple at execution time.
        let mut template_variables = HashSet::new();
        for template in modify
            .delete_templates
            .iter()
            .chain(&modify.insert_templates)
        {
            for component in [&template.subject, &template.predicate, &template.object] {
                if let TripleComponent::Variable(name) = component {
                    template_variables.insert(name.strip_prefix('?').unwrap_or(name));
                }
            }
            if let Some(graph) = template.graph.as_deref()
                && let Some(variable) = rdf_graph_variable_from_template(graph)
            {
                template_variables.insert(variable);
            }
        }
        for variable in template_variables {
            if sealed_identity
                && column_map.contains_key(variable)
                && !column_map.contains_key(&rdf_exact_term_column(variable))
            {
                return Err(Error::Query(
                    grafeo_common::utils::error::QueryError::new(
                        grafeo_common::utils::error::QueryErrorKind::Semantic,
                        format!(
                            "RDF mutation variable ?{variable} has no lossless term identity"
                        ),
                    )
                    .with_hint(
                        "Use an exact RDF-producing binding, or keep this variable outside the update template"
                            .to_string(),
                    ),
                ));
            }
        }

        let operator = Box::new(RdfModifyOperator::new(
            Arc::clone(&self.store),
            where_op,
            modify.delete_templates.clone(),
            modify.insert_templates.clone(),
            column_map,
            sealed_identity,
            RdfModifyContext {
                transaction_id: self.transaction_id,
                valid_time: self.valid_time,
                #[cfg(feature = "wal")]
                wal: self.wal.clone(),
                #[cfg(feature = "cdc")]
                cdc_log: self.cdc_log.clone(),
            },
        ));

        Ok((operator, Vec::new(), Vec::new()))
    }
}

/// A `MultiWayJoinOp` does not yet carry relation endpoint IDs. After the
/// planner reorders inputs, declared conditions are therefore unambiguous only
/// when they describe one homogeneous, same-named equivalence relation.
/// Binary plans retain every other condition shape until endpoint ownership is
/// represented explicitly.
fn validate_rdf_multiway_metadata(mwj: &crate::query::plan::MultiWayJoinOp) -> Result<()> {
    let mut shared = HashSet::<&str>::new();
    if mwj
        .shared_variables
        .iter()
        .any(|variable| !shared.insert(variable.as_str()))
    {
        return Err(Error::InvalidValue(
            "RDF MultiWayJoin shared variables must be unique".to_string(),
        ));
    }
    let Some(first) = mwj.conditions.first() else {
        if shared.is_empty() {
            return Ok(());
        }
        return Err(Error::InvalidValue(
            "RDF MultiWayJoin shared variables require explicit equality semantics".to_string(),
        ));
    };
    if first.semantics == JoinKeySemantics::SparqlCompatibility {
        return Err(Error::Query(grafeo_common::utils::error::QueryError::new(
            grafeo_common::utils::error::QueryErrorKind::Semantic,
            "SPARQL compatibility joins require the RDF compatibility operator",
        )));
    }
    let mut declared = HashSet::<&str>::new();
    for condition in &mwj.conditions {
        let variable = match (&condition.left, &condition.right) {
            (LogicalExpression::Variable(left), LogicalExpression::Variable(right))
                if left == right =>
            {
                left.as_str()
            }
            _ => {
                return Err(Error::InvalidValue(
                    "RDF MultiWayJoin requires homogeneous same-named Value or RDF-identity conditions; preserve mixed or owned conditions as binary joins"
                        .to_string(),
                ));
            }
        };
        if condition.semantics != first.semantics {
            return Err(Error::InvalidValue(
                "RDF MultiWayJoin requires homogeneous same-named Value or RDF-identity conditions; preserve mixed or owned conditions as binary joins"
                    .to_string(),
            ));
        }
        if !declared.insert(variable) {
            return Err(Error::InvalidValue(format!(
                "RDF MultiWayJoin has duplicate conditions for ?{variable}"
            )));
        }
    }
    if declared != shared {
        return Err(Error::InvalidValue(
            "RDF MultiWayJoin conditions must declare every shared variable exactly by name"
                .to_string(),
        ));
    }
    Ok(())
}

fn has_sparql_compatibility(conditions: &[JoinCondition]) -> bool {
    conditions
        .iter()
        .any(|condition| condition.semantics == JoinKeySemantics::SparqlCompatibility)
}

fn is_rdf_internal_physical_column(column: &str) -> bool {
    is_rdf_internal_term_column(column)
        || column.starts_with("__lang_")
        || column.starts_with("__datatype_")
}

/// Validates the public schema contract for a typed binary RDF join.
///
/// Physical join builders emit one column for each same-named overlap. That is
/// sound only when every such overlap has explicit typed equality metadata;
/// otherwise a public column could be silently discarded or coalesced without
/// ever being compared. Internal RDF companion columns are derived from their
/// visible owner and are deliberately excluded from this public contract.
fn validate_rdf_binary_join_metadata(
    left_columns: &[String],
    right_columns: &[String],
    conditions: &[JoinCondition],
    boundary: &str,
) -> Result<()> {
    fn public_names(columns: &[String], boundary: &str, side: &str) -> Result<HashSet<String>> {
        let mut names = HashSet::new();
        for column in columns {
            if is_rdf_internal_physical_column(column) {
                continue;
            }
            if !names.insert(column.clone()) {
                return Err(Error::InvalidValue(format!(
                    "RDF {boundary} {side} input has duplicate public column {column:?}"
                )));
            }
        }
        Ok(names)
    }

    let left = public_names(left_columns, boundary, "left")?;
    let right = public_names(right_columns, boundary, "right")?;
    let overlaps = left.intersection(&right).cloned().collect::<HashSet<_>>();

    let mut declared = HashSet::new();
    for condition in conditions {
        let (LogicalExpression::Variable(left), LogicalExpression::Variable(right)) =
            (&condition.left, &condition.right)
        else {
            return Err(Error::InvalidValue(format!(
                "RDF {boundary} metadata requires variable equality expressions"
            )));
        };
        if left == right && !declared.insert(left.clone()) {
            return Err(Error::InvalidValue(format!(
                "RDF {boundary} has duplicate conditions for public column {left:?}"
            )));
        }
    }

    if overlaps != declared {
        return Err(Error::InvalidValue(format!(
            "RDF {boundary} conditions must exactly declare every same-named public input column"
        )));
    }
    Ok(())
}

fn resolve_rdf_compatibility_keys(
    conditions: &[JoinCondition],
    left_columns: &[String],
    right_columns: &[String],
) -> Result<Vec<RdfCompatibilityKey>> {
    conditions
        .iter()
        .map(|condition| {
            let (left_name, right_name) = match (&condition.left, &condition.right) {
                (LogicalExpression::Variable(left), LogicalExpression::Variable(right)) => {
                    (left.as_str(), right.as_str())
                }
                _ => {
                    return Err(Error::Internal(
                        "RDF compatibility metadata requires variable expressions".to_string(),
                    ));
                }
            };
            let left_visible = left_columns
                .iter()
                .position(|column| column == left_name)
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "RDF compatibility variable ?{left_name} was not materialized on the left input"
                    ))
                })?;
            let right_visible = right_columns
                .iter()
                .position(|column| column == right_name)
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "RDF compatibility variable ?{right_name} was not materialized on the right input"
                    ))
                })?;
            let (left_group_key, right_group_key, left_identity, right_identity) =
                if condition.semantics == JoinKeySemantics::Value {
                    (None, None, None, None)
            } else {
                let left_group_key_name = rdf_group_key_column(left_name);
                let right_group_key_name = rdf_group_key_column(right_name);
                let left_identity_name = rdf_identity_key_column(left_name);
                let right_identity_name = rdf_identity_key_column(right_name);
                let left_group_key = left_columns
                    .iter()
                    .position(|column| column == &left_group_key_name);
                let right_group_key = right_columns
                    .iter()
                    .position(|column| column == &right_group_key_name);
                let left_identity = left_columns
                    .iter()
                    .position(|column| column == &left_identity_name);
                let right_identity = right_columns
                    .iter()
                    .position(|column| column == &right_identity_name);
                (
                    left_group_key,
                    right_group_key,
                    left_identity,
                    right_identity,
                )
            };
            Ok(RdfCompatibilityKey {
                left_visible,
                right_visible,
                left_group_key,
                right_group_key,
                left_identity,
                right_identity,
                semantics: condition.semantics,
            })
        })
        .collect()
}

struct PlannedRdfRelation {
    operator: Box<dyn Operator>,
    columns: Vec<String>,
    types: Vec<LogicalType>,
}

impl PlannedRdfRelation {
    fn new(operator: Box<dyn Operator>, columns: Vec<String>, types: Vec<LogicalType>) -> Self {
        Self {
            operator,
            columns,
            types,
        }
    }
}

fn build_rdf_compatibility_join(
    left: PlannedRdfRelation,
    right: PlannedRdfRelation,
    conditions: &[JoinCondition],
    mode: RdfCompatibilityMode,
) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
    let keys = resolve_rdf_compatibility_keys(conditions, &left.columns, &right.columns)?;
    let mut output_columns = Vec::new();
    let mut output_types = Vec::new();
    let mut output_layout = Vec::new();

    if matches!(
        mode,
        RdfCompatibilityMode::Semi | RdfCompatibilityMode::Anti { .. }
    ) {
        for (index, (column, ty)) in left.columns.iter().zip(&left.types).enumerate() {
            output_columns.push(column.clone());
            output_types.push(ty.clone());
            output_layout.push(RdfCompatibilityOutputColumn::Left(index));
        }
    } else {
        let normalized_helpers = conditions
            .iter()
            .zip(&keys)
            .filter_map(|(condition, key)| {
                if condition.semantics == JoinKeySemantics::Value {
                    return None;
                }
                let LogicalExpression::Variable(variable) = &condition.left else {
                    return None;
                };
                Some((rdf_group_key_column(variable), key.clone()))
            })
            .collect::<Vec<_>>();
        let normalized_helper_names = normalized_helpers
            .iter()
            .map(|(column, _)| column.clone())
            .collect::<HashSet<_>>();
        let mut seen = HashSet::new();
        for (left_index, (column, left_type)) in left.columns.iter().zip(&left.types).enumerate() {
            if normalized_helper_names.contains(column) {
                seen.insert(column.clone());
                continue;
            }
            if !seen.insert(column.clone()) {
                continue;
            }
            let right_index = right
                .columns
                .iter()
                .position(|right_column| right_column == column);
            if let Some(right_index) = right_index {
                let output_type = if *left_type == right.types[right_index] {
                    left_type.clone()
                } else {
                    LogicalType::Any
                };
                output_layout.push(RdfCompatibilityOutputColumn::Coalesce {
                    left: left_index,
                    right: right_index,
                });
                output_types.push(output_type);
            } else {
                output_layout.push(RdfCompatibilityOutputColumn::Left(left_index));
                output_types.push(left_type.clone());
            }
            output_columns.push(column.clone());
        }
        for (right_index, (column, right_type)) in
            right.columns.iter().zip(&right.types).enumerate()
        {
            if normalized_helper_names.contains(column) {
                seen.insert(column.clone());
                continue;
            }
            if seen.insert(column.clone()) {
                output_columns.push(column.clone());
                output_types.push(right_type.clone());
                output_layout.push(RdfCompatibilityOutputColumn::Right(right_index));
            }
        }
        for (column, key) in normalized_helpers {
            output_columns.push(column);
            output_types.push(LogicalType::Any);
            output_layout.push(RdfCompatibilityOutputColumn::NormalizedIdentity {
                left_visible: key.left_visible,
                right_visible: key.right_visible,
                left_group_key: key.left_group_key,
                right_group_key: key.right_group_key,
                left_identity: key.left_identity,
                right_identity: key.right_identity,
            });
        }
    }

    let operator = Box::new(RdfCompatibilityJoinOperator::new(
        left.operator,
        right.operator,
        keys,
        mode,
        output_layout,
        output_types.clone(),
    ));
    Ok((operator, output_columns, output_types))
}

/// Resolves logical RDF join semantics to canonical physical key columns.
/// Returns `None` only when no key semantics were declared. Declared ordinary
/// value keys resolve to their visible columns; they must not accidentally
/// absorb unrelated hidden RDF companions from the physical schema.
fn resolve_rdf_join_keys(
    conditions: &[JoinCondition],
    left_columns: &[String],
    right_columns: &[String],
) -> Result<Option<(Vec<usize>, Vec<usize>)>> {
    if conditions.is_empty() {
        return Ok(None);
    }

    let mut left_keys = Vec::with_capacity(conditions.len());
    let mut right_keys = Vec::with_capacity(conditions.len());
    for condition in conditions {
        let (left_name, right_name) = match (&condition.left, &condition.right) {
            (LogicalExpression::Variable(left), LogicalExpression::Variable(right)) => {
                (left.as_str(), right.as_str())
            }
            _ => {
                return Err(Error::Internal(
                    "RDF join-key metadata requires variable expressions".to_string(),
                ));
            }
        };
        let (left_name, right_name) = match condition.semantics {
            JoinKeySemantics::Value => (left_name.to_string(), right_name.to_string()),
            JoinKeySemantics::RdfTermIdentity => (
                rdf_identity_key_column(left_name),
                rdf_identity_key_column(right_name),
            ),
            JoinKeySemantics::SparqlCompatibility => {
                return Err(Error::Query(grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Semantic,
                    format!(
                        "SPARQL compatibility for possibly unbound shared variable ?{left_name} is not yet supported"
                    ),
                )));
            }
        };
        let left_index = left_columns
            .iter()
            .position(|column| column == &left_name)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "RDF join key {left_name:?} was not materialized on the left input"
                ))
            })?;
        let right_index = right_columns
            .iter()
            .position(|column| column == &right_name)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "RDF join key {right_name:?} was not materialized on the right input"
                ))
            })?;
        left_keys.push(left_index);
        right_keys.push(right_index);
    }
    Ok(Some((left_keys, right_keys)))
}

/// Resolves only the declared conditions crossing the current multi-way fold.
/// Conditions wholly inside the accumulated side have already been enforced.
fn resolve_rdf_multiway_join_keys(
    conditions: &[JoinCondition],
    left_columns: &[String],
    right_columns: &[String],
) -> Result<Option<(Vec<usize>, Vec<usize>)>> {
    if conditions.is_empty() {
        return Ok(None);
    }

    let mut keys = Vec::new();
    let mut seen = HashSet::new();
    for condition in conditions {
        let (left_variable, right_variable) = match (&condition.left, &condition.right) {
            (LogicalExpression::Variable(left), LogicalExpression::Variable(right)) => {
                (left.as_str(), right.as_str())
            }
            _ => {
                return Err(Error::Internal(
                    "RDF multi-way join metadata requires variable expressions".to_string(),
                ));
            }
        };
        let physical = |variable: &str| match condition.semantics {
            JoinKeySemantics::Value => variable.to_string(),
            JoinKeySemantics::RdfTermIdentity => rdf_identity_key_column(variable),
            JoinKeySemantics::SparqlCompatibility => variable.to_string(),
        };
        if condition.semantics == JoinKeySemantics::SparqlCompatibility {
            return Err(Error::Query(grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Semantic,
                format!(
                    "SPARQL compatibility for possibly unbound shared variable ?{left_variable} is not yet supported"
                ),
            )));
        }
        let orientation = left_columns
            .iter()
            .position(|column| column == left_variable)
            .zip(
                right_columns
                    .iter()
                    .position(|column| column == right_variable),
            )
            .map(|_| (left_variable, right_variable))
            .or_else(|| {
                left_columns
                    .iter()
                    .position(|column| column == right_variable)
                    .zip(
                        right_columns
                            .iter()
                            .position(|column| column == left_variable),
                    )
                    .map(|_| (right_variable, left_variable))
            });
        let Some((left_variable, right_variable)) = orientation else {
            // This condition is wholly inside the accumulated side or belongs
            // to a relation that has not entered the fold yet.
            continue;
        };
        let left_name = physical(left_variable);
        let right_name = physical(right_variable);
        let left_index = left_columns
            .iter()
            .position(|column| column == &left_name)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "RDF multi-way join key {left_name:?} was not materialized on the left input"
                ))
            })?;
        let right_index = right_columns
            .iter()
            .position(|column| column == &right_name)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "RDF multi-way join key {right_name:?} was not materialized on the right input"
                ))
            })?;
        if seen.insert((left_index, right_index)) {
            keys.push((left_index, right_index));
        }
    }

    if keys.is_empty() {
        return Ok(None);
    }
    Ok(Some(keys.into_iter().unzip()))
}

// ============================================================================
// RDF Insert Triple Operator
// ============================================================================

/// Operator that inserts a triple into the RDF store.
struct RdfInsertTripleOperator {
    store: Arc<RdfStore>,
    triple: Triple,
    graph_name: Option<String>,
    transaction_id: Option<TransactionId>,
    valid_time: Option<ValidTimeInterval>,
    inserted: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
    #[cfg(feature = "cdc")]
    cdc_log: Option<Arc<RdfCdcSink>>,
}

impl RdfInsertTripleOperator {
    fn new(
        store: Arc<RdfStore>,
        triple: Triple,
        graph_name: Option<String>,
        transaction_id: Option<TransactionId>,
        valid_time: Option<ValidTimeInterval>,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
        #[cfg(feature = "cdc")] cdc_log: Option<Arc<RdfCdcSink>>,
    ) -> Self {
        Self {
            store,
            triple,
            graph_name,
            transaction_id,
            valid_time,
            inserted: false,
            #[cfg(feature = "wal")]
            wal,
            #[cfg(feature = "cdc")]
            cdc_log,
        }
    }
}

impl Operator for RdfInsertTripleOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.inserted {
            return Ok(None);
        }

        // Resolve target store: named graph or default
        let target = match &self.graph_name {
            Some(name) => self
                .store
                .graph_or_create_in_tx(name, self.transaction_id)
                .map_err(|error| OperatorError::Execution(error.to_string()))?,
            None => Arc::clone(&self.store),
        };

        if rdf_visible_representative(&target, self.transaction_id, &self.triple).is_some() {
            self.inserted = true;
            return Ok(None);
        }

        #[cfg(feature = "wal")]
        {
            ensure_rdf_graph_high_water(&self.wal, &self.store, self.graph_name.as_deref())?;
            log_rdf_wal(
                &self.wal,
                &rdf_insert_wal_record(
                    &self.triple,
                    self.graph_name.as_deref(),
                    target.graph_incarnation(),
                    self.transaction_id.unwrap_or(TransactionId::SYSTEM),
                    self.valid_time,
                ),
            )?;
        }

        // Append the durable mutation frame before changing the transaction
        // overlay. A caught WAL error therefore cannot leave a pending triple
        // that a caller could accidentally publish later.
        if let Some(transaction_id) = self.transaction_id {
            target.insert_in_transaction_with_valid(
                transaction_id,
                self.triple.clone(),
                self.valid_time,
            );
        } else {
            target
                .try_insert_at_epoch_with_valid(
                    self.triple.clone(),
                    target.commit_epoch(),
                    self.valid_time,
                )
                .map_err(|error| OperatorError::Execution(error.to_string()))?;
        }

        #[cfg(feature = "cdc")]
        record_cdc_triple_insert(
            &self.cdc_log,
            self.triple.subject(),
            self.triple.predicate(),
            self.triple.object(),
            self.graph_name.as_deref(),
            target.graph_incarnation(),
        );

        self.inserted = true;

        // Return an empty result (INSERT doesn't produce rows)
        Ok(None)
    }

    fn reset(&mut self) {
        self.inserted = false;
    }

    fn name(&self) -> &'static str {
        "RdfInsertTriple"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Insert Pattern Operator
// ============================================================================

/// Operator that inserts triples based on a pattern from the RDF store.
/// Used for INSERT { } WHERE { } operations where the triple template contains variables.
struct RdfInsertPatternOperator {
    store: Arc<RdfStore>,
    input: Box<dyn Operator>,
    subject: TripleComponent,
    predicate: TripleComponent,
    object: TripleComponent,
    column_map: HashMap<String, usize>,
    graph: Option<String>,
    transaction_id: Option<TransactionId>,
    valid_time: Option<ValidTimeInterval>,
    done: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
    #[cfg(feature = "cdc")]
    cdc_log: Option<Arc<RdfCdcSink>>,
}

impl RdfInsertPatternOperator {
    fn new(
        store: Arc<RdfStore>,
        input: Box<dyn Operator>,
        operands: TripleOperands,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
        #[cfg(feature = "cdc")] cdc_log: Option<Arc<RdfCdcSink>>,
    ) -> Self {
        Self {
            store,
            input,
            subject: operands.subject,
            predicate: operands.predicate,
            object: operands.object,
            column_map: operands.column_map,
            graph: operands.graph,
            transaction_id: operands.transaction_id,
            valid_time: operands.valid_time,
            done: false,
            #[cfg(feature = "wal")]
            wal,
            #[cfg(feature = "cdc")]
            cdc_log,
        }
    }
}

impl Operator for RdfInsertPatternOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }

        // Collect all triples to insert
        let mut triples_to_insert = Vec::new();

        while let Some(chunk) = self.input.next()? {
            for row in 0..chunk.row_count() {
                let subject =
                    resolve_public_pattern_component(&self.subject, &self.column_map, &chunk, row)?;
                let predicate = resolve_public_pattern_component(
                    &self.predicate,
                    &self.column_map,
                    &chunk,
                    row,
                )?;
                let object =
                    resolve_public_pattern_component(&self.object, &self.column_map, &chunk, row)?;

                if let (Some(s), Some(p), Some(o)) = (subject, predicate, object)
                    && let Some(triple) = instantiate_mutation_triple(s, p, o)
                {
                    triples_to_insert.push(triple);
                }
            }
        }

        let target = match &self.graph {
            Some(name) => self
                .store
                .graph_or_create_in_tx(name, self.transaction_id)
                .map_err(|error| OperatorError::Execution(error.to_string()))?,
            None => Arc::clone(&self.store),
        };
        let mut seen = grafeo_common::utils::hash::FxHashSet::default();
        triples_to_insert.retain(|triple| {
            seen.insert(triple.canonical_identity_key())
                && rdf_visible_representative(&target, self.transaction_id, triple).is_none()
        });
        #[cfg(feature = "wal")]
        {
            ensure_rdf_graph_high_water(&self.wal, &self.store, self.graph.as_deref())?;
            for triple in &triples_to_insert {
                log_rdf_wal(
                    &self.wal,
                    &rdf_insert_wal_record(
                        triple,
                        self.graph.as_deref(),
                        target.graph_incarnation(),
                        self.transaction_id.unwrap_or(TransactionId::SYSTEM),
                        self.valid_time,
                    ),
                )?;
            }
        }

        for triple in &triples_to_insert {
            if let Some(tid) = self.transaction_id {
                target.insert_in_transaction_with_valid(tid, triple.clone(), self.valid_time);
            } else {
                target
                    .try_insert_at_epoch_with_valid(
                        triple.clone(),
                        target.commit_epoch(),
                        self.valid_time,
                    )
                    .map_err(|error| OperatorError::Execution(error.to_string()))?;
            }
        }

        #[cfg(feature = "cdc")]
        for triple in &triples_to_insert {
            record_cdc_triple_insert(
                &self.cdc_log,
                triple.subject(),
                triple.predicate(),
                triple.object(),
                self.graph.as_deref(),
                target.graph_incarnation(),
            );
        }

        self.done = true;
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
        self.input.reset();
    }

    fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> std::result::Result<(), QueryResourceContextError> {
        self.input.install_resource_context(resources)
    }

    fn name(&self) -> &'static str {
        "RdfInsertPattern"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Delete Triple Operator
// ============================================================================

/// Operator that deletes a triple from the RDF store.
struct RdfDeleteTripleOperator {
    store: Arc<RdfStore>,
    triple: Triple,
    graph_name: Option<String>,
    transaction_id: Option<TransactionId>,
    deleted: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
    #[cfg(feature = "cdc")]
    cdc_log: Option<Arc<RdfCdcSink>>,
}

impl RdfDeleteTripleOperator {
    fn new(
        store: Arc<RdfStore>,
        triple: Triple,
        graph_name: Option<String>,
        transaction_id: Option<TransactionId>,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
        #[cfg(feature = "cdc")] cdc_log: Option<Arc<RdfCdcSink>>,
    ) -> Self {
        Self {
            store,
            triple,
            graph_name,
            transaction_id,
            deleted: false,
            #[cfg(feature = "wal")]
            wal,
            #[cfg(feature = "cdc")]
            cdc_log,
        }
    }
}

impl Operator for RdfDeleteTripleOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.deleted {
            return Ok(None);
        }

        // DELETE is lookup-only: an absent named graph is an empty target and
        // must not acquire a lifecycle merely because it was mentioned.
        let Some(target) =
            rdf_delete_target(&self.store, self.graph_name.as_deref(), self.transaction_id)
        else {
            self.deleted = true;
            return Ok(None);
        };

        let Some(representative) =
            rdf_visible_representative(&target, self.transaction_id, &self.triple)
        else {
            self.deleted = true;
            return Ok(None);
        };
        self.triple = representative.as_ref().clone();

        #[cfg(feature = "wal")]
        {
            ensure_rdf_graph_high_water(&self.wal, &self.store, self.graph_name.as_deref())?;
            log_rdf_wal(
                &self.wal,
                &rdf_delete_wal_record(
                    &self.triple,
                    self.graph_name.as_deref(),
                    target.graph_incarnation(),
                    self.transaction_id.unwrap_or(TransactionId::SYSTEM),
                ),
            )?;
        }

        // Delete the triple (buffered if in a transaction)
        if let Some(transaction_id) = self.transaction_id {
            target.remove_in_transaction(transaction_id, self.triple.clone());
        } else {
            target.remove(&self.triple);
        }

        #[cfg(feature = "cdc")]
        record_cdc_triple_delete(
            &self.cdc_log,
            self.triple.subject(),
            self.triple.predicate(),
            self.triple.object(),
            self.graph_name.as_deref(),
            target.graph_incarnation(),
        );

        self.deleted = true;

        // Return an empty result (DELETE doesn't produce rows)
        Ok(None)
    }

    fn reset(&mut self) {
        self.deleted = false;
    }

    fn name(&self) -> &'static str {
        "RdfDeleteTriple"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Delete Pattern Operator
// ============================================================================

/// Operator that deletes triples matching a pattern from the RDF store.
/// Used for DELETE WHERE operations where the triple template contains variables.
struct RdfDeletePatternOperator {
    store: Arc<RdfStore>,
    input: Box<dyn Operator>,
    subject: TripleComponent,
    predicate: TripleComponent,
    object: TripleComponent,
    column_map: HashMap<String, usize>,
    graph: Option<String>,
    transaction_id: Option<TransactionId>,
    done: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
    #[cfg(feature = "cdc")]
    cdc_log: Option<Arc<RdfCdcSink>>,
}

impl RdfDeletePatternOperator {
    fn new(
        store: Arc<RdfStore>,
        input: Box<dyn Operator>,
        operands: TripleOperands,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
        #[cfg(feature = "cdc")] cdc_log: Option<Arc<RdfCdcSink>>,
    ) -> Self {
        Self {
            store,
            input,
            subject: operands.subject,
            predicate: operands.predicate,
            object: operands.object,
            column_map: operands.column_map,
            graph: operands.graph,
            transaction_id: operands.transaction_id,
            done: false,
            #[cfg(feature = "wal")]
            wal,
            #[cfg(feature = "cdc")]
            cdc_log,
        }
    }
}

impl Operator for RdfDeletePatternOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }

        // Collect all triples to delete
        let mut triples_to_delete = Vec::new();

        while let Some(chunk) = self.input.next()? {
            for row in 0..chunk.row_count() {
                let subject =
                    resolve_public_pattern_component(&self.subject, &self.column_map, &chunk, row)?;
                let predicate = resolve_public_pattern_component(
                    &self.predicate,
                    &self.column_map,
                    &chunk,
                    row,
                )?;
                let object =
                    resolve_public_pattern_component(&self.object, &self.column_map, &chunk, row)?;

                if let (Some(s), Some(p), Some(o)) = (subject, predicate, object)
                    && let Some(triple) = instantiate_mutation_triple(s, p, o)
                {
                    triples_to_delete.push(triple);
                }
            }
        }

        let Some(target) =
            rdf_delete_target(&self.store, self.graph.as_deref(), self.transaction_id)
        else {
            self.done = true;
            return Ok(None);
        };

        let mut seen = grafeo_common::utils::hash::FxHashSet::default();
        let triples_to_delete: Vec<_> = triples_to_delete
            .into_iter()
            .filter_map(|triple| rdf_visible_representative(&target, self.transaction_id, &triple))
            .filter(|triple| seen.insert(Arc::clone(triple)))
            .map(|triple| triple.as_ref().clone())
            .collect();

        #[cfg(feature = "wal")]
        {
            ensure_rdf_graph_high_water(&self.wal, &self.store, self.graph.as_deref())?;
            for triple in &triples_to_delete {
                log_rdf_wal(
                    &self.wal,
                    &rdf_delete_wal_record(
                        triple,
                        self.graph.as_deref(),
                        target.graph_incarnation(),
                        self.transaction_id.unwrap_or(TransactionId::SYSTEM),
                    ),
                )?;
            }
        }
        for triple in &triples_to_delete {
            if let Some(tid) = self.transaction_id {
                target.remove_in_transaction(tid, triple.clone());
            } else {
                target.remove(triple);
            }
        }

        #[cfg(feature = "cdc")]
        for triple in &triples_to_delete {
            record_cdc_triple_delete(
                &self.cdc_log,
                triple.subject(),
                triple.predicate(),
                triple.object(),
                self.graph.as_deref(),
                target.graph_incarnation(),
            );
        }

        self.done = true;
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
        self.input.reset();
    }

    fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> std::result::Result<(), QueryResourceContextError> {
        self.input.install_resource_context(resources)
    }

    fn name(&self) -> &'static str {
        "RdfDeletePattern"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Clear Graph Operator
// ============================================================================

/// Operator that clears triples from a graph in the RDF store.
struct RdfClearGraphOperator {
    store: Arc<RdfStore>,
    graph: Option<String>,
    transaction_id: Option<TransactionId>,
    cleared: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfClearGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        graph: Option<String>,
        _silent: bool,
        transaction_id: Option<TransactionId>,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
    ) -> Self {
        Self {
            store,
            graph,
            transaction_id,
            cleared: false,
            #[cfg(feature = "wal")]
            wal,
        }
    }
}

impl Operator for RdfClearGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.cleared {
            return Ok(None);
        }

        #[cfg(feature = "wal")]
        {
            let tid = self.transaction_id.unwrap_or(TransactionId::SYSTEM);
            if self.graph.as_deref() == Some("\u{1}NAMED") {
                let named: Vec<(
                    String,
                    grafeo_common::types::GraphIncarnationId,
                    Vec<Triple>,
                )> = self
                    .store
                    .graph_names_in_transaction(self.transaction_id)
                    .into_iter()
                    .map(|name| {
                        let incarnation = self
                            .store
                            .graph_in_transaction(&name, self.transaction_id)
                            .ok_or_else(|| {
                                OperatorError::Execution(format!(
                                    "RDF graph <{name}> disappeared before WAL framing"
                                ))
                            })?
                            .graph_incarnation();
                        let triples = self
                            .store
                            .visible_in_graph(Some(&name), self.transaction_id);
                        Ok((name, incarnation, triples))
                    })
                    .collect::<std::result::Result<_, OperatorError>>()?;
                for (name, incarnation, triples) in &named {
                    log_tagged_triples(
                        &self.wal,
                        &self.store,
                        Some(name),
                        *incarnation,
                        triples,
                        &[],
                        tid,
                    )?;
                }
                for (name, _, _) in &named {
                    self.store
                        .clear_graph_in_tx(Some(name.as_str()), self.transaction_id);
                }
            } else if self.graph.as_deref() == Some("") {
                let default_deleted = self.store.visible_in_graph(None, self.transaction_id);
                let named: Vec<(
                    String,
                    grafeo_common::types::GraphIncarnationId,
                    Vec<Triple>,
                )> = self
                    .store
                    .graph_names_in_transaction(self.transaction_id)
                    .into_iter()
                    .map(|name| {
                        let incarnation = self
                            .store
                            .graph_in_transaction(&name, self.transaction_id)
                            .ok_or_else(|| {
                                OperatorError::Execution(format!(
                                    "RDF graph <{name}> disappeared before WAL framing"
                                ))
                            })?
                            .graph_incarnation();
                        let triples = self
                            .store
                            .visible_in_graph(Some(&name), self.transaction_id);
                        Ok((name, incarnation, triples))
                    })
                    .collect::<std::result::Result<_, OperatorError>>()?;
                log_tagged_triples(
                    &self.wal,
                    &self.store,
                    None,
                    grafeo_common::types::GraphIncarnationId::DEFAULT_GRAPH,
                    &default_deleted,
                    &[],
                    tid,
                )?;
                for (name, incarnation, triples) in &named {
                    log_tagged_triples(
                        &self.wal,
                        &self.store,
                        Some(name),
                        *incarnation,
                        triples,
                        &[],
                        tid,
                    )?;
                }
                self.store
                    .clear_graph_in_tx(self.graph.as_deref(), self.transaction_id);
            } else {
                let deleted = self
                    .store
                    .visible_in_graph(self.graph.as_deref(), self.transaction_id);
                if !deleted.is_empty() {
                    let target =
                        rdf_delete_target(&self.store, self.graph.as_deref(), self.transaction_id)
                            .ok_or_else(|| {
                                OperatorError::Execution(
                                    "RDF clear target disappeared before WAL framing".to_string(),
                                )
                            })?;
                    log_tagged_triples(
                        &self.wal,
                        &self.store,
                        self.graph.as_deref(),
                        target.graph_incarnation(),
                        &deleted,
                        &[],
                        tid,
                    )?;
                }
                self.store
                    .clear_graph_in_tx(self.graph.as_deref(), self.transaction_id);
            }
        }

        #[cfg(not(feature = "wal"))]
        self.store
            .clear_graph_in_tx(self.graph.as_deref(), self.transaction_id);

        self.cleared = true;

        Ok(None)
    }

    fn reset(&mut self) {
        self.cleared = false;
    }

    fn name(&self) -> &'static str {
        "RdfClearGraph"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF CREATE/DROP Graph Operators
// ============================================================================

/// Operator that creates a named graph.
struct RdfCreateGraphOperator {
    store: Arc<RdfStore>,
    graph: String,
    silent: bool,
    transaction_id: Option<TransactionId>,
    done: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfCreateGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        graph: String,
        silent: bool,
        transaction_id: Option<TransactionId>,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
    ) -> Self {
        Self {
            store,
            graph,
            silent,
            transaction_id,
            done: false,
            #[cfg(feature = "wal")]
            wal,
        }
    }
}

impl Operator for RdfCreateGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        #[cfg(feature = "wal")]
        require_graph_wal_transaction(&self.wal, self.transaction_id)?;
        if self.done {
            return Ok(None);
        }
        self.done = true;
        let created = self
            .store
            .create_graph_in_tx(&self.graph, self.transaction_id);
        if !created && !self.silent {
            return Err(OperatorError::Execution(format!(
                "Graph <{}> already exists",
                self.graph
            )));
        }
        #[cfg(feature = "wal")]
        if created && let Some(tid) = self.transaction_id {
            let graph = self
                .store
                .graph_in_transaction(&self.graph, Some(tid))
                .ok_or_else(|| {
                    OperatorError::Execution(format!(
                        "created RDF graph <{}> disappeared before WAL framing",
                        self.graph
                    ))
                })?;
            if let Some(wal) = &self.wal {
                wal.ensure_graph_high_water(&self.store)?;
            }
            log_rdf_wal(
                &self.wal,
                &grafeo_storage::wal::WalRecord::CreateNamedRdfGraphV2 {
                    name: self.graph.clone(),
                    incarnation: graph.graph_incarnation(),
                    transaction_id: tid,
                },
            )?;
        }
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "RdfCreateGraph"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Operator that drops a named graph.
struct RdfDropGraphOperator {
    store: Arc<RdfStore>,
    graph: Option<String>,
    silent: bool,
    transaction_id: Option<TransactionId>,
    done: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfDropGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        graph: Option<String>,
        silent: bool,
        transaction_id: Option<TransactionId>,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
    ) -> Self {
        Self {
            store,
            graph,
            silent,
            transaction_id,
            done: false,
            #[cfg(feature = "wal")]
            wal,
        }
    }
}

impl Operator for RdfDropGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        #[cfg(feature = "wal")]
        require_graph_wal_transaction(&self.wal, self.transaction_id)?;
        if self.done {
            return Ok(None);
        }
        self.done = true;
        match &self.graph {
            None => {
                #[cfg(feature = "wal")]
                let deleted = self.store.visible_in_graph(None, self.transaction_id);
                #[cfg(feature = "wal")]
                log_tagged_triples(
                    &self.wal,
                    &self.store,
                    None,
                    grafeo_common::types::GraphIncarnationId::DEFAULT_GRAPH,
                    &deleted,
                    &[],
                    self.transaction_id.unwrap_or(TransactionId::SYSTEM),
                )?;
                self.store.clear_graph_in_tx(None, self.transaction_id);
            }
            Some(name) if name == "\u{1}NAMED" => {
                let names = self.store.graph_names_in_transaction(self.transaction_id);
                #[cfg(feature = "wal")]
                let named: Vec<(
                    String,
                    grafeo_common::types::GraphIncarnationId,
                    Vec<Triple>,
                )> = names
                    .iter()
                    .map(|n| {
                        Ok((
                            n.clone(),
                            self.store
                                .graph_in_transaction(n, self.transaction_id)
                                .ok_or_else(|| {
                                    OperatorError::Execution(format!(
                                        "RDF graph <{n}> disappeared before WAL framing"
                                    ))
                                })?
                                .graph_incarnation(),
                            self.store.visible_in_graph(Some(n), self.transaction_id),
                        ))
                    })
                    .collect::<std::result::Result<_, OperatorError>>()?;
                #[cfg(feature = "wal")]
                {
                    let tid = self.transaction_id.unwrap_or(TransactionId::SYSTEM);
                    for (n, incarnation, triples) in named {
                        log_tagged_triples(
                            &self.wal,
                            &self.store,
                            Some(&n),
                            incarnation,
                            &triples,
                            &[],
                            tid,
                        )?;
                        if self.transaction_id.is_some() {
                            log_rdf_wal(
                                &self.wal,
                                &grafeo_storage::wal::WalRecord::DropNamedRdfGraphV2 {
                                    name: n,
                                    incarnation,
                                    transaction_id: tid,
                                },
                            )?;
                        }
                    }
                }
                for n in &names {
                    let _ = self.store.drop_graph_in_tx(n, self.transaction_id);
                }
            }
            Some(name) if name.is_empty() => {
                #[cfg(feature = "wal")]
                let default_deleted = self.store.visible_in_graph(None, self.transaction_id);
                let names = self.store.graph_names_in_transaction(self.transaction_id);
                #[cfg(feature = "wal")]
                let named: Vec<(
                    String,
                    grafeo_common::types::GraphIncarnationId,
                    Vec<Triple>,
                )> = names
                    .iter()
                    .map(|n| {
                        Ok((
                            n.clone(),
                            self.store
                                .graph_in_transaction(n, self.transaction_id)
                                .ok_or_else(|| {
                                    OperatorError::Execution(format!(
                                        "RDF graph <{n}> disappeared before WAL framing"
                                    ))
                                })?
                                .graph_incarnation(),
                            self.store.visible_in_graph(Some(n), self.transaction_id),
                        ))
                    })
                    .collect::<std::result::Result<_, OperatorError>>()?;
                #[cfg(feature = "wal")]
                {
                    let tid = self.transaction_id.unwrap_or(TransactionId::SYSTEM);
                    log_tagged_triples(
                        &self.wal,
                        &self.store,
                        None,
                        grafeo_common::types::GraphIncarnationId::DEFAULT_GRAPH,
                        &default_deleted,
                        &[],
                        tid,
                    )?;
                    for (n, incarnation, triples) in named {
                        log_tagged_triples(
                            &self.wal,
                            &self.store,
                            Some(&n),
                            incarnation,
                            &triples,
                            &[],
                            tid,
                        )?;
                        if self.transaction_id.is_some() {
                            log_rdf_wal(
                                &self.wal,
                                &grafeo_storage::wal::WalRecord::DropNamedRdfGraphV2 {
                                    name: n,
                                    incarnation,
                                    transaction_id: tid,
                                },
                            )?;
                        }
                    }
                }
                self.store.clear_graph_in_tx(None, self.transaction_id);
                for n in &names {
                    let _ = self.store.drop_graph_in_tx(n, self.transaction_id);
                }
            }
            Some(name) => {
                if self
                    .store
                    .graph_in_transaction(name, self.transaction_id)
                    .is_none()
                {
                    if !self.silent {
                        return Err(OperatorError::Execution(format!(
                            "Graph <{name}> does not exist"
                        )));
                    }
                    return Ok(None);
                }
                #[cfg(feature = "wal")]
                let deleted = self.store.visible_in_graph(Some(name), self.transaction_id);
                #[cfg(feature = "wal")]
                let dropped_incarnation = self
                    .store
                    .graph_in_transaction(name, self.transaction_id)
                    .ok_or_else(|| {
                        OperatorError::Execution(format!(
                            "RDF graph <{name}> disappeared before WAL framing"
                        ))
                    })?
                    .graph_incarnation();
                #[cfg(feature = "wal")]
                {
                    let tid = self.transaction_id.unwrap_or(TransactionId::SYSTEM);
                    log_tagged_triples(
                        &self.wal,
                        &self.store,
                        Some(name),
                        dropped_incarnation,
                        &deleted,
                        &[],
                        tid,
                    )?;
                    if self.transaction_id.is_some() {
                        let record = grafeo_storage::wal::WalRecord::DropNamedRdfGraphV2 {
                            name: name.clone(),
                            incarnation: dropped_incarnation,
                            transaction_id: tid,
                        };
                        log_rdf_wal(&self.wal, &record)?;
                    }
                }
                let dropped = self.store.drop_graph_in_tx(name, self.transaction_id);
                if !dropped && !self.silent {
                    return Err(OperatorError::Execution(format!(
                        "Graph <{name}> does not exist"
                    )));
                }
            }
        }
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "RdfDropGraph"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF COPY/MOVE/ADD Graph Operators
// ============================================================================

/// Operator that copies all triples from one graph to another.
struct RdfCopyGraphOperator {
    store: Arc<RdfStore>,
    source: Option<String>,
    destination: Option<String>,
    silent: bool,
    transaction_id: Option<TransactionId>,
    done: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfCopyGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        source: Option<String>,
        destination: Option<String>,
        silent: bool,
        transaction_id: Option<TransactionId>,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
    ) -> Self {
        Self {
            store,
            source,
            destination,
            silent,
            transaction_id,
            done: false,
            #[cfg(feature = "wal")]
            wal,
        }
    }
}

impl Operator for RdfCopyGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        #[cfg(feature = "wal")]
        require_graph_wal_transaction(&self.wal, self.transaction_id)?;
        if self.done {
            return Ok(None);
        }
        self.done = true;

        // A missing source is an error with no side effects. SILENT suppresses
        // that error; it does not turn the missing graph into an empty source.
        if let Some(ref name) = self.source
            && self
                .store
                .graph_in_transaction(name, self.transaction_id)
                .is_none()
        {
            if self.silent {
                return Ok(None);
            }
            return Err(OperatorError::Execution(format!(
                "Source graph <{name}> does not exist"
            )));
        }

        let destination_created = ensure_graph_operation_destination(
            &self.store,
            self.destination.as_deref(),
            self.transaction_id,
        )?;
        #[cfg(not(feature = "wal"))]
        let _ = destination_created;
        #[cfg(feature = "wal")]
        log_graph_operation_destination_create(
            &self.wal,
            &self.store,
            self.destination.as_deref(),
            self.transaction_id,
            destination_created,
        )?;

        #[cfg(feature = "wal")]
        let dest_old = self
            .store
            .visible_in_graph(self.destination.as_deref(), self.transaction_id);
        #[cfg(feature = "wal")]
        let src = self
            .store
            .visible_with_valid_in_graph(self.source.as_deref(), self.transaction_id);
        #[cfg(feature = "wal")]
        let destination_incarnation = active_graph_incarnation(
            &self.store,
            self.destination.as_deref(),
            self.transaction_id,
        )?;
        #[cfg(feature = "wal")]
        log_tagged_triples(
            &self.wal,
            &self.store,
            self.destination.as_deref(),
            destination_incarnation,
            &dest_old,
            &src,
            self.transaction_id.unwrap_or(TransactionId::SYSTEM),
        )?;
        self.store.copy_graph_in_tx(
            self.source.as_deref(),
            self.destination.as_deref(),
            self.transaction_id,
        );
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "RdfCopyGraph"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Operator that moves all triples from one graph to another.
struct RdfMoveGraphOperator {
    store: Arc<RdfStore>,
    source: Option<String>,
    destination: Option<String>,
    silent: bool,
    transaction_id: Option<TransactionId>,
    done: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfMoveGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        source: Option<String>,
        destination: Option<String>,
        silent: bool,
        transaction_id: Option<TransactionId>,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
    ) -> Self {
        Self {
            store,
            source,
            destination,
            silent,
            transaction_id,
            done: false,
            #[cfg(feature = "wal")]
            wal,
        }
    }
}

impl Operator for RdfMoveGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        #[cfg(feature = "wal")]
        require_graph_wal_transaction(&self.wal, self.transaction_id)?;
        if self.done {
            return Ok(None);
        }
        self.done = true;

        // A missing source is an error with no side effects. SILENT suppresses
        // that error; it does not turn the missing graph into an empty source.
        if let Some(ref name) = self.source
            && self
                .store
                .graph_in_transaction(name, self.transaction_id)
                .is_none()
        {
            if self.silent {
                return Ok(None);
            }
            return Err(OperatorError::Execution(format!(
                "Source graph <{name}> does not exist"
            )));
        }

        let destination_created = ensure_graph_operation_destination(
            &self.store,
            self.destination.as_deref(),
            self.transaction_id,
        )?;
        #[cfg(not(feature = "wal"))]
        let _ = destination_created;
        #[cfg(feature = "wal")]
        log_graph_operation_destination_create(
            &self.wal,
            &self.store,
            self.destination.as_deref(),
            self.transaction_id,
            destination_created,
        )?;

        #[cfg(feature = "wal")]
        let dest_old = self
            .store
            .visible_in_graph(self.destination.as_deref(), self.transaction_id);
        #[cfg(feature = "wal")]
        let src = self
            .store
            .visible_with_valid_in_graph(self.source.as_deref(), self.transaction_id);
        #[cfg(feature = "wal")]
        let src_triples: Vec<_> = src.iter().map(|(triple, _)| triple.clone()).collect();
        #[cfg(feature = "wal")]
        let destination_incarnation = active_graph_incarnation(
            &self.store,
            self.destination.as_deref(),
            self.transaction_id,
        )?;
        #[cfg(feature = "wal")]
        let source_incarnation = match self.source.as_deref() {
            Some(name) => Some(
                self.store
                    .graph_in_transaction(name, self.transaction_id)
                    .ok_or_else(|| {
                        OperatorError::Execution(format!(
                            "RDF graph <{name}> disappeared before WAL framing"
                        ))
                    })?
                    .graph_incarnation(),
            ),
            None => None,
        };
        #[cfg(feature = "wal")]
        {
            let tid = self.transaction_id.unwrap_or(TransactionId::SYSTEM);
            log_tagged_triples(
                &self.wal,
                &self.store,
                self.destination.as_deref(),
                destination_incarnation,
                &dest_old,
                &src,
                tid,
            )?;
            log_tagged_triples(
                &self.wal,
                &self.store,
                self.source.as_deref(),
                source_incarnation
                    .unwrap_or(grafeo_common::types::GraphIncarnationId::DEFAULT_GRAPH),
                &src_triples,
                &[],
                tid,
            )?;
            if let Some(name) = &self.source
                && self.transaction_id.is_some()
            {
                log_rdf_wal(
                    &self.wal,
                    &grafeo_storage::wal::WalRecord::DropNamedRdfGraphV2 {
                        name: name.clone(),
                        incarnation: source_incarnation.ok_or_else(|| {
                            OperatorError::Execution(
                                "missing RDF source incarnation during MOVE framing".to_string(),
                            )
                        })?,
                        transaction_id: tid,
                    },
                )?;
            }
        }
        self.store.move_graph_in_tx(
            self.source.as_deref(),
            self.destination.as_deref(),
            self.transaction_id,
        );
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "RdfMoveGraph"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Operator that adds (merges) all triples from one graph into another.
struct RdfAddGraphOperator {
    store: Arc<RdfStore>,
    source: Option<String>,
    destination: Option<String>,
    silent: bool,
    transaction_id: Option<TransactionId>,
    done: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfAddGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        source: Option<String>,
        destination: Option<String>,
        silent: bool,
        transaction_id: Option<TransactionId>,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
    ) -> Self {
        Self {
            store,
            source,
            destination,
            silent,
            transaction_id,
            done: false,
            #[cfg(feature = "wal")]
            wal,
        }
    }
}

impl Operator for RdfAddGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        #[cfg(feature = "wal")]
        require_graph_wal_transaction(&self.wal, self.transaction_id)?;
        if self.done {
            return Ok(None);
        }
        self.done = true;

        // A missing source is an error with no side effects. SILENT suppresses
        // that error; it does not turn the missing graph into an empty source.
        if let Some(ref name) = self.source
            && self
                .store
                .graph_in_transaction(name, self.transaction_id)
                .is_none()
        {
            if self.silent {
                return Ok(None);
            }
            return Err(OperatorError::Execution(format!(
                "Source graph <{name}> does not exist"
            )));
        }

        let destination_created = ensure_graph_operation_destination(
            &self.store,
            self.destination.as_deref(),
            self.transaction_id,
        )?;
        #[cfg(not(feature = "wal"))]
        let _ = destination_created;
        #[cfg(feature = "wal")]
        log_graph_operation_destination_create(
            &self.wal,
            &self.store,
            self.destination.as_deref(),
            self.transaction_id,
            destination_created,
        )?;

        #[cfg(feature = "wal")]
        let src = {
            let present: grafeo_common::utils::hash::FxHashSet<_> = self
                .store
                .visible_in_graph(self.destination.as_deref(), self.transaction_id)
                .iter()
                .map(Triple::canonical_identity_key)
                .collect();
            self.store
                .visible_with_valid_in_graph(self.source.as_deref(), self.transaction_id)
                .into_iter()
                .filter(|(triple, _)| !present.contains(&triple.canonical_identity_key()))
                .collect::<Vec<_>>()
        };
        #[cfg(feature = "wal")]
        let destination_incarnation = active_graph_incarnation(
            &self.store,
            self.destination.as_deref(),
            self.transaction_id,
        )?;
        #[cfg(feature = "wal")]
        log_tagged_triples(
            &self.wal,
            &self.store,
            self.destination.as_deref(),
            destination_incarnation,
            &[],
            &src,
            self.transaction_id.unwrap_or(TransactionId::SYSTEM),
        )?;
        self.store.add_graph_in_tx(
            self.source.as_deref(),
            self.destination.as_deref(),
            self.transaction_id,
        );
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "RdfAddGraph"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Modify Operator (SPARQL DELETE/INSERT WHERE)
// ============================================================================

/// Transactional services used while applying a SPARQL MODIFY operation.
struct RdfModifyContext {
    transaction_id: Option<TransactionId>,
    valid_time: Option<ValidTimeInterval>,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
    #[cfg(feature = "cdc")]
    cdc_log: Option<Arc<RdfCdcSink>>,
}

/// Operator that handles SPARQL MODIFY operations (DELETE/INSERT WHERE).
///
/// Per SPARQL 1.1 Update spec:
/// 1. Evaluate WHERE clause once to get all bindings
/// 2. Apply DELETE templates to each binding
/// 3. Apply INSERT templates to each binding (using SAME bindings)
struct RdfModifyOperator {
    store: Arc<RdfStore>,
    input: Box<dyn Operator>,
    delete_templates: Vec<TripleTemplate>,
    insert_templates: Vec<TripleTemplate>,
    column_map: HashMap<String, usize>,
    sealed_identity: bool,
    done: bool,
    transaction_id: Option<TransactionId>,
    valid_time: Option<ValidTimeInterval>,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
    #[cfg(feature = "cdc")]
    cdc_log: Option<Arc<RdfCdcSink>>,
}

impl RdfModifyOperator {
    fn new(
        store: Arc<RdfStore>,
        input: Box<dyn Operator>,
        delete_templates: Vec<TripleTemplate>,
        insert_templates: Vec<TripleTemplate>,
        column_map: HashMap<String, usize>,
        sealed_identity: bool,
        context: RdfModifyContext,
    ) -> Self {
        Self {
            store,
            input,
            delete_templates,
            insert_templates,
            column_map,
            sealed_identity,
            done: false,
            transaction_id: context.transaction_id,
            valid_time: context.valid_time,
            #[cfg(feature = "wal")]
            wal: context.wal,
            #[cfg(feature = "cdc")]
            cdc_log: context.cdc_log,
        }
    }

    fn insert_target(
        &self,
        graph: Option<&str>,
    ) -> std::result::Result<Arc<RdfStore>, OperatorError> {
        match graph {
            Some(name) => self
                .store
                .graph_or_create_in_tx(name, self.transaction_id)
                .map_err(|error| OperatorError::Execution(error.to_string())),
            None => Ok(Arc::clone(&self.store)),
        }
    }

    fn insert_triple(
        &self,
        graph: Option<&str>,
        triple: Triple,
    ) -> std::result::Result<bool, OperatorError> {
        let target = self.insert_target(graph)?;
        let already_visible = !target
            .find_with_pending(
                &TriplePattern {
                    subject: Some(triple.subject().clone()),
                    predicate: Some(triple.predicate().clone()),
                    object: Some(triple.object().clone()),
                },
                self.transaction_id,
            )
            .is_empty();
        if already_visible {
            return Ok(false);
        }
        #[cfg(feature = "wal")]
        {
            ensure_rdf_graph_high_water(&self.wal, &self.store, graph)?;
            log_rdf_wal(
                &self.wal,
                &rdf_insert_wal_record(
                    &triple,
                    graph,
                    target.graph_incarnation(),
                    self.transaction_id.unwrap_or(TransactionId::SYSTEM),
                    self.valid_time,
                ),
            )?;
        }
        if let Some(tid) = self.transaction_id {
            target.insert_in_transaction_with_valid(tid, triple.clone(), self.valid_time);
        } else {
            target
                .try_insert_at_epoch_with_valid(
                    triple.clone(),
                    target.commit_epoch(),
                    self.valid_time,
                )
                .map_err(|error| OperatorError::Execution(error.to_string()))?;
        }
        #[cfg(feature = "cdc")]
        record_cdc_triple_insert(
            &self.cdc_log,
            triple.subject(),
            triple.predicate(),
            triple.object(),
            graph,
            target.graph_incarnation(),
        );
        Ok(true)
    }

    fn delete_triple(
        &self,
        graph: Option<&str>,
        triple: Triple,
    ) -> std::result::Result<Option<Arc<Triple>>, OperatorError> {
        let Some(target) = rdf_delete_target(&self.store, graph, self.transaction_id) else {
            return Ok(None);
        };
        let Some(triple) = rdf_visible_representative(&target, self.transaction_id, &triple) else {
            return Ok(None);
        };
        #[cfg(feature = "wal")]
        {
            ensure_rdf_graph_high_water(&self.wal, &self.store, graph)?;
            log_rdf_wal(
                &self.wal,
                &rdf_delete_wal_record(
                    &triple,
                    graph,
                    target.graph_incarnation(),
                    self.transaction_id.unwrap_or(TransactionId::SYSTEM),
                ),
            )?;
        }
        if let Some(tid) = self.transaction_id {
            target.remove_in_transaction(tid, triple.as_ref().clone());
        } else {
            target.remove(&triple);
        }
        #[cfg(feature = "cdc")]
        record_cdc_triple_delete(
            &self.cdc_log,
            triple.subject(),
            triple.predicate(),
            triple.object(),
            graph,
            target.graph_incarnation(),
        );
        Ok(Some(triple))
    }

    /// Resolves every fallible template substitution before the first write.
    /// This gives an explicit transaction statement atomicity for semantic
    /// failures: callers cannot catch a late bad binding and commit an earlier
    /// subset of the same MODIFY.
    fn materialize_templates(
        &self,
        templates: &[TripleTemplate],
        binding_chunks: &[DataChunk],
        blank_execution_scope: Option<&str>,
    ) -> std::result::Result<Vec<(Option<String>, Triple)>, OperatorError> {
        let mut materialized = Vec::new();
        let mut solution_id = 0usize;
        for chunk in binding_chunks {
            for row in chunk.selected_indices() {
                for template in templates {
                    let blank_scope = blank_execution_scope.map(|scope| (scope, solution_id));
                    let subject = resolve_mutation_template_component(
                        &template.subject,
                        &self.column_map,
                        chunk,
                        row,
                        blank_scope,
                        self.sealed_identity,
                    )?;
                    let predicate = resolve_mutation_template_component(
                        &template.predicate,
                        &self.column_map,
                        chunk,
                        row,
                        blank_scope,
                        self.sealed_identity,
                    )?;
                    let object = resolve_mutation_template_component(
                        &template.object,
                        &self.column_map,
                        chunk,
                        row,
                        blank_scope,
                        self.sealed_identity,
                    )?;
                    let (Some(subject), Some(predicate), Some(object)) =
                        (subject, predicate, object)
                    else {
                        continue;
                    };
                    let graph_name = match &template.graph {
                        None => None,
                        Some(graph_template) => {
                            let Some(graph) = resolve_mutation_graph(
                                graph_template,
                                &self.column_map,
                                chunk,
                                row,
                                self.sealed_identity,
                            )?
                            else {
                                continue;
                            };
                            Some(graph)
                        }
                    };
                    if let Some(triple) = instantiate_mutation_triple(subject, predicate, object) {
                        materialized.push((graph_name, triple));
                    }
                }
                solution_id += 1;
            }
        }
        Ok(materialized)
    }
}

impl Operator for RdfModifyOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }

        // Step 1: Retain each WHERE chunk exactly once before any modifications.
        // Selection vectors contain physical row indexes, so consumers must use
        // selected_indices() rather than treating row_count() as a dense range.
        let mut binding_chunks = Vec::new();
        while let Some(mut chunk) = self.input.next()? {
            chunk.flatten();
            binding_chunks.push(chunk);
        }
        let delete_candidates =
            self.materialize_templates(&self.delete_templates, &binding_chunks, None)?;
        use std::sync::atomic::{AtomicU64, Ordering};
        static MODIFY_BLANK_SCOPE: AtomicU64 = AtomicU64::new(0);
        let execution_scope = self.sealed_identity.then(|| {
            format!(
                "grafeo{}_e{}_x{}",
                self.store.store_id(),
                self.store.commit_epoch().as_u64(),
                MODIFY_BLANK_SCOPE.fetch_add(1, Ordering::Relaxed),
            )
        });
        let insert_candidates = self.materialize_templates(
            &self.insert_templates,
            &binding_chunks,
            execution_scope.as_deref(),
        )?;

        // The sealed store may defer publication until Session commit, so it is
        // not itself a reliable read-after-write cache while this physical
        // operator is framing a statement. Track the statement's logical set
        // image explicitly to deduplicate solutions and later templates.
        let mut presence: HashMap<(Option<String>, [String; 3]), bool> = HashMap::new();

        // Step 2: Apply DELETE templates using exact bound RDF terms.
        for (graph_name, triple) in delete_candidates {
            let graph = graph_name.as_deref();
            let key = (graph_name.clone(), triple.canonical_identity_key());
            let exact = *presence.entry(key.clone()).or_insert_with(|| {
                rdf_triple_visible(&self.store, graph, self.transaction_id, &triple)
            });
            if !exact {
                continue;
            }
            let Some(_deleted_triple) = self.delete_triple(graph, triple.clone())? else {
                continue;
            };
            presence.insert(key, false);
        }

        // Step 3: Apply INSERT templates using the SAME retained bindings.
        for (graph_name, triple) in insert_candidates {
            let graph = graph_name.as_deref();
            let key = (graph_name.clone(), triple.canonical_identity_key());
            let already_visible = *presence.entry(key.clone()).or_insert_with(|| {
                rdf_triple_visible(&self.store, graph, self.transaction_id, &triple)
            });
            if already_visible {
                continue;
            }
            if !self.insert_triple(graph, triple.clone())? {
                continue;
            }
            presence.insert(key, true);
        }

        self.done = true;
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
        self.input.reset();
    }

    fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> std::result::Result<(), QueryResourceContextError> {
        self.input.install_resource_context(resources)
    }

    fn name(&self) -> &'static str {
        "RdfModify"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF SPARQL Compatibility Join Operator
// ============================================================================

/// One declared shared-variable comparison at a SPARQL solution-mapping
/// boundary. Boundness is determined from the visible column; RDF term
/// identity is read from the canonical identity-key companion when required.
#[derive(Debug, Clone)]
struct RdfCompatibilityKey {
    left_visible: usize,
    right_visible: usize,
    left_group_key: Option<usize>,
    right_group_key: Option<usize>,
    left_identity: Option<usize>,
    right_identity: Option<usize>,
    semantics: JoinKeySemantics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RdfCompatibilityMode {
    Inner,
    Left,
    Semi,
    Anti { require_bound_overlap: bool },
}

#[derive(Debug, Clone)]
struct RdfCompatibilityRow {
    values: Vec<Value>,
    bound: Vec<bool>,
    keys: Vec<Option<HashableValue>>,
}

#[derive(Debug, Clone, Copy)]
enum RdfCompatibilityOutputColumn {
    Left(usize),
    Right(usize),
    Coalesce {
        left: usize,
        right: usize,
    },
    NormalizedIdentity {
        left_visible: usize,
        right_visible: usize,
        left_group_key: Option<usize>,
        right_group_key: Option<usize>,
        left_identity: Option<usize>,
        right_identity: Option<usize>,
    },
}

#[derive(Debug)]
struct RdfCompatibilityShapeIndex {
    bound: Vec<bool>,
    rows: Vec<usize>,
    /// One canonical-key hash table for each intersection shape induced by the left
    /// input. Query width and binding shapes are fixed-query constants, so
    /// this keeps data complexity linear in input rows plus emitted matches.
    indexes: HashMap<Vec<bool>, HashMap<Vec<HashableValue>, Vec<usize>>>,
}

#[derive(Debug, Default, Clone, Copy)]
struct RdfCompatibilityWork {
    left_rows: usize,
    right_rows: usize,
    indexed_rows: usize,
    lookups: usize,
    emitted_pairs: usize,
}

/// Implements SPARQL solution-mapping compatibility without cloning either
/// input. UNDEF is a wildcard, bound/bound RDF keys compare canonical term
/// identity, and shared output columns are coalesced from the side that binds
/// them. Inputs and output retain bag semantics.
struct RdfCompatibilityJoinOperator {
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    keys: Vec<RdfCompatibilityKey>,
    mode: RdfCompatibilityMode,
    output_columns: Vec<RdfCompatibilityOutputColumn>,
    output_types: Vec<LogicalType>,
    output_rows: Option<Vec<Vec<Value>>>,
    position: usize,
    work: RdfCompatibilityWork,
}

impl RdfCompatibilityJoinOperator {
    fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        keys: Vec<RdfCompatibilityKey>,
        mode: RdfCompatibilityMode,
        output_columns: Vec<RdfCompatibilityOutputColumn>,
        output_types: Vec<LogicalType>,
    ) -> Self {
        Self {
            left,
            right,
            keys,
            mode,
            output_columns,
            output_types,
            output_rows: None,
            position: 0,
            work: RdfCompatibilityWork::default(),
        }
    }

    fn materialize(
        operator: &mut dyn Operator,
    ) -> std::result::Result<Vec<Vec<Value>>, OperatorError> {
        let mut rows = Vec::new();
        while let Some(chunk) = operator.next()? {
            for row in chunk.selected_indices() {
                let mut values = Vec::with_capacity(chunk.column_count());
                for column in 0..chunk.column_count() {
                    values.push(
                        chunk
                            .column(column)
                            .and_then(|values| values.get_value(row))
                            .unwrap_or(Value::Null),
                    );
                }
                rows.push(values);
            }
        }
        Ok(rows)
    }

    fn annotate_rows(
        rows: Vec<Vec<Value>>,
        keys: &[RdfCompatibilityKey],
        left: bool,
    ) -> std::result::Result<Vec<RdfCompatibilityRow>, OperatorError> {
        rows.into_iter()
            .map(|values| {
                let mut bound = Vec::with_capacity(keys.len());
                let mut row_keys = Vec::with_capacity(keys.len());
                for key in keys {
                    let visible_index = if left {
                        key.left_visible
                    } else {
                        key.right_visible
                    };
                    let visible = values.get(visible_index).ok_or_else(|| {
                        OperatorError::Execution(format!(
                            "RDF compatibility visible column {visible_index} was absent"
                        ))
                    })?;
                    let is_bound = !visible.is_null();
                    bound.push(is_bound);
                    if !is_bound {
                        row_keys.push(None);
                        continue;
                    }

                    let comparison = match key.semantics {
                        JoinKeySemantics::Value => visible.clone(),
                        JoinKeySemantics::RdfTermIdentity
                        | JoinKeySemantics::SparqlCompatibility => {
                            let group_key = if left {
                                key.left_group_key
                            } else {
                                key.right_group_key
                            }
                            .and_then(|group_key_index| values.get(group_key_index));
                            let identity = if left {
                                key.left_identity
                            } else {
                                key.right_identity
                            }
                            .and_then(|identity_index| values.get(identity_index));
                            normalized_rdf_or_native_key(visible, group_key, identity)
                        }
                    };
                    row_keys.push(Some(HashableValue::new(comparison)));
                }
                Ok(RdfCompatibilityRow {
                    values,
                    bound,
                    keys: row_keys,
                })
            })
            .collect()
    }

    fn comparison_mask(
        left: &[bool],
        right: &[bool],
        keys: &[RdfCompatibilityKey],
    ) -> Option<Vec<bool>> {
        let mut mask = Vec::with_capacity(keys.len());
        for ((left_bound, right_bound), key) in left.iter().zip(right).zip(keys) {
            match key.semantics {
                JoinKeySemantics::SparqlCompatibility => {
                    mask.push(*left_bound && *right_bound);
                }
                JoinKeySemantics::Value | JoinKeySemantics::RdfTermIdentity => {
                    if !(*left_bound && *right_bound) {
                        return None;
                    }
                    mask.push(true);
                }
            }
        }
        Some(mask)
    }

    fn projected_key(
        row: &RdfCompatibilityRow,
        mask: &[bool],
    ) -> std::result::Result<Vec<HashableValue>, OperatorError> {
        row.keys
            .iter()
            .zip(mask)
            .filter_map(|(value, selected)| selected.then_some(value))
            .map(|value| {
                value.clone().ok_or_else(|| {
                    OperatorError::Execution(
                        "bound RDF compatibility comparison lacked a canonical RDF-or-native identity key"
                            .to_string(),
                    )
                })
            })
            .collect()
    }

    fn build_shape_indexes(
        left_rows: &[RdfCompatibilityRow],
        right_rows: &[RdfCompatibilityRow],
        keys: &[RdfCompatibilityKey],
        work: &mut RdfCompatibilityWork,
    ) -> std::result::Result<Vec<RdfCompatibilityShapeIndex>, OperatorError> {
        let left_shapes = left_rows
            .iter()
            .map(|row| row.bound.clone())
            .collect::<HashSet<_>>();
        let mut shape_positions = HashMap::<Vec<bool>, usize>::new();
        let mut shapes: Vec<RdfCompatibilityShapeIndex> = Vec::new();
        for (row_index, row) in right_rows.iter().enumerate() {
            let position = if let Some(position) = shape_positions.get(&row.bound) {
                *position
            } else {
                let position = shapes.len();
                shape_positions.insert(row.bound.clone(), position);
                shapes.push(RdfCompatibilityShapeIndex {
                    bound: row.bound.clone(),
                    rows: Vec::new(),
                    indexes: HashMap::new(),
                });
                position
            };
            shapes[position].rows.push(row_index);
        }

        for shape in &mut shapes {
            let intersections = left_shapes
                .iter()
                .filter_map(|left| Self::comparison_mask(left, &shape.bound, keys))
                .collect::<HashSet<_>>();
            for intersection in intersections {
                let mut index: HashMap<Vec<HashableValue>, Vec<usize>> = HashMap::new();
                for row_index in &shape.rows {
                    let key = Self::projected_key(&right_rows[*row_index], &intersection)?;
                    index.entry(key).or_default().push(*row_index);
                    work.indexed_rows += 1;
                }
                shape.indexes.insert(intersection, index);
            }
        }
        Ok(shapes)
    }

    fn matching_right_rows(
        left: &RdfCompatibilityRow,
        shapes: &[RdfCompatibilityShapeIndex],
        keys: &[RdfCompatibilityKey],
        require_bound_overlap: bool,
    ) -> std::result::Result<(Vec<usize>, usize), OperatorError> {
        let mut lists: Vec<&[usize]> = Vec::new();
        let mut lookups = 0;
        for shape in shapes {
            let Some(intersection) = Self::comparison_mask(&left.bound, &shape.bound, keys) else {
                continue;
            };
            if require_bound_overlap && !intersection.iter().any(|bound| *bound) {
                continue;
            }
            lookups += 1;
            let key = Self::projected_key(left, &intersection)?;
            if let Some(rows) = shape
                .indexes
                .get(&intersection)
                .and_then(|index| index.get(&key))
            {
                lists.push(rows);
            }
        }

        // Each right row belongs to exactly one binding shape. Merge the
        // shape-local, input-ordered lists so output follows right input order
        // without scanning mismatches or sorting every emitted pair.
        let mut heap = BinaryHeap::new();
        for (list_index, rows) in lists.iter().enumerate() {
            if let Some(row) = rows.first() {
                heap.push(Reverse((*row, list_index, 0usize)));
            }
        }
        let mut matches = Vec::new();
        while let Some(Reverse((row, list_index, position))) = heap.pop() {
            matches.push(row);
            let next = position + 1;
            if let Some(row) = lists[list_index].get(next) {
                heap.push(Reverse((*row, list_index, next)));
            }
        }
        Ok((matches, lookups))
    }

    fn output_row(
        &self,
        left: &RdfCompatibilityRow,
        right: Option<&RdfCompatibilityRow>,
    ) -> Vec<Value> {
        self.output_columns
            .iter()
            .map(|column| match *column {
                RdfCompatibilityOutputColumn::Left(index) => left.values[index].clone(),
                RdfCompatibilityOutputColumn::Right(index) => {
                    right.map_or(Value::Null, |row| row.values[index].clone())
                }
                RdfCompatibilityOutputColumn::Coalesce { left: l, right: r } => {
                    let left_value = &left.values[l];
                    if left_value.is_null() {
                        right.map_or(Value::Null, |row| row.values[r].clone())
                    } else {
                        left_value.clone()
                    }
                }
                RdfCompatibilityOutputColumn::NormalizedIdentity {
                    left_visible,
                    right_visible,
                    left_group_key,
                    right_group_key,
                    left_identity,
                    right_identity,
                } => {
                    let left_value = &left.values[left_visible];
                    if !left_value.is_null() {
                        normalized_rdf_or_native_key(
                            left_value,
                            left_group_key.and_then(|index| left.values.get(index)),
                            left_identity.and_then(|index| left.values.get(index)),
                        )
                    } else if let Some(right) = right {
                        let right_value = &right.values[right_visible];
                        normalized_rdf_or_native_key(
                            right_value,
                            right_group_key.and_then(|index| right.values.get(index)),
                            right_identity.and_then(|index| right.values.get(index)),
                        )
                    } else {
                        Value::Null
                    }
                }
            })
            .collect()
    }

    fn prepare(&mut self) -> std::result::Result<(), OperatorError> {
        let left_values = Self::materialize(self.left.as_mut())?;
        let right_values = Self::materialize(self.right.as_mut())?;
        self.work.left_rows = left_values.len();
        self.work.right_rows = right_values.len();
        let left_rows = Self::annotate_rows(left_values, &self.keys, true)?;
        let right_rows = Self::annotate_rows(right_values, &self.keys, false)?;
        let shapes =
            Self::build_shape_indexes(&left_rows, &right_rows, &self.keys, &mut self.work)?;

        let mut output = Vec::new();
        for left in &left_rows {
            let require_bound_overlap = matches!(
                self.mode,
                RdfCompatibilityMode::Anti {
                    require_bound_overlap: true
                }
            );
            let (matches, lookups) =
                Self::matching_right_rows(left, &shapes, &self.keys, require_bound_overlap)?;
            self.work.lookups += lookups;
            match self.mode {
                RdfCompatibilityMode::Inner => {
                    for right in matches {
                        output.push(self.output_row(left, Some(&right_rows[right])));
                        self.work.emitted_pairs += 1;
                    }
                }
                RdfCompatibilityMode::Left => {
                    if matches.is_empty() {
                        output.push(self.output_row(left, None));
                    } else {
                        for right in matches {
                            output.push(self.output_row(left, Some(&right_rows[right])));
                            self.work.emitted_pairs += 1;
                        }
                    }
                }
                RdfCompatibilityMode::Semi => {
                    if !matches.is_empty() {
                        output.push(self.output_row(left, None));
                    }
                }
                RdfCompatibilityMode::Anti { .. } => {
                    if matches.is_empty() {
                        output.push(self.output_row(left, None));
                    }
                }
            }
        }
        self.output_rows = Some(output);
        Ok(())
    }
}

impl Operator for RdfCompatibilityJoinOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.output_rows.is_none() {
            self.prepare()?;
        }
        let rows = self.output_rows.as_ref().ok_or_else(|| {
            OperatorError::Execution("RDF compatibility join has no prepared rows".to_string())
        })?;
        if self.position >= rows.len() {
            return Ok(None);
        }
        let end = (self.position + DEFAULT_CHUNK_SIZE).min(rows.len());
        let mut chunk = DataChunk::with_capacity(&self.output_types, end - self.position);
        for row in &rows[self.position..end] {
            if row.len() != self.output_types.len() {
                return Err(OperatorError::Execution(
                    "RDF compatibility join row does not match its output schema".to_string(),
                ));
            }
            for (column, value) in row.iter().enumerate() {
                chunk
                    .column_mut(column)
                    .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?
                    .push_value(value.clone());
            }
        }
        chunk.set_count(end - self.position);
        self.position = end;
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
        self.output_rows = None;
        self.position = 0;
        self.work = RdfCompatibilityWork::default();
    }

    fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> std::result::Result<(), QueryResourceContextError> {
        self.left.install_resource_context(resources)?;
        self.right.install_resource_context(resources)
    }

    fn name(&self) -> &'static str {
        "RdfCompatibilityJoin"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Union Operator
// ============================================================================

/// Operator that executes multiple operators in sequence.
/// Used for UNION of INSERT operations.
struct RdfUnionOperator {
    operators: Vec<Box<dyn Operator>>,
    current_idx: usize,
}

impl RdfUnionOperator {
    fn new(operators: Vec<Box<dyn Operator>>) -> Self {
        Self {
            operators,
            current_idx: 0,
        }
    }
}

impl Operator for RdfUnionOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        // Execute all operators
        while self.current_idx < self.operators.len() {
            let op = &mut self.operators[self.current_idx];
            match op.next()? {
                Some(chunk) => return Ok(Some(chunk)),
                None => self.current_idx += 1,
            }
        }
        Ok(None)
    }

    fn reset(&mut self) {
        self.current_idx = 0;
        for op in &mut self.operators {
            op.reset();
        }
    }

    fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> std::result::Result<(), QueryResourceContextError> {
        for operator in &mut self.operators {
            operator.install_resource_context(resources)?;
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "RdfUnion"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Bind Operator
// ============================================================================

/// Operator that appends a computed column to each row.
///
/// Used for SPARQL BIND expressions (e.g., `BIND (CONCAT(?x, ?y) AS ?z)`).
/// Evaluates an expression using `RdfExpressionPredicate` and appends the
/// result as a new column in the output chunk.
struct RdfBindOperator {
    /// Child operator providing input rows.
    child: Box<dyn Operator>,
    /// Expression to evaluate per row.
    expression: FilterExpression,
    /// Variable name to column index mapping for expression evaluation.
    variable_columns: HashMap<String, usize>,
}

impl RdfBindOperator {
    fn new(
        child: Box<dyn Operator>,
        expression: FilterExpression,
        variable_columns: HashMap<String, usize>,
    ) -> Self {
        Self {
            child,
            expression,
            variable_columns,
        }
    }
}

impl Operator for RdfBindOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        let Some(input) = self.child.next()? else {
            return Ok(None);
        };

        let input_col_count = input.column_count();
        let row_count = input.row_count();

        // Build output schema: preserve input column types + one extra column for BIND result
        let mut output_types: Vec<LogicalType> = Vec::with_capacity(input_col_count + 1);
        for col_idx in 0..input_col_count {
            if let Some(col) = input.column(col_idx) {
                output_types.push(col.logical_type());
            } else {
                output_types.push(LogicalType::Any);
            }
        }
        output_types.push(LogicalType::Any);

        let mut output = DataChunk::with_capacity(&output_types, row_count);

        // Copy existing columns
        for col_idx in 0..input_col_count {
            let output_col = output
                .column_mut(col_idx)
                .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {col_idx}")))?;
            if let Some(input_col) = input.column(col_idx) {
                for row in input.selected_indices() {
                    if let Some(value) = input_col.get_value(row) {
                        output_col.push_value(value);
                    } else {
                        output_col.push_value(Value::Null);
                    }
                }
            }
        }

        // Evaluate expression for each row and append as new column
        let evaluator =
            RdfExpressionPredicate::new(self.expression.clone(), self.variable_columns.clone());
        let bind_col = output
            .column_mut(input_col_count)
            .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {input_col_count}")))?;
        for row in input.selected_indices() {
            let value = evaluator.eval(&input, row).unwrap_or(Value::Null);
            bind_col.push_value(value);
        }

        output.set_count(row_count);
        Ok(Some(output))
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> std::result::Result<(), QueryResourceContextError> {
        self.child.install_resource_context(resources)
    }

    fn name(&self) -> &'static str {
        "RdfBind"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Project Operator
// ============================================================================

/// Projection variant for expression evaluation.
enum RdfProjectExpr {
    /// Reference to an input column.
    Column(usize),
    /// A constant value.
    Constant(Value),
    /// Full expression evaluation using `RdfExpressionPredicate`.
    Expression {
        /// The filter expression to evaluate.
        expr: FilterExpression,
        /// Variable name to column index mapping.
        variable_columns: HashMap<String, usize>,
    },
}

/// An RDF-specific project operator that uses `RdfExpressionPredicate` for
/// expression evaluation, giving access to SPARQL functions (STRLEN, UCASE,
/// LCASE, etc.) that the generic `ProjectOperator` does not support.
struct RdfProjectOperator {
    /// Child operator providing input rows.
    child: Box<dyn Operator>,
    /// Projection expressions.
    projections: Vec<RdfProjectExpr>,
    /// Output column types.
    output_types: Vec<LogicalType>,
}

impl RdfProjectOperator {
    fn new(
        child: Box<dyn Operator>,
        projections: Vec<RdfProjectExpr>,
        output_types: Vec<LogicalType>,
    ) -> Self {
        Self {
            child,
            projections,
            output_types,
        }
    }
}

impl Operator for RdfProjectOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.projections.len() != self.output_types.len() {
            return Err(OperatorError::Execution(
                "RDF projection expressions do not match the output schema".to_string(),
            ));
        }
        let Some(input) = self.child.next()? else {
            return Ok(None);
        };

        let mut output = DataChunk::with_capacity(&self.output_types, input.row_count());

        for (i, proj) in self.projections.iter().enumerate() {
            let output_col = output
                .column_mut(i)
                .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {i}")))?;

            match proj {
                RdfProjectExpr::Column(col_idx) => {
                    let input_col = input.column(*col_idx).ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("Column {col_idx}"))
                    })?;
                    for row in input.selected_indices() {
                        if let Some(value) = input_col.get_value(row) {
                            output_col.push_value(value);
                        } else {
                            output_col.push_value(Value::Null);
                        }
                    }
                }
                RdfProjectExpr::Constant(value) => {
                    for _ in input.selected_indices() {
                        output_col.push_value(value.clone());
                    }
                }
                RdfProjectExpr::Expression {
                    expr,
                    variable_columns,
                } => {
                    let evaluator =
                        RdfExpressionPredicate::new(expr.clone(), variable_columns.clone());
                    for row in input.selected_indices() {
                        let value = evaluator.eval(&input, row).unwrap_or(Value::Null);
                        output_col.push_value(value);
                    }
                }
            }
        }

        output.set_count(input.row_count());
        Ok(Some(output))
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> std::result::Result<(), QueryResourceContextError> {
        self.child.install_resource_context(resources)
    }

    fn name(&self) -> &'static str {
        "RdfProject"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Triple Scan Operator
// ============================================================================

/// Graph resolution context for a triple scan.
///
/// Groups the graph IRI, scan-all flag, and SPARQL dataset restriction
/// to keep the operator constructor argument count manageable.
struct GraphContext {
    /// Named graph to query. `None` = default graph.
    graph: Option<String>,
    /// Whether to scan ALL graphs (when GRAPH ?var is used).
    scan_all_graphs: bool,
    /// SPARQL dataset restriction from FROM / FROM NAMED clauses.
    dataset: Option<DatasetRestriction>,
}

/// LeapfrogRing WCOJ operator for Ring-backed multi-way joins.
///
/// Prepares query-local Ring tries and streams each native solution into a
/// `DataChunk` with scan-equivalent public and companion columns.
#[cfg(feature = "ring-index")]
struct RdfLeapfrogConfig {
    output_variables: Vec<String>,
    output_owners: Vec<(usize, u8)>,
    output_types: Vec<LogicalType>,
    emit_exact_term_columns: bool,
    emit_identity_key_columns: bool,
    chunk_size: usize,
    output_cap: Option<usize>,
}

#[cfg(feature = "ring-index")]
struct RdfLeapfrogOperator {
    ring: Arc<grafeo_core::index::ring::TripleRing>,
    annotated_patterns: Vec<grafeo_core::index::ring::AnnotatedPattern>,
    /// Output variables in stable, deduplicated order.
    output_variables: Vec<String>,
    /// Deterministic exact witness occurrence for each output variable.
    output_owners: Vec<(usize, u8)>,
    /// Scan-equivalent visible types followed by internal companion types.
    output_types: Vec<LogicalType>,
    /// Whether to append lossless companions for every variable.
    emit_exact_term_columns: bool,
    /// Whether to append canonical identity keys for every variable.
    emit_identity_key_columns: bool,
    /// Immutable query-local canonical tries, prepared on the first pull.
    prepared: Option<grafeo_core::index::ring::PreparedRingJoin>,
    /// Resumable iterative LFTJ state.
    state: Option<grafeo_core::index::ring::RingJoinState>,
    /// Maximum rows produced by one pull, preserving cancellation boundaries.
    chunk_size: usize,
    /// Planner-proven total output cap for a direct row-preserving LIMIT path.
    output_cap: Option<usize>,
}

#[cfg(feature = "ring-index")]
impl RdfLeapfrogOperator {
    fn new(
        ring: Arc<grafeo_core::index::ring::TripleRing>,
        annotated_patterns: Vec<grafeo_core::index::ring::AnnotatedPattern>,
        config: RdfLeapfrogConfig,
    ) -> Self {
        Self {
            ring,
            annotated_patterns,
            output_variables: config.output_variables,
            output_owners: config.output_owners,
            output_types: config.output_types,
            emit_exact_term_columns: config.emit_exact_term_columns,
            emit_identity_key_columns: config.emit_identity_key_columns,
            prepared: None,
            state: None,
            chunk_size: config.chunk_size,
            output_cap: config.output_cap,
        }
    }

    fn ensure_prepared(&mut self) -> std::result::Result<(), OperatorError> {
        if self.prepared.is_some() {
            return Ok(());
        }
        let mut guard = grafeo_core::index::ring::UnboundedRingJoinGuard::new();
        let prepared = grafeo_core::index::ring::PreparedRingJoin::prepare(
            &self.ring,
            &self.annotated_patterns,
            &mut guard,
        )
        .map_err(|error| OperatorError::Execution(error.to_string()))?;
        self.state = Some(prepared.new_state());
        self.prepared = Some(prepared);
        Ok(())
    }

    fn owner_term<'a>(
        &self,
        output_index: usize,
        solution: &'a grafeo_core::index::ring::RingSolution,
    ) -> Option<&'a Term> {
        let (pattern, component) = self.output_owners[output_index];
        let triple = solution.witnesses().get(pattern)?;
        match component {
            0 => Some(triple.subject()),
            1 => Some(triple.predicate()),
            2 => Some(triple.object()),
            _ => None,
        }
    }
}

#[cfg(feature = "ring-index")]
impl Operator for RdfLeapfrogOperator {
    fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
        if self.output_cap == Some(0) {
            return Ok(None);
        }
        self.ensure_prepared()?;
        let visible_count = self.output_variables.len();
        let col_count = self.output_types.len();
        debug_assert_eq!(self.output_types.len(), col_count);
        let mut chunk = DataChunk::with_capacity(&self.output_types, self.chunk_size);
        let mut batch_size = 0;
        let mut guard = grafeo_core::index::ring::UnboundedRingJoinGuard::new();
        if let Some(output_cap) = self.output_cap {
            guard = guard.with_output_cap(output_cap);
        }

        let batch_cap = self.output_cap.map_or(self.chunk_size, |output_cap| {
            self.chunk_size.min(output_cap)
        });
        while batch_size < batch_cap {
            let solution = {
                let prepared = self.prepared.as_ref().ok_or_else(|| {
                    OperatorError::Execution("RDF Ring join is not prepared".to_string())
                })?;
                let state = self.state.as_mut().ok_or_else(|| {
                    OperatorError::Execution("RDF Ring join has no execution state".to_string())
                })?;
                prepared
                    .next_solution(state, &mut guard)
                    .map_err(|error| OperatorError::Execution(error.to_string()))?
            };
            let Some(solution) = solution else {
                break;
            };
            let mut output_index = 0;
            for visible_index in 0..visible_count {
                let owner = self.owner_term(visible_index, &solution);
                if let Some(col) = chunk.column_mut(output_index) {
                    if let Some(term) = owner {
                        push_term_value(col, term);
                    } else {
                        col.push_value(Value::Null);
                    }
                }
                output_index += 1;
                if self.emit_exact_term_columns {
                    if let Some(column) = chunk.column_mut(output_index) {
                        if let Some(term) = owner {
                            column.push_string(term.to_ntriples());
                        } else {
                            column.push_value(Value::Null);
                        }
                    }
                    output_index += 1;
                }
                if self.emit_identity_key_columns {
                    if let Some(column) = chunk.column_mut(output_index) {
                        if let Some(term) = owner {
                            column.push_string(term.canonical_identity_key());
                        } else {
                            column.push_value(Value::Null);
                        }
                    }
                    output_index += 1;
                }
                if self.output_owners[visible_index].1 == 2 {
                    if let Some(column) = chunk.column_mut(output_index) {
                        let language = owner
                            .and_then(Term::as_literal)
                            .and_then(Literal::language)
                            .unwrap_or("");
                        column.push_string(language.to_string());
                    }
                    output_index += 1;
                }
            }
            debug_assert_eq!(output_index, col_count);
            batch_size += 1;
        }

        if batch_size == 0 {
            return Ok(None);
        }
        chunk.set_count(batch_size);
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.state = self.prepared.as_ref().map(|prepared| prepared.new_state());
    }

    fn name(&self) -> &'static str {
        "RdfLeapfrog"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// CONSTRUCT operator: instantiates triple templates from variable bindings.
///
/// For each input row, substitutes variables in each template triple to produce
/// (subject, predicate, object) output rows. Skips template triples where a
/// variable has no binding (unbound variables produce no triple).
struct ConstructOperator {
    input: Box<dyn Operator>,
    templates: Vec<TripleTemplate>,
    variable_columns: HashMap<String, usize>,
}

impl ConstructOperator {
    fn new(
        input: Box<dyn Operator>,
        templates: Vec<TripleTemplate>,
        variable_columns: HashMap<String, usize>,
    ) -> Self {
        Self {
            input,
            templates,
            variable_columns,
        }
    }

    /// Resolves a `TripleComponent` to a string using the current row bindings.
    fn resolve_component(
        &self,
        component: &TripleComponent,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<String> {
        match component {
            TripleComponent::Variable(name) => {
                let col_idx = *self.variable_columns.get(name)?;
                let col = chunk.column(col_idx)?;
                let val = col.get_value(row)?;
                if val.is_null() {
                    None
                } else {
                    Some(val.to_string())
                }
            }
            TripleComponent::Iri(iri) => Some(iri.clone()),
            TripleComponent::Literal(val) => Some(val.to_string()),
            TripleComponent::LangLiteral { value, lang } => Some(format!("\"{value}\"@{lang}")),
            TripleComponent::BlankNode(label) => Some(format!("_:{label}")),
        }
    }
}

impl Operator for ConstructOperator {
    fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
        loop {
            let Some(input_chunk) = self.input.next()? else {
                return Ok(None);
            };

            let row_count = input_chunk.row_count();
            if row_count == 0 {
                continue;
            }

            // Each input row can produce up to templates.len() output rows
            let max_output = row_count * self.templates.len();
            let schema = vec![LogicalType::String; 3];
            let mut output = DataChunk::with_capacity(&schema, max_output);
            let mut actual_count = 0;

            for row in 0..row_count {
                for template in &self.templates {
                    let Some(subject) =
                        self.resolve_component(&template.subject, &input_chunk, row)
                    else {
                        continue;
                    };
                    let Some(predicate) =
                        self.resolve_component(&template.predicate, &input_chunk, row)
                    else {
                        continue;
                    };
                    let Some(object) = self.resolve_component(&template.object, &input_chunk, row)
                    else {
                        continue;
                    };

                    if let Some(col) = output.column_mut(0) {
                        col.push_string(subject);
                    }
                    if let Some(col) = output.column_mut(1) {
                        col.push_string(predicate);
                    }
                    if let Some(col) = output.column_mut(2) {
                        col.push_string(object);
                    }
                    actual_count += 1;
                }
            }

            if actual_count > 0 {
                output.set_count(actual_count);
                return Ok(Some(output));
            }
        }
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> std::result::Result<(), QueryResourceContextError> {
        self.input.install_resource_context(resources)
    }

    fn name(&self) -> &'static str {
        "Construct"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Operator that produces a single pre-computed `DataChunk` and then stops.
///
/// Used for fast-path results such as `COUNT(*)` short-circuits where the
/// answer is known without scanning the data.
struct ConstantOperator {
    chunk: Option<DataChunk>,
}

impl ConstantOperator {
    fn new(chunk: DataChunk) -> Self {
        Self { chunk: Some(chunk) }
    }
}

impl Operator for ConstantOperator {
    fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
        Ok(self.chunk.take())
    }

    fn reset(&mut self) {
        // Cannot reset: the chunk was consumed. This is fine for single-use
        // fast-path results.
    }

    fn name(&self) -> &'static str {
        "Constant"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Resolves dictionary-encoded Int64 term IDs back to String values.
///
/// Placed at the result boundary (just before output) so that all intermediate
/// operators (joins, aggregates, sorts) work with compact Int64 keys.
struct DictResolveOperator {
    input: Box<dyn Operator>,
    dictionary: Arc<grafeo_core::graph::rdf::TermDictionary>,
    /// Column indices that carry encoded term IDs and need resolution.
    encoded_col_indices: Vec<usize>,
}

impl DictResolveOperator {
    fn new(
        input: Box<dyn Operator>,
        dictionary: Arc<grafeo_core::graph::rdf::TermDictionary>,
        encoded_col_indices: Vec<usize>,
    ) -> Self {
        Self {
            input,
            dictionary,
            encoded_col_indices,
        }
    }
}

impl Operator for DictResolveOperator {
    fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
        let Some(chunk) = self.input.next()? else {
            return Ok(None);
        };

        if self.encoded_col_indices.is_empty() {
            return Ok(Some(chunk));
        }

        let row_count = chunk.row_count();
        let col_count = chunk.column_count();

        // Build output schema: replace Int64 encoded columns with String
        let mut schema = Vec::with_capacity(col_count);
        for col_idx in 0..col_count {
            if self.encoded_col_indices.contains(&col_idx) {
                schema.push(LogicalType::String);
            } else if let Some(col) = chunk.column(col_idx) {
                schema.push(col.data_type().clone());
            } else {
                schema.push(LogicalType::String);
            }
        }

        let mut out = DataChunk::with_capacity(&schema, row_count);

        for row in 0..row_count {
            for col_idx in 0..col_count {
                let Some(in_col) = chunk.column(col_idx) else {
                    continue;
                };
                let Some(out_col) = out.column_mut(col_idx) else {
                    continue;
                };

                if self.encoded_col_indices.contains(&col_idx) {
                    // Resolve term ID to string
                    if in_col.is_null(row) {
                        out_col.push_value(Value::Null);
                    } else if let Some(term_id) = in_col.get_int64(row) {
                        // reason: term IDs are assigned sequentially from 0 and fit u32
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let tid = term_id as u32;
                        if let Some(term) = self.dictionary.get_term(tid) {
                            out_col.push_string(term_to_string(term));
                        } else {
                            out_col.push_value(Value::Null);
                        }
                    } else {
                        out_col.push_value(Value::Null);
                    }
                } else {
                    // Pass through non-encoded columns
                    if let Some(val) = in_col.get_value(row) {
                        out_col.push_value(val);
                    } else {
                        out_col.push_value(Value::Null);
                    }
                }
            }
        }

        out.set_count(row_count);
        Ok(Some(out))
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> std::result::Result<(), QueryResourceContextError> {
        self.input.install_resource_context(resources)
    }

    fn name(&self) -> &'static str {
        "DictResolve"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Native SPARQL `pred*` / `pred+` reachability (visited-set BFS, no hop cap).
struct RdfPropertyPathOperator {
    store: Arc<RdfStore>,
    path: PropertyPathOp,
    output_columns: Vec<String>,
    chunk_size: usize,
    transaction_id: Option<TransactionId>,
    emit_exact_term_columns: bool,
    emit_identity_key_columns: bool,
    rows: Option<Vec<(Term, Term)>>,
    position: usize,
}

impl RdfPropertyPathOperator {
    fn new(
        store: Arc<RdfStore>,
        path: PropertyPathOp,
        output_columns: Vec<String>,
        chunk_size: usize,
        transaction_id: Option<TransactionId>,
        emit_exact_term_columns: bool,
        emit_identity_key_columns: bool,
    ) -> Self {
        Self {
            store,
            path,
            output_columns,
            chunk_size,
            transaction_id,
            emit_exact_term_columns,
            emit_identity_key_columns,
            rows: None,
            position: 0,
        }
    }

    fn target_store(&self) -> Arc<RdfStore> {
        match &self.path.graph {
            Some(name) => self
                .store
                .graph_in_transaction(name, self.transaction_id)
                .unwrap_or_else(|| Arc::new(RdfStore::new())),
            None => Arc::clone(&self.store),
        }
    }

    fn expand_iri(
        store: &RdfStore,
        tid: Option<TransactionId>,
        node: &Term,
        iri: &str,
        inverse: bool,
    ) -> Vec<Term> {
        let pred = Term::iri(iri);
        let pattern = if inverse {
            TriplePattern {
                subject: None,
                predicate: Some(pred),
                object: Some(node.clone()),
            }
        } else {
            TriplePattern {
                subject: Some(node.clone()),
                predicate: Some(pred),
                object: None,
            }
        };
        store
            .find_with_pending(&pattern, tid)
            .into_iter()
            .map(|t| {
                if inverse {
                    t.subject().clone()
                } else {
                    t.object().clone()
                }
            })
            .collect()
    }

    fn expand_step(
        store: &RdfStore,
        tid: Option<TransactionId>,
        node: &Term,
        step: &PathStep,
    ) -> Vec<Term> {
        match step {
            PathStep::Iri { iri, inverse } => Self::expand_iri(store, tid, node, iri, *inverse),
            PathStep::Sequence(steps) => {
                let mut frontier = vec![node.clone()];
                for s in steps {
                    let mut next = Vec::new();
                    for n in &frontier {
                        next.extend(Self::expand_step(store, tid, n, s));
                    }
                    frontier = next;
                    if frontier.is_empty() {
                        break;
                    }
                }
                frontier
            }
            PathStep::Alternative(steps) => {
                let mut out = Vec::new();
                for s in steps {
                    out.extend(Self::expand_step(store, tid, node, s));
                }
                out
            }
        }
    }

    fn collect_step_iris(step: &PathStep, out: &mut Vec<(String, bool)>) {
        match step {
            PathStep::Iri { iri, inverse } => out.push((iri.clone(), *inverse)),
            PathStep::Sequence(steps) | PathStep::Alternative(steps) => {
                for s in steps {
                    Self::collect_step_iris(s, out);
                }
            }
        }
    }

    fn compute(&mut self) {
        let store = self.target_store();
        let start_bound = component_to_term(&self.path.subject);
        let end_bound = component_to_term(&self.path.object);
        let walk_inverse_from_end = start_bound.is_none() && end_bound.is_some();
        let hop = if walk_inverse_from_end {
            self.path.path.inverted()
        } else {
            self.path.path.clone()
        };

        let starts: Vec<Term> = if let Some(s) = start_bound.clone() {
            vec![s]
        } else if let Some(end) = end_bound.clone() {
            vec![end]
        } else {
            let mut iris = Vec::new();
            Self::collect_step_iris(&hop, &mut iris);
            let mut nodes = HashSet::new();
            for (iri, inverse) in iris {
                let pred = Term::iri(iri);
                let triples = store.find_with_pending(
                    &TriplePattern {
                        subject: None,
                        predicate: Some(pred),
                        object: None,
                    },
                    self.transaction_id,
                );
                for t in triples {
                    if inverse {
                        nodes.insert(t.object().clone());
                        nodes.insert(t.subject().clone());
                    } else {
                        nodes.insert(t.subject().clone());
                        nodes.insert(t.object().clone());
                    }
                }
            }
            nodes.into_iter().collect()
        };

        let mut out: Vec<(Term, Term)> = Vec::new();
        let mut seen_pairs: HashSet<(Term, Term)> = HashSet::new();
        let push_pair = |out: &mut Vec<(Term, Term)>,
                         seen: &mut HashSet<(Term, Term)>,
                         start: &Term,
                         node: &Term| {
            let pair = if walk_inverse_from_end {
                (node.clone(), start.clone())
            } else {
                (start.clone(), node.clone())
            };
            if seen.insert(pair.clone()) {
                out.push(pair);
            }
        };

        for start in starts {
            if self.path.min_hops == 0 {
                if let Some(ref end) = end_bound {
                    if walk_inverse_from_end || start == *end {
                        push_pair(&mut out, &mut seen_pairs, &start, &start);
                    }
                } else {
                    push_pair(&mut out, &mut seen_pairs, &start, &start);
                }
            }

            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back(start.clone());
            visited.insert(start.clone());
            while let Some(node) = queue.pop_front() {
                for next in Self::expand_step(&store, self.transaction_id, &node, &hop) {
                    let newly_visited = visited.insert(next.clone());
                    if let Some(ref end) = end_bound {
                        if walk_inverse_from_end || next == *end {
                            push_pair(&mut out, &mut seen_pairs, &start, &next);
                        }
                    } else {
                        push_pair(&mut out, &mut seen_pairs, &start, &next);
                    }
                    if newly_visited {
                        queue.push_back(next);
                    }
                }
            }
        }

        self.rows = Some(out);
    }
}

impl Operator for RdfPropertyPathOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.rows.is_none() {
            self.compute();
        }
        let rows = self.rows.as_ref().ok_or_else(|| {
            OperatorError::Execution("RDF property path has no computed rows".to_string())
        })?;
        if self.position >= rows.len() {
            return Ok(None);
        }
        let end = (self.position + self.chunk_size).min(rows.len());
        let schema: Vec<LogicalType> = self
            .output_columns
            .iter()
            .map(|_| LogicalType::String)
            .collect();
        let mut chunk = DataChunk::with_capacity(&schema, end - self.position);
        for (s, o) in &rows[self.position..end] {
            let mut col_idx = 0;
            if matches!(self.path.subject, TripleComponent::Variable(_)) {
                if let Some(col) = chunk.column_mut(col_idx) {
                    col.push_string(term_to_string(s));
                }
                col_idx += 1;
                if self.emit_exact_term_columns {
                    if let Some(col) = chunk.column_mut(col_idx) {
                        col.push_string(s.to_ntriples());
                    }
                    col_idx += 1;
                }
                if self.emit_identity_key_columns {
                    if let Some(col) = chunk.column_mut(col_idx) {
                        col.push_string(s.canonical_identity_key());
                    }
                    col_idx += 1;
                }
            }
            if matches!(self.path.object, TripleComponent::Variable(_)) {
                if let Some(col) = chunk.column_mut(col_idx) {
                    col.push_string(term_to_string(o));
                }
                col_idx += 1;
                if self.emit_exact_term_columns {
                    if let Some(col) = chunk.column_mut(col_idx) {
                        col.push_string(o.to_ntriples());
                    }
                    col_idx += 1;
                }
                if self.emit_identity_key_columns
                    && let Some(col) = chunk.column_mut(col_idx)
                {
                    col.push_string(o.canonical_identity_key());
                }
            }
        }
        chunk.set_count(end - self.position);
        self.position = end;
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "RdfPropertyPath"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Column layout emitted by an RDF triple scan.
#[derive(Clone, Copy)]
struct RdfTermCompanionOutput {
    lossless: bool,
    identity: bool,
}

struct RdfTripleScanOutput {
    mask: [bool; 4],
    companion_columns: bool,
    datatype_column: bool,
    term_companions: RdfTermCompanionOutput,
}

/// Lazy triple scan operator that processes triples in chunks.
///
/// This operator queries the RDF store and emits results in DataChunks
/// for efficient vectorized processing.
struct RdfTripleScanOperator {
    /// The RDF store to scan.
    store: Arc<RdfStore>,
    /// The pattern to match.
    pattern: TriplePattern,
    /// Which components to include in output [s, p, o, g].
    output_mask: [bool; 4],
    /// Graph resolution context (graph IRI, scan-all flag, dataset restriction).
    graph_context: GraphContext,
    /// Whether to emit a companion language-tag column after the object column.
    emit_companion_columns: bool,
    /// Whether to also emit a companion datatype column (only when DATATYPE() is used).
    emit_datatype_column: bool,
    /// Independently demanded lossless reconstruction and canonical identity
    /// companions for each S/P/O/G variable column.
    term_companions: RdfTermCompanionOutput,
    /// Chunk size for batching.
    chunk_size: usize,
    /// Cached matching triples with graph names (lazily populated).
    triples: Option<Vec<(Option<String>, Arc<Triple>)>>,
    /// Current position in the triples.
    position: usize,
    /// Optional term dictionary for dictionary-encoded output. When present,
    /// S/P/O variable columns emit Int64 term IDs instead of String values.
    dictionary: Option<Arc<grafeo_core::graph::rdf::TermDictionary>>,
    /// Session transaction, if any. Scans use `find_with_pending` so SPARQL
    /// reads its own uncommitted writes.
    transaction_id: Option<TransactionId>,
}

impl RdfTripleScanOperator {
    fn new(
        store: Arc<RdfStore>,
        pattern: TriplePattern,
        output: RdfTripleScanOutput,
        chunk_size: usize,
        graph_context: GraphContext,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        Self {
            store,
            pattern,
            output_mask: output.mask,
            graph_context,
            emit_companion_columns: output.companion_columns,
            emit_datatype_column: output.datatype_column,
            term_companions: output.term_companions,
            chunk_size,
            triples: None,
            position: 0,
            dictionary: None,
            transaction_id,
        }
    }

    /// Enables dictionary-encoded output for S/P/O columns.
    ///
    /// Infrastructure for Phase 4 (Ring Index) which will automatically enable
    /// dictionary encoding when the Ring provides native integer-keyed iteration.
    #[allow(dead_code)]
    fn with_dictionary(mut self, dict: Arc<grafeo_core::graph::rdf::TermDictionary>) -> Self {
        self.dictionary = Some(dict);
        self
    }

    /// Lazily load matching triples on first access.
    ///
    /// Respects SPARQL dataset clauses (FROM / FROM NAMED):
    /// - FROM: basic patterns (no graph context) scan the union of specified named graphs.
    /// - FROM NAMED: GRAPH patterns only iterate listed named graphs.
    fn ensure_triples(&mut self) {
        if self.triples.is_none() {
            let ctx = &self.graph_context;
            self.triples = Some(if ctx.scan_all_graphs {
                // GRAPH ?var: scan named graphs (restricted by FROM NAMED if present)
                if let Some(ref ds) = ctx.dataset {
                    if ds.named_graphs == [RDF_EXPLICIT_EMPTY_NAMED_DATASET] {
                        // An explicit dataset without FROM NAMED has no named graphs.
                        Vec::new()
                    } else if !ds.named_graphs.is_empty() {
                        let mut graph_refs: Vec<&str> =
                            ds.named_graphs.iter().map(String::as_str).collect();
                        graph_refs.sort_unstable();
                        graph_refs.dedup();
                        self.store.find_in_graphs_with_pending(
                            &self.pattern,
                            Some(&graph_refs),
                            self.transaction_id,
                        )
                    } else {
                        // Preserve the public DatasetRestriction convention:
                        // an empty named list means unrestricted named graphs.
                        self.store.find_in_graphs_with_pending(
                            &self.pattern,
                            Some(&[]),
                            self.transaction_id,
                        )
                    }
                } else {
                    // No dataset restriction: scan all graphs
                    self.store.find_in_graphs_with_pending(
                        &self.pattern,
                        Some(&[]),
                        self.transaction_id,
                    )
                }
            } else if let Some(ref graph_iri) = ctx.graph {
                // GRAPH <iri>: scan specific named graph (restricted by FROM NAMED if present)
                if let Some(ref ds) = ctx.dataset {
                    if !ds.named_graphs.is_empty()
                        && !ds.named_graphs.iter().any(|g| g == graph_iri)
                    {
                        // The specified graph is not in the FROM NAMED list: empty result
                        Vec::new()
                    } else {
                        self.store
                            .graph_in_transaction(graph_iri, self.transaction_id)
                            .map(|g| {
                                g.find_with_pending(&self.pattern, self.transaction_id)
                                    .into_iter()
                                    .map(|t| (Some(graph_iri.clone()), t))
                                    .collect()
                            })
                            .unwrap_or_default()
                    }
                } else {
                    self.store
                        .graph_in_transaction(graph_iri, self.transaction_id)
                        .map(|g| {
                            g.find_with_pending(&self.pattern, self.transaction_id)
                                .into_iter()
                                .map(|t| (Some(graph_iri.clone()), t))
                                .collect()
                        })
                        .unwrap_or_default()
                }
            } else {
                // No graph context (basic triple pattern).
                // FROM clauses redefine the default graph as the union of specified graphs.
                if let Some(ref ds) = ctx.dataset {
                    if ds.default_graphs == [RDF_EXPLICIT_EMPTY_DEFAULT_DATASET] {
                        Vec::new()
                    } else if !ds.default_graphs.is_empty() {
                        // FROM: default graph = union of specified named graphs.
                        // Deduplicate graph IRIs first so listing the same IRI
                        // twice does not produce duplicate triples.
                        let mut unique_graphs: Vec<&str> =
                            ds.default_graphs.iter().map(String::as_str).collect();
                        unique_graphs.sort_unstable();
                        unique_graphs.dedup();
                        let mut results = self.store.find_in_graphs_with_pending(
                            &self.pattern,
                            Some(&unique_graphs),
                            self.transaction_id,
                        );
                        // Clear graph names so results appear as default-graph triples
                        for item in &mut results {
                            item.0 = None;
                        }
                        // A SPARQL default graph is an RDF graph (a set), even
                        // when the same triple occurs in multiple source graphs.
                        let mut seen = HashSet::new();
                        results.retain(|(_, triple)| seen.insert(Arc::clone(triple)));
                        results
                    } else {
                        // Preserve the public DatasetRestriction convention:
                        // an empty default list means the actual default graph.
                        self.store
                            .find_with_pending(&self.pattern, self.transaction_id)
                            .into_iter()
                            .map(|t| (None, t))
                            .collect()
                    }
                } else {
                    // No dataset restriction: use actual default graph.
                    // Prefer Ring Index when available (O(log sigma) access)
                    // unless a transaction must see its own pending writes.
                    #[cfg(feature = "ring-index")]
                    {
                        if self.transaction_id.is_none()
                            && let Some(ring) = self.store.ring()
                        {
                            ring.find(&self.pattern)
                                .map(|t| (None, Arc::new(t)))
                                .collect()
                        } else {
                            self.store
                                .find_with_pending(&self.pattern, self.transaction_id)
                                .into_iter()
                                .map(|t| (None, t))
                                .collect()
                        }
                    }
                    #[cfg(not(feature = "ring-index"))]
                    {
                        self.store
                            .find_with_pending(&self.pattern, self.transaction_id)
                            .into_iter()
                            .map(|t| (None, t))
                            .collect()
                    }
                }
            });
        }
    }

    /// Count how many output columns we have.
    fn output_column_count(&self) -> usize {
        let base = self.output_mask.iter().filter(|&&b| b).count();
        let exact = if self.term_companions.lossless {
            self.output_mask.iter().filter(|&&enabled| enabled).count()
        } else {
            0
        };
        let identity = if self.term_companions.identity {
            self.output_mask.iter().filter(|&&enabled| enabled).count()
        } else {
            0
        };
        let literal_companions = if self.emit_companion_columns {
            if self.emit_datatype_column { 2 } else { 1 }
        } else {
            0
        };
        base + exact + identity + literal_companions
    }

    /// Builds the output schema. In dictionary mode, S/P/O variable columns
    /// are Int64 (term IDs), companion and graph columns remain String.
    fn build_output_schema(&self, col_count: usize, dict_mode: bool) -> Vec<LogicalType> {
        if !dict_mode {
            let mut schema = Vec::with_capacity(col_count);
            if self.output_mask[0] {
                schema.push(LogicalType::String);
                if self.term_companions.lossless {
                    schema.push(LogicalType::String);
                }
                if self.term_companions.identity {
                    schema.push(LogicalType::String);
                }
            }
            if self.output_mask[1] {
                schema.push(LogicalType::String);
                if self.term_companions.lossless {
                    schema.push(LogicalType::String);
                }
                if self.term_companions.identity {
                    schema.push(LogicalType::String);
                }
            }
            if self.output_mask[2] {
                schema.push(LogicalType::Any);
                if self.term_companions.lossless {
                    schema.push(LogicalType::String);
                }
                if self.term_companions.identity {
                    schema.push(LogicalType::String);
                }
            }
            if self.output_mask[2] && self.emit_companion_columns {
                schema.push(LogicalType::String);
                if self.emit_datatype_column {
                    schema.push(LogicalType::String);
                }
            }
            if self.output_mask[3] {
                schema.push(LogicalType::String);
                if self.term_companions.lossless {
                    schema.push(LogicalType::String);
                }
                if self.term_companions.identity {
                    schema.push(LogicalType::String);
                }
            }
            debug_assert_eq!(schema.len(), col_count);
            return schema;
        }
        let mut schema = Vec::with_capacity(col_count);
        // S, P, O variable columns get Int64; exact companions remain String.
        for i in 0..3 {
            if self.output_mask[i] {
                schema.push(LogicalType::Int64);
                if self.term_companions.lossless {
                    schema.push(LogicalType::String);
                }
                if self.term_companions.identity {
                    schema.push(LogicalType::String);
                }
            }
        }
        // Companion columns (lang, datatype) are always String
        if self.output_mask[2] && self.emit_companion_columns {
            schema.push(LogicalType::String); // lang
            if self.emit_datatype_column {
                schema.push(LogicalType::String); // datatype
            }
        }
        // Graph column is always String
        if self.output_mask[3] {
            schema.push(LogicalType::String);
            if self.term_companions.lossless {
                schema.push(LogicalType::String);
            }
            if self.term_companions.identity {
                schema.push(LogicalType::String);
            }
        }
        schema
    }
}

impl Operator for RdfTripleScanOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        self.ensure_triples();

        let triples = self.triples.as_ref().ok_or_else(|| {
            OperatorError::Execution("RDF triple scan has no prepared triples".to_string())
        })?;

        if self.position >= triples.len() {
            return Ok(None);
        }

        let end = (self.position + self.chunk_size).min(triples.len());
        let batch_size = end - self.position;
        let col_count = self.output_column_count();

        // Create output schema: Int64 for dictionary-encoded S/P/O, String otherwise
        let dict_mode = self.dictionary.is_some();
        let schema: Vec<LogicalType> = self.build_output_schema(col_count, dict_mode);
        let mut chunk = DataChunk::with_capacity(&schema, batch_size);

        // Fill the chunk
        for i in self.position..end {
            let (ref graph_name, ref triple) = triples[i];
            let mut col_idx = 0;

            if self.output_mask[0] {
                // Subject
                if let Some(col) = chunk.column_mut(col_idx) {
                    if let Some(ref dict) = self.dictionary {
                        if let Some(id) = dict.get_id(triple.subject()) {
                            col.push_value(Value::Int64(i64::from(id)));
                        } else {
                            col.push_value(Value::Null);
                        }
                    } else {
                        col.push_string(term_to_string(triple.subject()));
                    }
                }
                col_idx += 1;
                if self.term_companions.lossless {
                    if let Some(col) = chunk.column_mut(col_idx) {
                        col.push_string(triple.subject().to_ntriples());
                    }
                    col_idx += 1;
                }
                if self.term_companions.identity {
                    if let Some(col) = chunk.column_mut(col_idx) {
                        col.push_string(triple.subject().canonical_identity_key());
                    }
                    col_idx += 1;
                }
            }
            if self.output_mask[1] {
                // Predicate
                if let Some(col) = chunk.column_mut(col_idx) {
                    if let Some(ref dict) = self.dictionary {
                        if let Some(id) = dict.get_id(triple.predicate()) {
                            col.push_value(Value::Int64(i64::from(id)));
                        } else {
                            col.push_value(Value::Null);
                        }
                    } else {
                        col.push_string(term_to_string(triple.predicate()));
                    }
                }
                col_idx += 1;
                if self.term_companions.lossless {
                    if let Some(col) = chunk.column_mut(col_idx) {
                        col.push_string(triple.predicate().to_ntriples());
                    }
                    col_idx += 1;
                }
                if self.term_companions.identity {
                    if let Some(col) = chunk.column_mut(col_idx) {
                        col.push_string(triple.predicate().canonical_identity_key());
                    }
                    col_idx += 1;
                }
            }
            if self.output_mask[2] {
                // Object: dictionary encoding applies here too
                if let Some(col) = chunk.column_mut(col_idx) {
                    if let Some(ref dict) = self.dictionary {
                        if let Some(id) = dict.get_id(triple.object()) {
                            col.push_value(Value::Int64(i64::from(id)));
                        } else {
                            col.push_value(Value::Null);
                        }
                    } else {
                        push_term_value(col, triple.object());
                    }
                }
                col_idx += 1;

                if self.term_companions.lossless {
                    if let Some(col) = chunk.column_mut(col_idx) {
                        col.push_string(triple.object().to_ntriples());
                    }
                    col_idx += 1;
                }
                if self.term_companions.identity {
                    if let Some(col) = chunk.column_mut(col_idx) {
                        col.push_string(triple.object().canonical_identity_key());
                    }
                    col_idx += 1;
                }

                // Companion language-tag and datatype columns (always String)
                if self.emit_companion_columns {
                    if let Some(col) = chunk.column_mut(col_idx) {
                        let lang_tag = match triple.object() {
                            Term::Literal(lit) => lit.language().unwrap_or("").to_string(),
                            _ => String::new(),
                        };
                        col.push_string(lang_tag);
                    }
                    col_idx += 1;

                    if self.emit_datatype_column {
                        if let Some(col) = chunk.column_mut(col_idx) {
                            let datatype = match triple.object() {
                                Term::Literal(lit) => lit.datatype().to_string(),
                                _ => String::new(),
                            };
                            col.push_string(datatype);
                        }
                        col_idx += 1;
                    }
                }
            }
            if self.output_mask[3] {
                // Graph (always String)
                if let Some(col) = chunk.column_mut(col_idx) {
                    match graph_name {
                        Some(name) => col.push_string(name.clone()),
                        None => col.push_value(Value::Null),
                    }
                }
                col_idx += 1;
                if self.term_companions.lossless {
                    if let Some(col) = chunk.column_mut(col_idx) {
                        match graph_name {
                            Some(name) => col.push_string(Term::iri(name.clone()).to_ntriples()),
                            None => col.push_value(Value::Null),
                        }
                    }
                    col_idx += 1;
                }
                if self.term_companions.identity
                    && let Some(col) = chunk.column_mut(col_idx)
                {
                    match graph_name {
                        Some(name) => {
                            col.push_string(Term::iri(name.clone()).canonical_identity_key());
                        }
                        None => col.push_value(Value::Null),
                    }
                }
            }
        }

        chunk.set_count(batch_size);
        self.position = end;

        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.position = 0;
        // Keep triples cached for re-execution
    }

    fn name(&self) -> &'static str {
        "RdfTripleScan"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Expression Predicate
// ============================================================================

/// Builds the one rowwise key consumed by RDF GROUP, DISTINCT, and
/// compatibility. Existing discriminated provenance wins only when present
/// on this row; canonical RDF identity is the next choice, and every remaining
/// bound value is kept as a native value inside the discriminated envelope.
fn normalized_rdf_or_native_key(
    visible: &Value,
    existing: Option<&Value>,
    canonical_rdf: Option<&Value>,
) -> Value {
    if visible.is_null() {
        return Value::Null;
    }
    if let Some(existing) = existing.filter(|value| !value.is_null()) {
        return existing.clone();
    }
    if let Some(identity) = canonical_rdf.filter(|value| !value.is_null()) {
        return Value::List(vec![Value::Bool(true), identity.clone()].into());
    }
    Value::List(vec![Value::Bool(false), visible.clone()].into())
}

/// Finite dispatch after recognizing an internal RDF term-kind function.
enum RdfTermKindTest {
    Iri,
    Blank,
    Literal,
    Numeric,
}

/// Finite dispatch after recognizing an internal RDF term constructor.
enum RdfTermTagger {
    Iri,
    Blank,
    Literal,
}

/// Expression predicate for RDF queries.
///
/// Unlike the LPG predicate, this doesn't need a store reference because
/// RDF values are already materialized in the DataChunk columns.
struct RdfExpressionPredicate {
    expression: FilterExpression,
    variable_columns: HashMap<String, usize>,
}

impl RdfExpressionPredicate {
    fn new(expression: FilterExpression, variable_columns: HashMap<String, usize>) -> Self {
        Self {
            expression,
            variable_columns,
        }
    }

    fn eval(&self, chunk: &DataChunk, row: usize) -> Option<Value> {
        self.eval_expr(&self.expression, chunk, row)
    }

    /// Reads an authoritative lossless companion for a bound scalar operand.
    /// An absent term denotes a native/no-companion value; a malformed
    /// non-null RDF companion fails the expression instead
    /// of falling back to a guess from the visible host value.
    fn bound_scalar_term(
        &self,
        expression: &FilterExpression,
        chunk: &DataChunk,
        row: usize,
    ) -> std::result::Result<Option<Term>, ()> {
        let FilterExpression::Variable(variable) = expression else {
            return Ok(None);
        };
        let visible = self
            .variable_columns
            .get(variable)
            .and_then(|&column| chunk.column(column))
            .and_then(|column| column.get_value(row));
        if visible.as_ref().is_none_or(Value::is_null) {
            return Ok(None);
        }
        let Some((_, &column)) = self.variable_columns.iter().find(|(name, _)| {
            name.strip_prefix(RDF_EXACT_TERM_COLUMN_PREFIX) == Some(variable.as_str())
        }) else {
            return Ok(None);
        };
        let Some(exact) = chunk
            .column(column)
            .and_then(|column| column.get_value(row))
        else {
            return Err(());
        };
        if exact.is_null() {
            // A mixed RDF/native projection deliberately carries NULL in the
            // exact column for its native branch. Preserve that fallback.
            return Ok(None);
        }
        exact
            .as_str()
            .and_then(Term::from_ntriples)
            .map(Some)
            .ok_or(())
    }

    fn eval_expr(&self, expr: &FilterExpression, chunk: &DataChunk, row: usize) -> Option<Value> {
        match expr {
            FilterExpression::Literal(v) => (!v.is_null()).then(|| v.clone()),
            FilterExpression::Variable(name) => {
                let col_idx = *self.variable_columns.get(name)?;
                chunk
                    .column(col_idx)?
                    .get_value(row)
                    .filter(|value| !value.is_null())
            }
            FilterExpression::Property { variable, .. } => {
                // For RDF, treat property access as variable access
                let col_idx = *self.variable_columns.get(variable)?;
                chunk.column(col_idx)?.get_value(row)
            }
            FilterExpression::Binary { left, op, right } => {
                if matches!(op, BinaryFilterOp::And | BinaryFilterOp::Or) {
                    let left = self
                        .eval_expr(left, chunk, row)
                        .and_then(|value| rdf_effective_boolean_value(&value));
                    if (*op == BinaryFilterOp::And && left == Some(false))
                        || (*op == BinaryFilterOp::Or && left == Some(true))
                    {
                        return Some(Value::Bool(*op == BinaryFilterOp::Or));
                    }
                    let right = self
                        .eval_expr(right, chunk, row)
                        .and_then(|value| rdf_effective_boolean_value(&value));
                    return match (*op, left, right) {
                        (BinaryFilterOp::And, Some(true), Some(value))
                        | (BinaryFilterOp::Or, Some(false), Some(value)) => {
                            Some(Value::Bool(value))
                        }
                        (BinaryFilterOp::And, None, Some(false)) => Some(Value::Bool(false)),
                        (BinaryFilterOp::Or, None, Some(true)) => Some(Value::Bool(true)),
                        _ => None,
                    };
                }
                // IN operator: evaluate right side as a list, then check membership
                if *op == BinaryFilterOp::In {
                    let left_val = self.eval_expr(left, chunk, row)?;
                    let right_val = self.eval_expr(right, chunk, row)?;
                    return match right_val {
                        Value::List(items) => {
                            if left_val.is_null() {
                                return Some(Value::Null);
                            }
                            let mut has_null = false;
                            for item in items.iter() {
                                if item.is_null() {
                                    has_null = true;
                                } else if rdf_values_equal(&left_val, item) {
                                    return Some(Value::Bool(true));
                                }
                            }
                            if has_null {
                                Some(Value::Null)
                            } else {
                                Some(Value::Bool(false))
                            }
                        }
                        _ => None,
                    };
                }
                let left_val = self.eval_expr(left, chunk, row)?;
                let right_val = self.eval_expr(right, chunk, row)?;
                self.eval_binary_op(&left_val, *op, &right_val)
            }
            FilterExpression::Unary { op, operand } => {
                let val = self.eval_expr(operand, chunk, row);
                self.eval_unary_op(*op, val)
            }
            FilterExpression::Id(var)
            | FilterExpression::Labels(var)
            | FilterExpression::Type(var) => {
                // Treat Id/Labels/Type access as variable lookup for RDF
                let col_idx = *self.variable_columns.get(var)?;
                chunk.column(col_idx)?.get_value(row)
            }
            FilterExpression::FunctionCall { name, args } => {
                self.eval_function_call(name, args, chunk, row)
            }
            FilterExpression::List(items) => {
                let values: Vec<Value> = items
                    .iter()
                    .filter_map(|item| self.eval_expr(item, chunk, row))
                    .collect();
                Some(Value::List(values.into()))
            }
            FilterExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                if let Some(operand) = operand {
                    let operand = self.eval_expr(operand, chunk, row)?;
                    for (when, result) in when_clauses {
                        let when = self.eval_expr(when, chunk, row)?;
                        if rdf_values_equal(&operand, &when) {
                            return self.eval_expr(result, chunk, row);
                        }
                    }
                } else {
                    for (condition, result) in when_clauses {
                        let condition = self.eval_expr(condition, chunk, row)?;
                        if rdf_effective_boolean_value(&condition)? {
                            return self.eval_expr(result, chunk, row);
                        }
                    }
                }
                else_clause
                    .as_deref()
                    .and_then(|expression| self.eval_expr(expression, chunk, row))
                    .or(Some(Value::Null))
            }
            // These expression types are not commonly used in RDF FILTER clauses
            FilterExpression::Map(_)
            | FilterExpression::IndexAccess { .. }
            | FilterExpression::SliceAccess { .. }
            | FilterExpression::ListComprehension { .. }
            | FilterExpression::ListPredicate { .. }
            | FilterExpression::ExistsSubquery { .. }
            | FilterExpression::CountSubquery { .. }
            | FilterExpression::Reduce { .. } => None,
            _ => None,
        }
    }

    fn eval_binary_op(&self, left: &Value, op: BinaryFilterOp, right: &Value) -> Option<Value> {
        match op {
            BinaryFilterOp::And => Some(Value::Bool(
                rdf_effective_boolean_value(left)? && rdf_effective_boolean_value(right)?,
            )),
            BinaryFilterOp::Or => Some(Value::Bool(
                rdf_effective_boolean_value(left)? || rdf_effective_boolean_value(right)?,
            )),
            BinaryFilterOp::Xor => Some(Value::Bool(
                rdf_effective_boolean_value(left)? != rdf_effective_boolean_value(right)?,
            )),
            BinaryFilterOp::Eq => rdf_numeric_comparison(left, op, right)
                .or_else(|| compare_values(left, right, |ordering| ordering.is_eq())),
            BinaryFilterOp::Ne => rdf_numeric_comparison(left, op, right)
                .or_else(|| compare_values(left, right, |ordering| ordering.is_ne())),
            BinaryFilterOp::Lt => rdf_numeric_comparison(left, op, right)
                .or_else(|| compare_values(left, right, |ordering| ordering.is_lt())),
            BinaryFilterOp::Le => rdf_numeric_comparison(left, op, right)
                .or_else(|| compare_values(left, right, |ordering| ordering.is_le())),
            BinaryFilterOp::Gt => rdf_numeric_comparison(left, op, right)
                .or_else(|| compare_values(left, right, |ordering| ordering.is_gt())),
            BinaryFilterOp::Ge => rdf_numeric_comparison(left, op, right)
                .or_else(|| compare_values(left, right, |ordering| ordering.is_ge())),
            BinaryFilterOp::Add => RdfNumeric::from_compatible_value(left)?
                .checked_add(RdfNumeric::from_compatible_value(right)?)
                .map(RdfNumeric::into_value),
            BinaryFilterOp::Sub => RdfNumeric::from_compatible_value(left)?
                .checked_sub(RdfNumeric::from_compatible_value(right)?)
                .map(RdfNumeric::into_value),
            BinaryFilterOp::Mul => RdfNumeric::from_compatible_value(left)?
                .checked_mul(RdfNumeric::from_compatible_value(right)?)
                .map(RdfNumeric::into_value),
            BinaryFilterOp::Div => RdfNumeric::from_compatible_value(left)?
                .checked_div(RdfNumeric::from_compatible_value(right)?)
                .map(RdfNumeric::into_value),
            BinaryFilterOp::Mod => RdfNumeric::from_compatible_value(left)?
                .checked_rem(RdfNumeric::from_compatible_value(right)?)
                .map(RdfNumeric::into_value),
            BinaryFilterOp::Contains => match (left, right) {
                (Value::String(l), Value::String(r)) => Some(Value::Bool(l.contains(&**r))),
                _ => None,
            },
            BinaryFilterOp::StartsWith => match (left, right) {
                (Value::String(l), Value::String(r)) => Some(Value::Bool(l.starts_with(&**r))),
                _ => None,
            },
            BinaryFilterOp::EndsWith => match (left, right) {
                (Value::String(l), Value::String(r)) => Some(Value::Bool(l.ends_with(&**r))),
                _ => None,
            },
            BinaryFilterOp::In => {
                // Not implemented for RDF filter evaluation
                None
            }
            BinaryFilterOp::Regex => {
                // SPARQL REGEX(string, pattern) - returns true if string matches pattern
                match (left, right) {
                    (Value::String(text), Value::String(pattern)) => {
                        // Compile the regex pattern
                        match Regex::new(pattern) {
                            Ok(re) => Some(Value::Bool(re.is_match(text))),
                            Err(_) => None, // Invalid regex pattern
                        }
                    }
                    _ => None,
                }
            }
            BinaryFilterOp::Pow => {
                let base = RdfNumeric::from_compatible_value(left)?.as_f64();
                let exponent = RdfNumeric::from_compatible_value(right)?.as_f64();
                Some(Value::Float64(base.powf(exponent)))
            }
            BinaryFilterOp::Like => {
                match (left, right) {
                    (Value::String(s), Value::String(pattern)) => {
                        // Convert SQL LIKE pattern to regex
                        let mut re_pat = String::with_capacity(pattern.len() + 4);
                        re_pat.push('^');
                        let mut chars = pattern.chars().peekable();
                        while let Some(ch) = chars.next() {
                            match ch {
                                '%' => re_pat.push_str(".*"),
                                '_' => re_pat.push('.'),
                                '\\' => {
                                    if let Some(next) = chars.next() {
                                        if ".+*?^${}()|[]\\".contains(next) {
                                            re_pat.push('\\');
                                        }
                                        re_pat.push(next);
                                    }
                                }
                                _ => {
                                    if ".+*?^${}()|[]\\".contains(ch) {
                                        re_pat.push('\\');
                                    }
                                    re_pat.push(ch);
                                }
                            }
                        }
                        re_pat.push('$');
                        match Regex::new(&re_pat) {
                            Ok(re) => Some(Value::Bool(re.is_match(s))),
                            Err(_) => None,
                        }
                    }
                    _ => None,
                }
            }
            BinaryFilterOp::Concat => match (left, right) {
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
                        _ => return None,
                    };
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(a);
                    s.push_str(&b);
                    Some(Value::String(s.into()))
                }
                (other, Value::String(b)) => {
                    let a = match other {
                        Value::Int64(i) => i.to_string(),
                        Value::Float64(f) => f.to_string(),
                        Value::Bool(bo) => bo.to_string(),
                        _ => return None,
                    };
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(&a);
                    s.push_str(b);
                    Some(Value::String(s.into()))
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn eval_unary_op(&self, op: UnaryFilterOp, val: Option<Value>) -> Option<Value> {
        match op {
            UnaryFilterOp::Not => Some(Value::Bool(!rdf_effective_boolean_value(&val?)?)),
            UnaryFilterOp::IsNull => Some(Value::Bool(val.is_none())),
            UnaryFilterOp::IsNotNull => Some(Value::Bool(val.is_some())),
            UnaryFilterOp::Neg => Some(
                RdfNumeric::from_compatible_value(&val?)?
                    .negated()
                    .into_value(),
            ),
            _ => None,
        }
    }

    /// Evaluates SPARQL function calls.
    fn eval_function_call(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        if name == RDF_NUMERIC_VALUE {
            let value = self.eval_expr(args.first()?, chunk, row)?;
            return RdfNumeric::from_compatible_value(&value).map(RdfNumeric::into_value);
        }

        if name == RDF_TERM_IN {
            let left = self.eval_expr(args.first()?, chunk, row)?;
            let (left_visible, left_term) = decode_tagged_rdf_filter_term(&left)?;
            let mut saw_error = false;
            for argument in &args[1..] {
                let Some(right) = self.eval_expr(argument, chunk, row) else {
                    saw_error = true;
                    continue;
                };
                let Some((right_visible, right_term)) = decode_tagged_rdf_filter_term(&right)
                else {
                    saw_error = true;
                    continue;
                };
                match rdf_terms_value_equal(left_visible, &left_term, right_visible, &right_term) {
                    Some(true) => return Some(Value::Bool(true)),
                    Some(false) => {}
                    None => saw_error = true,
                }
            }
            return if saw_error {
                None
            } else {
                Some(Value::Bool(false))
            };
        }

        if matches!(name, RDF_TERM_EQUAL | RDF_SAME_TERM) {
            let left = self.eval_expr(args.first()?, chunk, row)?;
            let right = self.eval_expr(args.get(1)?, chunk, row)?;
            let (left_visible, left_term) = decode_tagged_rdf_filter_term(&left)?;
            let (right_visible, right_term) = decode_tagged_rdf_filter_term(&right)?;
            let equal = if name == RDF_SAME_TERM {
                rdf_terms_same(&left_term, &right_term)
            } else {
                rdf_terms_value_equal(left_visible, &left_term, right_visible, &right_term)?
            };
            return Some(Value::Bool(equal));
        }

        let kind_test = match name {
            RDF_IS_IRI => Some(RdfTermKindTest::Iri),
            RDF_IS_BLANK => Some(RdfTermKindTest::Blank),
            RDF_IS_LITERAL => Some(RdfTermKindTest::Literal),
            RDF_IS_NUMERIC => Some(RdfTermKindTest::Numeric),
            _ => None,
        };
        if let Some(kind_test) = kind_test {
            let value = self.eval_expr(args.first()?, chunk, row)?;
            let (_, term) = decode_tagged_rdf_filter_term(&value)?;
            return Some(Value::Bool(match kind_test {
                RdfTermKindTest::Iri => term.is_iri(),
                RdfTermKindTest::Blank => term.is_blank_node(),
                RdfTermKindTest::Literal => term.is_literal(),
                RdfTermKindTest::Numeric => match term {
                    Term::Literal(ref literal) => rdf_numeric_literal_is_valid(literal),
                    _ => false,
                },
            }));
        }

        if name == RDF_TERM_IDENTITY_KEY {
            let value = self.eval_expr(args.first()?, chunk, row)?;
            let (_, term) = decode_tagged_rdf_filter_term(&value)?;
            return Some(Value::String(rdf_term_identity_key(&term).into()));
        }

        if name == RDF_DISTINCT_TERM_OR_VALUE_KEY {
            let value = self.eval_expr(args.first()?, chunk, row)?;
            if value.is_null() {
                return Some(Value::Null);
            }
            let (canonical_rdf, identity) =
                if let Some((_, term)) = decode_tagged_rdf_filter_term(&value) {
                    (true, Value::String(rdf_term_identity_key(&term).into()))
                } else {
                    (false, value)
                };
            return Some(Value::List(
                vec![Value::Bool(canonical_rdf), identity].into(),
            ));
        }

        if name == RDF_IDENTITY_OR_NATIVE_KEY {
            let visible = self.eval_expr(args.first()?, chunk, row)?;
            let (existing, identity) = if args.len() >= 3 {
                (
                    args.get(1)
                        .and_then(|expression| self.eval_expr(expression, chunk, row)),
                    args.get(2)
                        .and_then(|expression| self.eval_expr(expression, chunk, row)),
                )
            } else {
                (
                    None,
                    args.get(1)
                        .and_then(|expression| self.eval_expr(expression, chunk, row)),
                )
            };
            return Some(normalized_rdf_or_native_key(
                &visible,
                existing.as_ref(),
                identity.as_ref(),
            ));
        }

        let tagger = match name {
            RDF_TAG_IRI_TERM => Some(RdfTermTagger::Iri),
            RDF_TAG_BLANK_TERM => Some(RdfTermTagger::Blank),
            RDF_TAG_LITERAL_TERM => Some(RdfTermTagger::Literal),
            _ => None,
        };
        if let Some(tagger) = tagger {
            let Some(value) = self.eval_expr(args.first()?, chunk, row) else {
                return Some(Value::Null);
            };
            if value.is_null() {
                return Some(Value::Null);
            }
            let term = match tagger {
                RdfTermTagger::Iri => Term::iri(value.as_str()?),
                RdfTermTagger::Blank => {
                    let label = value
                        .as_str()?
                        .strip_prefix("_:")
                        .unwrap_or(value.as_str()?);
                    Term::blank(label)
                }
                RdfTermTagger::Literal => value_as_rdf_term(&value),
            };
            return Some(tagged_rdf_term(value, term));
        }

        if matches!(name, RDF_TAG_LANG_LITERAL_TERM | RDF_TAG_TYPED_LITERAL_TERM) {
            let first = self.eval_expr(args.first()?, chunk, row)?;
            let second = self.eval_expr(args.get(1)?, chunk, row)?;
            if first.is_null() || second.is_null() {
                return Some(Value::Null);
            }
            let lexical = value_to_string(&first);
            let (visible, term) = if name == RDF_TAG_LANG_LITERAL_TERM {
                let annotation = value_to_string(&second);
                let normalized_language = annotation.to_ascii_lowercase();
                (
                    Value::RdfLiteral {
                        lexical: lexical.clone().into(),
                        language: Some(normalized_language.into()),
                        datatype: None,
                    },
                    Term::lang_literal(lexical, annotation),
                )
            } else {
                let (_, datatype_term) = decode_tagged_rdf_filter_term(&second)?;
                let Term::Iri(datatype) = datatype_term else {
                    return None;
                };
                let annotation = datatype.as_str().to_string();
                let visible = strdt_visible_value(&lexical, &annotation).unwrap_or_else(|| {
                    Value::RdfLiteral {
                        lexical: lexical.clone().into(),
                        datatype: Some(annotation.clone().into()),
                        language: None,
                    }
                });
                (visible, Term::typed_literal(lexical, annotation))
            };
            return Some(tagged_rdf_term(visible, term));
        }

        if name == RDF_TAG_BOUND_TERM {
            let Some(visible_expression) = args.first() else {
                return Some(Value::Null);
            };
            let Some(visible) = self.eval_expr(visible_expression, chunk, row) else {
                return Some(Value::Null);
            };
            if visible.is_null() {
                return Some(Value::Null);
            }
            let exact = args
                .get(1)
                .and_then(|expression| self.eval_expr(expression, chunk, row))
                .unwrap_or(Value::Null);
            return Some(Value::List(
                vec![
                    visible,
                    exact,
                    Value::String(INTERNAL_RDF_TAGGED_TERM_MARKER.into()),
                ]
                .into(),
            ));
        }

        if matches!(name, RDF_TERM_OR_NATIVE_VISIBLE | RDF_TERM_OR_NATIVE_EXACT) {
            let value = self.eval_expr(args.first()?, chunk, row)?;
            let Some((visible, _)) = decode_tagged_rdf_filter_term(&value) else {
                return Some(if name == RDF_TERM_OR_NATIVE_VISIBLE {
                    value
                } else {
                    Value::Null
                });
            };
            if name == RDF_TERM_OR_NATIVE_VISIBLE {
                return Some(visible.clone());
            }
            let Value::List(values) = value else {
                return None;
            };
            return values.get(1).cloned().or(Some(Value::Null));
        }

        if name == RDF_TERM_OR_NATIVE_VALUE {
            let value = self.eval_expr(args.first()?, chunk, row)?;
            if decode_tagged_rdf_filter_term(&value).is_some() {
                return Some(value);
            }
            if let Value::List(values) = &value
                && let [visible, _, Value::String(marker)] = values.as_ref()
                && marker.as_str() == INTERNAL_RDF_TAGGED_TERM_MARKER
            {
                return Some(visible.clone());
            }
            return Some(value);
        }

        if matches!(name, RDF_TAG_VALUE | RDF_TAG_EXACT) {
            let tagged = self.eval_expr(args.first()?, chunk, row)?;
            let Value::List(values) = tagged else {
                return Some(Value::Null);
            };
            let index = usize::from(name == RDF_TAG_EXACT);
            return values.get(index).cloned().or(Some(Value::Null));
        }

        // Normalize function name to uppercase for case-insensitive matching
        let func_name = name.to_uppercase();
        // Public SPARQL extraction receives one lossless tagged operand. Keep
        // the native fallback for internal callers, without evaluating twice.
        let temporal_value = if matches!(
            func_name.as_str(),
            "YEAR" | "MONTH" | "DAY" | "HOURS" | "MINUTES" | "SECONDS" | "TIMEZONE" | "TZ"
        ) {
            let value = self.eval_expr(args.first()?, chunk, row)?;
            if let Some((_, term)) = decode_tagged_rdf_filter_term(&value) {
                return exact_datetime_component(&func_name, &term);
            }
            if matches!(value, Value::RdfLiteral { .. }) {
                return exact_datetime_component(&func_name, &value_as_rdf_term(&value));
            }
            if matches!(value, Value::List(_)) {
                return None;
            }
            Some(value)
        } else {
            None
        };

        match func_name.as_str() {
            // CONCAT - concatenate multiple strings
            "CONCAT" => {
                let mut result = String::new();
                for arg in args {
                    if let Some(value) = self.eval_expr(arg, chunk, row) {
                        match value {
                            Value::String(string) => result.push_str(&string),
                            other => result.push_str(&value_to_string(&other)),
                        }
                    }
                }
                Some(Value::String(result.into()))
            }

            // REPLACE - replace occurrences of pattern with replacement
            "REPLACE" => {
                if args.len() < 3 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let pattern = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let replacement = match self.eval_expr(&args[2], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };

                // Check if the pattern should be treated as regex (4th argument with 'r' flag)
                if args.len() >= 4
                    && let Some(Value::String(flags)) = self.eval_expr(&args[3], chunk, row)
                    && (flags.contains('r') || flags.contains('i'))
                {
                    // Regex-based replace
                    let regex_pattern = if flags.contains('i') {
                        format!("(?i){}", pattern)
                    } else {
                        pattern.clone()
                    };
                    if let Ok(re) = Regex::new(&regex_pattern) {
                        return Some(Value::String(re.replace_all(&text, &replacement).into()));
                    }
                }

                // Simple string replace
                Some(Value::String(text.replace(&pattern, &replacement).into()))
            }

            // STRLEN - string length
            "STRLEN" => {
                if args.is_empty() {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                // reason: string length will not exceed i64::MAX
                #[allow(clippy::cast_possible_wrap)]
                Some(Value::Int64(text.chars().count() as i64))
            }

            // UCASE - uppercase
            "UCASE" | "UPPER" => {
                if args.is_empty() {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                Some(Value::String(text.to_uppercase().into()))
            }

            // LCASE - lowercase
            "LCASE" | "LOWER" => {
                if args.is_empty() {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                Some(Value::String(text.to_lowercase().into()))
            }

            // SUBSTR - substring extraction
            "SUBSTR" | "SUBSTRING" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let start = match self.eval_expr(&args[1], chunk, row)? {
                    // reason: clamped to >= 0 by .max(1) - 1
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    Value::Int64(i) => (i.max(1) - 1) as usize, // SPARQL uses 1-based indexing
                    _ => return None,
                };
                let len = if args.len() >= 3 {
                    match self.eval_expr(&args[2], chunk, row)? {
                        // reason: clamped to >= 0 by .max(0)
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        Value::Int64(i) => Some(i.max(0) as usize),
                        _ => return None,
                    }
                } else {
                    None
                };

                let chars: Vec<char> = text.chars().collect();
                let substr: String = if let Some(len) = len {
                    chars.iter().skip(start).take(len).collect()
                } else {
                    chars.iter().skip(start).collect()
                };
                Some(Value::String(substr.into()))
            }

            // STRSTARTS - check if string starts with prefix
            "STRSTARTS" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let prefix = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                Some(Value::Bool(text.starts_with(&prefix)))
            }

            // STRENDS - check if string ends with suffix
            "STRENDS" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let suffix = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                Some(Value::Bool(text.ends_with(&suffix)))
            }

            // CONTAINS - check if string contains substring
            "CONTAINS" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let pattern = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                Some(Value::Bool(text.contains(&pattern)))
            }

            // STRBEFORE - substring before pattern
            "STRBEFORE" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let pattern = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                if let Some(pos) = text.find(&pattern) {
                    Some(Value::String(text[..pos].to_string().into()))
                } else {
                    Some(Value::String("".into()))
                }
            }

            // STRAFTER - substring after pattern
            "STRAFTER" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let pattern = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                if let Some(pos) = text.find(&pattern) {
                    Some(Value::String(
                        text[pos + pattern.len()..].to_string().into(),
                    ))
                } else {
                    Some(Value::String("".into()))
                }
            }

            // ENCODE_FOR_URI - URL encode
            "ENCODE_FOR_URI" => {
                if args.is_empty() {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                // Simple URL encoding for common characters
                let encoded: String = text
                    .chars()
                    .map(|c| match c {
                        'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
                        _ => format!("%{:02X}", c as u32),
                    })
                    .collect();
                Some(Value::String(encoded.into()))
            }

            // COALESCE - return first non-null value
            "COALESCE" => {
                for arg in args {
                    if let Some(val) = self.eval_expr(arg, chunk, row)
                        && !matches!(val, Value::Null)
                    {
                        return Some(val);
                    }
                }
                None
            }

            // IF - conditional expression
            "IF" => {
                if args.len() < 3 {
                    return None;
                }
                let condition = self.eval_expr(&args[0], chunk, row)?;
                if rdf_effective_boolean_value(&condition)? {
                    self.eval_expr(&args[1], chunk, row)
                } else {
                    self.eval_expr(&args[2], chunk, row)
                }
            }

            // BOUND - check if variable is bound
            "BOUND" => {
                if args.is_empty() {
                    return None;
                }
                // For variable arguments, check the validity bitmap directly so
                // that NULL entries from LEFT JOIN (OPTIONAL) are recognized as
                // "unbound" rather than "bound to Null".
                if let FilterExpression::Variable(var_name) = &args[0] {
                    if let Some(&col_idx) = self.variable_columns.get(var_name)
                        && let Some(col) = chunk.column(col_idx)
                    {
                        return Some(Value::Bool(!col.is_null(row)));
                    }
                    // Variable not in column map: unbound
                    return Some(Value::Bool(false));
                }
                // Non-variable arguments: fall back to expression evaluation
                let is_bound = self.eval_expr(&args[0], chunk, row).is_some();
                Some(Value::Bool(is_bound))
            }

            // STR - convert to string
            "STR" => {
                let argument = args.first()?;
                if let Some(term) = self.bound_scalar_term(argument, chunk, row).ok()? {
                    return match term {
                        Term::Iri(iri) => Some(Value::from(iri.as_str())),
                        Term::Literal(literal) => Some(Value::from(literal.value())),
                        _ => None,
                    };
                }
                let val = self.eval_expr(argument, chunk, row)?;
                Some(Value::String(value_to_string(&val).into()))
            }

            // ISIRI / ISURI - check if value is an IRI
            "ISIRI" | "ISURI" => {
                let argument = args.first()?;
                if let Some(term) = self.bound_scalar_term(argument, chunk, row).ok()? {
                    return Some(Value::Bool(term.is_iri()));
                }
                let val = self.eval_expr(argument, chunk, row)?;
                if let Value::String(s) = val {
                    // Check if it looks like an IRI (starts with a scheme)
                    let is_iri = s.contains("://") || s.starts_with("urn:");
                    Some(Value::Bool(is_iri))
                } else {
                    Some(Value::Bool(false))
                }
            }

            // ISBLANK - check if value is a blank node
            "ISBLANK" => {
                if args.is_empty() {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                if let Value::String(s) = val {
                    Some(Value::Bool(s.starts_with("_:")))
                } else {
                    Some(Value::Bool(false))
                }
            }

            // ISLITERAL - check if value is a literal
            "ISLITERAL" => {
                if args.is_empty() {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                // In our model, non-IRI strings and other values are literals
                match &val {
                    Value::String(s) => {
                        Some(Value::Bool(!s.contains("://") && !s.starts_with("_:")))
                    }
                    _ => Some(Value::Bool(true)),
                }
            }

            // ISNUMERIC - check if value is numeric
            "ISNUMERIC" => {
                let val = self.eval_expr(&args[0], chunk, row)?;
                let is_numeric = match &val {
                    Value::Int64(_) | Value::Float64(_) => true,
                    Value::RdfLiteral {
                        lexical,
                        language: None,
                        datatype: Some(datatype),
                    } => rdf_numeric_literal_is_valid(&Literal::typed(
                        lexical.as_str(),
                        datatype.as_str(),
                    )),
                    _ => false,
                };
                Some(Value::Bool(is_numeric))
            }

            // ABS - absolute value
            "ABS" => {
                if args.is_empty() {
                    return None;
                }
                Some(
                    RdfNumeric::from_compatible_value(&self.eval_expr(&args[0], chunk, row)?)?
                        .absolute()
                        .into_value(),
                )
            }

            // CEIL - ceiling
            "CEIL" => {
                if args.is_empty() {
                    return None;
                }
                Some(
                    RdfNumeric::from_compatible_value(&self.eval_expr(&args[0], chunk, row)?)?
                        .ceiling()
                        .into_value(),
                )
            }

            // FLOOR - floor
            "FLOOR" => {
                if args.is_empty() {
                    return None;
                }
                Some(
                    RdfNumeric::from_compatible_value(&self.eval_expr(&args[0], chunk, row)?)?
                        .floor()
                        .into_value(),
                )
            }

            // ROUND - round to nearest integer
            "ROUND" => {
                if args.is_empty() {
                    return None;
                }
                Some(
                    RdfNumeric::from_compatible_value(&self.eval_expr(&args[0], chunk, row)?)?
                        .rounded()
                        .into_value(),
                )
            }

            // REGEX - regular expression matching
            "REGEX" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let pattern = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    _ => return None,
                };
                // Optional flags argument (3rd arg): "i" for case-insensitive
                let regex_pattern = if args.len() >= 3
                    && let Some(Value::String(flags)) = self.eval_expr(&args[2], chunk, row)
                    && flags.contains('i')
                {
                    format!("(?i){pattern}")
                } else {
                    pattern
                };
                match Regex::new(&regex_pattern) {
                    Ok(re) => Some(Value::Bool(re.is_match(&text))),
                    Err(_) => None,
                }
            }

            // ================================================================
            // Date/Time Functions (SPARQL 1.1 Section 17.4.5)
            // ================================================================

            // NOW - current datetime
            "NOW" => {
                let ts = grafeo_common::types::Timestamp::now();
                Some(Value::Timestamp(ts))
            }

            // YEAR - extract year from date/datetime
            "YEAR" => {
                let val = temporal_value?;
                match val {
                    Value::Date(d) => Some(Value::Int64(i64::from(d.year()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_date().year()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_date().year())))
                    }
                    Value::String(s) => {
                        // Try full dateTime parse first (handles "2024-06-15T10:30:45+02:00"),
                        // then fall back to date-only parse (handles "2024-06-15").
                        if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(&s) {
                            Some(Value::Int64(i64::from(zdt.to_local_date().year())))
                        } else {
                            parse_datetime_date(&s).map(|d| Value::Int64(i64::from(d.year())))
                        }
                    }
                    _ => None,
                }
            }

            // MONTH - extract month from date/datetime
            "MONTH" => {
                let val = temporal_value?;
                match val {
                    Value::Date(d) => Some(Value::Int64(i64::from(d.month()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_date().month()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_date().month())))
                    }
                    Value::String(s) => {
                        if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(&s) {
                            Some(Value::Int64(i64::from(zdt.to_local_date().month())))
                        } else {
                            parse_datetime_date(&s).map(|d| Value::Int64(i64::from(d.month())))
                        }
                    }
                    _ => None,
                }
            }

            // DAY - extract day from date/datetime
            "DAY" => {
                let val = temporal_value?;
                match val {
                    Value::Date(d) => Some(Value::Int64(i64::from(d.day()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_date().day()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_date().day())))
                    }
                    Value::String(s) => {
                        if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(&s) {
                            Some(Value::Int64(i64::from(zdt.to_local_date().day())))
                        } else {
                            parse_datetime_date(&s).map(|d| Value::Int64(i64::from(d.day())))
                        }
                    }
                    _ => None,
                }
            }

            // HOURS - extract hours from time/datetime
            "HOURS" => {
                let val = temporal_value?;
                match val {
                    Value::Time(t) => Some(Value::Int64(i64::from(t.hour()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_time().hour()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_time().hour())))
                    }
                    Value::String(s) => {
                        if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(&s) {
                            Some(Value::Int64(i64::from(zdt.to_local_time().hour())))
                        } else {
                            parse_datetime_time(&s).map(|t| Value::Int64(i64::from(t.hour())))
                        }
                    }
                    _ => None,
                }
            }

            // MINUTES - extract minutes from time/datetime
            "MINUTES" => {
                let val = temporal_value?;
                match val {
                    Value::Time(t) => Some(Value::Int64(i64::from(t.minute()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_time().minute()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_time().minute())))
                    }
                    Value::String(s) => {
                        if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(&s) {
                            Some(Value::Int64(i64::from(zdt.to_local_time().minute())))
                        } else {
                            parse_datetime_time(&s).map(|t| Value::Int64(i64::from(t.minute())))
                        }
                    }
                    _ => None,
                }
            }

            // SECONDS - extract seconds (with fractional) from time/datetime
            "SECONDS" => {
                let val = temporal_value?;
                let to_secs = |t: &grafeo_common::types::Time| {
                    f64::from(t.second()) + f64::from(t.nanosecond()) / 1_000_000_000.0
                };
                match val {
                    Value::Time(t) => Some(Value::Float64(to_secs(&t))),
                    Value::Timestamp(ts) => Some(Value::Float64(to_secs(&ts.to_time()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Float64(to_secs(&zdt.to_local_time())))
                    }
                    Value::String(s) => {
                        if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(&s) {
                            Some(Value::Float64(to_secs(&zdt.to_local_time())))
                        } else {
                            grafeo_common::types::Time::parse(&s)
                                .map(|t| Value::Float64(to_secs(&t)))
                        }
                    }
                    _ => None,
                }
            }

            // TIMEZONE - extract timezone as xsd:dayTimeDuration
            "TIMEZONE" => {
                let val = temporal_value?;
                match val {
                    Value::Time(t) => t.offset_seconds().map(|offset| {
                        Value::Duration(grafeo_common::types::Duration::from_seconds(i64::from(
                            offset,
                        )))
                    }),
                    Value::ZonedDatetime(zdt) => Some(Value::Duration(
                        grafeo_common::types::Duration::from_seconds(i64::from(
                            zdt.offset_seconds(),
                        )),
                    )),
                    Value::String(s) => grafeo_common::types::ZonedDatetime::parse(&s).map(|zdt| {
                        Value::Duration(grafeo_common::types::Duration::from_seconds(i64::from(
                            zdt.offset_seconds(),
                        )))
                    }),
                    _ => None,
                }
            }

            // TZ - extract timezone as string ("+05:00", "Z", "")
            "TZ" => {
                let val = temporal_value?;
                match val {
                    Value::Time(t) => {
                        if let Some(offset) = t.offset_seconds() {
                            Some(Value::String(format_tz_offset(offset).into()))
                        } else {
                            Some(Value::String(String::new().into()))
                        }
                    }
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::String(format_tz_offset(zdt.offset_seconds()).into()))
                    }
                    Value::String(s) => {
                        if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(&s) {
                            Some(Value::String(format_tz_offset(zdt.offset_seconds()).into()))
                        } else {
                            Some(Value::String(String::new().into()))
                        }
                    }
                    _ => Some(Value::String(String::new().into())),
                }
            }

            // ================================================================
            // Hash Functions (SPARQL 1.1 Section 17.4.4)
            // ================================================================
            "MD5" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let s = value_to_string(&val);
                let digest = md5::compute(s.as_bytes());
                Some(Value::String(format!("{digest:x}").into()))
            }

            "SHA1" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let s = value_to_string(&val);
                use sha1::Digest as _;
                let hash = sha1::Sha1::digest(s.as_bytes());
                let hex = hash.iter().fold(String::new(), |mut s, b| {
                    use std::fmt::Write as _;
                    let _ = write!(s, "{b:02x}");
                    s
                });
                Some(Value::String(hex.into()))
            }

            "SHA256" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let s = value_to_string(&val);
                use sha2::Digest as _;
                let hash = sha2::Sha256::digest(s.as_bytes());
                let hex = hash.iter().fold(String::new(), |mut s, b| {
                    use std::fmt::Write as _;
                    let _ = write!(s, "{b:02x}");
                    s
                });
                Some(Value::String(hex.into()))
            }

            "SHA384" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let s = value_to_string(&val);
                use sha2::Digest as _;
                let hash = sha2::Sha384::digest(s.as_bytes());
                let hex = hash.iter().fold(String::new(), |mut s, b| {
                    use std::fmt::Write as _;
                    let _ = write!(s, "{b:02x}");
                    s
                });
                Some(Value::String(hex.into()))
            }

            "SHA512" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let s = value_to_string(&val);
                use sha2::Digest as _;
                let hash = sha2::Sha512::digest(s.as_bytes());
                let hex = hash.iter().fold(String::new(), |mut s, b| {
                    use std::fmt::Write as _;
                    let _ = write!(s, "{b:02x}");
                    s
                });
                Some(Value::String(hex.into()))
            }

            // ================================================================
            // RDF Term Functions (SPARQL 1.1 Section 17.4.2)
            // ================================================================

            // LANG - language tag of a literal
            "LANG" => {
                let argument = args.first()?;
                if let Some(argument) = args.first()
                    && let Some(term) = self.bound_scalar_term(argument, chunk, row).ok()?
                {
                    return match term {
                        Term::Literal(literal) => {
                            Some(Value::from(literal.language().unwrap_or("")))
                        }
                        _ => None,
                    };
                }
                // An unbound variable is a type error; do not synthesize an
                // empty language tag from the absence of a companion column.
                self.eval_expr(argument, chunk, row)?;
                // Look up the companion language-tag column for the variable.
                // The triple scan emits a hidden __lang_<var> column alongside
                // each object variable.
                if let Some(FilterExpression::Variable(var_name)) = args.first() {
                    let lang_col_name = format!("__lang_{var_name}");
                    if let Some(&col_idx) = self.variable_columns.get(&lang_col_name)
                        && let Some(col) = chunk.column(col_idx)
                        && let Some(Value::String(tag)) = col.get_value(row)
                    {
                        return Some(Value::String(tag));
                    }
                }
                // No language tag found: return empty string per SPARQL spec
                Some(Value::String(String::new().into()))
            }

            // LANGMATCHES - BCP47 language range matching (Section 17.4.2.9)
            "LANGMATCHES" => {
                if args.len() < 2 {
                    return None;
                }
                let tag = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    _ => return None,
                };
                let range = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    _ => return None,
                };

                // SPARQL spec: LANGMATCHES(tag, "*") matches any non-empty tag
                if range == "*" {
                    return Some(Value::Bool(!tag.is_empty()));
                }

                // Case-insensitive prefix match per RFC 4647 basic filtering:
                // tag "en-US" matches range "en" because "en" is a prefix and
                // the next character in tag is '-'.
                let tag_lower = tag.to_lowercase();
                let range_lower = range.to_lowercase();
                let matches = tag_lower == range_lower
                    || (tag_lower.starts_with(&range_lower)
                        && tag_lower.as_bytes().get(range_lower.len()) == Some(&b'-'));
                Some(Value::Bool(matches))
            }

            // DATATYPE - datatype IRI of a literal
            "DATATYPE" => {
                if let Some(argument) = args.first()
                    && let Some(term) = self.bound_scalar_term(argument, chunk, row).ok()?
                {
                    return match term {
                        Term::Literal(literal) => Some(Value::from(literal.datatype())),
                        _ => None,
                    };
                }
                if let Some(FilterExpression::Variable(var_name)) = args.first() {
                    let dt_col_name = format!("__datatype_{var_name}");
                    if let Some(&col_idx) = self.variable_columns.get(&dt_col_name)
                        && let Some(col) = chunk.column(col_idx)
                        && let Some(Value::String(dt)) = col.get_value(row)
                        && !dt.is_empty()
                    {
                        return Some(Value::String(dt));
                    }
                }
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let dt = match &val {
                    Value::String(_) => "http://www.w3.org/2001/XMLSchema#string",
                    Value::Int64(_) => "http://www.w3.org/2001/XMLSchema#integer",
                    Value::Float64(_) => "http://www.w3.org/2001/XMLSchema#double",
                    Value::Bool(_) => "http://www.w3.org/2001/XMLSchema#boolean",
                    Value::Date(_) => "http://www.w3.org/2001/XMLSchema#date",
                    Value::Time(_) => "http://www.w3.org/2001/XMLSchema#time",
                    Value::Timestamp(_) => "http://www.w3.org/2001/XMLSchema#dateTime",
                    Value::Duration(_) => "http://www.w3.org/2001/XMLSchema#duration",
                    Value::RdfLiteral {
                        language: Some(_), ..
                    } => Literal::RDF_LANG_STRING,
                    Value::RdfLiteral {
                        language: None,
                        datatype: Some(datatype),
                        ..
                    } => datatype.as_str(),
                    _ => return None,
                };
                Some(Value::String(dt.to_string().into()))
            }

            // Native vector constructor extension. Each component follows the
            // RDF numeric conversion rules; an error in any component leaves
            // the SPARQL expression unbound.
            "VECTOR" => {
                let mut values = Vec::with_capacity(args.len());
                for argument in args {
                    let value = self.eval_expr(argument, chunk, row)?;
                    let numeric = RdfNumeric::from_compatible_value(&value)?;
                    values.push(numeric.as_f32()?);
                }
                Some(Value::Vector(values.into()))
            }

            // IRI / URI - construct an IRI
            "IRI" | "URI" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s)),
                    v => Some(Value::String(value_to_string(&v).into())),
                }
            }

            // BNODE - construct or retrieve a blank node
            "BNODE" => {
                if args.is_empty() {
                    // Generate a unique blank node ID
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static BNODE_COUNTER: AtomicU64 = AtomicU64::new(0);
                    let id = BNODE_COUNTER.fetch_add(1, Ordering::Relaxed);
                    Some(Value::String(format!("_:b{id}").into()))
                } else {
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    let label = value_to_string(&val);
                    Some(Value::String(format!("_:b{label}").into()))
                }
            }

            // STRDT - construct a typed literal
            "STRDT" => {
                if args.len() < 2 {
                    return None;
                }
                let lexical = self.eval_expr(&args[0], chunk, row)?;
                let datatype = self.eval_expr(&args[1], chunk, row)?;
                let lex_str = value_to_string(&lexical);
                let (_, datatype_term) = decode_tagged_rdf_filter_term(&datatype)?;
                let Term::Iri(datatype) = datatype_term else {
                    return None;
                };
                let dt_str = datatype.as_str().to_string();
                Some(
                    strdt_visible_value(&lex_str, &dt_str).unwrap_or_else(|| Value::RdfLiteral {
                        lexical: lex_str.into(),
                        datatype: Some(dt_str.into()),
                        language: None,
                    }),
                )
            }

            // STRLANG - construct a language-tagged literal
            "STRLANG" => {
                if args.len() < 2 {
                    return None;
                }
                let lexical = self.eval_expr(&args[0], chunk, row)?;
                let language = self.eval_expr(&args[1], chunk, row)?;
                if lexical.is_null() || language.is_null() {
                    return None;
                }
                let lexical = value_to_string(&lexical);
                let language = value_to_string(&language).to_ascii_lowercase();
                Some(Value::RdfLiteral {
                    lexical: lexical.into(),
                    language: Some(language.into()),
                    datatype: None,
                })
            }

            // UUID - generate a UUID IRI
            "UUID" => {
                use std::sync::atomic::{AtomicU64, Ordering};
                static UUID_COUNTER: AtomicU64 = AtomicU64::new(0);
                let id = UUID_COUNTER.fetch_add(1, Ordering::Relaxed);
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos());
                Some(Value::String(format!("urn:uuid:{ts:032x}-{id:04x}").into()))
            }

            // STRUUID - generate a UUID string (no urn: prefix)
            "STRUUID" => {
                use std::sync::atomic::{AtomicU64, Ordering};
                static STRUUID_COUNTER: AtomicU64 = AtomicU64::new(0);
                let id = STRUUID_COUNTER.fetch_add(1, Ordering::Relaxed);
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos());
                Some(Value::String(format!("{ts:032x}-{id:04x}").into()))
            }

            // sameTerm - strict RDF term equality
            "SAMETERM" => {
                if args.len() < 2 {
                    return None;
                }
                let a = self.eval_expr(&args[0], chunk, row)?;
                let b = self.eval_expr(&args[1], chunk, row)?;
                Some(Value::Bool(a == b))
            }

            // ================================================================
            // Numeric Functions (SPARQL 1.1 Section 17.4.4)
            // ================================================================

            // RAND - random double in [0, 1)
            "RAND" => {
                #[cfg(test)]
                RDF_VOLATILE_EVALUATIONS.with(|count| count.set(count.get().saturating_add(1)));
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                use std::sync::atomic::{AtomicU64, Ordering};
                static RAND_STATE: AtomicU64 = AtomicU64::new(0);
                let state = RAND_STATE.fetch_add(1, Ordering::Relaxed);
                let mut hasher = DefaultHasher::new();
                state.hash(&mut hasher);
                // reason: truncation is intentional for hash seed entropy
                #[allow(clippy::cast_possible_truncation)]
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0u64, |d| d.as_nanos() as u64)
                    .hash(&mut hasher);
                let bits = hasher.finish();
                let value = (bits >> 11) as f64 / (1u64 << 53) as f64;
                Some(Value::Float64(value))
            }

            // Unknown function
            _ => None,
        }
    }
}

/// Parses the date component from a dateTime or date string.
///
/// Handles both "YYYY-MM-DD" and "YYYY-MM-DDTHH:MM:SS..." forms by splitting
/// at 'T' and parsing only the date portion, then falling back to a plain date
/// parse.
fn parse_datetime_date(s: &str) -> Option<grafeo_common::types::Date> {
    if let Some(pos) = s.find('T').or_else(|| s.find('t')) {
        grafeo_common::types::Date::parse(&s[..pos])
    } else {
        grafeo_common::types::Date::parse(s)
    }
}

/// Parses the time component from a dateTime or time string.
///
/// Handles both "HH:MM:SS..." and "YYYY-MM-DDTHH:MM:SS..." forms by splitting
/// at 'T' and parsing only the time portion, then falling back to a plain time
/// parse.
fn parse_datetime_time(s: &str) -> Option<grafeo_common::types::Time> {
    if let Some(pos) = s.find('T').or_else(|| s.find('t')) {
        grafeo_common::types::Time::parse(&s[pos + 1..])
    } else {
        grafeo_common::types::Time::parse(s)
    }
}

/// Formats a timezone offset in seconds as "+HH:MM" or "Z".
fn format_tz_offset(offset_secs: i32) -> String {
    if offset_secs == 0 {
        return "Z".to_string();
    }
    let sign = if offset_secs >= 0 { '+' } else { '-' };
    let abs = offset_secs.unsigned_abs();
    let hours = abs / 3600;
    let minutes = (abs % 3600) / 60;
    format!("{sign}{hours:02}:{minutes:02}")
}

impl Predicate for RdfExpressionPredicate {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> std::result::Result<bool, OperatorError> {
        Ok(self
            .eval(chunk, row)
            .as_ref()
            .and_then(rdf_effective_boolean_value)
            == Some(true))
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Strips internal RDF companion columns from the final output.
///
/// If no internal columns are present, the operator and columns pass through
/// unchanged. Otherwise, a lightweight projection is inserted to remove them.
fn strip_internal_columns(
    operator: Box<dyn Operator>,
    columns: Vec<String>,
) -> (Box<dyn Operator>, Vec<String>) {
    let keep_indices: Vec<usize> = columns
        .iter()
        .enumerate()
        .filter(|(_, name)| !is_rdf_internal_physical_column(name))
        .map(|(i, _)| i)
        .collect();

    if keep_indices.len() == columns.len() {
        // Nothing to strip
        return (operator, columns);
    }

    let output_columns: Vec<String> = keep_indices.iter().map(|&i| columns[i].clone()).collect();
    let output_types: Vec<LogicalType> = keep_indices.iter().map(|_| LogicalType::Any).collect();

    let projections = keep_indices
        .into_iter()
        .map(RdfProjectExpr::Column)
        .collect();

    let stripped = Box::new(RdfProjectOperator::new(operator, projections, output_types));
    (stripped, output_columns)
}

/// Converts an RDF Term to a string for IRI/blank node representation.
fn term_to_string(term: &Term) -> String {
    match term {
        Term::Iri(iri) => iri.as_str().to_string(),
        Term::BlankNode(bnode) => format!("_:{}", bnode.id()),
        Term::Literal(lit) => lit.value().to_string(),
        _ => String::new(),
    }
}

/// Pushes an RDF term value to a column.
///
/// For RDF columns (which use String type), we always push as string to avoid
/// type mismatches. The typed literal's value is preserved as a string, and
/// numeric comparisons are handled at the filter level.
fn push_term_value(col: &mut grafeo_core::execution::ValueVector, term: &Term) {
    match term {
        Term::Iri(iri) => col.push_string(iri.as_str().to_string()),
        Term::BlankNode(bnode) => col.push_string(format!("_:{}", bnode.id())),
        Term::Literal(lit) => {
            if let Some(lang) = lit.language() {
                col.push_value(Value::RdfLiteral {
                    lexical: lit.value().into(),
                    language: Some(lang.into()),
                    datatype: None,
                });
            } else if lit.datatype() != Literal::XSD_STRING {
                col.push_value(Value::RdfLiteral {
                    lexical: lit.value().into(),
                    language: None,
                    datatype: Some(lit.datatype().into()),
                });
            } else {
                col.push_string(lit.value().to_string());
            }
        }
        _ => col.push_value(Value::Null),
    }
}

/// Converts a bound literal `Value` to an RDF term, preserving typed `RdfLiteral`s.
fn value_as_rdf_term(value: &Value) -> Term {
    match value {
        Value::String(s) => Term::literal(s.clone()),
        Value::Int64(n) => Term::Literal(Literal::integer(*n)),
        Value::Float64(f) => Term::typed_literal(xsd_double_lexical(*f), Literal::XSD_DOUBLE),
        Value::Bool(b) => Term::Literal(Literal::boolean(*b)),
        Value::Date(d) => Term::typed_literal(d.to_string(), Literal::XSD_DATE),
        Value::Time(t) => {
            Term::typed_literal(t.to_string(), "http://www.w3.org/2001/XMLSchema#time")
        }
        Value::Timestamp(ts) => Term::typed_literal(ts.to_string(), Literal::XSD_DATETIME),
        Value::ZonedDatetime(zdt) => Term::typed_literal(zdt.to_string(), Literal::XSD_DATETIME),
        Value::Duration(dur) => {
            Term::typed_literal(dur.to_string(), "http://www.w3.org/2001/XMLSchema#duration")
        }
        Value::RdfLiteral {
            lexical,
            language: Some(lang),
            ..
        } => Term::lang_literal(lexical.to_string(), lang.to_string()),
        Value::RdfLiteral {
            lexical,
            datatype: Some(dt),
            language: None,
        } => Term::typed_literal(lexical.to_string(), dt.to_string()),
        Value::RdfLiteral { lexical, .. } => Term::literal(lexical.to_string()),
        other => Term::literal(format!("{other:?}")),
    }
}

fn xsd_double_lexical(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value == f64::INFINITY {
        "INF".to_string()
    } else if value == f64::NEG_INFINITY {
        "-INF".to_string()
    } else {
        value.to_string()
    }
}

/// Seals a visible execution value together with its exact RDF identity.
fn tagged_rdf_term(visible: Value, term: Term) -> Value {
    Value::List(
        vec![
            visible,
            Value::String(term.to_ntriples().into()),
            Value::String(INTERNAL_RDF_TAGGED_TERM_MARKER.into()),
        ]
        .into(),
    )
}

fn decode_tagged_rdf_filter_term(value: &Value) -> Option<(&Value, Term)> {
    let Value::List(values) = value else {
        return None;
    };
    let [visible, Value::String(exact), Value::String(marker)] = values.as_ref() else {
        return None;
    };
    if marker.as_str() != INTERNAL_RDF_TAGGED_TERM_MARKER {
        return None;
    }
    Some((visible, Term::from_ntriples(exact.as_str())?))
}

fn rdf_term_identity_key(term: &Term) -> String {
    term.canonical_identity_key()
}

fn rdf_terms_value_equal(
    _left_visible: &Value,
    left: &Term,
    _right_visible: &Value,
    right: &Term,
) -> Option<bool> {
    match (left, right) {
        (Term::Iri(_), Term::Iri(_)) | (Term::BlankNode(_), Term::BlankNode(_)) => {
            Some(rdf_terms_same(left, right))
        }
        (Term::Literal(left), Term::Literal(right)) => {
            match (left.language(), right.language()) {
                (Some(left_language), Some(right_language)) => {
                    return Some(
                        left.value() == right.value()
                            && left_language.eq_ignore_ascii_case(right_language),
                    );
                }
                // Grafeo advertises the SPARQL LangTagAwareness extension:
                // a language-tagged literal and a non-language literal are
                // determinably different rather than an operator error.
                (Some(_), None) | (None, Some(_)) => return Some(false),
                (None, None) => {}
            }

            if left.datatype() == Literal::XSD_STRING && right.datatype() == Literal::XSD_STRING {
                return Some(left.value() == right.value());
            }

            if numeric_kind(left.datatype()).is_some() || numeric_kind(right.datatype()).is_some() {
                return rdf_numeric_literals_equal(left, right).or_else(|| {
                    rdf_terms_same(&Term::Literal(left.clone()), &Term::Literal(right.clone()))
                        .then_some(true)
                });
            }

            if left.datatype() == Literal::XSD_BOOLEAN && right.datatype() == Literal::XSD_BOOLEAN {
                return match (
                    parse_xsd_boolean(left.value()),
                    parse_xsd_boolean(right.value()),
                ) {
                    (Some(left), Some(right)) => Some(left == right),
                    _ => {
                        rdf_terms_same(&Term::Literal(left.clone()), &Term::Literal(right.clone()))
                            .then_some(true)
                    }
                };
            }

            if left.datatype() == Literal::XSD_DATE && right.datatype() == Literal::XSD_DATE {
                return xsd_dates_equal(left.value(), right.value()).or_else(|| {
                    rdf_terms_same(&Term::Literal(left.clone()), &Term::Literal(right.clone()))
                        .then_some(true)
                });
            }

            if left.datatype() == Literal::XSD_DATETIME && right.datatype() == Literal::XSD_DATETIME
            {
                return xsd_datetimes_equal(left.value(), right.value()).or_else(|| {
                    rdf_terms_same(&Term::Literal(left.clone()), &Term::Literal(right.clone()))
                        .then_some(true)
                });
            }

            if rdf_terms_same(&Term::Literal(left.clone()), &Term::Literal(right.clone())) {
                return Some(true);
            }

            // SPARQL operator mappings do not define value equality for
            // arbitrary typed literals. Preserve that expression error so
            // `!=` cannot accidentally select the row by negating `false`.
            None
        }
        _ => Some(false),
    }
}

fn rdf_terms_same(left: &Term, right: &Term) -> bool {
    match (left, right) {
        (Term::Literal(left), Term::Literal(right)) => {
            left.value() == right.value()
                && left.datatype() == right.datatype()
                && match (left.language(), right.language()) {
                    (Some(left), Some(right)) => left.eq_ignore_ascii_case(right),
                    (None, None) => true,
                    _ => false,
                }
        }
        _ => left == right,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum NumericKind {
    Integer,
    Decimal,
    Float,
    Double,
}

fn numeric_kind(datatype: &str) -> Option<NumericKind> {
    let local = datatype.strip_prefix(Literal::XSD)?;
    match local {
        "double" => Some(NumericKind::Double),
        "float" => Some(NumericKind::Float),
        "decimal" => Some(NumericKind::Decimal),
        "integer" | "nonPositiveInteger" | "negativeInteger" | "long" | "int" | "short"
        | "byte" | "nonNegativeInteger" | "unsignedLong" | "unsignedInt" | "unsignedShort"
        | "unsignedByte" | "positiveInteger" => Some(NumericKind::Integer),
        _ => None,
    }
}

pub(crate) fn rdf_numeric_literal_is_valid(literal: &Literal) -> bool {
    let Some(kind) = numeric_kind(literal.datatype()) else {
        return false;
    };
    match kind {
        NumericKind::Integer | NumericKind::Decimal => numeric_as_decimal(literal, kind).is_some(),
        NumericKind::Float => parse_xsd_float(literal.value()).is_some(),
        NumericKind::Double => parse_xsd_double(literal.value()).is_some(),
    }
}

#[derive(PartialEq, Eq)]
struct CanonicalDecimal {
    negative: bool,
    integer: String,
    fraction: String,
}

impl CanonicalDecimal {
    fn parse(lexical: &str, allow_fraction: bool) -> Option<Self> {
        let (negative, unsigned) = lexical.strip_prefix('-').map_or_else(
            || (false, lexical.strip_prefix('+').unwrap_or(lexical)),
            |rest| (true, rest),
        );
        let (integer, fraction) = match unsigned.split_once('.') {
            Some((integer, fraction)) if allow_fraction => (integer, fraction),
            Some(_) => return None,
            None => (unsigned, ""),
        };
        if (integer.is_empty() && fraction.is_empty())
            || !integer.bytes().all(|byte| byte.is_ascii_digit())
            || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        {
            return None;
        }
        let integer = integer.trim_start_matches('0');
        let fraction = fraction.trim_end_matches('0');
        let integer = if integer.is_empty() { "0" } else { integer };
        let zero = integer == "0" && fraction.is_empty();
        Some(Self {
            negative: negative && !zero,
            integer: integer.to_string(),
            fraction: fraction.to_string(),
        })
    }
}

fn rdf_numeric_literals_equal(left: &Literal, right: &Literal) -> Option<bool> {
    let left_kind = numeric_kind(left.datatype())?;
    let right_kind = numeric_kind(right.datatype())?;
    let promoted = left_kind.max(right_kind);

    match promoted {
        NumericKind::Integer | NumericKind::Decimal => {
            Some(numeric_as_decimal(left, left_kind)? == numeric_as_decimal(right, right_kind)?)
        }
        NumericKind::Float => {
            let left = numeric_as_f32(left, left_kind)?;
            let right = numeric_as_f32(right, right_kind)?;
            Some(left == right)
        }
        NumericKind::Double => {
            let left = numeric_as_f64_literal(left, left_kind)?;
            let right = numeric_as_f64_literal(right, right_kind)?;
            Some(left == right)
        }
    }
}

fn numeric_as_decimal(literal: &Literal, kind: NumericKind) -> Option<CanonicalDecimal> {
    let decimal = CanonicalDecimal::parse(literal.value(), kind == NumericKind::Decimal)?;
    if kind == NumericKind::Integer && !integer_facets_accept(literal, &decimal) {
        return None;
    }
    Some(decimal)
}

fn integer_facets_accept(literal: &Literal, value: &CanonicalDecimal) -> bool {
    let local = literal
        .datatype()
        .strip_prefix(Literal::XSD)
        .unwrap_or_default();
    let zero = value.integer == "0";
    match local {
        "integer" => true,
        "nonPositiveInteger" => value.negative || zero,
        "negativeInteger" => value.negative,
        "nonNegativeInteger" => !value.negative,
        "positiveInteger" => !value.negative && !zero,
        "long" => canonical_integer_in_range(value, i64::MIN as i128, i64::MAX as i128),
        "int" => canonical_integer_in_range(value, i32::MIN as i128, i32::MAX as i128),
        "short" => canonical_integer_in_range(value, i16::MIN as i128, i16::MAX as i128),
        "byte" => canonical_integer_in_range(value, i8::MIN as i128, i8::MAX as i128),
        "unsignedLong" => canonical_integer_in_range(value, 0, u64::MAX as i128),
        "unsignedInt" => canonical_integer_in_range(value, 0, u32::MAX as i128),
        "unsignedShort" => canonical_integer_in_range(value, 0, u16::MAX as i128),
        "unsignedByte" => canonical_integer_in_range(value, 0, u8::MAX as i128),
        _ => false,
    }
}

fn canonical_integer_in_range(value: &CanonicalDecimal, minimum: i128, maximum: i128) -> bool {
    let lexical = if value.negative {
        format!("-{}", value.integer)
    } else {
        value.integer.clone()
    };
    lexical
        .parse::<i128>()
        .is_ok_and(|value| (minimum..=maximum).contains(&value))
}

fn numeric_as_f32(literal: &Literal, kind: NumericKind) -> Option<f32> {
    match kind {
        NumericKind::Integer | NumericKind::Decimal => {
            numeric_as_decimal(literal, kind)?;
            literal.value().parse().ok()
        }
        NumericKind::Float => parse_xsd_float(literal.value()),
        NumericKind::Double => None,
    }
}

fn numeric_as_f64_literal(literal: &Literal, kind: NumericKind) -> Option<f64> {
    match kind {
        NumericKind::Integer | NumericKind::Decimal => {
            numeric_as_decimal(literal, kind)?;
            literal.value().parse().ok()
        }
        NumericKind::Float => parse_xsd_float(literal.value()).map(f64::from),
        NumericKind::Double => parse_xsd_double(literal.value()),
    }
}

fn parse_xsd_float(lexical: &str) -> Option<f32> {
    match lexical {
        "INF" | "+INF" => Some(f32::INFINITY),
        "-INF" => Some(f32::NEG_INFINITY),
        "NaN" => Some(f32::NAN),
        _ if is_xsd_finite_float_lexical(lexical) => lexical.parse().ok(),
        _ => None,
    }
}

fn parse_xsd_double(lexical: &str) -> Option<f64> {
    match lexical {
        "INF" | "+INF" => Some(f64::INFINITY),
        "-INF" => Some(f64::NEG_INFINITY),
        "NaN" => Some(f64::NAN),
        _ if is_xsd_finite_float_lexical(lexical) => lexical.parse().ok(),
        _ => None,
    }
}

fn is_xsd_finite_float_lexical(lexical: &str) -> bool {
    let bytes = lexical.as_bytes();
    let mut index = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    let integer_start = index;
    while bytes.get(index).is_some_and(u8::is_ascii_digit) {
        index += 1;
    }
    let integer_digits = index - integer_start;
    let mut fraction_digits = 0;
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        let fraction_start = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        fraction_digits = index - fraction_start;
    }
    if integer_digits == 0 && fraction_digits == 0 {
        return false;
    }
    if matches!(bytes.get(index), Some(b'e' | b'E')) {
        index += 1;
        if matches!(bytes.get(index), Some(b'+' | b'-')) {
            index += 1;
        }
        let exponent_start = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == exponent_start {
            return false;
        }
    }
    index == bytes.len()
}

fn parse_xsd_boolean(lexical: &str) -> Option<bool> {
    match lexical {
        "true" | "1" => Some(true),
        "false" | "0" => Some(false),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExactYear {
    negative: bool,
    magnitude: String,
}

impl ExactYear {
    fn parse(negative: bool, lexical: &str) -> Option<Self> {
        if lexical.len() < 4
            || !lexical.bytes().all(|byte| byte.is_ascii_digit())
            || (lexical.len() > 4 && lexical.starts_with('0'))
        {
            return None;
        }
        let magnitude = lexical.trim_start_matches('0');
        let magnitude = if magnitude.is_empty() { "0" } else { magnitude };
        Some(Self {
            negative: negative && magnitude != "0",
            magnitude: magnitude.to_string(),
        })
    }

    fn modulo(&self, modulus: u16) -> u16 {
        self.magnitude.bytes().fold(0, |remainder, digit| {
            (remainder * 10 + u16::from(digit - b'0')) % modulus
        })
    }

    fn is_leap(&self) -> bool {
        self.modulo(4) == 0 && (self.modulo(100) != 0 || self.modulo(400) == 0)
    }

    fn add_one(&mut self) {
        if self.negative {
            if self.magnitude == "1" {
                self.negative = false;
                self.magnitude = "0".to_string();
            } else {
                decrement_decimal_digits(&mut self.magnitude);
            }
        } else {
            increment_decimal_digits(&mut self.magnitude);
        }
    }

    fn subtract_one(&mut self) {
        if self.negative {
            increment_decimal_digits(&mut self.magnitude);
        } else if self.magnitude == "0" {
            self.negative = true;
            self.magnitude = "1".to_string();
        } else if self.magnitude == "1" {
            self.magnitude = "0".to_string();
        } else {
            decrement_decimal_digits(&mut self.magnitude);
        }
    }
}

// ExactYear constructs only nonempty ASCII decimal magnitudes. Rebuilding
// through chars preserves that representation without a fallible UTF-8 cast.
fn increment_decimal_digits(digits: &mut String) {
    let mut bytes = digits.as_bytes().to_vec();
    let mut index = bytes.len();
    while index > 0 {
        index -= 1;
        if bytes[index] < b'9' {
            bytes[index] += 1;
            digits.clear();
            digits.extend(bytes.into_iter().map(char::from));
            return;
        }
        bytes[index] = b'0';
    }
    bytes.insert(0, b'1');
    digits.clear();
    digits.extend(bytes.into_iter().map(char::from));
}

fn decrement_decimal_digits(digits: &mut String) {
    debug_assert!(digits.as_str() > "0");
    let mut bytes = digits.as_bytes().to_vec();
    let mut index = bytes.len();
    while index > 0 {
        index -= 1;
        if bytes[index] > b'0' {
            bytes[index] -= 1;
            break;
        }
        bytes[index] = b'9';
    }
    let first_nonzero = bytes
        .iter()
        .position(|byte| *byte != b'0')
        .unwrap_or(bytes.len() - 1);
    digits.clear();
    digits.extend(bytes[first_nonzero..].iter().copied().map(char::from));
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExactCalendarDate {
    year: ExactYear,
    month: u8,
    day: u8,
}

impl ExactCalendarDate {
    fn shift_days(mut self, days: i32) -> Self {
        match days.cmp(&0) {
            std::cmp::Ordering::Greater => {
                for _ in 0..days {
                    self.next_day();
                }
            }
            std::cmp::Ordering::Less => {
                for _ in days..0 {
                    self.previous_day();
                }
            }
            std::cmp::Ordering::Equal => {}
        }
        self
    }

    fn next_day(&mut self) {
        let days_in_month = days_in_gregorian_month(&self.year, self.month);
        if self.day < days_in_month {
            self.day += 1;
        } else if self.month < 12 {
            self.month += 1;
            self.day = 1;
        } else {
            self.year.add_one();
            self.month = 1;
            self.day = 1;
        }
    }

    fn previous_day(&mut self) {
        if self.day > 1 {
            self.day -= 1;
        } else if self.month > 1 {
            self.month -= 1;
            self.day = days_in_gregorian_month(&self.year, self.month);
        } else {
            self.year.subtract_one();
            self.month = 12;
            self.day = 31;
        }
    }
}

#[derive(PartialEq, Eq)]
struct ExactDate {
    date: ExactCalendarDate,
    offset_minutes: Option<i32>,
}

fn parse_exact_xsd_date(lexical: &str) -> Option<ExactDate> {
    let (date, suffix) = parse_xsd_date_prefix(lexical)?;
    let offset_minutes = match parse_xsd_timezone(suffix) {
        XsdTimezoneParse::Absent => None,
        XsdTimezoneParse::OffsetMinutes(offset) => Some(offset),
        XsdTimezoneParse::Invalid => return None,
    };
    Some(ExactDate {
        date,
        offset_minutes,
    })
}

fn parse_xsd_date_prefix(lexical: &str) -> Option<(ExactCalendarDate, &str)> {
    let bytes = lexical.as_bytes();
    let negative = bytes.first() == Some(&b'-');
    let mut index = usize::from(negative);
    let year_start = index;
    while bytes.get(index).is_some_and(u8::is_ascii_digit) {
        index += 1;
    }
    if index - year_start < 4 || bytes.get(index) != Some(&b'-') {
        return None;
    }
    let year_lexical = lexical.get(year_start..index)?;
    let year = ExactYear::parse(negative, year_lexical)?;
    index += 1;
    let month = parse_two_ascii_digits(bytes.get(index..index + 2)?)?;
    index += 2;
    if bytes.get(index) != Some(&b'-') {
        return None;
    }
    index += 1;
    let day = parse_two_ascii_digits(bytes.get(index..index + 2)?)?;
    index += 2;
    if !(1..=12).contains(&month) || !(1..=days_in_gregorian_month(&year, month)).contains(&day) {
        return None;
    }
    Some((
        ExactCalendarDate { year, month, day },
        lexical.get(index..)?,
    ))
}

fn parse_two_ascii_digits(bytes: &[u8]) -> Option<u8> {
    let [first, second] = bytes else {
        return None;
    };
    if !first.is_ascii_digit() || !second.is_ascii_digit() {
        return None;
    }
    Some((first - b'0') * 10 + (second - b'0'))
}

fn days_in_gregorian_month(year: &ExactYear, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if year.is_leap() => 29,
        2 => 28,
        _ => 0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum XsdTimezoneParse {
    Absent,
    OffsetMinutes(i32),
    Invalid,
}

fn parse_xsd_timezone(suffix: &str) -> XsdTimezoneParse {
    if suffix.is_empty() {
        return XsdTimezoneParse::Absent;
    }
    if suffix == "Z" {
        return XsdTimezoneParse::OffsetMinutes(0);
    }
    let bytes = suffix.as_bytes();
    if bytes.len() != 6 || !matches!(bytes[0], b'+' | b'-') || bytes[3] != b':' {
        return XsdTimezoneParse::Invalid;
    }
    let Some(hours) = parse_two_ascii_digits(&bytes[1..3]) else {
        return XsdTimezoneParse::Invalid;
    };
    let Some(minutes) = parse_two_ascii_digits(&bytes[4..6]) else {
        return XsdTimezoneParse::Invalid;
    };
    let hours = i32::from(hours);
    let minutes = i32::from(minutes);
    if hours > 14 || minutes > 59 || (hours == 14 && minutes != 0) {
        return XsdTimezoneParse::Invalid;
    }
    let sign = if bytes[0] == b'-' { -1 } else { 1 };
    XsdTimezoneParse::OffsetMinutes(sign * (hours * 60 + minutes))
}

fn xsd_dates_equal(left: &str, right: &str) -> Option<bool> {
    Some(compare_xsd_dates(left, right)? == Ordering::Equal)
}

fn normalize_exact_date(date: ExactDate) -> (ExactCalendarDate, i32) {
    // Grafeo's query context uses UTC as its implicit timezone when an XSD
    // date omits an offset.
    let utc_minutes = -date.offset_minutes.unwrap_or(0);
    (
        date.date.shift_days(utc_minutes.div_euclid(1_440)),
        utc_minutes.rem_euclid(1_440),
    )
}

pub(super) fn compare_xsd_dates(left: &str, right: &str) -> Option<Ordering> {
    let (left_date, left_minutes) = normalize_exact_date(parse_exact_xsd_date(left)?);
    let (right_date, right_minutes) = normalize_exact_date(parse_exact_xsd_date(right)?);
    Some(
        compare_exact_calendar_dates(&left_date, &right_date)
            .then_with(|| left_minutes.cmp(&right_minutes)),
    )
}

#[derive(PartialEq, Eq)]
struct ExactDateTime {
    date: ExactCalendarDate,
    second_of_day: i32,
    fraction: String,
    offset_minutes: Option<i32>,
}

fn parse_exact_xsd_datetime(lexical: &str) -> Option<ExactDateTime> {
    let (date, time_and_zone) = lexical.split_once('T')?;
    let (mut date, date_suffix) = parse_xsd_date_prefix(date)?;
    if !date_suffix.is_empty() {
        return None;
    }

    let bytes = time_and_zone.as_bytes();
    if bytes.len() < 8 || bytes.get(2) != Some(&b':') || bytes.get(5) != Some(&b':') {
        return None;
    }
    let mut hours = i32::from(parse_two_ascii_digits(&bytes[0..2])?);
    let minutes = i32::from(parse_two_ascii_digits(&bytes[3..5])?);
    let seconds = i32::from(parse_two_ascii_digits(&bytes[6..8])?);
    let mut index = 8;
    let fraction = if bytes.get(index) == Some(&b'.') {
        index += 1;
        let start = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == start {
            return None;
        }
        time_and_zone.get(start..index)?
    } else {
        ""
    };
    let offset_minutes = match parse_xsd_timezone(time_and_zone.get(index..)?) {
        XsdTimezoneParse::Absent => None,
        XsdTimezoneParse::OffsetMinutes(offset) => Some(offset),
        XsdTimezoneParse::Invalid => return None,
    };
    if minutes > 59 || seconds > 59 || hours > 24 {
        return None;
    }
    if hours == 24 {
        if minutes != 0 || seconds != 0 || fraction.bytes().any(|digit| digit != b'0') {
            return None;
        }
        date = date.shift_days(1);
        hours = 0;
    }
    Some(ExactDateTime {
        date,
        second_of_day: hours * 3_600 + minutes * 60 + seconds,
        fraction: fraction.trim_end_matches('0').to_string(),
        offset_minutes,
    })
}

/// Extracts local components without narrowing a year or decimal fraction.
fn exact_datetime_component(name: &str, term: &Term) -> Option<Value> {
    let Term::Literal(literal) = term else {
        return None;
    };
    if literal.datatype() != Literal::XSD_DATETIME || literal.language().is_some() {
        return None;
    }
    let lexical = literal.value();
    let datetime = parse_exact_xsd_datetime(lexical)?;
    let typed = |lexical: String, datatype: &str| Value::RdfLiteral {
        lexical: lexical.into(),
        datatype: Some(datatype.into()),
        language: None,
    };
    match name {
        "YEAR" => {
            let year = if datetime.date.year.negative {
                format!("-{}", datetime.date.year.magnitude)
            } else {
                datetime.date.year.magnitude
            };
            Some(
                year.parse::<i64>()
                    .map_or_else(|_| typed(year, Literal::XSD_INTEGER), Value::Int64),
            )
        }
        "MONTH" => Some(Value::Int64(i64::from(datetime.date.month))),
        "DAY" => Some(Value::Int64(i64::from(datetime.date.day))),
        "HOURS" => Some(Value::Int64(i64::from(datetime.second_of_day / 3_600))),
        "MINUTES" => Some(Value::Int64(i64::from(datetime.second_of_day / 60 % 60))),
        "SECONDS" => {
            let seconds = datetime.second_of_day % 60;
            let lexical = if datetime.fraction.is_empty() {
                seconds.to_string()
            } else {
                format!("{seconds}.{}", datetime.fraction)
            };
            Some(typed(lexical, Literal::XSD_DECIMAL))
        }
        "TIMEZONE" => {
            let seconds = i64::from(datetime.offset_minutes?) * 60;
            Some(typed(
                grafeo_common::types::Duration::from_seconds(seconds).to_string(),
                "http://www.w3.org/2001/XMLSchema#dayTimeDuration",
            ))
        }
        "TZ" => Some(Value::from(if datetime.offset_minutes.is_none() {
            ""
        } else if lexical.ends_with('Z') {
            "Z"
        } else {
            lexical.get(lexical.len().checked_sub(6)?..)?
        })),
        _ => None,
    }
}

fn xsd_datetimes_equal(left: &str, right: &str) -> Option<bool> {
    Some(compare_xsd_datetimes(left, right)? == Ordering::Equal)
}

fn normalize_exact_datetime(date_time: ExactDateTime) -> (ExactCalendarDate, i32, String) {
    // Grafeo's query context uses UTC as its implicit timezone when an XSD
    // dateTime omits an offset.
    let utc_seconds = date_time.second_of_day - date_time.offset_minutes.unwrap_or(0) * 60;
    (
        date_time.date.shift_days(utc_seconds.div_euclid(86_400)),
        utc_seconds.rem_euclid(86_400),
        date_time.fraction,
    )
}

pub(super) fn compare_xsd_datetimes(left: &str, right: &str) -> Option<Ordering> {
    let (left_date, left_seconds, left_fraction) =
        normalize_exact_datetime(parse_exact_xsd_datetime(left)?);
    let (right_date, right_seconds, right_fraction) =
        normalize_exact_datetime(parse_exact_xsd_datetime(right)?);
    Some(
        compare_exact_calendar_dates(&left_date, &right_date)
            .then_with(|| left_seconds.cmp(&right_seconds))
            .then_with(|| compare_decimal_fraction(&left_fraction, &right_fraction)),
    )
}

fn compare_exact_calendar_dates(left: &ExactCalendarDate, right: &ExactCalendarDate) -> Ordering {
    compare_exact_years(&left.year, &right.year)
        .then_with(|| left.month.cmp(&right.month))
        .then_with(|| left.day.cmp(&right.day))
}

fn compare_exact_years(left: &ExactYear, right: &ExactYear) -> Ordering {
    match (left.negative, right.negative) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => compare_unsigned_decimal(&left.magnitude, &right.magnitude),
        (true, true) => compare_unsigned_decimal(&right.magnitude, &left.magnitude),
    }
}

fn compare_unsigned_decimal(left: &str, right: &str) -> Ordering {
    left.len().cmp(&right.len()).then_with(|| left.cmp(right))
}

fn compare_decimal_fraction(left: &str, right: &str) -> Ordering {
    let width = left.len().max(right.len());
    (0..width)
        .map(|index| {
            left.as_bytes()
                .get(index)
                .copied()
                .unwrap_or(b'0')
                .cmp(&right.as_bytes().get(index).copied().unwrap_or(b'0'))
        })
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or(Ordering::Equal)
}

/// Instantiates a SPARQL update template only when the substituted terms are
/// legal in their RDF positions. Invalid solutions are omitted per SPARQL
/// Update semantics; they must never reach `Triple::new`'s debug assertions.
fn instantiate_mutation_triple(subject: Term, predicate: Term, object: Term) -> Option<Triple> {
    if !(subject.is_iri() || subject.is_blank_node()) || !predicate.is_iri() {
        return None;
    }
    Some(Triple::new(subject, predicate, object))
}

/// Resolves one RDF mutation-template component without discarding RDF term
/// identity. Every non-null variable binding must carry a sealed N-Triples
/// companion; visible strings are never guessed to be IRIs or literals.
fn resolve_mutation_component(
    component: &TripleComponent,
    column_map: &HashMap<String, usize>,
    chunk: &DataChunk,
    row: usize,
) -> std::result::Result<Option<Term>, OperatorError> {
    match component {
        TripleComponent::Iri(iri) => Ok(Some(Term::iri(iri.clone()))),
        TripleComponent::BlankNode(label) => Ok(Some(Term::blank(label.clone()))),
        TripleComponent::Literal(value) => Ok(Some(value_as_rdf_term(value))),
        TripleComponent::LangLiteral { value, lang } => {
            Ok(Some(Term::lang_literal(value.clone(), lang.clone())))
        }
        TripleComponent::Variable(name) => {
            let variable = name.strip_prefix('?').unwrap_or(name);
            let visible = column_map
                .get(variable)
                .and_then(|column_index| chunk.column(*column_index))
                .and_then(|column| column.get_value(row));
            if visible.as_ref().is_none_or(Value::is_null) {
                return Ok(None);
            }

            let exact_column = rdf_exact_term_column(variable);
            let Some(exact) = column_map
                .get(&exact_column)
                .and_then(|column_index| chunk.column(*column_index))
                .and_then(|column| column.get_value(row))
            else {
                return Err(OperatorError::Execution(format!(
                    "bound RDF mutation variable ?{variable} lacked sealed term identity"
                )));
            };
            if exact.is_null() {
                return Err(OperatorError::Execution(format!(
                    "bound RDF mutation variable ?{variable} had no exact RDF term identity"
                )));
            }
            let encoded = exact.as_str().ok_or_else(|| {
                OperatorError::Execution(format!(
                    "sealed RDF companion {exact_column} was not a string"
                ))
            })?;
            let term = Term::from_ntriples(encoded).ok_or_else(|| {
                OperatorError::Execution(format!(
                    "sealed RDF companion {exact_column} contained invalid N-Triples"
                ))
            })?;
            Ok(Some(term))
        }
    }
}

/// Preserves the stable programmatic InsertTripleOp/DeleteTripleOp behavior
/// for downstream logical plans that supply scalar bindings without the
/// parser-private sealed columns used by SPARQL MODIFY.
fn resolve_public_pattern_component(
    component: &TripleComponent,
    column_map: &HashMap<String, usize>,
    chunk: &DataChunk,
    row: usize,
) -> std::result::Result<Option<Term>, OperatorError> {
    let TripleComponent::Variable(name) = component else {
        return resolve_mutation_component(component, column_map, chunk, row);
    };
    let variable = name.strip_prefix('?').unwrap_or(name);
    let Some(value) = column_map
        .get(variable)
        .and_then(|column_index| chunk.column(*column_index))
        .and_then(|column| column.get_value(row))
    else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }

    let exact_column = rdf_exact_term_column(variable);
    if let Some(exact) = column_map
        .get(&exact_column)
        .and_then(|column_index| chunk.column(*column_index))
        .and_then(|column| column.get_value(row))
        .filter(|exact| !exact.is_null())
    {
        let encoded = exact.as_str().ok_or_else(|| {
            OperatorError::Execution(format!(
                "sealed RDF companion {exact_column} was not a string"
            ))
        })?;
        return Term::from_ntriples(encoded).map(Some).ok_or_else(|| {
            OperatorError::Execution(format!(
                "sealed RDF companion {exact_column} contained invalid N-Triples"
            ))
        });
    }

    legacy_bound_value_as_rdf_term(&value).map(Some)
}

fn legacy_bound_value_as_rdf_term(value: &Value) -> std::result::Result<Term, OperatorError> {
    match value {
        Value::String(value) => {
            if let Some(label) = value.strip_prefix("_:") {
                return Ok(Term::blank(label.to_string()));
            }
            let is_absolute_iri = value.split_once(':').is_some_and(|(scheme, remainder)| {
                !remainder.is_empty()
                    && scheme
                        .chars()
                        .next()
                        .is_some_and(|first| first.is_ascii_alphabetic())
                    && scheme
                        .chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '-' | '.'))
                    && !value.chars().any(char::is_whitespace)
            });
            Ok(if is_absolute_iri {
                Term::iri(value.to_string())
            } else {
                Term::literal(value.to_string())
            })
        }
        Value::Int64(_)
        | Value::Float64(_)
        | Value::Bool(_)
        | Value::Date(_)
        | Value::Time(_)
        | Value::Timestamp(_)
        | Value::ZonedDatetime(_)
        | Value::Duration(_)
        | Value::RdfLiteral { .. } => Ok(value_as_rdf_term(value)),
        unsupported => Err(OperatorError::Execution(format!(
            "bound value {unsupported:?} is not an RDF term"
        ))),
    }
}

/// Resolves a MODIFY template component, constructing INSERT-template blank
/// nodes freshly per solution while sharing a repeated label within it.
fn resolve_mutation_template_component(
    component: &TripleComponent,
    column_map: &HashMap<String, usize>,
    chunk: &DataChunk,
    row: usize,
    blank_solution: Option<(&str, usize)>,
    sealed_identity: bool,
) -> std::result::Result<Option<Term>, OperatorError> {
    if let TripleComponent::BlankNode(label) = component
        && let Some((execution, solution)) = blank_solution
    {
        return Ok(Some(Term::blank(format!(
            "{execution}_{label}_solution{solution}"
        ))));
    }
    if sealed_identity {
        resolve_mutation_component(component, column_map, chunk, row)
    } else {
        resolve_public_pattern_component(component, column_map, chunk, row)
    }
}

/// Resolves a GRAPH template exclusively from sealed RDF term identity.
/// Literal, blank-node, and unbound substitutions omit the template quad as
/// required by SPARQL Update; visible strings are never guessed to be IRIs.
fn resolve_mutation_graph(
    template: &str,
    column_map: &HashMap<String, usize>,
    chunk: &DataChunk,
    row: usize,
    sealed_identity: bool,
) -> std::result::Result<Option<String>, OperatorError> {
    let Some(variable) = rdf_graph_variable_from_template(template) else {
        return Ok(Some(template.to_string()));
    };
    if !sealed_identity {
        return match resolve_public_pattern_component(
            &TripleComponent::Variable(variable.to_string()),
            column_map,
            chunk,
            row,
        )? {
            Some(Term::Iri(iri)) => Ok(Some(iri.as_str().to_string())),
            Some(_) | None => Ok(None),
        };
    }

    let visible = column_map
        .get(variable)
        .and_then(|column_index| chunk.column(*column_index))
        .and_then(|column| column.get_value(row));
    if visible.as_ref().is_none_or(Value::is_null) {
        return Ok(None);
    }

    let exact_column = rdf_exact_term_column(variable);
    let Some(exact) = column_map
        .get(&exact_column)
        .and_then(|column_index| chunk.column(*column_index))
        .and_then(|column| column.get_value(row))
    else {
        return Err(OperatorError::Execution(format!(
            "bound graph variable ?{variable} lacked sealed RDF term identity"
        )));
    };
    if exact.is_null() {
        return Err(OperatorError::Execution(format!(
            "bound graph variable ?{variable} had no exact RDF term identity"
        )));
    }
    let encoded = exact.as_str().ok_or_else(|| {
        OperatorError::Execution(format!(
            "sealed RDF graph companion for ?{variable} was not a string"
        ))
    })?;
    let term = Term::from_ntriples(encoded).ok_or_else(|| {
        OperatorError::Execution(format!(
            "sealed RDF graph companion for ?{variable} contained invalid N-Triples"
        ))
    })?;
    match term {
        Term::Iri(iri) => Ok(Some(iri.as_str().to_string())),
        _ => Ok(None),
    }
}

/// Converts a TripleComponent to an Option<Term> for pattern matching.
fn component_to_term(component: &TripleComponent) -> Option<Term> {
    match component {
        TripleComponent::Variable(_) => None,
        TripleComponent::BlankNode(label) => Some(Term::blank(label.clone())),
        TripleComponent::Iri(iri) => Some(Term::iri(iri.clone())),
        TripleComponent::Literal(value) => Some(value_as_rdf_term(value)),
        TripleComponent::LangLiteral { value, lang } => {
            Some(Term::lang_literal(value.clone(), lang.clone()))
        }
    }
}

/// Whether a plan needs hidden exact RDF term companions either for a bound
/// mutation or for an ordinary exact-term consumer such as dynamic STRDT.
fn needs_exact_rdf_term_columns(op: &LogicalOperator) -> bool {
    contains_bound_rdf_mutation(op) || uses_bound_rdf_term_expression(op)
}

/// Whether a plan needs canonical RDF identity-key companions for relational
/// comparison. Unlike exact columns, these keys are never used to reconstruct
/// mutation terms.
fn needs_identity_rdf_term_columns(op: &LogicalOperator) -> bool {
    let local = match op {
        LogicalOperator::Join(join) => join
            .conditions
            .iter()
            .any(|condition| condition.semantics != JoinKeySemantics::Value),
        LogicalOperator::LeftJoin(join) => join
            .compatibility_conditions
            .iter()
            .any(|condition| condition.semantics != JoinKeySemantics::Value),
        LogicalOperator::AntiJoin(join) => join
            .compatibility_conditions
            .iter()
            .any(|condition| condition.semantics != JoinKeySemantics::Value),
        LogicalOperator::MultiWayJoin(join) => join
            .conditions
            .iter()
            .any(|condition| condition.semantics != JoinKeySemantics::Value),
        LogicalOperator::Aggregate(aggregate) => {
            aggregate
                .group_by
                .iter()
                .any(|expression| matches!(expression, LogicalExpression::Variable(_)))
                || aggregate.aggregates.iter().any(|expression| {
                    expression.function == LogicalAggregateFunction::Count
                        && expression.distinct
                        && expression.expression.is_none()
                })
        }
        LogicalOperator::Distinct(_) => true,
        _ => false,
    };
    local
        || op
            .children()
            .into_iter()
            .any(needs_identity_rdf_term_columns)
}

fn expression_contains_exists(expression: &LogicalExpression) -> bool {
    match expression {
        LogicalExpression::ExistsSubquery(_) => true,
        LogicalExpression::Binary { left, right, .. } => {
            expression_contains_exists(left) || expression_contains_exists(right)
        }
        LogicalExpression::Unary { operand, .. } => expression_contains_exists(operand),
        LogicalExpression::FunctionCall { args, .. } | LogicalExpression::List(args) => {
            args.iter().any(expression_contains_exists)
        }
        LogicalExpression::Map(entries) => entries
            .iter()
            .any(|(_, expression)| expression_contains_exists(expression)),
        LogicalExpression::IndexAccess { base, index } => {
            expression_contains_exists(base) || expression_contains_exists(index)
        }
        LogicalExpression::SliceAccess { base, start, end } => {
            expression_contains_exists(base)
                || start.as_deref().is_some_and(expression_contains_exists)
                || end.as_deref().is_some_and(expression_contains_exists)
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            operand.as_deref().is_some_and(expression_contains_exists)
                || when_clauses.iter().any(|(when, then)| {
                    expression_contains_exists(when) || expression_contains_exists(then)
                })
                || else_clause
                    .as_deref()
                    .is_some_and(expression_contains_exists)
        }
        LogicalExpression::ListComprehension {
            list_expr,
            filter_expr,
            map_expr,
            ..
        } => {
            expression_contains_exists(list_expr)
                || filter_expr
                    .as_deref()
                    .is_some_and(expression_contains_exists)
                || expression_contains_exists(map_expr)
        }
        LogicalExpression::ListPredicate {
            list_expr,
            predicate,
            ..
        } => expression_contains_exists(list_expr) || expression_contains_exists(predicate),
        LogicalExpression::Reduce {
            initial,
            list,
            expression,
            ..
        } => {
            expression_contains_exists(initial)
                || expression_contains_exists(list)
                || expression_contains_exists(expression)
        }
        LogicalExpression::PatternComprehension { projection, .. } => {
            expression_contains_exists(projection)
        }
        _ => false,
    }
}

fn is_direct_exists_filter(expression: &LogicalExpression) -> bool {
    matches!(expression, LogicalExpression::ExistsSubquery(_))
        || matches!(
            expression,
            LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand,
            }
                if matches!(operand.as_ref(), LogicalExpression::ExistsSubquery(_))
        )
}

/// RDF execution currently lowers a whole FILTER EXISTS/NOT EXISTS to a
/// typed semi/anti join. Embedded EXISTS in a compound expression or modifier
/// has no row-aware evaluator; reject it before planning instead of silently
/// treating it as an unbound/false scalar.
fn validate_rdf_exists_placement(op: &LogicalOperator) -> Result<()> {
    let unsupported = match op {
        LogicalOperator::Filter(filter) => {
            expression_contains_exists(&filter.predicate)
                && !is_direct_exists_filter(&filter.predicate)
        }
        LogicalOperator::Project(project) => project
            .projections
            .iter()
            .any(|projection| expression_contains_exists(&projection.expression)),
        LogicalOperator::Bind(bind) => expression_contains_exists(&bind.expression),
        LogicalOperator::Aggregate(aggregate) => {
            aggregate.group_by.iter().any(expression_contains_exists)
                || aggregate.aggregates.iter().any(|aggregate| {
                    [
                        aggregate.expression.as_ref(),
                        aggregate.expression2.as_ref(),
                        aggregate.distinct_key.as_ref(),
                    ]
                    .into_iter()
                    .flatten()
                    .any(expression_contains_exists)
                })
                || aggregate
                    .having
                    .as_ref()
                    .is_some_and(expression_contains_exists)
        }
        LogicalOperator::Sort(sort) => sort
            .keys
            .iter()
            .any(|key| expression_contains_exists(&key.expression)),
        LogicalOperator::Return(ret) => ret
            .items
            .iter()
            .any(|item| expression_contains_exists(&item.expression)),
        LogicalOperator::Join(join) => join.conditions.iter().any(|condition| {
            expression_contains_exists(&condition.left)
                || expression_contains_exists(&condition.right)
        }),
        LogicalOperator::LeftJoin(join) => {
            join.condition
                .as_ref()
                .is_some_and(expression_contains_exists)
                || join.compatibility_conditions.iter().any(|condition| {
                    expression_contains_exists(&condition.left)
                        || expression_contains_exists(&condition.right)
                })
        }
        LogicalOperator::AntiJoin(join) => join.compatibility_conditions.iter().any(|condition| {
            expression_contains_exists(&condition.left)
                || expression_contains_exists(&condition.right)
        }),
        LogicalOperator::Unwind(unwind) => expression_contains_exists(&unwind.expression),
        LogicalOperator::MultiWayJoin(join) => join.conditions.iter().any(|condition| {
            expression_contains_exists(&condition.left)
                || expression_contains_exists(&condition.right)
        }),
        _ => false,
    };
    if unsupported {
        return Err(Error::Query(grafeo_common::utils::error::QueryError::new(
            grafeo_common::utils::error::QueryErrorKind::Semantic,
            "compound or modifier EXISTS/NOT EXISTS is not yet supported by RDF execution",
        )));
    }
    for child in op.children() {
        validate_rdf_exists_placement(child)?;
    }
    Ok(())
}

fn validate_rdf_repeated_scan_variables(op: &LogicalOperator) -> Result<()> {
    if let LogicalOperator::TripleScan(scan) = op {
        let mut seen = HashSet::<&str>::new();
        for component in [&scan.subject, &scan.predicate, &scan.object]
            .into_iter()
            .chain(scan.graph.as_ref())
        {
            if let TripleComponent::Variable(variable) = component
                && !seen.insert(variable.as_str())
            {
                return Err(Error::InvalidValue(format!(
                    "RDF TripleScan contains repeated variable ?{variable} across scan positions; normalize repeated positions to distinct internal variables plus Filter/Project before planning"
                )));
            }
        }
    }
    for child in op.children() {
        validate_rdf_repeated_scan_variables(child)?;
    }
    Ok(())
}

fn uses_bound_rdf_term_expression(op: &LogicalOperator) -> bool {
    let local_use = match op {
        LogicalOperator::Filter(filter) => expr_uses_bound_rdf_term(&filter.predicate),
        LogicalOperator::Project(project) => project
            .projections
            .iter()
            .any(|projection| expr_uses_bound_rdf_term(&projection.expression)),
        LogicalOperator::Bind(bind) => expr_uses_bound_rdf_term(&bind.expression),
        LogicalOperator::Aggregate(aggregate) => {
            aggregate.group_by.iter().any(expr_uses_bound_rdf_term)
                || aggregate.aggregates.iter().any(|expression| {
                    [
                        &expression.expression,
                        &expression.expression2,
                        &expression.distinct_key,
                    ]
                    .into_iter()
                    .flatten()
                    .any(expr_uses_bound_rdf_term)
                })
                || aggregate
                    .having
                    .as_ref()
                    .is_some_and(expr_uses_bound_rdf_term)
        }
        LogicalOperator::Return(ret) => ret
            .items
            .iter()
            .any(|item| expr_uses_bound_rdf_term(&item.expression)),
        LogicalOperator::Sort(sort) => sort
            .keys
            .iter()
            .any(|key| expr_uses_bound_rdf_term(&key.expression)),
        LogicalOperator::Join(join) => join.conditions.iter().any(|condition| {
            expr_uses_bound_rdf_term(&condition.left) || expr_uses_bound_rdf_term(&condition.right)
        }),
        LogicalOperator::LeftJoin(join) => {
            join.compatibility_conditions.iter().any(|condition| {
                expr_uses_bound_rdf_term(&condition.left)
                    || expr_uses_bound_rdf_term(&condition.right)
            }) || join
                .condition
                .as_ref()
                .is_some_and(expr_uses_bound_rdf_term)
        }
        LogicalOperator::AntiJoin(join) => join.compatibility_conditions.iter().any(|condition| {
            expr_uses_bound_rdf_term(&condition.left) || expr_uses_bound_rdf_term(&condition.right)
        }),
        LogicalOperator::Unwind(unwind) => expr_uses_bound_rdf_term(&unwind.expression),
        LogicalOperator::MultiWayJoin(join) => join.conditions.iter().any(|condition| {
            expr_uses_bound_rdf_term(&condition.left) || expr_uses_bound_rdf_term(&condition.right)
        }),
        _ => false,
    };

    local_use
        || op
            .children()
            .into_iter()
            .any(uses_bound_rdf_term_expression)
}

fn expr_uses_bound_rdf_term(expression: &LogicalExpression) -> bool {
    match expression {
        LogicalExpression::FunctionCall { name, args, .. } => {
            name == RDF_TAG_BOUND_TERM
                // These public scalar consumers must retain the first RDF
                // witness through a projection/DISTINCT/subquery boundary.
                // Identity keys alone cannot replace lossless lexical state.
                || (["STR", "ISIRI", "ISURI", "LANG", "DATATYPE"]
                    .iter()
                    .any(|function| name.eq_ignore_ascii_case(function))
                    && matches!(args.first(), Some(LogicalExpression::Variable(_))))
                || args.iter().any(expr_uses_bound_rdf_term)
        }
        LogicalExpression::Binary { left, right, .. } => {
            expr_uses_bound_rdf_term(left) || expr_uses_bound_rdf_term(right)
        }
        LogicalExpression::Unary { operand, .. } => expr_uses_bound_rdf_term(operand),
        LogicalExpression::List(items) => items.iter().any(expr_uses_bound_rdf_term),
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            operand.as_deref().is_some_and(expr_uses_bound_rdf_term)
                || when_clauses.iter().any(|(when, then)| {
                    expr_uses_bound_rdf_term(when) || expr_uses_bound_rdf_term(then)
                })
                || else_clause.as_deref().is_some_and(expr_uses_bound_rdf_term)
        }
        _ => false,
    }
}

/// Whether a plan substitutes scanned bindings back into RDF mutation terms.
/// Such plans need hidden lossless term companions because visible string
/// values intentionally erase the IRI-versus-literal distinction.
fn contains_bound_rdf_mutation(op: &LogicalOperator) -> bool {
    use LogicalOperator::{
        Aggregate, AntiJoin, Bind, DeleteTriple, Distinct, Filter, InsertTriple, Join, LeftJoin,
        Limit, Modify, Project, Return, Skip, Sort, TripleScan, Union, Unwind,
    };
    match op {
        InsertTriple(insert) => insert.input.is_some(),
        DeleteTriple(delete) => delete.input.is_some(),
        Modify(_) => true,
        Filter(filter) => contains_bound_rdf_mutation(&filter.input),
        Project(project) => contains_bound_rdf_mutation(&project.input),
        Bind(bind) => contains_bound_rdf_mutation(&bind.input),
        Aggregate(aggregate) => contains_bound_rdf_mutation(&aggregate.input),
        Return(ret) => contains_bound_rdf_mutation(&ret.input),
        Sort(sort) => contains_bound_rdf_mutation(&sort.input),
        Join(join) => {
            contains_bound_rdf_mutation(&join.left) || contains_bound_rdf_mutation(&join.right)
        }
        LeftJoin(join) => {
            contains_bound_rdf_mutation(&join.left) || contains_bound_rdf_mutation(&join.right)
        }
        AntiJoin(join) => {
            contains_bound_rdf_mutation(&join.left) || contains_bound_rdf_mutation(&join.right)
        }
        Union(union) => union.inputs.iter().any(contains_bound_rdf_mutation),
        Distinct(distinct) => contains_bound_rdf_mutation(&distinct.input),
        Limit(limit) => contains_bound_rdf_mutation(&limit.input),
        Skip(skip) => contains_bound_rdf_mutation(&skip.input),
        Unwind(unwind) => contains_bound_rdf_mutation(&unwind.input),
        TripleScan(scan) => scan
            .input
            .as_ref()
            .is_some_and(|input| contains_bound_rdf_mutation(input)),
        LogicalOperator::MultiWayJoin(join) => join.inputs.iter().any(contains_bound_rdf_mutation),
        _ => false,
    }
}

/// Checks whether a logical plan tree references LANG, LANGMATCHES, or DATATYPE
/// functions. When true, the planner emits companion columns on triple scans.
fn uses_lang_or_datatype(op: &LogicalOperator) -> bool {
    use LogicalOperator::{
        Aggregate, AntiJoin, Bind, Distinct, Filter, Join, LeftJoin, Limit, Project, Return, Skip,
        Sort, TripleScan, Union, Unwind,
    };
    match op {
        Filter(f) => expr_uses_lang_or_datatype(&f.predicate) || uses_lang_or_datatype(&f.input),
        Project(p) => {
            p.projections
                .iter()
                .any(|proj| expr_uses_lang_or_datatype(&proj.expression))
                || uses_lang_or_datatype(&p.input)
        }
        Bind(b) => expr_uses_lang_or_datatype(&b.expression) || uses_lang_or_datatype(&b.input),
        Aggregate(a) => {
            a.aggregates.iter().any(|ae| {
                [&ae.expression, &ae.expression2, &ae.distinct_key]
                    .into_iter()
                    .flatten()
                    .any(expr_uses_lang_or_datatype)
            }) || a.having.as_ref().is_some_and(expr_uses_lang_or_datatype)
                || uses_lang_or_datatype(&a.input)
        }
        Return(r) => {
            r.items
                .iter()
                .any(|item| expr_uses_lang_or_datatype(&item.expression))
                || uses_lang_or_datatype(&r.input)
        }
        Sort(s) => {
            s.keys
                .iter()
                .any(|k| expr_uses_lang_or_datatype(&k.expression))
                || uses_lang_or_datatype(&s.input)
        }
        Join(j) => uses_lang_or_datatype(&j.left) || uses_lang_or_datatype(&j.right),
        LeftJoin(j) => {
            uses_lang_or_datatype(&j.left)
                || uses_lang_or_datatype(&j.right)
                || j.condition.as_ref().is_some_and(expr_uses_lang_or_datatype)
        }
        AntiJoin(j) => uses_lang_or_datatype(&j.left) || uses_lang_or_datatype(&j.right),
        Union(u) => u.inputs.iter().any(uses_lang_or_datatype),
        Distinct(d) => uses_lang_or_datatype(&d.input),
        Limit(l) => uses_lang_or_datatype(&l.input),
        Skip(s) => uses_lang_or_datatype(&s.input),
        Unwind(u) => uses_lang_or_datatype(&u.input),
        TripleScan(t) => t.input.as_ref().is_some_and(|i| uses_lang_or_datatype(i)),
        LogicalOperator::MultiWayJoin(mwj) => mwj.inputs.iter().any(uses_lang_or_datatype),
        // For any other operator, conservatively assume companion columns are needed
        _ => true,
    }
}

/// Checks whether a logical expression references LANG, LANGMATCHES, or DATATYPE.
fn expr_uses_lang_or_datatype(expr: &LogicalExpression) -> bool {
    match expr {
        LogicalExpression::FunctionCall { name, args, .. } => {
            let upper = name.to_uppercase();
            if upper == "LANG" || upper == "LANGMATCHES" || upper == "DATATYPE" {
                return true;
            }
            args.iter().any(expr_uses_lang_or_datatype)
        }
        LogicalExpression::Binary { left, right, .. } => {
            expr_uses_lang_or_datatype(left) || expr_uses_lang_or_datatype(right)
        }
        LogicalExpression::Unary { operand, .. } => expr_uses_lang_or_datatype(operand),
        LogicalExpression::List(items) => items.iter().any(expr_uses_lang_or_datatype),
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            operand.as_deref().is_some_and(expr_uses_lang_or_datatype)
                || when_clauses
                    .iter()
                    .any(|(w, t)| expr_uses_lang_or_datatype(w) || expr_uses_lang_or_datatype(t))
                || else_clause
                    .as_deref()
                    .is_some_and(expr_uses_lang_or_datatype)
        }
        _ => false,
    }
}

/// Quick cardinality estimate for a logical operator subtree.
///
/// Uses store index sizes to estimate how many rows an operator produces,
/// without collecting full statistics. Returns `None` if estimation is
/// not possible for this operator type.
fn estimate_operator_cardinality(
    op: &crate::query::plan::LogicalOperator,
    store: &RdfStore,
) -> Option<f64> {
    use crate::query::plan::LogicalOperator;
    match op {
        LogicalOperator::TripleScan(scan) => {
            let stats = store.stats();
            let total = stats.triple_count as f64;
            if total == 0.0 {
                return Some(0.0);
            }

            // Estimate based on which components are bound
            let s_bound = matches!(
                scan.subject,
                crate::query::plan::TripleComponent::Iri(_)
                    | crate::query::plan::TripleComponent::Literal(_)
                    | crate::query::plan::TripleComponent::LangLiteral { .. }
            );
            let p_bound = matches!(
                scan.predicate,
                crate::query::plan::TripleComponent::Iri(_)
                    | crate::query::plan::TripleComponent::Literal(_)
                    | crate::query::plan::TripleComponent::LangLiteral { .. }
            );
            let o_bound = matches!(
                scan.object,
                crate::query::plan::TripleComponent::Iri(_)
                    | crate::query::plan::TripleComponent::Literal(_)
                    | crate::query::plan::TripleComponent::LangLiteral { .. }
            );

            let estimate = match (s_bound, p_bound, o_bound) {
                (true, true, true) => 1.0,
                (true, true, false) => total / stats.subject_count.max(1) as f64,
                (true, false, true) => total / stats.subject_count.max(1) as f64,
                (false, true, true) => total / stats.predicate_count.max(1) as f64,
                (true, false, false) => total / stats.subject_count.max(1) as f64,
                (false, true, false) => total / stats.predicate_count.max(1) as f64,
                (false, false, true) => total / stats.object_count.max(1) as f64,
                (false, false, false) => total,
            };
            Some(estimate.max(1.0))
        }
        LogicalOperator::Filter(f) => {
            estimate_operator_cardinality(&f.input, store).map(|c| (c * 0.33).max(1.0))
        }
        LogicalOperator::Join(j) => {
            let left = estimate_operator_cardinality(&j.left, store)?;
            let right = estimate_operator_cardinality(&j.right, store)?;
            Some((left * right * 0.1).max(1.0))
        }
        LogicalOperator::Limit(l) => {
            if let crate::query::plan::CountExpr::Literal(n) = l.count {
                Some(n as f64)
            } else {
                estimate_operator_cardinality(&l.input, store)
            }
        }
        _ => None,
    }
}

/// Stable physical relation order shared by native Ring and typed hash
/// MultiWay execution. Using one order makes the first representative of a
/// canonical-equivalent RDF binding independent of Ring freshness while still
/// retaining the existing smallest-estimate-first join optimization.
fn rdf_multiway_join_order(
    join: &crate::query::plan::MultiWayJoinOp,
    store: &RdfStore,
) -> Vec<usize> {
    let mut order = (0..join.inputs.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        let left_cardinality =
            estimate_operator_cardinality(&join.inputs[*left], store).unwrap_or(1000.0);
        let right_cardinality =
            estimate_operator_cardinality(&join.inputs[*right], store).unwrap_or(1000.0);
        left_cardinality
            .partial_cmp(&right_cardinality)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    order
}

/// Resolves an expression to a column index.
fn resolve_expression(
    expr: &LogicalExpression,
    variable_columns: &HashMap<String, usize>,
) -> Result<usize> {
    crate::query::planner::common::resolve_expression_to_column(expr, variable_columns, "")
}

// expression_to_string is now in planner/common.rs
use crate::query::planner::common::{
    expression_to_string, output_column_name, resolved_column_name,
};

/// Converts a value to its string representation.
fn value_to_string(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Int64(i) => i.to_string(),
        Value::Float64(f) => f.to_string(),
        Value::String(s) => s.to_string(),
        Value::Bytes(b) => String::from_utf8_lossy(b).to_string(),
        Value::Timestamp(t) => t.to_string(),
        Value::Date(d) => d.to_string(),
        Value::Time(t) => t.to_string(),
        Value::Duration(d) => d.to_string(),
        Value::List(items) => {
            let parts: Vec<String> = items.iter().map(value_to_string).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Map(entries) => {
            let parts: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{}: {}", k, value_to_string(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        Value::Vector(v) => {
            let parts: Vec<String> = v.iter().map(|f| f.to_string()).collect();
            format!("vector([{}])", parts.join(", "))
        }
        Value::ZonedDatetime(zdt) => zdt.to_string(),
        Value::Path { nodes, edges } => {
            format!("<path: {} nodes, {} edges>", nodes.len(), edges.len())
        }
        Value::GCounter(counts) => {
            let total: u64 = counts.values().sum();
            format!("GCounter({total})")
        }
        Value::OnCounter { pos, neg } => {
            // reason: counter values are small increments, sum will not overflow i64
            #[allow(clippy::cast_possible_wrap)]
            let pos_sum: i64 = pos.values().copied().map(|v| v as i64).sum();
            // reason: value is a small counter, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            let neg_sum: i64 = neg.values().copied().map(|v| v as i64).sum();
            format!("OnCounter({})", pos_sum - neg_sum)
        }
        Value::RdfLiteral { lexical, .. } => lexical.to_string(),
        _ => value.to_string(),
    }
}

/// Returns a native visible value when the kernel can coerce the datatype.
/// Other datatypes retain exact RDF literal identity at the call site.
fn strdt_visible_value(lexical: &str, datatype: &str) -> Option<Value> {
    match datatype {
        Literal::XSD_STRING => Some(Value::String(lexical.into())),
        "http://www.w3.org/2001/XMLSchema#integer"
        | "http://www.w3.org/2001/XMLSchema#int"
        | "http://www.w3.org/2001/XMLSchema#long" => {
            rdf_numeric_literal_is_valid(&Literal::typed(lexical, datatype))
                .then(|| lexical.parse::<i64>().ok().map(Value::Int64))
                .flatten()
        }
        "http://www.w3.org/2001/XMLSchema#double"
        | "http://www.w3.org/2001/XMLSchema#float"
        | "http://www.w3.org/2001/XMLSchema#decimal" => {
            rdf_numeric_literal_is_valid(&Literal::typed(lexical, datatype))
                .then(|| lexical.parse::<f64>().ok().map(Value::Float64))
                .flatten()
        }
        "http://www.w3.org/2001/XMLSchema#boolean" => match lexical {
            "true" | "1" => Some(Value::Bool(true)),
            "false" | "0" => Some(Value::Bool(false)),
            _ => None,
        },
        _ => None,
    }
}

/// Compares two values and returns a boolean result.
fn date_value(value: &Value) -> Option<grafeo_common::types::Date> {
    match value {
        Value::Date(d) => Some(*d),
        Value::String(s) => grafeo_common::types::Date::parse(s),
        Value::RdfLiteral {
            lexical,
            language: None,
            datatype,
        } => {
            let dt = datatype.as_deref().unwrap_or("");
            if dt.is_empty() || dt == Literal::XSD_DATE || dt.ends_with("#date") {
                grafeo_common::types::Date::parse(lexical)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn is_xsd_date_value(value: &Value) -> bool {
    matches!(value, Value::Date(_))
        || matches!(
            value,
            Value::RdfLiteral {
                language: None,
                datatype: Some(dt),
                ..
            } if dt.as_str() == Literal::XSD_DATE || dt.ends_with("#date")
        )
}

fn parse_xsd_datetime(lexical: &str) -> Option<i64> {
    if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(lexical) {
        return Some(zdt.as_timestamp().as_micros());
    }
    if let Some(pos) = lexical.find('T')
        && let (Some(d), Some(t)) = (
            grafeo_common::types::Date::parse(&lexical[..pos]),
            grafeo_common::types::Time::parse(&lexical[pos + 1..]),
        )
    {
        return Some(grafeo_common::types::Timestamp::from_date_time(d, t).as_micros());
    }
    None
}

fn datetime_micros(value: &Value) -> Option<i64> {
    match value {
        Value::Timestamp(t) => Some(t.as_micros()),
        Value::ZonedDatetime(z) => Some(z.as_timestamp().as_micros()),
        Value::RdfLiteral {
            lexical,
            language: None,
            datatype,
        } => {
            let dt = datatype.as_deref().unwrap_or("");
            if dt == Literal::XSD_DATETIME || dt.ends_with("#dateTime") {
                parse_xsd_datetime(lexical)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn is_xsd_datetime_value(value: &Value) -> bool {
    matches!(value, Value::Timestamp(_) | Value::ZonedDatetime(_))
        || matches!(
            value,
            Value::RdfLiteral {
                language: None,
                datatype: Some(dt),
                ..
            } if dt.as_str() == Literal::XSD_DATETIME || dt.ends_with("#dateTime")
        )
}

fn rdf_effective_boolean_value(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(value) => Some(*value),
        Value::String(value) => Some(!value.is_empty()),
        Value::RdfLiteral {
            lexical,
            language: None,
            datatype: Some(datatype),
        } if datatype.as_str() == Literal::XSD_BOOLEAN => {
            Some(parse_xsd_boolean(lexical.as_str()).unwrap_or(false))
        }
        Value::RdfLiteral {
            lexical,
            language: None,
            datatype: Some(datatype),
        } if datatype.as_str() == Literal::XSD_STRING => Some(!lexical.is_empty()),
        Value::RdfLiteral {
            datatype: Some(datatype),
            ..
        } if numeric_kind(datatype.as_str()).is_some() => {
            Some(RdfNumeric::from_value(value).is_some_and(|value| value.effective_boolean_value()))
        }
        _ => RdfNumeric::from_value(value).map(|value| value.effective_boolean_value()),
    }
}

fn rdf_numeric_comparison(left: &Value, op: BinaryFilterOp, right: &Value) -> Option<Value> {
    let left = RdfNumeric::from_compatible_value(left)?;
    let right = RdfNumeric::from_compatible_value(right)?;
    let result = match op {
        BinaryFilterOp::Eq => left.equal(&right)?,
        BinaryFilterOp::Ne => !left.equal(&right)?,
        BinaryFilterOp::Lt => left.less_than(&right)?,
        BinaryFilterOp::Le => left.less_than(&right)? || left.equal(&right)?,
        BinaryFilterOp::Gt => left.greater_than(&right)?,
        BinaryFilterOp::Ge => left.greater_than(&right)? || left.equal(&right)?,
        _ => return None,
    };
    Some(Value::Bool(result))
}

fn is_invalid_rdf_numeric(value: &Value) -> bool {
    matches!(
        value,
        Value::RdfLiteral {
            lexical,
            language: None,
            datatype: Some(datatype),
        } if numeric_kind(datatype).is_some()
            && !rdf_numeric_literal_is_valid(&Literal::typed(lexical.as_str(), datatype.as_str()))
    )
}

fn compare_values<F>(left: &Value, right: &Value, cmp: F) -> Option<Value>
where
    F: Fn(std::cmp::Ordering) -> bool,
{
    if is_invalid_rdf_numeric(left) || is_invalid_rdf_numeric(right) {
        return None;
    }
    if is_xsd_date_value(left) || is_xsd_date_value(right) {
        let l = date_value(left)?;
        let r = date_value(right)?;
        return Some(Value::Bool(cmp(l.cmp(&r))));
    }
    if is_xsd_datetime_value(left) || is_xsd_datetime_value(right) {
        let l = datetime_micros(left)?;
        let r = datetime_micros(right)?;
        return Some(Value::Bool(cmp(l.cmp(&r))));
    }
    if let (Some(left), Some(right)) = (
        RdfNumeric::from_compatible_value(left),
        RdfNumeric::from_compatible_value(right),
    ) {
        return Some(Value::Bool(cmp(left.compare(&right)?)));
    }
    let ordering = match (left, right) {
        (
            Value::RdfLiteral {
                lexical: l,
                language: lang_l,
                datatype: dt_l,
            },
            Value::RdfLiteral {
                lexical: r,
                language: lang_r,
                datatype: dt_r,
            },
        ) => (l.as_str(), lang_l.as_deref(), dt_l.as_deref()).cmp(&(
            r.as_str(),
            lang_r.as_deref(),
            dt_r.as_deref(),
        )),
        // LANG-expansion compares the lexical form to a plain string.
        (
            Value::RdfLiteral {
                lexical,
                language: Some(_),
                ..
            },
            Value::String(s),
        ) => lexical.as_str().cmp(s.as_str()),
        (
            Value::String(s),
            Value::RdfLiteral {
                lexical,
                language: Some(_),
                ..
            },
        ) => s.as_str().cmp(lexical.as_str()),
        (Value::Int64(l), Value::Int64(r)) => l.cmp(r),
        (Value::Float64(l), Value::Float64(r)) => l.partial_cmp(r)?,
        (Value::String(l), Value::String(r)) => {
            // Try numeric comparison first if both look like numbers
            if let (Ok(l_num), Ok(r_num)) = (l.parse::<f64>(), r.parse::<f64>()) {
                l_num.partial_cmp(&r_num)?
            } else {
                l.cmp(r)
            }
        }
        (Value::Int64(l), Value::Float64(r)) => (*l as f64).partial_cmp(r)?,
        (Value::Float64(l), Value::Int64(r)) => l.partial_cmp(&(*r as f64))?,
        // RDF values are often stored as strings - try numeric conversion
        (Value::String(s), Value::Int64(r)) => {
            let l_num = s.parse::<f64>().ok()?;
            l_num.partial_cmp(&(*r as f64))?
        }
        (Value::String(s), Value::Float64(r)) => {
            let l_num = s.parse::<f64>().ok()?;
            l_num.partial_cmp(r)?
        }
        (Value::Int64(l), Value::String(s)) => {
            let r_num = s.parse::<f64>().ok()?;
            (*l as f64).partial_cmp(&r_num)?
        }
        (Value::Float64(l), Value::String(s)) => {
            let r_num = s.parse::<f64>().ok()?;
            l.partial_cmp(&r_num)?
        }
        _ => return None,
    };
    Some(Value::Bool(cmp(ordering)))
}

/// Checks equality of two RDF values, with cross-type numeric coercion.
///
/// Used by the IN operator to compare the left-hand value against each element
/// of the right-hand list.
fn rdf_values_equal(left: &Value, right: &Value) -> bool {
    if let (Some(left), Some(right)) = (
        RdfNumeric::from_compatible_value(left),
        RdfNumeric::from_compatible_value(right),
    ) {
        return left.equal(&right).unwrap_or(false);
    }
    match (left, right) {
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Int64(a), Value::Int64(b)) => a == b,
        (Value::Float64(a), Value::Float64(b)) => (a - b).abs() < f64::EPSILON,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Int64(a), Value::Float64(b)) | (Value::Float64(b), Value::Int64(a)) => {
            (*a as f64 - b).abs() < f64::EPSILON
        }
        // RDF stores numeric literals as strings: allow cross-type equality
        (Value::String(s), Value::Int64(i)) | (Value::Int64(i), Value::String(s)) => {
            s.parse::<i64>().is_ok_and(|n| n == *i)
        }
        (Value::String(s), Value::Float64(f)) | (Value::Float64(f), Value::String(s)) => {
            s.parse::<f64>().is_ok_and(|n| (n - f).abs() < f64::EPSILON)
        }
        _ => false,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{CountExpr, JoinOp, JoinType, LogicalPlan, ProjectOp, Projection};

    #[test]
    fn rdf_totality_projection_schema_error_precedes_child_consumption() {
        for (projection_count, type_count) in [(1, 0), (0, 1)] {
            let input = DataChunk::new(vec![ValueVector::from_values(&[Value::Int64(7)])]);
            let mut operator = RdfProjectOperator::new(
                Box::new(ConstantOperator::new(input)),
                (0..projection_count)
                    .map(|_| RdfProjectExpr::Column(0))
                    .collect(),
                vec![LogicalType::Int64; type_count],
            );
            assert!(matches!(operator.next(), Err(OperatorError::Execution(_))));
            assert_eq!(operator.child.next().unwrap().unwrap().row_count(), 1);
        }
    }

    #[test]
    fn rdf_totality_compatibility_schema_error_preserves_output_position() {
        for row in [Vec::new(), vec![Value::Int64(1), Value::Int64(2)]] {
            let mut operator = RdfCompatibilityJoinOperator::new(
                Box::new(SingleRowOperator::new()),
                Box::new(SingleRowOperator::new()),
                Vec::new(),
                RdfCompatibilityMode::Inner,
                Vec::new(),
                vec![LogicalType::Int64],
            );
            operator.output_rows = Some(vec![row]);
            assert!(matches!(operator.next(), Err(OperatorError::Execution(_))));
            assert_eq!(operator.position, 0);
        }
    }

    #[cfg(feature = "ring-index")]
    #[test]
    fn rdf_totality_ring_missing_state_is_typed_and_reset_recovers() {
        use grafeo_core::index::ring::AnnotatedPattern;

        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("urn:s"),
            Term::iri("urn:p"),
            Term::iri("urn:o"),
        ));
        store.rebuild_ring();
        let mut operator = RdfLeapfrogOperator::new(
            store.ring().unwrap(),
            vec![AnnotatedPattern {
                pattern: TriplePattern::any(),
                subject_var: Some("s".to_string()),
                predicate_var: Some("p".to_string()),
                object_var: Some("o".to_string()),
            }],
            RdfLeapfrogConfig {
                output_variables: vec!["s".to_string()],
                output_owners: vec![(0, 0)],
                output_types: vec![LogicalType::String],
                emit_exact_term_columns: false,
                emit_identity_key_columns: false,
                chunk_size: 1,
                output_cap: None,
            },
        );
        operator.ensure_prepared().unwrap();
        operator.state = None;
        assert!(matches!(operator.next(), Err(OperatorError::Execution(_))));
        operator.reset();
        let row = operator.next().unwrap().unwrap();
        assert_eq!(row.row_count(), 1);
        assert_eq!(
            row.column(0).unwrap().get_value(0),
            Some(Value::from("urn:s"))
        );
        assert!(operator.next().unwrap().is_none());
    }

    #[test]
    fn rdf_totality_exact_year_carry_borrow_and_sign_boundaries() {
        for width in [4, 128] {
            let digits = "9".repeat(width);
            let mut year = ExactYear::parse(false, &digits).unwrap();
            year.add_one();
            assert_eq!(year.magnitude, format!("1{}", "0".repeat(width)));
            year.subtract_one();
            assert_eq!(year.magnitude, digits);
            let mut negative = ExactYear::parse(true, &digits).unwrap();
            negative.subtract_one();
            negative.add_one();
            assert_eq!(negative.magnitude, digits);
            assert!(negative.negative);
        }
        let mut year = ExactYear::parse(false, "0000").unwrap();
        year.subtract_one();
        assert_eq!(year, ExactYear::parse(true, "0001").unwrap());
        year.add_one();
        assert_eq!(year, ExactYear::parse(false, "0000").unwrap());
        year.add_one();
        assert_eq!(year, ExactYear::parse(false, "0001").unwrap());
        assert_eq!(
            compare_xsd_datetimes("9999-12-31T24:00:00Z", "10000-01-01T00:00:00Z"),
            Some(Ordering::Equal)
        );
        assert_eq!(
            compare_xsd_datetimes("999x-12-31T24:00:00Z", "10000-01-01T00:00:00Z"),
            None
        );
    }

    #[test]
    fn rdf_totality_internal_term_dispatch_preserves_expression_errors() {
        let evaluator =
            RdfExpressionPredicate::new(FilterExpression::Literal(Value::Null), HashMap::new());
        let chunk = DataChunk::empty();
        for (tagger, value, expected, kind) in [
            (
                RDF_TAG_IRI_TERM,
                Value::from("urn:s"),
                Term::iri("urn:s"),
                RDF_IS_IRI,
            ),
            (
                RDF_TAG_BLANK_TERM,
                Value::from("_:b"),
                Term::blank("b"),
                RDF_IS_BLANK,
            ),
            (
                RDF_TAG_LITERAL_TERM,
                Value::Int64(7),
                value_as_rdf_term(&Value::Int64(7)),
                RDF_IS_LITERAL,
            ),
        ] {
            let tagged = evaluator
                .eval_function_call(tagger, &[FilterExpression::Literal(value)], &chunk, 0)
                .unwrap();
            assert_eq!(decode_tagged_rdf_filter_term(&tagged).unwrap().1, expected);
            assert_eq!(
                evaluator.eval_function_call(kind, &[FilterExpression::Literal(tagged)], &chunk, 0),
                Some(Value::Bool(true))
            );
        }
        let malformed = Value::List(
            vec![
                Value::Int64(7),
                Value::from("invalid exact term"),
                Value::from(INTERNAL_RDF_TAGGED_TERM_MARKER),
            ]
            .into(),
        );
        for kind in [RDF_IS_IRI, RDF_IS_BLANK, RDF_IS_LITERAL, RDF_IS_NUMERIC] {
            assert_eq!(
                evaluator.eval_function_call(
                    kind,
                    &[FilterExpression::Literal(malformed.clone())],
                    &chunk,
                    0
                ),
                None
            );
        }
        assert_eq!(
            evaluator.eval_function_call(
                RDF_TAG_IRI_TERM,
                &[FilterExpression::Literal(Value::Int64(7))],
                &chunk,
                0
            ),
            None
        );
    }

    #[cfg(feature = "cdc")]
    #[test]
    fn canonical_rdf_cdc_accumulator_stages_exact_events_once() {
        use crate::cdc::{CdcLog, ChangeKind, TransactionChangeAccumulator};
        use grafeo_common::types::{EpochId, GraphIncarnationId, HlcTimestamp};

        let log = Arc::new(CdcLog::new());
        let pending = Arc::new(TransactionChangeAccumulator::new(&log));
        let planner = RdfPlanner::new(Arc::new(RdfStore::new()))
            .with_transaction_id(Some(TransactionId::new(7)))
            .with_cdc_accumulator(Some(Arc::clone(&log)), Some(Arc::clone(&pending)));
        let sink = planner
            .cdc_log
            .as_ref()
            .expect("transaction-owned CDC sink");
        assert!(Arc::ptr_eq(&sink.pending_events, &pending));

        // A far-future physical component makes each mint's logical increment
        // deterministic without sleeping or changing the production clock.
        log.clock().update(HlcTimestamp::new(1 << 47, 0));
        for (index, (kind, graph, incarnation, object, encoded_object)) in [
            (
                ChangeKind::Create,
                Some("urn:graph"),
                GraphIncarnationId::FIRST_NAMED,
                Term::iri("urn:object"),
                "<urn:object>",
            ),
            (
                ChangeKind::Delete,
                None,
                GraphIncarnationId::DEFAULT_GRAPH,
                Term::lang_literal("colour", "en"),
                "\"colour\"@en",
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let before = log.clock().peek();
            sink.record(
                kind.clone(),
                &Term::iri("urn:subject"),
                &Term::iri("urn:predicate"),
                &object,
                (graph, incarnation),
            );

            let events = pending.lock();
            assert_eq!(events.len(), index + 1);
            let event = &events[index];
            assert_eq!(event.kind, kind);
            assert_eq!(event.epoch, EpochId::PENDING);
            assert_eq!(event.graph_incarnation, Some(incarnation));
            assert_eq!(event.triple_subject.as_deref(), Some("<urn:subject>"));
            assert_eq!(event.triple_predicate.as_deref(), Some("<urn:predicate>"));
            assert_eq!(event.triple_object.as_deref(), Some(encoded_object));
            assert_eq!(event.triple_graph.as_deref(), graph);
            assert!(event.lpg_graph.is_none());
            assert!(event.before.is_none());
            assert!(event.after.is_none());
            assert!(event.labels.is_none());
            assert!(event.edge_type.is_none());
            assert!(event.src_id.is_none());
            assert!(event.dst_id.is_none());
            assert_eq!(
                event.entity_id,
                log.triple_event(
                    kind,
                    "<urn:subject>",
                    "<urn:predicate>",
                    encoded_object,
                    graph,
                    EpochId::PENDING,
                )
                .entity_id,
            );
            assert_ne!(event.timestamp, HlcTimestamp::zero());
            assert_eq!(
                event.timestamp,
                HlcTimestamp::new(before.physical_ms(), before.logical() + 1),
                "only the accumulator may mint a timestamp for this event",
            );
            assert_eq!(event.timestamp, log.clock().peek());
            assert_eq!(log.event_count(), 0, "staged events are not committed");
            assert!(log.history_in_rdf_graph(event.entity_id, graph).is_empty());
        }
    }

    #[cfg(feature = "cdc")]
    #[test]
    fn rdf_cdc_sink_requires_transaction_and_enabled_capture() {
        use crate::cdc::{CdcLog, TransactionChangeAccumulator};

        for has_transaction in [false, true] {
            for has_log in [false, true] {
                for has_accumulator in [false, true] {
                    let store = Arc::new(RdfStore::new());
                    store.insert(Triple::new(
                        Term::iri("urn:subject"),
                        Term::iri("urn:predicate"),
                        Term::literal("value"),
                    ));
                    let log = Arc::new(CdcLog::new());
                    let initial_timestamp = log.clock().peek();
                    let pending = Arc::new(TransactionChangeAccumulator::new(&log));
                    let planner = RdfPlanner::new(Arc::clone(&store))
                        .with_transaction_id(has_transaction.then_some(TransactionId::new(7)))
                        .with_cdc_accumulator(
                            has_log.then(|| Arc::clone(&log)),
                            has_accumulator.then(|| Arc::clone(&pending)),
                        );
                    assert_eq!(
                        planner.cdc_log.is_some(),
                        has_transaction && has_log && has_accumulator,
                    );

                    // Session EXPLAIN uses profiled planning without executing
                    // the mutation, including when there is no transaction.
                    let _ = planner
                        .plan_profiled(&LogicalPlan::new(LogicalOperator::CreateGraph(
                            CreateGraphOp {
                                graph: "urn:uncreated".to_string(),
                                silent: false,
                            },
                        )))
                        .unwrap();
                    let mut read = planner
                        .plan(&LogicalPlan::new(LogicalOperator::TripleScan(
                            TripleScanOp {
                                subject: TripleComponent::Variable("s".to_string()),
                                predicate: TripleComponent::Variable("p".to_string()),
                                object: TripleComponent::Variable("o".to_string()),
                                graph: None,
                                input: None,
                                dataset: None,
                            },
                        )))
                        .unwrap();
                    let mut rows = 0;
                    while let Some(chunk) = read.operator.next().unwrap() {
                        rows += chunk.len();
                    }
                    assert_eq!(rows, 1);
                    assert_eq!(store.len(), 1);
                    assert!(store.graph("urn:uncreated").is_none());
                    assert_eq!(pending.position(), 0);
                    assert_eq!(log.event_count(), 0);
                    assert_eq!(log.clock().peek(), initial_timestamp);
                }
            }
        }
    }

    fn value_join_condition(variable: &str) -> JoinCondition {
        JoinCondition {
            left: LogicalExpression::Variable(variable.to_string()),
            right: LogicalExpression::Variable(variable.to_string()),
            semantics: JoinKeySemantics::Value,
        }
    }

    #[test]
    fn rdf_bound_scalar_companions_override_visible_values_and_reject_malformed_terms() {
        use grafeo_core::execution::ValueVector;

        let variables =
            HashMap::from([("term".to_string(), 0), (rdf_exact_term_column("term"), 1)]);
        let evaluate = |name: &str, visible: Value, exact: Value| {
            let expression = convert_filter_expression(&LogicalExpression::FunctionCall {
                name: name.to_string(),
                args: vec![LogicalExpression::Variable("term".to_string())],
                distinct: false,
            })
            .unwrap();
            let evaluator = RdfExpressionPredicate::new(expression, variables.clone());
            let chunk = DataChunk::new(vec![
                ValueVector::from_values(&[visible]),
                ValueVector::from_values(&[exact]),
            ]);
            evaluator.eval(&chunk, 0)
        };
        for function in ["ISIRI", "ISURI"] {
            assert_eq!(
                evaluate(function, Value::from("urn:x"), Value::from("\"urn:x\"")),
                Some(Value::Bool(false)),
            );
            // A NULL exact companion identifies the native branch of a mixed
            // row, whose existing no-companion scalar behavior is unchanged.
            assert_eq!(
                evaluate(function, Value::from("urn:x"), Value::Null),
                Some(Value::Bool(true)),
            );
        }
        assert_eq!(
            evaluate(
                "STR",
                Value::Int64(1),
                Value::from("\"01\"^^<http://www.w3.org/2001/XMLSchema#integer>")
            ),
            Some(Value::from("01")),
        );
        assert_eq!(
            evaluate("LANG", Value::from("colour"), Value::from("\"colour\"@en")),
            Some(Value::from("en")),
        );
        assert_eq!(
            evaluate(
                "DATATYPE",
                Value::Int64(1),
                Value::from("\"1\"^^<http://www.w3.org/2001/XMLSchema#int>")
            ),
            Some(Value::from("http://www.w3.org/2001/XMLSchema#int")),
        );
        for function in ["STR", "ISIRI", "ISURI", "LANG", "DATATYPE"] {
            assert_eq!(
                evaluate(function, Value::Null, Value::from("\"urn:x\""),),
                None,
                "NULL visible binding must not use an exact companion: {function}",
            );
        }
        for function in ["STR", "ISIRI", "ISURI", "LANG", "DATATYPE"] {
            assert_eq!(
                evaluate(
                    function,
                    Value::from("urn:x"),
                    Value::from("not an encoded RDF term")
                ),
                None,
                "an authoritative malformed companion must not become a native guess: {function}",
            );
        }
    }

    #[test]
    fn rdf_unbound_values_are_expression_errors_and_coalesce_falls_back() {
        use grafeo_core::execution::ValueVector;

        let variables =
            HashMap::from([("term".to_string(), 0), (rdf_exact_term_column("term"), 1)]);
        let chunk = DataChunk::new(vec![
            ValueVector::from_values(&[Value::Null]),
            ValueVector::from_values(&[Value::Null]),
        ]);
        for (label, argument) in [
            ("variable", LogicalExpression::Variable("term".to_string())),
            ("literal", LogicalExpression::Literal(Value::Null)),
        ] {
            for function in ["ISIRI", "ISURI", "STR", "LANG", "DATATYPE"] {
                let expression = convert_filter_expression(&LogicalExpression::FunctionCall {
                    name: function.to_string(),
                    args: vec![argument.clone()],
                    distinct: false,
                })
                .unwrap();
                let evaluator = RdfExpressionPredicate::new(expression, variables.clone());
                assert_eq!(
                    evaluator.eval(&chunk, 0),
                    None,
                    "{function} on {label} must type-error"
                );
            }
        }

        let bound = convert_filter_expression(&LogicalExpression::FunctionCall {
            name: "BOUND".to_string(),
            args: vec![LogicalExpression::Variable("term".to_string())],
            distinct: false,
        })
        .unwrap();
        assert_eq!(
            RdfExpressionPredicate::new(bound, variables.clone()).eval(&chunk, 0),
            Some(Value::Bool(false))
        );

        let coalesce = convert_filter_expression(&LogicalExpression::FunctionCall {
            name: "COALESCE".to_string(),
            args: vec![
                LogicalExpression::Variable("term".to_string()),
                LogicalExpression::Literal(Value::from("fallback")),
            ],
            distinct: false,
        })
        .unwrap();
        assert_eq!(
            RdfExpressionPredicate::new(coalesce, variables).eval(&chunk, 0),
            Some(Value::from("fallback"))
        );
    }

    #[test]
    fn rdf_concat_evaluates_each_volatile_argument_once() {
        RDF_VOLATILE_EVALUATIONS.with(|count| count.set(0));
        let expression = convert_filter_expression(&LogicalExpression::FunctionCall {
            name: "CONCAT".to_string(),
            args: vec![LogicalExpression::FunctionCall {
                name: "RAND".to_string(),
                args: Vec::new(),
                distinct: false,
            }],
            distinct: false,
        })
        .unwrap();
        let predicate = RdfExpressionPredicate::new(expression, HashMap::new());
        let mut chunk = DataChunk::with_capacity(&[], 1);
        chunk.set_count(1);

        assert!(matches!(predicate.eval(&chunk, 0), Some(Value::String(_))));
        assert_eq!(
            RDF_VOLATILE_EVALUATIONS.with(std::cell::Cell::get),
            1,
            "CONCAT must evaluate a non-string volatile argument exactly once"
        );
    }

    #[cfg(feature = "sparql")]
    #[test]
    fn rdf_distinct_evaluates_volatile_operand_once_before_keying() {
        RDF_VOLATILE_EVALUATIONS.with(|count| count.set(0));
        let db = crate::GrafeoDB::with_config(
            crate::Config::in_memory().with_graph_model(crate::GraphModel::Rdf),
        )
        .unwrap();
        let result = db
            .execute_sparql("SELECT DISTINCT (RAND() AS ?value) WHERE { VALUES ?seed { 1 } }")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(
            RDF_VOLATILE_EVALUATIONS.with(std::cell::Cell::get),
            1,
            "ordinary SELECT DISTINCT must evaluate its one source operand once"
        );
    }

    #[cfg(all(feature = "spill", feature = "sparql"))]
    #[test]
    fn rdf_distinct_volatile_values_spill_without_repeated_evaluation() {
        use std::fmt::Write as _;
        let mut query =
            String::from("SELECT DISTINCT ?seed (RAND() AS ?value) WHERE { VALUES ?seed {");
        for seed in 0..4096 {
            write!(query, " {seed}").unwrap();
        }
        query.push_str(" } }");

        let denied_dir = tempfile::tempdir().unwrap();
        let denied = crate::GrafeoDB::with_config(
            crate::Config::in_memory()
                .with_graph_model(crate::GraphModel::Rdf)
                .with_memory_limit(2 << 20)
                .with_spill_path(denied_dir.path())
                .with_max_query_spill_bytes(0),
        )
        .unwrap();
        RDF_VOLATILE_EVALUATIONS.with(|count| count.set(0));
        let error = denied
            .execute_sparql(&query)
            .expect_err("zero spill quota must reject the pressure query");
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );

        let spill_dir = tempfile::tempdir().unwrap();
        let db = crate::GrafeoDB::with_config(
            crate::Config::in_memory()
                .with_graph_model(crate::GraphModel::Rdf)
                .with_memory_limit(2 << 20)
                .with_spill_path(spill_dir.path()),
        )
        .unwrap();
        RDF_VOLATILE_EVALUATIONS.with(|count| count.set(0));
        let result = db.execute_sparql(&query).unwrap();
        assert_eq!(result.row_count(), 4096);
        assert_eq!(
            RDF_VOLATILE_EVALUATIONS.with(std::cell::Cell::get),
            4096,
            "one RAND source expression must be evaluated once per VALUES row"
        );
    }

    #[test]
    fn rdf_numeric_modulo_validates_typed_literals() {
        let predicate = RdfExpressionPredicate::new(
            FilterExpression::Literal(Value::Bool(true)),
            HashMap::new(),
        );
        let valid = Value::RdfLiteral {
            lexical: "8".into(),
            language: None,
            datatype: Some(format!("{}unsignedByte", Literal::XSD).into()),
        };
        let invalid = Value::RdfLiteral {
            lexical: "256".into(),
            language: None,
            datatype: Some(format!("{}unsignedByte", Literal::XSD).into()),
        };

        assert_eq!(
            predicate.eval_binary_op(&valid, BinaryFilterOp::Mod, &Value::Int64(3)),
            Some(Value::Int64(2))
        );
        assert_eq!(
            predicate.eval_binary_op(&invalid, BinaryFilterOp::Mod, &Value::Int64(3)),
            None,
            "an invalid numeric facet remains an expression error"
        );
    }

    #[test]
    fn test_rdf_planner_simple_scan() {
        let store = Arc::new(RdfStore::new());

        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));

        let planner = RdfPlanner::new(store);

        let scan = TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        };

        let plan = LogicalPlan::new(LogicalOperator::TripleScan(scan));
        let physical = planner.plan(&plan).unwrap();

        assert_eq!(physical.columns, vec!["s", "p", "o"]);
    }

    #[test]
    fn test_rdf_planner_with_pattern() {
        let store = Arc::new(RdfStore::new());

        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Gus"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let planner = RdfPlanner::new(store);

        let scan = TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        };

        let plan = LogicalPlan::new(LogicalOperator::TripleScan(scan));
        let physical = planner.plan(&plan).unwrap();

        // Only s and o are variables (predicate is fixed)
        assert_eq!(physical.columns, vec!["s", "o"]);
    }

    #[test]
    fn test_rdf_scan_operator_chunking() {
        let store = Arc::new(RdfStore::new());

        // Insert 100 triples
        for i in 0..100 {
            store.insert(Triple::new(
                Term::iri(format!("http://example.org/item{}", i)),
                Term::iri("http://example.org/value"),
                Term::literal(i.to_string()),
            ));
        }

        let pattern = TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        };

        let mut operator = RdfTripleScanOperator::new(
            Arc::clone(&store),
            pattern,
            RdfTripleScanOutput {
                mask: [true, true, true, false],
                companion_columns: false,
                datatype_column: false,
                term_companions: RdfTermCompanionOutput {
                    lossless: false,
                    identity: false,
                },
            },
            30,
            GraphContext {
                graph: None,
                scan_all_graphs: false,
                dataset: None,
            },
            None,
        );

        let mut total_rows = 0;
        while let Ok(Some(chunk)) = operator.next() {
            total_rows += chunk.row_count();
            assert!(chunk.row_count() <= 30); // Respects chunk size
        }

        assert_eq!(total_rows, 100);
    }

    #[test]
    #[cfg(feature = "wal")]
    fn wal_graph_mutations_require_transaction_before_any_side_effect() {
        fn wal_image(
            path: &std::path::Path,
        ) -> std::collections::BTreeMap<std::ffi::OsString, Vec<u8>> {
            std::fs::read_dir(path)
                .unwrap()
                .map(|entry| {
                    let entry = entry.unwrap();
                    (entry.file_name(), std::fs::read(entry.path()).unwrap())
                })
                .collect()
        }

        const SOURCE: &str = "urn:source";
        const TARGET: &str = "urn:target";
        const EMPTY: &str = "urn:empty";
        const MISSING: &str = "urn:missing";
        let store = Arc::new(RdfStore::new());
        assert!(store.insert(Triple::new(
            Term::iri("urn:default"),
            Term::iri("urn:p"),
            Term::literal("default")
        )));
        for name in [SOURCE, TARGET, EMPTY] {
            assert!(store.create_graph(name));
            if name != EMPTY {
                assert!(store.graph(name).unwrap().insert(Triple::new(
                    Term::iri(name),
                    Term::iri("urn:p"),
                    Term::literal(name),
                )));
            }
        }
        let temporary = tempfile::tempdir().unwrap();
        let wal_path = temporary.path().join("wal");
        let wal = Arc::new(grafeo_storage::wal::LpgWal::open(&wal_path).unwrap());
        wal.log(&grafeo_storage::wal::WalRecord::GraphModelMeta { model: 2 })
            .unwrap();
        wal.sync().unwrap();
        let before_wal = wal_image(&wal_path);
        let before_epoch = store.commit_epoch();
        let before = store.dataset_history().unwrap();
        let before_cut = before.cut(before_epoch).unwrap();
        let mut before_names = store.graph_names();
        before_names.sort();

        for silent in [false, true] {
            let mut cases = Vec::new();
            for name in [MISSING, SOURCE] {
                cases.push(LogicalOperator::CreateGraph(CreateGraphOp {
                    graph: name.to_string(),
                    silent,
                }));
            }
            for graph in [
                None,
                Some(SOURCE),
                Some(MISSING),
                Some("\u{1}NAMED"),
                Some(""),
            ] {
                cases.push(LogicalOperator::DropGraph(DropGraphOp {
                    graph: graph.map(str::to_string),
                    silent,
                }));
            }
            for (source, destination) in [
                (None, Some(MISSING)),
                (Some(SOURCE), None),
                (Some(SOURCE), Some(TARGET)),
                (Some(EMPTY), Some(MISSING)),
                (Some(SOURCE), Some(SOURCE)),
                (None, None),
                (Some(MISSING), Some(TARGET)),
            ] {
                cases.push(LogicalOperator::CopyGraph(CopyGraphOp {
                    source: source.map(str::to_string),
                    destination: destination.map(str::to_string),
                    silent,
                }));
                cases.push(LogicalOperator::MoveGraph(MoveGraphOp {
                    source: source.map(str::to_string),
                    destination: destination.map(str::to_string),
                    silent,
                }));
                cases.push(LogicalOperator::AddGraph(AddGraphOp {
                    source: source.map(str::to_string),
                    destination: destination.map(str::to_string),
                    silent,
                }));
            }
            for operator in cases {
                let logical = LogicalPlan::new(operator);
                for profiled in [false, true] {
                    let planner = RdfPlanner::new(Arc::clone(&store))
                        .with_transaction_id(None)
                        .with_wal(Some(Arc::clone(&wal)));
                    let mut physical = if profiled {
                        planner.plan_profiled(&logical).unwrap().0
                    } else {
                        planner.plan(&logical).unwrap()
                    };
                    for _ in 0..2 {
                        let error = physical.operator.next().unwrap_err();
                        assert!(
                            matches!(error, OperatorError::Execution(ref message)
                            if message == "WAL-backed RDF graph mutation requires an active transaction"),
                            "{error}; plan={:?}, profiled={profiled}",
                            logical.root
                        );
                        let after = store.dataset_history().unwrap();
                        assert_eq!(store.commit_epoch(), before_epoch);
                        assert_eq!(after.store_id(), before.store_id());
                        assert_eq!(after.completeness(), before.completeness());
                        assert_eq!(
                            after.next_graph_incarnation(),
                            before.next_graph_incarnation()
                        );
                        assert_eq!(
                            store.next_graph_incarnation(),
                            before.next_graph_incarnation()
                        );
                        assert_eq!(after.graph_lives(), before.graph_lives());
                        assert_eq!(after.quad_versions(), before.quad_versions());
                        assert_eq!(after.cut(before_epoch).unwrap(), before_cut);
                        let mut names = store.graph_names();
                        names.sort();
                        assert_eq!(names, before_names);
                        assert_eq!(
                            *planner.wal.as_ref().unwrap().logged_graph_high_water.lock(),
                            0
                        );
                        wal.sync().unwrap();
                        assert_eq!(wal_image(&wal_path), before_wal);
                    }
                }
            }
        }
    }

    #[test]
    fn test_copy_graph_operator() {
        let store = Arc::new(RdfStore::new());

        // Insert triples into a named graph
        store.create_graph("http://example.org/src");
        let src = store.graph("http://example.org/src").unwrap();
        src.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/p"),
            Term::literal("val"),
        ));
        assert_eq!(src.len(), 1);

        // Copy src -> dst via operator
        let plan = LogicalPlan::new(LogicalOperator::CopyGraph(CopyGraphOp {
            source: Some("http://example.org/src".to_string()),
            destination: Some("http://example.org/dst".to_string()),
            silent: false,
        }));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner.plan(&plan).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}

        // Source still has its data
        assert_eq!(store.graph("http://example.org/src").unwrap().len(), 1);
        // Destination has a copy
        assert_eq!(store.graph("http://example.org/dst").unwrap().len(), 1);
    }

    #[test]
    fn test_move_graph_operator() {
        let store = Arc::new(RdfStore::new());

        store.create_graph("http://example.org/src");
        let src = store.graph("http://example.org/src").unwrap();
        src.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/p"),
            Term::literal("val"),
        ));

        // Move src -> dst via operator
        let plan = LogicalPlan::new(LogicalOperator::MoveGraph(MoveGraphOp {
            source: Some("http://example.org/src".to_string()),
            destination: Some("http://example.org/dst".to_string()),
            silent: false,
        }));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner.plan(&plan).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}

        // Source is gone (move drops it)
        assert!(store.graph("http://example.org/src").is_none());
        // Destination has the data
        assert_eq!(store.graph("http://example.org/dst").unwrap().len(), 1);
    }

    #[test]
    fn test_add_graph_operator() {
        let store = Arc::new(RdfStore::new());

        // Create src with 1 triple
        store.create_graph("http://example.org/src");
        store
            .graph("http://example.org/src")
            .unwrap()
            .insert(Triple::new(
                Term::iri("http://example.org/a"),
                Term::iri("http://example.org/p"),
                Term::literal("from-src"),
            ));

        // Create dst with 1 different triple
        store.create_graph("http://example.org/dst");
        store
            .graph("http://example.org/dst")
            .unwrap()
            .insert(Triple::new(
                Term::iri("http://example.org/b"),
                Term::iri("http://example.org/q"),
                Term::literal("from-dst"),
            ));

        // Add src -> dst via operator (merges)
        let plan = LogicalPlan::new(LogicalOperator::AddGraph(AddGraphOp {
            source: Some("http://example.org/src".to_string()),
            destination: Some("http://example.org/dst".to_string()),
            silent: false,
        }));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner.plan(&plan).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}

        // Source unchanged
        assert_eq!(store.graph("http://example.org/src").unwrap().len(), 1);
        // Destination has both triples (union)
        assert_eq!(store.graph("http://example.org/dst").unwrap().len(), 2);
    }

    #[test]
    fn test_empty_source_graph_operations_create_named_destination() {
        for operation in ["copy", "move", "add"] {
            let store = Arc::new(RdfStore::new());
            assert!(store.create_graph("http://example.org/empty-source"));

            let operator = match operation {
                "copy" => LogicalOperator::CopyGraph(CopyGraphOp {
                    source: Some("http://example.org/empty-source".to_string()),
                    destination: Some("http://example.org/empty-destination".to_string()),
                    silent: false,
                }),
                "move" => LogicalOperator::MoveGraph(MoveGraphOp {
                    source: Some("http://example.org/empty-source".to_string()),
                    destination: Some("http://example.org/empty-destination".to_string()),
                    silent: false,
                }),
                "add" => LogicalOperator::AddGraph(AddGraphOp {
                    source: Some("http://example.org/empty-source".to_string()),
                    destination: Some("http://example.org/empty-destination".to_string()),
                    silent: false,
                }),
                _ => unreachable!("fixed operation table"),
            };
            let planner = RdfPlanner::new(Arc::clone(&store));
            let mut physical = planner.plan(&LogicalPlan::new(operator)).unwrap().operator;
            while physical.next().unwrap().is_some() {}

            assert!(
                store
                    .graph("http://example.org/empty-destination")
                    .is_some(),
                "{operation} must create an absent named destination even when the source is empty"
            );
            if operation == "move" {
                assert!(store.graph("http://example.org/empty-source").is_none());
            } else {
                assert!(store.graph("http://example.org/empty-source").is_some());
            }
        }
    }

    #[test]
    fn test_copy_nonexistent_source_errors_without_silent() {
        let store = Arc::new(RdfStore::new());

        let plan = LogicalPlan::new(LogicalOperator::CopyGraph(CopyGraphOp {
            source: Some("http://example.org/nope".to_string()),
            destination: Some("http://example.org/dst".to_string()),
            silent: false,
        }));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner.plan(&plan).unwrap();
        let mut op = physical.operator;

        // Should error because source doesn't exist
        let result = op.next();
        assert!(result.is_err());
    }

    #[test]
    fn test_copy_nonexistent_source_silent_ok() {
        let store = Arc::new(RdfStore::new());

        let plan = LogicalPlan::new(LogicalOperator::CopyGraph(CopyGraphOp {
            source: Some("http://example.org/nope".to_string()),
            destination: Some("http://example.org/dst".to_string()),
            silent: true,
        }));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner.plan(&plan).unwrap();
        let mut op = physical.operator;

        // Should succeed silently
        assert!(op.next().is_ok());
    }

    /// Triple scan: IRIs are String; object terms are Any (lang/datatype).
    #[test]
    fn test_type_propagation_triple_scan() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));

        let planner = RdfPlanner::new(store);
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });

        let (_op, columns, types) = planner.plan_operator(&scan).unwrap();
        assert_eq!(columns.len(), types.len());
        for (name, ty) in columns.iter().zip(types.iter()) {
            if name == "o" {
                assert_eq!(*ty, LogicalType::Any, "object column is Any");
            } else {
                assert_eq!(*ty, LogicalType::String, "column {name} should be String");
            }
        }
    }

    #[test]
    fn triple_scan_separates_lossless_terms_from_canonical_identity_keys() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("urn:subject"),
            Term::iri("urn:predicate"),
            Term::lang_literal("colour", "EN"),
        ));
        let planner = RdfPlanner::new(store);
        planner.needs_exact_term_columns.set(true);
        planner.needs_identity_key_columns.set(true);
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Iri("urn:subject".to_string()),
            predicate: TripleComponent::Iri("urn:predicate".to_string()),
            object: TripleComponent::Variable("value".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });

        let (mut operator, columns, _) = planner.plan_operator(&scan).unwrap();
        assert_eq!(
            columns,
            [
                "value".to_string(),
                rdf_exact_term_column("value"),
                rdf_identity_key_column("value"),
                "__lang_value".to_string(),
            ]
        );
        let chunk = operator.next().unwrap().expect("one matching RDF term");
        assert_eq!(
            chunk.column(1).unwrap().get_value(0),
            Some(Value::String("\"colour\"@EN".into())),
            "the reconstruction companion preserves the stored language spelling"
        );
        assert_eq!(
            chunk.column(2).unwrap().get_value(0),
            Some(Value::String("\"colour\"@en".into())),
            "the relational key canonicalizes case-insensitive RDF language identity"
        );
    }

    /// Join of two triple scans propagates String types through to output.
    #[test]
    fn test_type_propagation_join() {
        use crate::query::plan::JoinOp;

        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let planner = RdfPlanner::new(store);

        let left = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let right = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/age".to_string()),
            object: TripleComponent::Variable("age".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });

        let join = LogicalOperator::Join(JoinOp {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
            conditions: vec![value_join_condition("s")],
        });

        let (_op, columns, types) = planner.plan_operator(&join).unwrap();
        assert_eq!(columns.len(), types.len());
        for (name, ty) in columns.iter().zip(types.iter()) {
            if name == "name" || name == "age" {
                assert_eq!(*ty, LogicalType::Any, "object column {name} is Any");
            } else {
                assert_eq!(
                    *ty,
                    LogicalType::String,
                    "join column {name} should be String"
                );
            }
        }
    }

    #[test]
    fn rdf_identity_join_resolves_only_canonical_identity_keys() {
        let shared_identity = rdf_identity_key_column("shared");
        let left_columns = vec![
            "shared".to_string(),
            shared_identity.clone(),
            "left".to_string(),
        ];
        let right_columns = vec![
            "shared".to_string(),
            shared_identity.clone(),
            "right".to_string(),
        ];
        let identity = JoinCondition {
            left: LogicalExpression::Variable("shared".to_string()),
            right: LogicalExpression::Variable("shared".to_string()),
            semantics: JoinKeySemantics::RdfTermIdentity,
        };

        assert_eq!(
            resolve_rdf_join_keys(
                std::slice::from_ref(&identity),
                &left_columns,
                &right_columns
            )
            .unwrap(),
            Some((vec![1], vec![1])),
            "RDF identity must hash only the canonical identity key, not the visible value"
        );
        assert_eq!(
            resolve_rdf_join_keys(
                &[JoinCondition {
                    semantics: JoinKeySemantics::Value,
                    ..identity.clone()
                }],
                &left_columns,
                &right_columns,
            )
            .unwrap(),
            Some((vec![0], vec![0])),
            "declared value joins use only their visible key columns"
        );
        let error = resolve_rdf_join_keys(
            &[identity],
            &["shared".to_string()],
            &["shared".to_string()],
        )
        .unwrap_err();
        assert!(error.to_string().contains("was not materialized"));
    }

    #[test]
    fn rdf_binary_joins_reject_partial_or_duplicate_public_schemas() {
        fn projected(names: &[&str]) -> LogicalOperator {
            LogicalOperator::Project(ProjectOp {
                projections: names
                    .iter()
                    .map(|name| Projection {
                        expression: LogicalExpression::Literal(Value::String(
                            format!("urn:{name}").into(),
                        )),
                        alias: Some((*name).to_string()),
                    })
                    .collect(),
                input: Box::new(LogicalOperator::Empty),
                pass_through_input: false,
            })
        }

        fn condition(semantics: JoinKeySemantics) -> JoinCondition {
            JoinCondition {
                left: LogicalExpression::Variable("x".to_string()),
                right: LogicalExpression::Variable("x".to_string()),
                semantics,
            }
        }

        let planner = RdfPlanner::new(Arc::new(RdfStore::new()));
        let left = || projected(&["x", "y"]);
        let right = || projected(&["x", "y"]);
        let operators = [
            LogicalOperator::Join(JoinOp {
                left: Box::new(left()),
                right: Box::new(right()),
                join_type: JoinType::Inner,
                conditions: vec![condition(JoinKeySemantics::Value)],
            }),
            LogicalOperator::LeftJoin(LeftJoinOp {
                left: Box::new(left()),
                right: Box::new(right()),
                condition: None,
                compatibility_conditions: vec![condition(JoinKeySemantics::SparqlCompatibility)],
            }),
            LogicalOperator::AntiJoin(AntiJoinOp {
                left: Box::new(left()),
                right: Box::new(right()),
                compatibility_conditions: vec![condition(JoinKeySemantics::Value)],
                semantics: AntiJoinSemantics::Minus,
            }),
        ];
        for operator in operators {
            let Err(error) = planner.plan_operator(&operator) else {
                panic!("partial binary metadata was accepted");
            };
            assert!(
                error
                    .to_string()
                    .contains("exactly declare every same-named public input column"),
                "partial binary metadata must fail before schema coalescing: {error}"
            );
        }

        let duplicate = LogicalOperator::Join(JoinOp {
            left: Box::new(projected(&["duplicate", "duplicate"])),
            right: Box::new(LogicalOperator::Empty),
            join_type: JoinType::Cross,
            conditions: Vec::new(),
        });
        let Err(error) = planner.plan_operator(&duplicate) else {
            panic!("duplicate public input columns were accepted");
        };
        assert!(
            error.to_string().contains("duplicate public column"),
            "duplicate public input names must fail before keyed/coalesced planning: {error}"
        );
    }

    #[test]
    fn rdf_identity_join_materializes_each_identity_scan_column_once() {
        use crate::query::plan::{JoinOp, ProjectOp, ReturnOp, UnionOp, UnwindOp};

        let left = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("left".to_string()),
            predicate: TripleComponent::Iri("urn:p".to_string()),
            object: TripleComponent::Variable("shared".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let right = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("right".to_string()),
            predicate: TripleComponent::Iri("urn:q".to_string()),
            object: TripleComponent::Variable("shared".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let join = LogicalOperator::Join(JoinOp {
            left: Box::new(left.clone()),
            right: Box::new(right.clone()),
            join_type: JoinType::Inner,
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("shared".to_string()),
                right: LogicalExpression::Variable("shared".to_string()),
                semantics: JoinKeySemantics::RdfTermIdentity,
            }],
        });
        assert!(!needs_exact_rdf_term_columns(&join));
        assert!(needs_identity_rdf_term_columns(&join));
        assert!(needs_identity_rdf_term_columns(
            &LogicalOperator::Construct(ConstructOp {
                templates: Vec::new(),
                input: Box::new(join.clone()),
            })
        ));

        let wrappers = vec![
            (
                "filter",
                LogicalOperator::Filter(FilterOp {
                    predicate: LogicalExpression::Literal(Value::Bool(true)),
                    input: Box::new(join.clone()),
                    pushdown_hint: None,
                }),
            ),
            (
                "project",
                LogicalOperator::Project(ProjectOp {
                    projections: Vec::new(),
                    input: Box::new(join.clone()),
                    pass_through_input: false,
                }),
            ),
            (
                "aggregate",
                LogicalOperator::Aggregate(AggregateOp {
                    group_by: Vec::new(),
                    aggregates: Vec::new(),
                    input: Box::new(join.clone()),
                    having: None,
                }),
            ),
            (
                "sort",
                LogicalOperator::Sort(SortOp {
                    keys: Vec::new(),
                    input: Box::new(join.clone()),
                }),
            ),
            (
                "distinct",
                LogicalOperator::Distinct(DistinctOp {
                    input: Box::new(join.clone()),
                    columns: None,
                }),
            ),
            (
                "limit",
                LogicalOperator::Limit(LimitOp {
                    count: CountExpr::Literal(1),
                    input: Box::new(join.clone()),
                }),
            ),
            (
                "skip",
                LogicalOperator::Skip(SkipOp {
                    count: CountExpr::Literal(1),
                    input: Box::new(join.clone()),
                }),
            ),
            (
                "return",
                LogicalOperator::Return(ReturnOp {
                    items: Vec::new(),
                    distinct: false,
                    input: Box::new(join.clone()),
                }),
            ),
            (
                "construct",
                LogicalOperator::Construct(ConstructOp {
                    templates: Vec::new(),
                    input: Box::new(join.clone()),
                }),
            ),
            (
                "unwind",
                LogicalOperator::Unwind(UnwindOp {
                    expression: LogicalExpression::List(Vec::new()),
                    variable: "item".to_string(),
                    ordinality_var: None,
                    offset_var: None,
                    input: Box::new(join.clone()),
                }),
            ),
            (
                "union",
                LogicalOperator::Union(UnionOp {
                    inputs: vec![LogicalOperator::Empty, join.clone()],
                }),
            ),
            (
                "left join",
                LogicalOperator::LeftJoin(LeftJoinOp {
                    left: Box::new(LogicalOperator::Empty),
                    right: Box::new(join.clone()),
                    condition: None,
                    compatibility_conditions: Vec::new(),
                }),
            ),
            (
                "anti join",
                LogicalOperator::AntiJoin(AntiJoinOp {
                    left: Box::new(join.clone()),
                    right: Box::new(LogicalOperator::Empty),
                    compatibility_conditions: Vec::new(),
                    semantics: AntiJoinSemantics::NotExists,
                }),
            ),
        ];
        for (name, wrapper) in wrappers {
            assert!(
                needs_identity_rdf_term_columns(&wrapper),
                "canonical identity metadata was hidden by the {name} wrapper"
            );
        }

        let planner = RdfPlanner::new(Arc::new(RdfStore::new()));
        planner.needs_identity_key_columns.set(true);
        let exact = rdf_exact_term_column("shared");
        let identity = rdf_identity_key_column("shared");
        for input in [&left, &right] {
            let (_, columns, _) = planner.plan_operator(input).unwrap();
            assert_eq!(
                columns.iter().filter(|column| **column == identity).count(),
                1
            );
            assert_eq!(columns.iter().filter(|column| **column == exact).count(), 0);
        }
        let (_, columns, _) = planner.plan_operator(&join).unwrap();
        assert_eq!(
            columns.iter().filter(|column| **column == identity).count(),
            1
        );
        assert_eq!(columns.iter().filter(|column| **column == exact).count(), 0);
    }

    #[test]
    fn rdf_compatibility_join_indexes_mismatches_without_pairwise_scanning() {
        fn input(prefix: &str, count: usize) -> Box<dyn Operator> {
            let mut chunk =
                DataChunk::with_capacity(&[LogicalType::Any, LogicalType::String], count);
            for index in 0..count {
                chunk
                    .column_mut(0)
                    .unwrap()
                    .push_value(Value::String(format!("urn:{prefix}:{index}").into()));
                chunk
                    .column_mut(1)
                    .unwrap()
                    .push_value(Value::String(format!("<urn:{prefix}:{index}>").into()));
            }
            chunk.set_count(count);
            Box::new(ConstantOperator::new(chunk))
        }

        const ROWS: usize = 128;
        let mut operator = RdfCompatibilityJoinOperator::new(
            input("left", ROWS),
            input("right", ROWS),
            vec![RdfCompatibilityKey {
                left_visible: 0,
                right_visible: 0,
                left_group_key: None,
                right_group_key: None,
                left_identity: Some(1),
                right_identity: Some(1),
                semantics: JoinKeySemantics::SparqlCompatibility,
            }],
            RdfCompatibilityMode::Inner,
            vec![RdfCompatibilityOutputColumn::Coalesce { left: 0, right: 0 }],
            vec![LogicalType::Any],
        );

        assert!(operator.next().unwrap().is_none());
        assert_eq!(operator.work.left_rows, ROWS);
        assert_eq!(operator.work.right_rows, ROWS);
        assert_eq!(operator.work.indexed_rows, ROWS);
        assert_eq!(operator.work.lookups, ROWS);
        assert_eq!(operator.work.emitted_pairs, 0);
        assert!(
            operator.work.indexed_rows + operator.work.lookups < ROWS * ROWS,
            "fixed-shape mismatch work must be linear in input rows, not pairwise"
        );
    }

    #[test]
    fn rdf_compatibility_join_coalesces_visible_and_exact_columns() {
        fn one_row(visible: Value, exact: Value, identity: Value) -> Box<dyn Operator> {
            let mut chunk = DataChunk::with_capacity(
                &[LogicalType::Any, LogicalType::String, LogicalType::String],
                1,
            );
            chunk.column_mut(0).unwrap().push_value(visible);
            chunk.column_mut(1).unwrap().push_value(exact);
            chunk.column_mut(2).unwrap().push_value(identity);
            chunk.set_count(1);
            Box::new(ConstantOperator::new(chunk))
        }

        let columns = vec![
            "term".to_string(),
            rdf_exact_term_column("term"),
            rdf_identity_key_column("term"),
        ];
        let types = vec![LogicalType::Any, LogicalType::String, LogicalType::String];
        let conditions = vec![JoinCondition {
            left: LogicalExpression::Variable("term".to_string()),
            right: LogicalExpression::Variable("term".to_string()),
            semantics: JoinKeySemantics::SparqlCompatibility,
        }];
        let (mut operator, output_columns, _) = build_rdf_compatibility_join(
            PlannedRdfRelation::new(
                one_row(Value::Null, Value::Null, Value::Null),
                columns.clone(),
                types.clone(),
            ),
            PlannedRdfRelation::new(
                one_row(
                    Value::String("urn:right".into()),
                    Value::String("<urn:right>".into()),
                    Value::String("<urn:right>".into()),
                ),
                columns.clone(),
                types.clone(),
            ),
            &conditions,
            RdfCompatibilityMode::Inner,
        )
        .unwrap();

        let mut expected_columns = columns;
        expected_columns.push(rdf_group_key_column("term"));
        assert_eq!(output_columns, expected_columns, "output names stay unique");
        let chunk = operator.next().unwrap().expect("one compatible row");
        assert_eq!(
            chunk.column(0).unwrap().get_value(0),
            Some(Value::String("urn:right".into()))
        );
        assert_eq!(
            chunk.column(1).unwrap().get_value(0),
            Some(Value::String("<urn:right>".into()))
        );
        assert_eq!(
            chunk.column(2).unwrap().get_value(0),
            Some(Value::String("<urn:right>".into()))
        );
        assert_eq!(
            chunk.column(3).unwrap().get_value(0),
            Some(Value::List(
                vec![Value::Bool(true), Value::String("<urn:right>".into())].into()
            )),
            "the coalesced binding carries normalized row-level identity"
        );
    }

    #[test]
    fn rdf_compatibility_join_falls_back_to_discriminated_native_identity() {
        fn one_row(value: Value) -> Box<dyn Operator> {
            let mut chunk = DataChunk::with_capacity(&[LogicalType::Any], 1);
            chunk.column_mut(0).unwrap().push_value(value);
            chunk.set_count(1);
            Box::new(ConstantOperator::new(chunk))
        }

        let columns = vec!["term".to_string()];
        let types = vec![LogicalType::Any];
        let conditions = vec![JoinCondition {
            left: LogicalExpression::Variable("term".to_string()),
            right: LogicalExpression::Variable("term".to_string()),
            semantics: JoinKeySemantics::SparqlCompatibility,
        }];
        let build = |left, right| {
            build_rdf_compatibility_join(
                PlannedRdfRelation::new(left, columns.clone(), types.clone()),
                PlannedRdfRelation::new(right, columns.clone(), types.clone()),
                &conditions,
                RdfCompatibilityMode::Inner,
            )
            .expect("missing identity provenance is a row-level invariant")
            .0
        };

        let mut unbound = build(one_row(Value::Null), one_row(Value::Null));
        let row = unbound
            .next()
            .expect("unbound compatibility executes")
            .expect("the two unbound mappings are compatible");
        assert!(row.column(0).expect("coalesced column").is_null(0));

        let mut one_sided = build(
            one_row(Value::String("urn:bound".into())),
            one_row(Value::Null),
        );
        let row = one_sided
            .next()
            .expect("an unbound peer needs no identity comparison")
            .expect("the unbound peer is a compatibility wildcard");
        assert_eq!(
            row.column(0).unwrap().get_value(0),
            Some(Value::String("urn:bound".into()))
        );

        let mut compared = build(
            one_row(Value::String("urn:left".into())),
            one_row(Value::String("urn:right".into())),
        );
        assert!(
            compared.next().unwrap().is_none(),
            "different native values remain incompatible without RDF provenance"
        );

        let mut equal = build(
            one_row(Value::String("native".into())),
            one_row(Value::String("native".into())),
        );
        assert!(
            equal.next().unwrap().is_some(),
            "equal native values compare through their typed discriminated keys"
        );
    }

    #[test]
    fn rdf_compatibility_join_does_not_weaken_mixed_strict_keys() {
        fn row(values: Vec<Value>) -> Box<dyn Operator> {
            let types = vec![LogicalType::Any; values.len()];
            let mut chunk = DataChunk::with_capacity(&types, 1);
            for (column, value) in values.into_iter().enumerate() {
                chunk.column_mut(column).unwrap().push_value(value);
            }
            chunk.set_count(1);
            Box::new(ConstantOperator::new(chunk))
        }

        for strict_semantics in [JoinKeySemantics::Value, JoinKeySemantics::RdfTermIdentity] {
            let strict_exact = (strict_semantics == JoinKeySemantics::RdfTermIdentity).then_some(1);
            let mut operator = RdfCompatibilityJoinOperator::new(
                row(vec![Value::Null, Value::Null, Value::Null, Value::Null]),
                row(vec![
                    Value::String("strict".into()),
                    Value::String("\"strict\"".into()),
                    Value::String("urn:wildcard".into()),
                    Value::String("<urn:wildcard>".into()),
                ]),
                vec![
                    RdfCompatibilityKey {
                        left_visible: 0,
                        right_visible: 0,
                        left_group_key: None,
                        right_group_key: None,
                        left_identity: strict_exact,
                        right_identity: strict_exact,
                        semantics: strict_semantics,
                    },
                    RdfCompatibilityKey {
                        left_visible: 2,
                        right_visible: 2,
                        left_group_key: None,
                        right_group_key: None,
                        left_identity: Some(3),
                        right_identity: Some(3),
                        semantics: JoinKeySemantics::SparqlCompatibility,
                    },
                ],
                RdfCompatibilityMode::Inner,
                vec![RdfCompatibilityOutputColumn::Left(0)],
                vec![LogicalType::Any],
            );
            assert!(
                operator.next().unwrap().is_none(),
                "{strict_semantics:?} must remain bound/bound strict inside a compatibility join"
            );
        }
    }

    /// BIND appends an Any-typed column for the computed expression.
    #[test]
    fn test_type_propagation_bind() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));

        let planner = RdfPlanner::new(store);

        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });

        let bind = LogicalOperator::Bind(BindOp {
            input: Box::new(scan),
            variable: "label".to_string(),
            expression: crate::query::plan::LogicalExpression::Variable("name".to_string()),
        });

        let (_op, columns, types) = planner.plan_operator(&bind).unwrap();
        assert_eq!(columns.last(), Some(&"label".to_string()));
        // Input columns are String, BIND column is Any
        let last_idx = types.len() - 1;
        assert_eq!(
            types[last_idx],
            LogicalType::Any,
            "BIND column should be Any"
        );
        for (name, ty) in columns.iter().zip(types.iter()).take(last_idx) {
            if name == "name" {
                assert_eq!(*ty, LogicalType::Any, "object column is Any");
            } else {
                assert_eq!(*ty, LogicalType::String, "column {name} should be String");
            }
        }
    }

    /// Aggregate output preserves proven input types and fixed result types.
    #[test]
    fn test_type_propagation_aggregate() {
        use crate::query::plan::{AggregateExpr, AggregateFunction, AggregateOp};

        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));

        let planner = RdfPlanner::new(store);

        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });

        let agg = LogicalOperator::Aggregate(AggregateOp {
            input: Box::new(scan),
            group_by: vec![crate::query::plan::LogicalExpression::Variable(
                "name".to_string(),
            )],
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
            having: None,
        });

        assert!(
            needs_identity_rdf_term_columns(&agg),
            "a direct RDF variable GROUP BY demands canonical identity without relying on a join"
        );
        let (_op, columns, types) = planner.plan_operator(&agg).unwrap();
        assert_eq!(
            columns
                .iter()
                .filter(|column| !is_rdf_internal_physical_column(column))
                .cloned()
                .collect::<Vec<_>>(),
            vec!["name", "cnt"]
        );
        assert!(
            columns.contains(&rdf_group_key_column("name")),
            "a group output retains compositional RDF-or-native identity"
        );
        assert_eq!(
            types[columns.iter().position(|column| column == "name").unwrap()],
            LogicalType::Any,
            "an RDF object group can contain heterogeneous term value types"
        );
        assert_eq!(
            types[columns.iter().position(|column| column == "cnt").unwrap()],
            LogicalType::Int64,
            "COUNT produces Int64"
        );
    }

    /// Filter, Distinct, Limit, Skip all preserve input types.
    #[test]
    fn test_type_propagation_passthrough_operators() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));

        let planner = RdfPlanner::new(store);

        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });

        // Get baseline types from the scan
        let (_op, scan_columns, scan_types) = planner.plan_operator(&scan).unwrap();

        // Wrap in Limit
        let limited = LogicalOperator::Limit(LimitOp {
            input: Box::new(scan.clone()),
            count: CountExpr::Literal(10),
        });
        let (_op, _cols, limit_types) = planner.plan_operator(&limited).unwrap();
        assert_eq!(scan_types, limit_types, "Limit preserves types");

        // Wrap in Distinct
        let distinct = LogicalOperator::Distinct(DistinctOp {
            input: Box::new(scan.clone()),
            columns: None,
        });
        let (_op, distinct_columns, distinct_types) = planner.plan_operator(&distinct).unwrap();
        let public_distinct_types = distinct_columns
            .iter()
            .zip(&distinct_types)
            .filter_map(|(column, ty)| {
                (!is_rdf_internal_physical_column(column)).then_some(ty.clone())
            })
            .collect::<Vec<_>>();
        assert_eq!(scan_columns.len(), scan_types.len());
        let public_scan_types = scan_columns
            .iter()
            .zip(&scan_types)
            .filter_map(|(column, ty)| {
                (!is_rdf_internal_physical_column(column)).then_some(ty.clone())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            public_scan_types, public_distinct_types,
            "Distinct preserves public types while adding private normalized keys"
        );

        // Wrap in Skip
        let skipped = LogicalOperator::Skip(SkipOp {
            input: Box::new(scan),
            count: CountExpr::Literal(1),
        });
        let (_op, _cols, skip_types) = planner.plan_operator(&skipped).unwrap();
        assert_eq!(scan_types, skip_types, "Skip preserves types");
    }

    #[test]
    fn test_plan_filter_with_comparison() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("20", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/age".to_string()),
            object: TripleComponent::Variable("age".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let filter = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Variable("age".to_string())),
                op: crate::query::plan::BinaryOp::Gt,
                right: Box::new(LogicalExpression::Literal(Value::String("25".into()))),
            },
            input: Box::new(scan),
            pushdown_hint: None,
        });
        let physical = planner.plan(&LogicalPlan::new(filter)).unwrap();
        assert_eq!(physical.columns, vec!["s", "age"]);
        let mut op = physical.operator;
        let mut rows = 0;
        while let Ok(Some(c)) = op.next() {
            rows += c.row_count();
        }
        assert_eq!(rows, 1);
    }

    #[test]
    fn test_plan_aggregate_count_star_fast_path() {
        use crate::query::plan::{AggregateExpr, AggregateFunction, AggregateOp};
        let store = Arc::new(RdfStore::new());
        for i in 0..10 {
            store.insert(Triple::new(
                Term::iri(format!("http://example.org/item{i}")),
                Term::iri("http://example.org/value"),
                Term::literal(i.to_string()),
            ));
        }
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let agg = LogicalOperator::Aggregate(AggregateOp {
            input: Box::new(scan),
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
            having: None,
        });
        let physical = planner.plan(&LogicalPlan::new(agg)).unwrap();
        let mut op = physical.operator;
        let chunk = op.next().unwrap().unwrap();
        assert_eq!(
            chunk.column(0).unwrap().get_value(0),
            Some(Value::Int64(10))
        );
    }

    #[test]
    fn test_plan_aggregate_with_having() {
        use crate::query::plan::{AggregateExpr, AggregateFunction, AggregateOp};
        let store = Arc::new(RdfStore::new());
        for i in 0..3 {
            store.insert(Triple::new(
                Term::iri("http://example.org/alix"),
                Term::iri(format!("http://example.org/p{i}")),
                Term::literal(format!("val{i}")),
            ));
        }
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://example.org/p0"),
            Term::literal("gus_val"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let agg = LogicalOperator::Aggregate(AggregateOp {
            input: Box::new(scan),
            group_by: vec![LogicalExpression::Variable("s".to_string())],
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
            having: Some(LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Variable("cnt".to_string())),
                op: crate::query::plan::BinaryOp::Gt,
                right: Box::new(LogicalExpression::Literal(Value::String("1".into()))),
            }),
        });
        let physical = planner.plan(&LogicalPlan::new(agg)).unwrap();
        let mut op = physical.operator;
        let mut rows = 0;
        while let Ok(Some(c)) = op.next() {
            rows += c.row_count();
        }
        assert_eq!(rows, 1);
    }

    #[test]
    fn test_plan_aggregate_sum_avg() {
        use crate::query::plan::{AggregateExpr, AggregateFunction, AggregateOp};
        let store = Arc::new(RdfStore::new());
        for i in 1..=4 {
            store.insert(Triple::new(
                Term::iri("http://example.org/a"),
                Term::iri("http://example.org/val"),
                Term::typed_literal(i.to_string(), "http://www.w3.org/2001/XMLSchema#integer"),
            ));
        }
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/val".to_string()),
            object: TripleComponent::Variable("v".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let agg = LogicalOperator::Aggregate(AggregateOp {
            input: Box::new(scan),
            group_by: vec![],
            aggregates: vec![
                AggregateExpr {
                    function: AggregateFunction::Sum,
                    expression: Some(LogicalExpression::Variable("v".to_string())),
                    expression2: None,
                    distinct_key: None,
                    distinct: false,
                    alias: Some("total".to_string()),
                    percentile: None,
                    separator: None,
                },
                AggregateExpr {
                    function: AggregateFunction::Avg,
                    expression: Some(LogicalExpression::Variable("v".to_string())),
                    expression2: None,
                    distinct_key: None,
                    distinct: false,
                    alias: Some("average".to_string()),
                    percentile: None,
                    separator: None,
                },
            ],
            having: None,
        });
        let physical = planner.plan(&LogicalPlan::new(agg)).unwrap();
        assert_eq!(physical.columns, vec!["total", "average"]);
    }

    #[test]
    fn test_plan_join_execution() {
        use crate::query::plan::JoinOp;
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Gus"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let left = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let right = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/age".to_string()),
            object: TripleComponent::Variable("age".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let join = LogicalOperator::Join(JoinOp {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
            conditions: vec![value_join_condition("s")],
        });
        let physical = planner.plan(&LogicalPlan::new(join)).unwrap();
        let mut op = physical.operator;
        let mut rows = 0;
        while let Ok(Some(c)) = op.next() {
            rows += c.row_count();
        }
        assert_eq!(rows, 1);
    }

    #[test]
    fn test_plan_left_join() {
        use crate::query::plan::LeftJoinOp;
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Gus"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let left = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let right = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/age".to_string()),
            object: TripleComponent::Variable("age".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let join = LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(left),
            right: Box::new(right),
            condition: None,
            compatibility_conditions: vec![value_join_condition("s")],
        });
        let physical = planner.plan(&LogicalPlan::new(join)).unwrap();
        let mut op = physical.operator;
        let mut rows = 0;
        while let Ok(Some(c)) = op.next() {
            rows += c.row_count();
        }
        assert_eq!(rows, 2);
    }

    #[test]
    fn test_plan_anti_join() {
        use crate::query::plan::AntiJoinOp;
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Gus"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let left = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let right = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/age".to_string()),
            object: TripleComponent::Variable("age".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let anti = LogicalOperator::AntiJoin(AntiJoinOp {
            left: Box::new(left),
            right: Box::new(right),
            compatibility_conditions: vec![value_join_condition("s")],
            semantics: crate::query::plan::AntiJoinSemantics::Minus,
        });
        let physical = planner.plan(&LogicalPlan::new(anti)).unwrap();
        let mut op = physical.operator;
        let mut rows = 0;
        while let Ok(Some(c)) = op.next() {
            rows += c.row_count();
        }
        assert_eq!(rows, 1);
    }

    #[test]
    fn test_plan_sort() {
        use crate::query::plan::{SortKey, SortOp, SortOrder};
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/c"),
            Term::iri("http://example.org/val"),
            Term::literal("3"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/val"),
            Term::literal("1"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/b"),
            Term::iri("http://example.org/val"),
            Term::literal("2"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/val".to_string()),
            object: TripleComponent::Variable("v".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let sort = LogicalOperator::Sort(SortOp {
            keys: vec![SortKey {
                expression: LogicalExpression::Variable("v".to_string()),
                order: SortOrder::Ascending,
                nulls: None,
            }],
            input: Box::new(scan),
        });
        let physical = planner.plan(&LogicalPlan::new(sort)).unwrap();
        assert_eq!(physical.columns, vec!["s", "v"]);
        let mut op = physical.operator;
        let mut vals = Vec::new();
        while let Ok(Some(chunk)) = op.next() {
            if let Some(col) = chunk.column(1) {
                for row in 0..chunk.row_count() {
                    if let Some(v) = col.get_value(row) {
                        vals.push(v);
                    }
                }
            }
        }
        assert_eq!(
            vals,
            vec![
                Value::String("1".into()),
                Value::String("2".into()),
                Value::String("3".into())
            ]
        );
    }

    #[test]
    fn test_plan_insert_triple_concrete() {
        let store = Arc::new(RdfStore::new());
        let planner = RdfPlanner::new(Arc::clone(&store));
        let insert = LogicalOperator::InsertTriple(InsertTripleOp {
            subject: TripleComponent::Iri("http://example.org/alix".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Literal(Value::String("Alix".into())),
            graph: None,
            input: None,
        });
        let physical = planner.plan(&LogicalPlan::new(insert)).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_plan_insert_triple_into_named_graph() {
        let store = Arc::new(RdfStore::new());
        let planner = RdfPlanner::new(Arc::clone(&store));
        let insert = LogicalOperator::InsertTriple(InsertTripleOp {
            subject: TripleComponent::Iri("http://example.org/alix".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Literal(Value::String("Alix".into())),
            graph: Some("http://example.org/g1".to_string()),
            input: None,
        });
        let physical = planner.plan(&LogicalPlan::new(insert)).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
        assert_eq!(store.graph("http://example.org/g1").unwrap().len(), 1);
    }

    #[test]
    fn test_plan_delete_triple_concrete() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let delete = LogicalOperator::DeleteTriple(DeleteTripleOp {
            subject: TripleComponent::Iri("http://example.org/alix".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Literal(Value::String("Alix".into())),
            graph: None,
            input: None,
        });
        let physical = planner.plan(&LogicalPlan::new(delete)).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_plan_insert_pattern_from_where() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let where_scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let insert = LogicalOperator::InsertTriple(InsertTripleOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/label".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: Some(Box::new(where_scan)),
        });
        let physical = planner.plan(&LogicalPlan::new(insert)).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_plan_delete_pattern_from_where() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Gus"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let where_scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let delete = LogicalOperator::DeleteTriple(DeleteTripleOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: Some(Box::new(where_scan)),
        });
        let physical = planner.plan(&LogicalPlan::new(delete)).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_plan_modify_delete_insert_where() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let where_scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("old".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let modify = LogicalOperator::Modify(ModifyOp {
            delete_templates: vec![TripleTemplate {
                subject: TripleComponent::Variable("s".to_string()),
                predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
                object: TripleComponent::Variable("old".to_string()),
                graph: None,
            }],
            insert_templates: vec![TripleTemplate {
                subject: TripleComponent::Variable("s".to_string()),
                predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
                object: TripleComponent::Literal(Value::String("Alix R.".into())),
                graph: None,
            }],
            where_clause: Box::new(where_scan),
            graph: None,
        });
        let physical = planner.plan(&LogicalPlan::new(modify)).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
        assert_eq!(store.len(), 1);
        let all = store.find(&TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        });
        assert_eq!(all[0].object().to_string(), "\"Alix R.\"");
    }

    #[test]
    fn public_modify_retains_legacy_scalar_and_blank_template_semantics() {
        let store = Arc::new(RdfStore::new());
        let source_predicate = "http://example.org/public-modify-source";
        for subject in ["http://example.org/a", "http://example.org/b"] {
            store.insert(Triple::new(
                Term::iri(subject),
                Term::iri(source_predicate),
                Term::literal("source"),
            ));
        }
        let planner = RdfPlanner::new(Arc::clone(&store));
        let where_scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("source".to_string()),
            predicate: TripleComponent::Iri(source_predicate.to_string()),
            object: TripleComponent::Literal(Value::String("source".into())),
            graph: None,
            input: None,
            dataset: None,
        });
        let where_clause = LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::Literal(Value::String(
                "http://example.org/legacy-subject".into(),
            )),
            variable: "legacy_subject".to_string(),
            input: Box::new(where_scan),
        });
        let result_predicate = "http://example.org/public-modify-result";
        let modify = LogicalOperator::Modify(ModifyOp {
            delete_templates: Vec::new(),
            insert_templates: vec![
                TripleTemplate {
                    subject: TripleComponent::Variable("legacy_subject".to_string()),
                    predicate: TripleComponent::Iri(result_predicate.to_string()),
                    object: TripleComponent::Literal(Value::String("scalar".into())),
                    graph: None,
                },
                TripleTemplate {
                    subject: TripleComponent::BlankNode("legacy-blank".to_string()),
                    predicate: TripleComponent::Iri(result_predicate.to_string()),
                    object: TripleComponent::Literal(Value::String("blank".into())),
                    graph: None,
                },
            ],
            where_clause: Box::new(where_clause),
            graph: None,
        });

        let mut physical = planner.plan(&LogicalPlan::new(modify)).unwrap().operator;
        while physical.next().unwrap().is_some() {}

        assert!(store.contains(&Triple::new(
            Term::iri("http://example.org/legacy-subject"),
            Term::iri(result_predicate),
            Term::literal("scalar"),
        )));
        assert!(store.contains(&Triple::new(
            Term::blank("legacy-blank"),
            Term::iri(result_predicate),
            Term::literal("blank"),
        )));
    }

    #[test]
    fn modify_operator_respects_physical_selection_indices() {
        struct OneChunk {
            chunk: Option<DataChunk>,
        }

        impl Operator for OneChunk {
            fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
                Ok(self.chunk.take())
            }

            fn reset(&mut self) {}

            fn name(&self) -> &'static str {
                "OneChunk"
            }

            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }

        let first = Term::iri("http://example.org/physical-first");
        let selected = Term::iri("http://example.org/physical-selected");
        let predicate = Term::iri("http://example.org/physical-predicate");
        let object = Term::literal("old");
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            first.clone(),
            predicate.clone(),
            object.clone(),
        ));
        store.insert(Triple::new(
            selected.clone(),
            predicate.clone(),
            object.clone(),
        ));

        let mut chunk = DataChunk::new(vec![
            grafeo_core::execution::ValueVector::from_values(&[
                Value::String("http://example.org/physical-first".into()),
                Value::String("http://example.org/physical-selected".into()),
            ]),
            grafeo_core::execution::ValueVector::from_values(&[
                Value::String(first.to_ntriples().into()),
                Value::String(selected.to_ntriples().into()),
            ]),
        ]);
        let mut selection = grafeo_core::execution::SelectionVector::new_empty();
        selection.push(1);
        chunk.set_selection(selection);

        let mut operator = RdfModifyOperator::new(
            Arc::clone(&store),
            Box::new(OneChunk { chunk: Some(chunk) }),
            vec![TripleTemplate {
                subject: TripleComponent::Variable("subject".to_string()),
                predicate: TripleComponent::Iri(
                    "http://example.org/physical-predicate".to_string(),
                ),
                object: TripleComponent::Literal(Value::String("old".into())),
                graph: None,
            }],
            Vec::new(),
            HashMap::from([
                ("subject".to_string(), 0),
                (rdf_exact_term_column("subject"), 1),
            ]),
            true,
            RdfModifyContext {
                transaction_id: None,
                valid_time: None,
                #[cfg(feature = "wal")]
                wal: None,
                #[cfg(feature = "cdc")]
                cdc_log: None,
            },
        );
        assert!(operator.next().unwrap().is_none());

        assert!(store.contains(&Triple::new(first, predicate.clone(), object.clone())));
        assert!(!store.contains(&Triple::new(selected, predicate, object)));
    }

    #[test]
    fn test_plan_union() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://example.org/label"),
            Term::literal("Gus"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan1 = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let scan2 = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/label".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let union = LogicalOperator::Union(crate::query::plan::UnionOp {
            inputs: vec![scan1, scan2],
        });
        let physical = planner.plan(&LogicalPlan::new(union)).unwrap();
        let mut op = physical.operator;
        let mut rows = 0;
        while let Ok(Some(c)) = op.next() {
            rows += c.row_count();
        }
        assert_eq!(rows, 2);
    }

    #[test]
    fn test_plan_construct() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let construct = LogicalOperator::Construct(ConstructOp {
            templates: vec![TripleTemplate {
                subject: TripleComponent::Variable("s".to_string()),
                predicate: TripleComponent::Iri("http://example.org/label".to_string()),
                object: TripleComponent::Variable("name".to_string()),
                graph: None,
            }],
            input: Box::new(scan),
        });
        let physical = planner.plan(&LogicalPlan::new(construct)).unwrap();
        assert_eq!(physical.columns, vec!["subject", "predicate", "object"]);
        let mut op = physical.operator;
        let mut rows = 0;
        while let Ok(Some(c)) = op.next() {
            rows += c.row_count();
        }
        assert_eq!(rows, 1);
    }

    #[test]
    fn test_plan_bind_execution() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let bind = LogicalOperator::Bind(BindOp {
            input: Box::new(scan),
            variable: "upper".to_string(),
            expression: crate::query::plan::LogicalExpression::FunctionCall {
                name: "UCASE".to_string(),
                args: vec![crate::query::plan::LogicalExpression::Variable(
                    "name".to_string(),
                )],
                distinct: false,
            },
        });
        let physical = planner.plan(&LogicalPlan::new(bind)).unwrap();
        assert!(physical.columns.contains(&"upper".to_string()));
        let mut op = physical.operator;
        let chunk = op.next().unwrap().unwrap();
        let upper_idx = physical.columns.iter().position(|c| c == "upper").unwrap();
        assert_eq!(
            chunk.column(upper_idx).unwrap().get_value(0),
            Some(Value::String("ALIX".into()))
        );
    }

    #[test]
    fn test_plan_project_with_expression() {
        use crate::query::plan::{ProjectOp, Projection};
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let project = LogicalOperator::Project(ProjectOp {
            projections: vec![
                Projection {
                    expression: LogicalExpression::Variable("name".to_string()),
                    alias: Some("n".to_string()),
                },
                Projection {
                    expression: LogicalExpression::Literal(Value::String("const".into())),
                    alias: Some("c".to_string()),
                },
            ],
            input: Box::new(scan),
            pass_through_input: false,
        });
        let physical = planner.plan(&LogicalPlan::new(project)).unwrap();
        assert_eq!(physical.columns, vec!["n", "c"]);
        let mut op = physical.operator;
        let chunk = op.next().unwrap().unwrap();
        assert_eq!(
            chunk.column(0).unwrap().get_value(0),
            Some(Value::String("Alix".into()))
        );
        assert_eq!(
            chunk.column(1).unwrap().get_value(0),
            Some(Value::String("const".into()))
        );
    }

    #[test]
    fn test_plan_create_graph() {
        let store = Arc::new(RdfStore::new());
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::CreateGraph(
                CreateGraphOp {
                    graph: "http://example.org/g1".to_string(),
                    silent: false,
                },
            )))
            .unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
        assert!(store.graph("http://example.org/g1").is_some());
    }

    #[test]
    fn test_plan_create_graph_already_exists_errors() {
        let store = Arc::new(RdfStore::new());
        store.create_graph("http://example.org/g1");
        let planner = RdfPlanner::new(Arc::clone(&store));
        let mut physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::CreateGraph(
                CreateGraphOp {
                    graph: "http://example.org/g1".to_string(),
                    silent: false,
                },
            )))
            .unwrap();
        assert!(physical.operator.next().is_err());
    }

    #[test]
    fn test_plan_create_graph_silent() {
        let store = Arc::new(RdfStore::new());
        store.create_graph("http://example.org/g1");
        let planner = RdfPlanner::new(Arc::clone(&store));
        let mut physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::CreateGraph(
                CreateGraphOp {
                    graph: "http://example.org/g1".to_string(),
                    silent: true,
                },
            )))
            .unwrap();
        assert!(physical.operator.next().is_ok());
    }

    #[test]
    fn test_plan_drop_graph() {
        let store = Arc::new(RdfStore::new());
        store.create_graph("http://example.org/g1");
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::DropGraph(DropGraphOp {
                graph: Some("http://example.org/g1".to_string()),
                silent: false,
            })))
            .unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
        assert!(store.graph("http://example.org/g1").is_none());
    }

    #[test]
    fn test_plan_drop_nonexistent_graph_errors() {
        let store = Arc::new(RdfStore::new());
        let planner = RdfPlanner::new(Arc::clone(&store));
        let mut physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::DropGraph(DropGraphOp {
                graph: Some("http://example.org/nope".to_string()),
                silent: false,
            })))
            .unwrap();
        assert!(physical.operator.next().is_err());
    }

    #[test]
    fn test_plan_drop_default_graph() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/b"),
            Term::literal("c"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::DropGraph(DropGraphOp {
                graph: None,
                silent: false,
            })))
            .unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_plan_clear_graph_default() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/b"),
            Term::literal("c"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::ClearGraph(
                ClearGraphOp {
                    graph: None,
                    silent: false,
                },
            )))
            .unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_plan_clear_all_graphs() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/b"),
            Term::literal("c"),
        ));
        store.create_graph("http://example.org/g1");
        store
            .graph("http://example.org/g1")
            .unwrap()
            .insert(Triple::new(
                Term::iri("http://example.org/x"),
                Term::iri("http://example.org/y"),
                Term::literal("z"),
            ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::ClearGraph(
                ClearGraphOp {
                    graph: Some(String::new()),
                    silent: false,
                },
            )))
            .unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_plan_multi_way_join_hash_fallback() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/knows"),
            Term::iri("http://example.org/gus"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan1 = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let scan2 = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/age".to_string()),
            object: TripleComponent::Variable("age".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let scan3 = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/knows".to_string()),
            object: TripleComponent::Variable("friend".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let mwj = LogicalOperator::MultiWayJoin(crate::query::plan::MultiWayJoinOp {
            inputs: vec![scan1, scan2, scan3],
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("s".to_string()),
                right: LogicalExpression::Variable("s".to_string()),
                semantics: JoinKeySemantics::RdfTermIdentity,
            }],
            shared_variables: vec!["s".to_string()],
        });
        let physical = planner.plan(&LogicalPlan::new(mwj)).unwrap();
        let mut op = physical.operator;
        let mut rows = 0;
        while let Ok(Some(c)) = op.next() {
            rows += c.row_count();
        }
        assert_eq!(rows, 1);
    }

    #[cfg(feature = "ring-index")]
    #[test]
    fn rdf_multi_way_selects_ring_only_for_qualified_identity_scans() {
        let store = Arc::new(RdfStore::new());
        for triple in [
            Triple::new(Term::iri("urn:a"), Term::iri("urn:p"), Term::iri("urn:b")),
            Triple::new(Term::iri("urn:b"), Term::iri("urn:q"), Term::iri("urn:c")),
            Triple::new(Term::iri("urn:c"), Term::iri("urn:r"), Term::iri("urn:a")),
        ] {
            store.insert(triple);
        }
        store.rebuild_ring();
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
        let condition = |variable: &str, semantics| JoinCondition {
            left: LogicalExpression::Variable(variable.to_string()),
            right: LogicalExpression::Variable(variable.to_string()),
            semantics,
        };
        let plan = |semantics| {
            LogicalPlan::new(LogicalOperator::MultiWayJoin(
                crate::query::plan::MultiWayJoinOp {
                    inputs: vec![
                        scan("a", "urn:p", "b"),
                        scan("b", "urn:q", "c"),
                        scan("c", "urn:r", "a"),
                    ],
                    conditions: ["a", "b", "c"]
                        .into_iter()
                        .map(|variable| condition(variable, semantics))
                        .collect(),
                    shared_variables: vec!["a".to_string(), "b".to_string(), "c".to_string()],
                },
            ))
        };
        let planner = RdfPlanner::new(Arc::clone(&store));

        let (_, native_entries) = planner
            .plan_profiled(&plan(JoinKeySemantics::RdfTermIdentity))
            .unwrap();
        assert_eq!(
            native_entries.last().map(|entry| entry.name.as_str()),
            Some("RdfLeapfrog")
        );
        let fused_inputs = native_entries
            .iter()
            .filter(|entry| entry.name == "RdfRingTrieInput")
            .collect::<Vec<_>>();
        assert_eq!(fused_inputs.len(), 3);
        assert!(fused_inputs.iter().all(|entry| {
            entry
                .label
                .contains("[fused; stats unavailable, time in parent]")
        }));

        let (_, fallback_entries) = planner
            .plan_profiled(&plan(JoinKeySemantics::Value))
            .unwrap();
        assert!(
            fallback_entries
                .iter()
                .all(|entry| entry.name != "RdfLeapfrog")
        );

        let assert_fallback = |planner: RdfPlanner, logical: &LogicalPlan, reason: &str| {
            let (_, entries) = planner.plan_profiled(logical).unwrap();
            assert!(
                entries.iter().all(|entry| entry.name != "RdfLeapfrog"),
                "unsupported native shape selected Ring ({reason})"
            );
        };
        let mut graph_scoped = plan(JoinKeySemantics::RdfTermIdentity);
        if let LogicalOperator::MultiWayJoin(join) = &mut graph_scoped.root
            && let LogicalOperator::TripleScan(scan) = &mut join.inputs[0]
        {
            scan.graph = Some(TripleComponent::Iri("urn:g".to_string()));
        }
        assert_fallback(
            RdfPlanner::new(Arc::clone(&store)),
            &graph_scoped,
            "named graph",
        );

        let mut dataset_scoped = plan(JoinKeySemantics::RdfTermIdentity);
        if let LogicalOperator::MultiWayJoin(join) = &mut dataset_scoped.root
            && let LogicalOperator::TripleScan(scan) = &mut join.inputs[0]
        {
            scan.dataset = Some(DatasetRestriction {
                default_graphs: vec!["urn:g".to_string()],
                named_graphs: Vec::new(),
            });
        }
        assert_fallback(
            RdfPlanner::new(Arc::clone(&store)),
            &dataset_scoped,
            "dataset restriction",
        );
        assert_fallback(
            RdfPlanner::new(Arc::clone(&store)).with_transaction_id(Some(TransactionId::new(42))),
            &plan(JoinKeySemantics::RdfTermIdentity),
            "transactional overlay",
        );

        store.insert(Triple::new(
            Term::iri("urn:new"),
            Term::iri("urn:p"),
            Term::iri("urn:value"),
        ));
        assert_fallback(
            RdfPlanner::new(store),
            &plan(JoinKeySemantics::RdfTermIdentity),
            "stale Ring",
        );
    }

    #[cfg(feature = "ring-index")]
    #[test]
    fn rdf_native_ring_obeys_a_direct_planner_proven_limit_before_chunk_fill() {
        use grafeo_core::execution::operators::LimitOperator;

        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("urn:a"),
            Term::iri("urn:p"),
            Term::iri("urn:b"),
        ));
        for index in 0..100 {
            let closing = format!("urn:c{index:03}");
            store.insert(Triple::new(
                Term::iri("urn:b"),
                Term::iri("urn:q"),
                Term::iri(closing.as_str()),
            ));
            store.insert(Triple::new(
                Term::iri(closing.as_str()),
                Term::iri("urn:r"),
                Term::iri("urn:a"),
            ));
        }
        store.rebuild_ring();
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
        let join = LogicalOperator::MultiWayJoin(crate::query::plan::MultiWayJoinOp {
            inputs: vec![
                scan("a", "urn:p", "b"),
                scan("b", "urn:q", "c"),
                scan("c", "urn:r", "a"),
            ],
            conditions: ["a", "b", "c"]
                .into_iter()
                .map(|variable| JoinCondition {
                    left: LogicalExpression::Variable(variable.to_string()),
                    right: LogicalExpression::Variable(variable.to_string()),
                    semantics: JoinKeySemantics::RdfTermIdentity,
                })
                .collect(),
            shared_variables: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        });
        assert!(rdf_native_ring_limit_passthrough(&join));
        assert!(!rdf_native_ring_limit_passthrough(
            &LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Literal(Value::Bool(true)),
                input: Box::new(join.clone()),
                pushdown_hint: None,
            })
        ));
        assert!(!rdf_native_ring_limit_passthrough(
            &LogicalOperator::Distinct(DistinctOp {
                input: Box::new(join.clone()),
                columns: None,
            })
        ));
        let logical = LogicalPlan::new(LogicalOperator::Limit(LimitOp {
            count: crate::query::plan::CountExpr::Literal(1),
            input: Box::new(join),
        }));
        let planner = RdfPlanner::new(store);
        planner.needs_identity_key_columns.set(true);
        let (operator, _, _) = planner.plan_operator(&logical.root).unwrap();
        let limit = operator
            .into_any()
            .downcast::<LimitOperator>()
            .expect("physical LIMIT");
        let (mut child, limit) = limit.into_parts();

        assert_eq!(limit, 1);
        assert_eq!(child.name(), "RdfLeapfrog");
        let chunk = child.next().unwrap().expect("one native chunk");
        assert_eq!(
            chunk.row_count(),
            1,
            "the planner-proven LIMIT must cap native enumeration, not only truncate its chunk"
        );
        assert!(child.next().unwrap().is_none());
    }

    #[cfg(feature = "ring-index")]
    #[test]
    fn rdf_native_ring_and_fallback_share_canonical_pattern_constant_semantics() {
        let store = Arc::new(RdfStore::new());
        for triple in [
            Triple::new(
                Term::iri("urn:a"),
                Term::iri("urn:p"),
                Term::lang_literal("x", "EN"),
            ),
            Triple::new(Term::iri("urn:a"), Term::iri("urn:q"), Term::iri("urn:b")),
            Triple::new(Term::iri("urn:b"), Term::iri("urn:r"), Term::iri("urn:a")),
        ] {
            store.insert(triple);
        }
        store.rebuild_ring();
        let scan = |subject: TripleComponent, predicate: &str, object: TripleComponent| {
            LogicalOperator::TripleScan(TripleScanOp {
                subject,
                predicate: TripleComponent::Iri(predicate.to_string()),
                object,
                graph: None,
                input: None,
                dataset: None,
            })
        };
        let logical = LogicalPlan::new(LogicalOperator::MultiWayJoin(
            crate::query::plan::MultiWayJoinOp {
                inputs: vec![
                    scan(
                        TripleComponent::Variable("a".to_string()),
                        "urn:p",
                        TripleComponent::LangLiteral {
                            value: "x".to_string(),
                            lang: "en".to_string(),
                        },
                    ),
                    scan(
                        TripleComponent::Variable("a".to_string()),
                        "urn:q",
                        TripleComponent::Variable("b".to_string()),
                    ),
                    scan(
                        TripleComponent::Variable("b".to_string()),
                        "urn:r",
                        TripleComponent::Variable("a".to_string()),
                    ),
                ],
                conditions: ["a", "b"]
                    .into_iter()
                    .map(|variable| JoinCondition {
                        left: LogicalExpression::Variable(variable.to_string()),
                        right: LogicalExpression::Variable(variable.to_string()),
                        semantics: JoinKeySemantics::RdfTermIdentity,
                    })
                    .collect(),
                shared_variables: vec!["a".to_string(), "b".to_string()],
            },
        ));
        let execute = |planner: RdfPlanner| {
            let mut physical = planner.plan(&logical).unwrap().operator;
            let mut rows = 0;
            while let Some(chunk) = physical.next().unwrap() {
                rows += chunk.row_count();
            }
            rows
        };

        assert_eq!(execute(RdfPlanner::new(Arc::clone(&store))), 1);
        assert_eq!(
            execute(RdfPlanner::new(Arc::clone(&store)).with_native_ring_enabled(false),),
            1,
            "forced hash must use the same canonical RDF constant semantics"
        );
        store.insert(Triple::new(
            Term::iri("urn:stale"),
            Term::iri("urn:noise"),
            Term::iri("urn:value"),
        ));
        assert_eq!(
            execute(RdfPlanner::new(store)),
            1,
            "stale-Ring fallback must not change constant matching"
        );
    }

    #[cfg(feature = "ring-index")]
    #[test]
    fn rdf_native_ring_and_fallback_use_the_same_stable_physical_representative_owner() {
        let store = Arc::new(RdfStore::new());
        for triple in [
            Triple::new(
                Term::iri("urn:s1"),
                Term::iri("urn:p"),
                Term::lang_literal("x", "EN"),
            ),
            Triple::new(
                Term::iri("urn:s2"),
                Term::iri("urn:p"),
                Term::lang_literal("y", "EN"),
            ),
            Triple::new(
                Term::iri("urn:s3"),
                Term::iri("urn:q"),
                Term::lang_literal("x", "en"),
            ),
            Triple::new(
                Term::iri("urn:s4"),
                Term::iri("urn:r"),
                Term::lang_literal("x", "En"),
            ),
        ] {
            store.insert(triple);
        }
        store.rebuild_ring();
        let scan = |predicate: &str| {
            LogicalOperator::TripleScan(TripleScanOp {
                subject: if predicate == "p" {
                    TripleComponent::Variable("a".to_string())
                } else {
                    TripleComponent::Iri(format!("urn:s{}", if predicate == "q" { 3 } else { 4 }))
                },
                predicate: TripleComponent::Iri(format!("urn:{predicate}")),
                object: TripleComponent::Variable("term".to_string()),
                graph: None,
                input: None,
                dataset: None,
            })
        };
        let logical = |order: [&str; 3]| {
            LogicalOperator::MultiWayJoin(crate::query::plan::MultiWayJoinOp {
                inputs: order.into_iter().map(scan).collect(),
                conditions: vec![JoinCondition {
                    left: LogicalExpression::Variable("term".to_string()),
                    right: LogicalExpression::Variable("term".to_string()),
                    semantics: JoinKeySemantics::RdfTermIdentity,
                }],
                shared_variables: vec!["term".to_string()],
            })
        };
        let execute = |planner: RdfPlanner, logical: &LogicalOperator| {
            planner.needs_exact_term_columns.set(true);
            planner.needs_identity_key_columns.set(true);
            let (mut operator, columns, types) = planner.plan_operator(logical).unwrap();
            let term_index = columns.iter().position(|column| column == "term").unwrap();
            let exact_index = columns
                .iter()
                .position(|column| column == &rdf_exact_term_column("term"))
                .unwrap();
            let chunk = operator.next().unwrap().expect("one joined row");
            assert_eq!(chunk.row_count(), 1);
            (
                columns,
                types,
                chunk.column(term_index).unwrap().get_value(0).unwrap(),
                chunk.column(exact_index).unwrap().get_value(0).unwrap(),
            )
        };
        for (order, expected_language) in [(["p", "q", "r"], "en"), (["p", "r", "q"], "En")] {
            let logical = logical(order);
            let LogicalOperator::MultiWayJoin(join) = &logical else {
                unreachable!()
            };
            assert_eq!(
                rdf_multiway_join_order(join, &store),
                vec![1, 2, 0],
                "subject+predicate-bound relations must precede the predicate-only relation; ties remain stable"
            );
            let native = execute(RdfPlanner::new(Arc::clone(&store)), &logical);
            let forced_hash = execute(
                RdfPlanner::new(Arc::clone(&store)).with_native_ring_enabled(false),
                &logical,
            );

            assert_eq!(native, forced_hash);
            assert!(matches!(
                native.2,
                Value::RdfLiteral {
                    language: Some(ref language),
                    ..
                } if language.as_str() == expected_language
            ));
            assert_eq!(
                native.3,
                Value::String(format!("\"x\"@{expected_language}").into())
            );
        }
    }

    #[cfg(feature = "ring-index")]
    #[test]
    fn rdf_planner_rejects_unnormalized_repeated_variables_before_native_or_fallback() {
        let store = Arc::new(RdfStore::new());
        for triple in [
            Triple::new(Term::iri("urn:a"), Term::iri("urn:p"), Term::iri("urn:a")),
            Triple::new(Term::iri("urn:a"), Term::iri("urn:q"), Term::iri("urn:b")),
            Triple::new(Term::iri("urn:b"), Term::iri("urn:r"), Term::iri("urn:a")),
        ] {
            store.insert(triple);
        }
        store.rebuild_ring();
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
        let logical = LogicalPlan::new(LogicalOperator::MultiWayJoin(
            crate::query::plan::MultiWayJoinOp {
                inputs: vec![
                    scan("a", "urn:p", "a"),
                    scan("a", "urn:q", "b"),
                    scan("b", "urn:r", "a"),
                ],
                conditions: ["a", "b"]
                    .into_iter()
                    .map(|variable| JoinCondition {
                        left: LogicalExpression::Variable(variable.to_string()),
                        right: LogicalExpression::Variable(variable.to_string()),
                        semantics: JoinKeySemantics::RdfTermIdentity,
                    })
                    .collect(),
                shared_variables: vec!["a".to_string(), "b".to_string()],
            },
        ));
        let assert_rejected =
            |planner: RdfPlanner, plan: &LogicalPlan, variable: &str, mode: &str| {
                let error = planner
                    .plan(plan)
                    .err()
                    .unwrap_or_else(|| panic!("{mode} accepted an unnormalized repeated variable"));
                assert!(
                    error
                        .to_string()
                        .contains(&format!("repeated variable ?{variable}"))
                        && error.to_string().contains("normalize"),
                    "unexpected {mode} error: {error}"
                );
            };

        assert_rejected(
            RdfPlanner::new(Arc::clone(&store)),
            &logical,
            "a",
            "fresh native",
        );
        assert_rejected(
            RdfPlanner::new(Arc::clone(&store)).with_native_ring_enabled(false),
            &logical,
            "a",
            "forced typed fallback",
        );
        store.insert(Triple::new(
            Term::iri("urn:new"),
            Term::iri("urn:p"),
            Term::iri("urn:new"),
        ));
        assert_rejected(
            RdfPlanner::new(Arc::clone(&store)),
            &logical,
            "a",
            "stale typed fallback",
        );

        let repeated_graph = LogicalPlan::new(LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("g".to_string()),
            predicate: TripleComponent::Iri("urn:p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: Some(TripleComponent::Variable("g".to_string())),
            input: None,
            dataset: None,
        }));
        assert_rejected(
            RdfPlanner::new(store),
            &repeated_graph,
            "g",
            "graph-context typed fallback",
        );
    }

    #[cfg(feature = "ring-index")]
    #[test]
    fn rdf_native_ring_matches_forced_hash_for_the_exhaustive_triangle_lattice() {
        use std::collections::BTreeMap;

        let edges = [
            ("urn:a", "urn:p", "urn:b"),
            ("urn:d", "urn:p", "urn:b"),
            ("urn:b", "urn:q", "urn:c"),
            ("urn:b", "urn:q", "urn:d"),
            ("urn:c", "urn:r", "urn:a"),
            ("urn:d", "urn:r", "urn:a"),
            ("urn:c", "urn:r", "urn:d"),
        ];
        let orders = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let scan = |pattern: usize| {
            let (subject, predicate, object) = match pattern {
                0 => ("a", "urn:p", "b"),
                1 => ("b", "urn:q", "c"),
                _ => ("c", "urn:r", "a"),
            };
            LogicalOperator::TripleScan(TripleScanOp {
                subject: TripleComponent::Variable(subject.to_string()),
                predicate: TripleComponent::Iri(predicate.to_string()),
                object: TripleComponent::Variable(object.to_string()),
                graph: None,
                input: None,
                dataset: None,
            })
        };
        let logical = |order: [usize; 3]| {
            LogicalPlan::new(LogicalOperator::MultiWayJoin(
                crate::query::plan::MultiWayJoinOp {
                    inputs: order.into_iter().map(scan).collect(),
                    conditions: ["a", "b", "c"]
                        .into_iter()
                        .map(|variable| JoinCondition {
                            left: LogicalExpression::Variable(variable.to_string()),
                            right: LogicalExpression::Variable(variable.to_string()),
                            semantics: JoinKeySemantics::RdfTermIdentity,
                        })
                        .collect(),
                    shared_variables: vec!["a".to_string(), "b".to_string(), "c".to_string()],
                },
            ))
        };
        let execute = |mut physical: PhysicalPlan| {
            let indices = ["a", "b", "c"].map(|variable| {
                physical
                    .columns
                    .iter()
                    .position(|column| column == variable)
                    .expect("public triangle variable")
            });
            let mut bag = BTreeMap::<Vec<String>, usize>::new();
            while let Some(chunk) = physical.operator.next().unwrap() {
                for row in 0..chunk.row_count() {
                    let tuple = indices
                        .iter()
                        .map(|column| {
                            let value = chunk
                                .column(*column)
                                .and_then(|values| values.get_value(row))
                                .expect("bound triangle value");
                            match value {
                                Value::String(value) => value.to_string(),
                                other => other.to_string(),
                            }
                        })
                        .collect();
                    *bag.entry(tuple).or_default() += 1;
                }
            }
            bag
        };

        for mask in 0u16..128 {
            let store = Arc::new(RdfStore::new());
            for (bit, (subject, predicate, object)) in edges.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    store.insert(Triple::new(
                        Term::iri(*subject),
                        Term::iri(*predicate),
                        Term::iri(*object),
                    ));
                }
            }
            store.rebuild_ring();
            let mut expected = BTreeMap::new();
            for (required, tuple) in [
                (21, ["urn:a", "urn:b", "urn:c"]),
                (41, ["urn:a", "urn:b", "urn:d"]),
                (70, ["urn:d", "urn:b", "urn:c"]),
            ] {
                if mask & required == required {
                    *expected
                        .entry(tuple.into_iter().map(str::to_string).collect::<Vec<_>>())
                        .or_default() += 1;
                }
            }

            for order in orders {
                let plan = logical(order);
                let native = execute(RdfPlanner::new(Arc::clone(&store)).plan(&plan).unwrap());
                let forced_hash = execute(
                    RdfPlanner::new(Arc::clone(&store))
                        .with_native_ring_enabled(false)
                        .plan(&plan)
                        .unwrap(),
                );
                assert_eq!(native, expected, "native mask={mask}, order={order:?}");
                assert_eq!(forced_hash, expected, "hash mask={mask}, order={order:?}");
            }
        }
    }

    #[test]
    fn rdf_multi_way_identity_never_falls_back_to_visible_value_keys() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("urn:a"),
            Term::iri("urn:p"),
            Term::iri("urn:x"),
        ));
        store.insert(Triple::new(
            Term::iri("urn:b"),
            Term::iri("urn:q"),
            Term::literal("urn:x"),
        ));
        store.insert(Triple::new(
            Term::iri("urn:c"),
            Term::iri("urn:r"),
            Term::iri("urn:x"),
        ));
        let scan = |subject: &str, predicate: &str| {
            LogicalOperator::TripleScan(TripleScanOp {
                subject: TripleComponent::Iri(subject.to_string()),
                predicate: TripleComponent::Iri(predicate.to_string()),
                object: TripleComponent::Variable("term".to_string()),
                graph: None,
                input: None,
                dataset: None,
            })
        };
        let logical = LogicalPlan::new(LogicalOperator::MultiWayJoin(
            crate::query::plan::MultiWayJoinOp {
                inputs: vec![
                    scan("urn:a", "urn:p"),
                    scan("urn:b", "urn:q"),
                    scan("urn:c", "urn:r"),
                ],
                conditions: vec![JoinCondition {
                    left: LogicalExpression::Variable("term".to_string()),
                    right: LogicalExpression::Variable("term".to_string()),
                    semantics: JoinKeySemantics::RdfTermIdentity,
                }],
                shared_variables: vec!["term".to_string()],
            },
        ));
        let planner = RdfPlanner::new(store);
        let mut physical = planner.plan(&logical).unwrap().operator;
        assert!(physical.next().unwrap().is_none());
    }

    #[test]
    fn rdf_multi_way_rejects_metadata_that_loses_endpoint_ownership() {
        fn scan(variable: &str, predicate: &str) -> LogicalOperator {
            LogicalOperator::TripleScan(TripleScanOp {
                subject: TripleComponent::Variable(variable.to_string()),
                predicate: TripleComponent::Iri(predicate.to_string()),
                object: TripleComponent::Iri(format!("urn:{variable}:object")),
                graph: None,
                input: None,
                dataset: None,
            })
        }
        let planner = RdfPlanner::new(Arc::new(RdfStore::new()));
        let inputs = vec![scan("x", "urn:p"), scan("x", "urn:q"), scan("x", "urn:r")];
        let condition = |semantics| JoinCondition {
            left: LogicalExpression::Variable("x".to_string()),
            right: LogicalExpression::Variable("x".to_string()),
            semantics,
        };

        let mixed = LogicalPlan::new(LogicalOperator::MultiWayJoin(
            crate::query::plan::MultiWayJoinOp {
                inputs: inputs.clone(),
                conditions: vec![
                    condition(JoinKeySemantics::RdfTermIdentity),
                    condition(JoinKeySemantics::Value),
                ],
                shared_variables: vec!["x".to_string()],
            },
        ));
        assert!(planner.plan(&mixed).is_err());

        let owned = LogicalPlan::new(LogicalOperator::MultiWayJoin(
            crate::query::plan::MultiWayJoinOp {
                inputs: vec![scan("a", "urn:p"), scan("b", "urn:q"), scan("c", "urn:r")],
                conditions: vec![JoinCondition {
                    left: LogicalExpression::Variable("a".to_string()),
                    right: LogicalExpression::Variable("b".to_string()),
                    semantics: JoinKeySemantics::Value,
                }],
                shared_variables: Vec::new(),
            },
        ));
        assert!(planner.plan(&owned).is_err());

        let duplicate = LogicalPlan::new(LogicalOperator::MultiWayJoin(
            crate::query::plan::MultiWayJoinOp {
                inputs,
                conditions: vec![condition(JoinKeySemantics::RdfTermIdentity)],
                shared_variables: vec!["x".to_string(), "x".to_string()],
            },
        ));
        assert!(planner.plan(&duplicate).is_err());

        let duplicate_conditions = LogicalPlan::new(LogicalOperator::MultiWayJoin(
            crate::query::plan::MultiWayJoinOp {
                inputs: vec![scan("x", "urn:p"), scan("x", "urn:q"), scan("x", "urn:r")],
                conditions: vec![
                    condition(JoinKeySemantics::RdfTermIdentity),
                    condition(JoinKeySemantics::RdfTermIdentity),
                ],
                shared_variables: vec!["x".to_string()],
            },
        ));
        assert!(planner.plan(&duplicate_conditions).is_err());

        let absent_from_peers = LogicalPlan::new(LogicalOperator::MultiWayJoin(
            crate::query::plan::MultiWayJoinOp {
                inputs: vec![scan("a", "urn:p"), scan("b", "urn:q"), scan("c", "urn:r")],
                conditions: vec![JoinCondition {
                    left: LogicalExpression::Variable("a".to_string()),
                    right: LogicalExpression::Variable("a".to_string()),
                    semantics: JoinKeySemantics::RdfTermIdentity,
                }],
                shared_variables: vec!["a".to_string()],
            },
        ));
        assert!(planner.plan(&absent_from_peers).is_err());

        let undeclared_overlap = LogicalPlan::new(LogicalOperator::MultiWayJoin(
            crate::query::plan::MultiWayJoinOp {
                inputs: vec![
                    LogicalOperator::TripleScan(TripleScanOp {
                        subject: TripleComponent::Variable("x".to_string()),
                        predicate: TripleComponent::Iri("urn:p".to_string()),
                        object: TripleComponent::Variable("extra".to_string()),
                        graph: None,
                        input: None,
                        dataset: None,
                    }),
                    LogicalOperator::TripleScan(TripleScanOp {
                        subject: TripleComponent::Variable("x".to_string()),
                        predicate: TripleComponent::Iri("urn:q".to_string()),
                        object: TripleComponent::Variable("extra".to_string()),
                        graph: None,
                        input: None,
                        dataset: None,
                    }),
                    scan("x", "urn:r"),
                ],
                conditions: vec![condition(JoinKeySemantics::RdfTermIdentity)],
                shared_variables: vec!["x".to_string()],
            },
        ));
        assert!(planner.plan(&undeclared_overlap).is_err());

        let empty_metadata_overlap = LogicalPlan::new(LogicalOperator::MultiWayJoin(
            crate::query::plan::MultiWayJoinOp {
                inputs: vec![scan("x", "urn:p"), scan("x", "urn:q")],
                conditions: Vec::new(),
                shared_variables: Vec::new(),
            },
        ));
        assert!(planner.plan(&empty_metadata_overlap).is_err());
    }

    #[test]
    fn test_plan_profiled() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        let planner = RdfPlanner::new(store);
        let plan = LogicalPlan::new(LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        }));
        let (physical, entries) = planner.plan_profiled(&plan).unwrap();
        assert_eq!(physical.columns, vec!["s", "p", "o"]);
        assert!(!entries.is_empty());
    }

    #[test]
    fn test_plan_unsupported_operator_returns_error() {
        let store = Arc::new(RdfStore::new());
        let planner = RdfPlanner::new(store);
        let map = LogicalOperator::MapCollect(crate::query::plan::MapCollectOp {
            input: Box::new(LogicalOperator::Empty),
            key_var: "k".to_string(),
            value_var: "v".to_string(),
            alias: "m".to_string(),
        });
        assert!(planner.plan(&LogicalPlan::new(map)).is_err());
    }

    #[test]
    fn test_plan_empty_operator() {
        let store = Arc::new(RdfStore::new());
        let planner = RdfPlanner::new(store);
        let physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::Empty))
            .unwrap();
        assert!(physical.columns.is_empty());
    }

    #[test]
    fn test_component_to_term_conversions() {
        let store = Arc::new(RdfStore::new());
        let planner = RdfPlanner::new(store);
        assert!(matches!(
            planner
                .component_to_term(&TripleComponent::Iri("http://example.org/x".to_string()))
                .unwrap(),
            Term::Iri(_)
        ));
        assert!(matches!(
            planner
                .component_to_term(&TripleComponent::Literal(Value::String("hello".into())))
                .unwrap(),
            Term::Literal(_)
        ));
        assert!(matches!(
            planner
                .component_to_term(&TripleComponent::Literal(Value::Int64(42)))
                .unwrap(),
            Term::Literal(_)
        ));
        assert!(matches!(
            planner
                .component_to_term(&TripleComponent::Literal(Value::Float64(2.72)))
                .unwrap(),
            Term::Literal(_)
        ));
        assert!(matches!(
            planner
                .component_to_term(&TripleComponent::Literal(Value::Bool(true)))
                .unwrap(),
            Term::Literal(_)
        ));
        assert!(matches!(
            planner
                .component_to_term(&TripleComponent::LangLiteral {
                    value: "hello".to_string(),
                    lang: "en".to_string()
                })
                .unwrap(),
            Term::Literal(_)
        ));
        assert!(matches!(
            planner
                .component_to_term(&TripleComponent::BlankNode("b0".to_string()))
                .unwrap(),
            Term::BlankNode(_)
        ));
        assert!(
            planner
                .component_to_term(&TripleComponent::Variable("x".to_string()))
                .is_err()
        );
    }

    #[test]
    fn numeric_literal_validation_covers_xsd_numeric_family_and_facets() {
        for (datatype, lexical) in [
            ("integer", "+0"),
            ("decimal", ".5"),
            ("float", "INF"),
            ("double", "NaN"),
            ("nonPositiveInteger", "0"),
            ("negativeInteger", "-1"),
            ("long", "-9223372036854775808"),
            ("int", "2147483647"),
            ("short", "-32768"),
            ("byte", "127"),
            ("nonNegativeInteger", "0"),
            ("unsignedLong", "18446744073709551615"),
            ("unsignedInt", "4294967295"),
            ("unsignedShort", "65535"),
            ("unsignedByte", "255"),
            ("positiveInteger", "1"),
        ] {
            let literal = Literal::typed(lexical, format!("{}{datatype}", Literal::XSD));
            assert!(
                rdf_numeric_literal_is_valid(&literal),
                "valid xsd:{datatype} lexical form rejected: {lexical}",
            );
        }

        for (datatype, lexical) in [
            ("string", "12"),
            ("integer", "pumpkin"),
            ("decimal", "1e2"),
            ("float", "Infinity"),
            ("nonPositiveInteger", "1"),
            ("negativeInteger", "0"),
            ("long", "9223372036854775808"),
            ("int", "2147483648"),
            ("short", "32768"),
            ("byte", "128"),
            ("nonNegativeInteger", "-1"),
            ("unsignedLong", "18446744073709551616"),
            ("unsignedInt", "4294967296"),
            ("unsignedShort", "65536"),
            ("unsignedByte", "256"),
            ("positiveInteger", "0"),
        ] {
            let literal = Literal::typed(lexical, format!("{}{datatype}", Literal::XSD));
            assert!(
                !rdf_numeric_literal_is_valid(&literal),
                "invalid xsd:{datatype} lexical/facet form accepted: {lexical}",
            );
        }
    }

    #[test]
    fn test_count_fast_path_predicate_bound() {
        use crate::query::plan::{AggregateExpr, AggregateFunction, AggregateOp};
        let store = Arc::new(RdfStore::new());
        for i in 0..5 {
            store.insert(Triple::new(
                Term::iri(format!("http://example.org/item{i}")),
                Term::iri("http://example.org/type"),
                Term::literal(format!("val{i}")),
            ));
        }
        store.insert(Triple::new(
            Term::iri("http://example.org/other"),
            Term::iri("http://example.org/other_pred"),
            Term::literal("x"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/type".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let agg = LogicalOperator::Aggregate(AggregateOp {
            input: Box::new(scan),
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
            having: None,
        });
        let physical = planner.plan(&LogicalPlan::new(agg)).unwrap();
        let mut op = physical.operator;
        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.column(0).unwrap().get_value(0), Some(Value::Int64(5)));
    }

    #[test]
    fn count_fast_path_rejects_independent_distinct_key() {
        use crate::query::plan::{AggregateExpr, AggregateFunction, AggregateOp};

        let store = Arc::new(RdfStore::new());
        let planner = RdfPlanner::new(store);
        let aggregate = AggregateOp {
            input: Box::new(LogicalOperator::TripleScan(TripleScanOp {
                subject: TripleComponent::Variable("s".to_string()),
                predicate: TripleComponent::Variable("p".to_string()),
                object: TripleComponent::Variable("o".to_string()),
                graph: None,
                input: None,
                dataset: None,
            })),
            group_by: vec![],
            aggregates: vec![AggregateExpr {
                function: AggregateFunction::Count,
                expression: None,
                expression2: None,
                distinct_key: Some(LogicalExpression::Variable("rdf_identity".to_string())),
                distinct: false,
                alias: Some("cnt".to_string()),
                percentile: None,
                separator: None,
            }],
            having: None,
        };

        assert!(planner.try_count_fast_path(&aggregate).is_none());
    }

    // ---- into_any() coverage tests ----
    //
    // Each RDF operator implements `into_any()` for downcasting support in the
    // push pipeline. These tests construct minimal instances and verify the
    // method returns a valid `Box<dyn Any + Send>` that can be downcast back.

    #[test]
    fn test_into_any_rdf_insert_triple_operator() {
        let store = Arc::new(RdfStore::new());
        let triple = Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("o"),
        );
        let op: Box<dyn Operator> = Box::new(RdfInsertTripleOperator::new(
            store,
            triple,
            None,
            None,
            None,
            #[cfg(feature = "wal")]
            None,
            #[cfg(feature = "cdc")]
            None,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfInsertTripleOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_insert_pattern_operator() {
        let store = Arc::new(RdfStore::new());
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let operands = TripleOperands {
            subject: TripleComponent::Iri("http://example.org/s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/p".to_string()),
            object: TripleComponent::Literal(Value::String("o".into())),
            column_map: HashMap::new(),
            graph: None,
            transaction_id: None,
            valid_time: None,
        };
        let op: Box<dyn Operator> = Box::new(RdfInsertPatternOperator::new(
            store,
            child,
            operands,
            #[cfg(feature = "wal")]
            None,
            #[cfg(feature = "cdc")]
            None,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfInsertPatternOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_delete_triple_operator() {
        let store = Arc::new(RdfStore::new());
        let triple = Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("o"),
        );
        let op: Box<dyn Operator> = Box::new(RdfDeleteTripleOperator::new(
            store,
            triple,
            None,
            None,
            #[cfg(feature = "wal")]
            None,
            #[cfg(feature = "cdc")]
            None,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfDeleteTripleOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_delete_pattern_operator() {
        let store = Arc::new(RdfStore::new());
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let operands = TripleOperands {
            subject: TripleComponent::Iri("http://example.org/s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/p".to_string()),
            object: TripleComponent::Literal(Value::String("o".into())),
            column_map: HashMap::new(),
            graph: None,
            transaction_id: None,
            valid_time: None,
        };
        let op: Box<dyn Operator> = Box::new(RdfDeletePatternOperator::new(
            store,
            child,
            operands,
            #[cfg(feature = "wal")]
            None,
            #[cfg(feature = "cdc")]
            None,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfDeletePatternOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_clear_graph_operator() {
        let store = Arc::new(RdfStore::new());
        let op: Box<dyn Operator> = Box::new(RdfClearGraphOperator::new(
            store,
            None,
            false,
            None,
            #[cfg(feature = "wal")]
            None,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfClearGraphOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_create_graph_operator() {
        let store = Arc::new(RdfStore::new());
        let op: Box<dyn Operator> = Box::new(RdfCreateGraphOperator::new(
            store,
            "http://example.org/g".to_string(),
            true,
            None,
            #[cfg(feature = "wal")]
            None,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfCreateGraphOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_drop_graph_operator() {
        let store = Arc::new(RdfStore::new());
        let op: Box<dyn Operator> = Box::new(RdfDropGraphOperator::new(
            store,
            Some("http://example.org/g".to_string()),
            true,
            None,
            #[cfg(feature = "wal")]
            None,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfDropGraphOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_copy_graph_operator() {
        let store = Arc::new(RdfStore::new());
        let op: Box<dyn Operator> = Box::new(RdfCopyGraphOperator::new(
            store,
            Some("http://example.org/src".to_string()),
            Some("http://example.org/dst".to_string()),
            true,
            None,
            #[cfg(feature = "wal")]
            None,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfCopyGraphOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_move_graph_operator() {
        let store = Arc::new(RdfStore::new());
        let op: Box<dyn Operator> = Box::new(RdfMoveGraphOperator::new(
            store,
            Some("http://example.org/src".to_string()),
            Some("http://example.org/dst".to_string()),
            true,
            None,
            #[cfg(feature = "wal")]
            None,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfMoveGraphOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_add_graph_operator() {
        let store = Arc::new(RdfStore::new());
        let op: Box<dyn Operator> = Box::new(RdfAddGraphOperator::new(
            store,
            Some("http://example.org/src".to_string()),
            Some("http://example.org/dst".to_string()),
            true,
            None,
            #[cfg(feature = "wal")]
            None,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfAddGraphOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_modify_operator() {
        let store = Arc::new(RdfStore::new());
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let op: Box<dyn Operator> = Box::new(RdfModifyOperator::new(
            store,
            child,
            vec![],
            vec![],
            HashMap::new(),
            false,
            RdfModifyContext {
                transaction_id: None,
                valid_time: None,
                #[cfg(feature = "wal")]
                wal: None,
                #[cfg(feature = "cdc")]
                cdc_log: None,
            },
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfModifyOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_union_operator() {
        let op: Box<dyn Operator> = Box::new(RdfUnionOperator::new(vec![]));
        let any = op.into_any();
        assert!(any.downcast::<RdfUnionOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_bind_operator() {
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let expr = FilterExpression::Literal(Value::String("test".into()));
        let op: Box<dyn Operator> = Box::new(RdfBindOperator::new(child, expr, HashMap::new()));
        let any = op.into_any();
        assert!(any.downcast::<RdfBindOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_project_operator() {
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let op: Box<dyn Operator> = Box::new(RdfProjectOperator::new(
            child,
            vec![RdfProjectExpr::Constant(Value::String("x".into()))],
            vec![LogicalType::String],
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfProjectOperator>().is_ok());
    }

    #[test]
    fn test_into_any_construct_operator() {
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let op: Box<dyn Operator> = Box::new(ConstructOperator::new(child, vec![], HashMap::new()));
        let any = op.into_any();
        assert!(any.downcast::<ConstructOperator>().is_ok());
    }

    #[test]
    fn test_into_any_constant_operator() {
        let chunk = DataChunk::empty();
        let op: Box<dyn Operator> = Box::new(ConstantOperator::new(chunk));
        let any = op.into_any();
        assert!(any.downcast::<ConstantOperator>().is_ok());
    }

    #[test]
    fn test_into_any_dict_resolve_operator() {
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let dict = Arc::new(grafeo_core::graph::rdf::TermDictionary::new());
        let op: Box<dyn Operator> = Box::new(DictResolveOperator::new(child, dict, vec![]));
        let any = op.into_any();
        assert!(any.downcast::<DictResolveOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_triple_scan_operator() {
        let store = Arc::new(RdfStore::new());
        let pattern = TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        };
        let op: Box<dyn Operator> = Box::new(RdfTripleScanOperator::new(
            store,
            pattern,
            RdfTripleScanOutput {
                mask: [true, true, true, false],
                companion_columns: false,
                datatype_column: false,
                term_companions: RdfTermCompanionOutput {
                    lossless: false,
                    identity: false,
                },
            },
            1024,
            GraphContext {
                graph: None,
                scan_all_graphs: false,
                dataset: None,
            },
            None,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfTripleScanOperator>().is_ok());
    }
}
