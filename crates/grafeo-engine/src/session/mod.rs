//! Lightweight handles for database interaction.
//!
//! A session is your conversation with the database. Each session can have
//! its own transaction state, so concurrent sessions don't interfere with
//! each other. Sessions are cheap to create - spin up as many as you need.

#[cfg(feature = "lpg")]
mod catalog_transaction;
#[cfg(feature = "lpg")]
mod commit_publication;
#[cfg(all(test, feature = "lpg", feature = "gql"))]
mod external_store_tests;
#[cfg(feature = "lpg")]
mod graph_lifecycle;
#[cfg(all(test, feature = "lpg", feature = "gql"))]
mod graph_type_tests;
#[cfg(all(test, feature = "lpg"))]
mod index_owner_tests;
#[cfg(feature = "lpg")]
mod index_owners;
#[cfg(feature = "triple-store")]
mod rdf;
#[cfg(any(feature = "lpg", feature = "triple-store"))]
mod savepoint_rollback;
#[cfg(feature = "lpg")]
use catalog_transaction::TransactionCatalog;
#[cfg(feature = "lpg")]
use commit_publication::EngineCommitCapture;
mod results;
mod snapshot;
pub use snapshot::MixedSnapshot;
pub(crate) use snapshot::acquire_publication_read;
#[cfg(all(feature = "gql", feature = "lpg"))]
pub(crate) use snapshot::acquire_publication_read_with_checkpoint;

use std::sync::Arc;
#[cfg(any(feature = "lpg", feature = "triple-store"))]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
#[cfg(any(test, not(target_arch = "wasm32"), feature = "metrics"))]
use std::time::Instant;

#[cfg(feature = "lpg")]
use grafeo_common::grafeo_debug_span;
use grafeo_common::grafeo_info_span;
#[cfg(feature = "lpg")]
use grafeo_common::types::{EdgeId, NodeId};
use grafeo_common::types::{EpochId, GraphPath, TransactionId, Value};
use grafeo_common::utils::error::Result;
#[cfg(all(feature = "lpg", feature = "gql"))]
use grafeo_common::utils::hash::FxHashMap;
#[cfg(feature = "lpg")]
use grafeo_common::utils::hash::FxHashSet;
#[cfg(feature = "lpg")]
use grafeo_core::execution::operators::ConstraintValidator;
#[cfg(feature = "lpg")]
use grafeo_core::graph::Direction;
#[cfg(any(
    feature = "lpg",
    feature = "gql",
    feature = "cypher",
    feature = "gremlin",
    feature = "graphql",
    feature = "sql-pgq"
))]
use grafeo_core::graph::GraphStore;
#[cfg(all(
    feature = "triple-store",
    not(feature = "lpg"),
    any(
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    )
))]
use grafeo_core::graph::NullGraphStore;
#[cfg(feature = "lpg")]
use grafeo_core::graph::TxStructuralSnapshot;
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::TxDelta;
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::{Edge, Node};
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::{LpgCommitWorkspace, LpgStore, with_prepared_lpg_commit};
#[cfg(feature = "triple-store")]
use grafeo_core::graph::rdf::{RdfStore, RdfTransactionSavepoint};
use grafeo_core::graph::{GraphStoreMut, GraphStoreSearch};

#[cfg(any(
    feature = "lpg",
    feature = "gql",
    feature = "cypher",
    feature = "gremlin",
    feature = "graphql",
    feature = "sql-pgq"
))]
use crate::catalog::Catalog;
#[cfg(feature = "lpg")]
use crate::catalog::CatalogConstraintValidator;
#[cfg(feature = "lpg")]
use crate::catalog::IndexConfiguration;
#[cfg(feature = "lpg")]
use crate::catalog::{CatalogRead, CatalogReadGuard, CatalogWorkspace, ReadyCatalog};
use crate::config::{AdaptiveConfig, GraphModel};
use crate::database::QueryResult;
use crate::query::Executor;
#[cfg(any(feature = "gql", feature = "cypher"))]
use crate::query::cache::PhysicalCacheKey;
use crate::query::cache::{PhysicalPlanCache, QueryCache};
#[cfg(any(
    feature = "gql",
    feature = "cypher",
    feature = "gremlin",
    feature = "graphql",
    feature = "sql-pgq",
    feature = "sparql"
))]
use crate::query::planner::PhysicalPlan;
#[cfg(any(feature = "gql", feature = "cypher"))]
use crate::query::processor::QueryLanguage;
#[cfg(not(feature = "lpg"))]
use crate::transaction::TransactionManager;
#[cfg(feature = "lpg")]
use crate::transaction::prop_tag;
#[cfg(feature = "lpg")]
use crate::transaction::{ConflictGranularity, EntityId, TransactionManager};
#[cfg(feature = "lpg")]
use grafeo_common::types::IndexId;
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::{PhysicalIndexFamily, PhysicalIndexKey};

#[cfg(feature = "spill")]
fn finish_spill_execution<T>(
    execution: Result<T>,
    spill_manager: Option<Arc<grafeo_core::execution::spill::SpillManager>>,
) -> Result<T> {
    let cleanup = spill_manager
        .map(|manager| manager.finish_query())
        .transpose();
    match (execution, cleanup) {
        (Ok(output), Ok(_)) => Ok(output),
        (Ok(_), Err(cleanup_error)) => Err(cleanup_error.into()),
        (Err(primary), Ok(_)) => Err(primary),
        (Err(primary), Err(cleanup_error)) => {
            Err(primary.with_context(format!("spill query cleanup also failed: {cleanup_error}")))
        }
    }
}

/// Storage key suffix for the implicit default graph within a schema.
/// Auto-created by `CREATE SCHEMA` and auto-dropped by `DROP SCHEMA`.
pub(crate) const SCHEMA_DEFAULT_GRAPH: &str = "__default__";

/// Savepoint namespace reserved for engine-owned statement and nesting frames.
/// A leading NUL cannot be produced by the query grammars, and public API
/// calls reject it explicitly so caller names cannot collide with protocol
/// state.
const INTERNAL_SAVEPOINT_PREFIX: &str = "\0grafeo:";

/// Process-wide source of opaque, parser-inexpressible statement savepoint
/// names. A checked counter avoids silently reusing a live savepoint name.
#[cfg(any(feature = "lpg", feature = "triple-store"))]
static NEXT_STATEMENT_SAVEPOINT: AtomicU64 = AtomicU64::new(1);

/// Provenance for a process-local, read-only LPG projection.
///
/// Virtual projections are deliberately distinct from the durable RDF→LPG
/// projection registry. The exact named-store `Arc` is retained so graph
/// lifecycle publication can cascade only the incarnation the view was built
/// over; a caller that already cloned the view keeps a detached, readable
/// handle after its registry name is removed.
#[cfg(feature = "lpg")]
pub(crate) enum VirtualProjectionSource {
    /// The built-in default graph or a caller-supplied external read store.
    RootOrExternal,
    /// One exact built-in named-graph incarnation.
    #[cfg(feature = "gql")]
    Named {
        storage_key: GraphPath,
        incarnation: Arc<LpgStore>,
    },
}

/// One identity-bearing virtual-projection registry entry.
///
/// The outer `Arc` is also the compare-and-swap token for concurrent DDL.
pub(crate) struct RegisteredGraphProjection {
    projection: Arc<grafeo_core::graph::GraphProjection>,
    #[cfg(feature = "lpg")]
    source: VirtualProjectionSource,
}

impl RegisteredGraphProjection {
    pub(crate) fn root_or_external(projection: Arc<grafeo_core::graph::GraphProjection>) -> Self {
        Self {
            projection,
            #[cfg(feature = "lpg")]
            source: VirtualProjectionSource::RootOrExternal,
        }
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn named(
        projection: Arc<grafeo_core::graph::GraphProjection>,
        storage_key: GraphPath,
        incarnation: Arc<LpgStore>,
    ) -> Self {
        Self {
            projection,
            source: VirtualProjectionSource::Named {
                storage_key,
                incarnation,
            },
        }
    }

    pub(crate) fn view(&self) -> Arc<grafeo_core::graph::GraphProjection> {
        Arc::clone(&self.projection)
    }

    #[cfg(feature = "lpg")]
    fn is_owned_by(&self, storage_key: &GraphPath, incarnation: &Arc<LpgStore>) -> bool {
        let Some((path, owner)) = self.source_graph() else {
            return false;
        };
        if !path.components().starts_with(storage_key.components()) {
            return false;
        }
        let rest = &path.components()[storage_key.components().len()..];
        // Qualify the descendant against this exact incarnation, not its name.
        let mut target = Some(Arc::clone(incarnation));
        for component in rest {
            target = target.and_then(|store| store.graph(component));
        }
        target
            .as_ref()
            .is_some_and(|target| Arc::ptr_eq(target, owner))
    }

    #[cfg(feature = "lpg")]
    fn source_graph(&self) -> Option<(&GraphPath, &Arc<LpgStore>)> {
        match &self.source {
            VirtualProjectionSource::RootOrExternal => None,
            #[cfg(feature = "gql")]
            VirtualProjectionSource::Named {
                storage_key,
                incarnation,
            } => Some((storage_key, incarnation)),
        }
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn belongs_to_schema(&self, schema: &str) -> bool {
        self.source_graph().is_some_and(|(storage_key, _)| {
            storage_key
                .components()
                .first()
                .is_some_and(|name| CatalogDdlTarget::key_belongs_to_schema(name, schema))
        })
    }
}

pub(crate) type VirtualProjectionRegistry =
    Arc<parking_lot::RwLock<std::collections::HashMap<String, Arc<RegisteredGraphProjection>>>>;

#[cfg(any(
    feature = "lpg",
    feature = "gql",
    feature = "cypher",
    feature = "gremlin",
    feature = "graphql",
    feature = "sql-pgq"
))]
std::thread_local! {
    /// Temporary historical cuts belong to one invocation on one thread.
    ///
    /// Keeping them out of `Session` state prevents a concurrent query on the
    /// same Session from observing another caller's `execute_at_epoch` scope.
    static SCOPED_VIEWING_EPOCHS: std::cell::RefCell<Vec<(usize, EpochId)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Unwind-safe ownership of one thread-local historical execution scope.
#[cfg(feature = "gql")]
struct ScopedViewingEpochGuard {
    session_key: usize,
}

#[cfg(feature = "gql")]
impl Drop for ScopedViewingEpochGuard {
    fn drop(&mut self) {
        SCOPED_VIEWING_EPOCHS.with(|scopes| {
            let popped = scopes.borrow_mut().pop();
            debug_assert_eq!(popped.map(|(key, _)| key), Some(self.session_key));
        });
    }
}

/// One validated, linearizable LPG coordinate and its language selector metadata.
#[derive(Clone, Default, PartialEq, Eq)]
pub(crate) struct SessionGraphContext {
    pub(crate) graph: Option<String>,
    pub(crate) schema: Option<String>,
    pub(crate) storage_key: GraphPath,
    /// Native paths do not re-resolve when the language schema changes.
    pub(crate) native: bool,
}

/// Parses a DDL default-value literal string into a [`Value`].
///
/// Handles string literals (single- or double-quoted), integers, floats,
/// booleans (`true`/`false`), and `NULL`.
#[cfg(all(feature = "gql", feature = "lpg"))]
fn parse_default_literal(text: &str) -> Value {
    if text.eq_ignore_ascii_case("null") {
        return Value::Null;
    }
    if text.eq_ignore_ascii_case("true") {
        return Value::Bool(true);
    }
    if text.eq_ignore_ascii_case("false") {
        return Value::Bool(false);
    }
    // String literal: strip surrounding quotes
    if (text.starts_with('\'') && text.ends_with('\''))
        || (text.starts_with('"') && text.ends_with('"'))
    {
        return Value::String(text[1..text.len() - 1].into());
    }
    // Try integer, then float
    if let Ok(i) = text.parse::<i64>() {
        return Value::Int64(i);
    }
    if let Ok(f) = text.parse::<f64>() {
        return Value::Float64(f);
    }
    // Fallback: treat as string
    Value::String(text.into())
}

/// Runtime configuration for creating a new session.
///
/// Groups the shared parameters passed to all session constructors, keeping
/// call sites readable and avoiding long argument lists.
pub(crate) struct SessionConfig {
    pub transaction_manager: Arc<TransactionManager>,
    pub query_cache: Arc<QueryCache>,
    /// Shared with the owning database so one-shot `execute` hits the same cache.
    pub physical_cache: Arc<parking_lot::Mutex<PhysicalPlanCache>>,
    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    pub catalog: Arc<Catalog>,
    pub adaptive_config: AdaptiveConfig,
    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    pub factorized_execution: bool,
    pub graph_model: GraphModel,
    pub query_timeout: Option<Duration>,
    pub result_limits: crate::query::ResultLimits,
    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    pub max_property_size: Option<usize>,
    /// Buffer manager for memory-aware query execution.
    pub buffer_manager: Option<Arc<grafeo_common::memory::buffer::BufferManager>>,
    /// Database-owned authenticated root and this session's logical store.
    #[cfg(feature = "spill")]
    pub spill_root: Arc<crate::spill_crypto::DatabaseSpillRoot>,
    #[cfg(any(feature = "spill", feature = "cdc"))]
    pub world_identity: Arc<parking_lot::RwLock<grafeo_common::types::WorldIdentityMetadataV1>>,
    #[cfg(feature = "lpg")]
    pub commit_counter: Arc<AtomicUsize>,
    pub durability_poisoned: Arc<std::sync::atomic::AtomicBool>,
    /// Shared terminal lifecycle state owned by the database.
    pub database_open: Arc<parking_lot::RwLock<bool>>,
    /// Shared count used to exclude store-replacing maintenance while a
    /// Session could retain an old overlay pointer.
    pub active_sessions: Arc<AtomicUsize>,
    #[cfg(feature = "lpg")]
    pub gc_interval: usize,
    /// When true, the session permanently blocks all mutations.
    pub read_only: bool,
    /// The identity bound to this session (for permission checks).
    pub identity: crate::auth::Identity,
    /// Named graph projections shared with the database.
    #[cfg(feature = "lpg")]
    pub projections: VirtualProjectionRegistry,
}

/// Your handle to the database - execute queries and manage transactions.
///
/// Get one from [`GrafeoDB::session()`](crate::GrafeoDB::session). Each session
/// tracks its own transaction state, so you can have multiple concurrent
/// sessions without them interfering.
pub struct Session {
    /// The underlying store.
    #[cfg(feature = "lpg")]
    store: Arc<LpgStore>,
    /// Classifies the role of `store` for the active backend.
    /// Search procedures (CALL grafeo.search.*) only reach into `store` when
    /// this is `Active`. External-store sessions keep `store` as a placeholder
    /// and must not expose it: it has no indexes or data.
    #[cfg(feature = "lpg")]
    lpg_backend: LpgBackend,
    /// Graph store trait object for pluggable storage backends (read path).
    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    graph_store: Arc<dyn GraphStoreSearch>,
    /// Writable graph store (None for read-only databases).
    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    graph_store_mut: Option<Arc<dyn GraphStoreMut>>,
    /// Schema and metadata catalog shared across sessions.
    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    catalog: Arc<Catalog>,
    /// RDF triple store (if RDF feature is enabled).
    #[cfg(feature = "triple-store")]
    rdf_store: Arc<RdfStore>,
    /// Transaction manager.
    transaction_manager: Arc<TransactionManager>,
    /// Query cache shared across sessions.
    query_cache: Arc<QueryCache>,
    /// Physical operator trees, shared with the owning [`GrafeoDB`].
    /// One-shot `db.execute` and `session.execute` hit the same cache.
    physical_cache: Arc<parking_lot::Mutex<PhysicalPlanCache>>,
    /// Current transaction ID (if any). Behind a Mutex so that GQL commands
    /// (`START TRANSACTION`, `COMMIT`, `ROLLBACK`) can manage transactions
    /// from within `execute(&self)`.
    current_transaction: parking_lot::Mutex<Option<TransactionId>>,
    /// Private metadata cut, sharing immutable state until a real catalog edit.
    #[cfg(feature = "lpg")]
    transaction_catalog: parking_lot::Mutex<Option<TransactionCatalog>>,
    /// Serializes each store-backed Session's mutation, context, and
    /// transaction-boundary work.
    ///
    /// The lock is reentrant because auto-commit and nested transactions call
    /// the same private begin/commit/savepoint helpers. It remains held across
    /// state mutation, WAL framing, and CDC staging so savepoint snapshots can
    /// never split those three effects when a Session is shared across threads.
    /// Unrestricted sessions without runtime CDC use the same boundary.
    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    mutation_operation_gate: parking_lot::ReentrantMutex<()>,
    /// Serializes persistent historical-mode changes against LPG mutation and
    /// transaction-begin boundaries.
    ///
    /// This is separate from `mutation_operation_gate`. Lock order is session
    /// operation gate, then historical-view gate, then transaction / publication
    /// locks, including when runtime CDC is disabled.
    historical_view_operation_gate: parking_lot::ReentrantMutex<()>,
    /// Whether the current transaction is read-only (blocks mutations).
    read_only_tx: parking_lot::Mutex<bool>,
    /// Whether the database itself is read-only (set at open time, never changes).
    /// When true, `read_only_tx` is always true regardless of transaction flags.
    db_read_only: bool,
    /// The identity bound to this session (determines permission level).
    identity: crate::auth::Identity,
    /// Whether the session is in auto-commit mode.
    auto_commit: bool,
    /// Adaptive execution configuration.
    #[allow(dead_code)] // Stored for future adaptive re-optimization during execution
    adaptive_config: AdaptiveConfig,
    /// Whether to use factorized execution for multi-hop queries.
    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    factorized_execution: bool,
    /// The graph data model this session operates on.
    graph_model: GraphModel,
    /// Maximum time a query may run before being cancelled.
    query_timeout: Option<Duration>,
    result_limits: crate::query::ResultLimits,
    active_result_limits: parking_lot::Mutex<Option<crate::query::ResultLimits>>,
    active_result_admission: parking_lot::Mutex<Option<crate::query::executor::ResultAdmission>>,
    /// Caller-owned execution control installed for one public query.
    active_execution_control:
        parking_lot::Mutex<Option<grafeo_core::execution::QueryExecutionControl>>,
    active_execution_completed: std::sync::atomic::AtomicBool,
    active_execution_statement_depth: std::sync::atomic::AtomicUsize,
    #[cfg(feature = "testing-statement-injection")]
    query_cancellation_test_hook: parking_lot::Mutex<Option<QueryCancellationTestHook>>,
    /// Maximum size in bytes for a single property value.
    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    max_property_size: Option<usize>,
    /// Buffer manager for memory-aware execution (spill decisions).
    buffer_manager: Option<Arc<grafeo_common::memory::buffer::BufferManager>>,
    /// Shared root authority; each query receives its own authenticated lease.
    #[cfg(feature = "spill")]
    spill_root: Arc<crate::spill_crypto::DatabaseSpillRoot>,
    #[cfg(any(feature = "spill", feature = "cdc"))]
    world_identity: Arc<parking_lot::RwLock<grafeo_common::types::WorldIdentityMetadataV1>>,
    /// Shared commit counter for triggering auto-GC.
    #[cfg(feature = "lpg")]
    commit_counter: Arc<AtomicUsize>,
    /// Shared with [`GrafeoDB`]: abort/log/fsync failure poisons all sessions.
    durability_poisoned: Arc<std::sync::atomic::AtomicBool>,
    /// Terminal database lifecycle state. Once false, an existing Session may
    /// no longer begin transactions or execute queries against closed storage.
    database_open: Arc<parking_lot::RwLock<bool>>,
    /// Shared live-Session count; decremented by `Drop`.
    active_sessions: Arc<AtomicUsize>,
    /// GC every N commits (0 = disabled).
    #[cfg(feature = "lpg")]
    gc_interval: usize,
    /// Node count at the start of the current transaction (for PreparedCommit stats).
    #[cfg(feature = "lpg")]
    transaction_start_node_count: AtomicUsize,
    /// Edge count at the start of the current transaction (for PreparedCommit stats).
    #[cfg(feature = "lpg")]
    transaction_start_edge_count: AtomicUsize,
    /// WAL for logging schema changes.
    #[cfg(feature = "wal")]
    wal: Option<Arc<grafeo_storage::wal::LpgWal>>,
    /// CDC log for change tracking.
    #[cfg(feature = "cdc")]
    cdc_log: Arc<crate::cdc::CdcLog>,
    /// Buffered CDC events for the current transaction.
    /// Flushed to `cdc_log` on commit, discarded on rollback.
    #[cfg(feature = "cdc")]
    cdc_pending_events: Option<Arc<crate::cdc::TransactionChangeAccumulator>>,
    /// Typed view of the exact default CDC wrapper also stored in `graph_store_mut`.
    #[cfg(all(feature = "lpg", feature = "cdc"))]
    default_cdc_writer: Option<Arc<crate::database::cdc_store::CdcGraphStore>>,
    /// Resolved graph coordinate and language/schema metadata, published together.
    current_context: parking_lot::Mutex<SessionGraphContext>,
    /// Session time zone override.
    time_zone: parking_lot::Mutex<Option<String>>,
    /// Default application valid-time captured by RDF insert statements.
    ///
    /// This is a session setting, not transaction state. Each pending RDF op
    /// owns the value captured when it was inserted, so later scope changes and
    /// savepoint rollback cannot rewrite already-buffered history.
    #[cfg(feature = "triple-store")]
    rdf_valid_time: parking_lot::Mutex<Option<grafeo_common::types::ValidTimeInterval>>,
    /// Session-level parameters (SET PARAMETER).
    session_params:
        parking_lot::Mutex<std::collections::HashMap<String, grafeo_common::types::Value>>,
    /// Override epoch for time-travel queries (None = use transaction/current epoch).
    viewing_epoch_override: parking_lot::Mutex<Option<EpochId>>,
    /// Savepoints within the current transaction.
    savepoints: parking_lot::Mutex<Vec<SavepointState>>,
    /// Nesting depth for nested transactions (0 = outermost).
    /// Nested `START TRANSACTION` creates an auto-savepoint; nested `COMMIT`
    /// releases it, nested `ROLLBACK` rolls back to it.
    transaction_nesting_depth: parking_lot::Mutex<u32>,
    /// Named graphs touched during the current transaction (for cross-graph atomicity).
    /// `None` represents the default graph. Populated at `BEGIN` time and on each
    /// `USE GRAPH` / `SESSION SET GRAPH` switch within a transaction.
    touched_graphs: parking_lot::Mutex<Vec<GraphPath>>,
    /// LPG graphs created in the current transaction.
    ///
    /// These stores stay detached from the database graph map until the
    /// durable commit is published. That makes graph existence and contents
    /// transaction-local instead of leaking them to concurrent sessions.
    #[cfg(feature = "lpg")]
    pending_created_graphs:
        parking_lot::Mutex<std::collections::HashMap<GraphPath, PendingCreatedGraph>>,
    /// Committed LPG graph incarnations dropped in the current transaction.
    ///
    /// The captured `Arc` is also the compare-and-swap token used at commit:
    /// a transaction can never drop a concurrently replaced graph.
    #[cfg(feature = "lpg")]
    pending_dropped_graphs: parking_lot::Mutex<std::collections::HashMap<GraphPath, Arc<LpgStore>>>,
    /// Detached incarnations created and then dropped in this transaction.
    ///
    /// The stores remain retained until commit/rollback cleanup. A vector is
    /// required because one transaction may repeatedly CREATE/DROP the same
    /// name before selecting its final incarnation.
    #[cfg(feature = "lpg")]
    cancelled_created_graphs:
        parking_lot::Mutex<std::collections::HashMap<GraphPath, Vec<Arc<LpgStore>>>>,
    /// Exact named-graph incarnations touched by this transaction.
    ///
    /// Keeping the `Arc` prevents a concurrent `DROP GRAPH` from making
    /// rollback/finalization resolve the name to the default store. Commit
    /// validates pointer identity under the publication lock.
    #[cfg(feature = "lpg")]
    touched_named_graphs: parking_lot::Mutex<std::collections::HashMap<GraphPath, Arc<LpgStore>>>,
    /// Previously mutated exact targets displaced by this transaction's lifecycle.
    #[cfg(feature = "lpg")]
    superseded_graph_touches: parking_lot::Mutex<Vec<(GraphPath, Arc<LpgStore>)>>,
    /// Named coordinates first observed absent by a Snapshot Isolation or
    /// Serializable transaction.
    ///
    /// Absence is a lifecycle expectation, not permission to upgrade to a
    /// graph that appears later. Read Committed deliberately does not retain
    /// this set and may refresh at each mutating operation.
    #[cfg(feature = "lpg")]
    missing_named_graphs: parking_lot::Mutex<std::collections::HashSet<GraphPath>>,
    /// Transaction-local graph type binding changes, pinned to the catalog
    /// value observed on first lifecycle touch.
    #[cfg(feature = "lpg")]
    pending_graph_type_bindings:
        parking_lot::Mutex<std::collections::HashMap<GraphPath, PendingGraphTypeBinding>>,
    /// Index DDL staged by the current transaction.
    ///
    /// Definitions and pre-commit validation are session-private. Physical
    /// index registries and catalog names are published only after the durable
    /// commit marker succeeds.
    #[cfg(feature = "lpg")]
    pending_index_ddl: parking_lot::Mutex<Vec<PendingIndexDdl>>,
    /// Transaction-local compare-and-swap overlay for virtual projection DDL.
    ///
    /// Definitions remain process-local, but their visibility follows the
    /// surrounding transaction: rollback, savepoints, conflicts, and durable
    /// commit failure cannot leak a registry post-image.
    #[cfg(feature = "lpg")]
    pending_projection_ddl:
        parking_lot::Mutex<std::collections::HashMap<String, PendingProjectionChange>>,
    /// Committed virtual-projection registry captured at transaction start for
    /// Snapshot Isolation and Serializable transactions. Read Committed keeps
    /// this empty and resolves the live registry per statement.
    #[cfg(feature = "lpg")]
    projection_registry_snapshot: parking_lot::Mutex<
        Option<std::collections::HashMap<String, Arc<RegisteredGraphProjection>>>,
    >,
    /// Serializable `SHOW PROJECTIONS` reads the whole registry predicate.
    /// Commit validates that predicate conservatively against concurrent DDL.
    #[cfg(feature = "lpg")]
    projection_registry_read: std::sync::atomic::AtomicBool,
    /// Temporary compatibility records for one atomically framed standalone
    /// catalog statement. Index DDL uses the normal tagged transaction path.
    #[cfg(all(feature = "wal", feature = "lpg"))]
    catalog_wal_batch: parking_lot::Mutex<Option<Vec<grafeo_storage::wal::WalRecord>>>,
    /// Engine-only capability and exact target predicate for one RDF→LPG
    /// projection rebuild transaction. Public sessions never receive it.
    #[cfg(all(feature = "lpg", feature = "triple-store"))]
    rdf_projection_target: parking_lot::Mutex<Option<RdfProjectionTarget>>,
    /// Count of active `ResultStream`s pinned to this session. Commit and
    /// rollback block while any streams are outstanding so mid-iteration
    /// snapshots are not invalidated.
    #[cfg(feature = "lpg")]
    active_streams: AtomicUsize,
    /// Conflict-detection granularity used for the NEXT Serializable
    /// transaction begun on this session.  Changing this field mid-transaction
    /// has no effect until the next `BEGIN`.
    conflict_granularity: parking_lot::Mutex<crate::transaction::ConflictGranularity>,
    /// Shared metrics registry (populated when the `metrics` feature is enabled).
    #[cfg(feature = "metrics")]
    pub(crate) metrics: Option<Arc<crate::metrics::MetricsRegistry>>,
    /// Transaction start time for duration tracking.
    #[cfg(feature = "metrics")]
    tx_start_time: parking_lot::Mutex<Option<Instant>>,
    /// Named graph projections shared with the database.
    #[cfg(feature = "lpg")]
    projections: VirtualProjectionRegistry,
}

/// Role of the session's internal `LpgStore`.
#[cfg(feature = "lpg")]
#[derive(Clone, Copy)]
enum LpgBackend {
    /// The internal `LpgStore` is the session's active backing store (possibly
    /// wrapped by WAL/CDC/Layered decorators on the read/write path). Search
    /// procedures can reach its HNSW/BM25 indexes.
    Active,
    /// External storage owns administrative operations. The internal store
    /// is its exact mutation target when available, or an empty placeholder
    /// otherwise. Search procedures must not use either as an admin escape.
    Placeholder { commit_target_available: bool },
}

/// Per-graph savepoint snapshot, capturing the store state at the time of the savepoint.
#[cfg(feature = "lpg")]
#[derive(Clone)]
struct GraphSavepoint {
    /// Whether this exact store participated in commit/finalization at the
    /// savepoint, rather than being captured only as retained lifecycle state.
    was_touched: bool,
    /// Exact concrete graph incarnation captured at the savepoint.
    store: Arc<LpgStore>,
    /// Exact mutation layer that owns overlays/base tombstones for this graph.
    mutation_store: Arc<dyn GraphStoreMut>,
    next_node_id: u64,
    next_edge_id: u64,
    undo_log_position: usize,
    /// Snapshot of the per-transaction property delta at savepoint creation.
    ///
    /// `rollback_to_savepoint` replaces the live delta with this clone so that
    /// buffered property writes made *after* the savepoint are discarded.
    /// Labels and entity deletes still flow through the undo log; this only
    /// covers the buffered property overlay introduced in the unified-MVCC
    /// first increment.
    overlay_snapshot: TxDelta,
    /// Deferred creates/deletes and layered base tombstones at the savepoint.
    structural_snapshot: TxStructuralSnapshot,
}

/// Exact transaction-local LPG graph lifecycle state at a savepoint.
#[cfg(feature = "lpg")]
#[derive(Clone)]
struct LpgLifecycleSavepoint {
    pending_created_graphs: std::collections::HashMap<GraphPath, PendingCreatedGraph>,
    pending_dropped_graphs: std::collections::HashMap<GraphPath, Arc<LpgStore>>,
    cancelled_created_graphs: std::collections::HashMap<GraphPath, Vec<Arc<LpgStore>>>,
    touched_named_graphs: std::collections::HashMap<GraphPath, Arc<LpgStore>>,
    superseded_graph_touches: Vec<(GraphPath, Arc<LpgStore>)>,
    missing_named_graphs: std::collections::HashSet<GraphPath>,
    pending_graph_type_bindings: std::collections::HashMap<GraphPath, PendingGraphTypeBinding>,
}

/// Exact schema incarnation that owns a schema-qualified object.
///
/// The implicit default graph is created and dropped atomically with the
/// namespace, so its Arc is the namespace's compare-and-swap token and closes
/// DROP+recreate ABA races that a name-only check would admit.
#[cfg(feature = "lpg")]
#[derive(Clone)]
struct SchemaIncarnation {
    name: String,
    default_graph: Arc<LpgStore>,
}

/// Namespace expectation captured for a detached graph create.
///
/// Parser-free graph names may contain `/` (for example URI-shaped names), so
/// an unknown prefix remains legal but pins the fact that no schema with that
/// prefix exists. CREATE SCHEMA performs the converse collision check.
#[cfg(feature = "lpg")]
#[derive(Clone)]
enum PendingGraphNamespace {
    Root,
    Schema(SchemaIncarnation),
    UnregisteredPrefix(String),
}

/// Detached graph plus the namespace expectation it must still satisfy at
/// publication.
#[cfg(feature = "lpg")]
#[derive(Clone)]
struct PendingCreatedGraph {
    store: Arc<LpgStore>,
    parent: Arc<LpgStore>,
    namespace: PendingGraphNamespace,
    /// Exact source/index-registry cut consumed by `AS COPY OF`, if any.
    ///
    /// The target lifecycle owns this expectation so savepoint restore and a
    /// later COPY→DROP automatically retain or discard it with the detached
    /// target. Transaction-local source DDL is represented separately in the
    /// copied post-image; this baseline contains only physically published
    /// property-index keys.
    copy_source_indexes: Option<CopySourceIndexExpectation>,
}

/// Parent/store CAS tokens and dependency order are fixed before durability.
#[cfg(feature = "lpg")]
enum PreparedGraphLifecycleChange {
    Retire {
        parent: Arc<LpgStore>,
        name: String,
        expected: Arc<LpgStore>,
    },
    Install {
        parent: Arc<LpgStore>,
        name: String,
        store: Arc<LpgStore>,
    },
}

/// Published source metadata that one detached COPY target must still match.
#[cfg(feature = "lpg")]
#[derive(Clone)]
struct CopySourceIndexExpectation {
    /// `None` is the root default partition; `Some` is one exact named
    /// storage coordinate (including a schema default partition).
    source_name: Option<String>,
    source: Arc<LpgStore>,
    physical_keys: Vec<String>,
}

/// One publication-cut source bundle for `CREATE GRAPH ... LIKE/COPY OF`.
///
/// The concrete incarnation is the lifecycle/index compare-and-swap token.
/// The read view may additionally contain a compact cold tier and is therefore
/// the only sound source for graph-wide materialization.
#[cfg(all(feature = "lpg", feature = "gql"))]
struct GraphCreationSource {
    storage_key: Option<String>,
    incarnation: Arc<LpgStore>,
    read_view: Arc<dyn GraphStoreSearch>,
    graph_type_binding: Option<String>,
    copy: Option<CopySourceSnapshot>,
}

/// Transaction-visible source snapshot captured before the target exists.
#[cfg(all(feature = "lpg", feature = "gql"))]
struct CopySourceSnapshot {
    epoch: EpochId,
    transaction_id: TransactionId,
    graph: MaterializedLpgGraph,
    /// Physically published property-index registry used for commit-time CAS.
    physical_property_indexes: Vec<String>,
}

/// Complete transaction-visible LPG graph image, retaining source IDs until
/// the detached destination assigns its own IDs.
#[cfg(all(feature = "lpg", feature = "gql"))]
struct MaterializedLpgGraph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    property_indexes: Vec<String>,
}

/// Exact entity snapshots produced by one `CREATE GRAPH … AS COPY OF`.
///
/// The receipt stays transaction-local until every fallible copy/WAL/catalog
/// step succeeds. CDC then stages one complete Create snapshot per entity;
/// copied properties never appear as separate Update events.
#[cfg(all(feature = "lpg", feature = "gql"))]
struct CopiedLpgEntities {
    #[cfg(any(feature = "wal", feature = "cdc"))]
    nodes: Vec<Node>,
    #[cfg(any(feature = "wal", feature = "cdc"))]
    edges: Vec<Edge>,
    /// Transaction-visible property-index post-image inherited from the
    /// source. Text/vector index families require format-specific receipts and
    /// are deliberately outside this v1 copy substrate.
    property_indexes: Vec<String>,
}

/// One compare-and-swap graph type binding staged with graph lifecycle.
#[cfg(feature = "lpg")]
#[derive(Clone)]
struct PendingGraphTypeBinding {
    expected: Option<String>,
    replacement: Option<String>,
}

/// First-observed registry identity and final transaction-local post-image for
/// one virtual projection name.
#[cfg(feature = "lpg")]
#[derive(Clone)]
struct PendingProjectionChange {
    expected: Option<Arc<RegisteredGraphProjection>>,
    replacement: Option<Arc<RegisteredGraphProjection>>,
}

/// Physical index kind plus the configuration needed for exact WAL replay.
#[cfg(feature = "lpg")]
#[derive(Clone, Debug, PartialEq, Eq)]
enum PendingIndexKind {
    Property,
    BTree,
    Text {
        min_token_length: Option<usize>,
    },
    Vector {
        dimensions: Option<usize>,
        metric: Option<String>,
        m: Option<usize>,
        ef_construction: Option<usize>,
        ef: Option<usize>,
        quantization: Option<String>,
    },
}

#[cfg(feature = "lpg")]
impl PendingIndexKind {
    fn physical_kind(&self) -> PhysicalIndexFamily {
        match self {
            Self::Property | Self::BTree => PhysicalIndexFamily::Property,
            Self::Text { .. } => PhysicalIndexFamily::Text,
            Self::Vector { .. } => PhysicalIndexFamily::Vector,
        }
    }
}

/// A transaction-local index definition or removal.
#[cfg(feature = "lpg")]
#[derive(Clone)]
struct PendingIndexDdl {
    create: bool,
    graph: GraphPath,
    /// Exact schema namespace that owned the target when this DDL was staged.
    /// Revalidated at commit so DROP+recreate cannot publish an orphan index.
    owner_schema: Option<SchemaIncarnation>,
    name: Option<String>,
    /// Exact live logical identity, or None for a same-transaction/anonymous owner.
    expected_owner: Option<grafeo_common::types::IndexId>,
    /// Rebuild is a physical replacement, not a new logical allocation.
    rebuild: bool,
    /// Resolved by physical preparation, before catalog allocation/publication.
    configuration: Option<IndexConfiguration>,
    /// Private transaction result; observed by the direct caller only after
    /// successful commit. A prepared but aborted ID never escapes.
    owner_result: Arc<std::sync::OnceLock<IndexId>>,
    label: String,
    property: String,
    kind: PendingIndexKind,
    /// Exact physical registry that will receive the index at publication.
    target: Arc<LpgStore>,
}

/// Exact desired target generation for one RDF→LPG rebuild.
///
/// This value is installed before the rebuild transaction starts, threaded
/// into statement-time protected-row validation, and checked against the full
/// transaction-visible default graph while the publication write lock is held.
#[cfg(all(feature = "lpg", feature = "triple-store"))]
#[derive(Clone)]
struct RdfProjectionTarget {
    projection_id: u64,
    owner_marker: String,
    node_label: String,
    desired_iris: Arc<std::collections::BTreeSet<String>>,
    receipt: Option<RdfProjectionReceiptTarget>,
}

/// Immutable receipt coordinates captured from one coherent RDF source cut.
/// The target LPG epoch is deliberately absent until commit preparation
/// reserves it under the publication lock.
#[cfg(all(feature = "lpg", feature = "triple-store"))]
#[derive(Clone)]
struct RdfProjectionReceiptTarget {
    registry: Arc<grafeo_core::graph::rdf::RdfLpgProjectionRegistry>,
    store_id: grafeo_common::types::StoreId,
    mapping_digest: grafeo_common::types::Digest256,
    source_graph: grafeo_common::types::ProjectionSourceGraph,
    source_epoch: EpochId,
    generation: u64,
    row_count: u64,
}

#[cfg(feature = "lpg")]
enum PreparedLogicalCatalog<'catalog, 'workspace> {
    NoCatalogChange,
    /// Anonymous index actions pin current owner decisions without copying
    /// the catalog or introducing a checked logical-admission error.
    OwnersPinned(CatalogReadGuard<'catalog>),
    Changed(ReadyCatalog<'catalog, 'workspace>),
}

/// Savepoint state: name and per-graph transaction snapshots.
#[derive(Clone)]
struct SavepointState {
    name: String,
    #[cfg(feature = "lpg")]
    catalog_snapshot: Option<TransactionCatalog>,
    #[cfg(feature = "lpg")]
    graph_snapshots: Vec<GraphSavepoint>,
    /// Exact set of graph coordinates that participated in the transaction at
    /// savepoint creation.
    ///
    /// `graph_snapshots` is deliberately a superset: it also captures detached
    /// lifecycle incarnations that might first receive writes after the
    /// savepoint. Restoring `touched_graphs` from that superset would make an
    /// already-cancelled incarnation look live again.
    #[cfg(feature = "lpg")]
    touched_graphs: Vec<GraphPath>,
    /// CDC event buffer position at savepoint creation.
    /// On rollback-to-savepoint, the buffer is truncated to this position.
    #[cfg(feature = "cdc")]
    cdc_event_position: usize,
    /// Detached/pinned graph incarnations must roll back with graph data.
    #[cfg(feature = "lpg")]
    lpg_lifecycle: LpgLifecycleSavepoint,
    /// Transaction-local index DDL visible at the savepoint.
    #[cfg(feature = "lpg")]
    index_ddl_snapshot: Vec<PendingIndexDdl>,
    /// Exact virtual-projection DDL overlay visible at the savepoint.
    #[cfg(feature = "lpg")]
    projection_ddl_snapshot: std::collections::HashMap<String, PendingProjectionChange>,
    /// Pending RDF operations on the default and exact named partitions.
    #[cfg(feature = "triple-store")]
    rdf_snapshot: RdfTransactionSavepoint,
}

/// Deterministic rendezvous around cancellation's irreversible boundaries.
#[cfg(feature = "testing-statement-injection")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryCancellationTestPhase {
    /// The statement rollback boundary is ready; no body mutation has run.
    BeforeFirstMutation,
    /// The statement-entry savepoint exists and remains rollback-capable.
    AfterSavepointPrepared,
    /// Mutations are staged but the statement can still roll back.
    BeforeStatementCompletion,
    /// Cancellation can still defeat the durable commit fence.
    BeforeCommitFence,
    /// The durable commit marker has been acknowledged.
    AfterDurableMarker,
}

#[cfg(feature = "testing-statement-injection")]
struct QueryCancellationTestHook {
    phase: QueryCancellationTestPhase,
    reached: Arc<std::sync::Barrier>,
    release: Arc<std::sync::Barrier>,
}

#[cfg(any(feature = "lpg", feature = "triple-store"))]
struct ExecutionStatementGuard<'a> {
    depth: Option<&'a std::sync::atomic::AtomicUsize>,
    outermost: bool,
    mutating: bool,
}
#[cfg(any(feature = "lpg", feature = "triple-store"))]
impl Drop for ExecutionStatementGuard<'_> {
    fn drop(&mut self) {
        if let Some(depth) = self.depth {
            depth.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Publication and transaction semantics for schema-language statements.
///
/// Catalog statements use the private cut inside a transaction and atomic
/// standalone publication otherwise. Index definitions share commit staging.
#[cfg(all(feature = "lpg", feature = "gql"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SchemaCommandMode {
    TransactionalIndex,
    ReadOnlyShow,
    StandaloneCatalog,
}

/// Effects and immutable procedure bodies selected before a statement enters
/// authorization and transaction framing.
#[cfg(any(
    feature = "gql",
    feature = "cypher",
    feature = "gremlin",
    feature = "graphql",
    feature = "sql-pgq"
))]
#[derive(Clone)]
struct QualifiedLpgPlan {
    mutates: bool,
    contains_call: bool,
    #[cfg(all(any(feature = "lpg", feature = "algos"), feature = "gql"))]
    contains_catalog_call: bool,
    #[cfg(all(any(feature = "lpg", feature = "algos"), feature = "gql"))]
    procedures: Arc<crate::query::procedure_effect::ResolvedProcedureCatalog>,
}

/// Detached mutation targets for one standalone catalog statement.
///
/// SHOW and transactional index statements always use the live wrapper. Only
/// `StandaloneCatalog` receives a detached target, so the statement can finish
/// every fallible mutation before WAL publication touches live state.
#[cfg(all(feature = "lpg", feature = "gql"))]
#[derive(Clone, Copy)]
struct CatalogDdlTarget<'a> {
    catalog: &'a Catalog,
    store: &'a LpgStore,
    current_schema: &'a parking_lot::Mutex<Option<String>>,
    /// Live process-local virtual projections. Standalone catalog DDL already
    /// holds the publication write lock while consulting this registry.
    projections: &'a VirtualProjectionRegistry,
}

#[cfg(all(feature = "lpg", feature = "gql"))]
impl CatalogDdlTarget<'_> {
    fn effective_type_key(self, type_name: &str) -> String {
        match self.current_schema.lock().as_deref() {
            Some(schema) => format!("{schema}/{type_name}"),
            None => type_name.to_string(),
        }
    }

    fn key_belongs_to_schema(key: &str, schema: &str) -> bool {
        key.split_once('/')
            .is_some_and(|(prefix, _)| prefix.eq_ignore_ascii_case(schema))
    }

    fn graph_type_bindings(
        self,
        session: &Session,
    ) -> std::collections::HashMap<GraphPath, String> {
        let mut bindings: std::collections::HashMap<_, _> =
            self.catalog.all_graph_type_bindings().into_iter().collect();
        for (path, pending) in session.pending_graph_type_bindings.lock().iter() {
            match &pending.replacement {
                Some(graph_type) => {
                    bindings.insert(path.clone(), graph_type.clone());
                }
                None => {
                    bindings.remove(path);
                }
            }
        }
        bindings
    }

    fn has_schema_owned_catalog_objects(self, schema: &str, session: &Session) -> bool {
        let bindings = self.graph_type_bindings(session);
        let catalog = self.catalog.read();
        // Dependency checks observe the same final owner/binding/projection
        // view as commit, without mutating the catalog's owner preimage.
        let mut indexes: std::collections::HashSet<_> = catalog
            .all_indexes()
            .into_iter()
            .map(|index| index.key)
            .collect();
        for pending in session.pending_index_ddl.lock().iter() {
            let key = Session::physical_index_key(
                &pending.graph,
                &pending.label,
                &pending.property,
                &pending.kind,
            );
            if pending.create {
                indexes.insert(key);
            } else {
                indexes.remove(&key);
            }
        }
        let registry = self.projections.read();
        let snapshot = session.projection_registry_snapshot.lock();
        let projections = snapshot.as_ref().unwrap_or(&registry);
        let pending = session.pending_projection_ddl.lock();
        let has_projection = projections.iter().any(|(name, projection)| {
            !pending.contains_key(name) && projection.belongs_to_schema(schema)
        }) || pending.values().any(|change| {
            change
                .replacement
                .as_ref()
                .is_some_and(|projection| projection.belongs_to_schema(schema))
        });
        catalog
            .all_node_type_names()
            .iter()
            .any(|name| Self::key_belongs_to_schema(name, schema))
            || catalog
                .all_edge_type_names()
                .iter()
                .any(|name| Self::key_belongs_to_schema(name, schema))
            || catalog
                .all_graph_type_names()
                .iter()
                .any(|name| Self::key_belongs_to_schema(name, schema))
            || catalog.all_named_constraints().iter().any(|constraint| {
                Self::key_belongs_to_schema(&constraint.name, schema)
                    || Self::key_belongs_to_schema(&constraint.label, schema)
            })
            || indexes.iter().any(|index| {
                index
                    .graph()
                    .components()
                    .first()
                    .is_some_and(|graph| Self::key_belongs_to_schema(graph, schema))
            })
            || bindings.iter().any(|(graph, graph_type)| {
                graph
                    .components()
                    .first()
                    .is_some_and(|name| Self::key_belongs_to_schema(name, schema))
                    || Self::key_belongs_to_schema(graph_type, schema)
            })
            || has_projection
    }
}

/// A completely prepared catalog post-image.
///
/// Construction performs every statement operation that can return an error,
/// including validation and graph creation. After the WAL append succeeds,
/// publication only consumes these prebuilt values through installers that do
/// not return recoverable errors; a fatal process interruption is resolved by
/// replaying the already-durable post-image on reopen.
#[cfg(all(feature = "lpg", feature = "gql"))]
struct PreparedCatalogDdl<'catalog, 'workspace> {
    result: QueryResult,
    catalog: ReadyCatalog<'catalog, 'workspace>,
    graph_entries: FxHashMap<String, Arc<LpgStore>>,
    current_context: SessionGraphContext,
    epoch_targets: Vec<Arc<LpgStore>>,
    #[cfg(feature = "wal")]
    catalog_state: Vec<u8>,
    #[cfg(feature = "wal")]
    created_graphs: Vec<GraphPath>,
    #[cfg(feature = "wal")]
    dropped_graphs: Vec<GraphPath>,
}

/// Unwind-safe ownership of the compatibility WAL buffer used while a
/// standalone statement is prepared. Individual legacy records are discarded
/// after validation because the complete `CatalogBatchV3` post-image is the
/// only recovery unit.
#[cfg(all(feature = "wal", feature = "lpg", feature = "gql"))]
struct CatalogWalBatchGuard<'a> {
    slot: &'a parking_lot::Mutex<Option<Vec<grafeo_storage::wal::WalRecord>>>,
    active: bool,
}

#[cfg(all(feature = "wal", feature = "lpg", feature = "gql"))]
impl<'a> CatalogWalBatchGuard<'a> {
    fn begin(
        slot: &'a parking_lot::Mutex<Option<Vec<grafeo_storage::wal::WalRecord>>>,
    ) -> Result<Self> {
        let mut batch = slot.lock();
        if batch.is_some() {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "nested catalog WAL batch".to_string(),
                ),
            ));
        }
        *batch = Some(Vec::new());
        drop(batch);
        Ok(Self { slot, active: true })
    }

    fn finish(mut self) -> Vec<grafeo_storage::wal::WalRecord> {
        self.active = false;
        self.slot.lock().take().unwrap_or_default()
    }
}

#[cfg(all(feature = "wal", feature = "lpg", feature = "gql"))]
impl Drop for CatalogWalBatchGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            self.slot.lock().take();
        }
    }
}

/// An admitted writer whose compound CDC construction boundary is retained.
#[cfg(feature = "lpg")]
enum ResolvedLpgWriter {
    Plain(Arc<dyn GraphStoreMut>),
    #[cfg(feature = "cdc")]
    Cdc(Arc<crate::database::cdc_store::CdcGraphStore>),
}

#[cfg(feature = "lpg")]
impl ResolvedLpgWriter {
    fn into_store(self) -> Arc<dyn GraphStoreMut> {
        match self {
            Self::Plain(store) => store,
            #[cfg(feature = "cdc")]
            Self::Cdc(store) => store,
        }
    }
}

impl Session {
    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn catalog_view(&self) -> Arc<Catalog> {
        #[cfg(feature = "lpg")]
        if let Some(catalog) = self.transaction_catalog.lock().as_ref() {
            return Arc::clone(&catalog.current);
        }
        Arc::clone(&self.catalog)
    }

    /// Creates a new session with adaptive execution configuration.
    #[cfg(feature = "lpg")]
    #[allow(dead_code)] // Used when lpg enabled without triple-store
    pub(crate) fn with_adaptive(store: Arc<LpgStore>, cfg: SessionConfig) -> Self {
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreSearch>;
        let graph_store_mut = Some(Arc::clone(&store) as Arc<dyn GraphStoreMut>);
        Self {
            store,
            lpg_backend: LpgBackend::Active,
            graph_store,
            graph_store_mut,
            catalog: cfg.catalog,
            #[cfg(feature = "triple-store")]
            rdf_store: Arc::new(RdfStore::new()),
            transaction_manager: cfg.transaction_manager,
            query_cache: cfg.query_cache,
            physical_cache: cfg.physical_cache,
            current_transaction: parking_lot::Mutex::new(None),
            #[cfg(feature = "lpg")]
            transaction_catalog: parking_lot::Mutex::new(None),
            #[cfg(any(feature = "lpg", feature = "triple-store"))]
            mutation_operation_gate: parking_lot::ReentrantMutex::new(()),
            historical_view_operation_gate: parking_lot::ReentrantMutex::new(()),
            read_only_tx: parking_lot::Mutex::new(cfg.read_only),
            db_read_only: cfg.read_only,
            identity: cfg.identity,
            auto_commit: true,
            adaptive_config: cfg.adaptive_config,
            #[cfg(any(
                feature = "gql",
                feature = "cypher",
                feature = "gremlin",
                feature = "graphql",
                feature = "sql-pgq"
            ))]
            factorized_execution: cfg.factorized_execution,
            graph_model: cfg.graph_model,
            query_timeout: cfg.query_timeout,
            result_limits: cfg.result_limits,
            active_result_limits: parking_lot::Mutex::new(None),
            active_result_admission: parking_lot::Mutex::new(None),
            active_execution_control: parking_lot::Mutex::new(None),
            active_execution_completed: std::sync::atomic::AtomicBool::new(false),
            active_execution_statement_depth: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "testing-statement-injection")]
            query_cancellation_test_hook: parking_lot::Mutex::new(None),
            max_property_size: cfg.max_property_size,
            buffer_manager: cfg.buffer_manager,
            #[cfg(feature = "spill")]
            spill_root: cfg.spill_root,
            #[cfg(any(feature = "spill", feature = "cdc"))]
            world_identity: cfg.world_identity,
            commit_counter: cfg.commit_counter,
            durability_poisoned: cfg.durability_poisoned,
            database_open: cfg.database_open,
            active_sessions: cfg.active_sessions,
            gc_interval: cfg.gc_interval,
            transaction_start_node_count: AtomicUsize::new(0),
            transaction_start_edge_count: AtomicUsize::new(0),
            #[cfg(feature = "wal")]
            wal: None,
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            #[cfg(feature = "cdc")]
            cdc_pending_events: None,
            #[cfg(all(feature = "lpg", feature = "cdc"))]
            default_cdc_writer: None,
            current_context: parking_lot::Mutex::new(SessionGraphContext::default()),
            time_zone: parking_lot::Mutex::new(None),
            #[cfg(feature = "triple-store")]
            rdf_valid_time: parking_lot::Mutex::new(None),
            session_params: parking_lot::Mutex::new(std::collections::HashMap::new()),
            viewing_epoch_override: parking_lot::Mutex::new(None),
            savepoints: parking_lot::Mutex::new(Vec::new()),
            transaction_nesting_depth: parking_lot::Mutex::new(0),
            touched_graphs: parking_lot::Mutex::new(Vec::new()),
            #[cfg(feature = "lpg")]
            pending_created_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            pending_dropped_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            cancelled_created_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            touched_named_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            superseded_graph_touches: parking_lot::Mutex::new(Vec::new()),
            #[cfg(feature = "lpg")]
            missing_named_graphs: parking_lot::Mutex::new(std::collections::HashSet::new()),
            #[cfg(feature = "lpg")]
            pending_graph_type_bindings: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            pending_index_ddl: parking_lot::Mutex::new(Vec::new()),
            #[cfg(feature = "lpg")]
            pending_projection_ddl: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            projection_registry_snapshot: parking_lot::Mutex::new(None),
            #[cfg(feature = "lpg")]
            projection_registry_read: std::sync::atomic::AtomicBool::new(false),
            #[cfg(all(feature = "wal", feature = "lpg"))]
            catalog_wal_batch: parking_lot::Mutex::new(None),
            #[cfg(all(feature = "lpg", feature = "triple-store"))]
            rdf_projection_target: parking_lot::Mutex::new(None),
            active_streams: AtomicUsize::new(0),
            conflict_granularity: parking_lot::Mutex::new(
                crate::transaction::ConflictGranularity::Entity,
            ),
            #[cfg(feature = "metrics")]
            metrics: None,
            #[cfg(feature = "metrics")]
            tx_start_time: parking_lot::Mutex::new(None),
            projections: cfg.projections,
        }
    }

    /// Overrides the graph store and write store used by the query engine.
    ///
    /// Used by the layered store integration: the session's `store` field is
    /// the overlay `LpgStore` (for MVCC), but reads and writes should route
    /// through the `LayeredStore` (which merges base + overlay).
    /// Construction-only: call before attaching WAL and then CDC once/last.
    #[cfg(all(feature = "compact-store", feature = "lpg"))]
    pub(crate) fn override_stores(
        &mut self,
        read_store: Arc<dyn GraphStoreSearch>,
        write_store: Option<Arc<dyn GraphStoreMut>>,
    ) {
        self.graph_store = read_store;
        self.graph_store_mut = write_store;
    }

    /// RDF-only session: SPARQL/WAL without an LPG store or GQL.
    #[cfg(all(feature = "triple-store", not(feature = "lpg")))]
    pub(crate) fn with_rdf_only(rdf_store: Arc<RdfStore>, cfg: SessionConfig) -> Self {
        #[cfg(any(
            feature = "gql",
            feature = "cypher",
            feature = "gremlin",
            feature = "graphql",
            feature = "sql-pgq"
        ))]
        let graph_store = Arc::new(NullGraphStore) as Arc<dyn GraphStoreSearch>;
        Self {
            #[cfg(any(
                feature = "gql",
                feature = "cypher",
                feature = "gremlin",
                feature = "graphql",
                feature = "sql-pgq"
            ))]
            graph_store,
            #[cfg(any(
                feature = "gql",
                feature = "cypher",
                feature = "gremlin",
                feature = "graphql",
                feature = "sql-pgq"
            ))]
            graph_store_mut: None,
            #[cfg(any(
                feature = "gql",
                feature = "cypher",
                feature = "gremlin",
                feature = "graphql",
                feature = "sql-pgq"
            ))]
            catalog: cfg.catalog,
            rdf_store,
            transaction_manager: cfg.transaction_manager,
            query_cache: cfg.query_cache,
            physical_cache: cfg.physical_cache,
            current_transaction: parking_lot::Mutex::new(None),
            #[cfg(feature = "lpg")]
            transaction_catalog: parking_lot::Mutex::new(None),
            #[cfg(any(feature = "lpg", feature = "triple-store"))]
            mutation_operation_gate: parking_lot::ReentrantMutex::new(()),
            historical_view_operation_gate: parking_lot::ReentrantMutex::new(()),
            read_only_tx: parking_lot::Mutex::new(cfg.read_only),
            db_read_only: cfg.read_only,
            identity: cfg.identity,
            auto_commit: true,
            adaptive_config: cfg.adaptive_config,
            #[cfg(any(
                feature = "gql",
                feature = "cypher",
                feature = "gremlin",
                feature = "graphql",
                feature = "sql-pgq"
            ))]
            factorized_execution: cfg.factorized_execution,
            graph_model: cfg.graph_model,
            query_timeout: cfg.query_timeout,
            result_limits: cfg.result_limits,
            active_result_limits: parking_lot::Mutex::new(None),
            active_result_admission: parking_lot::Mutex::new(None),
            active_execution_control: parking_lot::Mutex::new(None),
            active_execution_completed: std::sync::atomic::AtomicBool::new(false),
            active_execution_statement_depth: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "testing-statement-injection")]
            query_cancellation_test_hook: parking_lot::Mutex::new(None),
            #[cfg(any(
                feature = "gql",
                feature = "cypher",
                feature = "gremlin",
                feature = "graphql",
                feature = "sql-pgq"
            ))]
            max_property_size: cfg.max_property_size,
            buffer_manager: cfg.buffer_manager,
            #[cfg(feature = "spill")]
            spill_root: cfg.spill_root,
            #[cfg(any(feature = "spill", feature = "cdc"))]
            world_identity: cfg.world_identity,
            #[cfg(feature = "lpg")]
            commit_counter: cfg.commit_counter,
            durability_poisoned: cfg.durability_poisoned,
            database_open: cfg.database_open,
            active_sessions: cfg.active_sessions,
            #[cfg(feature = "lpg")]
            gc_interval: cfg.gc_interval,
            #[cfg(feature = "lpg")]
            transaction_start_node_count: AtomicUsize::new(0),
            #[cfg(feature = "lpg")]
            transaction_start_edge_count: AtomicUsize::new(0),
            #[cfg(feature = "wal")]
            wal: None,
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            #[cfg(feature = "cdc")]
            cdc_pending_events: None,
            current_context: parking_lot::Mutex::new(SessionGraphContext::default()),
            time_zone: parking_lot::Mutex::new(None),
            rdf_valid_time: parking_lot::Mutex::new(None),
            session_params: parking_lot::Mutex::new(std::collections::HashMap::new()),
            viewing_epoch_override: parking_lot::Mutex::new(None),
            savepoints: parking_lot::Mutex::new(Vec::new()),
            transaction_nesting_depth: parking_lot::Mutex::new(0),
            touched_graphs: parking_lot::Mutex::new(Vec::new()),
            #[cfg(feature = "lpg")]
            pending_created_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            pending_dropped_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            cancelled_created_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            touched_named_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            superseded_graph_touches: parking_lot::Mutex::new(Vec::new()),
            #[cfg(feature = "lpg")]
            missing_named_graphs: parking_lot::Mutex::new(std::collections::HashSet::new()),
            #[cfg(feature = "lpg")]
            pending_graph_type_bindings: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            pending_index_ddl: parking_lot::Mutex::new(Vec::new()),
            #[cfg(feature = "lpg")]
            pending_projection_ddl: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            projection_registry_snapshot: parking_lot::Mutex::new(None),
            #[cfg(feature = "lpg")]
            projection_registry_read: std::sync::atomic::AtomicBool::new(false),
            #[cfg(all(feature = "wal", feature = "lpg"))]
            catalog_wal_batch: parking_lot::Mutex::new(None),
            #[cfg(all(feature = "lpg", feature = "triple-store"))]
            rdf_projection_target: parking_lot::Mutex::new(None),
            #[cfg(feature = "lpg")]
            active_streams: AtomicUsize::new(0),
            conflict_granularity: parking_lot::Mutex::new(
                crate::transaction::ConflictGranularity::Entity,
            ),
            #[cfg(feature = "metrics")]
            metrics: None,
            #[cfg(feature = "metrics")]
            tx_start_time: parking_lot::Mutex::new(None),
        }
    }

    /// Attaches WAL for RDF-only sessions (no LPG overlay wrapper).
    #[cfg(all(feature = "wal", not(feature = "lpg")))]
    pub(crate) fn set_wal_handle(&mut self, wal: Arc<grafeo_storage::wal::LpgWal>) {
        self.wal = Some(wal);
    }

    /// Sets the WAL for this session (shared with the database).
    ///
    /// This also wraps `graph_store` in a [`WalGraphStore`] so that mutation
    /// operators (INSERT, DELETE, SET via queries) log to the WAL.
    /// Construction-only: attach after selecting stores and before CDC.
    #[cfg(all(feature = "wal", feature = "lpg"))]
    pub(crate) fn set_wal(&mut self, wal: Arc<grafeo_storage::wal::LpgWal>) {
        // Wrap the write store selected by session construction so query-engine
        // mutations are WAL-logged. After compact() this is the LayeredStore;
        // wrapping `self.store` instead would collapse reads and writes back to
        // the empty LPG overlay and make the compact base disappear.
        let inner = self.graph_store_mut.as_ref().map_or_else(
            || Arc::clone(&self.store) as Arc<dyn GraphStoreMut>,
            Arc::clone,
        );
        let wal_store = Arc::new(
            crate::database::wal_store::WalGraphStore::new(
                inner,
                Arc::clone(&wal),
                GraphPath::root(),
            )
            .with_poison(Arc::clone(&self.durability_poisoned)),
        );
        self.graph_store = Arc::clone(&wal_store) as Arc<dyn GraphStoreSearch>;
        self.graph_store_mut = Some(wal_store as Arc<dyn GraphStoreMut>);
        self.wal = Some(wal);
    }

    /// Logs a WAL record if WAL is enabled. No-op for in-memory sessions.
    ///
    /// WAL write failures poison the session and return `Err`. Callers must
    /// not apply further mutations after that.
    #[cfg(feature = "wal")]
    pub(crate) fn log_wal_record(&self, record: &grafeo_storage::wal::WalRecord) -> Result<()> {
        if let Some(ref wal) = self.wal
            && let Err(e) = wal.log(record)
        {
            self.poison_durability();
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::DurabilityFailure(format!(
                    "WAL log failed: {e}"
                )),
            ));
        }
        Ok(())
    }

    #[cfg(all(feature = "wal", feature = "lpg"))]
    fn log_lpg_in_graph(
        &self,
        transaction_id: Option<TransactionId>,
        graph: &GraphPath,
        op: grafeo_storage::wal::LpgMutationOp,
    ) -> Result<()> {
        let tid = transaction_id.unwrap_or(TransactionId::SYSTEM);
        self.log_wal_record(&grafeo_storage::wal::WalRecord::lpg(tid, graph.clone(), op))
    }

    #[cfg(feature = "lpg")]
    fn rollback_pending_graph_lifecycle(&self, transaction_id: TransactionId) {
        // Some incarnations can disappear from the name-based transaction
        // view after CREATE/DROP replacement. Retained lifecycle Arcs are the
        // authoritative cleanup set, including zero-event writes.
        let mut stores = Vec::new();
        for pending in self.pending_created_graphs.lock().values() {
            Self::remember_lpg_incarnation(&mut stores, Arc::clone(&pending.store));
        }
        for store in self.pending_dropped_graphs.lock().values() {
            Self::remember_lpg_incarnation(&mut stores, Arc::clone(store));
        }
        for store in self.cancelled_created_graphs.lock().values().flatten() {
            Self::remember_lpg_incarnation(&mut stores, Arc::clone(store));
        }
        for store in stores {
            self.discard_lpg_graph_transaction(&store, transaction_id);
        }

        self.pending_created_graphs.lock().clear();
        self.pending_dropped_graphs.lock().clear();
        self.cancelled_created_graphs.lock().clear();
        self.missing_named_graphs.lock().clear();
        self.pending_graph_type_bindings.lock().clear();
    }

    #[cfg(feature = "lpg")]
    fn remember_lpg_incarnation(stores: &mut Vec<Arc<LpgStore>>, candidate: Arc<LpgStore>) {
        if !stores.iter().any(|known| Arc::ptr_eq(known, &candidate)) {
            stores.push(candidate);
        }
    }

    /// Discards transaction state owned by incarnations absent from the
    /// durable lifecycle post-image. The caller has a durable commit outcome
    /// and holds the publication write barrier.
    #[cfg(feature = "lpg")]
    fn discard_non_surviving_lpg_incarnations(&self, transaction_id: TransactionId) {
        let mut stores = Vec::new();
        for (_, store) in self.superseded_graph_touches.lock().iter() {
            Self::remember_lpg_incarnation(&mut stores, Arc::clone(store));
        }
        for store in self.pending_dropped_graphs.lock().values() {
            Self::remember_lpg_incarnation(&mut stores, Arc::clone(store));
        }
        for store in self.cancelled_created_graphs.lock().values().flatten() {
            Self::remember_lpg_incarnation(&mut stores, Arc::clone(store));
        }
        for store in stores {
            self.discard_lpg_graph_transaction(&store, transaction_id);
        }
    }

    /// Tests whether one exact store is selected by this transaction's final
    /// (not yet installed) graph-lifecycle post-image.
    #[cfg(feature = "lpg")]
    fn lpg_incarnation_survives_pending_lifecycle(
        &self,
        path: &GraphPath,
        candidate: &Arc<LpgStore>,
    ) -> bool {
        Self::resolve_lpg_lifecycle_path(
            &self.store,
            path,
            &self.pending_created_graphs.lock(),
            &self.pending_dropped_graphs.lock(),
            &self.cancelled_created_graphs.lock(),
        )
        .as_ref()
        .is_some_and(|live| Arc::ptr_eq(live, candidate))
    }

    /// Resolves the staged topology without substituting retained read views.
    #[cfg(feature = "lpg")]
    fn resolve_lpg_lifecycle_path(
        root: &Arc<LpgStore>,
        path: &GraphPath,
        created: &std::collections::HashMap<GraphPath, PendingCreatedGraph>,
        dropped: &std::collections::HashMap<GraphPath, Arc<LpgStore>>,
        cancelled: &std::collections::HashMap<GraphPath, Vec<Arc<LpgStore>>>,
    ) -> Option<Arc<LpgStore>> {
        let mut current = Arc::clone(root);
        let mut prefix = GraphPath::root();
        for name in path.components() {
            prefix = prefix.child(name).ok()?;
            if let Some(pending) = created.get(&prefix) {
                if !Arc::ptr_eq(&current, &pending.parent) {
                    return None;
                }
                current = Arc::clone(&pending.store);
            } else if dropped.contains_key(&prefix) || cancelled.contains_key(&prefix) {
                return None;
            } else {
                current = current.graph(name)?;
            }
        }
        Some(current)
    }

    #[cfg(feature = "lpg")]
    fn rollback_pending_index_ddl(&self) {
        self.pending_index_ddl.lock().clear();
    }

    #[cfg(feature = "lpg")]
    fn rollback_pending_projection_ddl(&self) {
        self.pending_projection_ddl.lock().clear();
        self.projection_registry_snapshot.lock().take();
        self.projection_registry_read
            .store(false, Ordering::Release);
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn projection_ddl_error(message: impl Into<String>) -> grafeo_common::utils::error::Error {
        grafeo_common::utils::error::Error::Query(grafeo_common::utils::error::QueryError::new(
            grafeo_common::utils::error::QueryErrorKind::Semantic,
            message,
        ))
    }

    /// Captures one coherent projection source and its exact lifecycle token.
    /// Missing built-in named graphs are errors; they never fall back to the
    /// detached empty read sentinel used by older infallible helper paths.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn capture_virtual_projection_source(
        &self,
        context: &SessionGraphContext,
    ) -> Result<(
        Arc<dyn GraphStoreSearch>,
        Option<(GraphPath, Arc<LpgStore>)>,
    )> {
        match self.lpg_backend {
            LpgBackend::Placeholder { .. } => Ok((Arc::clone(&self.graph_store), None)),
            LpgBackend::Active => {
                if context.storage_key.components().is_empty() {
                    return Ok((Arc::clone(&self.graph_store), None));
                }
                let incarnation =
                    self.session_graph_path(&context.storage_key)
                        .ok_or_else(|| {
                            Self::projection_ddl_error(format!(
                                "Graph {:?} does not exist",
                                context.storage_key
                            ))
                        })?;
                Ok((
                    Arc::clone(&incarnation) as Arc<dyn GraphStoreSearch>,
                    Some((context.storage_key.clone(), incarnation)),
                ))
            }
        }
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn stage_create_virtual_projection(
        &self,
        name: String,
        spec: grafeo_core::graph::ProjectionSpec,
    ) -> Result<bool> {
        use grafeo_core::graph::GraphProjection;

        // Snapshot selector locks before publication, then retain one read cut
        // through source resolution and projection-name expectation capture.
        let context = self.graph_context_snapshot();
        let _publication = self.publication_read_guard();
        self.require_graph_path_grant(&context.storage_key, crate::auth::Role::ReadWrite)?;
        let (source_store, owner) = self.capture_virtual_projection_source(&context)?;
        let projection = Arc::new(GraphProjection::new(source_store, spec));
        let replacement = Arc::new(match owner {
            Some((storage_key, incarnation)) => {
                RegisteredGraphProjection::named(projection, storage_key, incarnation)
            }
            None => RegisteredGraphProjection::root_or_external(projection),
        });

        // Capture the name's committed identity in the same publication cut as
        // staging. Later touches retain this first expectation and change only
        // the final transaction-local post-image.
        let registry = self.projections.read();
        let snapshot = self.projection_registry_snapshot.lock();
        let base = snapshot.as_ref().unwrap_or(&registry);
        let mut pending = self.pending_projection_ddl.lock();
        let effective = pending.get(&name).map_or_else(
            || base.get(&name).cloned(),
            |change| change.replacement.clone(),
        );
        if effective.is_some() {
            return Ok(false);
        }

        match pending.get_mut(&name) {
            Some(change) => change.replacement = Some(replacement),
            None => {
                pending.insert(
                    name.clone(),
                    PendingProjectionChange {
                        expected: base.get(&name).cloned(),
                        replacement: Some(replacement),
                    },
                );
            }
        }
        Ok(true)
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn stage_drop_virtual_projection(&self, name: &str) -> Result<bool> {
        let _publication = self.publication_read_guard();
        let registry = self.projections.read();
        let snapshot = self.projection_registry_snapshot.lock();
        let base = snapshot.as_ref().unwrap_or(&registry);
        let mut pending = self.pending_projection_ddl.lock();
        let effective = pending.get(name).map_or_else(
            || base.get(name).cloned(),
            |change| change.replacement.clone(),
        );
        let Some(effective) = effective else {
            return Ok(false);
        };
        let root = GraphPath::root();
        let source_graph = effective
            .source_graph()
            .map_or(&root, |(storage_key, _)| storage_key);
        self.require_graph_path_grant(source_graph, crate::auth::Role::ReadWrite)?;

        match pending.get_mut(name) {
            Some(change) => change.replacement = None,
            None => {
                pending.insert(
                    name.to_string(),
                    PendingProjectionChange {
                        expected: base.get(name).cloned(),
                        replacement: None,
                    },
                );
            }
        }
        Ok(true)
    }

    /// Stages removal of every registry entry owned by one exact graph
    /// incarnation or descendant. A replacement projection over a newly-created
    /// graph with the same storage key is deliberately retained.
    #[cfg(feature = "lpg")]
    fn stage_virtual_projection_cascade(
        &self,
        storage_key: &GraphPath,
        incarnation: &Arc<LpgStore>,
    ) {
        let _publication = self.publication_read_guard();
        let registry = self.projections.read();
        let snapshot = self.projection_registry_snapshot.lock();
        let base = snapshot.as_ref().unwrap_or(&registry);
        let mut pending = self.pending_projection_ddl.lock();
        let mut names: std::collections::HashSet<String> = base.keys().cloned().collect();
        names.extend(pending.keys().cloned());

        for name in names {
            let effective = pending.get(&name).map_or_else(
                || base.get(&name).cloned(),
                |change| change.replacement.clone(),
            );
            if !effective
                .as_ref()
                .is_some_and(|entry| entry.is_owned_by(storage_key, incarnation))
            {
                continue;
            }
            match pending.get_mut(&name) {
                Some(change) => change.replacement = None,
                None => {
                    pending.insert(
                        name.clone(),
                        PendingProjectionChange {
                            expected: base.get(&name).cloned(),
                            replacement: None,
                        },
                    );
                }
            }
        }
    }

    /// Conservatively validates the whole-registry predicate observed by
    /// `SHOW PROJECTIONS` in a Serializable transaction.
    #[cfg(feature = "lpg")]
    fn validate_serializable_projection_read(&self, transaction_id: TransactionId) -> Result<()> {
        use grafeo_common::utils::error::{Error, TransactionError};

        if self.transaction_manager.isolation_level(transaction_id)
            != Some(crate::transaction::IsolationLevel::Serializable)
            || !self.projection_registry_read.load(Ordering::Acquire)
        {
            return Ok(());
        }

        let registry = self.projections.read();
        let snapshot = self.projection_registry_snapshot.lock();
        let Some(snapshot) = snapshot.as_ref() else {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Serializable projection read has no transaction-start snapshot".to_string(),
            )));
        };
        let unchanged = registry.len() == snapshot.len()
            && snapshot.iter().all(|(name, expected)| {
                registry
                    .get(name)
                    .is_some_and(|current| Arc::ptr_eq(current, expected))
            });
        if !unchanged {
            return Err(Error::Transaction(TransactionError::SerializationFailure(
                "virtual projection registry changed after a Serializable predicate read"
                    .to_string(),
            )));
        }
        Ok(())
    }

    /// Prepares a complete, validated registry post-image while the caller
    /// holds the publication write lock. This is the only fallible phase.
    #[cfg(feature = "lpg")]
    fn prepare_pending_projection_ddl(
        &self,
    ) -> Result<Option<std::collections::HashMap<String, Arc<RegisteredGraphProjection>>>> {
        use grafeo_common::utils::error::{Error, TransactionError};

        let pending = self.pending_projection_ddl.lock().clone();
        let dropped = self.pending_dropped_graphs.lock().clone();
        if pending.is_empty() && dropped.is_empty() {
            return Ok(None);
        }

        let registry = self.projections.read();
        let mut prepared = registry.clone();
        let mut names: Vec<_> = pending.keys().cloned().collect();
        names.sort();
        for name in names {
            let change = &pending[&name];
            let current = registry.get(&name);
            let expectation_matches = match (&change.expected, current) {
                (None, None) => true,
                (Some(expected), Some(current)) => Arc::ptr_eq(expected, current),
                (None, Some(_)) | (Some(_), None) => false,
            };
            if !expectation_matches {
                return Err(Error::Transaction(TransactionError::WriteConflict(
                    format!("projection '{name}' changed during the transaction"),
                )));
            }
            match &change.replacement {
                Some(replacement) => {
                    prepared.insert(name, Arc::clone(replacement));
                }
                None => {
                    prepared.remove(&name);
                }
            }
        }
        drop(registry);

        // A graph-drop transaction also removes a projection committed after
        // its statement-time cascade but before this publication frontier.
        prepared.retain(|_, entry| {
            !dropped
                .iter()
                .any(|(key, graph)| entry.is_owned_by(key, graph))
        });

        for (name, entry) in &prepared {
            let Some((storage_key, incarnation)) = entry.source_graph() else {
                continue;
            };
            if !self.lpg_incarnation_survives_pending_lifecycle(storage_key, incarnation) {
                return Err(Error::Transaction(TransactionError::WriteConflict(
                    format!(
                        "projection '{name}' source graph {storage_key:?} was dropped or replaced during the transaction"
                    ),
                )));
            }
        }

        Ok(Some(prepared))
    }

    /// Installs a previously validated virtual-projection post-image. Callers
    /// hold the publication write lock, so readers observe it atomically with
    /// graph lifecycle and data visibility.
    #[cfg(feature = "lpg")]
    fn publish_prepared_projection_ddl(
        &self,
        prepared: Option<std::collections::HashMap<String, Arc<RegisteredGraphProjection>>>,
    ) {
        if let Some(prepared) = prepared {
            *self.projections.write() = prepared;
        }
        self.pending_projection_ddl.lock().clear();
        self.projection_registry_snapshot.lock().take();
        self.projection_registry_read
            .store(false, Ordering::Release);
    }

    #[cfg(feature = "lpg")]
    fn lpg_lifecycle_snapshot(&self) -> LpgLifecycleSavepoint {
        // Keep the same lock order as lifecycle validation so a Session used
        // from multiple threads cannot deadlock while a savepoint is captured.
        let created = self.pending_created_graphs.lock();
        let dropped = self.pending_dropped_graphs.lock();
        let cancelled = self.cancelled_created_graphs.lock();
        let touched = self.touched_named_graphs.lock();
        let missing = self.missing_named_graphs.lock();
        let bindings = self.pending_graph_type_bindings.lock();
        LpgLifecycleSavepoint {
            pending_created_graphs: created.clone(),
            pending_dropped_graphs: dropped.clone(),
            cancelled_created_graphs: cancelled.clone(),
            touched_named_graphs: touched.clone(),
            superseded_graph_touches: self.superseded_graph_touches.lock().clone(),
            missing_named_graphs: missing.clone(),
            pending_graph_type_bindings: bindings.clone(),
        }
    }

    #[cfg(feature = "lpg")]
    fn restore_lpg_lifecycle_snapshot(&self, snapshot: LpgLifecycleSavepoint) {
        let mut created = self.pending_created_graphs.lock();
        let mut dropped = self.pending_dropped_graphs.lock();
        let mut cancelled = self.cancelled_created_graphs.lock();
        let mut touched = self.touched_named_graphs.lock();
        let mut missing = self.missing_named_graphs.lock();
        let mut bindings = self.pending_graph_type_bindings.lock();
        *created = snapshot.pending_created_graphs;
        *dropped = snapshot.pending_dropped_graphs;
        *cancelled = snapshot.cancelled_created_graphs;
        *touched = snapshot.touched_named_graphs;
        *self.superseded_graph_touches.lock() = snapshot.superseded_graph_touches;
        *missing = snapshot.missing_named_graphs;
        *bindings = snapshot.pending_graph_type_bindings;
    }

    #[cfg(feature = "lpg")]
    fn session_graph_type_binding(&self, graph_name: &GraphPath) -> Option<String> {
        if let Some(replacement) = self
            .pending_graph_type_bindings
            .lock()
            .get(graph_name)
            .map(|pending| pending.replacement.clone())
        {
            return replacement;
        }
        self.catalog_view().get_graph_type_binding(graph_name)
    }

    #[cfg(all(
        not(feature = "lpg"),
        any(
            feature = "gql",
            feature = "cypher",
            feature = "gremlin",
            feature = "graphql",
            feature = "sql-pgq"
        )
    ))]
    fn session_graph_type_binding(&self, graph_name: &GraphPath) -> Option<String> {
        self.catalog_view().get_graph_type_binding(graph_name)
    }

    #[cfg(feature = "lpg")]
    fn stage_graph_type_binding(&self, graph_name: &GraphPath, replacement: Option<String>) {
        let mut pending = self.pending_graph_type_bindings.lock();
        let expected = pending.get(graph_name).map_or_else(
            || self.catalog_view().get_graph_type_binding(graph_name),
            |change| change.expected.clone(),
        );
        pending.insert(
            graph_name.clone(),
            PendingGraphTypeBinding {
                expected,
                replacement,
            },
        );
    }

    /// Stage the complete binding subtree, matching literal components rather
    /// than treating slashes in a name as topology. Preparation checks for any
    /// bindings added after this capture under the catalog publication writer.
    #[cfg(feature = "lpg")]
    fn stage_graph_type_binding_cascade(&self, root: &GraphPath) -> Result<()> {
        let mut paths: std::collections::HashSet<_> = self
            .catalog_view()
            .all_graph_type_bindings()
            .into_iter()
            .map(|(path, _)| path)
            .filter(|path| path.components().starts_with(root.components()))
            .collect();
        paths.extend(
            self.pending_graph_type_bindings
                .lock()
                .keys()
                .filter(|path| path.components().starts_with(root.components()))
                .cloned(),
        );
        paths.insert(root.clone());
        for path in paths {
            self.stage_graph_type_binding(&path, None);
        }
        Ok(())
    }

    #[cfg(feature = "lpg")]
    fn validate_pending_graph_lifecycle(&self) -> Result<Vec<PreparedGraphLifecycleChange>> {
        use grafeo_common::utils::error::{Error, TransactionError};

        // Index DDL precedes lifecycle locks in the session lock order. The
        // snapshot also lets dependency validation evaluate the transaction's
        // final post-image instead of rejecting an atomic DROP INDEX; DROP GRAPH.
        let pending_indexes = self.pending_index_ddl.lock().clone();
        let created = self.pending_created_graphs.lock();
        let dropped = self.pending_dropped_graphs.lock();
        let cancelled = self.cancelled_created_graphs.lock();

        // A DROP is tied to the exact graph incarnation observed by this
        // transaction. A missing or different Arc means another transaction
        // published a lifecycle change first.
        for (name, expected) in dropped.iter() {
            let (logical_indexes, has_physical_indexes) =
                self.graph_indexes_after_pending_ddl(name, expected, &pending_indexes)?;
            if !logical_indexes.is_empty() || has_physical_indexes {
                return Err(Error::Transaction(TransactionError::WriteConflict(
                    format!("graph {name:?} owns indexes that must be dropped before the graph"),
                )));
            }
            let current = self.live_graph_path(name);
            if !current
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, expected))
            {
                return Err(Error::Transaction(TransactionError::WriteConflict(
                    format!("graph {name:?} changed during the transaction"),
                )));
            }
        }

        // A detached CREATE that is not replacing a graph must still target an
        // absent name when its commit reaches the publication boundary.
        for (name, pending) in created.iter() {
            match &pending.namespace {
                PendingGraphNamespace::Root => {}
                PendingGraphNamespace::Schema(owner) => {
                    let default_key = format!("{}/{SCHEMA_DEFAULT_GRAPH}", owner.name);
                    let default_path = Self::graph_path_for_storage_key(Some(&default_key))?;
                    let live_default = created
                        .get(&default_path)
                        .map(|pending| Arc::clone(&pending.store))
                        .or_else(|| self.store.graph(&default_key));
                    if !self.catalog_view().schema_exists(&owner.name)
                        || !live_default
                            .as_ref()
                            .is_some_and(|graph| Arc::ptr_eq(graph, &owner.default_graph))
                    {
                        return Err(Error::Transaction(TransactionError::WriteConflict(
                            format!(
                                "owning schema '{}' was dropped or replaced while creating graph {name:?}",
                                owner.name
                            ),
                        )));
                    }
                }
                PendingGraphNamespace::UnregisteredPrefix(prefix) => {
                    if self
                        .catalog_view()
                        .schema_names()
                        .iter()
                        .any(|name| name.eq_ignore_ascii_case(prefix))
                    {
                        return Err(Error::Transaction(TransactionError::WriteConflict(
                            format!(
                                "schema '{prefix}' was created while graph {name:?} was being created"
                            ),
                        )));
                    }
                }
            }
            if let Some(expectation) = pending.copy_source_indexes.as_ref() {
                let source_is_retained = match expectation.source_name.as_deref() {
                    None => Arc::ptr_eq(&self.store, &expectation.source),
                    Some(source_name) => {
                        let source_path = Self::graph_path_for_storage_key(Some(source_name))?;
                        created
                            .get(&source_path)
                            .is_some_and(|pending| Arc::ptr_eq(&pending.store, &expectation.source))
                            || dropped
                                .get(&source_path)
                                .is_some_and(|source| Arc::ptr_eq(source, &expectation.source))
                            || cancelled.get(&source_path).is_some_and(|sources| {
                                sources
                                    .iter()
                                    .any(|source| Arc::ptr_eq(source, &expectation.source))
                            })
                            || self
                                .store
                                .graph(source_name)
                                .as_ref()
                                .is_some_and(|source| Arc::ptr_eq(source, &expectation.source))
                    }
                };
                let source_name = expectation.source_name.as_deref().unwrap_or("default");
                if !source_is_retained {
                    return Err(Error::Transaction(TransactionError::WriteConflict(
                        format!(
                            "COPY source graph '{}' was dropped or replaced during the transaction",
                            source_name
                        ),
                    )));
                }
                let mut current_physical_keys = expectation.source.property_index_keys();
                current_physical_keys.sort_unstable();
                current_physical_keys.dedup();
                if current_physical_keys != expectation.physical_keys {
                    return Err(Error::Transaction(TransactionError::WriteConflict(
                        format!(
                            "COPY source graph '{}' changed its physical property-index registry during the transaction",
                            source_name
                        ),
                    )));
                }
            }
            let parent_path = name
                .parent()
                .map_err(|error| Self::index_ddl_error(error.to_string()))?
                .ok_or_else(|| Self::index_ddl_error("root cannot be staged for creation"))?;
            let parent = Self::resolve_lpg_lifecycle_path(
                &self.store,
                &parent_path,
                &created,
                &dropped,
                &cancelled,
            );
            if !parent
                .as_ref()
                .is_some_and(|parent| Arc::ptr_eq(parent, &pending.parent))
            {
                return Err(Error::Transaction(TransactionError::WriteConflict(
                    format!("parent of graph {name:?} was dropped or replaced"),
                )));
            }
            let child_name = name
                .components()
                .last()
                .ok_or_else(|| Self::index_ddl_error("created graph has no child name"))?;
            let existing = pending.parent.graph(child_name);
            if existing.as_ref().is_some_and(|current| {
                !dropped
                    .get(name)
                    .is_some_and(|expected| Arc::ptr_eq(current, expected))
            }) {
                return Err(Error::Transaction(TransactionError::WriteConflict(
                    format!("graph {name:?} was created concurrently"),
                )));
            }
        }

        // Each observed prefix retains its committed incarnation. A transaction-
        // owned replacement is separately validated by its root lifecycle CAS.
        for (path, expected) in self.touched_named_graphs.lock().iter() {
            let own_lifecycle = created
                .keys()
                .chain(dropped.keys())
                .chain(cancelled.keys())
                .any(|prefix| path.components().starts_with(prefix.components()));
            if own_lifecycle {
                continue;
            }
            if !self
                .live_graph_path(path)
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, expected))
            {
                return Err(Error::Transaction(TransactionError::WriteConflict(
                    format!("graph {path:?} was dropped or replaced concurrently"),
                )));
            }
        }
        for path in self.missing_named_graphs.lock().iter() {
            let own_create = created
                .keys()
                .any(|prefix| path.components().starts_with(prefix.components()));
            if !own_create && self.live_graph_path(path).is_some() {
                return Err(Error::Transaction(TransactionError::WriteConflict(
                    format!("graph {path:?} was absent at first touch and appeared concurrently"),
                )));
            }
        }

        for (name, pending) in self.pending_graph_type_bindings.lock().iter() {
            if let Some(graph_type) = pending.replacement.as_deref()
                && let Err(error) = self
                    .catalog_view()
                    .validate_graph_type_binding_target(graph_type)
            {
                return Err(Error::Transaction(TransactionError::WriteConflict(
                    format!(
                        "graph type binding for {name:?} changed during the transaction: {error}"
                    ),
                )));
            }
            if !self
                .catalog
                .graph_type_binding_matches(name, pending.expected.as_deref())
            {
                return Err(Error::Transaction(TransactionError::WriteConflict(
                    format!("graph type binding for {name:?} was changed concurrently"),
                )));
            }
        }

        let mut changes = Vec::new();
        changes
            .try_reserve(created.len().saturating_add(dropped.len()))
            .map_err(|error| {
                Error::Internal(format!("cannot prepare graph publication: {error}"))
            })?;
        let mut retire: Vec<_> = dropped.iter().collect();
        retire.sort_unstable_by(|(a, _), (b, _)| {
            b.components()
                .len()
                .cmp(&a.components().len())
                .then_with(|| a.cmp(b))
        });
        for (path, expected) in retire {
            let parent_path = path
                .parent()
                .map_err(|error| Self::index_ddl_error(error.to_string()))?
                .ok_or_else(|| Self::index_ddl_error("root cannot be dropped"))?;
            let parent = self
                .live_graph_path(&parent_path)
                .ok_or_else(|| Self::index_ddl_error("dropped graph parent disappeared"))?;
            let name = path
                .components()
                .last()
                .ok_or_else(|| Self::index_ddl_error("dropped graph has no child name"))?
                .clone();
            changes.push(PreparedGraphLifecycleChange::Retire {
                parent,
                name,
                expected: Arc::clone(expected),
            });
        }
        let mut install: Vec<_> = created.iter().collect();
        install.sort_unstable_by(|(a, _), (b, _)| {
            a.components()
                .len()
                .cmp(&b.components().len())
                .then_with(|| a.cmp(b))
        });
        for (path, pending) in install {
            let name = path
                .components()
                .last()
                .ok_or_else(|| Self::index_ddl_error("created graph has no child name"))?
                .clone();
            changes.push(PreparedGraphLifecycleChange::Install {
                parent: Arc::clone(&pending.parent),
                name,
                store: Arc::clone(&pending.store),
            });
        }
        Ok(changes)
    }

    #[cfg(feature = "lpg")]
    fn commit_pending_graph_lifecycle(
        &self,
        commit_epoch: EpochId,
        changes: Vec<PreparedGraphLifecycleChange>,
    ) -> Result<()> {
        for change in changes {
            let (name, published) = match change {
                PreparedGraphLifecycleChange::Retire {
                    parent,
                    name,
                    expected,
                } => {
                    let published = parent.drop_graph_if_same(&name, &expected);
                    (name, published)
                }
                PreparedGraphLifecycleChange::Install {
                    parent,
                    name,
                    store,
                } => {
                    store.sync_epoch(commit_epoch);
                    let published = parent.install_graph_if_absent(&name, store);
                    (name, published)
                }
            };
            if !published {
                return Err(grafeo_common::utils::error::Error::Internal(format!(
                    "graph {:?} changed after lifecycle preparation",
                    name
                )));
            }
        }
        self.pending_created_graphs.lock().clear();
        self.pending_dropped_graphs.lock().clear();
        // Catalog bindings were consumed with the complete prepared logical
        // postimage. This tail only publishes graph-registry lifecycle.
        self.cancelled_created_graphs.lock().clear();
        self.missing_named_graphs.lock().clear();
        Ok(())
    }

    /// Allocates a graph that remains private to this session until commit.
    #[cfg(feature = "lpg")]
    fn new_detached_lpg_graph(&self) -> Result<Arc<LpgStore>> {
        let graph =
            Arc::new(self.store.new_named_graph_candidate().map_err(|error| {
                grafeo_common::utils::error::Error::Internal(error.to_string())
            })?);
        if !graph.seal_unframed_writes(self.transaction_manager.write_authority()) {
            return Err(grafeo_common::utils::error::Error::Internal(
                "failed to bind detached named graph to database write authority".to_string(),
            ));
        }
        Ok(graph)
    }

    /// Discards this transaction's pending data from one graph incarnation.
    ///
    /// Once an existing graph is queued for DROP, its earlier writes cannot
    /// survive a successful commit. Cleanup is nevertheless deferred until
    /// the transaction outcome is irrevocable: rollback-to-savepoint may
    /// restore that exact incarnation and its transaction-local state.
    #[cfg(feature = "lpg")]
    fn discard_lpg_graph_transaction(&self, store: &Arc<LpgStore>, tid: TransactionId) {
        let mutation_store: Arc<dyn GraphStoreMut> = Arc::clone(store) as Arc<dyn GraphStoreMut>;
        self.discard_lpg_graph_transaction_via(store, &mutation_store, tid);
    }

    /// Full transaction cleanup through the exact mutation layer that accepted
    /// the writes. This matters for a compacted default graph, where base-tier
    /// tombstones live on `LayeredStore`, not on the concrete LPG overlay.
    #[cfg(feature = "lpg")]
    fn discard_lpg_graph_transaction_via(
        &self,
        store: &Arc<LpgStore>,
        mutation_store: &Arc<dyn GraphStoreMut>,
        tid: TransactionId,
    ) {
        let (pending_nodes, pending_edges) = store.take_pending_creates(tid);
        store.discard_entities_by_id(tid, &pending_nodes, &pending_edges);
        store.rollback_transaction_properties(tid);
        mutation_store.drop_tx_overlay(tid);
        let pending_deletes = mutation_store.take_pending_deletes(tid);
        store.rollback_pending_deletes(tid, &pending_deletes);
        let pending_edge_deletes = mutation_store.take_pending_edge_deletes(tid);
        store.rollback_pending_edge_deletes(tid, &pending_edge_deletes);
        mutation_store.unregister_read_tracker(tid);
        mutation_store.unregister_write_tracker(tid);
    }

    fn poison_durability(&self) {
        self.durability_poisoned
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn database_closed_error() -> grafeo_common::utils::error::Error {
        grafeo_common::utils::error::Error::Transaction(
            grafeo_common::utils::error::TransactionError::InvalidState(
                "database is closed".to_string(),
            ),
        )
    }

    pub(crate) fn check_not_poisoned(&self) -> Result<()> {
        if !*self.database_open.read() {
            return Err(Self::database_closed_error());
        }

        self.check_durability_not_poisoned()
    }

    /// Rechecks only sticky durability state without taking the lifecycle
    /// lock. CDC readers call this after acquiring the publication read side:
    /// taking `database_open` there would invert close's documented
    /// lifecycle-before-publication order.
    fn check_durability_not_poisoned(&self) -> Result<()> {
        // TypedWal tracks failures at the lowest common logging boundary. Some
        // mutation operators can only report their own execution status, so
        // promote that sticky WAL state to the database-wide poison flag at
        // every checked Session boundary.
        #[cfg(feature = "wal")]
        if self.wal.as_ref().is_some_and(|wal| wal.is_poisoned()) {
            self.poison_durability();
        }

        if self
            .durability_poisoned
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::DurabilityFailure(
                    "session poisoned after a WAL log/fsync failure; reopen and recover before continuing"
                        .into(),
                ),
            ));
        }
        Ok(())
    }

    /// Enforces authorization and transaction mode after parsing has
    /// classified a logical plan. Text inspection is deliberately forbidden:
    /// a false negative would execute an unframed write.
    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn check_lpg_query_access(&self, has_mutations: bool) -> Result<()> {
        let graph_name = self.active_graph_storage_key();
        if has_mutations {
            self.require_permission(crate::auth::StatementKind::Write)?;
            self.require_graph_path_grant(&graph_name, crate::auth::Role::ReadWrite)?;
        } else {
            self.require_permission(crate::auth::StatementKind::Read)?;
            self.require_graph_path_grant(&graph_name, crate::auth::Role::ReadOnly)?;
        }
        if has_mutations && *self.read_only_tx.lock() {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::ReadOnly,
            ));
        }
        if has_mutations {
            self.reject_lpg_historical_mutation()?;
        }
        Ok(())
    }

    /// Resolves transitive procedure effects before choosing authorization or
    /// transaction framing. Catalog bodies are captured at the same
    /// publication cut and later handed unchanged to the physical planner.
    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn qualify_lpg_plan(
        &self,
        root: &crate::query::plan::LogicalOperator,
    ) -> Result<QualifiedLpgPlan> {
        #[cfg(all(any(feature = "lpg", feature = "algos"), feature = "gql"))]
        let qualified = {
            let _publication = self.publication_read_guard();
            let effects = crate::query::procedure_effect::analyze_procedure_effects(
                root,
                self.catalog_view().as_ref(),
            )?;
            QualifiedLpgPlan {
                mutates: effects.mutates,
                contains_call: effects.contains_call,
                contains_catalog_call: effects.contains_catalog_call(),
                procedures: effects.procedures,
            }
        };

        #[cfg(all(any(feature = "lpg", feature = "algos"), not(feature = "gql")))]
        let qualified = {
            let effects = crate::procedures::analyze_builtin_procedure_effects(root)?;
            QualifiedLpgPlan {
                mutates: effects.mutates,
                contains_call: effects.contains_call,
            }
        };

        #[cfg(not(any(feature = "lpg", feature = "algos")))]
        let qualified = QualifiedLpgPlan {
            mutates: root.has_mutations(),
            contains_call: root.contains_procedure_call(),
        };

        if self.effective_viewing_epoch().is_some() && qualified.contains_call {
            Self::reject_unclassified_procedure_call(root, "historical execution")?;
        }
        self.check_lpg_query_access(qualified.mutates)?;
        Ok(qualified)
    }

    /// Revalidates the captured catalog after auto-commit has opened its
    /// transaction. The caller holds the publication read guard through
    /// physical execution, so no DROP/REPLACE can race the authorized body.
    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn validate_qualified_lpg_plan(&self, _qualified: &QualifiedLpgPlan) -> Result<()> {
        #[cfg(all(any(feature = "lpg", feature = "algos"), feature = "gql"))]
        if _qualified.contains_catalog_call {
            _qualified
                .procedures
                .validate_live_catalog(&self.catalog_view())?;
        }
        Ok(())
    }

    #[cfg(all(
        any(feature = "lpg", feature = "algos"),
        any(
            feature = "gql",
            feature = "cypher",
            feature = "gremlin",
            feature = "graphql",
            feature = "sql-pgq"
        )
    ))]
    fn attach_qualified_procedures(
        &self,
        planner: crate::query::Planner,
        _qualified: &QualifiedLpgPlan,
    ) -> crate::query::Planner {
        #[cfg(feature = "gql")]
        let planner = planner.with_session_procedure_write_authority();
        #[cfg(feature = "gql")]
        let planner = planner.with_resolved_procedures(Arc::clone(&_qualified.procedures));
        planner
    }

    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn reject_unclassified_procedure_call(
        root: &crate::query::plan::LogicalOperator,
        execution_kind: &str,
    ) -> Result<()> {
        if !root.contains_procedure_call() {
            return Ok(());
        }
        Err(grafeo_common::utils::error::Error::Query(
            grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Unsupported,
                format!(
                    "procedure calls are not qualified for {execution_kind} until their transitive effects and snapshot context are sealed"
                ),
            ),
        ))
    }

    /// A query must never report success after a lower-level WAL operation has
    /// poisoned durability, even if an operator failed to propagate that I/O
    /// error directly.
    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq",
        feature = "sparql"
    ))]
    fn finish_query(&self, result: Result<QueryResult>) -> Result<QueryResult> {
        if result.is_ok() {
            self.check_not_poisoned()?;
        }
        result
    }

    /// Attaches the shared retained feed and optionally enables mutation capture.
    ///
    /// Wraps the current write store with a `CdcGraphStore` decorator so
    /// that all session mutations (INSERT, SET, DELETE via query execution)
    /// buffer CDC events. The buffer is flushed to the `CdcLog` on commit
    /// and discarded on rollback.
    /// Construction-only: attach once, after store selection and WAL wrapping.
    #[cfg(feature = "cdc")]
    pub(crate) fn set_cdc_log(&mut self, cdc_log: Arc<crate::cdc::CdcLog>, capture: bool) {
        if !capture {
            // Disabling future capture does not replace retained read authority.
            self.cdc_log = cdc_log;
            return;
        }
        // Wrap the WRITE store only with CdcGraphStore to intercept mutations.
        // The read store (self.graph_store) is left unchanged for zero read overhead.
        #[cfg(feature = "lpg")]
        let pending_events = if let Some(write_store) =
            self.graph_store_mut.as_ref().map(Arc::clone)
        {
            let cdc_store = Arc::new(crate::database::cdc_store::CdcGraphStore::new(
                write_store,
                Arc::clone(&cdc_log),
                Arc::clone(&self.store),
                GraphPath::root(),
            ));
            let pending_events = cdc_store.pending_events();
            self.default_cdc_writer = Some(Arc::clone(&cdc_store));
            self.graph_store_mut = Some(cdc_store as Arc<dyn grafeo_core::graph::GraphStoreMut>);
            pending_events
        } else {
            self.default_cdc_writer = None;
            Arc::new(crate::cdc::TransactionChangeAccumulator::new(&cdc_log))
        };
        #[cfg(not(feature = "lpg"))]
        let pending_events = Arc::new(crate::cdc::TransactionChangeAccumulator::new(&cdc_log));

        // RDF-only sessions have no LPG decorator. They still bind their
        // staging accumulator to this exact database log.
        self.cdc_pending_events = Some(pending_events);
        self.cdc_log = cdc_log;
    }

    /// Acquires the per-Session mutation/context gate for every store-backed
    /// session.
    ///
    /// Transaction state, graph context, WAL framing, CDC staging, and
    /// savepoint snapshots form one Session-local state machine. Serializing
    /// them unconditionally prevents unrestricted sessions from racing two
    /// transaction boundaries or splitting a savepoint snapshot from the
    /// mutation it governs.
    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    fn session_operation_guard(&self) -> Option<impl std::ops::Deref<Target = ()> + '_> {
        Some(self.mutation_operation_gate.lock())
    }

    /// Language-only feature slices have no graph state to serialize. Keep
    /// their parser/API surface compilable while preserving the zero-work
    /// behavior of a session without either storage model.
    #[cfg(not(any(feature = "lpg", feature = "triple-store")))]
    fn session_operation_guard(&self) -> Option<&()> {
        None
    }

    #[cfg(all(feature = "cdc", feature = "lpg"))]
    fn cdc_mutation_operation_guard(&self) -> Option<impl std::ops::Deref<Target = ()> + '_> {
        self.session_operation_guard()
    }

    #[cfg(all(feature = "cdc", feature = "lpg"))]
    fn stage_lpg_node_create(
        &self,
        id: NodeId,
        properties: Option<std::collections::HashMap<String, Value>>,
        labels: Option<Vec<String>>,
        graph: &GraphPath,
        graph_incarnation: Arc<LpgStore>,
    ) {
        let Some(ref accumulator) = self.cdc_pending_events else {
            return;
        };
        let event = self
            .cdc_log
            .node_create_event(id, EpochId::PENDING, properties, labels, graph);
        accumulator.stage_lpg(event, graph, graph_incarnation);
    }

    #[cfg(all(feature = "cdc", feature = "lpg"))]
    fn stage_lpg_edge_create(
        &self,
        id: grafeo_common::types::EdgeId,
        properties: Option<std::collections::HashMap<String, Value>>,
        endpoints: (NodeId, NodeId),
        edge_type: String,
        graph: &GraphPath,
        graph_incarnation: Arc<LpgStore>,
    ) {
        let Some(ref accumulator) = self.cdc_pending_events else {
            return;
        };
        let event = self.cdc_log.edge_create_event(
            id,
            EpochId::PENDING,
            properties,
            (endpoints.0.as_u64(), endpoints.1.as_u64()),
            edge_type,
            graph,
        );
        accumulator.stage_lpg(event, graph, graph_incarnation);
    }

    /// Sets the metrics registry for this session (shared with the database).
    #[cfg(feature = "metrics")]
    pub(crate) fn set_metrics(&mut self, metrics: Arc<crate::metrics::MetricsRegistry>) {
        self.metrics = Some(metrics);
    }

    /// Creates a session backed by an external graph store.
    ///
    /// The external store handles all data operations. Native LPG transactions
    /// require its mutable view to expose the exact owned commit target;
    /// arbitrary external backends remain queryable without that capability.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal arena allocation fails (out of memory).
    pub(crate) fn with_external_store(
        read_store: Arc<dyn GraphStoreSearch>,
        write_store: Option<Arc<dyn GraphStoreMut>>,
        cfg: SessionConfig,
    ) -> Result<Self> {
        #[cfg(not(any(
            feature = "lpg",
            feature = "gql",
            feature = "cypher",
            feature = "gremlin",
            feature = "graphql",
            feature = "sql-pgq"
        )))]
        let _ = (read_store, write_store);
        #[cfg(feature = "lpg")]
        let (store, commit_target_available) = match write_store
            .as_ref()
            .and_then(|writer| Arc::clone(writer).lpg_commit_store())
        {
            Some(store) => (store, true),
            None => (Arc::new(LpgStore::new()?), false),
        };
        Ok(Self {
            #[cfg(feature = "lpg")]
            store,
            #[cfg(feature = "lpg")]
            lpg_backend: LpgBackend::Placeholder {
                commit_target_available,
            },
            #[cfg(any(
                feature = "lpg",
                feature = "gql",
                feature = "cypher",
                feature = "gremlin",
                feature = "graphql",
                feature = "sql-pgq"
            ))]
            graph_store: read_store,
            #[cfg(any(
                feature = "lpg",
                feature = "gql",
                feature = "cypher",
                feature = "gremlin",
                feature = "graphql",
                feature = "sql-pgq"
            ))]
            graph_store_mut: write_store,
            #[cfg(any(
                feature = "lpg",
                feature = "gql",
                feature = "cypher",
                feature = "gremlin",
                feature = "graphql",
                feature = "sql-pgq"
            ))]
            catalog: cfg.catalog,
            #[cfg(feature = "triple-store")]
            rdf_store: Arc::new(RdfStore::new()),
            transaction_manager: cfg.transaction_manager,
            query_cache: cfg.query_cache,
            physical_cache: cfg.physical_cache,
            current_transaction: parking_lot::Mutex::new(None),
            #[cfg(feature = "lpg")]
            transaction_catalog: parking_lot::Mutex::new(None),
            #[cfg(any(feature = "lpg", feature = "triple-store"))]
            mutation_operation_gate: parking_lot::ReentrantMutex::new(()),
            historical_view_operation_gate: parking_lot::ReentrantMutex::new(()),
            read_only_tx: parking_lot::Mutex::new(cfg.read_only),
            db_read_only: cfg.read_only,
            identity: cfg.identity,
            auto_commit: true,
            adaptive_config: cfg.adaptive_config,
            #[cfg(any(
                feature = "gql",
                feature = "cypher",
                feature = "gremlin",
                feature = "graphql",
                feature = "sql-pgq"
            ))]
            factorized_execution: cfg.factorized_execution,
            graph_model: cfg.graph_model,
            query_timeout: cfg.query_timeout,
            result_limits: cfg.result_limits,
            active_result_limits: parking_lot::Mutex::new(None),
            active_result_admission: parking_lot::Mutex::new(None),
            active_execution_control: parking_lot::Mutex::new(None),
            active_execution_completed: std::sync::atomic::AtomicBool::new(false),
            active_execution_statement_depth: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "testing-statement-injection")]
            query_cancellation_test_hook: parking_lot::Mutex::new(None),
            #[cfg(any(
                feature = "lpg",
                feature = "gql",
                feature = "cypher",
                feature = "gremlin",
                feature = "graphql",
                feature = "sql-pgq"
            ))]
            max_property_size: cfg.max_property_size,
            buffer_manager: cfg.buffer_manager,
            #[cfg(feature = "spill")]
            spill_root: cfg.spill_root,
            #[cfg(any(feature = "spill", feature = "cdc"))]
            world_identity: cfg.world_identity,
            #[cfg(feature = "lpg")]
            commit_counter: cfg.commit_counter,
            durability_poisoned: cfg.durability_poisoned,
            database_open: cfg.database_open,
            active_sessions: cfg.active_sessions,
            #[cfg(feature = "lpg")]
            gc_interval: cfg.gc_interval,
            #[cfg(feature = "lpg")]
            transaction_start_node_count: AtomicUsize::new(0),
            #[cfg(feature = "lpg")]
            transaction_start_edge_count: AtomicUsize::new(0),
            #[cfg(feature = "wal")]
            wal: None,
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            #[cfg(feature = "cdc")]
            cdc_pending_events: None,
            #[cfg(all(feature = "lpg", feature = "cdc"))]
            default_cdc_writer: None,
            current_context: parking_lot::Mutex::new(SessionGraphContext::default()),
            time_zone: parking_lot::Mutex::new(None),
            #[cfg(feature = "triple-store")]
            rdf_valid_time: parking_lot::Mutex::new(None),
            session_params: parking_lot::Mutex::new(std::collections::HashMap::new()),
            viewing_epoch_override: parking_lot::Mutex::new(None),
            savepoints: parking_lot::Mutex::new(Vec::new()),
            transaction_nesting_depth: parking_lot::Mutex::new(0),
            touched_graphs: parking_lot::Mutex::new(Vec::new()),
            #[cfg(feature = "lpg")]
            pending_created_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            pending_dropped_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            cancelled_created_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            touched_named_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            superseded_graph_touches: parking_lot::Mutex::new(Vec::new()),
            #[cfg(feature = "lpg")]
            missing_named_graphs: parking_lot::Mutex::new(std::collections::HashSet::new()),
            #[cfg(feature = "lpg")]
            pending_graph_type_bindings: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            pending_index_ddl: parking_lot::Mutex::new(Vec::new()),
            #[cfg(feature = "lpg")]
            pending_projection_ddl: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "lpg")]
            projection_registry_snapshot: parking_lot::Mutex::new(None),
            #[cfg(feature = "lpg")]
            projection_registry_read: std::sync::atomic::AtomicBool::new(false),
            #[cfg(all(feature = "wal", feature = "lpg"))]
            catalog_wal_batch: parking_lot::Mutex::new(None),
            #[cfg(all(feature = "lpg", feature = "triple-store"))]
            rdf_projection_target: parking_lot::Mutex::new(None),
            #[cfg(feature = "lpg")]
            active_streams: AtomicUsize::new(0),
            conflict_granularity: parking_lot::Mutex::new(
                crate::transaction::ConflictGranularity::Entity,
            ),
            #[cfg(feature = "metrics")]
            metrics: None,
            #[cfg(feature = "metrics")]
            tx_start_time: parking_lot::Mutex::new(None),
            #[cfg(feature = "lpg")]
            projections: cfg.projections,
        })
    }

    /// Returns the graph model this session operates on.
    #[must_use]
    pub fn graph_model(&self) -> GraphModel {
        self.graph_model
    }

    /// Returns the identity bound to this session.
    #[must_use]
    pub fn identity(&self) -> &crate::auth::Identity {
        &self.identity
    }

    // === Session State Management ===

    /// Selects one exact LPG path, without interpreting names or separators.
    ///
    /// Native selection remains independent of later language schema changes.
    ///
    /// # Errors
    /// Rejects missing graphs and insufficient read permission without changing
    /// the selected context.
    #[cfg(feature = "lpg")]
    pub fn use_graph_path(&self, path: &GraphPath) -> Result<()> {
        let _operation = self.session_operation_guard();
        let _publication = self.publication_read_guard();
        if !self.index_path_grant_allows(path, crate::auth::Role::ReadOnly) {
            return Err(Self::index_ddl_error("graph selection is not permitted"));
        }
        let target = self.session_graph_path(path);
        if let Some(transaction_id) = self.current_transaction_id() {
            self.track_lpg_graph_coordinate(transaction_id, path.clone(), target.clone())?;
        }
        if target.is_none() {
            return Err(Self::index_ddl_error(format!(
                "Graph {path:?} does not exist"
            )));
        }
        let mut context = self.current_context.lock();
        context.storage_key = path.clone();
        context.graph = None;
        context.native = true;
        Ok(())
    }

    /// Returns the exact resolved LPG coordinate, including the root path.
    #[must_use]
    pub fn current_graph_path(&self) -> GraphPath {
        self.current_context.lock().storage_key.clone()
    }

    /// Resolves a language selector without treating a schema separator as topology.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn use_graph(&self, name: &str) -> Result<()> {
        let _operation = self.session_operation_guard();
        let mut next = self.graph_context_snapshot();
        next.graph = Some(name.to_owned());
        next.native = false;
        next.storage_key = Self::context_graph_path(next.schema.as_deref(), next.graph.as_deref())?;
        self.use_graph_path(&next.storage_key)?;
        *self.current_context.lock() = next;
        Ok(())
    }

    /// Sets the language schema; explicitly selected native paths stay fixed.
    ///
    /// # Errors
    /// Rejects an unrepresentable resolved path before changing the context.
    pub fn set_schema(&self, name: &str) -> Result<()> {
        #[cfg(any(feature = "lpg", feature = "triple-store"))]
        let _operation = self.session_operation_guard();
        let mut next = self.graph_context_snapshot();
        next.schema = Some(name.to_owned());
        Self::context_graph_path(next.schema.as_deref(), None)?;
        if !next.native {
            next.storage_key =
                Self::context_graph_path(next.schema.as_deref(), next.graph.as_deref())?;
        }
        *self.current_context.lock() = next;
        self.track_graph_touch()
    }

    /// Returns the language schema selector, independently of a native graph path.
    #[must_use]
    pub fn current_schema(&self) -> Option<String> {
        self.current_context.lock().schema.clone()
    }

    /// Resolves the current schema to its canonical name and exact namespace
    /// incarnation. The short publication read makes catalog membership and
    /// the implicit default-graph token one coherent observation; callers keep
    /// the Arc for commit-time ABA detection after the guard is released.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn current_schema_incarnation(&self) -> Result<Option<SchemaIncarnation>> {
        let requested = self.current_schema();
        let _publication = self.publication_read_guard();
        self.schema_incarnation_for_name(requested.as_deref())
    }

    /// Resolves a schema name while the caller holds a publication barrier.
    /// `None` denotes the root namespace.
    #[cfg(feature = "lpg")]
    fn schema_incarnation_for_name(
        &self,
        requested: Option<&str>,
    ) -> Result<Option<SchemaIncarnation>> {
        use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};

        let Some(requested) = requested else {
            return Ok(None);
        };
        let canonical = self
            .catalog_view()
            .schema_names()
            .into_iter()
            .find(|registered| registered.eq_ignore_ascii_case(requested))
            .ok_or_else(|| {
                Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!("Schema '{requested}' does not exist"),
                ))
            })?;
        let default_key = format!("{canonical}/{SCHEMA_DEFAULT_GRAPH}");
        let default_path = Self::graph_path_for_storage_key(Some(&default_key))?;
        let default_graph = self.session_graph_path(&default_path).ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!("Schema '{canonical}' does not have a live default graph"),
            ))
        })?;
        Ok(Some(SchemaIncarnation {
            name: canonical,
            default_graph,
        }))
    }

    /// Resolves the namespace semantics of the parser-free LPG graph API.
    ///
    /// A registered prefix is canonicalized and tied to its exact schema
    /// incarnation. An unknown prefix remains a legal URI-shaped/root name,
    /// but commit pins the fact that the prefix is not concurrently claimed by
    /// CREATE SCHEMA.
    #[cfg(feature = "lpg")]
    fn resolve_parser_free_graph_name(
        &self,
        requested: &str,
    ) -> Result<(String, PendingGraphNamespace)> {
        if requested.is_empty() {
            return Err(grafeo_common::utils::error::Error::Query(
                grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Semantic,
                    "Graph name must not be empty",
                ),
            ));
        }
        if requested.eq_ignore_ascii_case("default") {
            return Err(grafeo_common::utils::error::Error::Query(
                grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Semantic,
                    "Graph name 'default' is reserved for the root default partition",
                ),
            ));
        }
        let Some((prefix, suffix)) = requested.split_once('/') else {
            return Ok((requested.to_string(), PendingGraphNamespace::Root));
        };

        let _publication = self.publication_read_guard();
        let canonical = self
            .catalog_view()
            .schema_names()
            .into_iter()
            .find(|registered| registered.eq_ignore_ascii_case(prefix));
        match canonical {
            Some(canonical) => {
                if suffix.is_empty() {
                    return Err(grafeo_common::utils::error::Error::Query(
                        grafeo_common::utils::error::QueryError::new(
                            grafeo_common::utils::error::QueryErrorKind::Semantic,
                            "Graph name must not be empty",
                        ),
                    ));
                }
                if suffix.eq_ignore_ascii_case("default") {
                    return Err(grafeo_common::utils::error::Error::Query(
                        grafeo_common::utils::error::QueryError::new(
                            grafeo_common::utils::error::QueryErrorKind::Semantic,
                            "Graph name 'default' is reserved for the schema default partition",
                        ),
                    ));
                }
                let owner = self
                    .schema_incarnation_for_name(Some(&canonical))?
                    .ok_or_else(|| {
                        grafeo_common::utils::error::Error::Internal(format!(
                            "canonical schema '{canonical}' lost its namespace incarnation"
                        ))
                    })?;
                let suffix = if suffix.eq_ignore_ascii_case(SCHEMA_DEFAULT_GRAPH) {
                    SCHEMA_DEFAULT_GRAPH
                } else {
                    suffix
                };
                Ok((
                    format!("{canonical}/{suffix}"),
                    PendingGraphNamespace::Schema(owner),
                ))
            }
            None => Ok((
                requested.to_string(),
                PendingGraphNamespace::UnregisteredPrefix(prefix.to_string()),
            )),
        }
    }

    /// Resolves constraint ownership from the physical path, never from an
    /// independently selected language schema. Only a single flat component
    /// can belong to the catalog's schema-qualified namespace.
    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn constraint_schema_in_names<'a>(path: &GraphPath, schemas: &'a [String]) -> Option<&'a str> {
        let [name] = path.components() else {
            return None;
        };
        let (prefix, _) = name.split_once('/')?;
        schemas
            .iter()
            .find(|schema| schema.eq_ignore_ascii_case(prefix))
            .map(String::as_str)
    }

    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn constraint_schema_for_path(&self, path: &GraphPath) -> Option<String> {
        let [name] = path.components() else {
            return None;
        };
        if !name.contains('/') {
            return None;
        }
        Self::constraint_schema_in_names(path, &self.catalog_view().schema_names())
            .map(str::to_owned)
    }

    /// Validates existing rows before publishing a new named constraint.
    /// Constraint creation is DDL, so paying for a complete scan is preferable
    /// to installing metadata that the current graph already violates.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn validate_named_constraint_existing_data(
        &self,
        definition: &crate::catalog::NamedConstraintDefinition,
        catalog: CatalogRead<'_>,
    ) -> Result<()> {
        self.validate_named_constraint_existing_data_at(
            definition,
            catalog,
            self.transaction_manager.current_epoch(),
            self.current_transaction_id(),
        )
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn validate_named_constraint_existing_data_at(
        &self,
        definition: &crate::catalog::NamedConstraintDefinition,
        catalog: CatalogRead<'_>,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Result<()> {
        use crate::catalog::NamedConstraintKind;
        use grafeo_common::types::PropertyKey;
        use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};

        let registered_schemas = catalog.schema_names();
        let (definition_schema, label) = definition.label.split_once('/').map_or(
            (None, definition.label.as_str()),
            |(prefix, local)| {
                registered_schemas
                    .iter()
                    .find(|schema| schema.eq_ignore_ascii_case(prefix))
                    .map_or((None, definition.label.as_str()), |schema| {
                        (Some(schema.as_str()), local)
                    })
            },
        );
        let keys: Vec<PropertyKey> = definition.properties.iter().map(PropertyKey::new).collect();
        let mut seen: Vec<Vec<Value>> = Vec::new();

        // Type constraints are schema-wide, not scoped to whichever graph the
        // DDL session happened to have selected. Scan every graph in the
        // definition's schema and keep one shared tuple set across them.
        let mut stores = Vec::new();
        for (path, graph) in self.catalog_graphs()? {
            let graph_schema = Self::constraint_schema_in_names(&path, &registered_schemas);
            let belongs_to_schema = match (definition_schema, graph_schema) {
                (None, None) => true,
                (Some(expected), Some(actual)) => actual.eq_ignore_ascii_case(expected),
                (None, Some(_)) | (Some(_), None) => false,
            };
            if belongs_to_schema {
                let source = if path.components().is_empty() {
                    Arc::clone(&self.graph_store)
                } else {
                    graph as Arc<dyn GraphStoreSearch>
                };
                stores.push((path, source));
            }
        }

        for (graph_name, store) in stores {
            for node in store.prepare_index_node_rows(epoch, transaction_id)? {
                if !node.labels.iter().any(|name| name.as_str() == label) {
                    continue;
                }
                let node_id = node.id;
                let values: Vec<Option<Value>> = keys
                    .iter()
                    .map(|key| node.properties.get(key).cloned())
                    .collect();
                let requires_all = matches!(
                    definition.kind,
                    NamedConstraintKind::NodeKey
                        | NamedConstraintKind::NotNull
                        | NamedConstraintKind::Exists
                );
                if requires_all
                    && values
                        .iter()
                        .any(|value| value.as_ref().is_none_or(|value| *value == Value::Null))
                {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!(
                            "Cannot create constraint '{}': existing node {node_id} in graph {graph_name:?} with label :{label} is missing a required property",
                            definition.name,
                        ),
                    )));
                }

                if matches!(
                    definition.kind,
                    NamedConstraintKind::Unique | NamedConstraintKind::NodeKey
                ) {
                    // SQL/GQL uniqueness permits absent/null values; NODE KEY was
                    // rejected above. A composite key is compared as a tuple.
                    let Some(tuple) = values.into_iter().collect::<Option<Vec<_>>>() else {
                        continue;
                    };
                    if tuple.contains(&Value::Null) {
                        continue;
                    }
                    if seen.iter().any(|existing| existing == &tuple) {
                        return Err(Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            format!(
                                "Cannot create constraint '{}': duplicate value tuple already exists for :{label}({}) across schema graphs",
                                definition.name,
                                definition.properties.join(", ")
                            ),
                        )));
                    }
                    seen.push(tuple);
                }
            }
        }
        Ok(())
    }

    /// Revalidates every constraint-sensitive entity while publication is
    /// serialized, immediately before the durable commit marker is prepared.
    ///
    /// Statement-time validation gives useful early errors, but cannot close
    /// the classic write-skew window where two transactions independently
    /// choose the same UNIQUE/NODE KEY tuple. The publication write lock makes
    /// this final check authoritative: the first committer becomes visible and
    /// the second observes it and is rolled back before `Committed` reaches WAL.
    #[cfg(feature = "lpg")]
    fn revalidate_touched_constraints(
        &self,
        transaction_id: TransactionId,
        touched_graphs: &[GraphPath],
    ) -> Result<()> {
        use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, TransactionError};

        let validation_epoch = self.transaction_manager.current_epoch();
        for graph_name in touched_graphs {
            let store = self.resolve_mutation_store(graph_name)?;
            let mut touched_nodes = store.pending_node_creates(transaction_id);
            let mut touched_edges = store.pending_edge_creates(transaction_id);
            let pending_node_deletes: FxHashSet<NodeId> = store
                .pending_node_deletes_peek(transaction_id)
                .into_iter()
                .collect();
            let pending_edge_deletes: FxHashSet<EdgeId> = store
                .pending_edge_deletes_peek(transaction_id)
                .into_iter()
                .collect();
            let (overlay_nodes, overlay_edges) = store.overlay_touched_entities(transaction_id);
            touched_nodes.extend(overlay_nodes);
            touched_nodes.extend(pending_node_deletes.iter().copied());
            touched_edges.extend(overlay_edges);
            touched_nodes.sort_unstable();
            touched_nodes.dedup();
            touched_edges.sort_unstable();
            touched_edges.dedup();
            if touched_nodes.is_empty() && touched_edges.is_empty() {
                continue;
            }

            let concrete_store = self.resolve_store(graph_name)?;
            let read_store: Arc<dyn GraphStore> = concrete_store.clone();
            let schema = self.constraint_schema_for_path(graph_name);
            let mut validator = CatalogConstraintValidator::new(self.catalog_view())
                .with_store(read_store)
                .with_schema(schema)
                .with_max_property_size(self.max_property_size);
            #[cfg(feature = "triple-store")]
            if graph_name.components().is_empty()
                && let Some(target) = self.rdf_projection_target.lock().clone()
            {
                validator = validator.with_rdf_projection_authority(
                    target.owner_marker,
                    target.node_label,
                    target.desired_iris,
                );
            }
            validator = validator
                .with_graph_path(graph_name.clone())
                .with_graph_type_binding_override(self.session_graph_type_binding(graph_name));

            for node_id in touched_nodes {
                if pending_node_deletes.contains(&node_id) {
                    // Re-read committed adjacency at the publication frontier,
                    // ignoring this transaction's own PENDING tombstones, then
                    // subtract the exact detach set it staged. If another
                    // transaction committed a new incident edge after statement
                    // execution, this delete plan is stale. Abort before the WAL
                    // commit marker; the competing edge publisher has already
                    // won and a retry will enumerate it into a fresh detach set.
                    let remaining = Self::incident_edge_ids_versioned(
                        store.as_ref(),
                        node_id,
                        validation_epoch,
                        TransactionId::INVALID,
                        self.resolve_store(graph_name)?.has_backward_adjacency(),
                    );
                    if let Some(edge_id) = remaining
                        .into_iter()
                        .find(|edge_id| !pending_edge_deletes.contains(edge_id))
                    {
                        return Err(Error::Transaction(TransactionError::WriteConflict(
                            format!(
                                "node {node_id} gained incident edge {edge_id} after DETACH enumeration"
                            ),
                        )));
                    }
                    validator
                        .validate_node_delete(node_id, validation_epoch, Some(transaction_id))
                        .map_err(|error| {
                            Error::Query(QueryError::new(
                                QueryErrorKind::Semantic,
                                format!("Commit rejected protected-node deletion: {error}"),
                            ))
                        })?;
                    continue;
                }
                if store
                    .get_node_versioned(node_id, validation_epoch, transaction_id)
                    .is_none()
                {
                    // A deleted node has no post-image to constrain.
                    continue;
                }
                let labels: Vec<String> = concrete_store
                    .read_node_labels_visible_for_validation(
                        node_id,
                        validation_epoch,
                        transaction_id,
                    )
                    .iter()
                    .map(|label| label.as_str().to_string())
                    .collect();
                let properties: Vec<(String, Value)> = concrete_store
                    .read_node_properties_visible_for_validation(
                        node_id,
                        validation_epoch,
                        transaction_id,
                    )
                    .into_iter()
                    .map(|(key, value)| (key.as_str().to_string(), value))
                    .collect();
                validator
                    .validate_node_labels_allowed(&labels)
                    .and_then(|()| {
                        validator.validate_node_post_image(
                            Some(node_id),
                            &labels,
                            &properties,
                            validation_epoch,
                            Some(transaction_id),
                        )
                    })
                    .map_err(|error| {
                        Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            format!("Commit rejected by schema constraint: {error}"),
                        ))
                    })?;
            }

            for edge_id in touched_edges {
                let Some(edge) =
                    store.get_edge_versioned(edge_id, validation_epoch, transaction_id)
                else {
                    continue;
                };
                for endpoint in [edge.src, edge.dst] {
                    if store
                        .get_node_versioned(endpoint, validation_epoch, transaction_id)
                        .is_none()
                    {
                        return Err(Error::Transaction(TransactionError::WriteConflict(
                            format!(
                                "edge {edge_id} endpoint {endpoint} was deleted before publication"
                            ),
                        )));
                    }
                }
                let properties: Vec<(String, Value)> = concrete_store
                    .read_edge_properties_visible_for_validation(
                        edge_id,
                        validation_epoch,
                        transaction_id,
                    )
                    .into_iter()
                    .map(|(key, value)| (key.as_str().to_string(), value))
                    .collect();
                let source_labels: Vec<String> = concrete_store
                    .read_node_labels_visible_for_validation(
                        edge.src,
                        validation_epoch,
                        transaction_id,
                    )
                    .into_iter()
                    .map(|label| label.as_str().to_string())
                    .collect();
                let target_labels: Vec<String> = concrete_store
                    .read_node_labels_visible_for_validation(
                        edge.dst,
                        validation_epoch,
                        transaction_id,
                    )
                    .into_iter()
                    .map(|label| label.as_str().to_string())
                    .collect();
                validator
                    .validate_edge_type_allowed(edge.edge_type.as_str())
                    .and_then(|()| {
                        validator.validate_edge_endpoints(
                            edge.edge_type.as_str(),
                            &source_labels,
                            &target_labels,
                        )
                    })
                    .and_then(|()| {
                        validator.validate_edge_post_image(edge.edge_type.as_str(), &properties)
                    })
                    .map_err(|error| {
                        Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            format!("Commit rejected by schema constraint: {error}"),
                        ))
                    })?;
            }
        }
        Ok(())
    }

    /// Revalidates the complete RDF→LPG target post-image at the authoritative
    /// publication boundary.
    ///
    /// The rebuild's earlier scan is only a plan. A concurrent writer may have
    /// committed after that scan, so success is legal only when the writing
    /// transaction's visible post-image contains exactly one row for every
    /// desired IRI, no stale/duplicate rows for this owner, and exactly the
    /// declared label plus the two reserved properties on every row. The caller
    /// holds the publication write lock.
    #[cfg(all(feature = "lpg", feature = "triple-store"))]
    fn revalidate_rdf_projection_target(&self, transaction_id: TransactionId) -> Result<()> {
        use grafeo_common::types::PropertyKey;
        use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};
        use grafeo_core::graph::rdf::{
            RDF_LPG_PROJECTION_IRI_PROPERTY, RDF_LPG_PROJECTION_OWNER_PROPERTY,
        };

        let Some(target) = self.rdf_projection_target.lock().clone() else {
            return Ok(());
        };
        let store = self.resolve_mutation_store(&GraphPath::root())?;
        let validation_epoch = self.transaction_manager.current_epoch();
        let owner_key = PropertyKey::new(RDF_LPG_PROJECTION_OWNER_PROPERTY);
        let iri_key = PropertyKey::new(RDF_LPG_PROJECTION_IRI_PROPERTY);
        let mut node_ids = store.node_ids();
        node_ids.extend(store.pending_node_creates(transaction_id));
        node_ids.sort_unstable();
        node_ids.dedup();

        let mut actual = std::collections::BTreeMap::<String, usize>::new();
        for node_id in node_ids {
            if store
                .get_node_versioned(node_id, validation_epoch, transaction_id)
                .is_none()
            {
                continue;
            }
            let properties =
                store.read_node_properties_visible(node_id, validation_epoch, Some(transaction_id));
            if properties.get(&owner_key).and_then(Value::as_str)
                != Some(target.owner_marker.as_str())
            {
                continue;
            }
            let Some(iri) = properties.get(&iri_key).and_then(Value::as_str) else {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!(
                        "RDF→LPG projection {} target contains owned node {node_id} without a string IRI",
                        target.projection_id
                    ),
                )));
            };
            if !target.desired_iris.contains(iri) {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!(
                        "RDF→LPG projection {} target contains stale owned IRI '{iri}'",
                        target.projection_id
                    ),
                )));
            }
            if properties.len() != 2 {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!(
                        "RDF→LPG projection {} target node for '{iri}' has properties outside the projection metadata plane",
                        target.projection_id
                    ),
                )));
            }
            let labels =
                store.read_node_labels_visible(node_id, validation_epoch, Some(transaction_id));
            if labels.len() != 1
                || !labels
                    .iter()
                    .any(|label| label.as_str() == target.node_label)
            {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!(
                        "RDF→LPG projection {} target node for '{iri}' does not have exactly label :{}",
                        target.projection_id, target.node_label
                    ),
                )));
            }
            *actual.entry(iri.to_string()).or_default() += 1;
        }

        for iri in target.desired_iris.iter() {
            match actual.get(iri).copied().unwrap_or(0) {
                1 => {}
                0 => {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!(
                            "RDF→LPG projection {} target is missing desired IRI '{iri}'",
                            target.projection_id
                        ),
                    )));
                }
                count => {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!(
                            "RDF→LPG projection {} target has {count} owned rows for IRI '{iri}'",
                            target.projection_id
                        ),
                    )));
                }
            }
        }
        Ok(())
    }

    /// Returns the validated resolved coordinate from one context acquisition.
    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn active_graph_storage_key(&self) -> GraphPath {
        self.current_graph_path()
    }

    pub(crate) fn graph_context_snapshot(&self) -> SessionGraphContext {
        self.current_context.lock().clone()
    }

    /// Seeds a fresh Session from an already-validated database context.
    pub(crate) fn inherit_graph_context(&self, context: SessionGraphContext) {
        *self.current_context.lock() = context;
    }

    pub(crate) fn context_graph_path(
        schema: Option<&str>,
        graph: Option<&str>,
    ) -> Result<GraphPath> {
        if let Some(schema) = schema {
            GraphPath::from_components(&[&format!("{schema}/{SCHEMA_DEFAULT_GRAPH}")]).map_err(
                |error| grafeo_common::utils::error::Error::InvalidValue(error.to_string()),
            )?;
        }
        match Self::storage_key_for_context(schema, graph) {
            None => Ok(GraphPath::root()),
            Some(name) => GraphPath::from_components(&[&name]).map_err(|error| {
                grafeo_common::utils::error::Error::InvalidValue(error.to_string())
            }),
        }
    }

    pub(crate) fn storage_key_for_context(
        schema: Option<&str>,
        graph: Option<&str>,
    ) -> Option<String> {
        match (schema, graph) {
            (None, None) => None,
            (Some(s), None) => Some(format!("{s}/{SCHEMA_DEFAULT_GRAPH}")),
            (None, Some(name)) if name.eq_ignore_ascii_case("default") => None,
            (Some(s), Some(name)) if name.eq_ignore_ascii_case("default") => {
                Some(format!("{s}/{SCHEMA_DEFAULT_GRAPH}"))
            }
            (_, Some(name)) if name.contains('/') => Some(name.to_string()),
            (None, Some(name)) => Some(name.to_string()),
            (Some(s), Some(g)) => Some(format!("{s}/{g}")),
        }
    }

    /// Clears only the selector still addressing this exact flat lifecycle target.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn reset_graph_context_if_storage_key(&self, expected: &str) -> Result<()> {
        let _operation = self.session_operation_guard();
        let next = {
            let context = self.current_context.lock();
            let matches = matches!(context.storage_key.components(), [name] if name == expected);
            matches.then(|| context.clone())
        };
        if let Some(mut next) = next {
            next.graph = None;
            next.native = false;
            next.storage_key = Self::context_graph_path(next.schema.as_deref(), None)?;
            *self.current_context.lock() = next;
        }
        Ok(())
    }

    /// Resolves a literal path through staged lifecycle and retained observations.
    #[cfg(feature = "lpg")]
    fn live_graph_path(&self, path: &GraphPath) -> Option<Arc<LpgStore>> {
        let mut target = Arc::clone(&self.store);
        for name in path.components() {
            target = target.graph(name)?;
        }
        Some(target)
    }

    #[cfg(feature = "lpg")]
    fn session_graph_path(&self, path: &GraphPath) -> Option<Arc<LpgStore>> {
        let mut target = Arc::clone(&self.store);
        let mut prefix = GraphPath::root();
        let mut private_root = false;
        for name in path.components() {
            prefix = prefix.child(name).ok()?;
            if let Some(pending) = self.pending_created_graphs.lock().get(&prefix).cloned() {
                if !Arc::ptr_eq(&target, &pending.parent) {
                    return None;
                }
                target = pending.store;
                private_root = true;
                continue;
            }
            if self.cancelled_created_graphs.lock().contains_key(&prefix)
                || self.pending_dropped_graphs.lock().contains_key(&prefix)
            {
                return None;
            }
            if !private_root {
                if let Some(retained) = self.touched_named_graphs.lock().get(&prefix).cloned() {
                    target = retained;
                    continue;
                }
                if self.missing_named_graphs.lock().contains(&prefix) {
                    return None;
                }
            }
            target = target.graph(name)?;
        }
        Some(target)
    }

    /// Language/schema names are already resolved single child names.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn session_named_graph(&self, key: &str) -> Option<Arc<LpgStore>> {
        let path = Self::graph_path_for_storage_key(Some(key)).ok()?;
        self.session_graph_path(&path)
    }

    /// Returns an isolated empty LPG store for a missing named-graph view.
    ///
    /// Several internal read helpers predate fallible graph resolution and
    /// return trait objects directly. Their safe failure mode is an empty,
    /// detached store — never the default graph. Public query/session commands
    /// still report a structured semantic error when selecting a missing graph.
    #[cfg(feature = "lpg")]
    fn empty_lpg_view(&self) -> Arc<LpgStore> {
        let store = Arc::new(LpgStore::new().expect("allocate empty missing-graph view"));
        let _ = store.seal_unframed_writes(self.transaction_manager.write_authority());
        store
    }

    /// Resolves only a retained transaction incarnation, never a live replacement.
    #[cfg(feature = "lpg")]
    fn pinned_graph_path(&self, path: &GraphPath) -> Option<Arc<LpgStore>> {
        if path.components().is_empty() {
            return Some(Arc::clone(&self.store));
        }
        self.touched_named_graphs.lock().get(path).cloned()
    }

    /// Named graph is present and not pending-drop in this transaction.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn graph_visible(&self, key: &str) -> bool {
        self.session_named_graph(key).is_some()
    }

    /// Resolves one writable target from the exact captured coordinate.
    #[cfg(feature = "lpg")]
    fn require_lpg_store_for_storage_key(&self, path: &GraphPath) -> Result<Arc<LpgStore>> {
        self.require_index_path_write_grant(path)?;
        self.session_graph_path(path)
            .ok_or_else(|| Self::index_ddl_error(format!("Graph {path:?} does not exist")))
    }

    /// Returns the graph store for the currently active graph.
    ///
    /// If `current_graph` is `None` or `"default"`, returns the session's
    /// default `graph_store` (already WAL-wrapped for the default graph).
    /// Otherwise looks up the named graph in the root store and wraps it
    /// in a [`WalGraphStore`] so mutations are WAL-logged with the correct
    /// graph context.
    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn active_store(&self) -> Arc<dyn GraphStoreSearch> {
        self.active_store_with_graph_context().1
    }

    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn active_store_with_graph_context(&self) -> (SessionGraphContext, Arc<dyn GraphStoreSearch>) {
        let context = self.graph_context_snapshot();
        let store = self.store_for_graph_storage_key(&context.storage_key);
        (context, store)
    }

    /// Resolves a read store from an already-captured session storage key.
    ///
    /// Physical planning retains this same key for cache insertion so a
    /// concurrent context change cannot bind one graph's operators under
    /// another graph's cache entry.
    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn store_for_graph_storage_key(&self, path: &GraphPath) -> Arc<dyn GraphStoreSearch> {
        #[cfg(feature = "lpg")]
        {
            if !self.index_path_grant_allows(path, crate::auth::Role::ReadOnly) {
                return self.empty_lpg_view() as Arc<dyn GraphStoreSearch>;
            }
            if !path.components().is_empty() {
                return self.session_graph_path(path).map_or_else(
                    || self.empty_lpg_view() as Arc<dyn GraphStoreSearch>,
                    |store| store as Arc<dyn GraphStoreSearch>,
                );
            }
        }
        #[cfg(not(feature = "lpg"))]
        let _ = path;
        Arc::clone(&self.graph_store)
    }

    /// Returns the writable store for the active graph, if available.
    ///
    /// Returns `None` for read-only databases. For named graphs, wraps
    /// the store with WAL logging when durability is enabled.
    #[cfg(any(
        feature = "lpg",
        feature = "cypher",
        feature = "gremlin",
        feature = "sql-pgq"
    ))]
    fn active_write_store(&self) -> Option<Arc<dyn GraphStoreMut>> {
        let key = self.active_graph_storage_key();
        self.write_store_for_graph_storage_key(&key)
    }

    /// Resolves a write store from an already-captured session storage key.
    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn write_store_for_graph_storage_key(
        &self,
        path: &GraphPath,
    ) -> Option<Arc<dyn GraphStoreMut>> {
        #[cfg(feature = "lpg")]
        {
            if !self.index_path_grant_allows(path, crate::auth::Role::ReadWrite) {
                return None;
            }
            if !path.components().is_empty() {
                return self
                    .resolve_named_lpg_writer(path)
                    .map(ResolvedLpgWriter::into_store);
            }
        }
        #[cfg(not(feature = "lpg"))]
        let _ = path;
        self.graph_store_mut.as_ref().map(Arc::clone)
    }

    /// Builds every decorator from one captured named-graph incarnation.
    #[cfg(feature = "lpg")]
    fn resolve_named_lpg_writer(&self, path: &GraphPath) -> Option<ResolvedLpgWriter> {
        let named_store = self.session_graph_path(path)?;
        #[cfg(feature = "cdc")]
        let graph_incarnation = Arc::clone(&named_store);
        let store: Arc<dyn GraphStoreMut> = named_store;
        #[cfg(feature = "wal")]
        let store: Arc<dyn GraphStoreMut> = if let Some(wal) = &self.wal {
            Arc::new(
                crate::database::wal_store::WalGraphStore::new(
                    store,
                    Arc::clone(wal),
                    path.clone(),
                )
                .with_poison(Arc::clone(&self.durability_poisoned)),
            )
        } else {
            store
        };
        #[cfg(feature = "cdc")]
        if let Some(ref pending) = self.cdc_pending_events {
            return Some(ResolvedLpgWriter::Cdc(Arc::new(
                crate::database::cdc_store::CdcGraphStore::wrap(
                    store,
                    Arc::clone(pending),
                    graph_incarnation,
                    path.clone(),
                ),
            )));
        }
        Some(ResolvedLpgWriter::Plain(store))
    }

    /// Resolves the compound writer without erasing its CDC construction seam.
    #[cfg(feature = "lpg")]
    fn compound_write_store_for_graph_storage_key(
        &self,
        storage_key: &GraphPath,
    ) -> Result<ResolvedLpgWriter> {
        use grafeo_common::utils::error::Error;
        self.require_index_path_write_grant(storage_key)?;
        if !storage_key.components().is_empty() {
            return self.resolve_named_lpg_writer(storage_key).ok_or_else(|| {
                Error::Internal(format!("no admitted writer for graph {storage_key:?}"))
            });
        }
        let writer = self
            .graph_store_mut
            .as_ref()
            .ok_or_else(|| Error::Internal("no admitted default graph writer".to_string()))?;
        #[cfg(feature = "cdc")]
        match (&self.default_cdc_writer, &self.cdc_pending_events) {
            (Some(cached), Some(pending)) => {
                let erased: Arc<dyn GraphStoreMut> = Arc::clone(cached) as Arc<dyn GraphStoreMut>;
                if !Arc::ptr_eq(&erased, writer) || !Arc::ptr_eq(&cached.pending_events(), pending)
                {
                    return Err(Error::Internal(
                        "default compound CDC writer does not match the admitted writer"
                            .to_string(),
                    ));
                }
                return Ok(ResolvedLpgWriter::Cdc(Arc::clone(cached)));
            }
            (None, None) => {}
            _ => {
                return Err(Error::Internal(
                    "default compound CDC writer is missing or inconsistent".to_string(),
                ));
            }
        }
        Ok(ResolvedLpgWriter::Plain(Arc::clone(writer)))
    }

    /// Returns the concrete `LpgStore` for the currently active graph.
    ///
    /// Used by direct CRUD methods that need the concrete store type
    /// for versioned operations.
    #[cfg(feature = "lpg")]
    fn active_lpg_store(&self) -> Arc<LpgStore> {
        let path = self.current_graph_path();
        if !self.index_path_grant_allows(&path, crate::auth::Role::ReadOnly) {
            return self.empty_lpg_view();
        }
        self.session_graph_path(&path)
            .unwrap_or_else(|| self.empty_lpg_view())
    }

    /// Returns the tier-merged **read** view for the active graph.
    ///
    /// For the default graph this is `graph_store` — after
    /// [`compact()`](crate::GrafeoDB::compact) the `LayeredStore` (cold base +
    /// overlay), so base-resident entities stay visible to the direct lookup
    /// APIs. For a named graph it is that graph's own `LpgStore` (named graphs
    /// are never folded into the cold base). Mirrors
    /// [`active_lpg_store`](Self::active_lpg_store) / [`active_write_store`](Self::active_write_store).
    #[cfg(feature = "lpg")]
    fn active_read_store(&self) -> Arc<dyn GraphStoreSearch> {
        let path = self.current_graph_path();
        if !self.index_path_grant_allows(&path, crate::auth::Role::ReadOnly) {
            return self.empty_lpg_view() as Arc<dyn GraphStoreSearch>;
        }
        if path.components().is_empty() {
            return Arc::clone(&self.graph_store);
        }
        self.session_graph_path(&path).map_or_else(
            || self.empty_lpg_view() as Arc<dyn GraphStoreSearch>,
            |store| store as Arc<dyn GraphStoreSearch>,
        )
    }

    /// Builds the same catalog/protected-row validator used by planned LPG
    /// mutations for the direct Session CRUD surface.
    #[cfg(feature = "lpg")]
    fn direct_mutation_validator(&self) -> CatalogConstraintValidator {
        let path = self.current_graph_path();
        let store: Arc<dyn GraphStore> = self.active_read_store();
        let mut validator = CatalogConstraintValidator::new(self.catalog_view())
            .with_store(store)
            .with_max_property_size(self.max_property_size)
            .with_schema(self.constraint_schema_for_path(&path));
        #[cfg(all(feature = "lpg", feature = "triple-store"))]
        if path.components().is_empty()
            && let Some(target) = self.rdf_projection_target.lock().clone()
        {
            validator = validator.with_rdf_projection_authority(
                target.owner_marker,
                target.node_label,
                target.desired_iris,
            );
        }
        let binding = self.session_graph_type_binding(&path);
        validator = validator
            .with_graph_path(path)
            .with_graph_type_binding_override(binding);
        validator
    }

    /// Returns a complete transaction-visible node image for direct mutation
    /// validation.
    #[cfg(feature = "lpg")]
    fn direct_node_image(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Option<(Vec<String>, Vec<(String, Value)>)> {
        let store = self.active_read_store();
        store.get_node_versioned(id, epoch, transaction_id.unwrap_or(TransactionId::SYSTEM))?;
        let labels = store
            .read_node_labels_visible(id, epoch, transaction_id)
            .iter()
            .map(|label| label.as_str().to_string())
            .collect();
        let properties = store
            .read_node_properties_visible(id, epoch, transaction_id)
            .into_iter()
            .map(|(key, value)| (key.as_str().to_string(), value))
            .collect();
        Some((labels, properties))
    }

    /// Enumerates every transaction-visible incident edge, falling back to a
    /// complete outgoing structural scan when the concrete LPG store was
    /// configured without reverse adjacency.
    ///
    /// `Direction::Both` is O(degree) in the normal indexed topology. With
    /// `Config::without_backward_edges`, however, it intentionally yields only
    /// outgoing rows. A DETACH operation cannot inherit that query-performance
    /// tradeoff as a correctness hole, so the rare no-backward topology scans
    /// every visible source and this transaction's pending creates.
    #[cfg(feature = "lpg")]
    fn incident_edge_ids_versioned(
        store: &dyn GraphStore,
        node: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
        has_backward_adjacency: bool,
    ) -> Vec<EdgeId> {
        let mut incident: Vec<EdgeId> = store
            .edges_from_versioned(node, Direction::Both, epoch, transaction_id)
            .into_iter()
            .map(|(_, edge_id)| edge_id)
            .collect();
        if !has_backward_adjacency {
            let mut sources = store.node_ids();
            sources.extend(store.pending_node_creates(transaction_id));
            sources.sort_unstable();
            sources.dedup();
            for source in sources {
                incident.extend(
                    store
                        .edges_from_versioned(source, Direction::Outgoing, epoch, transaction_id)
                        .into_iter()
                        .filter_map(|(_, edge_id)| {
                            store
                                .get_edge_versioned(edge_id, epoch, transaction_id)
                                .is_some_and(|edge| edge.dst == node)
                                .then_some(edge_id)
                        }),
                );
            }
            // A newly-created edge can be visible to its owner even when its
            // newly-created source is not part of committed node enumeration.
            incident.extend(
                store
                    .pending_edge_creates(transaction_id)
                    .into_iter()
                    .filter(|edge_id| {
                        store
                            .get_edge_versioned(*edge_id, epoch, transaction_id)
                            .is_some_and(|edge| edge.src == node || edge.dst == node)
                    }),
            );
        }
        incident.sort_unstable();
        incident.dedup();
        incident
    }

    /// Applies a mutation through the active **write** store, which after
    /// `compact()` is the `LayeredStore` — so a base-resident entity is promoted
    /// into the overlay (or base-tombstoned on delete) before the write lands,
    /// keeping the direct mutation APIs correct on cold data. Falls back to the
    /// concrete overlay only when there is no writable trait store (e.g. a
    /// read-only/placeholder backend), preserving the prior behaviour.
    #[cfg(feature = "lpg")]
    fn with_write_store<R>(&self, f: impl FnOnce(&dyn GraphStoreMut) -> R) -> R {
        match self.active_write_store() {
            Some(write) => f(&*write),
            None => f(&*self.active_lpg_store()),
        }
    }

    /// Resolves the exact concrete incarnation retained by transaction tracking.
    #[cfg(feature = "lpg")]
    fn resolve_store(&self, path: &GraphPath) -> Result<Arc<LpgStore>> {
        self.pinned_graph_path(path)
            .ok_or_else(|| Self::index_ddl_error(format!("retained graph {path:?} is missing")))
    }

    /// Resolves the exact mutation view, preserving the root LayeredStore.
    #[cfg(feature = "lpg")]
    fn resolve_mutation_store(&self, path: &GraphPath) -> Result<Arc<dyn GraphStoreMut>> {
        if path.components().is_empty() {
            return Ok(self.graph_store_mut.as_ref().map_or_else(
                || Arc::clone(&self.store) as Arc<dyn GraphStoreMut>,
                Arc::clone,
            ));
        }
        self.resolve_store(path)
            .map(|store| store as Arc<dyn GraphStoreMut>)
    }

    /// Records the selected path while holding the current transaction stable.
    fn track_graph_touch(&self) -> Result<()> {
        let current = self.current_transaction.lock();
        let Some(transaction_id) = *current else {
            return Ok(());
        };
        let key = self.current_graph_path();
        #[cfg(feature = "lpg")]
        {
            let target = self.session_graph_path(&key);
            self.track_lpg_graph_coordinate(transaction_id, key, target)
        }
        #[cfg(not(feature = "lpg"))]
        {
            let _ = transaction_id;
            let mut touched = self.touched_graphs.lock();
            if !touched.contains(&key) {
                touched.push(key);
            }
            Ok(())
        }
    }

    /// Ensures the current mutation target is admitted and exactly retained.
    #[cfg(feature = "lpg")]
    fn ensure_current_mutation_touch(&self) -> Result<()> {
        let current = self.current_transaction.lock();
        let transaction_id = current
            .ok_or_else(|| Self::index_ddl_error("LPG mutation escaped transaction framing"))?;
        let key = self.current_graph_path();
        let target = self.session_graph_path(&key);
        self.track_lpg_graph_coordinate(transaction_id, key.clone(), target.clone())?;
        if target.is_none() {
            return Err(Self::index_ddl_error(format!(
                "Graph {key:?} does not exist"
            )));
        }
        Ok(())
    }

    /// Retains target and ancestor observations before registering SSI bridges.
    #[cfg(feature = "lpg")]
    fn track_lpg_graph_coordinate(
        &self,
        transaction_id: TransactionId,
        key: GraphPath,
        named: Option<Arc<LpgStore>>,
    ) -> Result<()> {
        // The Session root cannot be replaced and has no ancestor witnesses.
        // Repeated root writes need neither path-map work nor a clone of every
        // previously touched coordinate; its SSI bridges are already installed.
        if key.components().is_empty() && self.touched_graphs.lock().contains(&key) {
            return Ok(());
        }
        let read_committed = self.transaction_manager.isolation_level(transaction_id)
            == Some(crate::transaction::IsolationLevel::ReadCommitted);
        let owns_pending = self
            .pending_created_graphs
            .lock()
            .keys()
            .any(|prefix| key.components().starts_with(prefix.components()));
        let mut observations = Vec::new();
        let mut prefix = GraphPath::root();
        for name in key.components() {
            prefix = prefix
                .child(name)
                .map_err(|error| Self::index_ddl_error(error.to_string()))?;
            let observed = if prefix == key {
                named.clone()
            } else {
                self.session_graph_path(&prefix)
            };
            observations.push((prefix.clone(), observed));
        }
        let touched_before = self.touched_graphs.lock().clone();
        let mut superseded = Vec::new();
        let mut tracker_named = named.clone();
        let mut changed = false;
        {
            let mut pinned = self.touched_named_graphs.lock();
            let mut missing = self.missing_named_graphs.lock();
            for (path, observed) in observations {
                match observed {
                    Some(graph) if !missing.contains(&path) || read_committed || owns_pending => {
                        changed |= missing.remove(&path);
                        if !pinned
                            .get(&path)
                            .is_some_and(|old| Arc::ptr_eq(old, &graph))
                        {
                            if let Some(old) = pinned.insert(path.clone(), graph)
                                && touched_before.contains(&path)
                            {
                                superseded.push((path, old));
                            }
                            changed = true;
                        }
                    }
                    Some(_) => {
                        tracker_named = None;
                    }
                    None if !read_committed && !pinned.contains_key(&path) => {
                        changed |= missing.insert(path);
                    }
                    None => {}
                }
            }
        }
        if !superseded.is_empty() {
            self.superseded_graph_touches.lock().extend(superseded);
        }
        // Missing coordinates retain an absence expectation, not a fake data target.
        let target_present = key.components().is_empty() || tracker_named.is_some();
        if !target_present {
            return Ok(());
        }
        let newly_touched = {
            let mut touched = self.touched_graphs.lock();
            if touched.contains(&key) {
                false
            } else {
                touched.push(key.clone());
                true
            }
        };
        if !newly_touched && !changed {
            return Ok(());
        }

        // A detached root and its descendants are transaction-private. Preserve
        // the existing rule excluding them from the unqualified SSI entity space.
        let tracker_store: Option<Arc<dyn GraphStoreSearch>> = if owns_pending {
            None
        } else if key.components().is_empty() {
            Some(Arc::clone(&self.graph_store))
        } else {
            tracker_named.map(|store| store as Arc<dyn GraphStoreSearch>)
        };
        if self.transaction_manager.isolation_level(transaction_id)
            == Some(crate::transaction::IsolationLevel::Serializable)
            && let Some(store) = tracker_store
        {
            let granularity = *self.conflict_granularity.lock();
            let read_bridge = Arc::new(
                crate::transaction::TransactionReadTracker::with_granularity(
                    Arc::clone(&self.transaction_manager),
                    granularity,
                ),
            );
            let write_bridge = Arc::new(
                crate::transaction::TransactionWriteTracker::with_granularity(
                    Arc::clone(&self.transaction_manager),
                    granularity,
                ),
            );
            self.transaction_manager.with_write_authority(|| {
                store.register_read_tracker(transaction_id, read_bridge);
                store.register_write_tracker(transaction_id, write_bridge);
            });
        }
        Ok(())
    }

    /// Tracks one already-resolved graph incarnation without changing the
    /// Session's active graph/schema context. `None` denotes the root default
    /// partition; named coordinates retain their exact store Arc.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn track_graph_incarnation_touch(
        &self,
        key: Option<&str>,
        graph: &Arc<LpgStore>,
    ) -> Result<()> {
        let current = self.current_transaction.lock();
        let Some(transaction_id) = *current else {
            return Ok(());
        };
        self.track_lpg_graph_coordinate(
            transaction_id,
            Self::graph_path_for_storage_key(key)?,
            Some(Arc::clone(graph)),
        )
    }

    /// Records an already-resolved flat language lifecycle target.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn track_named_graph_incarnation_touch(&self, key: &str, graph: &Arc<LpgStore>) -> Result<()> {
        self.track_graph_incarnation_touch(Some(key), graph)
    }

    /// Sets the session time zone.
    pub fn set_time_zone(&self, tz: &str) {
        *self.time_zone.lock() = Some(tz.to_string());
    }

    /// Returns the session time zone, if set.
    #[must_use]
    pub fn time_zone(&self) -> Option<String> {
        self.time_zone.lock().clone()
    }

    /// Sets a session parameter.
    pub fn set_parameter(&self, key: &str, value: grafeo_common::types::Value) {
        self.session_params.lock().insert(key.to_string(), value);
    }

    /// Gets a session parameter by cloning it.
    #[must_use]
    pub fn get_parameter(&self, key: &str) -> Option<grafeo_common::types::Value> {
        self.session_params.lock().get(key).cloned()
    }

    /// Resets language and native graph selectors and other session settings.
    ///
    /// # Errors
    /// Returns a transaction tracking failure.
    pub fn reset_session(&self) -> Result<()> {
        #[cfg(any(feature = "lpg", feature = "triple-store"))]
        let _operation = self.session_operation_guard();
        let _historical = self.historical_view_operation_gate.lock();
        *self.current_context.lock() = SessionGraphContext::default();
        *self.time_zone.lock() = None;
        self.session_params.lock().clear();
        *self.viewing_epoch_override.lock() = None;
        self.track_graph_touch()
    }

    /// Resets the language schema without retargeting an explicit native path.
    ///
    /// # Errors
    /// Rejects invalid path resolution or transaction tracking.
    pub fn reset_schema(&self) -> Result<()> {
        #[cfg(any(feature = "lpg", feature = "triple-store"))]
        let _operation = self.session_operation_guard();
        let mut next = self.graph_context_snapshot();
        next.schema = None;
        if !next.native {
            next.storage_key = Self::context_graph_path(None, next.graph.as_deref())?;
        }
        *self.current_context.lock() = next;
        self.track_graph_touch()
    }

    /// Returns graph selection to the language schema's implicit default.
    ///
    /// # Errors
    /// Rejects invalid path resolution or transaction tracking.
    pub fn reset_graph(&self) -> Result<()> {
        #[cfg(any(feature = "lpg", feature = "triple-store"))]
        let _operation = self.session_operation_guard();
        let mut next = self.graph_context_snapshot();
        next.graph = None;
        next.native = false;
        next.storage_key = Self::context_graph_path(next.schema.as_deref(), None)?;
        *self.current_context.lock() = next;
        self.track_graph_touch()
    }

    /// Resets only the session time zone (Section 7.2 GR3).
    pub fn reset_time_zone(&self) {
        *self.time_zone.lock() = None;
    }

    /// Resets only session parameters (Section 7.2 GR4).
    pub fn reset_parameters(&self) {
        self.session_params.lock().clear();
    }

    // --- Time-travel API ---

    /// Sets a viewing epoch override for time-travel queries.
    ///
    /// While set, LPG reads outside a transaction see the database as it
    /// existed at the given epoch. Use
    /// [`clear_viewing_epoch`](Self::clear_viewing_epoch) to return to normal
    /// behavior. LPG mutations are rejected while the override is set. A
    /// transaction retains its fixed BEGIN snapshot; the override takes effect
    /// for LPG reads after that transaction ends. RDF valid/system-time
    /// selection is a separate contract.
    pub fn set_viewing_epoch(&self, epoch: EpochId) {
        #[cfg(any(feature = "lpg", feature = "triple-store"))]
        let _operation = self.session_operation_guard();
        let _historical = self.historical_view_operation_gate.lock();
        *self.viewing_epoch_override.lock() = Some(epoch);
    }

    /// Clears the viewing epoch override, returning to normal behavior.
    pub fn clear_viewing_epoch(&self) {
        #[cfg(any(feature = "lpg", feature = "triple-store"))]
        let _operation = self.session_operation_guard();
        let _historical = self.historical_view_operation_gate.lock();
        *self.viewing_epoch_override.lock() = None;
    }

    /// Returns the current viewing epoch override, if any.
    #[must_use]
    pub fn viewing_epoch(&self) -> Option<EpochId> {
        *self.viewing_epoch_override.lock()
    }

    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn session_scope_key(&self) -> usize {
        std::ptr::from_ref(self).addr()
    }

    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn scoped_viewing_epoch(&self) -> Option<EpochId> {
        let session_key = self.session_scope_key();
        SCOPED_VIEWING_EPOCHS.with(|scopes| {
            scopes
                .borrow()
                .iter()
                .rev()
                .find_map(|(key, epoch)| (*key == session_key).then_some(*epoch))
        })
    }

    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn effective_viewing_epoch(&self) -> Option<EpochId> {
        self.scoped_viewing_epoch()
            .or_else(|| *self.viewing_epoch_override.lock())
    }

    #[cfg(feature = "gql")]
    fn with_scoped_viewing_epoch<T>(&self, epoch: EpochId, body: impl FnOnce() -> T) -> T {
        let session_key = self.session_scope_key();
        SCOPED_VIEWING_EPOCHS.with(|scopes| scopes.borrow_mut().push((session_key, epoch)));
        let _scope = ScopedViewingEpochGuard { session_key };
        body()
    }

    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn reject_lpg_historical_mutation(&self) -> Result<()> {
        if self.effective_viewing_epoch().is_none() {
            return Ok(());
        }
        Err(grafeo_common::utils::error::Error::Transaction(
            grafeo_common::utils::error::TransactionError::InvalidState(
                "historical view is read-only; clear the viewing epoch before mutating LPG state"
                    .to_string(),
            ),
        ))
    }

    #[cfg(feature = "gql")]
    fn reject_scoped_historical_command(&self) -> Result<()> {
        if self.scoped_viewing_epoch().is_none() {
            return Ok(());
        }
        Err(grafeo_common::utils::error::Error::Query(
            grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Semantic,
                "historical execution accepts read queries only; session and schema commands are not historical",
            ),
        ))
    }

    /// Returns all versions of a node with their creation/deletion epochs.
    ///
    /// Properties and labels reflect the current state (not versioned per-epoch).
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        let _operation = self.session_operation_guard();
        self.active_lpg_store().get_node_history(id)
    }

    /// Returns all versions of an edge with their creation/deletion epochs.
    ///
    /// Properties reflect the current state (not versioned per-epoch).
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        let _operation = self.session_operation_guard();
        self.active_lpg_store().get_edge_history(id)
    }

    /// Checks that the session's graph model supports LPG operations.
    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn require_lpg(&self, language: &str) -> Result<()> {
        if self.graph_model == GraphModel::Rdf {
            return Err(grafeo_common::utils::error::Error::Internal(format!(
                "This is an RDF database. {language} queries require an LPG database."
            )));
        }
        Ok(())
    }

    /// Checks that the session's graph model supports RDF operations.
    #[cfg(feature = "triple-store")]
    fn require_rdf(&self, what: &str) -> Result<()> {
        if self.graph_model == GraphModel::Lpg {
            return Err(grafeo_common::utils::error::Error::Internal(format!(
                "This is an LPG database. {what} requires an RDF or Both database."
            )));
        }
        Ok(())
    }

    /// Checks that the session's identity is permitted to execute the given
    /// statement kind. Returns an error if the role is insufficient.
    ///
    /// Short-circuits for Admin identities (the common case for embedded use
    /// and benchmarks) to avoid any overhead on the hot path.
    #[inline]
    fn require_permission(&self, kind: crate::auth::StatementKind) -> Result<()> {
        // Fast path: Admin can do everything
        if self.identity.can_admin() {
            return Ok(());
        }
        crate::auth::check_permission(&self.identity, kind).map_err(|denied| {
            grafeo_common::utils::error::Error::Query(grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Semantic,
                denied.to_string(),
            ))
        })
    }

    /// Tests an exact graph storage coordinate against the identity's grants.
    ///
    /// Schema-local aliases are deliberately not accepted here: allowing a
    /// grant for `g` to follow `SESSION SET SCHEMA` would let one capability
    /// roam across every `*/g` namespace. The root default graph is the sole
    /// coordinate represented without a storage key.
    #[cfg(feature = "gql")]
    fn graph_grant_allows(&self, graph_name: Option<&str>, required: crate::auth::Role) -> bool {
        let path = match graph_name {
            None => GraphPath::root(),
            Some(name) => {
                let Ok(path) = GraphPath::from_components(&[name]) else {
                    return false;
                };
                path
            }
        };
        self.index_path_grant_allows(&path, required)
    }

    /// Enforces a capability for one exact RDF dataset graph.
    ///
    /// RDF grants deliberately bypass the LPG graph-grant matcher: named graph
    /// IRIs are case-sensitive, and `None` is the default graph rather than a
    /// named graph whose IRI happens to be `default`.
    #[cfg(any(feature = "triple-store", feature = "cdc"))]
    fn require_rdf_graph_grant(
        &self,
        graph: Option<&str>,
        required: crate::auth::Role,
        access: &str,
    ) -> Result<()> {
        if self.identity.can_access_rdf_graph(graph, required) {
            return Ok(());
        }

        let graph = match graph {
            Some(iri) => format!("named RDF graph '{iri}'"),
            None => "default RDF graph".to_owned(),
        };
        Err(grafeo_common::utils::error::Error::Query(
            grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Semantic,
                format!(
                    "permission denied: no {access} grant for {graph} (user: {})",
                    self.identity.user_id()
                ),
            ),
        ))
    }

    #[cfg(any(
        feature = "lpg",
        feature = "cdc",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn index_path_grant_allows(&self, path: &GraphPath, required: crate::auth::Role) -> bool {
        let role_allows = match required {
            crate::auth::Role::ReadOnly => self.identity.can_read(),
            crate::auth::Role::ReadWrite => self.identity.can_write(),
            crate::auth::Role::Admin => self.identity.can_admin(),
        };
        role_allows
            && (!self.identity.has_grants() || self.identity.can_access_graph(path, required))
    }

    #[cfg(any(
        feature = "lpg",
        feature = "cdc",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn require_graph_path_grant(&self, path: &GraphPath, role: crate::auth::Role) -> Result<()> {
        if self.index_path_grant_allows(path, role) {
            return Ok(());
        }
        Err(grafeo_common::utils::error::Error::Query(
            grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Semantic,
                format!(
                    "permission denied for graph {path:?} (user: {})",
                    self.identity.user_id()
                ),
            ),
        ))
    }

    #[cfg(feature = "gql")]
    fn require_graph_grant(
        &self,
        graph_name: Option<&str>,
        required: crate::auth::Role,
        access: &str,
    ) -> Result<()> {
        if self.graph_grant_allows(graph_name, required) {
            return Ok(());
        }
        let graph_name = graph_name.unwrap_or("default");
        Err(grafeo_common::utils::error::Error::Query(
            grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Semantic,
                format!(
                    "permission denied: no {access} grant for graph '{graph_name}' (user: {})",
                    self.identity.user_id()
                ),
            ),
        ))
    }

    /// Enforces an identity's exact graph read grant for APIs that bypass the
    /// normal query binder/active-graph authorization path.
    #[cfg(feature = "gql")]
    fn require_graph_read_grant(&self, graph_name: Option<&str>) -> Result<()> {
        self.require_graph_grant(graph_name, crate::auth::Role::ReadOnly, "read")
    }

    #[cfg(feature = "gql")]
    fn require_graph_write_grant(&self, graph_name: Option<&str>) -> Result<()> {
        self.require_graph_grant(graph_name, crate::auth::Role::ReadWrite, "write")
    }

    /// Requires write authority over every graph coordinate governed by one
    /// schema-scoped catalog operation.
    ///
    /// Node, edge, graph-type, and constraint DDL can inspect or govern every
    /// graph in the current schema. Checking only the schema's default graph
    /// would let a partially scoped administrator learn from, or impose
    /// metadata on, graphs outside their capability. Deliberately return one
    /// non-enumerating error so the authorization check does not disclose the
    /// name of the first inaccessible graph.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn require_schema_graph_write_scope(
        &self,
        schema: Option<&str>,
        catalog: CatalogRead<'_>,
    ) -> Result<()> {
        if !self.identity.has_grants() {
            return Ok(());
        }

        let registered_schemas = catalog.schema_names();
        let default_key = schema.map(|schema| format!("{schema}/{SCHEMA_DEFAULT_GRAPH}"));
        let default_allowed =
            self.graph_grant_allows(default_key.as_deref(), crate::auth::Role::ReadWrite);
        let all_named_allowed = self.store.graph_names().into_iter().all(|graph_name| {
            let graph_schema = graph_name.split_once('/').and_then(|(prefix, _)| {
                registered_schemas
                    .iter()
                    .find(|registered| registered.eq_ignore_ascii_case(prefix))
            });
            let belongs_to_schema = match (schema, graph_schema) {
                (None, None) => true,
                (Some(expected), Some(actual)) => actual.eq_ignore_ascii_case(expected),
                (None, Some(_)) | (Some(_), None) => false,
            };
            !belongs_to_schema
                || self.graph_grant_allows(Some(&graph_name), crate::auth::Role::ReadWrite)
        });
        if default_allowed && all_named_allowed {
            return Ok(());
        }

        let scope = schema.unwrap_or("default");
        Err(grafeo_common::utils::error::Error::Query(
            grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Semantic,
                format!(
                    "permission denied: complete write grants for schema '{scope}' are required (user: {})",
                    self.identity.user_id()
                ),
            ),
        ))
    }

    #[cfg(feature = "cdc")]
    fn cdc_event_is_visible_to_identity(&self, event: &crate::cdc::ChangeEvent) -> bool {
        if event.entity_id.is_triple() {
            return self.identity.can_read()
                && self.identity.can_access_rdf_graph(
                    event.triple_graph.as_deref(),
                    crate::auth::Role::ReadOnly,
                );
        }
        event
            .graph_path()
            .is_some_and(|path| self.index_path_grant_allows(path, crate::auth::Role::ReadOnly))
    }

    /// Graph lifecycle mutates durable state; virtual projection DDL mutates a
    /// process-local registry but still needs the transaction/publication path.
    /// Pure session-state commands need neither.
    #[cfg(feature = "gql")]
    fn session_command_mutates(cmd: &grafeo_adapters::query::gql::ast::SessionCommand) -> bool {
        use grafeo_adapters::query::gql::ast::SessionCommand::{
            CreateGraph, CreateProjection, DropGraph, DropProjection,
        };
        matches!(
            cmd,
            CreateGraph { .. } | DropGraph { .. } | CreateProjection { .. } | DropProjection { .. }
        )
    }

    /// Executes a session or transaction command, returning an empty result.
    #[cfg(feature = "gql")]
    fn execute_session_command(
        &self,
        cmd: grafeo_adapters::query::gql::ast::SessionCommand,
    ) -> Result<QueryResult> {
        use grafeo_adapters::query::gql::ast::SessionCommand;
        #[cfg(feature = "lpg")]
        use grafeo_adapters::query::gql::ast::TransactionIsolationLevel;
        use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};

        // Check role-based permission for graph management commands
        match &cmd {
            SessionCommand::CreateGraph { .. }
            | SessionCommand::DropGraph { .. }
            | SessionCommand::CreateProjection { .. }
            | SessionCommand::DropProjection { .. } => {
                self.require_permission(crate::auth::StatementKind::Write)?;
            }
            _ => {} // Session state + transaction control: always allowed
        }

        // Block DDL in read-only transactions (ISO/IEC 39075 Section 8)
        if *self.read_only_tx.lock() {
            match &cmd {
                SessionCommand::CreateGraph { .. }
                | SessionCommand::DropGraph { .. }
                | SessionCommand::CreateProjection { .. }
                | SessionCommand::DropProjection { .. } => {
                    return Err(Error::Transaction(
                        grafeo_common::utils::error::TransactionError::ReadOnly,
                    ));
                }
                _ => {} // Session state + transaction control allowed
            }
        }

        match cmd {
            #[cfg(feature = "lpg")]
            SessionCommand::CreateGraph {
                name,
                if_not_exists,
                typed,
                like_graph,
                copy_of,
                open: _,
            } => {
                // ISO/IEC 39075 Section 12.4: graphs are created within the current schema
                if name.is_empty() {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "Graph name must not be empty",
                    )));
                }
                if name.eq_ignore_ascii_case("default") {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "Graph name 'default' is reserved for the current default partition",
                    )));
                }
                if name.contains('/') {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!(
                            "Graph name '{name}' must not contain '/' (reserved as schema/graph separator)"
                        ),
                    )));
                }
                let owner_schema = self.current_schema_incarnation()?;
                let qualify = |local_name: &str| match owner_schema.as_ref() {
                    Some(owner) => format!("{}/{local_name}", owner.name),
                    None => local_name.to_string(),
                };
                let storage_key =
                    if owner_schema.is_some() && name.eq_ignore_ascii_case(SCHEMA_DEFAULT_GRAPH) {
                        qualify(SCHEMA_DEFAULT_GRAPH)
                    } else {
                        qualify(&name)
                    };
                self.require_graph_write_grant(Some(&storage_key))?;
                let graph_path = Self::graph_path_for_storage_key(Some(&storage_key))?;

                // ISO `IF NOT EXISTS` is a true no-op. Resolve the target
                // before validating COPY/LIKE/type inputs so an already-
                // existing target cannot fail because an otherwise-unused
                // source is missing or invalid, and cannot register source SSI
                // dependencies as a side effect.
                if if_not_exists {
                    let _target_cut = self.publication_read_guard();
                    if self.graph_visible(&storage_key) {
                        return Ok(QueryResult::empty());
                    }
                }

                // Resolve every inherited source coordinate as one coherent
                // publication-cut bundle. The exact store pin, transaction-
                // visible graph-type binding, and COPY property-index image
                // must never be assembled from different DROP/CREATE
                // incarnations. COPY takes precedence only defensively; the
                // grammar does not produce COPY and LIKE together.
                let source_bundle = if let Some(src) = copy_of.as_ref().or(like_graph.as_ref()) {
                    let _source_cut = self.publication_read_guard();
                    let src_key = Self::storage_key_for_context(
                        owner_schema.as_ref().map(|owner| owner.name.as_str()),
                        Some(src),
                    );
                    self.require_graph_read_grant(src_key.as_deref())?;
                    let (incarnation, read_view) = match src_key.as_deref() {
                        None => (Arc::clone(&self.store), Arc::clone(&self.graph_store)),
                        Some(src_key) => {
                            let source = self.session_named_graph(src_key).ok_or_else(|| {
                                Error::Query(QueryError::new(
                                    QueryErrorKind::Semantic,
                                    format!("Source graph '{src}' does not exist"),
                                ))
                            })?;
                            let read_view = Arc::clone(&source) as Arc<dyn GraphStoreSearch>;
                            (source, read_view)
                        }
                    };
                    self.track_graph_incarnation_touch(src_key.as_deref(), &incarnation)?;
                    let binding = self.session_graph_type_binding(
                        &Self::graph_path_for_storage_key(src_key.as_deref())?,
                    );
                    let copy = if copy_of.is_some() {
                        // `_source_cut` makes the Read Committed epoch below a
                        // real committed publication cut rather than a merely
                        // reserved epoch whose state is still being installed.
                        let (epoch, transaction_id) = self.copy_transaction_context();
                        let transaction_id = transaction_id.ok_or_else(|| {
                            Error::Internal(format!(
                                "CREATE GRAPH AS COPY OF '{storage_key}' escaped transaction framing"
                            ))
                        })?;
                        GraphStore::record_lpg_dataset_read(read_view.as_ref(), transaction_id);
                        let mut physical_property_indexes = incarnation.property_index_keys();
                        physical_property_indexes.sort_unstable();
                        physical_property_indexes.dedup();
                        let property_indexes = self.transaction_visible_property_index_keys(
                            src_key.as_deref(),
                            &incarnation,
                        );
                        Some(CopySourceSnapshot {
                            epoch,
                            transaction_id,
                            graph: Self::materialize_lpg_graph(
                                read_view.as_ref(),
                                epoch,
                                transaction_id,
                                property_indexes,
                            ),
                            physical_property_indexes,
                        })
                    } else {
                        None
                    };
                    Some(GraphCreationSource {
                        storage_key: src_key,
                        incarnation,
                        read_view,
                        graph_type_binding: binding,
                        copy,
                    })
                } else {
                    None
                };

                // Resolve and validate the binding before creating or WAL-
                // framing the graph. Publication remains transaction-local.
                let graph_type_binding = if let Some(type_name) = typed.as_ref() {
                    let resolved = if type_name.contains('/') {
                        type_name.clone()
                    } else {
                        qualify(type_name)
                    };
                    Some(resolved)
                } else if let Some(source) = source_bundle.as_ref() {
                    source.graph_type_binding.clone()
                } else {
                    None
                };
                if let Some(graph_type) = graph_type_binding.as_deref() {
                    self.catalog_view()
                        .validate_graph_type_binding_target(graph_type)
                        .map_err(|error| {
                            Error::Query(QueryError::new(
                                QueryErrorKind::Semantic,
                                error.to_string(),
                            ))
                        })?;
                }
                if let Some(source) = source_bundle.as_ref()
                    && let Some(copy) = source.copy.as_ref()
                {
                    self.validate_lpg_copy_snapshot(
                        &storage_key,
                        owner_schema.as_ref(),
                        graph_type_binding.clone(),
                        Arc::clone(&source.read_view),
                        copy,
                    )?;
                }

                // Build the complete destination privately. No session-visible
                // lifecycle state, WAL record, or CDC receipt exists until the
                // source has been materialized, validated, and installed.
                let mut copied = None;
                let (created, destination) = if self.graph_visible(&storage_key) {
                    (false, None)
                } else if self.in_transaction() {
                    let graph = self.new_detached_lpg_graph()?;
                    if copy_of.is_some() {
                        let source = source_bundle.as_ref().ok_or_else(|| {
                            Error::Internal(format!(
                                "CREATE GRAPH AS COPY OF '{storage_key}' lost its source publication cut"
                            ))
                        })?;
                        let copy = source.copy.as_ref().ok_or_else(|| {
                            Error::Internal(format!(
                                "CREATE GRAPH AS COPY OF '{storage_key}' lost its transaction-visible source metadata"
                            ))
                        })?;
                        copied = Some(Self::install_lpg_graph_snapshot(
                            &copy.graph,
                            &graph,
                            copy.epoch,
                            copy.transaction_id,
                        ));
                    }
                    let copy_source_indexes = source_bundle.as_ref().and_then(|source| {
                        source.copy.as_ref().map(|copy| CopySourceIndexExpectation {
                            source_name: source.storage_key.clone(),
                            source: Arc::clone(&source.incarnation),
                            physical_keys: copy.physical_property_indexes.clone(),
                        })
                    });
                    self.pending_created_graphs.lock().insert(
                        graph_path.clone(),
                        PendingCreatedGraph {
                            store: Arc::clone(&graph),
                            parent: Arc::clone(&self.store),
                            namespace: owner_schema
                                .clone()
                                .map_or(PendingGraphNamespace::Root, PendingGraphNamespace::Schema),
                            copy_source_indexes,
                        },
                    );
                    self.track_named_graph_incarnation_touch(&storage_key, &graph)?;
                    (true, Some(graph))
                } else {
                    return Err(Error::Internal(
                        "graph creation escaped transaction framing".into(),
                    ));
                };
                if !created && !if_not_exists {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Graph '{name}' already exists"),
                    )));
                }
                if created {
                    #[cfg(feature = "wal")]
                    self.log_schema_wal(&grafeo_storage::wal::WalRecord::CreateLpgGraph {
                        incarnation: destination
                            .as_ref()
                            .ok_or_else(|| {
                                Error::Internal("created graph has no destination".into())
                            })?
                            .graph_incarnation_id(),
                        graph: Self::graph_path_for_storage_key(Some(&storage_key))?,
                        transaction_id: self
                            .current_transaction_id()
                            .unwrap_or(TransactionId::SYSTEM),
                    })?;
                }
                if let (Some(destination), Some(copied)) = (destination.as_ref(), copied.as_ref()) {
                    #[cfg(all(feature = "wal", feature = "lpg"))]
                    self.log_copied_lpg_graph(&storage_key, copied)?;
                    self.stage_copied_property_indexes(
                        &storage_key,
                        owner_schema.as_ref(),
                        destination,
                        &copied.property_indexes,
                    )?;
                }

                if created {
                    if self.in_transaction() {
                        self.stage_graph_type_binding(&graph_path, graph_type_binding.clone());
                        #[cfg(feature = "wal")]
                        self.log_schema_wal(
                            &grafeo_storage::wal::WalRecord::SetGraphTypeBinding {
                                transaction_id: self
                                    .current_transaction_id()
                                    .unwrap_or(TransactionId::SYSTEM),
                                graph: Self::graph_path_for_storage_key(Some(&storage_key))?,
                                graph_type: graph_type_binding,
                            },
                        )?;
                    } else {
                        let expected = self.catalog_view().get_graph_type_binding(&graph_path);
                        let published = self.catalog_view().publish_graph_type_binding_if_same(
                            &graph_path,
                            expected.as_deref(),
                            graph_type_binding.clone(),
                        );
                        if !published {
                            return Err(Error::Internal(format!(
                                "graph type binding for '{storage_key}' changed during CREATE GRAPH"
                            )));
                        }
                        #[cfg(feature = "wal")]
                        self.log_schema_wal(
                            &grafeo_storage::wal::WalRecord::SetGraphTypeBinding {
                                transaction_id: TransactionId::SYSTEM,
                                graph: Self::graph_path_for_storage_key(Some(&storage_key))?,
                                graph_type: graph_type_binding,
                            },
                        )?;
                    }
                }

                // This is deliberately the final operation in the command:
                // every fallible copy, WAL, and graph-binding step above has
                // succeeded. The accumulator still owns rollback/savepoint and
                // final-incarnation filtering until durable publication.
                #[cfg(feature = "cdc")]
                if let (Some(destination), Some(copied)) = (destination, copied) {
                    self.stage_copied_lpg_graph(&graph_path, destination, copied);
                }

                Ok(QueryResult::empty())
            }
            #[cfg(feature = "lpg")]
            SessionCommand::DropGraph { name, if_exists } => {
                if name.is_empty() {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "Graph name must not be empty",
                    )));
                }
                if name.eq_ignore_ascii_case("default") {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "Graph name 'default' is reserved for the current default partition",
                    )));
                }
                let owner_schema = self.current_schema_incarnation()?;
                let storage_key = owner_schema
                    .as_ref()
                    .map_or_else(|| name.clone(), |owner| format!("{}/{name}", owner.name));
                self.require_graph_write_grant(Some(&storage_key))?;
                if owner_schema.is_some() && name.eq_ignore_ascii_case(SCHEMA_DEFAULT_GRAPH) {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!(
                            "Graph '{storage_key}' is the owning schema's default partition and cannot be dropped directly"
                        ),
                    )));
                }
                let visible = self.graph_visible(&storage_key);
                if !visible && !if_exists {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Graph '{name}' does not exist"),
                    )));
                }
                if visible {
                    let namespace = owner_schema
                        .map_or(PendingGraphNamespace::Root, PendingGraphNamespace::Schema);
                    let path = Self::graph_path_for_storage_key(Some(&storage_key))?;
                    self.stage_drop_graph_path(&path, &namespace)?;
                    self.reset_graph_context_if_storage_key(&storage_key)?;
                }
                Ok(QueryResult::empty())
            }
            #[cfg(feature = "lpg")]
            SessionCommand::UseGraph(name) | SessionCommand::SessionSetGraph(name) => {
                let admitted = self.admit_query_result(QueryResult::empty())?;
                self.use_graph(&name)?;
                Ok(admitted)
            }
            SessionCommand::SessionSetSchema(name) => {
                #[cfg(all(feature = "cdc", any(feature = "lpg", feature = "triple-store")))]
                let _operation = self.cdc_mutation_operation_guard();
                let _publication = self.publication_read_guard();
                // ISO/IEC 39075 Section 7.1 GR1: set session schema (independent of graph)
                if !self.catalog_view().schema_exists(&name) {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Schema '{name}' does not exist"),
                    )));
                }
                // Store the CANONICAL (registered-case) name, not the user-typed
                // case: `schema_exists` matches case-insensitively, but type/graph
                // storage keys are prefixed with the stored schema name, so a case
                // mismatch ("myschema" vs registered "MySchema") would silently
                // break every subsequent type lookup under the schema.
                let canonical = self
                    .catalog_view()
                    .schema_names()
                    .into_iter()
                    .find(|s| s.eq_ignore_ascii_case(&name))
                    .unwrap_or(name);
                let admitted = self.admit_query_result(QueryResult::empty())?;
                self.set_schema(&canonical)?;
                Ok(admitted)
            }
            SessionCommand::SessionSetTimeZone(tz) => {
                let admitted = self.admit_query_result(QueryResult::empty())?;
                self.set_time_zone(&tz);
                Ok(admitted)
            }
            #[cfg(feature = "gql")]
            SessionCommand::SessionSetParameter(key, expr) => {
                if key.eq_ignore_ascii_case("viewing_epoch") {
                    match Self::eval_integer_literal(&expr) {
                        Some(n) if n >= 0 => {
                            // reason: guard ensures n >= 0
                            #[allow(clippy::cast_sign_loss)]
                            let epoch = n as u64;
                            let result =
                                self.bounded_status(format_args!("Set viewing_epoch to {n}"))?;
                            self.set_viewing_epoch(EpochId::new(epoch));
                            Ok(result)
                        }
                        _ => Err(Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            "viewing_epoch must be a non-negative integer literal",
                        ))),
                    }
                } else {
                    // For now, store parameter name with Null value.
                    // Full expression evaluation would require building and executing a plan.
                    let admitted = self.admit_query_result(QueryResult::empty())?;
                    self.set_parameter(&key, Value::Null);
                    Ok(admitted)
                }
            }
            SessionCommand::SessionReset(target) => {
                let admitted = self.admit_query_result(QueryResult::empty())?;
                use grafeo_adapters::query::gql::ast::SessionResetTarget;
                match target {
                    SessionResetTarget::All => self.reset_session()?,
                    SessionResetTarget::Schema => self.reset_schema()?,
                    SessionResetTarget::Graph => self.reset_graph()?,
                    SessionResetTarget::TimeZone => self.reset_time_zone(),
                    SessionResetTarget::Parameters => self.reset_parameters(),
                }
                Ok(admitted)
            }
            SessionCommand::SessionClose => {
                let admitted = self.admit_query_result(QueryResult::empty())?;
                self.reset_session()?;
                Ok(admitted)
            }
            #[cfg(feature = "lpg")]
            SessionCommand::StartTransaction {
                read_only,
                isolation_level,
            } => {
                let engine_level = isolation_level.map(|l| match l {
                    TransactionIsolationLevel::ReadCommitted => {
                        crate::transaction::IsolationLevel::ReadCommitted
                    }
                    TransactionIsolationLevel::SnapshotIsolation => {
                        crate::transaction::IsolationLevel::SnapshotIsolation
                    }
                    TransactionIsolationLevel::Serializable => {
                        crate::transaction::IsolationLevel::Serializable
                    }
                });
                let result = self.bounded_status(format_args!("{}", "Transaction started"))?;
                self.begin_transaction_inner(read_only, engine_level)?;
                Ok(result)
            }
            #[cfg(feature = "lpg")]
            SessionCommand::Commit => {
                let result = self.bounded_status(format_args!("{}", "Transaction committed"))?;
                self.commit_inner()?;
                Ok(result)
            }
            #[cfg(feature = "lpg")]
            SessionCommand::Rollback => {
                let result = self.bounded_status(format_args!("{}", "Transaction rolled back"))?;
                self.rollback_inner()?;
                Ok(result)
            }
            #[cfg(feature = "lpg")]
            SessionCommand::Savepoint(name) => {
                let result = self.bounded_status(format_args!("Savepoint '{name}' created"))?;
                self.savepoint(&name)?;
                Ok(result)
            }
            #[cfg(feature = "lpg")]
            SessionCommand::RollbackToSavepoint(name) => {
                let result =
                    self.bounded_status(format_args!("Rolled back to savepoint '{name}'"))?;
                self.rollback_to_savepoint(&name)?;
                Ok(result)
            }
            #[cfg(feature = "lpg")]
            SessionCommand::ReleaseSavepoint(name) => {
                let result = self.bounded_status(format_args!("Savepoint '{name}' released"))?;
                self.release_savepoint(&name)?;
                Ok(result)
            }
            #[cfg(feature = "lpg")]
            SessionCommand::CreateProjection {
                name,
                node_labels,
                edge_types,
            } => {
                use grafeo_core::graph::ProjectionSpec;

                let spec = ProjectionSpec::new()
                    .with_node_labels(node_labels)
                    .with_edge_types(edge_types);
                if !self.stage_create_virtual_projection(name.clone(), spec)? {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Projection '{name}' already exists"),
                    )));
                }
                self.bounded_status(format_args!("Projection '{name}' created"))
            }
            #[cfg(feature = "lpg")]
            SessionCommand::DropProjection { name } => {
                if !self.stage_drop_virtual_projection(&name)? {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Projection '{name}' does not exist"),
                    )));
                }
                self.bounded_status(format_args!("Projection '{name}' dropped"))
            }
            #[cfg(feature = "lpg")]
            SessionCommand::ShowProjections => self.execute_show_projections(),
            #[cfg(not(feature = "lpg"))]
            _ => Err(grafeo_common::utils::error::Error::Internal(
                "This command requires the `lpg` feature".to_string(),
            )),
        }
    }

    /// Logs a WAL record for a schema change. Fail-closed: poisons on error.
    #[cfg(all(feature = "wal", feature = "lpg"))]
    fn log_schema_wal(&self, record: &grafeo_storage::wal::WalRecord) -> Result<()> {
        if let Some(records) = self.catalog_wal_batch.lock().as_mut() {
            records.push(record.clone());
            return Ok(());
        }
        self.log_wal_record(record)
    }

    /// WAL every node/edge copied by `CREATE GRAPH … AS COPY OF`.
    #[cfg(all(feature = "wal", feature = "lpg", feature = "gql"))]
    fn log_copied_lpg_graph(&self, graph_key: &str, copied: &CopiedLpgEntities) -> Result<()> {
        use grafeo_storage::wal::LpgMutationOp;
        let graph = Self::graph_path_for_storage_key(Some(graph_key))?;
        let tid = self
            .current_transaction_id()
            .unwrap_or(TransactionId::SYSTEM);
        for node in &copied.nodes {
            self.log_wal_record(&grafeo_storage::wal::WalRecord::lpg(
                tid,
                graph.clone(),
                LpgMutationOp::CreateNode {
                    id: node.id,
                    labels: node.labels.iter().map(|s| s.to_string()).collect(),
                },
            ))?;
            for (key, value) in &node.properties {
                self.log_wal_record(&grafeo_storage::wal::WalRecord::lpg(
                    tid,
                    graph.clone(),
                    LpgMutationOp::SetNodeProperty {
                        id: node.id,
                        key: key.to_string(),
                        value: value.clone(),
                    },
                ))?;
            }
        }
        for edge in &copied.edges {
            self.log_wal_record(&grafeo_storage::wal::WalRecord::lpg(
                tid,
                graph.clone(),
                LpgMutationOp::CreateEdge {
                    id: edge.id,
                    src: edge.src,
                    dst: edge.dst,
                    edge_type: edge.edge_type.to_string(),
                },
            ))?;
            for (key, value) in &edge.properties {
                self.log_wal_record(&grafeo_storage::wal::WalRecord::lpg(
                    tid,
                    graph.clone(),
                    LpgMutationOp::SetEdgeProperty {
                        id: edge.id,
                        key: key.to_string(),
                        value: value.clone(),
                    },
                ))?;
            }
        }
        Ok(())
    }

    /// Stages complete Create snapshots for one successfully copied graph.
    #[cfg(all(feature = "cdc", feature = "lpg"))]
    fn stage_copied_lpg_graph(
        &self,
        graph_key: &GraphPath,
        graph_incarnation: Arc<LpgStore>,
        copied: CopiedLpgEntities,
    ) {
        for node in copied.nodes {
            let properties: std::collections::HashMap<String, Value> = node
                .properties
                .iter()
                .map(|(key, value)| (key.as_str().to_string(), value.clone()))
                .collect();
            let labels = node.labels.iter().map(ToString::to_string).collect();
            self.stage_lpg_node_create(
                node.id,
                (!properties.is_empty()).then_some(properties),
                Some(labels),
                graph_key,
                Arc::clone(&graph_incarnation),
            );
        }
        for edge in copied.edges {
            let properties: std::collections::HashMap<String, Value> = edge
                .properties
                .iter()
                .map(|(key, value)| (key.as_str().to_string(), value.clone()))
                .collect();
            self.stage_lpg_edge_create(
                edge.id,
                (!properties.is_empty()).then_some(properties),
                (edge.src, edge.dst),
                edge.edge_type.to_string(),
                graph_key,
                Arc::clone(&graph_incarnation),
            );
        }
    }

    /// Materializes one complete transaction-visible LPG graph image.
    ///
    /// Candidate enumeration is deliberately separate from visibility: a
    /// layered store returns every retained identity from `all_node_ids`, then
    /// the versioned accessors decide whether that identity belongs to this
    /// snapshot. This preserves pre-delete rows for Snapshot Isolation while
    /// also including this transaction's pending creates.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn materialize_lpg_graph(
        source: &dyn GraphStoreSearch,
        epoch: EpochId,
        transaction_id: TransactionId,
        property_indexes: Vec<String>,
    ) -> MaterializedLpgGraph {
        let mut nodes: Vec<Node> = source
            .all_node_ids()
            .into_iter()
            .filter_map(|id| {
                let mut node = source.get_node_versioned(id, epoch, transaction_id)?;
                node.properties = source
                    .read_node_properties_visible(id, epoch, Some(transaction_id))
                    .into_iter()
                    .collect();
                node.labels = source
                    .read_node_labels_visible(id, epoch, Some(transaction_id))
                    .into_iter()
                    .collect();
                node.labels
                    .sort_unstable_by(|left, right| left.as_str().cmp(right.as_str()));
                Some(node)
            })
            .collect();

        let mut edge_ids: Vec<EdgeId> = nodes
            .iter()
            .flat_map(|node| {
                source
                    .edges_from_versioned(node.id, Direction::Outgoing, epoch, transaction_id)
                    .into_iter()
                    .map(|(_, edge_id)| edge_id)
            })
            .collect();
        edge_ids.sort_unstable();
        edge_ids.dedup();
        let mut edges: Vec<Edge> = edge_ids
            .into_iter()
            .filter_map(|id| {
                let mut edge = source.get_edge_versioned(id, epoch, transaction_id)?;
                edge.properties = source
                    .read_edge_properties_visible(id, epoch, Some(transaction_id))
                    .into_iter()
                    .collect();
                Some(edge)
            })
            .collect();
        nodes.sort_unstable_by_key(|node| node.id);
        edges.sort_unstable_by_key(|edge| edge.id);

        MaterializedLpgGraph {
            nodes,
            edges,
            property_indexes,
        }
    }

    /// Validates a COPY source as the complete post-image of its requested
    /// destination binding before any target graph, WAL record, or CDC event
    /// exists. Commit repeats these structural checks under the publication
    /// write lock to close catalog/binding races.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn validate_lpg_copy_snapshot(
        &self,
        target_graph: &str,
        owner_schema: Option<&SchemaIncarnation>,
        graph_type_binding: Option<String>,
        source: Arc<dyn GraphStoreSearch>,
        snapshot: &CopySourceSnapshot,
    ) -> Result<()> {
        use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};

        let constraint_store: Arc<dyn GraphStore> = source;
        let validator = CatalogConstraintValidator::new(self.catalog_view())
            .with_store(constraint_store)
            .with_max_property_size(self.max_property_size)
            .with_schema(owner_schema.map(|owner| owner.name.clone()))
            .with_graph_path(Self::graph_path_for_storage_key(Some(target_graph))?)
            .with_graph_type_binding_override(graph_type_binding);
        let reject = |error| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!("CREATE GRAPH AS COPY OF rejected by target schema: {error}"),
            ))
        };

        let labels_by_node: std::collections::HashMap<NodeId, Vec<String>> = snapshot
            .graph
            .nodes
            .iter()
            .map(|node| {
                (
                    node.id,
                    node.labels.iter().map(ToString::to_string).collect(),
                )
            })
            .collect();
        for node in &snapshot.graph.nodes {
            let labels = labels_by_node.get(&node.id).cloned().unwrap_or_default();
            let properties: Vec<(String, Value)> = node
                .properties
                .iter()
                .map(|(key, value)| (key.as_str().to_string(), value.clone()))
                .collect();
            validator
                .validate_node_labels_allowed(&labels)
                .and_then(|()| {
                    validator.validate_node_post_image(
                        Some(node.id),
                        &labels,
                        &properties,
                        snapshot.epoch,
                        Some(snapshot.transaction_id),
                    )
                })
                .map_err(&reject)?;
        }
        for edge in &snapshot.graph.edges {
            let source_labels = labels_by_node.get(&edge.src).ok_or_else(|| {
                Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!(
                        "CREATE GRAPH AS COPY OF source edge {} has missing source endpoint {}",
                        edge.id, edge.src
                    ),
                ))
            })?;
            let target_labels = labels_by_node.get(&edge.dst).ok_or_else(|| {
                Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!(
                        "CREATE GRAPH AS COPY OF source edge {} has missing target endpoint {}",
                        edge.id, edge.dst
                    ),
                ))
            })?;
            let properties: Vec<(String, Value)> = edge
                .properties
                .iter()
                .map(|(key, value)| (key.as_str().to_string(), value.clone()))
                .collect();
            validator
                .validate_edge_type_allowed(edge.edge_type.as_str())
                .and_then(|()| {
                    validator.validate_edge_endpoints(
                        edge.edge_type.as_str(),
                        source_labels,
                        target_labels,
                    )
                })
                .and_then(|()| {
                    validator.validate_edge_post_image(edge.edge_type.as_str(), &properties)
                })
                .map_err(&reject)?;
        }
        Ok(())
    }

    /// Installs a prevalidated materialized image into one detached graph.
    /// Destination entities remain PENDING under the creating transaction;
    /// WAL records are emitted separately using the newly assigned IDs.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn install_lpg_graph_snapshot(
        snapshot: &MaterializedLpgGraph,
        destination: &LpgStore,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> CopiedLpgEntities {
        let mut ids = std::collections::HashMap::with_capacity(snapshot.nodes.len());
        #[cfg(any(feature = "wal", feature = "cdc"))]
        let mut copied_nodes = Vec::with_capacity(snapshot.nodes.len());
        #[cfg(any(feature = "wal", feature = "cdc"))]
        let mut copied_edges = Vec::with_capacity(snapshot.edges.len());

        for node in &snapshot.nodes {
            let labels: Vec<&str> = node.labels.iter().map(|label| label.as_str()).collect();
            let copied_properties = node.properties.clone();
            let id = destination.create_node_versioned(&labels, epoch, transaction_id);
            for (key, value) in &copied_properties {
                destination.set_node_property_buffered(
                    id,
                    key.as_str(),
                    value.clone(),
                    transaction_id,
                );
            }
            ids.insert(node.id, id);
            #[cfg(any(feature = "wal", feature = "cdc"))]
            copied_nodes.push(Node {
                id,
                labels: node.labels.clone(),
                properties: copied_properties,
            });
        }
        for edge in &snapshot.edges {
            let src = *ids
                .get(&edge.src)
                .expect("prevalidated COPY source endpoint must have a destination ID");
            let dst = *ids
                .get(&edge.dst)
                .expect("prevalidated COPY target endpoint must have a destination ID");
            let copied_properties = edge.properties.clone();
            let id = destination.create_edge_versioned(
                src,
                dst,
                edge.edge_type.as_str(),
                epoch,
                transaction_id,
            );
            for (key, value) in &copied_properties {
                destination.set_edge_property_buffered(
                    id,
                    key.as_str(),
                    value.clone(),
                    transaction_id,
                );
            }
            #[cfg(any(feature = "wal", feature = "cdc"))]
            copied_edges.push(Edge {
                id,
                src,
                dst,
                edge_type: edge.edge_type.clone(),
                properties: copied_properties,
            });
        }
        CopiedLpgEntities {
            #[cfg(any(feature = "wal", feature = "cdc"))]
            nodes: copied_nodes,
            #[cfg(any(feature = "wal", feature = "cdc"))]
            edges: copied_edges,
            property_indexes: snapshot.property_indexes.clone(),
        }
    }

    /// Returns the source incarnation's exact property-index post-image after
    /// replaying this transaction's ordered physical index DDL.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn transaction_visible_property_index_keys(
        &self,
        graph_key: Option<&str>,
        source: &Arc<LpgStore>,
    ) -> Vec<String> {
        let mut keys: std::collections::BTreeSet<String> =
            source.property_index_keys().into_iter().collect();
        for ddl in self.pending_index_ddl.lock().iter().filter(|ddl| {
            Self::index_path_matches_flat(&ddl.graph, graph_key)
                && Arc::ptr_eq(&ddl.target, source)
                && matches!(
                    ddl.kind,
                    PendingIndexKind::Property | PendingIndexKind::BTree
                )
        }) {
            if ddl.create {
                keys.insert(ddl.property.clone());
            } else {
                keys.remove(&ddl.property);
            }
        }
        keys.into_iter().collect()
    }

    /// Stages COPY property indexes into the same canonical owner allocation
    /// and physical publication as ordinary creates. No predecessor WAL receipt.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn stage_copied_property_indexes(
        &self,
        graph_key: &str,
        owner_schema: Option<&SchemaIncarnation>,
        destination: &Arc<LpgStore>,
        properties: &[String],
    ) -> Result<()> {
        if properties.is_empty() {
            return Ok(());
        }
        if !self.in_transaction() {
            return Err(Self::index_ddl_error(
                "copied property indexes require an active transaction",
            ));
        }
        let graph = Self::graph_path_for_storage_key(Some(graph_key))?;
        let staged = properties.iter().map(|property| PendingIndexDdl {
            create: true,
            rebuild: false,
            expected_owner: None,
            configuration: None,
            owner_result: Arc::new(std::sync::OnceLock::new()),
            graph: graph.clone(),
            owner_schema: owner_schema.cloned(),
            name: None,
            label: String::new(),
            property: property.clone(),
            kind: PendingIndexKind::Property,
            target: Arc::clone(destination),
        });
        self.pending_index_ddl.lock().extend(staged);
        Ok(())
    }

    #[cfg(feature = "lpg")]
    fn index_ddl_error(message: impl Into<String>) -> grafeo_common::utils::error::Error {
        grafeo_common::utils::error::Error::Query(grafeo_common::utils::error::QueryError::new(
            grafeo_common::utils::error::QueryErrorKind::Semantic,
            message,
        ))
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn catalog_index_kind(kind: &PendingIndexKind) -> crate::catalog::IndexType {
        match kind {
            PendingIndexKind::Property => crate::catalog::IndexType::Hash,
            PendingIndexKind::BTree => crate::catalog::IndexType::BTree,
            PendingIndexKind::Text { .. } => crate::catalog::IndexType::FullText,
            PendingIndexKind::Vector { .. } => crate::catalog::IndexType::Vector,
        }
    }

    #[cfg(feature = "lpg")]
    fn pending_kind_from_catalog(kind: crate::catalog::IndexType) -> PendingIndexKind {
        match kind {
            crate::catalog::IndexType::Hash => PendingIndexKind::Property,
            crate::catalog::IndexType::BTree => PendingIndexKind::BTree,
            crate::catalog::IndexType::FullText => PendingIndexKind::Text {
                min_token_length: None,
            },
            crate::catalog::IndexType::Vector => PendingIndexKind::Vector {
                dimensions: None,
                metric: None,
                m: None,
                ef_construction: None,
                ef: None,
                quantization: None,
            },
        }
    }

    #[cfg(feature = "lpg")]
    fn physical_index_key(
        graph: &GraphPath,
        label: &str,
        property: &str,
        kind: &PendingIndexKind,
    ) -> PhysicalIndexKey {
        match kind.physical_kind() {
            PhysicalIndexFamily::Property => PhysicalIndexKey::property(graph.clone(), property),
            PhysicalIndexFamily::Text => PhysicalIndexKey::text(graph.clone(), label, property),
            PhysicalIndexFamily::Vector => PhysicalIndexKey::vector(graph.clone(), label, property),
        }
    }

    #[cfg(feature = "lpg")]
    fn catalog_physical_index_owner(
        &self,
        key: &PhysicalIndexKey,
        except_name: Option<&str>,
    ) -> Option<String> {
        Self::catalog_physical_index_owner_at(self.catalog_view().read().view(), key, except_name)
    }

    #[cfg(feature = "lpg")]
    fn catalog_physical_index_owner_at(
        catalog: CatalogRead<'_>,
        key: &PhysicalIndexKey,
        except_name: Option<&str>,
    ) -> Option<String> {
        catalog
            .physical_index_owner(key)
            .filter(|definition| except_name != Some(definition.name.as_str()))
            .map(|definition| definition.name)
    }

    #[cfg(feature = "lpg")]
    fn physical_index_exists(target: &LpgStore, key: &PhysicalIndexKey) -> bool {
        match key.family() {
            PhysicalIndexFamily::Property => target.has_property_index(key.property_name()),
            PhysicalIndexFamily::Text => {
                #[cfg(feature = "text-index")]
                {
                    key.label().is_some_and(|label| {
                        target.get_text_index(label, key.property_name()).is_some()
                    })
                }
                #[cfg(not(feature = "text-index"))]
                {
                    false
                }
            }
            PhysicalIndexFamily::Vector => {
                #[cfg(feature = "vector-index")]
                {
                    key.label().is_some_and(|label| {
                        target
                            .get_vector_index(label, key.property_name())
                            .is_some()
                    })
                }
                #[cfg(not(feature = "vector-index"))]
                {
                    false
                }
            }
        }
    }

    #[cfg(feature = "lpg")]
    fn physical_index_keys_for_store(
        graph: &GraphPath,
        target: &LpgStore,
    ) -> std::collections::HashSet<PhysicalIndexKey> {
        let property_keys: std::collections::HashSet<PhysicalIndexKey> = target
            .property_index_keys()
            .into_iter()
            .map(|property| PhysicalIndexKey::property(graph.clone(), property))
            .collect();
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        let mut keys = property_keys;
        #[cfg(not(any(feature = "text-index", feature = "vector-index")))]
        let keys = property_keys;
        #[cfg(feature = "text-index")]
        for (encoded, _) in target.text_index_entries() {
            if let Some((label, property)) = grafeo_core::graph::lpg::decode_index_key(&encoded) {
                keys.insert(PhysicalIndexKey::text(graph.clone(), label, property));
            }
        }
        #[cfg(feature = "vector-index")]
        for (encoded, _) in target.vector_index_entries() {
            if let Some((label, property)) = grafeo_core::graph::lpg::decode_index_key(&encoded) {
                keys.insert(PhysicalIndexKey::vector(graph.clone(), label, property));
            }
        }
        keys
    }

    #[cfg(feature = "lpg")]
    fn index_subtree(
        path: GraphPath,
        root: Arc<LpgStore>,
    ) -> Result<std::collections::HashMap<GraphPath, Arc<LpgStore>>> {
        let mut work = vec![(path, root)];
        let mut stores = std::collections::HashMap::new();
        let mut identities = std::collections::HashSet::new();
        while let Some((path, store)) = work.pop() {
            if !identities.insert(Arc::as_ptr(&store).addr()) {
                return Err(Self::index_ddl_error(
                    "index subtree contains a cycle or aliased store",
                ));
            }
            for name in store.graph_names() {
                let child = store
                    .graph(&name)
                    .ok_or_else(|| Self::index_ddl_error("index subtree changed during capture"))?;
                work.push((
                    path.child(&name)
                        .map_err(|error| Self::index_ddl_error(error.to_string()))?,
                    child,
                ));
            }
            stores.insert(path, store);
        }
        Ok(stores)
    }

    #[cfg(feature = "lpg")]
    fn graph_indexes_after_pending_ddl(
        &self,
        graph_name: &GraphPath,
        target: &Arc<LpgStore>,
        pending: &[PendingIndexDdl],
    ) -> Result<(std::collections::HashSet<String>, bool)> {
        let stores = Self::index_subtree(graph_name.clone(), Arc::clone(target))?;
        let catalog_owner = self.catalog_view();
        let catalog = catalog_owner.read();
        let target_is_live = self
            .live_graph_path(graph_name)
            .as_ref()
            .is_some_and(|live| Arc::ptr_eq(live, target));
        let mut logical_names: std::collections::HashSet<String> = catalog
            .all_indexes()
            .into_iter()
            .filter(|index| target_is_live && stores.contains_key(index.key.graph()))
            .map(|index| index.name)
            .collect();
        let mut physical = std::collections::HashSet::new();
        for (path, store) in &stores {
            physical.extend(Self::physical_index_keys_for_store(path, store));
        }
        for ddl in pending.iter().filter(|ddl| {
            stores
                .get(&ddl.graph)
                .is_some_and(|store| Arc::ptr_eq(store, &ddl.target))
        }) {
            let key = Self::physical_index_key(&ddl.graph, &ddl.label, &ddl.property, &ddl.kind);
            if ddl.create {
                if let Some(name) = &ddl.name {
                    logical_names.insert(name.clone());
                }
                physical.insert(key);
            } else {
                if let Some(name) = &ddl.name {
                    logical_names.remove(name);
                }
                physical.remove(&key);
            }
        }
        Ok((logical_names, !physical.is_empty()))
    }

    /// Cascade exact owners across the target subtree; never leave descendant
    /// owners attached to a removed graph or manufacture owners for raw orphans.
    #[cfg(feature = "lpg")]
    fn stage_graph_index_cascade(
        &self,
        graph_name: &GraphPath,
        target: &Arc<LpgStore>,
        owner_schema: Option<&SchemaIncarnation>,
    ) -> Result<()> {
        if !self.in_transaction() {
            return Err(Self::index_ddl_error(
                "graph lifecycle requires an active transaction",
            ));
        }
        let stores = Self::index_subtree(graph_name.clone(), Arc::clone(target))?;
        let target_is_live = self
            .live_graph_path(graph_name)
            .as_ref()
            .is_some_and(|live| Arc::ptr_eq(live, target));
        let mut logical = std::collections::HashMap::new();
        if target_is_live {
            let catalog_owner = self.catalog_view();
            let catalog = catalog_owner.read();
            for definition in catalog.all_indexes() {
                let Some(store) = stores.get(definition.key.graph()) else {
                    continue;
                };
                let label = catalog
                    .get_label_name(definition.label)
                    .ok_or_else(|| Self::index_ddl_error("index owner label is missing"))?
                    .to_string();
                let property = catalog
                    .get_property_key_name(definition.property_key)
                    .ok_or_else(|| Self::index_ddl_error("index owner property is missing"))?
                    .to_string();
                let kind = Self::pending_kind_from_catalog(definition.index_type);
                let key = definition.key.clone();
                logical.insert(
                    key,
                    PendingIndexDdl {
                        create: true,
                        rebuild: false,
                        graph: definition.key.graph().clone(),
                        expected_owner: Some(definition.id),
                        configuration: Some(definition.configuration),
                        owner_result: Arc::new(std::sync::OnceLock::new()),
                        owner_schema: owner_schema.cloned(),
                        name: Some(definition.name),
                        label,
                        property,
                        kind,
                        target: Arc::clone(store),
                    },
                );
            }
        }
        let mut physical = std::collections::HashSet::new();
        for (path, store) in &stores {
            physical.extend(Self::physical_index_keys_for_store(path, store));
        }
        for ddl in self.pending_index_ddl.lock().iter().filter(|ddl| {
            stores
                .get(&ddl.graph)
                .is_some_and(|store| Arc::ptr_eq(store, &ddl.target))
        }) {
            let key = Self::physical_index_key(&ddl.graph, &ddl.label, &ddl.property, &ddl.kind);
            if ddl.create {
                physical.insert(key.clone());
                logical.insert(key, ddl.clone());
            } else {
                physical.remove(&key);
                logical.remove(&key);
            }
        }
        if physical.iter().any(|key| !logical.contains_key(key)) {
            return Err(Self::index_ddl_error(
                "graph subtree contains an unowned physical index",
            ));
        }
        let mut drops: Vec<_> = logical.into_values().collect();
        drops.sort_by(|a, b| {
            (&a.graph, &a.name, &a.property).cmp(&(&b.graph, &b.name, &b.property))
        });
        for ddl in &mut drops {
            ddl.create = false;
            ddl.rebuild = false;
            ddl.owner_schema = owner_schema.cloned();
        }
        self.pending_index_ddl.lock().extend(drops);
        Ok(())
    }

    /// Captures one coherent selector, canonical schema incarnation, concrete
    /// target, and tier-merged source for index DDL.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn active_index_ddl_target(
        &self,
    ) -> Result<(Option<SchemaIncarnation>, GraphPath, Arc<LpgStore>)> {
        let context = self.graph_context_snapshot();
        let _publication = self.publication_read_guard();
        let owner = self.schema_incarnation_for_name(context.schema.as_deref())?;
        let graph = if context.native {
            context.storage_key
        } else {
            Self::context_graph_path(
                owner.as_ref().map(|schema| schema.name.as_str()),
                context.graph.as_deref(),
            )?
        };
        let target = self.require_lpg_store_for_storage_key(&graph)?;
        Ok((owner, graph, target))
    }

    /// Stages index DDL into the same owner/physical commit as direct requests.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn stage_create_index(
        &self,
        name: Option<&str>,
        label: &str,
        property: &str,
        kind: PendingIndexKind,
        if_not_exists: bool,
    ) -> Result<bool> {
        let (owner_schema, graph, target) = self.active_index_ddl_target()?;
        self.stage_create_index_on(
            PendingIndexDdl {
                create: true,
                rebuild: false,
                expected_owner: None,
                configuration: None,
                owner_result: Arc::new(std::sync::OnceLock::new()),
                owner_schema,
                graph,
                target,
                name: name.map(ToString::to_string),
                label: label.to_string(),
                property: property.to_string(),
                kind,
            },
            if_not_exists,
        )
    }

    #[cfg(feature = "lpg")]
    fn stage_create_index_on(&self, ddl: PendingIndexDdl, if_not_exists: bool) -> Result<bool> {
        let name = ddl.name.as_deref();
        let label = ddl.label.as_str();
        let property = ddl.property.as_str();
        let kind = &ddl.kind;
        let graph = &ddl.graph;
        let target = &ddl.target;
        if !self.in_transaction() {
            return Err(Self::index_ddl_error(
                "index DDL requires an active transaction",
            ));
        }
        if property.is_empty()
            || name.is_some_and(|name| name.is_empty() || name.starts_with("@grafeo-index:"))
        {
            return Err(Self::index_ddl_error(
                "invalid index property or reserved/empty index name",
            ));
        }

        #[cfg(not(feature = "text-index"))]
        if matches!(kind, PendingIndexKind::Text { .. }) {
            return Err(Self::index_ddl_error(
                "Text index support requires the 'text-index' feature",
            ));
        }
        #[cfg(not(feature = "vector-index"))]
        if matches!(kind, PendingIndexKind::Vector { .. }) {
            return Err(Self::index_ddl_error(
                "Vector index support requires the 'vector-index' feature",
            ));
        }

        if let Some(name) = name {
            let staged = self
                .pending_index_ddl
                .lock()
                .iter()
                .rev()
                .find(|ddl| ddl.name.as_deref() == Some(name))
                .map(|ddl| ddl.create);
            let exists =
                staged.unwrap_or_else(|| self.catalog_view().find_index_by_name(name).is_some());
            if exists {
                if if_not_exists {
                    return Ok(false);
                }
                return Err(Self::index_ddl_error(format!(
                    "Index '{name}' already exists"
                )));
            }
        }

        let physical_key = Self::physical_index_key(graph, label, property, kind);
        let staged_target = self
            .pending_index_ddl
            .lock()
            .iter()
            .rev()
            .find(|ddl| {
                Self::physical_index_key(&ddl.graph, &ddl.label, &ddl.property, &ddl.kind)
                    == physical_key
            })
            .cloned();
        if staged_target.as_ref().is_some_and(|ddl| ddl.create) {
            if name.is_none() && if_not_exists {
                return Ok(false);
            }
            return Err(Self::index_ddl_error(format!(
                "Physical index target {} is already staged by '{}'",
                physical_key.property_name(),
                staged_target
                    .and_then(|ddl| ddl.name)
                    .unwrap_or_else(|| "an unnamed index".to_string())
            )));
        }
        if staged_target.is_none()
            && let Some(owner) = self.catalog_physical_index_owner(&physical_key, name)
        {
            return Err(Self::index_ddl_error(format!(
                "Index target already belongs to logical index '{owner}'; one physical index cannot have multiple names"
            )));
        }
        if staged_target.is_none() && Self::physical_index_exists(target, &physical_key) {
            if name.is_none() && if_not_exists {
                return Ok(false);
            }
            return Err(Self::index_ddl_error(
                "index target already has a physical index without the requested owner",
            ));
        }

        self.pending_index_ddl.lock().push(ddl);
        Ok(true)
    }

    /// Resolves and stages a named query index drop. The target definition is
    /// copied into the WAL so replay does not depend on an older catalog image.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn stage_drop_named_index(&self, name: &str, if_exists: bool) -> Result<bool> {
        if !self.in_transaction() {
            return Err(Self::index_ddl_error(
                "index DDL requires an active transaction",
            ));
        }

        let staged = self
            .pending_index_ddl
            .lock()
            .iter()
            .rev()
            .find(|ddl| ddl.name.as_deref() == Some(name))
            .cloned();
        let catalog_owner = self.catalog_view();
        let catalog = catalog_owner.read();
        let mut ddl = if let Some(ddl) = staged {
            self.require_index_path_write_grant(&ddl.graph)?;
            if !ddl.create {
                if if_exists {
                    return Ok(false);
                }
                return Err(Self::index_ddl_error(format!(
                    "Index '{name}' does not exist"
                )));
            }
            ddl
        } else if let Some(index_id) = catalog.find_index_by_name(name) {
            let definition = catalog.get_index(index_id).ok_or_else(|| {
                Self::index_ddl_error(format!("Index '{name}' disappeared from the catalog"))
            })?;
            let label = catalog
                .get_label_name(definition.label)
                .ok_or_else(|| Self::index_ddl_error("index label is missing from the catalog"))?
                .to_string();
            let property = catalog
                .get_property_key_name(definition.property_key)
                .ok_or_else(|| Self::index_ddl_error("index property is missing from the catalog"))?
                .to_string();
            let graph = definition.key.graph().clone();
            self.require_index_path_write_grant(&graph)?;
            let target = self.resolve_index_graph_path(&graph)?;
            PendingIndexDdl {
                create: false,
                expected_owner: Some(index_id),
                rebuild: false,
                configuration: Some(definition.configuration.clone()),
                owner_result: Arc::new(std::sync::OnceLock::new()),
                graph,
                owner_schema: None,
                name: Some(name.to_string()),
                label,
                property,
                kind: Self::pending_kind_from_catalog(definition.index_type),
                target,
            }
        } else if if_exists {
            return Ok(false);
        } else {
            return Err(Self::index_ddl_error(format!(
                "Index '{name}' does not exist"
            )));
        };
        drop(catalog);
        ddl.create = false;
        ddl.rebuild = false;

        self.pending_index_ddl.lock().push(ddl);
        Ok(true)
    }

    #[cfg(all(feature = "lpg", feature = "vector-index"))]
    fn build_pending_vector_contents(
        ddl: &PendingIndexDdl,
        nodes: &[Node],
    ) -> Result<grafeo_core::index::vector::VectorIndexKind> {
        use grafeo_core::index::vector::{
            DistanceMetric, HnswConfig, HnswIndex, QuantizationType, QuantizedHnswIndex,
            VectorIndexKind, value_to_vector,
        };

        let PendingIndexKind::Vector {
            dimensions,
            metric,
            m,
            ef_construction,
            ef,
            quantization,
        } = &ddl.kind
        else {
            return Err(Self::index_ddl_error("invalid vector index definition"));
        };

        let metric = match metric.as_deref() {
            Some(value) => DistanceMetric::from_str(value).ok_or_else(|| {
                Self::index_ddl_error(format!(
                    "Unknown distance metric '{value}'. Use: cosine, euclidean, dot_product, manhattan"
                ))
            })?,
            None => DistanceMetric::Cosine,
        };
        let quantization = match quantization.as_deref() {
            None => QuantizationType::None,
            Some(value) => QuantizationType::from_str(value).ok_or_else(|| {
                Self::index_ddl_error(format!(
                    "Unknown quantization type '{value}'. Use: scalar, binary, product, pqN"
                ))
            })?,
        };

        let retained = match &ddl.configuration {
            Some(IndexConfiguration::Vector {
                config,
                quantization,
            }) => Some((config, *quantization)),
            Some(_) => {
                return Err(Self::index_ddl_error(
                    "vector owner has a different configuration family",
                ));
            }
            None => None,
        };
        let mut found_dimensions =
            retained.map_or(*dimensions, |(config, _)| Some(config.dimensions));
        let mut vectors = Vec::new();
        for node in nodes.iter().filter(|node| node.has_label(&ddl.label)) {
            if let Some(value) = node.get_property(&ddl.property)
                && let Some(vector) = value_to_vector(value)
            {
                if let Some(expected) = found_dimensions {
                    if vector.len() != expected {
                        return Err(Self::index_ddl_error(format!(
                            "Vector dimension mismatch: expected {expected}, found {} on node {}",
                            vector.len(),
                            node.id.0
                        )));
                    }
                } else {
                    found_dimensions = Some(vector.len());
                }
                vectors.push((node.id, vector.to_vec()));
            }
        }
        let dimensions = found_dimensions.ok_or_else(|| {
            Self::index_ddl_error(format!(
                "No vector properties found on :{}({}) and no dimensions specified",
                ddl.label, ddl.property
            ))
        })?;

        let mut config = HnswConfig::new(dimensions, metric);
        if let Some(value) = m {
            config = config.with_m(*value);
        }
        if let Some(value) = ef_construction {
            config = config.with_ef_construction(*value);
        }
        if let Some(value) = ef {
            config = config.with_ef(*value);
        }

        let (config, quantization) = retained
            .map_or((config, quantization), |(config, quantization)| {
                (config.clone(), quantization)
            });
        IndexConfiguration::Vector {
            config: config.clone(),
            quantization,
        }
        .validate()
        .map_err(|error| Self::index_ddl_error(error.to_string()))?;

        let index = match quantization {
            QuantizationType::None => {
                let hnsw = HnswIndex::with_capacity(config, vectors.len());
                let vector_map: std::collections::HashMap<_, Arc<[f32]>> = vectors
                    .iter()
                    .map(|(id, vector)| (*id, Arc::from(vector.clone())))
                    .collect();
                let accessor = |id| vector_map.get(&id).cloned();
                for (node_id, vector) in &vectors {
                    hnsw.insert(*node_id, vector, &accessor);
                }
                VectorIndexKind::Hnsw(hnsw)
            }
            quantization => {
                let index = QuantizedHnswIndex::new(config, quantization);
                for (node_id, vector) in &vectors {
                    index.insert(*node_id, vector);
                }
                VectorIndexKind::Quantized(index)
            }
        };
        Ok(index)
    }

    /// Validates catalog concurrency and builds every fallible index object
    /// while the publication write lock is held, but before durable commit.
    #[cfg(feature = "lpg")]
    fn validate_pending_index_ddl_at(
        &self,
        catalog: CatalogRead<'_>,
        pending: &[PendingIndexDdl],
    ) -> Result<()> {
        #[cfg(test)]
        tests::INDEX_OWNER_SCANS.with(|scans| scans.set(scans.get() + 1));
        let created_graphs = self.pending_created_graphs.lock().clone();
        let dropped_graphs = self.pending_dropped_graphs.lock().clone();
        let cancelled_graphs = self.cancelled_created_graphs.lock().clone();
        let final_operations: std::collections::HashMap<_, _> = pending
            .iter()
            .enumerate()
            .map(|(ordinal, ddl)| {
                (
                    (
                        Arc::as_ptr(&ddl.target).addr(),
                        Self::physical_index_key(&ddl.graph, &ddl.label, &ddl.property, &ddl.kind),
                    ),
                    ordinal,
                )
            })
            .collect();
        let mut names: std::collections::HashMap<String, Option<grafeo_common::types::IndexId>> =
            catalog
                .all_indexes()
                .into_iter()
                .map(|definition| (definition.name, Some(definition.id)))
                .collect();
        let mut physical_owners: std::collections::HashMap<
            PhysicalIndexKey,
            std::collections::HashSet<String>,
        > = std::collections::HashMap::new();
        let mut removed_physical = std::collections::HashSet::new();
        for definition in catalog.all_indexes() {
            physical_owners
                .entry(definition.key)
                .or_default()
                .insert(definition.name);
        }

        for (ordinal, ddl) in pending.iter().enumerate() {
            if let Some(owner) = ddl.owner_schema.as_ref() {
                let default_key = format!("{}/{SCHEMA_DEFAULT_GRAPH}", owner.name);
                let default_path = Self::graph_path_for_storage_key(Some(&default_key))?;
                let live_default = created_graphs
                    .get(&default_path)
                    .map(|pending| Arc::clone(&pending.store))
                    .or_else(|| self.store.graph(&default_key));
                if !catalog.schema_exists(&owner.name)
                    || !live_default
                        .as_ref()
                        .is_some_and(|graph| Arc::ptr_eq(graph, &owner.default_graph))
                {
                    return Err(grafeo_common::utils::error::Error::Transaction(
                        grafeo_common::utils::error::TransactionError::WriteConflict(format!(
                            "owning schema '{}' was dropped or replaced while staging index DDL",
                            owner.name
                        )),
                    ));
                }
            }

            if ddl.create
                && final_operations.get(&(
                    Arc::as_ptr(&ddl.target).addr(),
                    Self::physical_index_key(&ddl.graph, &ddl.label, &ddl.property, &ddl.kind),
                )) == Some(&ordinal)
                && !Self::resolve_lpg_lifecycle_path(
                    &self.store,
                    &ddl.graph,
                    &created_graphs,
                    &dropped_graphs,
                    &cancelled_graphs,
                )
                .as_ref()
                .is_some_and(|target| Arc::ptr_eq(target, &ddl.target))
            {
                return Err(grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::WriteConflict(
                        "cannot publish an index into a graph dropped by the same transaction"
                            .to_string(),
                    ),
                ));
            }
            let retained_target =
                self.live_graph_path(&ddl.graph)
                    .as_ref()
                    .is_some_and(|target| Arc::ptr_eq(target, &ddl.target))
                    || created_graphs
                        .iter()
                        .map(|(path, pending)| (path, &pending.store))
                        .chain(dropped_graphs.iter())
                        .chain(cancelled_graphs.iter().flat_map(|(path, stores)| {
                            stores.iter().map(move |store| (path, store))
                        }))
                        .any(|(prefix, root)| {
                            if !ddl.graph.components().starts_with(prefix.components()) {
                                return false;
                            }
                            let mut target = Some(Arc::clone(root));
                            for name in &ddl.graph.components()[prefix.components().len()..] {
                                target = target.and_then(|target| target.graph(name));
                            }
                            target
                                .as_ref()
                                .is_some_and(|target| Arc::ptr_eq(target, &ddl.target))
                        });
            if !retained_target {
                return Err(grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::WriteConflict(format!(
                        "index target graph {:?} was dropped or replaced during the transaction",
                        ddl.graph
                    )),
                ));
            }

            if ddl.rebuild {
                Self::validate_index_rebuild_owner(catalog, ddl)?;
                continue;
            }
            let physical_key =
                Self::physical_index_key(&ddl.graph, &ddl.label, &ddl.property, &ddl.kind);
            if let Some(name) = &ddl.name {
                if ddl.create {
                    if names.insert(name.clone(), None).is_some() {
                        return Err(grafeo_common::utils::error::Error::Transaction(
                            grafeo_common::utils::error::TransactionError::WriteConflict(format!(
                                "index '{name}' changed during the transaction"
                            )),
                        ));
                    }
                } else if names.remove(name) != Some(ddl.expected_owner) {
                    return Err(grafeo_common::utils::error::Error::Transaction(
                        grafeo_common::utils::error::TransactionError::WriteConflict(format!(
                            "index '{name}' changed during the transaction"
                        )),
                    ));
                }
            }

            let owners = physical_owners.entry(physical_key.clone()).or_default();
            if ddl.create {
                if !owners.is_empty() {
                    return Err(grafeo_common::utils::error::Error::Transaction(
                        grafeo_common::utils::error::TransactionError::WriteConflict(format!(
                            "physical index target changed during the transaction; current logical owner(s): {}",
                            owners.iter().cloned().collect::<Vec<_>>().join(", ")
                        )),
                    ));
                }
                if ddl.name.is_some()
                    && !removed_physical.contains(&physical_key)
                    && Self::physical_index_exists(&ddl.target, &physical_key)
                {
                    return Err(grafeo_common::utils::error::Error::Transaction(
                        grafeo_common::utils::error::TransactionError::WriteConflict(
                            "an unnamed physical index appeared on the target during the transaction"
                                .to_string(),
                        ),
                    ));
                }
                owners.insert(ddl.name.clone().unwrap_or_else(|| "<unnamed>".to_string()));
                removed_physical.remove(&physical_key);
            } else if let Some(name) = &ddl.name {
                owners.remove(name);
                if owners.is_empty() {
                    removed_physical.insert(physical_key);
                }
            } else {
                let named_owners: Vec<_> = owners
                    .iter()
                    .filter(|owner| owner.as_str() != "<unnamed>")
                    .cloned()
                    .collect();
                if !named_owners.is_empty() {
                    return Err(Self::index_ddl_error(format!(
                        "Cannot drop an unnamed physical index while logical owner(s) remain: {}",
                        named_owners.join(", ")
                    )));
                }
                owners.remove("<unnamed>");
                removed_physical.insert(physical_key);
            }
        }
        Ok(())
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn schema_command_mode(
        cmd: &grafeo_adapters::query::gql::ast::SchemaStatement,
    ) -> SchemaCommandMode {
        use grafeo_adapters::query::gql::ast::SchemaStatement;

        match cmd {
            SchemaStatement::CreateVectorIndex(_)
            | SchemaStatement::CreateIndex(_)
            | SchemaStatement::DropIndex { .. } => SchemaCommandMode::TransactionalIndex,
            SchemaStatement::ShowConstraints
            | SchemaStatement::ShowIndexes
            | SchemaStatement::ShowNodeTypes
            | SchemaStatement::ShowEdgeTypes
            | SchemaStatement::ShowGraphTypes
            | SchemaStatement::ShowGraphType(_)
            | SchemaStatement::ShowCurrentGraphType
            | SchemaStatement::ShowGraphs
            | SchemaStatement::ShowSchemas => SchemaCommandMode::ReadOnlyShow,
            _ => SchemaCommandMode::StandaloneCatalog,
        }
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn schema_command_uses_current_schema(
        cmd: &grafeo_adapters::query::gql::ast::SchemaStatement,
    ) -> bool {
        use grafeo_adapters::query::gql::ast::SchemaStatement;

        match cmd {
            SchemaStatement::CreateNodeType(_)
            | SchemaStatement::CreateEdgeType(_)
            | SchemaStatement::DropNodeType { .. }
            | SchemaStatement::DropEdgeType { .. }
            | SchemaStatement::CreateConstraint(_)
            | SchemaStatement::DropConstraint { .. }
            | SchemaStatement::CreateGraphType(_)
            | SchemaStatement::DropGraphType { .. }
            | SchemaStatement::AlterNodeType(_)
            | SchemaStatement::AlterEdgeType(_)
            | SchemaStatement::AlterGraphType(_) => true,
            SchemaStatement::CreateVectorIndex(_)
            | SchemaStatement::CreateIndex(_)
            | SchemaStatement::DropIndex { .. }
            | SchemaStatement::CreateSchema { .. }
            | SchemaStatement::DropSchema { .. }
            | SchemaStatement::CreateProcedure(_)
            | SchemaStatement::DropProcedure { .. }
            | SchemaStatement::ShowConstraints
            | SchemaStatement::ShowIndexes
            | SchemaStatement::ShowNodeTypes
            | SchemaStatement::ShowEdgeTypes
            | SchemaStatement::ShowGraphTypes
            | SchemaStatement::ShowGraphType(_)
            | SchemaStatement::ShowCurrentGraphType
            | SchemaStatement::ShowGraphs
            | SchemaStatement::ShowSchemas => false,
        }
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn schema_command_requires_unrestricted_admin(
        cmd: &grafeo_adapters::query::gql::ast::SchemaStatement,
    ) -> bool {
        use grafeo_adapters::query::gql::ast::SchemaStatement;

        matches!(
            cmd,
            SchemaStatement::CreateProcedure(_) | SchemaStatement::DropProcedure { .. }
        )
    }

    /// Runs a schema statement at the strongest boundary its implementation
    /// can honestly support.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn execute_schema_command_checked(
        &self,
        cmd: grafeo_adapters::query::gql::ast::SchemaStatement,
    ) -> Result<QueryResult> {
        let mut catalog_workspace = CatalogWorkspace::new();
        let _operation = self.session_operation_guard();
        let _catalog_cuts = self.pin_catalog_cuts();
        let mode = Self::schema_command_mode(&cmd);
        let _historical = (mode != SchemaCommandMode::ReadOnlyShow)
            .then(|| self.historical_view_operation_gate.lock());
        if mode != SchemaCommandMode::ReadOnlyShow {
            self.reject_lpg_historical_mutation()?;
        }
        match mode {
            SchemaCommandMode::ReadOnlyShow => {
                self.require_permission(crate::auth::StatementKind::Read)?;
                let _publication = self.publication_read_guard();
                self.execute_schema_command(cmd)
            }
            SchemaCommandMode::TransactionalIndex => {
                self.require_permission(crate::auth::StatementKind::Admin)?;
                if *self.read_only_tx.lock() {
                    return Err(grafeo_common::utils::error::Error::Transaction(
                        grafeo_common::utils::error::TransactionError::ReadOnly,
                    ));
                }
                self.with_lpg_plan_auto_commit(true, false, || self.execute_schema_command(cmd))
            }
            SchemaCommandMode::StandaloneCatalog => {
                #[cfg(feature = "cdc")]
                let _operation = self.cdc_mutation_operation_guard();
                self.require_permission(crate::auth::StatementKind::Admin)?;
                if Self::schema_command_requires_unrestricted_admin(&cmd)
                    && self.identity.has_grants()
                {
                    return Err(grafeo_common::utils::error::Error::Query(
                        grafeo_common::utils::error::QueryError::new(
                            grafeo_common::utils::error::QueryErrorKind::Semantic,
                            format!(
                                "permission denied: global procedure DDL requires unrestricted administrator authority (user: {})",
                                self.identity.user_id()
                            ),
                        ),
                    ));
                }
                self.check_not_in_mixed_snapshot()?;
                if *self.read_only_tx.lock() {
                    return Err(grafeo_common::utils::error::Error::Transaction(
                        grafeo_common::utils::error::TransactionError::ReadOnly,
                    ));
                }

                if self.current_transaction_id().is_some() {
                    return self.with_lpg_plan_auto_commit(true, false, || {
                        self.execute_transaction_catalog_command(cmd, &mut catalog_workspace)
                    });
                }

                // Keep close() outside the whole standalone publication and
                // preserve the lifecycle -> publication global lock order.
                let database_open = self.database_open.read();
                if !*database_open {
                    return Err(Self::database_closed_error());
                }
                self.transaction_manager.with_write_authority(|| {
                    let _publication = self.transaction_manager.publication().write();
                    if self.current_transaction.lock().is_some() {
                        return Err(grafeo_common::utils::error::Error::Transaction(
                            grafeo_common::utils::error::TransactionError::InvalidState(
                                "transaction state changed during standalone catalog admission"
                                    .to_string(),
                            ),
                        ));
                    }

                    // Pin the invoking session's complete context for the
                    // prepare/WAL/publish interval. Public SESSION SET/RESET
                    // calls use this same mutex, so they cannot race a DDL
                    // statement into resolving names in one schema and then
                    // publishing a different session context.
                    let mut live_context = self.current_context.lock();
                    let catalog_edit = self.catalog.prepare_edit(&mut catalog_workspace)
                        .map_err(|error| Self::index_ddl_error(error.to_string()))?;
                    let staged_current_schema = if Self::schema_command_uses_current_schema(&cmd) {
                        match live_context.schema.as_deref() {
                            Some(requested) => Some(
                                catalog_edit.view()
                                    .schema_names()
                                    .into_iter()
                                    .find(|registered| {
                                        registered.eq_ignore_ascii_case(requested)
                                    })
                                    .ok_or_else(|| {
                                        grafeo_common::utils::error::Error::Query(
                                            grafeo_common::utils::error::QueryError::new(
                                                grafeo_common::utils::error::QueryErrorKind::Semantic,
                                                format!("Schema '{requested}' does not exist"),
                                            ),
                                        )
                                    })?,
                            ),
                            None => None,
                        }
                    } else {
                        live_context.schema.clone()
                    };
                    if Self::schema_command_uses_current_schema(&cmd) {
                        self.require_schema_graph_write_scope(staged_current_schema.as_deref(), catalog_edit.view())?;
                    }
                    let catalog_state_before = catalog_edit.view()
                        .ddl_comparison_state()
                        .map_err(grafeo_common::utils::error::Error::Serialization)?;
                    let graphs_before = self.store.named_graph_entries();

                    // All fallible DDL work happens against detached state.
                    // Existing graph Arcs remain exact and new schema
                    // partitions are allocated before the durable marker. Do
                    // not seal this private registry: its existing entries are
                    // shared Arcs, and an in-memory database deliberately
                    // preserves raw mutation capability. The live root applies
                    // its own sealing policy atomically during final install.
                    let staged_catalog = catalog_edit.candidate();
                    let staged_store = self.store.new_graph_topology_candidate().map_err(|error| {
                        grafeo_common::utils::error::Error::Internal(format!(
                            "failed to allocate detached catalog DDL store: {error}"
                        ))
                    })?;
                    staged_store.install_named_graphs(graphs_before.clone());
                    let staged_schema = parking_lot::Mutex::new(staged_current_schema);

                    #[cfg(feature = "wal")]
                    let wal_batch = CatalogWalBatchGuard::begin(&self.catalog_wal_batch)?;

                    let result = self.execute_schema_command_against(
                        cmd,
                        CatalogDdlTarget {
                            catalog: staged_catalog,
                            store: &staged_store,
                            current_schema: &staged_schema,
                            projections: &self.projections,
                        },
                        false,
                    );
                    #[cfg(feature = "wal")]
                    let _compatibility_records = wal_batch.finish();
                    let result = result?;

                    let comparison_state = staged_catalog
                        .ddl_comparison_state()
                        .map_err(grafeo_common::utils::error::Error::Serialization)?;
                    let graphs_after = staged_store.named_graph_entries();

                    // Catalog statements only create or drop graph partitions;
                    // replacing an existing incarnation would require a data
                    // post-image and is therefore rejected as an invariant
                    // violation before publication.
                    if graphs_before.iter().any(|(name, before)| {
                        graphs_after
                            .get(name)
                            .is_some_and(|after| !Arc::ptr_eq(before, after))
                    }) {
                        return Err(grafeo_common::utils::error::Error::Internal(
                            "catalog DDL replaced an existing named graph incarnation"
                                .to_string(),
                        ));
                    }

                    let mut created_graphs: Vec<GraphPath> = graphs_after
                        .keys()
                        .filter(|name| !graphs_before.contains_key(*name))
                        .map(|name| Self::graph_path_for_storage_key(Some(name)))
                        .collect::<Result<_>>()?;
                    let mut dropped_graphs: Vec<GraphPath> = graphs_before
                        .keys()
                        .filter(|name| !graphs_after.contains_key(*name))
                        .map(|name| Self::graph_path_for_storage_key(Some(name)))
                        .collect::<Result<_>>()?;
                    created_graphs.sort_unstable();
                    dropped_graphs.sort_unstable();

                    let changed = comparison_state != catalog_state_before
                        || !created_graphs.is_empty()
                        || !dropped_graphs.is_empty();
                    if !changed {
                        return Ok(result);
                    }

                    // In-memory change detection retains native coordinates.
                    // Only an attached WAL needs the current canonical wire DTO;
                    // encode its complete checked image before publication.
                    #[cfg(feature = "wal")]
                    let catalog_state = if self.wal.is_some() {
                        staged_catalog.encode_wal_state_v1()
                            .map_err(grafeo_common::utils::error::Error::Serialization)?
                    } else {
                        Vec::new()
                    };

                    // Schema changes re-resolve only language selectors.
                    // Native paths remain exact, and every fallible path
                    // derivation precedes the durable publication marker.
                    let mut current_context = live_context.clone();
                    current_context.schema = staged_schema.into_inner();
                    if !current_context.native {
                        current_context.storage_key = Self::context_graph_path(
                            current_context.schema.as_deref(),
                            current_context.graph.as_deref(),
                        )?;
                    }

                    // Catalog publication advances every surviving graph, not
                    // just immediate children. Capture before WAL so the
                    // post-marker epoch publication remains infallible.
                    let mut epoch_targets = Vec::new();
                    for graph in graphs_after.values() {
                        for (_, descendant) in
                            grafeo_core::graph::lpg::LpgStoreSection::new(Arc::clone(graph))
                                .capture_graphs()?
                        {
                            epoch_targets.push(descendant);
                        }
                    }

                    // This value is the proof boundary: every statement step
                    // that can return an error is complete before WAL.
                    let prepared = PreparedCatalogDdl {
                        result,
                        catalog: catalog_edit.finish(),
                        epoch_targets,
                        graph_entries: graphs_after,
                        current_context,
                        #[cfg(feature = "wal")]
                        catalog_state,
                        #[cfg(feature = "wal")]
                        created_graphs,
                        #[cfg(feature = "wal")]
                        dropped_graphs,
                    };

                    let epoch = self.transaction_manager.reserve_publication_epoch()?;
                    #[cfg(feature = "wal")]
                    let wal_record = grafeo_storage::wal::WalRecord::CatalogBatchV3 {
                        version: 2,
                        created_graph_incarnations: prepared.created_graphs.iter().map(|path| prepared.graph_entries[path.components().last().expect("prepared named path")].graph_incarnation_id()).collect(),
                        dropped_graph_incarnations: prepared.dropped_graphs.iter().map(|path| graphs_before[path.components().last().expect("prepared named path")].graph_incarnation_id()).collect(),
                        epoch,
                        catalog_state: prepared.catalog_state,
                        created_graphs: prepared.created_graphs,
                        dropped_graphs: prepared.dropped_graphs,
                    };
                    #[cfg(feature = "wal")]
                    self.log_wal_record(&wal_record)?;

                    // WAL has acknowledged the complete post-image. The
                    // remaining installers cannot return statement errors and
                    // no statement logic is re-executed. A fatal interruption
                    // during installation is repaired from this WAL frame on
                    // reopen.
                    let catalog_fence = prepared.catalog.install();
                    self.store.install_named_graphs(prepared.graph_entries);
                    *live_context = prepared.current_context;
                    self.store.sync_epoch(epoch);
                    for graph in prepared.epoch_targets {
                        graph.sync_epoch(epoch);
                    }
                    #[cfg(feature = "triple-store")]
                    if let Err(error) = self.rdf_store.try_set_commit_epoch(epoch) {
                        self.poison_durability();
                        return Err(grafeo_common::utils::error::Error::Transaction(
                            grafeo_common::utils::error::TransactionError::DurabilityFailure(
                                format!(
                                    "durable catalog publication could not advance the RDF clock: {error}; reopen and recover before continuing"
                                ),
                            ),
                        ));
                    }
                    self.query_cache.clear();
                    self.physical_cache.lock().clear();
                    self.transaction_manager.publish_reserved_epoch(epoch);
                    catalog_fence.finish();
                    Ok(prepared.result)
                })
            }
        }
    }

    /// Executes a schema command after its transaction/publication boundary
    /// has been selected by [`Self::execute_schema_command_checked`].
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn execute_schema_command(
        &self,
        cmd: grafeo_adapters::query::gql::ast::SchemaStatement,
    ) -> Result<QueryResult> {
        // Only SHOW/index statements use the live catalog wrapper. Their
        // schema target is a detached snapshot, leaving the unified context
        // available to the query/index helpers they invoke.
        let current_schema = parking_lot::Mutex::new(self.current_schema());
        self.execute_schema_command_against(
            cmd,
            CatalogDdlTarget {
                catalog: &self.catalog_view(),
                store: &self.store,
                current_schema: &current_schema,
                projections: &self.projections,
            },
            true,
        )
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn execute_schema_command_against(
        &self,
        cmd: grafeo_adapters::query::gql::ast::SchemaStatement,
        target: CatalogDdlTarget<'_>,
        invalidate_live_cache: bool,
    ) -> Result<QueryResult> {
        use crate::catalog::{
            EdgeTypeDefinition, NamedConstraintDefinition, NamedConstraintKind, NodeTypeDefinition,
            PropertyDataType, TypedProperty,
        };
        use grafeo_adapters::query::gql::ast::SchemaStatement;
        #[cfg(feature = "wal")]
        use grafeo_common::types::TransactionId;
        use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};
        #[cfg(feature = "wal")]
        use grafeo_storage::wal::WalRecord;

        if !std::ptr::eq(target.catalog, self.catalog_view().as_ref())
            && Self::schema_command_mode(&cmd) != SchemaCommandMode::StandaloneCatalog
        {
            return Err(Error::Internal(
                "only standalone catalog DDL may execute against detached state".to_string(),
            ));
        }

        /// Logs a WAL record for schema changes. Compiles to nothing without `wal`.
        /// Fail-closed: WAL errors poison the session and abort the statement.
        macro_rules! wal_log {
            ($self:expr, $record:expr) => {
                #[cfg(feature = "wal")]
                $self.log_schema_wal(&$record)?;
            };
        }

        let result = match cmd {
            SchemaStatement::CreateNodeType(stmt) => {
                let effective_name = target.effective_type_key(&stmt.name);
                #[cfg(feature = "wal")]
                let props_for_wal: Vec<(String, String, bool)> = stmt
                    .properties
                    .iter()
                    .map(|p| (p.name.clone(), p.data_type.clone(), p.nullable))
                    .collect();
                let def = NodeTypeDefinition {
                    name: effective_name.clone(),
                    properties: stmt
                        .properties
                        .iter()
                        .map(|p| TypedProperty {
                            name: p.name.clone(),
                            data_type: PropertyDataType::from_type_name(&p.data_type),
                            nullable: p.nullable,
                            default_value: p
                                .default_value
                                .as_ref()
                                .map(|s| parse_default_literal(s)),
                        })
                        .collect(),
                    constraints: Vec::new(),
                    parent_types: stmt.parent_types.clone(),
                };
                let result = if stmt.or_replace {
                    target.catalog.register_or_replace_node_type(def);
                    Ok(())
                } else {
                    target.catalog.register_node_type(def)
                };
                match result {
                    Ok(()) => {
                        wal_log!(
                            self,
                            WalRecord::CreateNodeType {
                                name: effective_name.clone(),
                                properties: props_for_wal,
                                constraints: Vec::new(),
                            }
                        );
                        self.bounded_status(format_args!("Created node type '{}'", stmt.name))
                    }
                    Err(crate::catalog::CatalogError::TypeAlreadyExists(_))
                        if stmt.if_not_exists =>
                    {
                        self.bounded_status(format_args!("{}", "No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::CreateEdgeType(stmt) => {
                let effective_name = target.effective_type_key(&stmt.name);
                #[cfg(feature = "wal")]
                let props_for_wal: Vec<(String, String, bool)> = stmt
                    .properties
                    .iter()
                    .map(|p| (p.name.clone(), p.data_type.clone(), p.nullable))
                    .collect();
                let def = EdgeTypeDefinition {
                    name: effective_name.clone(),
                    properties: stmt
                        .properties
                        .iter()
                        .map(|p| TypedProperty {
                            name: p.name.clone(),
                            data_type: PropertyDataType::from_type_name(&p.data_type),
                            nullable: p.nullable,
                            default_value: p
                                .default_value
                                .as_ref()
                                .map(|s| parse_default_literal(s)),
                        })
                        .collect(),
                    constraints: Vec::new(),
                    source_node_types: stmt.source_node_types.clone(),
                    target_node_types: stmt.target_node_types.clone(),
                };
                let result = if stmt.or_replace {
                    target.catalog.register_or_replace_edge_type_def(def);
                    Ok(())
                } else {
                    target.catalog.register_edge_type_def(def)
                };
                match result {
                    Ok(()) => {
                        wal_log!(
                            self,
                            WalRecord::CreateEdgeType {
                                name: effective_name.clone(),
                                properties: props_for_wal,
                                constraints: Vec::new(),
                            }
                        );
                        self.bounded_status(format_args!("Created edge type '{}'", stmt.name))
                    }
                    Err(crate::catalog::CatalogError::TypeAlreadyExists(_))
                        if stmt.if_not_exists =>
                    {
                        self.bounded_status(format_args!("{}", "No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::CreateVectorIndex(stmt) => {
                self.stage_create_index(
                    Some(&stmt.name),
                    &stmt.node_label,
                    &stmt.property,
                    PendingIndexKind::Vector {
                        dimensions: stmt.dimensions,
                        metric: stmt.metric,
                        m: None,
                        ef_construction: None,
                        ef: None,
                        quantization: None,
                    },
                    false,
                )?;
                self.bounded_status(format_args!("Created vector index '{}'", stmt.name))
            }
            SchemaStatement::DropNodeType { name, if_exists } => {
                let effective_name = target.effective_type_key(&name);
                match target.catalog.drop_node_type(&effective_name) {
                    Ok(()) => {
                        wal_log!(
                            self,
                            WalRecord::DropNodeType {
                                name: effective_name
                            }
                        );
                        self.bounded_status(format_args!("Dropped node type '{name}'"))
                    }
                    Err(e) if if_exists => {
                        let _ = e;
                        self.bounded_status(format_args!("{}", "No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::DropEdgeType { name, if_exists } => {
                let effective_name = target.effective_type_key(&name);
                match target.catalog.drop_edge_type_def(&effective_name) {
                    Ok(()) => {
                        wal_log!(
                            self,
                            WalRecord::DropEdgeType {
                                name: effective_name
                            }
                        );
                        self.bounded_status(format_args!("Dropped edge type '{name}'"))
                    }
                    Err(e) if if_exists => {
                        let _ = e;
                        self.bounded_status(format_args!("{}", "No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::CreateIndex(stmt) => {
                use grafeo_adapters::query::gql::ast::IndexKind;
                if stmt.properties.len() != 1 {
                    return Err(Self::index_ddl_error(
                        "Composite indexes are not supported yet; specify exactly one property",
                    ));
                }
                let index_type_str = match stmt.index_kind {
                    IndexKind::Property => "property",
                    IndexKind::BTree => "btree",
                    IndexKind::Text => "text",
                    IndexKind::Vector => "vector",
                };
                let kind = match stmt.index_kind {
                    IndexKind::Property => PendingIndexKind::Property,
                    IndexKind::BTree => PendingIndexKind::BTree,
                    IndexKind::Text => PendingIndexKind::Text {
                        min_token_length: stmt.options.min_token_length,
                    },
                    IndexKind::Vector => PendingIndexKind::Vector {
                        dimensions: stmt.options.dimensions,
                        metric: stmt.options.metric.clone(),
                        m: None,
                        ef_construction: None,
                        ef: stmt.options.ef,
                        quantization: None,
                    },
                };
                let changed = self.stage_create_index(
                    Some(&stmt.name),
                    &stmt.label,
                    &stmt.properties[0],
                    kind,
                    stmt.if_not_exists,
                )?;
                if !changed {
                    return self.bounded_status(format_args!("{}", "No change"));
                }
                self.bounded_status(format_args!(
                    "Created {} index '{}'",
                    index_type_str, stmt.name
                ))
            }
            SchemaStatement::DropIndex { name, if_exists } => {
                if self.stage_drop_named_index(&name, if_exists)? {
                    self.bounded_status(format_args!("Dropped index '{name}'"))
                } else {
                    self.bounded_status(format_args!("{}", "No change"))
                }
            }
            SchemaStatement::CreateConstraint(stmt) => {
                use grafeo_adapters::query::gql::ast::ConstraintKind;
                let kind = match stmt.constraint_kind {
                    ConstraintKind::Unique => NamedConstraintKind::Unique,
                    ConstraintKind::NodeKey => NamedConstraintKind::NodeKey,
                    ConstraintKind::NotNull => NamedConstraintKind::NotNull,
                    ConstraintKind::Exists => NamedConstraintKind::Exists,
                };
                let display_name = stmt.name.clone().unwrap_or_else(|| {
                    // Anonymous constraint names are durable catalog identity,
                    // so derive them from the complete normalized target rather
                    // than only label+kind (which made two properties collide).
                    let mut properties = stmt.properties.clone();
                    properties.sort();
                    let signature = format!(
                        "{}\u{0}{}\u{0}{}",
                        stmt.label,
                        kind.as_str(),
                        properties.join("\u{0}")
                    );
                    format!(
                        "{}_{}_{:016x}",
                        stmt.label,
                        kind.as_str(),
                        prop_tag(&signature)
                    )
                });
                let definition = NamedConstraintDefinition {
                    name: target.effective_type_key(&display_name),
                    label: target.effective_type_key(&stmt.label),
                    properties: stmt.properties.clone(),
                    kind,
                };

                if target
                    .catalog
                    .get_named_constraint(&definition.name)
                    .is_some()
                    || target.catalog.has_equivalent_named_constraint(&definition)
                {
                    if stmt.if_not_exists {
                        return self.bounded_status(format_args!("{}", "No change"));
                    }
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Constraint '{}' already exists", display_name),
                    )));
                }

                self.validate_named_constraint_existing_data(
                    &definition,
                    target.catalog.read().view(),
                )?;
                target
                    .catalog
                    .create_named_constraint(definition.clone())
                    .map_err(|error| {
                        Error::Query(QueryError::new(QueryErrorKind::Semantic, error.to_string()))
                    })?;

                wal_log!(
                    self,
                    WalRecord::CreateConstraint {
                        name: definition.name,
                        label: definition.label,
                        properties: stmt.properties.clone(),
                        kind: kind.as_str().to_string(),
                    }
                );
                self.bounded_status(format_args!(
                    "Created {} constraint '{display_name}'",
                    kind.as_str()
                ))
            }
            SchemaStatement::DropConstraint { name, if_exists } => {
                let effective_name = target.effective_type_key(&name);
                match target.catalog.drop_named_constraint(&effective_name) {
                    Ok(_) => {}
                    Err(crate::catalog::CatalogError::ConstraintNotFound(_)) if if_exists => {
                        return self.bounded_status(format_args!("{}", "No change"));
                    }
                    Err(error) => {
                        return Err(Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            error.to_string(),
                        )));
                    }
                }
                wal_log!(
                    self,
                    WalRecord::DropConstraint {
                        name: effective_name
                    }
                );
                self.bounded_status(format_args!("Dropped constraint '{name}'"))
            }
            SchemaStatement::CreateGraphType(stmt) => {
                use crate::catalog::GraphTypeDefinition;
                use grafeo_adapters::query::gql::ast::InlineElementType;

                let effective_name = target.effective_type_key(&stmt.name);

                // GG04: LIKE clause copies type from existing graph
                let (mut node_types, mut edge_types, open) = if let Some(ref like_graph) =
                    stmt.like_graph
                {
                    let source_key = Self::storage_key_for_context(
                        target.current_schema.lock().as_deref(),
                        Some(like_graph),
                    );
                    // Authorize the exact canonical coordinate before
                    // existence or binding lookup so LIKE cannot be used as
                    // a graph/type metadata oracle.
                    self.require_graph_read_grant(source_key.as_deref())?;
                    if let Some(source_key) = source_key.as_deref()
                        && target.store.graph(source_key).is_none()
                    {
                        return Err(Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            format!("Source graph '{like_graph}' does not exist"),
                        )));
                    }

                    // Infer types from the exact graph's bound type, or—
                    // for an untyped graph—from definitions in that
                    // graph's schema only. A schema-local LIKE must never
                    // clone a root binding with the same local name or
                    // absorb definitions from unrelated schemas.
                    if let Some(type_name) =
                        target
                            .catalog
                            .get_graph_type_binding(&Self::graph_path_for_storage_key(
                                source_key.as_deref(),
                            )?)
                    {
                        let existing = target
                                .catalog
                                .get_graph_type_def(&type_name)
                                .ok_or_else(|| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        format!(
                                            "Source graph '{like_graph}' has an invalid graph type binding"
                                        ),
                                    ))
                                })?;
                        (
                            existing.allowed_node_types.clone(),
                            existing.allowed_edge_types.clone(),
                            existing.open,
                        )
                    } else {
                        // GG22: infer from registered element types, but
                        // only within the source graph's namespace.
                        let registered_schemas = target.catalog.schema_names();
                        let source_schema = source_key.as_deref().and_then(|key| {
                            key.split_once('/').and_then(|(prefix, _)| {
                                registered_schemas
                                    .iter()
                                    .find(|schema| schema.eq_ignore_ascii_case(prefix))
                            })
                        });
                        let belongs = |name: &str| {
                            let type_schema = name.split_once('/').and_then(|(prefix, _)| {
                                registered_schemas
                                    .iter()
                                    .find(|schema| schema.eq_ignore_ascii_case(prefix))
                            });
                            match (source_schema, type_schema) {
                                (None, None) => true,
                                (Some(expected), Some(actual)) => {
                                    actual.eq_ignore_ascii_case(expected)
                                }
                                (None, Some(_)) | (Some(_), None) => false,
                            }
                        };
                        let nt = target
                            .catalog
                            .all_node_type_names()
                            .into_iter()
                            .filter(|name| belongs(name))
                            .collect::<Vec<_>>();
                        let et = target
                            .catalog
                            .all_edge_type_names()
                            .into_iter()
                            .filter(|name| belongs(name))
                            .collect::<Vec<_>>();
                        if nt.is_empty() && et.is_empty() {
                            (Vec::new(), Vec::new(), true)
                        } else {
                            (nt, et, false)
                        }
                    }
                } else {
                    // Prefix element type names with schema for consistency
                    let nt = stmt
                        .node_types
                        .iter()
                        .map(|n| target.effective_type_key(n))
                        .collect();
                    let et = stmt
                        .edge_types
                        .iter()
                        .map(|n| target.effective_type_key(n))
                        .collect();
                    (nt, et, stmt.open)
                };

                // GG03: Process inline element type entries. Per ISO/IEC 39075,
                // a bare `NODE TYPE Name` or `EDGE TYPE Name` inside a graph
                // type body is a reference; anything with a property block or
                // a KEY clause is an inline declaration. See issue #316.
                for inline in &stmt.inline_types {
                    match inline {
                        InlineElementType::Node {
                            name,
                            properties,
                            key_labels,
                            is_reference,
                            ..
                        } => {
                            let inline_effective = target.effective_type_key(name);
                            if *is_reference {
                                // Reference: validate existence; do not register, do not WAL.
                                if target.catalog.get_node_type(&inline_effective).is_none() {
                                    return Err(Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        format!(
                                            "Referenced node type '{inline_effective}' does not exist"
                                        ),
                                    )));
                                }
                            } else {
                                let def = NodeTypeDefinition {
                                    name: inline_effective.clone(),
                                    properties: properties
                                        .iter()
                                        .map(|p| TypedProperty {
                                            name: p.name.clone(),
                                            data_type: PropertyDataType::from_type_name(
                                                &p.data_type,
                                            ),
                                            nullable: p.nullable,
                                            default_value: None,
                                        })
                                        .collect(),
                                    constraints: Vec::new(),
                                    parent_types: key_labels.clone(),
                                };
                                target.catalog.register_or_replace_node_type(def);
                                #[cfg(feature = "wal")]
                                {
                                    let props_for_wal: Vec<(String, String, bool)> = properties
                                        .iter()
                                        .map(|p| (p.name.clone(), p.data_type.clone(), p.nullable))
                                        .collect();
                                    self.log_schema_wal(&WalRecord::CreateNodeType {
                                        name: inline_effective.clone(),
                                        properties: props_for_wal,
                                        constraints: Vec::new(),
                                    })?;
                                }
                            }
                            if !node_types.contains(&inline_effective) {
                                node_types.push(inline_effective);
                            }
                        }
                        InlineElementType::Edge {
                            name,
                            properties,
                            source_node_types,
                            target_node_types,
                            is_reference,
                            ..
                        } => {
                            let inline_effective = target.effective_type_key(name);
                            if *is_reference {
                                if target
                                    .catalog
                                    .get_edge_type_def(&inline_effective)
                                    .is_none()
                                {
                                    return Err(Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        format!(
                                            "Referenced edge type '{inline_effective}' does not exist"
                                        ),
                                    )));
                                }
                            } else {
                                let def = EdgeTypeDefinition {
                                    name: inline_effective.clone(),
                                    properties: properties
                                        .iter()
                                        .map(|p| TypedProperty {
                                            name: p.name.clone(),
                                            data_type: PropertyDataType::from_type_name(
                                                &p.data_type,
                                            ),
                                            nullable: p.nullable,
                                            default_value: None,
                                        })
                                        .collect(),
                                    constraints: Vec::new(),
                                    source_node_types: source_node_types.clone(),
                                    target_node_types: target_node_types.clone(),
                                };
                                target.catalog.register_or_replace_edge_type_def(def);
                                #[cfg(feature = "wal")]
                                {
                                    let props_for_wal: Vec<(String, String, bool)> = properties
                                        .iter()
                                        .map(|p| (p.name.clone(), p.data_type.clone(), p.nullable))
                                        .collect();
                                    self.log_schema_wal(&WalRecord::CreateEdgeType {
                                        name: inline_effective.clone(),
                                        properties: props_for_wal,
                                        constraints: Vec::new(),
                                    })?;
                                }
                            }
                            if !edge_types.contains(&inline_effective) {
                                edge_types.push(inline_effective);
                            }
                        }
                    }
                }

                let def = GraphTypeDefinition {
                    name: effective_name.clone(),
                    allowed_node_types: node_types.clone(),
                    allowed_edge_types: edge_types.clone(),
                    open,
                };
                let result = if stmt.or_replace {
                    target.catalog.register_or_replace_graph_type(def);
                    Ok(())
                } else {
                    target.catalog.register_graph_type(def)
                };
                match result {
                    Ok(()) => {
                        wal_log!(
                            self,
                            WalRecord::CreateGraphType {
                                name: effective_name.clone(),
                                node_types,
                                edge_types,
                                open,
                            }
                        );
                        self.bounded_status(format_args!("Created graph type '{}'", stmt.name))
                    }
                    Err(crate::catalog::CatalogError::TypeAlreadyExists(_))
                        if stmt.if_not_exists =>
                    {
                        self.bounded_status(format_args!("{}", "No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::DropGraphType { name, if_exists } => {
                let effective_name = target.effective_type_key(&name);
                let bound_graphs: Vec<String> = target
                    .graph_type_bindings(self)
                    .into_iter()
                    .filter_map(|(graph, graph_type)| {
                        (graph_type == effective_name).then(|| format!("{graph:?}"))
                    })
                    .collect();
                if !bound_graphs.is_empty() {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!(
                            "Cannot drop graph type '{name}' while it is bound to graph(s): {}",
                            bound_graphs.join(", ")
                        ),
                    )));
                }
                match target.catalog.drop_graph_type(&effective_name) {
                    Ok(()) => {
                        wal_log!(
                            self,
                            WalRecord::DropGraphType {
                                name: effective_name
                            }
                        );
                        self.bounded_status(format_args!("Dropped graph type '{name}'"))
                    }
                    Err(e) if if_exists => {
                        let _ = e;
                        self.bounded_status(format_args!("{}", "No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::CreateSchema {
                name,
                if_not_exists,
            } => {
                if name.is_empty() {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "Schema name must not be empty",
                    )));
                }
                if name.contains('/') {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!(
                            "Schema name '{name}' must not contain '/' (reserved as schema/graph separator)"
                        ),
                    )));
                }
                let existing = target
                    .catalog
                    .schema_names()
                    .into_iter()
                    .find(|registered| registered.eq_ignore_ascii_case(&name));
                let canonical_name = existing.as_deref().unwrap_or(&name);
                let default_key = format!("{canonical_name}/{SCHEMA_DEFAULT_GRAPH}");
                self.require_graph_write_grant(Some(&default_key))?;
                if let Some(existing) = existing {
                    if if_not_exists {
                        return self.bounded_status(format_args!("{}", "No change"));
                    }
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Schema '{name}' conflicts with existing schema '{existing}'"),
                    )));
                }
                if target.store.graph_names().iter().any(|graph| {
                    graph
                        .split_once('/')
                        .is_some_and(|(prefix, _)| prefix.eq_ignore_ascii_case(&name))
                }) {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!(
                            "Schema '{name}' cannot claim a namespace containing existing root graphs"
                        ),
                    )));
                }
                if target.has_schema_owned_catalog_objects(&name, self) {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!(
                            "Schema '{name}' cannot claim a namespace containing existing catalog objects"
                        ),
                    )));
                }
                match target.catalog.register_schema_namespace(name.clone()) {
                    Ok(()) => {
                        wal_log!(self, WalRecord::CreateSchema { name: name.clone() });
                        // Auto-create the schema's default graph partition so that
                        // SESSION SET SCHEMA + queries work without an explicit graph.
                        let created = target.store.create_graph(&default_key).map_err(|error| {
                            Error::Internal(format!(
                                "failed to allocate default graph for schema '{name}': {error}"
                            ))
                        })?;
                        if created {
                            wal_log!(
                                self,
                                WalRecord::CreateLpgGraph {
                                    incarnation: target
                                        .store
                                        .graph(&default_key)
                                        .ok_or_else(|| Error::Internal(
                                            "created schema graph is absent".into()
                                        ))?
                                        .graph_incarnation_id(),
                                    graph: Self::graph_path_for_storage_key(Some(&default_key))?,
                                    transaction_id: self
                                        .current_transaction_id()
                                        .unwrap_or(TransactionId::SYSTEM),
                                }
                            );
                        }
                        self.bounded_status(format_args!("Created schema '{name}'"))
                    }
                    Err(crate::catalog::CatalogError::SchemaAlreadyExists(_)) if if_not_exists => {
                        self.bounded_status(format_args!("{}", "No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::DropSchema { name, if_exists } => {
                let requested_name = name;
                if requested_name.is_empty() {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "Schema name must not be empty",
                    )));
                }
                if requested_name.contains('/') {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!(
                            "Schema name '{requested_name}' must not contain '/' (reserved as schema/graph separator)"
                        ),
                    )));
                }
                let registered = target
                    .catalog
                    .schema_names()
                    .into_iter()
                    .find(|registered| registered.eq_ignore_ascii_case(&requested_name));
                let canonical_name = registered.as_deref().unwrap_or(&requested_name);
                let default_graph_key = format!("{canonical_name}/{SCHEMA_DEFAULT_GRAPH}");
                self.require_graph_write_grant(Some(&default_graph_key))?;
                let Some(name) = registered else {
                    if if_exists {
                        return self.bounded_status(format_args!("{}", "No change"));
                    }
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Schema '{requested_name}' does not exist"),
                    )));
                };
                // ISO/IEC 39075 Section 12.3: schema must be empty before dropping.
                // The auto-created __default__ partition is exempt only when
                // it contains no data, physical indexes or child graphs. Any
                // immediate child makes the schema nonempty, even if that
                // child's entire subtree currently contains no rows.
                let default_graph = target.store.graph(&default_graph_key);
                let default_path = Self::graph_path_for_storage_key(Some(&default_graph_key))?;
                let has_default_data_or_graphs = match default_graph.as_ref() {
                    Some(graph) => {
                        Self::catalog_graph_has_data(
                            graph,
                            self.transaction_manager.current_epoch(),
                            self.current_transaction_id(),
                        )? || self.catalog_graphs()?.iter().any(|(path, _)| {
                            path != &default_path
                                && path.components().starts_with(default_path.components())
                        })
                    }
                    None => false,
                };
                let has_default_physical_indexes = match default_graph.as_ref() {
                    Some(graph) if !self.in_transaction() => {
                        // Standalone DDL already owns the live catalog writer.
                        !Self::physical_index_keys_for_store(&default_path, graph).is_empty()
                    }
                    Some(graph) => {
                        self.graph_indexes_after_pending_ddl(
                            &default_path,
                            graph,
                            &self.pending_index_ddl.lock(),
                        )?
                        .1
                    }
                    None => false,
                };
                let has_graphs = target.store.graph_names().iter().any(|graph| {
                    CatalogDdlTarget::key_belongs_to_schema(graph, &name)
                        && *graph != default_graph_key
                });
                let has_catalog_objects = target.has_schema_owned_catalog_objects(&name, self);
                if has_default_data_or_graphs
                    || has_default_physical_indexes
                    || has_graphs
                    || has_catalog_objects
                {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!(
                            "Schema '{name}' is not empty: remove default-graph data and drop all graphs, projections, types, constraints, bindings, and indexes first"
                        ),
                    )));
                }
                match target.catalog.drop_schema_namespace(&name) {
                    Ok(()) => {
                        wal_log!(self, WalRecord::DropSchema { name: name.clone() });
                        // Drop the auto-created default graph partition
                        #[cfg(feature = "wal")]
                        let dropped_incarnation = target
                            .store
                            .graph(&default_graph_key)
                            .map(|graph| graph.graph_incarnation_id());
                        if target.store.drop_graph(&default_graph_key) {
                            wal_log!(
                                self,
                                WalRecord::DropLpgGraph {
                                    incarnation: dropped_incarnation.ok_or_else(|| {
                                        Error::Internal(
                                            "dropped schema graph had no incarnation".into(),
                                        )
                                    })?,
                                    graph: Self::graph_path_for_storage_key(Some(
                                        &default_graph_key
                                    ))?,
                                    transaction_id: self
                                        .current_transaction_id()
                                        .unwrap_or(TransactionId::SYSTEM),
                                }
                            );
                        }
                        // If this session was using the dropped schema, reset it
                        let mut current = target.current_schema.lock();
                        if current
                            .as_deref()
                            .is_some_and(|s| s.eq_ignore_ascii_case(&name))
                        {
                            *current = None;
                        }
                        self.bounded_status(format_args!("Dropped schema '{name}'"))
                    }
                    Err(e) if if_exists => {
                        let _ = e;
                        self.bounded_status(format_args!("{}", "No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::AlterNodeType(stmt) => {
                use grafeo_adapters::query::gql::ast::TypeAlteration;
                let effective_name = target.effective_type_key(&stmt.name);
                let mut wal_alts = Vec::new();
                for alt in &stmt.alterations {
                    match alt {
                        TypeAlteration::AddProperty(prop) => {
                            let typed = TypedProperty {
                                name: prop.name.clone(),
                                data_type: PropertyDataType::from_type_name(&prop.data_type),
                                nullable: prop.nullable,
                                default_value: prop
                                    .default_value
                                    .as_ref()
                                    .map(|s| parse_default_literal(s)),
                            };
                            target
                                .catalog
                                .alter_node_type_add_property(&effective_name, typed)
                                .map_err(|e| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        e.to_string(),
                                    ))
                                })?;
                            wal_alts.push((
                                "add".to_string(),
                                prop.name.clone(),
                                prop.data_type.clone(),
                                prop.nullable,
                            ));
                        }
                        TypeAlteration::DropProperty(name) => {
                            target
                                .catalog
                                .alter_node_type_drop_property(&effective_name, name)
                                .map_err(|e| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        e.to_string(),
                                    ))
                                })?;
                            wal_alts.push(("drop".to_string(), name.clone(), String::new(), false));
                        }
                    }
                }
                wal_log!(
                    self,
                    WalRecord::AlterNodeType {
                        name: effective_name,
                        alterations: wal_alts,
                    }
                );
                self.bounded_status(format_args!("Altered node type '{}'", stmt.name))
            }
            SchemaStatement::AlterEdgeType(stmt) => {
                use grafeo_adapters::query::gql::ast::TypeAlteration;
                let effective_name = target.effective_type_key(&stmt.name);
                let mut wal_alts = Vec::new();
                for alt in &stmt.alterations {
                    match alt {
                        TypeAlteration::AddProperty(prop) => {
                            let typed = TypedProperty {
                                name: prop.name.clone(),
                                data_type: PropertyDataType::from_type_name(&prop.data_type),
                                nullable: prop.nullable,
                                default_value: prop
                                    .default_value
                                    .as_ref()
                                    .map(|s| parse_default_literal(s)),
                            };
                            target
                                .catalog
                                .alter_edge_type_add_property(&effective_name, typed)
                                .map_err(|e| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        e.to_string(),
                                    ))
                                })?;
                            wal_alts.push((
                                "add".to_string(),
                                prop.name.clone(),
                                prop.data_type.clone(),
                                prop.nullable,
                            ));
                        }
                        TypeAlteration::DropProperty(name) => {
                            target
                                .catalog
                                .alter_edge_type_drop_property(&effective_name, name)
                                .map_err(|e| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        e.to_string(),
                                    ))
                                })?;
                            wal_alts.push(("drop".to_string(), name.clone(), String::new(), false));
                        }
                    }
                }
                wal_log!(
                    self,
                    WalRecord::AlterEdgeType {
                        name: effective_name,
                        alterations: wal_alts,
                    }
                );
                self.bounded_status(format_args!("Altered edge type '{}'", stmt.name))
            }
            SchemaStatement::AlterGraphType(stmt) => {
                use grafeo_adapters::query::gql::ast::GraphTypeAlteration;
                let effective_name = target.effective_type_key(&stmt.name);
                let mut wal_alts = Vec::new();
                for alt in &stmt.alterations {
                    match alt {
                        GraphTypeAlteration::AddNodeType(name) => {
                            target
                                .catalog
                                .alter_graph_type_add_node_type(&effective_name, name.clone())
                                .map_err(|e| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        e.to_string(),
                                    ))
                                })?;
                            wal_alts.push(("add_node_type".to_string(), name.clone()));
                        }
                        GraphTypeAlteration::DropNodeType(name) => {
                            target
                                .catalog
                                .alter_graph_type_drop_node_type(&effective_name, name)
                                .map_err(|e| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        e.to_string(),
                                    ))
                                })?;
                            wal_alts.push(("drop_node_type".to_string(), name.clone()));
                        }
                        GraphTypeAlteration::AddEdgeType(name) => {
                            target
                                .catalog
                                .alter_graph_type_add_edge_type(&effective_name, name.clone())
                                .map_err(|e| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        e.to_string(),
                                    ))
                                })?;
                            wal_alts.push(("add_edge_type".to_string(), name.clone()));
                        }
                        GraphTypeAlteration::DropEdgeType(name) => {
                            target
                                .catalog
                                .alter_graph_type_drop_edge_type(&effective_name, name)
                                .map_err(|e| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        e.to_string(),
                                    ))
                                })?;
                            wal_alts.push(("drop_edge_type".to_string(), name.clone()));
                        }
                    }
                }
                wal_log!(
                    self,
                    WalRecord::AlterGraphType {
                        name: effective_name,
                        alterations: wal_alts,
                    }
                );
                self.bounded_status(format_args!("Altered graph type '{}'", stmt.name))
            }
            SchemaStatement::CreateProcedure(stmt) => {
                use crate::catalog::ProcedureDefinition;

                let def = ProcedureDefinition {
                    name: stmt.name.clone(),
                    params: stmt
                        .params
                        .iter()
                        .map(|p| (p.name.clone(), p.param_type.clone()))
                        .collect(),
                    returns: stmt
                        .returns
                        .iter()
                        .map(|r| (r.name.clone(), r.return_type.clone()))
                        .collect(),
                    body: stmt.body.clone(),
                };

                if stmt.or_replace {
                    target.catalog.replace_procedure(def).map_err(|e| {
                        Error::Query(QueryError::new(QueryErrorKind::Semantic, e.to_string()))
                    })?;
                } else {
                    match target.catalog.register_procedure(def) {
                        Ok(()) => {}
                        Err(crate::catalog::CatalogError::TypeAlreadyExists(_))
                            if stmt.if_not_exists =>
                        {
                            return Ok(QueryResult::empty());
                        }
                        Err(e) => {
                            return Err(Error::Query(QueryError::new(
                                QueryErrorKind::Semantic,
                                e.to_string(),
                            )));
                        }
                    }
                }

                wal_log!(
                    self,
                    WalRecord::CreateProcedure {
                        name: stmt.name.clone(),
                        params: stmt
                            .params
                            .iter()
                            .map(|p| (p.name.clone(), p.param_type.clone()))
                            .collect(),
                        returns: stmt
                            .returns
                            .iter()
                            .map(|r| (r.name.clone(), r.return_type.clone()))
                            .collect(),
                        body: stmt.body,
                    }
                );
                self.bounded_status(format_args!("Created procedure '{}'", stmt.name))
            }
            SchemaStatement::DropProcedure { name, if_exists } => {
                match target.catalog.drop_procedure(&name) {
                    Ok(()) => {}
                    Err(_) if if_exists => {
                        return Ok(QueryResult::empty());
                    }
                    Err(e) => {
                        return Err(Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            e.to_string(),
                        )));
                    }
                }
                wal_log!(self, WalRecord::DropProcedure { name: name.clone() });
                self.bounded_status(format_args!("Dropped procedure '{name}'"))
            }
            SchemaStatement::ShowIndexes => {
                return self.execute_show_indexes();
            }
            SchemaStatement::ShowConstraints => {
                return self.execute_show_constraints();
            }
            SchemaStatement::ShowNodeTypes => {
                return self.execute_show_node_types();
            }
            SchemaStatement::ShowEdgeTypes => {
                return self.execute_show_edge_types();
            }
            SchemaStatement::ShowGraphTypes => {
                return self.execute_show_graph_types();
            }
            SchemaStatement::ShowGraphType(name) => {
                return self.execute_show_graph_type(&name);
            }
            SchemaStatement::ShowCurrentGraphType => {
                return self.execute_show_current_graph_type();
            }
            SchemaStatement::ShowGraphs => {
                return self.execute_show_graphs();
            }
            SchemaStatement::ShowSchemas => {
                return self.execute_show_schemas();
            }
        };

        // Invalidate all cached query plans after any successful DDL change.
        // DDL is rare, so clearing the entire cache is cheap and correct.
        if result.is_ok() && invalidate_live_cache {
            self.query_cache.clear();
            self.physical_cache.lock().clear();
        }

        result
    }

    /// Executes a GQL query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// // Create a node
    /// session.execute("INSERT (:Person {name: 'Alix', age: 30})")?;
    ///
    /// // Query nodes
    /// let result = session.execute("MATCH (n:Person) RETURN n.name, n.age")?;
    /// for row in result.rows() {
    ///     println!("{:?}", row);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "gql")]
    pub fn execute(&self, query: &str) -> Result<QueryResult> {
        let _execution_operation = self.session_operation_guard();
        if let Some(options) = self.fresh_execution_options(None) {
            return self.execute_with_options(query, std::collections::HashMap::new(), options);
        }
        let _operation = self.session_operation_guard();
        self.check_not_poisoned()?;
        self.require_lpg("GQL")?;

        #[cfg(feature = "testing-statement-injection")]
        grafeo_common::testing::statement_failure::maybe_fail_statement().map_err(|e| {
            grafeo_common::utils::error::Error::Internal(format!("injected failure: {e}"))
        })?;

        use crate::query::{
            binder::Binder, cache::CacheKey, optimizer::Optimizer, processor::QueryLanguage,
            translators::gql,
        };

        if let Some(cached) = self.execute_cached_physical(query, QueryLanguage::Gql) {
            return self.finish_query(cached);
        }

        let _span = grafeo_info_span!(
            "grafeo::session::execute",
            language = "gql",
            query_len = query.len(),
        );

        #[cfg(not(target_arch = "wasm32"))]
        let start_time = std::time::Instant::now();

        // Parse and translate, checking for session/schema commands first
        self.check_active_execution()?;
        let translation = gql::translate_full(query)?;
        self.check_active_execution()?;
        let logical_plan = match translation {
            gql::GqlTranslationResult::SessionCommand(cmd) => {
                self.reject_scoped_historical_command()?;
                let mutates = Self::session_command_mutates(&cmd);
                if mutates {
                    self.reject_lpg_historical_mutation()?;
                }
                // Lifecycle commands have their own explicit targets. A stale
                // current data selector must not prevent recreating that graph.
                let result = self.with_auto_commit(mutates, || self.execute_session_command(cmd));
                return self.finish_query(result);
            }
            #[cfg(feature = "lpg")]
            gql::GqlTranslationResult::SchemaCommand(cmd) => {
                self.reject_scoped_historical_command()?;
                let result = self.execute_schema_command_checked(cmd);
                return self.finish_query(result);
            }
            gql::GqlTranslationResult::Plan(plan) => plan,
            #[cfg(not(feature = "lpg"))]
            gql::GqlTranslationResult::SchemaCommand(_) => {
                return Err(grafeo_common::utils::error::Error::Internal(
                    "Schema commands require the `lpg` feature".to_string(),
                ));
            }
        };

        // Create cache key for this query
        let cache_key = CacheKey::with_graph(query, QueryLanguage::Gql, self.current_graph_path());

        // Try to get cached optimized plan, or use the plan we just translated
        let optimized_plan = if let Some(cached_plan) = self.query_cache.get_optimized(&cache_key) {
            cached_plan
        } else {
            // Semantic validation
            let mut binder = Binder::new();
            self.check_active_execution()?;
            let _binding_context = binder.bind(&logical_plan)?;
            self.check_active_execution()?;

            // Optimize the plan
            let active = self.active_store();
            let optimizer = Optimizer::from_graph_store(&*active);
            self.check_active_execution()?;
            let plan = optimizer.optimize(logical_plan)?;
            self.check_active_execution()?;

            // Cache the optimized plan for future use
            self.query_cache.put_optimized(cache_key, plan.clone());

            plan
        };
        let qualified = self.qualify_lpg_plan(&optimized_plan.root)?;

        // EXPLAIN: annotate pushdown hints and return the plan tree
        if optimized_plan.explain {
            use crate::query::processor::{annotate_pushdown_hints, explain_result};
            let _read_barrier = self.publication_read_guard();
            let active = self.active_store();
            let mut plan = optimized_plan;
            annotate_pushdown_hints(&mut plan.root, active.as_ref());
            return self.finish_query(explain_result(
                &plan,
                self.result_resources()?,
                self.effective_result_limits(),
            ));
        }

        // PROFILE: execute with per-operator instrumentation
        if optimized_plan.profile {
            let has_mutations = qualified.mutates;
            let result =
                self.with_lpg_plan_auto_commit(has_mutations, qualified.contains_call, || {
                    let _read_barrier = if !has_mutations || qualified.contains_call {
                        self.publication_read_guard()
                    } else {
                        None
                    };
                    self.validate_qualified_lpg_plan(&qualified)?;
                    let (graph_context, active) = self.active_store_with_graph_context();
                    let (viewing_epoch, transaction_id) = self.get_transaction_context();
                    let planner = self.create_planner_for_store_with_graph_context(
                        Arc::clone(&active),
                        viewing_epoch,
                        transaction_id,
                        false,
                        &graph_context,
                    );
                    #[cfg(any(feature = "lpg", feature = "algos"))]
                    let planner = self.attach_qualified_procedures(planner, &qualified);
                    self.check_active_execution()?;
                    let (physical_plan, entries) = planner.plan_profiled(&optimized_plan)?;
                    self.check_active_execution()?;
                    let _result = self.execute_physical_plan_with_checkpoint(
                        physical_plan,
                        self.installed_or_fresh_checkpoint(),
                        &entries,
                    )?;

                    let total_time_ms;
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        total_time_ms = 0.0;
                    }

                    let profile_tree = crate::query::profile::build_profile_tree(
                        &optimized_plan.root,
                        &mut entries.into_iter(),
                    );
                    crate::query::profile::profile_result(
                        &profile_tree,
                        total_time_ms,
                        self.result_resources()?,
                        self.effective_result_limits(),
                    )
                });
            return self.finish_query(result);
        }

        let has_mutations = qualified.mutates;

        let result = self.with_lpg_plan_auto_commit(has_mutations, qualified.contains_call, || {
            let _read_barrier = if !has_mutations || qualified.contains_call {
                self.publication_read_guard()
            } else {
                None
            };
            self.validate_qualified_lpg_plan(&qualified)?;
            let graph_context = self.graph_context_snapshot();
            let active = self.store_for_graph_storage_key(&graph_context.storage_key);
            // Get transaction context for MVCC visibility
            let (viewing_epoch, transaction_id) = self.get_transaction_context();

            // Convert to physical plan with transaction context.
            // Read-only, no-tx plans are cached on the database so the next
            // execute (any session, including one-shot `db.execute`) skips
            // parse + physicalize (see execute_cached_physical).
            let has_active_tx = self.current_transaction.lock().is_some();
            let read_only = !has_mutations && !has_active_tx;
            let planner = self.create_planner_for_store_with_graph_context(
                Arc::clone(&active),
                viewing_epoch,
                transaction_id,
                read_only,
                &graph_context,
            );
            #[cfg(any(feature = "lpg", feature = "algos"))]
            let planner = self.attach_qualified_procedures(planner, &qualified);
            self.check_active_execution()?;
            let mut physical_plan = planner.plan(&optimized_plan)?;
            self.check_active_execution()?;

            let cache_key = (!has_mutations && !qualified.contains_call)
                .then(|| {
                    Self::physical_cache_key_for_context(
                        query,
                        QueryLanguage::Gql,
                        graph_context,
                        viewing_epoch,
                        transaction_id,
                    )
                })
                .flatten();
            let mut result = if let Some(key) = cache_key {
                let out = self.execute_borrowed_physical_plan(&mut physical_plan)?;
                self.store_cached_physical(key, physical_plan);
                out
            } else {
                self.execute_physical_plan(physical_plan)?
            };

            // Add execution metrics
            let rows_scanned = result.row_count() as u64;
            #[cfg(not(target_arch = "wasm32"))]
            {
                let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                result.execution_time_ms = Some(elapsed_ms);
            }
            result.rows_scanned = Some(rows_scanned);

            Ok(result)
        });
        // Record metrics for this query execution.
        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("gql", elapsed_ms, &result);
        }

        self.finish_query(result)
    }

    /// Completes proven small resident work or prepares scheduled ORDER BY.
    /// Unsupported shapes leave request/options for this same Session's caller.
    #[cfg(all(
        feature = "gql",
        feature = "lpg",
        feature = "spill",
        feature = "async-storage"
    ))]
    pub(crate) fn try_prepare_async_sort(
        &self,
        query: &str,
        params: &std::collections::HashMap<String, Value>,
        options: &mut Option<crate::query::ExecutionOptions>,
    ) -> Result<Option<crate::query::executor::AsyncSortDispatch>> {
        use crate::query::{
            binder::Binder,
            cache::CacheKey,
            executor::AsyncSortDispatch,
            optimizer::Optimizer,
            plan::{LogicalExpression, LogicalOperator},
            processor::{QueryLanguage, substitute_params},
            translators::gql,
        };
        use grafeo_core::execution::operators::{Operator, SortOperator};

        // This is a work bound, not a cardinality estimate: filters, LIMIT
        // and SKIP never hide a larger input. Only literal finite sources and
        // named scalar wrappers qualify; scans and function evaluation do not.
        const SMALL_RESIDENT_ROWS: usize = 4096;
        fn literal(value: &Value) -> bool {
            matches!(
                value,
                Value::Null
                    | Value::Bool(_)
                    | Value::Int64(_)
                    | Value::Float64(_)
                    | Value::String(_)
            )
        }
        fn small_scalar(expression: &LogicalExpression) -> bool {
            match expression {
                LogicalExpression::Literal(value) => literal(value),
                LogicalExpression::Variable(_) => true,
                LogicalExpression::Binary { left, right, .. } => {
                    small_scalar(left) && small_scalar(right)
                }
                LogicalExpression::Unary { operand, .. } => small_scalar(operand),
                _ => false,
            }
        }
        fn literal_rows(expression: &LogicalExpression) -> Option<usize> {
            let rows = match expression {
                LogicalExpression::List(values) if values.iter().all(|value| matches!(value, LogicalExpression::Literal(value) if literal(value))) => values.len(),
                LogicalExpression::Literal(Value::List(values)) if values.iter().all(literal) => values.len(),
                LogicalExpression::FunctionCall { name, args, distinct: false }
                    if name.eq_ignore_ascii_case("range") => {
                    let [LogicalExpression::Literal(Value::Int64(start)), LogicalExpression::Literal(Value::Int64(end))] = args.as_slice() else { return None; };
                    // The evaluator advances past the inclusive end; its last
                    // increment must also be representable.
                    end.checked_add(1)?;
                    usize::try_from(end.checked_sub(*start)?.checked_add(1)?).ok()?
                }
                _ => return None,
            };
            (rows <= SMALL_RESIDENT_ROWS).then_some(rows)
        }
        fn small_input(operator: &LogicalOperator) -> Option<usize> {
            let rows = match operator {
                LogicalOperator::Empty => 1,
                LogicalOperator::Unwind(unwind) => {
                    small_input(&unwind.input)?.checked_mul(literal_rows(&unwind.expression)?)?
                }
                LogicalOperator::Return(ret)
                    if !ret.distinct
                        && ret.items.iter().all(|item| small_scalar(&item.expression)) =>
                {
                    small_input(&ret.input)?
                }
                LogicalOperator::Project(project)
                    if project
                        .projections
                        .iter()
                        .all(|item| small_scalar(&item.expression)) =>
                {
                    small_input(&project.input)?
                }
                LogicalOperator::Filter(filter) if small_scalar(&filter.predicate) => {
                    small_input(&filter.input)?
                }
                LogicalOperator::Limit(limit) => small_input(&limit.input)?,
                LogicalOperator::Skip(skip) => small_input(&skip.input)?,
                _ => return None,
            };
            (rows <= SMALL_RESIDENT_ROWS).then_some(rows)
        }
        fn small_sort(plan: &crate::query::plan::LogicalPlan) -> bool {
            matches!(&plan.root, LogicalOperator::Sort(sort) if !plan.explain && !plan.profile
                && sort.keys.iter().all(|key| small_scalar(&key.expression))
                && small_input(&sort.input).is_some())
        }

        // This is only a cheap refusal. The parsed/optimized tree owns admission.
        let Some(candidate) = options.as_ref() else {
            return Err(grafeo_common::utils::error::Error::Internal(
                "async query lost its options".into(),
            ));
        };
        if candidate
            .language
            .as_deref()
            .is_some_and(|name| !name.eq_ignore_ascii_case("gql"))
            || !query
                .as_bytes()
                .windows(5)
                .any(|word| word.eq_ignore_ascii_case(b"order"))
            || self.in_transaction()
        {
            return Ok(None);
        }
        self.check_not_poisoned()?;
        self.require_lpg("GQL")?;
        let resident_only = self
            .buffer_manager
            .as_ref()
            .is_some_and(|memory| memory.config().spill_path.is_none());
        let logical_cache_key = (resident_only && params.is_empty())
            .then(|| CacheKey::with_graph(query, QueryLanguage::Gql, self.current_graph_path()));
        if let Some(cached) = logical_cache_key
            .as_ref()
            .and_then(|key| self.query_cache.get_optimized(key))
            && small_sort(&cached)
        {
            // The ordinary path reuses the physical cache, with the original
            // options and exactly one composition of the execution control.
            return Ok(None);
        }
        let preparation_started = std::time::Instant::now();
        let Some(mut candidate) = options.take() else {
            return Err(grafeo_common::utils::error::Error::Internal(
                "async query lost its options".into(),
            ));
        };
        candidate.control = self.compose_execution_control(candidate.control)?;
        let checkpoint = candidate.control.checkpoint();
        *options = Some(candidate);
        Self::check_query_checkpoint(&checkpoint)?;
        let gql::GqlTranslationResult::Plan(mut logical) = gql::translate_full(query)? else {
            return Ok(None);
        };
        Self::check_query_checkpoint(&checkpoint)?;
        // Keep non-sort execution, mutation, profiling and other blocking
        // algorithms on their existing complete caller path.
        fn scalar(expression: &crate::query::plan::LogicalExpression) -> bool {
            use crate::query::plan::LogicalExpression as Expr;
            match expression {
                Expr::Literal(_)
                | Expr::Variable(_)
                | Expr::Property { .. }
                | Expr::Parameter(_)
                | Expr::Labels(_)
                | Expr::Type(_)
                | Expr::Id(_) => true,
                Expr::Binary { left, right, .. } => scalar(left) && scalar(right),
                Expr::Unary { operand, .. } => scalar(operand),
                Expr::FunctionCall { args, distinct, .. } => !distinct && args.iter().all(scalar),
                Expr::List(items) => items.iter().all(scalar),
                // Subqueries and composite expressions whose children can hide
                // blocking plans remain on the established execution path.
                _ => false,
            }
        }
        fn eligible_input(operator: &LogicalOperator) -> bool {
            let accepted = match operator {
                LogicalOperator::Return(ret) => {
                    !ret.distinct && ret.items.iter().all(|item| scalar(&item.expression))
                }
                LogicalOperator::Filter(filter) => scalar(&filter.predicate),
                LogicalOperator::Project(project) => project
                    .projections
                    .iter()
                    .all(|item| scalar(&item.expression)),
                LogicalOperator::Unwind(unwind) => scalar(&unwind.expression),
                LogicalOperator::NodeScan(_)
                | LogicalOperator::EdgeScan(_)
                | LogicalOperator::Limit(_)
                | LogicalOperator::Skip(_)
                | LogicalOperator::Empty => true,
                _ => false,
            };
            accepted && operator.children().into_iter().all(eligible_input)
        }
        let LogicalOperator::Sort(sort) = &logical.root else {
            return Ok(None);
        };
        if logical.explain
            || logical.profile
            || !eligible_input(&sort.input)
            || !sort.keys.iter().all(|key| scalar(&key.expression))
        {
            return Ok(None);
        }
        self.check_lpg_query_access(false)?;
        let candidate = options.as_ref().ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal("async query lost its options".into())
        })?;
        let publication = self.streaming_owned_publication_read_guard(&candidate.control)?;
        substitute_params(&mut logical, params)?;
        Binder::new().bind(&logical)?;
        Self::check_query_checkpoint(&checkpoint)?;
        let active = self.active_store();
        let optimized = Optimizer::from_graph_store(&*active).optimize(logical)?;
        Self::check_query_checkpoint(&checkpoint)?;
        let LogicalOperator::Sort(sort) = &optimized.root else {
            return Ok(None);
        };
        if !eligible_input(&sort.input) || !sort.keys.iter().all(|key| scalar(&key.expression)) {
            return Ok(None);
        }
        let qualified = self.qualify_lpg_plan(&optimized.root)?;
        if qualified.mutates || qualified.contains_call {
            return Ok(None);
        }
        self.validate_qualified_lpg_plan(&qualified)?;
        let (graph_context, active) = self.active_store_with_graph_context();
        let (epoch, transaction) = self.get_transaction_context();
        let planner = self.create_planner_for_store_with_graph_context(
            active,
            epoch,
            transaction,
            true,
            &graph_context,
        );
        if resident_only && small_sort(&optimized) {
            // The complete literal input is proven <=4096 rows. Use this
            // already-qualified plan once in the existing preparation worker;
            // no scheduler/task/error-ticket ownership is needed.
            let mut physical = planner.plan(&optimized)?;
            Self::check_query_checkpoint(&checkpoint)?;
            let resources = self.make_query_resource_context(Some(checkpoint.token()))?;
            let mut candidate = options.take().ok_or_else(|| {
                grafeo_common::utils::error::Error::Internal(
                    "small sort lost execution options".into(),
                )
            })?;
            let limits = candidate.result_limits.unwrap_or(self.result_limits);
            let executor =
                Executor::with_bounded_columns(&physical.columns, resources.clone(), limits)?
                    .with_execution_checkpoint(checkpoint.clone());
            physical
                .operator
                .install_resource_context(&resources)
                .map_err(Self::map_query_resource_context_error)?;
            let execution = executor.execute(physical.operator.as_mut());
            drop(executor);
            physical.operator.reset();
            let mut result = execution?;
            Self::check_query_checkpoint(&checkpoint)?;
            if let Some(key) = logical_cache_key {
                self.query_cache.put_optimized(key, optimized.clone());
                if let Some(key) = Self::physical_cache_key_for_context(
                    query,
                    QueryLanguage::Gql,
                    graph_context,
                    epoch,
                    transaction,
                ) {
                    self.store_cached_physical(key, physical);
                }
            }
            if let Some(admit) = candidate.result_admission {
                admit(&result, limits)?;
            }
            candidate
                .control
                .complete()
                .map_err(Self::map_query_lifecycle_error)?;
            result.execution_time_ms = Some(preparation_started.elapsed().as_secs_f64() * 1000.0);
            drop(publication);
            return Ok(Some(AsyncSortDispatch::Completed(result)));
        }
        let parts = planner.plan_sort_input(sort)?;
        Self::check_query_checkpoint(&checkpoint)?;
        let resources = self.make_query_resource_context(Some(checkpoint.token()))?;
        let mut operator = SortOperator::new(parts.input, parts.keys, parts.schema);
        operator
            .install_resource_context(&resources)
            .map_err(Self::map_query_resource_context_error)?;
        let candidate = options.take().ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal("async query lost its options".into())
        })?;
        crate::query::executor::PreparedAsyncSort::new(
            operator,
            parts.columns,
            resources,
            candidate.control,
            publication,
            candidate.result_limits.unwrap_or(self.result_limits),
            candidate.result_admission,
        )
        .map(|prepared| Some(AsyncSortDispatch::Prepared(prepared)))
    }

    /// Executes one explicitly controlled, read-only GQL plan.
    ///
    /// This is the qualification seam for propagating one cancellation state
    /// through Session planning, executor orchestration, and resource-aware
    /// blocking operators. It deliberately remains crate-private until the
    /// wider DB, streaming, binding, and mutation lifecycle contracts are
    /// qualified. In particular, it never enters [`Self::with_auto_commit`].
    /// The returned value is a materialized candidate: an outer owner holding
    /// the corresponding [`grafeo_core::execution::QueryExecutionControl`]
    /// must successfully complete that lifecycle before publishing it through
    /// a future public API.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "crate-private qualification seam; public facade follows mutation-fence qualification"
        )
    )]
    #[cfg(all(feature = "gql", feature = "lpg"))]
    pub(crate) fn execute_read_with_checkpoint(
        &self,
        query: &str,
        checkpoint: grafeo_core::execution::QueryExecutionCheckpoint,
    ) -> Result<QueryResult> {
        use crate::query::{
            binder::Binder, cache::CacheKey, optimizer::Optimizer, processor::QueryLanguage,
            translators::gql,
        };

        self.check_not_poisoned()?;
        self.require_lpg("GQL")?;
        let checkpoint = self.compose_query_checkpoint(checkpoint);
        Self::check_query_checkpoint(&checkpoint)?;

        #[cfg(feature = "testing-statement-injection")]
        grafeo_common::testing::statement_failure::maybe_fail_statement().map_err(|e| {
            grafeo_common::utils::error::Error::Internal(format!("injected failure: {e}"))
        })?;

        let _span = grafeo_info_span!(
            "grafeo::session::execute_read_with_checkpoint",
            language = "gql",
            query_len = query.len(),
        );

        #[cfg(not(target_arch = "wasm32"))]
        let start_time = std::time::Instant::now();

        let translation = gql::translate_full(query)?;
        Self::check_query_checkpoint(&checkpoint)?;
        let logical_plan = match translation {
            gql::GqlTranslationResult::SessionCommand(_) => {
                return Err(grafeo_common::utils::error::Error::Query(
                    grafeo_common::utils::error::QueryError::new(
                        grafeo_common::utils::error::QueryErrorKind::Semantic,
                        "session commands cannot run through read-only controlled execution",
                    ),
                ));
            }
            gql::GqlTranslationResult::SchemaCommand(_) => {
                return Err(grafeo_common::utils::error::Error::Query(
                    grafeo_common::utils::error::QueryError::new(
                        grafeo_common::utils::error::QueryErrorKind::Semantic,
                        "schema DDL cannot run through read-only controlled execution",
                    ),
                ));
            }
            gql::GqlTranslationResult::Plan(plan) => {
                if plan.root.has_mutations() {
                    return Err(grafeo_common::utils::error::Error::Query(
                        grafeo_common::utils::error::QueryError::new(
                            grafeo_common::utils::error::QueryErrorKind::Semantic,
                            "mutating queries cannot run through read-only controlled execution",
                        ),
                    ));
                }
                Self::reject_unclassified_procedure_call(&plan.root, "controlled read execution")?;
                self.check_lpg_query_access(false)?;
                plan
            }
        };

        let cache_key = CacheKey::with_graph(query, QueryLanguage::Gql, self.current_graph_path());
        let optimized_plan = if let Some(cached_plan) = self.query_cache.get_optimized(&cache_key) {
            cached_plan
        } else {
            Self::check_query_checkpoint(&checkpoint)?;
            let mut binder = Binder::new();
            let _binding_context = binder.bind(&logical_plan)?;
            Self::check_query_checkpoint(&checkpoint)?;

            let active = self.active_store();
            let optimizer = Optimizer::from_graph_store(&*active);
            let plan = optimizer.optimize(logical_plan)?;
            Self::check_query_checkpoint(&checkpoint)?;
            self.query_cache.put_optimized(cache_key, plan.clone());
            plan
        };
        Self::check_query_checkpoint(&checkpoint)?;

        if optimized_plan.profile {
            return Err(grafeo_common::utils::error::Error::Query(
                grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Semantic,
                    "PROFILE is not yet qualified for controlled read execution",
                ),
            ));
        }

        if optimized_plan.explain {
            use crate::query::processor::{annotate_pushdown_hints, explain_result};

            let _read_barrier =
                acquire_publication_read_with_checkpoint(&self.transaction_manager, &checkpoint)
                    .map_err(Self::map_query_cancellation_error)?;
            let active = self.active_store();
            let mut plan = optimized_plan;
            Self::check_query_checkpoint(&checkpoint)?;
            annotate_pushdown_hints(&mut plan.root, active.as_ref());
            Self::check_query_checkpoint(&checkpoint)?;
            return self.finish_query(explain_result(
                &plan,
                self.result_resources()?,
                self.effective_result_limits(),
            ));
        }

        let _read_barrier =
            acquire_publication_read_with_checkpoint(&self.transaction_manager, &checkpoint)
                .map_err(Self::map_query_cancellation_error)?;
        let graph_context = self.graph_context_snapshot();
        let active = self.store_for_graph_storage_key(&graph_context.storage_key);
        let (viewing_epoch, transaction_id) = self.get_transaction_context();
        let has_active_tx = self.current_transaction.lock().is_some();
        Self::check_query_checkpoint(&checkpoint)?;
        let planner = self.create_planner_for_store_with_graph_context(
            Arc::clone(&active),
            viewing_epoch,
            transaction_id,
            !has_active_tx,
            &graph_context,
        );
        let physical_plan = planner.plan(&optimized_plan)?;
        Self::check_query_checkpoint(&checkpoint)?;

        let mut result =
            self.execute_physical_plan_with_checkpoint(physical_plan, checkpoint.clone(), &[])?;

        let rows_scanned = result.row_count() as u64;
        #[cfg(not(target_arch = "wasm32"))]
        {
            result.execution_time_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
        }
        result.rows_scanned = Some(rows_scanned);
        self.finish_query(Ok(result))
    }

    /// Executes a read-only GQL query as a lazy, explicitly closeable stream.
    ///
    /// PROFILE yields query rows and exposes live metrics through the stream's
    /// `profile()` accessor. Mutations, commands and push pipelines remain
    /// outside this read cursor's admission contract.
    ///
    /// # Errors
    /// Returns parsing, admission, planning, resource or cancellation errors.
    #[cfg(all(feature = "gql", feature = "lpg"))]
    pub fn execute_streaming(
        &self,
        query: &str,
    ) -> Result<crate::query::executor::stream::ResultStream<'_>> {
        self.stream_with_options(query, std::collections::HashMap::new(), Default::default())
    }

    /// Opens a read-only GQL stream with an owned parameter snapshot and control.
    ///
    /// The same owner and deadline cover planning, every pull and close. Closing
    /// the stream releases its publication guard and Session stream count even
    /// while the terminal stream object remains alive.
    ///
    /// # Errors
    /// Returns typed cancellation or the ordinary parsing/admission/resource
    /// errors. An explicit language other than GQL is unsupported by this cursor.
    #[cfg(all(feature = "gql", feature = "lpg"))]
    pub fn stream_with_options(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
        options: crate::query::ExecutionOptions,
    ) -> Result<crate::query::executor::stream::ResultStream<'_>> {
        use crate::query::executor::stream::{ResultStream, StreamGuard};
        let _operation = self.session_operation_guard();
        self.check_not_poisoned()?;
        let crate::query::ExecutionOptions {
            control,
            language,
            result_limits,
            result_admission,
        } = options;
        if result_admission.is_some() {
            return Err(grafeo_common::utils::error::Error::Query(
                grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Unsupported,
                    "eager result admission is not a streaming chunk conversion policy",
                ),
            ));
        }
        let control = self.compose_execution_control(control)?;
        let publication = self.streaming_publication_read_guard(&control)?;
        let execution = self
            .build_streaming_plan(query, params, control, language.as_deref())?
            .with_result_limits(result_limits);
        Ok(ResultStream::new(
            execution,
            StreamGuard::new(&self.active_streams),
            publication,
        ))
    }

    /// Builds the one resource-owned read cursor used by both stream facades.
    #[cfg(all(feature = "gql", feature = "lpg"))]
    pub(crate) fn build_streaming_plan(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
        control: grafeo_core::execution::QueryExecutionControl,
        language: Option<&str>,
    ) -> Result<crate::query::executor::stream::StreamExecution> {
        use crate::query::executor::stream::StreamExecution;
        use crate::query::{
            binder::Binder,
            cache::CacheKey,
            optimizer::Optimizer,
            processor::{QueryLanguage, substitute_params},
            translators::gql,
        };
        control
            .check()
            .map_err(Self::map_query_cancellation_error)?;
        if language.is_some_and(|name| !name.eq_ignore_ascii_case("gql")) {
            return Err(grafeo_common::utils::error::Error::Query(
                grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Unsupported,
                    "this read cursor supports GQL; other language cursors are not yet qualified",
                ),
            ));
        }
        self.require_lpg("GQL")?;
        let translation = gql::translate_full(query)?;
        control
            .check()
            .map_err(Self::map_query_cancellation_error)?;
        let mut logical_plan = match translation {
            gql::GqlTranslationResult::SessionCommand(_)
            | gql::GqlTranslationResult::SchemaCommand(_) => {
                return Err(grafeo_common::utils::error::Error::Query(
                    grafeo_common::utils::error::QueryError::new(
                        grafeo_common::utils::error::QueryErrorKind::Semantic,
                        "schema/session commands cannot be streamed; use execute() instead",
                    ),
                ));
            }
            gql::GqlTranslationResult::Plan(plan) => {
                if plan.root.has_mutations() {
                    return Err(grafeo_common::utils::error::Error::Query(
                        grafeo_common::utils::error::QueryError::new(
                            grafeo_common::utils::error::QueryErrorKind::Semantic,
                            "mutating queries cannot be streamed; use execute() instead",
                        ),
                    ));
                }
                Self::reject_unclassified_procedure_call(&plan.root, "streaming execution")?;
                self.check_lpg_query_access(false)?;
                plan
            }
        };
        // Never cache a plan containing a particular caller's parameter values.
        let cache_key = params
            .is_empty()
            .then(|| CacheKey::with_graph(query, QueryLanguage::Gql, self.current_graph_path()));
        substitute_params(&mut logical_plan, &params)?;
        control
            .check()
            .map_err(Self::map_query_cancellation_error)?;
        let cached = cache_key
            .as_ref()
            .and_then(|key| self.query_cache.get_optimized(key));
        let optimized_plan = if let Some(cached) = cached {
            cached
        } else {
            let mut binder = Binder::new();
            let _binding_context = binder.bind(&logical_plan)?;
            control
                .check()
                .map_err(Self::map_query_cancellation_error)?;
            let active = self.active_store();
            let optimizer = Optimizer::from_graph_store(&*active);
            let plan = optimizer.optimize(logical_plan)?;
            control
                .check()
                .map_err(Self::map_query_cancellation_error)?;
            if let Some(key) = cache_key {
                self.query_cache.put_optimized(key, plan.clone());
            }
            plan
        };
        if optimized_plan.explain {
            return Err(grafeo_common::utils::error::Error::Query(
                grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Semantic,
                    "EXPLAIN cannot be streamed; use execute() instead",
                ),
            ));
        }
        control
            .check()
            .map_err(Self::map_query_cancellation_error)?;
        let (graph_context, active) = self.active_store_with_graph_context();
        let has_active_tx = self.current_transaction.lock().is_some();
        let (viewing_epoch, transaction_id) = self.get_transaction_context();
        let planner = self.create_planner_for_store_with_graph_context(
            Arc::clone(&active),
            viewing_epoch,
            transaction_id,
            !has_active_tx,
            &graph_context,
        );
        // Profiling wrappers are pull boundaries. Check the optimized logical
        // tree as well so profiling cannot conceal a blocking push stage.
        fn has_blocking_stage(operator: &crate::query::plan::LogicalOperator) -> bool {
            use crate::query::plan::LogicalOperator;
            matches!(
                operator,
                LogicalOperator::Sort(_) | LogicalOperator::Aggregate(_)
            ) || operator.children().into_iter().any(has_blocking_stage)
        }
        if optimized_plan.profile && has_blocking_stage(&optimized_plan.root) {
            return Err(grafeo_common::utils::error::Error::Query(
                grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Semantic,
                    "PROFILE query requires a blocking pipeline which cannot be streamed; use execute() instead",
                ),
            ));
        }
        fn has_distinct_stage(operator: &crate::query::plan::LogicalOperator) -> bool {
            use crate::query::plan::LogicalOperator;
            matches!(operator, LogicalOperator::Distinct(_))
                || matches!(operator, LogicalOperator::Return(result) if result.distinct)
                || operator.children().into_iter().any(has_distinct_stage)
        }
        // The shared DISTINCT state now has a fallible, grant-aware pull cursor.
        // Keep that cursor intact for streaming; sort/aggregate still retain
        // their existing admission boundary until their own bounded cuts land.
        let use_distinct_pull =
            has_distinct_stage(&optimized_plan.root) && !has_blocking_stage(&optimized_plan.root);
        let (physical_plan, profile) = if optimized_plan.profile {
            let (physical, entries) = planner.plan_profiled(&optimized_plan)?;
            let tree = crate::query::profile::build_profile_tree(
                &optimized_plan.root,
                &mut entries.into_iter(),
            );
            (physical, Some(tree))
        } else {
            (planner.plan(&optimized_plan)?, None)
        };
        control
            .check()
            .map_err(Self::map_query_cancellation_error)?;
        let columns = physical_plan.columns.clone();
        let resources = self.make_query_resource_context(Some(control.token()))?;
        let preparation = (|| {
            if use_distinct_pull {
                let mut source = physical_plan.into_operator();
                source
                    .install_resource_context(&resources)
                    .map_err(Self::map_query_resource_context_error)?;
                control
                    .check()
                    .map_err(Self::map_query_cancellation_error)?;
                return Ok(source);
            }
            let (source, push_ops) =
                grafeo_core::execution::pipeline_convert::convert_to_pipeline_with_resources(
                    physical_plan.into_operator(),
                    &resources,
                )
                .map_err(Self::map_query_resource_context_error)?;
            if !push_ops.is_empty() {
                return Err(grafeo_common::utils::error::Error::Query(
                    grafeo_common::utils::error::QueryError::new(
                        grafeo_common::utils::error::QueryErrorKind::Semantic,
                        "query requires a push-based pipeline (ORDER BY / aggregate / DISTINCT) which cannot be streamed; use execute() instead",
                    ),
                ));
            }
            control
                .check()
                .map_err(Self::map_query_cancellation_error)?;
            Ok(source)
        })();
        let source = match preparation {
            Ok(source) => source,
            Err(error) => {
                #[cfg(feature = "spill")]
                let spill_manager = resources.spill_manager().cloned();
                drop(resources);
                #[cfg(feature = "spill")]
                return finish_spill_execution(Err(error), spill_manager);
                #[cfg(not(feature = "spill"))]
                return Err(error);
            }
        };
        let execution = StreamExecution::new(source, columns, control, resources);
        Ok(match profile {
            Some(profile) => execution.with_profile(profile),
            None => execution,
        })
    }

    /// Executes a GQL query with visibility at the specified epoch.
    ///
    /// This enables time-travel queries: the query sees the database
    /// as it existed at the given epoch.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing or execution fails.
    #[cfg(feature = "gql")]
    pub fn execute_at_epoch(&self, query: &str, epoch: EpochId) -> Result<QueryResult> {
        let _operation = self.session_operation_guard();
        let _historical = self.historical_view_operation_gate.lock();
        self.check_not_poisoned()?;
        if self.current_transaction.lock().is_some() {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "execute_at_epoch cannot replace the snapshot of an active transaction"
                        .to_string(),
                ),
            ));
        }
        self.with_scoped_viewing_epoch(epoch, || self.execute(query))
    }

    /// Executes a GQL query at a specific epoch with optional parameters.
    ///
    /// Combines epoch-based time travel with parameterized queries.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing or execution fails.
    #[cfg(feature = "gql")]
    pub fn execute_at_epoch_with_params(
        &self,
        query: &str,
        epoch: EpochId,
        params: Option<std::collections::HashMap<String, Value>>,
    ) -> Result<QueryResult> {
        let _operation = self.session_operation_guard();
        let _historical = self.historical_view_operation_gate.lock();
        self.check_not_poisoned()?;
        if self.current_transaction.lock().is_some() {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "execute_at_epoch_with_params cannot replace the snapshot of an active transaction"
                        .to_string(),
                ),
            ));
        }
        self.with_scoped_viewing_epoch(epoch, || {
            if let Some(p) = params {
                self.execute_with_params(query, p)
            } else {
                self.execute(query)
            }
        })
    }

    /// Executes a historical query with a caller-owned cancellation and result policy.
    ///
    /// # Errors
    /// Returns an error for an active transaction, invalid query, cancellation,
    /// or rejected result admission. Restores the previous historical view.
    #[cfg(feature = "gql")]
    pub fn execute_at_epoch_with_options(
        &self,
        query: &str,
        epoch: EpochId,
        params: std::collections::HashMap<String, Value>,
        options: crate::query::ExecutionOptions,
    ) -> Result<QueryResult> {
        if options
            .language
            .as_deref()
            .is_some_and(|language| !language.eq_ignore_ascii_case("gql"))
        {
            return Err(grafeo_common::utils::error::Error::Query(
                grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Unsupported,
                    "historical execution options require the GQL epoch-aware planner",
                ),
            ));
        }
        let _operation = self.session_operation_guard();
        let _historical = self.historical_view_operation_gate.lock();
        self.check_not_poisoned()?;
        if self.current_transaction.lock().is_some() {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "execute_at_epoch_with_options cannot replace the snapshot of an active transaction".to_string(),
                ),
            ));
        }
        self.with_scoped_viewing_epoch(epoch, || self.execute_with_options(query, params, options))
    }

    /// Executes a GQL query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "gql")]
    pub fn execute_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        let _execution_operation = self.session_operation_guard();
        if let Some(options) = self.fresh_execution_options(None) {
            return self.execute_with_options(query, params, options);
        }
        let _operation = self.session_operation_guard();
        self.check_not_poisoned()?;
        self.require_lpg("GQL")?;

        use crate::query::processor::substitute_params;
        use crate::query::translators::gql;
        use crate::query::{binder::Binder, optimizer::Optimizer};

        // Parse before opening the execution boundary. Mutation framing is a
        // semantic property of the plan; keyword inspection can miss valid
        // writes hidden behind language syntax and would execute them without
        // a recoverable transaction marker.
        self.check_active_execution()?;
        let mut logical_plan = gql::translate(query)?;
        self.check_active_execution()?;
        substitute_params(&mut logical_plan, &params)?;
        self.check_active_execution()?;
        let mut binder = Binder::new();
        self.check_active_execution()?;
        let _binding_context = binder.bind(&logical_plan)?;
        self.check_active_execution()?;
        let optimized_plan = {
            let active = self.active_store();
            self.check_active_execution()?;
            let optimized = Optimizer::from_graph_store(active.as_ref()).optimize(logical_plan)?;
            self.check_active_execution()?;
            optimized
        };
        let qualified = self.qualify_lpg_plan(&optimized_plan.root)?;
        let has_mutations = qualified.mutates;

        let result = self.with_lpg_plan_auto_commit(has_mutations, qualified.contains_call, || {
            let _read_barrier = if !has_mutations || qualified.contains_call {
                self.publication_read_guard()
            } else {
                None
            };
            self.validate_qualified_lpg_plan(&qualified)?;
            let graph_context = self.graph_context_snapshot();
            let active = self.store_for_graph_storage_key(&graph_context.storage_key);
            let (viewing_epoch, transaction_id) = self.get_transaction_context();
            let planner = self.create_planner_for_store_with_graph_context(
                Arc::clone(&active),
                viewing_epoch,
                transaction_id,
                !has_mutations && transaction_id.is_none(),
                &graph_context,
            );
            #[cfg(any(feature = "lpg", feature = "algos"))]
            let planner = self.attach_qualified_procedures(planner, &qualified);
            self.check_active_execution()?;
            let physical_plan = planner.plan(&optimized_plan)?;
            self.check_active_execution()?;
            self.execute_physical_plan(physical_plan)
        });
        self.finish_query(result)
    }

    #[cfg(feature = "testing-statement-injection")]
    /// Installs one rendezvous for a controlled query boundary.
    pub fn set_query_cancellation_test_hook(
        &self,
        phase: QueryCancellationTestPhase,
        reached: Arc<std::sync::Barrier>,
        release: Arc<std::sync::Barrier>,
    ) {
        *self.query_cancellation_test_hook.lock() = Some(QueryCancellationTestHook {
            phase,
            reached,
            release,
        });
    }

    #[cfg(feature = "testing-statement-injection")]
    fn query_cancellation_test_boundary(&self, phase: QueryCancellationTestPhase) {
        let hook = {
            let mut slot = self.query_cancellation_test_hook.lock();
            if slot.as_ref().is_some_and(|hook| hook.phase == phase) {
                slot.take()
            } else {
                None
            }
        };
        if let Some(hook) = hook {
            hook.reached.wait();
            hook.release.wait();
        }
    }

    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    fn enter_execution_statement(&self, mutating: bool) -> Result<ExecutionStatementGuard<'_>> {
        if self.active_execution_control.lock().is_none() {
            return Ok(ExecutionStatementGuard {
                depth: None,
                outermost: false,
                mutating,
            });
        }
        let previous = self
            .active_execution_statement_depth
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |depth| {
                depth.checked_add(1)
            })
            .map_err(|_| {
                grafeo_common::utils::error::Error::Internal(
                    "execution statement depth exhausted".into(),
                )
            })?;
        Ok(ExecutionStatementGuard {
            depth: Some(&self.active_execution_statement_depth),
            outermost: previous == 0,
            mutating,
        })
    }

    /// Check only the installed owner: no fresh deadline, token or allocation.
    fn check_active_execution(&self) -> Result<()> {
        if let Some(control) = self.active_execution_control.lock().as_ref() {
            control
                .check()
                .map_err(Self::map_query_cancellation_error)?;
        }
        Ok(())
    }

    fn complete_active_execution(&self) -> Result<()> {
        let mut slot = self.active_execution_control.lock();
        if let Some(control) = slot.as_mut()
            && !self.active_execution_completed.load(Ordering::Acquire)
        {
            control
                .complete()
                .map_err(Self::map_query_lifecycle_error)?;
            self.active_execution_completed
                .store(true, Ordering::Release);
        }
        Ok(())
    }

    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    fn finish_controlled_statement<T>(
        &self,
        guard: &ExecutionStatementGuard<'_>,
        terminal: bool,
        body: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if guard.outermost && guard.mutating {
            #[cfg(feature = "testing-statement-injection")]
            self.query_cancellation_test_boundary(QueryCancellationTestPhase::BeforeFirstMutation);
            self.check_active_execution()?;
        }
        // Observe the body outcome before the rendezvous. A cancellation that
        // arrives afterwards cannot replace an already-observed primary error.
        let outcome = body();
        if guard.outermost {
            #[cfg(feature = "testing-statement-injection")]
            self.query_cancellation_test_boundary(
                QueryCancellationTestPhase::BeforeStatementCompletion,
            );
        }
        let value = outcome?;
        if guard.outermost {
            self.check_active_execution()?;
            if terminal {
                self.complete_active_execution()?;
            }
        }
        Ok(value)
    }

    fn fresh_execution_options(
        &self,
        language: Option<&str>,
    ) -> Option<crate::query::ExecutionOptions> {
        if self.active_execution_control.lock().is_some() {
            return None;
        }
        Some(crate::query::ExecutionOptions {
            control: grafeo_core::execution::QueryExecutionControl::new(),
            language: language.map(str::to_owned),
            result_limits: None,
            result_admission: None,
        })
    }

    /// Composes the Session deadline once into the execution's actual owner.
    pub(crate) fn compose_execution_control(
        &self,
        control: grafeo_core::execution::QueryExecutionControl,
    ) -> Result<grafeo_core::execution::QueryExecutionControl> {
        #[cfg(not(target_arch = "wasm32"))]
        let control = {
            let deadline = self
                .query_timeout
                .map(|timeout| {
                    std::time::Instant::now()
                        .checked_add(timeout)
                        .ok_or_else(|| {
                            grafeo_common::utils::error::Error::Query(
                                grafeo_common::utils::error::QueryError::new(
                                    grafeo_common::utils::error::QueryErrorKind::Semantic,
                                    "session query timeout exceeds the platform clock range",
                                ),
                            )
                        })
                })
                .transpose()?;
            control.with_additional_deadline(deadline, self.query_timeout)
        };

        #[cfg(target_arch = "wasm32")]
        if self.query_timeout.is_some() {
            return Err(grafeo_common::utils::error::Error::Query(
                grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Unsupported,
                    "session deadlines require a qualified monotonic clock on this target",
                ),
            ));
        }
        Ok(control)
    }

    /// Executes a query with caller-owned cancellation.
    ///
    /// The supplied [`crate::ExecutionOptions::control`] is consumed. A cancelled
    /// control cannot be reset; later queries need a fresh control.
    ///
    /// # Errors
    ///
    /// Returns a typed cancellation error when the control is already
    /// cancelled or a deadline has elapsed, and otherwise the same failures
    /// as [`Self::execute`] / [`Self::execute_language`].
    pub fn execute_with_options(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
        options: crate::query::ExecutionOptions,
    ) -> Result<QueryResult> {
        let _operation = self.session_operation_guard();
        self.check_not_poisoned()?;
        let crate::query::ExecutionOptions {
            control,
            language,
            result_limits,
            result_admission,
        } = options;
        let control = self.compose_execution_control(control)?;
        let previous_limits = self
            .active_result_limits
            .lock()
            .replace(result_limits.unwrap_or(self.result_limits));
        let previous_admission =
            std::mem::replace(&mut *self.active_result_admission.lock(), result_admission);
        let previous = self.active_execution_control.lock().replace(control);
        let previous_completed = self
            .active_execution_completed
            .swap(false, Ordering::AcqRel);

        let previous_depth = self
            .active_execution_statement_depth
            .swap(0, Ordering::AcqRel);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = (|| {
                let checkpoint = {
                    let guard = self.active_execution_control.lock();
                    let Some(control) = guard.as_ref() else {
                        return Err(grafeo_common::utils::error::Error::Internal(
                            "query execution control is not installed".to_string(),
                        ));
                    };
                    control.checkpoint()
                };
                Self::check_query_checkpoint(&checkpoint)?;
                match language.as_deref() {
                    None => {
                        if params.is_empty() {
                            self.execute(query)
                        } else {
                            self.execute_with_params(query, params)
                        }
                    }
                    Some(name) => {
                        let params = if params.is_empty() {
                            None
                        } else {
                            Some(params)
                        };
                        self.execute_language(query, name, params)
                    }
                }
            })();
            result.and_then(|value| {
                let value = self.admit_query_result(value)?;
                self.complete_active_execution()?;
                Ok(value)
            })
        }));
        *self.active_execution_control.lock() = previous;
        *self.active_result_limits.lock() = previous_limits;
        *self.active_result_admission.lock() = previous_admission;
        self.active_execution_completed
            .store(previous_completed, Ordering::Release);
        self.active_execution_statement_depth
            .store(previous_depth, Ordering::Release);
        match outcome {
            Ok(result) => result,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Executes a GQL query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if no query language is enabled.
    #[cfg(not(any(feature = "gql", feature = "cypher")))]
    pub fn execute_with_params(
        &self,
        _query: &str,
        _params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        self.check_not_poisoned()?;
        Err(grafeo_common::utils::error::Error::Query(
            grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Unsupported,
                "Default query execution requires GQL or Cypher",
            ),
        ))
    }

    /// Executes a GQL query.
    ///
    /// # Errors
    ///
    /// Returns an error if no query language is enabled.
    #[cfg(not(any(feature = "gql", feature = "cypher")))]
    pub fn execute(&self, _query: &str) -> Result<QueryResult> {
        self.check_not_poisoned()?;
        Err(grafeo_common::utils::error::Error::Query(
            grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Unsupported,
                "Default query execution requires GQL or Cypher",
            ),
        ))
    }

    /// Executes a Cypher query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "cypher")]
    pub fn execute_cypher(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            binder::Binder, cache::CacheKey, optimizer::Optimizer, processor::QueryLanguage,
            translators::cypher,
        };

        let _execution_operation = self.session_operation_guard();
        if let Some(options) = self.fresh_execution_options(Some("cypher")) {
            return self.execute_with_options(query, std::collections::HashMap::new(), options);
        }

        let _operation = self.session_operation_guard();
        self.check_not_poisoned()?;
        self.require_lpg("Cypher")?;

        if let Some(cached) = self.execute_cached_physical(query, QueryLanguage::Cypher) {
            return self.finish_query(cached);
        }

        // Handle schema DDL and SHOW commands before the normal query path
        self.check_active_execution()?;
        let translation = cypher::translate_full(query)?;
        self.check_active_execution()?;
        match translation {
            #[cfg(feature = "lpg")]
            cypher::CypherTranslationResult::SchemaCommand(cmd) => {
                let result = self.execute_schema_command_checked(cmd);
                return self.finish_query(result);
            }
            #[cfg(not(feature = "lpg"))]
            cypher::CypherTranslationResult::SchemaCommand(_) => {
                return Err(grafeo_common::utils::error::Error::Internal(
                    "Schema DDL requires the `lpg` feature".to_string(),
                ));
            }
            cypher::CypherTranslationResult::ShowIndexes => {
                let _read_barrier = self.publication_read_guard();
                return self.finish_query(self.execute_show_indexes());
            }
            cypher::CypherTranslationResult::ShowConstraints => {
                let _read_barrier = self.publication_read_guard();
                return self.finish_query(self.execute_show_constraints());
            }
            cypher::CypherTranslationResult::ShowCurrentGraphType => {
                let _read_barrier = self.publication_read_guard();
                return self.finish_query(self.execute_show_current_graph_type());
            }
            cypher::CypherTranslationResult::Plan(_) => {
                // Fall through to normal execution below
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        let start_time = std::time::Instant::now();

        // Create cache key for this query
        let cache_key =
            CacheKey::with_graph(query, QueryLanguage::Cypher, self.current_graph_path());

        // Try to get cached optimized plan
        let optimized_plan = if let Some(cached_plan) = self.query_cache.get_optimized(&cache_key) {
            cached_plan
        } else {
            // Parse and translate the query to a logical plan
            self.check_active_execution()?;
            let logical_plan = cypher::translate(query)?;
            self.check_active_execution()?;

            // Semantic validation
            let mut binder = Binder::new();
            self.check_active_execution()?;
            let _binding_context = binder.bind(&logical_plan)?;
            self.check_active_execution()?;

            // Optimize the plan
            let active = self.active_store();
            let optimizer = Optimizer::from_graph_store(&*active);
            self.check_active_execution()?;
            let plan = optimizer.optimize(logical_plan)?;
            self.check_active_execution()?;

            // Cache the optimized plan
            self.query_cache.put_optimized(cache_key, plan.clone());

            plan
        };

        let qualified = self.qualify_lpg_plan(&optimized_plan.root)?;

        // EXPLAIN
        if optimized_plan.explain {
            use crate::query::processor::{annotate_pushdown_hints, explain_result};
            let _read_barrier = self.publication_read_guard();
            let active = self.active_store();
            let mut plan = optimized_plan;
            annotate_pushdown_hints(&mut plan.root, active.as_ref());
            return self.finish_query(explain_result(
                &plan,
                self.result_resources()?,
                self.effective_result_limits(),
            ));
        }

        // PROFILE
        if optimized_plan.profile {
            let has_mutations = qualified.mutates;
            let result =
                self.with_lpg_plan_auto_commit(has_mutations, qualified.contains_call, || {
                    let _read_barrier = if !has_mutations || qualified.contains_call {
                        self.publication_read_guard()
                    } else {
                        None
                    };
                    self.validate_qualified_lpg_plan(&qualified)?;
                    let (graph_context, active) = self.active_store_with_graph_context();
                    let (viewing_epoch, transaction_id) = self.get_transaction_context();
                    let planner = self.create_planner_for_store_with_graph_context(
                        Arc::clone(&active),
                        viewing_epoch,
                        transaction_id,
                        false,
                        &graph_context,
                    );
                    #[cfg(any(feature = "lpg", feature = "algos"))]
                    let planner = self.attach_qualified_procedures(planner, &qualified);
                    self.check_active_execution()?;
                    let (physical_plan, entries) = planner.plan_profiled(&optimized_plan)?;
                    self.check_active_execution()?;
                    let _result = self.execute_physical_plan_with_checkpoint(
                        physical_plan,
                        self.installed_or_fresh_checkpoint(),
                        &entries,
                    )?;

                    let total_time_ms;
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        total_time_ms = 0.0;
                    }

                    let profile_tree = crate::query::profile::build_profile_tree(
                        &optimized_plan.root,
                        &mut entries.into_iter(),
                    );
                    crate::query::profile::profile_result(
                        &profile_tree,
                        total_time_ms,
                        self.result_resources()?,
                        self.effective_result_limits(),
                    )
                });
            return self.finish_query(result);
        }

        let has_mutations = qualified.mutates;

        let result = self.with_lpg_plan_auto_commit(has_mutations, qualified.contains_call, || {
            let _read_barrier = if !has_mutations || qualified.contains_call {
                self.publication_read_guard()
            } else {
                None
            };
            self.validate_qualified_lpg_plan(&qualified)?;
            let graph_context = self.graph_context_snapshot();
            let active = self.store_for_graph_storage_key(&graph_context.storage_key);
            // Get transaction context for MVCC visibility
            let (viewing_epoch, transaction_id) = self.get_transaction_context();

            // Convert to physical plan with transaction context
            let planner = self.create_planner_for_store_with_graph_context(
                Arc::clone(&active),
                viewing_epoch,
                transaction_id,
                false,
                &graph_context,
            );
            #[cfg(any(feature = "lpg", feature = "algos"))]
            let planner = self.attach_qualified_procedures(planner, &qualified);
            self.check_active_execution()?;
            let mut physical_plan = planner.plan(&optimized_plan)?;
            self.check_active_execution()?;
            let cache_key = (!has_mutations && !qualified.contains_call)
                .then(|| {
                    Self::physical_cache_key_for_context(
                        query,
                        QueryLanguage::Cypher,
                        graph_context,
                        viewing_epoch,
                        transaction_id,
                    )
                })
                .flatten();
            if let Some(key) = cache_key {
                let result = self.execute_borrowed_physical_plan(&mut physical_plan);
                self.store_cached_physical(key, physical_plan);
                result
            } else {
                self.execute_physical_plan(physical_plan)
            }
        });

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("cypher", elapsed_ms, &result);
        }

        self.finish_query(result)
    }

    /// Executes a Gremlin query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// // Create some nodes first
    /// session.create_node(&["Person"]);
    ///
    /// // Query using Gremlin
    /// let result = session.execute_gremlin("g.V().hasLabel('Person')")?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "gremlin")]
    pub fn execute_gremlin(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{binder::Binder, optimizer::Optimizer, translators::gremlin};

        let _execution_operation = self.session_operation_guard();
        if let Some(options) = self.fresh_execution_options(Some("gremlin")) {
            return self.execute_with_options(query, std::collections::HashMap::new(), options);
        }

        let _operation = self.session_operation_guard();
        self.check_not_poisoned()?;

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        // Parse and translate the query to a logical plan
        self.check_active_execution()?;
        let logical_plan = gremlin::translate(query)?;
        self.check_active_execution()?;

        // Semantic validation
        let mut binder = Binder::new();
        self.check_active_execution()?;
        let _binding_context = binder.bind(&logical_plan)?;
        self.check_active_execution()?;

        // Optimize the plan
        let optimized_plan = {
            let active = self.active_store();
            let optimizer = Optimizer::from_graph_store(&*active);
            self.check_active_execution()?;
            let optimized = optimizer.optimize(logical_plan)?;
            self.check_active_execution()?;
            optimized
        };

        let qualified = self.qualify_lpg_plan(&optimized_plan.root)?;
        let has_mutations = qualified.mutates;

        let result = self.with_lpg_plan_auto_commit(has_mutations, qualified.contains_call, || {
            let _read_barrier = if !has_mutations || qualified.contains_call {
                self.publication_read_guard()
            } else {
                None
            };
            self.validate_qualified_lpg_plan(&qualified)?;
            let (graph_context, active) = self.active_store_with_graph_context();
            // Get transaction context for MVCC visibility
            let (viewing_epoch, transaction_id) = self.get_transaction_context();

            // Convert to physical plan with transaction context
            let planner = self.create_planner_for_store_with_graph_context(
                Arc::clone(&active),
                viewing_epoch,
                transaction_id,
                false,
                &graph_context,
            );
            #[cfg(any(feature = "lpg", feature = "algos"))]
            let planner = self.attach_qualified_procedures(planner, &qualified);
            self.check_active_execution()?;
            let physical_plan = planner.plan(&optimized_plan)?;
            self.check_active_execution()?;
            self.execute_physical_plan(physical_plan)
        });

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("gremlin", elapsed_ms, &result);
        }

        self.finish_query(result)
    }

    /// Executes a Gremlin query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "gremlin")]
    pub fn execute_gremlin_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::{
            binder::Binder, optimizer::Optimizer, processor::substitute_params,
            translators::gremlin,
        };

        let _execution_operation = self.session_operation_guard();
        if let Some(options) = self.fresh_execution_options(Some("gremlin")) {
            return self.execute_with_options(query, params, options);
        }

        let _operation = self.session_operation_guard();
        self.check_not_poisoned()?;

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        // Parse and translate the query to a logical plan
        self.check_active_execution()?;
        let mut logical_plan = gremlin::translate(query)?;
        self.check_active_execution()?;

        // Substitute parameters
        self.check_active_execution()?;
        substitute_params(&mut logical_plan, &params)?;
        self.check_active_execution()?;

        // Semantic validation
        let mut binder = Binder::new();
        self.check_active_execution()?;
        let _binding_context = binder.bind(&logical_plan)?;
        self.check_active_execution()?;

        // Optimize the plan
        let optimized_plan = {
            let active = self.active_store();
            let optimizer = Optimizer::from_graph_store(&*active);
            self.check_active_execution()?;
            let optimized = optimizer.optimize(logical_plan)?;
            self.check_active_execution()?;
            optimized
        };

        let qualified = self.qualify_lpg_plan(&optimized_plan.root)?;
        let has_mutations = qualified.mutates;

        let result = self.with_lpg_plan_auto_commit(has_mutations, qualified.contains_call, || {
            let _read_barrier = if !has_mutations || qualified.contains_call {
                self.publication_read_guard()
            } else {
                None
            };
            self.validate_qualified_lpg_plan(&qualified)?;
            let (graph_context, active) = self.active_store_with_graph_context();
            let (viewing_epoch, transaction_id) = self.get_transaction_context();
            let planner = self.create_planner_for_store_with_graph_context(
                Arc::clone(&active),
                viewing_epoch,
                transaction_id,
                false,
                &graph_context,
            );
            #[cfg(any(feature = "lpg", feature = "algos"))]
            let planner = self.attach_qualified_procedures(planner, &qualified);
            self.check_active_execution()?;
            let physical_plan = planner.plan(&optimized_plan)?;
            self.check_active_execution()?;
            self.execute_physical_plan(physical_plan)
        });

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("gremlin", elapsed_ms, &result);
        }

        self.finish_query(result)
    }

    /// Executes a GraphQL query against the LPG store.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// // Create some nodes first
    /// session.create_node(&["User"]);
    ///
    /// // Query using GraphQL
    /// let result = session.execute_graphql("query { user { id name } }")?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "graphql")]
    pub fn execute_graphql(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            binder::Binder, optimizer::Optimizer, processor::substitute_params,
            translators::graphql,
        };

        let _execution_operation = self.session_operation_guard();
        if let Some(options) = self.fresh_execution_options(Some("graphql")) {
            return self.execute_with_options(query, std::collections::HashMap::new(), options);
        }

        let _operation = self.session_operation_guard();
        self.check_not_poisoned()?;
        self.require_lpg("GraphQL")?;

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        self.check_active_execution()?;
        let mut logical_plan = graphql::translate(query)?;
        self.check_active_execution()?;

        // Substitute default parameter values from variable declarations
        if !logical_plan.default_params.is_empty() {
            let defaults = logical_plan.default_params.clone();
            self.check_active_execution()?;
            substitute_params(&mut logical_plan, &defaults)?;
            self.check_active_execution()?;
        }

        let mut binder = Binder::new();
        self.check_active_execution()?;
        let _binding_context = binder.bind(&logical_plan)?;
        self.check_active_execution()?;

        let optimized_plan = {
            let active = self.active_store();
            let optimizer = Optimizer::from_graph_store(&*active);
            self.check_active_execution()?;
            let optimized = optimizer.optimize(logical_plan)?;
            self.check_active_execution()?;
            optimized
        };
        let qualified = self.qualify_lpg_plan(&optimized_plan.root)?;
        let has_mutations = qualified.mutates;

        let result = self.with_lpg_plan_auto_commit(has_mutations, qualified.contains_call, || {
            let _read_barrier = if !has_mutations || qualified.contains_call {
                self.publication_read_guard()
            } else {
                None
            };
            self.validate_qualified_lpg_plan(&qualified)?;
            let (graph_context, active) = self.active_store_with_graph_context();
            let (viewing_epoch, transaction_id) = self.get_transaction_context();
            let planner = self.create_planner_for_store_with_graph_context(
                Arc::clone(&active),
                viewing_epoch,
                transaction_id,
                false,
                &graph_context,
            );
            #[cfg(any(feature = "lpg", feature = "algos"))]
            let planner = self.attach_qualified_procedures(planner, &qualified);
            self.check_active_execution()?;
            let physical_plan = planner.plan(&optimized_plan)?;
            self.check_active_execution()?;
            self.execute_physical_plan(physical_plan)
        });

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("graphql", elapsed_ms, &result);
        }

        self.finish_query(result)
    }

    /// Executes a GraphQL query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::{
            binder::Binder, optimizer::Optimizer, processor::substitute_params,
            translators::graphql,
        };

        let _execution_operation = self.session_operation_guard();
        if let Some(options) = self.fresh_execution_options(Some("graphql")) {
            return self.execute_with_options(query, params, options);
        }

        let _operation = self.session_operation_guard();
        self.check_not_poisoned()?;
        self.require_lpg("GraphQL")?;

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        // Parse and translate the query to a logical plan
        self.check_active_execution()?;
        let mut logical_plan = graphql::translate(query)?;
        self.check_active_execution()?;

        // Merge default params with caller-supplied params
        if !logical_plan.default_params.is_empty() {
            let mut merged = logical_plan.default_params.clone();
            merged.extend(params.iter().map(|(k, v)| (k.clone(), v.clone())));
            self.check_active_execution()?;
            substitute_params(&mut logical_plan, &merged)?;
            self.check_active_execution()?;
        } else {
            self.check_active_execution()?;
            substitute_params(&mut logical_plan, &params)?;
            self.check_active_execution()?;
        }

        // Semantic validation
        let mut binder = Binder::new();
        self.check_active_execution()?;
        let _binding_context = binder.bind(&logical_plan)?;
        self.check_active_execution()?;

        // Optimize the plan
        let optimized_plan = {
            let active = self.active_store();
            let optimizer = Optimizer::from_graph_store(&*active);
            self.check_active_execution()?;
            let optimized = optimizer.optimize(logical_plan)?;
            self.check_active_execution()?;
            optimized
        };

        let qualified = self.qualify_lpg_plan(&optimized_plan.root)?;
        let has_mutations = qualified.mutates;

        let result = self.with_lpg_plan_auto_commit(has_mutations, qualified.contains_call, || {
            let _read_barrier = if !has_mutations || qualified.contains_call {
                self.publication_read_guard()
            } else {
                None
            };
            self.validate_qualified_lpg_plan(&qualified)?;
            let (graph_context, active) = self.active_store_with_graph_context();
            let (viewing_epoch, transaction_id) = self.get_transaction_context();
            let planner = self.create_planner_for_store_with_graph_context(
                Arc::clone(&active),
                viewing_epoch,
                transaction_id,
                false,
                &graph_context,
            );
            #[cfg(any(feature = "lpg", feature = "algos"))]
            let planner = self.attach_qualified_procedures(planner, &qualified);
            self.check_active_execution()?;
            let physical_plan = planner.plan(&optimized_plan)?;
            self.check_active_execution()?;
            self.execute_physical_plan(physical_plan)
        });

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("graphql", elapsed_ms, &result);
        }

        self.finish_query(result)
    }

    /// Executes a SQL/PGQ query (SQL:2023 GRAPH_TABLE).
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// let result = session.execute_sql(
    ///     "SELECT * FROM GRAPH_TABLE (
    ///         MATCH (n:Person)
    ///         COLUMNS (n.name AS name)
    ///     )"
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "sql-pgq")]
    pub fn execute_sql(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            binder::Binder, cache::CacheKey, optimizer::Optimizer, plan::LogicalOperator,
            processor::QueryLanguage, translators::sql_pgq,
        };

        let _execution_operation = self.session_operation_guard();
        if let Some(options) = self.fresh_execution_options(Some("sql")) {
            return self.execute_with_options(query, std::collections::HashMap::new(), options);
        }

        let _operation = self.session_operation_guard();
        self.check_not_poisoned()?;

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        // Parse and translate (always needed to check for DDL)
        self.check_active_execution()?;
        let logical_plan = sql_pgq::translate(query)?;
        self.check_active_execution()?;

        // Handle DDL statements directly (they don't go through the query pipeline)
        if let LogicalOperator::CreatePropertyGraph(ref cpg) = logical_plan.root {
            self.require_permission(crate::auth::StatementKind::Admin)?;
            return self.finish_query(crate::query::executor::bounded_text_result(
                "status",
                self.result_resources()?,
                self.effective_result_limits(),
                |writer| {
                    write!(
                        writer,
                        "Property graph '{}' created ({} node tables, {} edge tables)",
                        cpg.name,
                        cpg.node_tables.len(),
                        cpg.edge_tables.len()
                    )
                },
            ));
        }

        let cache_key =
            CacheKey::with_graph(query, QueryLanguage::SqlPgq, self.current_graph_path());

        let optimized_plan = if let Some(cached_plan) = self.query_cache.get_optimized(&cache_key) {
            cached_plan
        } else {
            let mut binder = Binder::new();
            self.check_active_execution()?;
            let _binding_context = binder.bind(&logical_plan)?;
            self.check_active_execution()?;
            let active = self.active_store();
            let optimizer = Optimizer::from_graph_store(&*active);
            self.check_active_execution()?;
            let plan = optimizer.optimize(logical_plan)?;
            self.check_active_execution()?;
            self.query_cache.put_optimized(cache_key, plan.clone());
            plan
        };

        let qualified = self.qualify_lpg_plan(&optimized_plan.root)?;
        let has_mutations = qualified.mutates;

        let result = self.with_lpg_plan_auto_commit(has_mutations, qualified.contains_call, || {
            let _read_barrier = if !has_mutations || qualified.contains_call {
                self.publication_read_guard()
            } else {
                None
            };
            self.validate_qualified_lpg_plan(&qualified)?;
            let (graph_context, active) = self.active_store_with_graph_context();
            let (viewing_epoch, transaction_id) = self.get_transaction_context();
            let planner = self.create_planner_for_store_with_graph_context(
                Arc::clone(&active),
                viewing_epoch,
                transaction_id,
                false,
                &graph_context,
            );
            #[cfg(any(feature = "lpg", feature = "algos"))]
            let planner = self.attach_qualified_procedures(planner, &qualified);
            self.check_active_execution()?;
            let physical_plan = planner.plan(&optimized_plan)?;
            self.check_active_execution()?;
            self.execute_physical_plan(physical_plan)
        });

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("sql", elapsed_ms, &result);
        }

        self.finish_query(result)
    }

    /// Executes a SQL/PGQ query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "sql-pgq")]
    pub fn execute_sql_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::processor::substitute_params;

        let _execution_operation = self.session_operation_guard();
        if let Some(options) = self.fresh_execution_options(Some("sql")) {
            return self.execute_with_options(query, params, options);
        }
        use crate::query::translators::sql_pgq;
        use crate::query::{binder::Binder, optimizer::Optimizer};

        let _operation = self.session_operation_guard();
        self.check_not_poisoned()?;

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        self.check_active_execution()?;
        let mut logical_plan = sql_pgq::translate(query)?;
        self.check_active_execution()?;
        substitute_params(&mut logical_plan, &params)?;
        self.check_active_execution()?;
        let mut binder = Binder::new();
        self.check_active_execution()?;
        let _binding_context = binder.bind(&logical_plan)?;
        self.check_active_execution()?;
        let optimized_plan = {
            let active = self.active_store();
            self.check_active_execution()?;
            let optimized = Optimizer::from_graph_store(active.as_ref()).optimize(logical_plan)?;
            self.check_active_execution()?;
            optimized
        };
        let qualified = self.qualify_lpg_plan(&optimized_plan.root)?;
        let has_mutations = qualified.mutates;

        let result = self.with_lpg_plan_auto_commit(has_mutations, qualified.contains_call, || {
            let _read_barrier = if !has_mutations || qualified.contains_call {
                self.publication_read_guard()
            } else {
                None
            };
            self.validate_qualified_lpg_plan(&qualified)?;
            let graph_context = self.graph_context_snapshot();
            let active = self.store_for_graph_storage_key(&graph_context.storage_key);
            let (viewing_epoch, transaction_id) = self.get_transaction_context();
            let planner = self.create_planner_for_store_with_graph_context(
                Arc::clone(&active),
                viewing_epoch,
                transaction_id,
                !has_mutations && transaction_id.is_none(),
                &graph_context,
            );
            #[cfg(any(feature = "lpg", feature = "algos"))]
            let planner = self.attach_qualified_procedures(planner, &qualified);
            self.check_active_execution()?;
            let physical_plan = planner.plan(&optimized_plan)?;
            self.check_active_execution()?;
            self.execute_physical_plan(physical_plan)
        });

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("sql", elapsed_ms, &result);
        }

        self.finish_query(result)
    }

    /// Executes a query in the specified language by name.
    ///
    /// Supported language names: `"gql"`, `"cypher"`, `"gremlin"`, `"graphql"`,
    /// `"graphql-rdf"`, `"sparql"`, `"sql"`. Each requires the corresponding feature flag.
    ///
    /// # Errors
    ///
    /// Returns an error if the language is unknown/disabled or the query fails.
    pub fn execute_language(
        &self,
        query: &str,
        language: &str,
        params: Option<std::collections::HashMap<String, Value>>,
    ) -> Result<QueryResult> {
        let _execution_operation = self.session_operation_guard();
        if let Some(options) = self.fresh_execution_options(Some(language)) {
            return self.execute_with_options(query, params.unwrap_or_default(), options);
        }
        #[cfg(any(feature = "lpg", feature = "triple-store"))]
        let _operation = self.session_operation_guard();
        self.check_not_poisoned()?;
        let _span = grafeo_info_span!(
            "grafeo::session::execute",
            language,
            query_len = query.len(),
        );
        match language {
            #[cfg(feature = "gql")]
            "gql" => {
                if let Some(p) = params {
                    self.execute_with_params(query, p)
                } else {
                    self.execute(query)
                }
            }
            #[cfg(feature = "cypher")]
            "cypher" => {
                if let Some(p) = params {
                    use crate::query::processor::substitute_params;
                    use crate::query::translators::cypher;
                    use crate::query::{binder::Binder, optimizer::Optimizer};

                    #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
                    let start_time = Instant::now();

                    self.check_active_execution()?;
                    let mut logical_plan = cypher::translate(query)?;
                    self.check_active_execution()?;
                    substitute_params(&mut logical_plan, &p)?;
                    self.check_active_execution()?;
                    let mut binder = Binder::new();
                    self.check_active_execution()?;
                    let _binding_context = binder.bind(&logical_plan)?;
                    self.check_active_execution()?;
                    let optimized_plan = {
                        let active = self.active_store();
                        self.check_active_execution()?;
                        let optimized =
                            Optimizer::from_graph_store(active.as_ref()).optimize(logical_plan)?;
                        self.check_active_execution()?;
                        optimized
                    };
                    let qualified = self.qualify_lpg_plan(&optimized_plan.root)?;
                    let has_mutations = qualified.mutates;
                    let result = self.with_lpg_plan_auto_commit(
                        has_mutations,
                        qualified.contains_call,
                        || {
                            let _read_barrier = if !has_mutations || qualified.contains_call {
                                self.publication_read_guard()
                            } else {
                                None
                            };
                            self.validate_qualified_lpg_plan(&qualified)?;
                            let graph_context = self.graph_context_snapshot();
                            let active =
                                self.store_for_graph_storage_key(&graph_context.storage_key);
                            let (viewing_epoch, transaction_id) = self.get_transaction_context();
                            let planner = self.create_planner_for_store_with_graph_context(
                                Arc::clone(&active),
                                viewing_epoch,
                                transaction_id,
                                !has_mutations && transaction_id.is_none(),
                                &graph_context,
                            );
                            #[cfg(any(feature = "lpg", feature = "algos"))]
                            let planner = self.attach_qualified_procedures(planner, &qualified);
                            self.check_active_execution()?;
                            let physical_plan = planner.plan(&optimized_plan)?;
                            self.check_active_execution()?;
                            self.execute_physical_plan(physical_plan)
                        },
                    );

                    #[cfg(feature = "metrics")]
                    {
                        #[cfg(not(target_arch = "wasm32"))]
                        let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
                        #[cfg(target_arch = "wasm32")]
                        let elapsed_ms = None;
                        self.record_query_metrics("cypher", elapsed_ms, &result);
                    }

                    self.finish_query(result)
                } else {
                    self.execute_cypher(query)
                }
            }
            #[cfg(feature = "gremlin")]
            "gremlin" => {
                if let Some(p) = params {
                    self.execute_gremlin_with_params(query, p)
                } else {
                    self.execute_gremlin(query)
                }
            }
            #[cfg(feature = "graphql")]
            "graphql" => {
                if let Some(p) = params {
                    self.execute_graphql_with_params(query, p)
                } else {
                    self.execute_graphql(query)
                }
            }
            #[cfg(all(feature = "graphql", feature = "triple-store"))]
            "graphql-rdf" => {
                if let Some(p) = params {
                    self.execute_graphql_rdf_with_params(query, p)
                } else {
                    self.execute_graphql_rdf(query)
                }
            }
            #[cfg(feature = "sql-pgq")]
            "sql" | "sql-pgq" => {
                if let Some(p) = params {
                    self.execute_sql_with_params(query, p)
                } else {
                    self.execute_sql(query)
                }
            }
            #[cfg(all(feature = "sparql", feature = "triple-store"))]
            "sparql" => {
                if let Some(p) = params {
                    self.execute_sparql_with_params(query, p)
                } else {
                    self.execute_sparql(query)
                }
            }
            other => Err(grafeo_common::utils::error::Error::Query(
                grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Semantic,
                    format!("Unknown query language: '{other}'"),
                ),
            )),
        }
    }

    /// Begins a new transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if the database is closed or durability-poisoned, a
    /// mixed snapshot is active, or nested savepoint creation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let mut session = db.session();
    ///
    /// session.begin_transaction()?;
    /// session.execute("INSERT (:Person {name: 'Alix'})")?;
    /// session.execute("INSERT (:Person {name: 'Gus'})")?;
    /// session.commit()?; // Both inserts committed atomically
    /// # Ok(())
    /// # }
    /// ```
    /// Clears all cached query plans.
    ///
    /// The plan cache is shared across all sessions on the same database,
    /// so clearing from one session affects all sessions.
    pub fn clear_plan_cache(&self) {
        self.query_cache.clear();
        self.physical_cache.lock().clear();
    }

    #[cfg(any(feature = "gql", feature = "cypher"))]
    fn physical_cache_key(&self, query: &str, language: QueryLanguage) -> Option<PhysicalCacheKey> {
        let graph_context = self.graph_context_snapshot();
        let (epoch, tx) = self.get_transaction_context();
        Self::physical_cache_key_for_context(query, language, graph_context, epoch, tx)
    }

    #[cfg(any(feature = "gql", feature = "cypher"))]
    fn physical_cache_key_for_context(
        query: &str,
        language: QueryLanguage,
        graph_context: SessionGraphContext,
        epoch: EpochId,
        tx: Option<TransactionId>,
    ) -> Option<PhysicalCacheKey> {
        PhysicalCacheKey::new(
            query,
            language,
            graph_context.storage_key,
            graph_context.schema,
            graph_context.graph,
            epoch,
            tx,
        )
    }

    #[cfg(any(feature = "gql", feature = "cypher"))]
    fn take_cached_physical(&self, key: &PhysicalCacheKey) -> Option<PhysicalPlan> {
        self.physical_cache.lock().take(key)
    }

    #[cfg(any(feature = "gql", feature = "cypher"))]
    fn store_cached_physical(&self, key: PhysicalCacheKey, mut plan: PhysicalPlan) {
        plan.operator.reset();
        self.physical_cache.lock().insert(key, plan);
    }

    #[cfg(any(feature = "gql", feature = "cypher"))]
    fn execute_cached_physical(
        &self,
        query: &str,
        language: QueryLanguage,
    ) -> Option<Result<QueryResult>> {
        let _operation = self.session_operation_guard();
        let _historical = self.historical_view_operation_gate.lock();
        // Both initial cache writers admit only classified nonmutating plans
        // without CALL. Reinsertion preserves that invariant, so historical
        // reads can use their epoch-qualified key too. Procedure calls always
        // return through logical qualification and remain rejected at an epoch.
        // Catalog publication clears this cache while holding the exclusive
        // barrier. Acquire its read side before deriving the epoch key or
        // removing a plan, and retain it through reset, execution, and
        // reinsertion. Otherwise a DROP/REPLACE PROCEDURE could clear the
        // cache while an extracted old body executes and is reinserted.
        let _read_barrier = self.publication_read_guard();
        let key = self.physical_cache_key(query, language)?;
        let mut plan = self.take_cached_physical(&key)?;
        if let Err(error) = self.check_lpg_query_access(false) {
            self.store_cached_physical(key, plan);
            return Some(Err(error));
        }
        plan.operator.reset();
        let result = self.execute_borrowed_physical_plan(&mut plan);
        self.store_cached_physical(key, plan);
        Some(result)
    }

    /// Begins a new transaction on this session.
    ///
    /// Uses the default isolation level (`SnapshotIsolation`).
    ///
    /// # Errors
    ///
    /// Returns an error if the database is closed or durability-poisoned, a
    /// mixed snapshot is active, or nested savepoint creation fails.
    #[cfg(feature = "lpg")]
    pub fn begin_transaction(&mut self) -> Result<()> {
        self.begin_transaction_inner(false, None)
    }

    /// Begins an RDF-only transaction at the default Snapshot Isolation level.
    ///
    /// # Errors
    ///
    /// Returns an error if the database is closed or durability-poisoned, a
    /// mixed snapshot is active, or nested savepoint creation fails.
    #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
    pub fn begin_transaction(&mut self) -> Result<()> {
        self.begin_rdf_auto()
    }

    /// Begins a transaction with a specific isolation level.
    ///
    /// See [`begin_transaction`](Self::begin_transaction) for the default (`SnapshotIsolation`).
    ///
    /// # Errors
    ///
    /// Returns an error if the database is closed or durability-poisoned, a
    /// mixed snapshot is active, or nested savepoint creation fails.
    #[cfg(feature = "lpg")]
    pub fn begin_transaction_with_isolation(
        &mut self,
        isolation_level: crate::transaction::IsolationLevel,
    ) -> Result<()> {
        self.begin_transaction_inner(false, Some(isolation_level))
    }

    /// Begins an RDF-only transaction with an explicit isolation level.
    ///
    /// # Errors
    ///
    /// Returns an error if the database is closed or poisoned, a mixed
    /// snapshot is active, or nested savepoint creation fails.
    #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
    pub fn begin_transaction_with_isolation(
        &mut self,
        isolation_level: crate::transaction::IsolationLevel,
    ) -> Result<()> {
        self.begin_rdf_with_isolation(isolation_level)
    }

    /// Sets the conflict-detection granularity for the NEXT Serializable
    /// transaction begun on this session.
    ///
    /// Changing this setting mid-transaction has no effect until the next
    /// `begin_transaction_with_isolation` call.  The granularity is ignored
    /// for `SnapshotIsolation` and `ReadCommitted` transactions.
    ///
    /// See [`ConflictGranularity`] for
    /// the tradeoffs between `Entity` (the conservative default) and `Property`
    /// (finer-grained, fewer false aborts).
    pub fn set_conflict_granularity(&self, granularity: crate::transaction::ConflictGranularity) {
        *self.conflict_granularity.lock() = granularity;
    }

    /// Core transaction begin logic, usable from both `&mut self` and `&self` paths.
    #[cfg(feature = "lpg")]
    fn begin_transaction_inner(
        &self,
        read_only: bool,
        isolation_level: Option<crate::transaction::IsolationLevel>,
    ) -> Result<()> {
        let _operation = self.session_operation_guard();
        let _historical = self.historical_view_operation_gate.lock();
        self.transaction_manager.with_write_authority(|| {
            self.begin_transaction_inner_authorized(read_only, isolation_level)
        })
    }

    #[cfg(feature = "lpg")]
    fn begin_transaction_inner_authorized(
        &self,
        read_only: bool,
        isolation_level: Option<crate::transaction::IsolationLevel>,
    ) -> Result<()> {
        self.check_not_poisoned()?;
        self.check_not_in_mixed_snapshot()?;
        if matches!(
            self.lpg_backend,
            LpgBackend::Placeholder {
                commit_target_available: false,
            }
        ) {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "external store does not support native LPG transactions".to_owned(),
                ),
            ));
        }
        // Lifecycle precedes publication in the global lock order. Retaining
        // this guard through either nested savepoint creation or transaction
        // allocation prevents close() from admitting work after shutdown.
        let database_open = self.database_open.read();
        if !*database_open {
            return Err(Self::database_closed_error());
        }
        let nested = self.current_transaction.lock().is_some();
        if nested {
            let mut depth = self.transaction_nesting_depth.lock();
            let next_depth = depth.checked_add(1).ok_or_else(|| {
                grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::InvalidState(
                        "nested transaction depth exhausted".to_string(),
                    ),
                )
            })?;
            let savepoint_name = format!("{INTERNAL_SAVEPOINT_PREFIX}nested:{next_depth}");
            self.savepoint_authorized(&savepoint_name)?;
            *depth = next_depth;
            return Ok(());
        }
        let _publication = self.transaction_manager.publication().read();
        let _span = grafeo_debug_span!("grafeo::tx::begin", read_only);
        let mut current = self.current_transaction.lock();
        if current.is_some() {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "transaction state changed while beginning a transaction".to_string(),
                ),
            ));
        }

        let transaction_id = if let Some(level) = isolation_level {
            self.transaction_manager.begin_with_isolation(level)
        } else {
            self.transaction_manager.begin()
        };
        *current = Some(transaction_id);
        *self.transaction_catalog.lock() = Some(TransactionCatalog::begin(&self.catalog));
        *self.read_only_tx.lock() = read_only || self.db_read_only;
        #[cfg(feature = "lpg")]
        {
            let snapshot = if self.transaction_manager.isolation_level(transaction_id)
                == Some(crate::transaction::IsolationLevel::ReadCommitted)
            {
                None
            } else {
                Some(self.projections.read().clone())
            };
            *self.projection_registry_snapshot.lock() = snapshot;
            self.projection_registry_read
                .store(false, Ordering::Release);
        }

        // RDF's indexes hold only the latest committed state. Pin the
        // transaction's MVCC start epoch in the RDF store so every planner and
        // direct read carrying this tid resolves history at the same cut, then
        // overlays only this transaction's pending operations. Read Committed
        // deliberately stays unregistered and reads the live indexes.
        #[cfg(feature = "triple-store")]
        if self.transaction_manager.isolation_level(transaction_id)
            != Some(crate::transaction::IsolationLevel::ReadCommitted)
            && let Some(epoch) = self.transaction_manager.start_epoch(transaction_id)
        {
            self.rdf_store
                .register_transaction_snapshot(transaction_id, epoch);
        }

        // Clear the preceding transaction's lifecycle view before resolving the
        // initial coordinate. In particular, a stale Missing expectation must
        // never hide a graph from the next transaction while its exact/missing
        // expectation and Serializable bridges are being established.
        let key = self.active_graph_storage_key();
        self.touched_graphs.lock().clear();
        self.pending_created_graphs.lock().clear();
        self.pending_dropped_graphs.lock().clear();
        self.cancelled_created_graphs.lock().clear();
        self.touched_named_graphs.lock().clear();
        self.superseded_graph_touches.lock().clear();
        self.missing_named_graphs.lock().clear();
        self.pending_graph_type_bindings.lock().clear();
        self.pending_index_ddl.lock().clear();
        self.pending_projection_ddl.lock().clear();

        // Resolve the root registry directly while the publication read guard
        // is held. `track_lpg_graph_coordinate` records either Exact(Arc) or,
        // for SI/Serializable, Missing, and installs the Serializable tracker
        // bridges on that same exact store. Read Committed deliberately keeps
        // absence refreshable.
        let named = self.session_graph_path(&key);
        #[cfg(feature = "metrics")]
        {
            crate::metrics::record_metric!(self.metrics, tx_active, inc);
            #[cfg(not(target_arch = "wasm32"))]
            {
                *self.tx_start_time.lock() = Some(Instant::now());
            }
        }

        if let Err(error) = self.track_lpg_graph_coordinate(transaction_id, key, named) {
            // Release BEGIN's guards before the ordinary all-model abort path.
            // Failed admission must not strand an active TM transaction or SSI
            // bridges installed before a later prefix could be qualified.
            drop(current);
            drop(_publication);
            drop(database_open);
            return match self.rollback_inner_authorized() {
                Ok(()) => Err(error),
                Err(rollback_error) => {
                    self.poison_durability();
                    Err(error.with_context(format!(
                        "transaction admission rollback also failed: {rollback_error}; reopen and recover before continuing"
                    )))
                }
            };
        }

        let active = self.active_read_store();
        self.transaction_start_node_count
            .store(active.node_count(), Ordering::Relaxed);
        self.transaction_start_edge_count
            .store(active.edge_count(), Ordering::Relaxed);

        Ok(())
    }

    /// Commits the current transaction.
    ///
    /// Makes all changes since [`begin_transaction`](Self::begin_transaction) permanent.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active, commit validation or
    /// constraint publication fails, or the durable WAL boundary cannot be
    /// written and flushed according to the configured durability mode.
    #[cfg(feature = "lpg")]
    pub fn commit(&mut self) -> Result<grafeo_common::types::EpochId> {
        self.commit_inner()
    }

    /// Commits the active RDF-only transaction and returns its durable epoch.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active, lifecycle/Serializable
    /// validation fails, or the durable WAL boundary cannot be written and
    /// flushed according to the configured durability mode.
    #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
    pub fn commit(&mut self) -> Result<grafeo_common::types::EpochId> {
        self.commit_rdf_auto()
    }

    /// Core commit logic, usable from both `&mut self` and `&self` paths.
    #[cfg(feature = "lpg")]
    fn commit_inner(&self) -> Result<grafeo_common::types::EpochId> {
        let mut capture = EngineCommitCapture::default();
        let mut core_workspace = None;
        let mut transaction_workspace = crate::transaction::TransactionFinalizationWorkspace::new();
        let mut catalog_workspace = CatalogWorkspace::new();
        let _operation = self.session_operation_guard();
        let _catalog_cuts = self.pin_catalog_cuts();
        self.transaction_manager.with_write_authority(|| {
            self.commit_inner_authorized(
                &mut catalog_workspace,
                &mut transaction_workspace,
                &mut capture,
                &mut core_workspace,
            )
        })
    }

    #[cfg(feature = "lpg")]
    fn commit_inner_authorized<'capture>(
        &self,
        catalog_workspace: &mut CatalogWorkspace,
        transaction_workspace: &mut crate::transaction::TransactionFinalizationWorkspace,
        capture: &'capture mut EngineCommitCapture,
        core_workspace: &mut Option<LpgCommitWorkspace<'capture>>,
    ) -> Result<grafeo_common::types::EpochId> {
        self.check_not_poisoned()?;
        self.check_not_in_mixed_snapshot()?;
        let _span = grafeo_debug_span!("grafeo::tx::commit");

        #[cfg(feature = "testing-statement-injection")]
        if let Err(e) = grafeo_common::testing::statement_failure::maybe_fail_commit() {
            // Commit fails before any state is finalized. Treat it like any
            // other pre-prepare commit failure and auto-rollback so the
            // session returns to a clean, consistent state (matches real
            // DB semantics: commit failure implies the transaction is
            // aborted, not left in-flight).
            let injected = grafeo_common::utils::error::Error::Internal(format!(
                "injected commit failure: {e}"
            ));
            return match self.rollback_entire_transaction_authorized() {
                Ok(()) => Err(injected),
                Err(rollback_error) => {
                    self.poison_durability();
                    Err(injected.with_context(format!(
                        "full transaction rollback also failed: {rollback_error}; reopen and recover before continuing"
                    )))
                }
            };
        }

        self.check_no_active_streams("commit")?;
        // Nested transaction: release the auto-savepoint (changes are preserved).
        {
            let mut depth = self.transaction_nesting_depth.lock();
            if *depth > 0 {
                let sp_name = format!("{INTERNAL_SAVEPOINT_PREFIX}nested:{depth}");
                self.release_savepoint_authorized(&sp_name)?;
                *depth -= 1;
                return Ok(self.transaction_manager.current_epoch());
            }
        }

        // Resolve exact cleanup targets while a resolution failure can still
        // leave the transaction active. No fallible lookup follows the marker.
        let mut touched_stores: Vec<(Arc<LpgStore>, Arc<dyn GraphStoreMut>)> = self
            .touched_graphs
            .lock()
            .iter()
            .map(|graph| {
                Ok((
                    self.resolve_store(graph)?,
                    self.resolve_mutation_store(graph)?,
                ))
            })
            .collect::<Result<_>>()?;
        // Current paths stay first for the zipped publication loops below.
        // Replaced targets still own pending state, even if their ancestors
        // are the only incarnations retained by the lifecycle maps.
        touched_stores.extend(
            self.superseded_graph_touches
                .lock()
                .iter()
                .map(|(_, store)| {
                    (
                        Arc::clone(store),
                        Arc::clone(store) as Arc<dyn GraphStoreMut>,
                    )
                }),
        );
        let transaction_id = self.current_transaction.lock().take().ok_or_else(|| {
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;

        // Validate the transaction first (conflict detection) before committing data.
        // If this fails, we rollback the data changes instead of making them permanent.
        //
        // Take ownership of the touched graphs in one lock acquisition. Since
        // current_transaction was .take()'d above, no concurrent thread can call
        // track_graph_touch() for this transaction (it checks current_transaction
        // first), so this is safe.
        let touched = std::mem::take(&mut *self.touched_graphs.lock());

        // Increment 2e (Part E): complete the write-set from the store chokepoints
        // (complete by construction) before validation. Non-draining peeks; the
        // existing take_*/finalize_* below still consume them.
        //
        // Part G Task 4: under Property granularity the overlay entries are tagged
        // with their per-property hash rather than collapsing to entity-level None,
        // so disjoint-property concurrent writes do not form false rw-antidependencies.
        {
            let granularity = *self.conflict_granularity.lock();
            // Detached CREATE targets are invisible until this commit's
            // lifecycle publication. Their graph-local entity IDs cannot
            // conflict with any concurrent transaction; entering them into
            // the shared, graph-unqualified keyspace creates false W-W/SSI
            // conflicts with equal IDs in unrelated visible graphs.
            let private_graphs: std::collections::HashSet<&GraphPath> = {
                let pending = self.pending_created_graphs.lock();
                touched
                    .iter()
                    .filter(|graph| {
                        pending
                            .keys()
                            .any(|prefix| graph.components().starts_with(prefix.components()))
                    })
                    .collect()
            };
            let conflict_visible = |graph: &GraphPath| !private_graphs.contains(graph);
            // Structural writes (creates, deletes) are always entity-level (None).
            let mut structural_ws: Vec<EntityId> = Vec::new();
            for (_, (_, store)) in touched
                .iter()
                .zip(&touched_stores)
                .filter(|(graph, _)| conflict_visible(graph))
            {
                structural_ws.extend(
                    store
                        .pending_node_creates(transaction_id)
                        .into_iter()
                        .map(EntityId::Node),
                );
                structural_ws.extend(
                    store
                        .pending_edge_creates(transaction_id)
                        .into_iter()
                        .map(EntityId::Edge),
                );
                structural_ws.extend(
                    store
                        .pending_node_deletes_peek(transaction_id)
                        .into_iter()
                        .map(EntityId::Node),
                );
                structural_ws.extend(
                    store
                        .pending_edge_deletes_peek(transaction_id)
                        .into_iter()
                        .map(EntityId::Edge),
                );
            }
            self.transaction_manager
                .extend_write_set(transaction_id, structural_ws);

            // Overlay touches: property-tagged under Property granularity, entity-level
            // (None) under Entity granularity.
            if granularity == ConflictGranularity::Property {
                let mut tagged: Vec<(EntityId, crate::transaction::PropTag)> = Vec::new();
                for (_, (_, store)) in touched
                    .iter()
                    .zip(&touched_stores)
                    .filter(|(graph, _)| conflict_visible(graph))
                {
                    let (node_props, edge_props) = store.overlay_touched_properties(transaction_id);
                    for (node_id, opt_key) in node_props {
                        let tag = opt_key.as_deref().map(prop_tag);
                        tagged.push((EntityId::Node(node_id), tag));
                    }
                    for (edge_id, opt_key) in edge_props {
                        let tag = opt_key.as_deref().map(prop_tag);
                        tagged.push((EntityId::Edge(edge_id), tag));
                    }
                }
                self.transaction_manager
                    .extend_write_set_tagged(transaction_id, tagged);
            } else {
                let mut ws: Vec<EntityId> = Vec::new();
                for (_, (_, store)) in touched
                    .iter()
                    .zip(&touched_stores)
                    .filter(|(graph, _)| conflict_visible(graph))
                {
                    let (on, oe) = store.overlay_touched_entities(transaction_id);
                    ws.extend(on.into_iter().map(EntityId::Node));
                    ws.extend(oe.into_iter().map(EntityId::Edge));
                }
                self.transaction_manager
                    .extend_write_set(transaction_id, ws);
            }
        }

        // Epoch assignment, durable Committed, both-model apply, and
        // visibility share one write lock so concurrent commits cannot
        // publish out of epoch order.
        #[cfg(feature = "triple-store")]
        let _rdf_gate = self.rdf_store.lock_commit();
        let _publication = self.transaction_manager.publication().write();

        #[cfg(feature = "cdc")]
        let mut prepared_cdc = None;
        #[cfg(all(feature = "cdc", feature = "wal"))]
        let mut cdc_models = None;
        let mut prepared_virtual_projection_ddl = None;
        let mut prepared_graph_lifecycle = Vec::new();
        #[cfg(all(feature = "lpg", feature = "triple-store"))]
        let mut prepared_projection_receipt = None;
        let publication_epoch = self.transaction_manager.current_epoch();
        let prepared = self
            .validate_pending_graph_lifecycle()
            .map(|changes| prepared_graph_lifecycle = changes)
            .and_then(|()| self.validate_transaction_catalog(transaction_id))
            .and_then(|()| self.validate_serializable_projection_read(transaction_id));
        #[cfg(feature = "triple-store")]
        let prepared =
            prepared.and_then(|()| self.validate_rdf_transaction_lifecycle(transaction_id));
        let prepared =
            prepared.and_then(|()| self.revalidate_touched_constraints(transaction_id, &touched));
        #[cfg(feature = "triple-store")]
        let prepared =
            prepared.and_then(|()| self.revalidate_rdf_projection_target(transaction_id));
        let (commit_epoch, outcome): (_, Result<_>) = match prepared
            .and_then(|()| {
                self.prepare_pending_projection_ddl().map(|projections| {
                    prepared_virtual_projection_ddl = projections;
                })
            })
            .and_then(|()| {
                self.transaction_manager
                    .prepare_durable_commit(transaction_id)
            })
            .and_then(|epoch| {
                // Resolve graph survival before final core/catalog writers;
                // destination reservation below performs no graph reads.
                #[cfg(feature = "cdc")]
                let cdc_batch = self.cdc_pending_events.as_ref().map(|pending| {
                    pending.prepare_committed_lpg(epoch, |path, incarnation| {
                        self.lpg_incarnation_survives_pending_lifecycle(path, incarnation)
                    })
                }).transpose()?;
                #[cfg(all(feature = "cdc", feature = "wal"))]
                let cdc_records = if self.wal.is_some() {
                    cdc_batch.as_ref().map(|batch| batch.wal_records(transaction_id, epoch)).transpose()?
                } else { None };
                capture.prepare(self, &touched, transaction_id, publication_epoch, epoch)?;
                let (workspace, logical) = capture.workspace(transaction_id, publication_epoch, epoch);
                let workspace = core_workspace.insert(workspace);
                #[cfg(feature = "wal")]
                if self.wal.is_some() {
                    workspace.capture_label_images()?;
                    #[cfg(feature = "text-index")]
                    workspace.capture_text_postimages()?;
                    #[cfg(feature = "vector-index")]
                    workspace.capture_vector_postimages()?;
                }
                // Reserve TM map capacity before catalog's final writer. The
                // released companion keeps no inner TM locks while the other
                // preparation steps run; final binding is the last abortable
                // step before the durable marker.
                let transaction = self.transaction_manager.prepare_finalization(
                    transaction_id,
                    epoch,
                    transaction_workspace,
                )?;
                #[cfg(all(feature = "lpg", feature = "triple-store"))]
                if let Some((projection_id, receipt_target)) = self
                    .rdf_projection_target
                    .lock()
                    .as_ref()
                    .and_then(|target| {
                        target
                            .receipt
                            .clone()
                            .map(|receipt| (target.projection_id, receipt))
                    })
                {
                    use grafeo_common::types::ProjectionReconciliationState;
                    use grafeo_common::utils::error::{Error, TransactionError};
                    use grafeo_core::graph::rdf::RdfLpgProjectionReceipt;

                    let receipt = RdfLpgProjectionReceipt::new(
                        receipt_target.store_id,
                        receipt_target.mapping_digest,
                        projection_id,
                        receipt_target.source_graph,
                        receipt_target.source_epoch,
                        epoch,
                        receipt_target.generation,
                        receipt_target.row_count,
                        ProjectionReconciliationState::Reconciled,
                    )
                    .map_err(|message| {
                        Error::Transaction(TransactionError::InvalidState(message))
                    })?;
                    let prepared = receipt_target
                        .registry
                        .prepare_receipt(receipt_target.store_id, receipt)
                        .map_err(|message| {
                            Error::Transaction(TransactionError::InvalidState(message))
                        })?;

                    #[cfg(feature = "testing-crash-injection")]
                    grafeo_common::testing::crash::maybe_crash("projection:before_receipt");

                    #[cfg(feature = "wal")]
                    if let Some(ref wal) = self.wal {
                        use grafeo_storage::wal::WalRecord;
                        #[cfg(feature = "testing-crash-injection")]
                        grafeo_common::testing::wal_failure::maybe_fail_projection_receipt_log()
                            .map_err(|error| Error::Internal(error.to_string()))?;
                        wal.log(&WalRecord::RdfLpgProjectionPublishedV3 {
                            transaction_id,
                            receipt: prepared.receipt().encode(),
                        })?;
                    }

                    #[cfg(feature = "testing-crash-injection")]
                    grafeo_common::testing::crash::maybe_crash(
                        "projection:after_receipt_before_commit",
                    );

                    prepared_projection_receipt = Some(prepared);
                }
                with_prepared_lpg_commit(
                    workspace,
                    self.transaction_manager.write_authority(),
                    |released| {
                        #[cfg(feature = "wal")]
                        logical.write_wal_publication(self, &released, transaction_id)?;
                        // Core has drained Vector readers before this catalog
                        // acquisition. No physical/source read is permitted
                        // here; every subsequent binding is try-only.
                        let publication = logical.prepare(self, catalog_workspace)?;
                        #[cfg(feature = "wal")]
                        if self.catalog_changed()
                            && let Some(wal) = &self.wal
                            && let PreparedLogicalCatalog::Changed(ready) = &publication
                        {
                            wal.log(&grafeo_storage::wal::WalRecord::CatalogPostimage {
                                transaction_id,
                                payload: ready.encode_transaction_metadata()
                                    .map_err(grafeo_common::utils::error::Error::Serialization)?,
                            })?;
                        }
                        #[cfg(feature = "wal")]
                        logical.write_wal_owners(self, &publication, &released, transaction_id)?;
                        let prepared_core = match released.rebind() {
                            Ok(prepared) => prepared,
                            Err(error) => {
                                drop(publication);
                                return Err(error.into_error());
                            }
                        };
                        let prepared_transaction = match transaction.rebind() {
                            Ok(prepared) => prepared,
                            Err(error) => {
                                drop(prepared_core);
                                drop(publication);
                                return Err(error.into_error());
                            }
                        };

                        #[cfg(feature = "cdc")]
                        {
                            prepared_cdc = cdc_batch
                                .map(crate::cdc::PreparedCdcBatch::prepare_publication)
                                .transpose()?;
                        }

                        #[cfg(all(feature = "cdc", feature = "wal"))]
                        if let (Some(wal), Some(records)) = (&self.wal, cdc_records) {
                            let mut models = 0;
                            for record in records {
                                if let grafeo_storage::wal::WalRecord::CdcBatch { model, .. } = &record { models |= model; }
                                wal.log(&record)?;
                            }
                            cdc_models = Some(models);
                        }
                        #[cfg(feature = "testing-statement-injection")]
                        self.query_cancellation_test_boundary(QueryCancellationTestPhase::BeforeCommitFence);
                        if let Some(control) = self.active_execution_control.lock().as_mut()
                            && let Err(error) = control.try_begin_commit() {
                                drop(prepared_transaction);
                                drop(prepared_core);
                                drop(publication);
                                return Err(Self::map_query_lifecycle_error(error));
                        }

                        // A rejected final binding is abortable. A failed WAL
                        // acknowledgement is not: return it as an inner
                        // outcome so the premarker rollback branch cannot
                        // manufacture a contradictory Abort record.
                        #[cfg(feature = "wal")]
                        if let Some(ref wal) = self.wal {
                            use grafeo_storage::wal::WalRecord;
                            #[cfg(feature = "testing-crash-injection")]
                            grafeo_common::testing::crash::maybe_crash("commit:before_marker");
                            let marker = WalRecord::Committed { transaction_id, epoch };
                            #[cfg(feature = "cdc")]
                            let marker = cdc_models.map_or(marker, |models| WalRecord::CommittedWithCdc { transaction_id, epoch, models });
                            if let Err(error) = wal.log(&marker) {
                                drop(prepared_transaction);
                                drop(prepared_core);
                                drop(publication);
                                self.poison_durability();
                                return Ok(Err(grafeo_common::utils::error::Error::Transaction(
                                    grafeo_common::utils::error::TransactionError::DurabilityFailure(format!(
                                        "WAL commit acknowledgement failed and the transaction outcome is unknown: {error}; reopen and recover before continuing"
                                    )),
                                )));
                            }
                            #[cfg(feature = "testing-crash-injection")]
                            grafeo_common::testing::crash::maybe_crash(
                                "commit:after_marker_before_publication",
                            );
                            #[cfg(feature = "testing-statement-injection")]
                            self.query_cancellation_test_boundary(QueryCancellationTestPhase::AfterDurableMarker);
                        }

                        #[cfg(all(feature = "triple-store", feature = "testing-crash-injection"))]
                        if prepared_projection_receipt.is_some() {
                            grafeo_common::testing::crash::maybe_crash(
                                "projection:after_commit_before_publication",
                            );
                        }
                        let installed_core = prepared_core.install();
                        let (owners, installed_catalog) = match publication {
                            PreparedLogicalCatalog::NoCatalogChange => (None, None),
                            PreparedLogicalCatalog::OwnersPinned(owners) => (Some(owners), None),
                            PreparedLogicalCatalog::Changed(ready) => (None, Some(ready.install())),
                        };
                        let installed_transaction = prepared_transaction.install();
                        // Keep every final writer through every companion's
                        // installation; only outer owners retire payloads.
                        drop(installed_catalog);
                        drop(owners);
                        drop(installed_core);
                        Ok(Ok(installed_transaction.release()))
                    },
                ).map(|outcome| (epoch, outcome))
            }) {
            Ok(prepared) => prepared,
            Err(e) => {
                #[cfg(feature = "cdc")]
                drop(prepared_cdc.take());
                // Conflict detected: discard the transaction's uncommitted
                // (PENDING) versions and replay its property undo log, then mark
                // it aborted. rollback_transaction_properties alone leaks the
                // PENDING node/edge versions; skipping abort leaves the tx Active
                // forever, pinning min_active_epoch and stalling MVCC GC.
                for (store, mutation_store) in &touched_stores {
                    let (pending_nodes, pending_edges) = store.take_pending_creates(transaction_id);
                    store.discard_entities_by_id(transaction_id, &pending_nodes, &pending_edges);
                    store.rollback_transaction_properties(transaction_id);
                    mutation_store.drop_tx_overlay(transaction_id);
                    // Clear deferred pending deletes (PENDING path): unmark their
                    // version chains so the nodes remain visible after conflict rollback.
                    let pending_deletes = mutation_store.take_pending_deletes(transaction_id);
                    store.rollback_pending_deletes(transaction_id, &pending_deletes);
                    // Same for deferred pending EDGE deletes (MVCC increment 2b):
                    // unmark so the edges remain visible after conflict rollback.
                    let pending_edge_deletes =
                        mutation_store.take_pending_edge_deletes(transaction_id);
                    store.rollback_pending_edge_deletes(transaction_id, &pending_edge_deletes);
                    // Unregister the Serializable read/write trackers (no-op for SI/RC).
                    mutation_store.unregister_read_tracker(transaction_id);
                    mutation_store.unregister_write_tracker(transaction_id);
                }
                self.rollback_pending_graph_lifecycle(transaction_id);
                self.rollback_pending_index_ddl();
                self.rollback_pending_projection_ddl();
                self.transaction_catalog.lock().take();
                let _ = self.transaction_manager.abort(transaction_id);
                #[cfg(feature = "triple-store")]
                self.rollback_rdf_transaction(transaction_id);
                // Discard buffered CDC events on conflict rollback
                #[cfg(feature = "cdc")]
                if let Some(ref pending) = self.cdc_pending_events {
                    pending.clear();
                }
                *self.read_only_tx.lock() = self.db_read_only;
                self.savepoints.lock().clear();
                self.touched_graphs.lock().clear();
                self.touched_named_graphs.lock().clear();
                self.superseded_graph_touches.lock().clear();
                #[cfg(feature = "triple-store")]
                self.rdf_projection_target.lock().take();

                // The transaction's mutation records may already be present in
                // the WAL.  Record the abort before allowing any later
                // transaction to proceed so recovery cannot ever mistake this
                // rejected transaction for committed work.  An abort-marker
                // failure is itself outcome-ambiguous, so fail closed and make
                // recovery mandatory instead of returning the validation error
                // as though the durable state were known.
                #[cfg(feature = "wal")]
                if let Some(ref wal) = self.wal {
                    use grafeo_storage::wal::WalRecord;
                    if let Err(abort_error) =
                        wal.log(&WalRecord::TransactionAbort { transaction_id })
                    {
                        self.poison_durability();
                        return Err(grafeo_common::utils::error::Error::Transaction(
                            grafeo_common::utils::error::TransactionError::DurabilityFailure(
                                format!(
                                    "transaction preparation failed ({e}); WAL abort acknowledgement also failed ({abort_error}); reopen and recover before continuing"
                                ),
                            ),
                        ));
                    }
                }
                #[cfg(feature = "metrics")]
                {
                    crate::metrics::record_metric!(self.metrics, tx_active, dec);
                    crate::metrics::record_metric!(self.metrics, tx_conflicts, inc);
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(start) = self.tx_start_time.lock().take() {
                        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                        crate::metrics::record_metric!(
                            self.metrics,
                            tx_duration,
                            observe duration_ms
                        );
                    }
                }
                return Err(e);
            }
        };

        let transaction_cleanup = outcome?;
        self.transaction_catalog.lock().take();

        // The durable outcome is now irrevocable. Remove transaction-local
        // state from every dropped/cancelled incarnation. The connected core
        // segment has released topology, so the existing lifecycle tail can
        // use its ordinary APIs beneath the enclosing publication gate.
        self.discard_non_surviving_lpg_incarnations(transaction_id);
        for (graph, (store, mutation_store)) in touched.iter().zip(&touched_stores) {
            if !self.lpg_incarnation_survives_pending_lifecycle(graph, store) {
                // A dropped ancestor owns only its own queues. Descendants
                // keep independent pending state and SSI bridges, so retire
                // each exact touched target through its captured writer.
                self.discard_lpg_graph_transaction_via(store, mutation_store, transaction_id);
                continue;
            }
            // Tracker retirement remains outside all core final fences.
            mutation_store.unregister_read_tracker(transaction_id);
            mutation_store.unregister_write_tracker(transaction_id);
        }

        // The durable marker owns both the LPG post-image and this receipt.
        // Readers cannot observe the ordering inside this publication write
        // lock, and the prepared token has no fallible post-marker work.
        #[cfg(all(feature = "lpg", feature = "triple-store"))]
        if let Some(prepared) = prepared_projection_receipt.take() {
            prepared.publish();
        }

        #[cfg(feature = "lpg")]
        if let Err(error) =
            self.commit_pending_graph_lifecycle(commit_epoch, prepared_graph_lifecycle)
        {
            // The commit marker and transaction epoch are already durable at
            // this point. A lifecycle publication failure is therefore an
            // invariant violation, not an ordinary statement error: block all
            // further access so reopen/WAL recovery can reconstruct the single
            // committed outcome instead of exposing a partially published cut.
            self.poison_durability();
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::DurabilityFailure(format!(
                    "durable graph lifecycle commit could not be published: {error}; reopen and recover before continuing"
                )),
            ));
        }
        self.publish_prepared_projection_ddl(prepared_virtual_projection_ddl);

        #[cfg(feature = "testing-crash-injection")]
        grafeo_common::testing::crash::maybe_crash("commit:after_lpg_before_rdf");

        #[cfg(feature = "triple-store")]
        if let Err(error) = self.commit_rdf_transaction(transaction_id, commit_epoch) {
            self.poison_durability();
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::DurabilityFailure(format!(
                    "durable RDF transaction could not be published: {error}; reopen and recover before continuing"
                )),
            ));
        }

        // The root store is the clock authority for the complete LPG world,
        // including named-graph and catalog-only commits. Keep it at the
        // publication frontier even when the transaction touched only a child;
        // authenticated container metadata compares this clock with the shared
        // transaction epoch. The operation is one atomic max on the hot path.
        let current_epoch = self.transaction_manager.current_epoch();
        self.store.sync_epoch(current_epoch);

        // Child-store convenience lookups (edge_type, get_edge, get_node) also
        // need the latest epoch when that exact graph was touched.
        for (graph, (store, _)) in touched.iter().zip(&touched_stores) {
            if !graph.components().is_empty() {
                store.sync_epoch(current_epoch);
            }
        }

        // Every feed allocation completed before the durable marker. Native
        // publication succeeded, and the reserved log writer is still held.
        #[cfg(feature = "cdc")]
        if let Some(prepared) = prepared_cdc.take() {
            prepared.publish();
        }

        // Reset read-only flag and clear savepoints.
        // touched_graphs was already emptied by mem::take above.
        *self.read_only_tx.lock() = self.db_read_only;
        self.savepoints.lock().clear();

        // All current data/catalog publication calls have released their
        // inner fences. Retain SSI readers until this point, still beneath
        // the database publication authority shared with Session begin.
        transaction_cleanup.finish();

        // Auto-GC: periodically prune old MVCC versions
        if self.gc_interval > 0 {
            let count = self.commit_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if count.is_multiple_of(self.gc_interval) {
                #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
                let gc_start = std::time::Instant::now();

                let min_epoch = self.transaction_manager.min_active_epoch();
                for (store, _) in &touched_stores {
                    store.gc_versions(min_epoch);
                }
                self.transaction_manager.gc();

                #[cfg(feature = "metrics")]
                {
                    crate::metrics::record_metric!(self.metrics, gc_runs, inc);
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let gc_duration_ms = gc_start.elapsed().as_secs_f64() * 1000.0;
                        crate::metrics::record_metric!(
                            self.metrics,
                            gc_duration,
                            observe gc_duration_ms
                        );
                    }
                }
            }
        }

        #[cfg(feature = "metrics")]
        {
            crate::metrics::record_metric!(self.metrics, tx_active, dec);
            crate::metrics::record_metric!(self.metrics, tx_committed, inc);
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(start) = self.tx_start_time.lock().take() {
                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                crate::metrics::record_metric!(self.metrics, tx_duration, observe duration_ms);
            }
        }

        // Keep pinned named-graph incarnations through sync/GC: a committed
        // DROP has already removed its name from the shared map, but these
        // final maintenance steps must still address the exact old store.
        self.touched_named_graphs.lock().clear();
        self.superseded_graph_touches.lock().clear();
        #[cfg(feature = "triple-store")]
        self.rdf_projection_target.lock().take();

        Ok(commit_epoch)
    }

    /// Aborts the current transaction.
    ///
    /// Discards all changes since [`begin_transaction`](Self::begin_transaction).
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active or the durable abort
    /// boundary cannot be recorded.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let mut session = db.session();
    ///
    /// session.begin_transaction()?;
    /// session.execute("INSERT (:Person {name: 'Alix'})")?;
    /// session.rollback()?; // Insert is discarded
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "lpg")]
    pub fn rollback(&mut self) -> Result<()> {
        self.rollback_inner()
    }

    /// Rolls back the active RDF-only transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active or the durable abort
    /// boundary cannot be recorded.
    #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
    pub fn rollback(&mut self) -> Result<()> {
        self.rollback_rdf_auto()
    }

    /// Core rollback logic, usable from both `&mut self` and `&self` paths.
    #[cfg(feature = "lpg")]
    fn rollback_inner(&self) -> Result<()> {
        let _operation = self.session_operation_guard();
        let _catalog_cuts = self.pin_catalog_cuts();
        self.transaction_manager
            .with_write_authority(|| self.rollback_inner_authorized())
    }

    #[cfg(feature = "lpg")]
    fn rollback_inner_authorized(&self) -> Result<()> {
        let _span = grafeo_debug_span!("grafeo::tx::rollback");
        self.check_no_active_streams("rollback")?;
        // Nested transaction: rollback to the auto-savepoint.
        {
            let mut depth = self.transaction_nesting_depth.lock();
            if *depth > 0 {
                let sp_name = format!("{INTERNAL_SAVEPOINT_PREFIX}nested:{depth}");
                self.rollback_to_savepoint_authorized(&sp_name)?;
                self.release_savepoint_authorized(&sp_name)?;
                *depth -= 1;
                return Ok(());
            }
        }

        let mut touched_stores: Vec<(Arc<LpgStore>, Arc<dyn GraphStoreMut>)> = self
            .touched_graphs
            .lock()
            .iter()
            .map(|graph| {
                Ok((
                    self.resolve_store(graph)?,
                    self.resolve_mutation_store(graph)?,
                ))
            })
            .collect::<Result<_>>()?;
        touched_stores.extend(
            self.superseded_graph_touches
                .lock()
                .iter()
                .map(|(_, store)| {
                    (
                        Arc::clone(store),
                        Arc::clone(store) as Arc<dyn GraphStoreMut>,
                    )
                }),
        );
        let transaction_id = self.current_transaction.lock().take().ok_or_else(|| {
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;

        // Reset read-only flag
        *self.read_only_tx.lock() = self.db_read_only;

        // Discard uncommitted versions in ALL touched LPG stores (cross-graph
        // atomicity). Write-set-scoped via the store's pending-create index:
        // discard only the entities this transaction created, then replay the
        // property/label undo log (which covers sets, labels, and deletes).
        for (store, mutation_store) in &touched_stores {
            let (pending_nodes, pending_edges) = store.take_pending_creates(transaction_id);
            store.discard_entities_by_id(transaction_id, &pending_nodes, &pending_edges);
            store.rollback_transaction_properties(transaction_id);
            mutation_store.drop_tx_overlay(transaction_id);
            // Clear deferred pending deletes (PENDING path): unmark their version
            // chains so the nodes remain visible after rollback.
            let pending_deletes = mutation_store.take_pending_deletes(transaction_id);
            store.rollback_pending_deletes(transaction_id, &pending_deletes);
            // Same for deferred pending EDGE deletes (MVCC increment 2b): unmark
            // their version chains so the edges remain visible after rollback.
            let pending_edge_deletes = mutation_store.take_pending_edge_deletes(transaction_id);
            store.rollback_pending_edge_deletes(transaction_id, &pending_edge_deletes);
            // Unregister the Serializable read/write trackers on rollback.
            // No-op for SI/RC (trackers were never registered for those levels).
            mutation_store.unregister_read_tracker(transaction_id);
            mutation_store.unregister_write_tracker(transaction_id);
        }

        #[cfg(feature = "lpg")]
        self.rollback_pending_graph_lifecycle(transaction_id);
        #[cfg(feature = "lpg")]
        self.rollback_pending_index_ddl();
        #[cfg(feature = "lpg")]
        self.rollback_pending_projection_ddl();

        #[cfg(feature = "lpg")]
        self.transaction_catalog.lock().take();

        // Discard pending operations in the RDF store
        #[cfg(feature = "triple-store")]
        self.rollback_rdf_transaction(transaction_id);

        // Discard buffered CDC events on rollback
        #[cfg(feature = "cdc")]
        if let Some(ref pending) = self.cdc_pending_events {
            pending.clear();
        }

        // Clear savepoints and touched graphs
        self.savepoints.lock().clear();
        self.touched_graphs.lock().clear();
        self.touched_named_graphs.lock().clear();
        self.superseded_graph_touches.lock().clear();
        #[cfg(feature = "triple-store")]
        self.rdf_projection_target.lock().take();

        // Mark transaction as aborted in the manager
        let result = self.transaction_manager.abort(transaction_id);

        // Log transaction abort to WAL so recovery clears any data records
        // emitted during this transaction (e.g. via session-direct mutation
        // APIs). Without this marker, recovery's per-transaction buffer
        // would carry the rolled-back records into the next
        // `TransactionCommit` and resurrect them on reopen.
        #[cfg(feature = "wal")]
        if let Some(ref wal) = self.wal {
            use grafeo_storage::wal::WalRecord;
            if let Err(e) = wal.log(&WalRecord::TransactionAbort { transaction_id }) {
                self.poison_durability();
                return Err(grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::DurabilityFailure(format!(
                        "WAL abort log failed: {e}"
                    )),
                ));
            }
        }

        #[cfg(feature = "metrics")]
        if result.is_ok() {
            crate::metrics::record_metric!(self.metrics, tx_active, dec);
            crate::metrics::record_metric!(self.metrics, tx_rolled_back, inc);
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(start) = self.tx_start_time.lock().take() {
                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                crate::metrics::record_metric!(self.metrics, tx_duration, observe duration_ms);
            }
        }

        result
    }

    /// Creates a named savepoint within the current transaction.
    ///
    /// The savepoint captures LPG graph state and pending RDF operations so
    /// [`rollback_to_savepoint`](Self::rollback_to_savepoint) can restore the
    /// complete transaction-local state for every enabled graph model.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active.
    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    pub fn savepoint(&self, name: &str) -> Result<()> {
        Self::require_public_savepoint_name(name)?;
        let _operation = self.session_operation_guard();
        self.transaction_manager
            .with_write_authority(|| self.savepoint_authorized(name))
    }

    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    fn savepoint_authorized(&self, name: &str) -> Result<()> {
        let tx_id = self.current_transaction.lock().ok_or_else(|| {
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;

        let mut savepoints = self.savepoints.lock();
        if savepoints.iter().any(|savepoint| savepoint.name == name) {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(format!(
                    "Savepoint '{name}' already exists"
                )),
            ));
        }

        // Capture state for every graph touched so far and every exact
        // lifecycle incarnation that exists at the savepoint. A detached graph
        // may be created before SAVEPOINT and first receive writes afterwards;
        // name-only lifecycle capture cannot rewind those post-savepoint
        // PENDING entities.
        #[cfg(feature = "lpg")]
        let touched = self.touched_graphs.lock().clone();
        #[cfg(feature = "lpg")]
        let lpg_lifecycle = self.lpg_lifecycle_snapshot();
        #[cfg(feature = "lpg")]
        let mut graph_snapshots = Vec::new();
        #[cfg(feature = "lpg")]
        let mut remember =
            |was_touched: bool, store: Arc<LpgStore>, mutation_store: Arc<dyn GraphStoreMut>| {
                if graph_snapshots.iter().any(|known: &GraphSavepoint| {
                    Arc::ptr_eq(&known.store, &store)
                        && Arc::ptr_eq(&known.mutation_store, &mutation_store)
                }) {
                    return;
                }
                graph_snapshots.push(GraphSavepoint {
                    was_touched,
                    next_node_id: store.peek_next_node_id(),
                    next_edge_id: store.peek_next_edge_id(),
                    undo_log_position: store.property_undo_log_position(tx_id),
                    overlay_snapshot: mutation_store.tx_overlay_snapshot(tx_id),
                    structural_snapshot: mutation_store.tx_structural_snapshot(tx_id),
                    store,
                    mutation_store,
                });
            };
        #[cfg(feature = "lpg")]
        for graph_name in &touched {
            remember(
                true,
                self.resolve_store(graph_name)?,
                self.resolve_mutation_store(graph_name)?,
            );
        }
        #[cfg(feature = "lpg")]
        for (_, store) in &lpg_lifecycle.superseded_graph_touches {
            remember(
                true,
                Arc::clone(store),
                Arc::clone(store) as Arc<dyn GraphStoreMut>,
            );
        }
        #[cfg(feature = "lpg")]
        for pending in lpg_lifecycle.pending_created_graphs.values() {
            remember(
                false,
                Arc::clone(&pending.store),
                Arc::clone(&pending.store) as Arc<dyn GraphStoreMut>,
            );
        }
        #[cfg(feature = "lpg")]
        for store in lpg_lifecycle
            .pending_dropped_graphs
            .values()
            .chain(lpg_lifecycle.touched_named_graphs.values())
        {
            remember(
                false,
                Arc::clone(store),
                Arc::clone(store) as Arc<dyn GraphStoreMut>,
            );
        }
        #[cfg(feature = "lpg")]
        for stores in lpg_lifecycle.cancelled_created_graphs.values() {
            for store in stores {
                remember(
                    false,
                    Arc::clone(store),
                    Arc::clone(store) as Arc<dyn GraphStoreMut>,
                );
            }
        }
        #[cfg(feature = "lpg")]
        let index_ddl_snapshot = self.pending_index_ddl.lock().clone();
        #[cfg(feature = "lpg")]
        let projection_ddl_snapshot = self.pending_projection_ddl.lock().clone();
        #[cfg(feature = "triple-store")]
        let rdf_snapshot = self.rdf_store.transaction_savepoint(tx_id);

        // The marker must precede every mutation governed by this savepoint.
        // A logging failure poisons durability and leaves the runtime stack
        // unchanged, so a caller cannot continue with memory/WAL disagreement.
        #[cfg(feature = "wal")]
        self.log_wal_record(&grafeo_storage::wal::WalRecord::TransactionSavepoint {
            transaction_id: tx_id,
            name: name.to_string(),
        })?;

        savepoints.push(SavepointState {
            name: name.to_string(),
            #[cfg(feature = "lpg")]
            catalog_snapshot: self.transaction_catalog.lock().clone(),
            #[cfg(feature = "lpg")]
            graph_snapshots,
            #[cfg(feature = "lpg")]
            touched_graphs: touched,
            #[cfg(feature = "cdc")]
            cdc_event_position: self
                .cdc_pending_events
                .as_ref()
                .map_or(0, |pending| pending.position()),
            #[cfg(feature = "lpg")]
            lpg_lifecycle,
            #[cfg(feature = "lpg")]
            index_ddl_snapshot,
            #[cfg(feature = "lpg")]
            projection_ddl_snapshot,
            #[cfg(feature = "triple-store")]
            rdf_snapshot,
        });
        Ok(())
    }

    /// Rolls back to a named savepoint, undoing all writes made after it.
    ///
    /// Savepoints created after the target are removed; the target remains
    /// active and may be rolled back to again.
    /// Entities with IDs >= the savepoint snapshot are discarded.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active or the savepoint does not exist.
    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    pub fn rollback_to_savepoint(&self, name: &str) -> Result<()> {
        Self::require_public_savepoint_name(name)?;
        let _operation = self.session_operation_guard();
        #[cfg(feature = "lpg")]
        let _catalog_cuts = self.pin_catalog_cuts();
        self.transaction_manager
            .with_write_authority(|| self.rollback_to_savepoint_authorized(name))
    }

    /// Releases (removes) a named savepoint without rolling back.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active or the savepoint does not exist.
    pub fn release_savepoint(&self, name: &str) -> Result<()> {
        Self::require_public_savepoint_name(name)?;
        #[cfg(any(feature = "lpg", feature = "triple-store"))]
        let result = {
            let _operation = self.session_operation_guard();
            #[cfg(feature = "lpg")]
            let _catalog_cuts = self.pin_catalog_cuts();
            self.transaction_manager
                .with_write_authority(|| self.release_savepoint_authorized(name))
        };

        #[cfg(not(any(feature = "lpg", feature = "triple-store")))]
        let result = self.release_savepoint_authorized(name);

        result
    }

    fn require_public_savepoint_name(name: &str) -> Result<()> {
        if name.starts_with(INTERNAL_SAVEPOINT_PREFIX) {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "savepoint name uses Grafeo's reserved internal namespace".to_string(),
                ),
            ));
        }
        Ok(())
    }

    fn release_savepoint_authorized(&self, name: &str) -> Result<()> {
        let _tx_id = self.current_transaction.lock().ok_or_else(|| {
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;

        let mut savepoints = self.savepoints.lock();
        let pos = savepoints
            .iter()
            .rposition(|sp| sp.name == name)
            .ok_or_else(|| {
                grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::InvalidState(format!(
                        "Savepoint '{name}' not found"
                    )),
                )
            })?;
        savepoints.remove(pos);
        Ok(())
    }

    /// Returns whether a transaction is active.
    #[must_use]
    pub fn in_transaction(&self) -> bool {
        self.current_transaction.lock().is_some()
    }

    /// Returns the current transaction ID, if any.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub(crate) fn current_transaction_id(&self) -> Option<TransactionId> {
        *self.current_transaction.lock()
    }

    /// Installs the engine-only ownership capability and exact target predicate
    /// for one RDF→LPG rebuild transaction.
    ///
    /// Must be called on a fresh, default-graph session before `BEGIN`. The
    /// capability is intentionally unavailable outside this crate and is
    /// cleared when the transaction commits or rolls back.
    #[cfg(all(test, feature = "lpg", feature = "triple-store"))]
    pub(crate) fn authorize_rdf_projection_rebuild(
        &self,
        projection_id: u64,
        owner_marker: String,
        node_label: String,
        desired_iris: std::collections::BTreeSet<String>,
    ) -> Result<()> {
        self.install_rdf_projection_target(RdfProjectionTarget {
            projection_id,
            owner_marker,
            node_label,
            desired_iris: Arc::new(desired_iris),
            receipt: None,
        })
    }

    /// Installs the engine-only ownership capability and the complete receipt
    /// draft for one production RDF→LPG rebuild.
    ///
    /// Every coordinate known before LPG epoch reservation is checked here.
    /// Commit later supplies the target epoch, validates the final receipt, and
    /// appends it before the transaction's durable `Committed` marker.
    #[cfg(all(feature = "lpg", feature = "triple-store"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn authorize_rdf_projection_rebuild_v3(
        &self,
        projection_id: u64,
        owner_marker: String,
        node_label: String,
        desired_iris: std::collections::BTreeSet<String>,
        registry: Arc<grafeo_core::graph::rdf::RdfLpgProjectionRegistry>,
        store_id: grafeo_common::types::StoreId,
        mapping_digest: grafeo_common::types::Digest256,
        source_graph: grafeo_common::types::ProjectionSourceGraph,
        source_epoch: EpochId,
        generation: u64,
        row_count: u64,
    ) -> Result<()> {
        use grafeo_common::utils::error::{Error, TransactionError};

        if owner_marker != mapping_digest.to_string() {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "RDF→LPG rebuild owner marker does not contain the full mapping digest".into(),
            )));
        }
        if generation == 0 || source_epoch == EpochId::PENDING {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "RDF→LPG rebuild receipt has an invalid source epoch or generation".into(),
            )));
        }
        if store_id != self.rdf_store.store_id() {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "RDF→LPG rebuild receipt store identity disagrees with the session dataset".into(),
            )));
        }
        if source_epoch > self.rdf_store.commit_epoch() {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "RDF→LPG rebuild receipt source cut is newer than the session dataset".into(),
            )));
        }
        if row_count != u64::try_from(desired_iris.len()).unwrap_or(u64::MAX) {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "RDF→LPG rebuild receipt row count does not match its desired source set".into(),
            )));
        }
        let definition = registry.get(projection_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(format!(
                "RDF→LPG rebuild receipt references unknown projection {projection_id}"
            )))
        })?;
        if definition.mapping_digest() != mapping_digest
            || definition.source_graph() != source_graph.name()
            || definition.node_label() != node_label
            || definition.generation().checked_add(1) != Some(generation)
        {
            return Err(Error::Transaction(TransactionError::InvalidState(format!(
                "RDF→LPG rebuild receipt coordinates disagree with projection {projection_id}"
            ))));
        }

        self.install_rdf_projection_target(RdfProjectionTarget {
            projection_id,
            owner_marker,
            node_label,
            desired_iris: Arc::new(desired_iris),
            receipt: Some(RdfProjectionReceiptTarget {
                registry,
                store_id,
                mapping_digest,
                source_graph,
                source_epoch,
                generation,
                row_count,
            }),
        })
    }

    #[cfg(all(feature = "lpg", feature = "triple-store"))]
    fn install_rdf_projection_target(&self, target: RdfProjectionTarget) -> Result<()> {
        use grafeo_common::utils::error::{Error, TransactionError};

        if self.in_transaction() || !self.active_graph_storage_key().components().is_empty() {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "RDF→LPG rebuild authority requires a fresh default-graph session".into(),
            )));
        }
        let mut slot = self.rdf_projection_target.lock();
        if slot.is_some() {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "RDF→LPG rebuild authority is already installed on this session".into(),
            )));
        }
        *slot = Some(target);
        Ok(())
    }

    /// Returns the current transaction ID (public accessor for tests).
    ///
    /// Returns `None` if no transaction is active. Useful for integration
    /// tests that need to inspect the write-set via the transaction manager
    /// after committing.
    #[doc(hidden)]
    #[must_use]
    pub fn active_transaction_id(&self) -> Option<TransactionId> {
        *self.current_transaction.lock()
    }

    /// Returns a read-only view of transaction status and epochs.
    ///
    /// Mutation authority, transaction allocation, commit/abort, and the
    /// publication lock remain private to the Session/WAL protocol.
    #[doc(hidden)]
    #[must_use]
    pub fn transaction_manager_ref(&self) -> crate::transaction::TransactionManagerView<'_> {
        crate::transaction::TransactionManagerView::new(&self.transaction_manager)
    }

    /// Returns a reference to the transaction manager.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub(crate) fn transaction_manager(&self) -> &TransactionManager {
        &self.transaction_manager
    }

    /// Returns the store's current node count and the count at transaction start.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub(crate) fn node_count_delta(&self) -> (usize, usize) {
        (
            self.transaction_start_node_count.load(Ordering::Relaxed),
            self.active_read_store().node_count(),
        )
    }

    /// Returns the store's current edge count and the count at transaction start.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub(crate) fn edge_count_delta(&self) -> (usize, usize) {
        (
            self.transaction_start_edge_count.load(Ordering::Relaxed),
            self.active_read_store().edge_count(),
        )
    }

    /// Prepares the current transaction for a two-phase commit.
    ///
    /// Returns a [`PreparedCommit`](crate::transaction::PreparedCommit) that
    /// lets you inspect pending changes and attach metadata before finalizing.
    /// The mutable borrow prevents concurrent operations while the commit is
    /// pending.
    ///
    /// If the `PreparedCommit` is dropped without calling `commit()` or
    /// `abort()`, the transaction is automatically rolled back.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let mut session = db.session();
    ///
    /// session.begin_transaction()?;
    /// session.execute("INSERT (:Person {name: 'Alix'})")?;
    ///
    /// let mut prepared = session.prepare_commit()?;
    /// println!("Nodes written: {}", prepared.info().nodes_written);
    /// prepared.set_metadata("audit_user", "admin");
    /// prepared.commit()?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "lpg")]
    pub fn prepare_commit(&mut self) -> Result<crate::transaction::PreparedCommit<'_>> {
        crate::transaction::PreparedCommit::new(self)
    }

    /// Sets auto-commit mode.
    pub fn set_auto_commit(&mut self, auto_commit: bool) {
        self.auto_commit = auto_commit;
    }

    /// Returns whether auto-commit is enabled.
    #[must_use]
    pub fn auto_commit(&self) -> bool {
        self.auto_commit
    }

    /// Returns `true` when a mutating statement needs an implicit transaction.
    ///
    /// With auto-commit enabled that transaction is committed before the
    /// statement returns. With auto-commit disabled it remains open for an
    /// explicit commit or rollback.
    /// Either mode must frame the write: falling through to a SYSTEM mutation
    /// would leak uncommitted state and leave no recoverable commit marker.
    fn needs_implicit_transaction(&self, has_mutations: bool) -> bool {
        has_mutations && self.current_transaction.lock().is_none()
    }

    /// Adds the LPG historical-view fence around the shared transaction
    /// wrapper. RDF execution deliberately continues to call
    /// [`Self::with_auto_commit`] directly: `viewing_epoch` is an LPG MVCC
    /// contract, not an RDF valid/system-time selector.
    #[cfg(any(
        all(test, feature = "lpg"),
        any(
            feature = "gql",
            feature = "cypher",
            feature = "gremlin",
            feature = "graphql",
            feature = "sql-pgq"
        )
    ))]
    fn with_lpg_plan_auto_commit<F>(
        &self,
        has_mutations: bool,
        contains_procedure_call: bool,
        body: F,
    ) -> Result<QueryResult>
    where
        F: FnOnce() -> Result<QueryResult>,
    {
        if !has_mutations && !contains_procedure_call {
            return self.with_auto_commit(false, body);
        }
        let _operation = self.session_operation_guard();
        let _historical = self.historical_view_operation_gate.lock();
        if contains_procedure_call && self.effective_viewing_epoch().is_some() {
            return Err(grafeo_common::utils::error::Error::Query(
                grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Unsupported,
                    "procedure calls are not qualified for historical execution until their transitive effects and snapshot context are sealed",
                ),
            ));
        }
        if has_mutations {
            self.reject_lpg_historical_mutation()?;
        }
        self.with_auto_commit(has_mutations, || {
            #[cfg(feature = "lpg")]
            if has_mutations {
                self.ensure_current_mutation_touch()?;
            }
            body()
        })
    }

    /// Wraps `body` in an automatic begin/commit when [`needs_auto_commit`]
    /// returns `true`. On error the transaction is rolled back.
    #[cfg(feature = "lpg")]
    fn with_auto_commit<F>(&self, has_mutations: bool, body: F) -> Result<QueryResult>
    where
        F: FnOnce() -> Result<QueryResult>,
    {
        self.with_auto_commit_guarded(has_mutations, body)
    }

    /// Shared LPG auto-commit path with statement atomicity inside an existing
    /// caller-owned transaction.
    #[cfg(feature = "lpg")]
    fn with_auto_commit_guarded<F>(&self, has_mutations: bool, body: F) -> Result<QueryResult>
    where
        F: FnOnce() -> Result<QueryResult>,
    {
        let mut capture = EngineCommitCapture::default();
        let mut core_workspace = None;
        let mut transaction_workspace = crate::transaction::TransactionFinalizationWorkspace::new();
        let mut catalog_workspace = CatalogWorkspace::new();
        let _operation = self.session_operation_guard();
        let execution_statement = self.enter_execution_statement(has_mutations)?;
        let terminal_statement =
            has_mutations && (!self.needs_implicit_transaction(has_mutations) || !self.auto_commit);
        let body = || self.admit_query_result(body()?);
        let body =
            || self.finish_controlled_statement(&execution_statement, terminal_statement, body);
        let _catalog_cuts = has_mutations.then(|| self.pin_catalog_cuts());
        // Both-mode RDF statements share this transaction wrapper. They take
        // the gate to preserve H -> write-authority lock order, but only the
        // LPG-specific wrapper rejects a configured LPG historical view.
        let _historical = has_mutations.then(|| self.historical_view_operation_gate.lock());
        let run = || {
            if self.needs_implicit_transaction(has_mutations) {
                self.begin_transaction_inner(false, None)?;
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
                    Ok(Ok(result)) => {
                        if self.auto_commit {
                            self.commit_inner_authorized(
                                &mut catalog_workspace,
                                &mut transaction_workspace,
                                &mut capture,
                                &mut core_workspace,
                            )?;
                        }
                        Ok(result)
                    }
                    Ok(Err(primary)) => match self.rollback_inner() {
                        Ok(()) => Err(primary),
                        Err(rollback_error) => {
                            self.poison_durability();
                            Err(primary.with_context(format!(
                                "automatic statement rollback also failed: {rollback_error}"
                            )))
                        }
                    },
                    Err(payload) => {
                        let rollback =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                self.rollback_inner()
                            }));
                        if !matches!(rollback, Ok(Ok(()))) {
                            self.poison_durability();
                        }
                        std::panic::resume_unwind(payload)
                    }
                }
            } else {
                if has_mutations {
                    return self.with_statement_savepoint(body);
                }
                body()
            }
        };
        if has_mutations {
            #[cfg(feature = "cdc")]
            let _operation = self.cdc_mutation_operation_guard();
            self.check_not_in_mixed_snapshot()?;
            self.transaction_manager.with_write_authority(run)
        } else {
            run()
        }
    }

    /// Makes one mutating statement atomic inside a caller-owned transaction.
    ///
    /// Physical operators can discover a late error after earlier operators
    /// staged writes (for example, a stored-procedure return-contract error).
    /// An internal savepoint rewinds graph/RDF overlays, WAL intent, CDC,
    /// lifecycle, index, and projection state without discarding valid work
    /// from earlier statements. Serializable conflict evidence remains
    /// conservatively retained, which can only cause a safe retry.
    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    fn with_statement_savepoint<T>(&self, body: impl FnOnce() -> Result<T>) -> Result<T> {
        let sequence = NEXT_STATEMENT_SAVEPOINT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| {
                grafeo_common::utils::error::Error::Internal(
                    "statement savepoint identifier space exhausted".to_string(),
                )
            })?;
        // NUL cannot be expressed by the GQL savepoint grammar. The process
        // counter also prevents collisions across concurrent Session handles.
        let name = format!("{INTERNAL_SAVEPOINT_PREFIX}statement:{sequence}");
        self.savepoint_authorized(&name)?;

        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            #[cfg(feature = "testing-statement-injection")]
            if self
                .active_execution_statement_depth
                .load(Ordering::Acquire)
                == 1
            {
                self.query_cancellation_test_boundary(
                    QueryCancellationTestPhase::AfterSavepointPrepared,
                );
            }
            self.check_active_execution()?;
            body()
        })) {
            Ok(Ok(value)) => match self.release_savepoint_authorized(&name) {
                Ok(()) => Ok(value),
                Err(release_error) => {
                    let rollback = self.rollback_entire_transaction_authorized();
                    if rollback.is_err() {
                        self.poison_durability();
                    }
                    let context = match rollback {
                        Ok(()) => {
                            "statement savepoint release failed; the transaction was rolled back"
                                .to_string()
                        }
                        Err(rollback_error) => format!(
                            "statement savepoint release failed and full rollback also failed: {rollback_error}"
                        ),
                    };
                    Err(release_error.with_context(context))
                }
            },
            Ok(Err(primary)) => {
                let cleanup = self
                    .rollback_to_savepoint_authorized(&name)
                    .and_then(|()| self.release_savepoint_authorized(&name));
                match cleanup {
                    Ok(()) => Err(primary),
                    Err(cleanup_error) => {
                        let rollback = self.rollback_entire_transaction_authorized();
                        if rollback.is_err() {
                            self.poison_durability();
                        }
                        let context = match rollback {
                            Ok(()) => format!(
                                "statement savepoint cleanup failed ({cleanup_error}); the transaction was rolled back"
                            ),
                            Err(rollback_error) => format!(
                                "statement savepoint cleanup failed ({cleanup_error}) and full rollback also failed: {rollback_error}"
                            ),
                        };
                        Err(primary.with_context(context))
                    }
                }
            }
            Err(payload) => {
                let cleanup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.rollback_to_savepoint_authorized(&name)
                        .and_then(|()| self.release_savepoint_authorized(&name))
                }));
                if !matches!(cleanup, Ok(Ok(()))) {
                    let rollback = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        self.rollback_entire_transaction_authorized()
                    }));
                    if !matches!(rollback, Ok(Ok(()))) {
                        self.poison_durability();
                    }
                }
                std::panic::resume_unwind(payload)
            }
        }
    }

    /// Aborts the outer transaction even when the Session is currently inside
    /// one or more nested transaction frames.
    ///
    /// This is the fail-closed fallback for an internal savepoint protocol
    /// failure. A normal rollback intentionally unwinds only one nested frame;
    /// after statement-savepoint cleanup fails, retaining any part of the
    /// transaction would make its exact state unknowable to the caller.
    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    fn rollback_entire_transaction(&self) -> Result<()> {
        let _operation = self.session_operation_guard();
        #[cfg(feature = "lpg")]
        let _catalog_cuts = self.pin_catalog_cuts();
        self.transaction_manager
            .with_write_authority(|| self.rollback_entire_transaction_authorized())
    }

    /// Authorized half of [`Self::rollback_entire_transaction`].
    ///
    /// Commit and statement-savepoint failure paths already hold the Session
    /// operation gate and store write authority. They call this directly so
    /// cleanup is one explicit transition rather than re-entering the public
    /// transaction wrapper while handling a failure.
    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    fn rollback_entire_transaction_authorized(&self) -> Result<()> {
        *self.transaction_nesting_depth.lock() = 0;
        #[cfg(feature = "lpg")]
        {
            self.rollback_inner_authorized()
        }
        #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
        {
            self.rollback_rdf_auto_authorized()
        }
    }

    /// Auto-commit wrapper for Session LPG CRUD (nodes, edges, properties).
    #[cfg(feature = "lpg")]
    fn with_lpg_auto_commit<T>(&self, body: impl FnOnce() -> Result<T>) -> Result<T> {
        self.with_lpg_mutation_transaction(true, body)
    }

    /// Explicit lifecycle targets own their admission; an unrelated selected
    /// graph may be absent or ungranted. Transaction/historical safety is shared.
    #[cfg(feature = "lpg")]
    fn with_lpg_graph_lifecycle<T>(&self, body: impl FnOnce() -> Result<T>) -> Result<T> {
        self.with_lpg_mutation_transaction(false, body)
    }

    #[cfg(feature = "lpg")]
    fn with_lpg_mutation_transaction<T>(
        &self,
        selected_graph: bool,
        body: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let mut capture = EngineCommitCapture::default();
        let mut core_workspace = None;
        let mut transaction_workspace = crate::transaction::TransactionFinalizationWorkspace::new();
        let mut catalog_workspace = CatalogWorkspace::new();
        let _operation = self.session_operation_guard();
        let execution_statement = self.enter_execution_statement(true)?;
        let terminal_statement = !self.needs_implicit_transaction(true) || !self.auto_commit;
        let body =
            || self.finish_controlled_statement(&execution_statement, terminal_statement, body);
        let _catalog_cuts = self.pin_catalog_cuts();
        let _historical = self.historical_view_operation_gate.lock();
        self.reject_lpg_historical_mutation()?;
        self.check_not_in_mixed_snapshot()?;
        self.require_lpg("LPG mutation")?;
        self.check_not_poisoned()?;
        self.require_permission(crate::auth::StatementKind::Write)?;
        if selected_graph {
            let graph_path = self.active_graph_storage_key();
            self.require_graph_path_grant(&graph_path, crate::auth::Role::ReadWrite)?;
        }
        if *self.read_only_tx.lock() {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::ReadOnly,
            ));
        }
        self.transaction_manager.with_write_authority(|| {
            if self.needs_implicit_transaction(true) {
                self.begin_transaction_inner(false, None)?;
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if selected_graph {
                        self.ensure_current_mutation_touch()?;
                    }
                    body()
                })) {
                    Ok(Ok(value)) => {
                        if self.auto_commit {
                            self.commit_inner_authorized(
                                &mut catalog_workspace,
                                &mut transaction_workspace,
                                &mut capture,
                                &mut core_workspace,
                            )?;
                        }
                        Ok(value)
                    }
                    Ok(Err(primary)) => match self.rollback_inner() {
                        Ok(()) => Err(primary),
                        Err(rollback_error) => {
                            self.poison_durability();
                            Err(primary.with_context(format!(
                                "automatic LPG mutation rollback also failed: {rollback_error}"
                            )))
                        }
                    },
                    Err(payload) => {
                        let rollback =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                self.rollback_inner()
                            }));
                        if !matches!(rollback, Ok(Ok(()))) {
                            self.poison_durability();
                        }
                        std::panic::resume_unwind(payload)
                    }
                }
            } else {
                if selected_graph {
                    self.ensure_current_mutation_touch()?;
                }
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if selected_graph && !execution_statement.outermost {
                        body()
                    } else {
                        self.with_statement_savepoint(body)
                    }
                })) {
                    Ok(result) => result,
                    Err(payload) => {
                        let rollback =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                self.rollback_inner()
                            }));
                        if !matches!(rollback, Ok(Ok(()))) {
                            self.poison_durability();
                        }
                        std::panic::resume_unwind(payload)
                    }
                }
            }
        })
    }

    /// RDF-only auto-commit: one-statement tx so SPARQL UPDATE logs WAL and RYW.
    #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
    fn with_auto_commit<F>(&self, has_mutations: bool, body: F) -> Result<QueryResult>
    where
        F: FnOnce() -> Result<QueryResult>,
    {
        let _operation = self.session_operation_guard();
        let execution_statement = self.enter_execution_statement(has_mutations)?;
        let terminal_statement =
            has_mutations && (!self.needs_implicit_transaction(has_mutations) || !self.auto_commit);
        let body = || self.admit_query_result(body()?);
        let body =
            || self.finish_controlled_statement(&execution_statement, terminal_statement, body);
        let mut transaction_workspace = crate::transaction::TransactionFinalizationWorkspace::new();
        let run = || {
            if self.needs_implicit_transaction(has_mutations) {
                self.begin_rdf_auto()?;
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
                    Ok(Ok(result)) => {
                        if self.auto_commit {
                            self.commit_rdf_auto_authorized(&mut transaction_workspace)?;
                        }
                        Ok(result)
                    }
                    Ok(Err(primary)) => match self.rollback_rdf_auto() {
                        Ok(()) => Err(primary),
                        Err(rollback_error) => {
                            self.poison_durability();
                            Err(primary.with_context(format!(
                                "automatic RDF mutation rollback also failed: {rollback_error}"
                            )))
                        }
                    },
                    Err(payload) => {
                        let rollback =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                self.rollback_rdf_auto()
                            }));
                        if !matches!(rollback, Ok(Ok(()))) {
                            self.poison_durability();
                        }
                        std::panic::resume_unwind(payload)
                    }
                }
            } else if has_mutations {
                self.with_statement_savepoint(body)
            } else {
                body()
            }
        };
        if has_mutations {
            let _operation = self.session_operation_guard();
            self.check_not_in_mixed_snapshot()?;
            self.transaction_manager.with_write_authority(run)
        } else {
            run()
        }
    }

    /// No LPG and no RDF: nothing to wrap.
    #[cfg(all(not(feature = "lpg"), not(feature = "triple-store")))]
    fn with_auto_commit<F>(&self, _has_mutations: bool, body: F) -> Result<QueryResult>
    where
        F: FnOnce() -> Result<QueryResult>,
    {
        body()
    }

    #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
    fn begin_rdf_auto(&self) -> Result<()> {
        let _operation = self.session_operation_guard();
        self.transaction_manager
            .with_write_authority(|| self.begin_rdf_auto_authorized(None))
    }

    #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
    fn begin_rdf_with_isolation(
        &self,
        isolation_level: crate::transaction::IsolationLevel,
    ) -> Result<()> {
        let _operation = self.session_operation_guard();
        self.transaction_manager
            .with_write_authority(|| self.begin_rdf_auto_authorized(Some(isolation_level)))
    }

    #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
    fn begin_rdf_auto_authorized(
        &self,
        isolation_level: Option<crate::transaction::IsolationLevel>,
    ) -> Result<()> {
        self.check_not_poisoned()?;
        self.check_not_in_mixed_snapshot()?;
        // Match the store-backed lifecycle-before-publication lock order and
        // reject nested work after the owning database has closed.
        let database_open = self.database_open.read();
        if !*database_open {
            return Err(Self::database_closed_error());
        }
        let nested = self.current_transaction.lock().is_some();
        if nested {
            let mut depth = self.transaction_nesting_depth.lock();
            let next_depth = depth.checked_add(1).ok_or_else(|| {
                grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::InvalidState(
                        "nested transaction depth exhausted".to_string(),
                    ),
                )
            })?;
            let savepoint_name = format!("{INTERNAL_SAVEPOINT_PREFIX}nested:{next_depth}");
            self.savepoint_authorized(&savepoint_name)?;
            *depth = next_depth;
            return Ok(());
        }
        let _publication = self.transaction_manager.publication().read();
        let mut current = self.current_transaction.lock();
        if current.is_some() {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "transaction state changed while beginning a transaction".to_string(),
                ),
            ));
        }
        let transaction_id = isolation_level.map_or_else(
            || self.transaction_manager.begin(),
            |level| self.transaction_manager.begin_with_isolation(level),
        );
        *current = Some(transaction_id);
        if self.transaction_manager.isolation_level(transaction_id)
            != Some(crate::transaction::IsolationLevel::ReadCommitted)
            && let Some(epoch) = self.transaction_manager.start_epoch(transaction_id)
        {
            self.rdf_store
                .register_transaction_snapshot(transaction_id, epoch);
        }
        Ok(())
    }

    #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
    fn commit_rdf_auto(&self) -> Result<grafeo_common::types::EpochId> {
        let mut transaction_workspace = crate::transaction::TransactionFinalizationWorkspace::new();
        let _operation = self.session_operation_guard();
        self.transaction_manager
            .with_write_authority(|| self.commit_rdf_auto_authorized(&mut transaction_workspace))
    }

    #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
    fn commit_rdf_auto_authorized(
        &self,
        transaction_workspace: &mut crate::transaction::TransactionFinalizationWorkspace,
    ) -> Result<grafeo_common::types::EpochId> {
        self.check_not_poisoned()?;
        self.check_not_in_mixed_snapshot()?;
        {
            let mut depth = self.transaction_nesting_depth.lock();
            if *depth > 0 {
                let savepoint_name = format!("{INTERNAL_SAVEPOINT_PREFIX}nested:{depth}");
                self.release_savepoint_authorized(&savepoint_name)?;
                *depth -= 1;
                return Ok(self.transaction_manager.current_epoch());
            }
        }
        let transaction_id = self.current_transaction.lock().take().ok_or_else(|| {
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;
        let _gate = self.rdf_store.lock_commit();
        let _publication = self.transaction_manager.publication().write();
        #[cfg(feature = "cdc")]
        let mut prepared_cdc = None;
        #[cfg(all(feature = "cdc", feature = "wal"))]
        let mut cdc_models = None;
        let (epoch, prepared_transaction) = match self
            .validate_rdf_transaction_lifecycle(transaction_id)
            .and_then(|()| {
                self.transaction_manager
                    .prepare_durable_commit(transaction_id)
            })
            .and_then(|epoch| {
                #[cfg(feature = "cdc")]
                let cdc_batch = self
                    .cdc_pending_events
                    .as_ref()
                    .map(|pending| pending.prepare_committed(epoch))
                    .transpose()?;
                #[cfg(all(feature = "cdc", feature = "wal"))]
                let cdc_records = if self.wal.is_some() {
                    cdc_batch
                        .as_ref()
                        .map(|batch| batch.wal_records(transaction_id, epoch))
                        .transpose()?
                } else {
                    None
                };
                let released = self.transaction_manager.prepare_finalization(
                    transaction_id,
                    epoch,
                    transaction_workspace,
                )?;
                let prepared = released.rebind().map_err(|error| error.into_error())?;
                #[cfg(feature = "cdc")]
                {
                    prepared_cdc = cdc_batch
                        .map(crate::cdc::PreparedCdcBatch::prepare_publication)
                        .transpose()?;
                }
                #[cfg(all(feature = "cdc", feature = "wal"))]
                if let (Some(wal), Some(records)) = (&self.wal, cdc_records) {
                    let mut models = 0;
                    for record in records {
                        if let grafeo_storage::wal::WalRecord::CdcBatch { model, .. } = &record {
                            models |= model;
                        }
                        wal.log(&record)?;
                    }
                    cdc_models = Some(models);
                }
                #[cfg(feature = "testing-statement-injection")]
                self.query_cancellation_test_boundary(
                    QueryCancellationTestPhase::BeforeCommitFence,
                );
                if let Some(control) = self.active_execution_control.lock().as_mut() {
                    control
                        .try_begin_commit()
                        .map_err(Self::map_query_lifecycle_error)?;
                }
                Ok((epoch, prepared))
            }) {
            Ok(prepared) => prepared,
            Err(error) => {
                #[cfg(feature = "cdc")]
                drop(prepared_cdc.take());
                self.rollback_rdf_transaction(transaction_id);
                let _ = self.transaction_manager.abort(transaction_id);

                #[cfg(feature = "cdc")]
                if let Some(ref pending) = self.cdc_pending_events {
                    pending.clear();
                }
                *self.read_only_tx.lock() = self.db_read_only;
                self.savepoints.lock().clear();

                #[cfg(feature = "wal")]
                if let Some(ref wal) = self.wal {
                    use grafeo_storage::wal::WalRecord;
                    if let Err(abort_error) =
                        wal.log(&WalRecord::TransactionAbort { transaction_id })
                    {
                        self.poison_durability();
                        return Err(grafeo_common::utils::error::Error::Transaction(
                            grafeo_common::utils::error::TransactionError::DurabilityFailure(
                                format!(
                                    "RDF transaction preparation failed ({error}); WAL abort acknowledgement also failed ({abort_error}); reopen and recover before continuing"
                                ),
                            ),
                        ));
                    }
                }
                return Err(error);
            }
        };
        #[cfg(feature = "wal")]
        if let Some(ref wal) = self.wal {
            use grafeo_storage::wal::WalRecord;
            let marker = WalRecord::Committed {
                transaction_id,
                epoch,
            };
            #[cfg(feature = "cdc")]
            let marker = cdc_models.map_or(marker, |models| WalRecord::CommittedWithCdc {
                transaction_id,
                epoch,
                models,
            });
            if let Err(e) = wal.log(&marker) {
                drop(prepared_transaction);
                // As in the dual-model path, a failed acknowledgement may
                // follow a successfully synced commit marker.  Preserve the
                // prepared transaction and ask recovery to resolve it.
                self.poison_durability();
                return Err(grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::DurabilityFailure(format!(
                        "WAL commit acknowledgement failed and the RDF transaction outcome is unknown: {e}; reopen and recover before continuing"
                    )),
                ));
            }
        }
        #[cfg(all(feature = "wal", feature = "testing-statement-injection"))]
        if self.wal.is_some() {
            self.query_cancellation_test_boundary(QueryCancellationTestPhase::AfterDurableMarker);
        }
        let transaction_cleanup = prepared_transaction.install().release();
        if let Err(error) = self.commit_rdf_transaction(transaction_id, epoch) {
            self.poison_durability();
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::DurabilityFailure(format!(
                    "durable RDF transaction could not be published: {error}; reopen and recover before continuing"
                )),
            ));
        }

        transaction_cleanup.finish();
        #[cfg(feature = "cdc")]
        if let Some(prepared) = prepared_cdc.take() {
            prepared.publish();
        }
        *self.read_only_tx.lock() = self.db_read_only;
        self.savepoints.lock().clear();
        Ok(epoch)
    }

    #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
    fn rollback_rdf_auto(&self) -> Result<()> {
        let _operation = self.session_operation_guard();
        self.transaction_manager
            .with_write_authority(|| self.rollback_rdf_auto_authorized())
    }

    #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
    fn rollback_rdf_auto_authorized(&self) -> Result<()> {
        {
            let mut depth = self.transaction_nesting_depth.lock();
            if *depth > 0 {
                let savepoint_name = format!("{INTERNAL_SAVEPOINT_PREFIX}nested:{depth}");
                self.rollback_to_savepoint_authorized(&savepoint_name)?;
                self.release_savepoint_authorized(&savepoint_name)?;
                *depth -= 1;
                return Ok(());
            }
        }
        let transaction_id = self.current_transaction.lock().take().ok_or_else(|| {
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;
        let _ = self.transaction_manager.abort(transaction_id);
        self.rollback_rdf_transaction(transaction_id);
        #[cfg(feature = "cdc")]
        if let Some(ref pending) = self.cdc_pending_events {
            pending.clear();
        }
        *self.read_only_tx.lock() = self.db_read_only;
        self.savepoints.lock().clear();
        #[cfg(feature = "wal")]
        if let Some(ref wal) = self.wal {
            use grafeo_storage::wal::WalRecord;
            if let Err(e) = wal.log(&WalRecord::TransactionAbort { transaction_id }) {
                self.poison_durability();
                return Err(grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::DurabilityFailure(format!(
                        "WAL abort log failed: {e}"
                    )),
                ));
            }
        }
        Ok(())
    }

    /// Returns `Err(Transaction(InvalidState))` if any `ResultStream` is
    /// still outstanding for this session.
    #[cfg(feature = "lpg")]
    fn check_no_active_streams(&self, op: &str) -> Result<()> {
        if self.active_streams.load(Ordering::Acquire) > 0 {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(format!(
                    "Cannot {op} while streaming results are active; drop the stream first"
                )),
            ));
        }
        Ok(())
    }

    /// Computes the wall-clock deadline for query execution.
    #[cfg(any(test, not(target_arch = "wasm32")))]
    #[must_use]
    fn query_deadline(&self) -> Option<Instant> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.query_timeout
                .and_then(|duration| Instant::now().checked_add(duration))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = &self.query_timeout;
            None
        }
    }

    /// Composes the Session compatibility timeout exactly once at the
    /// controlled-query orchestration boundary.
    fn compose_query_checkpoint(
        &self,
        checkpoint: grafeo_core::execution::QueryExecutionCheckpoint,
    ) -> grafeo_core::execution::QueryExecutionCheckpoint {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(deadline) = self.query_deadline() {
            return checkpoint.with_additional_deadline(deadline, self.query_timeout);
        }

        checkpoint
    }

    fn check_query_checkpoint(
        checkpoint: &grafeo_core::execution::QueryExecutionCheckpoint,
    ) -> Result<()> {
        checkpoint
            .check()
            .map_err(Self::map_query_cancellation_error)
    }

    fn map_query_cancellation_error(
        error: grafeo_core::execution::QueryCancellationError,
    ) -> grafeo_common::utils::error::Error {
        match error {
            grafeo_core::execution::QueryCancellationError::Cancelled => {
                grafeo_common::utils::error::Error::Query(
                    grafeo_common::utils::error::QueryError::cancelled(),
                )
            }
            grafeo_core::execution::QueryCancellationError::DeadlineExceeded {
                timeout: Some(timeout),
            } => grafeo_common::utils::error::Error::Query(
                grafeo_common::utils::error::QueryError::timeout_with_limit(timeout),
            ),
            grafeo_core::execution::QueryCancellationError::DeadlineExceeded { timeout: None } => {
                grafeo_common::utils::error::Error::Query(
                    grafeo_common::utils::error::QueryError::timeout(),
                )
            }
            grafeo_core::execution::QueryCancellationError::InvalidState { phase } => {
                grafeo_common::utils::error::Error::Internal(format!(
                    "query execution state is invalid: {phase}"
                ))
            }
            _ => grafeo_common::utils::error::Error::Internal(format!(
                "unknown query cancellation reason: {error}"
            )),
        }
    }

    fn map_query_lifecycle_error(
        error: grafeo_core::execution::QueryLifecycleError,
    ) -> grafeo_common::utils::error::Error {
        match error {
            grafeo_core::execution::QueryLifecycleError::Cancelled(cancelled) => {
                Self::map_query_cancellation_error(cancelled)
            }
            grafeo_core::execution::QueryLifecycleError::CommitAlreadyStarted => {
                grafeo_common::utils::error::Error::Internal(
                    "query commit has already begun".to_string(),
                )
            }
            grafeo_core::execution::QueryLifecycleError::ExecutionAlreadyFinished => {
                grafeo_common::utils::error::Error::Internal(
                    "query execution has already finished".to_string(),
                )
            }
            grafeo_core::execution::QueryLifecycleError::InvalidState { phase } => {
                grafeo_common::utils::error::Error::Internal(format!(
                    "query execution state is invalid: {phase}"
                ))
            }
            other => grafeo_common::utils::error::Error::Internal(format!(
                "unknown query lifecycle error: {other}"
            )),
        }
    }

    fn result_resources(&self) -> Result<grafeo_core::execution::QueryResourceContext> {
        let checkpoint = self.installed_or_fresh_checkpoint();
        let buffer = self.buffer_manager.as_ref().ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "session has no buffer manager for result accounting".into(),
            )
        })?;
        grafeo_core::execution::QueryResourceContext::new_with_cancellation(
            Arc::clone(buffer),
            checkpoint.token(),
        )
        .map_err(Self::map_query_resource_context_error)
    }

    fn admit_query_result(&self, result: QueryResult) -> Result<QueryResult> {
        let admission = *self.active_result_admission.lock();
        if let Some(admit) = admission {
            admit(&result, self.effective_result_limits())?;
        }
        Ok(result)
    }

    #[cfg(feature = "gql")]
    fn bounded_status(&self, message: std::fmt::Arguments<'_>) -> Result<QueryResult> {
        self.admit_query_result(crate::query::executor::bounded_status_result(
            self.result_resources()?,
            message,
        )?)
    }

    fn effective_result_limits(&self) -> crate::query::ResultLimits {
        self.active_result_limits
            .lock()
            .unwrap_or(self.result_limits)
    }

    /// Executes a cacheable physical tree with the same query resource and
    /// cancellation identity as owned execution, without consuming the tree.
    /// Reset drains resident scratch before spill finalization or cache return.
    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "sparql",
        feature = "graphql"
    ))]
    fn execute_borrowed_physical_plan(&self, plan: &mut PhysicalPlan) -> Result<QueryResult> {
        self.execute_borrowed_profiled_plan(plan, &[])
    }

    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "sparql",
        feature = "graphql"
    ))]
    fn execute_borrowed_profiled_plan(
        &self,
        plan: &mut PhysicalPlan,
        profile: &[crate::query::profile::ProfileEntry],
    ) -> Result<QueryResult> {
        let checkpoint = self.installed_or_fresh_checkpoint();
        Self::check_query_checkpoint(&checkpoint)?;
        let resources = self.make_query_resource_context(Some(checkpoint.token()))?;
        let execution = (|| {
            let executor = Executor::with_bounded_columns(
                &plan.columns,
                resources.clone(),
                self.effective_result_limits(),
            )?
            .with_execution_checkpoint(checkpoint.clone());
            plan.operator
                .install_resource_context(&resources)
                .map_err(Self::map_query_resource_context_error)?;
            Self::check_query_checkpoint(&checkpoint)?;
            executor.execute(plan.operator.as_mut())
        })();
        plan.operator.reset();
        #[cfg(feature = "spill")]
        let spill_manager = resources.spill_manager().cloned();
        // Preserve ordinary execution's release order; only profiling needs
        // its memory account alive through the final snapshot.
        let resources = (!profile.is_empty()).then_some(resources);
        #[cfg(feature = "spill")]
        let result = finish_spill_execution(execution, spill_manager);
        #[cfg(not(feature = "spill"))]
        let result = execution;
        if let Some(resources) = &resources {
            let snapshot = resources.profile_stats();
            for entry in profile {
                entry.stats.lock().query_resources = Some(snapshot);
            }
        }
        drop(resources);
        let result = result?;
        Self::check_query_checkpoint(&checkpoint)?;
        Ok(result)
    }

    /// Executes one owned physical plan under a single resource and
    /// cancellation identity.
    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn execute_physical_plan(&self, physical_plan: PhysicalPlan) -> Result<QueryResult> {
        let checkpoint = self.installed_or_fresh_checkpoint();
        self.execute_physical_plan_with_checkpoint(physical_plan, checkpoint, &[])
    }

    fn installed_or_fresh_checkpoint(&self) -> grafeo_core::execution::QueryExecutionCheckpoint {
        if let Some(control) = self.active_execution_control.lock().as_ref() {
            control.checkpoint()
        } else {
            self.compose_query_checkpoint(
                grafeo_core::execution::QueryExecutionControl::new().checkpoint(),
            )
        }
    }

    /// Executes one owned physical plan with an already-composed checkpoint.
    ///
    /// The executor, every blocking operator, and spill cleanup share the
    /// exact same checkpoint token. The final check catches a deadline or
    /// cancellation that wins during a last, zero-row pull.
    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn execute_physical_plan_with_checkpoint(
        &self,
        physical_plan: PhysicalPlan,
        checkpoint: grafeo_core::execution::QueryExecutionCheckpoint,
        profile: &[crate::query::profile::ProfileEntry],
    ) -> Result<QueryResult> {
        Self::check_query_checkpoint(&checkpoint)?;
        let resources = self.make_query_resource_context(Some(checkpoint.token()))?;
        let PhysicalPlan {
            operator, columns, ..
        } = physical_plan;
        let execution = (|| {
            let executor = Executor::with_bounded_columns(
                &columns,
                resources.clone(),
                self.effective_result_limits(),
            )?
            .with_execution_checkpoint(checkpoint.clone());
            let conversion =
                grafeo_core::execution::pipeline_convert::convert_to_pipeline_with_resources(
                    operator, &resources,
                )
                .map_err(Self::map_query_resource_context_error);
            match conversion {
                Ok((mut source, push_ops)) if push_ops.is_empty() => {
                    let execution = Self::check_query_checkpoint(&checkpoint)
                        .and_then(|()| executor.execute(source.as_mut()));
                    drop(source);
                    drop(push_ops);
                    execution
                }
                Ok((source, push_ops)) => Self::check_query_checkpoint(&checkpoint)
                    .and_then(|()| executor.execute_pipeline(source, push_ops)),
                Err(error) => Err(error),
            }
        })();
        #[cfg(feature = "spill")]
        let spill_manager = resources.spill_manager().cloned();
        // Preserve ordinary execution's release order; only profiling needs
        // its memory account alive through the final snapshot.
        let resources = (!profile.is_empty()).then_some(resources);
        #[cfg(feature = "spill")]
        let result = finish_spill_execution(execution, spill_manager);
        #[cfg(not(feature = "spill"))]
        let result = execution;
        if let Some(resources) = &resources {
            let snapshot = resources.profile_stats();
            for entry in profile {
                entry.stats.lock().query_resources = Some(snapshot);
            }
        }
        drop(resources);
        let result = result?;
        Self::check_query_checkpoint(&checkpoint)?;
        Ok(result)
    }

    fn map_query_resource_context_error(
        error: grafeo_core::execution::QueryResourceContextError,
    ) -> grafeo_common::utils::error::Error {
        let message = error.to_string();
        match error {
            grafeo_core::execution::QueryResourceContextError::Memory(
                grafeo_common::memory::buffer::MemoryGrantError::LimitExceeded { .. },
            ) => grafeo_common::utils::error::Error::Storage(
                grafeo_common::utils::error::StorageError::Full,
            )
            .with_context(message),
            #[cfg(feature = "spill")]
            grafeo_core::execution::QueryResourceContextError::SpillAdmission {
                kind,
                message,
                cancellation,
            } => match cancellation {
                Some(reason) => Self::map_query_cancellation_error(reason).with_context(message),
                None => std::io::Error::new(kind, message).into(),
            },
            _ => grafeo_common::utils::error::Error::Internal(message),
        }
    }

    /// Creates one always-on resource context for the current query execution.
    ///
    /// Absence of a spill path selects resident-only resources, not absence of
    /// accounting. A configured root is created fallibly and never downgraded
    /// after setup failure. Encrypted databases install an authenticated
    /// record provider before any query directory is created.
    fn make_query_resource_context(
        &self,
        cancellation: Option<grafeo_core::execution::QueryCancellationToken>,
    ) -> Result<grafeo_core::execution::QueryResourceContext> {
        let bm = self.buffer_manager.as_ref().ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "session has no buffer manager for query resource accounting".to_string(),
            )
        })?;

        #[cfg(feature = "spill")]
        if let Some(spill_path) = bm.config().spill_path.as_ref() {
            let cancellation = cancellation
                .unwrap_or_else(|| grafeo_core::execution::QueryExecutionControl::new().token());
            cancellation
                .check()
                .map_err(Self::map_query_cancellation_error)?;
            // Public callers already own their publication barrier. Read the
            // shared identity here so an idle Session follows snapshot restore.
            let world = self.world_identity.read();
            let root = self
                .spill_root
                .open(spill_path, world.store_id(), &cancellation)
                .map_err(|error| {
                    match error.get_ref().and_then(|source| {
                        source.downcast_ref::<grafeo_core::execution::QueryCancellationError>()
                    }) {
                        Some(reason) => Self::map_query_cancellation_error(*reason),
                        None => error.into(),
                    }
                })?;
            return grafeo_core::execution::QueryResourceContext::with_spill_root(
                Arc::clone(bm),
                &root,
                cancellation,
            )
            .map_err(Self::map_query_resource_context_error);
        }

        match cancellation {
            Some(cancellation) => {
                grafeo_core::execution::QueryResourceContext::new_with_cancellation(
                    std::sync::Arc::clone(bm),
                    cancellation,
                )
            }
            None => grafeo_core::execution::QueryResourceContext::new(std::sync::Arc::clone(bm)),
        }
        .map_err(Self::map_query_resource_context_error)
    }

    /// Checks that a property value does not exceed the configured size limit.
    #[cfg(feature = "lpg")]
    fn check_property_size(&self, key: &str, value: &Value) -> Result<()> {
        if let Some(limit) = self.max_property_size {
            let size = value.estimated_size_bytes();
            if size > limit {
                let limit_display = if limit >= 1024 * 1024 && limit % (1024 * 1024) == 0 {
                    format!("{} MiB", limit / (1024 * 1024))
                } else if limit >= 1024 && limit % 1024 == 0 {
                    format!("{} KiB", limit / 1024)
                } else {
                    format!("{limit} bytes")
                };
                return Err(grafeo_common::utils::error::Error::Query(
                    grafeo_common::utils::error::QueryError::new(
                        grafeo_common::utils::error::QueryErrorKind::Execution,
                        format!(
                            "Property '{key}' value exceeds maximum size of {limit_display} ({size} bytes)"
                        ),
                    )
                    .with_hint(
                        "Increase with Config::with_max_property_size() or disable with Config::without_max_property_size()".to_string(),
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Records query metrics for any language.
    ///
    /// Called after query execution to update counters, latency histogram,
    /// and per-language tracking. `elapsed_ms` should be `None` on WASM
    /// where `Instant` is unavailable.
    #[cfg(feature = "metrics")]
    fn record_query_metrics(
        &self,
        language: &str,
        elapsed_ms: Option<f64>,
        result: &Result<crate::database::QueryResult>,
    ) {
        use crate::metrics::record_metric;

        record_metric!(self.metrics, query_count, inc);
        if let Some(ref reg) = self.metrics {
            reg.query_count_by_language.increment(language);
        }
        if let Some(ms) = elapsed_ms {
            record_metric!(self.metrics, query_latency, observe ms);
        }
        match result {
            Ok(r) => {
                let returned = r.row_count() as u64;
                record_metric!(self.metrics, rows_returned, add returned);
                if let Some(scanned) = r.rows_scanned {
                    record_metric!(self.metrics, rows_scanned, add scanned);
                }
            }
            Err(e) => {
                record_metric!(self.metrics, query_errors, inc);
                // Detect timeout errors
                let msg = e.to_string();
                if msg.contains("exceeded timeout") {
                    record_metric!(self.metrics, query_timeouts, inc);
                }
            }
        }
    }

    /// Evaluates a simple integer literal from a session parameter expression.
    #[cfg(feature = "gql")]
    fn eval_integer_literal(expr: &grafeo_adapters::query::gql::ast::Expression) -> Option<i64> {
        use grafeo_adapters::query::gql::ast::{Expression, Literal};
        match expr {
            Expression::Literal(Literal::Integer(n)) => Some(*n),
            _ => None,
        }
    }

    /// Returns the current transaction context for MVCC visibility.
    ///
    /// Returns `(viewing_epoch, transaction_id)` where:
    /// - `viewing_epoch` is the epoch at which to check version visibility
    /// - `transaction_id` is the current transaction ID (if in a transaction)
    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    #[must_use]
    fn get_transaction_context(&self) -> (EpochId, Option<TransactionId>) {
        if let Some(transaction_id) = *self.current_transaction.lock() {
            // A transaction's BEGIN cut is immutable. A persistent historical
            // override applies again only after the transaction closes; it
            // must never erase the real ID and lose read-your-writes.
            let epoch = self
                .transaction_manager
                .start_epoch(transaction_id)
                .unwrap_or_else(|| self.transaction_manager.current_epoch());
            return (epoch, Some(transaction_id));
        }

        // PENDING is an internal uncommitted-version sentinel. Public
        // historical APIs have long treated it as "current"; normalize it to
        // the committed publication cut so it cannot expose another writer.
        let epoch = self
            .effective_viewing_epoch()
            .filter(|epoch| *epoch != EpochId::PENDING)
            .unwrap_or_else(|| self.transaction_manager.current_epoch());
        (epoch, None)
    }

    /// Returns the COPY statement's MVCC cut.
    ///
    /// The caller must hold a publication read guard. That guard is what turns
    /// `current_epoch()` into a fully installed state boundary: commit reserves
    /// its epoch before final publication while holding the opposing write
    /// guard. Snapshot Isolation and Serializable retain their BEGIN cut.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn copy_transaction_context(&self) -> (EpochId, Option<TransactionId>) {
        if let Some(transaction_id) = *self.current_transaction.lock()
            && self.transaction_manager.isolation_level(transaction_id)
                == Some(crate::transaction::IsolationLevel::ReadCommitted)
        {
            return (
                self.transaction_manager.current_epoch(),
                Some(transaction_id),
            );
        }
        self.get_transaction_context()
    }

    /// Creates a planner from one linearizable graph/schema context snapshot.
    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn create_planner_for_store_with_graph_context(
        &self,
        store: Arc<dyn GraphStoreSearch>,
        viewing_epoch: EpochId,
        transaction_id: Option<TransactionId>,
        read_only: bool,
        graph_context: &SessionGraphContext,
    ) -> crate::query::Planner {
        use crate::catalog::CatalogConstraintValidator;
        use crate::query::Planner;
        use grafeo_core::execution::operators::{LazyValue, SessionContext};

        // Capture store reference for lazy introspection (only computed if info()/schema() called).
        let info_store = Arc::clone(&store);
        let schema_store = Arc::clone(&store);

        let session_context = SessionContext {
            current_schema: graph_context.schema.clone(),
            current_graph: graph_context.graph.clone(),
            db_info: LazyValue::new(move || Self::build_info_value(&*info_store)),
            schema_info: LazyValue::new(move || Self::build_schema_value(&*schema_store)),
        };

        let write_store = self.write_store_for_graph_storage_key(&graph_context.storage_key);

        // A pending CREATE graph is an exact detached incarnation owned only
        // by this transaction. Preserve its transaction ID for MVCC/WAL/CDC,
        // but do not project its graph-local numeric IDs into the shared,
        // graph-unqualified conflict keyspace before lifecycle publication.
        #[cfg(feature = "lpg")]
        let transaction_private_store = self.pending_created_graphs.lock().keys().any(|prefix| {
            graph_context
                .storage_key
                .components()
                .starts_with(prefix.components())
        });

        let granularity = *self.conflict_granularity.lock();
        let mut planner = Planner::with_context(
            Arc::clone(&store),
            write_store,
            Arc::clone(&self.transaction_manager),
            transaction_id,
            viewing_epoch,
        )
        .with_conflict_granularity(granularity)
        .with_factorized_execution(self.factorized_execution)
        .with_catalog(self.catalog_view())
        .with_session_context(session_context)
        .with_read_only(read_only);
        #[cfg(feature = "lpg")]
        if transaction_private_store {
            // Apply this after conflict granularity: Property mode replaces
            // the tracker and must not accidentally re-enable it.
            planner = planner.without_entity_conflict_tracking();
        }

        // Attach the exact captured LPG incarnation so CALL grafeo.search.*
        // procedures use indexes from the same authorized graph as every
        // other operator. Skip external, missing, and denied stores rather
        // than falling back to the root/default store.
        #[cfg(feature = "lpg")]
        if matches!(self.lpg_backend, LpgBackend::Active)
            && self.index_path_grant_allows(&graph_context.storage_key, crate::auth::Role::ReadOnly)
        {
            let procedure_store = self.session_graph_path(&graph_context.storage_key);
            if let Some(procedure_store) = procedure_store {
                planner = planner.with_lpg_store(procedure_store);
            }
        }

        // Attach the constraint validator for schema enforcement and property size limits
        let mut validator = CatalogConstraintValidator::new(self.catalog_view())
            .with_store(store)
            .with_max_property_size(self.max_property_size)
            .with_schema(self.constraint_schema_for_path(&graph_context.storage_key));
        #[cfg(all(feature = "lpg", feature = "triple-store"))]
        if graph_context.storage_key.components().is_empty()
            && let Some(target) = self.rdf_projection_target.lock().clone()
        {
            validator = validator.with_rdf_projection_authority(
                target.owner_marker,
                target.node_label,
                target.desired_iris,
            );
        }
        let binding = self.session_graph_type_binding(&graph_context.storage_key);
        validator = validator
            .with_graph_path(graph_context.storage_key.clone())
            .with_graph_type_binding_override(binding);
        planner = planner.with_validator(Arc::new(validator));

        planner
    }

    /// Builds a `Value::Map` for the `info()` introspection function.
    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn build_info_value(store: &dyn GraphStore) -> Value {
        use grafeo_common::types::PropertyKey;
        use std::collections::BTreeMap;

        let mut map = BTreeMap::new();
        map.insert(PropertyKey::from("mode"), Value::String("lpg".into()));
        // reason: node/edge counts will not exceed i64::MAX
        #[allow(clippy::cast_possible_wrap)]
        let node_count = store.node_count() as i64;
        // reason: value is a small counter, well within i64::MAX
        #[allow(clippy::cast_possible_wrap)]
        let edge_count = store.edge_count() as i64;
        map.insert(PropertyKey::from("node_count"), Value::Int64(node_count));
        map.insert(PropertyKey::from("edge_count"), Value::Int64(edge_count));
        map.insert(
            PropertyKey::from("version"),
            Value::String(env!("CARGO_PKG_VERSION").into()),
        );
        Value::Map(map.into())
    }

    /// Builds a `Value::Map` for the `schema()` introspection function.
    #[cfg(any(
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    fn build_schema_value(store: &dyn GraphStore) -> Value {
        use grafeo_common::types::PropertyKey;
        use std::collections::BTreeMap;

        let labels: Vec<Value> = store
            .all_labels()
            .into_iter()
            .map(|l| Value::String(l.into()))
            .collect();
        let edge_types: Vec<Value> = store
            .all_edge_types()
            .into_iter()
            .map(|t| Value::String(t.into()))
            .collect();
        let property_keys: Vec<Value> = store
            .all_property_keys()
            .into_iter()
            .map(|k| Value::String(k.into()))
            .collect();

        let mut map = BTreeMap::new();
        map.insert(PropertyKey::from("labels"), Value::List(labels.into()));
        map.insert(
            PropertyKey::from("edge_types"),
            Value::List(edge_types.into()),
        );
        map.insert(
            PropertyKey::from("property_keys"),
            Value::List(property_keys.into()),
        );
        Value::Map(map.into())
    }

    /// Language/schema name adapter over the canonical graph lifecycle.
    #[cfg(feature = "lpg")]
    pub(crate) fn create_named_graph(&self, name: &str) -> Result<bool> {
        self.with_lpg_graph_lifecycle(|| {
            let _publication = self.publication_read_guard();
            let (storage_key, namespace) = self.resolve_parser_free_graph_name(name)?;
            let path = Self::graph_path_for_storage_key(Some(&storage_key))?;
            self.stage_create_graph_path(&path, namespace)
        })
    }

    /// Resolves the language/schema name once and returns that exact flat key.
    #[cfg(feature = "lpg")]
    pub(crate) fn drop_named_graph_resolved(&self, name: &str) -> Result<(bool, String)> {
        self.with_lpg_graph_lifecycle(|| {
            let _publication = self.publication_read_guard();
            let (storage_key, namespace) = self.resolve_parser_free_graph_name(name)?;
            let path = Self::graph_path_for_storage_key(Some(&storage_key))?;
            let dropped = self.stage_drop_graph_path(&path, &namespace)?;
            Ok((dropped, storage_key))
        })
    }

    /// Creates a node directly (bypassing query execution).
    ///
    /// This is a low-level API for testing and direct manipulation.
    /// If a transaction is active, the node will be versioned with the transaction ID.
    #[cfg(feature = "lpg")]
    pub fn create_node(&self, labels: &[&str]) -> NodeId {
        self.with_lpg_auto_commit(|| self.create_node_in_tx(labels))
            .unwrap_or(NodeId::INVALID)
    }

    #[cfg(feature = "lpg")]
    fn create_node_in_tx(&self, labels: &[&str]) -> Result<NodeId> {
        let (epoch, transaction_id) = self.get_transaction_context();
        let graph_path = self.active_graph_storage_key();
        let store = self.require_lpg_store_for_storage_key(&graph_path)?;
        let id = store.create_node_versioned(
            labels,
            epoch,
            transaction_id.unwrap_or(TransactionId::SYSTEM),
        );

        #[cfg(feature = "wal")]
        if self.wal.is_some() {
            self.log_lpg_in_graph(
                transaction_id,
                &graph_path,
                grafeo_storage::wal::LpgMutationOp::CreateNode {
                    id,
                    labels: labels.iter().map(|s| (*s).to_string()).collect(),
                },
            )
            .inspect_err(|_| {
                let _ = store.delete_node(id);
            })?;
        }

        #[cfg(feature = "cdc")]
        self.stage_lpg_node_create(
            id,
            None,
            Some(labels.iter().map(|label| (*label).to_string()).collect()),
            &graph_path,
            Arc::clone(&store),
        );

        Ok(id)
    }

    /// Creates a node with properties.
    ///
    /// If a transaction is active, the node will be versioned with the transaction ID.
    ///
    /// # Errors
    ///
    /// Returns an error if any property value exceeds the configured `max_property_size`.
    #[cfg(feature = "lpg")]
    pub fn create_node_with_props<'a>(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (&'a str, Value)>,
    ) -> Result<NodeId> {
        let props: Vec<(&str, Value)> = properties.into_iter().collect();
        for (key, value) in &props {
            self.check_property_size(key, value)?;
        }

        // Snapshot the props for WAL before passing them to the LPG store.
        // The LPG store consumes `props` by value; we need owned copies for
        // post-hoc WAL emission.
        #[cfg(feature = "wal")]
        let wal_props: Vec<(String, Value)> = props
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect();

        self.with_lpg_auto_commit(|| {
            let (epoch, transaction_id) = self.get_transaction_context();
            let graph_path = self.active_graph_storage_key();
            let label_names: Vec<String> =
                labels.iter().map(|label| (*label).to_string()).collect();
            let property_values: Vec<(String, Value)> = props
                .iter()
                .map(|(key, value)| ((*key).to_string(), value.clone()))
                .collect();
            self.direct_mutation_validator()
                .validate_node_post_image(
                    None,
                    &label_names,
                    &property_values,
                    epoch,
                    transaction_id,
                )
                .map_err(|error| {
                    grafeo_common::utils::error::Error::Query(
                        grafeo_common::utils::error::QueryError::new(
                            grafeo_common::utils::error::QueryErrorKind::Semantic,
                            format!("Direct node creation rejected: {error}"),
                        ),
                    )
                })?;
            let store = self.require_lpg_store_for_storage_key(&graph_path)?;
            let id = store.create_node_versioned(
                labels,
                epoch,
                transaction_id.unwrap_or(TransactionId::SYSTEM),
            );
            if let Some(transaction_id) = transaction_id {
                // Keep property values and every derived index out of committed
                // state until publication. This is the same overlay used by
                // query CREATE/SET, so the creating transaction still gets RYW
                // while another session cannot observe an uncommitted property
                // index hit (or a stale hit after rollback).
                for (key, value) in &props {
                    store.set_node_property_buffered(id, key, value.clone(), transaction_id);
                }
            } else {
                for (key, value) in &props {
                    store.set_node_property(id, key, value.clone());
                }
            }

            #[cfg(feature = "wal")]
            if self.wal.is_some() {
                self.log_lpg_in_graph(
                    transaction_id,
                    &graph_path,
                    grafeo_storage::wal::LpgMutationOp::CreateNode {
                        id,
                        labels: labels.iter().map(|s| (*s).to_string()).collect(),
                    },
                )?;
                for (key, value) in &wal_props {
                    self.log_lpg_in_graph(
                        transaction_id,
                        &graph_path,
                        grafeo_storage::wal::LpgMutationOp::SetNodeProperty {
                            id,
                            key: key.clone(),
                            value: value.clone(),
                        },
                    )?;
                }
            }

            #[cfg(feature = "cdc")]
            self.stage_lpg_node_create(
                id,
                (!property_values.is_empty()).then(|| property_values.into_iter().collect()),
                Some(label_names),
                &graph_path,
                Arc::clone(&store),
            );

            Ok(id)
        })
    }

    /// Creates an edge between two nodes.
    ///
    /// This is a low-level API for testing and direct manipulation.
    /// If a transaction is active, the edge will be versioned with the transaction ID.
    #[cfg(feature = "lpg")]
    pub fn create_edge(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
    ) -> grafeo_common::types::EdgeId {
        self.with_lpg_auto_commit(|| self.create_edge_on_admitted_writer(src, dst, edge_type, &[]))
            .unwrap_or(grafeo_common::types::EdgeId::INVALID)
    }

    /// Creates an edge with properties within the active transaction context.
    ///
    /// # Errors
    ///
    /// Returns an error if an endpoint is not visible, the writer rejects the
    /// creation, durability fails, or a property exceeds `max_property_size`.
    #[cfg(feature = "lpg")]
    pub fn create_edge_with_props<'a>(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: impl IntoIterator<Item = (&'a str, Value)>,
    ) -> Result<grafeo_common::types::EdgeId> {
        let props: Vec<(&str, Value)> = properties.into_iter().collect();
        for (key, value) in &props {
            self.check_property_size(key, value)?;
        }
        self.with_lpg_auto_commit(|| {
            self.create_edge_on_admitted_writer(src, dst, edge_type, &props)
        })
    }

    /// The caller retains Session framing and write authority for this complete
    /// construction. The selected writer owns WAL; CDC receives one final image.
    #[cfg(feature = "lpg")]
    fn create_edge_on_admitted_writer(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: &[(&str, Value)],
    ) -> Result<grafeo_common::types::EdgeId> {
        use grafeo_common::utils::error::Error;
        let (epoch, transaction_id) = self.get_transaction_context();
        let transaction_id = transaction_id.ok_or_else(|| {
            Error::Internal("compound edge creation escaped transaction framing".to_string())
        })?;
        let graph_path = self.active_graph_storage_key();
        let writer = self.compound_write_store_for_graph_storage_key(&graph_path)?;
        let construct = |store: &dyn GraphStoreMut| {
            // Check both endpoints before the LayeredStore can promote either.
            // This is transaction visibility, not presence in the hot overlay.
            for endpoint in [src, dst] {
                if !store.is_node_visible_versioned(endpoint, epoch, transaction_id) {
                    return Err(Error::NodeNotFound(endpoint));
                }
            }
            let id = store.create_edge_versioned(src, dst, edge_type, epoch, transaction_id);
            if !id.is_valid() {
                return Err(Error::Internal(
                    "admitted graph writer rejected edge creation".to_string(),
                ));
            }
            self.check_not_poisoned()?;
            for (key, value) in properties {
                store.set_edge_property_buffered(id, key, value.clone(), transaction_id);
                self.check_not_poisoned()?;
            }
            Ok(id)
        };
        match writer {
            ResolvedLpgWriter::Plain(store) => construct(store.as_ref()),
            #[cfg(feature = "cdc")]
            ResolvedLpgWriter::Cdc(store) => {
                store.create_edge_compound(src, dst, edge_type, properties, construct)
            }
        }
    }

    /// Sets a node property within the active transaction context.
    ///
    /// # Errors
    ///
    /// Returns an error if the node is not visible to this transaction, the
    /// value or post-image violates a configured limit or schema, or write
    /// admission or durability fails.
    #[cfg(feature = "lpg")]
    pub fn set_node_property(&self, id: NodeId, key: &str, value: Value) -> Result<()> {
        self.check_property_size(key, &value)?;
        self.with_lpg_auto_commit(|| {
            let (epoch, transaction_id) = self.get_transaction_context();
            let (labels, mut properties) = self
                .direct_node_image(id, epoch, transaction_id)
                .ok_or(grafeo_common::utils::error::Error::NodeNotFound(id))?;
            if let Some((_, existing)) = properties.iter_mut().find(|(property, _)| property == key)
            {
                *existing = value.clone();
            } else {
                properties.push((key.to_string(), value.clone()));
            }
            self.direct_mutation_validator()
                .validate_node_post_image(Some(id), &labels, &properties, epoch, transaction_id)
                .map_err(|error| {
                    grafeo_common::utils::error::Error::Query(
                        grafeo_common::utils::error::QueryError::new(
                            grafeo_common::utils::error::QueryErrorKind::Semantic,
                            format!("Direct node property mutation rejected: {error}"),
                        ),
                    )
                })?;
            {
                // A Layered writer must not silently skip a snapshot-visible
                // entity deleted since BEGIN. Keep the current-target check
                // and decorated buffering at one publication cut; release it
                // before the lifecycle/poison check and automatic commit.
                let _publication = self.transaction_manager.publication().read();
                self.with_write_store(|w| {
                    if !w.is_node_visible_versioned(
                        id,
                        self.transaction_manager.current_epoch(),
                        transaction_id.unwrap_or(TransactionId::SYSTEM),
                    ) {
                        return Err(grafeo_common::utils::error::Error::Transaction(
                            grafeo_common::utils::error::TransactionError::WriteConflict(format!(
                                "node {id} was deleted after the transaction snapshot"
                            )),
                        ));
                    }
                    if let Some(tid) = transaction_id {
                        w.set_node_property_buffered(id, key, value, tid);
                    } else {
                        w.set_node_property(id, key, value);
                    }
                    Ok(())
                })?;
            }
            self.check_not_poisoned()
        })
    }

    /// Sets an edge property within the active transaction context.
    ///
    /// # Errors
    ///
    /// Returns an error if the edge is not visible to this transaction, the
    /// value exceeds `max_property_size`, or write admission or durability fails.
    #[cfg(feature = "lpg")]
    pub fn set_edge_property(
        &self,
        id: grafeo_common::types::EdgeId,
        key: &str,
        value: Value,
    ) -> Result<()> {
        self.check_property_size(key, &value)?;
        self.with_lpg_auto_commit(|| {
            let (epoch, transaction_id) = self.get_transaction_context();
            if !self.active_read_store().is_edge_visible_versioned(
                id,
                epoch,
                transaction_id.unwrap_or(TransactionId::SYSTEM),
            ) {
                return Err(grafeo_common::utils::error::Error::EdgeNotFound(id));
            }
            {
                // Match the node setter's stable current-target admission.
                let _publication = self.transaction_manager.publication().read();
                self.with_write_store(|w| {
                    if !w.is_edge_visible_versioned(
                        id,
                        self.transaction_manager.current_epoch(),
                        transaction_id.unwrap_or(TransactionId::SYSTEM),
                    ) {
                        return Err(grafeo_common::utils::error::Error::Transaction(
                            grafeo_common::utils::error::TransactionError::WriteConflict(format!(
                                "edge {id} was deleted after the transaction snapshot"
                            )),
                        ));
                    }
                    if let Some(tid) = transaction_id {
                        w.set_edge_property_buffered(id, key, value, tid);
                    } else {
                        w.set_edge_property(id, key, value);
                    }
                    Ok(())
                })?;
            }
            self.check_not_poisoned()
        })
    }

    /// Removes a node property within the active transaction context.
    #[cfg(feature = "lpg")]
    pub fn remove_node_property(&self, id: NodeId, key: &str) -> bool {
        self.with_lpg_auto_commit(|| {
            let (epoch, transaction_id) = self.get_transaction_context();
            let mut existed = false;
            if let Some((labels, mut properties)) =
                self.direct_node_image(id, epoch, transaction_id)
            {
                let before = properties.len();
                properties.retain(|(property, _)| property != key);
                existed = properties.len() != before;
                self.direct_mutation_validator()
                    .validate_node_post_image(Some(id), &labels, &properties, epoch, transaction_id)
                    .map_err(|error| {
                        grafeo_common::utils::error::Error::Query(
                            grafeo_common::utils::error::QueryError::new(
                                grafeo_common::utils::error::QueryErrorKind::Semantic,
                                format!("Direct node property removal rejected: {error}"),
                            ),
                        )
                    })?;
            }
            self.with_write_store(|w| {
                if let Some(tid) = transaction_id {
                    w.remove_node_property_buffered(id, key, tid);
                } else {
                    let _ = w.remove_node_property(id, key);
                }
            });
            self.check_not_poisoned()?;
            Ok(existed)
        })
        .unwrap_or(false)
    }

    /// Removes an edge property within the active transaction context.
    #[cfg(feature = "lpg")]
    pub fn remove_edge_property(&self, id: EdgeId, key: &str) -> bool {
        self.with_lpg_auto_commit(|| {
            let existed = self.get_edge(id).is_some_and(|edge| {
                edge.properties
                    .contains_key(&grafeo_common::types::PropertyKey::new(key))
            });
            let (_, transaction_id) = self.get_transaction_context();
            self.with_write_store(|w| {
                if let Some(tid) = transaction_id {
                    w.remove_edge_property_buffered(id, key, tid);
                } else {
                    let _ = w.remove_edge_property(id, key);
                }
            });
            self.check_not_poisoned()?;
            Ok(existed)
        })
        .unwrap_or(false)
    }

    /// Adds a label within the active transaction context.
    #[cfg(feature = "lpg")]
    pub fn add_node_label(&self, id: NodeId, label: &str) -> bool {
        self.with_lpg_auto_commit(|| {
            let (epoch, transaction_id) = self.get_transaction_context();
            let mut changed = false;
            if let Some((mut labels, properties)) =
                self.direct_node_image(id, epoch, transaction_id)
            {
                changed = !labels.iter().any(|existing| existing == label);
                if changed {
                    labels.push(label.to_string());
                }
                let validator = self.direct_mutation_validator();
                validator
                    .validate_node_labels_allowed(&labels)
                    .and_then(|()| {
                        validator.validate_node_post_image(
                            Some(id),
                            &labels,
                            &properties,
                            epoch,
                            transaction_id,
                        )
                    })
                    .map_err(|error| {
                        grafeo_common::utils::error::Error::Query(
                            grafeo_common::utils::error::QueryError::new(
                                grafeo_common::utils::error::QueryErrorKind::Semantic,
                                format!("Direct node label mutation rejected: {error}"),
                            ),
                        )
                    })?;
            }
            self.with_write_store(|w| {
                if let Some(tid) = transaction_id {
                    w.add_label_buffered(id, label, tid);
                } else {
                    let _ = w.add_label(id, label);
                }
            });
            self.check_not_poisoned()?;
            Ok(changed)
        })
        .unwrap_or(false)
    }

    /// Removes a label within the active transaction context.
    #[cfg(feature = "lpg")]
    pub fn remove_node_label(&self, id: NodeId, label: &str) -> bool {
        self.with_lpg_auto_commit(|| {
            let (epoch, transaction_id) = self.get_transaction_context();
            let mut changed = false;
            if let Some((mut labels, properties)) =
                self.direct_node_image(id, epoch, transaction_id)
            {
                let before = labels.len();
                labels.retain(|existing| existing != label);
                changed = labels.len() != before;
                let validator = self.direct_mutation_validator();
                validator
                    .validate_node_labels_allowed(&labels)
                    .and_then(|()| {
                        validator.validate_node_post_image(
                            Some(id),
                            &labels,
                            &properties,
                            epoch,
                            transaction_id,
                        )
                    })
                    .map_err(|error| {
                        grafeo_common::utils::error::Error::Query(
                            grafeo_common::utils::error::QueryError::new(
                                grafeo_common::utils::error::QueryErrorKind::Semantic,
                                format!("Direct node label removal rejected: {error}"),
                            ),
                        )
                    })?;
            }
            self.with_write_store(|w| {
                if let Some(tid) = transaction_id {
                    w.remove_label_buffered(id, label, tid);
                } else {
                    let _ = w.remove_label(id, label);
                }
            });
            self.check_not_poisoned()?;
            Ok(changed)
        })
        .unwrap_or(false)
    }

    /// Atomically deletes a node and every incident edge within the active
    /// transaction context.
    ///
    /// Incoming, outgoing, and self-loop edges are staged through the same
    /// versioned WAL/CDC path as the node tombstone. If an error occurs after
    /// staging begins, the active transaction (or nested transaction scope) is
    /// rolled back so a caller can never commit a proper prefix of the detach.
    #[cfg(feature = "lpg")]
    pub fn delete_node(&self, id: NodeId) -> bool {
        self.with_lpg_auto_commit(|| {
            // A post-mutation rollback must not be rejected by an outstanding
            // result stream. Reject before the first tombstone instead.
            self.check_no_active_streams("directly delete a node")?;
            let (epoch, transaction_id) = self.get_transaction_context();
            let transaction_id = transaction_id.ok_or_else(|| {
                grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::InvalidState(
                        "direct DETACH DELETE requires a framed transaction".to_string(),
                    ),
                )
            })?;
            let read_store = self.active_read_store();
            if read_store
                .get_node_versioned(id, epoch, transaction_id)
                .is_none()
            {
                return Ok(false);
            }
            self.direct_mutation_validator()
                .validate_node_delete(id, epoch, Some(transaction_id))
                .map_err(|error| {
                    grafeo_common::utils::error::Error::Query(
                        grafeo_common::utils::error::QueryError::new(
                            grafeo_common::utils::error::QueryErrorKind::Semantic,
                            format!("Direct node deletion rejected: {error}"),
                        ),
                    )
                })?;

            // The normal backward index makes this O(degree). Configurations
            // that intentionally omit it take the complete outgoing-scan
            // fallback so DETACH semantics never lose incoming edges.
            let incident_edges = Self::incident_edge_ids_versioned(
                read_store.as_ref(),
                id,
                epoch,
                transaction_id,
                self.active_lpg_store().has_backward_adjacency(),
            );

            let detach = self.with_write_store(|write_store| -> Result<()> {
                for edge_id in incident_edges {
                    if !write_store.delete_edge_versioned(edge_id, epoch, transaction_id) {
                        return Err(grafeo_common::utils::error::Error::Transaction(
                            grafeo_common::utils::error::TransactionError::WriteConflict(format!(
                                "incident edge {edge_id} changed during DETACH DELETE of node {id}"
                            )),
                        ));
                    }
                    // GraphStoreMut cannot return WAL errors directly. Its WAL
                    // decorator sets the shared poison flag, so check after
                    // every mutation; the error path below rolls the whole
                    // pending detach back before returning `false`.
                    self.check_not_poisoned()?;
                }

                if !write_store.delete_node_versioned(id, epoch, transaction_id) {
                    return Err(grafeo_common::utils::error::Error::Transaction(
                        grafeo_common::utils::error::TransactionError::WriteConflict(format!(
                            "node {id} changed during DETACH DELETE"
                        )),
                    ));
                }
                self.check_not_poisoned()
            });
            if let Err(error) = detach {
                // `with_lpg_auto_commit` owns rollback only when it began the
                // transaction itself. Roll back here while the failing
                // transaction is still current so a caller-owned transaction
                // cannot later commit an already-deleted prefix of incident
                // edges. In-memory cleanup precedes the WAL abort marker, so
                // even an abort-log failure cannot expose a partial live
                // post-image.
                let _ = self.rollback_inner();
                return Err(error);
            }
            Ok(true)
        })
        .unwrap_or(false)
    }

    /// Deletes an edge within the active transaction context.
    #[cfg(feature = "lpg")]
    pub fn delete_edge(&self, id: grafeo_common::types::EdgeId) -> bool {
        self.with_lpg_auto_commit(|| {
            let (epoch, transaction_id) = self.get_transaction_context();
            let deleted = self.with_write_store(|w| {
                if let Some(tid) = transaction_id {
                    w.delete_edge_versioned(id, epoch, tid)
                } else {
                    w.delete_edge(id)
                }
            });
            self.check_not_poisoned()?;
            Ok(deleted)
        })
        .unwrap_or(false)
    }

    // =========================================================================
    // Direct Lookup APIs (bypass query planning for O(1) point reads)
    // =========================================================================

    /// Gets a node by ID directly, bypassing query planning.
    ///
    /// This is the fastest way to retrieve a single node when you know its ID.
    /// Skips parsing, binding, optimization, and physical planning entirely.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(1) average case
    /// - No lock contention (uses DashMap internally)
    /// - ~20-30x faster than equivalent MATCH query
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use grafeo_engine::GrafeoDB;
    /// # let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    /// let node_id = session.create_node(&["Person"]);
    ///
    /// // Direct lookup - O(1), no query planning
    /// let node = session.get_node(node_id);
    /// assert!(node.is_some());
    /// ```
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        let _operation = self.session_operation_guard();
        let _publication = self.publication_read_guard();
        let (epoch, transaction_id) = self.get_transaction_context();
        let mut node = self.active_read_store().get_node_versioned(
            id,
            epoch,
            transaction_id.unwrap_or(TransactionId::SYSTEM),
        )?;
        // Read-your-writes: the committed materialization above does not include
        // this transaction's pending buffered writes, so overlay them.
        if let Some(tx) = transaction_id {
            self.active_lpg_store().apply_node_tx_delta(&mut node, tx);
        }
        Some(node)
    }

    /// Gets a single property from a node by ID, bypassing query planning.
    ///
    /// More efficient than `get_node()` when you only need one property,
    /// as it avoids loading the full node with all properties.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(1) average case
    /// - No query planning overhead
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use grafeo_engine::GrafeoDB;
    /// # use grafeo_common::types::Value;
    /// # let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    /// let id = session.create_node_with_props(&["Person"], [("name", "Alix".into())]).unwrap();
    ///
    /// // Direct property access - O(1)
    /// let name = session.get_node_property(id, "name");
    /// assert_eq!(name, Some(Value::String("Alix".into())));
    /// ```
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        self.get_node(id)
            .and_then(|node| node.get_property(key).cloned())
    }

    /// Gets an edge by ID directly, bypassing query planning.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(1) average case
    /// - No lock contention
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        let _operation = self.session_operation_guard();
        let _publication = self.publication_read_guard();
        let (epoch, transaction_id) = self.get_transaction_context();
        let active = self.active_read_store();
        let mut edge = active.get_edge_versioned(
            id,
            epoch,
            transaction_id.unwrap_or(TransactionId::SYSTEM),
        )?;
        // Read-your-writes: overlay this transaction's pending edge-property delta.
        if let Some(tx) = transaction_id {
            self.active_lpg_store().apply_edge_tx_delta(&mut edge, tx);
        }
        Some(edge)
    }

    /// Gets outgoing neighbors of a node directly, bypassing query planning.
    ///
    /// Returns (neighbor_id, edge_id) pairs for all outgoing edges.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(degree) where degree is the number of outgoing edges
    /// - Uses adjacency index for direct access
    /// - ~10-20x faster than equivalent MATCH query
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use grafeo_engine::GrafeoDB;
    /// # let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    /// let alix = session.create_node(&["Person"]);
    /// let gus = session.create_node(&["Person"]);
    /// session.create_edge(alix, gus, "KNOWS");
    ///
    /// // Direct neighbor lookup - O(degree)
    /// let neighbors = session.get_neighbors_outgoing(alix);
    /// assert_eq!(neighbors.len(), 1);
    /// assert_eq!(neighbors[0].0, gus);
    /// ```
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_neighbors_outgoing(&self, node: NodeId) -> Vec<(NodeId, EdgeId)> {
        let _operation = self.session_operation_guard();
        let _publication = self.publication_read_guard();
        let (epoch, transaction_id) = self.get_transaction_context();
        self.active_read_store().edges_from_versioned(
            node,
            Direction::Outgoing,
            epoch,
            transaction_id.unwrap_or(TransactionId::SYSTEM),
        )
    }

    /// Gets incoming neighbors of a node directly, bypassing query planning.
    ///
    /// Returns (neighbor_id, edge_id) pairs for all incoming edges.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(degree) where degree is the number of incoming edges
    /// - Uses backward adjacency index for direct access
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_neighbors_incoming(&self, node: NodeId) -> Vec<(NodeId, EdgeId)> {
        let _operation = self.session_operation_guard();
        let _publication = self.publication_read_guard();
        let (epoch, transaction_id) = self.get_transaction_context();
        self.active_read_store().edges_from_versioned(
            node,
            Direction::Incoming,
            epoch,
            transaction_id.unwrap_or(TransactionId::SYSTEM),
        )
    }

    /// Gets outgoing neighbors filtered by edge type, bypassing query planning.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use grafeo_engine::GrafeoDB;
    /// # let db = GrafeoDB::new_in_memory();
    /// # let session = db.session();
    /// # let alix = session.create_node(&["Person"]);
    /// let neighbors = session.get_neighbors_outgoing_by_type(alix, "KNOWS");
    /// ```
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_neighbors_outgoing_by_type(
        &self,
        node: NodeId,
        edge_type: &str,
    ) -> Vec<(NodeId, EdgeId)> {
        let _operation = self.session_operation_guard();
        let _publication = self.publication_read_guard();
        let (epoch, transaction_id) = self.get_transaction_context();
        let transaction = transaction_id.unwrap_or(TransactionId::SYSTEM);
        let active = self.active_read_store();
        let overlay = transaction_id.map(|_| self.active_lpg_store());

        active
            .edges_from_versioned(node, Direction::Outgoing, epoch, transaction)
            .into_iter()
            .filter(|(_, edge_id)| {
                let Some(mut edge) = active.get_edge_versioned(*edge_id, epoch, transaction) else {
                    return false;
                };
                if let (Some(store), Some(tid)) = (&overlay, transaction_id) {
                    store.apply_edge_tx_delta(&mut edge, tid);
                }
                edge.edge_type.as_str() == edge_type
            })
            .collect()
    }

    /// Checks if a node exists, bypassing query planning.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(1)
    /// - Fastest existence check available
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn node_exists(&self, id: NodeId) -> bool {
        self.get_node(id).is_some()
    }

    /// Checks if an edge exists, bypassing query planning.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn edge_exists(&self, id: EdgeId) -> bool {
        self.get_edge(id).is_some()
    }

    /// Gets the degree (number of edges) of a node.
    ///
    /// Returns (outgoing_degree, incoming_degree).
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_degree(&self, node: NodeId) -> (usize, usize) {
        let _operation = self.session_operation_guard();
        let _publication = self.publication_read_guard();
        let (epoch, transaction_id) = self.get_transaction_context();
        let transaction = transaction_id.unwrap_or(TransactionId::SYSTEM);
        let active = self.active_read_store();
        let out = active
            .edges_from_versioned(node, Direction::Outgoing, epoch, transaction)
            .len();
        let in_degree = active
            .edges_from_versioned(node, Direction::Incoming, epoch, transaction)
            .len();
        (out, in_degree)
    }

    /// Batch lookup of multiple nodes by ID.
    ///
    /// More efficient than calling `get_node()` in a loop because it
    /// amortizes overhead.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(n) where n is the number of IDs
    /// - Better cache utilization than individual lookups
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_nodes_batch(&self, ids: &[NodeId]) -> Vec<Option<Node>> {
        let _operation = self.session_operation_guard();
        let _publication = self.publication_read_guard();
        let (epoch, transaction_id) = self.get_transaction_context();
        let tx = transaction_id.unwrap_or(TransactionId::SYSTEM);
        let active = self.active_read_store();
        // Read-your-writes: overlay the pending delta when in a transaction.
        let overlay = transaction_id.map(|_| self.active_lpg_store());
        ids.iter()
            .map(|&id| {
                let mut node = active.get_node_versioned(id, epoch, tx)?;
                if let (Some(ov), Some(t)) = (&overlay, transaction_id) {
                    ov.apply_node_tx_delta(&mut node, t);
                }
                Some(node)
            })
            .collect()
    }

    // ── Change Data Capture ─────────────────────────────────────────────

    /// Reads a bounded retained feed page using this Session's read grants.
    ///
    /// Both limits are required and positive. `max_events` bounds inspected
    /// rows, so denied graphs cannot cause an unbounded scan. Hidden rows are
    /// neither cloned nor charged against the serialized event byte budget.
    /// An empty filtered page may advance `next`; an unchanged cursor means EOF.
    /// Canonical cursor positions describe the shared store feed, not a private
    /// per-identity sequence. The cursor is not an authorization credential.
    /// Future capture may be disabled without losing retained read authority.
    ///
    /// # Errors
    /// Returns lifecycle, durability, permission, cursor, or page-limit errors.
    #[cfg(feature = "cdc")]
    pub fn changes_after(
        &self,
        cursor: Option<&grafeo_common::types::DurableCursor>,
        max_events: usize,
        max_bytes: usize,
    ) -> Result<crate::cdc::ChangePage> {
        self.check_not_poisoned()?;
        self.require_permission(crate::auth::StatementKind::Read)?;
        let _publication = self.publication_read_guard();
        self.check_durability_not_poisoned()?;
        self.cdc_log.page(
            (
                self.world_identity.read().store_id(),
                grafeo_common::types::FeedId::new(self.graph_model.as_u8() + 1, 0)?,
                self.transaction_manager.current_epoch(),
            ),
            None,
            cursor,
            max_events,
            max_bytes,
            |event| self.cdc_event_is_visible_to_identity(event),
        )
    }

    /// Reads bounded indexed history using the shared durable feed cursor.
    ///
    /// The row limit caps inspected entity candidates, including hidden or
    /// filtered rows. Epoch/graph predicates run before payload accounting.
    /// An exhausted entity index may advance an empty page to the shared tail;
    /// an unchanged cursor marks EOF. Widening the query requires an earlier
    /// cursor or `None`. Both limits are positive and required.
    ///
    /// # Errors
    /// Returns lifecycle, durability, cursor, permission or resource errors.
    #[cfg(feature = "cdc")]
    pub fn history_after(
        &self,
        query: &crate::cdc::EntityHistoryQuery,
        cursor: Option<&grafeo_common::types::DurableCursor>,
        max_events: usize,
        max_bytes: usize,
    ) -> Result<crate::cdc::ChangePage> {
        self.check_not_poisoned()?;
        self.require_permission(crate::auth::StatementKind::Read)?;
        match &query.graph {
            crate::cdc::HistoryGraph::All => {}
            crate::cdc::HistoryGraph::Lpg(path) => {
                self.require_graph_path_grant(path, crate::auth::Role::ReadOnly)?;
            }
            crate::cdc::HistoryGraph::Rdf(name) => {
                self.require_rdf_graph_grant(name.as_deref(), crate::auth::Role::ReadOnly, "read")?;
            }
        }
        let _publication = self.publication_read_guard();
        self.check_durability_not_poisoned()?;
        self.cdc_log.page(
            (
                self.world_identity.read().store_id(),
                grafeo_common::types::FeedId::new(self.graph_model.as_u8() + 1, 0)?,
                self.transaction_manager.current_epoch(),
            ),
            Some(query.entity_id),
            cursor,
            max_events,
            max_bytes,
            |event| self.cdc_event_is_visible_to_identity(event) && query.matches(event),
        )
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Auto-rollback any active transaction to prevent leaked MVCC state,
        // dangling write locks, and uncommitted versions lingering in the
        // store. Drop is a scope abort, not a semantic nested rollback: force
        // the outermost frame so one rollback discards every nested level.
        #[cfg(any(feature = "lpg", feature = "triple-store"))]
        if self.in_transaction() {
            let _ = self.rollback_entire_transaction();
        }

        #[cfg(feature = "metrics")]
        if let Some(ref reg) = self.metrics {
            reg.session_active
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }

        self.active_sessions
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    #[cfg(all(
        feature = "lpg",
        feature = "gql",
        feature = "wal",
        feature = "testing-statement-injection",
        feature = "testing-crash-injection"
    ))]
    #[test]
    fn primary_statement_error_survives_cancel_and_failed_full_rollback() {
        use super::QueryCancellationTestPhase;
        use grafeo_common::testing::wal_failure::{
            disable_abort_log_failure, enable_abort_log_failure_once,
        };
        use grafeo_common::utils::error::Error;

        let directory = tempfile::tempdir().unwrap();
        let db = crate::GrafeoDB::with_config(crate::Config::persistent(
            directory.path().join("cleanup.grafeo"),
        ))
        .unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute("INSERT (:KeptBeforeFailure)").unwrap();
        let control = grafeo_core::execution::QueryExecutionControl::new();
        let handle = control.cancellation_handle();
        *session.active_execution_control.lock() = Some(control);
        session
            .active_execution_completed
            .store(false, std::sync::atomic::Ordering::Release);
        let reached = std::sync::Arc::new(std::sync::Barrier::new(2));
        let release = std::sync::Arc::new(std::sync::Barrier::new(2));
        session.set_query_cancellation_test_hook(
            QueryCancellationTestPhase::BeforeStatementCompletion,
            std::sync::Arc::clone(&reached),
            std::sync::Arc::clone(&release),
        );
        let canceller = std::thread::spawn(move || {
            reached.wait();
            handle.cancel();
            release.wait();
        });
        let result = session.with_auto_commit_guarded(true, || {
            session.execute("INSERT (:FailedStatementResidue)")?;
            // Reuse the existing savepoint-loss qualification seam, then the
            // existing WAL abort injector to fail its full-rollback fallback.
            session.savepoints.lock().clear();
            enable_abort_log_failure_once();
            Err(Error::Internal(
                "primary statement failure before cancellation".into(),
            ))
        });
        disable_abort_log_failure();
        canceller.join().unwrap();
        session.active_execution_control.lock().take();
        let Error::Context { source, context } = result.unwrap_err() else {
            panic!("primary error must retain attached cleanup diagnostics");
        };
        assert!(
            matches!(*source, Error::Internal(ref message) if message == "primary statement failure before cancellation")
        );
        assert!(context.contains("savepoint cleanup failed"), "{context}");
        assert!(context.contains("full rollback also failed"), "{context}");
        assert!(db.is_durability_poisoned());
        assert!(session.execute("RETURN 1").is_err(), "poison rejects reuse");
        assert_eq!(
            session
                .active_execution_statement_depth
                .load(std::sync::atomic::Ordering::Acquire),
            0
        );
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn nested_controlled_statement_cancellation_rewinds_only_outer_statement() {
        let db = crate::GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Owned {name: 'prior'})").unwrap();
        let control = grafeo_core::execution::QueryExecutionControl::new();
        let handle = control.cancellation_handle();
        *session.active_execution_control.lock() = Some(control);
        session
            .active_execution_completed
            .store(false, std::sync::atomic::Ordering::Release);
        let result = session.with_auto_commit_guarded(true, || {
            session.execute("INSERT (:Owned {name: 'nested'})")?;
            assert!(
                !session
                    .active_execution_completed
                    .load(std::sync::atomic::Ordering::Acquire),
                "a nested statement must not finish the outer owner's control"
            );
            handle.cancel();
            Ok(crate::database::QueryResult::empty())
        });
        assert!(
            matches!(result, Err(grafeo_common::utils::error::Error::Query(ref error))
            if error.kind == grafeo_common::utils::error::QueryErrorKind::Cancelled)
        );
        session.active_execution_control.lock().take();
        assert_eq!(
            session
                .active_execution_statement_depth
                .load(std::sync::atomic::Ordering::Acquire),
            0
        );
        let rows = session.execute("MATCH (n:Owned) RETURN n.name").unwrap();
        assert_eq!(
            rows.rows(),
            vec![vec![grafeo_common::types::Value::from("prior")]]
        );
        session.commit().unwrap();
        assert_eq!(db.node_count(), 1);
    }

    #[cfg(feature = "lpg")]
    thread_local! {
        pub(super) static INDEX_OWNER_SCANS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn nested_projection_cascade_preserves_literal_path_and_rejects_stale_source()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use grafeo_common::types::GraphPath;

        let db = GrafeoDB::new_in_memory();
        assert!(db.create_graph("parent")?);
        let parent = crate::database::testing::root_lpg_store(&db)
            .graph("parent")
            .ok_or("missing parent graph")?;
        assert!(
            db.transaction_manager
                .with_write_authority(|| parent.create_graph("leaf"))?
        );
        assert!(db.create_graph("parent/leaf")?);
        let nested = GraphPath::from_components(&["parent", "leaf"])?;
        let literal = GraphPath::from_components(&["parent/leaf"])?;

        let seed = db.session();
        seed.use_graph_path(&nested)?;
        seed.execute("INSERT (:Person {name: 'nested'})")?;
        seed.execute("CREATE PROJECTION nested_people LABELS (Person)")?;
        let held = db
            .projection("nested_people")
            .ok_or("missing nested projection")?;
        assert_eq!(held.node_count(), 1);
        seed.use_graph_path(&literal)?;
        seed.execute("INSERT (:Person {name: 'literal'})")?;
        seed.execute("CREATE PROJECTION literal_people LABELS (Person)")?;

        let stale = db.session();
        stale.use_graph_path(&nested)?;
        stale.execute("START TRANSACTION")?;
        stale.execute("CREATE PROJECTION stale_people LABELS (Person)")?;

        let dropper = db.session();
        dropper.execute("DROP GRAPH parent")?;
        assert!(db.projection("nested_people").is_none());
        assert_eq!(held.node_count(), 1);
        let literal_view = db
            .projection("literal_people")
            .ok_or("literal projection was cascaded")?;
        assert_eq!(literal_view.node_count(), 1);

        assert!(db.create_graph("parent")?);
        let replacement = crate::database::testing::root_lpg_store(&db)
            .graph("parent")
            .ok_or("missing replacement parent")?;
        assert!(
            db.transaction_manager
                .with_write_authority(|| replacement.create_graph("leaf"))?
        );
        assert!(stale.execute("COMMIT").is_err());
        assert!(db.projection("stale_people").is_none());
        assert!(db.projection("literal_people").is_some());
        Ok(())
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn native_constraint_schema_comes_from_path_at_statement_and_commit()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use grafeo_common::types::GraphPath;

        let db = GrafeoDB::new_in_memory();
        let admin = db.session();
        admin.execute("CREATE NODE TYPE Item (root_required INTEGER NOT NULL)")?;
        admin.execute("CREATE SCHEMA Scoped")?;
        admin.execute("SESSION SET SCHEMA Scoped")?;
        admin.execute("CREATE NODE TYPE Item (schema_required INTEGER NOT NULL)")?;
        assert!(db.create_graph("Scoped/flat")?);
        db.transaction_manager.with_write_authority(
            || -> std::result::Result<(), Box<dyn std::error::Error>> {
                assert!(crate::database::testing::root_lpg_store(&db).create_graph("a")?);
                assert!(
                    crate::database::testing::root_lpg_store(&db)
                        .graph("a")
                        .ok_or("missing parent")?
                        .create_graph("b")?
                );
                Ok(())
            },
        )?;
        let cases = [
            (
                GraphPath::root(),
                Some("Scoped"),
                "root_required",
                "schema_required",
            ),
            (
                GraphPath::from_components(&["a", "b"])?,
                Some("Scoped"),
                "root_required",
                "schema_required",
            ),
            (
                GraphPath::from_components(&["Scoped/flat"])?,
                None,
                "schema_required",
                "root_required",
            ),
        ];
        for (path, schema, required, unrelated) in cases {
            let mut session = db.session();
            if let Some(schema) = schema {
                session.set_schema(schema)?;
            }
            session.use_graph_path(&path)?;
            session.begin_transaction()?;
            assert!(
                session
                    .create_node_with_props(&["Item"], [(unrelated, Value::Int64(1))])
                    .is_err(),
                "direct statement must enforce {required} at {path:?}"
            );
            assert!(
                session
                    .execute(&format!("INSERT (:Item {{{unrelated}: 1}})"))
                    .is_err(),
                "planned statement must enforce {required} at {path:?}"
            );
            let node = session.create_node_with_props(&["Item"], [(required, Value::Int64(2))])?;
            session.execute(&format!("INSERT (:Item {{{required}: 3}})"))?;
            session.commit()?;
            assert_eq!(
                session.get_node_property(node, required),
                Some(Value::Int64(2))
            );
            assert_eq!(session.current_schema().as_deref(), schema);
            assert_eq!(session.current_graph_path(), path);
        }
        Ok(())
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn constraint_creation_scans_nested_rows_without_flattening_schema_ownership()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use grafeo_common::types::GraphPath;

        let db = GrafeoDB::new_in_memory();
        let admin = db.session();
        admin.execute("CREATE SCHEMA Scoped")?;
        db.transaction_manager.with_write_authority(
            || -> std::result::Result<(), Box<dyn std::error::Error>> {
                assert!(crate::database::testing::root_lpg_store(&db).create_graph("a/b")?);
                assert!(crate::database::testing::root_lpg_store(&db).create_graph("a")?);
                assert!(
                    crate::database::testing::root_lpg_store(&db)
                        .graph("a")
                        .ok_or("missing parent")?
                        .create_graph("b")?
                );
                assert!(
                    crate::database::testing::root_lpg_store(&db)
                        .graph("Scoped/__default__")
                        .ok_or("missing schema partition")?
                        .create_graph("child")?
                );
                Ok(())
            },
        )?;
        let flat = GraphPath::from_components(&["a/b"])?;
        let nested = GraphPath::from_components(&["a", "b"])?;
        let beneath_schema = GraphPath::from_components(&["Scoped/__default__", "child"])?;
        let writer = db.session();
        writer.use_graph_path(&flat)?;
        writer.create_node_with_props(
            &["Existing"],
            [("key", Value::Int64(1)), ("required", Value::Int64(1))],
        )?;
        writer.use_graph_path(&nested)?;
        let nested_node =
            writer.create_node_with_props(&["Existing"], [("key", Value::Int64(1))])?;
        writer.use_graph_path(&beneath_schema)?;
        let beneath_node =
            writer.create_node_with_props(&["Existing"], [("key", Value::Int64(2))])?;

        // A native descendant does not inherit its parent's flat schema name.
        admin.execute("SESSION SET SCHEMA Scoped")?;
        admin.execute(
            "CREATE CONSTRAINT scoped_required FOR (n:Existing) ON (n.required) NOT NULL",
        )?;
        admin.execute("SESSION RESET SCHEMA")?;
        assert!(
            admin
                .execute(
                    "CREATE CONSTRAINT root_required FOR (n:Existing) ON (n.required) NOT NULL"
                )
                .is_err()
        );
        assert!(
            admin
                .execute("CREATE CONSTRAINT root_unique FOR (n:Existing) ON (n.key) UNIQUE")
                .is_err()
        );
        assert_eq!(
            admin.execute("SHOW CONSTRAINTS")?.row_count(),
            0,
            "failed validation must not publish root constraints"
        );

        writer.use_graph_path(&nested)?;
        writer.set_node_property(nested_node, "required", Value::Int64(1))?;
        writer.set_node_property(nested_node, "key", Value::Int64(3))?;
        writer.use_graph_path(&beneath_schema)?;
        writer.set_node_property(beneath_node, "required", Value::Int64(1))?;
        admin
            .execute("CREATE CONSTRAINT root_required FOR (n:Existing) ON (n.required) NOT NULL")?;
        admin.execute("CREATE CONSTRAINT root_unique FOR (n:Existing) ON (n.key) UNIQUE")?;
        assert_eq!(admin.execute("SHOW CONSTRAINTS")?.row_count(), 2);
        Ok(())
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn dropped_ancestor_retires_exact_descendant_state_and_ssi_after_savepoint()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use grafeo_common::types::GraphPath;
        use std::sync::Arc;

        for replace in [false, true] {
            let db = GrafeoDB::new_in_memory();
            let leaf = db.transaction_manager.with_write_authority(|| -> std::result::Result<
                Arc<grafeo_core::graph::lpg::LpgStore>, Box<dyn std::error::Error>,
            > {
                assert!(crate::database::testing::root_lpg_store(&db).create_graph("parent")?);
                let parent = crate::database::testing::root_lpg_store(&db).graph("parent").ok_or("missing parent")?;
                assert!(parent.create_graph("leaf")?);
                parent.graph("leaf").ok_or_else(|| "missing leaf".into())
            })?;
            let path = GraphPath::from_components(&["parent", "leaf"])?;
            let mut session = db.session();
            session.use_graph_path(&path)?;
            let source = session.create_node_with_props(&["Kept"], [("value", Value::Int64(1))])?;
            let target = session.create_node_with_props(&["Kept"], [])?;
            let victim = session.create_node_with_props(&["Victim"], [])?;
            let original_edge = session.create_edge_with_props(source, target, "LINK", [])?;
            let manager_owners = Arc::strong_count(&db.transaction_manager);

            session.begin_transaction_with_isolation(
                crate::transaction::IsolationLevel::Serializable,
            )?;
            let tx = session
                .current_transaction_id()
                .ok_or("missing transaction")?;
            assert!(Arc::strong_count(&db.transaction_manager) >= manager_owners + 2);
            let pending_node =
                session.create_node_with_props(&["Pending"], [("value", Value::Int64(2))])?;
            let pending_edge = session.create_edge_with_props(source, pending_node, "TEMP", [])?;
            session.set_node_property(source, "value", Value::Int64(99))?;
            assert!(session.add_node_label(source, "Temporary"));
            assert!(session.delete_node(victim));
            assert!(session.delete_edge(original_edge));
            session.savepoint("before_drop")?;
            session.use_graph_path(&GraphPath::root())?;
            session.execute("DROP GRAPH parent")?;
            session.rollback_to_savepoint("before_drop")?;
            assert!(leaf.pending_node_creates(tx).contains(&pending_node));
            assert!(leaf.pending_edge_creates(tx).contains(&pending_edge));
            assert!(leaf.pending_node_deletes_peek(tx).contains(&victim));
            assert!(leaf.pending_edge_deletes_peek(tx).contains(&original_edge));

            session.execute("DROP GRAPH parent")?;
            if replace {
                session.execute("CREATE GRAPH parent")?;
                assert!(session.create_graph_path(&path)?);
                session.use_graph_path(&path)?;
                session.create_node_with_props(&["Replacement"], [])?;
            }
            session.commit()?;

            assert!(leaf.pending_node_creates(tx).is_empty());
            assert!(leaf.pending_edge_creates(tx).is_empty());
            assert!(leaf.pending_node_deletes_peek(tx).is_empty());
            assert!(leaf.pending_edge_deletes_peek(tx).is_empty());
            let (nodes, edges) = leaf.overlay_touched_entities(tx);
            assert!(nodes.is_empty() && edges.is_empty());
            assert!(
                leaf.get_node_at_epoch(pending_node, db.current_epoch())
                    .is_none()
            );
            assert!(
                leaf.get_edge_at_epoch(pending_edge, db.current_epoch())
                    .is_none()
            );
            assert!(leaf.get_node_at_epoch(victim, db.current_epoch()).is_some());
            assert!(
                leaf.get_edge_at_epoch(original_edge, db.current_epoch())
                    .is_some()
            );
            let source = leaf
                .get_node_at_epoch(source, db.current_epoch())
                .ok_or("missing retained source")?;
            assert_eq!(source.get_property("value"), Some(&Value::Int64(1)));
            assert!(!source.has_label("Temporary"));
            assert_eq!(Arc::strong_count(&db.transaction_manager), manager_owners);
            if replace {
                let replacement = session
                    .session_graph_path(&path)
                    .ok_or("missing published replacement")?;
                assert!(!Arc::ptr_eq(&replacement, &leaf));
                assert_eq!(replacement.node_count(), 1);
            } else {
                assert!(
                    crate::database::testing::root_lpg_store(&db)
                        .graph("parent")
                        .is_none()
                );
            }
        }
        Ok(())
    }

    #[cfg(all(feature = "lpg", feature = "gql", feature = "wal"))]
    #[test]
    fn nested_wal_mutation_admission_uses_exact_transaction_target()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use grafeo_common::types::GraphPath;
        use grafeo_core::graph::lpg::LpgStore;
        use std::sync::Arc;

        let db = GrafeoDB::new_in_memory();
        let nested = db.transaction_manager.with_write_authority(
            || -> std::result::Result<Arc<LpgStore>, Box<dyn std::error::Error>> {
                assert!(crate::database::testing::root_lpg_store(&db).create_graph("parent")?);
                let parent = crate::database::testing::root_lpg_store(&db)
                    .graph("parent")
                    .ok_or("missing parent")?;
                assert!(parent.create_graph("child")?);
                parent
                    .graph("child")
                    .ok_or_else(|| "missing nested graph".into())
            },
        )?;
        let directory = tempfile::tempdir()?;
        let wal = Arc::new(grafeo_storage::wal::LpgWal::open(
            directory.path().join("wal"),
        )?);
        let mut session = db.session();
        session.set_wal(Arc::clone(&wal));
        let path = GraphPath::from_components(&["parent", "child"])?;
        for explicit in [true, false] {
            session.use_graph_path(&GraphPath::root())?;
            if explicit {
                session.begin_transaction()?;
            }
            session.use_graph_path(&path)?;
            let before = nested.node_count();
            let records = wal.record_count();
            session.create_node_with_props(&["Accepted"], [("value", Value::Int64(1))])?;
            session.execute("INSERT (:Accepted {value: 2})")?;
            assert_eq!(session.current_graph_path(), path);
            assert!(wal.record_count() > records);
            assert_eq!(
                crate::database::testing::root_lpg_store(&db).node_count(),
                0,
                "nested writes cannot target root"
            );
            if explicit {
                session.rollback()?;
                assert_eq!(nested.node_count(), before);
            } else {
                assert_eq!(nested.node_count(), before + 2);
            }
            assert!(!session.in_transaction());
        }
        Ok(())
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn native_graph_paths_isolate_nested_reads_writes_and_rollback()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use grafeo_common::types::GraphPath;

        let db = GrafeoDB::new_in_memory();
        db.transaction_manager.with_write_authority(
            || -> std::result::Result<(), Box<dyn std::error::Error>> {
                for name in ["", "default", "a/b", "a"] {
                    assert!(crate::database::testing::root_lpg_store(&db).create_graph(name)?);
                }
                assert!(
                    crate::database::testing::root_lpg_store(&db)
                        .graph("a")
                        .ok_or("missing parent")?
                        .create_graph("b")?
                );
                Ok(())
            },
        )?;
        let paths = [
            GraphPath::root(),
            GraphPath::from_components(&[""])?,
            GraphPath::from_components(&["default"])?,
            GraphPath::from_components(&["a/b"])?,
            GraphPath::from_components(&["a", "b"])?,
        ];
        let mut session = db.session();
        let mut nodes = Vec::new();
        for (index, path) in paths.iter().enumerate() {
            session.use_graph_path(path)?;
            assert_eq!(session.current_graph_path(), *path);
            let value = Value::String(format!("path-{index}").into());
            let node =
                session.create_node_with_props(&["PathIdentity"], [("value", value.clone())])?;
            assert_eq!(session.get_node_property(node, "value"), Some(value));
            nodes.push(node);
        }
        assert!(
            nodes.windows(2).all(|pair| pair[0] == pair[1]),
            "local IDs must collide across distinct stores"
        );

        session.begin_transaction()?;
        for (path, node) in paths.iter().zip(&nodes) {
            session.use_graph_path(path)?;
            session.set_node_property(*node, "value", Value::String("pending".into()))?;
            assert_eq!(
                session.get_node_property(*node, "value"),
                Some(Value::String("pending".into()))
            );
            assert!(session.touched_graphs.lock().contains(path));
        }
        assert!(session.touched_named_graphs.lock().contains_key(&paths[4]));
        session.rollback()?;
        assert!(session.touched_graphs.lock().is_empty());
        assert!(session.touched_named_graphs.lock().is_empty());
        for (index, (path, node)) in paths.iter().zip(&nodes).enumerate() {
            session.use_graph_path(path)?;
            assert_eq!(
                session.get_node_property(*node, "value"),
                Some(Value::String(format!("path-{index}").into()))
            );
        }
        Ok(())
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn failed_implicit_mutation_on_a_missing_native_path_retires_its_transaction()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use grafeo_common::types::GraphPath;

        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH vanishing")?;
        let path = GraphPath::from_components(&["vanishing"])?;
        session.use_graph_path(&path)?;
        db.session().execute("DROP GRAPH vanishing")?;
        assert!(session.create_node_with_props(&["Missing"], []).is_err());
        assert!(!session.in_transaction());
        assert!(session.touched_graphs.lock().is_empty());
        assert!(session.missing_named_graphs.lock().is_empty());
        assert_eq!(session.current_graph_path(), path);
        assert!(session.execute("INSERT (:Missing)").is_err());
        assert!(!session.in_transaction());
        assert!(session.touched_graphs.lock().is_empty());
        assert!(session.missing_named_graphs.lock().is_empty());
        assert_eq!(session.current_graph_path(), path);
        session.execute("CREATE GRAPH vanishing")?;
        session.create_node_with_props(&["Recreated"], [])?;
        assert!(!session.in_transaction());
        Ok(())
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn historical_session_lifecycle_admission_precedes_transaction()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH preserved")?;
        session.set_viewing_epoch(grafeo_common::types::EpochId::new(0));
        assert!(session.execute("CREATE GRAPH forbidden").is_err());
        assert!(session.execute("DROP GRAPH preserved").is_err());
        assert!(!session.in_transaction());
        assert_eq!(db.list_graphs(), vec!["preserved"]);
        Ok(())
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn catalog_foundation_actual_standalone_races_direct_catalog_admission()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use std::sync::{Arc, mpsc};
        let db = Arc::new(GrafeoDB::new_in_memory());
        let (prepared_tx, prepared_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let preparing = Arc::clone(&db);
        let ddl = std::thread::spawn(move || -> std::result::Result<(), String> {
            crate::catalog::install_preparation_rendezvous(prepared_tx, release_rx)
                .map_err(|error| error.to_string())?;
            preparing
                .session()
                .execute("CREATE SCHEMA prepared")
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        prepared_rx.recv()?;
        assert!(
            db.catalog.writer_is_held_for_test(),
            "actual standalone must retain catalog writer after cloning"
        );
        let (attempt_tx, attempt_rx) = mpsc::channel();
        let direct_catalog = Arc::clone(&db.catalog);
        let direct = std::thread::spawn(move || -> std::result::Result<(), String> {
            attempt_tx.send(()).map_err(|error| error.to_string())?;
            direct_catalog
                .get_or_create_label("direct")
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        attempt_rx.recv()?;
        release_tx.send(())?;
        ddl.join().map_err(|_| "standalone DDL worker panicked")??;
        direct
            .join()
            .map_err(|_| "direct admission worker panicked")??;
        let view = db.catalog.read();
        assert!(view.schema_exists("prepared"));
        assert!(view.get_label_id("direct").is_some());
        assert!(db.session().store.graph("prepared/__default__").is_some());
        Ok(())
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn catalog_foundation_show_indexes_never_mixes_replacement_cuts()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use std::sync::{Arc, Barrier};
        fn candidate(
            name: &str,
        ) -> std::result::Result<crate::catalog::Catalog, crate::catalog::CatalogError> {
            let catalog = crate::catalog::Catalog::new();
            let label = catalog.get_or_create_label(name)?;
            let property = catalog.get_or_create_property_key(name)?;
            catalog.create_index(
                Some(name),
                label,
                property,
                grafeo_common::types::GraphPath::root(),
                crate::catalog::IndexConfiguration::Property,
            )?;
            Ok(catalog)
        }
        let db = Arc::new(GrafeoDB::new_in_memory());
        let mut initial = crate::catalog::CatalogWorkspace::replacement(candidate("A")?);
        db.catalog
            .prepare_replacement(&mut initial)?
            .install()
            .finish();
        let start = Arc::new(Barrier::new(2));
        let writer_db = Arc::clone(&db);
        let writer_start = Arc::clone(&start);
        let writer = std::thread::spawn(
            move || -> std::result::Result<(), crate::catalog::CatalogError> {
                writer_start.wait();
                for iteration in 0..100 {
                    let name = if iteration % 2 == 0 { "A" } else { "B" };
                    let mut workspace =
                        crate::catalog::CatalogWorkspace::replacement(candidate(name)?);
                    writer_db
                        .catalog
                        .prepare_replacement(&mut workspace)?
                        .install()
                        .finish();
                }
                Ok(())
            },
        );
        start.wait();
        let session = db.session();
        for _ in 0..200 {
            let rows = session.execute("SHOW INDEXES")?;
            assert_eq!(rows.rows().len(), 1);
            let row = rows.rows().first().ok_or("missing coherent index row")?;
            let name = row
                .first()
                .and_then(|value| value.as_str())
                .ok_or("missing owner name")?;
            assert!(matches!(name, "A" | "B"));
            assert_eq!(row.get(2).and_then(|value| value.as_str()), Some(name));
            assert_eq!(row.get(3).and_then(|value| value.as_str()), Some(name));
        }
        writer.join().map_err(|_| "replacement worker panicked")??;
        assert!(db.catalog.find_index_by_name("B").is_some());
        Ok(())
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn catalog_foundation_show_current_graph_type_uses_one_cut_and_pending_overlay()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use std::sync::{Arc, Barrier};
        fn candidate(
            name: &str,
        ) -> std::result::Result<crate::catalog::Catalog, crate::catalog::CatalogError> {
            let catalog = crate::catalog::Catalog::new();
            for type_name in [name, "Overlay"] {
                catalog.register_graph_type(crate::catalog::GraphTypeDefinition {
                    name: type_name.to_string(),
                    allowed_node_types: Vec::new(),
                    allowed_edge_types: Vec::new(),
                    open: type_name != "B",
                })?;
            }
            let chosen = grafeo_common::types::GraphPath::from_components(&["chosen"])
                .map_err(|error| crate::catalog::CatalogError::InvalidState(error.to_string()))?;
            catalog.bind_graph_type(&chosen, name.to_string())?;
            Ok(catalog)
        }
        let db = Arc::new(GrafeoDB::new_in_memory());
        let session = db.session();
        session.execute("CREATE GRAPH chosen")?;
        session.execute("SESSION SET GRAPH chosen")?;
        let mut initial = crate::catalog::CatalogWorkspace::replacement(candidate("A")?);
        db.catalog
            .prepare_replacement(&mut initial)?
            .install()
            .finish();
        let start = Arc::new(Barrier::new(2));
        let writer_db = Arc::clone(&db);
        let writer_start = Arc::clone(&start);
        let writer = std::thread::spawn(
            move || -> std::result::Result<(), crate::catalog::CatalogError> {
                writer_start.wait();
                for iteration in 0..100 {
                    let name = if iteration % 2 == 0 { "A" } else { "B" };
                    let mut workspace =
                        crate::catalog::CatalogWorkspace::replacement(candidate(name)?);
                    writer_db
                        .catalog
                        .prepare_replacement(&mut workspace)?
                        .install()
                        .finish();
                }
                Ok(())
            },
        );
        start.wait();
        for _ in 0..200 {
            let result = session.execute("SHOW CURRENT GRAPH TYPE")?;
            let row = result
                .rows()
                .first()
                .ok_or("missing current graph type row")?;
            let name = row
                .get(1)
                .and_then(|value| value.as_str())
                .ok_or("missing bound type")?;
            assert!(matches!(name, "A" | "B"));
            assert_eq!(row.get(2), Some(&Value::from(name == "A")));
        }
        writer
            .join()
            .map_err(|_| "graph type replacement worker panicked")??;
        let chosen = grafeo_common::types::GraphPath::from_components(&["chosen"])?;
        session.stage_graph_type_binding(&chosen, Some("Overlay".to_string()));
        let replacement = session.execute("SHOW CURRENT GRAPH TYPE")?;
        assert_eq!(
            replacement
                .rows()
                .first()
                .and_then(|row| row.get(1))
                .and_then(|value| value.as_str()),
            Some("Overlay")
        );
        session.stage_graph_type_binding(&chosen, None);
        let cleared = session.execute("SHOW CURRENT GRAPH TYPE")?;
        assert_eq!(
            cleared.rows().first().and_then(|row| row.get(1)),
            Some(&Value::Null)
        );
        Ok(())
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn catalog_foundation_anonymous_owners_copy_catalog_but_data_only_commits_do_not()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let copies = crate::catalog::state_copy_count();
        let owner = session.create_index_durable(crate::CreateIndexRequest {
            graph: grafeo_common::types::GraphPath::root(),
            name: None,
            label: None,
            property: "anonymous".into(),
            kind: crate::IndexCreateKind::Property,
        })?;
        assert!(session.store.has_property_index("anonymous"));
        assert!(session.catalog.get_index(owner).is_some());
        assert!(session.drop_index_durable(owner)?);
        assert!(!session.store.has_property_index("anonymous"));
        session.with_lpg_auto_commit(|| Ok(()))?;
        assert_eq!(crate::catalog::state_copy_count(), copies + 2);
        assert_eq!(session.catalog.index_count(), 0);
        assert_eq!(session.catalog.index_allocator_high_water(), 1);
        assert_eq!(session.catalog.label_count(), 1);
        assert_eq!(session.catalog.property_key_count(), 1);
        let label = session.catalog.get_or_create_label("Owned")?;
        let property = session.catalog.get_or_create_property_key("owned")?;
        session.catalog.create_index(
            Some("owner"),
            label,
            property,
            grafeo_common::types::GraphPath::root(),
            crate::catalog::IndexConfiguration::Property,
        )?;
        let scans = INDEX_OWNER_SCANS.with(std::cell::Cell::get);
        session.with_lpg_auto_commit(|| Ok(()))?;
        assert_eq!(INDEX_OWNER_SCANS.with(std::cell::Cell::get), scans);
        assert_eq!(crate::catalog::state_copy_count(), copies + 2);
        assert!(session.catalog.find_index_by_name("owner").is_some());
        Ok(())
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn catalog_foundation_named_index_and_binding_share_one_prepared_successor()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = GrafeoDB::new_in_memory();
        db.catalog
            .register_graph_type(crate::catalog::GraphTypeDefinition {
                name: "BoundType".to_string(),
                allowed_node_types: Vec::new(),
                allowed_edge_types: Vec::new(),
                open: true,
            })?;
        let mut session = db.session();
        session.begin_transaction()?;
        session.execute("CREATE GRAPH bound")?;
        session.execute("CREATE INDEX prepared_owner FOR (n:Person) ON (n.email)")?;
        session.pending_graph_type_bindings.lock().insert(
            grafeo_common::types::GraphPath::from_components(&["bound"])?,
            super::PendingGraphTypeBinding {
                expected: None,
                replacement: Some("BoundType".to_string()),
            },
        );
        let copies = crate::catalog::state_copy_count();
        session.commit()?;
        assert_eq!(crate::catalog::state_copy_count(), copies + 1);
        let catalog = db.catalog.read();
        assert!(catalog.find_index_by_name("prepared_owner").is_some());
        assert_eq!(
            catalog
                .get_graph_type_binding(&grafeo_common::types::GraphPath::from_components(&[
                    "bound"
                ])?)
                .as_deref(),
            Some("BoundType")
        );
        assert!(session.store.has_property_index("email"));
        assert!(session.store.graph("bound").is_some());
        Ok(())
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn catalog_foundation_late_binding_failure_cannot_publish_prepared_index()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.begin_transaction()?;
        session.execute("CREATE INDEX rejected_owner FOR (n:Person) ON (n.email)")?;
        session.pending_graph_type_bindings.lock().insert(
            grafeo_common::types::GraphPath::from_components(&["bound"])?,
            super::PendingGraphTypeBinding {
                expected: None,
                replacement: Some("MissingType".to_string()),
            },
        );
        let before = db.catalog.encode_wal_state_v1()?;
        let epoch = db.transaction_manager.current_epoch();
        let commit_epoch = grafeo_common::types::EpochId::new(epoch.0 + 1);
        let mut capture = super::EngineCommitCapture::default();
        let mut core_workspace = None;
        let mut workspace = crate::catalog::CatalogWorkspace::new();
        let transaction_id = session
            .current_transaction
            .lock()
            .ok_or("missing transaction")?;
        let mut reached_logical = false;
        let result = db.transaction_manager.with_write_authority(|| {
            let _publication = db.transaction_manager.publication().write();
            capture.prepare(
                &session,
                &[grafeo_common::types::GraphPath::root()],
                transaction_id,
                epoch,
                commit_epoch,
            )?;
            let (core, logical) = capture.workspace(transaction_id, epoch, commit_epoch);
            let core = core_workspace.insert(core);
            super::with_prepared_lpg_commit(
                core,
                db.transaction_manager.write_authority(),
                |released| {
                    reached_logical = true;
                    let result = logical.prepare(&session, &mut workspace);
                    drop(released);
                    result.map(drop)
                },
            )
        });
        assert!(
            reached_logical,
            "the real core aggregate must prepare before late catalog rejection"
        );
        assert!(result.is_err());
        drop(result);
        assert_eq!(db.catalog.encode_wal_state_v1()?, before);
        assert_eq!(db.catalog.find_index_by_name("rejected_owner"), None);
        assert!(!session.store.has_property_index("email"));
        assert_eq!(db.transaction_manager.current_epoch(), epoch);
        session.rollback()?;
        Ok(())
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn catalog_foundation_named_drop_rejects_same_name_recreated_owner()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.execute("CREATE INDEX owner FOR (n:Person) ON (n.email)")?;
        session.begin_transaction()?;
        session.execute("DROP INDEX owner")?;
        let old = db
            .catalog
            .find_index_by_name("owner")
            .ok_or("missing owner")?;
        let definition = db.catalog.get_index(old).ok_or("missing definition")?;
        assert!(db.catalog.drop_index(old));
        let replacement = db.catalog.create_index(
            Some("owner"),
            definition.label,
            definition.property_key,
            definition.key.graph().clone(),
            definition.configuration,
        )?;
        assert_ne!(replacement, old);
        assert!(session.commit().is_err());
        assert_eq!(db.catalog.find_index_by_name("owner"), Some(replacement));
        assert!(session.store.has_property_index("email"));
        Ok(())
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn catalog_rejects_shared_property_owner_and_drop_removes_exact_index()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        let before = Value::from("before@example.com");
        let after = Value::from("after@example.com");
        let changed = session.create_node_with_props(&["Person"], [("email", before.clone())])?;
        let untouched = session.create_node_with_props(&["Person"], [("email", before.clone())])?;
        session.execute("CREATE INDEX first_owner FOR (n:Person) ON (n.email)")?;
        let first = db
            .catalog
            .find_index_by_name("first_owner")
            .ok_or("missing first owner")?;
        let definition = db
            .catalog
            .get_index(first)
            .ok_or("missing first definition")?;
        // The catalog allocator itself enforces the same physical bijection.
        assert!(
            db.catalog
                .create_index(
                    Some("retained_owner"),
                    definition.label,
                    definition.property_key,
                    definition.key.graph().clone(),
                    definition.configuration,
                )
                .is_err()
        );
        let observed = session
            .store
            .observe_property_index("email")
            .ok_or("missing physical index")?;
        session.begin_transaction()?;
        session.execute("DROP INDEX first_owner")?;
        session.set_node_property(changed, "email", after.clone())?;
        session.commit()?;

        assert_eq!(db.catalog.find_index_by_name("first_owner"), None);
        assert_eq!(db.catalog.find_index_by_name("retained_owner"), None);
        assert_eq!(db.catalog.index_count(), 0);
        assert!(!session.store.has_property_index("email"));
        assert!(
            db.transaction_manager
                .with_write_authority(|| session.store.validate_index_registration(&observed))
                .is_err()
        );
        assert_eq!(
            session.store.find_nodes_by_property("email", &before),
            vec![untouched]
        );
        assert_eq!(
            session.store.find_nodes_by_property("email", &after),
            vec![changed]
        );
        Ok(())
    }

    #[cfg(all(feature = "wal", feature = "lpg", feature = "gql"))]
    use super::CatalogWalBatchGuard;
    #[cfg(all(feature = "gql", feature = "lpg"))]
    use super::parse_default_literal;
    use crate::database::GrafeoDB;
    #[cfg(feature = "lpg")]
    use grafeo_common::types::Value;

    #[cfg(all(feature = "lpg", feature = "cdc"))]
    #[test]
    fn compact_edge_creation_rejects_replaced_default_writer_without_mutation() {
        use grafeo_core::graph::lpg::LpgStore;
        use std::sync::Arc;
        let db = GrafeoDB::with_config(crate::Config::in_memory().with_cdc()).expect("database");
        let a = db.create_node(&["A"]);
        let b = db.create_node(&["B"]);
        let mut session = db.session();
        let foreign = Arc::new(LpgStore::new().expect("foreign store"));
        let cached = session
            .default_cdc_writer
            .as_ref()
            .expect("cached default CDC writer");
        let erased: Arc<dyn grafeo_core::graph::GraphStoreMut> = cached.clone();
        assert!(Arc::ptr_eq(
            &erased,
            session.graph_store_mut.as_ref().expect("actual writer")
        ));
        assert_eq!(foreign.create_node(&["ForeignA"]), a);
        assert_eq!(foreign.create_node(&["ForeignB"]), b);
        session.graph_store_mut = Some(foreign.clone());
        let events_before = db.memory_usage().cdc.event_count;
        assert!(
            session
                .create_edge_with_props(a, b, "WRONG_TARGET", [])
                .is_err()
        );
        assert_eq!(db.edge_count(), 0);
        assert_eq!(foreign.edge_count(), 0);
        assert_eq!(db.memory_usage().cdc.event_count, events_before);
    }

    #[cfg(all(feature = "lpg", feature = "cdc"))]
    #[test]
    fn compact_edge_creation_rejects_missing_cdc_cache_without_mutation() {
        let db = GrafeoDB::with_config(crate::Config::in_memory().with_cdc()).expect("database");
        let a = db.create_node(&["A"]);
        let b = db.create_node(&["B"]);
        let mut session = db.session();
        assert!(session.default_cdc_writer.take().is_some());
        let events_before = db.memory_usage().cdc.event_count;
        assert!(
            session
                .create_edge_with_props(a, b, "MISSING_CACHE", [])
                .is_err()
        );
        assert_eq!(db.edge_count(), 0);
        assert_eq!(db.memory_usage().cdc.event_count, events_before);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn no_backward_adjacency_detach_removes_incoming_outgoing_and_self_edges() {
        let database = GrafeoDB::with_config(crate::Config::in_memory().without_backward_edges())
            .expect("no-backward database");
        let incoming_source = database.create_node(&["Incoming"]);
        let victim = database.create_node(&["Victim"]);
        let outgoing_target = database.create_node(&["Outgoing"]);
        let incoming = database.create_edge(incoming_source, victim, "IN");
        let outgoing = database.create_edge(victim, outgoing_target, "OUT");
        let self_loop = database.create_edge(victim, victim, "SELF");

        assert!(database.session().delete_node(victim));
        assert!(database.get_node(victim).is_none());
        assert!(database.get_edge(incoming).is_none());
        assert!(database.get_edge(outgoing).is_none());
        assert!(database.get_edge(self_loop).is_none());
        assert_eq!(database.edge_count(), 0);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn no_backward_adjacency_commit_rejects_new_concurrent_incoming_edge() {
        let database = GrafeoDB::with_config(crate::Config::in_memory().without_backward_edges())
            .expect("no-backward database");
        let source = database.create_node(&["Source"]);
        let victim = database.create_node(&["Victim"]);

        let mut deleter = database.session();
        deleter.begin_transaction().expect("begin delete");
        assert!(deleter.delete_node(victim));

        let winning_edge = database.create_edge(source, victim, "LATE_INCOMING");
        let error = deleter
            .commit()
            .expect_err("stale DETACH enumeration must lose at publication")
            .to_string();
        assert!(error.contains("gained incident edge"), "{error}");
        assert!(database.get_node(victim).is_some());
        assert!(database.get_edge(winning_edge).is_some());
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn bulk_detach_commit_uses_set_membership_for_pending_node_deletes() {
        const NODE_COUNT: usize = 2_048;

        let database = GrafeoDB::new_in_memory();
        let mut creator = database.session();
        creator.begin_transaction().expect("begin bulk create");
        let nodes: Vec<_> = (0..NODE_COUNT)
            .map(|_| creator.create_node(&["BulkDelete"]))
            .collect();
        creator.commit().expect("commit bulk create");

        let mut deleter = database.session();
        deleter.begin_transaction().expect("begin bulk detach");
        for node in &nodes {
            assert!(deleter.delete_node(*node));
        }
        deleter.commit().expect("commit bulk detach");

        assert_eq!(
            nodes
                .iter()
                .filter(|&&node| database.get_node(node).is_some())
                .count(),
            0
        );
    }

    #[cfg(feature = "spill")]
    struct FailSpillQueryRemoval;

    #[cfg(feature = "spill")]
    impl grafeo_core::execution::spill::SpillIo for FailSpillQueryRemoval {
        fn check(
            &self,
            operation: grafeo_core::execution::spill::SpillIoOperation,
        ) -> std::io::Result<()> {
            if operation == grafeo_core::execution::spill::SpillIoOperation::RemoveQueryDirectory {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "session spill finalization failpoint",
                ));
            }
            Ok(())
        }
    }

    #[cfg(feature = "spill")]
    fn manager_with_failing_query_removal(
        root: &std::path::Path,
    ) -> std::sync::Arc<grafeo_core::execution::spill::SpillManager> {
        let (_resources, manager) = crate::spill_crypto::admitted_spill_test_resources(
            root,
            grafeo_common::memory::buffer::BufferManager::with_budget(1 << 20),
            grafeo_core::execution::QueryExecutionControl::new().token(),
            std::sync::Arc::new(grafeo_core::execution::spill::CleartextSpillRecordProvider),
            grafeo_core::execution::spill::SpillFrameLimits::format_max(),
            std::sync::Arc::new(FailSpillQueryRemoval),
            grafeo_core::execution::spill::SpillDiskQuota::new(u64::MAX),
        );
        manager
    }

    #[cfg(feature = "spill")]
    #[test]
    fn spill_query_finalization_blocks_success_and_preserves_primary_error_kind() {
        let success_root = tempfile::tempdir().unwrap();
        let success = super::finish_spill_execution(
            Ok(7u8),
            Some(manager_with_failing_query_removal(success_root.path())),
        )
        .unwrap_err();
        assert!(matches!(
            success,
            grafeo_common::utils::error::Error::Io(ref error)
                if error.kind() == std::io::ErrorKind::PermissionDenied
        ));

        let primary_root = tempfile::tempdir().unwrap();
        let primary = grafeo_common::utils::error::Error::Io(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "primary execution failure",
        ));
        let combined = super::finish_spill_execution::<u8>(
            Err(primary),
            Some(manager_with_failing_query_removal(primary_root.path())),
        )
        .unwrap_err();
        assert_eq!(
            combined.error_code(),
            grafeo_common::utils::error::ErrorCode::IoError
        );
        assert!(combined.to_string().contains("primary execution failure"));
        assert!(combined.to_string().contains("cleanup also failed"));
        let source = std::error::Error::source(&combined)
            .and_then(|source| source.downcast_ref::<grafeo_common::utils::error::Error>())
            .expect("combined error retains the structured primary source");
        assert!(matches!(
            source,
            grafeo_common::utils::error::Error::Io(error)
                if error.kind() == std::io::ErrorKind::BrokenPipe
        ));

        let node_root = tempfile::tempdir().unwrap();
        let node_primary =
            grafeo_common::utils::error::Error::NodeNotFound(grafeo_common::types::NodeId::new(7));
        let node_combined = super::finish_spill_execution::<u8>(
            Err(node_primary),
            Some(manager_with_failing_query_removal(node_root.path())),
        )
        .unwrap_err();
        assert_eq!(
            node_combined.error_code(),
            grafeo_common::utils::error::ErrorCode::NodeNotFound
        );
        assert!(node_combined.to_string().contains("Node not found: 7"));
        assert!(node_combined.to_string().contains("cleanup also failed"));
        let node_source = std::error::Error::source(&node_combined)
            .and_then(|source| source.downcast_ref::<grafeo_common::utils::error::Error>())
            .expect("non-textual primary remains recoverable from the source chain");
        assert!(matches!(
            node_source,
            grafeo_common::utils::error::Error::NodeNotFound(id)
                if *id == grafeo_common::types::NodeId::new(7)
        ));
    }

    #[cfg(all(target_os = "linux", feature = "spill"))]
    fn assert_no_query_spill_leaves(root: &std::path::Path, database: &GrafeoDB) {
        let namespace = root.join(format!("grafeo-store-{}", database.store_id()));
        let mut names: Vec<_> = std::fs::read_dir(namespace)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                ".grafeo-spill-quota",
                ".grafeo-spill-quota.lock",
                ".grafeo-spill-root"
            ]
            .map(std::ffi::OsString::from)
        );
    }

    #[cfg(all(
        target_os = "linux",
        feature = "spill",
        feature = "lpg",
        feature = "gql"
    ))]
    #[test]
    fn synchronous_non_cached_query_finalizes_its_spill_directory_before_success() {
        let directory = tempfile::tempdir().unwrap();
        let spill_root = directory.path().join("spill");
        let database =
            GrafeoDB::with_config(crate::Config::in_memory().with_spill_path(spill_root.clone()))
                .unwrap();
        let mut session = database.session();
        session.begin_transaction().unwrap();
        let baseline_consumers = database.buffer_manager().stats().consumer_count;

        session.execute("MATCH (n) RETURN n ORDER BY n").unwrap();

        assert!(spill_root.is_dir());
        assert_no_query_spill_leaves(&spill_root, &database);
        assert_eq!(
            database.buffer_manager().stats().consumer_count,
            baseline_consumers
        );
        session.rollback().unwrap();
    }

    #[cfg(all(
        target_os = "linux",
        feature = "spill",
        feature = "lpg",
        feature = "gql"
    ))]
    #[test]
    fn cacheable_blocking_plan_retains_no_query_resources_or_spill_leaf() {
        let directory = tempfile::tempdir().unwrap();
        let spill_root = directory.path().join("spill");
        let database =
            GrafeoDB::with_config(crate::Config::in_memory().with_spill_path(spill_root.clone()))
                .unwrap();
        let session = database.session();
        let baseline_consumers = database.buffer_manager().stats().consumer_count;
        const QUERY: &str = "MATCH (n) RETURN n ORDER BY n";

        session.execute(QUERY).unwrap();
        assert_eq!(session.physical_cache.lock().len(), 1);
        session.execute(QUERY).unwrap();
        assert_eq!(session.physical_cache.lock().len(), 1);

        assert_eq!(
            database.buffer_manager().stats().consumer_count,
            baseline_consumers
        );
        assert_no_query_spill_leaves(&spill_root, &database);
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn cached_literal_lookup_installs_and_releases_each_query_context()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use grafeo_common::memory::buffer::BufferManager;
        use grafeo_core::execution::QueryResourceContext;
        use std::sync::Arc;
        const QUERY: &str = "UNWIND ['Alix', 'Vincent'] AS target_name MATCH (n:Person {name: target_name}) RETURN n.name, n.age ORDER BY n.name";
        let database = GrafeoDB::with_config(crate::Config::in_memory().with_gc_interval(0))?;
        let mut session = database.session();
        for (name, age) in [("Alix", 30), ("Vincent", 31)] {
            session.create_node_with_props(
                &["Person"],
                [("name", Value::from(name)), ("age", Value::Int64(age))],
            )?;
        }
        let manager = Arc::clone(
            session
                .buffer_manager
                .as_ref()
                .ok_or("query buffer manager missing")?,
        );
        let baseline_bytes = manager.allocated();
        let baseline_limit = QueryResourceContext::new(Arc::clone(&manager))?
            .query_stats()
            .growth_limit_bytes;
        let expected = vec![
            vec![Value::from("Alix"), Value::Int64(30)],
            vec![Value::from("Vincent"), Value::Int64(31)],
        ];
        for _ in 0..2 {
            assert_eq!(session.execute(QUERY)?.rows(), expected.as_slice());
            assert_eq!(session.physical_cache.lock().len(), 1);
            assert_eq!(manager.allocated(), baseline_bytes);
            assert_eq!(
                QueryResourceContext::new(Arc::clone(&manager))?
                    .query_stats()
                    .growth_limit_bytes,
                baseline_limit,
                "the cached plan must not keep a completed query pool active"
            );
        }
        #[cfg(feature = "cypher")]
        for _ in 0..2 {
            assert_eq!(session.execute_cypher(QUERY)?.rows(), expected.as_slice());
            assert_eq!(session.physical_cache.lock().len(), 2);
            assert_eq!(manager.allocated(), baseline_bytes);
        }

        // Change only this fixture's execution account: storage remains on its
        // real database manager. The warm cached tree must fail admission too.
        let tiny = BufferManager::with_budget(128);
        session.buffer_manager = Some(Arc::clone(&tiny));
        let error = session
            .execute(QUERY)
            .err()
            .ok_or("cached lookup unexpectedly bypassed its budget")?;
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::Error::Storage(
                grafeo_common::utils::error::StorageError::Full,
            )
            .error_code()
        );
        assert_eq!(tiny.allocated(), 0);
        // Even a scalar now admits its output metadata and row storage. The
        // old free-output success under 128 bytes would bypass that contract.
        let scalar_error = session.execute("RETURN 1").unwrap_err();
        assert_eq!(
            scalar_error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert_eq!(tiny.allocated(), 0);
        session.buffer_manager = Some(manager);
        assert_eq!(
            session.execute("RETURN 1")?.rows(),
            &[vec![Value::Int64(1)]]
        );
        assert_eq!(session.execute(QUERY)?.rows(), expected.as_slice());
        Ok(())
    }

    #[cfg(feature = "spill")]
    #[test]
    fn configured_spill_setup_failure_is_not_silently_disabled() {
        let directory = tempfile::tempdir().unwrap();
        let invalid_root = directory.path().join("not-a-directory");
        std::fs::write(&invalid_root, b"occupied").unwrap();
        let database =
            GrafeoDB::with_config(crate::Config::in_memory().with_spill_path(invalid_root))
                .unwrap();
        let session = database.session();

        let Err(error) = session.make_query_resource_context(None) else {
            panic!("configured spill setup failure must reach the query boundary")
        };

        assert!(matches!(error, grafeo_common::utils::error::Error::Io(_)));
    }

    #[cfg(all(target_os = "linux", feature = "spill"))]
    #[test]
    fn configured_query_spill_quota_reaches_the_exclusive_manager() {
        let directory = tempfile::tempdir().unwrap();
        let spill_root = directory.path().join("spill");
        let database = GrafeoDB::with_config(
            crate::Config::in_memory()
                .with_spill_path(spill_root.clone())
                .with_max_query_spill_bytes(1234),
        )
        .unwrap();
        let session = database.session();

        {
            let resources = session.make_query_resource_context(None).unwrap();
            let manager = resources
                .ensure_spill_manager()
                .unwrap()
                .expect("configured spill path admits an exclusive manager");
            assert_eq!(manager.disk_stats().limit_bytes, 1234);
        }

        assert!(spill_root.is_dir());
        assert_no_query_spill_leaves(&spill_root, &database);
    }

    #[cfg(all(feature = "spill", feature = "encryption"))]
    #[test]
    fn encrypted_session_spill_is_sealed_and_round_trips_without_cleartext_fallback() {
        let directory = tempfile::tempdir().unwrap();
        let spill_root = directory.path().join("spill");
        let mut config = crate::Config::in_memory()
            .with_spill_path(spill_root.clone())
            .with_max_query_spill_bytes(1234);
        config.encryption = Some(crate::config::EncryptionConfig {
            key_chain: std::sync::Arc::new(grafeo_common::encryption::KeyChain::new([0x2a; 32])),
        });
        let database = GrafeoDB::with_config(config).unwrap();
        let session = database.session();
        let resources = session.make_query_resource_context(None).unwrap();
        let manager = resources
            .ensure_spill_manager()
            .unwrap()
            .expect("configured encrypted spill admits a query manager");
        assert_eq!(manager.disk_stats().limit_bytes, 1234);
        let mut file = manager
            .create_file(grafeo_core::execution::spill::SpillFileRole::SortRun)
            .unwrap();
        let plaintext = b"grafeo encrypted spill plaintext sentinel";
        file.write_sort_run_start(1, 1).unwrap();
        file.write_sort_row(plaintext).unwrap();
        file.finish_write().unwrap();

        let raw = std::fs::read(file.path()).unwrap();
        assert!(
            !raw.windows(plaintext.len())
                .any(|window| window == plaintext),
            "an encrypted database must never downgrade spill records to cleartext"
        );
        let mut reader = file.reader().unwrap();
        assert_eq!(reader.read_sort_run_start().unwrap(), (1, 1));
        assert_eq!(reader.read_sort_row().unwrap(), plaintext);
        reader.finish().unwrap();
    }

    #[cfg(all(feature = "spill", feature = "encryption"))]
    #[test]
    fn encrypted_session_without_spill_path_remains_resident() {
        let mut config = crate::Config::in_memory();
        config.encryption = Some(crate::config::EncryptionConfig {
            key_chain: std::sync::Arc::new(grafeo_common::encryption::KeyChain::new([0x2b; 32])),
        });
        let database = GrafeoDB::with_config(config).unwrap();
        let session = database.session();

        let resources = session.make_query_resource_context(None).unwrap();
        assert!(resources.spill_manager().is_none());
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn session_query_contexts_are_fresh_and_independent_of_transaction_commits() {
        let database = GrafeoDB::new_in_memory();
        let mut session = database.session();
        let commit_count = session
            .commit_counter
            .load(std::sync::atomic::Ordering::Acquire);
        let baseline_consumers = database.buffer_manager().stats().consumer_count;

        let before_transaction = session.make_query_resource_context(None).unwrap();
        session.begin_transaction().unwrap();
        let during_transaction = session.make_query_resource_context(None).unwrap();

        assert!(during_transaction.query_id() > before_transaction.query_id());
        assert_eq!(
            session
                .commit_counter
                .load(std::sync::atomic::Ordering::Acquire),
            commit_count
        );
        assert_eq!(
            database.buffer_manager().stats().consumer_count,
            baseline_consumers
        );
        #[cfg(feature = "spill")]
        {
            assert!(before_transaction.spill_manager().is_none());
            assert!(during_transaction.spill_manager().is_none());
        }
        session.rollback().unwrap();
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn qualified_read_reports_pre_cancellation_without_publishing_write_state() {
        use grafeo_common::utils::error::{Error, QueryErrorKind};
        use grafeo_core::execution::QueryExecutionControl;

        let database = GrafeoDB::new_in_memory();
        let mut session = database.session();
        session.begin_transaction().unwrap();
        let baseline_commits = session
            .commit_counter
            .load(std::sync::atomic::Ordering::Acquire);
        let baseline_nodes = database.node_count();

        let control = QueryExecutionControl::new();
        control.cancellation_handle().cancel();
        let error = session
            .execute_read_with_checkpoint(
                "UNWIND [3, 1, 2] AS x RETURN x ORDER BY x",
                control.checkpoint(),
            )
            .unwrap_err();

        assert!(matches!(
            error,
            Error::Query(ref query) if query.kind == QueryErrorKind::Cancelled
        ));
        assert!(
            session.in_transaction(),
            "a cancelled read must not close its caller's transaction"
        );
        assert_eq!(database.node_count(), baseline_nodes);
        assert_eq!(
            session
                .commit_counter
                .load(std::sync::atomic::Ordering::Acquire),
            baseline_commits,
            "a cancelled read must not cross a commit boundary"
        );
        session.rollback().unwrap();
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "lpg", feature = "gql"))]
    #[test]
    fn qualified_read_preserves_typed_deadline_error() {
        use grafeo_common::utils::error::{Error, QueryErrorKind};
        use grafeo_core::execution::QueryExecutionControl;
        use std::time::Duration;

        let database = GrafeoDB::new_in_memory();
        let session = database.session();
        let control = QueryExecutionControl::with_timeout(Duration::ZERO).unwrap();

        let error = session
            .execute_read_with_checkpoint("RETURN 1", control.checkpoint())
            .unwrap_err();

        assert!(matches!(
            error,
            Error::Query(ref query)
                if query.kind == QueryErrorKind::Timeout
                    && query.message.contains("0ms")
        ));
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "lpg", feature = "gql"))]
    #[test]
    fn qualified_read_composes_session_timeout_with_caller_control() {
        use grafeo_common::utils::error::{Error, QueryErrorKind};
        use grafeo_core::execution::QueryExecutionControl;
        use std::time::Duration;

        let database =
            GrafeoDB::with_config(crate::Config::in_memory().with_query_timeout(Duration::ZERO))
                .unwrap();
        let session = database.session();
        let control = QueryExecutionControl::new();

        let error = session
            .execute_read_with_checkpoint("RETURN 1", control.checkpoint())
            .unwrap_err();

        assert!(matches!(
            error,
            Error::Query(ref query)
                if query.kind == QueryErrorKind::Timeout
                    && query.message.contains("0ms")
        ));
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn qualified_read_rejects_mutations_before_transaction_framing() {
        use grafeo_common::utils::error::{Error, QueryErrorKind};
        use grafeo_core::execution::QueryExecutionControl;

        let database = GrafeoDB::new_in_memory();
        let session = database.session();
        let baseline_commits = session
            .commit_counter
            .load(std::sync::atomic::Ordering::Acquire);

        let control = QueryExecutionControl::new();
        let error = session
            .execute_read_with_checkpoint("INSERT (:MustNotExist)", control.checkpoint())
            .unwrap_err();

        assert!(matches!(
            error,
            Error::Query(ref query) if query.kind == QueryErrorKind::Semantic
        ));
        assert_eq!(database.node_count(), 0);
        assert!(!session.in_transaction());
        assert_eq!(
            session
                .commit_counter
                .load(std::sync::atomic::Ordering::Acquire),
            baseline_commits
        );
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn qualified_read_rejects_unclassified_procedure_before_execution() {
        use grafeo_common::utils::error::{Error, QueryErrorKind};
        use grafeo_core::execution::QueryExecutionControl;

        let database = GrafeoDB::new_in_memory();
        let session = database.session();
        session
            .execute(
                "CREATE PROCEDURE plant() RETURNS (n NODE) \
                 AS { INSERT (n:Secret) RETURN n }",
            )
            .expect("create mutating procedure fixture");
        let baseline_commits = session
            .commit_counter
            .load(std::sync::atomic::Ordering::Acquire);

        let control = QueryExecutionControl::new();
        let error = session
            .execute_read_with_checkpoint("CALL plant()", control.checkpoint())
            .expect_err("controlled read execution must reject unknown procedure effects");

        assert!(matches!(
            error,
            Error::Query(ref query)
                if query.kind == QueryErrorKind::Unsupported
                    && query.message.contains("controlled read execution")
        ));
        assert_eq!(database.node_count(), 0);
        assert!(!session.in_transaction());
        assert_eq!(
            session
                .commit_counter
                .load(std::sync::atomic::Ordering::Acquire),
            baseline_commits
        );
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn qualified_blocking_reads_share_control_and_tokenless_execute_stays_inert() {
        use grafeo_core::execution::QueryExecutionControl;

        let database = GrafeoDB::new_in_memory();
        let session = database.session();
        let control = QueryExecutionControl::new();

        let sorted = session
            .execute_read_with_checkpoint(
                "UNWIND [3, 1, 2] AS x RETURN x ORDER BY x",
                control.checkpoint(),
            )
            .unwrap();
        assert_eq!(
            sorted.rows(),
            &[
                vec![Value::Int64(1)],
                vec![Value::Int64(2)],
                vec![Value::Int64(3)]
            ]
        );

        let aggregate_control = QueryExecutionControl::new();
        let aggregate = session
            .execute_read_with_checkpoint(
                "UNWIND [1, 2, 3] AS x RETURN COUNT(*) AS count",
                aggregate_control.checkpoint(),
            )
            .unwrap();
        assert_eq!(aggregate.rows(), &[vec![Value::Int64(3)]]);

        let unrelated = QueryExecutionControl::new();
        unrelated.cancellation_handle().cancel();
        let compatibility = session.execute("RETURN 7").unwrap();
        assert_eq!(compatibility.rows(), &[vec![Value::Int64(7)]]);
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn session_resource_context_retains_the_controlled_read_token() {
        use grafeo_core::execution::{QueryCancellationError, QueryExecutionControl};

        let database = GrafeoDB::new_in_memory();
        let session = database.session();
        let control = QueryExecutionControl::new();
        let cancellation = control.cancellation_handle();
        let resources = session
            .make_query_resource_context(Some(control.token()))
            .unwrap();

        cancellation.cancel();

        assert_eq!(
            resources.check_cancelled(),
            Err(QueryCancellationError::Cancelled)
        );
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn assert_publication_wait_observes_controlled_stop<MakeStop, Stop, AssertError>(
        query: &'static str,
        make_stop: MakeStop,
        assert_error: AssertError,
    ) where
        MakeStop: FnOnce() -> (grafeo_core::execution::QueryExecutionCheckpoint, Stop),
        Stop: FnOnce(),
        AssertError: FnOnce(&grafeo_common::utils::error::Error),
    {
        use std::sync::Arc;
        use std::time::{Duration, Instant};

        let database = Arc::new(GrafeoDB::new_in_memory());
        let session = Arc::new(database.session());
        session.execute(query).unwrap();
        session.query_cache.reset_stats();
        let baseline_nodes = database.node_count();
        let baseline_commits = session
            .commit_counter
            .load(std::sync::atomic::Ordering::Acquire);

        let publication = database.transaction_manager.publication().write();
        let (checkpoint, stop) = make_stop();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let worker_session = Arc::clone(&session);
        let worker = std::thread::spawn(move || {
            let result = worker_session.execute_read_with_checkpoint(query, checkpoint);
            let _ = result_tx.send(result);
        });

        let wait_until = Instant::now() + Duration::from_secs(2);
        while session.query_cache.stats().optimized_hits == 0 {
            assert!(
                Instant::now() < wait_until,
                "controlled read did not reach its optimized-plan boundary"
            );
            std::thread::yield_now();
        }
        // The exclusive publication guard prevents the worker from entering
        // physical planning or execution after this observable cache boundary.
        std::thread::sleep(Duration::from_millis(10));
        stop();
        let error = result_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("controlled stop must interrupt publication-barrier acquisition")
            .unwrap_err();
        assert_error(&error);
        assert_eq!(
            session
                .commit_counter
                .load(std::sync::atomic::Ordering::Acquire),
            baseline_commits
        );
        assert!(!session.in_transaction());

        drop(publication);
        worker.join().unwrap();
        assert_eq!(database.node_count(), baseline_nodes);
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn cancellation_while_a_read_waits_for_publication_cannot_change_graph_state() {
        use grafeo_common::utils::error::{Error, QueryErrorKind};
        use grafeo_core::execution::QueryExecutionControl;

        assert_publication_wait_observes_controlled_stop(
            "UNWIND [3, 1, 2] AS x RETURN x ORDER BY x",
            || {
                let control = QueryExecutionControl::new();
                let cancellation = control.cancellation_handle();
                (control.checkpoint(), move || {
                    cancellation.cancel();
                })
            },
            |error| {
                assert!(matches!(
                    error,
                    Error::Query(query) if query.kind == QueryErrorKind::Cancelled
                ));
            },
        );
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn explain_cancellation_while_waiting_for_publication_preserves_graph_state() {
        use grafeo_common::utils::error::{Error, QueryErrorKind};
        use grafeo_core::execution::QueryExecutionControl;

        assert_publication_wait_observes_controlled_stop(
            "EXPLAIN MATCH (n:Person) RETURN n",
            || {
                let control = QueryExecutionControl::new();
                let cancellation = control.cancellation_handle();
                (control.checkpoint(), move || {
                    cancellation.cancel();
                })
            },
            |error| {
                assert!(matches!(
                    error,
                    Error::Query(query) if query.kind == QueryErrorKind::Cancelled
                ));
            },
        );
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "lpg", feature = "gql"))]
    #[test]
    fn deadline_while_a_read_waits_for_publication_preserves_timeout_and_graph_state() {
        use grafeo_common::utils::error::{Error, QueryErrorKind};
        use grafeo_core::execution::QueryExecutionControl;
        use std::time::Duration;

        const TIMEOUT: Duration = Duration::from_millis(200);
        assert_publication_wait_observes_controlled_stop(
            "UNWIND [3, 1, 2] AS x RETURN x ORDER BY x",
            || {
                let control = QueryExecutionControl::with_timeout(TIMEOUT).unwrap();
                (control.checkpoint(), || {})
            },
            |error| {
                assert!(matches!(
                    error,
                    Error::Query(query)
                        if query.kind == QueryErrorKind::Timeout
                            && query.message.contains("200ms")
                ));
            },
        );
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    struct ResetProbeOperator {
        reset_count: usize,
        signal_at: usize,
        entered: std::sync::mpsc::Sender<()>,
        release: Option<std::sync::Mutex<std::sync::mpsc::Receiver<()>>>,
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    impl grafeo_core::execution::operators::Operator for ResetProbeOperator {
        fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
            Ok(None)
        }

        fn reset(&mut self) {
            self.reset_count += 1;
            if self.reset_count == self.signal_at {
                self.entered.send(()).unwrap();
                if let Some(release) = &self.release {
                    release.lock().unwrap().recv().unwrap();
                }
            }
        }

        fn name(&self) -> &'static str {
            "ResetProbe"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn install_reset_probe(
        session: &super::Session,
        query: &str,
        signal_at: usize,
        entered: std::sync::mpsc::Sender<()>,
        release: Option<std::sync::mpsc::Receiver<()>>,
    ) {
        use crate::query::planner::PhysicalPlan;
        use crate::query::processor::QueryLanguage;

        let key = session
            .physical_cache_key(query, QueryLanguage::Gql)
            .unwrap();
        session.store_cached_physical(
            key,
            PhysicalPlan {
                operator: Box::new(ResetProbeOperator {
                    reset_count: 0,
                    signal_at,
                    entered,
                    release: release.map(std::sync::Mutex::new),
                }),
                columns: Vec::new(),
                adaptive_context: None,
            },
        );
    }

    #[cfg(all(feature = "wal", feature = "lpg", feature = "gql"))]
    #[test]
    fn catalog_wal_batch_guard_clears_a_caught_panic() {
        let slot = parking_lot::Mutex::new(None);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _batch = CatalogWalBatchGuard::begin(&slot).unwrap();
            panic!("injected catalog preparation panic");
        }));
        assert!(panic.is_err());
        assert!(slot.lock().is_none());

        let batch = CatalogWalBatchGuard::begin(&slot)
            .expect("a caught panic must not strand nested-batch state");
        assert!(batch.finish().is_empty());
        assert!(slot.lock().is_none());
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn cached_physical_plan_is_not_extracted_before_publication_read() {
        use std::sync::{Arc, mpsc};
        use std::time::Duration;

        const QUERY: &str = "RETURN 1";
        let db = Arc::new(GrafeoDB::new_in_memory());
        let session = Arc::new(db.session());
        let (entered_tx, entered_rx) = mpsc::channel();
        install_reset_probe(&session, QUERY, 2, entered_tx, None);
        assert_eq!(session.physical_cache.lock().len(), 1);

        let publication = db.transaction_manager.publication().write();
        let worker_session = Arc::clone(&session);
        let worker = std::thread::spawn(move || worker_session.execute(QUERY));

        assert!(matches!(
            entered_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        assert_eq!(
            session.physical_cache.lock().len(),
            1,
            "a query waiting for publication must leave its cached plan available for DDL invalidation"
        );
        drop(publication);

        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.join().unwrap().unwrap();
        assert_eq!(session.physical_cache.lock().len(), 1);
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn cached_physical_plan_holds_publication_through_reinsertion() {
        use std::sync::{Arc, mpsc};
        use std::time::Duration;

        const QUERY: &str = "RETURN 1";
        let db = Arc::new(GrafeoDB::new_in_memory());
        let session = Arc::new(db.session());
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        // Cache insertion resets once, cached execution resets a second time,
        // and reinsertion performs the third reset where this probe blocks.
        install_reset_probe(&session, QUERY, 3, entered_tx, Some(release_rx));

        let worker_session = Arc::clone(&session);
        let worker = std::thread::spawn(move || worker_session.execute(QUERY));
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let writer_entered = db.transaction_manager.publication().try_write().is_some();
        release_tx.send(()).unwrap();
        worker.join().unwrap().unwrap();

        assert!(
            !writer_entered,
            "catalog publication must remain excluded until the cached plan is safely reinserted"
        );
        assert_eq!(session.physical_cache.lock().len(), 1);
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn assert_graph_selection_serializes_with_drop(command: &'static str) {
        use std::sync::{Arc, mpsc};
        use std::time::Duration;

        let db = Arc::new(GrafeoDB::new_in_memory());
        db.create_graph("vanishing").unwrap();
        let session = Arc::new(db.session());
        let publication = db.transaction_manager.publication().write();
        let (attempting_tx, attempting_rx) = mpsc::sync_channel(0);
        let (completed_tx, completed_rx) = mpsc::sync_channel(0);
        let worker_session = Arc::clone(&session);
        let worker = std::thread::spawn(move || {
            attempting_tx.send(()).unwrap();
            let result = worker_session.execute(command);
            completed_tx
                .send((result, worker_session.current_graph_path()))
                .unwrap();
        });

        attempting_rx.recv().unwrap();
        assert!(matches!(
            completed_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        assert!(db.transaction_manager.with_write_authority(|| {
            crate::database::testing::root_lpg_store(&db).drop_graph("vanishing")
        }));
        drop(publication);

        let (result, current) = completed_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.join().unwrap();
        assert!(result.is_err());
        assert_eq!(current, grafeo_common::types::GraphPath::root());
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn use_graph_serializes_validation_and_assignment_with_drop() {
        assert_graph_selection_serializes_with_drop("USE GRAPH vanishing");
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn session_set_graph_serializes_validation_and_assignment_with_drop() {
        assert_graph_selection_serializes_with_drop("SESSION SET GRAPH vanishing");
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn session_set_schema_serializes_validation_and_assignment_with_drop() {
        use std::sync::{Arc, mpsc};
        use std::time::Duration;

        let db = Arc::new(GrafeoDB::new_in_memory());
        db.catalog
            .register_schema_namespace("Vanishing".to_string())
            .unwrap();
        let session = Arc::new(db.session());
        let publication = db.transaction_manager.publication().write();
        let (attempting_tx, attempting_rx) = mpsc::sync_channel(0);
        let (completed_tx, completed_rx) = mpsc::sync_channel(0);
        let worker_session = Arc::clone(&session);
        let worker = std::thread::spawn(move || {
            attempting_tx.send(()).unwrap();
            let result = worker_session.execute("SESSION SET SCHEMA vanishing");
            completed_tx
                .send((result, worker_session.current_schema()))
                .unwrap();
        });

        attempting_rx.recv().unwrap();
        assert!(matches!(
            completed_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        db.catalog.drop_schema_namespace("Vanishing").unwrap();
        drop(publication);

        let (result, current) = completed_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.join().unwrap();
        assert!(result.is_err());
        assert_eq!(current, None);
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn parser_free_create_conflicts_when_prefix_becomes_a_schema() {
        let db = GrafeoDB::new_in_memory();
        let mut creator = db.session();
        creator.set_auto_commit(false);

        assert!(creator.create_named_graph("future/root").unwrap());

        let claimant = db.session();
        claimant
            .execute("CREATE SCHEMA future")
            .expect("the detached graph is not visible before its commit");

        let error = creator
            .commit()
            .expect_err("the root prefix was claimed before graph publication");
        assert!(
            matches!(
                error,
                grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::WriteConflict(_)
                )
            ),
            "expected a namespace write conflict, got: {error}"
        );
        assert!(
            crate::database::testing::root_lpg_store(&db)
                .graph("future/root")
                .is_none(),
            "the conflicting detached graph must not publish beneath the live schema"
        );
    }

    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn failed_commit_aborts_tx_and_does_not_pin_gc() {
        // Two transactions begin at the same epoch. T1 writes+commits. T2 then
        // writes the same entity (admitted, since T1 is no longer Active) and
        // commits -> commit-time write-write conflict. The failed commit MUST
        // abort the transaction; leaving it Active pins min_active_epoch and
        // stalls MVCC GC forever.
        let db = GrafeoDB::new_in_memory();
        let mut s1 = db.session();
        s1.execute("CREATE (:Acct {id: 1, bal: 100})").unwrap();

        let mut s2 = db.session();
        s1.begin_transaction().unwrap();
        s2.begin_transaction().unwrap();

        s1.execute("MATCH (a:Acct {id: 1}) SET a.bal = 50").unwrap();
        s1.commit().unwrap();

        s2.execute("MATCH (a:Acct {id: 1}) SET a.bal = 60").unwrap();
        let r = s2.commit();
        assert!(r.is_err(), "expected a commit-time write-write conflict");

        // White-box: no zombie Active transaction left behind.
        assert_eq!(
            s2.transaction_manager.active_count(),
            0,
            "failed commit left a zombie Active transaction (GC-pinning leak)"
        );

        // Data is consistent: T1's committed value is visible exactly once.
        let s3 = db.session();
        let q = s3.execute("MATCH (a:Acct {id: 1}) RETURN a.bal").unwrap();
        assert_eq!(q.row_count(), 1);
        assert_eq!(q.rows()[0][0], Value::Int64(50));
    }

    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn writeset_scoped_commit_finalizes_query_and_session_direct_paths() {
        // The store-level pending-create index must finalize entities touched via
        // BOTH the query (execute) path and the session-direct mutation APIs. A
        // gap surfaces here as committed-but-invisible data. (MERGE/LOAD DATA
        // coverage is provided by the coverage_patterns suite under all-features.)
        let db = GrafeoDB::new_in_memory();
        let mut s = db.session();
        s.begin_transaction().unwrap();

        s.execute("CREATE (:QPath {name: 'q'})").unwrap();
        let n1 = s.create_node(&["Direct"]);
        s.set_node_property(n1, "k", Value::Int64(42)).unwrap();
        let n2 = s.create_node(&["Direct"]);
        let _e = s.create_edge(n1, n2, "REL");

        s.commit().unwrap();

        let r = db.session();
        assert_eq!(
            r.execute("MATCH (:QPath) RETURN count(*) AS c")
                .unwrap()
                .rows()[0][0],
            Value::Int64(1),
            "query-path create must be finalized"
        );
        assert_eq!(
            r.execute("MATCH (n:Direct) RETURN count(n) AS c")
                .unwrap()
                .rows()[0][0],
            Value::Int64(2),
            "session-direct creates must be finalized"
        );
        assert_eq!(
            r.execute("MATCH ()-[e:REL]->() RETURN count(e) AS c")
                .unwrap()
                .rows()[0][0],
            Value::Int64(1),
            "session-direct edge must be finalized"
        );
        assert_eq!(
            r.execute("MATCH (n:Direct) WHERE n.k = 42 RETURN count(n) AS c")
                .unwrap()
                .rows()[0][0],
            Value::Int64(1),
            "session-direct property must be finalized"
        );
    }

    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn writeset_scoped_rollback_discards_both_paths() {
        let db = GrafeoDB::new_in_memory();
        let mut s = db.session();
        s.begin_transaction().unwrap();
        s.execute("CREATE (:QPath2 {name: 'q'})").unwrap();
        let _n1 = s.create_node(&["Direct2"]);
        let _n2 = s.create_node(&["Direct2"]);
        s.rollback().unwrap();

        let r = db.session();
        assert_eq!(
            r.execute("MATCH (:QPath2) RETURN count(*) AS c")
                .unwrap()
                .rows()[0][0],
            Value::Int64(0),
            "query-path create must be discarded on rollback"
        );
        assert_eq!(
            r.execute("MATCH (:Direct2) RETURN count(*) AS c")
                .unwrap()
                .rows()[0][0],
            Value::Int64(0),
            "session-direct creates must be discarded on rollback"
        );
    }

    // -----------------------------------------------------------------------
    // parse_default_literal
    // -----------------------------------------------------------------------

    #[cfg(all(feature = "gql", feature = "lpg"))]
    #[test]
    fn parse_default_literal_null() {
        assert_eq!(parse_default_literal("null"), Value::Null);
        assert_eq!(parse_default_literal("NULL"), Value::Null);
        assert_eq!(parse_default_literal("Null"), Value::Null);
    }

    #[cfg(all(feature = "gql", feature = "lpg"))]
    #[test]
    fn parse_default_literal_bool() {
        assert_eq!(parse_default_literal("true"), Value::Bool(true));
        assert_eq!(parse_default_literal("TRUE"), Value::Bool(true));
        assert_eq!(parse_default_literal("false"), Value::Bool(false));
        assert_eq!(parse_default_literal("FALSE"), Value::Bool(false));
    }

    #[cfg(all(feature = "gql", feature = "lpg"))]
    #[test]
    fn parse_default_literal_string_single_quoted() {
        assert_eq!(
            parse_default_literal("'hello'"),
            Value::String("hello".into())
        );
    }

    #[cfg(all(feature = "gql", feature = "lpg"))]
    #[test]
    fn parse_default_literal_string_double_quoted() {
        assert_eq!(
            parse_default_literal("\"world\""),
            Value::String("world".into())
        );
    }

    #[cfg(all(feature = "gql", feature = "lpg"))]
    #[test]
    fn parse_default_literal_integer() {
        assert_eq!(parse_default_literal("42"), Value::Int64(42));
        assert_eq!(parse_default_literal("-7"), Value::Int64(-7));
        assert_eq!(parse_default_literal("0"), Value::Int64(0));
    }

    #[cfg(all(feature = "gql", feature = "lpg"))]
    #[test]
    fn parse_default_literal_float() {
        assert_eq!(parse_default_literal("9.81"), Value::Float64(9.81_f64));
        assert_eq!(parse_default_literal("-0.5"), Value::Float64(-0.5));
    }

    #[cfg(all(feature = "gql", feature = "lpg"))]
    #[test]
    fn parse_default_literal_fallback_string() {
        // Not a recognized literal, not quoted, not a number
        assert_eq!(
            parse_default_literal("some_identifier"),
            Value::String("some_identifier".into())
        );
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_session_create_node() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        let id = session.create_node(&["Person"]);
        assert!(id.is_valid());
        assert_eq!(db.node_count(), 1);
    }

    #[test]
    fn test_session_transaction() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        assert!(!session.in_transaction());

        session.begin_transaction().unwrap();
        assert!(session.in_transaction());

        session.commit().unwrap();
        assert!(!session.in_transaction());
    }

    #[cfg(all(feature = "triple-store", not(feature = "lpg")))]
    #[test]
    fn rdf_nested_transactions_preserve_outer_state_and_depth() {
        use grafeo_core::graph::rdf::{Term, Triple};

        let db = GrafeoDB::with_config(
            crate::Config::in_memory().with_graph_model(crate::GraphModel::Rdf),
        )
        .unwrap();
        let mut session = db.session();
        let statement = |subject: &str| {
            Triple::new(
                Term::iri(subject),
                Term::iri("http://example.com/p"),
                Term::literal("value"),
            )
        };
        let outer = statement("http://example.com/outer");
        let rolled_back = statement("http://example.com/rolled-back-inner");
        let committed_inner = statement("http://example.com/committed-inner");

        session.begin_transaction().unwrap();
        session.insert_rdf_batch([outer.clone()]).unwrap();

        session.begin_transaction().unwrap();
        session.insert_rdf_batch([rolled_back.clone()]).unwrap();
        session.rollback().unwrap();
        assert!(
            session.in_transaction(),
            "nested rollback keeps the outer tx"
        );

        session.begin_transaction().unwrap();
        session.insert_rdf_batch([committed_inner.clone()]).unwrap();
        session.commit().unwrap();
        assert!(session.in_transaction(), "nested commit keeps the outer tx");

        session.commit().unwrap();
        assert!(db.rdf_store().contains(&outer));
        assert!(db.rdf_store().contains(&committed_inner));
        assert!(!db.rdf_store().contains(&rolled_back));
    }

    #[cfg(all(feature = "triple-store", not(feature = "lpg")))]
    #[test]
    fn rdf_statement_savepoint_rewinds_panic_without_losing_nested_scope() {
        use grafeo_core::graph::rdf::{Quad, Term, Triple};

        let db = GrafeoDB::with_config(
            crate::Config::in_memory().with_graph_model(crate::GraphModel::Rdf),
        )
        .unwrap();
        let mut session = db.session();
        let statement = |subject: &str| {
            Triple::new(
                Term::iri(subject),
                Term::iri("http://example.com/p"),
                Term::literal("value"),
            )
        };
        let retained_before = statement("http://example.com/before-panic");
        let panic_residue = statement("http://example.com/panic-residue");
        let retained_after = statement("http://example.com/after-panic");

        session.begin_transaction().unwrap();
        session.insert_rdf_batch([retained_before.clone()]).unwrap();
        session.begin_transaction().unwrap();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = session.with_auto_commit(true, || {
                session.insert_rdf_batch([panic_residue.clone()])?;
                panic!("injected RDF statement panic")
            });
        }));

        assert!(panic.is_err());
        assert!(session.in_transaction());
        assert!(session.contains_rdf_quad(&Quad::new(retained_before.clone())));
        assert!(!session.contains_rdf_quad(&Quad::new(panic_residue.clone())));

        session.commit().unwrap();
        assert!(
            session.in_transaction(),
            "statement rewind must not consume the caller's nested transaction frame"
        );
        session.insert_rdf_batch([retained_after.clone()]).unwrap();
        session.commit().unwrap();

        assert!(db.rdf_store().contains(&retained_before));
        assert!(db.rdf_store().contains(&retained_after));
        assert!(!db.rdf_store().contains(&panic_residue));
    }

    #[cfg(all(feature = "triple-store", not(feature = "lpg")))]
    #[test]
    fn rdf_statement_savepoint_cleanup_failure_aborts_every_nested_scope() {
        use grafeo_core::graph::rdf::{Term, Triple};

        let db = GrafeoDB::with_config(
            crate::Config::in_memory().with_graph_model(crate::GraphModel::Rdf),
        )
        .unwrap();
        let starting_epoch = db.rdf_store().commit_epoch();
        let mut session = db.session();
        let statement = |subject: &str| {
            Triple::new(
                Term::iri(subject),
                Term::iri("http://example.com/p"),
                Term::literal("value"),
            )
        };

        session.begin_transaction().unwrap();
        let transaction_id = session.current_transaction.lock().unwrap();
        session
            .insert_rdf_batch([statement("http://example.com/outer")])
            .unwrap();
        session.begin_transaction().unwrap();
        session
            .insert_rdf_batch([statement("http://example.com/inner")])
            .unwrap();

        let error = session
            .with_auto_commit(true, || {
                session.insert_rdf_batch([statement("http://example.com/residue")])?;
                // Deterministically model internal savepoint-state loss after
                // execution. The wrapper must fail closed by aborting the
                // outer transaction, not merely one nested frame.
                session.savepoints.lock().clear();
                Err(grafeo_common::utils::error::Error::Internal(
                    "injected statement error after savepoint loss".to_string(),
                ))
            })
            .unwrap_err();

        assert!(error.to_string().contains("savepoint cleanup failed"));
        assert!(error.to_string().contains("transaction was rolled back"));
        assert!(!session.in_transaction());
        assert!(!db.rdf_store().has_pending_ops(transaction_id));
        assert!(db.rdf_store().is_empty());
        assert_eq!(db.rdf_store().commit_epoch(), starting_epoch);
    }

    #[cfg(all(feature = "lpg", feature = "triple-store"))]
    #[test]
    fn mixed_statement_savepoint_cleanup_failure_aborts_both_models_and_all_scopes() {
        use grafeo_core::graph::rdf::{Term, Triple};

        let db = GrafeoDB::with_config(
            crate::Config::in_memory().with_graph_model(crate::GraphModel::Both),
        )
        .unwrap();
        let starting_epoch = db.current_epoch();
        let mut session = db.session();
        let statement = |subject: &str| {
            Triple::new(
                Term::iri(subject),
                Term::iri("http://example.com/p"),
                Term::literal("value"),
            )
        };

        session.begin_transaction().unwrap();
        let transaction_id = session.current_transaction.lock().unwrap();
        assert!(session.create_node(&["Outer"]).is_valid());
        session
            .insert_rdf_batch([statement("http://example.com/outer")])
            .unwrap();
        session.begin_transaction().unwrap();
        assert!(session.create_node(&["Inner"]).is_valid());
        session
            .insert_rdf_batch([statement("http://example.com/inner")])
            .unwrap();

        let error = session
            .with_auto_commit(true, || {
                assert!(session.create_node(&["Residue"]).is_valid());
                session.insert_rdf_batch([statement("http://example.com/residue")])?;
                session.savepoints.lock().clear();
                Err(grafeo_common::utils::error::Error::Internal(
                    "injected mixed statement error after savepoint loss".to_string(),
                ))
            })
            .unwrap_err();

        assert!(error.to_string().contains("savepoint cleanup failed"));
        assert!(error.to_string().contains("transaction was rolled back"));
        assert!(!session.in_transaction());
        assert_eq!(db.node_count(), 0);
        assert!(!db.rdf_store().has_pending_ops(transaction_id));
        assert!(db.rdf_store().is_empty());
        assert_eq!(db.current_epoch(), starting_epoch);
    }

    #[cfg(any(
        feature = "lpg",
        feature = "gql",
        feature = "cypher",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    ))]
    #[test]
    fn test_session_transaction_context() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        // Without transaction - context should have current epoch and no transaction_id
        let (_epoch1, transaction_id1) = session.get_transaction_context();
        assert!(transaction_id1.is_none());

        // Start a transaction
        session.begin_transaction().unwrap();
        let (epoch2, transaction_id2) = session.get_transaction_context();
        assert!(transaction_id2.is_some());
        // Transaction should have a valid epoch
        let _ = epoch2; // Use the variable

        // Commit and verify
        session.commit().unwrap();
        let (epoch3, tx_id3) = session.get_transaction_context();
        assert!(tx_id3.is_none());
        // Epoch should have advanced after commit
        assert!(epoch3.as_u64() >= epoch2.as_u64());
    }

    #[cfg(feature = "gql")]
    #[test]
    fn scoped_viewing_epoch_is_thread_local_and_unwind_safe() {
        let db = GrafeoDB::new_in_memory();
        let session = std::sync::Arc::new(db.session());
        let persistent = grafeo_common::types::EpochId::new(7);
        let temporary = grafeo_common::types::EpochId::new(11);
        session.set_viewing_epoch(persistent);

        let entered = std::sync::Arc::new(std::sync::Barrier::new(2));
        let release = std::sync::Arc::new(std::sync::Barrier::new(2));
        let worker_session = std::sync::Arc::clone(&session);
        let worker_entered = std::sync::Arc::clone(&entered);
        let worker_release = std::sync::Arc::clone(&release);
        let worker = std::thread::spawn(move || {
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                worker_session.with_scoped_viewing_epoch(temporary, || {
                    assert_eq!(worker_session.effective_viewing_epoch(), Some(temporary));
                    worker_entered.wait();
                    worker_release.wait();
                    panic!("historical scope unwind witness");
                });
            }));
            assert!(panic.is_err());
            assert_eq!(
                worker_session.effective_viewing_epoch(),
                Some(persistent),
                "unwinding must restore the persistent override"
            );
        });

        entered.wait();
        assert_eq!(
            session.effective_viewing_epoch(),
            Some(persistent),
            "another thread must not observe the temporary scope"
        );
        release.wait();
        worker.join().unwrap();
        assert_eq!(session.effective_viewing_epoch(), Some(persistent));
    }

    #[test]
    fn test_session_rollback() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        session.begin_transaction().unwrap();
        session.rollback().unwrap();
        assert!(!session.in_transaction());
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_session_rollback_discards_versions() {
        use grafeo_common::types::TransactionId;

        let db = GrafeoDB::new_in_memory();

        // Create a node outside of any transaction (at system level)
        let node_before = crate::database::testing::root_lpg_store(&db).create_node(&["Person"]);
        assert!(node_before.is_valid());
        assert_eq!(db.node_count(), 1, "Should have 1 node before transaction");

        // Start a transaction
        let mut session = db.session();
        session.begin_transaction().unwrap();
        let transaction_id = session.current_transaction.lock().unwrap();

        // Create a node versioned with the transaction's ID
        let epoch = crate::database::testing::root_lpg_store(&db).current_epoch();
        let node_in_tx = crate::database::testing::root_lpg_store(&db).create_node_versioned(
            &["Person"],
            epoch,
            transaction_id,
        );
        assert!(node_in_tx.is_valid());

        // Uncommitted nodes use EpochId::PENDING, so they are invisible to
        // non-versioned lookups like node_count(). Verify the node is visible
        // only through the owning transaction.
        assert_eq!(
            db.node_count(),
            1,
            "PENDING nodes should be invisible to non-versioned node_count()"
        );
        assert!(
            crate::database::testing::root_lpg_store(&db)
                .get_node_versioned(node_in_tx, epoch, transaction_id)
                .is_some(),
            "Transaction node should be visible to its own transaction"
        );

        // Rollback the transaction
        session.rollback().unwrap();
        assert!(!session.in_transaction());

        // The node created in the transaction should be discarded
        // Only the first node should remain visible
        let count_after = db.node_count();
        assert_eq!(
            count_after, 1,
            "Rollback should discard uncommitted node, but got {count_after}"
        );

        // The original node should still be accessible
        let current_epoch = crate::database::testing::root_lpg_store(&db).current_epoch();
        assert!(
            crate::database::testing::root_lpg_store(&db)
                .get_node_versioned(node_before, current_epoch, TransactionId::SYSTEM)
                .is_some(),
            "Original node should still exist"
        );

        // The node created in the transaction should not be accessible
        assert!(
            crate::database::testing::root_lpg_store(&db)
                .get_node_versioned(node_in_tx, current_epoch, TransactionId::SYSTEM)
                .is_none(),
            "Transaction node should be gone"
        );
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_session_create_node_in_transaction() {
        // Test that session.create_node() is transaction-aware
        let db = GrafeoDB::new_in_memory();

        // Create a node outside of any transaction
        let node_before = db.create_node(&["Person"]);
        assert!(node_before.is_valid());
        assert_eq!(db.node_count(), 1, "Should have 1 node before transaction");

        // Start a transaction and create a node through the session
        let mut session = db.session();
        session.begin_transaction().unwrap();
        let transaction_id = session.current_transaction.lock().unwrap();

        // Create a node through session.create_node() - should be versioned with tx
        let node_in_tx = session.create_node(&["Person"]);
        assert!(node_in_tx.is_valid());

        // Uncommitted nodes use EpochId::PENDING, so they are invisible to
        // non-versioned lookups. Verify the node is visible only to its own tx.
        assert_eq!(
            db.node_count(),
            1,
            "PENDING nodes should be invisible to non-versioned node_count()"
        );
        let epoch = crate::database::testing::root_lpg_store(&db).current_epoch();
        assert!(
            crate::database::testing::root_lpg_store(&db)
                .get_node_versioned(node_in_tx, epoch, transaction_id)
                .is_some(),
            "Transaction node should be visible to its own transaction"
        );

        // Rollback the transaction
        session.rollback().unwrap();

        // The node created via session.create_node() should be discarded
        let count_after = db.node_count();
        assert_eq!(
            count_after, 1,
            "Rollback should discard node created via session.create_node(), but got {count_after}"
        );
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_session_create_node_with_props_in_transaction() {
        use grafeo_common::types::Value;

        // Test that session.create_node_with_props() is transaction-aware
        let db = GrafeoDB::new_in_memory();

        // Create a node outside of any transaction
        db.create_node(&["Person"]);
        assert_eq!(db.node_count(), 1, "Should have 1 node before transaction");

        // Start a transaction and create a node with properties
        let mut session = db.session();
        session.begin_transaction().unwrap();
        let transaction_id = session.current_transaction.lock().unwrap();

        let node_in_tx = session
            .create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))])
            .unwrap();
        assert!(node_in_tx.is_valid());

        // Uncommitted nodes use EpochId::PENDING, so they are invisible to
        // non-versioned lookups. Verify the node is visible only to its own tx.
        assert_eq!(
            db.node_count(),
            1,
            "PENDING nodes should be invisible to non-versioned node_count()"
        );
        let epoch = crate::database::testing::root_lpg_store(&db).current_epoch();
        assert!(
            crate::database::testing::root_lpg_store(&db)
                .get_node_versioned(node_in_tx, epoch, transaction_id)
                .is_some(),
            "Transaction node should be visible to its own transaction"
        );

        // Rollback the transaction
        session.rollback().unwrap();

        // The node should be discarded
        let count_after = db.node_count();
        assert_eq!(
            count_after, 1,
            "Rollback should discard node created via session.create_node_with_props()"
        );
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn implicit_mutation_boundary_rolls_back_before_resuming_a_panic() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let epoch = db.current_epoch();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = session.with_lpg_plan_auto_commit(true, false, || {
                let node = session.create_node(&["ImplicitPanicResidue"]);
                assert!(node.is_valid());
                panic!("injected implicit statement panic")
            });
        }));

        assert!(panic.is_err());
        assert!(!session.in_transaction());
        assert_eq!(db.node_count(), 0);
        assert_eq!(db.current_epoch(), epoch);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn statement_savepoint_rewinds_before_resuming_a_panic() {
        use grafeo_common::types::NodeId;

        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        let preserved = session.create_node(&["BeforeProcedurePanic"]);
        let mut residue = NodeId::INVALID;

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = session.with_lpg_plan_auto_commit(true, true, || {
                residue = session.create_node(&["ProcedurePanicResidue"]);
                assert!(residue.is_valid());
                panic!("injected procedure statement panic")
            });
        }));

        assert!(panic.is_err());
        assert!(session.in_transaction());
        assert!(session.get_node(preserved).is_some());
        assert!(session.get_node(residue).is_none());
        session.commit().unwrap();
        assert!(db.get_node(preserved).is_some());
        assert!(db.get_node(residue).is_none());
    }

    #[cfg(all(feature = "gql", feature = "lpg"))]
    mod gql_tests {
        use super::*;

        #[test]
        fn test_gql_query_execution() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create some test data
            session.create_node(&["Person"]);
            session.create_node(&["Person"]);
            session.create_node(&["Animal"]);

            // Execute a GQL query
            let result = session.execute("MATCH (n:Person) RETURN n").unwrap();

            // Should return 2 Person nodes
            assert_eq!(result.row_count(), 2);
            assert_eq!(result.column_count(), 1);
            assert_eq!(result.columns[0], "n");
        }

        #[test]
        fn test_gql_empty_result() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // No data in database
            let result = session.execute("MATCH (n:Person) RETURN n").unwrap();

            assert_eq!(result.row_count(), 0);
        }

        #[test]
        fn test_gql_parse_error() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Invalid GQL syntax
            let result = session.execute("MATCH (n RETURN n");

            assert!(result.is_err());
        }

        #[test]
        fn test_gql_relationship_traversal() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create a graph: Alix -> Gus, Alix -> Vincent
            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let vincent = session.create_node(&["Person"]);

            session.create_edge(alix, gus, "KNOWS");
            session.create_edge(alix, vincent, "KNOWS");

            // Execute a path query: MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b
            let result = session
                .execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b")
                .unwrap();

            // Should return 2 rows (Alix->Gus, Alix->Vincent)
            assert_eq!(result.row_count(), 2);
            assert_eq!(result.column_count(), 2);
            assert_eq!(result.columns[0], "a");
            assert_eq!(result.columns[1], "b");
        }

        #[test]
        fn test_gql_relationship_with_type_filter() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create a graph: Alix -KNOWS-> Gus, Alix -WORKS_WITH-> Vincent
            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let vincent = session.create_node(&["Person"]);

            session.create_edge(alix, gus, "KNOWS");
            session.create_edge(alix, vincent, "WORKS_WITH");

            // Query only KNOWS relationships
            let result = session
                .execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b")
                .unwrap();

            // Should return only 1 row (Alix->Gus)
            assert_eq!(result.row_count(), 1);
        }

        #[test]
        fn test_gql_semantic_error_undefined_variable() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Reference undefined variable 'x' in RETURN
            let result = session.execute("MATCH (n:Person) RETURN x");

            // Should fail with semantic error
            assert!(result.is_err());
            let Err(err) = result else {
                panic!("Expected error")
            };
            assert!(
                err.to_string().contains("Undefined variable"),
                "Expected undefined variable error, got: {}",
                err
            );
        }

        #[test]
        fn test_gql_where_clause_property_filter() {
            use grafeo_common::types::Value;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create people with ages
            session
                .create_node_with_props(&["Person"], [("age", Value::Int64(25))])
                .unwrap();
            session
                .create_node_with_props(&["Person"], [("age", Value::Int64(35))])
                .unwrap();
            session
                .create_node_with_props(&["Person"], [("age", Value::Int64(45))])
                .unwrap();

            // Query with WHERE clause: age > 30
            let result = session
                .execute("MATCH (n:Person) WHERE n.age > 30 RETURN n")
                .unwrap();

            // Should return 2 people (ages 35 and 45)
            assert_eq!(result.row_count(), 2);
        }

        #[test]
        fn test_gql_where_clause_equality() {
            use grafeo_common::types::Value;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create people with names
            session
                .create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))])
                .unwrap();
            session
                .create_node_with_props(&["Person"], [("name", Value::String("Gus".into()))])
                .unwrap();
            session
                .create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))])
                .unwrap();

            // Query with WHERE clause: name = "Alix"
            let result = session
                .execute("MATCH (n:Person) WHERE n.name = \"Alix\" RETURN n")
                .unwrap();

            // Should return 2 people named Alix
            assert_eq!(result.row_count(), 2);
        }

        #[test]
        fn test_gql_return_property_access() {
            use grafeo_common::types::Value;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create people with names and ages
            session
                .create_node_with_props(
                    &["Person"],
                    [
                        ("name", Value::String("Alix".into())),
                        ("age", Value::Int64(30)),
                    ],
                )
                .unwrap();
            session
                .create_node_with_props(
                    &["Person"],
                    [
                        ("name", Value::String("Gus".into())),
                        ("age", Value::Int64(25)),
                    ],
                )
                .unwrap();

            // Query returning properties
            let result = session
                .execute("MATCH (n:Person) RETURN n.name, n.age")
                .unwrap();

            // Should return 2 rows with name and age columns
            assert_eq!(result.row_count(), 2);
            assert_eq!(result.column_count(), 2);
            assert_eq!(result.columns[0], "n.name");
            assert_eq!(result.columns[1], "n.age");

            // Check that we get actual values
            let names: Vec<&Value> = result.rows.iter().map(|r| &r[0]).collect();
            assert!(names.contains(&&Value::String("Alix".into())));
            assert!(names.contains(&&Value::String("Gus".into())));
        }

        #[test]
        fn test_gql_return_mixed_expressions() {
            use grafeo_common::types::Value;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create a person
            session
                .create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))])
                .unwrap();

            // Query returning both node and property
            let result = session
                .execute("MATCH (n:Person) RETURN n, n.name")
                .unwrap();

            assert_eq!(result.row_count(), 1);
            assert_eq!(result.column_count(), 2);
            assert_eq!(result.columns[0], "n");
            assert_eq!(result.columns[1], "n.name");

            // Second column should be the name
            assert_eq!(result.rows[0][1], Value::String("Alix".into()));
        }
    }

    #[cfg(all(feature = "cypher", feature = "lpg"))]
    mod cypher_tests {
        use super::*;

        #[test]
        fn test_cypher_query_execution() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create some test data
            session.create_node(&["Person"]);
            session.create_node(&["Person"]);
            session.create_node(&["Animal"]);

            // Execute a Cypher query
            let result = session.execute_cypher("MATCH (n:Person) RETURN n").unwrap();

            // Should return 2 Person nodes
            assert_eq!(result.row_count(), 2);
            assert_eq!(result.column_count(), 1);
            assert_eq!(result.columns[0], "n");
        }

        #[test]
        fn test_cypher_empty_result() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // No data in database
            let result = session.execute_cypher("MATCH (n:Person) RETURN n").unwrap();

            assert_eq!(result.row_count(), 0);
        }

        #[test]
        fn test_cypher_parse_error() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Invalid Cypher syntax
            let result = session.execute_cypher("MATCH (n RETURN n");

            assert!(result.is_err());
        }
    }

    // ==================== Direct Lookup API Tests ====================

    #[cfg(feature = "lpg")]
    mod direct_lookup_tests {
        use super::*;
        use grafeo_common::types::Value;

        #[test]
        fn test_get_node() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let id = session.create_node(&["Person"]);
            let node = session.get_node(id);

            assert!(node.is_some());
            let node = node.unwrap();
            assert_eq!(node.id, id);
        }

        #[test]
        fn test_get_node_not_found() {
            use grafeo_common::types::NodeId;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Try to get a non-existent node
            let node = session.get_node(NodeId::new(9999));
            assert!(node.is_none());
        }

        #[test]
        fn test_get_node_property() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let id = session
                .create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))])
                .unwrap();

            let name = session.get_node_property(id, "name");
            assert_eq!(name, Some(Value::String("Alix".into())));

            // Non-existent property
            let missing = session.get_node_property(id, "missing");
            assert!(missing.is_none());
        }

        #[test]
        fn test_get_edge() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let edge_id = session.create_edge(alix, gus, "KNOWS");

            let edge = session.get_edge(edge_id);
            assert!(edge.is_some());
            let edge = edge.unwrap();
            assert_eq!(edge.id, edge_id);
            assert_eq!(edge.src, alix);
            assert_eq!(edge.dst, gus);
        }

        #[test]
        fn test_get_edge_not_found() {
            use grafeo_common::types::EdgeId;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let edge = session.get_edge(EdgeId::new(9999));
            assert!(edge.is_none());
        }

        #[test]
        fn test_get_neighbors_outgoing() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let harm = session.create_node(&["Person"]);

            session.create_edge(alix, gus, "KNOWS");
            session.create_edge(alix, harm, "KNOWS");

            let neighbors = session.get_neighbors_outgoing(alix);
            assert_eq!(neighbors.len(), 2);

            let neighbor_ids: Vec<_> = neighbors.iter().map(|(node_id, _)| *node_id).collect();
            assert!(neighbor_ids.contains(&gus));
            assert!(neighbor_ids.contains(&harm));
        }

        #[test]
        fn test_get_neighbors_incoming() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let harm = session.create_node(&["Person"]);

            session.create_edge(gus, alix, "KNOWS");
            session.create_edge(harm, alix, "KNOWS");

            let neighbors = session.get_neighbors_incoming(alix);
            assert_eq!(neighbors.len(), 2);

            let neighbor_ids: Vec<_> = neighbors.iter().map(|(node_id, _)| *node_id).collect();
            assert!(neighbor_ids.contains(&gus));
            assert!(neighbor_ids.contains(&harm));
        }

        #[test]
        fn test_get_neighbors_outgoing_by_type() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let company = session.create_node(&["Company"]);

            session.create_edge(alix, gus, "KNOWS");
            session.create_edge(alix, company, "WORKS_AT");

            let knows_neighbors = session.get_neighbors_outgoing_by_type(alix, "KNOWS");
            assert_eq!(knows_neighbors.len(), 1);
            assert_eq!(knows_neighbors[0].0, gus);

            let works_neighbors = session.get_neighbors_outgoing_by_type(alix, "WORKS_AT");
            assert_eq!(works_neighbors.len(), 1);
            assert_eq!(works_neighbors[0].0, company);

            // No edges of this type
            let no_neighbors = session.get_neighbors_outgoing_by_type(alix, "LIKES");
            assert!(no_neighbors.is_empty());
        }

        #[test]
        fn test_node_exists() {
            use grafeo_common::types::NodeId;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let id = session.create_node(&["Person"]);

            assert!(session.node_exists(id));
            assert!(!session.node_exists(NodeId::new(9999)));
        }

        #[test]
        fn test_edge_exists() {
            use grafeo_common::types::EdgeId;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let edge_id = session.create_edge(alix, gus, "KNOWS");

            assert!(session.edge_exists(edge_id));
            assert!(!session.edge_exists(EdgeId::new(9999)));
        }

        #[test]
        fn test_get_degree() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let harm = session.create_node(&["Person"]);

            // Alix knows Gus and Harm (2 outgoing)
            session.create_edge(alix, gus, "KNOWS");
            session.create_edge(alix, harm, "KNOWS");
            // Gus knows Alix (1 incoming for Alix)
            session.create_edge(gus, alix, "KNOWS");

            let (out_degree, in_degree) = session.get_degree(alix);
            assert_eq!(out_degree, 2);
            assert_eq!(in_degree, 1);

            // Node with no edges
            let lonely = session.create_node(&["Person"]);
            let (out, in_deg) = session.get_degree(lonely);
            assert_eq!(out, 0);
            assert_eq!(in_deg, 0);
        }

        #[test]
        fn test_get_nodes_batch() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let harm = session.create_node(&["Person"]);

            let nodes = session.get_nodes_batch(&[alix, gus, harm]);
            assert_eq!(nodes.len(), 3);
            assert!(nodes[0].is_some());
            assert!(nodes[1].is_some());
            assert!(nodes[2].is_some());

            // With non-existent node
            use grafeo_common::types::NodeId;
            let nodes_with_missing = session.get_nodes_batch(&[alix, NodeId::new(9999), harm]);
            assert_eq!(nodes_with_missing.len(), 3);
            assert!(nodes_with_missing[0].is_some());
            assert!(nodes_with_missing[1].is_none()); // Missing node
            assert!(nodes_with_missing[2].is_some());
        }

        #[test]
        fn test_auto_commit_setting() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();

            // Default is auto-commit enabled
            assert!(session.auto_commit());

            session.set_auto_commit(false);
            assert!(!session.auto_commit());

            session.set_auto_commit(true);
            assert!(session.auto_commit());
        }

        #[test]
        fn test_transaction_double_begin_nests() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();

            session.begin_transaction().unwrap();
            // Second begin_transaction creates a nested transaction (auto-savepoint)
            let result = session.begin_transaction();
            assert!(result.is_ok());
            // Commit the inner (releases savepoint)
            session.commit().unwrap();
            // Commit the outer
            session.commit().unwrap();
        }

        #[test]
        fn test_commit_without_transaction_error() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();

            let result = session.commit();
            assert!(result.is_err());
        }

        #[test]
        fn test_rollback_without_transaction_error() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();

            let result = session.rollback();
            assert!(result.is_err());
        }

        #[test]
        fn test_create_edge_in_transaction() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();

            // Create nodes outside transaction
            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);

            // Create edge in transaction
            session.begin_transaction().unwrap();
            let edge_id = session.create_edge(alix, gus, "KNOWS");

            // Edge should be visible in the transaction
            assert!(session.edge_exists(edge_id));

            // Commit
            session.commit().unwrap();

            // Edge should still be visible
            assert!(session.edge_exists(edge_id));
        }

        #[test]
        fn test_neighbors_empty_node() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let lonely = session.create_node(&["Person"]);

            assert!(session.get_neighbors_outgoing(lonely).is_empty());
            assert!(session.get_neighbors_incoming(lonely).is_empty());
            assert!(
                session
                    .get_neighbors_outgoing_by_type(lonely, "KNOWS")
                    .is_empty()
            );
        }
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_auto_gc_triggers_on_commit_interval() {
        use crate::config::Config;

        let config = Config::in_memory().with_gc_interval(2);
        let db = GrafeoDB::with_config(config).unwrap();
        let mut session = db.session();

        // First commit: counter = 1, no GC (not a multiple of 2)
        session.begin_transaction().unwrap();
        session.create_node(&["A"]);
        session.commit().unwrap();

        // Second commit: counter = 2, GC should trigger (multiple of 2)
        session.begin_transaction().unwrap();
        session.create_node(&["B"]);
        session.commit().unwrap();

        // Verify the database is still functional after GC
        assert_eq!(db.node_count(), 2);
    }

    #[cfg(all(feature = "lpg", feature = "cdc"))]
    #[test]
    fn savepoint_capture_linearizes_before_a_waiting_mutation() {
        use std::sync::{Arc, mpsc};
        use std::time::Duration;

        let db = GrafeoDB::with_config(crate::Config::in_memory().with_cdc()).unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        let session = Arc::new(session);

        // Hold the private operation gate so the worker can announce its
        // mutation attempt but cannot touch state, WAL, or CDC yet.
        let held = session.mutation_operation_gate.lock();
        let worker_session = Arc::clone(&session);
        let (attempted_tx, attempted_rx) = mpsc::sync_channel(0);
        let (done_tx, done_rx) = mpsc::sync_channel(0);
        let worker = std::thread::spawn(move || {
            attempted_tx.send(()).unwrap();
            let node = worker_session.create_node(&["AfterBoundary"]);
            done_tx.send(node).unwrap();
        });
        attempted_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("mutation worker started");
        assert_eq!(
            done_rx.recv_timeout(Duration::from_millis(50)),
            Err(mpsc::RecvTimeoutError::Timeout),
            "the mutation must remain outside the savepoint boundary"
        );

        // Reentrant acquisition on this thread is intentional: nested
        // transaction helpers can create savepoints while an outer mutation
        // operation owns the same gate.
        session.savepoint("before_waiting_mutation").unwrap();
        drop(held);
        let node = done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("mutation completed after savepoint");
        worker.join().unwrap();

        session
            .rollback_to_savepoint("before_waiting_mutation")
            .unwrap();
        session.commit_inner().unwrap();
        assert!(db.get_node(node).is_none());
        assert!(
            db.history_in_graph(node, &grafeo_common::types::GraphPath::root())
                .unwrap()
                .is_empty()
        );
    }

    #[cfg(all(feature = "lpg", feature = "cdc"))]
    #[test]
    fn graph_context_mutation_uses_the_cdc_operation_gate() {
        use std::sync::{Arc, mpsc};
        use std::time::Duration;

        let db = GrafeoDB::with_config(crate::Config::in_memory().with_cdc()).unwrap();
        db.create_graph("waiting_context")
            .expect("create context target");
        let session = Arc::new(db.session());
        let held = session.mutation_operation_gate.lock();
        let worker_session = Arc::clone(&session);
        let (attempted_tx, attempted_rx) = mpsc::sync_channel(0);
        let (done_tx, done_rx) = mpsc::sync_channel(0);
        let worker = std::thread::spawn(move || {
            attempted_tx.send(()).unwrap();
            worker_session
                .use_graph_path(
                    &grafeo_common::types::GraphPath::from_components(&["waiting_context"])
                        .expect("literal graph path"),
                )
                .expect("select context target");
            done_tx.send(()).unwrap();
        });
        attempted_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("context worker started");
        assert_eq!(
            done_rx.recv_timeout(Duration::from_millis(50)),
            Err(mpsc::RecvTimeoutError::Timeout),
            "graph context must not bisect a CDC mutation operation"
        );
        drop(held);
        done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("context changed after operation boundary");
        worker.join().unwrap();
        assert_eq!(
            session.current_graph_path(),
            grafeo_common::types::GraphPath::from_components(&["waiting_context"])
                .expect("literal graph path")
        );
    }

    #[cfg(all(feature = "lpg", not(feature = "cdc")))]
    #[test]
    fn governed_savepoint_uses_the_operation_gate_without_cdc() {
        use crate::auth::{Grant, Identity, Role};
        use std::sync::{Arc, mpsc};
        use std::time::Duration;

        let db = GrafeoDB::new_in_memory();
        let mut session = db.session_with_identity(
            Identity::new("governed-savepoint", [Role::ReadWrite]).with_grants([Grant::new(
                grafeo_common::types::GraphPath::root(),
                Role::ReadWrite,
            )]),
        );
        session.begin_transaction().unwrap();
        let session = Arc::new(session);
        let held = session.mutation_operation_gate.lock();
        let worker_session = Arc::clone(&session);
        let (done_tx, done_rx) = mpsc::sync_channel(0);
        let worker = std::thread::spawn(move || {
            done_tx
                .send(worker_session.savepoint("governed_boundary"))
                .unwrap();
        });
        assert!(
            matches!(
                done_rx.recv_timeout(Duration::from_millis(50)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ),
            "a governed savepoint must not bisect a concurrent mutation boundary"
        );
        drop(held);
        done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("savepoint completed after operation boundary")
            .unwrap();
        worker.join().unwrap();
    }

    #[cfg(all(feature = "lpg", feature = "cdc"))]
    #[test]
    fn runtime_disabled_cdc_still_serializes_session_mutations() {
        use std::sync::{Arc, mpsc};
        use std::time::Duration;

        let db = GrafeoDB::new_in_memory();
        let session = Arc::new(db.session());
        assert!(!session.identity.has_grants());
        assert!(session.identity.can_write());
        assert!(session.cdc_pending_events.is_none());
        assert!(session.default_cdc_writer.is_none());
        assert!(!session.in_transaction());
        assert_eq!(db.node_count(), 0);
        assert_eq!(db.memory_usage().cdc.event_count, 0);

        let held = session.mutation_operation_gate.lock();
        let worker_session = Arc::clone(&session);
        // Buffered channels let the worker finish even if a bounded receive
        // times out. After spawning, assert only after gate release and join.
        let (attempted_tx, attempted_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let gate_is_held = worker_session.mutation_operation_gate.try_lock().is_none();
            attempted_tx.send(gate_is_held).expect("gate observation");
            done_tx
                .send(worker_session.create_node(&["SerializedWithoutCdc"]))
                .expect("mutation result");
        });

        let gate_observation = attempted_rx.recv_timeout(Duration::from_secs(2));
        let premature_completion = done_rx.recv_timeout(Duration::from_millis(50));
        let remained_blocked =
            matches!(&premature_completion, Err(mpsc::RecvTimeoutError::Timeout));
        let node_count_before_release = db.node_count();
        let transaction_before_release = session.in_transaction();
        let events_before_release = db.memory_usage().cdc.event_count;
        drop(held);
        let completion = match premature_completion {
            Ok(node) => Ok(node),
            Err(_) => done_rx.recv_timeout(Duration::from_secs(2)),
        };
        let joined = worker.join();

        assert_eq!(gate_observation, Ok(true), "worker observed the held gate");
        assert!(
            remained_blocked,
            "runtime-disabled mutation must wait for the Session boundary"
        );
        assert_eq!(node_count_before_release, 0);
        assert!(!transaction_before_release);
        assert_eq!(events_before_release, 0);
        joined.expect("mutation worker joined after gate release");
        let node = completion.expect("mutation completed after gate release");
        assert!(node.is_valid());
        assert!(db.get_node(node).is_some());
        assert_eq!(db.node_count(), 1);
        assert!(!session.in_transaction());
        assert!(session.cdc_pending_events.is_none());
        assert!(session.default_cdc_writer.is_none());
        assert_eq!(db.memory_usage().cdc.event_count, 0);
    }

    #[cfg(all(
        feature = "lpg",
        feature = "triple-store",
        feature = "sparql",
        feature = "cdc"
    ))]
    #[test]
    fn both_mode_runtime_disabled_rdf_mutation_emits_no_cdc() {
        use grafeo_common::types::EpochId;

        let db = GrafeoDB::with_config(
            crate::Config::in_memory().with_graph_model(crate::GraphModel::Both),
        )
        .unwrap();
        let session = db.session();
        assert!(session.cdc_pending_events.is_none());

        session
            .execute_sparql(r#"INSERT DATA { <urn:no-cdc-both> <urn:p> "value" . }"#)
            .expect("runtime-disabled Both-mode RDF mutation");

        assert!(
            db.changes_between(EpochId::INITIAL, EpochId::new(u64::MAX))
                .unwrap()
                .is_empty(),
            "a runtime-disabled RDF planner must not install a direct compatibility sink"
        );
    }

    #[cfg(all(
        not(feature = "lpg"),
        feature = "triple-store",
        feature = "sparql",
        feature = "cdc"
    ))]
    #[test]
    fn rdf_only_runtime_disabled_mutation_emits_no_cdc() {
        use grafeo_common::types::EpochId;

        let db = GrafeoDB::with_config(
            crate::Config::in_memory().with_graph_model(crate::GraphModel::Rdf),
        )
        .unwrap();
        let session = db.session();
        assert!(session.cdc_pending_events.is_none());

        session
            .execute_sparql(r#"INSERT DATA { <urn:no-cdc-rdf> <urn:p> "value" . }"#)
            .expect("runtime-disabled RDF-only mutation");

        assert!(
            db.changes_between(EpochId::INITIAL, EpochId::new(u64::MAX))
                .unwrap()
                .is_empty(),
            "a runtime-disabled RDF planner must not install a direct compatibility sink"
        );
    }

    #[cfg(all(
        not(feature = "lpg"),
        feature = "triple-store",
        feature = "sparql",
        feature = "cdc"
    ))]
    #[test]
    fn rdf_only_gc_applies_retention_after_default_is_disabled() {
        use grafeo_common::types::EpochId;

        let mut config = crate::Config::in_memory()
            .with_graph_model(crate::GraphModel::Rdf)
            .with_cdc();
        config.cdc_retention = crate::cdc::CdcRetentionConfig {
            max_epochs: None,
            max_events: Some(1),
        };
        let db = GrafeoDB::with_config(config).unwrap();
        let session = db.session();
        db.set_cdc_enabled(false);

        for suffix in ["a", "b", "c"] {
            session
                .execute_sparql(&format!(
                    r#"INSERT DATA {{ <urn:retained-{suffix}> <urn:p> "value" . }}"#
                ))
                .expect("existing CDC session remains enabled");
        }
        assert_eq!(
            db.changes_between(EpochId::INITIAL, EpochId::new(u64::MAX))
                .unwrap()
                .len(),
            3
        );

        db.gc().expect("collect retained history");

        assert_eq!(
            db.changes_between(EpochId::INITIAL, EpochId::new(u64::MAX))
                .unwrap()
                .len(),
            1,
            "RDF-only GC must apply retention even after the default for future sessions is disabled"
        );
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn savepoint_rollback_unregisters_post_boundary_graph_trackers() {
        use crate::transaction::IsolationLevel;
        use std::sync::Arc;

        let db = GrafeoDB::new_in_memory();
        db.execute("CREATE GRAPH tracker_graph")
            .expect("create committed tracker target");
        let mut session = db.session();
        let baseline = Arc::strong_count(&session.transaction_manager);

        session
            .begin_transaction_with_isolation(IsolationLevel::Serializable)
            .unwrap();
        assert_eq!(
            Arc::strong_count(&session.transaction_manager),
            baseline + 2
        );
        session.savepoint("before_touch").unwrap();
        session
            .execute("USE GRAPH tracker_graph")
            .expect("select tracker graph");
        assert!(session.create_node(&["Transient"]).is_valid());
        assert_eq!(
            Arc::strong_count(&session.transaction_manager),
            baseline + 4
        );

        session.reset_graph().expect("reset tracker graph");
        session.rollback_to_savepoint("before_touch").unwrap();
        assert_eq!(
            Arc::strong_count(&session.transaction_manager),
            baseline + 2,
            "rollback must unregister exact-store SSI bridges first installed after the savepoint"
        );
        session.commit().unwrap();
        assert_eq!(Arc::strong_count(&session.transaction_manager), baseline);
    }

    #[cfg(all(
        not(feature = "lpg"),
        feature = "triple-store",
        feature = "sparql",
        feature = "cdc"
    ))]
    #[test]
    fn rdf_only_concurrent_auto_commit_uses_the_same_session_operation_gate() {
        use grafeo_common::types::EpochId;
        use grafeo_core::graph::rdf::{Quad, Term, Triple};
        use std::sync::{Arc, Barrier};

        let db = GrafeoDB::with_config(
            crate::Config::in_memory()
                .with_graph_model(crate::GraphModel::Rdf)
                .with_cdc(),
        )
        .unwrap();
        let session = Arc::new(db.session());
        let barrier = Arc::new(Barrier::new(3));
        let mut workers = Vec::new();
        for query in [
            r#"INSERT DATA { <urn:rdf-a> <urn:p> "a" . }"#,
            r#"INSERT DATA { <urn:rdf-b> <urn:p> "b" . }"#,
        ] {
            let session = Arc::clone(&session);
            let barrier = Arc::clone(&barrier);
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                session.execute_sparql(query)
            }));
        }
        barrier.wait();
        for worker in workers {
            worker
                .join()
                .expect("RDF-only mutation worker")
                .expect("RDF-only auto-commit mutation");
        }

        for (subject, object) in [("urn:rdf-a", "a"), ("urn:rdf-b", "b")] {
            let quad = Quad::new(Triple::new(
                Term::iri(subject),
                Term::iri("urn:p"),
                Term::literal(object),
            ));
            assert!(session.try_contains_rdf_quad(&quad).unwrap());
        }
        let changes = db
            .changes_between(EpochId::INITIAL, EpochId::new(u64::MAX))
            .unwrap();
        assert_eq!(changes.len(), 2);
        assert!(changes.iter().all(|event| {
            matches!(event.entity_id, crate::cdc::EntityId::Triple(_))
                && event.epoch != EpochId::PENDING
        }));
        let commit_epochs: std::collections::HashSet<_> =
            changes.iter().map(|event| event.epoch).collect();
        assert_eq!(
            commit_epochs.len(),
            2,
            "concurrent RDF auto-commits must not accidentally join one transaction"
        );
    }

    #[test]
    fn test_query_timeout_config_propagates_to_session() {
        use crate::config::Config;
        use std::time::Duration;

        let config = Config::in_memory().with_query_timeout(Duration::from_secs(5));
        let db = GrafeoDB::with_config(config).unwrap();
        let session = db.session();

        // Verify the session has a query deadline (timeout was set)
        assert!(session.query_deadline().is_some());
    }

    #[test]
    fn test_default_query_timeout_returns_deadline() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        // Default config has 30s timeout
        assert!(session.query_deadline().is_some());
    }

    #[test]
    fn test_no_query_timeout_returns_no_deadline() {
        use crate::config::Config;

        let config = Config::in_memory().without_query_timeout();
        let db = GrafeoDB::with_config(config).unwrap();
        let session = db.session();

        assert!(session.query_deadline().is_none());
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_graph_model_accessor() {
        use crate::config::GraphModel;

        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        assert_eq!(session.graph_model(), GraphModel::Lpg);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_reject_oversized_property() {
        use crate::config::Config;

        let config = Config::in_memory().with_max_property_size(100);
        let db = GrafeoDB::with_config(config).unwrap();
        let session = db.session();

        let node = session.create_node(&["Test"]);

        // Small property should succeed
        session
            .set_node_property(node, "small", Value::from("hello"))
            .unwrap();

        // Large property should be rejected
        let big = "x".repeat(200);
        let result = session.set_node_property(node, "big", Value::from(big.as_str()));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("exceeds maximum size"),
            "Expected size error, got: {err}"
        );
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_no_property_size_limit() {
        use crate::config::Config;

        let config = Config::in_memory().without_max_property_size();
        let db = GrafeoDB::with_config(config).unwrap();
        let session = db.session();

        let node = session.create_node(&["Test"]);

        // Even large properties should succeed with no limit
        let big = "x".repeat(10_000);
        session
            .set_node_property(node, "big", Value::from(big.as_str()))
            .unwrap();
    }

    #[cfg(all(feature = "gql", feature = "lpg"))]
    #[test]
    fn test_external_store_session() {
        use grafeo_core::graph::GraphStoreMut;
        use std::sync::Arc;

        let config = crate::config::Config::in_memory();
        let store = Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap());
        store.create_property_index("name");
        let db =
            GrafeoDB::with_store(Arc::clone(&store) as Arc<dyn GraphStoreMut>, config).unwrap();

        let mut session = db.session();
        assert!(Arc::ptr_eq(&session.store, &store));
        assert!(matches!(
            session.lpg_backend,
            super::LpgBackend::Placeholder {
                commit_target_available: true,
            }
        ));

        // Use an explicit transaction so that INSERT and MATCH share the same
        // transaction context. With PENDING epochs, uncommitted versions are
        // only visible to the owning transaction.
        session.begin_transaction().unwrap();
        let transaction = session.current_transaction.lock().unwrap();
        session.execute("INSERT (:Test {name: 'hello'})").unwrap();

        // Verify we can query through it within the same transaction
        let result = session.execute("MATCH (n:Test) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1);

        session.commit().unwrap();
        assert!(store.pending_node_creates(transaction).is_empty());
        let committed = store.find_nodes_by_property("name", &Value::from("hello"));
        assert_eq!(committed.len(), 1);
        assert_eq!(
            db.session()
                .execute("MATCH (n:Test {name: 'hello'}) RETURN n.name")
                .unwrap()
                .row_count(),
            1,
            "a fresh reader must see the external store's committed index and row"
        );

        session.begin_transaction().unwrap();
        let aborted = session.current_transaction.lock().unwrap();
        session.execute("INSERT (:Test {name: 'aborted'})").unwrap();
        session
            .execute("MATCH (n:Test {name: 'hello'}) SET n.name = 'changed'")
            .unwrap();
        session.rollback().unwrap();
        assert!(store.pending_node_creates(aborted).is_empty());
        assert_eq!(
            store.find_nodes_by_property("name", &Value::from("hello")),
            committed
        );
        assert!(
            store
                .find_nodes_by_property("name", &Value::from("aborted"))
                .is_empty()
        );
        assert_eq!(
            db.session()
                .execute("MATCH (n:Test) RETURN n.name")
                .unwrap()
                .rows,
            vec![vec![Value::from("hello")]],
            "rollback must target the external store, not the placeholder"
        );

        session
            .execute("INSERT (:Test {name: 'automatic'})")
            .unwrap();
        let direct = session.create_node(&["Test"]);
        session
            .set_node_property(direct, "name", Value::from("direct"))
            .unwrap();
        assert_eq!(
            store.find_nodes_by_property("name", &Value::from("direct")),
            vec![direct]
        );
        assert_eq!(
            db.session()
                .execute("MATCH (n:Test) RETURN n.name")
                .unwrap()
                .row_count(),
            3,
            "query and direct auto-commit must publish to the same external target"
        );
    }

    // ==================== Session Command Tests ====================

    #[cfg(all(feature = "gql", feature = "lpg"))]
    mod session_command_tests {
        use super::*;
        use grafeo_common::types::Value;

        #[test]
        fn test_use_graph_sets_current_graph() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create the graph first, then USE it
            session.execute("CREATE GRAPH mydb").unwrap();
            session.execute("USE GRAPH mydb").unwrap();

            assert_eq!(
                session.current_graph_path(),
                grafeo_common::types::GraphPath::from_components(&["mydb"])
                    .expect("literal graph path")
            );
        }

        #[test]
        fn test_use_graph_nonexistent_errors() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("USE GRAPH doesnotexist");
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("does not exist"),
                "Expected 'does not exist' error, got: {err}"
            );
        }

        #[test]
        fn test_use_graph_default_always_valid() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // "default" is always valid, even without CREATE GRAPH
            session.execute("USE GRAPH default").unwrap();
            assert_eq!(
                session.current_graph_path(),
                grafeo_common::types::GraphPath::root()
            );
        }

        #[test]
        fn test_session_set_graph() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH analytics").unwrap();
            session.execute("SESSION SET GRAPH analytics").unwrap();
            assert_eq!(
                session.current_graph_path(),
                grafeo_common::types::GraphPath::from_components(&["analytics"])
                    .expect("literal graph path")
            );
        }

        #[test]
        fn test_session_set_graph_nonexistent_errors() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("SESSION SET GRAPH nosuchgraph");
            assert!(result.is_err());
        }

        #[test]
        fn test_session_set_time_zone() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            assert_eq!(session.time_zone(), None);

            session.execute("SESSION SET TIME ZONE 'UTC'").unwrap();
            assert_eq!(session.time_zone(), Some("UTC".to_string()));

            session
                .execute("SESSION SET TIME ZONE 'America/New_York'")
                .unwrap();
            assert_eq!(session.time_zone(), Some("America/New_York".to_string()));
        }

        #[test]
        fn test_session_set_parameter() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session
                .execute("SESSION SET PARAMETER $timeout = 30")
                .unwrap();

            // Parameter is stored (value is Null for now, since expression
            // evaluation is not yet wired up)
            assert!(session.get_parameter("timeout").is_some());
        }

        #[test]
        fn test_session_reset_clears_all_state() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Set various session state
            session.execute("CREATE GRAPH analytics").unwrap();
            session.execute("SESSION SET GRAPH analytics").unwrap();
            session.execute("SESSION SET TIME ZONE 'UTC'").unwrap();
            session
                .execute("SESSION SET PARAMETER $limit = 100")
                .unwrap();

            // Verify state was set
            assert!(!session.current_graph_path().components().is_empty());
            assert!(session.time_zone().is_some());
            assert!(session.get_parameter("limit").is_some());

            // Reset everything
            session.execute("SESSION RESET").unwrap();

            assert_eq!(
                session.current_graph_path(),
                grafeo_common::types::GraphPath::root()
            );
            assert_eq!(session.time_zone(), None);
            assert!(session.get_parameter("limit").is_none());
        }

        #[test]
        fn test_session_close_clears_state() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH analytics").unwrap();
            session.execute("SESSION SET GRAPH analytics").unwrap();
            session.execute("SESSION SET TIME ZONE 'UTC'").unwrap();

            session.execute("SESSION CLOSE").unwrap();

            assert_eq!(
                session.current_graph_path(),
                grafeo_common::types::GraphPath::root()
            );
            assert_eq!(session.time_zone(), None);
        }

        #[test]
        fn test_create_graph() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH mydb").unwrap();

            // Should be able to USE it now
            session.execute("USE GRAPH mydb").unwrap();
            assert_eq!(
                session.current_graph_path(),
                grafeo_common::types::GraphPath::from_components(&["mydb"])
                    .expect("literal graph path")
            );
        }

        #[test]
        fn test_create_graph_duplicate_errors() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH mydb").unwrap();
            let result = session.execute("CREATE GRAPH mydb");

            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("already exists"),
                "Expected 'already exists' error, got: {err}"
            );
        }

        #[test]
        fn test_create_graph_if_not_exists() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH mydb").unwrap();
            // Should succeed silently with IF NOT EXISTS
            session.execute("CREATE GRAPH IF NOT EXISTS mydb").unwrap();
        }

        #[test]
        fn test_drop_graph() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH mydb").unwrap();
            session.execute("DROP GRAPH mydb").unwrap();

            // Should no longer be usable
            let result = session.execute("USE GRAPH mydb");
            assert!(result.is_err());
        }

        #[test]
        fn test_drop_graph_nonexistent_errors() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("DROP GRAPH nosuchgraph");
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("does not exist"),
                "Expected 'does not exist' error, got: {err}"
            );
        }

        #[test]
        fn test_drop_graph_if_exists() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Should succeed silently with IF EXISTS
            session.execute("DROP GRAPH IF EXISTS nosuchgraph").unwrap();
        }

        #[test]
        fn test_start_transaction_via_gql() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("START TRANSACTION").unwrap();
            assert!(session.in_transaction());
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            session.execute("COMMIT").unwrap();
            assert!(!session.in_transaction());

            let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
            assert_eq!(result.rows.len(), 1);
        }

        #[test]
        fn test_start_transaction_read_only_blocks_insert() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("START TRANSACTION READ ONLY").unwrap();
            let result = session.execute("INSERT (:Person {name: 'Alix'})");
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("read-only"),
                "Expected read-only error, got: {err}"
            );
            session.execute("ROLLBACK").unwrap();
        }

        #[test]
        fn test_start_transaction_read_only_allows_reads() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            session.commit().unwrap();

            session.execute("START TRANSACTION READ ONLY").unwrap();
            let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
            assert_eq!(result.rows.len(), 1);
            session.execute("COMMIT").unwrap();
        }

        #[test]
        fn test_rollback_via_gql() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("START TRANSACTION").unwrap();
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            session.execute("ROLLBACK").unwrap();

            let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
            assert!(result.rows.is_empty());
        }

        #[test]
        fn test_start_transaction_with_isolation_level() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Increment 2f: SERIALIZABLE is now a fully supported OCC-validated
            // isolation level; START TRANSACTION must succeed.
            session
                .execute("START TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                .expect("START TRANSACTION ISOLATION LEVEL SERIALIZABLE must succeed");
            assert!(session.in_transaction());
            session.execute("ROLLBACK").unwrap();

            // Other supported isolation levels continue to work.
            session
                .execute("START TRANSACTION ISOLATION LEVEL READ COMMITTED")
                .unwrap();
            assert!(session.in_transaction());
            session.execute("ROLLBACK").unwrap();
        }

        #[test]
        fn test_session_commands_return_empty_result() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH test").unwrap();
            let result = session.execute("SESSION SET GRAPH test").unwrap();
            assert_eq!(result.row_count(), 0);
            assert_eq!(result.column_count(), 0);
        }

        #[test]
        fn test_current_graph_path_default_is_root() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            assert_eq!(
                session.current_graph_path(),
                grafeo_common::types::GraphPath::root()
            );
        }

        #[test]
        fn test_time_zone_default_is_none() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            assert_eq!(session.time_zone(), None);
        }

        #[test]
        fn test_session_state_independent_across_sessions() {
            let db = GrafeoDB::new_in_memory();
            let session1 = db.session();
            let session2 = db.session();

            session1.execute("CREATE GRAPH first").unwrap();
            session1.execute("CREATE GRAPH second").unwrap();
            session1.execute("SESSION SET GRAPH first").unwrap();
            session2.execute("SESSION SET GRAPH second").unwrap();

            assert_eq!(
                session1.current_graph_path(),
                grafeo_common::types::GraphPath::from_components(&["first"])
                    .expect("literal graph path")
            );
            assert_eq!(
                session2.current_graph_path(),
                grafeo_common::types::GraphPath::from_components(&["second"])
                    .expect("literal graph path")
            );
        }

        #[test]
        fn test_show_node_types() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session
                .execute("CREATE NODE TYPE Person (name STRING NOT NULL, age INTEGER)")
                .unwrap();

            let result = session.execute("SHOW NODE TYPES").unwrap();
            assert_eq!(
                result.columns,
                vec!["name", "properties", "constraints", "parents"]
            );
            assert_eq!(result.rows.len(), 1);
            // First column is the type name
            assert_eq!(result.rows[0][0], Value::from("Person"));
        }

        #[test]
        fn test_show_edge_types() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session
                .execute("CREATE EDGE TYPE KNOWS CONNECTING (Person) TO (Person) (since INTEGER)")
                .unwrap();

            let result = session.execute("SHOW EDGE TYPES").unwrap();
            assert_eq!(
                result.columns,
                vec!["name", "properties", "source_types", "target_types"]
            );
            assert_eq!(result.rows.len(), 1);
            assert_eq!(result.rows[0][0], Value::from("KNOWS"));
        }

        #[test]
        fn test_show_graph_types() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session
                .execute("CREATE NODE TYPE Person (name STRING)")
                .unwrap();
            session
                .execute(
                    "CREATE GRAPH TYPE social (\
                        NODE TYPE Person (name STRING)\
                    )",
                )
                .unwrap();

            let result = session.execute("SHOW GRAPH TYPES").unwrap();
            assert_eq!(
                result.columns,
                vec!["name", "open", "node_types", "edge_types"]
            );
            assert_eq!(result.rows.len(), 1);
            assert_eq!(result.rows[0][0], Value::from("social"));
        }

        #[test]
        fn test_show_graph_type_named() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session
                .execute("CREATE NODE TYPE Person (name STRING)")
                .unwrap();
            session
                .execute(
                    "CREATE GRAPH TYPE social (\
                        NODE TYPE Person (name STRING)\
                    )",
                )
                .unwrap();

            let result = session.execute("SHOW GRAPH TYPE social").unwrap();
            assert_eq!(result.rows.len(), 1);
            assert_eq!(result.rows[0][0], Value::from("social"));
        }

        #[test]
        fn test_show_graph_type_not_found() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("SHOW GRAPH TYPE nonexistent");
            assert!(result.is_err());
        }

        #[test]
        fn test_show_indexes_via_gql() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("SHOW INDEXES").unwrap();
            assert_eq!(result.columns, vec!["name", "type", "label", "property"]);
        }

        #[test]
        fn test_show_constraints_via_gql() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("SHOW CONSTRAINTS").unwrap();
            assert_eq!(result.columns, vec!["name", "type", "label", "properties"]);
        }

        #[test]
        fn test_pattern_form_graph_type_roundtrip() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Register the types first
            session
                .execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
                .unwrap();
            session
                .execute("CREATE NODE TYPE City (name STRING)")
                .unwrap();
            session
                .execute("CREATE EDGE TYPE KNOWS (since INTEGER)")
                .unwrap();
            session.execute("CREATE EDGE TYPE LIVES_IN").unwrap();

            // Create graph type using pattern form
            session
                .execute(
                    "CREATE GRAPH TYPE social (\
                        (:Person {name STRING NOT NULL})-[:KNOWS {since INTEGER}]->(:Person),\
                        (:Person)-[:LIVES_IN]->(:City)\
                    )",
                )
                .unwrap();

            // Verify it was created
            let result = session.execute("SHOW GRAPH TYPE social").unwrap();
            assert_eq!(result.rows.len(), 1);
            assert_eq!(result.rows[0][0], Value::from("social"));
        }

        /// Two independently prepared target generations can both be valid
        /// against their earlier inventory. Publication-time revalidation must
        /// allow exactly one winner and reject the later commit after it sees
        /// the winner's now-committed duplicate.
        #[cfg(all(feature = "triple-store", feature = "sparql"))]
        #[test]
        fn concurrent_rdf_projection_target_commits_have_one_winner() {
            use std::collections::BTreeSet;

            use crate::{Config, GraphModel};
            use grafeo_core::graph::rdf::{
                RDF_LPG_PROJECTION_IRI_PROPERTY, RDF_LPG_PROJECTION_OWNER_PROPERTY,
            };

            const TYPE_IRI: &str = "http://ex.org/Person";
            const IRI: &str = "http://ex.org/alix";

            let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both))
                .unwrap();
            let projection_id = db.declare_rdf_lpg_projection(TYPE_IRI, "Person").unwrap();
            let definition = db.rdf_lpg_projection(projection_id).unwrap();
            let desired = BTreeSet::from([IRI.to_string()]);

            let mut earlier = db.session();
            earlier
                .authorize_rdf_projection_rebuild(
                    projection_id,
                    definition.owner_marker(),
                    "Person".to_string(),
                    desired.clone(),
                )
                .unwrap();
            earlier.begin_transaction().unwrap();
            earlier
                .create_node_with_props(
                    &["Person"],
                    [
                        (RDF_LPG_PROJECTION_IRI_PROPERTY, Value::from(IRI)),
                        (
                            RDF_LPG_PROJECTION_OWNER_PROPERTY,
                            Value::from(definition.owner_marker()),
                        ),
                    ],
                )
                .unwrap();

            // This generation commits after `earlier` has inventoried/planned
            // the empty target, but before `earlier` reaches publication.
            let mut winner = db.session();
            winner
                .authorize_rdf_projection_rebuild(
                    projection_id,
                    definition.owner_marker(),
                    "Person".to_string(),
                    desired,
                )
                .unwrap();
            winner.begin_transaction().unwrap();
            winner
                .create_node_with_props(
                    &["Person"],
                    [
                        (RDF_LPG_PROJECTION_IRI_PROPERTY, Value::from(IRI)),
                        (
                            RDF_LPG_PROJECTION_OWNER_PROPERTY,
                            Value::from(definition.owner_marker()),
                        ),
                    ],
                )
                .unwrap();
            winner.commit().expect("first publisher wins");

            let error = earlier
                .commit()
                .expect_err("stale target plan must not publish a duplicate");
            assert!(
                error.to_string().contains("2 owned rows"),
                "unexpected exact-target rejection: {error}"
            );
            assert_eq!(
                db.node_count(),
                1,
                "losing generation must roll back its pending duplicate"
            );
        }

        /// A generation that retained an existing row from its inventory must
        /// not report success if another authorized generation commits a delete
        /// before the first reaches publication.
        #[cfg(all(feature = "triple-store", feature = "sparql"))]
        #[test]
        fn rdf_projection_target_rejects_delete_after_inventory() {
            use std::collections::BTreeSet;

            use crate::{Config, GraphModel};
            use grafeo_core::graph::rdf::{
                RDF_LPG_PROJECTION_IRI_PROPERTY, RDF_LPG_PROJECTION_OWNER_PROPERTY,
            };

            const TYPE_IRI: &str = "http://ex.org/Person";
            const IRI: &str = "http://ex.org/alix";

            let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both))
                .unwrap();
            let projection_id = db.declare_rdf_lpg_projection(TYPE_IRI, "Person").unwrap();
            let definition = db.rdf_lpg_projection(projection_id).unwrap();
            let owner_marker = definition.owner_marker();
            let node_id = {
                let mut seed = db.session();
                seed.authorize_rdf_projection_rebuild(
                    projection_id,
                    owner_marker.clone(),
                    "Person".to_string(),
                    BTreeSet::from([IRI.to_string()]),
                )
                .unwrap();
                seed.begin_transaction().unwrap();
                let node_id = seed
                    .create_node_with_props(
                        &["Person"],
                        [
                            (RDF_LPG_PROJECTION_IRI_PROPERTY, Value::from(IRI)),
                            (
                                RDF_LPG_PROJECTION_OWNER_PROPERTY,
                                Value::from(owner_marker.clone()),
                            ),
                        ],
                    )
                    .unwrap();
                seed.commit().unwrap();
                node_id
            };

            // `retainer` represents a rebuild that inventoried the row and
            // therefore planned no target mutation.
            let mut retainer = db.session();
            retainer
                .authorize_rdf_projection_rebuild(
                    projection_id,
                    owner_marker.clone(),
                    "Person".to_string(),
                    BTreeSet::from([IRI.to_string()]),
                )
                .unwrap();
            retainer.begin_transaction().unwrap();

            // A newer authorized generation publishes an empty desired set.
            let mut remover = db.session();
            remover
                .authorize_rdf_projection_rebuild(
                    projection_id,
                    owner_marker,
                    "Person".to_string(),
                    BTreeSet::new(),
                )
                .unwrap();
            remover.begin_transaction().unwrap();
            assert!(remover.delete_node(node_id));
            remover.commit().expect("delete generation publishes first");

            let error = retainer
                .commit()
                .expect_err("retained stale inventory must not publish success");
            assert!(
                error.to_string().contains("missing desired IRI"),
                "unexpected exact-target rejection: {error}"
            );
            assert_eq!(db.node_count(), 0);
        }
    }
}
