//! Transaction support for the Node.js API.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use parking_lot::{Mutex, MutexGuard, RwLock};

use grafeo_engine::database::GrafeoDB;

use crate::error::{NodeGrafeoError, NodeResult};
use crate::query::QueryResult;

#[cfg(not(any(
    feature = "lpg",
    feature = "triple-store",
    feature = "embedded",
    feature = "edge",
    feature = "compact-store"
)))]
fn transactions_unavailable() -> grafeo_common::Error {
    use grafeo_common::utils::error::{QueryError, QueryErrorKind};
    grafeo_common::Error::Query(QueryError::new(
        QueryErrorKind::Unsupported,
        "transactions require a native LPG or RDF store feature",
    ))
}

/// A database transaction with explicit commit/rollback.
///
/// In Node.js 22+, use with `using` for automatic cleanup:
/// ```js
/// using tx = db.beginTransaction();
/// await tx.execute("INSERT (:Person {name: 'Alix'})");
/// tx.commit();
/// // auto-rollback if commit not called
/// ```
#[napi]
pub struct Transaction {
    state: Arc<Mutex<TransactionState>>,
}

struct TransactionState {
    // The session drops before the database keepalive, including when the last
    // owner is an execution worker after the JavaScript wrapper was collected.
    session: grafeo_engine::session::Session,
    committed: bool,
    rolled_back: bool,
    busy: bool,
    _database: Arc<RwLock<GrafeoDB>>,
}

/// Reserves the transaction synchronously, before option getters or Promise
/// scheduling can reenter commit/rollback. Only owned native data crosses threads.
struct ExecutionLease {
    state: Arc<Mutex<TransactionState>>,
}

impl Drop for ExecutionLease {
    fn drop(&mut self) {
        // The worker's guard is scoped inside its callback and has dropped
        // before the lease, including during unwind. On admission failure no
        // worker exists yet, so releasing this lease cannot block JavaScript.
        self.state.lock().busy = false;
    }
}

#[napi]
impl Transaction {
    /// Execute a GQL query within this transaction.
    #[napi(
        ts_args_type = "query: string, params?: any | undefined | null, options?: ExecutionOptions | undefined | null"
    )]
    pub fn execute<'env>(
        &self,
        env: &'env Env,
        query: String,
        params: Option<serde_json::Value>,
        options: Option<Object<'_>>,
    ) -> Result<PromiseRaw<'env, QueryResult>> {
        self.execute_language_impl(env, "gql", query, params, options)
    }

    /// Commit the transaction. Fails promptly while a query is queued/running.
    #[napi]
    pub fn commit(&self, env: Env) -> Result<i64> {
        #[cfg(any(
            feature = "lpg",
            feature = "triple-store",
            feature = "embedded",
            feature = "edge",
            feature = "compact-store"
        ))]
        {
            let mut state = self.idle_state()?;
            let epoch = state
                .session
                .commit()
                .map_err(|error| crate::error::native_to_js_error(&env, error))?;
            state.committed = true;
            Ok(epoch.as_u64().cast_signed())
        }
        #[cfg(not(any(
            feature = "lpg",
            feature = "triple-store",
            feature = "embedded",
            feature = "edge",
            feature = "compact-store"
        )))]
        {
            Err(crate::error::native_to_js_error(
                &env,
                transactions_unavailable(),
            ))
        }
    }

    /// Roll back the transaction. Fails promptly while a query is queued/running.
    #[napi]
    pub fn rollback(&self, env: Env) -> Result<()> {
        #[cfg(any(
            feature = "lpg",
            feature = "triple-store",
            feature = "embedded",
            feature = "edge",
            feature = "compact-store"
        ))]
        {
            let mut state = self.idle_state()?;
            state
                .session
                .rollback()
                .map_err(|error| crate::error::native_to_js_error(&env, error))?;
            state.rolled_back = true;
            Ok(())
        }
        #[cfg(not(any(
            feature = "lpg",
            feature = "triple-store",
            feature = "embedded",
            feature = "edge",
            feature = "compact-store"
        )))]
        {
            Err(crate::error::native_to_js_error(
                &env,
                transactions_unavailable(),
            ))
        }
    }

    /// Whether this transaction remains active (including during a query).
    #[napi(getter, js_name = "isActive")]
    pub fn is_active(&self) -> bool {
        self.state
            .try_lock()
            .is_none_or(|state| !state.committed && !state.rolled_back)
    }
}

impl Transaction {
    fn idle_state(&self) -> Result<MutexGuard<'_, TransactionState>> {
        let state = self.state.try_lock().ok_or_else(|| {
            napi::Error::from(NodeGrafeoError::Transaction(
                "Transaction is busy executing a query".into(),
            ))
        })?;
        if state.busy {
            return Err(NodeGrafeoError::Transaction(
                "Transaction is busy executing a query".into(),
            )
            .into());
        }
        if state.committed {
            return Err(NodeGrafeoError::Transaction("Already committed".into()).into());
        }
        if state.rolled_back {
            return Err(NodeGrafeoError::Transaction("Already rolled back".into()).into());
        }
        Ok(state)
    }

    /// Shared worker ownership for every language-specific query facade.
    fn execute_language_impl<'env>(
        &self,
        env: &'env Env,
        language: &str,
        query: String,
        params: Option<serde_json::Value>,
        options: Option<Object<'_>>,
    ) -> Result<PromiseRaw<'env, QueryResult>> {
        {
            let mut state = self.idle_state()?;
            state.busy = true;
        }
        let lease = ExecutionLease {
            state: Arc::clone(&self.state),
        };
        let (params, options) = crate::database::prepare_node_query(language, params, options)?;
        crate::error::spawn_execution(env, async move {
            tokio::task::spawn_blocking(move || -> NodeResult<QueryResult> {
                let result = {
                    let state = lease.state.lock();
                    let result = state
                        .session
                        .execute_with_options(&query, params, options.native)
                        .map_err(NodeGrafeoError::from)?;
                    crate::database::finish_node_result(result, options.max_bytes)
                };
                // Release before Promise resolution so the continuation may
                // immediately commit or issue the next query.
                drop(lease);
                result
            })
            .await
            .map_err(|error| NodeGrafeoError::Database(error.to_string()))?
        })
    }

    #[cfg(any(
        feature = "lpg",
        feature = "triple-store",
        feature = "embedded",
        feature = "edge",
        feature = "compact-store"
    ))]
    pub(crate) fn new(
        db: Arc<RwLock<GrafeoDB>>,
        isolation_level: Option<&str>,
    ) -> NodeResult<Self> {
        // Parse isolation level string
        let level = match isolation_level {
            Some("read_committed") => {
                Some(grafeo_engine::transaction::IsolationLevel::ReadCommitted)
            }
            Some("serializable") => Some(grafeo_engine::transaction::IsolationLevel::Serializable),
            Some("snapshot") | None => None, // snapshot is the default
            Some(other) => {
                return Err(NodeGrafeoError::InvalidArgument(format!(
                    "Unknown isolation level '{}'. Use 'read_committed', 'snapshot', or 'serializable'",
                    other
                )));
            }
        };

        let mut session = {
            let db_guard = db.read();
            db_guard.session()
        };

        if let Some(level) = level {
            session
                .begin_transaction_with_isolation(level)
                .map_err(NodeGrafeoError::from)?;
        } else {
            session.begin_transaction().map_err(NodeGrafeoError::from)?;
        }

        Ok(Self {
            state: Arc::new(Mutex::new(TransactionState {
                session,
                committed: false,
                rolled_back: false,
                busy: false,
                _database: db,
            })),
        })
    }

    #[cfg(not(any(
        feature = "lpg",
        feature = "triple-store",
        feature = "embedded",
        feature = "edge",
        feature = "compact-store"
    )))]
    pub(crate) fn new(
        _db: Arc<RwLock<GrafeoDB>>,
        _isolation_level: Option<&str>,
    ) -> NodeResult<Self> {
        Err(NodeGrafeoError::Native(transactions_unavailable()))
    }
}

// There is deliberately no locking Drop on the JavaScript wrapper. An active
// ExecutionLease retains state; the last owner drops Session, whose native Drop
// rolls back an uncommitted transaction after any worker has finished.

#[cfg(feature = "triple-store")]
#[napi]
impl Transaction {
    /// Insert one RDF quad in this transaction.
    #[napi(js_name = "insertRdfQuad")]
    pub fn insert_rdf_quad(
        &self,
        subject: String,
        predicate: String,
        object: String,
        graph: Option<String>,
    ) -> Result<u32> {
        let state = self.idle_state()?;
        let quad =
            crate::database::parse_node_rdf_quad(&subject, &predicate, &object, graph.as_deref())?;
        let session = &state.session;
        let n = session
            .insert_rdf_quads([quad])
            .map_err(NodeGrafeoError::from)?;
        crate::database::rdf_insert_count(n)
    }

    /// Bulk-insert RDF quads. Each item is `[subject, predicate, object]` or
    /// `[subject, predicate, object, graph]`.
    #[napi(
        js_name = "insertRdfQuads",
        ts_args_type = "quads: Array<[string, string, string] | [string, string, string, string]>"
    )]
    pub fn insert_rdf_quads(&self, quads: Vec<Vec<String>>) -> Result<u32> {
        let state = self.idle_state()?;
        let parsed = crate::database::parse_node_quad_list(&quads)?;
        let session = &state.session;
        let n = session
            .insert_rdf_quads(parsed)
            .map_err(NodeGrafeoError::from)?;
        crate::database::rdf_insert_count(n)
    }

    /// Exact typed-quad membership in this transaction.
    #[napi(js_name = "containsRdfQuad")]
    pub fn contains_rdf_quad(
        &self,
        subject: String,
        predicate: String,
        object: String,
        graph: Option<String>,
    ) -> Result<bool> {
        let state = self.idle_state()?;
        let quad =
            crate::database::parse_node_rdf_quad(&subject, &predicate, &object, graph.as_deref())?;
        let session = &state.session;
        Ok(session
            .try_contains_rdf_quad(&quad)
            .map_err(NodeGrafeoError::from)?)
    }
}

#[cfg(any(
    feature = "lpg",
    feature = "embedded",
    feature = "edge",
    feature = "native",
    feature = "compact-store"
))]
#[napi]
impl Transaction {
    /// Create a node inside this transaction (parser-free LPG mutation).
    #[napi(js_name = "createNode")]
    pub fn create_node(&self, labels: Vec<String>) -> Result<i64> {
        let state = self.idle_state()?;
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let session = &state.session;
        let id = session.create_node(&label_refs);
        if !id.is_valid() {
            return Err(NodeGrafeoError::Database("Failed to create node".into()).into());
        }
        i64::try_from(id.as_u64()).map_err(|_| {
            NodeGrafeoError::Database("Node ID exceeds the signed 64-bit result range".into())
                .into()
        })
    }
}

// Language-specific execute methods in separate impl blocks so `#[napi]`
// only generates C callback symbols when the feature is active.

#[cfg(feature = "cypher")]
#[napi]
impl Transaction {
    /// Execute a Cypher query within this transaction.
    #[napi(
        js_name = "executeCypher",
        ts_args_type = "query: string, params?: any | undefined | null, options?: ExecutionOptions | undefined | null"
    )]
    pub fn execute_cypher<'env>(
        &self,
        env: &'env Env,
        query: String,
        params: Option<serde_json::Value>,
        options: Option<Object<'_>>,
    ) -> Result<PromiseRaw<'env, QueryResult>> {
        self.execute_language_impl(env, "cypher", query, params, options)
    }
}

#[cfg(feature = "sql-pgq")]
#[napi]
impl Transaction {
    /// Execute a SQL/PGQ query (SQL:2023 GRAPH_TABLE) within this transaction.
    #[napi(
        js_name = "executeSql",
        ts_args_type = "query: string, params?: any | undefined | null, options?: ExecutionOptions | undefined | null"
    )]
    pub fn execute_sql<'env>(
        &self,
        env: &'env Env,
        query: String,
        params: Option<serde_json::Value>,
        options: Option<Object<'_>>,
    ) -> Result<PromiseRaw<'env, QueryResult>> {
        self.execute_language_impl(env, "sql", query, params, options)
    }
}

#[cfg(feature = "gremlin")]
#[napi]
impl Transaction {
    /// Execute a Gremlin query within this transaction.
    #[napi(
        js_name = "executeGremlin",
        ts_args_type = "query: string, params?: any | undefined | null, options?: ExecutionOptions | undefined | null"
    )]
    pub fn execute_gremlin<'env>(
        &self,
        env: &'env Env,
        query: String,
        params: Option<serde_json::Value>,
        options: Option<Object<'_>>,
    ) -> Result<PromiseRaw<'env, QueryResult>> {
        self.execute_language_impl(env, "gremlin", query, params, options)
    }
}

#[cfg(feature = "graphql")]
#[napi]
impl Transaction {
    /// Execute a GraphQL query within this transaction.
    #[napi(
        js_name = "executeGraphql",
        ts_args_type = "query: string, params?: any | undefined | null, options?: ExecutionOptions | undefined | null"
    )]
    pub fn execute_graphql<'env>(
        &self,
        env: &'env Env,
        query: String,
        params: Option<serde_json::Value>,
        options: Option<Object<'_>>,
    ) -> Result<PromiseRaw<'env, QueryResult>> {
        self.execute_language_impl(env, "graphql", query, params, options)
    }
}

#[cfg(feature = "sparql")]
#[napi]
impl Transaction {
    /// Execute a SPARQL query within this transaction.
    #[napi(
        js_name = "executeSparql",
        ts_args_type = "query: string, params?: any | undefined | null, options?: ExecutionOptions | undefined | null"
    )]
    pub fn execute_sparql<'env>(
        &self,
        env: &'env Env,
        query: String,
        params: Option<serde_json::Value>,
        options: Option<Object<'_>>,
    ) -> Result<PromiseRaw<'env, QueryResult>> {
        self.execute_language_impl(env, "sparql", query, params, options)
    }
}
