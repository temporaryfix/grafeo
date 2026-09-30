//! RDF-specific session methods.
//!
//! This module consolidates all RDF functionality from the session layer.
//! The entire module is gated behind `#[cfg(feature = "triple-store")]` in the parent.

use std::sync::Arc;
#[cfg(feature = "lpg")]
use std::sync::atomic::AtomicUsize;
#[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
use std::time::Instant;

use grafeo_common::types::TransactionId;
#[cfg(any(feature = "sparql", feature = "graphql"))]
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, Result};
#[cfg(feature = "sparql")]
use grafeo_common::utils::error::{QueryError, QueryErrorKind};
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::LpgStore;
#[cfg(feature = "lpg")]
use grafeo_core::graph::rdf::RdfStore;
#[cfg(feature = "lpg")]
use grafeo_core::graph::{GraphStoreMut, GraphStoreSearch};

use crate::database::QueryResult;

use super::Session;
#[cfg(feature = "lpg")]
use super::SessionConfig;

impl Session {
    /// Adds the RDF-specific post-execution qualification boundary around the
    /// model-dispatched transaction wrapper.
    ///
    /// This seam is shared by RDF-only and dual-model Sessions. Keeping the
    /// test hook here avoids changing GQL's independent entry-counter semantics
    /// when both models are enabled.
    fn with_rdf_auto_commit<F>(&self, has_mutations: bool, body: F) -> Result<QueryResult>
    where
        F: FnOnce() -> Result<QueryResult>,
    {
        #[cfg(not(feature = "testing-statement-injection"))]
        {
            self.with_auto_commit(has_mutations, body)
        }
        #[cfg(feature = "testing-statement-injection")]
        {
            self.with_auto_commit(has_mutations, || {
                let result = body()?;
                if has_mutations {
                    grafeo_common::testing::statement_failure::maybe_fail_statement().map_err(
                        |error| {
                            Error::Internal(format!(
                                "injected post-execution statement failure: {error}"
                            ))
                        },
                    )?;
                }
                Ok(result)
            })
        }
    }

    /// Records the conservative dataset-wide RDF predicate read used by SSI.
    ///
    /// RDF patterns include arbitrary terms, named graphs, property paths, and
    /// graph-lifecycle observations. One coarse key is intentionally used until
    /// finer keys can prove complete coverage: a false conflict may reduce
    /// concurrency, while a missed key can admit write skew or a phantom.
    fn record_rdf_serializable_read(&self) -> Result<()> {
        let Some(transaction_id) = *self.current_transaction.lock() else {
            return Ok(());
        };
        if self.transaction_manager.isolation_level(transaction_id)
            == Some(crate::transaction::IsolationLevel::Serializable)
        {
            self.transaction_manager.record_read(
                transaction_id,
                crate::transaction::EntityId::RdfDataset,
                None,
            )?;
        }
        Ok(())
    }

    /// Records a dataset mutation as a coarse SSI predicate write.
    fn record_rdf_serializable_write(&self) -> Result<()> {
        let Some(transaction_id) = *self.current_transaction.lock() else {
            return Ok(());
        };
        if self.transaction_manager.isolation_level(transaction_id)
            == Some(crate::transaction::IsolationLevel::Serializable)
        {
            self.transaction_manager.record_coarse_write(
                transaction_id,
                crate::transaction::EntityId::RdfDataset,
                None,
            )?;
        }
        Ok(())
    }

    /// Records an RDF read dependency and, for mutating plans, its write.
    fn record_rdf_serializable_access(&self, has_mutations: bool) -> Result<()> {
        self.record_rdf_serializable_read()?;
        if has_mutations {
            self.record_rdf_serializable_write()?;
        }
        Ok(())
    }

    /// Executes an optimized GraphQL-RDF plan at the Session transaction and
    /// publication boundary. The planner is intentionally created inside the
    /// auto-commit closure so a mutating plan receives its implicit RDF
    /// transaction ID before physical operators are built.
    #[cfg(feature = "graphql")]
    fn execute_graphql_rdf_plan(
        &self,
        optimized_plan: crate::query::plan::LogicalPlan,
    ) -> Result<QueryResult> {
        use crate::query::planner::rdf::RdfPlanner;

        let has_mutations = optimized_plan.root.has_mutations();
        if has_mutations {
            self.require_permission(crate::auth::StatementKind::Write)?;
            if *self.read_only_tx.lock() {
                return Err(Error::Transaction(
                    grafeo_common::utils::error::TransactionError::ReadOnly,
                ));
            }
        }

        let make_planner = || {
            let planner = RdfPlanner::new(Arc::clone(&self.rdf_store))
                .with_transaction_id(*self.current_transaction.lock())
                .with_valid_time(*self.rdf_valid_time.lock());
            #[cfg(feature = "wal")]
            let planner =
                planner.with_wal_poison(self.wal.clone(), Arc::clone(&self.durability_poisoned));
            #[cfg(feature = "cdc")]
            let planner = {
                let pending_events = self.cdc_pending_events.clone();
                let cdc_log = pending_events.as_ref().map(|_| Arc::clone(&self.cdc_log));
                planner.with_cdc_accumulator(cdc_log, pending_events)
            };
            planner
        };

        if optimized_plan.explain {
            let _read_barrier = self.publication_read_guard();
            self.check_active_execution()?;
            let (_, entries) = make_planner().plan_profiled(&optimized_plan)?;
            self.check_active_execution()?;
            use crate::query::processor::physical_explain_result;
            return self.finish_query(physical_explain_result(
                &optimized_plan,
                entries,
                self.result_resources()?,
                self.effective_result_limits(),
            ));
        }

        let result = self.with_rdf_auto_commit(has_mutations, || {
            self.record_rdf_serializable_access(has_mutations)?;
            let _read_barrier = if !has_mutations {
                self.publication_read_guard()
            } else {
                None
            };

            if optimized_plan.profile {
                self.check_active_execution()?;
                let (mut physical_plan, entries) = make_planner().plan_profiled(&optimized_plan)?;
                self.check_active_execution()?;
                let profile_start = std::time::Instant::now();
                let _ = self.execute_borrowed_profiled_plan(&mut physical_plan, &entries)?;
                let elapsed_ms = profile_start.elapsed().as_secs_f64() * 1000.0;
                // Stats are retained by entries. Release the finished query's
                // account before admitting the separate diagnostic result.
                drop(physical_plan);
                let tree = crate::query::profile::build_profile_tree(
                    &optimized_plan.root,
                    &mut entries.into_iter(),
                );
                return crate::query::profile::profile_result(
                    &tree,
                    elapsed_ms,
                    self.result_resources()?,
                    self.effective_result_limits(),
                );
            }

            self.check_active_execution()?;
            let mut physical_plan = make_planner().plan(&optimized_plan)?;
            self.check_active_execution()?;
            self.execute_borrowed_physical_plan(&mut physical_plan)
        });

        self.finish_query(result)
    }

    /// Creates a new session with RDF store and adaptive configuration.
    #[cfg(feature = "lpg")]
    pub(crate) fn with_rdf_store_and_adaptive(
        store: Arc<LpgStore>,
        rdf_store: Arc<RdfStore>,
        cfg: SessionConfig,
    ) -> Self {
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreSearch>;
        let graph_store_mut = Some(Arc::clone(&store) as Arc<dyn GraphStoreMut>);
        Self {
            store,
            lpg_backend: super::LpgBackend::Active,
            graph_store,
            graph_store_mut,
            catalog: cfg.catalog,
            rdf_store,
            transaction_manager: cfg.transaction_manager,
            query_cache: cfg.query_cache,
            physical_cache: cfg.physical_cache,
            current_transaction: parking_lot::Mutex::new(None),
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
            active_streams: AtomicUsize::new(0),
            #[cfg(feature = "wal")]
            wal: None,
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            #[cfg(feature = "cdc")]
            cdc_pending_events: None,
            #[cfg(all(feature = "lpg", feature = "cdc"))]
            default_cdc_writer: None,
            current_context: parking_lot::Mutex::new(super::SessionGraphContext::default()),
            time_zone: parking_lot::Mutex::new(None),
            rdf_valid_time: parking_lot::Mutex::new(None),
            session_params: parking_lot::Mutex::new(std::collections::HashMap::new()),
            viewing_epoch_override: parking_lot::Mutex::new(None),
            savepoints: parking_lot::Mutex::new(Vec::new()),
            transaction_nesting_depth: parking_lot::Mutex::new(0),
            touched_graphs: parking_lot::Mutex::new(Vec::new()),
            pending_created_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            pending_dropped_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            cancelled_created_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            touched_named_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            missing_named_graphs: parking_lot::Mutex::new(std::collections::HashSet::new()),
            superseded_graph_touches: parking_lot::Mutex::new(Vec::new()),
            pending_graph_type_bindings: parking_lot::Mutex::new(std::collections::HashMap::new()),
            pending_index_ddl: parking_lot::Mutex::new(Vec::new()),
            pending_projection_ddl: parking_lot::Mutex::new(std::collections::HashMap::new()),
            projection_registry_snapshot: parking_lot::Mutex::new(None),
            projection_registry_read: std::sync::atomic::AtomicBool::new(false),
            #[cfg(all(feature = "wal", feature = "lpg"))]
            catalog_wal_batch: parking_lot::Mutex::new(None),
            rdf_projection_target: parking_lot::Mutex::new(None),
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

    /// Executes a GraphQL query against the RDF store.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql_rdf(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            optimizer::Optimizer, processor::substitute_params, translators::graphql_rdf,
        };

        let _execution_operation = self.session_operation_guard();
        if let Some(options) = self.fresh_execution_options(Some("graphql-rdf")) {
            return self.execute_with_options(query, std::collections::HashMap::new(), options);
        }

        self.check_not_poisoned()?;
        self.require_rdf("GraphQL-RDF")?;
        self.record_rdf_serializable_read()?;
        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        self.check_active_execution()?;
        let mut logical_plan = graphql_rdf::translate(query, "http://example.org/")?;
        self.check_active_execution()?;
        if !logical_plan.default_params.is_empty() {
            let defaults = logical_plan.default_params.clone();
            self.check_active_execution()?;
            substitute_params(&mut logical_plan, &defaults)?;
            self.check_active_execution()?;
        }
        let rdf_stats = self.rdf_store.get_or_collect_statistics();
        let optimizer = Optimizer::from_rdf_statistics((*rdf_stats).clone());
        self.check_active_execution()?;
        let optimized_plan = optimizer.optimize(logical_plan)?;
        self.check_active_execution()?;
        let result = self.execute_graphql_rdf_plan(optimized_plan);

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("graphql-rdf", elapsed_ms, &result);
        }

        result
    }

    /// Executes a GraphQL query against the RDF store with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql_rdf_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::{
            optimizer::Optimizer, processor::substitute_params, translators::graphql_rdf,
        };

        let _execution_operation = self.session_operation_guard();
        if let Some(options) = self.fresh_execution_options(Some("graphql-rdf")) {
            return self.execute_with_options(query, params, options);
        }

        self.check_not_poisoned()?;
        self.require_rdf("GraphQL-RDF")?;
        self.record_rdf_serializable_read()?;
        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        // Parse and translate the query to a logical plan
        self.check_active_execution()?;
        let mut logical_plan = graphql_rdf::translate(query, "http://example.org/")?;
        self.check_active_execution()?;

        // Caller values override GraphQL variable defaults.
        if logical_plan.default_params.is_empty() {
            self.check_active_execution()?;
            substitute_params(&mut logical_plan, &params)?;
            self.check_active_execution()?;
        } else {
            let mut merged = logical_plan.default_params.clone();
            merged.extend(
                params
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone())),
            );
            self.check_active_execution()?;
            substitute_params(&mut logical_plan, &merged)?;
            self.check_active_execution()?;
        }

        // Optimize the plan
        let rdf_stats = self.rdf_store.get_or_collect_statistics();
        let optimizer = Optimizer::from_rdf_statistics((*rdf_stats).clone());
        self.check_active_execution()?;
        let optimized_plan = optimizer.optimize(logical_plan)?;
        self.check_active_execution()?;

        let result = self.execute_graphql_rdf_plan(optimized_plan);

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("graphql-rdf", elapsed_ms, &result);
        }

        result
    }

    /// Executes a SPARQL query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "sparql")]
    pub fn execute_sparql(&self, query: &str) -> Result<QueryResult> {
        let _execution_operation = self.session_operation_guard();
        if let Some(options) = self.fresh_execution_options(Some("sparql")) {
            return self.execute_with_options(query, std::collections::HashMap::new(), options);
        }
        self.execute_sparql_with_publication(query, true)
    }

    /// Executes SPARQL while a caller-owned publication guard pins the cut.
    /// Used by SHACL-SPARQL so recursive query execution does not attempt a
    /// second read lock behind a waiting writer.
    #[cfg(all(feature = "sparql", feature = "shacl"))]
    pub(crate) fn execute_sparql_at_pinned_publication(&self, query: &str) -> Result<QueryResult> {
        self.execute_sparql_with_publication(query, false)
    }

    #[cfg(feature = "sparql")]
    fn execute_sparql_with_publication(
        &self,
        query: &str,
        pin_publication: bool,
    ) -> Result<QueryResult> {
        use crate::query::{optimizer::Optimizer, planner::rdf::RdfPlanner, translators::sparql};

        self.require_rdf("SPARQL")?;
        self.check_not_poisoned()?;
        self.record_rdf_serializable_read()?;
        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        self.check_active_execution()?;
        let logical_plan = sparql::translate(query)?;
        self.check_active_execution()?;
        let has_mutations = logical_plan.root.has_mutations();
        if !pin_publication && has_mutations {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Unsupported,
                "SHACL-SPARQL constraints must be read-only".to_string(),
            )));
        }
        // Planning collects live RDF statistics. Pin the same publication cut
        // through planning and execution for reads/EXPLAIN so the plan cannot
        // describe one dataset and execute against another.
        let _read_barrier = if pin_publication && (!has_mutations || logical_plan.explain) {
            self.publication_read_guard()
        } else {
            None
        };
        let rdf_stats = self.rdf_store.get_or_collect_statistics();
        let optimizer = Optimizer::from_rdf_statistics((*rdf_stats).clone());
        self.check_active_execution()?;
        let optimized_plan = optimizer.optimize(logical_plan)?;
        self.check_active_execution()?;

        // Check role-based permission for mutations (skip tree walk for admin)
        if !self.identity.can_admin() && optimized_plan.root.has_mutations() {
            self.require_permission(crate::auth::StatementKind::Write)?;
        }

        let make_planner = || {
            let planner = RdfPlanner::new(Arc::clone(&self.rdf_store))
                .with_transaction_id(*self.current_transaction.lock())
                .with_valid_time(*self.rdf_valid_time.lock());
            #[cfg(feature = "wal")]
            let planner =
                planner.with_wal_poison(self.wal.clone(), Arc::clone(&self.durability_poisoned));
            #[cfg(feature = "cdc")]
            let planner = {
                let pending_events = self.cdc_pending_events.clone();
                let cdc_log = pending_events.as_ref().map(|_| Arc::clone(&self.cdc_log));
                planner.with_cdc_accumulator(cdc_log, pending_events)
            };
            planner
        };

        // EXPLAIN is planned at the same checked Session boundary as execution.
        if optimized_plan.explain {
            self.check_active_execution()?;
            let (_, entries) = make_planner().plan_profiled(&optimized_plan)?;
            self.check_active_execution()?;
            use crate::query::processor::physical_explain_result;
            return self.finish_query(physical_explain_result(
                &optimized_plan,
                entries,
                self.result_resources()?,
                self.effective_result_limits(),
            ));
        }

        debug_assert_eq!(has_mutations, optimized_plan.root.has_mutations());
        let result = self.with_rdf_auto_commit(has_mutations, || {
            self.record_rdf_serializable_access(has_mutations)?;
            if optimized_plan.profile {
                self.check_active_execution()?;
                let (mut physical_plan, entries) = make_planner().plan_profiled(&optimized_plan)?;
                self.check_active_execution()?;
                let profile_start = std::time::Instant::now();
                let _ = self.execute_borrowed_profiled_plan(&mut physical_plan, &entries)?;
                let elapsed_ms = profile_start.elapsed().as_secs_f64() * 1000.0;
                // Stats are retained by entries. Release the finished query's
                // account before admitting the separate diagnostic result.
                drop(physical_plan);
                let tree = crate::query::profile::build_profile_tree(
                    &optimized_plan.root,
                    &mut entries.into_iter(),
                );
                return crate::query::profile::profile_result(
                    &tree,
                    elapsed_ms,
                    self.result_resources()?,
                    self.effective_result_limits(),
                );
            }

            self.check_active_execution()?;
            let mut physical_plan = make_planner().plan(&optimized_plan)?;
            self.check_active_execution()?;
            self.execute_borrowed_physical_plan(&mut physical_plan)
        });
        self.check_not_poisoned()?;

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("sparql", elapsed_ms, &result);
        }

        self.finish_query(result)
    }

    /// Executes a SPARQL query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "sparql")]
    pub fn execute_sparql_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::{
            optimizer::Optimizer, planner::rdf::RdfPlanner, processor::substitute_params,
            translators::sparql,
        };

        let _execution_operation = self.session_operation_guard();
        if let Some(options) = self.fresh_execution_options(Some("sparql")) {
            return self.execute_with_options(query, params, options);
        }

        self.require_rdf("SPARQL")?;
        self.check_not_poisoned()?;
        self.record_rdf_serializable_read()?;
        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        self.check_active_execution()?;
        let mut logical_plan = sparql::translate(query)?;
        self.check_active_execution()?;
        substitute_params(&mut logical_plan, &params)?;
        self.check_active_execution()?;

        let has_mutations = logical_plan.root.has_mutations();
        let _read_barrier = if !has_mutations || logical_plan.explain {
            self.publication_read_guard()
        } else {
            None
        };

        let rdf_stats = self.rdf_store.get_or_collect_statistics();
        let optimizer = Optimizer::from_rdf_statistics((*rdf_stats).clone());
        self.check_active_execution()?;
        let optimized_plan = optimizer.optimize(logical_plan)?;
        self.check_active_execution()?;

        // Check role-based permission for mutations (skip tree walk for admin)
        if !self.identity.can_admin() && optimized_plan.root.has_mutations() {
            self.require_permission(crate::auth::StatementKind::Write)?;
        }

        let make_planner = || {
            let planner = RdfPlanner::new(Arc::clone(&self.rdf_store))
                .with_transaction_id(*self.current_transaction.lock())
                .with_valid_time(*self.rdf_valid_time.lock());
            #[cfg(feature = "wal")]
            let planner =
                planner.with_wal_poison(self.wal.clone(), Arc::clone(&self.durability_poisoned));
            #[cfg(feature = "cdc")]
            let planner = {
                let pending_events = self.cdc_pending_events.clone();
                let cdc_log = pending_events.as_ref().map(|_| Arc::clone(&self.cdc_log));
                planner.with_cdc_accumulator(cdc_log, pending_events)
            };
            planner
        };

        if optimized_plan.explain {
            self.check_active_execution()?;
            let (_, entries) = make_planner().plan_profiled(&optimized_plan)?;
            self.check_active_execution()?;
            use crate::query::processor::physical_explain_result;
            return self.finish_query(physical_explain_result(
                &optimized_plan,
                entries,
                self.result_resources()?,
                self.effective_result_limits(),
            ));
        }

        debug_assert_eq!(has_mutations, optimized_plan.root.has_mutations());
        let result = self.with_rdf_auto_commit(has_mutations, || {
            self.record_rdf_serializable_access(has_mutations)?;
            if optimized_plan.profile {
                self.check_active_execution()?;
                let (mut physical_plan, entries) = make_planner().plan_profiled(&optimized_plan)?;
                self.check_active_execution()?;
                let profile_start = std::time::Instant::now();
                let _ = self.execute_borrowed_profiled_plan(&mut physical_plan, &entries)?;
                let elapsed_ms = profile_start.elapsed().as_secs_f64() * 1000.0;
                // Stats are retained by entries. Release the finished query's
                // account before admitting the separate diagnostic result.
                drop(physical_plan);
                let tree = crate::query::profile::build_profile_tree(
                    &optimized_plan.root,
                    &mut entries.into_iter(),
                );
                return crate::query::profile::profile_result(
                    &tree,
                    elapsed_ms,
                    self.result_resources()?,
                    self.effective_result_limits(),
                );
            }

            self.check_active_execution()?;
            let mut physical_plan = make_planner().plan(&optimized_plan)?;
            self.check_active_execution()?;
            self.execute_borrowed_physical_plan(&mut physical_plan)
        });
        self.check_not_poisoned()?;

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("sparql", elapsed_ms, &result);
        }

        self.finish_query(result)
    }

    /// Inserts RDF triples through the Session mutation chokepoint.
    ///
    /// Autocommit when no tx is open. Logs WAL when attached. Skips duplicates.
    ///
    /// # Errors
    ///
    /// Returns an error if RDF is not enabled, the session lacks a write grant
    /// for the default graph, the session is poisoned, or WAL fails.
    pub fn insert_rdf_batch(
        &self,
        triples: impl IntoIterator<Item = grafeo_core::graph::rdf::Triple>,
    ) -> Result<usize> {
        let valid = *self.rdf_valid_time.lock();
        self.insert_rdf_batch_with_valid(triples, valid)
    }

    /// Sets the application valid-time captured by subsequent RDF inserts.
    ///
    /// The scope applies uniformly to statement-creating SPARQL INSERT paths,
    /// default-graph triples, and typed quads in named graphs. SPARQL
    /// COPY/MOVE/ADD preserve each source statement's valid-time instead of
    /// relabelling copied facts with the current scope. It is a session setting:
    /// pending operations retain the interval captured when they were inserted,
    /// while rollback discards those operations normally.
    pub fn set_rdf_valid_time(&self, valid: Option<grafeo_common::types::ValidTimeInterval>) {
        *self.rdf_valid_time.lock() = valid;
    }

    /// Valid-time currently captured by RDF insert operations.
    #[must_use]
    pub fn rdf_valid_time(&self) -> Option<grafeo_common::types::ValidTimeInterval> {
        *self.rdf_valid_time.lock()
    }

    /// Sets a canonical signed TAI-nanosecond valid-time interval.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error unless `valid_from_tai_ns < valid_to_tai_ns`.
    pub fn set_rdf_valid_time_tai_ns(
        &self,
        valid_from_tai_ns: i128,
        valid_to_tai_ns: i128,
    ) -> Result<()> {
        let valid = grafeo_common::types::ValidTimeInterval::from_tai_nanoseconds(
            valid_from_tai_ns,
            valid_to_tai_ns,
        )
        .map_err(|error| Error::InvalidValue(error.to_string()))?;
        self.set_rdf_valid_time(Some(valid));
        Ok(())
    }

    /// Clears the session valid-time scope so subsequent RDF inserts are
    /// always-valid on the application-time axis.
    pub fn clear_rdf_valid_time(&self) {
        self.set_rdf_valid_time(None);
    }

    /// Inserts RDF triples with an explicitly captured application valid-time.
    pub(crate) fn insert_rdf_batch_with_valid(
        &self,
        triples: impl IntoIterator<Item = grafeo_core::graph::rdf::Triple>,
        valid: Option<grafeo_common::types::ValidTimeInterval>,
    ) -> Result<usize> {
        use grafeo_core::graph::rdf::TriplePattern;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let triples: Vec<_> = triples.into_iter().collect();
        if triples.is_empty() {
            return Ok(0);
        }
        self.require_rdf("RDF insert")?;
        self.check_not_poisoned()?;
        self.require_rdf_graph_grant(None, crate::auth::Role::ReadWrite, "write")?;
        let inserted = AtomicUsize::new(0);
        self.with_rdf_auto_commit(true, || {
            self.record_rdf_serializable_access(true)?;
            let tid = *self.current_transaction.lock();
            let mut n = 0usize;
            for t in &triples {
                let pattern = TriplePattern {
                    subject: Some(t.subject().clone()),
                    predicate: Some(t.predicate().clone()),
                    object: Some(t.object().clone()),
                };
                if !self.rdf_store.find_with_pending(&pattern, tid).is_empty() {
                    continue;
                }
                if let Some(id) = tid {
                    self.rdf_store
                        .insert_in_transaction_with_valid(id, t.clone(), valid);
                } else {
                    self.rdf_store.try_insert_at_epoch_with_valid(
                        t.clone(),
                        self.rdf_store.commit_epoch(),
                        valid,
                    )?;
                }
                #[cfg(feature = "wal")]
                if self.wal.is_some() {
                    let transaction_id = tid.unwrap_or(TransactionId::SYSTEM);
                    let (valid_from_tai_ns, valid_to_tai_ns) = valid
                        .map_or((None, None), |valid| {
                            (Some(valid.from().as_i128()), Some(valid.to().as_i128()))
                        });
                    let record = grafeo_storage::wal::WalRecord::InsertRdfQuadV3 {
                        subject: t.subject().to_string(),
                        predicate: t.predicate().to_string(),
                        object: t.object().to_string(),
                        graph: None,
                        graph_incarnation: grafeo_common::types::GraphIncarnationId::DEFAULT_GRAPH,
                        valid_from_tai_ns,
                        valid_to_tai_ns,
                        transaction_id,
                    };
                    self.log_wal_record(&record)?;
                }
                n += 1;
            }
            inserted.store(n, Ordering::Relaxed);
            Ok(crate::database::QueryResult::empty())
        })?;
        Ok(inserted.load(Ordering::Relaxed))
    }

    /// Inserts RDF quads (triple + optional named graph) through the Session/WAL chokepoint.
    ///
    /// Autocommit when no tx is open. Skips duplicates already visible in the
    /// target graph. Returns the number of newly inserted quads.
    ///
    /// # Errors
    ///
    /// Returns an error if RDF is not enabled, the session lacks a write grant
    /// for any target graph, the session is poisoned, or WAL fails. Every
    /// target grant is checked before an implicit transaction is opened.
    pub fn insert_rdf_quads(
        &self,
        quads: impl IntoIterator<Item = grafeo_core::graph::rdf::Quad>,
    ) -> Result<usize> {
        use grafeo_core::graph::rdf::TriplePattern;
        #[cfg(feature = "wal")]
        use std::sync::atomic::AtomicU64;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let quads: Vec<_> = quads.into_iter().collect();
        if quads.is_empty() {
            return Ok(0);
        }
        self.require_rdf("RDF insert")?;
        self.check_not_poisoned()?;
        let mut targets = grafeo_common::utils::hash::FxHashSet::default();
        for quad in &quads {
            if targets.insert(quad.graph()) {
                self.require_rdf_graph_grant(quad.graph(), crate::auth::Role::ReadWrite, "write")?;
            }
        }
        let valid = *self.rdf_valid_time.lock();
        let inserted = AtomicUsize::new(0);
        #[cfg(feature = "wal")]
        let logged_graph_high_water = AtomicU64::new(0);
        self.with_rdf_auto_commit(true, || {
            self.record_rdf_serializable_access(true)?;
            let tid = *self.current_transaction.lock();
            let mut n = 0usize;
            let mut seen = grafeo_common::utils::hash::FxHashSet::default();
            for q in &quads {
                if !seen.insert(q) {
                    continue;
                }
                let t = q.triple();
                let target = match q.graph() {
                    Some(name) => self
                        .rdf_store
                        .graph_or_create_in_tx(name, tid)
                        .map_err(|error| Error::Internal(error.to_string()))?,
                    None => Arc::clone(&self.rdf_store),
                };
                let already = if tid.is_some() {
                    let pattern = TriplePattern {
                        subject: Some(t.subject().clone()),
                        predicate: Some(t.predicate().clone()),
                        object: Some(t.object().clone()),
                    };
                    !target.find_with_pending(&pattern, tid).is_empty()
                } else {
                    target.contains(t)
                };
                if already {
                    continue;
                }
                if let Some(id) = tid {
                    target.insert_in_transaction_with_valid(id, t.clone(), valid);
                } else {
                    target.try_insert_at_epoch_with_valid(
                        t.clone(),
                        target.commit_epoch(),
                        valid,
                    )?;
                }
                #[cfg(feature = "wal")]
                if self.wal.is_some() {
                    let transaction_id = tid.unwrap_or(TransactionId::SYSTEM);
                    if q.graph().is_some() {
                        let next = self.rdf_store.next_graph_incarnation();
                        if next.as_u64() > logged_graph_high_water.load(Ordering::Relaxed) {
                            self.log_wal_record(
                                &grafeo_storage::wal::WalRecord::RdfGraphIncarnationHighWaterMeta {
                                    store_id: self.rdf_store.store_id(),
                                    next_incarnation: next,
                                },
                            )?;
                            logged_graph_high_water.store(next.as_u64(), Ordering::Relaxed);
                        }
                    }
                    let (valid_from_tai_ns, valid_to_tai_ns) = valid
                        .map_or((None, None), |valid| {
                            (Some(valid.from().as_i128()), Some(valid.to().as_i128()))
                        });
                    let record = grafeo_storage::wal::WalRecord::InsertRdfQuadV3 {
                        subject: t.subject().to_string(),
                        predicate: t.predicate().to_string(),
                        object: t.object().to_string(),
                        graph: q.graph().map(str::to_string),
                        graph_incarnation: target.graph_incarnation(),
                        valid_from_tai_ns,
                        valid_to_tai_ns,
                        transaction_id,
                    };
                    self.log_wal_record(&record)?;
                }
                n += 1;
            }
            inserted.store(n, Ordering::Relaxed);
            Ok(crate::database::QueryResult::empty())
        })?;
        Ok(inserted.load(Ordering::Relaxed))
    }

    /// Exact typed-quad membership (lexical form + datatype + graph).
    ///
    /// Read-only. An open transaction sees its own pending writes and hides
    /// pending deletes. Other sessions see only committed quads. Serializable
    /// transactions also record the conservative dataset predicate read used
    /// by SSI. This infallible method is retained for source compatibility and
    /// returns `false` when lifecycle or SSI tracking fails; correctness-
    /// sensitive callers should use [`try_contains_rdf_quad`](Self::try_contains_rdf_quad).
    #[must_use]
    pub fn contains_rdf_quad(&self, quad: &grafeo_core::graph::rdf::Quad) -> bool {
        self.try_contains_rdf_quad(quad).unwrap_or(false)
    }

    fn contains_rdf_quad_unchecked(&self, quad: &grafeo_core::graph::rdf::Quad) -> bool {
        use grafeo_core::graph::rdf::TriplePattern;
        let tid = *self.current_transaction.lock();
        let t = quad.triple();
        let pattern = TriplePattern {
            subject: Some(t.subject().clone()),
            predicate: Some(t.predicate().clone()),
            object: Some(t.object().clone()),
        };
        let target = match quad.graph() {
            Some(name) => match self.rdf_store.graph_in_transaction(name, tid) {
                Some(g) => g,
                None => return false,
            },
            None => Arc::clone(&self.rdf_store),
        };
        !target.find_with_pending(&pattern, tid).is_empty()
    }

    /// Fallible exact typed-quad membership, including Serializable SSI read
    /// tracking.
    ///
    /// # Errors
    ///
    /// Returns an error if the database is LPG-only or closed, the session
    /// lacks an exact read grant for the target graph, durability is poisoned
    /// after a WAL failure, or Serializable SSI tracking finds that the current
    /// transaction is missing or no longer active.
    pub fn try_contains_rdf_quad(&self, quad: &grafeo_core::graph::rdf::Quad) -> Result<bool> {
        self.require_rdf("RDF quad membership")?;
        self.check_not_poisoned()?;
        self.require_rdf_graph_grant(quad.graph(), crate::auth::Role::ReadOnly, "read")?;
        self.record_rdf_serializable_read()?;
        let _publication = self.publication_read_guard();
        Ok(self.contains_rdf_quad_unchecked(quad))
    }

    /// Commits RDF transaction state.
    ///
    /// Called from the main commit path to finalize RDF changes.
    /// Active RDF transaction id, if any (for pending SHACL / search snapshots).
    #[cfg(feature = "shacl")]
    #[must_use]
    pub(crate) fn current_rdf_transaction(&self) -> Option<TransactionId> {
        *self.current_transaction.lock()
    }

    pub(super) fn commit_rdf_transaction(
        &self,
        transaction_id: TransactionId,
        epoch: grafeo_common::types::EpochId,
    ) -> Result<usize> {
        self.transaction_manager.with_write_authority(|| {
            self.rdf_store
                .try_commit_dataset_at_under_gate(transaction_id, epoch)
        })
    }

    pub(super) fn validate_rdf_transaction_lifecycle(
        &self,
        transaction_id: TransactionId,
    ) -> Result<()> {
        self.rdf_store
            .validate_transaction_lifecycle(transaction_id)
            .map_err(|message| {
                grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::WriteConflict(message),
                )
            })
    }

    /// Rolls back RDF transaction state.
    ///
    /// Called from the main commit-conflict and rollback paths to discard RDF changes.
    pub(super) fn rollback_rdf_transaction(&self, transaction_id: TransactionId) {
        self.rdf_store.rollback_dataset(transaction_id);
    }

    /// Validates the default graph against SHACL shapes in a named graph.
    ///
    /// # Errors
    ///
    /// Returns an error if shape parsing fails or the shapes graph doesn't exist.
    #[cfg(feature = "shacl")]
    pub fn validate_shacl(
        &self,
        shapes_graph: &str,
    ) -> Result<grafeo_core::graph::rdf::shacl::ValidationReport> {
        self.require_rdf("SHACL validation")?;
        self.check_not_poisoned()?;
        self.record_rdf_serializable_read()?;
        let _publication = self.publication_read_guard();
        crate::validation::validate_shacl(self, &self.rdf_store, shapes_graph)
    }

    /// Validates a named data graph against shapes in another named graph.
    ///
    /// Both SHACL Core constraints and SHACL-SPARQL constraints are scoped to
    /// the named data graph (SPARQL queries receive `FROM <data_graph_name>`).
    ///
    /// # Errors
    ///
    /// Returns an error if shape parsing fails or either graph doesn't exist.
    #[cfg(feature = "shacl")]
    pub fn validate_shacl_graph(
        &self,
        data_graph_name: &str,
        shapes_graph_name: &str,
    ) -> Result<grafeo_core::graph::rdf::shacl::ValidationReport> {
        self.require_rdf("SHACL validation")?;
        self.check_not_poisoned()?;
        self.record_rdf_serializable_read()?;
        let _publication = self.publication_read_guard();
        let tid = self.current_rdf_transaction();
        let data_store = self
            .rdf_store
            .graph_in_transaction(data_graph_name, tid)
            .ok_or_else(|| {
                grafeo_common::utils::error::Error::Internal(format!(
                    "Named graph '{data_graph_name}' not found"
                ))
            })?;
        let shapes_store = self
            .rdf_store
            .graph_in_transaction(shapes_graph_name, tid)
            .ok_or_else(|| {
                grafeo_common::utils::error::Error::Internal(format!(
                    "Named graph '{shapes_graph_name}' not found"
                ))
            })?;
        let executor =
            crate::validation::SessionSparqlExecutor::with_graph(self, data_graph_name.to_string());
        let snap = std::sync::Arc::new(grafeo_core::graph::rdf::RdfStore::new());
        let pattern = grafeo_core::graph::rdf::TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        };
        for t in data_store.find_with_pending(&pattern, tid) {
            snap.insert((*t).clone());
        }
        let shapes_snapshot = crate::validation::rdf_snapshot_with_pending(&shapes_store, tid);
        grafeo_core::graph::rdf::shacl::validate(&snap, &shapes_snapshot, Some(&executor))
            .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))
    }
}

#[cfg(all(test, feature = "graphql"))]
mod graphql_rdf_boundary_tests {
    use super::*;
    use crate::query::plan::{InsertTripleOp, LogicalOperator, LogicalPlan, TripleComponent};
    use crate::{Config, GrafeoDB, GraphModel};

    fn insert_plan() -> LogicalPlan {
        LogicalPlan::new(LogicalOperator::InsertTriple(InsertTripleOp {
            subject: TripleComponent::Iri("http://example.org/alix".to_string()),
            predicate: TripleComponent::Iri("http://example.org/name".to_string()),
            object: TripleComponent::Literal(Value::String("Alix".into())),
            graph: None,
            input: None,
        }))
    }

    #[test]
    fn graphql_rdf_boundary_autocommits_a_mutating_logical_plan() {
        let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))
            .expect("open RDF database");
        let session = db.session();

        session
            .execute_graphql_rdf_plan(insert_plan())
            .expect("execute mutating RDF plan");
        assert!(
            !session.in_transaction(),
            "auto-commit must close the implicit RDF transaction"
        );

        let quad = grafeo_core::graph::rdf::Quad::new(grafeo_core::graph::rdf::Triple::new(
            grafeo_core::graph::rdf::Term::iri("http://example.org/alix"),
            grafeo_core::graph::rdf::Term::iri("http://example.org/name"),
            grafeo_core::graph::rdf::Term::literal("Alix"),
        ));
        assert!(
            session
                .try_contains_rdf_quad(&quad)
                .expect("read committed quad"),
            "the auto-committed plan must be visible at the published epoch"
        );
    }

    #[test]
    fn graphql_rdf_boundary_rejects_mutation_without_write_permission() {
        let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))
            .expect("open RDF database");
        let session = db.session_with_role(crate::auth::Role::ReadOnly);

        let error = session
            .execute_graphql_rdf_plan(insert_plan())
            .expect_err("a Session without write permission must reject an RDF mutation plan");
        assert!(
            error
                .to_string()
                .to_ascii_lowercase()
                .contains("permission denied"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn graphql_rdf_boundary_rejects_mutation_in_read_only_transaction() {
        let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))
            .expect("open RDF database");
        let session = db.session();
        *session.read_only_tx.lock() = true;

        let error = session
            .execute_graphql_rdf_plan(insert_plan())
            .expect_err("a read-only transaction must reject an RDF mutation plan");
        assert!(
            error.to_string().to_ascii_lowercase().contains("read-only"),
            "unexpected error: {error}"
        );
    }
}
