//! Query execution methods for GrafeoDB.

use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};

use super::{FromValue, QueryResult};

impl super::GrafeoDB {
    /// Executes a closure with a one-shot session, syncing graph context back
    /// to the database afterward. This ensures `USE GRAPH`, `SESSION SET GRAPH`,
    /// and `SESSION RESET` persist across one-shot `execute()` calls.
    fn with_session<T, F>(&self, func: F) -> Result<T>
    where
        F: FnOnce(&crate::session::Session) -> Result<T>,
    {
        let session = self.session();
        let initial_context = session.graph_context_snapshot();
        let result = func(&session);
        let merge = self.merge_session_context_changes(&session, &initial_context);
        match result {
            Ok(result) => {
                merge?;
                Ok(result)
            }
            Err(error) => {
                // Preserve the query's original error. A valid earlier
                // session-state change is still merged; an invalidated
                // selection is simply declined.
                let _ = merge;
                Err(error)
            }
        }
    }

    /// Merges only context fields changed by this one-shot call.
    ///
    /// Temporary sessions take a coherent copy of both independent fields.
    /// Blindly copying both fields back would let an unchanged stale copy
    /// overwrite a concurrent `SET`/`RESET` of the other field. Changes are
    /// merged even when query execution returned an error because an earlier
    /// session command in that call may already have succeeded.
    fn merge_session_context_changes(
        &self,
        session: &crate::session::Session,
        initial: &crate::session::SessionGraphContext,
    ) -> Result<()> {
        let final_context = session.graph_context_snapshot();
        let graph_changed = final_context.graph != initial.graph
            || final_context.native != initial.native
            || (final_context.native && final_context.storage_key != initial.storage_key);
        let schema_changed = final_context.schema != initial.schema;
        if !graph_changed && !schema_changed {
            return Ok(());
        }

        let _publication = crate::session::acquire_publication_read(&self.transaction_manager);
        let mut current = self.current_context.write();
        let mut next = current.clone();
        if graph_changed {
            next.graph.clone_from(&final_context.graph);
            next.native = final_context.native;
            if next.native {
                next.storage_key.clone_from(&final_context.storage_key);
            }
        }
        if schema_changed {
            next.schema.clone_from(&final_context.schema);
        }

        // The Session selector released its read barrier before this facade
        // merge. Revalidate only selections made by this call so a concurrent
        // DROP cannot publish between those two boundaries and leave a stale
        // context. The unchanged sibling field remains independent by GQL.
        if schema_changed && let Some(name) = next.schema.as_deref() {
            let canonical = self
                .catalog
                .schema_names()
                .into_iter()
                .find(|registered| registered.eq_ignore_ascii_case(name))
                .ok_or_else(|| {
                    Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Schema '{name}' was dropped before session context publication"),
                    ))
                })?;
            next.schema = Some(canonical);
        }

        // Resolve against the latest sibling field, not the temporary
        // session's stale snapshot. Native paths never acquire a schema prefix.
        if !next.native || schema_changed {
            let path = crate::session::Session::context_graph_path(
                next.schema.as_deref(),
                next.graph.as_deref(),
            )?;
            if !next.native {
                next.storage_key = path;
            }
        }

        #[cfg(feature = "lpg")]
        if graph_changed && let Some(store) = &self.store {
            let mut target = std::sync::Arc::clone(store);
            for name in next.storage_key.components() {
                target = target.graph(name).ok_or_else(|| {
                    Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!(
                            "Graph {:?} does not exist at session context publication",
                            next.storage_key
                        ),
                    ))
                })?;
            }
        }

        *current = next;
        Ok(())
    }

    /// Runs a query directly on the database.
    ///
    /// Creates a temporary session for the call. Parsed and physical plans
    /// are cached on the database, so repeating the same read query via
    /// `execute` skips parse + physicalize. Grab a
    /// [`session()`](Self::session) for transactions or other session-local
    /// state.
    ///
    /// Graph context commands (`USE GRAPH`, `SESSION SET GRAPH`, `SESSION RESET`)
    /// persist across calls: running `execute("USE GRAPH analytics")` followed
    /// by `execute("MATCH (n) RETURN n")` routes the second query to the
    /// analytics graph.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing or execution fails.
    pub fn execute(&self, query: &str) -> Result<QueryResult> {
        self.with_session(|s| s.execute(query))
    }

    /// Opens a lazy read-only GQL stream with explicit close and terminal errors.
    ///
    /// PROFILE exposes live metrics through `profile()` alongside query rows.
    ///
    /// # Errors
    /// Returns parsing, admission, planning, resource or cancellation errors.
    #[cfg(all(feature = "gql", feature = "lpg"))]
    pub fn execute_streaming(
        &self,
        query: &str,
    ) -> Result<crate::query::executor::stream::OwnedResultStream> {
        self.stream_with_options(query, std::collections::HashMap::new(), Default::default())
    }

    /// Opens an owned GQL read cursor with parameters and cancellation control.
    ///
    /// The cursor retains its publication cut independently of the temporary
    /// Session. Explicit close releases that cut without dropping the cursor.
    ///
    /// # Errors
    /// Returns the same typed errors as [`crate::Session::stream_with_options`].
    #[cfg(all(feature = "gql", feature = "lpg"))]
    pub fn stream_with_options(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
        options: crate::query::ExecutionOptions,
    ) -> Result<crate::query::executor::stream::OwnedResultStream> {
        use crate::query::executor::stream::OwnedResultStream;
        let session = self.session();
        session.check_not_poisoned()?;
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
        let control = session.compose_execution_control(control)?;
        let publication = session.streaming_owned_publication_read_guard(&control)?;
        let execution = session
            .build_streaming_plan(query, params, control, language.as_deref())?
            .with_result_limits(result_limits);
        Ok(OwnedResultStream::new(execution, publication))
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
    pub fn execute_at_epoch(
        &self,
        query: &str,
        epoch: grafeo_common::types::EpochId,
    ) -> Result<QueryResult> {
        self.with_session(|s| s.execute_at_epoch(query, epoch))
    }

    /// Executes a query with parameters and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn execute_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
    ) -> Result<QueryResult> {
        self.with_session(|s| s.execute_with_params(query, params))
    }

    /// Executes a query with caller-owned cancellation and language selection.
    ///
    /// # Errors
    ///
    /// Returns query admission, execution, deadline or cancellation errors.
    pub fn execute_with_options(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
        options: crate::query::ExecutionOptions,
    ) -> Result<QueryResult> {
        self.with_session(|s| s.execute_with_options(query, params, options))
    }

    /// Executes an ordinary request or prepares owned, scheduled GQL ORDER BY work.
    ///
    /// Call preparation on a blocking worker. The prepared branch owns its
    /// publication guard and may then execute without borrowing this database.
    /// Other shapes use the existing execution and context-publication path.
    ///
    /// # Errors
    /// Returns the ordinary planning, execution, cancellation or admission error.
    #[cfg(all(
        feature = "gql",
        feature = "lpg",
        feature = "spill",
        feature = "async-storage"
    ))]
    pub fn execute_or_prepare_async_sort(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
        options: crate::query::ExecutionOptions,
    ) -> Result<crate::query::executor::AsyncSortDispatch> {
        use crate::query::executor::AsyncSortDispatch;
        if options
            .language
            .as_deref()
            .is_some_and(|name| !name.eq_ignore_ascii_case("gql"))
            || !query
                .as_bytes()
                .windows(5)
                .any(|word| word.eq_ignore_ascii_case(b"order"))
        {
            return self
                .execute_with_options(query, params, options)
                .map(AsyncSortDispatch::Completed);
        }
        self.with_session(|session| {
            let mut options = Some(options);
            if let Some(dispatch) = session.try_prepare_async_sort(query, &params, &mut options)? {
                return Ok(dispatch);
            }
            let options = options
                .ok_or_else(|| Error::Internal("async fallback lost execution options".into()))?;
            session
                .execute_with_options(query, params, options)
                .map(AsyncSortDispatch::Completed)
        })
    }

    /// Executes a Cypher query and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "cypher")]
    pub fn execute_cypher(&self, query: &str) -> Result<QueryResult> {
        self.with_session(|s| s.execute_cypher(query))
    }

    /// Executes a Cypher query with parameters and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "cypher")]
    pub fn execute_cypher_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
    ) -> Result<QueryResult> {
        self.with_session(|s| s.execute_language(query, "cypher", Some(params)))
    }

    /// Executes a Gremlin query and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "gremlin")]
    pub fn execute_gremlin(&self, query: &str) -> Result<QueryResult> {
        self.with_session(|s| s.execute_gremlin(query))
    }

    /// Executes a Gremlin query with parameters and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "gremlin")]
    pub fn execute_gremlin_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
    ) -> Result<QueryResult> {
        self.with_session(|s| s.execute_gremlin_with_params(query, params))
    }

    /// Executes a GraphQL query and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql(&self, query: &str) -> Result<QueryResult> {
        self.with_session(|s| s.execute_graphql(query))
    }

    /// Executes a GraphQL query with parameters and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
    ) -> Result<QueryResult> {
        self.with_session(|s| s.execute_graphql_with_params(query, params))
    }

    /// Executes a SQL/PGQ query (SQL:2023 GRAPH_TABLE) and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "sql-pgq")]
    pub fn execute_sql(&self, query: &str) -> Result<QueryResult> {
        self.with_session(|s| s.execute_sql(query))
    }

    /// Executes a SQL/PGQ query with parameters and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "sql-pgq")]
    pub fn execute_sql_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
    ) -> Result<QueryResult> {
        self.with_session(|s| s.execute_sql_with_params(query, params))
    }

    /// Executes a query in the specified language by name.
    ///
    /// Supported language names: `"gql"`, `"cypher"`, `"gremlin"`, `"graphql"`,
    /// `"sparql"`, `"sql"`. Each requires the corresponding feature flag.
    ///
    /// # Errors
    ///
    /// Returns an error if the language is unknown/disabled, or if the query
    /// fails.
    pub fn execute_language(
        &self,
        query: &str,
        language: &str,
        params: Option<std::collections::HashMap<String, grafeo_common::types::Value>>,
    ) -> Result<QueryResult> {
        self.with_session(|s| s.execute_language(query, language, params))
    }

    /// Executes a query and returns a single scalar value.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails or doesn't return exactly one row.
    pub fn query_scalar<T: FromValue>(&self, query: &str) -> Result<T> {
        let result = self.execute(query)?;
        result.scalar()
    }
}

#[cfg(all(test, feature = "lpg", feature = "gql"))]
mod tests {
    use std::sync::{Arc, Barrier, mpsc};

    use super::*;
    use crate::GrafeoDB;

    #[test]
    fn concurrent_one_shot_context_changes_merge_independent_fields() {
        let db = Arc::new(GrafeoDB::new_in_memory());
        db.session().execute("CREATE SCHEMA Shared").unwrap();
        db.create_graph("shared_graph").unwrap();
        db.create_graph("Shared/shared_graph")
            .expect("merged schema target");

        // Both temporary sessions must capture the same initial (None, None)
        // context before either command can run.
        let ready = Arc::new(Barrier::new(3));
        let schema_db = Arc::clone(&db);
        let schema_ready = Arc::clone(&ready);
        let schema_worker = std::thread::spawn(move || {
            schema_db.with_session(|session| {
                schema_ready.wait();
                session.execute("SESSION SET SCHEMA shared")
            })
        });
        let graph_db = Arc::clone(&db);
        let graph_ready = Arc::clone(&ready);
        let graph_worker = std::thread::spawn(move || {
            graph_db.with_session(|session| {
                graph_ready.wait();
                session.execute("SESSION SET GRAPH shared_graph")
            })
        });
        ready.wait();

        schema_worker.join().unwrap().unwrap();
        graph_worker.join().unwrap().unwrap();
        assert_eq!(db.current_schema().as_deref(), Some("Shared"));
        assert_eq!(db.current_graph().as_deref(), Some("shared_graph"));
        assert_eq!(
            db.session().current_graph_path(),
            grafeo_common::types::GraphPath::from_components(&["Shared/shared_graph"])
                .expect("literal graph path")
        );
    }

    #[test]
    fn graph_merge_rejects_a_missing_concurrently_resolved_target()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Arc::new(GrafeoDB::new_in_memory());
        db.session().execute("CREATE SCHEMA Changed")?;
        assert!(db.create_graph("root_only")?);
        let (selected_tx, selected_rx) = mpsc::sync_channel(0);
        let (release_tx, release_rx) = mpsc::sync_channel(0);
        let worker_db = Arc::clone(&db);
        let worker = std::thread::spawn(move || {
            worker_db.with_session(|session| {
                let result = session.execute("USE GRAPH root_only");
                selected_tx
                    .send(result.is_ok())
                    .expect("report selected root target");
                release_rx
                    .recv()
                    .expect("wait for concurrent schema update");
                result
            })
        });
        assert!(selected_rx.recv()?);
        db.set_current_schema(Some("Changed"))?;
        release_tx.send(())?;
        assert!(
            worker
                .join()
                .map_err(|_| "selector thread panicked")?
                .is_err()
        );
        assert_eq!(db.current_schema().as_deref(), Some("Changed"));
        assert_eq!(db.current_graph(), None);
        assert_eq!(
            db.session().current_graph_path(),
            grafeo_common::types::GraphPath::from_components(&["Changed/__default__"])?
        );
        assert!(
            crate::database::testing::root_lpg_store(&db)
                .graph("root_only")
                .is_some()
        );
        Ok(())
    }

    #[test]
    fn native_one_shot_context_survives_schema_changes_and_session_inheritance()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use grafeo_common::types::GraphPath;

        let db = GrafeoDB::new_in_memory();
        db.session().execute("CREATE SCHEMA Language")?;
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
        let nested = GraphPath::from_components(&["a", "b"])?;
        db.with_session(|session| {
            session.use_graph_path(&nested)?;
            Ok(QueryResult::empty())
        })?;
        db.set_current_schema(Some("Language"))?;
        let inherited = db.session();
        assert_eq!(inherited.current_graph_path(), nested);
        assert_eq!(inherited.current_schema().as_deref(), Some("Language"));
        let node = inherited.create_node_with_props(
            &["Nested"],
            [("value", grafeo_common::types::Value::Int64(7))],
        )?;
        assert!(db.session().get_node(node).is_some());
        assert!(
            crate::database::testing::root_lpg_store(&db)
                .get_node(node)
                .is_none()
        );
        db.execute("SESSION RESET SCHEMA")?;
        assert_eq!(db.session().current_graph_path(), nested);
        assert_eq!(db.current_schema(), None);
        Ok(())
    }

    #[test]
    fn unchanged_one_shot_context_cannot_resurrect_a_concurrent_reset() {
        let db = Arc::new(GrafeoDB::new_in_memory());
        db.session().execute("CREATE SCHEMA Retained").unwrap();
        db.set_current_schema(Some("Retained")).unwrap();

        let (entered_tx, entered_rx) = mpsc::sync_channel(0);
        let (release_tx, release_rx) = mpsc::sync_channel(0);
        let stale_db = Arc::clone(&db);
        let stale_worker = std::thread::spawn(move || {
            stale_db.with_session(|_| {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(QueryResult::empty())
            })
        });

        entered_rx.recv().unwrap();
        db.execute("SESSION RESET SCHEMA").unwrap();
        release_tx.send(()).unwrap();
        stale_worker.join().unwrap().unwrap();

        assert_eq!(db.current_schema(), None);
    }

    #[test]
    fn one_shot_schema_selection_loses_cleanly_to_drop_before_merge() {
        let db = Arc::new(GrafeoDB::new_in_memory());
        db.session().execute("CREATE SCHEMA Vanishing").unwrap();
        let (selected_tx, selected_rx) = mpsc::sync_channel(0);
        let (release_tx, release_rx) = mpsc::sync_channel(0);
        let selecting_db = Arc::clone(&db);
        let selector = std::thread::spawn(move || {
            selecting_db.with_session(|session| {
                let result = session.execute("SESSION SET SCHEMA vanishing");
                selected_tx.send(result.is_ok()).unwrap();
                release_rx.recv().unwrap();
                result
            })
        });

        assert!(
            selected_rx.recv().unwrap(),
            "initial schema selection must succeed before the concurrent DROP"
        );
        db.execute("DROP SCHEMA Vanishing").unwrap();
        release_tx.send(()).unwrap();

        assert!(selector.join().unwrap().is_err());
        assert_eq!(db.current_schema(), None);
    }

    #[test]
    fn one_shot_graph_selection_loses_cleanly_to_drop_before_merge() {
        let db = Arc::new(GrafeoDB::new_in_memory());
        db.create_graph("vanishing").unwrap();
        let (selected_tx, selected_rx) = mpsc::sync_channel(0);
        let (release_tx, release_rx) = mpsc::sync_channel(0);
        let selecting_db = Arc::clone(&db);
        let selector = std::thread::spawn(move || {
            selecting_db.with_session(|session| {
                let result = session.execute("SESSION SET GRAPH vanishing");
                selected_tx.send(result.is_ok()).unwrap();
                release_rx.recv().unwrap();
                result
            })
        });

        assert!(
            selected_rx.recv().unwrap(),
            "initial graph selection must succeed before the concurrent DROP"
        );
        db.execute("DROP GRAPH vanishing").unwrap();
        release_tx.send(()).unwrap();

        assert!(selector.join().unwrap().is_err());
        assert_eq!(db.current_graph(), None);
    }
}
