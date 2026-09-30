//! Public execution-owned cancellation and statement/commit fences.

#![cfg(any(
    all(feature = "lpg", feature = "gql"),
    all(feature = "triple-store", feature = "sparql")
))]
#![allow(missing_docs)]

#[cfg(all(feature = "lpg", feature = "gql"))]
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryErrorKind};
use grafeo_core::execution::QueryExecutionControl;
#[cfg(all(feature = "wal", feature = "testing-statement-injection"))]
use grafeo_engine::DurabilityMode;
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::executor::ExecutionOptions;
#[cfg(any(feature = "wal", feature = "triple-store"))]
use grafeo_engine::{Config, GraphModel};
use std::collections::HashMap;
#[cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "testing-statement-injection"
))]
use std::sync::{Arc, Barrier};
#[cfg(all(feature = "lpg", feature = "gql"))]
use std::time::Duration;

#[cfg(all(feature = "lpg", feature = "gql"))]
mod gql_cancellation {
    use super::*;

    fn cancelled(err: &Error) -> bool {
        matches!(err, Error::Query(query) if query.kind == QueryErrorKind::Cancelled)
    }

    #[test]
    fn execute_with_options_reports_pre_cancellation_without_side_effects() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let options = ExecutionOptions {
            control: QueryExecutionControl::new(),
            language: None,
            result_limits: None,
            result_admission: None,
        };
        let handle = options.control.cancellation_handle();
        handle.cancel();

        let error = session
            .execute_with_options("RETURN 1", HashMap::new(), options)
            .unwrap_err();
        assert!(cancelled(&error), "{error:?}");
        assert_eq!(db.node_count(), 0);

        let ok = session.execute("RETURN 1").unwrap();
        assert_eq!(ok.row_count(), 1);

        let cancelled_again = ExecutionOptions {
            control: QueryExecutionControl::new(),
            language: None,
            result_limits: None,
            result_admission: None,
        };
        cancelled_again.control.cancellation_handle().cancel();
        session
            .execute_with_options("RETURN 1", HashMap::new(), cancelled_again)
            .unwrap_err();
    }

    #[test]
    fn database_execute_with_options_observes_the_same_control() {
        let db = GrafeoDB::new_in_memory();
        let options = ExecutionOptions {
            control: QueryExecutionControl::new(),
            language: None,
            result_limits: None,
            result_admission: None,
        };
        options.control.cancellation_handle().cancel();
        let error = db
            .execute_with_options("RETURN 1", HashMap::new(), options)
            .unwrap_err();
        assert!(cancelled(&error), "{error:?}");
    }

    #[test]
    fn autocommit_cancel_before_execution_leaves_no_mutation() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let options = ExecutionOptions {
            control: QueryExecutionControl::new(),
            language: None,
            result_limits: None,
            result_admission: None,
        };
        options.control.cancellation_handle().cancel();
        let error = session
            .execute_with_options("INSERT (:Person {name: 'Ada'})", HashMap::new(), options)
            .unwrap_err();
        assert!(cancelled(&error), "{error:?}");
        assert_eq!(db.node_count(), 0);
        assert!(!session.in_transaction());
    }

    #[test]
    fn explicit_transaction_keeps_prior_state_when_a_later_statement_is_cancelled() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Kept'})").unwrap();
        let names = session
            .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
            .unwrap();
        assert_eq!(names.rows().len(), 1);
        assert_eq!(
            names.rows()[0][0],
            grafeo_common::types::Value::String("Kept".into())
        );

        let options = ExecutionOptions {
            control: QueryExecutionControl::new(),
            language: None,
            result_limits: None,
            result_admission: None,
        };
        options.control.cancellation_handle().cancel();
        let error = session
            .execute_with_options(
                "INSERT (:Person {name: 'Dropped'})",
                HashMap::new(),
                options,
            )
            .unwrap_err();
        assert!(cancelled(&error), "{error:?}");
        assert!(session.in_transaction());
        assert_eq!(
            session
                .execute("MATCH (n:Person) RETURN n")
                .unwrap()
                .row_count(),
            1
        );

        session.commit().unwrap();
        assert_eq!(db.node_count(), 1);
    }

    #[test]
    fn committed_autocommit_is_not_rolled_back_by_later_cancellation() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let options = ExecutionOptions::default();
        let handle = options.control.cancellation_handle();
        session
            .execute_with_options(
                "INSERT (:Person {name: 'Durable'})",
                HashMap::new(),
                options,
            )
            .unwrap();
        handle.cancel();
        assert_eq!(db.node_count(), 1);
        assert!(!session.in_transaction());
    }

    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn second_thread_cancel_at_statement_completion_keeps_autocommit_atomic() {
        use grafeo_engine::QueryCancellationTestPhase;
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let reached = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        session.set_query_cancellation_test_hook(
            QueryCancellationTestPhase::BeforeStatementCompletion,
            Arc::clone(&reached),
            Arc::clone(&release),
        );
        let options = ExecutionOptions::default();
        let cancel = options.control.cancellation_handle();
        let join = std::thread::spawn(move || {
            reached.wait();
            cancel.cancel();
            release.wait();
        });
        let result = session.execute_with_options(
            "INSERT (:Person {name: 'late'})",
            HashMap::new(),
            options,
        );
        join.join().unwrap();
        assert!(
            result.is_err(),
            "completion cancellation must abort statement"
        );
        assert_eq!(db.node_count(), 0, "cancelled autocommit must not publish");
    }

    #[cfg(feature = "testing-statement-injection")]
    fn hooked_cancel(
        session: &grafeo_engine::Session,
        phase: grafeo_engine::QueryCancellationTestPhase,
        query: &'static str,
    ) -> Result<grafeo_engine::database::QueryResult, Error> {
        hooked_cancel_with_params(session, phase, query, HashMap::new())
    }

    #[cfg(feature = "testing-statement-injection")]
    fn hooked_cancel_with_params(
        session: &grafeo_engine::Session,
        phase: grafeo_engine::QueryCancellationTestPhase,
        query: &'static str,
        params: HashMap<String, Value>,
    ) -> Result<grafeo_engine::database::QueryResult, Error> {
        let reached = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        session.set_query_cancellation_test_hook(phase, Arc::clone(&reached), Arc::clone(&release));
        let options = ExecutionOptions::default();
        let cancel = options.control.cancellation_handle();
        let join = std::thread::spawn(move || {
            reached.wait();
            cancel.cancel();
            release.wait();
        });
        let result = session.execute_with_options(query, params, options);
        join.join().expect("cancellation worker");
        result
    }

    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn second_thread_cancelled_explicit_statement_rolls_back_only_new_write() {
        use grafeo_engine::QueryCancellationTestPhase;
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Kept'})").unwrap();
        let result = hooked_cancel(
            &session,
            QueryCancellationTestPhase::BeforeStatementCompletion,
            "INSERT (:Person {name: 'Dropped'})",
        );
        assert!(result.as_ref().err().is_some_and(cancelled));
        assert_eq!(
            session
                .execute("MATCH (n:Person) RETURN n")
                .unwrap()
                .row_count(),
            1
        );
        session.execute("INSERT (:Person {name: 'After'})").unwrap();
        session.commit().unwrap();
        let names = session
            .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
            .unwrap();
        let names: Vec<_> = names.rows().iter().map(|row| row[0].clone()).collect();
        assert_eq!(
            names,
            vec![Value::String("After".into()), Value::String("Kept".into())]
        );
        assert_eq!(db.node_count(), 2);
    }

    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn cancellation_before_mutation_and_after_savepoint_preserves_prior_work() {
        use grafeo_engine::QueryCancellationTestPhase::{
            AfterSavepointPrepared, BeforeFirstMutation,
        };
        for phase in [BeforeFirstMutation, AfterSavepointPrepared] {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session.execute("INSERT (:Early {name: 'Kept'})").unwrap();
            let error =
                hooked_cancel(&session, phase, "INSERT (:Early {name: 'Dropped'})").unwrap_err();
            assert!(cancelled(&error), "{error:?}");
            assert!(session.in_transaction());
            assert_eq!(
                session
                    .execute("MATCH (n:Early) RETURN n.name")
                    .unwrap()
                    .rows(),
                &[vec![Value::from("Kept")]]
            );
            session.commit().unwrap();
            assert_eq!(db.node_count(), 1);
        }
    }

    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn cancelled_parameterized_procedure_retains_only_prior_statement() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.execute("CREATE PROCEDURE plant_named(name STRING) RETURNS (name STRING) AS { INSERT (n:ParamMutation {name: $name}) RETURN n.name AS name }").unwrap();
        session.begin_transaction().unwrap();
        session
            .execute("INSERT (:ParamMutation {name: 'Kept'})")
            .unwrap();
        let error = hooked_cancel_with_params(
            &session,
            grafeo_engine::QueryCancellationTestPhase::BeforeStatementCompletion,
            "CALL plant_named($requested) YIELD name",
            [("requested".to_owned(), Value::from("Dropped"))].into(),
        )
        .unwrap_err();
        assert!(cancelled(&error), "{error:?}");
        assert_eq!(
            session
                .execute("MATCH (n:ParamMutation) RETURN n.name")
                .unwrap()
                .rows(),
            &[vec![Value::from("Kept")]]
        );
        session.commit().unwrap();
        assert_eq!(db.node_count(), 1);
    }

    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn cancelled_graph_ddl_does_not_publish_after_transaction_commit() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.execute("CREATE GRAPH kept_graph").unwrap();
        session.begin_transaction().unwrap();
        session.execute("INSERT (:KeptBeforeGraph)").unwrap();
        let error = hooked_cancel(
            &session,
            grafeo_engine::QueryCancellationTestPhase::BeforeStatementCompletion,
            "CREATE GRAPH cancelled_graph",
        )
        .unwrap_err();
        assert!(cancelled(&error), "{error:?}");
        assert_eq!(
            session
                .execute("MATCH (n:KeptBeforeGraph) RETURN n")
                .unwrap()
                .row_count(),
            1
        );
        session.commit().unwrap();
        assert!(db.list_graphs().contains(&"kept_graph".to_owned()));
        assert!(!db.list_graphs().contains(&"cancelled_graph".to_owned()));
        assert_eq!(db.node_count(), 1);
    }

    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn second_thread_cancel_before_commit_fence_rolls_back_autocommit() {
        use grafeo_engine::QueryCancellationTestPhase;
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = hooked_cancel(
            &session,
            QueryCancellationTestPhase::BeforeCommitFence,
            "INSERT (:Person {name: 'Fence'})",
        );
        assert!(result.as_ref().err().is_some_and(cancelled));
        assert_eq!(db.node_count(), 0);
    }

    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn cancelled_property_index_ddl_leaves_prior_transaction_usable() {
        use grafeo_engine::QueryCancellationTestPhase;
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Kept'})").unwrap();
        let result = hooked_cancel(
            &session,
            QueryCancellationTestPhase::BeforeStatementCompletion,
            "CREATE INDEX cancelled_name FOR (n:Person) ON (n.name)",
        );
        assert!(result.as_ref().err().is_some_and(cancelled));
        let indexes = session.execute("SHOW INDEXES").unwrap();
        assert!(!indexes.rows().iter().flatten().any(|value| {
            matches!(value, Value::String(name) if name.as_str() == "cancelled_name")
        }));
        assert_eq!(
            session
                .execute("MATCH (n:Person) RETURN n.name")
                .unwrap()
                .row_count(),
            1
        );
        session.execute("INSERT (:Person {name: 'After'})").unwrap();
        session.commit().unwrap();
        assert_eq!(db.node_count(), 2);
    }

    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn nested_procedure_cancellation_preserves_prior_transaction_work() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.execute("CREATE PROCEDURE plant() RETURNS (name STRING) AS { INSERT (n:Framed {name: 'Dropped'}) RETURN n.name AS name }").unwrap();
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Framed {name: 'Kept'})").unwrap();
        let result = hooked_cancel(
            &session,
            grafeo_engine::QueryCancellationTestPhase::BeforeStatementCompletion,
            "CALL plant() YIELD name",
        );
        assert!(result.as_ref().err().is_some_and(cancelled), "{result:?}");
        let remaining = session
            .execute("MATCH (n:Framed) RETURN n.name ORDER BY n.name")
            .unwrap();
        assert_eq!(
            remaining.rows(),
            &[vec![grafeo_common::types::Value::from("Kept")]]
        );
        session.commit().unwrap();
        assert_eq!(db.node_count(), 1);
    }

    #[cfg(all(feature = "testing-statement-injection", feature = "cdc"))]
    #[test]
    fn cancelled_autocommit_publishes_no_cdc_and_fresh_query_publishes_once() {
        use grafeo_common::types::EpochId;
        for phase in [
            grafeo_engine::QueryCancellationTestPhase::BeforeStatementCompletion,
            grafeo_engine::QueryCancellationTestPhase::BeforeCommitFence,
        ] {
            let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
            let session = db.session();
            let result = hooked_cancel(&session, phase, "INSERT (:CancelledCdc)");
            assert!(result.as_ref().err().is_some_and(cancelled), "{result:?}");
            assert!(
                db.fixture_changes(EpochId::INITIAL..=EpochId::new(u64::MAX))
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(db.node_count(), 0);
            session.execute("INSERT (:PublishedCdc)").unwrap();
            let events = db
                .fixture_changes(EpochId::INITIAL..=EpochId::new(u64::MAX))
                .unwrap();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].kind, grafeo_engine::cdc::ChangeKind::Create);
            assert!(
                events[0]
                    .labels
                    .as_ref()
                    .unwrap()
                    .iter()
                    .any(|label| label == "PublishedCdc")
            );
        }
    }

    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn postwrite_procedure_error_remains_primary_when_cancellation_arrives() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.execute("CREATE PROCEDURE wrong_return_type() RETURNS (name STRING) AS { INSERT (n:WrongReturnResidue {name: 'temporary'}) RETURN 7 AS name }").unwrap();
        session.begin_transaction().unwrap();
        session.execute("INSERT (:KeptBeforeFailure)").unwrap();
        let error = hooked_cancel(
            &session,
            grafeo_engine::QueryCancellationTestPhase::BeforeStatementCompletion,
            "CALL wrong_return_type()",
        )
        .unwrap_err();
        assert!(
            !cancelled(&error),
            "earlier return-contract failure is primary: {error:?}"
        );
        assert!(
            error
                .to_string()
                .contains("violates its declared return type"),
            "{error:?}"
        );
        assert_eq!(
            session
                .execute("MATCH (n:WrongReturnResidue) RETURN n")
                .unwrap()
                .row_count(),
            0
        );
        assert_eq!(
            session
                .execute("MATCH (n:KeptBeforeFailure) RETURN n")
                .unwrap()
                .row_count(),
            1
        );
        session.commit().unwrap();
        assert_eq!(db.node_count(), 1);
    }

    #[cfg(all(feature = "testing-statement-injection", feature = "wal"))]
    #[test]
    fn cancellation_before_durable_marker_recovers_no_statement() {
        for phase in [
            grafeo_engine::QueryCancellationTestPhase::BeforeStatementCompletion,
            grafeo_engine::QueryCancellationTestPhase::BeforeCommitFence,
        ] {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("db");
            let config = Config::persistent(&path).with_wal_durability(DurabilityMode::Sync);
            let db = GrafeoDB::with_config(config.clone()).unwrap();
            let session = db.session();
            let error = hooked_cancel(&session, phase, "INSERT (:CancelledDurable)").unwrap_err();
            assert!(cancelled(&error), "{error:?}");
            drop(session);
            drop(db);
            let reopened = GrafeoDB::with_config(config).unwrap();
            assert_eq!(reopened.node_count(), 0);
            assert_eq!(
                reopened
                    .execute("MATCH (n:CancelledDurable) RETURN n")
                    .unwrap()
                    .row_count(),
                0
            );
        }
    }

    #[cfg(all(feature = "testing-statement-injection", feature = "wal"))]
    #[test]
    fn cancel_after_durable_marker_succeeds_and_reopens_data() {
        use grafeo_engine::QueryCancellationTestPhase;
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("db");
        let db = GrafeoDB::with_config(
            Config::persistent(&path)
                .with_graph_model(GraphModel::Lpg)
                .with_wal_durability(DurabilityMode::Sync),
        )
        .unwrap();
        let session = db.session();
        let result = hooked_cancel(
            &session,
            QueryCancellationTestPhase::AfterDurableMarker,
            "INSERT (:Person {name: 'Durable'})",
        );
        assert!(
            result.is_ok(),
            "durable marker means cancellation is too late"
        );
        drop(session);
        drop(db);
        let reopened = GrafeoDB::with_config(
            Config::persistent(&path)
                .with_graph_model(GraphModel::Lpg)
                .with_wal_durability(DurabilityMode::Sync),
        )
        .unwrap();
        assert_eq!(reopened.node_count(), 1);
        drop(reopened);
    }
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn gql_parameter_values_do_not_leak_between_cached_plans_and_profile_runs() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for value in ["first", "second"] {
        let mut params = HashMap::new();
        params.insert("value".into(), Value::String(value.into()));
        let result = session
            .execute_with_options(
                "RETURN $value AS value",
                params,
                ExecutionOptions::default(),
            )
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::String(value.into()));
    }
    let profile = session
        .execute_with_options(
            "PROFILE RETURN 1 AS value",
            HashMap::new(),
            ExecutionOptions::default(),
        )
        .unwrap();
    assert_eq!(profile.row_count(), 1);
    let text = profile.rows()[0][0].as_str().unwrap();
    assert!(
        text.lines()
            .any(|line| line.contains("SingleRow") && line.contains("rows=1")),
        "{text}"
    );
    #[cfg(feature = "cypher")]
    let with_profile = session
        .execute_with_options(
            "PROFILE WITH 1 AS value RETURN value",
            HashMap::new(),
            ExecutionOptions {
                language: Some("cypher".into()),
                ..ExecutionOptions::default()
            },
        )
        .unwrap();
    #[cfg(feature = "cypher")]
    let text = with_profile.rows()[0][0].as_str().unwrap();
    #[cfg(feature = "cypher")]
    assert!(
        text.lines()
            .any(|line| line.contains("SingleRow") && line.contains("rows=1")),
        "{text}"
    );
    let options = ExecutionOptions::default();
    options.control.cancellation_handle().cancel();
    let error = session
        .execute_with_options("PROFILE RETURN 1 AS value", HashMap::new(), options)
        .unwrap_err();
    assert!(matches!(error, Error::Query(query) if query.kind == QueryErrorKind::Cancelled));
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn gql_zero_timeout_is_typed_timeout() {
    let db = GrafeoDB::new_in_memory();
    let control = QueryExecutionControl::with_timeout(Duration::ZERO).unwrap();
    let error = db
        .session()
        .execute_with_options(
            "RETURN 1",
            HashMap::new(),
            ExecutionOptions {
                control,
                ..ExecutionOptions::default()
            },
        )
        .unwrap_err();
    assert!(matches!(error, Error::Query(query) if query.kind == QueryErrorKind::Timeout));
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
mod rdf_cancellation {
    use super::*;
    #[cfg(feature = "testing-statement-injection")]
    use std::sync::{Arc, Barrier};

    fn cancelled(err: &Error) -> bool {
        matches!(err, Error::Query(query) if query.kind == QueryErrorKind::Cancelled)
    }

    #[cfg(feature = "testing-statement-injection")]
    fn hooked_rdf_cancel(
        session: &grafeo_engine::Session,
        phase: grafeo_engine::QueryCancellationTestPhase,
        query: &'static str,
    ) -> Result<grafeo_engine::database::QueryResult, Error> {
        let reached = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        session.set_query_cancellation_test_hook(phase, Arc::clone(&reached), Arc::clone(&release));
        let control = QueryExecutionControl::new();
        let cancel = control.cancellation_handle();
        let join = std::thread::spawn(move || {
            reached.wait();
            cancel.cancel();
            release.wait();
        });
        let result = session.execute_with_options(
            query,
            HashMap::new(),
            ExecutionOptions {
                control,
                language: Some("sparql".into()),
                ..ExecutionOptions::default()
            },
        );
        join.join().unwrap();
        result
    }

    #[test]
    fn sparql_fresh_owner_select_and_insert_controls() {
        let db =
            GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
        let session = db.session();
        let result = session.execute_with_options(
            "SELECT ?s WHERE { ?s ?p ?o }",
            HashMap::new(),
            ExecutionOptions {
                language: Some("sparql".into()),
                ..ExecutionOptions::default()
            },
        );
        assert!(result.is_ok());
        session
            .execute_with_options(
                "INSERT DATA { <urn:s> <urn:p> <urn:o> . }",
                HashMap::new(),
                ExecutionOptions {
                    language: Some("sparql".into()),
                    ..ExecutionOptions::default()
                },
            )
            .unwrap();
        assert_eq!(
            session
                .execute_sparql("SELECT ?s WHERE { ?s <urn:p> ?o }")
                .unwrap()
                .row_count(),
            1
        );
    }

    #[test]
    fn sparql_precancelled_insert_has_no_quads() {
        let db =
            GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
        let session = db.session();
        let control = QueryExecutionControl::new();
        control.cancellation_handle().cancel();
        let error = session
            .execute_with_options(
                "INSERT DATA { <urn:s> <urn:p> <urn:o> . }",
                HashMap::new(),
                ExecutionOptions {
                    control,
                    language: Some("sparql".into()),
                    ..ExecutionOptions::default()
                },
            )
            .unwrap_err();
        assert!(cancelled(&error));
        assert_eq!(
            session
                .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
                .unwrap()
                .row_count(),
            0
        );
    }

    #[cfg(not(feature = "gql"))]
    #[test]
    fn explicit_unavailable_language_keeps_structured_admission_error() {
        let db =
            GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
        let error = db
            .execute_with_options(
                "RETURN 1",
                HashMap::new(),
                ExecutionOptions {
                    language: Some("gql".into()),
                    ..ExecutionOptions::default()
                },
            )
            .unwrap_err();
        assert!(
            matches!(error, Error::Query(query) if query.kind == QueryErrorKind::Semantic && query.message.contains("Unknown query language: 'gql'"))
        );
    }

    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn sparql_explicit_cancel_rolls_back_only_attempted_triple() {
        use grafeo_engine::QueryCancellationTestPhase;
        let db =
            GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute_sparql("INSERT DATA { <urn:kept> <urn:p> <urn:o> . }")
            .unwrap();
        let result = hooked_rdf_cancel(
            &session,
            QueryCancellationTestPhase::BeforeStatementCompletion,
            "INSERT DATA { <urn:dropped> <urn:p> <urn:o> . }",
        );
        assert!(result.as_ref().err().is_some_and(cancelled));
        let check = session
            .execute_sparql("SELECT ?s WHERE { ?s <urn:p> ?o }")
            .unwrap();
        assert_eq!(check.row_count(), 1);
        assert_eq!(
            session
                .execute_sparql("SELECT ?o WHERE { <urn:kept> <urn:p> ?o }")
                .unwrap()
                .row_count(),
            1
        );
        assert_eq!(
            session
                .execute_sparql("SELECT ?o WHERE { <urn:dropped> <urn:p> ?o }")
                .unwrap()
                .row_count(),
            0
        );
        session.commit().unwrap();
        assert_eq!(
            session
                .execute_sparql("SELECT ?o WHERE { <urn:kept> <urn:p> ?o }")
                .unwrap()
                .row_count(),
            1
        );
    }

    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn rdf_early_cancellation_preserves_prepared_transaction() {
        use grafeo_engine::QueryCancellationTestPhase::{
            AfterSavepointPrepared, BeforeFirstMutation,
        };
        for phase in [BeforeFirstMutation, AfterSavepointPrepared] {
            let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))
                .unwrap();
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session
                .execute_sparql("INSERT DATA { <urn:kept> <urn:p> <urn:o> . }")
                .unwrap();
            let error = hooked_rdf_cancel(
                &session,
                phase,
                "INSERT DATA { <urn:dropped> <urn:p> <urn:o> . }",
            )
            .unwrap_err();
            assert!(cancelled(&error), "{error:?}");
            assert_eq!(
                session
                    .execute_sparql("SELECT ?o WHERE { <urn:kept> <urn:p> ?o }")
                    .unwrap()
                    .row_count(),
                1
            );
            assert_eq!(
                session
                    .execute_sparql("SELECT ?o WHERE { <urn:dropped> <urn:p> ?o }")
                    .unwrap()
                    .row_count(),
                0
            );
            session.commit().unwrap();
            assert_eq!(
                db.execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
                    .unwrap()
                    .row_count(),
                1
            );
        }
    }

    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn sparql_cancel_before_commit_fence_rolls_back_autocommit() {
        use grafeo_engine::QueryCancellationTestPhase;
        let db =
            GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
        let session = db.session();
        let result = hooked_rdf_cancel(
            &session,
            QueryCancellationTestPhase::BeforeCommitFence,
            "INSERT DATA { <urn:fence> <urn:p> <urn:o> . }",
        );
        assert!(result.as_ref().err().is_some_and(cancelled));
        assert_eq!(
            session
                .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
                .unwrap()
                .row_count(),
            0
        );
    }

    #[cfg(all(
        feature = "lpg",
        feature = "gql",
        feature = "testing-statement-injection"
    ))]
    #[test]
    fn mixed_transaction_retains_both_models_after_cancelled_rdf_statement() {
        let db =
            GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute("INSERT (:MixedKept)").unwrap();
        session
            .execute_sparql("INSERT DATA { <urn:kept> <urn:p> <urn:o> . }")
            .unwrap();
        let result = hooked_rdf_cancel(
            &session,
            grafeo_engine::QueryCancellationTestPhase::BeforeStatementCompletion,
            "INSERT DATA { <urn:dropped> <urn:p> <urn:o> . }",
        );
        assert!(result.as_ref().err().is_some_and(cancelled), "{result:?}");
        assert_eq!(
            session
                .execute("MATCH (n:MixedKept) RETURN n")
                .unwrap()
                .row_count(),
            1
        );
        assert_eq!(
            session
                .execute_sparql("SELECT ?o WHERE { <urn:kept> <urn:p> ?o }")
                .unwrap()
                .row_count(),
            1
        );
        assert_eq!(
            session
                .execute_sparql("SELECT ?o WHERE { <urn:dropped> <urn:p> ?o }")
                .unwrap()
                .row_count(),
            0
        );
        session.commit().unwrap();
        assert_eq!(db.node_count(), 1);
        assert_eq!(
            db.execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
                .unwrap()
                .row_count(),
            1
        );
    }

    #[cfg(all(feature = "testing-statement-injection", feature = "wal"))]
    #[test]
    fn sparql_after_durable_marker_succeeds_and_reopens() {
        use grafeo_engine::QueryCancellationTestPhase;
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("db");
        let db = GrafeoDB::with_config(
            Config::persistent(&path)
                .with_graph_model(GraphModel::Rdf)
                .with_wal_durability(DurabilityMode::Sync),
        )
        .unwrap();
        let session = db.session();
        let result = hooked_rdf_cancel(
            &session,
            QueryCancellationTestPhase::AfterDurableMarker,
            "INSERT DATA { <urn:durable> <urn:p> <urn:o> . }",
        );
        assert!(result.is_ok());
        drop(session);
        drop(db);
        let reopened = GrafeoDB::with_config(
            Config::persistent(&path)
                .with_graph_model(GraphModel::Rdf)
                .with_wal_durability(DurabilityMode::Sync),
        )
        .unwrap();
        assert_eq!(
            reopened
                .execute_sparql("SELECT ?o WHERE { <urn:durable> <urn:p> ?o }")
                .unwrap()
                .row_count(),
            1
        );
    }
}

#[cfg(all(
    feature = "cdc",
    feature = "lpg",
    feature = "gql",
    feature = "testing-statement-injection"
))]
#[path = "support/cdc_pages.rs"]
mod cdc_pages;
#[cfg(all(
    feature = "cdc",
    feature = "lpg",
    feature = "gql",
    feature = "testing-statement-injection"
))]
use cdc_pages::CdcFixtureChanges;
