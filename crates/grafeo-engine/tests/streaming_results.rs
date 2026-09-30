//! Integration tests for `execute_streaming()` / `OwnedResultStream`.
//!
//! Focus of this first pass:
//! - Equivalence: a streamed result collected back matches the materialized one.
//! - Early drop does not leak (counter returns to 0, subsequent commit is fine).
//! - Non-streamable queries (mutations, EXPLAIN, ORDER BY, aggregate, session
//!   commands) are rejected with a clear error.

#![cfg(all(feature = "gql", feature = "lpg"))]

use grafeo_common::types::{LogicalType, Value};
use grafeo_common::utils::error::{Error, QueryErrorKind};
use grafeo_core::execution::QueryExecutionControl;
use grafeo_engine::query::executor::ExecutionOptions;
use grafeo_engine::{Config, GrafeoDB};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;
use std::time::Instant;

/// Seeds a small Person/KNOWS graph. Test data names follow the repo
/// convention (Alix, Gus, Tarantino characters) from CODE_STYLE.md.
fn seed_people(db: &GrafeoDB) {
    let alix = db.create_node(&["Person"]);
    let gus = db.create_node(&["Person"]);
    let vincent = db.create_node(&["Person"]);
    let jules = db.create_node(&["Person"]);
    let mia = db.create_node(&["Person"]);

    db.set_node_property(alix, "name", Value::String("Alix".into()))
        .expect("set node property");
    db.set_node_property(alix, "age", Value::Int64(32))
        .expect("set node property");
    db.set_node_property(gus, "name", Value::String("Gus".into()))
        .expect("set node property");
    db.set_node_property(gus, "age", Value::Int64(28))
        .expect("set node property");
    db.set_node_property(vincent, "name", Value::String("Vincent".into()))
        .expect("set node property");
    db.set_node_property(vincent, "age", Value::Int64(45))
        .expect("set node property");
    db.set_node_property(jules, "name", Value::String("Jules".into()))
        .expect("set node property");
    db.set_node_property(jules, "age", Value::Int64(40))
        .expect("set node property");
    db.set_node_property(mia, "name", Value::String("Mia".into()))
        .expect("set node property");
    db.set_node_property(mia, "age", Value::Int64(24))
        .expect("set node property");

    db.create_edge(alix, gus, "KNOWS");
    db.create_edge(vincent, jules, "KNOWS");
    db.create_edge(jules, mia, "KNOWS");
}

#[test]
fn streaming_matches_materialized_scan() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);

    let query = "MATCH (p:Person) RETURN p.name AS name, p.age AS age";

    let materialized = db.execute(query).expect("execute");
    let streamed = db
        .execute_streaming(query)
        .expect("execute_streaming")
        .collect(grafeo_engine::ResultLimits::default())
        .expect("collect");

    // Streaming does not guarantee a specific row order vs materialized in the
    // absence of ORDER BY, but it must yield the same multiset of rows.
    let mut mat_sorted = materialized.rows().to_vec();
    let mut str_sorted = streamed.rows().to_vec();
    mat_sorted.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    str_sorted.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

    assert_eq!(
        mat_sorted, str_sorted,
        "streaming must produce the same rows as materialized execution"
    );
    assert_eq!(
        streamed.columns,
        vec!["name".to_string(), "age".to_string()]
    );
}

#[test]
fn streaming_matches_materialized_filter() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);

    let query = "MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS name";

    let materialized = db.execute(query).expect("execute");
    let streamed = db
        .execute_streaming(query)
        .expect("execute_streaming")
        .collect(grafeo_engine::ResultLimits::default())
        .expect("collect");

    assert_eq!(materialized.rows().len(), streamed.rows().len());

    let mut mat_sorted = materialized.rows().to_vec();
    let mut str_sorted = streamed.rows().to_vec();
    mat_sorted.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    str_sorted.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    assert_eq!(mat_sorted, str_sorted);
}

#[test]
fn streaming_row_iter_yields_every_row() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);

    let stream = db
        .execute_streaming("MATCH (p:Person) RETURN p.name")
        .expect("execute_streaming");
    let cols = stream.columns().to_vec();
    assert_eq!(cols, vec!["p.name".to_string()]);

    let rows: Vec<_> = stream
        .into_row_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("row iter");
    assert_eq!(rows.len(), 5);
}

#[test]
fn streaming_early_drop_releases_counter() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);

    // Build a stream, pull one chunk, then drop the stream without exhausting.
    {
        let mut stream = db
            .execute_streaming("MATCH (p:Person) RETURN p.name")
            .expect("execute_streaming");
        let _first = stream.next_chunk().expect("first chunk");
        // stream drops here mid-iteration
    }

    // A subsequent full execute must still work and return all five rows.
    let result = db
        .execute("MATCH (p:Person) RETURN p.name")
        .expect("execute");
    assert_eq!(result.rows().len(), 5);
}

#[test]
fn owned_stream_pins_publication_until_drop() {
    let db = Arc::new(GrafeoDB::new_in_memory());
    seed_people(&db);
    let stream = db
        .execute_streaming("MATCH (p:Person) RETURN p.name")
        .expect("stream");

    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let writer_db = Arc::clone(&db);
    let writer = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        let id = writer_db.create_node(&["Concurrent"]);
        done_tx.send(id).unwrap();
    });
    started_rx.recv().unwrap();
    assert!(
        done_rx.recv_timeout(Duration::from_millis(100)).is_err(),
        "a commit must not publish while a lazy stream still owns its read cut"
    );

    drop(stream);
    let id = done_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("writer should publish after stream drop");
    assert!(id.is_valid());
    writer.join().unwrap();
}

#[test]
fn streaming_rejects_borrowing_a_shorter_mixed_snapshot_guard() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let session = db.session();
    let snapshot = session.snapshot().unwrap();

    let result = session.execute_streaming("MATCH (p:Person) RETURN p.name");
    assert!(
        result.is_err(),
        "the stream must own its publication guard beyond snapshot lifetime"
    );
    drop(snapshot);
}

#[test]
fn streaming_rejects_mutation() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);

    let err = db
        .execute_streaming("INSERT (:Person {name: 'Butch'})")
        .expect_err("should reject mutations");
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("mutat")
            || msg.to_lowercase().contains("execute() instead")
            || msg.to_lowercase().contains("cannot be streamed"),
        "expected rejection message, got: {msg}"
    );
}

#[test]
fn streaming_rejects_order_by() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);

    // ORDER BY compiles to a push-based pipeline (Sort is a pipeline breaker).
    let err = db
        .execute_streaming("MATCH (p:Person) RETURN p.name AS n ORDER BY n")
        .expect_err("should reject push pipelines");
    assert!(
        err.to_string().to_lowercase().contains("push")
            || err
                .to_string()
                .to_lowercase()
                .contains("cannot be streamed"),
        "expected push-pipeline rejection, got: {err}"
    );
}

#[test]
fn streaming_rejects_session_command() {
    let db = GrafeoDB::new_in_memory();
    let err = db
        .execute_streaming("SESSION SET GRAPH analytics")
        .expect_err("should reject session commands");
    assert!(err.to_string().to_lowercase().contains("session"));
}

#[test]
fn streaming_rejects_explain() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let err = db
        .execute_streaming("EXPLAIN MATCH (p:Person) RETURN p.name")
        .expect_err("should reject EXPLAIN");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("explain") || msg.contains("cannot be streamed"),
        "expected EXPLAIN rejection, got: {msg}"
    );
}

#[test]
fn streaming_empty_result_yields_no_rows() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);

    let rows: Vec<_> = db
        .execute_streaming("MATCH (p:Person) WHERE p.age > 999 RETURN p.name")
        .expect("execute_streaming")
        .into_row_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("row iter");
    assert!(rows.is_empty());
}

// -------- Session-scoped streaming (ResultStream<'s> / RowIterator<'s>) ----

#[test]
fn session_streaming_yields_expected_rows() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let session = db.session();

    let stream = session
        .execute_streaming("MATCH (p:Person) RETURN p.name")
        .expect("session execute_streaming");
    assert_eq!(stream.columns(), &["p.name".to_string()]);

    let rows: Vec<_> = stream
        .into_row_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("rows");
    assert_eq!(rows.len(), 5);
}

#[test]
fn session_streaming_row_iterator_exposes_columns() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let session = db.session();

    let iter = session
        .execute_streaming("MATCH (p:Person) RETURN p.name AS n, p.age AS a")
        .expect("execute_streaming")
        .into_row_iter();
    assert_eq!(iter.columns(), &["n".to_string(), "a".to_string()]);
}

#[test]
fn session_streaming_next_chunk_then_exhaustion_returns_none() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let session = db.session();

    let mut stream = session
        .execute_streaming("MATCH (p:Person) RETURN p.name")
        .expect("execute_streaming");

    let first = stream.next_chunk().expect("first chunk");
    assert!(first.is_some(), "first chunk should yield rows");

    while stream.next_chunk().expect("chunk").is_some() {}
    // Exhaustion is idempotent: further next_chunk calls stay at None.
    assert!(stream.next_chunk().expect("post-exhaustion").is_none());
}

#[test]
fn session_streaming_collect_matches_execute() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let session = db.session();

    let materialized = session
        .execute("MATCH (p:Person) RETURN p.name")
        .expect("execute");
    let streamed = session
        .execute_streaming("MATCH (p:Person) RETURN p.name")
        .expect("execute_streaming")
        .collect(grafeo_engine::ResultLimits::default())
        .expect("collect");
    assert_eq!(materialized.rows().len(), streamed.rows().len());
}

#[test]
fn session_streaming_rejects_schema_command() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // ResultStream doesn't impl Debug, so we can't use expect_err here.
    let Err(err) = session.execute_streaming("CREATE GRAPH analytics") else {
        panic!("schema DDL must not be streamable");
    };
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("schema") || msg.contains("cannot be streamed"),
        "expected schema DDL rejection, got: {msg}"
    );
}

#[test]
fn session_streaming_rejects_unclassified_procedure_before_execution() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE plant() RETURNS (n NODE) \
             AS { INSERT (n:Secret) RETURN n }",
        )
        .expect("create mutating procedure fixture");

    let Err(error) = session.execute_streaming("CALL plant()") else {
        panic!("lazy read-only execution must reject unclassified procedure effects");
    };
    assert!(
        error
            .to_string()
            .contains("procedure calls are not qualified for streaming execution"),
        "unexpected error: {error}"
    );
    assert_eq!(db.node_count(), 0);
    assert!(!session.in_transaction());
}

#[test]
fn session_streaming_second_call_hits_plan_cache() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let session = db.session();

    let query = "MATCH (p:Person) WHERE p.age > 30 RETURN p.name";
    let first: Vec<_> = session
        .execute_streaming(query)
        .expect("first")
        .into_row_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("first rows");
    // Second call goes through the cache-hit branch in build_streaming_plan.
    let second: Vec<_> = session
        .execute_streaming(query)
        .expect("second")
        .into_row_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("second rows");
    assert_eq!(first.len(), second.len());
}

// -------- OwnedResultStream / OwnedRowIterator specifics ------------------

#[test]
fn owned_stream_column_types_start_as_any() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let stream = db
        .execute_streaming("MATCH (p:Person) RETURN p.name AS name, p.age AS age")
        .expect("execute_streaming");
    let types = stream.column_types();
    assert_eq!(types.len(), 2);
    assert!(types.iter().all(|t| matches!(t, LogicalType::Any)));
}

#[test]
fn owned_stream_column_types_refine_after_first_chunk() {
    // refine_column_types runs after every non-empty chunk. Property accesses
    // like `p.name` stay as `Any` because GQL values are dynamically typed per
    // row, so we only assert that the slot count stays consistent and the
    // refinement path is actually exercised.
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let mut stream = db
        .execute_streaming("MATCH (p:Person) RETURN p.name AS name, p.age AS age")
        .expect("execute_streaming");
    let _ = stream.next_chunk().expect("chunk");
    assert_eq!(stream.column_types().len(), 2);
}

#[test]
fn owned_stream_debug_lists_columns() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let stream = db
        .execute_streaming("MATCH (p:Person) RETURN p.name AS name")
        .expect("execute_streaming");
    let dbg = format!("{stream:?}");
    assert!(dbg.contains("OwnedResultStream"));
    assert!(dbg.contains("name"));
}

#[test]
fn owned_row_iterator_exposes_columns() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let iter = db
        .execute_streaming("MATCH (p:Person) RETURN p.name AS n")
        .expect("execute_streaming")
        .into_row_iter();
    assert_eq!(iter.columns(), &["n".to_string()]);
}

#[test]
fn owned_stream_next_chunk_exhaustion_is_idempotent() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let mut stream = db
        .execute_streaming("MATCH (p:Person) RETURN p.name")
        .expect("execute_streaming");
    while stream.next_chunk().expect("chunk").is_some() {}
    assert!(stream.next_chunk().expect("after exhaustion").is_none());
    assert!(
        stream
            .next_chunk()
            .expect("after exhaustion again")
            .is_none()
    );
}

#[test]
fn streams_with_options_snapshot_parameters_and_cache_values() {
    let db = GrafeoDB::new_in_memory();
    let query = "RETURN $value AS value";
    let first = db
        .stream_with_options(
            query,
            HashMap::from([("value".to_string(), Value::Int64(11))]),
            ExecutionOptions::default(),
        )
        .expect("first stream")
        .collect(grafeo_engine::ResultLimits::default())
        .expect("first rows");
    let second = db
        .stream_with_options(
            query,
            HashMap::from([("value".to_string(), Value::Int64(22))]),
            ExecutionOptions::default(),
        )
        .expect("cached stream")
        .collect(grafeo_engine::ResultLimits::default())
        .expect("second rows");
    assert_eq!(first.rows(), &vec![vec![Value::Int64(11)]]);
    assert_eq!(second.rows(), &vec![vec![Value::Int64(22)]]);
}

#[test]
fn cancelled_before_stream_construction_is_reported() {
    let db = GrafeoDB::new_in_memory();
    let control = QueryExecutionControl::new();
    control.cancellation_handle().cancel();
    let error = db
        .stream_with_options(
            "RETURN 1",
            HashMap::new(),
            ExecutionOptions {
                control,
                language: None,
                result_limits: None,
                result_admission: None,
            },
        )
        .expect_err("pre-cancelled stream");
    assert!(matches!(error, Error::Query(query) if query.kind == QueryErrorKind::Cancelled));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn zero_deadline_stream_reports_timeout_diagnostic() {
    let db = GrafeoDB::new_in_memory();
    let control = QueryExecutionControl::with_deadline(Instant::now());
    let error = db
        .stream_with_options(
            "RETURN 1",
            HashMap::new(),
            ExecutionOptions {
                control,
                language: None,
                result_limits: None,
                result_admission: None,
            },
        )
        .expect_err("expired stream");
    assert!(matches!(&error, Error::Query(query) if query.kind == QueryErrorKind::Timeout));
    assert!(error.to_string().to_lowercase().contains("timeout"));
}

#[test]
fn session_zero_timeout_is_preserved_by_stream_admission() {
    let db = GrafeoDB::with_config(Config::in_memory().with_query_timeout(Duration::ZERO))
        .expect("database");
    let error = db
        .stream_with_options("RETURN 1", HashMap::new(), ExecutionOptions::default())
        .expect_err("session timeout");
    assert!(matches!(&error, Error::Query(query) if query.kind == QueryErrorKind::Timeout));
    assert!(error.to_string().to_lowercase().contains("0ms"));
}

#[test]
fn explicit_close_is_idempotent_and_next_is_none() {
    let db = GrafeoDB::new_in_memory();
    let mut stream = db
        .stream_with_options("RETURN 1", HashMap::new(), ExecutionOptions::default())
        .expect("stream");
    stream.close().expect("close");
    stream.close().expect("idempotent close");
    assert!(stream.next_chunk().expect("closed next").is_none());
    assert_eq!(
        stream.status(),
        grafeo_engine::query::executor::stream::StreamStatus::Closed
    );
}

#[test]
fn borrowed_stream_close_releases_commit_and_rollback_fence() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("START TRANSACTION").expect("begin");
    let mut stream = session
        .stream_with_options("RETURN 1", HashMap::new(), ExecutionOptions::default())
        .expect("stream");
    assert!(
        session.execute("COMMIT").is_err(),
        "open stream must fence commit"
    );
    stream.close().expect("close");
    assert!(stream.next_chunk().expect("closed next").is_none());
    session.execute("COMMIT").expect("commit after close");

    session.execute("START TRANSACTION").expect("begin again");
    let mut stream = session
        .stream_with_options("RETURN 1", HashMap::new(), ExecutionOptions::default())
        .expect("second stream");
    assert!(
        session.execute("ROLLBACK").is_err(),
        "open stream must fence rollback"
    );
    stream.close().expect("second close");
    assert!(stream.next_chunk().expect("closed next").is_none());
    session.execute("ROLLBACK").expect("rollback after close");
}

#[test]
fn profiled_push_pipelines_remain_rejected() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    for query in [
        "PROFILE MATCH (p:Person) RETURN p.name ORDER BY p.name",
        "PROFILE MATCH (p:Person) RETURN count(p)",
    ] {
        let error = db
            .stream_with_options(query, HashMap::new(), ExecutionOptions::default())
            .expect_err("profiled push pipeline");
        assert!(
            matches!(error, Error::Query(ref query_error) if query_error.kind == QueryErrorKind::Semantic),
            "unexpected error for {query}: {error}"
        );
    }
}

#[test]
fn stream_options_reject_non_gql_language() {
    let db = GrafeoDB::new_in_memory();
    let error = db
        .stream_with_options(
            "RETURN 1",
            HashMap::new(),
            ExecutionOptions {
                control: QueryExecutionControl::new(),
                language: Some("sparql".to_string()),
                result_limits: None,
                result_admission: None,
            },
        )
        .expect_err("non-GQL stream language");
    assert!(matches!(error, Error::Query(ref query) if query.kind == QueryErrorKind::Unsupported));
}

#[test]
fn borrowed_close_after_pull_releases_session_fence() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("START TRANSACTION").expect("begin");
    let mut stream = session
        .stream_with_options("RETURN 1", HashMap::new(), ExecutionOptions::default())
        .expect("stream");
    assert_eq!(
        stream
            .next_chunk()
            .expect("first chunk")
            .expect("row chunk")
            .row_count(),
        1
    );
    assert!(session.execute("COMMIT").is_err());
    stream.close().expect("close");
    assert!(stream.next_chunk().expect("closed next").is_none());
    session.execute("COMMIT").expect("commit after close");
}

#[test]
fn borrowed_stream_honors_named_graph_and_parameters() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE GRAPH social")
        .expect("create graph");
    session.execute("USE GRAPH social").expect("use graph");
    session
        .execute("INSERT (:Person {name: 'Ada'})")
        .expect("seed graph");
    let result = session
        .stream_with_options(
            "MATCH (p:Person) WHERE p.name = $name RETURN p.name",
            HashMap::from([("name".to_string(), Value::String("Ada".into()))]),
            ExecutionOptions::default(),
        )
        .expect("named graph stream")
        .collect(grafeo_engine::ResultLimits::default())
        .expect("named graph rows");
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Ada".into()));
}

#[test]
fn cancellation_after_first_chunk_fails_and_stays_failed() {
    let db = GrafeoDB::new_in_memory();
    for _ in 0..4097 {
        db.create_node(&["Person"]);
    }
    let control = QueryExecutionControl::new();
    let handle = control.cancellation_handle();
    let mut stream = db
        .stream_with_options(
            "MATCH (p:Person) RETURN p",
            HashMap::new(),
            ExecutionOptions {
                control,
                language: None,
                result_limits: None,
                result_admission: None,
            },
        )
        .expect("stream");
    assert!(stream.next_chunk().expect("first chunk").is_some());
    handle.cancel();
    let error = stream.next_chunk().expect_err("cancellation");
    assert!(matches!(error, Error::Query(query) if query.kind == QueryErrorKind::Cancelled));
    assert_eq!(
        stream.status(),
        grafeo_engine::query::executor::stream::StreamStatus::Failed
    );
    assert!(stream.next_chunk().expect("after error").is_none());
}

#[test]
fn cancelled_profile_retains_partial_stats_and_failed_status() {
    let db = GrafeoDB::new_in_memory();
    for _ in 0..4097 {
        db.create_node(&["Person"]);
    }
    let control = QueryExecutionControl::new();
    let handle = control.cancellation_handle();
    let mut stream = db
        .stream_with_options(
            "PROFILE MATCH (p:Person) RETURN p",
            HashMap::new(),
            ExecutionOptions {
                control,
                language: None,
                result_limits: None,
                result_admission: None,
            },
        )
        .expect("profile stream");
    assert!(stream.next_chunk().expect("first chunk").is_some());
    handle.cancel();
    assert!(stream.next_chunk().is_err());
    assert_eq!(
        stream.status(),
        grafeo_engine::query::executor::stream::StreamStatus::Failed
    );
    let profile = stream.profile().expect("partial profile");
    assert!(profile.stats.lock().rows_out > 0);
    assert_ne!(
        stream.status(),
        grafeo_engine::query::executor::stream::StreamStatus::Completed
    );
}

#[test]
fn profile_stream_keeps_rows_and_actual_root_stats() {
    let db = GrafeoDB::new_in_memory();
    let mut stream = db
        .stream_with_options(
            "PROFILE RETURN 1 AS value",
            HashMap::new(),
            ExecutionOptions::default(),
        )
        .expect("profile stream");
    let first = stream
        .next_chunk()
        .expect("profile first chunk")
        .expect("row chunk");
    assert_eq!(first.row_count(), 1);
    assert!(stream.next_chunk().expect("profile exhaustion").is_none());
    let profile = stream.profile().expect("profile tree");
    assert_eq!(profile.stats.lock().rows_out, 1);
    assert_eq!(
        stream.status(),
        grafeo_engine::query::executor::stream::StreamStatus::Completed
    );
}

#[test]
fn session_stream_with_options_accepts_parameters() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session
        .stream_with_options(
            "RETURN $value AS value",
            HashMap::from([("value".to_string(), Value::String("session".into()))]),
            ExecutionOptions::default(),
        )
        .expect("session stream")
        .collect(grafeo_engine::ResultLimits::default())
        .expect("session rows");
    assert_eq!(result.rows(), &vec![vec![Value::String("session".into())]]);
}

#[test]
fn owned_explicit_close_releases_publication_while_value_remains_alive() {
    let db = Arc::new(GrafeoDB::new_in_memory());
    seed_people(&db);
    let mut stream = db
        .stream_with_options(
            "MATCH (p:Person) RETURN p.name",
            HashMap::new(),
            ExecutionOptions::default(),
        )
        .expect("stream");
    let writer_db = Arc::clone(&db);
    let (started_tx, started_rx) = mpsc::channel();
    let (completed, completion) = mpsc::channel();
    let writer = std::thread::spawn(move || {
        started_tx.send(()).expect("started");
        let id = writer_db.create_node(&["Concurrent"]);
        completed.send(id).expect("completed");
    });
    started_rx.recv().expect("writer started");
    assert!(completion.recv_timeout(Duration::from_millis(100)).is_err());
    stream.close().expect("close");
    let id = completion
        .recv_timeout(Duration::from_secs(5))
        .expect("writer released");
    writer.join().expect("writer");
    assert!(id.is_valid());
    assert!(stream.next_chunk().expect("closed next").is_none());
}

#[test]
fn concurrent_stream_cancellation_is_isolated() {
    let db = GrafeoDB::new_in_memory();
    for _ in 0..4097 {
        db.create_node(&["Person"]);
    }
    let first_control = QueryExecutionControl::new();
    let first_handle = first_control.cancellation_handle();
    let second_control = QueryExecutionControl::new();
    let mut first = db
        .stream_with_options(
            "MATCH (p:Person) RETURN p",
            HashMap::new(),
            ExecutionOptions {
                control: first_control,
                language: None,
                result_limits: None,
                result_admission: None,
            },
        )
        .expect("first stream");
    let mut second = db
        .stream_with_options(
            "MATCH (p:Person) RETURN p",
            HashMap::new(),
            ExecutionOptions {
                control: second_control,
                language: None,
                result_limits: None,
                result_admission: None,
            },
        )
        .expect("second stream");
    assert_ne!(first.query_id(), second.query_id());
    assert!(first.next_chunk().expect("first chunk").is_some());
    assert!(second.next_chunk().expect("second chunk").is_some());
    first_handle.cancel();
    assert!(first.next_chunk().is_err());
    assert!(second.next_chunk().expect("uncancelled stream").is_some());
}

#[test]
fn distinct_pull_and_profile_cancel_once_with_buffered_witnesses() {
    use grafeo_engine::query::executor::stream::StreamStatus;
    for profile in [false, true] {
        let db = GrafeoDB::new_in_memory();
        let control = QueryExecutionControl::new();
        let cancel = control.cancellation_handle();
        let query = if profile {
            "PROFILE UNWIND $values AS value RETURN DISTINCT value"
        } else {
            "UNWIND $values AS value RETURN DISTINCT value"
        };
        let mut stream = db
            .stream_with_options(
                query,
                HashMap::from([(
                    "values".to_string(),
                    Value::List(
                        (0..97)
                            .chain(0..97)
                            .map(Value::Int64)
                            .collect::<Vec<_>>()
                            .into(),
                    ),
                )]),
                ExecutionOptions {
                    control,
                    language: None,
                    result_limits: None,
                    result_admission: None,
                },
            )
            .expect("qualified DISTINCT pull stream");
        let chunk = stream
            .next_chunk()
            .expect("first pull")
            .expect("first witnesses");
        assert!(chunk.row_count() > 0 && chunk.row_count() < 97);
        drop(chunk);
        cancel.cancel();
        let error = stream
            .next_chunk()
            .expect_err("cancel buffered DISTINCT witnesses");
        assert!(matches!(error, Error::Query(query) if query.kind == QueryErrorKind::Cancelled));
        assert_eq!(stream.status(), StreamStatus::Failed);
        assert!(
            stream
                .next_chunk()
                .expect("terminal error delivered once")
                .is_none()
        );
        if profile {
            let stats = stream
                .profile()
                .unwrap()
                .stats
                .lock()
                .query_resources
                .unwrap();
            assert_eq!(
                stats.resident_granted_bytes, 0,
                "terminal cancellation releases grants"
            );
            assert!(stats.resident_peak_bytes > 0);
            assert!(
                stream
                    .profile()
                    .expect("actual DISTINCT profile")
                    .stats
                    .lock()
                    .rows_out
                    > 0
            );
        }
        stream.close().expect("explicit cancellation cleanup");
        stream.close().expect("stable cleanup resolution");
        drop(stream);
        assert_eq!(
            db.execute("RETURN 1").unwrap().rows(),
            &[vec![Value::Int64(1)]]
        );
    }
}
