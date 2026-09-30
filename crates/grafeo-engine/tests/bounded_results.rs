//! Public row/byte bounded-result controls.

#![cfg(any(
    all(feature = "lpg", feature = "gql"),
    all(feature = "triple-store", feature = "sparql")
))]

use grafeo_common::utils::error::{Error, ErrorCode, StorageError};
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;
use grafeo_engine::query::executor::{ExecutionOptions, ResultLimits};
use std::collections::HashMap;

#[cfg(all(feature = "lpg", feature = "gql"))]
fn graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE (:Person {name:'a'}), (:Person {name:'b'}), (:Person {name:'c'})")
        .unwrap();
    db
}

#[cfg(all(feature = "lpg", feature = "gql"))]
fn options(limits: Option<ResultLimits>) -> ExecutionOptions {
    ExecutionOptions {
        result_limits: limits,
        ..ExecutionOptions::default()
    }
}

fn bounded_error(error: Error) -> bool {
    error.error_code() == ErrorCode::StorageFull
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn stream_chunk_grant_is_held_only_while_chunk_is_live() {
    use grafeo_common::types::Value;
    use grafeo_engine::config::Config;

    let db = GrafeoDB::with_config(Config::in_memory().with_memory_limit(2 * 1024 * 1024))
        .expect("bounded database");
    let payload = "x".repeat(256);
    let shared = Value::from(payload);
    for _ in 0..16384 {
        let node = db.create_node(&["Person"]);
        assert!(node.is_valid());
        db.set_node_property(node, "name", shared.clone())
            .expect("set name");
    }

    let manager = db.buffer_manager();
    let baseline = manager.allocated();
    let query = "MATCH (n:Person) RETURN n.name";
    assert!(bounded_error(
        db.execute(query)
            .expect_err("eager result exceeds the small grant")
    ));
    assert_eq!(manager.allocated(), baseline);
    let mut stream = db
        .stream_with_options(query, HashMap::new(), ExecutionOptions::default())
        .expect("stream");

    let chunk = stream
        .next_chunk()
        .expect("first chunk")
        .expect("non-empty first chunk");
    let first_rows = chunk.row_count();
    let during = manager.allocated();
    assert!(
        during > baseline,
        "live chunk must hold a buffer grant: baseline={baseline}, during={during}"
    );
    drop(chunk);
    assert!(
        manager.allocated() < during,
        "dropping the chunk must release its grant"
    );

    let mut rows = 0;
    while let Some(chunk) = stream.next_chunk().expect("next chunk") {
        rows += chunk.row_count();
    }
    assert_eq!(rows + first_rows, 16384, "stream must deliver all rows");
    stream.close().expect("close at EOF");
    assert_eq!(
        manager.allocated(),
        baseline,
        "EOF must leave no query charge"
    );
    let mut peak = baseline;
    let mut rows = 0;
    for row in db.execute_streaming(query).unwrap().into_row_iter() {
        assert_eq!(row.unwrap(), vec![shared.clone()]);
        rows += 1;
        peak = peak.max(manager.allocated());
    }
    assert_eq!(rows, 16384);
    assert!(peak > baseline && peak <= 2 * 1024 * 1024);
    assert_eq!(manager.allocated(), baseline);
}

#[cfg(all(feature = "lpg", feature = "gql"))]
fn deny_nonempty_result(
    result: &QueryResult,
    _limits: ResultLimits,
) -> grafeo_common::utils::error::Result<()> {
    if result.row_count() != 0 {
        Err(Error::Storage(StorageError::Full))
    } else {
        Ok(())
    }
}

fn deny_all_results(
    _result: &QueryResult,
    _limits: ResultLimits,
) -> grafeo_common::utils::error::Result<()> {
    Err(Error::Storage(StorageError::Full))
}

#[cfg(all(feature = "lpg", feature = "gql"))]
fn admitting_options() -> ExecutionOptions {
    ExecutionOptions {
        result_admission: Some(deny_nonempty_result),
        ..options(None)
    }
}

#[cfg(all(feature = "lpg", feature = "gql"))]
fn denying_options() -> ExecutionOptions {
    ExecutionOptions {
        result_admission: Some(deny_all_results),
        ..options(None)
    }
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn execute_rows_limit_is_exact_and_atomic() {
    let db = graph();
    let query = "MATCH (n:Person) RETURN n.name ORDER BY n.name";
    let error = db
        .execute_with_options(
            query,
            HashMap::new(),
            options(Some(ResultLimits {
                max_rows: 2,
                max_bytes: 1 << 20,
            })),
        )
        .unwrap_err();
    assert!(bounded_error(error));
    assert_eq!(db.execute(query).unwrap().row_count(), 3);
    let result = db
        .execute_with_options(
            query,
            HashMap::new(),
            options(Some(ResultLimits {
                max_rows: 3,
                max_bytes: 1 << 20,
            })),
        )
        .unwrap();
    assert_eq!(result.row_count(), 3);
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn owned_stream_collect_enforces_rows_and_bytes() {
    let db = graph();
    let query = "MATCH (n:Person) RETURN n.name";
    let stream = db
        .stream_with_options(query, HashMap::new(), options(None))
        .unwrap();
    let result = stream
        .collect(ResultLimits {
            max_rows: 3,
            max_bytes: 1 << 20,
        })
        .unwrap();
    assert_eq!(result.row_count(), 3);
    let stream = db
        .stream_with_options(query, HashMap::new(), options(None))
        .unwrap();
    let error = stream
        .collect(ResultLimits {
            max_rows: 2,
            max_bytes: 1 << 20,
        })
        .unwrap_err();
    assert!(bounded_error(error));
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn zero_limits_distinguish_empty_result_from_nonempty_result() {
    let db = graph();
    let empty = db
        .execute_with_options(
            "MATCH (n:Person) WHERE false RETURN n",
            HashMap::new(),
            options(Some(ResultLimits {
                max_rows: 0,
                max_bytes: 0,
            })),
        )
        .unwrap();
    assert_eq!(empty.row_count(), 0);
    let error = db
        .execute_with_options(
            "MATCH (n:Person) RETURN n",
            HashMap::new(),
            options(Some(ResultLimits {
                max_rows: 0,
                max_bytes: usize::MAX,
            })),
        )
        .unwrap_err();
    assert!(bounded_error(error));
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn parameterized_cached_shape_is_bounded_per_call_and_profile_is_checked() {
    let db = graph();
    for value in ["a", "b"] {
        let mut params = HashMap::new();
        params.insert(
            "name".into(),
            grafeo_common::types::Value::String(value.into()),
        );
        let result = db
            .execute_with_options(
                "MATCH (n:Person {name:$name}) RETURN n",
                params,
                options(Some(ResultLimits {
                    max_rows: 1,
                    max_bytes: 4096,
                })),
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }
    let profile = db
        .execute_with_options(
            "PROFILE MATCH (n:Person) RETURN n",
            HashMap::new(),
            options(Some(ResultLimits {
                max_rows: 3,
                max_bytes: 1 << 20,
            })),
        )
        .unwrap();
    assert_eq!(profile.row_count(), 1);
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn mutation_over_row_limit_does_not_publish_autocommit_write() {
    let db = GrafeoDB::new_in_memory();
    let error = db
        .execute_with_options(
            "UNWIND [1,2,3] AS i CREATE (:Item {i:i}) RETURN i",
            HashMap::new(),
            options(Some(ResultLimits {
                max_rows: 2,
                max_bytes: 1 << 20,
            })),
        )
        .unwrap_err();
    assert!(bounded_error(error));
    assert_eq!(
        db.execute("MATCH (n:Item) RETURN n").unwrap().row_count(),
        0
    );
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn distinct_mutation_output_denial_rolls_back_rows_and_bytes() {
    let db = GrafeoDB::new_in_memory();
    let error = db
        .execute_with_options(
            "UNWIND [1,2,3] AS i CREATE (:Item {i:i}) RETURN DISTINCT i",
            HashMap::new(),
            options(Some(ResultLimits {
                max_rows: 2,
                max_bytes: 1 << 20,
            })),
        )
        .unwrap_err();
    assert!(bounded_error(error));
    assert_eq!(
        db.execute("MATCH (n:Item) RETURN n").unwrap().row_count(),
        0,
        "row-limit failure must not publish the mutation"
    );

    let db = GrafeoDB::new_in_memory();
    let error = db
        .execute_with_options(
            "UNWIND ['a long distinct output value'] AS value CREATE (:Item {value:value}) RETURN DISTINCT value",
            HashMap::new(),
            options(Some(ResultLimits {
                max_rows: 1,
                max_bytes: 1,
            })),
        )
        .unwrap_err();
    assert!(bounded_error(error));
    assert_eq!(
        db.execute("MATCH (n:Item) RETURN n").unwrap().row_count(),
        0,
        "byte-limit failure must not publish the mutation"
    );
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_bounded_rows_use_fresh_rdf_owner() {
    let db = GrafeoDB::with_config(
        grafeo_engine::Config::in_memory().with_graph_model(grafeo_engine::GraphModel::Rdf),
    )
    .unwrap();
    db.execute_sparql("INSERT DATA { <urn:a> <urn:p> <urn:o> . <urn:b> <urn:p> <urn:o> . <urn:c> <urn:p> <urn:o> . }").unwrap();
    let result = db
        .execute_with_options(
            "SELECT ?s WHERE { ?s <urn:p> ?o }",
            HashMap::new(),
            ExecutionOptions {
                result_limits: Some(ResultLimits {
                    max_rows: 3,
                    max_bytes: 1 << 20,
                }),
                language: Some("sparql".into()),
                ..ExecutionOptions::default()
            },
        )
        .unwrap();
    assert_eq!(result.row_count(), 3);
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn retained_nested_payload_denies_bytes_and_releases_for_the_next_query() {
    let db = GrafeoDB::new_in_memory();
    let mut params = HashMap::new();
    params.insert(
        "payload".into(),
        grafeo_common::types::Value::from(vec![grafeo_common::types::Value::from(
            "x".repeat(8192),
        )]),
    );
    let error = db
        .execute_with_options(
            "RETURN $payload AS payload",
            params.clone(),
            options(Some(ResultLimits {
                max_rows: 1,
                max_bytes: 1024,
            })),
        )
        .unwrap_err();
    assert!(bounded_error(error));
    let result = db
        .execute_with_options(
            "RETURN $payload AS payload",
            params.clone(),
            options(Some(ResultLimits {
                max_rows: 1,
                max_bytes: 32768,
            })),
        )
        .unwrap();
    assert_eq!(result.rows()[0][0], params["payload"]);
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn configured_limits_can_be_overridden_but_stream_options_cannot_be_bypassed() {
    let db = GrafeoDB::with_config(grafeo_engine::Config::in_memory().with_result_limits(
        ResultLimits {
            max_rows: 2,
            max_bytes: 1 << 20,
        },
    ))
    .unwrap();
    let query = "UNWIND [1,2,3] AS x RETURN x";
    assert!(bounded_error(db.execute(query).unwrap_err()));
    assert_eq!(
        db.execute_with_options(
            query,
            HashMap::new(),
            options(Some(ResultLimits {
                max_rows: 3,
                max_bytes: 1 << 20,
            }))
        )
        .unwrap()
        .row_count(),
        3
    );
    let stream = db
        .stream_with_options(
            query,
            HashMap::new(),
            options(Some(ResultLimits {
                max_rows: 2,
                max_bytes: 1 << 20,
            })),
        )
        .unwrap();
    assert!(bounded_error(
        stream
            .collect(ResultLimits {
                max_rows: 3,
                max_bytes: 1 << 20
            })
            .unwrap_err()
    ));
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn output_denial_rolls_back_only_the_statement_inside_an_explicit_transaction() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("START TRANSACTION").unwrap();
    session.execute("CREATE (:Retained {i: 9})").unwrap();
    let error = session
        .execute_with_options(
            "UNWIND [1,2,3] AS i CREATE (:Rejected {i:i}) RETURN i",
            HashMap::new(),
            options(Some(ResultLimits {
                max_rows: 2,
                max_bytes: 1 << 20,
            })),
        )
        .unwrap_err();
    assert!(bounded_error(error));
    assert_eq!(
        session
            .execute("MATCH (n:Rejected) RETURN n")
            .unwrap()
            .row_count(),
        0
    );
    assert_eq!(
        session
            .execute("MATCH (n:Retained) RETURN n")
            .unwrap()
            .row_count(),
        1
    );
    session.execute("COMMIT").unwrap();
    assert_eq!(
        db.execute("MATCH (n:Retained) RETURN n")
            .unwrap()
            .row_count(),
        1
    );
    assert_eq!(
        db.execute("MATCH (n:Rejected) RETURN n")
            .unwrap()
            .row_count(),
        0
    );
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn result_admission_denial_rolls_back_an_implicit_mutation() {
    let db = GrafeoDB::new_in_memory();
    let error = db
        .execute_with_options(
            "UNWIND [1] AS i CREATE (:Rejected {i:i}) RETURN i",
            HashMap::new(),
            admitting_options(),
        )
        .unwrap_err();
    assert!(bounded_error(error));
    assert_eq!(
        db.execute("MATCH (n:Rejected) RETURN n")
            .unwrap()
            .row_count(),
        0,
        "admission denial must precede implicit commit"
    );
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn result_admission_denial_rolls_back_only_the_current_explicit_statement() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("START TRANSACTION").unwrap();
    session.execute("CREATE (:Retained {i: 9})").unwrap();

    let error = session
        .execute_with_options(
            "UNWIND [1] AS i CREATE (:Rejected {i:i}) RETURN i",
            HashMap::new(),
            admitting_options(),
        )
        .unwrap_err();
    assert!(bounded_error(error));
    assert_eq!(
        session
            .execute("MATCH (n:Rejected) RETURN n")
            .unwrap()
            .row_count(),
        0
    );
    assert_eq!(
        session
            .execute("MATCH (n:Retained) RETURN n")
            .unwrap()
            .row_count(),
        1
    );
    session.execute("COMMIT").unwrap();
    assert_eq!(
        db.execute("MATCH (n:Retained) RETURN n")
            .unwrap()
            .row_count(),
        1
    );
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn result_admission_preserves_begin_commit_status_and_rejects_stream_policy() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute_with_options("START TRANSACTION", HashMap::new(), admitting_options())
        .unwrap();
    assert!(session.in_transaction());
    session.execute("CREATE (:Committed)").unwrap();
    session
        .execute_with_options("COMMIT", HashMap::new(), admitting_options())
        .unwrap();
    assert!(!session.in_transaction());
    assert_eq!(
        db.execute("MATCH (n:Committed) RETURN n")
            .unwrap()
            .row_count(),
        1
    );

    let error = db
        .stream_with_options("RETURN 1 AS value", HashMap::new(), admitting_options())
        .unwrap_err();
    assert_eq!(error.error_code(), ErrorCode::QueryUnsupported);
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn denying_begin_and_commit_preserves_transaction_boundaries() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let error = session
        .execute_with_options("START TRANSACTION", HashMap::new(), denying_options())
        .unwrap_err();
    assert!(bounded_error(error));
    assert!(
        !session.in_transaction(),
        "denied BEGIN must not open a transaction"
    );

    session.execute("START TRANSACTION").unwrap();
    session.execute("CREATE (:CommittedLater)").unwrap();
    let error = session
        .execute_with_options("COMMIT", HashMap::new(), denying_options())
        .unwrap_err();
    assert!(bounded_error(error));
    assert!(
        session.in_transaction(),
        "denied COMMIT must retain the transaction"
    );
    assert_eq!(
        db.execute("MATCH (n:CommittedLater) RETURN n")
            .unwrap()
            .row_count(),
        0
    );
    session.execute("COMMIT").unwrap();
    assert_eq!(
        db.execute("MATCH (n:CommittedLater) RETURN n")
            .unwrap()
            .row_count(),
        1
    );
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn denied_session_timezone_change_preserves_previous_state() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("SESSION SET TIME ZONE 'UTC+2'").unwrap();

    let error = session
        .execute_with_options(
            "SESSION SET TIME ZONE 'UTC+5'",
            HashMap::new(),
            denying_options(),
        )
        .unwrap_err();
    assert!(bounded_error(error));
    assert_eq!(session.time_zone(), Some("UTC+2".to_owned()));
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn admission_panic_restores_the_next_query_context() {
    fn panic_admission(
        _result: &QueryResult,
        _limits: ResultLimits,
    ) -> grafeo_common::utils::error::Result<()> {
        panic!("test admission panic")
    }

    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        session.execute_with_options(
            "RETURN 1 AS value",
            HashMap::new(),
            ExecutionOptions {
                result_admission: Some(panic_admission),
                ..options(None)
            },
        )
    }));
    assert!(result.is_err());
    assert_eq!(session.execute("RETURN 2 AS value").unwrap().row_count(), 1);
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn rdf_update_admission_denial_rolls_back_the_update() {
    let db = GrafeoDB::with_config(
        grafeo_engine::Config::in_memory().with_graph_model(grafeo_engine::GraphModel::Rdf),
    )
    .unwrap();
    let error = db
        .execute_with_options(
            "INSERT DATA { <urn:subject> <urn:predicate> <urn:object> }",
            HashMap::new(),
            ExecutionOptions {
                language: Some("sparql".to_owned()),
                result_admission: Some(deny_all_results),
                ..ExecutionOptions::default()
            },
        )
        .unwrap_err();
    assert!(bounded_error(error));
    assert_eq!(
        db.execute_sparql("SELECT ?s WHERE { ?s <urn:predicate> <urn:object> }")
            .unwrap()
            .row_count(),
        0
    );
}

#[cfg(all(feature = "gql", feature = "lpg"))]
#[test]
fn historical_options_reject_a_language_without_the_epoch_planner() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let error = session
        .execute_at_epoch_with_options(
            "INSERT DATA { <urn:unpublished> <urn:p> <urn:o> }",
            grafeo_common::types::EpochId::new(0),
            HashMap::new(),
            ExecutionOptions {
                language: Some("sparql".into()),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(error.error_code(), ErrorCode::QueryUnsupported);
    assert_eq!(session.execute("RETURN 1").unwrap().row_count(), 1);
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn retained_stream_chunks_keep_their_grants_after_close_and_database_drop() {
    let db = GrafeoDB::with_config(
        grafeo_engine::Config::in_memory().with_memory_limit(2 * 1024 * 1024),
    )
    .unwrap();
    for _ in 0..32 {
        let _ = db.create_node(&["Held"]);
    }
    let manager = std::sync::Arc::clone(db.buffer_manager());
    let baseline = manager.allocated();
    let mut stream = db
        .execute_streaming("MATCH (n:Held) RETURN 7 AS value")
        .unwrap();
    let chunk = stream.next_chunk().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 32);
    assert!(chunk.granted_bytes() > 0);
    stream.close().unwrap();
    drop(stream);
    drop(db);
    assert!(manager.allocated() >= baseline + chunk.granted_bytes());
    assert!(
        chunk
            .rows()
            .unwrap()
            .iter()
            .all(|row| row[0] == grafeo_common::types::Value::Int64(7))
    );
    drop(chunk);
    assert_eq!(manager.allocated(), baseline);
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn retaining_chunks_exhausts_the_shared_grant_without_invalidating_prior_output() {
    let limit = 2 * 1024 * 1024;
    let db =
        GrafeoDB::with_config(grafeo_engine::Config::in_memory().with_memory_limit(limit)).unwrap();
    let payload = grafeo_common::types::Value::from("x".repeat(256));
    for _ in 0..16384 {
        let node = db.create_node(&["Held"]);
        db.set_node_property(node, "name", payload.clone()).unwrap();
    }
    let manager = db.buffer_manager();
    let baseline = manager.allocated();
    let mut stream = db
        .execute_streaming("MATCH (n:Held) RETURN n.name")
        .unwrap();
    let mut held = Vec::new();
    loop {
        match stream.next_chunk() {
            Ok(Some(chunk)) => {
                held.push(chunk);
                assert!(manager.allocated() <= limit);
            }
            Ok(None) => panic!("retaining the full result must exceed this grant"),
            Err(error) => {
                assert!(bounded_error(error));
                break;
            }
        }
    }
    let rows: usize = held.iter().map(|chunk| chunk.row_count()).sum();
    assert!(rows > 0 && rows < 16384);
    assert!(
        held.iter()
            .flat_map(|chunk| chunk.rows().unwrap())
            .all(|row| row[0] == payload)
    );
    assert!(stream.next_chunk().unwrap().is_none());
    stream.close().unwrap();
    drop(stream);
    assert!(manager.allocated() > baseline);
    drop(held);
    assert_eq!(manager.allocated(), baseline);
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn selected_stream_rows_share_one_admitted_cache() {
    use grafeo_common::types::Value;
    let db = GrafeoDB::with_config(
        grafeo_engine::Config::in_memory().with_memory_limit(4 * 1024 * 1024),
    )
    .unwrap();
    let payload = Value::from("selected-".repeat(32));
    for id in 0..4096 {
        let node = db.create_node(&["Selected"]);
        db.set_node_property(node, "id", Value::Int64(id)).unwrap();
        db.set_node_property(node, "name", payload.clone()).unwrap();
    }
    let manager = db.buffer_manager();
    let baseline = manager.allocated();
    let mut stream = db
        .execute_streaming("MATCH (n:Selected) WHERE n.id % 3 = 1 RETURN n.id, n.name")
        .unwrap();
    let mut seen = std::collections::BTreeSet::new();
    while let Some(chunk) = stream.next_chunk().unwrap() {
        assert_eq!(chunk.columns().len(), 2);
        let admitted = manager.allocated();
        let addresses = std::thread::scope(|scope| {
            let left = scope.spawn(|| chunk.rows().unwrap().as_ptr() as usize);
            let right = scope.spawn(|| chunk.rows().unwrap().as_ptr() as usize);
            (left.join().unwrap(), right.join().unwrap())
        });
        assert_eq!(addresses.0, addresses.1);
        assert_eq!(manager.allocated(), admitted);
        assert_eq!(chunk.rows().unwrap().len(), chunk.row_count());
        for row in chunk.rows().unwrap() {
            let Value::Int64(id) = row[0] else {
                panic!("integer id")
            };
            assert_eq!(id % 3, 1);
            assert_eq!(row[1], payload);
            assert!(seen.insert(id), "duplicate logical row");
        }
    }
    assert_eq!(seen, (0..4096).filter(|id| id % 3 == 1).collect());
    assert_eq!(manager.allocated(), baseline);
}

#[cfg(all(
    target_os = "linux",
    feature = "spill",
    feature = "lpg",
    feature = "gql"
))]
fn bounded_lpg_sort_query() -> &'static str {
    "UNWIND range(0, 65535) AS i RETURN 32767 - (i % 32768) AS value ORDER BY value LIMIT 768"
}

#[cfg(all(
    target_os = "linux",
    feature = "spill",
    feature = "lpg",
    feature = "gql"
))]
fn assert_sort_disk_denial(error: Error) {
    assert_eq!(
        error.error_code(),
        ErrorCode::StorageFull,
        "{error:?}; {error}"
    );
    assert!(
        error.to_string().contains("spill disk quota exceeded"),
        "{error}"
    );
}

#[cfg(all(
    target_os = "linux",
    feature = "spill",
    feature = "lpg",
    feature = "gql"
))]
fn assert_sort_rows_and_cleanup(result: &QueryResult, root: &std::path::Path, database: &GrafeoDB) {
    assert_eq!(result.row_count(), 768);
    for (index, row) in result.rows().iter().enumerate() {
        assert_eq!(
            row,
            &vec![grafeo_common::types::Value::Int64(
                i64::try_from(index / 2).unwrap()
            )]
        );
    }
    assert_sort_cleanup(root, database);
}

#[cfg(all(
    target_os = "linux",
    feature = "spill",
    feature = "lpg",
    feature = "gql"
))]
fn assert_sort_cleanup(root: &std::path::Path, database: &GrafeoDB) {
    let namespace = root.join(format!("grafeo-store-{}", database.store_id()));
    let entries: Vec<_> = std::fs::read_dir(namespace)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(
        entries,
        vec![std::ffi::OsString::from(".grafeo-spill-root")]
    );
}

#[cfg(all(
    target_os = "linux",
    feature = "spill",
    feature = "lpg",
    feature = "gql"
))]
#[test]
fn lpg_sort_spill_quota_covers_cold_warm_and_bounded_output() {
    let denied_root = tempfile::tempdir().unwrap();
    let config = grafeo_engine::Config::in_memory().with_memory_limit(2 << 20);
    let denied = GrafeoDB::with_config(
        config
            .clone()
            .with_spill_path(denied_root.path())
            .with_max_query_spill_bytes(0),
    )
    .unwrap();
    assert_sort_disk_denial(
        denied
            .session()
            .execute(bounded_lpg_sort_query())
            .expect_err("cold sort must require spill"),
    );
    assert_sort_cleanup(denied_root.path(), &denied);

    let positive_root = tempfile::tempdir().unwrap();
    let database = GrafeoDB::with_config(
        config
            .with_spill_path(positive_root.path())
            .with_max_query_spill_bytes(64 << 20),
    )
    .unwrap();
    let session = database.session();
    // Only a successful first call primes the physical cache. Reuse this exact
    // Session/query for the second call; failed cold calls do not prove warmth.
    for _ in 0..2 {
        let result = session.execute(bounded_lpg_sort_query()).unwrap();
        assert_sort_rows_and_cleanup(&result, positive_root.path(), &database);
    }
}

#[cfg(all(
    target_os = "linux",
    feature = "spill",
    feature = "lpg",
    feature = "gql",
    feature = "cypher"
))]
#[test]
fn cypher_sort_spill_quota_covers_cold_and_warm_cache() {
    let denied_root = tempfile::tempdir().unwrap();
    let config = grafeo_engine::Config::in_memory().with_memory_limit(2 << 20);
    let denied = GrafeoDB::with_config(
        config
            .clone()
            .with_spill_path(denied_root.path())
            .with_max_query_spill_bytes(0),
    )
    .unwrap();
    assert_sort_disk_denial(
        denied
            .session()
            .execute_cypher(bounded_lpg_sort_query())
            .expect_err("cold Cypher sort must require spill"),
    );
    assert_sort_cleanup(denied_root.path(), &denied);

    let positive_root = tempfile::tempdir().unwrap();
    let database = GrafeoDB::with_config(
        config
            .with_spill_path(positive_root.path())
            .with_max_query_spill_bytes(64 << 20),
    )
    .unwrap();
    let session = database.session();
    for _ in 0..2 {
        let result = session.execute_cypher(bounded_lpg_sort_query()).unwrap();
        assert_sort_rows_and_cleanup(&result, positive_root.path(), &database);
    }
}

#[cfg(all(
    target_os = "linux",
    feature = "spill",
    feature = "lpg",
    feature = "gql"
))]
fn assert_sort_wrapper_preserves_prior_write(query: &str, skip: usize) {
    let directory = tempfile::tempdir().unwrap();
    let database = GrafeoDB::with_config(
        grafeo_engine::Config::in_memory()
            .with_memory_limit(2 << 20)
            .with_spill_path(directory.path())
            .with_max_query_spill_bytes(0),
    )
    .unwrap();
    let mut session = database.session();
    session.begin_transaction().unwrap();
    session.execute("CREATE (:PriorSortWrite)").unwrap();
    assert_sort_disk_denial(
        session
            .execute(query)
            .expect_err("wrapped sort must require spill"),
    );
    assert_sort_cleanup(directory.path(), &database);
    assert_eq!(
        session
            .execute("MATCH (n:PriorSortWrite) RETURN count(n)")
            .unwrap()
            .rows(),
        &[vec![grafeo_common::types::Value::Int64(1)]]
    );
    session.rollback().unwrap();

    let positive_root = tempfile::tempdir().unwrap();
    let positive = GrafeoDB::with_config(
        grafeo_engine::Config::in_memory()
            .with_memory_limit(2 << 20)
            .with_spill_path(positive_root.path())
            .with_max_query_spill_bytes(64 << 20),
    )
    .unwrap();
    let mut positive_session = positive.session();
    positive_session.begin_transaction().unwrap();
    positive_session
        .execute("CREATE (:PriorSortWrite)")
        .unwrap();
    let result = positive_session.execute(query).unwrap();
    assert_eq!(result.row_count(), 768);
    for (index, row) in result.rows().iter().enumerate() {
        assert_eq!(
            row,
            &vec![grafeo_common::types::Value::Int64(
                i64::try_from(usize::midpoint(index, skip)).unwrap()
            )]
        );
    }
    assert_sort_cleanup(positive_root.path(), &positive);
    assert_eq!(
        positive_session
            .execute("MATCH (n:PriorSortWrite) RETURN count(n)")
            .unwrap()
            .rows(),
        &[vec![grafeo_common::types::Value::Int64(1)]]
    );
    positive_session.rollback().unwrap();
}

#[cfg(all(
    target_os = "linux",
    feature = "spill",
    feature = "lpg",
    feature = "gql"
))]
#[test]
fn lpg_sort_skip_preserves_transaction_on_quota_denial() {
    assert_sort_wrapper_preserves_prior_write(
        "UNWIND range(0, 65535) AS i RETURN 32767 - (i % 32768) AS value ORDER BY value SKIP 5 LIMIT 768",
        5,
    );
}

#[cfg(all(
    target_os = "linux",
    feature = "spill",
    feature = "lpg",
    feature = "gql"
))]
#[test]
fn lpg_sort_hidden_key_preserves_transaction_on_quota_denial() {
    assert_sort_wrapper_preserves_prior_write(
        "UNWIND range(0, 65535) AS i WITH 32767 - (i % 32768) AS value, i AS sort_key RETURN value ORDER BY value, sort_key LIMIT 768",
        0,
    );
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn lpg_sort_without_spill_rejects_resident_budget_overflow() {
    let database =
        GrafeoDB::with_config(grafeo_engine::Config::in_memory().with_memory_limit(2 << 20))
            .unwrap();
    let error = database
        .session()
        .execute(
            "UNWIND range(0, 65535) AS i RETURN 32767 - (i % 32768) AS value ORDER BY value LIMIT 768",
        )
        .expect_err("cold sort without spill must obey the resident budget");
    assert_eq!(
        error.error_code(),
        ErrorCode::StorageFull,
        "resident sort denial must remain structured: {error:?}; {error}"
    );
}
