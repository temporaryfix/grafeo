//! Targeted regression benchmarks for documented performance vectors.
//!
//! These benchmarks cover the three regression categories identified in
//! performance-degradation.md (50-120% multi-hop regression between v0.5.6
//! and v0.5.21):
//!
//! 1. Multi-hop traversal: vtable dispatch + MVCC compounding at depth
//! 2. Repeated execution: parser overhead compounding across many queries
//! 3. Edge type filtering: MVCC edge_type_versioned lookup overhead
//!
//! Run with: cargo bench -p grafeo-engine --bench regression_bench
//! All features: cargo bench --all-features --bench regression_bench
// reason: criterion_group! expansion from codspeed-criterion-compat does not
// carry doc comments on the generated wrapper functions.
#![allow(missing_docs)]
#![allow(
    unexpected_cfgs,
    reason = "CodSpeed supplies its custom instrumentation cfg"
)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group};

#[cfg(any(
    all(feature = "lpg", feature = "gql"),
    all(feature = "triple-store", feature = "sparql")
))]
use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
#[cfg(all(feature = "lpg", feature = "gql"))]
use grafeo_engine::ResultLimits;

// ============================================================================
// Setup helpers
// ============================================================================

/// Sets up a social graph with Person nodes and KNOWS edges.
fn setup_social_graph(node_count: usize, edge_multiplier: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    for i in 0..node_count {
        let query = format!(
            "INSERT (:Person {{id: {}, name: 'User{}', age: {}}})",
            i,
            i,
            20 + (i % 50)
        );
        session.execute(&query).unwrap();
    }

    let edge_count = node_count * edge_multiplier;
    for i in 0..edge_count {
        let src = i % node_count;
        let dst = (i * 7 + 13) % node_count;
        if src != dst {
            let query = format!(
                "MATCH (a:Person {{id: {}}}), (b:Person {{id: {}}}) CREATE (a)-[:KNOWS]->(b)",
                src, dst
            );
            let _ = session.execute(&query);
        }
    }

    db
}

/// Sets up a graph with multiple edge types for filtering benchmarks.
/// Creates KNOWS, FOLLOWS, and LIKES edges between Person nodes.
fn setup_multi_type_graph(node_count: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    for i in 0..node_count {
        let query = format!("INSERT (:Person {{id: {}, name: 'User{}'}})", i, i);
        session.execute(&query).unwrap();
    }

    let types = ["KNOWS", "FOLLOWS", "LIKES"];
    let edges_per_type = node_count * 3;
    for (type_idx, edge_type) in types.iter().enumerate() {
        for i in 0..edges_per_type {
            let src = i % node_count;
            let dst = (i * (type_idx + 3) + 7) % node_count;
            if src != dst {
                let query = format!(
                    "MATCH (a:Person {{id: {}}}), (b:Person {{id: {}}}) CREATE (a)-[:{}]->(b)",
                    src, dst, edge_type
                );
                let _ = session.execute(&query);
            }
        }
    }

    db
}

// ============================================================================
// Multi-hop traversal benchmarks
// ============================================================================

fn bench_1hop_1k(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    let mut group = c.benchmark_group("multihop");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_1hop_1k", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (a:Person {id: 0})-[:KNOWS]->(b) RETURN b.id")
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

fn bench_2hop_1k(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    let mut group = c.benchmark_group("multihop");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_2hop_1k", |b| {
        b.iter(|| {
            let result = session
                .execute(
                    "MATCH (a:Person {id: 0})-[:KNOWS]->(b)-[:KNOWS]->(c) \
                     RETURN DISTINCT c.id",
                )
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

fn bench_3hop_1k(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    let mut group = c.benchmark_group("multihop");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_3hop_1k", |b| {
        b.iter(|| {
            let result = session
                .execute(
                    "MATCH (a:Person {id: 0})-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(d) \
                     RETURN DISTINCT d.id LIMIT 5000",
                )
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

fn bench_1hop_5k(c: &mut Criterion) {
    let db = setup_social_graph(5_000, 5);
    let session = db.session();

    let mut group = c.benchmark_group("multihop");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_1hop_5k", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (a:Person {id: 0})-[:KNOWS]->(b) RETURN b.id")
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

fn bench_fan_out_5k(c: &mut Criterion) {
    let db = setup_social_graph(5_000, 5);
    let session = db.session();

    let mut group = c.benchmark_group("multihop");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_fan_out_5k", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (a:Person)-[:KNOWS]->(b) RETURN COUNT(b)")
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

// ============================================================================
// Repeated execution benchmarks
// ============================================================================

fn bench_repeat_unique_100(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    let mut group = c.benchmark_group("repeated");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_repeat_unique_100", |b| {
        b.iter(|| {
            for i in 0..100 {
                let query = format!("MATCH (n:Person {{id: {}}}) RETURN n.name", i);
                let result = session.execute(&query).unwrap();
                black_box(result);
            }
        });
    });

    group.finish();
}

fn bench_repeat_unique_500(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    let mut group = c.benchmark_group("repeated");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_repeat_unique_500", |b| {
        b.iter(|| {
            for i in 0..500 {
                let query = format!("MATCH (n:Person {{id: {}}}) RETURN n.name", i);
                let result = session.execute(&query).unwrap();
                black_box(result);
            }
        });
    });

    group.finish();
}

fn bench_repeat_cached_500(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    let mut group = c.benchmark_group("repeated");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_repeat_cached_500", |b| {
        b.iter(|| {
            for _ in 0..500 {
                let result = session
                    .execute("MATCH (n:Person {id: 42}) RETURN n.name")
                    .unwrap();
                black_box(result);
            }
        });
    });

    group.finish();
}

// ============================================================================
// Edge type filtering benchmarks
// ============================================================================

fn bench_edge_filter_single(c: &mut Criterion) {
    let db = setup_multi_type_graph(2_000);
    let session = db.session();

    let mut group = c.benchmark_group("edge_filter");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_edge_filter_single", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (a:Person {id: 0})-[:KNOWS]->(b) RETURN b.id")
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

fn bench_edge_filter_follows(c: &mut Criterion) {
    let db = setup_multi_type_graph(2_000);
    let session = db.session();

    let mut group = c.benchmark_group("edge_filter");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    // Second edge type filter: tests MVCC edge_type_versioned with a
    // different type in the same graph (compare against single/KNOWS).
    group.bench_function("regression_edge_filter_follows", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (a:Person {id: 0})-[:FOLLOWS]->(b) RETURN b.id")
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

fn bench_edge_filter_any(c: &mut Criterion) {
    let db = setup_multi_type_graph(2_000);
    let session = db.session();

    let mut group = c.benchmark_group("edge_filter");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_edge_filter_any", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (a:Person {id: 0})-->(b) RETURN b.id")
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

// ============================================================================
// Stream output benchmarks
// ============================================================================

fn bench_stream_output(c: &mut Criterion) {
    #[cfg(all(feature = "lpg", feature = "gql"))]
    {
        const ROWS: usize = 4_096;
        const QUERY: &str = "MATCH (p:Person) RETURN p.id, p.name";

        // Direct native setup keeps query parsing and writes out of the measurement.
        let db = GrafeoDB::new_in_memory();
        let suffix = "x".repeat(54);
        for id in 0..ROWS {
            let node = db.create_node(&["Person"]);
            db.set_node_property(node, "id", Value::Int64(i64::try_from(id).unwrap()))
                .unwrap();
            let name = format!("Alix-{id:04}-{suffix}");
            assert_eq!(name.len(), 64);
            db.set_node_property(node, "name", Value::String(name.into()))
                .unwrap();
        }

        let mut group = c.benchmark_group("stream_output");
        group.measurement_time(Duration::from_secs(10));
        group.sample_size(50);
        group.warm_up_time(Duration::from_secs(3));

        group.bench_function("chunks_4096", |b| {
            b.iter(|| {
                let mut stream = db.execute_streaming(black_box(QUERY)).unwrap();
                let mut rows = 0;
                while let Some(chunk) = stream.next_chunk().unwrap() {
                    rows += black_box(chunk.row_count());
                }
                assert_eq!(rows, ROWS);
                black_box(rows)
            });
        });

        group.bench_function("rows_4096", |b| {
            b.iter(|| {
                let stream = db.execute_streaming(black_box(QUERY)).unwrap();
                let mut rows = 0;
                for row in stream.into_row_iter() {
                    black_box(row.unwrap());
                    rows += 1;
                }
                assert_eq!(rows, ROWS);
                black_box(rows)
            });
        });

        group.bench_function("collect_4096", |b| {
            b.iter(|| {
                let result = db
                    .execute_streaming(black_box(QUERY))
                    .unwrap()
                    .collect(ResultLimits::default())
                    .unwrap();
                assert_eq!(result.row_count(), ROWS);
                black_box(result)
            });
        });

        group.finish();
    }
    #[cfg(not(all(feature = "lpg", feature = "gql")))]
    let _ = c;
}

// ============================================================================
// Groups and main
// ============================================================================

fn bench_spill_root(c: &mut Criterion) {
    let resident = GrafeoDB::new_in_memory();
    let mut group = c.benchmark_group("spill_root");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));
    group.bench_function("resident_scalar", |b| {
        b.iter(|| black_box(resident.execute(black_box("RETURN 1")).unwrap()));
    });
    #[cfg(feature = "spill")]
    {
        let parent = tempfile::tempdir().unwrap();
        let configured = GrafeoDB::with_config(
            grafeo_engine::Config::in_memory().with_spill_path(parent.path()),
        )
        .unwrap();
        group.bench_function("configured_scalar", |b| {
            b.iter(|| black_box(configured.execute(black_box("RETURN 1")).unwrap()));
        });
    }
    #[cfg(all(feature = "spill", feature = "sparql", feature = "triple-store"))]
    {
        use std::fmt::Write as _;
        let parent = tempfile::tempdir().unwrap();
        let config = grafeo_engine::Config::in_memory()
            .with_graph_model(grafeo_engine::GraphModel::Rdf)
            .with_memory_limit(2 << 20)
            .with_spill_path(parent.path());
        let mut query =
            String::from("SELECT DISTINCT (STR(?term) AS ?value) WHERE { VALUES ?term { ");
        for index in 0..4096 {
            write!(query, "<urn:pressure:{}> ", (index * 37) % 4096).unwrap();
        }
        query.push_str("} }");
        let denied = GrafeoDB::with_config(config.clone().with_max_query_spill_bytes(0))
            .unwrap()
            .execute_sparql(&query)
            .unwrap_err();
        assert_eq!(
            denied.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        let spilled = GrafeoDB::with_config(config).unwrap();
        group.bench_function("forced_distinct_4096", |b| {
            b.iter(|| {
                let result = spilled.execute_sparql(black_box(&query)).unwrap();
                assert_eq!(result.row_count(), 4096);
                black_box(result)
            });
        });
    }
    group.finish();
}

#[cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "spill",
    feature = "async-storage"
))]
async fn execute_async_sort_benchmark(
    database: std::sync::Arc<GrafeoDB>,
    query: std::sync::Arc<str>,
) -> grafeo_engine::database::QueryResult {
    use grafeo_engine::query::executor::AsyncSortDispatch;
    let dispatch = tokio::task::spawn_blocking(move || {
        database.execute_or_prepare_async_sort(&query, Default::default(), Default::default())
    })
    .await
    .unwrap()
    .unwrap();
    match dispatch {
        AsyncSortDispatch::Completed(result) => result,
        AsyncSortDispatch::Prepared(prepared) => prepared.execute().await.unwrap(),
    }
}

fn bench_async_sort_resource(c: &mut Criterion) {
    #[cfg(all(
        feature = "lpg",
        feature = "gql",
        feature = "spill",
        feature = "async-storage"
    ))]
    {
        use std::sync::Arc;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(1)
            .enable_time()
            .build()
            .unwrap();
        let mut group = c.benchmark_group("async_sort_resource");
        group.measurement_time(Duration::from_secs(10));
        group.sample_size(50);
        group.warm_up_time(Duration::from_secs(3));
        let scalar_database = Arc::new(GrafeoDB::new_in_memory());
        let scalar_query: Arc<str> = "RETURN 1 AS value".into();
        group.bench_function("scalar_fallback", |b| {
            b.iter(|| {
                let result = runtime.block_on(execute_async_sort_benchmark(
                    Arc::clone(&scalar_database),
                    Arc::clone(&scalar_query),
                ));
                assert_eq!(result.row_count(), 1);
                black_box(result)
            });
        });
        for rows in [64, 4096] {
            let database = Arc::new(GrafeoDB::new_in_memory());
            let query: Arc<str> = format!(
                "UNWIND range(0, {}) AS i RETURN {} - i AS value ORDER BY value",
                rows - 1,
                rows - 1
            )
            .into();
            let result = runtime.block_on(execute_async_sort_benchmark(
                Arc::clone(&database),
                Arc::clone(&query),
            ));
            assert_eq!(result.row_count(), rows);
            assert_eq!(result.rows()[0], vec![Value::Int64(0)]);
            assert_eq!(
                result.rows()[rows - 1],
                vec![Value::Int64((rows - 1) as i64)]
            );
            group.bench_function(format!("resident_{rows}"), |b| {
                b.iter(|| {
                    let result = runtime.block_on(execute_async_sort_benchmark(
                        Arc::clone(&database),
                        Arc::clone(&query),
                    ));
                    assert_eq!(result.row_count(), rows);
                    black_box(result)
                });
            });
        }
        // Preflight controls may select a common parent/candidate spill fixture
        // without rebuilding. Timed comparisons must use identical recorded inputs.
        let memory_mib = std::env::var("GRAFEO_ASYNC_SORT_BENCH_MEMORY_MIB")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(2);
        let key_bytes = std::env::var("GRAFEO_ASYNC_SORT_BENCH_KEY_BYTES")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(512);
        assert!((1..=64).contains(&memory_mib));
        assert!((64..=4096).contains(&key_bytes));
        eprintln!("async sort fixture: memory_mib={memory_mib}, key_bytes={key_bytes}");
        let directory = tempfile::tempdir().unwrap();
        let config = grafeo_engine::Config::in_memory()
            .with_memory_limit(memory_mib << 20)
            .with_spill_path(directory.path());
        let seed = |database: &GrafeoDB| {
            for value in 0..4096_i64 {
                database.create_node_with_props(
                    &["AsyncSort"],
                    [
                        ("value", Value::Int64(value)),
                        (
                            "key",
                            Value::from(format!("{:04}-{}", 4095 - value, "x".repeat(key_bytes))),
                        ),
                    ],
                );
            }
        };
        let query: Arc<str> = "MATCH (n:AsyncSort) RETURN n.value AS value ORDER BY n.key".into();
        let denied =
            Arc::new(GrafeoDB::with_config(config.clone().with_max_query_spill_bytes(0)).unwrap());
        seed(&denied);
        let error = denied.execute(&query).unwrap_err();
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert!(error.to_string().contains("spill disk quota exceeded"));
        // Candidate-only preflight: prove this exact fixture reaches the scheduled
        // sorter and its disk denial before the timed workload begins.
        let denied_owner = Arc::clone(&denied);
        let denied_query = Arc::clone(&query);
        runtime.block_on(async move {
            use grafeo_engine::query::executor::AsyncSortDispatch;
            let dispatch = tokio::task::spawn_blocking(move || {
                denied_owner.execute_or_prepare_async_sort(
                    &denied_query,
                    Default::default(),
                    Default::default(),
                )
            })
            .await
            .unwrap()
            .unwrap();
            let AsyncSortDispatch::Prepared(prepared) = dispatch else {
                panic!("forced async sort benchmark fell back to synchronous execution");
            };
            let error = prepared.execute().await.unwrap_err();
            assert_eq!(
                error.error_code(),
                grafeo_common::utils::error::ErrorCode::StorageFull
            );
            assert!(error.to_string().contains("spill disk quota exceeded"));
        });
        // End candidate-only preflight.
        drop(denied);
        let database = Arc::new(GrafeoDB::with_config(config).unwrap());
        seed(&database);
        let result = runtime.block_on(execute_async_sort_benchmark(
            Arc::clone(&database),
            Arc::clone(&query),
        ));
        assert_eq!(result.row_count(), 4096);
        for (index, row) in result.rows().iter().enumerate() {
            assert_eq!(row, &vec![Value::Int64(4095 - index as i64)]);
        }
        group.bench_function("forced_hidden_key_4096", |b| {
            b.iter(|| {
                let result = runtime.block_on(execute_async_sort_benchmark(
                    Arc::clone(&database),
                    Arc::clone(&query),
                ));
                assert_eq!(result.row_count(), 4096);
                black_box(result)
            });
        });
        group.finish();
    }
    #[cfg(not(all(
        feature = "lpg",
        feature = "gql",
        feature = "spill",
        feature = "async-storage"
    )))]
    let _ = c;
}

fn bench_sort_resource(c: &mut Criterion) {
    const RESIDENT: &str =
        "UNWIND range(0, 4095) AS i RETURN 2047 - (i % 2048) AS value ORDER BY value LIMIT 128";
    let database = GrafeoDB::new_in_memory();
    let cached = database.session();
    assert_eq!(cached.execute(RESIDENT).unwrap().row_count(), 128);
    let mut owned = database.session();
    owned.begin_transaction().unwrap();

    let mut group = c.benchmark_group("sort_resource");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));
    group.bench_function("resident_cached_4096", |b| {
        b.iter(|| {
            let result = cached.execute(black_box(RESIDENT)).unwrap();
            assert_eq!(result.row_count(), 128);
            black_box(result)
        });
    });
    group.bench_function("resident_owned_4096", |b| {
        b.iter(|| {
            let result = owned.execute(black_box(RESIDENT)).unwrap();
            assert_eq!(result.row_count(), 128);
            black_box(result)
        });
    });
    owned.rollback().unwrap();

    #[cfg(feature = "spill")]
    {
        const FORCED: &str = "UNWIND range(0, 24575) AS i RETURN 12287 - (i % 12288) AS value ORDER BY value LIMIT 768";
        let parent = tempfile::tempdir().unwrap();
        let config = grafeo_engine::Config::in_memory()
            .with_memory_limit(2 << 20)
            .with_spill_path(parent.path());
        let denied_database =
            GrafeoDB::with_config(config.clone().with_max_query_spill_bytes(0)).unwrap();
        let mut denied = denied_database.session();
        denied.begin_transaction().unwrap();
        let error = denied.execute(FORCED).unwrap_err();
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert!(error.to_string().contains("spill disk quota exceeded"));
        denied.rollback().unwrap();
        let database = GrafeoDB::with_config(config).unwrap();
        let mut session = database.session();
        session.begin_transaction().unwrap();
        group.bench_function("forced_owned_24576", |b| {
            b.iter(|| {
                let result = session.execute(black_box(FORCED)).unwrap();
                assert_eq!(result.row_count(), 768);
                black_box(result)
            });
        });
        session.rollback().unwrap();
    }
    group.finish();
}

#[cfg(all(feature = "triple-store", feature = "sparql"))]
fn rdf_order_query(keys: usize) -> String {
    use std::fmt::Write as _;

    let mut query = String::from("SELECT ?key ?ordinal WHERE { VALUES (?key ?ordinal) { ");
    for key in 0..keys {
        write!(query, "(\"k{key:04}\" 0) (\"k{key:04}\" 1) ").unwrap();
    }
    query.push_str("} } ORDER BY ASC(?key) DESC(?ordinal)");
    query
}

fn bench_rdf_sort_resource(c: &mut Criterion) {
    #[cfg(all(feature = "triple-store", feature = "sparql"))]
    {
        let resident_query = rdf_order_query(2048);
        let resident = GrafeoDB::with_config(
            grafeo_engine::Config::in_memory().with_graph_model(grafeo_engine::GraphModel::Rdf),
        )
        .unwrap();
        let resident_result = resident.execute_sparql(&resident_query).unwrap();
        assert_eq!(resident_result.row_count(), 4096);
        assert_eq!(
            resident_result.rows().first().unwrap()[0],
            Value::from("k0000")
        );
        assert_eq!(resident_result.rows().first().unwrap()[1], Value::Int64(1));
        assert_eq!(
            resident_result.rows().last().unwrap()[0],
            Value::from("k2047")
        );
        assert_eq!(resident_result.rows().last().unwrap()[1], Value::Int64(0));

        drop(resident_result);
        let mut resident_group = c.benchmark_group("rdf_sort_resource/resident");
        resident_group.measurement_time(Duration::from_secs(10));
        resident_group.sample_size(50);
        resident_group.warm_up_time(Duration::from_secs(3));
        resident_group.bench_function("order_by_4096", |b| {
            b.iter(|| {
                let result = resident.execute_sparql(black_box(&resident_query)).unwrap();
                assert_eq!(result.row_count(), 4096);
                assert_eq!(result.rows().first().unwrap()[0], Value::from("k0000"));
                assert_eq!(result.rows().first().unwrap()[1], Value::Int64(1));
                assert_eq!(result.rows().last().unwrap()[0], Value::from("k2047"));
                assert_eq!(result.rows().last().unwrap()[1], Value::Int64(0));
                black_box(result)
            });
        });
        resident_group.finish();

        #[cfg(feature = "spill")]
        {
            let configured_query = rdf_order_query(4096);
            let parent = tempfile::tempdir().unwrap();
            let configured = GrafeoDB::with_config(
                grafeo_engine::Config::in_memory()
                    .with_graph_model(grafeo_engine::GraphModel::Rdf)
                    .with_memory_limit(2 << 20)
                    .with_spill_path(parent.path())
                    .with_max_query_spill_bytes(64 << 20),
            )
            .unwrap();
            // Prove the timed label crosses disk, independently of elapsed time.
            #[cfg(feature = "spill")]
            {
                let denied_parent = tempfile::tempdir().unwrap();
                let denied = GrafeoDB::with_config(
                    grafeo_engine::Config::in_memory()
                        .with_graph_model(grafeo_engine::GraphModel::Rdf)
                        .with_memory_limit(2 << 20)
                        .with_spill_path(denied_parent.path())
                        .with_max_query_spill_bytes(0),
                )
                .unwrap();
                let error = denied
                    .execute_sparql(&configured_query)
                    .expect_err("configured RDF sort must cross the spill quota");
                assert_eq!(
                    error.error_code(),
                    grafeo_common::utils::error::ErrorCode::StorageFull
                );
                assert!(
                    error.to_string().contains("spill disk quota exceeded"),
                    "{error}"
                );
            }
            let expected: Vec<Vec<Value>> = (0..4096)
                .flat_map(|key| {
                    [
                        vec![Value::from(format!("k{key:04}")), Value::Int64(1)],
                        vec![Value::from(format!("k{key:04}")), Value::Int64(0)],
                    ]
                })
                .collect();
            assert_eq!(
                resident.execute_sparql(&configured_query).unwrap().rows(),
                expected.as_slice()
            );
            let configured_result = configured.execute_sparql(&configured_query).unwrap();
            assert_eq!(configured_result.rows(), expected.as_slice());
            assert_eq!(configured_result.row_count(), 8192);
            assert_eq!(
                configured_result.rows().first().unwrap()[0],
                Value::from("k0000")
            );
            assert_eq!(
                configured_result.rows().first().unwrap()[1],
                Value::Int64(1)
            );
            assert_eq!(
                configured_result.rows().last().unwrap()[0],
                Value::from("k4095")
            );
            assert_eq!(configured_result.rows().last().unwrap()[1], Value::Int64(0));

            drop(configured_result);
            let mut configured_group = c.benchmark_group("rdf_sort_resource/configured");
            configured_group.measurement_time(Duration::from_secs(10));
            configured_group.sample_size(50);
            configured_group.warm_up_time(Duration::from_secs(3));
            configured_group.bench_function("order_by_8192_resident", |b| {
                b.iter(|| {
                    let result = resident
                        .execute_sparql(black_box(&configured_query))
                        .unwrap();
                    assert_eq!(result.row_count(), 8192);
                    assert_eq!(result.rows().first().unwrap()[0], Value::from("k0000"));
                    assert_eq!(result.rows().first().unwrap()[1], Value::Int64(1));
                    assert_eq!(result.rows().last().unwrap()[0], Value::from("k4095"));
                    assert_eq!(result.rows().last().unwrap()[1], Value::Int64(0));
                    black_box(result)
                });
            });
            configured_group.bench_function("order_by_8192_forced_spill", |b| {
                b.iter(|| {
                    let result = configured
                        .execute_sparql(black_box(&configured_query))
                        .unwrap();
                    assert_eq!(result.row_count(), 8192);
                    assert_eq!(result.rows().first().unwrap()[0], Value::from("k0000"));
                    assert_eq!(result.rows().first().unwrap()[1], Value::Int64(1));
                    assert_eq!(result.rows().last().unwrap()[0], Value::from("k4095"));
                    assert_eq!(result.rows().last().unwrap()[1], Value::Int64(0));
                    black_box(result)
                });
            });
            configured_group.finish();
        }
    }
    #[cfg(not(all(feature = "triple-store", feature = "sparql")))]
    let _ = c;
}

fn bench_native_aggregate_resource(c: &mut Criterion) {
    #[cfg(all(feature = "lpg", feature = "gql", feature = "spill"))]
    {
        use grafeo_common::memory::buffer::MemoryRegion;

        const GROUPS: usize = 1024;
        const QUERY: &str = "MATCH (n:AggregateCost) RETURN n.bucket AS bucket, sum(n.value) AS total, count(*) AS count";
        const MEMORY_BYTES: usize = 32 << 20;
        const PRESSURE_BYTES: usize = 28 << 20;
        const SPILL_BYTES: u64 = 64 << 20;

        // Four round-robin passes revisit every group. Seeding and all resource
        // setup stay outside timing; there is no sort or output DISTINCT which
        // could independently satisfy the forced-spill preflight.
        let seed = |database: &GrafeoDB| {
            for value in 1..=4_i64 {
                for bucket in 0..GROUPS {
                    database.create_node_with_props(
                        &["AggregateCost"],
                        [
                            ("bucket", Value::Int64(i64::try_from(bucket).unwrap())),
                            ("value", Value::Int64(value)),
                        ],
                    );
                }
            }
        };
        let validate = |result: &grafeo_engine::database::QueryResult| {
            assert_eq!(result.row_count(), GROUPS);
            let mut seen = [false; GROUPS];
            for row in result.rows() {
                assert_eq!(row.len(), 3);
                let Value::Int64(bucket) = row[0] else {
                    panic!("aggregate group key must remain an integer");
                };
                let bucket = usize::try_from(bucket).unwrap();
                assert!(bucket < GROUPS && !seen[bucket]);
                seen[bucket] = true;
                assert_eq!(row[1], Value::Int64(10));
                assert_eq!(row[2], Value::Int64(4));
            }
            assert!(seen.into_iter().all(|present| present));
        };
        let config = grafeo_engine::Config::in_memory().with_memory_limit(MEMORY_BYTES);
        let resident = GrafeoDB::with_config(config.clone()).unwrap();
        seed(&resident);
        let resident_directory = tempfile::tempdir().unwrap();
        let configured = GrafeoDB::with_config(
            config
                .clone()
                .with_spill_path(resident_directory.path())
                .with_max_query_spill_bytes(0),
        )
        .unwrap();
        seed(&configured);

        // This fixed COST fixture models competing resident work. It is not the
        // separate 3 MiB N/2N/4N aggregate retained-memory qualification. The
        // public RAII grant keeps pressure above 85%, with about 2.4 MiB below
        // the 95% hard limit for the same grouped caller on parent and candidate.
        let forced_directory = tempfile::tempdir().unwrap();
        let forced_config = config
            .with_spill_path(forced_directory.path())
            .with_max_query_spill_bytes(SPILL_BYTES);
        let forced = GrafeoDB::with_config(forced_config.clone()).unwrap();
        seed(&forced);
        let _forced_pressure = forced
            .buffer_manager()
            .try_allocate(PRESSURE_BYTES, MemoryRegion::ExecutionBuffers)
            .expect("admit fixed competing resident pressure");
        {
            let denied =
                GrafeoDB::with_config(forced_config.with_max_query_spill_bytes(0)).unwrap();
            seed(&denied);
            let _denied_pressure = denied
                .buffer_manager()
                .try_allocate(PRESSURE_BYTES, MemoryRegion::ExecutionBuffers)
                .expect("admit identical denial-control pressure");
            let mut session = denied.session();
            session.begin_transaction().unwrap();
            let error = session
                .execute(QUERY)
                .expect_err("grouped aggregate must cross the spill quota");
            assert_eq!(
                error.error_code(),
                grafeo_common::utils::error::ErrorCode::StorageFull
            );
            assert!(
                error.to_string().contains("spill disk quota exceeded"),
                "{error}"
            );
            session.rollback().unwrap();
        }

        // Explicit transactions select the owned pipeline, including resource
        // GROUP BY decomposition. Cached pull execution and PROFILE wrappers do
        // not establish this caller's cost or spill behavior.
        let mut plain_session = resident.session();
        let mut configured_session = configured.session();
        let mut forced_session = forced.session();
        plain_session.begin_transaction().unwrap();
        configured_session.begin_transaction().unwrap();
        forced_session.begin_transaction().unwrap();
        validate(&plain_session.execute(QUERY).unwrap());
        // Zero disk quota makes any accidental configured-resident spill fail.
        validate(&configured_session.execute(QUERY).unwrap());
        validate(&forced_session.execute(QUERY).unwrap());

        let mut group = c.benchmark_group("native_aggregate_resource");
        group.measurement_time(Duration::from_secs(10));
        group.sample_size(50);
        group.warm_up_time(Duration::from_secs(3));
        for (name, session) in [
            ("groups_1024_rows_4096_resident", &plain_session),
            (
                "groups_1024_rows_4096_configured_resident",
                &configured_session,
            ),
            ("groups_1024_rows_4096_forced_spill", &forced_session),
        ] {
            group.bench_function(name, |b| {
                b.iter(|| {
                    let result = session.execute(black_box(QUERY)).unwrap();
                    assert_eq!(result.row_count(), GROUPS);
                    black_box(result)
                });
            });
        }
        group.finish();
        plain_session.rollback().unwrap();
        configured_session.rollback().unwrap();
        forced_session.rollback().unwrap();
    }
    #[cfg(not(all(feature = "lpg", feature = "gql", feature = "spill")))]
    let _ = c;
}

fn bench_rdf_aggregate_resource(c: &mut Criterion) {
    #[cfg(all(feature = "triple-store", feature = "sparql"))]
    {
        use std::fmt::Write;

        let db = GrafeoDB::with_config(
            grafeo_engine::Config::in_memory().with_graph_model(grafeo_engine::GraphModel::Rdf),
        )
        .unwrap();
        let mut grouped = String::from(
            "SELECT ?group (SUM(?value) AS ?sum) (COUNT(?value) AS ?count) WHERE { VALUES (?group ?value) { ",
        );
        for key in 0..1024 {
            for value in 1..=4 {
                write!(grouped, "(\"g{key:04}\" {value}) ").unwrap();
            }
        }
        grouped.push_str("} } GROUP BY ?group");
        let mut distinct = String::from(
            "SELECT (COUNT(DISTINCT ?value) AS ?count) (SUM(DISTINCT ?value) AS ?sum) WHERE { VALUES ?value { ",
        );
        for value in 0..2048 {
            write!(distinct, "{value} {value} ").unwrap();
        }
        distinct.push_str("} }");

        let validate = |result: &grafeo_engine::database::QueryResult, many_groups: bool| {
            if many_groups {
                assert_eq!(result.row_count(), 1024);
                assert!(result.rows().iter().all(|row| {
                    row.len() == 3 && row[1] == Value::Int64(10) && row[2] == Value::Int64(4)
                }));
            } else {
                assert_eq!(
                    result.rows(),
                    &[vec![Value::Int64(2048), Value::Int64(2_096_128)]]
                );
            }
        };
        let mut group = c.benchmark_group("rdf_aggregate_resource/resident");
        group.measurement_time(Duration::from_secs(10));
        group.sample_size(50);
        group.warm_up_time(Duration::from_secs(3));
        for (name, query, many_groups) in [
            ("groups_1024_rows_4096", &grouped, true),
            ("hot_distinct_2048_rows_4096", &distinct, false),
        ] {
            validate(&db.execute_sparql(query).unwrap(), many_groups);
            group.bench_function(name, |b| {
                b.iter(|| {
                    let result = db.execute_sparql(black_box(query)).unwrap();
                    validate(&result, many_groups);
                    black_box(result)
                });
            });
        }
        group.finish();

        #[cfg(feature = "spill")]
        {
            // Same public fixtures as rdf_exact_spill: exact decimal AVG,
            // encounter-ordered groups and hot DISTINCT membership.
            let many_query = grouped.replace(
                "(COUNT(?value) AS ?count)",
                "(AVG(?value) AS ?avg) (COUNT(?value) AS ?count)",
            );
            let many_expected: Vec<Vec<Value>> = (0..1024)
                .map(|key| {
                    vec![
                        Value::from(format!("g{key:04}")),
                        Value::Int64(10),
                        Value::RdfLiteral {
                            lexical: "2.5".into(),
                            language: None,
                            datatype: Some("http://www.w3.org/2001/XMLSchema#decimal".into()),
                        },
                        Value::Int64(4),
                    ]
                })
                .collect();
            let mut hot_query = String::from(
                "SELECT (COUNT(DISTINCT ?value) AS ?count) (SUM(DISTINCT ?value) AS ?sum) WHERE { VALUES ?value { ",
            );
            for value in 0..4096 {
                write!(hot_query, "{value} {value} ").unwrap();
            }
            hot_query.push_str("} }");
            // Mirror HotConcatInput: cold groups force the transition before
            // the growing hot group's encounter-ordered fold completes.
            let mut concat_query = String::from(
                "SELECT ?group (GROUP_CONCAT(?value; SEPARATOR=\"|\") AS ?joined) WHERE { VALUES (?group ?value) { ",
            );
            let mut concat_expected: Vec<Vec<Value>> = (0..512)
                .map(|group| {
                    write!(concat_query, "({group} \"cold\") ").unwrap();
                    vec![Value::Int64(group), Value::from("cold")]
                })
                .collect();
            let mut joined = String::new();
            for ordinal in 0..4096 {
                write!(concat_query, "(512 \"{ordinal:04}\") ").unwrap();
                if ordinal != 0 {
                    joined.push('|');
                }
                write!(joined, "{ordinal:04}").unwrap();
            }
            concat_query.push_str("} } GROUP BY ?group");
            concat_expected.push(vec![Value::Int64(512), Value::from(joined)]);
            let cases = [
                ("groups_1024_rows_4096", many_query, many_expected),
                (
                    "hot_distinct_4096_rows_8192",
                    hot_query,
                    vec![vec![Value::Int64(4096), Value::Int64(8_386_560)]],
                ),
                (
                    "hot_concat_4096_cold_groups_512",
                    concat_query,
                    concat_expected,
                ),
            ];
            let parent = tempfile::tempdir().unwrap();
            let config = grafeo_engine::Config::in_memory()
                .with_graph_model(grafeo_engine::GraphModel::Rdf)
                .with_memory_limit(2 << 20)
                .with_spill_path(parent.path());
            let denied =
                GrafeoDB::with_config(config.clone().with_max_query_spill_bytes(0)).unwrap();
            for (name, query, expected) in &cases {
                assert_eq!(
                    db.execute_sparql(query).unwrap().rows(),
                    expected.as_slice(),
                    "{name}"
                );
                let error = denied
                    .execute_sparql(query)
                    .expect_err("aggregate fixture must cross the spill quota");
                assert_eq!(
                    error.error_code(),
                    grafeo_common::utils::error::ErrorCode::StorageFull
                );
                assert!(
                    error.to_string().contains("spill disk quota exceeded"),
                    "{name}: {error}"
                );
            }
            drop(denied);
            let configured =
                GrafeoDB::with_config(config.with_max_query_spill_bytes(64 << 20)).unwrap();
            for (name, query, expected) in &cases {
                assert_eq!(
                    configured.execute_sparql(query).unwrap().rows(),
                    expected.as_slice(),
                    "{name}"
                );
            }
            let mut group = c.benchmark_group("rdf_aggregate_resource/configured");
            group.measurement_time(Duration::from_secs(10));
            group.sample_size(50);
            group.warm_up_time(Duration::from_secs(3));
            for (name, query, expected) in &cases {
                for (mode, database) in [("resident", &db), ("forced_spill", &configured)] {
                    group.bench_function(format!("{name}_{mode}"), |b| {
                        b.iter(|| {
                            let result = database.execute_sparql(black_box(query)).unwrap();
                            assert_eq!(result.row_count(), expected.len());
                            black_box(result)
                        });
                    });
                }
            }
            group.finish();
        }
    }
    #[cfg(not(all(feature = "triple-store", feature = "sparql")))]
    let _ = c;
}

// Candidate-only timing: the parent cached route ignores the memory budget.
// Compare this bounded route with the qualified parent owned-spill measurement.
fn bench_sort_cached_spill(c: &mut Criterion) {
    #[cfg(not(feature = "spill"))]
    let _ = c;
    #[cfg(feature = "spill")]
    {
        const QUERY: &str = "UNWIND range(0, 24575) AS i RETURN 12287 - (i % 12288) AS value ORDER BY value LIMIT 768";
        let parent = tempfile::tempdir().unwrap();
        let config = grafeo_engine::Config::in_memory()
            .with_memory_limit(2 << 20)
            .with_spill_path(parent.path());
        let denied = GrafeoDB::with_config(config.clone().with_max_query_spill_bytes(0)).unwrap();
        let error = denied.execute(QUERY).unwrap_err();
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert!(error.to_string().contains("spill disk quota exceeded"));
        let database = GrafeoDB::with_config(config).unwrap();
        let session = database.session();
        assert_eq!(session.execute(QUERY).unwrap().row_count(), 768);
        let mut group = c.benchmark_group("sort_cached_spill");
        group.measurement_time(Duration::from_secs(10));
        group.sample_size(50);
        group.warm_up_time(Duration::from_secs(3));
        group.bench_function("forced_cached_24576", |b| {
            b.iter(|| {
                let result = session.execute(black_box(QUERY)).unwrap();
                assert_eq!(result.row_count(), 768);
                black_box(result)
            });
        });
        group.finish();
    }
}

criterion_group!(
    multihop_benches,
    bench_1hop_1k,
    bench_2hop_1k,
    bench_3hop_1k,
    bench_1hop_5k,
    bench_fan_out_5k,
);

criterion_group!(
    repeated_benches,
    bench_repeat_unique_100,
    bench_repeat_unique_500,
    bench_repeat_cached_500,
);

criterion_group!(
    edge_filter_benches,
    bench_edge_filter_single,
    bench_edge_filter_follows,
    bench_edge_filter_any,
);

criterion_group!(stream_output_benches, bench_stream_output);
criterion_group!(spill_root_benches, bench_spill_root);
criterion_group!(sort_resource_benches, bench_sort_resource);
criterion_group!(async_sort_resource_benches, bench_async_sort_resource);
criterion_group!(rdf_sort_resource_benches, bench_rdf_sort_resource);
criterion_group!(
    native_aggregate_resource_benches,
    bench_native_aggregate_resource
);
criterion_group!(rdf_aggregate_resource_benches, bench_rdf_aggregate_resource);
criterion_group!(sort_cached_spill_benches, bench_sort_cached_spill);

// CodSpeed's instrumented macro supplies a different group calling convention.
#[cfg(codspeed)]
criterion::criterion_main!(
    multihop_benches,
    repeated_benches,
    edge_filter_benches,
    stream_output_benches,
    spill_root_benches,
    sort_resource_benches,
    async_sort_resource_benches,
    rdf_sort_resource_benches,
    native_aggregate_resource_benches,
    rdf_aggregate_resource_benches,
    sort_cached_spill_benches,
);

#[cfg(not(codspeed))]
fn main() {
    // Dispatch before group setup for the focused output-cost comparison. Check
    // the leading filter only: a baseline value named stream_output is not one.
    if std::env::args().nth(1).as_deref() == Some("stream_output") {
        stream_output_benches();
    } else if std::env::args().nth(1).as_deref() == Some("spill_root") {
        spill_root_benches();
    } else if std::env::args().nth(1).as_deref() == Some("sort_cached_spill") {
        sort_cached_spill_benches();
    } else if std::env::args().nth(1).as_deref() == Some("sort_resource") {
        sort_resource_benches();
    } else if std::env::args().nth(1).as_deref() == Some("async_sort_resource") {
        async_sort_resource_benches();
    } else if std::env::args().nth(1).as_deref() == Some("rdf_sort_resource") {
        rdf_sort_resource_benches();
    } else if std::env::args().nth(1).as_deref() == Some("rdf_aggregate_resource") {
        rdf_aggregate_resource_benches();
    } else if std::env::args().nth(1).as_deref() == Some("native_aggregate_resource") {
        native_aggregate_resource_benches();
    } else {
        multihop_benches();
        repeated_benches();
        edge_filter_benches();
        stream_output_benches();
        spill_root_benches();
        sort_resource_benches();
        async_sort_resource_benches();
        rdf_sort_resource_benches();
        native_aggregate_resource_benches();
        rdf_aggregate_resource_benches();
        sort_cached_spill_benches();
    }

    Criterion::default().configure_from_args().final_summary();
}
