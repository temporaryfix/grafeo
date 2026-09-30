//! Serialization benchmarks for snapshot export/import and Value encoding.
//!
//! Covers the bincode hot paths used by persistence, WAL, and spill-to-disk.
//!
//! Run with: cargo bench -p grafeo-engine --bench serialization_bench
// Bench values are small known constants
#![allow(clippy::cast_possible_wrap)]
// reason: criterion_group! expansion from codspeed-criterion-compat does not
// carry doc comments on the generated wrapper functions.
#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Build a small graph (~50 nodes, ~100 edges) representative of typical workloads.
fn build_bench_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for i in 0..50u64 {
        let n = db.create_node(&["Person"]);
        db.set_node_property(n, "name", Value::String(format!("User{i}").into()))
            .expect("set node property");
        db.set_node_property(n, "age", Value::Int64(20 + (i % 50) as i64))
            .expect("set node property");
        db.set_node_property(
            n,
            "bio",
            Value::String("A short biography for benchmarking serialization throughput.".into()),
        )
        .expect("set node property");
    }
    for i in 0..100u64 {
        let src = grafeo_common::types::NodeId::new(i % 50);
        let dst = grafeo_common::types::NodeId::new((i * 7 + 13) % 50);
        let e = db.create_edge(src, dst, "KNOWS");
        db.set_edge_property(e, "weight", Value::Float64(i as f64 * 0.1))
            .expect("set edge property");
    }
    db
}

// ---------------------------------------------------------------------------
// Snapshot export / import
// ---------------------------------------------------------------------------

fn bench_snapshot_export(c: &mut Criterion) {
    // Hash tables receive fresh randomized layouts at construction. Averaging
    // equivalent fixtures keeps one process's layout from deciding the entire
    // export measurement; fixture construction stays outside the timed loop.
    let fixtures: [GrafeoDB; 32] = std::array::from_fn(|_| build_bench_db());
    let mut next = 0;
    c.bench_function("snapshot_export_50n_100e", |b| {
        b.iter(|| {
            let db = &fixtures[next];
            next = (next + 1) % fixtures.len();
            black_box(db.export_snapshot().unwrap())
        });
    });
}

fn bench_snapshot_import(c: &mut Criterion) {
    let db = build_bench_db();
    let bytes = db.export_snapshot().unwrap();
    c.bench_function("snapshot_import_50n_100e", |b| {
        b.iter(|| black_box(GrafeoDB::import_snapshot(&bytes).unwrap()));
    });
}

fn bench_snapshot_roundtrip(c: &mut Criterion) {
    let db = build_bench_db();
    c.bench_function("snapshot_roundtrip_50n_100e", |b| {
        b.iter(|| {
            let bytes = db.export_snapshot().unwrap();
            black_box(GrafeoDB::import_snapshot(&bytes).unwrap());
        });
    });
}

// ---------------------------------------------------------------------------
// Value encoding / decoding (bincode hot path)
// ---------------------------------------------------------------------------

fn bench_value_encode(c: &mut Criterion) {
    let values: Vec<Value> = vec![
        Value::Int64(42),
        Value::Float64(9.81),
        Value::String("hello world".into()),
        Value::Bool(true),
        Value::Null,
        Value::List(vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)].into()),
    ];

    c.bench_function("value_encode_mixed_6", |b| {
        b.iter(|| {
            for v in &values {
                black_box(bincode::serde::encode_to_vec(v, bincode::config::standard()).unwrap());
            }
        });
    });
}

fn bench_value_decode(c: &mut Criterion) {
    let values: Vec<Value> = vec![
        Value::Int64(42),
        Value::Float64(9.81),
        Value::String("hello world".into()),
        Value::Bool(true),
        Value::Null,
        Value::List(vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)].into()),
    ];

    let encoded: Vec<Vec<u8>> = values
        .iter()
        .map(|v| bincode::serde::encode_to_vec(v, bincode::config::standard()).unwrap())
        .collect();

    c.bench_function("value_decode_mixed_6", |b| {
        b.iter(|| {
            for bytes in &encoded {
                let (v, _): (Value, _) =
                    bincode::serde::decode_from_slice(bytes, bincode::config::standard()).unwrap();
                black_box(v);
            }
        });
    });
}

fn bench_cdc_commit_retention(c: &mut Criterion) {
    #[cfg(feature = "cdc")]
    {
        let mut config = grafeo_engine::Config::in_memory().with_cdc();
        config.cdc_retention.max_epochs = None;
        config.cdc_retention.max_events = Some(128);
        let db = GrafeoDB::with_config(config).unwrap();
        let id = db.session().create_node(&["CdcCost"]);
        c.bench_function("cdc_commit_retention_64_updates", |b| {
            b.iter(|| {
                let mut session = db.session();
                session.begin_transaction().unwrap();
                for value in 0..64 {
                    session
                        .set_node_property(id, "value", Value::Int64(value))
                        .unwrap();
                }
                let epoch = session.commit().unwrap();
                db.gc().unwrap();
                black_box(epoch)
            });
        });
    }
    #[cfg(not(feature = "cdc"))]
    let _ = c;
}

fn bench_cdc_entity_history(c: &mut Criterion) {
    #[cfg(feature = "cdc")]
    {
        // Average independent map/heap layouts while each read still returns
        // one event from a 4096-event feed. Eight stores were too noisy across
        // process launches on two otherwise quiet pinned CPU cores.
        let fixtures: Vec<_> = (0..64)
            .map(|_| {
                let db =
                    GrafeoDB::with_config(grafeo_engine::Config::in_memory().with_cdc()).unwrap();
                let mut session = db.session();
                session.begin_transaction().unwrap();
                let ids: Vec<_> = (0..4096)
                    .map(|_| session.create_node(&["CdcCost"]))
                    .collect();
                session.commit().unwrap();
                (db, grafeo_engine::cdc::EntityId::Node(ids[2048]))
            })
            .collect();
        let mut next = 0;
        c.bench_function("cdc_entity_history_1_of_4096", |b| {
            b.iter(|| {
                let (db, id) = &fixtures[next];
                next = (next + 1) % fixtures.len();
                let rows = db
                    .history_after(
                        &grafeo_engine::cdc::EntityHistoryQuery::new(*id),
                        None,
                        1,
                        4096,
                    )
                    .unwrap();
                assert_eq!(rows.events.len(), 1);
                // Keep destruction timed, but return () so the harness's
                // volatile result sink does not charge per byte of ChangePage.
                drop(black_box(rows));
            });
        });
    }
    #[cfg(not(feature = "cdc"))]
    let _ = c;
}

criterion_group!(
    serialization,
    bench_snapshot_export,
    bench_snapshot_import,
    bench_snapshot_roundtrip,
    bench_value_encode,
    bench_value_decode,
    bench_cdc_commit_retention,
    bench_cdc_entity_history,
);
criterion_main!(serialization);
