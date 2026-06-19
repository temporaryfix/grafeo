//! Benchmark: the cost of Serializable (SSI) isolation vs SnapshotIsolation.
//!
//! The whole MVCC/Serializable arc was correctness-first; this puts numbers on
//! the overhead of the SSI read-recording (read-set + sharded registry) and the
//! commit-time dangerous-structure check. All single-threaded: the recording cost
//! is per-operation, so single-threaded isolates it without scheduler noise. The
//! ratio (Serializable / SnapshotIsolation) is the headline.
//!
//! Run: `cargo bench --bench serializable_overhead`
// reason: criterion_group! expansion from codspeed-criterion-compat does not
// satisfy this lint in some configurations.
#![allow(clippy::incompatible_msrv)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use grafeo_engine::GrafeoDB;
use grafeo_engine::transaction::IsolationLevel;

/// Build an in-memory DB with `n` `:Node {id, val}` nodes.
fn setup_nodes(n: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for i in 0..n {
        session
            .execute(&format!("CREATE (:Node {{id: {i}, val: {i}}})"))
            .expect("create");
    }
    db
}

/// Run one transaction at `level`: begin → execute `query` → commit.
fn run_txn(db: &GrafeoDB, level: IsolationLevel, query: &str) {
    let mut s = db.session();
    s.begin_transaction_with_isolation(level)
        .expect("begin");
    let r = s.execute(query).expect("execute");
    black_box(r);
    s.commit().expect("commit");
}

const SI: IsolationLevel = IsolationLevel::SnapshotIsolation;
const SER: IsolationLevel = IsolationLevel::Serializable;

/// (1) Read-recording overhead: scan N nodes (records N reads under Serializable).
fn bench_read_scan(c: &mut Criterion) {
    let mut g = c.benchmark_group("read_scan");
    g.sample_size(20);
    for n in [100usize, 1_000, 10_000] {
        let db = setup_nodes(n);
        let q = "MATCH (n:Node) RETURN n.id, n.val";
        g.bench_with_input(BenchmarkId::new("snapshot", n), &n, |b, _| {
            b.iter(|| run_txn(&db, SI, q));
        });
        g.bench_with_input(BenchmarkId::new("serializable", n), &n, |b, _| {
            b.iter(|| run_txn(&db, SER, q));
        });
    }
    g.finish();
}

/// (2) Write transaction: SET val on N nodes (read-set + write-set + commit check).
fn bench_write(c: &mut Criterion) {
    let mut g = c.benchmark_group("write_set");
    g.sample_size(20);
    for n in [100usize, 1_000] {
        let q = "MATCH (n:Node) SET n.val = n.val + 1";
        // Fresh DB per iteration-set so writes don't accumulate unbounded.
        g.bench_with_input(BenchmarkId::new("snapshot", n), &n, |b, &n| {
            b.iter_batched(
                || setup_nodes(n),
                |db| run_txn(&db, SI, q),
                criterion::BatchSize::LargeInput,
            );
        });
        g.bench_with_input(BenchmarkId::new("serializable", n), &n, |b, &n| {
            b.iter_batched(
                || setup_nodes(n),
                |db| run_txn(&db, SER, q),
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

/// (3) Empty transaction — begin + commit, no reads/writes. The pure fixed SSI
/// setup cost (read-set/registry alloc + empty commit-time dangerous-structure check).
/// Combined with `read_scan` (the per-read recording cost), this brackets any
/// workload: overhead ≈ fixed + (reads × per-read). A true point read (~1 recorded
/// read) therefore sits a hair above the empty-txn cost.
fn bench_empty(c: &mut Criterion) {
    let mut g = c.benchmark_group("empty_txn");
    g.sample_size(50);
    let db = setup_nodes(0);
    let run_empty = |level: IsolationLevel| {
        let mut s = db.session();
        s.begin_transaction_with_isolation(level).expect("begin");
        s.commit().expect("commit");
    };
    g.bench_function("snapshot", |b| b.iter(|| run_empty(SI)));
    g.bench_function("serializable", |b| b.iter(|| run_empty(SER)));
    g.finish();
}

criterion_group!(benches, bench_read_scan, bench_write, bench_empty);
criterion_main!(benches);
