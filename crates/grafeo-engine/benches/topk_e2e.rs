//! End-to-end §2.9 sanity benchmark for the heap top-K rewrite.
//!
//! Runs the same query shape via two paths:
//!   - `LIMIT 5` literal — exercises the rewrite (TopK fires).
//!   - `LIMIT $k` parameter — falls through to the unfused Limit ← Sort path
//!     since the rewrite refuses non-literal counts.
//!
//! Acceptance: literal path is no slower than parameter path. If literal is
//! slower, the rewrite is causing a regression and PR2 is held until it's
//! narrowed.
#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn seed(n: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for i in 0..n {
        let r = ((i as u64).wrapping_mul(2_654_435_761) % 1_000_000) as i64;
        let x = (i % 100) as i64;
        session
            .execute(&format!("INSERT (:Item {{id: {i}, r: {r}, x: {x}}})"))
            .unwrap();
    }
    db
}

fn bench_topk_e2e(c: &mut Criterion) {
    let db = seed(100_000);

    let mut group = c.benchmark_group("top_k_filtered_e2e");
    group.sample_size(20);

    group.bench_function("literal", |b| {
        b.iter(|| {
            let session = db.session();
            let result = session
                .execute("MATCH (n:Item) WHERE n.x > 50 RETURN n.r ORDER BY n.r DESC LIMIT 5")
                .unwrap();
            black_box(result.row_count());
        });
    });

    group.bench_function("parameter", |b| {
        b.iter(|| {
            let session = db.session();
            // Parameter LIMIT prevents the rewrite (CountExpr::Parameter falls
            // through). Same query shape otherwise; performance delta is
            // attributable to the rewrite vs. the unfused path.
            let mut params = std::collections::HashMap::new();
            params.insert("k".to_string(), Value::Int64(5));
            let result = session
                .execute_with_params(
                    "MATCH (n:Item) WHERE n.x > 50 RETURN n.r ORDER BY n.r DESC LIMIT $k",
                    params,
                )
                .unwrap();
            black_box(result.row_count());
        });
    });

    group.finish();
}

criterion_group!(benches, bench_topk_e2e);
criterion_main!(benches);
