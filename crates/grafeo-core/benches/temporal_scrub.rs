//! Whole-state as-of scrub throughput at POC scale (SP3 slice 7).
//!
//! Run: `cargo bench -p grafeo-core --bench temporal_scrub --features compact-store,lpg`
//!
//! Builds a synthetic 200K-node temporal cold base (via the public
//! LayeredStore + temporal-merge path — no external fixture, so OPSEC and the
//! one-way rule hold) and times two whole-state as-of reads at a past epoch:
//!
//! * `columnar_scrub` — one `get_node_property_at_epoch` per node, no Node
//!   materialization. This is the storage's scrub cost: zone-pruned validity
//!   lookups over the temporal columns. The 60fps budget is 16.67 ms / frame.
//! * `nodes_at_epoch` — the full `Vec<Node>` scrub API (materializes a Node
//!   per node), for comparison; the per-node allocation is the consumer's cost.
// Benchmarks are not public API — suppress doc and pedantic lints.
#![allow(
    missing_docs,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use grafeo_common::types::{EpochId, PropertyKey, Value};
use grafeo_core::graph::compact::from_graph_store_preserving_ids;
use grafeo_core::graph::compact::layered::LayeredStore;
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::graph::traits::GraphStore;

/// Builds a temporal base of `n` nodes, each with a 3-version `score` history at
/// epochs 10/20/30, by writing into the overlay and compacting it.
fn build_temporal_base(n: usize) -> LayeredStore {
    let empty = from_graph_store_preserving_ids(&LpgStore::new().unwrap()).unwrap();
    let layered = LayeredStore::new(empty, n as u64 + 10, n as u64 + 10).unwrap();
    let overlay = layered.overlay_store();
    for i in 0..n {
        let id = overlay.create_node(&["Entity"]);
        overlay.set_node_property_at_epoch(id, "score", Value::Int64(i as i64), EpochId::new(10));
        overlay.set_node_property_at_epoch(
            id,
            "score",
            Value::Int64((i as i64) * 2),
            EpochId::new(20),
        );
        overlay.set_node_property_at_epoch(
            id,
            "score",
            Value::Int64((i as i64) * 3),
            EpochId::new(30),
        );
    }
    overlay.set_epoch(EpochId::new(30));
    layered.merge_overlay_temporal().unwrap();
    layered
}

fn bench_scrub(c: &mut Criterion) {
    const N: usize = 200_000;
    let layered = build_temporal_base(N);
    let base = layered.base_store_arc();
    let key = PropertyKey::new("score");
    let ids = base.node_ids();
    assert_eq!(ids.len(), N);
    let epoch = EpochId::new(15); // a past epoch -> the [10,20) version

    let mut group = c.benchmark_group("temporal_scrub_200k");
    group.sample_size(20);

    group.bench_function("columnar_scrub", |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for id in &ids {
                if let Some(Value::Int64(v)) =
                    base.get_node_property_at_epoch(*id, &key, black_box(epoch))
                {
                    sum = sum.wrapping_add(v);
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("scrub_at_epoch_api", |b| {
        b.iter(|| {
            let frames = base.scrub_at_epoch(black_box(epoch));
            let mut count = 0usize;
            for frame in &frames {
                count += frame.node_ids.len();
            }
            black_box(count)
        });
    });

    group.bench_function("nodes_at_epoch", |b| {
        b.iter(|| black_box(layered.nodes_at_epoch(black_box(epoch)).len()));
    });

    group.finish();
}

criterion_group!(benches, bench_scrub);
criterion_main!(benches);
