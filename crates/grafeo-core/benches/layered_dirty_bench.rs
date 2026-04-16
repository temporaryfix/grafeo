//! Benchmarks for LayeredStore dirty tracking.
//!
//! Measures the cost of dirty-check lookups during reads at various
//! dirty ratios (0%, 1%, 10%, 50%). Uses generic entity names.
#![allow(clippy::cast_possible_truncation)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_core::graph::compact::from_graph_store_preserving_ids;
use grafeo_core::graph::compact::layered::LayeredStore;
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::graph::traits::{GraphStore, GraphStoreMut};

/// Builds a LayeredStore with `n` base nodes and promotes `dirty_pct`% of them.
fn build_layered(n: usize, dirty_pct: usize) -> (LayeredStore, Vec<NodeId>) {
    let store = LpgStore::new().unwrap();

    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let id = store.create_node(&["Item"]);
        store.set_node_property(id, "score", Value::Int64(i as i64));
        store.set_node_property(id, "name", Value::from(format!("item_{i}")));
        ids.push(id);
    }

    let compact = from_graph_store_preserving_ids(&store).unwrap();
    let max_nid = ids.iter().map(|id| id.as_u64()).max().unwrap_or(0);
    let layered = LayeredStore::new(compact, max_nid, 0).unwrap();

    // Promote a fraction of base nodes to the overlay (makes them dirty).
    let dirty_count = n * dirty_pct / 100;
    for &id in ids.iter().take(dirty_count) {
        layered.set_node_property(id, "score", Value::Int64(999));
    }

    (layered, ids)
}

fn bench_get_node(c: &mut Criterion) {
    let mut group = c.benchmark_group("layered_get_node");

    for &(n, dirty_pct) in &[(10_000, 0), (10_000, 1), (10_000, 10), (10_000, 50)] {
        let (layered, ids) = build_layered(n, dirty_pct);
        group.bench_with_input(
            BenchmarkId::new(format!("{n}_nodes"), format!("{dirty_pct}%_dirty")),
            &(),
            |b, _| {
                b.iter(|| {
                    for &id in &ids {
                        black_box(layered.get_node(id));
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_get_node_property(c: &mut Criterion) {
    let mut group = c.benchmark_group("layered_get_node_property");
    let key = PropertyKey::new("score");

    for &(n, dirty_pct) in &[(10_000, 0), (10_000, 1), (10_000, 10), (10_000, 50)] {
        let (layered, ids) = build_layered(n, dirty_pct);
        group.bench_with_input(
            BenchmarkId::new(format!("{n}_nodes"), format!("{dirty_pct}%_dirty")),
            &(),
            |b, _| {
                b.iter(|| {
                    for &id in &ids {
                        black_box(layered.get_node_property(id, &key));
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_find_nodes_by_property(c: &mut Criterion) {
    let mut group = c.benchmark_group("layered_find_nodes_by_property");

    for &(n, dirty_pct) in &[(10_000, 0), (10_000, 1), (10_000, 10), (10_000, 50)] {
        let (layered, _ids) = build_layered(n, dirty_pct);
        let target = Value::Int64(42);
        group.bench_with_input(
            BenchmarkId::new(format!("{n}_nodes"), format!("{dirty_pct}%_dirty")),
            &(),
            |b, _| {
                b.iter(|| {
                    black_box(layered.find_nodes_by_property("score", &target));
                });
            },
        );
    }
    group.finish();
}

fn bench_nodes_by_label(c: &mut Criterion) {
    let mut group = c.benchmark_group("layered_nodes_by_label");

    for &(n, dirty_pct) in &[(10_000, 0), (10_000, 1), (10_000, 10), (10_000, 50)] {
        let (layered, _ids) = build_layered(n, dirty_pct);
        group.bench_with_input(
            BenchmarkId::new(format!("{n}_nodes"), format!("{dirty_pct}%_dirty")),
            &(),
            |b, _| {
                b.iter(|| {
                    black_box(layered.nodes_by_label("Item"));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_get_node,
    bench_get_node_property,
    bench_find_nodes_by_property,
    bench_nodes_by_label,
);
criterion_main!(benches);
