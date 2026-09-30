//! View vs Owned query latency for the three Plan 2 codecs.
//!
//! Run: `cargo bench -p grafeo-core --bench codec_views`
//!
//! The output informs whether the WASM bindings should be migrated to
//! the View types. A meaningful latency penalty (e.g., > 1.5×) means the
//! current owned-codec WASM bindings are the right default; a comparable
//! or smaller latency means the views are a Pareto improvement.
// Benchmarks are not public API — suppress doc and name-length lints.
#![allow(missing_docs, clippy::many_single_char_names)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use grafeo_common::types::NodeId;
use grafeo_core::codec::{
    BitVector, FsstCodec, FsstView, WebGraphBuilder, WebGraphCodec, WebGraphView,
};
use grafeo_core::index::vector::{RabitqView, TwoStageVectorIndex};
use std::hint::black_box;

struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f32(&mut self) -> f32 {
        (self.u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn gaussian(&mut self) -> f32 {
        let u1 = self.f32().max(f32::MIN_POSITIVE);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

fn bench_rabitq(c: &mut Criterion) {
    const DIM: usize = 128;
    const COUNT: usize = 2_000;
    let mut rng = Rng(42);
    let vectors: Vec<(NodeId, Vec<f32>)> = (0..COUNT)
        .map(|i| {
            let v: Vec<f32> = (0..DIM).map(|_| rng.gaussian()).collect();
            (NodeId::new(i as u64 + 1), v)
        })
        .collect();
    let owned = TwoStageVectorIndex::build(&vectors, DIM, 1).expect("valid index");
    let blob = bytes::Bytes::from(owned.to_bytes().expect("serialize index"));
    let view = RabitqView::open(blob.clone()).expect("open");
    let query = vectors[0].1.clone();

    let mut open_group = c.benchmark_group("rabitq_open_2k_128d");
    open_group.throughput(Throughput::Elements(COUNT as u64));
    open_group.bench_function("owned", |b| {
        b.iter(|| TwoStageVectorIndex::from_bytes(black_box(&blob)).expect("open owned"));
    });
    open_group.bench_function("view", |b| {
        b.iter(|| RabitqView::open(black_box(blob.clone())).expect("open view"));
    });
    open_group.finish();

    let mut group = c.benchmark_group("rabitq_search_2k_128d");
    group.bench_function("owned", |b| {
        b.iter(|| {
            black_box(
                owned
                    .search(black_box(&query), 10, 8)
                    .expect("matching query dimension"),
            )
        });
    });
    group.bench_function("view", |b| {
        b.iter(|| {
            black_box(
                view.search(black_box(&query), 10, 8)
                    .expect("matching query dimension"),
            )
        });
    });
    group.finish();
}

fn bench_fsst(c: &mut Criterion) {
    let mut rng = Rng(99);
    let strings: Vec<Vec<u8>> = (0..1000)
        .map(|_| {
            let len = (rng.u64() % 32) as usize + 4;
            (0..len).map(|_| (rng.u64() & 0x7F) as u8 + 32).collect()
        })
        .collect();
    let refs: Vec<&[u8]> = strings.iter().map(Vec::as_slice).collect();
    let owned = FsstCodec::build(&refs);
    let blob = bytes::Bytes::from(owned.to_bytes());
    let view = FsstView::open(blob.clone()).expect("open");

    let mut open_group = c.benchmark_group("fsst_open_1k");
    open_group.throughput(Throughput::Elements(strings.len() as u64));
    open_group.bench_function("owned", |b| {
        b.iter(|| FsstCodec::from_bytes(black_box(&blob)).expect("open owned"));
    });
    open_group.bench_function("view", |b| {
        b.iter(|| FsstView::open(black_box(blob.clone())).expect("open view"));
    });
    open_group.finish();

    let mut group = c.benchmark_group("fsst_get_random_1k");
    group.bench_function("owned", |b| {
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 31) % 1000;
            black_box(owned.get(black_box(i)).unwrap().unwrap())
        });
    });
    group.bench_function("view", |b| {
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 31) % 1000;
            black_box(view.get(black_box(i)).unwrap().unwrap())
        });
    });
    group.finish();
}

fn bench_webgraph(c: &mut Criterion) {
    let mut rng = Rng(7);
    let num_nodes: u64 = 1000;
    let mut builder = WebGraphBuilder::new(num_nodes);
    let mut input_edges = Vec::with_capacity(15_000);
    for _ in 0..15_000 {
        let src = rng.u64() % num_nodes;
        let dst = rng.u64() % num_nodes;
        input_edges.push((src, dst));
        builder.add_edge(src, dst).unwrap();
    }
    let owned = builder.build().expect("build webgraph");
    let owned_blob = owned.to_bytes().expect("serialize webgraph");
    let blob = bytes::Bytes::copy_from_slice(&owned_blob);
    let view = WebGraphView::open(blob).expect("open");

    let mut build_group = c.benchmark_group("webgraph_build_1k_nodes_15k_edges");
    build_group.throughput(Throughput::Elements(15_000));
    build_group.bench_function("owned", |b| {
        b.iter(|| {
            let mut candidate = WebGraphBuilder::new(num_nodes);
            for &(source, destination) in black_box(&input_edges) {
                candidate
                    .add_edge(source, destination)
                    .expect("valid benchmark edge");
            }
            black_box(candidate.build().expect("build webgraph"))
        });
    });
    build_group.finish();

    let mut open_group = c.benchmark_group("webgraph_open_1k_nodes_15k_edges");
    open_group.throughput(Throughput::Elements(15_000));
    open_group.bench_function("owned", |b| {
        b.iter(|| WebGraphCodec::from_bytes(black_box(&owned_blob)).expect("open owned"));
    });
    let view_blob = bytes::Bytes::copy_from_slice(&owned_blob);
    open_group.bench_function("view", |b| {
        b.iter(|| WebGraphView::open(black_box(view_blob.clone())).expect("open view"));
    });
    open_group.finish();

    let mut validate_group = c.benchmark_group("webgraph_validate_1k_nodes_15k_edges");
    validate_group.throughput(Throughput::Elements(15_000));
    validate_group.bench_function("owned", |b| {
        b.iter(|| {
            let reopened = WebGraphCodec::from_bytes(black_box(&owned_blob)).expect("open owned");
            reopened.validate().expect("validate owned");
        });
    });
    let validation_blob = bytes::Bytes::copy_from_slice(&owned_blob);
    validate_group.bench_function("view", |b| {
        b.iter(|| {
            let reopened =
                WebGraphView::open(black_box(validation_blob.clone())).expect("open view");
            reopened.validate().expect("validate view");
        });
    });
    validate_group.finish();

    let mut group = c.benchmark_group("webgraph_successors_random_1k");
    group.bench_function("owned", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i = (i + 17) % 1000;
            black_box(
                owned
                    .successors(black_box(i))
                    .expect("owned successors")
                    .collect::<Vec<_>>(),
            )
        });
    });
    group.bench_function("view", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i = (i + 17) % 1000;
            black_box(
                view.successors(black_box(i))
                    .expect("view successors")
                    .collect::<Vec<_>>(),
            )
        });
    });
    group.finish();
}

fn bench_bitvector(c: &mut Criterion) {
    const BIT_COUNT: usize = 1 << 20;
    let bits: Vec<bool> = (0..BIT_COUNT).map(|index| index % 7 == 0).collect();
    let bitvector = BitVector::from_bools(&bits);

    let mut group = c.benchmark_group("bitvector_scan_1m_bits");
    group.throughput(Throughput::Elements(BIT_COUNT as u64));
    group.bench_function("checked_get_baseline", |b| {
        b.iter(|| {
            black_box(
                (0..BIT_COUNT)
                    .filter(|&index| bitvector.get(black_box(index)) == Some(true))
                    .count(),
            )
        });
    });
    group.bench_function("word_iterator", |b| {
        b.iter(|| black_box(bitvector.iter().filter(|set| *set).count()));
    });
    group.bench_function("count_ones", |b| {
        b.iter(|| black_box(bitvector.count_ones()));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_rabitq,
    bench_fsst,
    bench_webgraph,
    bench_bitvector
);
criterion_main!(benches);
