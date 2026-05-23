//! View vs Owned query latency for the three Plan 2 codecs.
//!
//! Run: `cargo bench -p grafeo-core --bench codec_views`
//!
//! The output informs whether the WASM bindings should be migrated to
//! the View types. A meaningful latency penalty (e.g., > 1.5×) means the
//! current owned-codec WASM bindings are the right default; a comparable
//! or smaller latency means the views are a Pareto improvement.

use criterion::{Criterion, criterion_group, criterion_main};
use grafeo_common::types::NodeId;
use grafeo_core::codec::{FsstCodec, FsstView, WebGraphBuilder, WebGraphView};
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
    let owned = TwoStageVectorIndex::build(&vectors, DIM, 1);
    let blob = bytes::Bytes::from(owned.to_bytes());
    let view = RabitqView::open(blob).expect("open");
    let query = vectors[0].1.clone();

    let mut group = c.benchmark_group("rabitq_search_2k_128d");
    group.bench_function("owned", |b| {
        b.iter(|| black_box(owned.search(black_box(&query), 10, 8)));
    });
    group.bench_function("view", |b| {
        b.iter(|| black_box(view.search(black_box(&query), 10, 8)));
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
    let view = FsstView::open(blob).expect("open");

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
    let n: u64 = 1000;
    let mut b = WebGraphBuilder::new(n);
    for _ in 0..15_000 {
        let s = rng.u64() % n;
        let d = rng.u64() % n;
        b.add_edge(s, d).unwrap();
    }
    let owned = b.build();
    let blob = bytes::Bytes::from(owned.to_bytes());
    let view = WebGraphView::open(blob).expect("open");

    let mut group = c.benchmark_group("webgraph_successors_random_1k");
    group.bench_function("owned", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i = (i + 17) % 1000;
            black_box(owned.successors(black_box(i)).collect::<Vec<_>>())
        });
    });
    group.bench_function("view", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i = (i + 17) % 1000;
            black_box(view.successors(black_box(i)).collect::<Vec<_>>())
        });
    });
    group.finish();
}

criterion_group!(benches, bench_rabitq, bench_fsst, bench_webgraph);
criterion_main!(benches);
