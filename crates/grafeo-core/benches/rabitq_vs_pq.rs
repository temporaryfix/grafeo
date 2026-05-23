//! RaBitQ two-stage vs Product Quantization: query latency and recall@10.
//!
//! Run: `cargo bench -p grafeo-core --bench rabitq_vs_pq`
//!
//! The recall numbers are printed once before the timed loops; use them
//! together with the criterion latency report to decide whether RaBitQ
//! replaces PQ or PQ remains the fallback.
#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};
use grafeo_common::types::NodeId;
use grafeo_core::index::vector::ProductQuantizer;
use grafeo_core::index::vector::TwoStageVectorIndex;
use std::hint::black_box;

const DIM: usize = 256;
const COUNT: usize = 5_000;
const K: usize = 10;

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

fn dataset() -> Vec<(NodeId, Vec<f32>)> {
    let mut rng = Rng(99);
    let centres: Vec<Vec<f32>> = (0..40)
        .map(|_| (0..DIM).map(|_| rng.gaussian() * 6.0).collect())
        .collect();
    let mut out = Vec::with_capacity(COUNT);
    for i in 0..COUNT {
        let c = &centres[i % centres.len()];
        let v = c.iter().map(|&x| x + rng.gaussian() * 0.5).collect();
        out.push((NodeId::new(i as u64 + 1), v));
    }
    out
}

fn brute_force_top_k(vectors: &[(NodeId, Vec<f32>)], query: &[f32]) -> Vec<NodeId> {
    let mut s: Vec<(NodeId, f32)> = vectors
        .iter()
        .map(|(id, v)| {
            (*id, v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum())
        })
        .collect();
    s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    s.into_iter().take(K).map(|(id, _)| id).collect()
}

fn recall(truth: &[NodeId], got: &[NodeId]) -> f32 {
    got.iter().filter(|id| truth.contains(id)).count() as f32 / truth.len() as f32
}

fn bench(c: &mut Criterion) {
    let vectors = dataset();
    let query = vectors[7].1.clone();
    let truth = brute_force_top_k(&vectors, &query);

    // RaBitQ two-stage.
    let rabitq = TwoStageVectorIndex::build(&vectors, DIM, 1);
    let rabitq_hits: Vec<NodeId> =
        rabitq.search(&query, K, 16).into_iter().map(|(id, _)| id).collect();

    // PQ baseline: 32 subvectors, asymmetric distance scan.
    let refs: Vec<&[f32]> = vectors.iter().map(|(_, v)| v.as_slice()).collect();
    let pq = ProductQuantizer::train(&refs, 32, 256, 10);
    let pq_codes: Vec<(NodeId, Vec<u8>)> =
        vectors.iter().map(|(id, v)| (*id, pq.quantize(v))).collect();
    let pq_search = |q: &[f32]| -> Vec<NodeId> {
        let table = pq.build_distance_table(q);
        let mut s: Vec<(NodeId, f32)> = pq_codes
            .iter()
            .map(|(id, codes)| (*id, pq.distance_with_table(&table, codes)))
            .collect();
        s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        s.into_iter().take(K).map(|(id, _)| id).collect()
    };
    let pq_hits = pq_search(&query);

    eprintln!("── recall@{K} (oracle = exact Euclidean) ──");
    eprintln!("  RaBitQ two-stage : {:.3}", recall(&truth, &rabitq_hits));
    eprintln!("  Product Quant.   : {:.3}", recall(&truth, &pq_hits));
    eprintln!("  RaBitQ code size : 32 B/vec (+8 B factors)  PQ: 32 B/vec");

    let mut group = c.benchmark_group("vector_search_5k_256d");
    group.bench_function("rabitq_two_stage", |b| {
        b.iter(|| black_box(rabitq.search(black_box(&query), K, 16)));
    });
    group.bench_function("product_quantization", |b| {
        b.iter(|| black_box(pq_search(black_box(&query))));
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
