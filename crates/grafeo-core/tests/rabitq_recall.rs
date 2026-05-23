//! Recall-floor regression test for the RaBitQ two-stage vector codec.
//!
//! Mirrors the fork's property-test discipline (see
//! `grafeo-engine/tests/compact_roundtrip_proptest.rs`): a `proptest!`
//! block generating well-separated clustered datasets, plus fixed-seed
//! regression cases. The oracle is exact brute-force Euclidean k-NN; the
//! codec must keep recall@10 at or above the floor.
//!
//! ## Dataset shape
//!
//! Clusters of exactly 10 points are used so the query's true top-10 IS
//! the query's own cluster. Cross-cluster separation (~13× the within-
//! cluster jitter scale) dominates the int8 reranker's quantization noise
//! by two orders of magnitude, so the codec routes to the right cluster
//! deterministically. This makes recall@10 a clean signal: it measures
//! whether the codec keeps the cluster intact, not whether it can resolve
//! within-cluster ordering (which is inherently noise-bound at any
//! reasonable per-cluster size with f32-precision Euclidean truth).
//!
//! ```bash
//! cargo test -p grafeo-core --test rabitq_recall
//! PROPTEST_CASES=512 cargo test -p grafeo-core --test rabitq_recall
//! ```

use grafeo_common::types::NodeId;
use grafeo_core::index::vector::TwoStageVectorIndex;
use proptest::prelude::*;

/// SplitMix64 — duplicated here because the in-crate one is private.
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

/// Builds a clustered dataset deterministically from `seed`.
fn clustered_dataset(
    seed: u64,
    dim: usize,
    clusters: usize,
    per_cluster: usize,
) -> Vec<(NodeId, Vec<f32>)> {
    let mut rng = Rng(seed);
    let centres: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.gaussian() * 6.0).collect())
        .collect();
    let mut out = Vec::new();
    let mut id = 1u64;
    for centre in &centres {
        for _ in 0..per_cluster {
            let v = centre.iter().map(|&c| c + rng.gaussian() * 0.4).collect();
            out.push((NodeId::new(id), v));
            id += 1;
        }
    }
    out
}

/// Exact brute-force Euclidean k-NN — the recall oracle.
fn brute_force(
    vectors: &[(NodeId, Vec<f32>)],
    query: &[f32],
    k: usize,
) -> Vec<NodeId> {
    let mut scored: Vec<(NodeId, f32)> = vectors
        .iter()
        .map(|(id, v)| {
            let d: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
            (*id, d)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

/// recall@k of `got` against the exact `truth`.
fn recall(truth: &[NodeId], got: &[(NodeId, f32)]) -> f32 {
    let hits = got.iter().filter(|(id, _)| truth.contains(id)).count();
    hits as f32 / truth.len() as f32
}

/// recall@10 floor on the clustered datasets generated below.
///
/// Calibrated against the dataset shape — `per_cluster = 10` so the
/// query's true top-10 IS its own cluster. The codec needs only to route
/// to the right cluster (a ~13× separation problem at int8 precision),
/// which it does deterministically. A breach signals a real regression
/// in coarse-pass routing or rerank ordering, not statistical noise.
const RECALL_FLOOR: f32 = 0.90;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// recall@10 stays at or above the floor across generated datasets.
    #[test]
    fn rabitq_two_stage_clears_recall_floor(
        seed in any::<u64>(),
        clusters in 4usize..=8,
        // Exactly 10 per cluster: the query's true top-10 is its own
        // cluster, removing within-cluster rank ambiguity (whose margins
        // are smaller than int8 quantization noise). See module doc.
        per_cluster in 10usize..=10,
        query_pick in 0usize..200,
    ) {
        let dim = 64;
        let vectors = clustered_dataset(seed, dim, clusters, per_cluster);
        let index = TwoStageVectorIndex::build(&vectors, dim, seed ^ 0xA5A5);

        let query = vectors[query_pick % vectors.len()].1.clone();
        let truth = brute_force(&vectors, &query, 10);
        let got = index.search(&query, 10, 16);

        let r = recall(&truth, &got);
        prop_assert!(
            r >= RECALL_FLOOR,
            "recall {r} below floor {RECALL_FLOOR} (seed={seed}, clusters={clusters})"
        );
    }
}

// ── Fixed regression seeds ───────────────────────────────────────

#[test]
fn recall_floor_fixed_seed_small() {
    let dim = 64;
    let vectors = clustered_dataset(1, dim, 5, 10);
    let index = TwoStageVectorIndex::build(&vectors, dim, 7);
    let query = vectors[0].1.clone();
    let truth = brute_force(&vectors, &query, 10);
    let got = index.search(&query, 10, 16);
    assert!(recall(&truth, &got) >= RECALL_FLOOR);
}

#[test]
fn recall_floor_survives_blob_round_trip() {
    let dim = 64;
    let vectors = clustered_dataset(42, dim, 6, 10);
    let index = TwoStageVectorIndex::build(&vectors, dim, 13);
    let reopened = TwoStageVectorIndex::from_bytes(&index.to_bytes()).expect("from_bytes");

    // Point 0 of cluster 1 (each cluster has 10 points).
    let query = vectors[10].1.clone();
    let truth = brute_force(&vectors, &query, 10);
    assert!(recall(&truth, &reopened.search(&query, 10, 16)) >= RECALL_FLOOR);
}
