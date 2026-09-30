//! Content-addressed persistence + cross-time (cross-generation) dedup of the
//! temporal cold base, at POC scale. Complements `temporal_scrub` (which times
//! whole-state as-of reads): this file times the *storage* side — serializing a
//! temporal base content-addressed into a [`BlockPool`], reloading it, and the
//! whole-pool round-trip — and measures the headline cross-time dedup ratio.
//!
//! Run: `cargo bench -p grafeo-core --bench temporal_persist --features compact-store,lpg`
//!
//! Builds a synthetic 100K-node temporal cold base the same way `temporal_scrub`
//! does — overlay writes at several epochs + `merge_overlay_temporal` (no
//! external fixture, so OPSEC and the one-way rule hold) — then:
//!
//! * `serialize_content_addressed` — interns every node value block into a fresh
//!   pool (deduped by content id) and writes only the 32-byte content ids inline.
//! * `deserialize_content_addressed` — reconstructs the whole store from
//!   (section bytes + pool).
//! * `pool_to_bytes` / `pool_from_bytes` — whole-pool persistence round-trip.
//!
//! Cross-time dedup (the headline) is a *measured ratio*, not a timing: gen1 is
//! serialized into a pool P, then a small fraction of nodes are updated at a new
//! epoch and re-merged into gen2, which is serialized into the **same** pool P.
//! The share of blocks/bytes shared across the two generations (only changed
//! columns are new) is logged via `eprintln!` at 1% and 10% change rates.
// Benchmarks are not public API — suppress doc and pedantic lints.
#![allow(
    missing_docs,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use std::hint::black_box;

use bytes::Bytes;
use criterion::{Criterion, criterion_group, criterion_main};
use grafeo_common::types::{EpochId, NodeId, Value};
use grafeo_core::graph::compact::content_dedup::BlockPool;
use grafeo_core::graph::compact::from_graph_store_preserving_ids;
use grafeo_core::graph::compact::layered::LayeredStore;
use grafeo_core::graph::compact::section::{CompactStoreSection, deserialize_content_addressed};
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::graph::traits::{GraphStore, GraphStoreMut};
use std::sync::Arc;

/// Nodes in the synthetic temporal base. 100K (per the bench spec); each node
/// carries a 3-version `score` history at epochs 10/20/30 plus two static
/// properties (`bucket`, `tier`) so there are several columns to content-address.
const N: usize = 100_000;

/// Builds a temporal base of `n` nodes via the public LayeredStore + temporal
/// merge path: a 3-version `score` history at epochs 10/20/30, and two static
/// columns. After `merge_overlay_temporal` the base is the temporal fold and the
/// overlay is reset — ready for a follow-on generation.
fn build_temporal_base(n: usize) -> LayeredStore {
    let empty = from_graph_store_preserving_ids(&LpgStore::new().unwrap()).unwrap();
    let layered = LayeredStore::new(empty, n as u64 + 10, n as u64 + 10).unwrap();
    let overlay = layered.overlay_store();
    // Tiered node records share one fixed arena per birth epoch. Ten cohorts
    // keep the 100K fixture within those arenas and precede every score version.
    let cohort_size = n.div_ceil(10).max(1);
    for (birth_epoch, start) in (0..n).step_by(cohort_size).enumerate() {
        overlay.set_epoch(EpochId::new(birth_epoch as u64));
        for i in start..start.saturating_add(cohort_size).min(n) {
            let id = overlay.create_node(&["Entity"]);
            assert_ne!(id, NodeId::INVALID, "fixture node {i} must be allocated");
            overlay.set_node_property(id, "bucket", Value::Int64((i % 256) as i64));
            overlay.set_node_property(id, "tier", Value::Int64((i % 8) as i64));
            overlay.set_node_property_at_epoch(
                id,
                "score",
                Value::Int64(i as i64),
                EpochId::new(10),
            );
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
    }
    assert_eq!(
        overlay.node_count(),
        n,
        "fixture must be complete before folding"
    );
    overlay.set_epoch(EpochId::new(30));
    layered.merge_overlay_temporal().unwrap();
    layered
}

/// Advances `layered` to a new generation: bumps `score` for the first
/// `changed` nodes at a fresh epoch, then re-folds into the cold base. The
/// changed nodes' `score` column gets a new value block (new content id); the
/// unchanged static columns (`bucket`, `tier`) hash identically and are shared.
///
/// Writes go through the [`LayeredStore`] (`GraphStoreMut`) so each touched base
/// node is *promoted* into the overlay and its new version recorded at the
/// freshly-advanced epoch — only then does the re-merge fold the new history
/// into the base. (Writing the raw overlay `LpgStore` directly would not promote
/// the base node, so the new version would be dropped by the merge.)
fn advance_generation(layered: &LayeredStore, changed: usize, epoch: u64) {
    // Advance the overlay clock so promoted writes land at the new epoch.
    layered.overlay_store().set_epoch(EpochId::new(epoch));
    let ids = layered.base_store_arc().node_ids();
    for id in ids.iter().take(changed) {
        layered.set_node_property(
            *id,
            "score",
            Value::Int64(epoch as i64 * 1000 + id.as_u64() as i64),
        );
    }
    layered.merge_overlay_temporal().unwrap();
}

/// Serializes a store content-addressed into `pool`, returning the section bytes.
fn serialize_ca(
    store: Arc<grafeo_core::graph::compact::CompactStore>,
    pool: &mut BlockPool,
) -> Vec<u8> {
    let section = CompactStoreSection::new(store);
    section.serialize_content_addressed(pool).unwrap()
}

/// Measures and logs the cross-time dedup at one change rate: serialize gen1 into
/// a fresh pool, advance a fraction of nodes to gen2, serialize gen2 into the
/// SAME pool, and report the blocks/bytes shared across the two generations.
fn report_dedup(change_pct: usize) {
    let changed = (N * change_pct).div_ceil(100);
    let layered = build_temporal_base(N);

    let mut pool = BlockPool::new();
    let _gen1_bytes = serialize_ca(layered.base_store_arc(), &mut pool);
    let blocks_1 = pool.block_count();
    let bytes_1 = pool.total_bytes();

    advance_generation(&layered, changed, 40);
    let _gen2_bytes = serialize_ca(layered.base_store_arc(), &mut pool);
    let blocks_after = pool.block_count();
    let bytes_after = pool.total_bytes();

    let new_blocks = blocks_after - blocks_1;
    let new_bytes = bytes_after - bytes_1;

    // gen2's own footprint: serialize it alone into a fresh pool — that pool's
    // block_count/total_bytes are gen2's distinct blocks/bytes, and `shared =
    // total - new`. Uses the same content-addressing as the shared pool above, so
    // the two measurements are on one consistent ContentId scheme.
    let mut gen2_only = BlockPool::new();
    let _ = serialize_ca(layered.base_store_arc(), &mut gen2_only);
    let gen2_total_blocks = gen2_only.block_count();
    let gen2_total_bytes = gen2_only.total_bytes();
    let shared_blocks = gen2_total_blocks.saturating_sub(new_blocks);
    let shared_bytes = gen2_total_bytes.saturating_sub(new_bytes);
    let shared_block_pct = 100.0 * shared_blocks as f64 / gen2_total_blocks as f64;
    let shared_byte_pct = 100.0 * shared_bytes as f64 / gen2_total_bytes as f64;

    eprintln!(
        "[dedup @ {change_pct}% change of {N} nodes] gen1: {blocks_1} blocks / {bytes_1} B; \
         gen2 references {gen2_total_blocks} blocks / {gen2_total_bytes} B; \
         NEW in gen2: {new_blocks} blocks / {new_bytes} B; \
         SHARED across gens: {shared_blocks}/{gen2_total_blocks} blocks ({shared_block_pct:.1}%), \
         {shared_bytes}/{gen2_total_bytes} B ({shared_byte_pct:.1}%); \
         pool after both gens: {blocks_after} blocks / {bytes_after} B"
    );
}

fn bench_persist(c: &mut Criterion) {
    let layered = build_temporal_base(N);
    let base = layered.base_store_arc();
    assert_eq!(base.node_ids().len(), N);

    // One-shot size capture (logged once, outside the timing loop).
    let mut size_pool = BlockPool::new();
    let section_bytes = serialize_ca(base.clone(), &mut size_pool);
    eprintln!(
        "[sizes @ {N} nodes] section bytes: {} B; pool: {} blocks / {} B (to_bytes blob: {} B)",
        section_bytes.len(),
        size_pool.block_count(),
        size_pool.total_bytes(),
        size_pool.to_bytes().len(),
    );

    let mut group = c.benchmark_group("temporal_persist_100k");
    group.sample_size(20);

    // 1. serialize_content_addressed — fresh pool each iter (interning cost).
    group.bench_function("serialize_content_addressed", |b| {
        b.iter(|| {
            let mut pool = BlockPool::new();
            let bytes = serialize_ca(base.clone(), &mut pool);
            black_box((bytes.len(), pool.block_count()))
        });
    });

    // 2. deserialize_content_addressed — reconstruct the whole store.
    let de_section = Bytes::from(section_bytes.clone());
    group.bench_function("deserialize_content_addressed", |b| {
        b.iter(|| {
            let store = deserialize_content_addressed(black_box(&de_section), &size_pool).unwrap();
            black_box(store.node_count())
        });
    });

    // 4. BlockPool whole-pool round-trip.
    let pool_blob = size_pool.to_bytes();
    group.bench_function("pool_to_bytes", |b| {
        b.iter(|| black_box(size_pool.to_bytes().len()));
    });
    group.bench_function("pool_from_bytes", |b| {
        b.iter(|| {
            let p = BlockPool::from_bytes(black_box(&pool_blob)).unwrap();
            black_box(p.block_count())
        });
    });

    group.finish();

    // 3. Cross-time dedup ratios (measured, logged — not timed).
    report_dedup(1);
    report_dedup(10);
}

criterion_group!(benches, bench_persist);
criterion_main!(benches);
