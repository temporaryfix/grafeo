//! Track 3 pins: 1-hop / 2-hop factorized expand vs flatten, and triangle
//! leapfrog vs nested-loop expand.
//!
//! Run:
//! `CARGO_TARGET_DIR=/tmp/grafeo-gate cargo bench -p grafeo-core --features lpg,compact-store --bench factorized_expand_gate`
//!
//! Host pins (this machine, 2026-08-15). Does not fail the process on a
//! timing miss.
//!
//! | Path | Graph | ms/iter |
//! |------|-------|---------|
//! | 1-hop flatten Expand | 2k nodes × deg 8 | 0.706 |
//! | 1-hop factorized | 2k × 8 | 0.578 (0.82× flat) |
//! | 2-hop flatten Expand | 2k × 8 | 7.356 |
//! | 2-hop factorized | 2k × 8 | 5.833 (0.79× flat) |
//! | triangle nested-loop kernel | circulant 80 × 10 | 1.541 |
//! | triangle leapfrog kernel | circulant 80 × 10 | 0.185 (0.12× nested) |
//! | triangle nested Expand ops | circulant 80 × 10 | 6.064 |
//! | triangle LeapfrogExpand op | circulant 80 × 10 | 0.151 (0.02× nested) |

#![allow(
    missing_docs,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use grafeo_common::types::LogicalType;
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{
    ExpandOperator, FactorizedExpandChain, LeapfrogExpandOperator, Operator, ScanOperator,
    count_leapfrog_triangles, count_nested_loop_triangles,
};
use grafeo_core::graph::Direction;
use grafeo_core::graph::GraphStoreSearch;
use grafeo_core::graph::lpg::LpgStore;

fn pin_ms(label: &str, warmup: u32, iters: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let ms = start.elapsed().as_secs_f64() * 1_000.0 / f64::from(iters);
    eprintln!("[pin] {label}: {ms:.3} ms/iter (warmup={warmup} iters={iters})");
    ms
}

fn dest(src: u32, k: u32, n: u32) -> u32 {
    let mixed = src
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(k.wrapping_mul(0x85EB_CA6B));
    let mut dst = mixed % n;
    if dst == src {
        dst = (dst + 1) % n;
    }
    dst
}

fn build_fanout(n: u32, degree: u32) -> Arc<LpgStore> {
    let store = Arc::new(LpgStore::new().unwrap());
    let mut ids = Vec::with_capacity(n as usize);
    for _ in 0..n {
        ids.push(store.create_node(&["V"]));
    }
    for s in 0..n {
        for k in 0..degree {
            let d = dest(s, k, n) as usize;
            store.create_edge(ids[s as usize], ids[d], "R");
        }
    }
    store
}

/// Research `factorized_2hop_hub` (scaled: 200×150, not 500×300).
fn build_hub() -> Arc<LpgStore> {
    let store = Arc::new(LpgStore::new().unwrap());
    let hub = store.create_node(&["Hub"]);
    let mut ins = Vec::with_capacity(200);
    for _ in 0..200 {
        let n = store.create_node(&["In"]);
        store.create_edge(n, hub, "R");
        ins.push(n);
    }
    for _ in 0..150 {
        let n = store.create_node(&["Out"]);
        store.create_edge(hub, n, "R");
    }
    let _ = ins;
    store
}

fn build_triangles() -> Arc<LpgStore> {
    // Circulant: n nodes, each points at the next `degree` nodes.
    // Overlapping wedges so leapfrog intersection has real work (unlike
    // 200 disjoint 3-cycles, where both kernels are ~degree 1).
    let n = 80u32;
    let degree = 10u32;
    let store = Arc::new(LpgStore::new().unwrap());
    let mut ids = Vec::with_capacity(n as usize);
    for _ in 0..n {
        ids.push(store.create_node(&["V"]));
    }
    for s in 0..n {
        for k in 1..=degree {
            let d = ((s + k) % n) as usize;
            store.create_edge(ids[s as usize], ids[d], "R");
        }
    }
    store
}

struct SingleChunk(Option<DataChunk>);

impl Operator for SingleChunk {
    fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
        Ok(self.0.take())
    }
    fn reset(&mut self) {}
    fn name(&self) -> &'static str {
        "SingleChunk"
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

fn all_nodes_chunk(store: &dyn GraphStoreSearch) -> DataChunk {
    let ids = store.node_ids();
    let mut chunk = DataChunk::with_capacity(&[LogicalType::Node], ids.len());
    if let Some(col) = chunk.column_mut(0) {
        for id in &ids {
            col.push_node_id(*id);
        }
    }
    chunk.set_count(ids.len());
    chunk
}

fn count_op(mut op: Box<dyn Operator>) -> usize {
    let mut n = 0usize;
    while let Some(c) = op.next().unwrap() {
        n += c.row_count();
    }
    n
}

fn main() {
    let fan = build_fanout(2_000, 8);
    let compact_fan = grafeo_core::graph::compact::from_graph_store_preserving_ids(fan.as_ref())
        .expect("compact fanout");
    let store: Arc<dyn GraphStoreSearch> = fan;

    let hop1_flat = pin_ms("1-hop flatten Expand", 3, 12, || {
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "V"));
        let mut op = ExpandOperator::new(
            Arc::clone(&store),
            scan,
            0,
            Direction::Outgoing,
            vec!["R".to_string()],
        );
        let mut n = 0usize;
        while let Some(c) = op.next().unwrap() {
            n += c.row_count();
        }
        black_box(n);
    });

    let hop1_fact = pin_ms("1-hop factorized", 3, 12, || {
        let src = Box::new(SingleChunk(Some(all_nodes_chunk(store.as_ref()))));
        let n = FactorizedExpandChain::new(Arc::clone(&store), src)
            .expand(0, Direction::Outgoing, vec!["R".to_string()])
            .unwrap()
            .finish()
            .map_or(0, |c| c.logical_row_count());
        black_box(n);
    });

    let hop2_flat = pin_ms("2-hop flatten Expand", 2, 6, || {
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "V"));
        let hop1 = Box::new(ExpandOperator::new(
            Arc::clone(&store),
            scan,
            0,
            Direction::Outgoing,
            vec!["R".to_string()],
        ));
        let mut hop2 = ExpandOperator::new(
            Arc::clone(&store),
            hop1,
            2,
            Direction::Outgoing,
            vec!["R".to_string()],
        );
        let mut n = 0usize;
        while let Some(c) = hop2.next().unwrap() {
            n += c.row_count();
        }
        black_box(n);
    });

    let hop2_fact = pin_ms("2-hop factorized", 2, 6, || {
        let src = Box::new(SingleChunk(Some(all_nodes_chunk(store.as_ref()))));
        let n = FactorizedExpandChain::new(Arc::clone(&store), src)
            .expand(0, Direction::Outgoing, vec!["R".to_string()])
            .unwrap()
            .expand(1, Direction::Outgoing, vec!["R".to_string()])
            .unwrap()
            .finish()
            .map_or(0, |c| c.logical_row_count());
        black_box(n);
    });

    let tri = build_triangles();
    let compact_tri_store =
        grafeo_core::graph::compact::from_graph_store_preserving_ids(tri.as_ref())
            .expect("compact triangles");
    let tri_store: Arc<dyn GraphStoreSearch> = tri;
    let nested = count_nested_loop_triangles(tri_store.as_ref());
    let leap = count_leapfrog_triangles(tri_store.as_ref());
    assert_eq!(nested, leap, "triangle kernels must agree");

    let tri_nested = pin_ms("triangle nested-loop kernel", 4, 16, || {
        black_box(count_nested_loop_triangles(tri_store.as_ref()));
    });
    let tri_leap = pin_ms("triangle leapfrog kernel", 4, 16, || {
        black_box(count_leapfrog_triangles(tri_store.as_ref()));
    });

    let tri_nested_op = pin_ms("triangle nested Expand ops", 2, 8, || {
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&tri_store), "V"));
        let hop1 = Box::new(ExpandOperator::new(
            Arc::clone(&tri_store),
            scan,
            0,
            Direction::Outgoing,
            vec!["R".to_string()],
        ));
        let hop2 = Box::new(ExpandOperator::new(
            Arc::clone(&tri_store),
            hop1,
            2,
            Direction::Outgoing,
            vec!["R".to_string()],
        ));
        let mut hop3 = ExpandOperator::new(
            Arc::clone(&tri_store),
            hop2,
            4,
            Direction::Outgoing,
            vec!["R".to_string()],
        );
        let mut n = 0usize;
        while let Some(c) = hop3.next().unwrap() {
            let a = c.column(0).unwrap();
            let a2 = c.column(6).unwrap();
            for i in 0..c.row_count() {
                if a.get_node_id(i) == a2.get_node_id(i) {
                    n += 1;
                }
            }
        }
        black_box(n);
    });

    let tri_leap_op = pin_ms("triangle LeapfrogExpand op", 2, 8, || {
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&tri_store), "V"));
        let hop1 = Box::new(ExpandOperator::new(
            Arc::clone(&tri_store),
            scan,
            0,
            Direction::Outgoing,
            vec!["R".to_string()],
        ));
        let leap = Box::new(LeapfrogExpandOperator::directed_triangle_close(
            Arc::clone(&tri_store),
            hop1,
            0,
            2,
            vec!["R".to_string()],
        ));
        black_box(count_op(leap));
    });

    eprintln!(
        "[gate] 1-hop fact/flat={:.2}  2-hop fact/flat={:.2}  tri leap/nested={:.2}  tri op leap/nested={:.2}",
        hop1_fact / hop1_flat,
        hop2_fact / hop2_flat,
        tri_leap / tri_nested,
        tri_leap_op / tri_nested_op
    );

    // CompactStore path — same graphs after ID-preserving compact.
    let compact: Arc<dyn GraphStoreSearch> = Arc::new(compact_fan);
    let c_hop2 = pin_ms("2-hop factorized CompactStore", 2, 6, || {
        let src = Box::new(SingleChunk(Some(all_nodes_chunk(compact.as_ref()))));
        let n = FactorizedExpandChain::new(Arc::clone(&compact), src)
            .expand(0, Direction::Outgoing, vec!["R".to_string()])
            .unwrap()
            .expand(1, Direction::Outgoing, vec!["R".to_string()])
            .unwrap()
            .finish()
            .map_or(0, |c| c.logical_row_count());
        black_box(n);
    });
    let compact_tri: Arc<dyn GraphStoreSearch> = Arc::new(compact_tri_store);
    let c_leap = pin_ms("triangle leapfrog CompactStore kernel", 4, 16, || {
        black_box(count_leapfrog_triangles(compact_tri.as_ref()));
    });
    eprintln!(
        "[gate] compact 2-hop fact={:.3} ms  compact tri leap={:.3} ms",
        c_hop2, c_leap
    );

    // kuzu_parity factorized_2hop_hub: physical values vs flattened cells.
    let hub_store: Arc<dyn GraphStoreSearch> = build_hub();
    let src = Box::new(SingleChunk(Some({
        let mut chunk = DataChunk::with_capacity(&[grafeo_common::types::LogicalType::Node], 200);
        for id in hub_store.nodes_by_label("In") {
            if let Some(col) = chunk.column_mut(0) {
                col.push_node_id(id);
            }
        }
        chunk.set_count(200);
        chunk
    })));
    let fact = FactorizedExpandChain::new(Arc::clone(&hub_store), src)
        .expand(0, Direction::Outgoing, vec!["R".to_string()])
        .unwrap()
        .expand(1, Direction::Outgoing, vec!["R".to_string()])
        .unwrap()
        .finish()
        .expect("hub 2-hop");
    let phys = fact.physical_size();
    let logical = fact.logical_row_count();
    let flat_cells = logical.saturating_mul(5);
    eprintln!(
        "[pin] hub 2-hop (200×150): physical={phys} logical_rows={logical} flat_cells={flat_cells} ratio={:.3}",
        phys as f64 / flat_cells.max(1) as f64
    );
    eprintln!(
        "[gate] hub factorized physical/flat_cells={:.3} (research wants << 1)",
        phys as f64 / flat_cells.max(1) as f64
    );
}
