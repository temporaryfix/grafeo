//! LDBC-shaped kernel pins (not official LDBC SNB SF1, not Ladybug).
//!
//! Synthetic social-network graph: `Person`—`KNOWS`→`Person` and
//! `Person`—`LIKES`→`Post`. Queries follow LDBC IC / graph-bench shapes
//! (1-hop, 2-hop FoF, directed triangles, common neighbors). Scale is
//! in-tree (2k persons), not the official generator.
//!
//! Run:
//! `CARGO_TARGET_DIR=/tmp/grafeo-gate cargo bench -p grafeo-core --features lpg,compact-store --bench ldbc_shaped_gate`
//!
//! Host pins are printed as `[pin]`. This harness does not fail the process
//! on a timing miss.
//!
//! Host pins (this machine, 2026-08-15). Not official LDBC SF1.
//!
//! | Path | Graph | ms/iter |
//! |------|-------|---------|
//! | KNOWS 1-hop flatten | 2k Person × deg 8 | 0.664 |
//! | KNOWS 1-hop factorized | 2k × 8 | 0.605 (0.91× flat) |
//! | KNOWS 2-hop flatten | 2k × 8 | 6.546 |
//! | KNOWS 2-hop factorized | 2k × 8 | 5.649 (0.86× flat) |
//! | LIKES 1-hop flatten | 2k × 1 post | 0.514 |
//! | KNOWS triangle nested kernel | circulant 80 × 10 | 1.121 |
//! | KNOWS triangle leapfrog kernel | circulant 80 × 10 | 0.136 (0.12× nested) |
//! | KNOWS common-neighbors sample | 64 pairs | 0.016 |
//! | CompactStore KNOWS 1-hop | 2k × 8 | 0.151 |

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
    ExpandOperator, FactorizedExpandChain, Operator, ScanOperator, count_leapfrog_triangles,
    count_nested_loop_triangles,
};
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::graph::{Direction, GraphStore, GraphStoreSearch};

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

/// LDBC-like KNOWS: each person points at `degree` others (directed).
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

struct SocialGraph {
    store: Arc<LpgStore>,
    persons: Vec<grafeo_common::types::NodeId>,
}

fn build_social(n_person: u32, knows_degree: u32, posts_per: u32) -> SocialGraph {
    let store = Arc::new(LpgStore::new().unwrap());
    let mut persons = Vec::with_capacity(n_person as usize);
    for _ in 0..n_person {
        persons.push(store.create_node(&["Person"]));
    }
    for s in 0..n_person {
        for k in 0..knows_degree {
            let d = dest(s, k, n_person) as usize;
            store.create_edge(persons[s as usize], persons[d], "KNOWS");
        }
        for p in 0..posts_per {
            let post = store.create_node(&["Post"]);
            store.create_edge(persons[s as usize], post, "LIKES");
            let _ = p;
        }
    }
    SocialGraph { store, persons }
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

fn person_chunk(ids: &[grafeo_common::types::NodeId]) -> DataChunk {
    let mut chunk = DataChunk::with_capacity(&[LogicalType::Node], ids.len());
    if let Some(col) = chunk.column_mut(0) {
        for id in ids {
            col.push_node_id(*id);
        }
    }
    chunk.set_count(ids.len());
    chunk
}

fn main() {
    let social = build_social(2_000, 8, 1);
    let compact =
        grafeo_core::graph::compact::from_graph_store_preserving_ids(social.store.as_ref())
            .expect("compact social");
    let store: Arc<dyn GraphStoreSearch> = social.store;

    let hop1 = pin_ms("ldbc/knows/1-hop flatten", 3, 12, || {
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "Person"));
        let mut op = ExpandOperator::new(
            Arc::clone(&store),
            scan,
            0,
            Direction::Outgoing,
            vec!["KNOWS".to_string()],
        );
        let mut n = 0usize;
        while let Some(c) = op.next().unwrap() {
            n += c.row_count();
        }
        black_box(n);
    });

    let hop1_fact = pin_ms("ldbc/knows/1-hop factorized", 3, 12, || {
        let src = Box::new(SingleChunk(Some(person_chunk(&social.persons))));
        let n = FactorizedExpandChain::new(Arc::clone(&store), src)
            .expand(0, Direction::Outgoing, vec!["KNOWS".to_string()])
            .unwrap()
            .finish()
            .map_or(0, |c| c.logical_row_count());
        black_box(n);
    });

    let hop2 = pin_ms("ldbc/knows/2-hop flatten", 2, 6, || {
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "Person"));
        let hop1 = Box::new(ExpandOperator::new(
            Arc::clone(&store),
            scan,
            0,
            Direction::Outgoing,
            vec!["KNOWS".to_string()],
        ));
        let mut hop2 = ExpandOperator::new(
            Arc::clone(&store),
            hop1,
            2,
            Direction::Outgoing,
            vec!["KNOWS".to_string()],
        );
        let mut n = 0usize;
        while let Some(c) = hop2.next().unwrap() {
            n += c.row_count();
        }
        black_box(n);
    });

    let hop2_fact = pin_ms("ldbc/knows/2-hop factorized", 2, 6, || {
        let src = Box::new(SingleChunk(Some(person_chunk(&social.persons))));
        let n = FactorizedExpandChain::new(Arc::clone(&store), src)
            .expand(0, Direction::Outgoing, vec!["KNOWS".to_string()])
            .unwrap()
            .expand(1, Direction::Outgoing, vec!["KNOWS".to_string()])
            .unwrap()
            .finish()
            .map_or(0, |c| c.logical_row_count());
        black_box(n);
    });

    let likes = pin_ms("ldbc/likes/1-hop flatten", 3, 12, || {
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "Person"));
        let mut op = ExpandOperator::new(
            Arc::clone(&store),
            scan,
            0,
            Direction::Outgoing,
            vec!["LIKES".to_string()],
        );
        let mut n = 0usize;
        while let Some(c) = op.next().unwrap() {
            n += c.row_count();
        }
        black_box(n);
    });

    // Circulant Person-KNOWS for triangle / common-neighbor shapes.
    let n = 80u32;
    let degree = 10u32;
    let tri_store = Arc::new(LpgStore::new().unwrap());
    let mut ids = Vec::with_capacity(n as usize);
    for _ in 0..n {
        ids.push(tri_store.create_node(&["Person"]));
    }
    for s in 0..n {
        for k in 1..=degree {
            let d = ((s + k) % n) as usize;
            tri_store.create_edge(ids[s as usize], ids[d], "KNOWS");
        }
    }
    let tri: Arc<dyn GraphStoreSearch> = tri_store;
    let nested = count_nested_loop_triangles(tri.as_ref());
    let leap = count_leapfrog_triangles(tri.as_ref());
    assert_eq!(nested, leap, "KNOWS triangles must agree");

    let tri_nested = pin_ms("ldbc/knows/triangle nested kernel", 4, 16, || {
        black_box(count_nested_loop_triangles(tri.as_ref()));
    });
    let tri_leap = pin_ms("ldbc/knows/triangle leapfrog kernel", 4, 16, || {
        black_box(count_leapfrog_triangles(tri.as_ref()));
    });

    let common = pin_ms("ldbc/knows/common-neighbors sample", 4, 16, || {
        let mut acc = 0u64;
        for i in 0..64 {
            let a = ids[i];
            let b = ids[(i + 3) % ids.len()];
            let na = tri.neighbors(a, Direction::Outgoing);
            let mut nb = tri.neighbors(b, Direction::Outgoing);
            nb.sort_unstable();
            for n in na {
                if nb.binary_search(&n).is_ok() {
                    acc += 1;
                }
            }
        }
        black_box(acc);
    });

    let compact_hop = pin_ms("ldbc/knows/compact 1-hop", 3, 12, || {
        let mut acc = 0u64;
        for id in &social.persons {
            acc = acc.wrapping_add(compact.neighbors(*id, Direction::Outgoing).len() as u64);
        }
        black_box(acc);
    });

    eprintln!(
        "[summary] 1-hop flat {hop1:.3} / fact {hop1_fact:.3}; 2-hop flat {hop2:.3} / fact {hop2_fact:.3}; likes {likes:.3}; tri nested {tri_nested:.3} / leap {tri_leap:.3}; cn {common:.3}; compact {compact_hop:.3}"
    );
    eprintln!(
        "[note] not official LDBC SF1; not a Ladybug/Kuzu head-to-head. Same kernel as native + WASM."
    );
}
