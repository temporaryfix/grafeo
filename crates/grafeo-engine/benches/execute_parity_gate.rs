//! Live `session.execute` pins vs the flat planner (not official LDBC SF1).
//!
//! Same query strings as `tests/execute_parity.rs`. Host pins are printed
//! as `[pin]`. This harness does not fail the process on a timing miss.
//!
//! Run:
//! `CARGO_TARGET_DIR=/tmp/grafeo-gate cargo bench -p grafeo-engine --bench execute_parity_gate`
//!
//! Host pins (this machine, 2026-08-15). Not official LDBC SF1.
//! Parse+plan+execute on a small hub (8 in × 12 out × 1 leaf). Kernel-only
//! 2k×8 pins live in `grafeo-core` `ldbc_shaped_gate`.
//!
//! | Path | Graph | ms/iter |
//! |------|-------|---------|
//! | 2-hop id fact | hub 8×12 | 0.075 |
//! | 2-hop id flat | hub 8×12 | 0.074 |
//! | 2-hop entity fact | hub 8×12 | 0.150 |
//! | 2-hop property fact | hub 8×12 | 0.091 |
//! | 2-hop WHERE fact | hub 8×12 | 0.070 |
//! | 3-hop id fact | hub 8×12×1 | 0.236 |
//! | triangle COUNT fact | circulant 8×2 | 0.053 |
//!
//! ## Fair execute card vs LadybugDB 0.15.3 (this machine, 2026-08-16)
//!
//! `graph-bench fair --nodes 2000 --degree 8` (Grafeo 0.5.43 release
//! bindings). Same edge set on both; count must match the Python
//! reference or the row is INVALID (no speedup). Not official LDBC.
//!
//! | Bench | Expect | g_ms | l_ms | g/l | Status |
//! |-------|-------:|-----:|-----:|----:|--------|
//! | count_persons | 2000 | 0.110 | 0.725 | 0.15× | OK |
//! | hop1_from_seed | 8 | 0.153 | 1.970 | 0.078× | OK |
//! | hop2_from_seed | 64 | 0.085 | 8.273 | 0.010× | OK |
//! | hop2_filter | 42 | 0.091 | 8.881 | 0.010× | OK |
//! | hop2_all | 128000 | 0.213 | 13.207 | **0.016×** | OK |
//! | triangle_count | 495 | 2.630 | 22.176 | 0.12× | OK |
//! | pagerank_n | 2000 | 0.962 | 31.735 | 0.030× | OK |
//! | wcc_sizes | 1 part | 1.742 | 11.616 | 0.15× | OK |
//!
//! `hop2_all` was 13.2 ms (1.76× Ladybug) when the last hop walked 16k
//! adjacency lists. COUNT of a 2-hop over every node is now
//! Σ_b indeg(b)·outdeg(b) (0.213 ms). Seeded 2-hop stays factorized
//! `execute`. Triangle COUNT(*) is `|out(b) ∩ in(a)|` from the start
//! scan (1.52 ms LpgStore / 0.75 ms after `compact()` CSR at 2k×8),
//! not 16k Leapfrog rows. Not official LDBC.
//!
//! Older graph-bench `run -c traversal` numbers are Python `get_neighbors`
//! loops, not this card.

#![allow(
    missing_docs,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::hint::black_box;
use std::time::Instant;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

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

fn load_hub(db: &GrafeoDB) {
    let hub = db.create_node_with_props(
        &["Person"],
        [
            ("name", Value::from("hub")),
            ("age", Value::Int64(40)),
            ("id", Value::Int64(0)),
        ],
    );
    for i in 0..8 {
        let src = db.create_node_with_props(
            &["Person"],
            [
                ("name", Value::from(format!("in{i}"))),
                ("age", Value::Int64(20 + i)),
                ("id", Value::Int64(100 + i)),
            ],
        );
        db.create_edge(src, hub, "KNOWS");
    }
    for j in 0..12 {
        let mid = db.create_node_with_props(
            &["Person"],
            [
                ("name", Value::from(format!("out{j}"))),
                ("age", Value::Int64(30 + j)),
                ("id", Value::Int64(200 + j)),
            ],
        );
        db.create_edge(hub, mid, "KNOWS");
        let leaf = db.create_node_with_props(
            &["Person"],
            [
                ("name", Value::from(format!("leaf{j}"))),
                ("age", Value::Int64(10)),
                ("id", Value::Int64(300 + j)),
            ],
        );
        db.create_edge(mid, leaf, "KNOWS");
    }
}

fn load_triangles(db: &GrafeoDB) {
    let mut ids = Vec::with_capacity(8);
    for i in 0..8 {
        ids.push(db.create_node_with_props(
            &["Tri"],
            [
                ("name", Value::from(format!("t{i}"))),
                ("id", Value::Int64(i)),
            ],
        ));
    }
    for s in 0..8 {
        for k in 1..=2 {
            let d = (s + k) % 8;
            db.create_edge(ids[s], ids[d], "R");
        }
    }
}

fn pair() -> (GrafeoDB, GrafeoDB) {
    let fact = GrafeoDB::new_in_memory();
    let flat = GrafeoDB::with_config(Config::default().without_factorized_execution()).unwrap();
    load_hub(&fact);
    load_hub(&flat);
    load_triangles(&fact);
    load_triangles(&flat);
    (fact, flat)
}

fn exec(db: &GrafeoDB, q: &str) -> usize {
    db.session().execute(q).unwrap().row_count()
}

fn main() {
    let (fact, flat) = pair();

    const Q2_ID: &str = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN id(c)";
    const Q2_ENT: &str = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN c";
    const Q2_PROP: &str =
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN c.name";
    const Q2_WHERE: &str =
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) WHERE c.age > 35 RETURN c.name";
    const Q3_ID: &str = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(d:Person) RETURN id(d)";
    const Q_TRI: &str = "MATCH (a:Tri)-[:R]->(b:Tri)-[:R]->(c:Tri)-[:R]->(a) RETURN COUNT(a)";

    let n_id = exec(&fact, Q2_ID);
    let n_id_flat = exec(&flat, Q2_ID);
    assert_eq!(n_id, n_id_flat, "2-hop id row counts must agree");
    assert_eq!(exec(&fact, Q2_ENT), exec(&flat, Q2_ENT));
    assert_eq!(exec(&fact, Q2_PROP), exec(&flat, Q2_PROP));
    assert_eq!(exec(&fact, Q2_WHERE), exec(&flat, Q2_WHERE));
    assert_eq!(exec(&fact, Q3_ID), exec(&flat, Q3_ID));
    assert_eq!(exec(&fact, Q_TRI), exec(&flat, Q_TRI));

    let hop2_id_fact = pin_ms("execute/2-hop id fact", 4, 20, || {
        black_box(exec(&fact, Q2_ID));
    });
    let hop2_id_flat = pin_ms("execute/2-hop id flat", 4, 20, || {
        black_box(exec(&flat, Q2_ID));
    });
    let hop2_ent = pin_ms("execute/2-hop entity fact", 3, 12, || {
        black_box(exec(&fact, Q2_ENT));
    });
    let hop2_prop = pin_ms("execute/2-hop property fact", 4, 20, || {
        black_box(exec(&fact, Q2_PROP));
    });
    let hop2_where = pin_ms("execute/2-hop WHERE fact", 4, 20, || {
        black_box(exec(&fact, Q2_WHERE));
    });
    let hop3 = pin_ms("execute/3-hop id fact", 3, 12, || {
        black_box(exec(&fact, Q3_ID));
    });
    let tri = pin_ms("execute/triangle COUNT fact", 4, 16, || {
        black_box(exec(&fact, Q_TRI));
    });

    eprintln!(
        "[summary] 2-hop id fact {hop2_id_fact:.3} / flat {hop2_id_flat:.3}; entity {hop2_ent:.3}; prop {hop2_prop:.3}; where {hop2_where:.3}; 3-hop {hop3:.3}; tri {tri:.3}; rows(id)={n_id}"
    );
    eprintln!(
        "[note] not official LDBC SF1; session.execute path (parse+plan+run), same kernel as native + WASM."
    );
}
