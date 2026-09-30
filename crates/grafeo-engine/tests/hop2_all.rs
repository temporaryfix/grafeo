//! Fair-card `hop2_all`: COUNT of `(a)-[:KNOWS]->(b)-[:KNOWS]->(c)` over every
//! Person. Uses Σ_b indeg(b)·outdeg(b) when the scan is the whole graph.

#![cfg(all(feature = "gql", feature = "lpg"))]

use std::time::Instant;

use grafeo_engine::GrafeoDB;

fn dest(src: u32, k: u32, n: u32) -> u32 {
    let mixed = src
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(k.wrapping_mul(0x85EB_CA6B));
    let mut out = mixed % n;
    if out == src {
        out = (out + 1) % n;
    }
    out
}

#[test]
fn hop2_all_count_two_thousand_by_eight() {
    const N: u32 = 2000;
    const DEG: u32 = 8;
    let db = GrafeoDB::new_in_memory();
    let mut ids = Vec::with_capacity(N as usize);
    for i in 0..N {
        ids.push(db.create_node_with_props(
            &["Person"],
            [
                ("id", grafeo_common::types::Value::from(i.to_string())),
                (
                    "age",
                    grafeo_common::types::Value::Int64(i64::from(20 + i % 50)),
                ),
            ],
        ));
    }
    let mut seen = std::collections::HashSet::new();
    for src in 0..N {
        for k in 0..DEG {
            let dst = dest(src, k, N);
            if seen.insert((src, dst)) {
                db.create_edge(ids[src as usize], ids[dst as usize], "KNOWS");
            }
        }
    }

    let q = "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN count(c) AS n";
    let session = db.session();
    let n = session.execute(q).unwrap().rows()[0][0].as_int64().unwrap();
    assert_eq!(n, 128_000);

    // Anonymous last hop: dest nodes only (no last-hop EdgeIds).
    let q_ret = "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN c";
    // Materializing 128k complete node maps exceeds the default 64 MiB
    // retained-output cap. Keep the complete path witness with an explicit cap.
    let ret_n = session
        .execute_with_options(
            q_ret,
            Default::default(),
            grafeo_engine::ExecutionOptions {
                result_limits: Some(grafeo_engine::ResultLimits {
                    max_rows: 128_000,
                    max_bytes: 512 * 1024 * 1024,
                }),
                ..Default::default()
            },
        )
        .unwrap()
        .row_count();
    assert_eq!(
        ret_n, 128_000,
        "RETURN c must emit the same 128k paths as count(c)"
    );

    let ids = session
        .execute("MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN id(c)")
        .unwrap();
    assert_eq!(ids.row_count(), 128_000);
    assert!(
        ids.is_int64_columnar(),
        "RETURN id(c) must stay one Int64 column, not 128k row vecs"
    );

    for _ in 0..2 {
        let _ = session.execute(q).unwrap();
    }
    let t0 = Instant::now();
    const ITERS: u32 = 8;
    for _ in 0..ITERS {
        let _ = session.execute(q).unwrap();
    }
    let ms = t0.elapsed().as_secs_f64() * 1_000.0 / f64::from(ITERS);
    eprintln!("[pin] hop2_all 2k×8 execute: {ms:.3} ms/iter (Ladybug card 7.503)");
    assert!(
        ms < 7.503,
        "hop2_all must beat Ladybug 7.503 ms, got {ms:.3}"
    );
}
