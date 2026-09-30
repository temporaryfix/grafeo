//! Fair-card triangle COUNT: `|out(b) ∩ in(a)|` over every Person, not
//! 80k first-hop rows + LeapfrogExpand materialize + SimpleAggregate.

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

fn expected_triangles(n: u32, adj: &[Vec<u32>]) -> i64 {
    let mut n_tri = 0i64;
    for a in 0..n {
        for &b in &adj[a as usize] {
            for &c in &adj[b as usize] {
                if adj[c as usize].binary_search(&a).is_ok() {
                    n_tri += 1;
                }
            }
        }
    }
    n_tri
}

#[test]
fn triangle_count_two_thousand_by_eight() {
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
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); N as usize];
    let mut seen = std::collections::HashSet::new();
    for src in 0..N {
        for k in 0..DEG {
            let dst = dest(src, k, N);
            if seen.insert((src, dst)) {
                db.create_edge(ids[src as usize], ids[dst as usize], "KNOWS");
                adj[src as usize].push(dst);
            }
        }
    }
    for nbrs in &mut adj {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    let expect = expected_triangles(N, &adj);

    let session = db.session();
    let comma =
        "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c), (c)-[:KNOWS]->(a) RETURN count(*) AS n";
    let linear = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(a) RETURN count(*) AS n";
    let cn = session.execute(comma).unwrap().rows()[0][0]
        .as_int64()
        .unwrap();
    let ln = session.execute(linear).unwrap().rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(cn, expect, "comma-join triangle COUNT");
    assert_eq!(ln, expect, "linear triangle COUNT");

    for _ in 0..2 {
        let _ = session.execute(comma).unwrap();
    }
    let t0 = Instant::now();
    const ITERS: u32 = 8;
    for _ in 0..ITERS {
        let _ = session.execute(comma).unwrap();
    }
    let ms = t0.elapsed().as_secs_f64() * 1_000.0 / f64::from(ITERS);
    eprintln!(
        "[debug] triangle COUNT 2k×8 execute: {ms:.3} ms/iter (uncalibrated; full sf1 benchmark is the performance gate)"
    );
}

#[cfg(feature = "compact-store")]
#[test]
fn compact_exposes_csr_triangle_kernel() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["V"]);
    let b = db.create_node(&["V"]);
    let c = db.create_node(&["V"]);
    db.create_edge(a, b, "R");
    db.create_edge(b, c, "R");
    db.create_edge(c, a, "R");
    db.compact().expect("compact");
    let layered = db.layered_store().expect("layered after compact");
    let graph = layered.graph_store();
    let n = graph.try_count_directed_triangles(&graph.node_ids(), None);
    assert_eq!(n, Some(3), "cold CSR kernel must fire after compact()");
}

#[cfg(feature = "compact-store")]
#[test]
fn triangle_count_compact_csr() {
    const N: u32 = 2000;
    const DEG: u32 = 8;
    let mut db = GrafeoDB::new_in_memory();
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
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); N as usize];
    let mut seen = std::collections::HashSet::new();
    for src in 0..N {
        for k in 0..DEG {
            let dst = dest(src, k, N);
            if seen.insert((src, dst)) {
                db.create_edge(ids[src as usize], ids[dst as usize], "KNOWS");
                adj[src as usize].push(dst);
            }
        }
    }
    for nbrs in &mut adj {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    let expect = expected_triangles(N, &adj);
    db.compact().expect("compact to CSR base");

    let session = db.session();
    let comma =
        "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c), (c)-[:KNOWS]->(a) RETURN count(*) AS n";
    let before_work = grafeo_engine::database::testing::root_lpg_store(&db).work_snapshot();
    let n = session.execute(comma).unwrap().rows()[0][0]
        .as_int64()
        .unwrap();
    let work = grafeo_engine::database::testing::root_lpg_store(&db)
        .work_snapshot()
        .since(before_work);
    assert_eq!(n, expect, "CSR triangle COUNT after compact()");
    assert!(
        work.label_scan_ids + work.full_scan_ids <= u64::from(N),
        "triangle count scan work must remain bounded by one node population: {work:?}"
    );

    for _ in 0..2 {
        let _ = session.execute(comma).unwrap();
    }
    let t0 = Instant::now();
    const ITERS: u32 = 8;
    for _ in 0..ITERS {
        let _ = session.execute(comma).unwrap();
    }
    let ms = t0.elapsed().as_secs_f64() * 1_000.0 / f64::from(ITERS);
    eprintln!(
        "[debug] triangle COUNT compact CSR 2k×8: {ms:.3} ms/iter (uncalibrated; full sf1 benchmark is the performance gate)"
    );
}
