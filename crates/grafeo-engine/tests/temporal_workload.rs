//! Temporal workload as-of queries on Grafeo.
//!
//! 1. Sources that observed handle H at epoch T (incoming OBSERVED_BY)
//! 2. Correlation cluster of H at T (2-hop CORRELATED_WITH)
//! 3. Provenance 3-hop DERIVED_FROM COUNT and id list
//!
//! After `compact()`, mid-history as-of uses packed / CSR fill. Seeded hops
//! must stay under 0.2 ms.

#![cfg(all(feature = "compact-store", feature = "gql", feature = "lpg"))]

use std::time::Instant;

use grafeo_common::types::{EpochId, NodeId, Value};
use grafeo_core::graph::Direction;
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

fn bump(db: &GrafeoDB) {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.commit().unwrap();
}

fn pin_ms(label: &str, warmup: u32, iters: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let ms = t0.elapsed().as_secs_f64() * 1_000.0 / f64::from(iters);
    eprintln!("[pin] {label}: {ms:.3} ms/iter (warmup={warmup} iters={iters})");
    ms
}

struct TemporalGraphFixture {
    db: GrafeoDB,
    entities: Vec<NodeId>,
    epoch_open: EpochId,
    epoch_mid: EpochId,
}

fn load_temporal_graph(n_ent: u32, n_src: u32) -> TemporalGraphFixture {
    let mut db = GrafeoDB::new_in_memory();
    let mut entities = Vec::with_capacity(n_ent as usize);
    for i in 0..n_ent {
        entities.push(db.create_node_with_props(&["Entity"], [("id", Value::from(i.to_string()))]));
    }
    let mut sources = Vec::with_capacity(n_src as usize);
    for i in 0..n_src {
        sources
            .push(db.create_node_with_props(&["Source"], [("id", Value::from(format!("s{i}")))]));
    }
    for (i, &ent) in entities.iter().enumerate() {
        let s0 = sources[i % sources.len()];
        let s1 = sources[(i + 1) % sources.len()];
        db.create_edge(s0, ent, "OBSERVED_BY");
        db.create_edge(s1, ent, "OBSERVED_BY");
        let c0 = entities[dest(u32::try_from(i).unwrap(), 0, n_ent) as usize];
        let c1 = entities[dest(u32::try_from(i).unwrap(), 1, n_ent) as usize];
        if c0 != ent {
            db.create_edge(ent, c0, "CORRELATED_WITH");
        }
        if c1 != ent && c1 != c0 {
            db.create_edge(ent, c1, "CORRELATED_WITH");
        }
        let d0 = entities[dest(u32::try_from(i).unwrap(), 2, n_ent) as usize];
        if d0 != ent {
            db.create_edge(ent, d0, "DERIVED_FROM");
        }
    }
    let epoch_open = db.current_epoch();
    bump(&db);
    // Close the first observation of even entities so as-of mid ≠ current.
    for (i, &ent) in entities.iter().enumerate() {
        if i % 2 != 0 {
            continue;
        }
        let src = sources[i % sources.len()];
        let edges = db.graph_store().edges_from(src, Direction::Outgoing);
        if let Some((_, eid)) = edges.iter().find(|(dst, _)| *dst == ent) {
            db.delete_edge(*eid);
        }
    }
    let epoch_mid = db.current_epoch();
    bump(&db);
    db.compact().expect("compact Temporal workload graph");
    TemporalGraphFixture {
        db,
        entities,
        epoch_open,
        epoch_mid,
    }
}

fn observers_of(db: &GrafeoDB, h: NodeId, epoch: EpochId) -> Vec<NodeId> {
    let types = vec!["OBSERVED_BY".to_string()];
    let mut buf = Vec::new();
    db.fill_neighbors_of_types_at_epoch(h, Direction::Incoming, epoch, &types, &mut buf);
    buf.sort_unstable();
    buf
}

#[test]
fn observers_of_h_at_t_is_incoming_asof() {
    let g = load_temporal_graph(64, 8);
    let h = g.entities[0];
    let at_open = observers_of(&g.db, h, g.epoch_open);
    let at_mid = observers_of(&g.db, h, g.epoch_mid);
    let pending = observers_of(&g.db, h, EpochId::PENDING);
    assert!(at_open.len() >= 2, "even entity starts with two observers");
    assert!(
        at_mid.len() < at_open.len() || at_mid != at_open,
        "closing one OBSERVED_BY must change incoming as-of"
    );
    assert_eq!(
        at_mid, pending,
        "after compact, mid-close == current incoming"
    );
}

#[test]
fn execute_at_epoch_second_call_reuses_physical_plan() {
    let g = load_temporal_graph(64, 8);
    let session = g.db.session();
    let q = "MATCH (h:Entity {id: '1'})-[:CORRELATED_WITH]->(x) RETURN count(*) AS n";
    let a = session.execute_at_epoch(q, g.epoch_open).unwrap();
    let warm_stats = g.db.query_cache().stats();
    let b = session.execute_at_epoch(q, g.epoch_open).unwrap();
    let hit_stats = g.db.query_cache().stats();
    assert_eq!(
        (hit_stats.optimized_hits, hit_stats.optimized_misses),
        (warm_stats.optimized_hits, warm_stats.optimized_misses),
        "same-epoch physical hits must bypass logical/physical replanning"
    );
    assert_eq!(a.rows(), b.rows());
    assert!(a.rows()[0][0].as_int64().unwrap() > 0);
    // Different viewing epochs must not share a physical tree.
    let q_obs = "MATCH (h:Entity {id: '0'})<-[:OBSERVED_BY]-(s) RETURN count(*) AS n";
    let open = session
        .execute_at_epoch(q_obs, g.epoch_open)
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    let mid = session.execute_at_epoch(q_obs, g.epoch_mid).unwrap().rows()[0][0]
        .as_int64()
        .unwrap();
    assert!(open >= 2, "even entity starts with two observers");
    assert!(
        mid < open,
        "closing one OBSERVED_BY must drop the as-of count"
    );
}

#[test]
fn temporal_workload_seeded_hops_under_point_two_ms() {
    const N: u32 = 2000;
    const S: u32 = 100;
    let g = load_temporal_graph(N, S);
    let h = g.entities[1]; // odd: both observations still live
    let session = g.db.session();

    let obs_ty = vec!["OBSERVED_BY".to_string()];
    let corr_ty = vec!["CORRELATED_WITH".to_string()];
    let mut buf = Vec::new();
    let obs = pin_ms("keep/observers incoming fill 2k", 8, 32, || {
        g.db.fill_neighbors_of_types_at_epoch(
            h,
            Direction::Incoming,
            g.epoch_open,
            &obs_ty,
            &mut buf,
        );
        assert!(!buf.is_empty(), "odd entity keeps observers at open epoch");
    });
    assert!(
        obs < 0.2,
        "observers of H at T must stay under 0.2 ms, got {obs:.3}"
    );

    let cluster_fill = pin_ms("keep/cluster 2-hop fill 2k", 8, 32, || {
        g.db.fill_neighbors_of_types_at_epoch(
            h,
            Direction::Outgoing,
            g.epoch_open,
            &corr_ty,
            &mut buf,
        );
        let mid = buf.clone();
        let mut acc = 0usize;
        for m in &mid {
            g.db.fill_neighbors_of_types_at_epoch(
                *m,
                Direction::Outgoing,
                g.epoch_open,
                &corr_ty,
                &mut buf,
            );
            acc += buf.len();
        }
        assert!(acc > 0, "correlation 2-hop fill must produce dests");
    });
    assert!(
        cluster_fill < 0.2,
        "correlation cluster fill at T must stay under 0.2 ms, got {cluster_fill:.3}"
    );

    let cluster_q = "MATCH (h:Entity {id: '1'})-[:CORRELATED_WITH]->(x)-[:CORRELATED_WITH]->(y) RETURN count(*) AS n";
    let n_open = session
        .execute_at_epoch(cluster_q, g.epoch_open)
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert!(n_open > 0, "correlation 2-hop must produce paths");
    let cluster = pin_ms("keep/cluster 2-hop COUNT 2k", 4, 16, || {
        let _ = session.execute_at_epoch(cluster_q, g.epoch_open).unwrap();
    });
    // Physical plan is cached after the first execute (warmup).
    let cluster_budget = if cfg!(debug_assertions) { 1.0 } else { 0.2 };
    assert!(
        cluster < cluster_budget,
        "correlation cluster COUNT must stay under {cluster_budget} ms, got {cluster:.3}"
    );

    let hop3_count = "MATCH (h:Entity {id: '1'})-[:DERIVED_FROM]->()-[:DERIVED_FROM]->()-[:DERIVED_FROM]->(d) RETURN count(*) AS n";
    let hop3_ids = "MATCH (h:Entity {id: '1'})-[:DERIVED_FROM]->()-[:DERIVED_FROM]->()-[:DERIVED_FROM]->(d) RETURN id(d)";
    let c3 = session
        .execute_at_epoch(hop3_count, g.epoch_open)
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    let ids = session.execute_at_epoch(hop3_ids, g.epoch_open).unwrap();
    assert_eq!(
        i64::try_from(ids.row_count()).unwrap(),
        c3,
        "3-hop id list matches COUNT"
    );
    assert!(c3 > 0, "provenance 3-hop must produce paths");

    let count_ms = pin_ms("keep/provenance 3-hop COUNT 2k", 4, 16, || {
        let _ = session.execute_at_epoch(hop3_count, g.epoch_open).unwrap();
    });
    let ids_ms = pin_ms("keep/provenance 3-hop id(d) 2k", 4, 16, || {
        let _ = session.execute_at_epoch(hop3_ids, g.epoch_open).unwrap();
    });
    eprintln!(
        "[pin] keep provenance 3-hop 2k: COUNT {count_ms:.3} ms, id list {ids_ms:.3} ms (n={c3})"
    );
    let hop3_budget = if cfg!(debug_assertions) { 1.0 } else { 0.2 };
    assert!(
        count_ms < hop3_budget,
        "seeded 3-hop COUNT should stay under {hop3_budget} ms, got {count_ms:.3}"
    );
}
