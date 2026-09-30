//! Live `session.execute` parity: factorized hops vs flat expand.
//!
//! Pins the query path applications and benchmarks call, not kernel-only operators.
//! Not official LDBC SF1 and not a Ladybug head-to-head.

#![cfg(all(feature = "gql", feature = "lpg"))]

use grafeo_common::types::Value;
use grafeo_engine::database::QueryResult;
use grafeo_engine::{Config, GrafeoDB};

fn sorted_rows(result: &QueryResult) -> Vec<Vec<Value>> {
    let mut rows = result.rows().to_vec();
    rows.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    rows
}

fn assert_same(label: &str, fact: &QueryResult, flat: &QueryResult) {
    assert_eq!(
        fact.row_count(),
        flat.row_count(),
        "{label}: row count fact={} flat={}",
        fact.row_count(),
        flat.row_count()
    );
    assert_eq!(
        sorted_rows(fact),
        sorted_rows(flat),
        "{label}: rows differ\nfact={:?}\nflat={:?}",
        fact.rows(),
        flat.rows()
    );
}

/// Hub: 8 inbound → hub → 12 outbound → 1 leaf each. Fan-out > 1 so the
/// planner picks the factorized chain.
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

/// Circulant triangle graph (8 nodes, each points at the next 2).
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

fn exec(db: &GrafeoDB, q: &str) -> QueryResult {
    db.session()
        .execute(q)
        .unwrap_or_else(|e| panic!("{q}: {e}"))
}

#[test]
fn two_hop_id_matches_flat() {
    let (fact, flat) = pair();
    let q = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN id(c)";
    assert_same("2-hop id(c)", &exec(&fact, q), &exec(&flat, q));
}

#[test]
fn two_hop_entity_matches_flat() {
    let (fact, flat) = pair();
    let q = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN c";
    assert_same("2-hop RETURN c", &exec(&fact, q), &exec(&flat, q));
}

#[test]
fn two_hop_property_matches_flat() {
    let (fact, flat) = pair();
    let q = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN c.name";
    assert_same("2-hop c.name", &exec(&fact, q), &exec(&flat, q));
}

#[test]
fn two_hop_mid_filter_matches_flat() {
    let (fact, flat) = pair();
    let q =
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) WHERE c.age > 35 RETURN c.name";
    assert_same("2-hop WHERE c.age", &exec(&fact, q), &exec(&flat, q));
}

#[test]
fn three_hop_id_matches_flat() {
    let (fact, flat) = pair();
    let q = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(d:Person) RETURN id(d)";
    assert_same("3-hop id(d)", &exec(&fact, q), &exec(&flat, q));
}

#[test]
fn triangle_count_matches_flat() {
    let (fact, flat) = pair();
    let q = "MATCH (a:Tri)-[:R]->(b:Tri)-[:R]->(c:Tri)-[:R]->(a) RETURN COUNT(a)";
    assert_same("triangle COUNT", &exec(&fact, q), &exec(&flat, q));
}

#[test]
fn two_hop_mixed_return_matches_flat() {
    let (fact, flat) = pair();
    let q = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN id(a), b.name, c";
    assert_same("2-hop mixed RETURN", &exec(&fact, q), &exec(&flat, q));
}
