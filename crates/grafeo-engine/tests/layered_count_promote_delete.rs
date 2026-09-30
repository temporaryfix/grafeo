//! Regression: `node_count`/`edge_count` must not double-subtract a base entity
//! that was promoted (written) and then deleted after `compact()`.
//!
//! `LayeredStore::node_count` = `base - deleted - promoted + overlay`. A
//! promoted-then-deleted node sits in BOTH `deleted_from_base_nodes` and
//! `dirty_node_ids`, so it was subtracted twice → undercount.

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn gql_count(db: &GrafeoDB, q: &str) -> i64 {
    match &db.session().execute(q).unwrap().rows()[0][0] {
        Value::Int64(n) => *n,
        other => panic!("expected Int64, got {other:?}"),
    }
}

#[test]
fn node_count_after_promote_then_delete() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    db.create_node(&["A"]);
    db.create_node(&["A"]);
    db.compact().expect("compact");

    db.set_node_property(a, "v", Value::Int64(1))
        .expect("set node property"); // promote `a` into the overlay
    db.delete_node(a); // then delete it

    assert_eq!(db.node_count(), 2, "3 created - 1 deleted = 2 live nodes");
    assert_eq!(gql_count(&db, "MATCH (n) RETURN count(n)"), 2, "GQL oracle");
}

#[test]
fn edge_count_after_promote_then_delete() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    let e = db.create_edge(a, b, "T");
    db.create_edge(a, b, "T");
    db.compact().expect("compact");

    db.set_edge_property(e, "w", Value::Int64(1))
        .expect("set edge property"); // promote `e`
    db.delete_edge(e); // then delete it

    assert_eq!(db.edge_count(), 1, "2 created - 1 deleted = 1 live edge");
    assert_eq!(
        gql_count(&db, "MATCH ()-[r]->() RETURN count(r)"),
        1,
        "GQL oracle"
    );
}
