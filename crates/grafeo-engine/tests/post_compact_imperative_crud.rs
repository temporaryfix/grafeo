//! Regression tests: the imperative DB CRUD surface (`db.get_node`,
//! `db.set_node_property`, `db.delete_node`, …) must keep working on
//! **base-resident** entities after [`GrafeoDB::compact`]/`recompact`.
//!
//! Pre-fix behaviour: the imperative CRUD routed through `lpg_store()` — the raw
//! overlay `LpgStore`. After `compact()` that overlay is empty (all data folded
//! into the columnar base), so reads of a base entity returned `None`/stale and
//! writes landed on an overlay that has no such node and were silently dropped
//! (the `LayeredStore`'s `dirty_node_ids` was never updated, so the merged read
//! path kept serving the stale base value). Only the GQL/session path — which
//! routes through the `LayeredStore`'s `GraphStoreMut` — was safe.
//!
//! The fix routes the imperative surface through the `LayeredStore` when one is
//! present: reads use the tier-merged view, writes go through the layered
//! `GraphStoreMut` (which promotes base entities into the overlay / sets base
//! tombstones), index maintenance runs against the *live* overlay.
//!
//! ```bash
//! cargo nextest run -p grafeo-engine --features "compact-store lpg gql" \
//!     --test post_compact_imperative_crud
//! ```

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "gql"))]

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

fn gql_int(db: &GrafeoDB, q: &str) -> i64 {
    let s = db.session();
    match &s.execute(q).unwrap().rows()[0][0] {
        Value::Int64(n) => *n,
        other => panic!("expected Int64 for `{q}`, got {other:?}"),
    }
}

fn gql_rows(db: &GrafeoDB, q: &str) -> usize {
    let s = db.session();
    s.execute(q).unwrap().rows().len()
}

fn prop(db: &GrafeoDB, id: grafeo_common::types::NodeId, key: &str) -> Option<Value> {
    db.get_node(id)
        .and_then(|n| n.properties.get(&PropertyKey::new(key)).cloned())
}

// ── reads ──────────────────────────────────────────────────────────

#[test]
fn get_node_on_base_node_after_compact() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    db.set_node_property(a, "v", Value::Int64(1))
        .expect("set node property");
    db.compact().expect("compact");

    let n = db
        .get_node(a)
        .expect("base node must be readable via db.get_node after compact");
    assert!(n.labels.iter().any(|l| l.as_str() == "A"));
    assert_eq!(prop(&db, a, "v"), Some(Value::Int64(1)));
}

#[test]
fn get_edge_on_base_edge_after_compact() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    let e = db.create_edge(a, b, "T");
    db.compact().expect("compact");

    let edge = db
        .get_edge(e)
        .expect("base edge must be readable via db.get_edge after compact");
    assert_eq!(edge.edge_type.as_str(), "T");
}

// ── writes ─────────────────────────────────────────────────────────

#[test]
fn set_property_on_base_node_persists_after_compact() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    db.set_node_property(a, "v", Value::Int64(1))
        .expect("set node property");
    db.compact().expect("compact");

    db.set_node_property(a, "v", Value::Int64(2))
        .expect("set node property");

    // Visible through the imperative read path …
    assert_eq!(prop(&db, a, "v"), Some(Value::Int64(2)));
    // … and through the safe GQL/session path.
    assert_eq!(gql_int(&db, "MATCH (n:A) RETURN n.v"), 2);
}

#[test]
fn add_label_on_base_node_persists_after_compact() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    db.compact().expect("compact");

    assert!(db.add_node_label(a, "B"), "label should be added");
    assert_eq!(gql_rows(&db, "MATCH (n:B) RETURN n"), 1);
    let n = db.get_node(a).expect("node still present");
    assert!(n.labels.iter().any(|l| l.as_str() == "B"));
}

#[test]
fn remove_property_on_base_node_after_compact() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    db.set_node_property(a, "v", Value::Int64(1))
        .expect("set node property");
    db.compact().expect("compact");

    assert!(
        db.remove_node_property(a, "v"),
        "property should be removed"
    );
    assert_eq!(prop(&db, a, "v"), None);
    assert_eq!(
        gql_rows(&db, "MATCH (n:A) WHERE n.v IS NOT NULL RETURN n"),
        0
    );
}

#[test]
fn delete_base_node_after_compact() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["A"]);
    db.compact().expect("compact");

    assert!(db.delete_node(a), "base node delete should report success");
    assert!(
        db.get_node(a).is_none(),
        "deleted node must not be readable"
    );
    assert!(db.get_node(b).is_some(), "sibling base node survives");
    assert_eq!(gql_int(&db, "MATCH (n) RETURN count(n)"), 1);
}

#[test]
fn delete_base_edge_after_compact() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    let e = db.create_edge(a, b, "T");
    db.compact().expect("compact");

    assert!(db.delete_edge(e), "base edge delete should report success");
    assert!(
        db.get_edge(e).is_none(),
        "deleted edge must not be readable"
    );
    assert_eq!(gql_int(&db, "MATCH ()-[r]->() RETURN count(r)"), 0);
}

#[test]
fn create_edge_between_base_nodes_after_compact() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    db.compact().expect("compact");

    let e = db.create_edge(a, b, "T");
    assert!(
        db.get_edge(e).is_some(),
        "new edge between base nodes readable"
    );
    assert_eq!(gql_int(&db, "MATCH ()-[r:T]->() RETURN count(r)"), 1);
    assert_eq!(
        gql_rows(&db, "MATCH (a:A)-[:T]->(b:B) RETURN a, b"),
        1,
        "edge must connect the two base endpoints"
    );
}

// ── recompact (stale-overlay-handle regression) ────────────────────

#[test]
fn set_property_on_base_node_persists_after_recompact() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    db.compact().expect("compact");
    db.set_node_property(a, "v", Value::Int64(1))
        .expect("set node property");
    db.compact().expect("recompact");

    // After recompact the overlay was swapped; the imperative handle must not be
    // stale. `a` is base-resident again, so this write must promote + persist.
    db.set_node_property(a, "v", Value::Int64(2))
        .expect("set node property");

    assert_eq!(prop(&db, a, "v"), Some(Value::Int64(2)));
    assert_eq!(gql_int(&db, "MATCH (n:A) RETURN n.v"), 2);
}
