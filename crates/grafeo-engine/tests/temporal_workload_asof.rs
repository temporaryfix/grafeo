//! Temporal workload as-of queries on `GrafeoDB` (Track A public API).
//!
//! Observed-by, correlation, and N-hop provenance at a viewing epoch,
//! including after an edge is closed and compacted.
//!
//! Requires `compact-store` (LayeredStore merge). Not official LDBC.

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "gql"))]

use grafeo_common::types::{EpochId, Value};
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

fn bump_epoch(db: &GrafeoDB) {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.commit().unwrap();
}

fn names(result: &QueryResult) -> Vec<String> {
    let mut out: Vec<String> = result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(s) => s.to_string(),
            other => format!("{other:?}"),
        })
        .collect();
    out.sort();
    out
}

fn pair_names(result: &QueryResult) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = result
        .rows()
        .iter()
        .map(|row| {
            let a = match &row[0] {
                Value::String(s) => s.to_string(),
                other => format!("{other:?}"),
            };
            let b = match &row[1] {
                Value::String(s) => s.to_string(),
                other => format!("{other:?}"),
            };
            (a, b)
        })
        .collect();
    out.sort();
    out
}

/// Observer —OBSERVED_BY→ Entity; one observation is later closed.
#[test]
fn observed_by_at_epoch_hides_closed_edge() {
    let mut db = GrafeoDB::new_in_memory();
    let radar = db.create_node_with_props(&["Observer"], [("name", Value::from("radar"))]);
    let ship = db.create_node_with_props(&["Entity"], [("name", Value::from("ship"))]);
    let plane = db.create_node_with_props(&["Entity"], [("name", Value::from("plane"))]);
    db.create_edge(radar, ship, "OBSERVED_BY");
    let dead = db.create_edge(radar, plane, "OBSERVED_BY");
    let e_open = db.current_epoch();

    bump_epoch(&db);
    assert!(db.delete_edge(dead));
    let e_del = db.current_epoch();
    bump_epoch(&db);
    db.compact().expect("compact");

    let q = "MATCH (o:Observer)-[:OBSERVED_BY]->(e:Entity) RETURN e.name";
    let session = db.session();

    let current = session.execute(q).unwrap();
    assert_eq!(names(&current), vec!["ship".to_string()]);

    let at_open = session.execute_at_epoch(q, e_open).unwrap();
    assert_eq!(
        names(&at_open),
        vec!["plane".to_string(), "ship".to_string()],
        "as-of before delete must still show the closed observation"
    );

    let at_del = session.execute_at_epoch(q, e_del).unwrap();
    assert_eq!(
        names(&at_del),
        vec!["ship".to_string()],
        "closed OBSERVED_BY absent at/after delete epoch"
    );

    let pending = session.execute_at_epoch(q, EpochId::PENDING).unwrap();
    assert_eq!(names(&pending), names(&current));
}

/// Entity —CORRELATED→ Entity at T; closing one correlation drops it after T.
#[test]
fn correlation_at_epoch() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node_with_props(&["Entity"], [("name", Value::from("alpha"))]);
    let b = db.create_node_with_props(&["Entity"], [("name", Value::from("bravo"))]);
    let c = db.create_node_with_props(&["Entity"], [("name", Value::from("charlie"))]);
    db.create_edge(a, b, "CORRELATED");
    let dead = db.create_edge(a, c, "CORRELATED");
    let e_open = db.current_epoch();

    bump_epoch(&db);
    assert!(db.delete_edge(dead));
    let e_del = db.current_epoch();
    bump_epoch(&db);
    db.compact().expect("compact");

    let q = "MATCH (x:Entity)-[:CORRELATED]->(y:Entity) RETURN x.name, y.name";
    let session = db.session();

    assert_eq!(
        pair_names(&session.execute_at_epoch(q, e_open).unwrap()),
        vec![
            ("alpha".to_string(), "bravo".to_string()),
            ("alpha".to_string(), "charlie".to_string()),
        ]
    );
    assert_eq!(
        pair_names(&session.execute_at_epoch(q, e_del).unwrap()),
        vec![("alpha".to_string(), "bravo".to_string())]
    );
}

/// Claim —DERIVED_FROM→ Claim —DERIVED_FROM→ Claim (2-hop provenance).
#[test]
fn provenance_two_hop_asof_after_close() {
    let mut db = GrafeoDB::new_in_memory();
    let root = db.create_node_with_props(&["Claim"], [("name", Value::from("root"))]);
    let mid = db.create_node_with_props(&["Claim"], [("name", Value::from("mid"))]);
    let leaf_a = db.create_node_with_props(&["Claim"], [("name", Value::from("leafA"))]);
    let leaf_b = db.create_node_with_props(&["Claim"], [("name", Value::from("leafB"))]);
    db.create_edge(mid, root, "DERIVED_FROM");
    db.create_edge(leaf_a, mid, "DERIVED_FROM");
    let dead = db.create_edge(leaf_b, mid, "DERIVED_FROM");
    let e_open = db.current_epoch();

    bump_epoch(&db);
    assert!(db.delete_edge(dead));
    let e_del = db.current_epoch();
    bump_epoch(&db);
    db.compact().expect("compact");

    let q = "MATCH (c:Claim)-[:DERIVED_FROM]->(m:Claim)-[:DERIVED_FROM]->(r:Claim) RETURN c.name, r.name";
    let session = db.session();

    assert_eq!(
        pair_names(&session.execute_at_epoch(q, e_open).unwrap()),
        vec![
            ("leafA".to_string(), "root".to_string()),
            ("leafB".to_string(), "root".to_string()),
        ],
        "as-of before close includes both provenance paths"
    );
    assert_eq!(
        pair_names(&session.execute_at_epoch(q, e_del).unwrap()),
        vec![("leafA".to_string(), "root".to_string())],
        "closed DERIVED_FROM drops that 2-hop path"
    );
}
