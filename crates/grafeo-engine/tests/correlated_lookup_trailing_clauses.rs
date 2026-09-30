//! A clause after `CREATE` in an `UNWIND` batch must keep the correlated
//! endpoint lookup and must still execute. The planner-level witnesses live in
//! `query::planner::lpg::correlated_lookup::tests`; these run the same three
//! trailing-clause forms through the public `Session` so that admission is
//! proved against the operator tree the engine actually installs resources on.
//!
//! ```bash
//! cargo test -p grafeo-engine --test correlated_lookup_trailing_clauses
//! ```

#![cfg(feature = "lpg")]

use std::collections::{BTreeMap, HashMap};

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

const BATCH: i64 = 16;
const NODES: i64 = 8;

/// `NODES` `:Node` rows keyed by `id`, plus a `$es` batch of `BATCH` maps each
/// carrying endpoint keys `s`/`t` and a property map `p`.
fn batch_fixture() -> (GrafeoDB, HashMap<String, Value>) {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for id in 0..NODES {
        session
            .create_node_with_props(&["Node"], [("id", Value::Int64(id))])
            .unwrap();
    }
    let rows = (0..BATCH)
        .map(|i| {
            Value::Map(
                BTreeMap::from([
                    (PropertyKey::new("s"), Value::Int64(i % NODES)),
                    (PropertyKey::new("t"), Value::Int64((i + 1) % NODES)),
                    (
                        PropertyKey::new("p"),
                        Value::Map(
                            BTreeMap::from([(PropertyKey::new("w"), Value::Int64(i))]).into(),
                        ),
                    ),
                ])
                .into(),
            )
        })
        .collect::<Vec<_>>();
    let params = HashMap::from([("es".to_string(), Value::List(rows.into()))]);
    (db, params)
}

#[test]
fn batched_create_with_trailing_edge_set_map_executes() {
    let (db, params) = batch_fixture();
    let session = db.session();
    session
        .execute_with_params(
            "UNWIND $es AS e MATCH (s:Node {id:e.s}), (t:Node {id:e.t}) \
             CREATE (s)-[r:REL]->(t) SET r = e.p",
            params,
        )
        .unwrap();
    let counted = session
        .execute("MATCH ()-[r:REL]->() WHERE r.w >= 0 RETURN count(r) AS n")
        .unwrap();
    assert_eq!(counted.rows()[0][0], Value::Int64(BATCH));
}

#[test]
fn batched_create_with_trailing_edge_set_property_executes() {
    let (db, params) = batch_fixture();
    let session = db.session();
    session
        .execute_with_params(
            "UNWIND $es AS e MATCH (s:Node {id:e.s}), (t:Node {id:e.t}) \
             CREATE (s)-[r:REL]->(t) SET r.k = e.s",
            params,
        )
        .unwrap();
    let counted = session
        .execute("MATCH ()-[r:REL]->() WHERE r.k >= 0 RETURN count(r) AS n")
        .unwrap();
    assert_eq!(counted.rows()[0][0], Value::Int64(BATCH));
}

#[test]
fn batched_create_with_trailing_aggregate_executes() {
    let (db, params) = batch_fixture();
    let session = db.session();
    let result = session
        .execute_with_params(
            "UNWIND $es AS e MATCH (s:Node {id:e.s}), (t:Node {id:e.t}) \
             CREATE (s)-[r:REL]->(t) RETURN count(r) AS n",
            params,
        )
        .unwrap();
    assert_eq!(result.rows()[0][0], Value::Int64(BATCH));
}

/// A trailing SET on a *node* property is not admitted, because it can change
/// the very property an endpoint lookup key filters on. It must still produce
/// the right rows on the fallback path.
#[test]
fn batched_create_with_trailing_node_set_still_correct() {
    let (db, params) = batch_fixture();
    let session = db.session();
    session
        .execute_with_params(
            "UNWIND $es AS e MATCH (s:Node {id:e.s}), (t:Node {id:e.t}) \
             CREATE (s)-[r:REL]->(t) SET s.seen = e.t",
            params,
        )
        .unwrap();
    let counted = session
        .execute("MATCH ()-[r:REL]->() RETURN count(r) AS n")
        .unwrap();
    assert_eq!(counted.rows()[0][0], Value::Int64(BATCH));
    let seen = session
        .execute("MATCH (n:Node) WHERE n.seen >= 0 RETURN count(n) AS n")
        .unwrap();
    assert_eq!(seen.rows()[0][0], Value::Int64(NODES));
}
