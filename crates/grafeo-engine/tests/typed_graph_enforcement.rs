//! F1/F2: typed-graph (graph-type-bound) and named-schema constraint enforcement.
//!
//! Previously these were scaffolded but unwired: the validator's graph_name was
//! never set (graph-type label/edge enforcement never fired), and under a named
//! schema bare labels didn't match the schema-qualified type keys (NOT NULL /
//! type / CHECK silently skipped). The critical risk in wiring it is FALSE
//! REJECTION of valid data under a named schema (qualified-vs-bare mismatch), so
//! every test below asserts the allowed case is ACCEPTED as well as the
//! forbidden case rejected.

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_engine::GrafeoDB;

#[test]
fn typed_graph_enforces_node_labels_default_schema() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.execute("CREATE NODE TYPE Person (name STRING)").unwrap();
    s.execute("CREATE NODE TYPE Animal (species STRING)")
        .unwrap();
    s.execute("CREATE GRAPH TYPE PeopleOnly (NODE TYPE Person)")
        .unwrap();
    s.execute("CREATE GRAPH g TYPED PeopleOnly").unwrap();
    s.execute("USE GRAPH g").unwrap();

    assert!(
        s.execute("INSERT (:Animal {species: 'dog'})").is_err(),
        "forbidden label Animal must be rejected by the closed graph type PeopleOnly"
    );
    assert!(
        s.execute("INSERT (:Person {name: 'Alice'})").is_ok(),
        "allowed label Person must be accepted"
    );
}

#[test]
fn typed_graph_enforces_node_labels_named_schema_no_false_reject() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.execute("CREATE SCHEMA hr").unwrap();
    s.execute("SESSION SET SCHEMA hr").unwrap();
    s.execute("CREATE NODE TYPE Person (name STRING)").unwrap();
    s.execute("CREATE NODE TYPE Animal (species STRING)")
        .unwrap();
    s.execute("CREATE GRAPH TYPE PeopleOnly (NODE TYPE Person)")
        .unwrap();
    s.execute("CREATE GRAPH g TYPED PeopleOnly").unwrap();
    s.execute("USE GRAPH g").unwrap();

    // The allowed label must NOT be falsely rejected under a named schema
    // (qualified "hr/Person" allowed-type vs bare "Person" label).
    let allowed = s.execute("INSERT (:Person {name: 'Alice'})");
    assert!(
        allowed.is_ok(),
        "allowed Person must not be falsely rejected under a schema, got {allowed:?}"
    );
    assert!(
        s.execute("INSERT (:Animal {species: 'dog'})").is_err(),
        "forbidden Animal must be rejected under a schema"
    );
}

#[test]
fn node_property_not_null_enforced_under_named_schema() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.execute("CREATE SCHEMA hr").unwrap();
    s.execute("SESSION SET SCHEMA hr").unwrap();
    s.execute("CREATE NODE TYPE Employee (id INTEGER NOT NULL, name STRING)")
        .unwrap();

    // Missing the NOT NULL `id` must be rejected (F1: was silently skipped under
    // a named schema because resolved_node_type("Employee") missed "hr/Employee").
    let bad = s.execute("INSERT (:Employee {name: 'Bob'})");
    assert!(
        bad.is_err(),
        "missing NOT NULL id must be rejected under a named schema, got {bad:?}"
    );
    // A complete Employee is accepted (no false rejection).
    let good = s.execute("INSERT (:Employee {id: 1, name: 'Bob'})");
    assert!(
        good.is_ok(),
        "valid Employee must be accepted, got {good:?}"
    );
}

#[test]
fn edge_property_not_null_enforced_under_named_schema() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.execute("CREATE SCHEMA hr").unwrap();
    s.execute("SESSION SET SCHEMA hr").unwrap();
    s.execute("CREATE NODE TYPE Person (name STRING)").unwrap();
    s.execute("CREATE EDGE TYPE Knows (since INTEGER NOT NULL)")
        .unwrap();
    s.execute("INSERT (:Person {name: 'A'})").unwrap();
    s.execute("INSERT (:Person {name: 'B'})").unwrap();

    // Edge missing the NOT NULL `since` must be rejected under a named schema
    // (F1 edge symmetry: get_edge_type_def previously missed "hr/Knows").
    let bad =
        s.execute("MATCH (a:Person {name: 'A'}), (b:Person {name: 'B'}) INSERT (a)-[:Knows]->(b)");
    assert!(
        bad.is_err(),
        "edge missing NOT NULL since must be rejected under a named schema, got {bad:?}"
    );
    // A complete edge is accepted (no false rejection).
    let good = s.execute(
        "MATCH (a:Person {name: 'A'}), (b:Person {name: 'B'}) INSERT (a)-[:Knows {since: 2020}]->(b)",
    );
    assert!(good.is_ok(), "valid edge must be accepted, got {good:?}");
}

#[test]
fn edge_endpoint_labels_enforced_under_named_schema() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.execute("CREATE SCHEMA hr").unwrap();
    s.execute("SESSION SET SCHEMA hr").unwrap();
    s.execute("CREATE NODE TYPE Person (name STRING)").unwrap();
    s.execute("CREATE NODE TYPE Company (name STRING)").unwrap();
    s.execute("CREATE EDGE TYPE WorksAt CONNECTING (Person) TO (Company)")
        .unwrap();
    s.execute("INSERT (:Person {name: 'A'})").unwrap();
    s.execute("INSERT (:Company {name: 'C'})").unwrap();

    // Wrong source endpoint (Company, not Person) must be rejected under a named
    // schema (the follow-up: get_edge_type_def previously missed "hr/WorksAt").
    let bad = s.execute(
        "MATCH (a:Company {name: 'C'}), (b:Company {name: 'C'}) INSERT (a)-[:WorksAt]->(b)",
    );
    assert!(
        bad.is_err(),
        "WorksAt source must be Person, not Company, under a named schema; got {bad:?}"
    );
    // Valid Person -> Company endpoints accepted (no false rejection).
    let good = s.execute(
        "MATCH (a:Person {name: 'A'}), (b:Company {name: 'C'}) INSERT (a)-[:WorksAt]->(b)",
    );
    assert!(
        good.is_ok(),
        "valid Person -> Company endpoints must be accepted, got {good:?}"
    );
}
