//! Regressions for three catalog/schema bugs found in review:
//! - F4: `SHOW GRAPH TYPE <name>` failed under a named schema (bare lookup).
//! - F5: `SESSION SET SCHEMA` stored the user-typed case, breaking type lookups
//!   for mixed-case schema names.
//! - F3: `CREATE OR REPLACE NODE TYPE` was lost on WAL replay (old def won).

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[test]
fn show_graph_type_singular_resolves_under_schema() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.execute("CREATE SCHEMA hr").unwrap();
    s.execute("SESSION SET SCHEMA hr").unwrap();
    s.execute("CREATE NODE TYPE Person (name STRING)").unwrap();
    s.execute("CREATE GRAPH TYPE social (NODE TYPE Person)")
        .unwrap();

    let r = s
        .execute("SHOW GRAPH TYPE social")
        .expect("SHOW GRAPH TYPE <name> must resolve under a named schema");
    assert_eq!(r.rows().len(), 1, "one graph-type row");
    assert_eq!(
        r.rows()[0][0],
        Value::from("social"),
        "unqualified name shown"
    );
}

#[test]
fn set_schema_case_insensitive_resolves_canonical() {
    let db = GrafeoDB::new_in_memory();
    {
        let s = db.session();
        s.execute("CREATE SCHEMA MySchema").unwrap();
        s.execute("SESSION SET SCHEMA MySchema").unwrap(); // canonical case
        s.execute("CREATE NODE TYPE Person (name STRING)").unwrap(); // -> "MySchema/Person"
    }
    {
        let s = db.session();
        s.execute("SESSION SET SCHEMA myschema").unwrap(); // different case
        // With the canonicalization fix, the type created under "MySchema" is visible.
        let r = s.execute("SHOW NODE TYPES").unwrap();
        assert_eq!(
            r.rows().len(),
            1,
            "case-different SESSION SET SCHEMA must resolve to the canonical schema, got {:?}",
            r.rows()
        );
    }
}

#[test]
fn create_or_replace_node_type_survives_wal_replay() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("orreplace.grafeo");
    {
        let db = GrafeoDB::open(&path).expect("open");
        let s = db.session();
        s.execute("CREATE NODE TYPE Person (name STRING)").unwrap();
        s.execute("CREATE OR REPLACE NODE TYPE Person (name STRING, age INTEGER)")
            .unwrap();
        db.close().expect("close");
    }

    let db = GrafeoDB::open(&path).expect("reopen");
    let s = db.session();
    let r = s.execute("SHOW NODE TYPES").unwrap();
    let person = r
        .rows()
        .iter()
        .find(|row| row[0] == Value::from("Person"))
        .expect("Person type present after WAL replay");
    let props = match &person[1] {
        Value::String(p) => p.to_string(),
        other => format!("{other:?}"),
    };
    assert!(
        props.contains("age"),
        "CREATE OR REPLACE must survive WAL replay (replacement def wins); props = {props}"
    );
}
