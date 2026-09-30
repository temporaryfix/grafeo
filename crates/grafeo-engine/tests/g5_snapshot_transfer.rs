//! G5 — snapshot transfer SPARQL identity (native). WASM identity is in
//! `crates/bindings/wasm` unit tests.
//!
//! ```text
//! cargo test -p grafeo-engine --features "triple-store,sparql,wal,grafeo-file" \
//!   --test g5_snapshot_transfer -- --test-threads=1
//! ```

#![cfg(all(
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file"
))]

use grafeo_engine::{Config, GrafeoDB, GraphModel};

#[test]
fn compact_snapshot_reopen_same_sparql_answers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g5.grafeo");
    let before;
    {
        let db = GrafeoDB::with_config(Config::persistent(&path).with_graph_model(GraphModel::Rdf))
            .unwrap();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/a> <http://ex.org/name> "Alix"@en .
                <http://ex.org/a> <http://ex.org/p> <http://ex.org/b> .
            }"#,
        )
        .unwrap();
        before = db
            .execute_sparql("SELECT ?n WHERE { ?s <http://ex.org/name> ?n }")
            .unwrap()
            .row_count();
        assert_eq!(before, 1);
        db.close().unwrap();
    }
    let db =
        GrafeoDB::with_config(Config::persistent(&path).with_graph_model(GraphModel::Rdf)).unwrap();
    let after = db
        .execute_sparql("SELECT ?n WHERE { ?s <http://ex.org/name> ?n }")
        .unwrap()
        .row_count();
    assert_eq!(after, before, "snapshot reopen must keep SPARQL answers");
    let lang = db
        .execute_sparql("SELECT ?n WHERE { ?s <http://ex.org/name> ?n }")
        .unwrap();
    match &lang.rows()[0][0] {
        grafeo_common::types::Value::RdfLiteral {
            language: Some(l), ..
        } => assert_eq!(l.as_str(), "en"),
        other => panic!("lang tag must survive snapshot, got {other:?}"),
    }
}
