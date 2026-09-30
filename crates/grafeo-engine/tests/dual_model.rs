//! One physical database holds native LPG and native RDF. No silent IRI≡node mirror.
//!
//! ```text
//! cargo test -p grafeo-engine --features "lpg,gql,triple-store,sparql,wal,grafeo-file" \
//!   --test dual_model -- --test-threads=1
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file"
))]

use grafeo_engine::{Config, GrafeoDB, GraphModel};

fn both_persistent(path: &std::path::Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path).with_graph_model(GraphModel::Both))
        .expect("open dual-model db")
}

fn gql_count(db: &GrafeoDB) -> i64 {
    db.session()
        .execute("MATCH (n) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap()
}

fn sparql_count(db: &GrafeoDB) -> usize {
    db.execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
        .unwrap()
        .row_count()
}

/// GQL and SPARQL both work on GraphModel::Both; RDF IRIs are not LPG nodes.
#[test]
fn both_models_native_no_silent_mirror() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap();
    db.create_node(&["Person"]);
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "v" }"#)
        .unwrap();
    assert_eq!(gql_count(&db), 1, "LPG node visible to GQL");
    assert_eq!(sparql_count(&db), 1, "RDF triple visible to SPARQL");
    assert_eq!(
        db.node_count(),
        1,
        "RDF insert must not create an LPG node (no silent mirror)"
    );
}

/// One .grafeo file round-trips LPG + RDF.
#[test]
fn both_models_persist_one_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("both.grafeo");
    {
        let db = both_persistent(&path);
        db.create_node(&["Person"]);
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "v" }"#)
            .unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(gql_count(&db), 1);
    assert_eq!(sparql_count(&db), 1);
    assert_eq!(db.node_count(), 1, "reopen must not mirror RDF into LPG");
}

/// RDF-only GraphModel still rejects GQL.
#[test]
fn rdf_model_still_rejects_gql() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let err = db.session().execute("MATCH (n) RETURN n");
    assert!(err.is_err(), "RDF model must still reject GQL, got {err:?}");
}

/// LPG-labelled databases reject RDF mutations; RDF-labelled reject LPG CRUD.
#[test]
fn graph_model_enforced_at_runtime() {
    let lpg = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg)).unwrap();
    let rdf_err = lpg.execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "v" }"#);
    assert!(
        rdf_err.is_err(),
        "LPG database must reject SPARQL UPDATE, got {rdf_err:?}"
    );
    let quad_err = lpg.insert_rdf_quads([grafeo_engine::Quad::new(grafeo_engine::Triple::new(
        grafeo_engine::Term::iri("http://ex.org/s"),
        grafeo_engine::Term::iri("http://ex.org/p"),
        grafeo_engine::Term::literal("v"),
    ))]);
    assert!(
        quad_err.is_err(),
        "LPG database must reject insert_rdf_quads, got {quad_err:?}"
    );

    let rdf = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let id = rdf.create_node(&["Person"]);
    assert!(
        !id.is_valid(),
        "RDF database must not create LPG nodes, got {id:?}"
    );
    let eid = rdf.create_edge(id, id, "KNOWS");
    assert!(
        !eid.is_valid(),
        "RDF database must not create LPG edges, got {eid:?}"
    );
    let sess = rdf.session();
    let sid = sess.create_node(&["Person"]);
    assert!(
        !sid.is_valid(),
        "RDF session must not create LPG nodes, got {sid:?}"
    );
}

/// One Session transaction mutating LPG and RDF is atomic on rollback/reopen.
#[test]
fn both_models_mixed_tx_rollback_not_durable() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("both_rollback.grafeo");
    {
        let db = both_persistent(&path);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session
            .execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "v" }"#)
            .unwrap();
        assert_eq!(
            session
                .execute("MATCH (n:Person) RETURN count(n)")
                .unwrap()
                .rows()[0][0]
                .as_int64()
                .unwrap(),
            1,
            "LPG RYW in mixed tx"
        );
        assert_eq!(
            session
                .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
                .unwrap()
                .row_count(),
            1,
            "RDF RYW in mixed tx"
        );
        session.rollback().unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(gql_count(&db), 0, "rolled-back LPG must not survive reopen");
    assert_eq!(
        sparql_count(&db),
        0,
        "rolled-back RDF must not survive reopen"
    );
}

/// Committed mixed LPG+RDF transaction is crash-stable in one .grafeo file.
#[test]
fn both_models_mixed_tx_commit_durable() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("both_commit.grafeo");
    {
        let db = both_persistent(&path);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session
            .execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "v" }"#)
            .unwrap();
        let epoch = session.commit().unwrap();
        assert!(epoch.as_u64() > 0);
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(gql_count(&db), 1, "committed LPG must survive reopen");
    assert_eq!(sparql_count(&db), 1, "committed RDF must survive reopen");
    assert_eq!(
        db.node_count(),
        1,
        "committed mixed tx must not mirror RDF into LPG"
    );
}
