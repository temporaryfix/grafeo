//! Parser-free `native` profile: LPG CRUD + RDF quads + WAL, no query languages.
//!
//! ```text
//! cargo test -p grafeo-engine --no-default-features --features native \
//!   --test native_no_parsers -- --test-threads=1
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "triple-store",
    feature = "wal",
    feature = "grafeo-file",
))]

use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel, Quad, Term, Triple};

fn both_sync(path: &std::path::Path) -> GrafeoDB {
    let config = Config::persistent(path)
        .with_graph_model(GraphModel::Both)
        .with_wal_durability(DurabilityMode::Sync);
    GrafeoDB::with_config(config).expect("open both db")
}

fn named_quad(s: &str, p: &str, o: &str, g: &str) -> Quad {
    Quad::named(Triple::new(Term::iri(s), Term::iri(p), Term::literal(o)), g)
}

#[test]
fn native_profile_excludes_parsers() {
    // File-level `not(gql)` used to skip the whole binary (0 tests, exit 0)
    // when a parser leaked in. Functional tests below always run. Isolation
    // is asserted unless this is a full languages / `--all-features` build.
    const {
        let all_languages = cfg!(feature = "gql")
            && cfg!(feature = "cypher")
            && cfg!(feature = "sparql")
            && cfg!(feature = "gremlin")
            && cfg!(feature = "graphql")
            && cfg!(feature = "sql-pgq");
        let no_languages = !cfg!(feature = "gql")
            && !cfg!(feature = "cypher")
            && !cfg!(feature = "sparql")
            && !cfg!(feature = "gremlin")
            && !cfg!(feature = "graphql")
            && !cfg!(feature = "sql-pgq");
        assert!(
            all_languages || no_languages,
            "a query-language feature leaked into this native_no_parsers build; \
             compile with --no-default-features --features native"
        );
    }
}

#[test]
fn both_create_node_and_insert_quad_in_memory() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both))
        .expect("both");
    let id = db.create_node(&["Person"]);
    assert_ne!(id.as_u64(), u64::MAX);
    assert_eq!(db.node_count(), 1);

    let quad = named_quad(
        "http://ex.org/s",
        "http://ex.org/p",
        "hello",
        "http://ex.org/g",
    );
    let (n, epoch) = db.insert_rdf_quads([quad.clone()]).expect("insert");
    assert_eq!(n, 1);
    assert!(epoch.as_u64() > 0);
    assert!(db.contains_rdf_quad(&quad));
}

#[test]
fn session_quad_rollback_not_visible() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both))
        .expect("both");
    let quad = named_quad(
        "http://ex.org/s",
        "http://ex.org/p",
        "pending",
        "http://ex.org/g",
    );
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.insert_rdf_quads([quad.clone()]).unwrap();
    assert!(
        session.contains_rdf_quad(&quad),
        "open tx must see its own quad"
    );
    session.rollback().unwrap();
    assert!(
        !db.contains_rdf_quad(&quad),
        "rolled-back quad must not be visible"
    );
}

#[test]
fn mixed_lpg_and_rdf_survive_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("native.grafeo");
    let quad = named_quad(
        "http://ex.org/s",
        "http://ex.org/name",
        "Alix",
        "http://ex.org/g",
    );
    {
        let db = both_sync(&path);
        db.create_node(&["Person"]);
        db.insert_rdf_quads([quad.clone()]).unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(db.graph_model(), GraphModel::Both);
    assert_eq!(db.node_count(), 1, "LPG node must survive reopen");
    assert!(
        db.contains_rdf_quad(&quad),
        "RDF quad must survive reopen without SPARQL"
    );
}

#[cfg(not(any(feature = "gql", feature = "cypher")))]
#[test]
fn default_query_entrypoints_are_structured_unsupported() {
    let db = GrafeoDB::new_in_memory();
    db.create_node(&["Before"]);
    let session = db.session();
    for result in [
        session.execute("RETURN 1"),
        session.execute_with_params(
            "RETURN $value",
            [("value".to_owned(), grafeo_common::types::Value::Int64(1))].into(),
        ),
    ] {
        let error = result.expect_err("default parser is unavailable");
        assert_eq!(error.error_code().as_str(), "GRAFEO-Q004");
    }
    assert_eq!(db.node_count(), 1);
    session.create_node(&["After"]);
    assert_eq!(db.node_count(), 2);
}
