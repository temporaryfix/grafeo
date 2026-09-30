//! Facade `rdf` must compile and transact SPARQL/WAL without `gql`/`graphql`.
//!
//! ```text
//! cargo test -p grafeo-engine --no-default-features \
//!   --features "triple-store,sparql,wal,grafeo-file" \
//!   --test rdf_no_gql -- --test-threads=1
//! ```

#![cfg(all(
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file",
    not(feature = "gql"),
    not(feature = "graphql"),
))]

use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};

fn persistent_rdf_sync(path: &std::path::Path) -> GrafeoDB {
    let config = Config::persistent(path)
        .with_graph_model(GraphModel::Rdf)
        .with_wal_durability(DurabilityMode::Sync);
    GrafeoDB::with_config(config).expect("open rdf db")
}

fn count_names(db: &GrafeoDB) -> usize {
    db.execute_sparql("SELECT ?n WHERE { ?s <http://ex.org/name> ?n }")
        .expect("select")
        .row_count()
}

/// SPARQL INSERT DATA + close + reopen must recover without GQL in the compile set.
#[test]
fn sparql_insert_wal_recovers_without_gql() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("rdf_no_gql.grafeo");
    {
        let db = persistent_rdf_sync(&path);
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/a> <http://ex.org/name> "Alix" . }"#)
            .unwrap();
        assert_eq!(count_names(&db), 1);
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        count_names(&db),
        1,
        "SPARQL WAL/snapshot must recover without gql"
    );
}

fn count_in_graph(db: &GrafeoDB, graph: &str) -> usize {
    db.execute_sparql(&format!(
        "SELECT ?o WHERE {{ GRAPH <{graph}> {{ ?s <http://ex.org/p> ?o }} }}"
    ))
    .expect("select")
    .row_count()
}

fn insert_named(session: &grafeo_engine::Session, graph: &str, subject: &str, value: &str) {
    session
        .execute_sparql(&format!(
            r#"INSERT DATA {{ GRAPH <{graph}> {{ <{subject}> <http://ex.org/p> "{value}" }} }}"#
        ))
        .unwrap_or_else(|e| panic!("insert into {graph}: {e}"));
}

/// Explicit RDF-only tx: two named-graph inserts, RYW, rollback, reopen absent.
#[test]
fn explicit_tx_two_named_graphs_rollback_not_durable() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("rdf_tx_rollback.grafeo");
    {
        let db = persistent_rdf_sync(&path);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        insert_named(&session, "http://ex.org/g1", "http://ex.org/s1", "a");
        insert_named(&session, "http://ex.org/g2", "http://ex.org/s2", "b");
        assert_eq!(
            session
                .execute_sparql(
                    "SELECT ?o WHERE { GRAPH <http://ex.org/g1> { ?s <http://ex.org/p> ?o } }"
                )
                .unwrap()
                .row_count(),
            1,
            "RYW g1"
        );
        assert_eq!(
            session
                .execute_sparql(
                    "SELECT ?o WHERE { GRAPH <http://ex.org/g2> { ?s <http://ex.org/p> ?o } }"
                )
                .unwrap()
                .row_count(),
            1,
            "RYW g2"
        );
        session.rollback().unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        count_in_graph(&db, "http://ex.org/g1"),
        0,
        "rolled-back g1 must not survive reopen"
    );
    assert_eq!(
        count_in_graph(&db, "http://ex.org/g2"),
        0,
        "rolled-back g2 must not survive reopen"
    );
}

/// Explicit RDF-only tx: two named-graph inserts, commit, reopen both visible.
#[test]
fn explicit_tx_two_named_graphs_commit_durable() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("rdf_tx_commit.grafeo");
    {
        let db = persistent_rdf_sync(&path);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        insert_named(&session, "http://ex.org/g1", "http://ex.org/s1", "a");
        insert_named(&session, "http://ex.org/g2", "http://ex.org/s2", "b");
        let epoch = session.commit().unwrap();
        assert!(epoch.as_u64() > 0, "commit must return EpochId");
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        count_in_graph(&db, "http://ex.org/g1"),
        1,
        "committed g1 must survive reopen"
    );
    assert_eq!(
        count_in_graph(&db, "http://ex.org/g2"),
        1,
        "committed g2 must survive reopen"
    );
}

/// Successful abort then a later successful commit must not resurrect the aborted insert.
#[test]
fn aborted_insert_not_resurrected_after_successful_abort_and_later_commit() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("rdf_tx_no_resurrect.grafeo");
    {
        let db = persistent_rdf_sync(&path);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute_sparql(r#"INSERT DATA { <http://ex.org/aborted> <http://ex.org/p> "dead" . }"#)
            .unwrap();
        session.rollback().expect("normal abort must succeed");

        session.begin_transaction().unwrap();
        session
            .execute_sparql(r#"INSERT DATA { <http://ex.org/live> <http://ex.org/p> "ok" . }"#)
            .unwrap();
        session.commit().expect("later commit must succeed");
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        db.execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/p> ?o }")
            .unwrap()
            .row_count(),
        1,
        "only the committed triple must survive"
    );
    assert_eq!(
        db.execute_sparql("SELECT ?o WHERE { <http://ex.org/live> <http://ex.org/p> ?o }")
            .unwrap()
            .row_count(),
        1,
        "live insert must be present"
    );
    assert_eq!(
        db.execute_sparql("SELECT ?o WHERE { <http://ex.org/aborted> <http://ex.org/p> ?o }")
            .unwrap()
            .row_count(),
        0,
        "aborted insert must not be resurrected"
    );
}

/// Unicode literals in a named graph must survive close/open (WAL + RDF section).
#[test]
fn unicode_named_graph_literals_survive_reopen() {
    use grafeo_engine::{Quad, Term, Triple};

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("rdf_unicode.grafeo");
    let graph = "http://ex.org/g-ja";
    let quads = [
        Quad::named(
            Triple::new(
                Term::iri("http://ex.org/plain"),
                Term::iri("http://ex.org/p"),
                Term::literal("café 日本語"),
            ),
            graph,
        ),
        Quad::named(
            Triple::new(
                Term::iri("http://ex.org/lang"),
                Term::iri("http://ex.org/p"),
                Term::lang_literal("日本語", "ja"),
            ),
            graph,
        ),
        Quad::named(
            Triple::new(
                Term::iri("http://ex.org/typed"),
                Term::iri("http://ex.org/p"),
                Term::typed_literal("Москва", "http://ex.org/City"),
            ),
            graph,
        ),
    ];
    {
        let db = persistent_rdf_sync(&path);
        db.insert_rdf_quads(quads.clone()).unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    let named = db
        .rdf_store()
        .graph(graph)
        .expect("named graph must exist after reopen");
    let pred = Term::iri("http://ex.org/p");
    let exact = |s: &str, o: Term| {
        !named
            .find(&grafeo_core::graph::rdf::TriplePattern {
                subject: Some(Term::iri(s)),
                predicate: Some(pred.clone()),
                object: Some(o),
            })
            .is_empty()
    };
    assert!(
        exact("http://ex.org/plain", Term::literal("café 日本語")),
        "plain Unicode literal must survive reopen"
    );
    assert!(
        exact("http://ex.org/lang", Term::lang_literal("日本語", "ja")),
        "language-tagged Unicode literal must survive reopen"
    );
    assert!(
        exact(
            "http://ex.org/typed",
            Term::typed_literal("Москва", "http://ex.org/City")
        ),
        "typed Unicode literal must survive reopen"
    );
}

/// CLEAR ALL in a transaction must WAL-tag deletes per graph, not as default-graph.
#[test]
fn clear_all_tags_named_graph_deletes() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("clear_all.grafeo");
    {
        let db = persistent_rdf_sync(&path);
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/def> <http://ex.org/p> "default" .
                GRAPH <http://ex.org/g1> { <http://ex.org/s1> <http://ex.org/p> "named" }
            }"#,
        )
        .unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute_sparql("CLEAR ALL").expect("clear all");
        session.commit().unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        db.execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/p> ?o }")
            .unwrap()
            .row_count(),
        0,
        "CLEAR ALL must persist default-graph delete"
    );
    assert_eq!(
        count_in_graph(&db, "http://ex.org/g1"),
        0,
        "CLEAR ALL must persist named-graph delete with graph tagging"
    );
}

/// Graceful close/open must not reissue RDF commit epochs.
#[test]
fn rdf_epoch_does_not_repeat_after_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("epoch.grafeo");
    let first = {
        let db = persistent_rdf_sync(&path);
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/a> <http://ex.org/name> "Alix" . }"#)
            .unwrap();
        let epoch = db.rdf_store_commit_epoch();
        assert!(epoch.as_u64() > 0, "first commit must advance RDF epoch");
        db.close().unwrap();
        epoch
    };
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        db.rdf_store_commit_epoch(),
        first,
        "reopen must restore RDF commit epoch"
    );
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/b> <http://ex.org/name> "Gus" . }"#)
        .unwrap();
    let second = db.rdf_store_commit_epoch();
    assert!(
        second > first,
        "next commit must assign an epoch above the restored high-water, got {second:?} after {first:?}"
    );
    assert_eq!(count_names(&db), 2);
}
