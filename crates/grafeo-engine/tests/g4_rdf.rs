//! G4 — RDF search, transactional SHACL, versioned projections.
//!
//! ```text
//! cargo test -p grafeo-engine --features "triple-store,sparql,wal,grafeo-file,shacl,lpg" \
//!   --test g4_rdf -- --test-threads=1
//! ```

#![cfg(all(feature = "triple-store", feature = "sparql"))]

use grafeo_engine::{Config, GrafeoDB, GraphModel};

#[cfg(all(feature = "lpg", feature = "gql"))]
use grafeo_core::graph::rdf::{Quad, Term, Triple};
#[cfg(all(feature = "lpg", feature = "gql"))]
use grafeo_engine::transaction::IsolationLevel;

fn rdf_db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap()
}

#[cfg(feature = "lpg")]
fn both_db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap()
}

#[test]
fn search_rdf_hits_same_graph_as_sparql() {
    let db = rdf_db();
    db.execute_sparql(
        r#"INSERT DATA { <http://ex.org/a> <http://ex.org/name> "Alix of Amsterdam" . }"#,
    )
    .unwrap();
    let sparql = db
        .execute_sparql(r#"SELECT ?s WHERE { ?s <http://ex.org/name> ?n }"#)
        .unwrap();
    assert_eq!(sparql.row_count(), 1);
    let hits = db.search_rdf("Alix Amsterdam");
    assert_eq!(hits.len(), 1, "search must see the SPARQL-visible triple");
    assert!(hits[0].to_string().contains("ex.org/a"));
}

#[cfg(feature = "shacl")]
const PERSON_NAME_SHAPE: &str = r#"
INSERT DATA { GRAPH <http://ex.org/shapes> {
    <http://ex.org/S> a <http://www.w3.org/ns/shacl#NodeShape> ;
        <http://www.w3.org/ns/shacl#targetClass> <http://ex.org/Person> ;
        <http://www.w3.org/ns/shacl#property> [
            <http://www.w3.org/ns/shacl#path> <http://ex.org/name> ;
            <http://www.w3.org/ns/shacl#minCount> 1
        ] .
} }
"#;

/// Pending Person without `name` must violate. An empty graph (or a Session
/// that cannot see the uncommitted insert) still conforms — so this fails if
/// SHACL ignores pending writes.
#[cfg(feature = "shacl")]
#[test]
fn shacl_sees_uncommitted_named_graph_writes() {
    let db = rdf_db();
    db.execute_sparql(PERSON_NAME_SHAPE).unwrap();
    db.execute_sparql("CREATE GRAPH <http://ex.org/data>")
        .unwrap();
    let empty = db
        .session()
        .validate_shacl_graph("http://ex.org/data", "http://ex.org/shapes")
        .unwrap();
    assert!(
        empty.conforms,
        "empty named graph has no targets, so minCount cannot fire: {empty}"
    );

    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute_sparql(
            r#"INSERT DATA { GRAPH <http://ex.org/data> {
                <http://ex.org/alix> a <http://ex.org/Person> .
            } }"#,
        )
        .unwrap();
    let report = writer
        .validate_shacl_graph("http://ex.org/data", "http://ex.org/shapes")
        .unwrap();
    assert!(
        !report.conforms,
        "pending Person without name must violate minCount (empty graph would still conform): {report}"
    );
    let outsider = db
        .session()
        .validate_shacl_graph("http://ex.org/data", "http://ex.org/shapes")
        .unwrap();
    assert!(
        outsider.conforms,
        "another Session must not see the uncommitted Person: {outsider}"
    );
}

/// SHACL graph lookup must use the same transaction-local lifecycle view as
/// SPARQL. Both graphs here are detached until commit, and disappear on abort.
#[cfg(feature = "shacl")]
#[test]
fn shacl_sees_transaction_local_named_graph_creation() {
    let db = rdf_db();
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute_sparql("CREATE GRAPH <http://ex.org/shapes>")
        .unwrap();
    writer.execute_sparql(PERSON_NAME_SHAPE).unwrap();
    writer
        .execute_sparql("CREATE GRAPH <http://ex.org/data>")
        .unwrap();
    writer
        .execute_sparql(
            r#"INSERT DATA { GRAPH <http://ex.org/data> {
                <http://ex.org/alix> a <http://ex.org/Person> .
            } }"#,
        )
        .unwrap();

    let report = writer
        .validate_shacl_graph("http://ex.org/data", "http://ex.org/shapes")
        .unwrap();
    assert!(
        !report.conforms,
        "SHACL must see detached data and shapes graphs owned by its transaction: {report}"
    );
    writer.rollback().unwrap();

    let outsider = db.session();
    assert!(
        outsider
            .validate_shacl_graph("http://ex.org/data", "http://ex.org/shapes")
            .is_err(),
        "rolled-back RDF graph creation must remain unpublished"
    );
}

/// Same violation-if-visible check on the default graph via `Session::validate_shacl`.
#[cfg(feature = "shacl")]
#[test]
fn shacl_sees_uncommitted_default_graph_writes() {
    let db = rdf_db();
    db.execute_sparql(PERSON_NAME_SHAPE).unwrap();
    let empty = db.session().validate_shacl("http://ex.org/shapes").unwrap();
    assert!(
        empty.conforms,
        "empty default graph has no targets, so minCount cannot fire: {empty}"
    );

    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute_sparql(r#"INSERT DATA { <http://ex.org/alix> a <http://ex.org/Person> . }"#)
        .unwrap();
    let report = writer.validate_shacl("http://ex.org/shapes").unwrap();
    assert!(
        !report.conforms,
        "pending default-graph Person without name must violate minCount: {report}"
    );
    let outsider = db.session().validate_shacl("http://ex.org/shapes").unwrap();
    assert!(
        outsider.conforms,
        "another Session must not see the uncommitted Person: {outsider}"
    );
}

#[cfg(feature = "lpg")]
#[test]
fn projection_is_declared_rebuildable_not_silent() {
    let db = both_db();
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/alix> a <http://ex.org/Person> . }"#)
        .unwrap();
    assert_eq!(
        db.node_count(),
        0,
        "IRIs must not silently become LPG nodes"
    );
    let id = db
        .declare_rdf_lpg_projection("http://ex.org/Person", "Person")
        .unwrap();
    let n = db.rebuild_rdf_lpg_projection(id).unwrap();
    assert_eq!(n, 1);
    assert_eq!(db.node_count(), 1);
    assert_eq!(db.rdf_projection_lag(id), Some(0));
    let id2 = db
        .declare_rdf_lpg_projection("http://ex.org/Person", "Person")
        .unwrap();
    assert_eq!(id, id2, "ProjectionId is a content hash of the mapping");
}

#[test]
fn snapshot_isolation_repeats_rdf_reads_and_sees_own_writes() {
    use grafeo_common::utils::error::ErrorCode;

    let db = rdf_db();
    let graph = "http://ex.org/g";
    db.execute_sparql(&format!(
        r#"INSERT DATA {{
            <http://ex.org/a> <http://ex.org/p> "default-a" .
            GRAPH <{graph}> {{ <http://ex.org/a> <http://ex.org/p> "named-a" . }}
        }}"#
    ))
    .unwrap();

    let default_count = |session: &grafeo_engine::Session| {
        session
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/p> ?o }")
            .unwrap()
            .row_count()
    };
    let named_count = |session: &grafeo_engine::Session| {
        session
            .execute_sparql(&format!(
                "SELECT ?s WHERE {{ GRAPH <{graph}> {{ ?s <http://ex.org/p> ?o }} }}"
            ))
            .unwrap()
            .row_count()
    };

    let mut reader = db.session();
    let mut writer = db.session();
    reader.begin_transaction().unwrap();
    assert_eq!(default_count(&reader), 1);
    assert_eq!(named_count(&reader), 1);

    writer.begin_transaction().unwrap();
    writer
        .execute_sparql(&format!(
            r#"INSERT DATA {{
                <http://ex.org/b> <http://ex.org/p> "default-b" .
                GRAPH <{graph}> {{ <http://ex.org/b> <http://ex.org/p> "named-b" . }}
            }}"#
        ))
        .unwrap();
    let committed = writer.commit().unwrap();

    assert_eq!(
        default_count(&reader),
        1,
        "a repeated default-graph read must retain the transaction start cut"
    );
    assert_eq!(
        named_count(&reader),
        1,
        "a repeated named-graph read must retain the transaction start cut"
    );

    let own_write = format!(
        r#"INSERT DATA {{
            <http://ex.org/c> <http://ex.org/p> "default-c" .
            GRAPH <{graph}> {{ <http://ex.org/c> <http://ex.org/p> "named-c" . }}
        }}"#
    );
    reader.execute_sparql(&own_write).unwrap();
    assert_eq!(
        default_count(&reader),
        2,
        "default graph must read own write"
    );
    assert_eq!(named_count(&reader), 2, "named graph must read own write");

    let outsider = db.session();
    assert_eq!(
        default_count(&outsider),
        2,
        "outsider sees A+B, not pending C"
    );
    assert_eq!(
        named_count(&outsider),
        2,
        "outsider sees A+B, not pending C"
    );

    let error = reader.commit().unwrap_err();
    assert_eq!(error.error_code(), ErrorCode::TransactionConflict);
    assert!(error.error_code().is_retryable());
    assert!(!reader.in_transaction());
    assert_eq!(db.current_epoch(), committed);
    assert_eq!(db.rdf_store_commit_epoch(), committed);
    assert_eq!(default_count(&db.session()), 2);
    assert_eq!(named_count(&db.session()), 2);

    reader.begin_transaction().unwrap();
    assert_eq!(default_count(&reader), 2);
    assert_eq!(named_count(&reader), 2);
    reader.execute_sparql(&own_write).unwrap();
    assert_eq!(default_count(&reader), 3);
    assert_eq!(named_count(&reader), 3);
    let retried = reader.commit().unwrap();
    assert!(retried > committed);
    assert_eq!(db.current_epoch(), retried);
    assert_eq!(db.rdf_store_commit_epoch(), retried);
    assert_eq!(default_count(&db.session()), 3);
    assert_eq!(named_count(&db.session()), 3);
}

#[test]
fn snapshot_reader_pins_named_graph_incarnation_across_drop_recreate() {
    let db = rdf_db();
    let graph = "http://ex.org/replaced";
    let old_subject = "http://ex.org/old";
    let new_subject = "http://ex.org/new";

    db.execute_sparql(&format!(
        r#"INSERT DATA {{ GRAPH <{graph}> {{
            <{old_subject}> <http://ex.org/p> "old" .
        }} }}"#
    ))
    .unwrap();

    let count_subject = |session: &grafeo_engine::Session, subject: &str| {
        session
            .execute_sparql(&format!(
                "SELECT ?o WHERE {{ GRAPH <{graph}> {{ <{subject}> <http://ex.org/p> ?o }} }}"
            ))
            .unwrap()
            .row_count()
    };

    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    assert_eq!(count_subject(&reader, old_subject), 1);

    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute_sparql(&format!("DROP GRAPH <{graph}>"))
        .unwrap();
    writer
        .execute_sparql(&format!("CREATE GRAPH <{graph}>"))
        .unwrap();
    writer
        .execute_sparql(&format!(
            r#"INSERT DATA {{ GRAPH <{graph}> {{
                <{new_subject}> <http://ex.org/p> "new" .
            }} }}"#
        ))
        .unwrap();
    writer.commit().unwrap();

    assert_eq!(
        count_subject(&reader, old_subject),
        1,
        "the reader must retain the exact committed graph Arc from its first access"
    );
    assert_eq!(
        count_subject(&reader, new_subject),
        0,
        "the replacement graph must not leak into the reader's snapshot"
    );
    reader
        .commit()
        .expect("a pure read pin must not become a lifecycle write conflict");

    let outsider = db.session();
    assert_eq!(count_subject(&outsider, old_subject), 0);
    assert_eq!(count_subject(&outsider, new_subject), 1);
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn rdf_and_lpg_share_one_serializable_transaction() {
    let db = both_db();
    let mut session = db.session();
    session
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .unwrap();

    let sparql = session
        .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
        .expect("RDF reads participate in the shared SSI transaction");
    assert_eq!(sparql.row_count(), 0);

    let quad = Quad::new(Triple::new(
        Term::iri("http://ex.org/a"),
        Term::iri("http://ex.org/p"),
        Term::literal("a"),
    ));
    assert!(!session.try_contains_rdf_quad(&quad).unwrap());
    session
        .insert_rdf_quads([quad.clone()])
        .expect("RDF writes participate in the shared SSI transaction");
    assert!(session.try_contains_rdf_quad(&quad).unwrap());

    session
        .execute("CREATE (:SerializableGuard {enabled: true})")
        .expect("LPG writes remain in the same Serializable transaction");
    session.commit().unwrap();

    assert!(db.contains_rdf_quad(&quad));
    assert_eq!(db.node_count(), 1);
}
