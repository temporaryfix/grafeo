//! G3 — RDF transaction-time and optional valid-time.
//!
//! ```text
//! cargo test -p grafeo-engine --features "triple-store,sparql,wal,grafeo-file" \
//!   --test g3_rdf_time -- --test-threads=1
//! ```

#![cfg(all(
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file"
))]

use grafeo_common::types::EpochId;
use grafeo_core::graph::rdf::{Quad, Term, Triple};
use grafeo_engine::{
    Config, DurabilityMode, GrafeoDB, GraphIncarnationId, GraphModel, HistoryCompleteness,
    RdfGraphIdentity, RdfHistoryCut, RdfHistoryTransition, RdfHistoryTransitionKind,
    TaiNanoseconds, ValidTimeInterval,
};

/// Check the complete public cut without projecting away graph or time metadata.
fn assert_exact_cut(
    db: &GrafeoDB,
    cut: &RdfHistoryCut,
    epoch: EpochId,
    expected: &[(Quad, Option<ValidTimeInterval>)],
) {
    assert_eq!(cut.store_id, db.rdf_store().store_id());
    assert_eq!(cut.epoch, epoch);
    assert_eq!(cut.completeness, HistoryCompleteness::Complete);
    assert_eq!(cut.quads.len(), expected.len());
    for (quad, valid) in expected {
        let actual = cut
            .quads
            .iter()
            .find(|row| &row.quad == quad)
            .expect("exact typed quad");
        let incarnation = match quad.graph() {
            None => GraphIncarnationId::DEFAULT_GRAPH,
            Some(name) => {
                let graph = cut
                    .named_graphs
                    .iter()
                    .find(|graph| graph.name() == Some(name))
                    .expect("named quad's graph lifetime");
                assert!(!graph.incarnation().is_default_graph());
                graph.incarnation()
            }
        };
        assert_eq!(actual.graph_incarnation, incarnation);
        assert_eq!(actual.valid, *valid);
        assert_eq!(
            actual.statement,
            db.rdf_statement_handle(quad, incarnation).unwrap()
        );
    }
}

fn persistent_rdf(path: &std::path::Path) -> GrafeoDB {
    persistent_model(path, GraphModel::Rdf)
}

fn persistent_model(path: &std::path::Path, model: GraphModel) -> GrafeoDB {
    let config = Config::persistent(path)
        .with_graph_model(model)
        .with_wal_durability(DurabilityMode::Sync);
    GrafeoDB::with_config(config).expect("open rdf db")
}

fn sidecar_wal_dir(path: &std::path::Path) -> std::path::PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".wal");
    std::path::PathBuf::from(p)
}

fn copy_live_database(src: &std::path::Path, dst: &std::path::Path) {
    std::fs::copy(src, dst).expect("copy live .grafeo container");
    let src_wal = sidecar_wal_dir(src);
    if src_wal.exists() {
        let dst_wal = sidecar_wal_dir(dst);
        std::fs::create_dir_all(&dst_wal).expect("create copied WAL directory");
        for entry in std::fs::read_dir(src_wal).expect("read live WAL directory") {
            let entry = entry.expect("WAL directory entry");
            if entry.path().is_file() {
                std::fs::copy(entry.path(), dst_wal.join(entry.file_name()))
                    .expect("copy live WAL segment");
            }
        }
    }
}

fn copy_sidecar_wal(src: &std::path::Path, dst: &std::path::Path) {
    let src_wal = sidecar_wal_dir(src);
    let dst_wal = sidecar_wal_dir(dst);
    if dst_wal.exists() {
        std::fs::remove_dir_all(&dst_wal).expect("remove destination WAL directory");
    }
    std::fs::create_dir_all(&dst_wal).expect("create destination WAL directory");
    for entry in std::fs::read_dir(src_wal).expect("read source WAL directory") {
        let entry = entry.expect("WAL directory entry");
        if entry.path().is_file() {
            std::fs::copy(entry.path(), dst_wal.join(entry.file_name())).expect("copy WAL segment");
        }
    }
}

fn triple() -> Triple {
    Triple::new(
        Term::iri("http://ex.org/s"),
        Term::iri("http://ex.org/p"),
        Term::literal("v"),
    )
}

fn triple_with_subject(subject: &str) -> Triple {
    Triple::new(
        Term::iri(subject),
        Term::iri("http://ex.org/p"),
        Term::literal("v"),
    )
}

#[test]
fn empty_authoritative_container_rejects_foreign_wal_identity() {
    let dir = tempfile::tempdir().expect("create temporary directory");
    let authoritative_path = dir.path().join("authoritative-empty.grafeo");
    let foreign_path = dir.path().join("foreign.grafeo");

    let authoritative = persistent_rdf(&authoritative_path);
    let authoritative_store_id = authoritative.store_id();
    authoritative
        .close()
        .expect("persist empty authoritative container");

    let foreign = persistent_rdf(&foreign_path);
    assert_ne!(
        foreign.store_id(),
        authoritative_store_id,
        "test precondition: the foreign WAL must identify another store"
    );
    foreign
        .wal()
        .expect("foreign database WAL")
        .sync()
        .expect("sync foreign identity metadata");
    copy_sidecar_wal(&foreign_path, &authoritative_path);

    let reopen = Config::persistent(&authoritative_path)
        .with_graph_model(GraphModel::Rdf)
        .with_wal_durability(DurabilityMode::Sync);
    let Err(error) = GrafeoDB::with_config(reopen) else {
        panic!("an authoritative container must reject foreign WAL identity metadata");
    };
    assert!(
        error.to_string().contains("identity"),
        "mismatch must be reported as an identity recovery failure: {error}"
    );
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn mixed_savepoint_pending_ops_match_live_and_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed_savepoint.grafeo");
    let copy = dir.path().join("mixed_savepoint_copy.grafeo");
    let db = persistent_model(&path, GraphModel::Both);
    db.execute_sparql("CREATE GRAPH <http://ex.org/named>")
        .unwrap();
    let session = db.session();

    session.execute("START TRANSACTION").unwrap();
    session.execute("INSERT (:Keep {id: 1})").unwrap();
    session
        .execute_sparql(
            r#"INSERT DATA { <http://ex.org/default-before> <http://ex.org/p> "keep" . }"#,
        )
        .unwrap();
    session
        .execute_sparql(
            r#"INSERT DATA { GRAPH <http://ex.org/named> { <http://ex.org/named-before> <http://ex.org/p> "keep" . } }"#,
        )
        .unwrap();
    session.execute("SAVEPOINT stable").unwrap();
    session.execute("INSERT (:Discard {id: 2})").unwrap();
    session
        .execute_sparql(
            r#"INSERT DATA { <http://ex.org/default-after> <http://ex.org/p> "discard" . }"#,
        )
        .unwrap();
    session
        .execute_sparql(
            r#"INSERT DATA { GRAPH <http://ex.org/named> { <http://ex.org/named-after> <http://ex.org/p> "discard" . } }"#,
        )
        .unwrap();
    session.execute("ROLLBACK TO SAVEPOINT stable").unwrap();
    session.execute("COMMIT").unwrap();

    let lpg = session.execute("MATCH (n) RETURN count(n)").unwrap();
    let default = session
        .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/p> ?o }")
        .unwrap();
    let named = session
        .execute_sparql(
            "SELECT ?s WHERE { GRAPH <http://ex.org/named> { ?s <http://ex.org/p> ?o } }",
        )
        .unwrap();
    assert_eq!(lpg.rows()[0][0].as_int64(), Some(1));
    assert_eq!(default.row_count(), 1);
    assert_eq!(named.row_count(), 1);

    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &copy);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent_model(&copy, GraphModel::Both);
    let lpg = recovered
        .session()
        .execute("MATCH (n) RETURN count(n)")
        .unwrap();
    let default = recovered
        .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/p> ?o }")
        .unwrap();
    let named = recovered
        .execute_sparql(
            "SELECT ?s WHERE { GRAPH <http://ex.org/named> { ?s <http://ex.org/p> ?o } }",
        )
        .unwrap();
    assert_eq!(lpg.rows()[0][0].as_int64(), Some(1));
    assert_eq!(default.row_count(), 1);
    assert_eq!(named.row_count(), 1);
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn rdf_graph_lifecycle_savepoint_matches_live_and_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rdf_graph_savepoint.grafeo");
    let copy = dir.path().join("rdf_graph_savepoint_copy.grafeo");
    let db = persistent_model(&path, GraphModel::Both);
    db.execute_sparql("CREATE GRAPH <http://ex.org/existing>")
        .unwrap();
    db.execute_sparql(
        r#"INSERT DATA { GRAPH <http://ex.org/existing> {
          <http://ex.org/before> <http://ex.org/p> "keep" .
        } }"#,
    )
    .unwrap();
    let session = db.session();
    session.execute("START TRANSACTION").unwrap();
    session.execute("SAVEPOINT stable").unwrap();
    session
        .execute_sparql("CREATE GRAPH <http://ex.org/post-savepoint>")
        .unwrap();
    session
        .execute_sparql(
            r#"INSERT DATA { GRAPH <http://ex.org/post-savepoint> {
              <http://ex.org/after> <http://ex.org/p> "discard" .
            } }"#,
        )
        .unwrap();
    session
        .execute_sparql("DROP GRAPH <http://ex.org/existing>")
        .unwrap();

    let owner_created = session
        .execute_sparql("SELECT ?s WHERE { GRAPH <http://ex.org/post-savepoint> { ?s ?p ?o } }")
        .unwrap();
    let owner_dropped = session
        .execute_sparql("SELECT ?s WHERE { GRAPH <http://ex.org/existing> { ?s ?p ?o } }")
        .unwrap();
    assert_eq!(owner_created.row_count(), 1);
    assert_eq!(owner_dropped.row_count(), 0);

    let observer = db.session();
    assert_eq!(
        observer
            .execute_sparql(
                "SELECT ?s WHERE { GRAPH <http://ex.org/post-savepoint> { ?s ?p ?o } }",
            )
            .unwrap()
            .row_count(),
        0,
        "detached CREATE must be invisible to other sessions"
    );
    assert_eq!(
        observer
            .execute_sparql("SELECT ?s WHERE { GRAPH <http://ex.org/existing> { ?s ?p ?o } }",)
            .unwrap()
            .row_count(),
        1,
        "staged DROP must not hide the committed partition from other sessions"
    );

    session.execute("ROLLBACK TO SAVEPOINT stable").unwrap();
    assert_eq!(
        session
            .execute_sparql(
                "SELECT ?s WHERE { GRAPH <http://ex.org/post-savepoint> { ?s ?p ?o } }",
            )
            .unwrap()
            .row_count(),
        0
    );
    assert_eq!(
        session
            .execute_sparql("SELECT ?s WHERE { GRAPH <http://ex.org/existing> { ?s ?p ?o } }",)
            .unwrap()
            .row_count(),
        1
    );
    session.execute("COMMIT").unwrap();

    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &copy);
    drop(observer);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent_model(&copy, GraphModel::Both);
    assert!(
        recovered
            .rdf_store()
            .graph("http://ex.org/post-savepoint")
            .is_none()
    );
    assert_eq!(
        recovered
            .execute_sparql("SELECT ?s WHERE { GRAPH <http://ex.org/existing> { ?s ?p ?o } }",)
            .unwrap()
            .row_count(),
        1
    );
}

#[test]
fn rdf_only_savepoint_pending_ops_match_live_and_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rdf_only_savepoint.grafeo");
    let copy = dir.path().join("rdf_only_savepoint_copy.grafeo");
    let db = persistent_rdf(&path);
    db.execute_sparql("CREATE GRAPH <http://ex.org/existing>")
        .unwrap();

    let retained_valid_time = ValidTimeInterval::from_tai_nanoseconds(1_000, 2_000).unwrap();
    let later_valid_time = ValidTimeInterval::from_tai_nanoseconds(3_000, 4_000).unwrap();
    let mut session = db.session();
    session.set_rdf_valid_time(Some(retained_valid_time));
    session.begin_transaction().unwrap();
    session
        .execute_sparql(
            r#"INSERT DATA {
              <http://ex.org/default-before> <http://ex.org/p> "keep" .
              GRAPH <http://ex.org/existing> {
                <http://ex.org/named-before> <http://ex.org/p> "keep" .
              }
            }"#,
        )
        .unwrap();
    session.savepoint("stable").unwrap();

    session.set_rdf_valid_time(Some(later_valid_time));
    session
        .execute_sparql(
            r#"INSERT DATA {
              <http://ex.org/default-after> <http://ex.org/p> "discard" .
            }"#,
        )
        .unwrap();
    session
        .execute_sparql("CREATE GRAPH <http://ex.org/post-savepoint>")
        .unwrap();
    session
        .execute_sparql(
            r#"INSERT DATA { GRAPH <http://ex.org/post-savepoint> {
              <http://ex.org/named-after> <http://ex.org/p> "discard" .
            } }"#,
        )
        .unwrap();
    session
        .execute_sparql("DROP GRAPH <http://ex.org/existing>")
        .unwrap();
    assert_eq!(
        session
            .execute_sparql(
                "SELECT ?s WHERE { GRAPH <http://ex.org/post-savepoint> { ?s ?p ?o } }",
            )
            .unwrap()
            .row_count(),
        1
    );
    assert_eq!(
        session
            .execute_sparql("SELECT ?s WHERE { GRAPH <http://ex.org/existing> { ?s ?p ?o } }")
            .unwrap()
            .row_count(),
        0
    );

    session.rollback_to_savepoint("stable").unwrap();
    assert_eq!(
        session.rdf_valid_time(),
        Some(later_valid_time),
        "savepoint rollback rewinds captured operations, not the Session setting"
    );
    assert_eq!(
        session
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/p> ?o }")
            .unwrap()
            .row_count(),
        1
    );
    assert_eq!(
        session
            .execute_sparql("SELECT ?s WHERE { GRAPH <http://ex.org/existing> { ?s ?p ?o } }")
            .unwrap()
            .row_count(),
        1
    );
    assert_eq!(
        session
            .execute_sparql(
                "SELECT ?s WHERE { GRAPH <http://ex.org/post-savepoint> { ?s ?p ?o } }",
            )
            .unwrap()
            .row_count(),
        0
    );
    session.release_savepoint("stable").unwrap();
    session.commit().unwrap();

    let retained = Triple::new(
        Term::iri("http://ex.org/default-before"),
        Term::iri("http://ex.org/p"),
        Term::literal("keep"),
    );
    let (_, lives) = db
        .rdf_store()
        .quad_history()
        .into_iter()
        .find(|(candidate, _)| candidate.as_ref() == &retained)
        .expect("retained default-graph history row");
    assert_eq!(lives.len(), 1);
    assert_eq!(lives[0].valid, Some(retained_valid_time));

    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &copy);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent_rdf(&copy);
    assert_eq!(
        recovered
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/p> ?o }")
            .unwrap()
            .row_count(),
        1
    );
    assert_eq!(
        recovered
            .execute_sparql("SELECT ?s WHERE { GRAPH <http://ex.org/existing> { ?s ?p ?o } }")
            .unwrap()
            .row_count(),
        1
    );
    assert!(
        recovered
            .rdf_store()
            .graph("http://ex.org/post-savepoint")
            .is_none()
    );
    let (_, recovered_lives) = recovered
        .rdf_store()
        .quad_history()
        .into_iter()
        .find(|(candidate, _)| candidate.as_ref() == &retained)
        .expect("recovered retained default-graph history row");
    assert_eq!(recovered_lives.len(), 1);
    assert_eq!(recovered_lives[0].valid, Some(retained_valid_time));
}

#[cfg(feature = "testing-statement-injection")]
#[test]
fn rdf_statement_error_rewinds_only_that_statement_live_and_after_recovery() {
    use grafeo_common::testing::statement_failure::with_statement_failure_after;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rdf_statement_atomicity.grafeo");
    let crash_copy = dir.path().join("rdf_statement_atomicity_copy.grafeo");
    let db = persistent_rdf(&path);
    let mut session = db.session();

    let retained_before = triple_with_subject("http://ex.org/retained-before");
    let discarded_default = triple_with_subject("http://ex.org/discarded-default");
    let discarded_named = triple_with_subject("http://ex.org/discarded-named");
    let retained_after = triple_with_subject("http://ex.org/retained-after");
    let discarded_graph = "http://ex.org/discarded-graph";

    session.begin_transaction().unwrap();
    session
        .execute_sparql(
            r#"INSERT DATA {
              <http://ex.org/retained-before> <http://ex.org/p> "v" .
            }"#,
        )
        .unwrap();
    session
        .begin_transaction()
        .expect("open a nested caller-owned frame around the failing statement");

    let error = with_statement_failure_after(1, || {
        session.execute_sparql(
            r#"INSERT DATA {
              <http://ex.org/discarded-default> <http://ex.org/p> "v" .
              GRAPH <http://ex.org/discarded-graph> {
                <http://ex.org/discarded-named> <http://ex.org/p> "v" .
              }
            }"#,
        )
    })
    .expect_err("the post-execution failure must escape to the caller");
    assert!(
        error
            .to_string()
            .contains("post-execution statement failure"),
        "unexpected injected error: {error}"
    );
    assert!(
        !db.is_durability_poisoned(),
        "a successful statement-savepoint rewind must leave durability healthy"
    );
    assert!(
        session.in_transaction(),
        "an ordinary statement error must not discard the caller-owned transaction"
    );
    assert_eq!(
        session
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/p> ?o }")
            .unwrap()
            .row_count(),
        1,
        "earlier work survives while every default-graph effect of the failed statement rewinds"
    );
    assert_eq!(
        session
            .execute_sparql(&format!(
                "SELECT ?s WHERE {{ GRAPH <{discarded_graph}> {{ ?s ?p ?o }} }}"
            ))
            .unwrap()
            .row_count(),
        0,
        "the failed statement's named-graph effects must rewind"
    );
    assert!(
        db.rdf_store().graph(discarded_graph).is_none(),
        "the failed statement must not publish a named-graph incarnation"
    );

    session.commit().unwrap();
    assert!(
        session.in_transaction(),
        "statement rewind must preserve the caller's nested transaction depth"
    );

    session
        .execute_sparql(
            r#"INSERT DATA {
              <http://ex.org/retained-after> <http://ex.org/p> "v" .
            }"#,
        )
        .unwrap();
    session.commit().unwrap();

    for retained in [&retained_before, &retained_after] {
        assert!(db.contains_rdf_quad(&Quad::new(retained.clone())));
    }
    assert!(!db.contains_rdf_quad(&Quad::new(discarded_default.clone())));
    assert!(!db.contains_rdf_quad(&Quad::named(discarded_named.clone(), discarded_graph,)));
    assert!(db.rdf_store().graph(discarded_graph).is_none());

    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &crash_copy);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent_rdf(&crash_copy);
    for retained in [retained_before, retained_after] {
        assert!(recovered.contains_rdf_quad(&Quad::new(retained)));
    }
    assert!(!recovered.contains_rdf_quad(&Quad::new(discarded_default)));
    assert!(!recovered.contains_rdf_quad(&Quad::named(discarded_named, discarded_graph,)));
    assert!(recovered.rdf_store().graph(discarded_graph).is_none());
}

#[test]
fn dropping_rdf_session_aborts_every_nested_transaction_level() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let observer = db.session();
    let outer = Quad::new(triple_with_subject("http://ex.org/drop-outer"));
    let inner = Quad::new(triple_with_subject("http://ex.org/drop-inner"));

    {
        let mut owner = db.session();
        owner.begin_transaction().unwrap();
        owner.insert_rdf_quads([outer.clone()]).unwrap();
        owner.begin_transaction().unwrap();
        owner.insert_rdf_quads([inner.clone()]).unwrap();

        assert!(owner.in_transaction());
        assert_eq!(owner.transaction_manager_ref().active_count(), 1);
        assert!(owner.try_contains_rdf_quad(&outer).unwrap());
        assert!(owner.try_contains_rdf_quad(&inner).unwrap());
    }

    assert_eq!(
        observer.transaction_manager_ref().active_count(),
        0,
        "Session::drop must abort the outer transaction, not only unwind one nested savepoint"
    );
    assert!(!observer.try_contains_rdf_quad(&outer).unwrap());
    assert!(!observer.try_contains_rdf_quad(&inner).unwrap());
    assert!(db.rdf_store().is_empty());
}

#[test]
fn rdf_public_savepoint_names_cannot_collide_with_nested_protocol_state() {
    const RESERVED: &str = "\0grafeo:caller";
    const LEGACY_COLLISION: &str = "_nested_tx_1";

    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let mut session = db.session();
    let retained = Quad::new(triple_with_subject("http://ex.org/savepoint-retained"));
    let outer_discarded = Quad::new(triple_with_subject("http://ex.org/savepoint-outer"));
    let inner_discarded = Quad::new(triple_with_subject("http://ex.org/savepoint-inner"));

    session.begin_transaction().unwrap();
    for error in [
        session.savepoint(RESERVED).unwrap_err(),
        session.rollback_to_savepoint(RESERVED).unwrap_err(),
        session.release_savepoint(RESERVED).unwrap_err(),
    ] {
        assert!(
            error.to_string().contains("reserved internal namespace"),
            "reserved names must fail explicitly: {error}"
        );
    }

    session.insert_rdf_quads([retained.clone()]).unwrap();
    session.savepoint(LEGACY_COLLISION).unwrap();
    session.insert_rdf_quads([outer_discarded.clone()]).unwrap();

    session
        .begin_transaction()
        .expect("the engine's nested frame must not collide with a caller savepoint");
    session.insert_rdf_quads([inner_discarded.clone()]).unwrap();
    session
        .rollback()
        .expect("nested rollback must target only the engine-owned frame");

    assert!(session.try_contains_rdf_quad(&retained).unwrap());
    assert!(session.try_contains_rdf_quad(&outer_discarded).unwrap());
    assert!(!session.try_contains_rdf_quad(&inner_discarded).unwrap());

    session
        .rollback_to_savepoint(LEGACY_COLLISION)
        .expect("the caller savepoint must survive nested rollback");
    assert!(session.try_contains_rdf_quad(&retained).unwrap());
    assert!(!session.try_contains_rdf_quad(&outer_discarded).unwrap());
    session.release_savepoint(LEGACY_COLLISION).unwrap();
    session.commit().unwrap();

    assert!(db.rdf_store().contains(retained.triple()));
    assert!(!db.rdf_store().contains(outer_discarded.triple()));
    assert!(!db.rdf_store().contains(inner_discarded.triple()));
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn rdf_named_graph_commit_cas_prevents_orphans_and_lost_writes() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap();
    let first = db.session();
    let second = db.session();
    first.execute("START TRANSACTION").unwrap();
    second.execute("START TRANSACTION").unwrap();

    first
        .execute_sparql("CREATE GRAPH <http://ex.org/raced>")
        .unwrap();
    second
        .execute_sparql("CREATE GRAPH <http://ex.org/raced>")
        .unwrap();
    first
        .execute_sparql(
            r#"INSERT DATA { GRAPH <http://ex.org/raced> {
              <http://ex.org/first> <http://ex.org/p> "first" .
            } }"#,
        )
        .unwrap();
    second
        .execute_sparql(
            r#"INSERT DATA { GRAPH <http://ex.org/raced> {
              <http://ex.org/second> <http://ex.org/p> "second" .
            } }"#,
        )
        .unwrap();
    assert!(
        db.rdf_store().graph("http://ex.org/raced").is_none(),
        "neither detached create may be visible before commit"
    );

    first.execute("COMMIT").unwrap();
    let error = second.execute("COMMIT").unwrap_err();
    assert!(error.error_code().is_retryable(), "{error}");
    assert!(!second.in_transaction());
    assert!(
        error.to_string().contains("created concurrently"),
        "{error}"
    );
    assert_eq!(
        db.execute_sparql("SELECT ?s WHERE { GRAPH <http://ex.org/raced> { ?s ?p ?o } }")
            .unwrap()
            .row_count(),
        1,
        "the losing CREATE must not replace the winner's exact partition"
    );

    let dropper = db.session();
    let writer = db.session();
    dropper.execute("START TRANSACTION").unwrap();
    writer.execute("START TRANSACTION").unwrap();
    dropper
        .execute_sparql("DROP GRAPH <http://ex.org/raced>")
        .unwrap();
    writer
        .execute_sparql(
            r#"INSERT DATA { GRAPH <http://ex.org/raced> {
              <http://ex.org/writer> <http://ex.org/p> "writer" .
            } }"#,
        )
        .unwrap();
    writer.execute("COMMIT").unwrap();
    let error = dropper.execute("COMMIT").unwrap_err();
    assert!(error.error_code().is_retryable(), "{error}");
    assert!(!dropper.in_transaction());
    assert!(
        error.to_string().contains("changed concurrently"),
        "{error}"
    );
    assert_eq!(
        db.execute_sparql("SELECT ?s WHERE { GRAPH <http://ex.org/raced> { ?s ?p ?o } }")
            .unwrap()
            .row_count(),
        2,
        "a stale DROP must not erase a concurrently committed graph write"
    );
}

#[test]
fn rdf_revision_conflict_is_retryable_and_rolls_back_before_publication() {
    use grafeo_common::utils::error::ErrorCode;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rdf-conflict.grafeo");
    let db = persistent_rdf(&path);
    let mut first = db.session();
    let mut second = db.session();
    let first_write = r#"INSERT DATA { <urn:first> <urn:p> "first" }"#;
    let second_write = r#"INSERT DATA { <urn:second> <urn:p> "second" }"#;
    first.begin_transaction().unwrap();
    second.begin_transaction().unwrap();
    first.execute_sparql(first_write).unwrap();
    second.execute_sparql(second_write).unwrap();
    let committed = first.commit().unwrap();
    let error = second.commit().unwrap_err();
    assert_eq!(error.error_code(), ErrorCode::TransactionConflict);
    assert!(error.error_code().is_retryable());
    assert!(!second.in_transaction());
    assert_eq!(db.current_epoch(), committed);
    assert_eq!(db.rdf_store_commit_epoch(), committed);
    let first_quad = Quad::new(Triple::new(
        Term::iri("urn:first"),
        Term::iri("urn:p"),
        Term::literal("first"),
    ));
    let committed_cut = db.rdf_history_cut(committed).unwrap();
    assert_exact_cut(
        &db,
        &committed_cut,
        committed,
        &[(first_quad.clone(), None)],
    );
    assert!(committed_cut.named_graphs.is_empty());
    assert!(!db.contains_rdf_quad(&Quad::new(Triple::new(
        Term::iri("urn:second"),
        Term::iri("urn:p"),
        Term::literal("second"),
    ))));

    second.begin_transaction().unwrap();
    second.execute_sparql(second_write).unwrap();
    let retried = second.commit().unwrap();
    assert!(retried > committed);
    assert_eq!(db.rdf_history_cut(committed).unwrap(), committed_cut);
    let retried_cut = db.rdf_history_cut(retried).unwrap();
    assert_exact_cut(
        &db,
        &retried_cut,
        retried,
        &[
            (first_quad, None),
            (
                Quad::new(Triple::new(
                    Term::iri("urn:second"),
                    Term::iri("urn:p"),
                    Term::literal("second"),
                )),
                None,
            ),
        ],
    );
    assert!(retried_cut.named_graphs.is_empty());
    drop(first);
    drop(second);
    db.close().unwrap();
    let reopened = persistent_rdf(&path);
    assert_eq!(reopened.rdf_store_commit_epoch(), retried);
    assert_eq!(reopened.rdf_history_cut(committed).unwrap(), committed_cut);
    assert_eq!(reopened.rdf_history_cut(retried).unwrap(), retried_cut);
}

#[test]
fn rdf_tx_time_cut_diff_and_persist() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g3.grafeo");
    let insert_cut;
    let delete_cut;
    let diff;
    {
        let db = persistent_rdf(&path);
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "v" . }"#)
            .unwrap();
        let inserted = db.rdf_store_commit_epoch();
        insert_cut = db.rdf_history_cut(inserted).unwrap();
        assert_exact_cut(&db, &insert_cut, inserted, &[(Quad::new(triple()), None)]);
        assert!(insert_cut.named_graphs.is_empty());
        db.execute_sparql(r#"DELETE DATA { <http://ex.org/s> <http://ex.org/p> "v" . }"#)
            .unwrap();
        let deleted = db.rdf_store_commit_epoch();
        assert!(deleted > inserted);
        assert_eq!(db.rdf_history_cut(inserted).unwrap(), insert_cut);
        delete_cut = db.rdf_history_cut(deleted).unwrap();
        assert_exact_cut(&db, &delete_cut, deleted, &[]);
        assert!(delete_cut.named_graphs.is_empty());
        diff = db.rdf_history_diff(inserted, deleted).unwrap();
        assert_eq!(diff.store_id, insert_cut.store_id);
        assert_eq!(diff.completeness, HistoryCompleteness::Complete);
        assert_eq!((diff.from, diff.through), (inserted, deleted));
        assert_eq!(
            diff.transitions,
            vec![RdfHistoryTransition {
                epoch: deleted,
                graph: RdfGraphIdentity::default_graph(),
                kind: RdfHistoryTransitionKind::StatementRetracted {
                    quad: Quad::new(triple()),
                    statement: insert_cut.quads[0].statement,
                    valid: None,
                },
            }]
        );
        db.close().unwrap();
    }
    let db = persistent_rdf(&path);
    assert_eq!(db.rdf_history_cut(insert_cut.epoch).unwrap(), insert_cut);
    assert_eq!(db.rdf_history_cut(delete_cut.epoch).unwrap(), delete_cut);
    assert_eq!(db.rdf_history_diff(diff.from, diff.through).unwrap(), diff);
}

#[test]
fn rdf_valid_time_is_not_only_filter() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g3v.grafeo");
    let inside;
    let after;
    {
        let db = persistent_rdf(&path);
        db.insert_rdf_valid([triple()], 1_000, 2_000).unwrap();
        let epoch = db.rdf_store_commit_epoch();
        inside = db
            .rdf_history_cut_at(epoch, Some(TaiNanoseconds::new(1_500_000)))
            .unwrap();
        assert_exact_cut(
            &db,
            &inside,
            epoch,
            &[(
                Quad::new(triple()),
                Some(ValidTimeInterval::from_tai_nanoseconds(1_000_000, 2_000_000).unwrap()),
            )],
        );
        assert!(inside.named_graphs.is_empty());
        after = db
            .rdf_history_cut_at(epoch, Some(TaiNanoseconds::new(2_500_000)))
            .unwrap();
        assert_exact_cut(&db, &after, epoch, &[]);
        assert!(after.named_graphs.is_empty());
        db.close().unwrap();
    }
    let db = persistent_rdf(&path);
    assert_eq!(
        db.rdf_history_cut_at(inside.epoch, Some(TaiNanoseconds::new(1_500_000)))
            .unwrap(),
        inside
    );
    assert_eq!(
        db.rdf_history_cut_at(after.epoch, Some(TaiNanoseconds::new(2_500_000)))
            .unwrap(),
        after
    );
    let before = db
        .rdf_history_cut_at(inside.epoch, Some(TaiNanoseconds::new(500_000)))
        .unwrap();
    assert_exact_cut(&db, &before, inside.epoch, &[]);
}

#[test]
fn rdf_tx_time_wal_only_recovery_matches_live_intervals() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g3_crash.grafeo");
    let crash_copy = dir.path().join("g3_crash_copy.grafeo");
    let db = persistent_rdf(&path);

    db.execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "v" . }"#)
        .unwrap();
    let insert_epoch = db.rdf_store_commit_epoch();
    db.execute_sparql(r#"DELETE DATA { <http://ex.org/s> <http://ex.org/p> "v" . }"#)
        .unwrap();
    let delete_epoch = db.rdf_store_commit_epoch();
    assert!(delete_epoch > insert_epoch);
    let insert_cut = db.rdf_history_cut(insert_epoch).unwrap();
    let delete_cut = db.rdf_history_cut(delete_epoch).unwrap();
    assert_exact_cut(
        &db,
        &insert_cut,
        insert_epoch,
        &[(Quad::new(triple()), None)],
    );
    assert_exact_cut(&db, &delete_cut, delete_epoch, &[]);
    assert!(insert_cut.named_graphs.is_empty());
    assert!(delete_cut.named_graphs.is_empty());
    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &crash_copy);
    std::mem::forget(db); // simulate process loss: no close/checkpoint

    let recovered = persistent_rdf(&crash_copy);
    assert_eq!(recovered.rdf_history_cut(insert_epoch).unwrap(), insert_cut);
    assert_eq!(recovered.rdf_history_cut(delete_epoch).unwrap(), delete_cut);
    let history = recovered.rdf_store().quad_history();
    let (_, lives) = history
        .iter()
        .find(|(candidate, _)| candidate.as_ref() == &triple())
        .expect("recovered triple history");
    assert_eq!(lives.len(), 1);
    assert_eq!(lives[0].tx.from(), insert_epoch);
    assert_eq!(lives[0].tx.to(), delete_epoch);
    assert_eq!(lives[0].valid, None);
}

#[test]
fn rdf_valid_time_wal_only_recovery_matches_live_interval() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g3_valid_crash.grafeo");
    let crash_copy = dir.path().join("g3_valid_crash_copy.grafeo");
    let db = persistent_rdf(&path);

    db.insert_rdf_valid([triple()], 1_000, 2_000).unwrap();
    let insert_epoch = db.rdf_store_commit_epoch();
    let insert_cut = db.rdf_history_cut(insert_epoch).unwrap();
    assert_exact_cut(
        &db,
        &insert_cut,
        insert_epoch,
        &[(
            Quad::new(triple()),
            Some(ValidTimeInterval::from_tai_nanoseconds(1_000_000, 2_000_000).unwrap()),
        )],
    );
    assert!(insert_cut.named_graphs.is_empty());
    let inside = db
        .rdf_history_cut_at(insert_epoch, Some(TaiNanoseconds::new(1_500_000)))
        .unwrap();
    assert_eq!(inside, insert_cut);
    let outside = db
        .rdf_history_cut_at(insert_epoch, Some(TaiNanoseconds::new(2_500_000)))
        .unwrap();
    assert_exact_cut(&db, &outside, insert_epoch, &[]);
    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &crash_copy);
    std::mem::forget(db); // simulate process loss: no close/checkpoint

    let recovered = persistent_rdf(&crash_copy);
    assert_eq!(recovered.rdf_history_cut(insert_epoch).unwrap(), insert_cut);
    assert_eq!(
        recovered
            .rdf_history_cut_at(insert_epoch, Some(TaiNanoseconds::new(1_500_000)))
            .unwrap(),
        inside
    );
    assert_eq!(
        recovered
            .rdf_history_cut_at(insert_epoch, Some(TaiNanoseconds::new(2_500_000)))
            .unwrap(),
        outside
    );
    let history = recovered.rdf_store().quad_history();
    let (_, lives) = history
        .iter()
        .find(|(candidate, _)| candidate.as_ref() == &triple())
        .expect("recovered valid-time history");
    assert_eq!(lives.len(), 1);
    assert_eq!(lives[0].tx.from(), insert_epoch);
    assert!(lives[0].tx.is_open());
    assert_eq!(
        lives[0].valid,
        Some(ValidTimeInterval::from_legacy_micros(1_000, 2_000).unwrap())
    );
}

#[test]
fn session_tai_nanosecond_scope_covers_quads_sparql_transactions_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g3_tai_scope.grafeo");
    let from = i128::from(i64::MAX) * 1_000 + 17;
    let to = from + 29;
    let interval = ValidTimeInterval::from_tai_nanoseconds(from, to).unwrap();
    let after_from = to + 100;
    let after = ValidTimeInterval::from_tai_nanoseconds(after_from, after_from + 7).unwrap();
    let default_direct = triple_with_subject("http://ex.org/default-direct");
    let default_sparql = triple_with_subject("http://ex.org/default-sparql");
    let derived_direct = Triple::new(
        Term::iri("http://ex.org/default-direct"),
        Term::iri("http://ex.org/derived"),
        Term::literal("derived"),
    );
    let derived_sparql = Triple::new(
        Term::iri("http://ex.org/default-sparql"),
        Term::iri("http://ex.org/derived"),
        Term::literal("derived"),
    );
    let named_direct = triple_with_subject("http://ex.org/named-direct");
    let discarded = triple_with_subject("http://ex.org/discarded");
    let inside_cut;
    let upper_cut;

    {
        let db = persistent_rdf(&path);
        let mut session = db.session();
        session.set_rdf_valid_time_tai_ns(from, to).unwrap();
        assert_eq!(session.rdf_valid_time(), Some(interval));
        session.begin_transaction().unwrap();
        assert_eq!(
            session
                .insert_rdf_quads([
                    Quad::new(default_direct.clone()),
                    Quad::named(named_direct.clone(), "http://ex.org/claims"),
                ])
                .unwrap(),
            2
        );
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/default-sparql> <http://ex.org/p> "v" .
                }"#,
            )
            .unwrap();
        session
            .execute_sparql(
                r#"INSERT { ?s <http://ex.org/derived> "derived" }
                   WHERE  { ?s <http://ex.org/p> "v" }"#,
            )
            .unwrap();
        session.commit().unwrap();
        assert_eq!(
            session.rdf_valid_time(),
            Some(interval),
            "commit does not rewind the Session valid-time setting"
        );

        session.set_rdf_valid_time(Some(after));
        session.begin_transaction().unwrap();
        session
            .execute_sparql(
                r#"INSERT DATA { GRAPH <http://ex.org/discarded-graph> {
                    <http://ex.org/discarded> <http://ex.org/p> "v" .
                } }"#,
            )
            .unwrap();
        session.rollback().unwrap();
        assert_eq!(
            session.rdf_valid_time(),
            Some(after),
            "valid time is a Session setting, not transaction state"
        );

        let default_history = db.rdf_store().quad_history();
        for expected in [
            &default_direct,
            &default_sparql,
            &derived_direct,
            &derived_sparql,
        ] {
            let (_, lives) = default_history
                .iter()
                .find(|(candidate, _)| candidate.as_ref() == expected)
                .expect("default-graph history row");
            assert_eq!(lives.len(), 1);
            assert_eq!(lives[0].valid, Some(interval));
        }
        let named = db
            .rdf_store()
            .graph("http://ex.org/claims")
            .expect("named graph");
        let (_, named_lives) = named
            .quad_history()
            .into_iter()
            .find(|(candidate, _)| candidate.as_ref() == &named_direct)
            .expect("named-graph history row");
        assert_eq!(named_lives[0].valid, Some(interval));
        assert!(
            db.rdf_store()
                .graph("http://ex.org/discarded-graph")
                .is_none(),
            "full rollback discards the graph and its captured interval"
        );
        let epoch = db.rdf_store_commit_epoch();
        inside_cut = db
            .rdf_history_cut_at(epoch, Some(TaiNanoseconds::new(from + 1)))
            .unwrap();
        assert_exact_cut(
            &db,
            &inside_cut,
            epoch,
            &[
                (Quad::new(default_direct.clone()), Some(interval)),
                (Quad::new(default_sparql.clone()), Some(interval)),
                (Quad::new(derived_direct.clone()), Some(interval)),
                (Quad::new(derived_sparql.clone()), Some(interval)),
                (
                    Quad::named(named_direct.clone(), "http://ex.org/claims"),
                    Some(interval),
                ),
            ],
        );
        assert_eq!(
            inside_cut.named_graphs,
            vec![
                RdfGraphIdentity::named("http://ex.org/claims", named.graph_incarnation(),)
                    .unwrap()
            ]
        );
        upper_cut = db
            .rdf_history_cut_at(epoch, Some(TaiNanoseconds::new(to)))
            .unwrap();
        assert_exact_cut(&db, &upper_cut, epoch, &[]);
        assert_eq!(
            upper_cut.named_graphs, inside_cut.named_graphs,
            "valid time filters statements without erasing graph lifetimes"
        );

        assert!(!db.rdf_store().contains(&discarded));
        db.close().unwrap();
    }

    let reopened = persistent_rdf(&path);
    assert_eq!(
        reopened
            .rdf_history_cut_at(inside_cut.epoch, Some(TaiNanoseconds::new(from + 1)))
            .unwrap(),
        inside_cut
    );
    assert_eq!(
        reopened
            .rdf_history_cut_at(upper_cut.epoch, Some(TaiNanoseconds::new(to)))
            .unwrap(),
        upper_cut
    );
    let default_history = reopened.rdf_store().quad_history();
    for expected in [
        &default_direct,
        &default_sparql,
        &derived_direct,
        &derived_sparql,
    ] {
        let (_, lives) = default_history
            .iter()
            .find(|(candidate, _)| candidate.as_ref() == expected)
            .expect("reopened default-graph history row");
        assert_eq!(lives[0].valid, Some(interval));
    }
    let named = reopened
        .rdf_store()
        .graph("http://ex.org/claims")
        .expect("reopened named graph");
    let (_, lives) = named
        .quad_history()
        .into_iter()
        .find(|(candidate, _)| candidate.as_ref() == &named_direct)
        .expect("reopened named-graph history row");
    assert_eq!(lives[0].valid, Some(interval));
}

#[cfg(feature = "lpg")]
#[test]
fn savepoint_rollback_discards_captured_ops_but_not_session_valid_time() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let mut session = db.session();
    let before = triple_with_subject("http://ex.org/before-savepoint");
    let discarded = triple_with_subject("http://ex.org/after-savepoint");
    let first = ValidTimeInterval::from_tai_nanoseconds(100, 200).unwrap();
    let second = ValidTimeInterval::from_tai_nanoseconds(300, 400).unwrap();

    session.set_rdf_valid_time(Some(first));
    session.begin_transaction().unwrap();
    session
        .insert_rdf_quads([Quad::new(before.clone())])
        .unwrap();
    session.savepoint("stable").unwrap();
    session.set_rdf_valid_time(Some(second));
    session
        .insert_rdf_quads([Quad::new(discarded.clone())])
        .unwrap();
    session.rollback_to_savepoint("stable").unwrap();
    assert_eq!(session.rdf_valid_time(), Some(second));
    session.commit().unwrap();

    let history = db.rdf_store().quad_history();
    let (_, lives) = history
        .iter()
        .find(|(candidate, _)| candidate.as_ref() == &before)
        .unwrap();
    assert_eq!(lives[0].valid, Some(first));
    assert!(!db.rdf_store().contains(&discarded));
}

#[test]
fn invalid_valid_time_intervals_fail_before_mutation() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let session = db.session();
    let valid = ValidTimeInterval::from_tai_nanoseconds(10, 20).unwrap();
    session.set_rdf_valid_time(Some(valid));

    for (from, to) in [(20, 20), (21, 20)] {
        let error = session
            .set_rdf_valid_time_tai_ns(from, to)
            .expect_err("invalid interval must fail");
        assert_eq!(error.error_code().as_str(), "GRAFEO-V001");
        assert_eq!(session.rdf_valid_time(), Some(valid));
    }

    assert!(
        db.insert_rdf_valid_tai_ns([triple()], 7, 7).is_err(),
        "direct TAI API rejects an empty interval"
    );
    assert!(
        db.insert_rdf_valid([triple()], 9, 8).is_err(),
        "legacy microsecond API rejects an inverted interval"
    );
    assert!(db.rdf_store().is_empty());
    session.clear_rdf_valid_time();
    assert_eq!(session.rdf_valid_time(), None);
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn explicit_transaction_rdf_wal_failure_poisons_and_cannot_commit_or_recover() {
    use grafeo_common::testing::wal_failure::{
        disable_mutation_log_failure, enable_mutation_log_failure_once,
    };

    #[derive(Clone, Copy, Debug)]
    enum MutationPath {
        DirectDefault,
        DirectNamedValid,
        SparqlNamedValid,
    }

    for path_kind in [
        MutationPath::DirectDefault,
        MutationPath::DirectNamedValid,
        MutationPath::SparqlNamedValid,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir
            .path()
            .join(format!("rdf_wal_poison_{path_kind:?}.grafeo"));
        let crash_copy = dir
            .path()
            .join(format!("rdf_wal_poison_{path_kind:?}_copy.grafeo"));
        let db = persistent_rdf(&path);
        let mut session = db.session();
        let statement = triple_with_subject(&format!("http://ex.org/{path_kind:?}"));
        let named_graph = "http://ex.org/poisoned-claims";
        let valid = ValidTimeInterval::from_tai_nanoseconds(1_001, 1_002).unwrap();
        session.begin_transaction().unwrap();

        if !matches!(path_kind, MutationPath::DirectDefault) {
            session.set_rdf_valid_time(Some(valid));
        }
        enable_mutation_log_failure_once();
        let result = match path_kind {
            MutationPath::DirectDefault => session
                .insert_rdf_quads([Quad::new(statement.clone())])
                .map(|_| ()),
            MutationPath::DirectNamedValid => session
                .insert_rdf_quads([Quad::named(statement.clone(), named_graph)])
                .map(|_| ()),
            MutationPath::SparqlNamedValid => session
                .execute_sparql(&format!(
                    "INSERT DATA {{ GRAPH <{named_graph}> {{ {} {} {} . }} }}",
                    statement.subject(),
                    statement.predicate(),
                    statement.object()
                ))
                .map(|_| ()),
        };
        disable_mutation_log_failure();

        let error = result.expect_err("injected RDF WAL failure must reach the caller");
        assert!(error.to_string().contains("WAL"), "{path_kind:?}: {error}");
        assert!(
            db.is_durability_poisoned(),
            "{path_kind:?}: RDF WAL failure must poison the database immediately"
        );
        assert!(
            db.rdf_store().is_empty(),
            "{path_kind:?}: WAL failure must precede live RDF insertion"
        );
        assert!(
            db.rdf_store().graph(named_graph).is_none(),
            "{path_kind:?}: WAL failure must not publish a named-graph lifetime"
        );
        assert!(
            session.commit().is_err(),
            "{path_kind:?}: a caught mutation-log error must never be committable"
        );

        copy_live_database(&path, &crash_copy);
        std::mem::forget(session);
        std::mem::forget(db);

        let recovered = persistent_rdf(&crash_copy);
        let quad = match path_kind {
            MutationPath::DirectDefault => Quad::new(statement),
            MutationPath::DirectNamedValid | MutationPath::SparqlNamedValid => {
                Quad::named(statement, named_graph)
            }
        };
        assert!(
            !recovered.contains_rdf_quad(&quad),
            "{path_kind:?}: failed/uncommitted RDF mutation must not recover"
        );
        assert!(
            recovered.rdf_store().graph(named_graph).is_none(),
            "{path_kind:?}: a failed named-graph mutation must not create graph identity"
        );
    }
}

#[test]
fn tai_nanosecond_named_quad_survives_wal_only_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g3_tai_crash.grafeo");
    let crash_copy = dir.path().join("g3_tai_crash_copy.grafeo");
    let db = persistent_rdf(&path);
    let session = db.session();
    let from = i128::from(i64::MAX) * 1_000 + 1;
    let interval = ValidTimeInterval::from_tai_nanoseconds(from, from + 2).unwrap();
    let named = triple_with_subject("http://ex.org/wal-named");
    session.set_rdf_valid_time(Some(interval));
    session
        .insert_rdf_quads([Quad::named(named.clone(), "http://ex.org/wal-claims")])
        .unwrap();
    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &crash_copy);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent_rdf(&crash_copy);
    let graph = recovered
        .rdf_store()
        .graph("http://ex.org/wal-claims")
        .expect("recovered named graph");
    let (_, lives) = graph
        .quad_history()
        .into_iter()
        .find(|(candidate, _)| candidate.as_ref() == &named)
        .expect("recovered named quad");
    assert_eq!(lives[0].valid, Some(interval));
}

#[test]
fn sparql_copy_move_add_preserve_source_valid_time_not_current_scope() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g3_graph_ops_tai.grafeo");
    let crash_copy = dir.path().join("g3_graph_ops_tai_crash.grafeo");
    let db = persistent_rdf(&path);
    let session = db.session();
    let source = ValidTimeInterval::from_tai_nanoseconds(101, 109).unwrap();
    let unrelated_scope = ValidTimeInterval::from_tai_nanoseconds(501, 509).unwrap();
    session.set_rdf_valid_time(Some(source));

    let copy = triple_with_subject("http://ex.org/copy");
    let add = triple_with_subject("http://ex.org/add");
    let moved = triple_with_subject("http://ex.org/move");
    session
        .insert_rdf_quads([
            Quad::named(copy.clone(), "http://ex.org/copy-src"),
            Quad::named(add.clone(), "http://ex.org/add-src"),
            Quad::named(moved.clone(), "http://ex.org/move-src"),
        ])
        .unwrap();

    session.set_rdf_valid_time(Some(unrelated_scope));
    session
        .execute_sparql("COPY GRAPH <http://ex.org/copy-src> TO GRAPH <http://ex.org/copy-dst>")
        .unwrap();
    session
        .execute_sparql("ADD GRAPH <http://ex.org/add-src> TO GRAPH <http://ex.org/add-dst>")
        .unwrap();
    session
        .execute_sparql("MOVE GRAPH <http://ex.org/move-src> TO GRAPH <http://ex.org/move-dst>")
        .unwrap();

    let assert_destinations = |database: &GrafeoDB| {
        for (graph_name, expected) in [
            ("http://ex.org/copy-dst", &copy),
            ("http://ex.org/add-dst", &add),
            ("http://ex.org/move-dst", &moved),
        ] {
            let graph = database
                .rdf_store()
                .graph(graph_name)
                .expect("destination graph");
            let (_, lives) = graph
                .quad_history()
                .into_iter()
                .find(|(candidate, _)| candidate.as_ref() == expected)
                .expect("destination quad history");
            assert_eq!(lives[0].valid, Some(source), "graph operation {graph_name}");
        }
        assert!(
            database
                .rdf_store()
                .graph("http://ex.org/move-src")
                .is_none()
        );
    };
    assert_destinations(&db);

    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &crash_copy);
    drop(session);
    std::mem::forget(db);
    let recovered = persistent_rdf(&crash_copy);
    assert_destinations(&recovered);
}

#[test]
fn sparql_empty_source_graph_operations_create_durable_destinations() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g3_empty_graph_ops.grafeo");
    let crash_copy = dir.path().join("g3_empty_graph_ops_crash.grafeo");
    let db = persistent_rdf(&path);
    let session = db.session();

    for source in ["copy-empty", "add-empty", "move-empty"] {
        session
            .execute_sparql(&format!("CREATE GRAPH <http://ex.org/{source}-src>"))
            .unwrap();
    }
    session
        .execute_sparql(
            "COPY GRAPH <http://ex.org/copy-empty-src> TO GRAPH <http://ex.org/copy-empty-dst>",
        )
        .unwrap();
    session
        .execute_sparql(
            "ADD GRAPH <http://ex.org/add-empty-src> TO GRAPH <http://ex.org/add-empty-dst>",
        )
        .unwrap();
    session
        .execute_sparql(
            "MOVE GRAPH <http://ex.org/move-empty-src> TO GRAPH <http://ex.org/move-empty-dst>",
        )
        .unwrap();

    let assert_lifecycles = |database: &GrafeoDB| {
        for destination in ["copy-empty-dst", "add-empty-dst", "move-empty-dst"] {
            let graph = database
                .rdf_store()
                .graph(&format!("http://ex.org/{destination}"))
                .unwrap_or_else(|| panic!("missing empty destination {destination}"));
            assert!(
                graph.is_empty(),
                "destination {destination} must stay empty"
            );
        }
        for retained_source in ["copy-empty-src", "add-empty-src"] {
            assert!(
                database
                    .rdf_store()
                    .graph(&format!("http://ex.org/{retained_source}"))
                    .is_some(),
                "{retained_source} must survive its graph operation"
            );
        }
        assert!(
            database
                .rdf_store()
                .graph("http://ex.org/move-empty-src")
                .is_none(),
            "MOVE must drop its empty source"
        );
    };
    assert_lifecycles(&db);

    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &crash_copy);
    drop(session);
    std::mem::forget(db);
    let recovered = persistent_rdf(&crash_copy);
    assert_lifecycles(&recovered);
}

#[cfg(feature = "lpg")]
#[test]
fn portable_snapshot_round_trips_submicrosecond_i128_valid_time() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let from = i128::from(i64::MAX) * 1_000 + 333;
    let interval = ValidTimeInterval::from_tai_nanoseconds(from, from + 1).unwrap();
    let statement = triple_with_subject("http://ex.org/portable-tai");
    let session = db.session();
    session.set_rdf_valid_time(Some(interval));
    session
        .insert_rdf_quads([Quad::named(
            statement.clone(),
            "http://ex.org/portable-claims",
        )])
        .unwrap();

    let snapshot = db.export_snapshot().unwrap();
    assert_eq!(grafeo_engine::snapshot_info(&snapshot).unwrap().version, 12);
    let restored = GrafeoDB::import_snapshot(&snapshot).unwrap();
    let graph = restored
        .rdf_store()
        .graph("http://ex.org/portable-claims")
        .expect("portable named graph");
    let (_, lives) = graph
        .quad_history()
        .into_iter()
        .find(|(candidate, _)| candidate.as_ref() == &statement)
        .expect("portable quad history");
    assert_eq!(lives[0].valid, Some(interval));
}

#[cfg(all(feature = "lpg", feature = "gql"))]
#[test]
fn both_model_mixed_commit_wal_only_keeps_rdf_epoch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g3_both_crash.grafeo");
    let crash_copy = dir.path().join("g3_both_crash_copy.grafeo");
    let db = persistent_model(&path, GraphModel::Both);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Anchor {name: 'mixed'})").unwrap();
    session
        .execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "v" . }"#)
        .unwrap();
    let epoch = session.commit().unwrap();
    let expected_cut = db.rdf_history_cut(epoch).unwrap();
    assert_exact_cut(&db, &expected_cut, epoch, &[(Quad::new(triple()), None)]);
    assert!(expected_cut.named_graphs.is_empty());
    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &crash_copy);
    std::mem::forget(session);
    std::mem::forget(db); // simulate process loss: no close/checkpoint

    let recovered = persistent_model(&crash_copy, GraphModel::Both);
    assert_eq!(recovered.rdf_history_cut(epoch).unwrap(), expected_cut);
    let nodes = recovered
        .execute("MATCH (n:Anchor) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(nodes, 1, "mixed LPG half must recover with the RDF half");
}
