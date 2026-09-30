//! Mixed-model compact checkpoint persistence regressions.
//!
//! A compacted database writes a `CompactStore` base plus an LPG overlay.
//! Those LPG-specific sections must not exclude the catalog, search indexes,
//! RDF dataset/history, or Ring from the same whole-container snapshot.
//!
//! ```text
//! cargo test -p grafeo-engine \
//!   --features "compact-store,triple-store,sparql,ring-index,vector-index,text-index,testing-crash-injection" \
//!   --test mixed_compact_persistence -- --test-threads=1
//! ```

#![cfg(all(
    feature = "compact-store",
    feature = "lpg",
    feature = "gql",
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file"
))]

use grafeo_common::storage::SectionType;
use grafeo_common::types::{GraphIncarnationId, HistoryCompleteness, Value};
use grafeo_core::graph::rdf::{Quad, RdfHistoricalQuad, Term, Triple};
use grafeo_engine::{
    Config, DurabilityMode, GrafeoDB, GraphModel, TaiNanoseconds, ValidTimeInterval,
};

fn both_sync(path: &std::path::Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .expect("open GraphModel::Both")
}

fn triple(subject: &str, object: &str) -> Triple {
    Triple::new(
        Term::iri(subject),
        Term::iri("http://ex.org/p"),
        Term::literal(object),
    )
}

fn gql_count(db: &GrafeoDB, label: &str) -> i64 {
    db.session()
        .execute(&format!("MATCH (n:{label}) RETURN count(n)"))
        .expect("GQL count")
        .rows()[0][0]
        .as_int64()
        .expect("integer count")
}

fn sparql_count(db: &GrafeoDB, query: &str) -> usize {
    db.execute_sparql(query).expect("SPARQL count").row_count()
}

fn assert_container_sections(db: &GrafeoDB) {
    let directory = db
        .file_manager()
        .expect("single-file database")
        .read_section_directory()
        .expect("read section directory")
        .expect("v2 section directory");

    for section in [
        SectionType::CompactStore,
        SectionType::LpgStore,
        SectionType::OverlayDeletions,
        SectionType::Catalog,
        SectionType::RdfStore,
    ] {
        assert!(
            directory.find(section).is_some(),
            "compacted Both checkpoint must contain {section:?}"
        );
    }

    #[cfg(feature = "ring-index")]
    assert!(
        directory.find(SectionType::RdfRing).is_some(),
        "compacted Both checkpoint must contain the RDF Ring"
    );
    #[cfg(feature = "vector-index")]
    assert!(
        directory.find(SectionType::VectorStore).is_some(),
        "compacted Both checkpoint must contain vector index data"
    );
    #[cfg(feature = "text-index")]
    assert!(
        directory.find(SectionType::TextIndex).is_some(),
        "compacted Both checkpoint must contain text index data"
    );
}

#[test]
fn both_compact_close_reopen_preserves_every_section_and_rdf_history() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("both_compact.grafeo");

    let history = triple("http://ex.org/history", "old");
    let valid = triple("http://ex.org/valid", "windowed");
    let live_default = triple("http://ex.org/default", "live");
    let live_named = triple("http://ex.org/named-subject", "named-live");
    let post_compact_default = triple("http://ex.org/post-compact", "default-after");
    let post_compact_named = triple("http://ex.org/post-compact-named", "named-after");
    let history_insert_epoch;
    let history_delete_epoch;
    let history_insert_cut;
    let history_delete_cut;
    let valid_cut;
    let outside_valid_cut;

    {
        let mut db = both_sync(&path);
        db.session()
            .execute("CREATE NODE TYPE Person (name STRING)")
            .expect("create catalog entry");

        db.create_node_with_props(&["Base"], [("name", Value::from("kept"))]);
        let deleted = db.create_node_with_props(&["Deleted"], [("name", Value::from("removed"))]);
        let deleted_via_db =
            db.create_node_with_props(&["DeletedViaDb"], [("name", Value::from("removed-too"))]);
        let rolled_back =
            db.create_node_with_props(&["RolledBack"], [("name", Value::from("restored"))]);
        assert_ne!(deleted, grafeo_common::types::NodeId::INVALID);
        assert_ne!(deleted_via_db, grafeo_common::types::NodeId::INVALID);
        assert_ne!(rolled_back, grafeo_common::types::NodeId::INVALID);
        assert_eq!(gql_count(&db, "Deleted"), 1);
        assert_eq!(gql_count(&db, "DeletedViaDb"), 1);

        db.execute_sparql(r#"INSERT DATA { <http://ex.org/history> <http://ex.org/p> "old" }"#)
            .unwrap();
        history_insert_epoch = db.rdf_store_commit_epoch();
        db.execute_sparql(r#"DELETE DATA { <http://ex.org/history> <http://ex.org/p> "old" }"#)
            .unwrap();
        history_delete_epoch = db.rdf_store_commit_epoch();

        db.insert_rdf_valid([valid.clone()], 1_000, 2_000)
            .expect("valid-time insert");
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/default> <http://ex.org/p> "live" .
                GRAPH <http://ex.org/named> {
                    <http://ex.org/named-subject> <http://ex.org/p> "named-live" .
                }
            }"#,
        )
        .unwrap();

        let tm_epoch_before_compact = db.session().transaction_manager_ref().current_epoch();
        assert_eq!(
            db.current_epoch(),
            tm_epoch_before_compact,
            "RDF-only commits must keep the LPG epoch aligned before compact"
        );

        db.compact().expect("compact LPG tier");
        assert!(
            db.get_node(deleted).is_some(),
            "compact must retain base node {deleted:?} (LPG epoch {:?}, TM epoch {:?}, RDF epoch {:?})",
            db.current_epoch(),
            db.session().transaction_manager_ref().current_epoch(),
            db.rdf_store_commit_epoch()
        );
        let post_compact_epoch = db.session().transaction_manager_ref().current_epoch();
        assert!(
            db.get_node_at_epoch(deleted, post_compact_epoch).is_some(),
            "compact-base node must be visible at the current transaction epoch {post_compact_epoch:?}"
        );
        db.create_node_with_props(
            &["Overlay"],
            [
                ("name", Value::from("after-compact")),
                ("embedding", Value::Vector(vec![1.0, 0.0, 0.0].into())),
                ("body", Value::from("compact checkpoints keep indexes")),
            ],
        );
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/post-compact> <http://ex.org/p> "default-after" .
                GRAPH <http://ex.org/post-compact-graph> {
                    <http://ex.org/post-compact-named> <http://ex.org/p> "named-after" .
                }
            }"#,
        )
        .expect("post-compact SPARQL uses the canonical RDF store and WAL");
        assert_eq!(
            gql_count(&db, "Deleted"),
            1,
            "base node exists after compact"
        );
        let mut delete_session = db.session();
        assert!(
            delete_session.get_node(deleted).is_some(),
            "layered session can read the compact-base node before deleting it"
        );
        delete_session.begin_transaction().unwrap();
        assert!(
            delete_session.delete_node(deleted),
            "base deletion must create an overlay tombstone"
        );
        delete_session
            .commit()
            .expect("publish compact-base tombstone");
        drop(delete_session);
        assert!(
            db.delete_node(deleted_via_db),
            "GrafeoDB convenience API must publish a compact-base tombstone"
        );
        let mut rollback_session = db.session();
        rollback_session.begin_transaction().unwrap();
        assert!(
            rollback_session.delete_node(rolled_back),
            "transactional compact-base delete must be visible to rollback"
        );
        rollback_session
            .rollback()
            .expect("rollback compact-base tombstone");
        drop(rollback_session);
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: None,
            property: "name".into(),
            kind: grafeo_engine::IndexCreateKind::Property,
        })
        .expect("create property index");

        #[cfg(feature = "vector-index")]
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Overlay".into()),
            property: "embedding".into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(3),
                metric: Some("cosine".into()),
                m: None,
                ef_construction: None,
                ef: None,
                quantization: None,
            },
        })
        .expect("create vector index in overlay");
        #[cfg(feature = "text-index")]
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Overlay".into()),
            property: "body".into(),
            kind: grafeo_engine::IndexCreateKind::Text {
                min_token_length: None,
            },
        })
        .expect("create text index in overlay");
        #[cfg(feature = "ring-index")]
        {
            db.rdf_store().rebuild_ring();
            assert!(db.rdf_store().ring().is_some(), "Ring built before close");
        }

        assert_eq!(gql_count(&db, "Base"), 1);
        assert_eq!(gql_count(&db, "Overlay"), 1);
        assert_eq!(gql_count(&db, "Deleted"), 0);
        assert_eq!(gql_count(&db, "DeletedViaDb"), 0);
        assert_eq!(gql_count(&db, "RolledBack"), 1);
        assert!(
            db.rdf_store().contains(&live_default),
            "SPARQL update publishes its default-graph triple before close"
        );
        assert!(
            db.rdf_store()
                .graph("http://ex.org/named")
                .is_some_and(|graph| graph.contains(&live_named)),
            "SPARQL update publishes its named-graph triple before close"
        );
        assert!(
            db.rdf_store().contains(&post_compact_default),
            "post-compact default-graph update reaches the canonical RDF store"
        );
        assert!(
            db.rdf_store()
                .graph("http://ex.org/post-compact-graph")
                .is_some_and(|graph| graph.contains(&post_compact_named)),
            "post-compact named-graph update reaches the canonical RDF store"
        );
        assert_eq!(
            sparql_count(
                &db,
                "SELECT ?o WHERE { <http://ex.org/default> <http://ex.org/p> ?o }"
            ),
            1,
            "default RDF graph exists before close"
        );
        assert_eq!(
            sparql_count(
                &db,
                "SELECT ?o WHERE { GRAPH <http://ex.org/named> { \
                 <http://ex.org/named-subject> <http://ex.org/p> ?o } }"
            ),
            1,
            "named RDF graph exists before close"
        );
        history_insert_cut = db.rdf_history_cut(history_insert_epoch).unwrap();
        history_delete_cut = db.rdf_history_cut(history_delete_epoch).unwrap();
        assert_eq!(
            history_insert_cut.completeness,
            HistoryCompleteness::Complete
        );
        assert_eq!(history_insert_cut.store_id, db.rdf_store().store_id());
        assert_eq!(history_insert_cut.epoch, history_insert_epoch);
        assert!(history_insert_cut.named_graphs.is_empty());
        let historical_quad = Quad::new(history.clone());
        assert_eq!(
            history_insert_cut.quads,
            vec![RdfHistoricalQuad {
                statement: db
                    .rdf_statement_handle(&historical_quad, GraphIncarnationId::DEFAULT_GRAPH)
                    .unwrap(),
                quad: historical_quad,
                graph_incarnation: GraphIncarnationId::DEFAULT_GRAPH,
                valid: None,
            }]
        );
        let mut expected_deleted = history_insert_cut.clone();
        expected_deleted.epoch = history_delete_epoch;
        expected_deleted.quads.clear();
        assert_eq!(history_delete_cut, expected_deleted);

        let epoch = db.rdf_store_commit_epoch();
        valid_cut = db
            .rdf_history_cut_at(epoch, Some(TaiNanoseconds::new(1_500_000)))
            .unwrap();
        outside_valid_cut = db
            .rdf_history_cut_at(epoch, Some(TaiNanoseconds::new(2_500_000)))
            .unwrap();
        assert_eq!(valid_cut.completeness, HistoryCompleteness::Complete);
        assert_eq!(valid_cut.store_id, history_insert_cut.store_id);
        assert_eq!(valid_cut.epoch, epoch);
        let valid_quad = Quad::new(valid.clone());
        let valid_row = valid_cut
            .quads
            .iter()
            .find(|row| row.quad == valid_quad)
            .expect("valid-time assertion survives compaction");
        assert_eq!(
            valid_row.graph_incarnation,
            GraphIncarnationId::DEFAULT_GRAPH
        );
        assert_eq!(
            valid_row.statement,
            db.rdf_statement_handle(&valid_quad, GraphIncarnationId::DEFAULT_GRAPH)
                .unwrap()
        );
        assert_eq!(
            valid_row.valid,
            Some(ValidTimeInterval::from_tai_nanoseconds(1_000_000, 2_000_000).unwrap())
        );
        for expected in [&live_default, &post_compact_default] {
            let quad = Quad::new(expected.clone());
            let row = valid_cut.quads.iter().find(|row| row.quad == quad).unwrap();
            assert_eq!(row.graph_incarnation, GraphIncarnationId::DEFAULT_GRAPH);
            assert_eq!(
                row.statement,
                db.rdf_statement_handle(&quad, GraphIncarnationId::DEFAULT_GRAPH)
                    .unwrap()
            );
            assert_eq!(row.valid, None);
        }
        for (name, expected) in [
            ("http://ex.org/named", &live_named),
            ("http://ex.org/post-compact-graph", &post_compact_named),
        ] {
            let incarnation = db.rdf_store().graph(name).unwrap().graph_incarnation();
            let identity = valid_cut
                .named_graphs
                .iter()
                .find(|graph| graph.name() == Some(name))
                .unwrap();
            assert_eq!(identity.incarnation(), incarnation);
            let quad = Quad::named(expected.clone(), name);
            let row = valid_cut.quads.iter().find(|row| row.quad == quad).unwrap();
            assert_eq!(row.graph_incarnation, incarnation);
            assert_eq!(
                row.statement,
                db.rdf_statement_handle(&quad, incarnation).unwrap()
            );
            assert_eq!(row.valid, None);
        }
        assert_eq!(valid_cut.named_graphs.len(), 2);
        assert_eq!(valid_cut.quads.len(), 5);
        let mut expected_outside = valid_cut.clone();
        expected_outside.quads.retain(|row| row.quad != valid_quad);
        assert_eq!(
            outside_valid_cut, expected_outside,
            "valid time removes only the windowed statement, retaining graph identities"
        );
        db.close().expect("close compacted Both database");
    }

    let db = GrafeoDB::open(&path).expect("reopen compacted Both database");
    assert_eq!(db.graph_model(), GraphModel::Both);
    assert_container_sections(&db);

    assert_eq!(gql_count(&db, "Base"), 1, "compact base survives");
    assert_eq!(gql_count(&db, "Overlay"), 1, "LPG overlay survives");
    assert_eq!(gql_count(&db, "Deleted"), 0, "base tombstone survives");
    assert_eq!(
        gql_count(&db, "DeletedViaDb"),
        0,
        "convenience-API base tombstone survives"
    );
    assert_eq!(
        gql_count(&db, "RolledBack"),
        1,
        "rolled-back base tombstone stays absent"
    );

    let node_types = db
        .session()
        .execute("SHOW NODE TYPES")
        .expect("catalog survives compact checkpoint");
    assert!(
        node_types
            .rows()
            .iter()
            .any(|row| row.first() == Some(&Value::from("Person"))),
        "catalog must retain Person after compact close/reopen"
    );

    assert_eq!(
        sparql_count(
            &db,
            "SELECT ?o WHERE { <http://ex.org/default> <http://ex.org/p> ?o }"
        ),
        1,
        "default RDF graph survives compact close/reopen"
    );
    assert_eq!(
        sparql_count(
            &db,
            "SELECT ?o WHERE { GRAPH <http://ex.org/named> { \
             <http://ex.org/named-subject> <http://ex.org/p> ?o } }"
        ),
        1,
        "named RDF graph survives compact close/reopen"
    );
    assert_eq!(
        sparql_count(
            &db,
            "SELECT ?o WHERE { <http://ex.org/post-compact> <http://ex.org/p> ?o }"
        ),
        1,
        "post-compact default RDF survives close/reopen"
    );
    assert_eq!(
        sparql_count(
            &db,
            "SELECT ?o WHERE { GRAPH <http://ex.org/post-compact-graph> { \
             <http://ex.org/post-compact-named> <http://ex.org/p> ?o } }"
        ),
        1,
        "post-compact named RDF survives close/reopen"
    );
    assert_eq!(
        db.rdf_history_cut(history_insert_epoch).unwrap(),
        history_insert_cut,
        "complete graph-qualified history at insert survives reopen"
    );
    assert_eq!(
        db.rdf_history_cut(history_delete_epoch).unwrap(),
        history_delete_cut,
        "complete graph-qualified history remains closed at delete"
    );
    assert_eq!(
        db.rdf_history_cut_at(valid_cut.epoch, Some(TaiNanoseconds::new(1_500_000)))
            .unwrap(),
        valid_cut,
        "valid-time quads, graph incarnations, handles and completeness survive reopen"
    );
    assert_eq!(
        db.rdf_history_cut_at(
            outside_valid_cut.epoch,
            Some(TaiNanoseconds::new(2_500_000))
        )
        .unwrap(),
        outside_valid_cut,
        "valid-time upper bound and unfiltered graph lifetimes survive reopen"
    );

    #[cfg(feature = "ring-index")]
    assert!(
        db.rdf_store().ring().is_some(),
        "persisted Ring is restored rather than silently omitted"
    );
}

#[cfg(feature = "testing-crash-injection")]
fn sidecar_wal_dir(path: &std::path::Path) -> std::path::PathBuf {
    let mut wal = path.as_os_str().to_owned();
    wal.push(".wal");
    std::path::PathBuf::from(wal)
}

#[cfg(feature = "testing-crash-injection")]
fn copy_tree(src: &std::path::Path, dst: &std::path::Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let target = dst.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[cfg(feature = "testing-crash-injection")]
fn snapshot_db(src: &std::path::Path, dst: &std::path::Path) {
    std::fs::copy(src, dst).unwrap();
    let wal = sidecar_wal_dir(src);
    if wal.exists() {
        copy_tree(&wal, &sidecar_wal_dir(dst));
    }
}

/// The first checkpoint retires its WAL, so a bad compact snapshot cannot be
/// rescued by the second generation's retained WAL after installation crashes.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn both_compact_last_good_snapshot_plus_wal_survives_interrupted_checkpoint() {
    use grafeo_common::testing::crash::{CrashResult, with_crash_named};

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("both_compact_crash.grafeo");
    let copy = dir.path().join("both_compact_crash_copy.grafeo");
    let mut db = both_sync(&path);

    db.create_node(&["GenerationOne"]);
    db.execute_sparql(
        r#"INSERT DATA {
            <http://ex.org/generation-one> <http://ex.org/p> "default-one" .
            GRAPH <http://ex.org/generation-one-graph> {
                <http://ex.org/named-one> <http://ex.org/p> "named-one" .
            }
        }"#,
    )
    .unwrap();
    db.compact().expect("compact first generation");
    db.wal_checkpoint()
        .expect("install compact first-generation snapshot and retire its WAL");

    db.create_node(&["GenerationTwo"]);
    db.execute_sparql(
        r#"INSERT DATA {
            <http://ex.org/generation-two> <http://ex.org/p> "default-two" .
            GRAPH <http://ex.org/generation-two-graph> {
                <http://ex.org/named-two> <http://ex.org/p> "named-two" .
            }
        }"#,
    )
    .unwrap();
    db.wal()
        .expect("WAL")
        .sync()
        .expect("second generation durable before checkpoint crash");

    let crashed = with_crash_named("write_sections:after_data", || {
        let _ = db.wal_checkpoint();
    });
    assert!(
        matches!(crashed, CrashResult::Crashed),
        "second checkpoint must crash before installing its temp container"
    );

    snapshot_db(&path, &copy);
    std::mem::forget(db);

    let db = both_sync(&copy);
    assert_eq!(gql_count(&db, "GenerationOne"), 1);
    assert_eq!(gql_count(&db, "GenerationTwo"), 1);
    assert_eq!(
        sparql_count(
            &db,
            "SELECT ?o WHERE { <http://ex.org/generation-one> <http://ex.org/p> ?o }"
        ),
        1,
        "last-good compact snapshot must contain first-generation RDF"
    );
    assert_eq!(
        sparql_count(
            &db,
            "SELECT ?o WHERE { <http://ex.org/generation-two> <http://ex.org/p> ?o }"
        ),
        1,
        "retained WAL must recover second-generation RDF"
    );
    assert_eq!(
        sparql_count(
            &db,
            "SELECT ?o WHERE { GRAPH <http://ex.org/generation-two-graph> { \
             <http://ex.org/named-two> <http://ex.org/p> ?o } }"
        ),
        1,
        "retained WAL must recover post-compact named RDF"
    );
    assert_eq!(
        sparql_count(
            &db,
            "SELECT ?o WHERE { GRAPH <http://ex.org/generation-one-graph> { \
             <http://ex.org/named-one> <http://ex.org/p> ?o } }"
        ),
        1,
        "last-good compact snapshot must contain first-generation named RDF"
    );
}
