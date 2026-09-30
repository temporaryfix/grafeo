//! WAL-backed public raw stores must not publish unframed mutations.
//!
//! ```text
//! cargo test -p grafeo-engine --features "lpg,gql,triple-store,sparql,wal,grafeo-file" --test raw_mutation_fail_closed -- --test-threads=1
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use grafeo_engine::{Config, DurabilityMode, GrafeoDB};

fn sidecar_wal_dir(path: &std::path::Path) -> std::path::PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".wal");
    std::path::PathBuf::from(p)
}

fn snapshot_db(src: &std::path::Path, dst: &std::path::Path) {
    std::fs::copy(src, dst).unwrap();
    let wal = sidecar_wal_dir(src);
    if wal.exists() {
        std::fs::create_dir_all(sidecar_wal_dir(dst)).unwrap();
        for e in std::fs::read_dir(&wal).unwrap() {
            let e = e.unwrap();
            let to = sidecar_wal_dir(dst).join(e.file_name());
            if e.path().is_dir() {
                continue;
            }
            std::fs::copy(e.path(), to).unwrap();
        }
    }
}

fn persistent(path: &std::path::Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path).with_wal_durability(DurabilityMode::Sync))
        .expect("open")
}

#[test]
fn builtin_graph_store_mutation_trait_is_not_reachable() {
    let in_memory = GrafeoDB::new_in_memory();
    assert!(
        in_memory.graph_store_mut().is_none(),
        "the built-in in-memory store must still route writes through Session"
    );

    let dir = tempfile::TempDir::new().unwrap();
    let durable = persistent(&dir.path().join("no_graph_store_mut.grafeo"));
    assert!(
        durable.graph_store_mut().is_none(),
        "a WAL-backed store must never expose GraphStoreMut"
    );
}

#[cfg(feature = "compact-store")]
#[test]
fn single_file_without_wal_still_seals_default_and_named_stores() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("no_wal_sealed.grafeo");
    let mut config = Config::persistent(&path);
    config.wal_enabled = false;
    let mut db = GrafeoDB::with_config(config.clone()).expect("open no-WAL single-file database");

    db.session().execute("INSERT (:Committed)").unwrap();
    db.create_graph("owned").unwrap();
    db.set_current_graph(Some("owned")).unwrap();
    db.session().execute("INSERT (:NamedCommitted)").unwrap();
    db.set_current_graph(None).unwrap();
    db.compact()
        .expect("compact no-WAL database before probing retained handles");

    assert!(db.graph_store_mut().is_none());
    assert!(
        !grafeo_engine::database::testing::root_lpg_store(&db)
            .create_node(&["Leaked"])
            .is_valid(),
        "single-file persistence must seal the default store even without WAL"
    );
    let named = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("owned")
        .expect("named graph exists");
    assert!(
        !named.create_node(&["NamedLeaked"]).is_valid(),
        "single-file persistence must recursively seal named stores"
    );

    db.wal_checkpoint().expect("publish no-WAL checkpoint");
    db.close().expect("close no-WAL database");
    drop(db);

    let reopened = GrafeoDB::with_config(config).expect("reopen no-WAL single-file database");
    let default_count = reopened
        .session()
        .execute("MATCH (n:Leaked) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(default_count, 0);
    reopened.set_current_graph(Some("owned")).unwrap();
    let named_count = reopened
        .session()
        .execute("MATCH (n:NamedLeaked) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(named_count, 0);
}

#[cfg(all(feature = "triple-store", feature = "sparql"))]
#[test]
fn single_file_without_wal_still_seals_default_and_named_rdf_stores() {
    use grafeo_core::graph::rdf::{Term, Triple};
    use grafeo_engine::GraphModel;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("no_wal_rdf_sealed.grafeo");
    let mut config = Config::persistent(&path).with_graph_model(GraphModel::Both);
    config.wal_enabled = false;
    let db = GrafeoDB::with_config(config.clone()).expect("open no-WAL RDF database");
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/kept> <http://ex.org/p> "v" }"#)
        .unwrap();
    db.execute_sparql("CREATE GRAPH <http://ex.org/owned>")
        .unwrap();

    let leaked = Triple::new(
        Term::iri("http://ex.org/leaked"),
        Term::iri("http://ex.org/p"),
        Term::literal("x"),
    );
    assert!(!db.rdf_store().insert(leaked.clone()));
    let named = db
        .rdf_store()
        .graph("http://ex.org/owned")
        .expect("named RDF graph exists");
    assert!(!named.insert(leaked));

    db.wal_checkpoint().expect("publish no-WAL RDF checkpoint");
    db.close().expect("close no-WAL RDF database");
    drop(db);

    let reopened = GrafeoDB::with_config(config).expect("reopen no-WAL RDF database");
    assert_eq!(
        reopened
            .execute_sparql("SELECT ?o WHERE { <http://ex.org/leaked> <http://ex.org/p> ?o }",)
            .unwrap()
            .row_count(),
        0
    );
    assert_eq!(
        reopened
            .execute_sparql(
                "SELECT ?o WHERE { GRAPH <http://ex.org/owned> { <http://ex.org/leaked> <http://ex.org/p> ?o } }",
            )
            .unwrap()
            .row_count(),
        0
    );
}

#[test]
fn downstream_foreign_authority_cannot_mutate_default_or_named_lpg_stores() {
    use grafeo_common::types::{EpochId, Value};
    use grafeo_core::graph::write_permit::{WriteAuthority, with_authority};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let dir = tempfile::TempDir::new().unwrap();
    let db = persistent(&dir.path().join("foreign_lpg.grafeo"));
    let default_id = db.create_node(&["Seed"]);
    db.set_node_property(default_id, "version", Value::from(1_i64))
        .expect("set node property");
    db.set_node_property(default_id, "version", Value::from(2_i64))
        .expect("set node property");
    db.create_graph("owned").unwrap();
    db.set_current_graph(Some("owned")).unwrap();
    let named_id = db.create_node(&["Seed"]);
    db.set_node_property(named_id, "version", Value::from(1_i64))
        .expect("set node property");
    db.set_node_property(named_id, "version", Value::from(2_i64))
        .expect("set node property");
    db.set_current_graph(None).unwrap();
    let named = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("owned")
        .unwrap();
    let default_history = grafeo_engine::database::testing::root_lpg_store(&db)
        .node_property_history_for_key(default_id, "version");
    let named_history = named.node_property_history_for_key(named_id, "version");
    let foreign = WriteAuthority::new();

    with_authority(&foreign, || {
        assert!(
            !grafeo_engine::database::testing::root_lpg_store(&db)
                .create_node(&["Foreign"])
                .is_valid()
        );
        assert!(!named.create_node(&["Foreign"]).is_valid());
        grafeo_engine::database::testing::root_lpg_store(&db).create_property_index("forged");
        named.create_property_index("forged");
        grafeo_engine::database::testing::root_lpg_store(&db).gc_versions(EpochId::new(u64::MAX));
        named.gc_versions(EpochId::new(u64::MAX));
    });
    assert!(!grafeo_engine::database::testing::root_lpg_store(&db).has_property_index("forged"));
    assert!(!named.has_property_index("forged"));
    assert_eq!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .node_property_history_for_key(default_id, "version"),
        default_history,
        "foreign authority must not prune default-graph MVCC history"
    );
    assert_eq!(
        named.node_property_history_for_key(named_id, "version"),
        named_history,
        "foreign authority must not prune named-graph MVCC history"
    );
    let panic = catch_unwind(AssertUnwindSafe(|| {
        with_authority(&foreign, || panic!("hostile downstream panic"));
    }));
    assert!(panic.is_err());
    assert!(
        !grafeo_engine::database::testing::root_lpg_store(&db)
            .create_node(&["AfterPanic"])
            .is_valid(),
        "a caught foreign-authority panic must not leave raw writes enabled"
    );
    assert!(!named.create_node(&["AfterPanic"]).is_valid());
}

#[test]
fn downstream_cannot_restore_transaction_structure_outside_the_session_protocol() {
    use std::sync::Arc;

    use grafeo_core::graph::GraphStoreMut;
    use grafeo_core::graph::lpg::LpgStore;
    use grafeo_core::graph::write_permit::{WriteAuthority, with_authority};

    for graph_name in [None, Some("owned")] {
        let dir = tempfile::TempDir::new().unwrap();
        let suffix = graph_name.unwrap_or("default");
        let path = dir
            .path()
            .join(format!("structural_restore_{suffix}.grafeo"));
        let copy = dir
            .path()
            .join(format!("structural_restore_{suffix}_copy.grafeo"));
        let db = persistent(&path);
        if let Some(name) = graph_name {
            assert!(db.create_graph(name).unwrap());
        }

        let mut session = db.session();
        if let Some(name) = graph_name {
            session
                .use_graph_path(
                    &grafeo_common::types::GraphPath::from_components(&[name])
                        .expect("literal graph path"),
                )
                .expect("select existing graph");
        }
        let survivor = session.create_node(&["Survivor"]);
        let doomed = session.create_node(&["Doomed"]);
        let edge_src = session.create_node(&["EdgeSource"]);
        let edge_dst = session.create_node(&["EdgeTarget"]);
        let doomed_edge = session.create_edge(edge_src, edge_dst, "DOOMED_EDGE");

        session.begin_transaction().unwrap();
        let transaction_id = session
            .active_transaction_id()
            .expect("explicit transaction has an id");
        let raw_store: Arc<LpgStore> = graph_name.map_or_else(
            || Arc::clone(grafeo_engine::database::testing::root_lpg_store(&db)),
            |name| {
                grafeo_engine::database::testing::root_lpg_store(&db)
                    .graph(name)
                    .expect("named graph exists")
            },
        );
        let before = GraphStoreMut::tx_structural_snapshot(raw_store.as_ref(), transaction_id);

        let created = session.create_node(&["Created"]);
        let created_edge = session.create_edge(survivor, created, "CREATED_EDGE");
        assert!(session.delete_edge(doomed_edge));
        assert!(session.delete_node(doomed));

        // The token is intentionally opaque, but the public trait method used
        // to accept it under a foreign authority and truncate the transaction's
        // pending create/delete queues after their WAL records already existed.
        // Committing that divergent queue state made the live cut disagree with
        // recovery. Only Session's database-scoped authority may restore it.
        let foreign = WriteAuthority::new();
        let restore = with_authority(&foreign, || {
            GraphStoreMut::tx_structural_restore(raw_store.as_ref(), transaction_id, before)
        });
        assert!(
            restore.is_err(),
            "foreign structural restore must fail closed for {suffix} graph"
        );

        session.commit().unwrap();
        assert!(raw_store.get_node(created).is_some());
        assert!(raw_store.get_edge(created_edge).is_some());
        assert!(raw_store.get_node(doomed).is_none());
        assert!(raw_store.get_edge(doomed_edge).is_none());

        db.wal().unwrap().sync().unwrap();
        snapshot_db(&path, &copy);
        drop((session, raw_store));
        std::mem::forget(db);

        let reopened = persistent(&copy);
        let reopened_store = graph_name.map_or_else(
            || Arc::clone(grafeo_engine::database::testing::root_lpg_store(&reopened)),
            |name| {
                grafeo_engine::database::testing::root_lpg_store(&reopened)
                    .graph(name)
                    .expect("named graph recovers")
            },
        );
        assert!(reopened_store.get_node(created).is_some());
        assert!(reopened_store.get_edge(created_edge).is_some());
        assert!(reopened_store.get_node(doomed).is_none());
        assert!(reopened_store.get_edge(doomed_edge).is_none());
    }
}

#[cfg(feature = "compact-store")]
#[test]
fn sealed_compact_savepoint_restore_preserves_live_and_recovered_state() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("compact_savepoint_restore.grafeo");
    let copy = dir.path().join("compact_savepoint_restore_copy.grafeo");
    let mut db = persistent(&path);

    let survivor = db.create_node(&["Survivor"]);
    let doomed = db.create_node(&["Doomed"]);
    let edge_src = db.create_node(&["EdgeSource"]);
    let edge_dst = db.create_node(&["EdgeTarget"]);
    let cold_edge = db.create_edge(edge_src, edge_dst, "COLD_EDGE");
    db.compact().expect("install a cold compact base");

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.savepoint("stable").unwrap();
    let discarded = session.create_node(&["Discarded"]);
    let discarded_edge = session.create_edge(survivor, discarded, "DISCARDED_EDGE");
    assert!(session.delete_edge(cold_edge));
    assert!(session.delete_node(doomed));

    session.rollback_to_savepoint("stable").unwrap();
    assert!(session.get_node(discarded).is_none());
    assert!(session.get_edge(discarded_edge).is_none());
    assert!(session.get_node(doomed).is_some());
    assert!(session.get_edge(cold_edge).is_some());
    session.commit().unwrap();

    assert!(db.get_node(discarded).is_none());
    assert!(db.get_edge(discarded_edge).is_none());
    assert!(db.get_node(doomed).is_some());
    assert!(db.get_edge(cold_edge).is_some());
    db.wal().unwrap().sync().unwrap();
    snapshot_db(&path, &copy);
    drop(session);
    std::mem::forget(db);

    let reopened = persistent(&copy);
    assert!(reopened.get_node(discarded).is_none());
    assert!(reopened.get_edge(discarded_edge).is_none());
    assert!(reopened.get_node(doomed).is_some());
    assert!(reopened.get_edge(cold_edge).is_some());
}

#[cfg(feature = "triple-store")]
#[test]
fn downstream_foreign_authority_cannot_mutate_default_or_named_rdf_stores() {
    use grafeo_core::graph::rdf::{Term, Triple};
    use grafeo_core::graph::write_permit::{WriteAuthority, with_authority};
    use grafeo_engine::GraphModel;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let dir = tempfile::TempDir::new().unwrap();
    let db = GrafeoDB::with_config(
        Config::persistent(dir.path().join("foreign_rdf.grafeo"))
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .unwrap();
    db.session()
        .execute_sparql("CREATE GRAPH <http://ex.org/owned>")
        .unwrap();
    let named = db.rdf_store().graph("http://ex.org/owned").unwrap();
    let candidate = Triple::new(
        Term::iri("http://ex.org/foreign"),
        Term::iri("http://ex.org/p"),
        Term::literal("x"),
    );
    let foreign = WriteAuthority::new();

    with_authority(&foreign, || {
        assert!(!db.rdf_store().insert(candidate.clone()));
        assert!(!named.insert(candidate.clone()));
    });
    let panic = catch_unwind(AssertUnwindSafe(|| {
        with_authority(&foreign, || panic!("hostile downstream panic"));
    }));
    assert!(panic.is_err());
    assert!(!db.rdf_store().insert(candidate.clone()));
    assert!(!named.insert(candidate));
}

#[test]
fn raw_lpg_store_create_does_not_survive_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("raw_lpg.grafeo");
    let copy = dir.path().join("raw_lpg_copy.grafeo");
    let db = persistent(&path);
    db.session()
        .execute("INSERT (:Committed {name: 'ok'})")
        .unwrap();
    let id = grafeo_engine::database::testing::root_lpg_store(&db).create_node(&["Leaked"]);
    assert!(
        !id.is_valid(),
        "unframed store().create_node on WAL must not allocate a live node"
    );
    db.wal().unwrap().sync().unwrap();
    snapshot_db(&path, &copy);
    std::mem::forget(db);
    let db = persistent(&copy);
    let n = db
        .session()
        .execute("MATCH (n:Leaked) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(n, 0, "raw LPG mutation must be absent after reopen");
}

#[test]
fn transaction_manager_handle_is_observation_only_and_reopen_state_is_unchanged() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("transaction_view.grafeo");
    let copy = dir.path().join("transaction_view_copy.grafeo");
    let db = persistent(&path);

    db.session()
        .execute("INSERT (:Committed {name: 'transaction-view'})")
        .unwrap();
    let session = db.session();
    let view = session.transaction_manager_ref();
    let epoch = view.current_epoch();
    let last_tid = view
        .last_assigned_transaction_id()
        .expect("the committed insert assigned a transaction id");
    assert_eq!(view.active_count(), 0);

    // Compile-fail API tests on TransactionManagerView prove begin/commit,
    // sync_epoch, transaction-id advancement, and publication().write() are
    // unreachable. This runtime half proves mere observation changes nothing.
    assert_eq!(session.transaction_manager_ref().current_epoch(), epoch);
    assert_eq!(
        session
            .transaction_manager_ref()
            .last_assigned_transaction_id(),
        Some(last_tid)
    );

    db.wal().unwrap().sync().unwrap();
    snapshot_db(&path, &copy);
    drop(session);
    std::mem::forget(db);

    let reopened = persistent(&copy);
    let count = reopened
        .session()
        .execute("MATCH (n:Committed {name: 'transaction-view'}) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(count, 1);
    assert!(
        reopened.session().transaction_manager_ref().current_epoch() >= epoch,
        "recovery may advance but must never regress the durable epoch"
    );
}

#[cfg(all(feature = "vector-index", feature = "text-index"))]
#[test]
fn public_index_handles_and_alias_injection_are_fail_closed_in_default_and_named_graphs() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::Arc;

    use grafeo_common::types::Value;
    use grafeo_core::graph::write_permit::{WriteAuthority, with_authority};
    use grafeo_core::index::text::{BM25Config, InvertedIndex};
    use grafeo_core::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
    use parking_lot::RwLock;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("index_views.grafeo");
    let copy = dir.path().join("index_views_copy.grafeo");
    let db = persistent(&path);

    let default_id = db.create_node_with_props(
        &["Doc"],
        [
            ("embedding", Value::Vector(vec![1.0, 0.0].into())),
            ("body", Value::from("default durable text")),
        ],
    );
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
        property: "embedding".into(),
        kind: grafeo_engine::IndexCreateKind::Vector {
            dimensions: Some(2),
            metric: Some("cosine".into()),
            m: None,
            ef_construction: None,
            ef: None,
            quantization: None,
        },
    })
    .unwrap();
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
        property: "body".into(),
        kind: grafeo_engine::IndexCreateKind::Text {
            min_token_length: None,
        },
    })
    .unwrap();

    db.create_graph("named").unwrap();
    db.set_current_graph(Some("named")).unwrap();
    let named_id = db.create_node_with_props(
        &["Note"],
        [
            ("embedding", Value::Vector(vec![0.0, 1.0].into())),
            ("body", Value::from("named durable text")),
        ],
    );
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: grafeo_common::types::GraphPath::from_components(&["named"])
            .expect("exact named graph path"),
        name: None,
        label: Some("Note".into()),
        property: "embedding".into(),
        kind: grafeo_engine::IndexCreateKind::Vector {
            dimensions: Some(2),
            metric: Some("cosine".into()),
            m: None,
            ef_construction: None,
            ef: None,
            quantization: None,
        },
    })
    .unwrap();
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: grafeo_common::types::GraphPath::from_components(&["named"])
            .expect("exact named graph path"),
        name: None,
        label: Some("Note".into()),
        property: "body".into(),
        kind: grafeo_engine::IndexCreateKind::Text {
            min_token_length: None,
        },
    })
    .unwrap();
    db.set_current_graph(None).unwrap();

    let named = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("named")
        .expect("named graph exists");
    let default_vector = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_vector_index("Doc", "embedding")
        .expect("default vector view");
    let default_text = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_text_index("Doc", "body")
        .expect("default text view");
    let named_vector = named
        .get_vector_index("Note", "embedding")
        .expect("named vector view");
    let named_text = named
        .get_text_index("Note", "body")
        .expect("named text view");
    assert!(default_vector.contains(default_id));
    assert!(default_text.read().contains(default_id));
    assert!(named_vector.contains(named_id));
    assert!(named_text.read().contains(named_id));

    // A downstream-created alias cannot be injected into either sealed store,
    // even while an unrelated authority is held.
    let foreign = WriteAuthority::new();
    let forged_vector = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
        2,
        DistanceMetric::Cosine,
    ))));
    let forged_text = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    with_authority(&foreign, || {
        grafeo_engine::database::testing::root_lpg_store(&db).add_vector_index(
            "Forged",
            "embedding",
            Arc::clone(&forged_vector),
        );
        named.add_text_index("Forged", "body", Arc::clone(&forged_text));
    });
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .get_vector_index("Forged", "embedding")
            .is_none()
    );
    assert!(named.get_text_index("Forged", "body").is_none());

    let panic = catch_unwind(AssertUnwindSafe(|| {
        with_authority(&foreign, || panic!("caught hostile index panic"));
    }));
    assert!(panic.is_err());
    grafeo_engine::database::testing::root_lpg_store(&db).add_vector_index(
        "AfterPanic",
        "embedding",
        forged_vector,
    );
    named.add_text_index("AfterPanic", "body", forged_text);
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .get_vector_index("AfterPanic", "embedding")
            .is_none()
    );
    assert!(named.get_text_index("AfterPanic", "body").is_none());

    drop((
        default_vector,
        default_text,
        named_vector,
        named_text,
        named,
    ));
    db.wal().unwrap().sync().unwrap();
    snapshot_db(&path, &copy);
    std::mem::forget(db);

    let reopened = persistent(&copy);
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&reopened)
            .get_vector_index("Forged", "embedding")
            .is_none()
    );
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&reopened)
            .get_vector_index("Doc", "embedding")
            .expect("default vector index recovered")
            .contains(default_id)
    );
    let named = grafeo_engine::database::testing::root_lpg_store(&reopened)
        .graph("named")
        .expect("named graph recovered");
    assert!(named.get_text_index("Forged", "body").is_none());
    assert!(
        named
            .get_text_index("Note", "body")
            .expect("named text index recovered")
            .read()
            .contains(named_id)
    );
}

#[cfg(feature = "compact-store")]
#[test]
fn compact_and_file_views_cannot_publish_tombstones_or_replace_storage() {
    use grafeo_engine::GraphStore;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("read_only_storage_views.grafeo");
    let copy = dir.path().join("read_only_storage_views_copy.grafeo");
    let mut db = persistent(&path);

    let kept = db.create_node(&["Kept"]);
    let deleted = db.create_node(&["Deleted"]);
    db.compact().expect("install compact base");
    assert!(db.delete_node(deleted));

    let layered = db.layered_store().expect("read-only layered view");
    assert_eq!(
        layered.graph_store().get_node(kept).map(|n| n.id),
        Some(kept)
    );
    assert!(layered.graph_store().get_node(deleted).is_none());
    assert!(
        layered
            .deleted_nodes()
            .iter()
            .any(|(node, _)| *node == deleted),
        "the committed base tombstone is observable"
    );
    let base = layered.base_store();
    assert!(
        base.node_count() >= 2,
        "cold base remains immutable and complete"
    );

    #[cfg(feature = "mmap")]
    {
        let tier = db.compact_tiered().expect("read-only compact tier view");
        assert_eq!(tier.store().node_count(), base.node_count());
        assert!(tier.memory_bytes() > 0);
    }

    let file = db.file_manager().expect("read-only database-file view");
    assert_eq!(file.path(), path.canonicalize().unwrap().as_path());
    assert!(file.file_size().unwrap() > 0);
    let _ = file.active_header();

    // Compile-fail API tests on the three view types prove reset_overlay,
    // seed_deleted_from_base, swap_base, tier persistence/reload, snapshot
    // writes, WAL removal, and close are absent from downstream capabilities.
    drop((file, layered, base));
    db.wal().unwrap().sync().unwrap();
    snapshot_db(&path, &copy);
    std::mem::forget(db);

    let reopened = persistent(&copy);
    assert!(reopened.get_node(kept).is_some());
    assert!(reopened.get_node(deleted).is_none());
}

#[cfg(feature = "triple-store")]
#[test]
fn raw_rdf_store_insert_does_not_survive_reopen() {
    use grafeo_core::graph::rdf::{Term, Triple};
    use grafeo_engine::GraphModel;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("raw_rdf.grafeo");
    let copy = dir.path().join("raw_rdf_copy.grafeo");
    let db = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .unwrap();
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/ok> <http://ex.org/p> "v" }"#)
        .unwrap();
    let leaked = Triple::new(
        Term::iri("http://ex.org/leaked"),
        Term::iri("http://ex.org/p"),
        Term::literal("x"),
    );
    assert!(
        !db.rdf_store().insert(leaked),
        "unframed rdf_store().insert on WAL must not publish"
    );
    db.wal().unwrap().sync().unwrap();
    snapshot_db(&path, &copy);
    std::mem::forget(db);
    let db = GrafeoDB::with_config(
        Config::persistent(&copy)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .unwrap();
    let n = db
        .execute_sparql("SELECT ?o WHERE { <http://ex.org/leaked> <http://ex.org/p> ?o }")
        .unwrap()
        .row_count();
    assert_eq!(n, 0, "raw RDF insert must be absent after reopen");
}

#[test]
fn restore_snapshot_refused_on_wal_database() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("restore.grafeo");
    let db = persistent(&path);
    db.session()
        .execute("INSERT (:Committed {name: 'ok'})")
        .unwrap();
    let err = db.restore_snapshot(&[0u8; 8]);
    assert!(
        err.is_err(),
        "restore_snapshot on WAL-backed DB must fail closed, got {err:?}"
    );
}

#[test]
fn raw_lpg_store_create_edge_does_not_survive_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("raw_edge.grafeo");
    let copy = dir.path().join("raw_edge_copy.grafeo");
    let db = persistent(&path);
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    assert!(a.is_valid() && b.is_valid());
    let eid = grafeo_engine::database::testing::root_lpg_store(&db).create_edge(a, b, "LEAKED");
    assert!(
        !eid.is_valid(),
        "unframed store().create_edge on WAL must not allocate a live edge"
    );
    db.wal().unwrap().sync().unwrap();
    snapshot_db(&path, &copy);
    std::mem::forget(db);
    let db = persistent(&copy);
    let n = db
        .session()
        .execute("MATCH ()-[r:LEAKED]->() RETURN count(r)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(n, 0, "raw LPG create_edge must be absent after reopen");
}

#[test]
fn raw_lpg_store_set_node_property_does_not_survive_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("raw_prop.grafeo");
    let copy = dir.path().join("raw_prop_copy.grafeo");
    let db = persistent(&path);
    let a = db.create_node(&["Person"]);
    grafeo_engine::database::testing::root_lpg_store(&db).set_node_property(
        a,
        "leaked",
        grafeo_common::types::Value::from("x"),
    );
    let live = db
        .session()
        .execute("MATCH (n:Person) WHERE n.leaked IS NOT NULL RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(
        live, 0,
        "unframed set_node_property must not publish on a WAL database"
    );
    db.wal().unwrap().sync().unwrap();
    snapshot_db(&path, &copy);
    std::mem::forget(db);
    let db = persistent(&copy);
    let n = db
        .session()
        .execute("MATCH (n:Person) WHERE n.leaked IS NOT NULL RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(n, 0, "raw set_node_property must be absent after reopen");
}

#[cfg(feature = "triple-store")]
#[test]
fn raw_rdf_store_remove_does_not_survive_reopen() {
    use grafeo_core::graph::rdf::{Term, Triple};
    use grafeo_engine::GraphModel;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("raw_rdf_del.grafeo");
    let copy = dir.path().join("raw_rdf_del_copy.grafeo");
    let db = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .unwrap();
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/keep> <http://ex.org/p> "v" }"#)
        .unwrap();
    let keep = Triple::new(
        Term::iri("http://ex.org/keep"),
        Term::iri("http://ex.org/p"),
        Term::literal("v"),
    );
    assert!(
        !db.rdf_store().remove(&keep),
        "unframed rdf_store().remove on WAL must not publish a delete"
    );
    db.wal().unwrap().sync().unwrap();
    snapshot_db(&path, &copy);
    std::mem::forget(db);
    let db = GrafeoDB::with_config(
        Config::persistent(&copy)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .unwrap();
    let n = db
        .execute_sparql("SELECT ?o WHERE { <http://ex.org/keep> <http://ex.org/p> ?o }")
        .unwrap()
        .row_count();
    assert_eq!(
        n, 1,
        "committed RDF triple must survive unframed remove after reopen"
    );
}

#[test]
fn batch_create_nodes_are_framed_and_recover_in_default_and_named_graphs() {
    use std::collections::HashMap;

    use grafeo_common::types::{PropertyKey, Value};

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("batch.grafeo");
    let copy = dir.path().join("batch_copy.grafeo");
    let db = persistent(&path);

    let default_ids = db.batch_create_nodes(
        "DefaultDoc",
        "embedding",
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
    );
    assert!(
        default_ids.iter().all(|id| id.is_valid()),
        "framed default-graph batch must return live ids, got {default_ids:?}"
    );

    db.create_graph("batch_named").unwrap();
    db.set_current_graph(Some("batch_named")).unwrap();
    let named_ids = db.batch_create_nodes_with_props(
        "NamedDoc",
        vec![HashMap::from([
            (PropertyKey::new("title"), Value::from("named")),
            (
                PropertyKey::new("embedding"),
                Value::Vector(vec![0.5, 0.5].into()),
            ),
        ])],
    );
    assert!(
        named_ids.iter().all(|id| id.is_valid()),
        "framed named-graph batch must return live ids, got {named_ids:?}"
    );
    db.set_current_graph(None).unwrap();

    db.wal().unwrap().sync().unwrap();
    snapshot_db(&path, &copy);
    std::mem::forget(db);
    let db = persistent(&copy);

    let default = db.session();
    let default_count = default
        .execute("MATCH (n:DefaultDoc) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(default_count, 2);
    assert_eq!(
        default.get_node_property(default_ids[0], "embedding"),
        Some(Value::Vector(vec![1.0, 0.0].into())),
        "default-graph vector property must recover from retained WAL"
    );

    let named = db.session();
    named
        .use_graph_path(
            &grafeo_common::types::GraphPath::from_components(&["batch_named"])
                .expect("literal graph path"),
        )
        .expect("select existing graph");
    let named_count = named
        .execute("MATCH (n:NamedDoc) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(named_count, 1);
    assert_eq!(
        named.get_node_property(named_ids[0], "title"),
        Some(Value::from("named")),
        "named-graph property map must recover from retained WAL"
    );
    assert_eq!(
        default
            .execute("MATCH (n:NamedDoc) RETURN count(n)")
            .unwrap()
            .rows()[0][0]
            .as_int64(),
        Some(0),
        "named batch must not leak into the default graph"
    );
}

#[test]
fn import_tsv_str_is_framed_and_recovers_in_default_and_named_graphs() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("import.grafeo");
    let copy = dir.path().join("import_copy.grafeo");
    let db = persistent(&path);

    assert_eq!(
        db.import_tsv_str("1 2\n", "DEFAULT_E", true).unwrap(),
        (2, 1)
    );
    db.create_graph("import_named").unwrap();
    db.set_current_graph(Some("import_named")).unwrap();
    assert_eq!(
        db.import_tsv_str("3 4\n4 5\n", "NAMED_E", false).unwrap(),
        (3, 4)
    );
    db.set_current_graph(None).unwrap();

    db.wal().unwrap().sync().unwrap();
    snapshot_db(&path, &copy);
    std::mem::forget(db);
    let db = persistent(&copy);

    let default = db.session();
    let default_nodes = default
        .execute("MATCH (n:_Imported) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    let default_edges = default
        .execute("MATCH ()-[r:DEFAULT_E]->() RETURN count(r)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(default_nodes, 2);
    assert_eq!(default_edges, 1);

    let named = db.session();
    named
        .use_graph_path(
            &grafeo_common::types::GraphPath::from_components(&["import_named"])
                .expect("literal graph path"),
        )
        .expect("select existing graph");
    let named_nodes = named
        .execute("MATCH (n:_Imported) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    let named_edges = named
        .execute("MATCH ()-[r:NAMED_E]->() RETURN count(r)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(named_nodes, 3);
    assert_eq!(named_edges, 4);
    assert_eq!(
        default
            .execute("MATCH ()-[r:NAMED_E]->() RETURN count(r)")
            .unwrap()
            .rows()[0][0]
            .as_int64(),
        Some(0),
        "named import must not leak into the default graph"
    );
}

#[test]
fn raw_lpg_graph_or_create_does_not_survive_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("raw_named.grafeo");
    let copy = dir.path().join("raw_named_copy.grafeo");
    let db = persistent(&path);
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .graph_or_create("leaked")
            .is_err(),
        "unframed graph creation must reject before returning a writable store"
    );
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .graph("leaked")
            .is_none()
    );
    db.wal().unwrap().sync().unwrap();
    snapshot_db(&path, &copy);
    std::mem::forget(db);
    let db = persistent(&copy);
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .graph("leaked")
            .is_none(),
        "unframed graph_or_create must not leave a named graph after reopen"
    );
}

#[test]
fn raw_lpg_named_graph_after_wal_replay_stays_sealed() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("named_replay.grafeo");
    let copy = dir.path().join("named_replay_copy.grafeo");
    let copy2 = dir.path().join("named_replay_copy2.grafeo");
    {
        let db = persistent(&path);
        let session = db.session();
        session.execute("CREATE GRAPH g").unwrap();
        session.execute("USE GRAPH g").unwrap();
        session.execute("INSERT (:Committed {name: 'ok'})").unwrap();
        db.wal().unwrap().sync().unwrap();
        snapshot_db(&path, &copy);
        std::mem::forget(db);
    }
    let db = persistent(&copy);
    let g = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("g")
        .expect("WAL replay must restore named graph g");
    let id = g.create_node(&["Leaked"]);
    assert!(
        !id.is_valid(),
        "named-graph store after WAL reopen must stay sealed"
    );
    db.wal().unwrap().sync().unwrap();
    snapshot_db(&copy, &copy2);
    std::mem::forget(db);
    let db = persistent(&copy2);
    let session = db.session();
    session.execute("USE GRAPH g").unwrap();
    let leaked = session
        .execute("MATCH (n:Leaked) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    let kept = session
        .execute("MATCH (n:Committed) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(leaked, 0, "unframed named-graph create_node must be absent");
    assert_eq!(kept, 1, "framed named-graph insert must survive reopen");
}

#[cfg(feature = "triple-store")]
#[test]
fn raw_rdf_named_graph_or_create_does_not_survive_reopen() {
    use grafeo_core::graph::rdf::{Term, Triple};
    use grafeo_engine::GraphModel;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("raw_rdf_named.grafeo");
    let copy = dir.path().join("raw_rdf_named_copy.grafeo");
    let db = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .unwrap();
    let g = db
        .rdf_store()
        .graph_or_create("http://ex.org/leaked")
        .unwrap();
    let leaked = Triple::new(
        Term::iri("http://ex.org/s"),
        Term::iri("http://ex.org/p"),
        Term::literal("x"),
    );
    assert!(
        !g.insert(leaked.clone()),
        "unframed rdf named-graph insert on WAL must not publish"
    );
    db.wal().unwrap().sync().unwrap();
    snapshot_db(&path, &copy);
    std::mem::forget(db);
    let db = GrafeoDB::with_config(
        Config::persistent(&copy)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .unwrap();
    let n = db
        .execute_sparql(
            "SELECT ?o WHERE { GRAPH <http://ex.org/leaked> { <http://ex.org/s> <http://ex.org/p> ?o } }",
        )
        .unwrap()
        .row_count();
    assert_eq!(
        n, 0,
        "unframed RDF named-graph insert must be absent after reopen"
    );
}

#[cfg(feature = "triple-store")]
#[test]
fn raw_rdf_named_graph_after_wal_replay_stays_sealed() {
    use grafeo_core::graph::rdf::{Term, Triple};
    use grafeo_engine::GraphModel;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("rdf_named_replay.grafeo");
    let copy = dir.path().join("rdf_named_replay_copy.grafeo");
    let copy2 = dir.path().join("rdf_named_replay_copy2.grafeo");
    let both = |p: &std::path::Path| {
        GrafeoDB::with_config(
            Config::persistent(p)
                .with_graph_model(GraphModel::Both)
                .with_wal_durability(DurabilityMode::Sync),
        )
        .unwrap()
    };
    {
        let db = both(&path);
        db.execute_sparql(
            r#"INSERT DATA { GRAPH <http://ex.org/g> { <http://ex.org/keep> <http://ex.org/p> "v" } }"#,
        )
        .unwrap();
        db.wal().unwrap().sync().unwrap();
        snapshot_db(&path, &copy);
        std::mem::forget(db);
    }
    let db = both(&copy);
    let g = db
        .rdf_store()
        .graph("http://ex.org/g")
        .expect("WAL replay must restore named RDF graph");
    let keep = Triple::new(
        Term::iri("http://ex.org/keep"),
        Term::iri("http://ex.org/p"),
        Term::literal("v"),
    );
    let leaked = Triple::new(
        Term::iri("http://ex.org/leaked"),
        Term::iri("http://ex.org/p"),
        Term::literal("x"),
    );
    assert!(
        !g.insert(leaked),
        "named RDF graph after WAL reopen must refuse unframed insert"
    );
    assert!(
        !g.remove(&keep),
        "named RDF graph after WAL reopen must refuse unframed remove"
    );
    db.wal().unwrap().sync().unwrap();
    snapshot_db(&copy, &copy2);
    std::mem::forget(db);
    let db = both(&copy2);
    let kept = db
        .execute_sparql(
            "SELECT ?o WHERE { GRAPH <http://ex.org/g> { <http://ex.org/keep> <http://ex.org/p> ?o } }",
        )
        .unwrap()
        .row_count();
    let leaked_n = db
        .execute_sparql(
            "SELECT ?o WHERE { GRAPH <http://ex.org/g> { <http://ex.org/leaked> <http://ex.org/p> ?o } }",
        )
        .unwrap()
        .row_count();
    assert_eq!(kept, 1, "framed RDF named-graph insert must survive reopen");
    assert_eq!(
        leaked_n, 0,
        "unframed RDF named-graph insert must be absent"
    );
}
