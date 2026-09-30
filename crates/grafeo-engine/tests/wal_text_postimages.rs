//! Actual WAL commits transport exact Text state, not a second tokenization pass.

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "text-index",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use grafeo_common::storage::Section;
use grafeo_common::types::{EpochId, GraphPath, IndexId, NodeId, TransactionId, Value};
use grafeo_core::graph::lpg::{LpgStore, LpgStoreSection, PhysicalIndexKey, decode_index_key};
use grafeo_core::index::text::TextIndexSection;
use grafeo_engine::{
    Config, CreateIndexRequest, DurabilityMode, GrafeoDB, GraphModel, IndexCreateKind,
};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn persistent(path: &Path) -> TestResult<GrafeoDB> {
    Ok(GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync)
            .with_gc_interval(0),
    )?)
}

fn target(db: &GrafeoDB, path: &GraphPath) -> TestResult<Arc<LpgStore>> {
    let mut store = Arc::clone(grafeo_engine::database::testing::root_lpg_store(db));
    for component in path.components() {
        store = store.graph(component).ok_or("missing exact graph path")?;
    }
    Ok(store)
}

fn create_text(db: &GrafeoDB, graph: GraphPath, name: &str) -> TestResult<IndexId> {
    Ok(db.create_index(CreateIndexRequest {
        graph,
        name: Some(name.into()),
        label: Some("Doc".into()),
        property: "body".into(),
        kind: IndexCreateKind::Text {
            min_token_length: None,
        },
    })?)
}

fn exact_text(db: &GrafeoDB) -> TestResult<Vec<u8>> {
    let graphs = LpgStoreSection::new(Arc::clone(
        grafeo_engine::database::testing::root_lpg_store(db),
    ))
    .capture_graphs()?;
    let mut indexes = Vec::new();
    for (path, store) in graphs {
        for (key, view) in store.text_index_entries() {
            let (label, property) = decode_index_key(&key).ok_or("invalid local Text key")?;
            indexes.push((PhysicalIndexKey::text(path.clone(), label, property), view));
        }
    }
    Ok(TextIndexSection::from_views(indexes).serialize()?)
}

fn score_bits(
    db: &GrafeoDB,
    graph: &GraphPath,
    nodes: &[NodeId],
    epochs: &[EpochId],
) -> TestResult<Vec<Option<u64>>> {
    let store = target(db, graph)?;
    let view = store
        .get_text_index("Doc", "body")
        .ok_or("missing Text owner")?;
    let text = view.read();
    let mut scores = Vec::new();
    for &epoch in epochs {
        for &node in nodes {
            for query in ["ancient", "modern"] {
                scores.push(
                    text.score_document_visible(
                        node,
                        query,
                        epoch,
                        TransactionId::INVALID,
                        None,
                        false,
                    )?
                    .map(f64::to_bits),
                );
            }
        }
    }
    Ok(scores)
}

fn sidecar(path: &Path) -> PathBuf {
    let mut path = path.as_os_str().to_owned();
    path.push(".wal");
    PathBuf::from(path)
}

fn recover_tail(db: GrafeoDB, source: &Path, copy: &Path, baseline: &[u8]) -> TestResult<GrafeoDB> {
    db.wal().ok_or("fixture requires WAL")?.sync()?;
    assert_eq!(
        std::fs::read(source)?,
        baseline,
        "owner/data changes must remain in the WAL tail"
    );
    std::fs::copy(source, copy)?;
    let destination = sidecar(copy);
    std::fs::create_dir_all(&destination)?;
    let mut segments = 0;
    for entry in std::fs::read_dir(sidecar(source))? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            std::fs::copy(entry.path(), destination.join(entry.file_name()))?;
            segments += 1;
        }
    }
    assert!(segments > 0);
    // Checkpointing the original now cannot change the captured crash image.
    db.close()?;
    drop(db);
    persistent(copy)
}

fn sparse_text_tail(compact: bool) -> TestResult {
    let temp = tempfile::tempdir()?;
    let source = temp.path().join("text.grafeo");
    let copy = temp.path().join("text-tail.grafeo");
    let db = persistent(&source)?;
    let root = GraphPath::root();
    let named_path = GraphPath::from_components(&["literal/slash"])?;
    assert!(db.create_graph("literal/slash")?);
    let node = db
        .session()
        .create_node_with_props(&["Doc"], [("body", Value::from("ancient alpha"))])?;
    let deleted = db
        .session()
        .create_node_with_props(&["Doc"], [("body", Value::from("ancient beta"))])?;
    let named = db.session();
    named.use_graph_path(&named_path)?;
    let named_node =
        named.create_node_with_props(&["Doc"], [("body", Value::from("ancient named"))])?;
    drop(named);
    db.wal_checkpoint()?;
    let baseline = std::fs::read(&source)?;
    #[cfg(feature = "compact-store")]
    let db = {
        let mut db = db;
        if compact {
            db.compact()?;
        }
        db
    };
    #[cfg(not(feature = "compact-store"))]
    if compact {
        return Err("compact fixture needs compact-store".into());
    }
    let root_owner = create_text(&db, root.clone(), "root_text")?;
    let named_owner = create_text(&db, named_path.clone(), "named_text")?;
    let mut epochs = vec![db.current_epoch()];

    // On the compact variant this is the first hydration of the indexed row.
    let before_hydration = exact_text(&db)?;
    db.set_node_property(node, "unrelated", Value::Bool(true))?;
    assert_eq!(
        exact_text(&db)?,
        before_hydration,
        "data-only hydration cannot alter Text"
    );
    db.set_node_property(node, "body", Value::from("modern alpha alpha"))?;
    let named = db.session();
    named.use_graph_path(&named_path)?;
    named.set_node_property(named_node, "body", Value::from("modern named"))?;
    drop(named);
    epochs.push(db.current_epoch());

    let before_noop = exact_text(&db)?;
    let mut transaction = db.session();
    transaction.begin_transaction()?;
    transaction.set_node_property(node, "body", Value::from("transienttoken"))?;
    transaction.set_node_property(node, "body", Value::from("modern alpha alpha"))?;
    assert!(transaction.remove_node_label(node, "Doc"));
    assert!(transaction.add_node_label(node, "Doc"));
    transaction.commit()?;
    drop(transaction);
    assert_eq!(
        exact_text(&db)?,
        before_noop,
        "normalized String/label no-op must not manufacture history"
    );
    db.set_node_property(node, "body", Value::from("modern alpha alpha"))?;
    assert_eq!(
        exact_text(&db)?,
        before_noop,
        "equal String SET must not manufacture history"
    );

    let named = db.session();
    named.use_graph_path(&named_path)?;
    assert!(named.remove_node_label(named_node, "Doc"));
    epochs.push(db.current_epoch());
    assert!(named.add_node_label(named_node, "Doc"));
    epochs.push(db.current_epoch());
    drop(named);
    assert!(db.session().delete_node(deleted));
    epochs.push(db.current_epoch());

    let before_rollback = exact_text(&db)?;
    let owners_before_rollback = db.execute("SHOW INDEXES")?.rows().to_vec();
    let mut transaction = db.session();
    transaction.begin_transaction()?;
    transaction.set_node_property(node, "body", Value::from("abortedtext"))?;
    assert!(transaction.remove_node_label(node, "Doc"));
    transaction.execute("CREATE INDEX aborted_text FOR (n:Doc) ON (n.uncommitted) USING TEXT")?;
    transaction.rollback()?;
    drop(transaction);
    assert_eq!(exact_text(&db)?, before_rollback, "rollback retains Text");
    assert_eq!(db.execute("SHOW INDEXES")?.rows(), owners_before_rollback);
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .get_text_index("Doc", "uncommitted")
            .is_none()
    );

    let mut transaction = db.session();
    transaction.begin_transaction()?;
    transaction.savepoint("text_baseline")?;
    transaction.set_node_property(node, "body", Value::from("discardedtext"))?;
    assert!(transaction.remove_node_label(node, "Doc"));
    transaction.rollback_to_savepoint("text_baseline")?;
    transaction.commit()?;
    drop(transaction);
    assert_eq!(
        exact_text(&db)?,
        before_rollback,
        "savepoint rollback must not publish discarded Text postimages"
    );

    let root_scores_before = score_bits(&db, &root, &[node, deleted], &epochs)?;
    let named_scores_before = score_bits(&db, &named_path, &[named_node], &epochs)?;
    db.rebuild_index(root_owner)?;
    db.rebuild_index(named_owner)?;
    // Rebuild publishes a fresh current population at C, retaining all earlier
    // postings. It is not a byte-identical no-op; old-cut answers must be exact.
    assert_eq!(
        score_bits(&db, &root, &[node, deleted], &epochs)?,
        root_scores_before
    );
    assert_eq!(
        score_bits(&db, &named_path, &[named_node], &epochs)?,
        named_scores_before
    );
    epochs.push(db.current_epoch());
    let root_scores = score_bits(&db, &root, &[node, deleted], &epochs)?;
    let named_scores = score_bits(&db, &named_path, &[named_node], &epochs)?;
    assert!(root_scores.iter().any(Option::is_some));
    assert!(named_scores.iter().any(Option::is_some));
    let expected = exact_text(&db)?;

    let recovered = recover_tail(db, &source, &copy, &baseline)?;
    assert_eq!(exact_text(&recovered)?, expected);
    assert_eq!(
        recovered.execute("SHOW INDEXES")?.rows(),
        owners_before_rollback
    );
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&recovered)
            .get_text_index("Doc", "uncommitted")
            .is_none()
    );
    assert_eq!(
        score_bits(&recovered, &root, &[node, deleted], &epochs)?,
        root_scores
    );
    assert_eq!(
        score_bits(&recovered, &named_path, &[named_node], &epochs)?,
        named_scores
    );
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&recovered)
            .get_node(deleted)
            .is_none()
    );
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&recovered)
            .graph("literal")
            .is_none(),
        "slash is one literal component"
    );
    recovered.rebuild_index(root_owner)?;
    recovered.rebuild_index(named_owner)?;
    assert_eq!(
        score_bits(&recovered, &root, &[node, deleted], &epochs)?,
        root_scores
    );
    assert_eq!(
        score_bits(&recovered, &named_path, &[named_node], &epochs)?,
        named_scores
    );
    recovered.close()?;
    Ok(())
}

#[test]
fn wal_text_sparse_postimages_and_historical_score_bits_are_exact() -> TestResult {
    sparse_text_tail(false)
}

#[cfg(feature = "compact-store")]
#[test]
fn wal_text_postimages_do_not_retokenize_compact_hydration() -> TestResult {
    sparse_text_tail(true)
}

#[test]
fn wal_text_drop_recreate_same_graph_name_preserves_only_replacement() -> TestResult {
    let temp = tempfile::tempdir()?;
    let source = temp.path().join("replacement.grafeo");
    let copy = temp.path().join("replacement-tail.grafeo");
    let db = persistent(&source)?;
    assert!(db.create_graph("replaceable")?);
    let path = GraphPath::from_components(&["replaceable"])?;
    let session = db.session();
    session.use_graph_path(&path)?;
    session.create_node_with_props(&["Doc"], [("body", Value::from("obsoleteword"))])?;
    drop(session);
    db.wal_checkpoint()?;
    let baseline = std::fs::read(&source)?;
    let obsolete = create_text(&db, path.clone(), "obsolete_text")?;
    let mut session = db.session();
    session.begin_transaction()?;
    session.execute("DROP GRAPH replaceable")?;
    session.execute("CREATE GRAPH replaceable")?;
    session.use_graph_path(&path)?;
    let node =
        session.create_node_with_props(&["Doc"], [("body", Value::from("modern replacement"))])?;
    session.execute("CREATE INDEX replacement_text FOR (n:Doc) ON (n.body) USING TEXT")?;
    session.commit()?;
    drop(session);
    let epoch = db.current_epoch();
    let expected = exact_text(&db)?;
    let scores = score_bits(&db, &path, &[node], &[epoch])?;
    assert!(scores.iter().any(Option::is_some));
    assert!(!db.drop_index(obsolete)?);
    let recovered = recover_tail(db, &source, &copy, &baseline)?;
    assert_eq!(exact_text(&recovered)?, expected);
    assert_eq!(score_bits(&recovered, &path, &[node], &[epoch])?, scores);
    assert!(!recovered.drop_index(obsolete)?);
    assert_eq!(exact_text(&recovered)?, expected);
    assert!(recovered.drop_index(IndexId::new(obsolete.as_u32() + 1))?);
    assert!(
        target(&recovered, &path)?
            .get_text_index("Doc", "body")
            .is_none()
    );
    recovered.close()?;
    Ok(())
}

#[test]
fn wal_text_gc_guard_precedes_pruning_for_root_and_named_only_indexes() -> TestResult {
    for named_only in [false, true] {
        let temp = tempfile::tempdir()?;
        let source = temp.path().join("gc.grafeo");
        let copy = temp.path().join("gc-tail.grafeo");
        let db = persistent(&source)?;
        let path = if named_only {
            assert!(db.create_graph("only_named")?);
            GraphPath::from_components(&["only_named"])?
        } else {
            GraphPath::root()
        };
        let session = db.session();
        session.use_graph_path(&path)?;
        let node =
            session.create_node_with_props(&["Doc"], [("body", Value::from("ancient archive"))])?;
        drop(session);
        db.wal_checkpoint()?;
        let baseline = std::fs::read(&source)?;
        create_text(&db, path.clone(), "guarded_text")?;
        let historical = db.current_epoch();
        let session = db.session();
        session.use_graph_path(&path)?;
        session.set_node_property(node, "body", Value::from("modern archive"))?;
        drop(session);
        let epoch = db.current_epoch();
        let before = exact_text(&db)?;
        let properties = target(&db, &path)?.node_property_history(node);
        let scores = score_bits(&db, &path, &[node], &[historical, epoch])?;
        let error = db
            .gc()
            .expect_err("WAL Text pruning needs its own durable transition");
        assert!(error.to_string().contains("WAL"), "{error}");
        assert_eq!(db.current_epoch(), epoch);
        assert_eq!(exact_text(&db)?, before);
        assert_eq!(
            target(&db, &path)?.node_property_history(node),
            properties,
            "guard must precede data pruning too"
        );
        #[cfg(feature = "vector-index")]
        {
            let vector_owner = db.create_index(CreateIndexRequest {
                graph: path.clone(),
                name: Some("coexisting_vector".into()),
                label: Some("Doc".into()),
                property: "embedding".into(),
                kind: IndexCreateKind::Vector {
                    dimensions: Some(2),
                    metric: Some("cosine".into()),
                    m: None,
                    ef_construction: None,
                    ef: None,
                    quantization: None,
                },
            })?;
            assert!(
                target(&db, &path)?
                    .get_vector_index("Doc", "embedding")
                    .is_some()
            );
            db.rebuild_index(vector_owner)?;
            assert_eq!(exact_text(&db)?, before);
            let before_gc = db.current_epoch();
            assert!(db.gc().is_err(), "mixed owners still require durable GC");
            assert_eq!(db.current_epoch(), before_gc);
        }
        let recovered = recover_tail(db, &source, &copy, &baseline)?;
        assert_eq!(exact_text(&recovered)?, before);
        assert_eq!(
            score_bits(&recovered, &path, &[node], &[historical, epoch])?,
            scores
        );
        recovered.close()?;
    }
    Ok(())
}
