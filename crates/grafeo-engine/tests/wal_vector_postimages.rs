//! Real uncheckpointed WAL recovery preserves physical Vector state and continuation.
#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "vector-index",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use grafeo_common::storage::Section;
use grafeo_common::types::{GraphPath, IndexId, NodeId, TransactionId, Value};
use grafeo_core::graph::GraphStoreSearch;
use grafeo_core::graph::lpg::{LpgStore, LpgStoreSection, PhysicalIndexKey, decode_index_key};
use grafeo_core::index::vector::VectorStoreSection;
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

fn vector(value: u16) -> Value {
    let n = f32::from(value);
    Value::Vector(Arc::from([
        n,
        n + 1.0,
        n * 0.5,
        n + 3.0,
        n * 0.25,
        n + 5.0,
        n * 0.125,
        n + 7.0,
    ]))
}

fn request(graph: GraphPath, name: &str, quantization: Option<&str>) -> CreateIndexRequest {
    CreateIndexRequest {
        graph,
        name: Some(name.into()),
        label: Some("Doc".into()),
        property: "embedding".into(),
        kind: IndexCreateKind::Vector {
            dimensions: Some(8),
            metric: Some("euclidean".into()),
            m: Some(8),
            ef_construction: Some(32),
            ef: None,
            quantization: quantization.map(str::to_owned),
        },
    }
}

fn target(db: &GrafeoDB, path: &GraphPath) -> TestResult<Arc<LpgStore>> {
    let mut store = Arc::clone(grafeo_engine::database::testing::root_lpg_store(db));
    for component in path.components() {
        store = store.graph(component).ok_or("missing exact graph path")?;
    }
    Ok(store)
}

fn exact(db: &GrafeoDB) -> TestResult<Vec<u8>> {
    let mut indexes = Vec::new();
    for (path, store) in LpgStoreSection::new(Arc::clone(
        grafeo_engine::database::testing::root_lpg_store(db),
    ))
    .capture_graphs()?
    {
        for (key, view) in store.vector_index_entries() {
            let (label, property) = decode_index_key(&key).ok_or("invalid physical Vector key")?;
            indexes.push((
                PhysicalIndexKey::vector(path.clone(), label, property),
                view,
            ));
        }
    }
    Ok(VectorStoreSection::from_views(indexes).serialize()?)
}

fn search(db: &GrafeoDB, path: &GraphPath) -> TestResult<Vec<(NodeId, u64)>> {
    let query = [10.0, 11.0, 5.0, 13.0, 2.5, 15.0, 1.25, 17.0];
    let rows = if path.components().is_empty() {
        db.graph_store().vector_search_visible(
            "Doc",
            "embedding",
            &query,
            20,
            db.current_epoch(),
            TransactionId::INVALID,
        )
    } else {
        target(db, path)?.vector_search_visible(
            "Doc",
            "embedding",
            &query,
            20,
            db.current_epoch(),
            TransactionId::INVALID,
        )
    };
    Ok(rows
        .into_iter()
        .map(|(id, score)| (id, score.to_bits()))
        .collect())
}

fn sidecar(path: &Path) -> PathBuf {
    let mut path = path.as_os_str().to_owned();
    path.push(".wal");
    PathBuf::from(path)
}

fn copy_tail(db: &GrafeoDB, source: &Path, copy: &Path, baseline: &[u8]) -> TestResult {
    db.wal().ok_or("fixture needs WAL")?.sync()?;
    assert_eq!(
        std::fs::read(source)?,
        baseline,
        "must recover the tail, not a later checkpoint"
    );
    std::fs::copy(source, copy)?;
    let destination = sidecar(copy);
    std::fs::create_dir_all(&destination)?;
    let mut count = 0;
    for entry in std::fs::read_dir(sidecar(source))? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            std::fs::copy(entry.path(), destination.join(entry.file_name()))?;
            count += 1;
        }
    }
    assert!(count > 0);
    Ok(())
}

fn continuation(db: &GrafeoDB) -> TestResult<NodeId> {
    Ok(db
        .session()
        .create_node_with_props(&["Doc"], [("embedding", vector(51))])?)
}

fn sparse_tail(compact: bool, quantization: Option<&str>) -> TestResult {
    let temp = tempfile::tempdir()?;
    let source = temp.path().join("vectors.grafeo");
    let copy = temp.path().join("tail.grafeo");
    let db = persistent(&source)?;
    let root = GraphPath::root();
    let named_path = GraphPath::from_components(&["literal/slash"])?;
    assert!(db.create_graph("literal/slash")?);
    let mut nodes = Vec::new();
    for value in 1..=4 {
        nodes.push(db.session().create_node_with_props(
            &["Doc"],
            [
                ("embedding", vector(value)),
                ("body", Value::from("original text")),
            ],
        )?);
    }
    let node = nodes[0];
    let named = db.session();
    named.use_graph_path(&named_path)?;
    let named_node = named.create_node_with_props(&["Doc"], [("embedding", vector(7))])?;
    assert_eq!(
        named_node, node,
        "graph-local identities deliberately collide"
    );
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
    assert!(!compact);
    let root_owner = db.create_index(request(root.clone(), "root_vector", quantization))?;
    let named_owner = db.create_index(request(named_path.clone(), "named_vector", quantization))?;
    #[cfg(feature = "text-index")]
    db.create_index(CreateIndexRequest {
        graph: root.clone(),
        name: Some("mixed_text".into()),
        label: Some("Doc".into()),
        property: "body".into(),
        kind: IndexCreateKind::Text {
            min_token_length: None,
        },
    })?;
    let hydration = exact(&db)?;
    db.set_node_property(node, "unrelated", Value::Bool(true))?;
    assert_eq!(
        exact(&db)?,
        hydration,
        "cold hydration cannot rederive Vector"
    );

    let mut transaction = db.session();
    transaction.begin_transaction()?;
    transaction.set_node_property(node, "embedding", vector(10))?;
    transaction.set_node_property(node, "body", Value::from("mixed committed text"))?;
    assert!(transaction.remove_node_property(nodes[1], "embedding"));
    assert!(transaction.remove_node_label(nodes[2], "Doc"));
    assert!(transaction.delete_node(nodes[3]));
    transaction.commit()?;
    drop(transaction);
    let unchanged = exact(&db)?;
    let mut transaction = db.session();
    transaction.begin_transaction()?;
    transaction.set_node_property(node, "embedding", vector(99))?;
    transaction.set_node_property(node, "embedding", vector(10))?;
    assert!(transaction.remove_node_label(node, "Doc"));
    assert!(transaction.add_node_label(node, "Doc"));
    transaction.commit()?;
    drop(transaction);
    assert_eq!(
        exact(&db)?,
        unchanged,
        "normalized no-op cannot consume topology RNG"
    );
    let mut transaction = db.session();
    transaction.begin_transaction()?;
    transaction.set_node_property(node, "embedding", vector(98))?;
    transaction.execute(
        "CREATE INDEX aborted_vector FOR (n:Doc) ON (n.uncommitted) USING VECTOR {dimensions: 8}",
    )?;
    transaction.rollback()?;
    drop(transaction);
    assert_eq!(exact(&db)?, unchanged);
    let mut transaction = db.session();
    transaction.begin_transaction()?;
    transaction.savepoint("vector_baseline")?;
    transaction.set_node_property(node, "embedding", vector(97))?;
    assert!(transaction.remove_node_label(node, "Doc"));
    transaction.rollback_to_savepoint("vector_baseline")?;
    transaction.commit()?;
    drop(transaction);
    assert_eq!(exact(&db)?, unchanged);
    let named = db.session();
    named.use_graph_path(&named_path)?;
    named.set_node_property(named_node, "embedding", vector(11))?;
    drop(named);
    // Root remains sparse; only the named owner exercises a full rebuild image.
    db.rebuild_index(named_owner)?;
    let expected = exact(&db)?;
    let rows = search(&db, &root)?;
    assert_eq!(
        rows.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![node]
    );
    let named_rows = search(&db, &named_path)?;
    assert_eq!(named_rows.len(), 1);
    let owners = db.execute("SHOW INDEXES")?.rows().to_vec();
    #[cfg(feature = "text-index")]
    let text = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_text_index("Doc", "body")
        .ok_or("missing Text")?
        .read()
        .encode_wal_birth()?;
    copy_tail(&db, &source, &copy, &baseline)?;
    let next = continuation(&db)?;
    let continued = exact(&db)?;
    let continued_rows = search(&db, &root)?;
    db.close()?;
    drop(db);
    let recovered = persistent(&copy)?;
    assert_eq!(
        exact(&recovered)?,
        expected,
        "compact={compact}, quantization={quantization:?}"
    );
    assert_eq!(search(&recovered, &root)?, rows);
    assert_eq!(search(&recovered, &named_path)?, named_rows);
    assert_eq!(recovered.execute("SHOW INDEXES")?.rows(), owners);
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&recovered)
            .get_vector_index("Doc", "uncommitted")
            .is_none()
    );
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&recovered)
            .graph("literal")
            .is_none()
    );
    #[cfg(feature = "text-index")]
    assert_eq!(
        grafeo_engine::database::testing::root_lpg_store(&recovered)
            .get_text_index("Doc", "body")
            .ok_or("missing recovered Text")?
            .read()
            .encode_wal_birth()?,
        text
    );
    assert_eq!(continuation(&recovered)?, next);
    assert_eq!(
        exact(&recovered)?,
        continued,
        "next insertion must resume exact topology RNG"
    );
    assert_eq!(search(&recovered, &root)?, continued_rows);
    assert!(recovered.drop_index(root_owner)?);
    assert!(!recovered.drop_index(root_owner)?);
    recovered.close()?;
    Ok(())
}

#[test]
fn wal_vector_live_tail_is_exact_for_all_families() -> TestResult {
    for compact in [
        false,
        #[cfg(feature = "compact-store")]
        true,
    ] {
        for quantization in [None, Some("scalar"), Some("binary"), Some("product")] {
            sparse_tail(compact, quantization)?;
        }
    }
    Ok(())
}

#[test]
fn wal_scalar_sparse_commit_crosses_default_training_threshold() -> TestResult {
    let temp = tempfile::tempdir()?;
    let source = temp.path().join("training.grafeo");
    let copy = temp.path().join("training-tail.grafeo");
    let db = persistent(&source)?;
    db.wal_checkpoint()?;
    let baseline = std::fs::read(&source)?;
    db.create_index(request(GraphPath::root(), "trained_scalar", Some("scalar")))?;
    let mut transaction = db.session();
    transaction.begin_transaction()?;
    for value in 0..995 {
        transaction.create_node_with_props(&["Doc"], [("embedding", vector(value))])?;
    }
    transaction.commit()?;
    drop(transaction);
    assert_eq!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .get_vector_index("Doc", "embedding")
            .ok_or("missing scalar")?
            .len(),
        995
    );
    let mut transaction = db.session();
    transaction.begin_transaction()?;
    // The first five finish training; the following outliers use the frozen model.
    for value in [995, 996, 997, 998, 999, 1000, 2000, 4000] {
        transaction.create_node_with_props(&["Doc"], [("embedding", vector(value))])?;
    }
    transaction.commit()?;
    drop(transaction);
    assert_eq!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .get_vector_index("Doc", "embedding")
            .ok_or("missing trained scalar")?
            .len(),
        1003
    );
    let expected = exact(&db)?;
    let rows = search(&db, &GraphPath::root())?;
    assert!(!rows.is_empty());
    copy_tail(&db, &source, &copy, &baseline)?;
    let next = continuation(&db)?;
    let continued = exact(&db)?;
    db.close()?;
    drop(db);
    let recovered = persistent(&copy)?;
    assert_eq!(exact(&recovered)?, expected);
    assert_eq!(search(&recovered, &GraphPath::root())?, rows);
    assert_eq!(continuation(&recovered)?, next);
    assert_eq!(exact(&recovered)?, continued);
    recovered.close()?;
    Ok(())
}

#[test]
fn wal_vector_graph_replacement_retires_only_the_old_owner() -> TestResult {
    let temp = tempfile::tempdir()?;
    let source = temp.path().join("replacement.grafeo");
    let copy = temp.path().join("replacement-tail.grafeo");
    let db = persistent(&source)?;
    assert!(db.create_graph("replaceable")?);
    db.wal_checkpoint()?;
    let baseline = std::fs::read(&source)?;
    let path = GraphPath::from_components(&["replaceable"])?;
    let old = db.create_index(request(path.clone(), "old_vector", None))?;
    let mut transaction = db.session();
    transaction.begin_transaction()?;
    transaction.execute("DROP GRAPH replaceable")?;
    transaction.execute("CREATE GRAPH replaceable")?;
    transaction.use_graph_path(&path)?;
    let node = transaction.create_node_with_props(&["Doc"], [("embedding", vector(10))])?;
    transaction.execute("CREATE INDEX replacement_vector FOR (n:Doc) ON (n.embedding) USING VECTOR {dimensions: 8, metric: 'euclidean'}")?;
    transaction.commit()?;
    drop(transaction);
    let replacement = IndexId::new(old.as_u32() + 1);
    assert!(!db.drop_index(old)?);
    let expected = exact(&db)?;
    let rows = search(&db, &path)?;
    assert_eq!(
        rows.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![node]
    );
    copy_tail(&db, &source, &copy, &baseline)?;
    db.close()?;
    drop(db);
    let recovered = persistent(&copy)?;
    assert_eq!(exact(&recovered)?, expected);
    assert_eq!(search(&recovered, &path)?, rows);
    assert!(!recovered.drop_index(old)?);
    assert!(recovered.drop_index(replacement)?);
    recovered.close()?;
    Ok(())
}

#[test]
fn wal_vector_gc_refusal_precedes_root_and_named_history_pruning() -> TestResult {
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
        let node = session.create_node_with_props(&["Doc"], [("embedding", vector(1))])?;
        drop(session);
        db.wal_checkpoint()?;
        let baseline = std::fs::read(&source)?;
        db.create_index(request(path.clone(), "guarded_vector", None))?;
        let session = db.session();
        session.use_graph_path(&path)?;
        session.set_node_property(node, "embedding", vector(2))?;
        drop(session);
        let before = exact(&db)?;
        let properties = target(&db, &path)?.node_property_history(node);
        let epoch = db.current_epoch();
        let sequence = db.wal().ok_or("missing WAL")?.current_sequence();
        let error = db.gc().err().ok_or("WAL Vector GC must reject")?;
        assert!(error.to_string().contains("WAL"), "{error}");
        assert_eq!(db.current_epoch(), epoch);
        assert_eq!(db.wal().ok_or("missing WAL")?.current_sequence(), sequence);
        assert_eq!(exact(&db)?, before);
        assert_eq!(target(&db, &path)?.node_property_history(node), properties);
        copy_tail(&db, &source, &copy, &baseline)?;
        db.close()?;
        drop(db);
        let recovered = persistent(&copy)?;
        assert_eq!(exact(&recovered)?, before);
        assert_eq!(
            target(&recovered, &path)?.node_property_history(node),
            properties
        );
        recovered.close()?;
    }
    Ok(())
}
