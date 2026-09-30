//! Real Session commits recover by literal graph path from a copied WAL tail.

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use grafeo_common::types::{EdgeId, GraphPath, NodeId, TransactionId, Value};
use grafeo_core::graph::lpg::LpgStore;
use grafeo_engine::{
    Config, CreateIndexRequest, DurabilityMode, GrafeoDB, GraphModel, IndexCreateKind,
};
use grafeo_storage::wal::{LpgWal, WalRecord};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn persistent(path: &Path) -> TestResult<GrafeoDB> {
    Ok(GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync)
            .with_gc_interval(0),
    )?)
}

fn paths() -> TestResult<Vec<GraphPath>> {
    Ok(vec![
        GraphPath::root(),
        GraphPath::from_components(&[""])?,
        GraphPath::from_components(&["a/b"])?,
        GraphPath::from_components(&["a"])?,
        GraphPath::from_components(&["a", "b"])?,
    ])
}

fn sidecar(path: &Path) -> PathBuf {
    let mut path = path.as_os_str().to_owned();
    path.push(".wal");
    PathBuf::from(path)
}

fn target(db: &GrafeoDB, path: &GraphPath) -> TestResult<Arc<LpgStore>> {
    let mut store = Arc::clone(grafeo_engine::database::testing::root_lpg_store(db));
    for component in path.components() {
        store = store.graph(component).ok_or("missing exact graph path")?;
    }
    Ok(store)
}

fn bootstrap(path: &Path) -> TestResult<GrafeoDB> {
    let db = persistent(path)?;
    db.wal_checkpoint()?;
    let epoch = db.current_epoch().next();
    db.close()?;
    drop(db);
    // The public lifecycle grammar does not expose every literal coordinate.
    // Establish topology through checked current WAL records, then exercise
    // real managed callers only after reopening and checkpointing it.
    let wal = LpgWal::open(sidecar(path))?;
    let transaction_id = TransactionId::new(8001);
    for (id, graph) in paths()?.into_iter().enumerate().skip(1) {
        wal.log(&WalRecord::CreateLpgGraph {
            incarnation: grafeo_common::types::GraphIncarnationId::new(id as u64),
            graph,
            transaction_id,
        })?;
    }
    wal.log(&WalRecord::Committed {
        transaction_id,
        epoch,
    })?;
    wal.close()?;
    drop(wal);
    let db = persistent(path)?;
    for graph in paths()? {
        assert_eq!(target(&db, &graph)?.node_count(), 0);
    }
    db.wal_checkpoint()?;
    Ok(db)
}

fn recover_tail(db: GrafeoDB, source: &Path, copy: &Path, baseline: &[u8]) -> TestResult<GrafeoDB> {
    db.wal().ok_or("missing real WAL")?.sync()?;
    assert_eq!(
        std::fs::read(source)?,
        baseline,
        "the container must precede all tested commits"
    );
    std::fs::copy(source, copy)?;
    let destination = sidecar(copy);
    std::fs::create_dir_all(&destination)?;
    let mut copied = 0;
    for entry in std::fs::read_dir(sidecar(source))? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            std::fs::copy(entry.path(), destination.join(entry.file_name()))?;
            copied += 1;
        }
    }
    assert!(copied > 0, "recovery must consume a real copied WAL tail");
    db.close()?;
    drop(db);
    persistent(copy)
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
fn exact_auxiliary_indexes(db: &GrafeoDB) -> TestResult<Vec<Vec<u8>>> {
    use grafeo_common::storage::Section;
    use grafeo_core::graph::lpg::{LpgStoreSection, PhysicalIndexKey, decode_index_key};

    let mut images = Vec::new();
    #[cfg(feature = "text-index")]
    {
        let mut indexes = Vec::new();
        for (path, store) in LpgStoreSection::new(Arc::clone(
            grafeo_engine::database::testing::root_lpg_store(db),
        ))
        .capture_graphs()?
        {
            for (key, view) in store.text_index_entries() {
                let (label, property) = decode_index_key(&key).ok_or("invalid Text key")?;
                indexes.push((PhysicalIndexKey::text(path.clone(), label, property), view));
            }
        }
        images.push(grafeo_core::index::text::TextIndexSection::from_views(indexes).serialize()?);
    }
    #[cfg(feature = "vector-index")]
    {
        let mut indexes = Vec::new();
        for (path, store) in LpgStoreSection::new(Arc::clone(
            grafeo_engine::database::testing::root_lpg_store(db),
        ))
        .capture_graphs()?
        {
            for (key, view) in store.vector_index_entries() {
                let (label, property) = decode_index_key(&key).ok_or("invalid Vector key")?;
                indexes.push((
                    PhysicalIndexKey::vector(path.clone(), label, property),
                    view,
                ));
            }
        }
        images
            .push(grafeo_core::index::vector::VectorStoreSection::from_views(indexes).serialize()?);
    }
    Ok(images)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Seed {
    node: NodeId,
    destination: NodeId,
    victim: NodeId,
    edge: EdgeId,
    deleted_edge: EdgeId,
}

fn seed(db: &GrafeoDB, graph: &GraphPath, scope: usize) -> TestResult<Seed> {
    let session = db.session();
    session.use_graph_path(graph)?;
    let node = session.create_node_with_props(
        &["Doc", "RemoveMe"],
        [
            ("scope", Value::from(format!("before-{scope}"))),
            ("body", Value::from("ancient text")),
            ("scratch", Value::Bool(true)),
            ("embedding", Value::Vector(Arc::from([1.0_f32, 2.0, 3.0]))),
        ],
    )?;
    let destination = session.create_node_with_props(&["Destination"], [])?;
    let victim = session.create_node_with_props(&["Victim"], [])?;
    let edge = session.create_edge_with_props(
        node,
        destination,
        "CONNECTS",
        [
            ("scope", Value::from("before")),
            ("scratch", Value::Bool(true)),
        ],
    )?;
    let deleted_edge = session.create_edge_with_props(node, victim, "DELETE_ME", [])?;
    Ok(Seed {
        node,
        destination,
        victim,
        edge,
        deleted_edge,
    })
}

fn nested_tail(compact: bool) -> TestResult {
    let directory = tempfile::tempdir()?;
    let source = directory.path().join("nested.grafeo");
    let copy = directory.path().join("nested-tail.grafeo");
    let db = bootstrap(&source)?;
    let paths = paths()?;
    let seeds = paths
        .iter()
        .enumerate()
        .map(|(scope, path)| seed(&db, path, scope))
        .collect::<TestResult<Vec<_>>>()?;
    assert!(
        seeds.windows(2).all(|pair| pair[0] == pair[1]),
        "all local entity IDs must collide across graphs"
    );
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

    let mut owners = Vec::new();
    for (scope, graph) in paths.iter().enumerate() {
        owners.push(db.create_index(CreateIndexRequest {
            graph: graph.clone(),
            name: Some(format!("scope_{scope}")),
            label: None,
            property: "scope".into(),
            kind: if scope % 2 == 0 {
                IndexCreateKind::Property
            } else {
                IndexCreateKind::BTree
            },
        })?);
        #[cfg(feature = "text-index")]
        owners.push(db.create_index(CreateIndexRequest {
            graph: graph.clone(),
            name: Some(format!("text_{scope}")),
            label: Some("Doc".into()),
            property: "body".into(),
            kind: IndexCreateKind::Text {
                min_token_length: None,
            },
        })?);
        #[cfg(feature = "vector-index")]
        owners.push(db.create_index(CreateIndexRequest {
            graph: graph.clone(),
            name: Some(format!("vector_{scope}")),
            label: Some("Doc".into()),
            property: "embedding".into(),
            kind: IndexCreateKind::Vector {
                dimensions: Some(3),
                metric: Some("euclidean".into()),
                m: Some(8),
                ef_construction: Some(32),
                ef: None,
                quantization: None,
            },
        })?);
    }

    let mut session = db.session();
    session.begin_transaction()?;
    let mut tail_ids = Vec::new();
    for (scope, (graph, seed)) in paths.iter().zip(&seeds).enumerate() {
        session.use_graph_path(graph)?;
        session.set_node_property(seed.node, "scope", Value::from(format!("after-{scope}")))?;
        session.set_node_property(
            seed.node,
            "body",
            Value::from(format!("modern text scope{scope}")),
        )?;
        session.set_node_property(
            seed.node,
            "embedding",
            Value::Vector(Arc::from([4.0_f32, 5.0, 6.0])),
        )?;
        assert!(session.remove_node_property(seed.node, "scratch"));
        assert!(session.remove_node_label(seed.node, "RemoveMe"));
        assert!(session.add_node_label(seed.node, "Committed"));
        session.set_edge_property(seed.edge, "scope", Value::from(format!("edge-{scope}")))?;
        assert!(session.remove_edge_property(seed.edge, "scratch"));
        assert!(session.delete_edge(seed.deleted_edge));
        assert!(
            session.get_node(seed.victim).is_some(),
            "victim absent in {graph:?}"
        );
        assert!(
            session.delete_node(seed.victim),
            "victim deletion failed in {graph:?}; transaction={:?}",
            session.active_transaction_id()
        );
        let tail_node = session
            .create_node_with_props(&["Tail"], [("scope", Value::from(format!("tail-{scope}")))])?;
        let tail_edge = session.create_edge_with_props(
            seed.destination,
            tail_node,
            "TAIL",
            [("scope", Value::from(format!("tail-edge-{scope}")))],
        )?;
        tail_ids.push((tail_node, tail_edge));
    }
    session.commit()?;
    assert!(tail_ids.windows(2).all(|pair| pair[0] == pair[1]));

    session.begin_transaction()?;
    for (graph, seed) in paths.iter().zip(&seeds) {
        session.use_graph_path(graph)?;
        session.set_node_property(seed.node, "scope", Value::from("aborted"))?;
        session.set_node_property(seed.node, "body", Value::from("aborted token"))?;
        assert!(session.add_node_label(seed.node, "Aborted"));
    }
    session.execute("CREATE INDEX aborted_owner FOR (n:Doc) ON (n.aborted)")?;
    session.rollback()?;

    session.begin_transaction()?;
    session.savepoint("before_changes")?;
    for ((graph, seed), (_, tail_edge)) in paths.iter().zip(&seeds).zip(&tail_ids) {
        session.use_graph_path(graph)?;
        session.set_node_property(seed.node, "scope", Value::from("rolled-back"))?;
        session.set_node_property(
            seed.node,
            "embedding",
            Value::Vector(Arc::from([99.0_f32, 99.0, 99.0])),
        )?;
        assert!(session.remove_node_label(seed.node, "Doc"));
        assert!(session.delete_edge(*tail_edge));
    }
    session.execute("CREATE INDEX savepoint_owner FOR (n:Doc) ON (n.rolled_back)")?;
    session.rollback_to_savepoint("before_changes")?;
    session.commit()?;
    drop(session);

    let epoch = db.current_epoch();
    let catalog = db.execute("SHOW INDEXES")?.rows().to_vec();
    assert_eq!(catalog.len(), owners.len());
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    let exact = exact_auxiliary_indexes(&db)?;
    let recovered = recover_tail(db, &source, &copy, &baseline)?;
    assert_eq!(recovered.current_epoch(), epoch);
    assert_eq!(recovered.execute("SHOW INDEXES")?.rows(), catalog);
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    assert_eq!(exact_auxiliary_indexes(&recovered)?, exact);
    for (scope, ((graph, seed), (tail_node, tail_edge))) in
        paths.iter().zip(&seeds).zip(&tail_ids).enumerate()
    {
        let session = recovered.session();
        session.use_graph_path(graph)?;
        assert_eq!(session.current_graph_path(), *graph);
        let node = session
            .get_node(seed.node)
            .ok_or("lost graph-qualified node")?;
        assert_eq!(
            node.get_property("scope"),
            Some(&Value::from(format!("after-{scope}")))
        );
        assert_eq!(
            node.get_property("body"),
            Some(&Value::from(format!("modern text scope{scope}")))
        );
        assert!(node.get_property("scratch").is_none());
        assert!(node.has_label("Doc") && node.has_label("Committed"));
        assert!(!node.has_label("RemoveMe") && !node.has_label("Aborted"));
        assert!(session.get_node(seed.victim).is_none());
        assert!(session.get_edge(seed.deleted_edge).is_none());
        assert_eq!(
            session.get_node_property(*tail_node, "scope"),
            Some(Value::from(format!("tail-{scope}")))
        );
        let edge = session
            .get_edge(seed.edge)
            .ok_or("lost graph-qualified edge")?;
        assert_eq!(
            edge.get_property("scope"),
            Some(&Value::from(format!("edge-{scope}")))
        );
        assert!(edge.get_property("scratch").is_none());
        let tail = session
            .get_edge(*tail_edge)
            .ok_or("savepoint lost surviving edge")?;
        assert_eq!(
            tail.get_property("scope"),
            Some(&Value::from(format!("tail-edge-{scope}")))
        );
        let store = target(&recovered, graph)?;
        assert!(store.has_property_index("scope"));
        assert!(!store.has_property_index("aborted"));
        assert!(!store.has_property_index("rolled_back"));
        assert_eq!(
            store.find_nodes_by_property("scope", &Value::from(format!("after-{scope}"))),
            vec![seed.node]
        );
        for other_scope in 0..paths.len() {
            if other_scope != scope {
                assert!(
                    store
                        .find_nodes_by_property(
                            "scope",
                            &Value::from(format!("after-{other_scope}"))
                        )
                        .is_empty()
                );
            }
        }
    }
    for owner in owners {
        assert!(
            recovered.drop_index(owner)?,
            "exact owner ID must survive replay"
        );
    }
    assert!(recovered.execute("SHOW INDEXES")?.rows().is_empty());
    recovered.close()?;
    Ok(())
}

#[test]
fn literal_and_nested_paths_recover_real_session_mutations_and_owners() -> TestResult {
    nested_tail(false)
}

#[cfg(feature = "compact-store")]
#[test]
fn literal_and_nested_paths_recover_compact_session_mutations_and_owners() -> TestResult {
    nested_tail(true)
}

#[test]
fn standalone_catalog_publication_preserves_nested_graph_epochs_on_recovery() -> TestResult {
    let directory = tempfile::tempdir()?;
    let source = directory.path().join("catalog-epochs.grafeo");
    let copy = directory.path().join("catalog-epochs-tail.grafeo");
    let db = bootstrap(&source)?;
    let baseline = std::fs::read(&source)?;
    let before = db.current_epoch();
    db.session().execute("CREATE SCHEMA clock")?;
    let publication = db.current_epoch();
    assert!(publication > before);
    for graph in paths()? {
        assert_eq!(
            target(&db, &graph)?.current_epoch(),
            publication,
            "standalone catalog publication must advance every live descendant: {graph:?}"
        );
    }
    let recovered = recover_tail(db, &source, &copy, &baseline)?;
    assert_eq!(recovered.current_epoch(), publication);
    for graph in paths()? {
        assert_eq!(
            target(&recovered, &graph)?.current_epoch(),
            publication,
            "replay must preserve the live graph epoch: {graph:?}"
        );
    }
    recovered.close()?;
    Ok(())
}

#[test]
fn stale_nested_session_cannot_publish_into_a_recreated_parent() -> TestResult {
    let directory = tempfile::tempdir()?;
    let source = directory.path().join("stale.grafeo");
    let copy = directory.path().join("stale-tail.grafeo");
    let db = bootstrap(&source)?;
    let nested = GraphPath::from_components(&["a", "b"])?;
    let literal = GraphPath::from_components(&["a/b"])?;
    let literal_seed = seed(&db, &literal, 1)?;
    let seed = seed(&db, &nested, 0)?;
    db.wal_checkpoint()?;
    let baseline = std::fs::read(&source)?;
    let retired = target(&db, &nested)?;
    let mut stale = db.session();
    stale.use_graph_path(&nested)?;
    stale.begin_transaction()?;
    stale.set_node_property(seed.node, "scope", Value::from("stale"))?;
    stale.execute("CREATE INDEX stale_owner FOR (n:Doc) ON (n.scope)")?;

    let replacement = db.session();
    replacement.execute("DROP GRAPH a")?;
    replacement.execute("CREATE GRAPH a")?;
    drop(replacement);
    let epoch = db.current_epoch();
    assert!(
        stale.commit().is_err(),
        "retired nested incarnation must reject publication"
    );
    drop(stale);
    assert_eq!(db.current_epoch(), epoch);
    assert!(db.execute("SHOW INDEXES")?.rows().is_empty());
    assert_eq!(
        retired
            .get_node(seed.node)
            .ok_or("lost retained node")?
            .get_property("scope"),
        Some(&Value::from("before-0"))
    );
    drop(retired);
    let recovered = recover_tail(db, &source, &copy, &baseline)?;
    let parent = target(&recovered, &GraphPath::from_components(&["a"])?)?;
    assert_eq!(parent.node_count(), 0);
    assert!(
        parent.graph("b").is_none(),
        "stale child must not be resurrected"
    );
    assert!(recovered.session().use_graph_path(&nested).is_err());
    let literal_session = recovered.session();
    literal_session.use_graph_path(&literal)?;
    assert_eq!(
        literal_session.get_node_property(literal_seed.node, "scope"),
        Some(Value::from("before-1"))
    );
    drop(literal_session);
    assert!(recovered.execute("SHOW INDEXES")?.rows().is_empty());
    recovered.close()?;
    Ok(())
}
