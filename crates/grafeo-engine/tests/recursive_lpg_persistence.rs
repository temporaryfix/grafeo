//! One recursive Both-model cut through durable recovery and exact copies.
//!
//! Topology and per-graph Simple tokenizer configurations use public native
//! APIs. The optional named hard-crash matrix covers mutation, mixed commit,
//! checkpoint and first/repeat representation transfer. Existing exact-history
//! restore constructs same-ID multi-lifetime input; ordinary writes never recycle IDs.
#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file",
    feature = "compact-store",
    feature = "text-index",
    feature = "vector-index"
))]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use grafeo_common::storage::Section;
use grafeo_common::types::{
    EdgeId, EpochId, GraphPath, IndexId, MAX_GRAPH_PATH_COMPONENTS, NodeId,
    ProjectionReconciliationState, PropertyKey, TransactionId, Value,
};
use grafeo_core::graph::GraphStoreSearch;
use grafeo_core::graph::lpg::{LpgStore, LpgStoreSection, PhysicalIndexKey, decode_index_key};
use grafeo_core::graph::rdf::{RdfLpgProjectionDefinition, RdfQuadVersion};
use grafeo_core::index::text::{BM25Config, TextIndexSection};
use grafeo_core::index::vector::{DistanceMetric, QuantizationType, VectorStoreSection};
use grafeo_engine::{
    Config, CreateIndexRequest, DurabilityMode, GrafeoDB, GraphModel, IndexCreateKind, Session,
    ValidTimeInterval,
};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

#[path = "support/recursive_same_id.rs"]
mod same_id_history;

#[cfg(feature = "async-storage")]
#[path = "support/recursive_wal_parity.rs"]
mod wal_parity;

const CHILD: &str = "GRAFEO_RECURSIVE_LIFECYCLE_CHILD";
const DATABASE: &str = "GRAFEO_RECURSIVE_LIFECYCLE_DATABASE";
const ORACLE: &str = "GRAFEO_RECURSIVE_LIFECYCLE_ORACLE";
const TEST: &str = "recursive_both_lifecycle_preserves_one_exact_cut";
const CLASS: &str = "http://recursive.test/Person";
const NAMED: &str = "http://recursive.test/named";
const TEXT_MINIMUMS: [usize; 6] = [3, 2, 0, 1, 4, 7];
const TEXT_QUERIES: [&str; 7] = [
    "x",
    "ox",
    "baseline",
    "tail",
    "archive",
    "companion",
    "refreshed",
];

fn persistent(path: &Path) -> TestResult<GrafeoDB> {
    Ok(GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync)
            .with_gc_interval(0),
    )?)
}

fn data_paths() -> TestResult<Vec<GraphPath>> {
    Ok(vec![
        GraphPath::root(),
        GraphPath::from_components(&[""])?,
        GraphPath::from_components(&["a/b"])?,
        GraphPath::from_components(&["a", "b"])?,
        GraphPath::from_components(&["日本語", "📚"])?,
        GraphPath::from_components(&vec!["deep"; MAX_GRAPH_PATH_COMPONENTS])?,
    ])
}

fn topology_paths() -> TestResult<Vec<GraphPath>> {
    let mut paths = std::collections::BTreeSet::new();
    for path in data_paths()? {
        let components: Vec<_> = path.components().iter().map(String::as_str).collect();
        for depth in 0..=components.len() {
            paths.insert(GraphPath::from_components(&components[..depth])?);
        }
    }
    Ok(paths.into_iter().collect())
}

fn target(db: &GrafeoDB, path: &GraphPath) -> TestResult<Arc<LpgStore>> {
    let mut store = Arc::clone(grafeo_engine::database::testing::root_lpg_store(db));
    for component in path.components() {
        store = store
            .graph(component)
            .ok_or("missing literal graph component")?;
    }
    Ok(store)
}

fn view(db: &GrafeoDB, path: &GraphPath) -> TestResult<Arc<dyn GraphStoreSearch>> {
    if path.components().is_empty() {
        Ok(db.graph_store())
    } else {
        Ok(target(db, path)?)
    }
}

fn projection_id() -> u64 {
    RdfLpgProjectionDefinition::new(CLASS, "Projected").id()
}

fn blank_recursive_memory(path: &Path) -> TestResult<GrafeoDB> {
    let db = persistent(path)?;
    db.wal_checkpoint()?;
    let checkpoint = std::fs::read(path)?;
    let mut session = db.session();
    session.begin_transaction()?;
    for graph in topology_paths()? {
        assert_eq!(
            session.create_graph_path(&graph)?,
            !graph.components().is_empty()
        );
    }
    session.commit()?;
    drop(session);
    let expected = db.export_snapshot()?;
    assert_eq!(
        std::fs::read(path)?,
        checkpoint,
        "native topology remains in the WAL tail"
    );
    let replay_path = path.with_extension("tail-copy.grafeo");
    std::fs::copy(path, &replay_path)?;
    let sidecar = |base: &Path| {
        let mut name = base.as_os_str().to_owned();
        name.push(".wal");
        PathBuf::from(name)
    };
    let replay_wal = sidecar(&replay_path);
    std::fs::create_dir(&replay_wal)?;
    for entry in std::fs::read_dir(sidecar(path))? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            std::fs::copy(entry.path(), replay_wal.join(entry.file_name()))?;
        }
    }
    db.close()?;
    drop(db);
    let db = persistent(&replay_path)?;
    assert_eq!(db.export_snapshot()?, expected);
    let memory = GrafeoDB::import_snapshot(&db.export_snapshot()?)?;
    db.close()?;
    Ok(memory)
}

fn vector(first: bool) -> Value {
    Value::Vector(if first {
        Arc::from([1.0_f32, 0.0, 0.0])
    } else {
        Arc::from([0.0_f32, 1.0, 0.0])
    })
}

fn index_request(graph: GraphPath, scope: usize, family: &str) -> CreateIndexRequest {
    CreateIndexRequest {
        graph,
        name: Some(format!("scope_{scope}_{family}")),
        label: (family != "property").then(|| "Doc".into()),
        property: if family == "vector" {
            "embedding"
        } else {
            "body"
        }
        .into(),
        kind: match family {
            "property" => IndexCreateKind::Property,
            "text" => IndexCreateKind::Text {
                // The empty-name graph is the omitted-option default control.
                min_token_length: (scope != 1).then_some(TEXT_MINIMUMS[scope]),
            },
            _ => IndexCreateKind::Vector {
                dimensions: Some(3),
                metric: Some("euclidean".into()),
                m: Some(4),
                ef_construction: Some(32),
                ef: None,
                quantization: Some("binary".into()),
            },
        },
    }
}

fn seed(db: &GrafeoDB) -> TestResult<EpochId> {
    let gap = db.create_index(CreateIndexRequest {
        graph: GraphPath::root(),
        name: Some("retired_owner".into()),
        label: None,
        property: "reserved".into(),
        kind: IndexCreateKind::Property,
    })?;
    assert_eq!(gap, IndexId::new(0));
    assert!(db.drop_index(gap)?);
    for (scope, path) in data_paths()?.iter().enumerate() {
        let mut session = db.session();
        session.use_graph_path(path)?;
        session.begin_transaction()?;
        let node = session.create_node_with_props(
            &["Doc", "Old"],
            [
                ("body", Value::from("old x ox archive")),
                ("scope", Value::Int64(i64::try_from(scope)?)),
                ("embedding", vector(true)),
            ],
        )?;
        let destination = session.create_node(&["Destination"]);
        let victim = session.create_node(&["Victim"]);
        let edge = session.create_edge_with_props(
            node,
            destination,
            "LINK",
            [("weight", Value::Int64(1))],
        )?;
        let doomed = session.create_edge(node, victim, "GONE");
        assert_eq!(
            (node, destination, victim),
            (NodeId::new(0), NodeId::new(1), NodeId::new(2))
        );
        assert_eq!((edge, doomed), (EdgeId::new(0), EdgeId::new(1)));
        session.commit()?;
        session.begin_transaction()?;
        session.set_node_property(node, "body", Value::from("baseline x ox archive"))?;
        assert!(session.remove_node_label(node, "Old"));
        assert!(session.add_node_label(node, "Base"));
        assert!(session.delete_node(victim));
        session.commit()?;
        session.begin_transaction()?;
        assert_eq!(session.create_node(&["Aborted"]), NodeId::new(3));
        assert_eq!(
            session.create_edge(node, destination, "ABORTED"),
            EdgeId::new(2)
        );
        session.rollback()?;
        drop(session);
        for (family_offset, family) in ["property", "text", "vector"].iter().enumerate() {
            let owner = db.create_index(index_request(path.clone(), scope, family))?;
            assert_eq!(
                owner.as_u32(),
                u32::try_from(1 + scope * 3 + family_offset)?
            );
        }
    }
    // Produce actual obsolete Text postings, then prune while the fixture is
    // memory-only. WAL-backed index GC is intentionally a separate refusal.
    let old_text_epoch = db.current_epoch();
    let before_gc = auxiliary(db)?.1;
    db.session().set_node_property(
        NodeId::new(0),
        "body",
        Value::from("baseline x ox archive refreshed"),
    )?;
    db.session()
        .set_node_property(NodeId::new(0), "body", Value::from("baseline x ox archive"))?;
    db.gc()?;
    assert_ne!(
        auxiliary(db)?.1,
        before_gc,
        "real Text GC must change the retained image"
    );
    let text = grafeo_engine::database::testing::root_lpg_store(db)
        .get_text_index("Doc", "body")
        .ok_or("root Text index missing")?;
    assert!(
        text.read()
            .doc_count_at(old_text_epoch, TransactionId::INVALID)
            .is_err()
    );

    let mut rdf = db.session();
    rdf.set_rdf_valid_time(Some(ValidTimeInterval::from_tai_nanoseconds(1_000, 2_000)?));
    rdf.begin_transaction()?;
    rdf.execute_sparql(&format!("CREATE GRAPH <{NAMED}>"))?;
    rdf.execute_sparql(&format!(
        r#"INSERT DATA {{
        <http://recursive.test/default> <http://recursive.test/p> "seed" .
        <http://recursive.test/person> a <{CLASS}> .
        GRAPH <{NAMED}> {{ <http://recursive.test/named-node> <http://recursive.test/p> "seed" . }}
    }}"#
    ))?;
    rdf.commit()?;
    drop(rdf);
    assert_eq!(
        db.declare_rdf_lpg_projection(CLASS, "Projected")?,
        projection_id()
    );
    assert_eq!(db.rebuild_rdf_lpg_projection(projection_id())?, 1);
    same_id_history::install(db)?;
    Ok(old_text_epoch)
}

fn tail(db: &GrafeoDB) -> TestResult {
    let mut session = db.session();
    session.begin_transaction()?;
    stage_tail(&session)?;
    let epoch = session.commit()?;
    drop(session);
    assert_eq!(db.current_epoch(), epoch);
    assert_eq!(db.rdf_store().commit_epoch(), epoch);
    for path in data_paths()? {
        assert_eq!(target(db, &path)?.current_epoch(), epoch);
    }
    Ok(())
}

fn stage_tail(session: &Session) -> TestResult {
    for path in &data_paths()? {
        session.use_graph_path(path)?;
        session.set_node_property(NodeId::new(0), "body", Value::from("tail x ox archive"))?;
        session.set_node_property(NodeId::new(100), "life", Value::from("tail-second"))?;
        session.set_edge_property(EdgeId::new(100), "life", Value::from("tail-second"))?;
        session.set_node_property(NodeId::new(0), "embedding", vector(false))?;
        assert!(session.remove_node_label(NodeId::new(0), "Base"));
        assert!(session.add_node_label(NodeId::new(0), "Tail"));
        session.set_edge_property(EdgeId::new(0), "weight", Value::Int64(2))?;
        let added = session.create_node_with_props(
            &["Doc"],
            [
                ("body", Value::from("tail x ox companion")),
                ("embedding", vector(true)),
            ],
        )?;
        assert_eq!(added, NodeId::new(102));
        assert_eq!(
            session.create_edge(NodeId::new(0), added, "REBORN"),
            EdgeId::new(101)
        );
        // All six LPG paths, both RDF lanes and the new owner share one marker.
        if path == &GraphPath::from_components(&["a", "b"])? {
            session
                .set_rdf_valid_time(Some(ValidTimeInterval::from_tai_nanoseconds(3_000, 4_000)?));
            session.execute_sparql(&format!(
                r#"DELETE DATA {{ GRAPH <{NAMED}> {{
                <http://recursive.test/named-node> <http://recursive.test/p> "seed" .
            }} }}"#
            ))?;
            session.execute_sparql(&format!(r#"INSERT DATA {{
                <http://recursive.test/tail> <http://recursive.test/p> "committed" .
                GRAPH <{NAMED}> {{ <http://recursive.test/named-node> <http://recursive.test/p> "tail" . }}
            }}"#))?;
        }
    }
    session.use_graph_path(&GraphPath::from_components(&["a", "b"])?)?;
    session.execute("CREATE INDEX tail_owner FOR (n:Doc) ON (n.tail_flag)")?;
    Ok(())
}

fn auxiliary(db: &GrafeoDB) -> TestResult<(Vec<u8>, Vec<u8>)> {
    let mut vectors = Vec::new();
    let mut texts = Vec::new();
    for (path, store) in LpgStoreSection::new(Arc::clone(
        grafeo_engine::database::testing::root_lpg_store(db),
    ))
    .capture_graphs()?
    {
        for (key, index) in store.vector_index_entries() {
            let (label, property) = decode_index_key(&key).ok_or("invalid Vector key")?;
            vectors.push((
                PhysicalIndexKey::vector(path.clone(), label, property),
                index,
            ));
        }
        for (key, index) in store.text_index_entries() {
            let (label, property) = decode_index_key(&key).ok_or("invalid Text key")?;
            texts.push((PhysicalIndexKey::text(path.clone(), label, property), index));
        }
    }
    Ok((
        VectorStoreSection::from_views(vectors).serialize()?,
        TextIndexSection::from_views(texts).serialize()?,
    ))
}

struct Witness {
    snapshot: Vec<u8>,
    auxiliary: (Vec<u8>, Vec<u8>),
    rdf: Vec<RdfQuadVersion>,
    projection: RdfLpgProjectionDefinition,
    scores: Vec<TextCutWitness>,
}

#[derive(Debug, PartialEq, Eq)]
struct TextCutWitness {
    documents: u64,
    total_length: u64,
    average_length_bits: u64,
    node0_score_bits: Vec<Option<u64>>,
    matches: Vec<Vec<(NodeId, u64)>>,
}

fn scores(db: &GrafeoDB, baseline: EpochId) -> TestResult<Vec<TextCutWitness>> {
    let mut scores = Vec::new();
    for path in data_paths()? {
        let store = target(db, &path)?;
        let graph = view(db, &path)?;
        let text = store.get_text_index("Doc", "body").ok_or("Text missing")?;
        for epoch in [baseline, db.current_epoch()] {
            let mut witness = {
                let text = text.read();
                let mut node0_score_bits = Vec::new();
                for query in TEXT_QUERIES {
                    node0_score_bits.push(
                        text.score_document_visible(
                            NodeId::new(0),
                            query,
                            epoch,
                            TransactionId::INVALID,
                            None,
                            false,
                        )?
                        .map(f64::to_bits),
                    );
                }
                TextCutWitness {
                    documents: text.doc_count_at(epoch, TransactionId::INVALID)?,
                    total_length: text.total_length_at(epoch, TransactionId::INVALID)?,
                    average_length_bits: text.avgdl_at(epoch, TransactionId::INVALID)?.to_bits(),
                    node0_score_bits,
                    matches: Vec::new(),
                }
            };
            for query in TEXT_QUERIES {
                let mut matches: Vec<_> = graph
                    .text_search_visible("Doc", "body", query, 10, epoch, TransactionId::INVALID)?
                    .into_iter()
                    .map(|(node, score)| (node, score.to_bits()))
                    .collect();
                matches.sort_by_key(|(node, _)| *node);
                witness.matches.push(matches);
            }
            scores.push(witness);
        }
    }
    Ok(scores)
}

fn expected_text_matches(query: &str, minimum: usize, has_tail: bool) -> Vec<NodeId> {
    let added = NodeId::new(102);
    let all = || {
        if has_tail {
            vec![NodeId::new(0), added]
        } else {
            vec![NodeId::new(0)]
        }
    };
    match query {
        "x" if minimum <= 1 => all(),
        "ox" if minimum <= 2 => all(),
        "baseline" if !has_tail => vec![NodeId::new(0)],
        "tail" if has_tail && minimum <= 4 => all(),
        "archive" => vec![NodeId::new(0)],
        "companion" if has_tail => vec![added],
        _ => Vec::new(),
    }
}

fn capture(db: &GrafeoDB, baseline: EpochId) -> TestResult<Witness> {
    Ok(Witness {
        snapshot: db.export_snapshot()?,
        auxiliary: auxiliary(db)?,
        rdf: db.rdf_dataset_history()?.quad_versions().to_vec(),
        projection: db
            .rdf_lpg_projection(projection_id())
            .ok_or("projection missing")?,
        scores: scores(db, baseline)?,
    })
}

fn assert_literal(
    db: &GrafeoDB,
    baseline: EpochId,
    old_text_epoch: EpochId,
    has_tail: bool,
    fork: bool,
) -> TestResult {
    assert_eq!(db.graph_model(), GraphModel::Both);
    same_id_history::assert_lives(db, baseline, has_tail)?;
    let topology: Vec<_> = LpgStoreSection::new(Arc::clone(
        grafeo_engine::database::testing::root_lpg_store(db),
    ))
    .capture_graphs()?
    .into_iter()
    .map(|(path, _)| path)
    .collect();
    assert_eq!(topology, topology_paths()?);
    for (scope, path) in data_paths()?.iter().enumerate() {
        let graph = view(db, path)?;
        let store = target(db, path)?;
        let node = graph
            .get_node(NodeId::new(0))
            .ok_or("literal node0 missing")?;
        assert_eq!(
            node.get_property("scope"),
            Some(&Value::Int64(i64::try_from(scope)?))
        );
        assert_eq!(
            node.get_property("body"),
            Some(&Value::from(if has_tail {
                "tail x ox archive"
            } else {
                "baseline x ox archive"
            }))
        );
        assert!(node.has_label(if has_tail { "Tail" } else { "Base" }));
        assert!(!node.has_label("Old"));
        let historical = graph
            .get_node_at_epoch(NodeId::new(0), baseline)
            .ok_or("retained baseline node missing")?;
        assert_eq!(
            historical.get_property("body"),
            Some(&Value::from("baseline x ox archive"))
        );
        assert!(historical.has_label("Base"));
        assert!(graph.get_node(NodeId::new(2)).is_none());
        assert!(
            graph.get_node(NodeId::new(3)).is_none(),
            "aborted identity must remain a gap"
        );
        assert!(graph.get_edge(EdgeId::new(1)).is_none());
        assert!(graph.get_edge(EdgeId::new(2)).is_none());
        let edge = graph
            .get_edge(EdgeId::new(0))
            .ok_or("literal edge0 missing")?;
        assert_eq!(
            edge.properties.get(&PropertyKey::new("weight")),
            Some(&Value::Int64(if has_tail { 2 } else { 1 }))
        );
        assert_eq!(store.next_edge_id(), 101 + u64::from(has_tail));
        let expected_next = 102 + u64::from(has_tail);
        assert_eq!(store.next_node_id(), expected_next);
        assert_eq!(
            graph.node_count(),
            4 + usize::from(has_tail) + usize::from(scope == 0 && !fork)
        );
        assert!(store.has_property_index("body"));
        assert_eq!(
            graph.find_nodes_by_property(
                "body",
                &Value::from(if has_tail {
                    "tail x ox archive"
                } else {
                    "baseline x ox archive"
                })
            ),
            vec![NodeId::new(0)]
        );
        let vector = store
            .get_vector_index("Doc", "embedding")
            .ok_or("Vector missing")?;
        assert_eq!(vector.config().dimensions, 3);
        assert_eq!(vector.config().metric, DistanceMetric::Euclidean);
        assert_eq!(vector.config().m, 4);
        assert_eq!(vector.config().ef_construction, 32);
        assert_eq!(vector.quantization_type(), Some(QuantizationType::Binary));
        assert_eq!(vector.len(), 1 + usize::from(has_tail));
        let hits = graph.vector_search_visible(
            "Doc",
            "embedding",
            if has_tail {
                &[0.0, 1.0, 0.0]
            } else {
                &[1.0, 0.0, 0.0]
            },
            10,
            db.current_epoch(),
            TransactionId::INVALID,
        );
        assert_eq!(hits.first().map(|(id, _)| *id), Some(NodeId::new(0)));
        let text = store.get_text_index("Doc", "body").ok_or("Text missing")?;
        let text = text.read();
        let minimum = TEXT_MINIMUMS[scope];
        assert!(
            text.has_simple_tokenizer(minimum),
            "wrong descriptor at {path:?}"
        );
        assert!(!text.has_simple_tokenizer(minimum + 1));
        let config = text.config();
        assert_eq!(
            (config.k1.to_bits(), config.b.to_bits()),
            (
                BM25Config::default().k1.to_bits(),
                BM25Config::default().b.to_bits()
            )
        );
        for (epoch, tail_visible) in [(baseline, false), (db.current_epoch(), has_tail)] {
            // Literal counts, independent of the implementation's tokenizer:
            // baseline/ox/archive; optionally x; min3/4 remove ox; min7 also
            // removes tail after the acknowledged mutation, retaining archive
            // and the new document's companion token.
            let document_length = match minimum {
                0 | 1 => 4_u32,
                2 => 3,
                3 | 4 => 2,
                7 => {
                    if tail_visible {
                        1
                    } else {
                        2
                    }
                }
                _ => return Err("unexpected fixture tokenizer minimum".into()),
            };
            let documents = 1 + u64::from(tail_visible);
            assert_eq!(
                text.doc_count_at(epoch, TransactionId::INVALID)?,
                documents,
                "document count at {path:?}, epoch {epoch}"
            );
            assert_eq!(
                text.total_length_at(epoch, TransactionId::INVALID)?,
                documents * u64::from(document_length),
                "total token count at {path:?}, epoch {epoch}"
            );
            assert_eq!(
                text.avgdl_at(epoch, TransactionId::INVALID)?.to_bits(),
                f64::from(document_length).to_bits(),
                "average token count at {path:?}, epoch {epoch}"
            );
            for query in TEXT_QUERIES {
                let expected = expected_text_matches(query, minimum, tail_visible);
                let score = text.score_document_visible(
                    NodeId::new(0),
                    query,
                    epoch,
                    TransactionId::INVALID,
                    None,
                    false,
                )?;
                assert_eq!(
                    score.is_some(),
                    query.len() >= minimum,
                    "visible node0 has a score for tokenized query {query:?} at {path:?}, epoch {epoch}"
                );
                assert_eq!(
                    score.is_some_and(|score| score > 0.0),
                    expected.contains(&NodeId::new(0)),
                    "node0 query {query:?} at {path:?}, epoch {epoch}"
                );
                assert!(score.is_none_or(|score| score.is_finite() && score >= 0.0));
            }
        }
        if scope == 0 {
            assert!(
                text.doc_count_at(old_text_epoch, TransactionId::INVALID)
                    .is_err()
            );
        }
        drop(text);
        for (epoch, tail_visible) in [(baseline, false), (db.current_epoch(), has_tail)] {
            for query in TEXT_QUERIES {
                let matches = graph.text_search_visible(
                    "Doc",
                    "body",
                    query,
                    10,
                    epoch,
                    TransactionId::INVALID,
                )?;
                assert!(
                    matches
                        .iter()
                        .all(|(_, score)| score.is_finite() && *score > 0.0)
                );
                let mut ids: Vec<_> = matches.into_iter().map(|(node, _)| node).collect();
                ids.sort_unstable();
                assert_eq!(
                    ids,
                    expected_text_matches(query, minimum, tail_visible),
                    "query {query:?} at {path:?}, epoch {epoch}"
                );
            }
        }
    }
    let mut names: Vec<_> = db
        .list_indexes()
        .into_iter()
        .map(|index| index.name)
        .collect();
    names.sort();
    let mut expected: Vec<_> = (0..6)
        .flat_map(|scope| {
            ["property", "text", "vector"].map(move |family| format!("scope_{scope}_{family}"))
        })
        .collect();
    if has_tail {
        expected.push("tail_owner".into());
    }
    expected.sort();
    assert_eq!(names, expected);
    let history = db.rdf_dataset_history()?;
    assert_eq!(history.graph_lives().len(), 1);
    assert_eq!(history.quad_versions().len(), if has_tail { 5 } else { 3 });
    assert_eq!(
        history
            .quad_versions()
            .iter()
            .filter(|version| !version.tx().is_open())
            .count(),
        usize::from(has_tail)
    );
    assert_eq!(
        history
            .quad_versions()
            .iter()
            .filter(|version| version.valid()
                == ValidTimeInterval::from_tai_nanoseconds(1_000, 2_000).ok())
            .count(),
        3
    );
    let definition = db
        .rdf_lpg_projection(projection_id())
        .ok_or("projection missing")?;
    if fork {
        assert_eq!(
            definition.reconciliation(),
            ProjectionReconciliationState::Pending
        );
        assert!(definition.receipt().is_none());
        assert!(db.graph_store().nodes_by_label("Projected").is_empty());
    } else {
        assert_eq!(definition.row_count(), 1);
        assert!(definition.receipt().is_some());
        assert_eq!(
            db.graph_store().nodes_by_label("Projected"),
            vec![NodeId::new(4)]
        );
    }
    Ok(())
}

fn assert_exact(
    db: &GrafeoDB,
    witness: &Witness,
    baseline: EpochId,
    old_text_epoch: EpochId,
    has_tail: bool,
) -> TestResult {
    assert_eq!(db.export_snapshot()?, witness.snapshot);
    assert_eq!(auxiliary(db)?, witness.auxiliary);
    assert_eq!(db.rdf_dataset_history()?.quad_versions(), witness.rdf);
    assert_eq!(
        db.rdf_lpg_projection(projection_id()).as_ref(),
        Some(&witness.projection)
    );
    assert_eq!(scores(db, baseline)?, witness.scores);
    assert_literal(db, baseline, old_text_epoch, has_tail, false)
}

fn run_child(path: &Path, oracle: &Path) -> TestResult {
    let mut child = Command::new(std::env::current_exe()?)
        .arg(TEST)
        .arg("--exact")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(CHILD, "1")
        .env(DATABASE, path)
        .env(ORACLE, oracle)
        .spawn()?;
    let deadline = Instant::now() + Duration::from_mins(1);
    loop {
        if let Some(status) = child.try_wait()? {
            assert!(status.success(), "acknowledged-tail child failed: {status}");
            return Ok(());
        }
        if Instant::now() >= deadline {
            child.kill()?;
            child.wait()?;
            return Err("recursive lifecycle child timed out".into());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn recursive_both_lifecycle_preserves_one_exact_cut() -> TestResult {
    if std::env::var(CHILD).ok().as_deref() == Some("1") {
        let db = persistent(Path::new(&std::env::var(DATABASE)?))?;
        tail(&db)?;
        db.wal().ok_or("child WAL missing")?.sync()?;
        std::fs::write(std::env::var(ORACLE)?, db.export_snapshot()?)?;
        // Sync commits have acknowledged; no database destructor/checkpoint runs.
        std::process::exit(0);
    }
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("recursive.grafeo");
    let oracle = temp.path().join("acknowledged.snapshot");
    let memory = blank_recursive_memory(&temp.path().join("topology.grafeo"))?;
    let old_text_epoch = seed(&memory)?;
    let baseline = memory.current_epoch();
    let initial = capture(&memory, baseline)?;
    assert_exact(&memory, &initial, baseline, old_text_epoch, false)?;
    memory.save(&path)?;
    memory.close()?;
    drop(memory);

    eprintln!("recursive lifecycle: checkpoint/close baseline");
    let db = persistent(&path)?;
    assert_exact(&db, &initial, baseline, old_text_epoch, false)?;
    db.wal_checkpoint()?;
    db.close()?;
    drop(db);
    let checkpoint = std::fs::read(&path)?;
    run_child(&path, &oracle)?;
    assert_eq!(
        std::fs::read(&path)?,
        checkpoint,
        "child must leave its acknowledged changes exclusively in the WAL tail"
    );

    eprintln!("recursive lifecycle: crash recovery and first compact");
    let mut db = persistent(&path)?;
    assert_eq!(db.export_snapshot()?, std::fs::read(&oracle)?);
    let recovered = capture(&db, baseline)?;
    assert_exact(&db, &recovered, baseline, old_text_epoch, true)?;
    db.compact()?;
    assert!(db.layered_store().is_some());
    assert_exact(&db, &recovered, baseline, old_text_epoch, true)?;
    db.wal_checkpoint()?;
    db.close()?;
    drop(db);

    eprintln!("recursive lifecycle: compact reopen and nonempty recompact");
    let mut db = persistent(&path)?;
    assert_exact(&db, &recovered, baseline, old_text_epoch, true)?;
    let before_base = db
        .layered_store()
        .ok_or("compact layer missing after reopen")?
        .base_store();
    same_id_history::mutate_for_recompact(&db)?;
    assert!(
        db.layered_store()
            .ok_or("layer missing")?
            .overlay_mutation_count()
            > 0
    );
    let changed = capture(&db, baseline)?;
    db.compact()?;
    let layer = db.layered_store().ok_or("recompact removed layer")?;
    assert!(
        !Arc::ptr_eq(&before_base, &layer.base_store()),
        "second compact must publish a new base"
    );
    assert_eq!(layer.overlay_mutation_count(), 0);
    assert_exact(&db, &changed, baseline, old_text_epoch, true)?;

    eprintln!("recursive lifecycle: exact portable import, save/open and fork");
    let imported = GrafeoDB::import_snapshot(&changed.snapshot)?;
    assert_eq!(imported.store_id(), db.store_id());
    assert_exact(&imported, &changed, baseline, old_text_epoch, true)?;
    let saved_path = temp.path().join("exact-extensionless-save");
    imported.save(&saved_path)?;
    let saved = GrafeoDB::open(&saved_path)?;
    assert_eq!(saved.store_id(), db.store_id());
    assert_exact(&saved, &changed, baseline, old_text_epoch, true)?;
    let fork = saved.to_memory()?;
    assert_ne!(fork.store_id(), saved.store_id());
    assert_eq!(auxiliary(&fork)?, changed.auxiliary);
    assert_eq!(scores(&fork, baseline)?, changed.scores);
    assert_literal(&fork, baseline, old_text_epoch, true, true)?;
    let fork_history = fork.rdf_dataset_history()?;
    // History is ordered by store-bound statement identity. Reminting that
    // identity may reorder records: compare a complete one-to-one semantic
    // history, including multiplicity, instead of pairing unrelated positions.
    let mut unmatched: Vec<_> = fork_history.quad_versions().iter().collect();
    assert_eq!(unmatched.len(), changed.rdf.len());
    for original in &changed.rdf {
        let position = unmatched
            .iter()
            .position(|copied| {
                (original.quad(), original.tx(), original.valid())
                    == (copied.quad(), copied.tx(), copied.valid())
            })
            .ok_or("fork lost or changed an RDF history record")?;
        let copied = unmatched.swap_remove(position);
        assert_ne!(original.statement(), copied.statement());
    }
    assert!(unmatched.is_empty());
    // Explicit owner and allocator witnesses after every preceding copy route.
    assert!(fork.drop_index(IndexId::new(19))?);
    let next_owner = fork.create_index(CreateIndexRequest {
        graph: GraphPath::root(),
        name: Some("floor_probe".into()),
        label: None,
        property: "floor_probe".into(),
        kind: IndexCreateKind::Property,
    })?;
    assert_eq!(next_owner.as_u32(), 20);
    assert_eq!(
        saved.export_snapshot()?,
        changed.snapshot,
        "fork edits cannot mutate source"
    );
    saved.close()?;
    imported.close()?;
    db.close()?;
    Ok(())
}

#[cfg(feature = "testing-crash-injection")]
mod hard_crash {
    use super::{
        EpochId, GrafeoDB, IndexId, Path, PathBuf, Session, TestResult, assert_exact,
        assert_literal, blank_recursive_memory, capture, persistent, seed, stage_tail, tail,
        target,
    };
    use grafeo_common::testing::crash::{disable_crash, enable_crash_named};
    use grafeo_common::types::GraphPath;
    use std::io::Write;
    use std::process::Command;
    use std::time::{Duration, Instant};

    const TEST: &str = "hard_crash::named_recursive_hard_crash_matrix";
    const CASE: &str = "GRAFEO_RECURSIVE_CRASH_CASE";
    const DATABASE: &str = "GRAFEO_RECURSIVE_CRASH_DATABASE";
    const MARKER: &str = "GRAFEO_RECURSIVE_CRASH_MARKER";
    const CLEANUP: &str = "GRAFEO_RECURSIVE_CRASH_CLEANUP";
    const EXPECTED_EXIT: i32 = 86;

    #[derive(Clone, Copy)]
    enum Phase {
        Tail,
        Checkpoint,
        FirstCompact,
        Recompact,
    }

    #[derive(Clone, Copy)]
    struct Case {
        site: &'static str,
        phase: Phase,
        committed: bool,
    }

    const CASES: [Case; 15] = [
        Case {
            site: "wal_before_write",
            phase: Phase::Tail,
            committed: false,
        },
        Case {
            site: "commit:before_marker",
            phase: Phase::Tail,
            committed: false,
        },
        Case {
            site: "commit:after_marker_before_publication",
            phase: Phase::Tail,
            committed: true,
        },
        Case {
            site: "commit:after_lpg_before_rdf",
            phase: Phase::Tail,
            committed: true,
        },
        Case {
            site: "write_sections:before_data",
            phase: Phase::Checkpoint,
            committed: true,
        },
        Case {
            site: "write_sections:after_data",
            phase: Phase::Checkpoint,
            committed: true,
        },
        Case {
            site: "write_sections:after_directory",
            phase: Phase::Checkpoint,
            committed: true,
        },
        Case {
            site: "write_sections:after_fsync",
            phase: Phase::Checkpoint,
            committed: true,
        },
        Case {
            site: "checkpoint:after_snapshot_before_wal_retire",
            phase: Phase::Checkpoint,
            committed: true,
        },
        Case {
            site: "wal_checkpoint_after_metadata",
            phase: Phase::Checkpoint,
            committed: true,
        },
        Case {
            site: "wal_checkpoint_after_truncate",
            phase: Phase::Checkpoint,
            committed: true,
        },
        Case {
            site: "compact:before_representation_transfer",
            phase: Phase::FirstCompact,
            committed: true,
        },
        Case {
            site: "compact:after_representation_transfer",
            phase: Phase::FirstCompact,
            committed: true,
        },
        Case {
            site: "compact:before_representation_transfer",
            phase: Phase::Recompact,
            committed: true,
        },
        Case {
            site: "compact:after_representation_transfer",
            phase: Phase::Recompact,
            committed: true,
        },
    ];

    struct UnexpectedCleanup(PathBuf);

    impl Drop for UnexpectedCleanup {
        fn drop(&mut self) {
            if let Err(error) = std::fs::write(&self.0, b"unexpected unwind or normal cleanup") {
                eprintln!("could not record unexpected cleanup: {error}");
            }
        }
    }

    fn arm(site: &'static str, marker: PathBuf) {
        let expected = format!("crash injection at: {site}");
        std::panic::set_hook(Box::new(move |info| {
            let message = info
                .payload()
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| info.payload().downcast_ref::<&str>().copied());
            if message == Some(expected.as_str()) {
                let written = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&marker)
                    .and_then(|mut file| {
                        file.write_all(site.as_bytes())?;
                        file.sync_all()
                    });
                if written.is_ok() {
                    // Deliberately exit inside the hook, before stack unwinding
                    // or database/session cleanup. This is a process-crash
                    // control, not a power-loss simulation.
                    std::process::exit(EXPECTED_EXIT);
                }
                std::process::exit(87);
            }
            eprintln!("unexpected panic while awaiting {site}: {info}");
            std::process::exit(88);
        }));
        enable_crash_named(site);
    }

    fn prepare_recompact(db: &mut GrafeoDB) -> TestResult {
        db.compact()?;
        assert!(db.layered_store().is_some());
        super::same_id_history::mutate_for_recompact(db)?;
        assert!(
            db.layered_store()
                .ok_or("recompact layer missing")?
                .overlay_mutation_count()
                > 0
        );
        Ok(())
    }

    fn crash_operation(
        db: &mut GrafeoDB,
        session: Option<&mut Session>,
        phase: Phase,
    ) -> TestResult {
        match phase {
            Phase::Tail => {
                let session = session.ok_or("tail transaction missing")?;
                stage_tail(session)?;
                session.commit()?;
            }
            Phase::Checkpoint => db.wal_checkpoint()?,
            Phase::FirstCompact | Phase::Recompact => db.compact()?,
        }
        Ok(())
    }

    fn child(case: Case) -> TestResult {
        let mut db = persistent(Path::new(&std::env::var(DATABASE)?))?;
        let mut transaction = if matches!(case.phase, Phase::Tail) {
            let mut session = db.session();
            session.begin_transaction()?;
            Some(session)
        } else {
            tail(&db)?;
            if matches!(case.phase, Phase::Recompact) {
                prepare_recompact(&mut db)?;
            }
            db.wal().ok_or("child WAL missing")?.sync()?;
            None
        };
        let _cleanup = UnexpectedCleanup(PathBuf::from(std::env::var(CLEANUP)?));
        // Open, BEGIN and acknowledged-tail preparation cannot consume the
        // injection: only this thread is armed, immediately before the target.
        arm(case.site, PathBuf::from(std::env::var(MARKER)?));
        let result = crash_operation(&mut db, transaction.as_mut(), case.phase);
        disable_crash();
        eprintln!("named crash site {} was not reached: {result:?}", case.site);
        // A missing site or earlier operation error must not run destructors,
        // where a close/checkpoint could otherwise falsely hit the target.
        std::process::exit(89);
    }

    fn run_child(case_index: usize, path: &Path, marker: &Path, cleanup: &Path) -> TestResult {
        assert!(!marker.exists() && !cleanup.exists());
        let mut child = Command::new(std::env::current_exe()?)
            .arg(TEST)
            .arg("--exact")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(CASE, case_index.to_string())
            .env(DATABASE, path)
            .env(MARKER, marker)
            .env(CLEANUP, cleanup)
            .env_remove("GRAFEO_CRASH_NAMED")
            .spawn()?;
        let deadline = Instant::now() + Duration::from_mins(1);
        loop {
            if let Some(status) = child.try_wait()? {
                assert_eq!(
                    status.code(),
                    Some(EXPECTED_EXIT),
                    "wrong exit at {}",
                    CASES[case_index].site
                );
                assert_eq!(std::fs::read(marker)?, CASES[case_index].site.as_bytes());
                assert!(
                    !cleanup.exists(),
                    "crash must not unwind or close the database"
                );
                return Ok(());
            }
            if Instant::now() >= deadline {
                child.kill()?;
                child.wait()?;
                return Err(
                    format!("named crash {} timed out after 60s", CASES[case_index].site).into(),
                );
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn assert_owner19(snapshot: &[u8]) -> TestResult {
        let replica = GrafeoDB::import_snapshot(snapshot)?;
        let nested = GraphPath::from_components(&["a", "b"])?;
        assert!(target(&replica, &nested)?.has_property_index("tail_flag"));
        assert!(replica.drop_index(IndexId::new(19))?);
        assert!(!target(&replica, &nested)?.has_property_index("tail_flag"));
        replica.close()?;
        Ok(())
    }

    #[test]
    fn named_recursive_hard_crash_matrix() -> TestResult {
        if let Ok(case) = std::env::var(CASE) {
            let case: usize = case.parse()?;
            return child(*CASES.get(case).ok_or("unknown crash case")?);
        }
        let temp = tempfile::tempdir()?;
        let seeded = blank_recursive_memory(&temp.path().join("topology.grafeo"))?;
        let old_text_epoch = seed(&seeded)?;
        let baseline: EpochId = seeded.current_epoch();
        let seed_image = seeded.export_snapshot()?;
        seeded.close()?;
        for (case_index, case) in CASES.into_iter().enumerate() {
            eprintln!("recursive hard crash {case_index}: {}", case.site);
            let case_dir = temp.path().join(format!("case-{case_index}"));
            std::fs::create_dir(&case_dir)?;
            let path = case_dir.join("crashed.grafeo");
            let oracle_path = case_dir.join("oracle.grafeo");
            // Both independent containers start with one exact imported cut
            // and the same persisted transaction allocator, not just matching
            // current rows or independently entropy-seeded indexes.
            let source = GrafeoDB::import_snapshot(&seed_image)?;
            source.save(&path)?;
            source.save(&oracle_path)?;
            source.close()?;
            let mut oracle = persistent(&oracle_path)?;
            let old = capture(&oracle, baseline)?;
            assert_literal(&oracle, baseline, old_text_epoch, false, false)?;
            tail(&oracle)?;
            if matches!(case.phase, Phase::Recompact) {
                prepare_recompact(&mut oracle)?;
            }
            match case.phase {
                Phase::Checkpoint => oracle.wal_checkpoint()?,
                Phase::FirstCompact | Phase::Recompact => oracle.compact()?,
                Phase::Tail => {}
            }
            let committed = capture(&oracle, baseline)?;
            assert_literal(&oracle, baseline, old_text_epoch, true, false)?;
            assert_owner19(&committed.snapshot)?;
            oracle.close()?;
            drop(oracle);
            run_child(
                case_index,
                &path,
                &case_dir.join("site.marker"),
                &case_dir.join("cleanup.marker"),
            )?;
            let expected = if case.committed { &committed } else { &old };
            let recovered = persistent(&path)?;
            assert_exact(
                &recovered,
                expected,
                baseline,
                old_text_epoch,
                case.committed,
            )?;
            recovered.close()?;
            drop(recovered);
            let reopened = persistent(&path)?;
            assert_exact(
                &reopened,
                expected,
                baseline,
                old_text_epoch,
                case.committed,
            )?;
            reopened.close()?;
        }
        Ok(())
    }
}
