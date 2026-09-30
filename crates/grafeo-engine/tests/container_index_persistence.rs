//! End-to-end index persistence through the `.grafeo` container loader.
//!
//! Snapshot-object round trips exercise a different restore path. These tests
//! deliberately close and reopen the single-file database so the catalog,
//! LPG, vector-topology, and text-postings sections must cooperate.

#![cfg(all(feature = "lpg", feature = "wal", feature = "grafeo-file"))]

use grafeo_common::types::Value;
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};

#[cfg(all(feature = "vector-index", feature = "text-index"))]
use grafeo_common::storage::{Section, SectionType};
#[cfg(all(feature = "vector-index", feature = "text-index"))]
use grafeo_core::graph::lpg::{LpgStoreSection, PhysicalIndexKey, decode_index_key};
#[cfg(all(feature = "vector-index", feature = "text-index"))]
use grafeo_storage::file::GrafeoFileManager;

#[cfg(all(feature = "vector-index", feature = "text-index"))]
type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn persistent_sync(path: &std::path::Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .expect("open persistent LPG database")
}

#[cfg(all(feature = "vector-index", feature = "text-index"))]
fn exact_auxiliary_images(db: &GrafeoDB, named_graph: &str) -> TestResult<(Vec<u8>, Vec<u8>)> {
    grafeo_engine::database::testing::root_lpg_store(db)
        .graph(named_graph)
        .ok_or("missing named graph store")?;
    recursive_auxiliary_images(db)
}

#[cfg(all(feature = "vector-index", feature = "text-index"))]
fn recursive_auxiliary_images(db: &GrafeoDB) -> TestResult<(Vec<u8>, Vec<u8>)> {
    // These fixtures are quiescent. Production capture additionally retains
    // the engine publication barrier across all of these section reads.
    let graphs = LpgStoreSection::new(std::sync::Arc::clone(
        grafeo_engine::database::testing::root_lpg_store(db),
    ))
    .capture_graphs()?;
    let mut vectors = Vec::new();
    let mut texts = Vec::new();
    for (path, graph) in graphs {
        for (key, index) in graph.vector_index_entries() {
            let (label, property) = decode_index_key(&key).ok_or("invalid local vector key")?;
            vectors.push((
                PhysicalIndexKey::vector(path.clone(), label, property),
                index,
            ));
        }
        for (key, index) in graph.text_index_entries() {
            let (label, property) = decode_index_key(&key).ok_or("invalid local text key")?;
            texts.push((PhysicalIndexKey::text(path.clone(), label, property), index));
        }
    }
    let vector = grafeo_core::index::vector::VectorStoreSection::from_views(vectors).serialize()?;
    let text = grafeo_core::index::text::TextIndexSection::from_views(texts).serialize()?;
    Ok((vector, text))
}

#[cfg(all(feature = "vector-index", feature = "text-index"))]
fn persisted_catalog_and_index_images(
    path: &std::path::Path,
) -> TestResult<Vec<(SectionType, u8, Vec<u8>)>> {
    let manager = GrafeoFileManager::open_read_only(path)?;
    let directory = manager
        .read_section_directory()?
        .ok_or("missing persisted section directory")?;
    let images = directory
        .entries()
        .iter()
        .filter(|entry| {
            matches!(
                entry.section_type,
                SectionType::Catalog | SectionType::VectorStore | SectionType::TextIndex
            )
        })
        .map(|entry| {
            Ok((
                entry.section_type,
                entry.version,
                manager.read_section_data(entry)?,
            ))
        })
        .collect::<grafeo_common::utils::error::Result<Vec<_>>>()?;
    manager.close()?;
    Ok(images)
}

#[test]
fn property_index_survives_container_close_and_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("property-index.grafeo");
    let indexed;

    {
        let db = persistent_sync(&path);
        indexed =
            db.create_node_with_props(&["Person"], [("email", Value::from("alix@example.com"))]);
        db.create_node_with_props(&["Person"], [("email", Value::from("gus@example.com"))]);
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: None,
            property: "email".into(),
            kind: grafeo_engine::IndexCreateKind::Property,
        })
        .expect("create property index");
        db.close().expect("checkpoint property index");
    }

    let db = GrafeoDB::open(&path).expect("reopen property index");
    assert!(db.has_property_index("email"));
    assert_eq!(
        db.find_nodes_by_property("email", &Value::from("alix@example.com")),
        vec![indexed]
    );
}

#[cfg(feature = "vector-index")]
#[test]
fn vector_index_survives_container_close_and_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("vector-index.grafeo");
    let nearest;

    {
        let db = persistent_sync(&path);
        nearest = db.create_node_with_props(
            &["Doc"],
            [("embedding", Value::Vector(vec![1.0, 0.0, 0.0].into()))],
        );
        db.create_node_with_props(
            &["Doc"],
            [("embedding", Value::Vector(vec![0.0, 1.0, 0.0].into()))],
        );
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Doc".into()),
            property: "embedding".into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(3),
                metric: Some("cosine".into()),
                m: Some(8),
                ef_construction: Some(64),
                ef: None,
                quantization: None,
            },
        })
        .expect("create vector index");
        db.close().expect("checkpoint vector index");
    }

    let db = GrafeoDB::open(&path).expect("reopen vector index");
    let results = db
        .vector_search("Doc", "embedding", &[1.0, 0.0, 0.0], 2, None, None)
        .expect("search restored vector index");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, nearest);
}

#[cfg(all(feature = "vector-index", feature = "gql"))]
#[test]
fn quantized_vector_index_rebuilds_gql_list_vectors_on_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("quantized-list-index.grafeo");

    {
        let db = persistent_sync(&path);
        db.execute("CREATE (:Doc {embedding: [1.0, 0.0, 0.0]})")
            .expect("create first list-backed vector");
        db.execute("CREATE (:Doc {embedding: [0.0, 1.0, 0.0]})")
            .expect("create second list-backed vector");
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Doc".into()),
            property: "embedding".into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(3),
                metric: Some("cosine".into()),
                m: Some(8),
                ef_construction: Some(64),
                ef: None,
                quantization: Some("scalar".into()),
            },
        })
        .expect("create quantized vector index over GQL list values");
        db.close().expect("checkpoint quantized vector index");
    }

    let db = GrafeoDB::open(&path).expect("reopen quantized vector index");
    let results = db
        .vector_search("Doc", "embedding", &[1.0, 0.0, 0.0], 2, None, None)
        .expect("search rebuilt quantized vector index");
    assert_eq!(results.len(), 2, "all list-backed vectors were rebuilt");
    assert!(
        results[0].1 <= 0.01,
        "the exact list-backed vector remains the nearest neighbour"
    );
}

#[cfg(feature = "text-index")]
#[test]
fn text_index_survives_container_close_and_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("text-index.grafeo");
    let matching;

    {
        let db = persistent_sync(&path);
        matching =
            db.create_node_with_props(&["Article"], [("body", Value::from("rust graph database"))]);
        db.create_node_with_props(
            &["Article"],
            [("body", Value::from("python web framework"))],
        );
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Article".into()),
            property: "body".into(),
            kind: grafeo_engine::IndexCreateKind::Text {
                min_token_length: None,
            },
        })
        .expect("create text index");
        db.close().expect("checkpoint text index");
    }

    let db = GrafeoDB::open(&path).expect("reopen text index");
    let results = db
        .text_search("Article", "body", "graph database", 10)
        .expect("search restored text index");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, matching);
}

#[cfg(all(feature = "vector-index", feature = "text-index"))]
#[test]
fn named_and_default_index_images_remain_exact_and_scope_distinct_on_reopen() -> TestResult {
    use grafeo_core::index::vector::PropertyVectorAccessor;

    // The parser-free create API intentionally rejects an empty name. Seed the
    // legal low-level graph through an unsealed in-memory store so persistence
    // still proves that `None` (default) and `Some("")` are never conflated.
    const NAMED: &str = "";

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("graph-qualified-indexes.grafeo");
    let default_vector;
    let named_vector;
    let before;

    {
        let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg))
            .expect("create unsealed fixture database");
        default_vector = db.create_node_with_props(
            &["Doc"],
            [
                ("embedding", Value::Vector(vec![0.0, 1.0, 0.0].into())),
                ("body", Value::from("defaultscope needle")),
            ],
        );
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Doc".into()),
            property: "embedding".into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(3),
                metric: Some("cosine".into()),
                m: Some(8),
                ef_construction: Some(64),
                ef: None,
                quantization: None,
            },
        })
        .expect("create default vector index");
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Doc".into()),
            property: "body".into(),
            kind: grafeo_engine::IndexCreateKind::Text {
                min_token_length: None,
            },
        })
        .expect("create default text index");

        assert!(
            grafeo_engine::database::testing::root_lpg_store(&db)
                .create_graph(NAMED)
                .expect("seed empty-name graph")
        );
        db.set_current_graph(Some(NAMED))
            .expect("select empty-name graph");
        assert_eq!(db.current_graph().as_deref(), Some(NAMED));
        named_vector = db.create_node_with_props(
            &["Doc"],
            [
                ("embedding", Value::Vector(vec![1.0, 0.0, 0.0].into())),
                ("body", Value::from("namedscope needle")),
            ],
        );
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: grafeo_common::types::GraphPath::from_components(&[NAMED])
                .expect("exact named graph path"),
            name: None,
            label: Some("Doc".into()),
            property: "embedding".into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(3),
                metric: Some("cosine".into()),
                m: Some(8),
                ef_construction: Some(64),
                ef: None,
                quantization: None,
            },
        })
        .expect("create named vector index");
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: grafeo_common::types::GraphPath::from_components(&[NAMED])
                .expect("exact named graph path"),
            name: None,
            label: Some("Doc".into()),
            property: "body".into(),
            kind: grafeo_engine::IndexCreateKind::Text {
                min_token_length: None,
            },
        })
        .expect("create named text index");

        before = exact_auxiliary_images(&db, NAMED)?;
        db.save(&path).expect("persist graph-qualified fixture");
        db.close().expect("close graph-qualified fixture");
    }
    let persisted_before = persisted_catalog_and_index_images(&path)?;

    let db = GrafeoDB::open(&path).expect("reopen graph-qualified indexes");
    let after = exact_auxiliary_images(&db, NAMED)?;
    assert_eq!(after, before, "exact index section images must round-trip");

    db.set_current_graph(Some(NAMED))
        .expect("restore empty-name graph context");
    assert_eq!(
        db.current_graph().as_deref(),
        Some(NAMED),
        "an empty named scope must remain distinct from the default graph"
    );

    db.set_current_graph(None)
        .expect("restore default graph context");
    assert_eq!(db.current_graph(), None);

    let default_results = db
        .vector_search("Doc", "embedding", &[0.0, 1.0, 0.0], 1, None, None)
        .expect("search restored default vector index");
    assert_eq!(default_results[0].0, default_vector);
    assert!(
        db.text_search("Doc", "body", "namedscope", 10)
            .expect("search restored default text index")
            .is_empty(),
        "the default index must not absorb the named graph image"
    );

    let named = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph(NAMED)
        .expect("restored named graph");
    let named_index = named
        .get_vector_index("Doc", "embedding")
        .expect("restored named vector index");
    let accessor = PropertyVectorAccessor::new(named.as_ref(), "embedding");
    let direct_named_results = named_index.search(&[1.0, 0.0, 0.0], 1, &accessor);
    assert_eq!(direct_named_results[0].0, named_vector);
    let named_text = named
        .get_text_index("Doc", "body")
        .expect("restored named text index");
    assert_eq!(named_text.read().search("namedscope", 10).len(), 1);

    db.close().expect("republish restored exact indexes");
    assert_eq!(
        persisted_catalog_and_index_images(&path)?,
        persisted_before,
        "close/reopen must republish byte-exact catalog and index sections"
    );
    Ok(())
}
