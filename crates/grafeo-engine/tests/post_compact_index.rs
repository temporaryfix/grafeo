//! Tests for index operations after compact().
//!
//! Validates that vector and text indexes work correctly when the database
//! is in layered mode (after compact()), including the full cycle of
//! insert → compact → create index → search.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test post_compact_index
//! ```

#![cfg(all(feature = "compact-store", feature = "lpg"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[cfg(feature = "vector-index")]
fn vec3(x: f32, y: f32, z: f32) -> Value {
    Value::Vector(vec![x, y, z].into())
}

/// Helper: create a DB with vector data, compact it, then return it.
#[cfg(feature = "vector-index")]
fn setup_compacted_vector_db() -> GrafeoDB {
    let mut db = GrafeoDB::new_in_memory();

    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "embedding", vec3(1.0, 0.0, 0.0))
        .expect("set node property");
    db.set_node_property(n1, "title", Value::String("alpha".into()))
        .expect("set node property");

    let n2 = db.create_node(&["Doc"]);
    db.set_node_property(n2, "embedding", vec3(0.0, 1.0, 0.0))
        .expect("set node property");
    db.set_node_property(n2, "title", Value::String("beta".into()))
        .expect("set node property");

    let n3 = db.create_node(&["Doc"]);
    db.set_node_property(n3, "embedding", vec3(0.0, 0.0, 1.0))
        .expect("set node property");
    db.set_node_property(n3, "title", Value::String("gamma".into()))
        .expect("set node property");

    db.compact().expect("compact");
    db
}

// ── Vector index after compact ─────────────────────────────────────

#[test]
#[cfg(feature = "vector-index")]
fn vector_index_after_compact_returns_results() {
    let db = setup_compacted_vector_db();

    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
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
    .expect("create vector index after compact");

    let results = db
        .vector_search("Doc", "embedding", &[1.0, 0.0, 0.0], 3, None, None)
        .expect("vector search after compact");

    assert_eq!(results.len(), 3, "should find all 3 pre-compact nodes");
}

#[test]
#[cfg(feature = "vector-index")]
fn vector_index_after_compact_nearest_neighbor_is_correct() {
    let db = setup_compacted_vector_db();

    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
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
    .expect("create vector index");

    let results = db
        .vector_search("Doc", "embedding", &[1.0, 0.0, 0.0], 1, None, None)
        .expect("search");

    assert_eq!(results.len(), 1);
    // Nearest to [1,0,0] should be node 1 (the one with [1,0,0])
    let (nearest_id, distance) = results[0];
    assert!(
        distance < 0.01,
        "exact match should have near-zero distance, got {distance}"
    );

    // Verify it's the right node by checking its property
    let title = db
        .graph_store()
        .get_node_property(nearest_id, &grafeo_common::types::PropertyKey::new("title"));
    assert_eq!(title, Some(Value::String("alpha".into())));
}

#[test]
#[cfg(feature = "vector-index")]
fn rebuild_vector_index_after_compact() {
    let db = setup_compacted_vector_db();

    let owner = db
        .create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Doc".into()),
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
        .expect("create");

    db.rebuild_index(owner).expect("rebuild after compact");

    let results = db
        .vector_search("Doc", "embedding", &[0.0, 1.0, 0.0], 3, None, None)
        .expect("search after rebuild");

    assert_eq!(results.len(), 3);
}

#[test]
#[cfg(feature = "vector-index")]
fn vector_search_with_filter_after_compact() {
    let mut db = GrafeoDB::new_in_memory();

    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "embedding", vec3(1.0, 0.0, 0.0))
        .expect("set node property");
    db.set_node_property(n1, "category", Value::String("science".into()))
        .expect("set node property");

    let n2 = db.create_node(&["Doc"]);
    db.set_node_property(n2, "embedding", vec3(0.0, 1.0, 0.0))
        .expect("set node property");
    db.set_node_property(n2, "category", Value::String("art".into()))
        .expect("set node property");

    let n3 = db.create_node(&["Doc"]);
    db.set_node_property(n3, "embedding", vec3(0.0, 0.0, 1.0))
        .expect("set node property");
    db.set_node_property(n3, "category", Value::String("science".into()))
        .expect("set node property");

    db.compact().expect("compact");
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: None,
        property: "category".into(),
        kind: grafeo_engine::IndexCreateKind::Property,
    })
    .expect("create property index");
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
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
    .expect("create vector index");

    let mut filters = std::collections::HashMap::new();
    filters.insert("category".to_string(), Value::String("science".into()));

    let results = db
        .vector_search(
            "Doc",
            "embedding",
            &[1.0, 0.0, 0.0],
            3,
            None,
            Some(&filters),
        )
        .expect("filtered search");

    assert_eq!(results.len(), 2, "should only return science nodes");
}

// ── Text index after compact ───────────────────────────────────────

#[test]
#[cfg(feature = "text-index")]
fn text_index_after_compact() {
    let mut db = GrafeoDB::new_in_memory();

    let n1 = db.create_node(&["Article"]);
    db.set_node_property(
        n1,
        "body",
        Value::String("the quick brown fox jumps over the lazy dog".into()),
    )
    .expect("set node property");

    let n2 = db.create_node(&["Article"]);
    db.set_node_property(
        n2,
        "body",
        Value::String("a fast brown fox leaps over a sleepy hound".into()),
    )
    .expect("set node property");

    let n3 = db.create_node(&["Article"]);
    db.set_node_property(n3, "body", Value::String("the cat sat on the mat".into()))
        .expect("set node property");

    db.compact().expect("compact");

    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Article".into()),
        property: "body".into(),
        kind: grafeo_engine::IndexCreateKind::Text {
            min_token_length: None,
        },
    })
    .expect("create text index after compact");

    let results = db
        .text_search("Article", "body", "fox", 10)
        .expect("text search");
    assert_eq!(
        results.len(),
        2,
        "should find both fox articles from pre-compact data"
    );
}

// ── Snapshot round-trip ────────────────────────────────────────────

#[test]
#[cfg(feature = "vector-index")]
fn snapshot_compact_vector_index_round_trip() {
    // Phase 1: build DB, export snapshot
    let snapshot_bytes = {
        let db = GrafeoDB::new_in_memory();
        let n1 = db.create_node(&["Doc"]);
        db.set_node_property(n1, "embedding", vec3(1.0, 0.0, 0.0))
            .expect("set node property");
        let n2 = db.create_node(&["Doc"]);
        db.set_node_property(n2, "embedding", vec3(0.0, 1.0, 0.0))
            .expect("set node property");
        let n3 = db.create_node(&["Doc"]);
        db.set_node_property(n3, "embedding", vec3(0.0, 0.0, 1.0))
            .expect("set node property");
        db.export_snapshot().expect("export")
    };

    // Phase 2: import → compact → create index → search
    let mut db = GrafeoDB::import_snapshot(&snapshot_bytes).expect("import");
    db.compact().expect("compact after import");

    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
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
    .expect("create vector index after snapshot+compact");

    let results = db
        .vector_search("Doc", "embedding", &[1.0, 0.0, 0.0], 3, None, None)
        .expect("vector search after snapshot+compact");

    assert_eq!(results.len(), 3, "should find all nodes from snapshot");
}

// ── LayeredStore trait method forwarding ────────────────────────────

#[test]
#[cfg(feature = "vector-index")]
fn layered_store_has_vector_index_forwards_to_overlay() {
    let mut db = GrafeoDB::new_in_memory();

    let n = db.create_node(&["Doc"]);
    db.set_node_property(n, "embedding", vec3(1.0, 0.0, 0.0))
        .expect("set node property");

    db.compact().expect("compact");

    // No index yet — graph_store (LayeredStore) should report false
    let gs = db.graph_store();
    assert!(!gs.has_vector_index("Doc", "embedding"));

    // Create index on the overlay via the imperative API
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
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
    .expect("create");

    // Now LayeredStore should forward to overlay and report true
    let gs = db.graph_store();
    assert!(gs.has_vector_index("Doc", "embedding"));
    assert!(gs.vector_index_metric("Doc", "embedding").is_some());
}

#[test]
#[cfg(feature = "text-index")]
fn layered_store_has_text_index_forwards_to_overlay() {
    let mut db = GrafeoDB::new_in_memory();

    let n = db.create_node(&["Article"]);
    db.set_node_property(n, "body", Value::String("hello world".into()))
        .expect("set node property");

    db.compact().expect("compact");

    let gs = db.graph_store();
    assert!(!gs.has_text_index("Article", "body"));

    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Article".into()),
        property: "body".into(),
        kind: grafeo_engine::IndexCreateKind::Text {
            min_token_length: None,
        },
    })
    .expect("create");

    let gs = db.graph_store();
    assert!(gs.has_text_index("Article", "body"));
}

// ── recompact() ────────────────────────────────────────────────────

#[test]
fn property_index_created_before_compaction_survives_both_merges() {
    let mut db = GrafeoDB::new_in_memory();
    let first = db.create_node_with_props(&["Doc"], [("kind", Value::from("kept"))]);
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: None,
        property: "kind".into(),
        kind: grafeo_engine::IndexCreateKind::Property,
    })
    .expect("create property index");

    db.compact().expect("compact with property index");
    assert!(db.has_property_index("kind"));
    assert_eq!(
        db.find_nodes_by_property("kind", &Value::from("kept")),
        vec![first]
    );

    let second = db.create_node_with_props(&["Doc"], [("kind", Value::from("kept"))]);
    db.compact().expect("recompact with property index");
    assert!(db.has_property_index("kind"));
    let mut found = db.find_nodes_by_property("kind", &Value::from("kept"));
    found.sort_unstable();
    assert_eq!(found, vec![first, second]);
}

#[test]
#[cfg(feature = "vector-index")]
fn compact_moves_exact_physical_index_without_descriptor_rebuild() {
    use grafeo_core::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
    use std::sync::Arc;

    let mut db = GrafeoDB::new_in_memory();
    let first = db.create_node_with_props(&["Doc"], [("embedding", vec3(1.0, 0.0, 0.0))]);
    let second = db.create_node_with_props(&["Doc"], [("embedding", vec3(0.0, 1.0, 0.0))]);

    // Build a deliberately sparse raw registry whose declared capacity is
    // below the authoritative matching-row count. This cannot be created
    // through Session DDL. Runtime compaction must nevertheless move this
    // exact physical object; rebuilding from its descriptor would either fail
    // or silently manufacture a different index.
    let config = HnswConfig::new(3, DistanceMetric::Cosine).with_max_elements(1);
    grafeo_engine::database::testing::root_lpg_store(&db).add_vector_index(
        "Doc",
        "embedding",
        Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config))),
    );

    let before = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_vector_index("Doc", "embedding")
        .expect("raw vector index before compact");
    assert_eq!(before.config().max_elements, Some(1));
    assert_eq!(before.len(), 0, "raw fixture deliberately has no vectors");

    db.compact().expect("move exact sparse physical index");

    let after = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_vector_index("Doc", "embedding")
        .expect("vector index after compact");
    assert_eq!(after.config().max_elements, Some(1));
    assert_eq!(
        after.len(),
        0,
        "compaction must not descriptor-rebuild matching graph rows"
    );
    assert!(db.get_node(first).is_some());
    assert!(db.get_node(second).is_some());
    let admitted = db.create_node_with_props(&["Doc"], [("embedding", vec3(0.0, 0.0, 1.0))]);
    assert!(admitted.is_valid());
    assert!(before.contains(admitted));
    assert!(after.contains(admitted));
    assert_eq!(before.len(), 1);
    assert_eq!(after.len(), 1);
}

#[test]
#[cfg(feature = "text-index")]
fn text_index_created_before_compaction_survives_both_merges() {
    let mut db = GrafeoDB::new_in_memory();
    let first = db.create_node_with_props(
        &["Article"],
        [("body", Value::from("durable temporalmodel search"))],
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
    .expect("create text index before compact");

    db.compact().expect("compact with text index");
    assert_eq!(
        db.text_search("Article", "body", "temporalmodel", 10)
            .expect("search after compact")[0]
            .0,
        first
    );

    let second = db.create_node_with_props(
        &["Article"],
        [("body", Value::from("temporalmodel overlay document"))],
    );
    db.compact().expect("recompact with text index");
    let mut found: Vec<_> = db
        .text_search("Article", "body", "temporalmodel", 10)
        .expect("search after recompact")
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    found.sort_unstable();
    assert_eq!(found, vec![first, second]);
}

#[test]
#[cfg(feature = "vector-index")]
fn quantized_index_created_before_compaction_survives_both_merges() {
    let mut db = GrafeoDB::new_in_memory();
    let first = db.create_node_with_props(&["Doc"], [("embedding", vec3(1.0, 0.0, 0.0))]);
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
    .expect("create quantized index before compact");

    db.compact().expect("compact with quantized index");
    let after_compact = db
        .vector_search("Doc", "embedding", &[1.0, 0.0, 0.0], 1, None, None)
        .expect("quantized search after compact");
    assert_eq!(after_compact[0].0, first);

    let second = db.create_node_with_props(&["Doc"], [("embedding", vec3(0.0, 1.0, 0.0))]);
    db.compact().expect("recompact with quantized index");
    let mut found: Vec<_> = db
        .vector_search("Doc", "embedding", &[1.0, 0.0, 0.0], 2, None, None)
        .expect("quantized search after recompact")
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    found.sort_unstable();
    assert_eq!(found, vec![first, second]);
}

#[test]
#[cfg(feature = "vector-index")]
fn vector_index_after_recompact() {
    let mut db = GrafeoDB::new_in_memory();

    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "embedding", vec3(1.0, 0.0, 0.0))
        .expect("set node property");
    let n2 = db.create_node(&["Doc"]);
    db.set_node_property(n2, "embedding", vec3(0.0, 1.0, 0.0))
        .expect("set node property");

    db.compact().expect("first compact");

    // Insert a third node after compact, then recompact to merge all three.
    let n3 = db.create_node(&["Doc"]);
    db.set_node_property(n3, "embedding", vec3(0.0, 0.0, 1.0))
        .expect("set node property");

    db.compact().expect("recompact");

    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
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
    .expect("create index after recompact");

    let results = db
        .vector_search("Doc", "embedding", &[1.0, 0.0, 0.0], 3, None, None)
        .expect("search after recompact");

    assert_eq!(
        results.len(),
        3,
        "should find nodes from both compaction rounds"
    );
}
