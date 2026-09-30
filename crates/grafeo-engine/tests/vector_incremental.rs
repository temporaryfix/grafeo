//! Integration tests for incremental vector index operations.
//!
//! Tests that nodes added/deleted after `create_vector_index` are
//! automatically indexed/removed, and that drop/rebuild work correctly.

#![cfg(all(feature = "lpg", feature = "vector-index"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Helper: create a 3D vector value.
fn vec3(x: f32, y: f32, z: f32) -> Value {
    Value::Vector(vec![x, y, z].into())
}

#[test]
fn test_incremental_insert_via_set_property() {
    let db = GrafeoDB::new_in_memory();

    // Create initial nodes and build index
    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
        .expect("set node property");
    let n2 = db.create_node(&["Doc"]);
    db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0))
        .expect("set node property");

    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
        property: "emb".into(),
        kind: grafeo_engine::IndexCreateKind::Vector {
            dimensions: Some(3),
            metric: Some("cosine".into()),
            m: None,
            ef_construction: None,
            ef: None,
            quantization: None,
        },
    })
    .expect("create index");

    // Add a new node AFTER index creation
    let n3 = db.create_node(&["Doc"]);
    db.set_node_property(n3, "emb", vec3(0.9, 0.1, 0.0))
        .expect("set node property");

    // Search should find the new node (closest to [1, 0, 0])
    let results = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 3, None, None)
        .expect("search");

    assert_eq!(results.len(), 3, "should find all 3 nodes");
    // n1 and n3 should be closest to query [1, 0, 0]
    let ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
    assert!(ids.contains(&n3.as_u64()), "n3 should be in results");
}

#[test]
fn test_incremental_batch_create_after_index() {
    let db = GrafeoDB::new_in_memory();

    // Create initial node and build index
    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
        .expect("set node property");

    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
        property: "emb".into(),
        kind: grafeo_engine::IndexCreateKind::Vector {
            dimensions: Some(3),
            metric: Some("euclidean".into()),
            m: None,
            ef_construction: None,
            ef: None,
            quantization: None,
        },
    })
    .expect("create index");

    // Batch-create nodes AFTER index
    let new_ids =
        db.batch_create_nodes("Doc", "emb", vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]]);

    assert_eq!(new_ids.len(), 2);

    // Search should find all 3 nodes
    let results = db
        .vector_search("Doc", "emb", &[0.5, 0.5, 0.5], 10, None, None)
        .expect("search");

    assert_eq!(results.len(), 3, "should find original + batch nodes");
}

#[test]
fn test_delete_removes_from_index() {
    let db = GrafeoDB::new_in_memory();

    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
        .expect("set node property");
    let n2 = db.create_node(&["Doc"]);
    db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0))
        .expect("set node property");
    let n3 = db.create_node(&["Doc"]);
    db.set_node_property(n3, "emb", vec3(0.0, 0.0, 1.0))
        .expect("set node property");

    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
        property: "emb".into(),
        kind: grafeo_engine::IndexCreateKind::Vector {
            dimensions: Some(3),
            metric: Some("euclidean".into()),
            m: None,
            ef_construction: None,
            ef: None,
            quantization: None,
        },
    })
    .expect("create index");

    // Delete n2
    assert!(db.delete_node(n2));

    // Search should NOT return n2
    let results = db
        .vector_search("Doc", "emb", &[0.0, 1.0, 0.0], 10, None, None)
        .expect("search");

    let ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
    assert!(
        !ids.contains(&n2.as_u64()),
        "deleted node should not appear"
    );
    assert_eq!(results.len(), 2, "should find only 2 remaining nodes");
}

#[test]
fn test_label_after_vector_triggers_index() {
    let db = GrafeoDB::new_in_memory();

    // Build index on "Doc:emb" with an initial node
    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
        .expect("set node property");

    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
        property: "emb".into(),
        kind: grafeo_engine::IndexCreateKind::Vector {
            dimensions: Some(3),
            metric: Some("cosine".into()),
            m: None,
            ef_construction: None,
            ef: None,
            quantization: None,
        },
    })
    .expect("create index");

    // Create a node WITHOUT the "Doc" label, set vector, THEN add label
    let n2 = db.create_node(&["Other"]);
    db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0))
        .expect("set node property");
    // At this point n2 has label "Other", not "Doc": no index match
    db.add_node_label(n2, "Doc");
    // Now n2 has "Doc" label, should trigger auto-insert

    let results = db
        .vector_search("Doc", "emb", &[0.0, 1.0, 0.0], 10, None, None)
        .expect("search");

    let ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
    assert!(
        ids.contains(&n2.as_u64()),
        "label-after-vector node should be found"
    );
}

#[test]
#[cfg(feature = "gql")]
fn session_transaction_property_commit_maintains_vector_index() {
    let db = GrafeoDB::new_in_memory();
    let anchor = db.create_node(&["Doc"]);
    db.set_node_property(anchor, "emb", vec3(1.0, 0.0, 0.0))
        .expect("set node property");
    let target = db.create_node_with_props(&["Doc"], [("name", Value::from("target"))]);

    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
        property: "emb".into(),
        kind: grafeo_engine::IndexCreateKind::Vector {
            dimensions: Some(3),
            metric: Some("cosine".into()),
            m: None,
            ef_construction: None,
            ef: None,
            quantization: None,
        },
    })
    .expect("create index");
    let index = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_vector_index("Doc", "emb")
        .expect("registered vector index");
    assert!(!index.contains(target));

    let mut session = db.session();
    session.begin_transaction().expect("begin SET transaction");
    session
        .execute("MATCH (n:Doc {name: 'target'}) SET n.emb = [0.0, 1.0, 0.0]")
        .expect("transactional vector SET");
    session.commit().expect("commit vector SET");

    assert!(
        index.contains(target),
        "Session/GQL SET commit must publish the node to the HNSW index"
    );
    let ids: Vec<_> = db
        .vector_search("Doc", "emb", &[0.0, 1.0, 0.0], 10, None, None)
        .expect("search after SET")
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert!(ids.contains(&target));

    let mut session = db.session();
    session
        .begin_transaction()
        .expect("begin REMOVE transaction");
    session
        .execute("MATCH (n:Doc {name: 'target'}) REMOVE n.emb")
        .expect("transactional vector REMOVE");
    session.commit().expect("commit vector REMOVE");

    assert!(
        !index.contains(target),
        "Session/GQL REMOVE commit must remove the node from the HNSW index"
    );
}

#[test]
#[cfg(feature = "gql")]
fn session_transaction_label_commit_maintains_vector_index() {
    let db = GrafeoDB::new_in_memory();
    let anchor = db.create_node(&["Doc"]);
    db.set_node_property(anchor, "emb", vec3(1.0, 0.0, 0.0))
        .expect("set node property");
    let target = db.create_node_with_props(
        &["Other"],
        [
            ("name", Value::from("label-target")),
            ("emb", vec3(0.0, 1.0, 0.0)),
        ],
    );

    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
        property: "emb".into(),
        kind: grafeo_engine::IndexCreateKind::Vector {
            dimensions: Some(3),
            metric: Some("cosine".into()),
            m: None,
            ef_construction: None,
            ef: None,
            quantization: None,
        },
    })
    .expect("create index");
    let index = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_vector_index("Doc", "emb")
        .expect("registered vector index");
    assert!(!index.contains(target));

    let mut session = db.session();
    session
        .begin_transaction()
        .expect("begin label-add transaction");
    session
        .execute("MATCH (n:Other {name: 'label-target'}) SET n:Doc")
        .expect("transactional label add");
    session.commit().expect("commit label add");
    assert!(
        index.contains(target),
        "Session/GQL label-add commit must publish the existing vector"
    );

    let mut session = db.session();
    session
        .begin_transaction()
        .expect("begin label-remove transaction");
    session
        .execute("MATCH (n:Doc {name: 'label-target'}) REMOVE n:Doc")
        .expect("transactional label remove");
    session.commit().expect("commit label remove");
    assert!(
        !index.contains(target),
        "Session/GQL label-remove commit must remove the vector"
    );
}

#[test]
#[cfg(feature = "gql")]
fn session_transaction_delete_cleans_vector_and_property_indexes() {
    let db = GrafeoDB::new_in_memory();
    let target = db.create_node_with_props(
        &["Doc"],
        [
            ("name", Value::from("delete-target")),
            ("emb", vec3(0.0, 1.0, 0.0)),
        ],
    );
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: None,
        property: "name".into(),
        kind: grafeo_engine::IndexCreateKind::Property,
    })
    .expect("create property index");
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
        property: "emb".into(),
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
    let index = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_vector_index("Doc", "emb")
        .expect("registered vector index");
    let name = Value::from("delete-target");
    assert!(index.contains(target));
    assert_eq!(db.find_nodes_by_property("name", &name), vec![target]);

    let mut session = db.session();
    session.begin_transaction().expect("begin rollback delete");
    session
        .execute("MATCH (n:Doc {name: 'delete-target'}) DELETE n")
        .expect("transactional DELETE before rollback");
    assert!(
        index.contains(target),
        "a pending delete must not hide the committed HNSW entry from other sessions"
    );
    session.rollback().expect("rollback delete");
    assert!(index.contains(target));
    assert_eq!(db.find_nodes_by_property("name", &name), vec![target]);

    let mut session = db.session();
    session.begin_transaction().expect("begin committed delete");
    session
        .execute("MATCH (n:Doc {name: 'delete-target'}) DELETE n")
        .expect("transactional DELETE");
    session.commit().expect("commit DELETE");

    assert!(
        !index.contains(target),
        "Session/GQL DELETE commit must remove HNSW membership"
    );
    assert!(
        db.find_nodes_by_property("name", &name).is_empty(),
        "Session/GQL DELETE commit must remove ordinary property-index membership"
    );
}

#[test]
fn test_drop_vector_index() {
    let db = GrafeoDB::new_in_memory();

    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
        .expect("set node property");

    let owner = db
        .create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Doc".into()),
            property: "emb".into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(3),
                metric: Some("cosine".into()),
                m: None,
                ef_construction: None,
                ef: None,
                quantization: None,
            },
        })
        .expect("create index");

    // Search works
    assert!(
        db.vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 1, None, None)
            .is_ok()
    );

    // Drop
    assert!(db.drop_index(owner).expect("drop vector owner"));
    assert!(!db.drop_index(owner).expect("drop vector owner")); // second drop returns false

    // Search now fails
    assert!(
        db.vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 1, None, None)
            .is_err()
    );
}

#[test]
fn test_rebuild_vector_index() {
    let db = GrafeoDB::new_in_memory();

    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
        .expect("set node property");

    let owner = db
        .create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Doc".into()),
            property: "emb".into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(3),
                metric: Some("cosine".into()),
                m: None,
                ef_construction: None,
                ef: None,
                quantization: None,
            },
        })
        .expect("create index");

    // Add more nodes
    let n2 = db.create_node(&["Doc"]);
    db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0))
        .expect("set node property");
    let n3 = db.create_node(&["Doc"]);
    db.set_node_property(n3, "emb", vec3(0.0, 0.0, 1.0))
        .expect("set node property");

    // Rebuild rescans all nodes
    db.rebuild_index(owner).expect("rebuild");

    let results = db
        .vector_search("Doc", "emb", &[0.5, 0.5, 0.5], 10, None, None)
        .expect("search");

    assert_eq!(results.len(), 3, "rebuild should include all nodes");
}

#[test]
fn test_rebuild_nonexistent_index_fails() {
    let db = GrafeoDB::new_in_memory();
    assert!(
        db.rebuild_index(grafeo_common::types::IndexId::new(u32::MAX))
            .is_err()
    );
}

#[test]
fn test_set_vector_without_index_is_noop() {
    let db = GrafeoDB::new_in_memory();

    // No vector index exists: setting a vector property should not crash
    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
        .expect("set node property");

    // Node should exist with the property
    let node = db.get_node(n1);
    assert!(node.is_some());
}
