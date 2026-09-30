//! Tests for text index mutation hooks.
//!
//! These verify that text indexes are automatically kept in sync
//! when nodes are created, updated, or deleted.

#![cfg(all(feature = "lpg", feature = "text-index"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Helper: create a DB with a text index on :Doc(content).
fn setup_db_with_text_index() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
        property: "content".into(),
        kind: grafeo_engine::IndexCreateKind::Text {
            min_token_length: None,
        },
    })
    .unwrap();
    db
}

#[test]
fn test_text_index_auto_insert_via_create_node() {
    let db = setup_db_with_text_index();

    // Create a node AFTER the text index exists
    let id = db.create_node_with_props(
        &["Doc"],
        [(
            "content",
            Value::String("The quick brown fox jumps over the lazy dog".into()),
        )],
    );

    // Should be searchable immediately
    let results = db
        .text_search("Doc", "content", "quick brown fox", 10)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, id);
}

#[test]
fn test_text_index_auto_update_via_set_property() {
    let db = setup_db_with_text_index();

    let id = db.create_node_with_props(
        &["Doc"],
        [(
            "content",
            Value::String("original text about databases".into()),
        )],
    );

    // Verify initial text is searchable
    let results = db.text_search("Doc", "content", "databases", 10).unwrap();
    assert_eq!(results.len(), 1);

    // Update the property
    db.set_node_property(
        id,
        "content",
        Value::String("updated text about graph theory".into()),
    )
    .expect("set node property");

    // Old text should no longer match
    let results = db.text_search("Doc", "content", "databases", 10).unwrap();
    assert!(
        results.is_empty(),
        "Old text should not be found after update"
    );

    // New text should be searchable
    let results = db
        .text_search("Doc", "content", "graph theory", 10)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, id);

    assert!(db.remove_node_property(id, "content"));
    assert!(
        db.text_search("Doc", "content", "graph theory", 10)
            .unwrap()
            .is_empty(),
        "removed text properties must be removed by the central index hook"
    );
}

#[test]
fn test_text_index_auto_delete() {
    let db = setup_db_with_text_index();

    let id = db.create_node_with_props(
        &["Doc"],
        [(
            "content",
            Value::String("searchable document content".into()),
        )],
    );

    // Verify it's searchable
    let results = db.text_search("Doc", "content", "searchable", 10).unwrap();
    assert_eq!(results.len(), 1);

    // Delete the node
    let deleted = db.delete_node(id);
    assert!(deleted);

    // Should no longer appear in search
    let results = db.text_search("Doc", "content", "searchable", 10).unwrap();
    assert!(
        results.is_empty(),
        "Deleted node should not appear in text search"
    );
}

#[test]
fn test_text_index_add_label() {
    let db = GrafeoDB::new_in_memory();
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Article".into()),
        property: "content".into(),
        kind: grafeo_engine::IndexCreateKind::Text {
            min_token_length: None,
        },
    })
    .unwrap();

    // Create a node WITHOUT the Article label
    let id = db.create_node_with_props(
        &["Doc"],
        [(
            "content",
            Value::String("important article about Rust programming".into()),
        )],
    );

    // Not searchable under Article index
    let results = db.text_search("Article", "content", "Rust", 10).unwrap();
    assert!(results.is_empty());

    // Add the Article label
    let added = db.add_node_label(id, "Article");
    assert!(added);

    // Now it should be searchable under the Article text index
    let results = db
        .text_search("Article", "content", "Rust programming", 10)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, id);
}

#[test]
#[cfg(feature = "gql")]
fn session_transaction_label_and_delete_maintain_text_index() {
    let db = GrafeoDB::new_in_memory();
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Article".into()),
        property: "content".into(),
        kind: grafeo_engine::IndexCreateKind::Text {
            min_token_length: None,
        },
    })
    .unwrap();
    let id = db.create_node_with_props(
        &["Other"],
        [
            ("name", Value::from("tx-text-target")),
            ("content", Value::from("transactional text indexing")),
        ],
    );

    let mut session = db.session();
    session.begin_transaction().expect("begin label add");
    session
        .execute("MATCH (n:Other {name: 'tx-text-target'}) SET n:Article")
        .expect("transactional text-index label add");
    session.commit().expect("commit label add");
    assert_eq!(
        db.text_search("Article", "content", "transactional", 10)
            .unwrap()
            .iter()
            .map(|(node, _)| *node)
            .collect::<Vec<_>>(),
        vec![id]
    );

    let mut session = db.session();
    session.begin_transaction().expect("begin label remove");
    session
        .execute("MATCH (n:Article {name: 'tx-text-target'}) REMOVE n:Article")
        .expect("transactional text-index label remove");
    session.commit().expect("commit label remove");
    assert!(
        db.text_search("Article", "content", "transactional", 10)
            .unwrap()
            .is_empty()
    );

    let mut session = db.session();
    session.begin_transaction().expect("begin label re-add");
    session
        .execute("MATCH (n:Other {name: 'tx-text-target'}) SET n:Article")
        .expect("transactional text-index label re-add");
    session.commit().expect("commit label re-add");

    let mut session = db.session();
    session.begin_transaction().expect("begin delete");
    session
        .execute("MATCH (n:Article {name: 'tx-text-target'}) DELETE n")
        .expect("transactional text-index delete");
    session.commit().expect("commit delete");
    assert!(
        db.text_search("Article", "content", "transactional", 10)
            .unwrap()
            .is_empty(),
        "Session/GQL DELETE commit must remove text-index membership"
    );
}

#[test]
fn test_text_index_non_string_property_ignored() {
    let db = setup_db_with_text_index();

    // Set a non-String property: should not crash or pollute the index
    let id = db.create_node_with_props(&["Doc"], [("content", Value::Int64(42))]);

    let results = db.text_search("Doc", "content", "42", 10).unwrap();
    assert!(
        results.is_empty(),
        "Non-string values should not be indexed"
    );

    // Now set it to a string
    db.set_node_property(
        id,
        "content",
        Value::String("now it is a string value".into()),
    )
    .expect("set node property");
    let results = db
        .text_search("Doc", "content", "string value", 10)
        .unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_text_index_multiple_nodes() {
    let db = setup_db_with_text_index();

    let _id1 = db.create_node_with_props(
        &["Doc"],
        [(
            "content",
            Value::String("machine learning with neural networks".into()),
        )],
    );
    let _id2 = db.create_node_with_props(
        &["Doc"],
        [(
            "content",
            Value::String("graph databases and knowledge graphs".into()),
        )],
    );
    let id3 = db.create_node_with_props(
        &["Doc"],
        [(
            "content",
            Value::String("machine learning on graphs".into()),
        )],
    );

    // "machine learning" should match 2 docs
    let results = db
        .text_search("Doc", "content", "machine learning", 10)
        .unwrap();
    assert_eq!(results.len(), 2);

    // Delete one
    db.delete_node(id3);
    let results = db
        .text_search("Doc", "content", "machine learning", 10)
        .unwrap();
    assert_eq!(results.len(), 1);
}
