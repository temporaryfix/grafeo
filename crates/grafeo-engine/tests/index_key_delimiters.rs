//! Regression coverage for collision-free vector/text index registry keys.

#![cfg(all(
    feature = "lpg",
    feature = "wal",
    feature = "vector-index",
    feature = "text-index"
))]

use grafeo_common::types::Value;
use grafeo_engine::config::StorageFormat;
use grafeo_engine::{Config, GrafeoDB};

#[test]
fn colon_scoped_indexes_survive_directory_save_and_reopen() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("colon-indexes");
    let (vector_label_node, vector_property_node, text_label_node, text_property_node);

    {
        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let db = GrafeoDB::with_config(config).expect("open");

        // Each pair collided under the old `format!("{label}:{property}")`
        // registry representation.
        vector_label_node = db.create_node_with_props(
            &["tenant:Doc"],
            [("embedding", Value::Vector(vec![1.0, 0.0].into()))],
        );
        vector_property_node = db.create_node_with_props(
            &["tenant"],
            [("Doc:embedding", Value::Vector(vec![0.0, 1.0].into()))],
        );
        text_label_node = db.create_node_with_props(
            &["tenant:Article"],
            [("body", Value::from("label colon document"))],
        );
        text_property_node = db.create_node_with_props(
            &["tenant"],
            [("Article:body", Value::from("property colon document"))],
        );

        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("tenant:Doc".into()),
            property: "embedding".into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(2),
                metric: Some("cosine".into()),
                m: Some(8),
                ef_construction: Some(32),
                ef: None,
                quantization: None,
            },
        })
        .expect("create colon-label vector index");
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("tenant".into()),
            property: "Doc:embedding".into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(2),
                metric: Some("cosine".into()),
                m: Some(8),
                ef_construction: Some(32),
                ef: None,
                quantization: None,
            },
        })
        .expect("create colon-property vector index");
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("tenant:Article".into()),
            property: "body".into(),
            kind: grafeo_engine::IndexCreateKind::Text {
                min_token_length: None,
            },
        })
        .expect("create colon-label text index");
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("tenant".into()),
            property: "Article:body".into(),
            kind: grafeo_engine::IndexCreateKind::Text {
                min_token_length: None,
            },
        })
        .expect("create colon-property text index");
        db.close().expect("save directory database");
    }

    let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
    let db = GrafeoDB::with_config(config).expect("reopen");

    let label_vector = db
        .vector_search("tenant:Doc", "embedding", &[1.0, 0.0], 1, None, None)
        .expect("search colon-label vector index");
    let property_vector = db
        .vector_search("tenant", "Doc:embedding", &[0.0, 1.0], 1, None, None)
        .expect("search colon-property vector index");
    assert_eq!(label_vector[0].0, vector_label_node);
    assert_eq!(property_vector[0].0, vector_property_node);

    let label_text = db
        .text_search("tenant:Article", "body", "label", 10)
        .expect("search colon-label text index");
    let property_text = db
        .text_search("tenant", "Article:body", "property", 10)
        .expect("search colon-property text index");
    assert_eq!(label_text[0].0, text_label_node);
    assert_eq!(property_text[0].0, text_property_node);
    db.close().expect("close reopened database");
}
