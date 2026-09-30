//! Owning fixtures for exact save configurations absent from the public DDL API.

#![cfg(all(feature = "lpg", feature = "grafeo-file", feature = "wal"))]

#[cfg(any(feature = "vector-index", feature = "text-index"))]
use crate::GrafeoDB;

#[cfg(any(feature = "vector-index", feature = "text-index"))]
fn assert_exact_saves(db: &GrafeoDB, inspect: impl Fn(&GrafeoDB)) {
    let snapshot = db.export_snapshot().expect("capture exact owned fixture");
    let cut = db.world_cut().expect("capture exact source world");
    let owners = db.catalog.all_indexes();
    let floor = db.catalog.index_allocator_high_water();
    let temp = tempfile::tempdir().unwrap();
    inspect(db);
    let assert_copy = |restored: &GrafeoDB| {
        assert_eq!(restored.export_snapshot().unwrap(), snapshot);
        assert_eq!(restored.world_cut().unwrap(), cut);
        assert_eq!(restored.catalog.index_allocator_high_water(), floor);
        for owner in &owners {
            assert_eq!(restored.catalog.get_index(owner.id), Some(owner.clone()));
        }
        inspect(restored);
    };
    let imported = GrafeoDB::import_snapshot(&snapshot).expect("import exact owned state");
    assert_copy(&imported);
    imported.close().unwrap();
    let live = GrafeoDB::new_in_memory();
    live.restore_snapshot(&snapshot)
        .expect("replace with exact owned state");
    assert_copy(&live);
    live.close().unwrap();
    for name in ["saved", "saved.grafeo"] {
        let path = temp.path().join(name);
        db.save(&path).expect("save exact owned state");
        assert!(path.is_file());
        let restored = GrafeoDB::open(&path).expect("open exact owned state");
        assert_copy(&restored);
        restored.close().unwrap();
    }
    assert_eq!(db.export_snapshot().unwrap(), snapshot);
    assert_eq!(db.world_cut().unwrap(), cut);
}

#[cfg(feature = "vector-index")]
#[test]
fn exact_save_preserves_quantized_none_and_nondefault_search_knobs() {
    use crate::catalog::IndexConfiguration;
    use grafeo_common::types::{GraphPath, Value};
    use grafeo_core::index::vector::{
        DistanceMetric, HnswConfig, QuantizationType, QuantizedHnswIndex, VectorIndexKind,
    };
    use std::sync::Arc;

    for case in 0..4 {
        let db = GrafeoDB::new_in_memory();
        let node = db.create_node_with_props(
            &["Doc"],
            [("embedding", Value::Vector(vec![1.0, 0.0, 0.0].into()))],
        );
        let config = HnswConfig::new(3, DistanceMetric::Cosine);
        let quantization = if case == 0 {
            QuantizationType::None
        } else {
            QuantizationType::Binary
        };
        let index = QuantizedHnswIndex::new(config.clone(), quantization);
        let index = match case {
            1 => index.without_rescore(),
            2 => index.with_rescore_factor(7),
            3 => index.with_training_threshold(10),
            _ => index,
        };
        let expected = (
            index.rescoring_enabled(),
            index.rescore_factor(),
            index.training_threshold(),
        );
        match case {
            1 => assert!(!expected.0),
            2 => assert_eq!(expected.1, 7),
            3 => assert_eq!(expected.2, 10),
            _ => assert_eq!(index.quantization_type(), QuantizationType::None),
        }
        index.insert(node, &[1.0, 0.0, 0.0]);
        db.transaction_manager.with_write_authority(|| {
            let label = db.catalog.get_or_create_label("Doc").unwrap();
            let property = db.catalog.get_or_create_property_key("embedding").unwrap();
            db.catalog
                .create_index(
                    Some("quantized_owner"),
                    label,
                    property,
                    GraphPath::root(),
                    IndexConfiguration::Vector {
                        config,
                        quantization,
                    },
                )
                .unwrap();
            db.store_arc().add_vector_index(
                "Doc",
                "embedding",
                Arc::new(VectorIndexKind::Quantized(index)),
            );
        });
        assert_exact_saves(&db, |copy| {
            let registered = copy
                .store_arc()
                .get_vector_index("Doc", "embedding")
                .expect("fixture vector actually installed");
            assert_eq!(
                registered.quantization_type(),
                Some(quantization),
                "Quantized variant survives even for None"
            );
            assert_eq!(registered.len(), 1);
        });
    }
}

#[cfg(feature = "text-index")]
#[test]
fn exact_save_preserves_owned_versioned_postings() {
    use crate::catalog::IndexConfiguration;
    use grafeo_common::types::{EpochId, GraphPath, TransactionId, Value};
    use grafeo_common::utils::hash::FxHashSet;
    use grafeo_core::index::text::{BM25Config, InvertedIndex};
    use parking_lot::RwLock;
    use std::sync::Arc;

    let db = GrafeoDB::new_in_memory();
    let node = db.create_node_with_props(&["Article"], [("body", Value::from("ancient graph"))]);
    assert_eq!(db.current_epoch(), EpochId::new(1));
    db.set_node_property(node, "body", Value::from("modern database"))
        .unwrap();
    assert_eq!(db.current_epoch(), EpochId::new(2));
    let mut index = InvertedIndex::new(BM25Config::default());
    index.insert_versioned(node, "ancient graph", EpochId::new(1), None);
    index.insert_versioned(node, "modern database", EpochId::new(2), None);
    db.transaction_manager.with_write_authority(|| {
        let label = db.catalog.get_or_create_label("Article").unwrap();
        let property = db.catalog.get_or_create_property_key("body").unwrap();
        db.catalog
            .create_index(
                Some("history_owner"),
                label,
                property,
                GraphPath::root(),
                IndexConfiguration::Text {
                    config: BM25Config::default(),
                    min_token_length: 2,
                },
            )
            .unwrap();
        db.store_arc()
            .add_text_index("Article", "body", Arc::new(RwLock::new(index)));
    });
    assert_exact_saves(&db, |copy| {
        let registered = copy
            .store_arc()
            .get_text_index("Article", "body")
            .expect("fixture Text actually installed");
        let index = registered.read();
        let removed = FxHashSet::default();
        let old = index
            .search_visible(
                "ancient",
                10,
                EpochId::new(1),
                TransactionId::INVALID,
                &[],
                &removed,
            )
            .unwrap();
        assert_eq!(
            old.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![node]
        );
        assert!(
            index
                .search_visible(
                    "ancient",
                    10,
                    EpochId::new(2),
                    TransactionId::INVALID,
                    &[],
                    &removed
                )
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            index
                .search_visible(
                    "modern",
                    10,
                    EpochId::new(2),
                    TransactionId::INVALID,
                    &[],
                    &removed
                )
                .unwrap()
                .len(),
            1
        );
    });
}

#[cfg(feature = "text-index")]
#[test]
fn exact_save_rejects_owned_opaque_tokenizer_before_destination_publication() {
    use crate::catalog::IndexConfiguration;
    use grafeo_common::storage::Section;
    use grafeo_common::types::{GraphPath, Value};
    use grafeo_core::graph::lpg::PhysicalIndexKey;
    use grafeo_core::index::text::{BM25Config, InvertedIndex, SimpleTokenizer, TextIndexSection};
    use parking_lot::RwLock;
    use std::sync::Arc;

    let db = GrafeoDB::new_in_memory();
    let node = db.create_node_with_props(
        &["Article"],
        [("body", Value::from("tiny persistence document"))],
    );
    let mut index = InvertedIndex::with_tokenizer(
        BM25Config::default(),
        Box::new(SimpleTokenizer::with_min_length(7)),
    );
    index.insert(node, "tiny persistence document");
    db.transaction_manager.with_write_authority(|| {
        let label = db.catalog.get_or_create_label("Article").unwrap();
        let property = db.catalog.get_or_create_property_key("body").unwrap();
        db.catalog
            .create_index(
                Some("opaque_owner"),
                label,
                property,
                GraphPath::root(),
                IndexConfiguration::Text {
                    config: BM25Config::default(),
                    min_token_length: 7,
                },
            )
            .unwrap();
        db.store_arc()
            .add_text_index("Article", "body", Arc::new(RwLock::new(index)));
    });
    assert!(db.store_arc().get_text_index("Article", "body").is_some());
    assert_eq!(db.catalog.index_count(), 1);
    let registered = db.store_arc().get_text_index("Article", "body").unwrap();
    let codec_error = TextIndexSection::from_views(vec![(
        PhysicalIndexKey::text(GraphPath::root(), "Article", "body"),
        registered,
    )])
    .serialize()
    .expect_err("fixture must carry opaque tokenizer provenance");
    assert!(codec_error.to_string().contains("opaque custom tokenizer"));
    let temp = tempfile::tempdir().unwrap();
    for name in ["opaque", "opaque.grafeo"] {
        let destination = temp.path().join(name);
        let error = db
            .save(&destination)
            .expect_err("opaque tokenizer cannot become a default tokenizer");
        // Catalog validation rejects the opaque descriptor before the codec:
        // its behavior cannot be proven equal to a named SimpleTokenizer.
        assert!(
            error
                .to_string()
                .contains("physical Text owner/config mismatch"),
            "{error}"
        );
        assert!(!destination.exists());
        assert!(!std::fs::read_dir(temp.path()).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".grafeo-file-save-")
        }));
        assert_eq!(db.node_count(), 1);
        assert!(db.store_arc().get_text_index("Article", "body").is_some());
    }
}
