//! Real Session publication of sparse surviving indexes and replacement DDL.

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "vector-index",
    feature = "text-index"
))]

use std::sync::Arc;

use grafeo_common::types::{TransactionId, Value};
use grafeo_common::utils::hash::FxHashSet;
use grafeo_engine::GrafeoDB;

fn vector(value: f32) -> Value {
    Value::Vector(Arc::from([value; 8]))
}

fn surviving_indexes(
    quantization: Option<&str>,
    compact: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let changed = session.create_node_with_props(
        &["Doc"],
        [
            ("name", Value::from("before")),
            ("body", Value::from("oldtoken")),
            ("emb", vector(1.0)),
        ],
    )?;
    let deleted = session.create_node_with_props(
        &["Doc"],
        [
            ("name", Value::from("delete")),
            ("body", Value::from("retiretoken")),
            ("emb", vector(-1.0)),
        ],
    )?;
    let untouched = session.create_node_with_props(
        &["Doc"],
        [
            ("name", Value::from("untouched")),
            ("body", Value::from("untouchedtoken")),
            ("emb", vector(0.5)),
        ],
    )?;
    session.execute("CREATE INDEX idx_name FOR (n:Doc) ON (n.name)")?;
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
        property: "body".into(),
        kind: grafeo_engine::IndexCreateKind::Text {
            min_token_length: None,
        },
    })?;
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
        property: "emb".into(),
        kind: grafeo_engine::IndexCreateKind::Vector {
            dimensions: Some(8),
            metric: Some("euclidean".into()),
            m: None,
            ef_construction: None,
            ef: None,
            quantization: quantization.map(str::to_owned),
        },
    })?;
    drop(session);
    #[cfg(feature = "compact-store")]
    let mut db = db;
    #[cfg(feature = "compact-store")]
    if compact {
        db.compact()?;
    }
    #[cfg(not(feature = "compact-store"))]
    assert!(!compact);

    let text = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_text_index("Doc", "body")
        .ok_or("Text index is absent")?;
    let vector = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_vector_index("Doc", "emb")
        .ok_or("Vector index is absent")?;
    let mut session = db.session();
    session.begin_transaction()?;
    let publication = session.commit()?;
    session.begin_transaction()?;
    session.set_node_property(changed, "name", Value::from("after"))?;
    session.set_node_property(changed, "body", Value::from("newtoken"))?;
    session.set_node_property(changed, "emb", Value::Vector(Arc::from([2.0_f32; 8])))?;
    session.execute("MATCH (n:Doc {name: 'delete'}) DELETE n")?;
    let created = session.create_node_with_props(
        &["Doc"],
        [
            ("name", Value::from("created")),
            ("body", Value::from("createdtoken")),
            ("emb", Value::Vector(Arc::from([-2.0_f32; 8]))),
        ],
    )?;
    let committed = session.commit()?;

    // These are retained precommit handles, not fresh registry lookups. A
    // whole-index replacement would leave these assertions stale.
    assert!(vector.contains(changed));
    assert!(vector.contains(untouched));
    assert!(vector.contains(created));
    assert!(!vector.contains(deleted));
    assert_eq!(vector.len(), 3);
    let current_hits = db.vector_search("Doc", "emb", &[2.0; 8], 10, None, None)?;
    assert_eq!(current_hits.len(), 3);
    assert!(!current_hits.iter().any(|(id, _)| *id == deleted));

    let guard = text.read();
    let removed = FxHashSet::default();
    assert_eq!(
        guard
            .search_visible(
                "oldtoken",
                10,
                publication,
                TransactionId::INVALID,
                &[],
                &removed
            )?
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>(),
        vec![changed]
    );
    assert!(
        guard
            .search_visible(
                "newtoken",
                10,
                publication,
                TransactionId::INVALID,
                &[],
                &removed
            )?
            .is_empty()
    );
    assert!(
        guard
            .search_visible(
                "oldtoken",
                10,
                committed,
                TransactionId::INVALID,
                &[],
                &removed
            )?
            .is_empty()
    );
    assert_eq!(
        guard
            .search_visible(
                "newtoken",
                10,
                committed,
                TransactionId::INVALID,
                &[],
                &removed
            )?
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>(),
        vec![changed]
    );
    assert_eq!(guard.doc_count_at(publication, TransactionId::INVALID)?, 3);
    assert_eq!(guard.doc_count_at(committed, TransactionId::INVALID)?, 3);
    drop(guard);
    assert_eq!(
        session
            .execute("MATCH (n:Doc {name: 'after'}) RETURN n.name")?
            .rows(),
        vec![vec![Value::from("after")]]
    );
    assert_eq!(
        session
            .execute("MATCH (n:Doc {name: 'before'}) RETURN n.name")?
            .row_count(),
        0
    );
    assert_eq!(
        session
            .execute("MATCH (n:Doc {name: 'delete'}) RETURN n.name")?
            .row_count(),
        0
    );
    Ok(())
}

#[test]
fn native_survivors_keep_handles_and_history_for_every_vector_kind()
-> Result<(), Box<dyn std::error::Error>> {
    for quantization in [
        None,
        Some("scalar"),
        Some("binary"),
        Some("product"),
        Some("pq4"),
    ] {
        surviving_indexes(quantization, false)?;
    }
    Ok(())
}

#[cfg(feature = "compact-store")]
#[test]
fn layered_survivors_publish_cold_promotion_and_delete_for_every_vector_kind()
-> Result<(), Box<dyn std::error::Error>> {
    for quantization in [
        None,
        Some("scalar"),
        Some("binary"),
        Some("product"),
        Some("pq4"),
    ] {
        surviving_indexes(quantization, true)?;
    }
    Ok(())
}

#[test]
fn declared_vector_search_depth_survives_commit_rollback_and_reopen()
-> Result<(), Box<dyn std::error::Error>> {
    use grafeo_core::index::vector::{DistanceMetric, HnswConfig};
    for ef in [None, Some(128)] {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.create_node_with_props(&["Doc"], [("emb", vector(1.0))])?;
        let suffix = ef.map_or_else(String::new, |value| format!(", ef: {value}"));
        session.begin_transaction()?;
        session.execute(&format!("CREATE INDEX idx_depth FOR (n:Doc) ON (n.emb) USING VECTOR {{dimensions: 8, metric: 'euclidean'{suffix}}}"))?;
        session.commit()?;
        let expected = ef.unwrap_or_else(|| HnswConfig::new(8, DistanceMetric::Euclidean).ef);
        assert_eq!(
            grafeo_engine::database::testing::root_lpg_store(&db)
                .get_vector_index("Doc", "emb")
                .ok_or("missing committed index")?
                .config()
                .ef,
            expected
        );
        session.begin_transaction()?;
        session.execute("DROP INDEX idx_depth")?;
        session.execute("CREATE INDEX idx_depth FOR (n:Doc) ON (n.emb) USING VECTOR {dimensions: 8, metric: 'euclidean', ef: 7}")?;
        session.rollback()?;
        assert_eq!(
            grafeo_engine::database::testing::root_lpg_store(&db)
                .get_vector_index("Doc", "emb")
                .ok_or("rollback lost index")?
                .config()
                .ef,
            expected
        );
        #[cfg(all(feature = "wal", feature = "grafeo-file"))]
        {
            let directory = tempfile::tempdir()?;
            let path = directory.path().join("vector-depth.grafeo");
            db.save(&path)?;
            let restored = GrafeoDB::open(&path)?;
            assert_eq!(
                grafeo_engine::database::testing::root_lpg_store(&restored)
                    .get_vector_index("Doc", "emb")
                    .ok_or("reopen lost index")?
                    .config()
                    .ef,
                expected
            );
            let indexes = restored.session().execute("SHOW INDEXES")?;
            assert_eq!(indexes.row_count(), 1);
            assert_eq!(indexes.rows()[0][0], Value::from("idx_depth"));
        }
    }
    Ok(())
}

#[test]
fn named_vector_drop_create_uses_only_final_registration_and_dimensions()
-> Result<(), Box<dyn std::error::Error>> {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    let id = session.create_node_with_props(&["Doc"], [("emb", vector(1.0))])?;
    session.execute("CREATE INDEX idx_vector FOR (n:Doc) ON (n.emb) USING VECTOR {dimensions: 8, metric: 'euclidean'}")?;
    let before = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_vector_index("Doc", "emb")
        .ok_or("old Vector index is absent")?;
    session.begin_transaction()?;
    session.execute("DROP INDEX idx_vector")?;
    session.set_node_property(id, "emb", Value::Vector(Arc::from([0.0_f32, 1.0])))?;
    session.execute("CREATE INDEX idx_vector FOR (n:Doc) ON (n.emb) USING VECTOR {dimensions: 2, metric: 'euclidean'}")?;
    session.commit()?;
    assert_eq!(before.config().dimensions, 8);
    assert!(before.contains(id));
    assert_eq!(
        db.vector_search("Doc", "emb", &[0.0, 1.0], 10, None, None)?,
        vec![(id, 0.0)]
    );
    let names = session.execute("SHOW INDEXES")?;
    assert_eq!(names.row_count(), 1);
    assert_eq!(names.rows()[0][0], Value::from("idx_vector"));
    Ok(())
}

#[test]
fn fresh_text_index_final_membership_is_born_at_commit_epoch()
-> Result<(), Box<dyn std::error::Error>> {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    let id = session.create_node_with_props(&["Doc"], [("body", Value::from("beforetoken"))])?;
    session.begin_transaction()?;
    let publication = session.commit()?;
    session.begin_transaction()?;
    session.set_node_property(id, "body", Value::from("committedtoken"))?;
    session.execute("CREATE INDEX idx_text_birth FOR (n:Doc) ON (n.body) USING TEXT")?;
    let committed = session.commit()?;
    let index = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_text_index("Doc", "body")
        .ok_or("Text index is absent")?;
    let index = index.read();
    let removed = FxHashSet::default();
    assert_eq!(
        index.doc_count_at(
            grafeo_common::types::EpochId::INITIAL,
            TransactionId::INVALID
        )?,
        0
    );
    assert_eq!(index.doc_count_at(publication, TransactionId::INVALID)?, 0);
    assert_eq!(index.doc_count_at(committed, TransactionId::INVALID)?, 1);
    assert!(
        index
            .search_visible(
                "committedtoken",
                10,
                publication,
                TransactionId::INVALID,
                &[],
                &removed
            )?
            .is_empty()
    );
    assert!(
        index
            .search_visible(
                "beforetoken",
                10,
                committed,
                TransactionId::INVALID,
                &[],
                &removed
            )?
            .is_empty()
    );
    assert_eq!(
        index
            .search_visible(
                "committedtoken",
                10,
                committed,
                TransactionId::INVALID,
                &[],
                &removed
            )?
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>(),
        vec![id]
    );
    drop(index);

    // Snapshot10 carries the exact Text birth history, as does the container.
    let imported = GrafeoDB::import_snapshot(&db.export_snapshot()?)?;
    let imported_text = grafeo_engine::database::testing::root_lpg_store(&imported)
        .get_text_index("Doc", "body")
        .ok_or("imported Text index is absent")?;
    assert_eq!(
        imported_text
            .read()
            .doc_count_at(publication, TransactionId::INVALID)?,
        0
    );
    assert_eq!(
        imported_text
            .read()
            .doc_count_at(committed, TransactionId::INVALID)?,
        1
    );
    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    {
        session.set_node_property(id, "body", Value::from("latertoken"))?;
        let images = |db: &GrafeoDB| -> grafeo_common::utils::error::Result<Vec<u8>> {
            use grafeo_common::storage::Section;
            grafeo_core::index::text::TextIndexSection::from_views(
                grafeo_engine::database::testing::root_lpg_store(db)
                    .text_index_entries()
                    .into_iter()
                    .map(|(key, view)| {
                        let (label, property) = grafeo_core::graph::lpg::decode_index_key(&key)
                            .ok_or_else(|| {
                                grafeo_common::utils::error::Error::Serialization(
                                    "invalid test index key".into(),
                                )
                            })?;
                        Ok((
                            grafeo_core::graph::lpg::PhysicalIndexKey::text(
                                grafeo_common::types::GraphPath::root(),
                                label,
                                property,
                            ),
                            view,
                        ))
                    })
                    .collect::<grafeo_common::utils::error::Result<Vec<_>>>()?,
            )
            .serialize()
        };
        let before = images(&db)?;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("commit-born-text.grafeo");
        db.save(&path)?;
        let restored = GrafeoDB::open(&path)?;
        assert_eq!(images(&restored)?, before);
        let restored_index = grafeo_engine::database::testing::root_lpg_store(&restored)
            .get_text_index("Doc", "body")
            .ok_or("restored Text index is absent")?;
        let restored_index = restored_index.read();
        assert_eq!(
            restored_index.doc_count_at(publication, TransactionId::INVALID)?,
            0
        );
        assert_eq!(
            restored_index.doc_count_at(committed, TransactionId::INVALID)?,
            1
        );
        assert_eq!(
            restored_index
                .search_visible(
                    "committedtoken",
                    10,
                    committed,
                    TransactionId::INVALID,
                    &[],
                    &removed
                )?
                .iter()
                .map(|(id, _)| *id)
                .collect::<Vec<_>>(),
            vec![id]
        );
        drop(restored_index);
        assert_eq!(
            restored.text_search("Doc", "body", "latertoken", 10)?[0].0,
            id
        );
        restored.close()?;
    }
    Ok(())
}

#[test]
fn removed_vector_membership_keeps_committed_routing_for_same_batch_insert()
-> Result<(), Box<dyn std::error::Error>> {
    fn check(compact: bool) -> Result<(), Box<dyn std::error::Error>> {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // The sole old node is necessarily the HNSW entry point. Removing
        // membership keeps it as a routing hop for the new node below.
        let removed = session.create_node_with_props(
            &["Doc"],
            [("emb", Value::Vector(Arc::from([1.0_f32, 0.0])))],
        )?;
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Doc".into()),
            property: "emb".into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(2),
                metric: Some("euclidean".into()),
                m: None,
                ef_construction: None,
                ef: None,
                quantization: None,
            },
        })?;
        drop(session);
        #[cfg(feature = "compact-store")]
        let mut db = db;
        #[cfg(feature = "compact-store")]
        if compact {
            db.compact()?;
        }
        #[cfg(not(feature = "compact-store"))]
        assert!(!compact);
        let index = grafeo_engine::database::testing::root_lpg_store(&db)
            .get_vector_index("Doc", "emb")
            .ok_or("Vector index is absent")?;
        let mut session = db.session();
        session.begin_transaction()?;
        session.set_node_property(
            removed,
            "emb",
            Value::Vector(Arc::from([0.0_f32, 1.0, 0.0])),
        )?;
        session.execute("MATCH (n:Doc) REMOVE n:Doc")?;
        let created = session.create_node_with_props(
            &["Doc"],
            [("emb", Value::Vector(Arc::from([0.0_f32, 1.0])))],
        )?;
        session.commit()?;
        assert!(!index.contains(removed));
        assert!(index.contains(created));
        assert_eq!(index.len(), 1);
        let node = db
            .get_node(removed)
            .ok_or("label removal deleted its node")?;
        assert!(!node.has_label("Doc"));
        assert_eq!(
            node.get_property("emb"),
            Some(&Value::Vector(Arc::from([0.0_f32, 1.0, 0.0])))
        );
        assert_eq!(
            db.vector_search("Doc", "emb", &[0.0, 1.0], 10, None, None)?,
            vec![(created, 0.0)],
        );
        // The next commit has no touched-row override for the removed entry
        // point. Its committed, now-incompatible property must not break
        // either preparation's lazy routing or ordinary query traversal.
        let later = session.create_node_with_props(
            &["Doc"],
            [("emb", Value::Vector(Arc::from([1.0_f32, 0.0])))],
        )?;
        assert!(index.contains(later));
        assert_eq!(index.len(), 2);
        let hits = db.vector_search("Doc", "emb", &[1.0, 0.0], 10, None, None)?;
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0], (later, 0.0));
        assert!(hits.iter().any(|(id, _)| *id == created));
        assert!(!hits.iter().any(|(id, _)| *id == removed));
        Ok(())
    }
    check(false)?;
    #[cfg(feature = "compact-store")]
    check(true)?;
    Ok(())
}
