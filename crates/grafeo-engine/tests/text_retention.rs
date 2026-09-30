//! Text GC must retain an honest historical boundary through actual copies.

#![cfg(all(feature = "lpg", feature = "text-index", feature = "grafeo-file"))]

use grafeo_common::types::{EpochId, NodeId, TransactionId, Value};
use grafeo_core::graph::GraphStoreSearch;
use grafeo_engine::{Config, CreateIndexRequest, GrafeoDB, IndexCreateKind};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn check_retained_copy(
    db: &GrafeoDB,
    node: NodeId,
    before: EpochId,
    floor: EpochId,
    score_bits: u64,
) -> TestResult {
    let store = grafeo_engine::database::testing::root_lpg_store(db);
    let text = store.get_text_index("Doc", "body").ok_or("missing Text")?;
    let index = text.read();
    assert_eq!(index.retained_from(), floor);
    assert!(index.doc_count_at(before, TransactionId::INVALID).is_err());
    assert!(
        index
            .score_document_visible(node, "", before, TransactionId::INVALID, None, false)
            .is_err()
    );
    assert_eq!(
        index
            .score_document_visible(node, "modern", floor, TransactionId::INVALID, None, false)?
            .map(f64::to_bits),
        Some(score_bits)
    );
    drop(index);
    assert!(
        store
            .text_search_visible("Doc", "body", "", 0, before, TransactionId::INVALID)
            .is_err()
    );
    assert_eq!(db.text_search("Doc", "body", "modern", 10)?[0].0, node);
    #[cfg(feature = "gql")]
    {
        let session = db.session();
        session.set_viewing_epoch(before);
        for query in [
            "MATCH (n:Doc) WHERE text_match(n.body, 'ancient') RETURN n.body",
            "MATCH (n:Doc) WHERE text_score(n.body, 'ancient') > 0 RETURN n.body",
            "MATCH (n:Doc) WHERE coalesce(text_score(n.body, 'ancient'), 0) > 0 RETURN n.body",
        ] {
            let error = session
                .execute(query)
                .expect_err("expired Text query must fail");
            assert!(error.to_string().contains("retained"), "{error}");
        }
    }
    Ok(())
}

#[test]
fn gc_respects_active_reader_then_persists_floor_through_snapshot_and_container() -> TestResult {
    let db = GrafeoDB::with_config(Config::in_memory().with_gc_interval(0))?;
    let node = db.create_node_with_props(&["Doc"], [("body", Value::from("ancient graph"))]);
    db.create_index(CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
        property: "body".into(),
        kind: IndexCreateKind::Text {
            min_token_length: None,
        },
    })?;
    let before = db.current_epoch();
    let mut reader = db.session();
    reader.begin_transaction()?;
    db.set_node_property(node, "body", Value::from("modern graph database"))?;
    let floor = db.current_epoch();
    assert!(floor > before);
    db.gc()?;
    let text = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_text_index("Doc", "body")
        .ok_or("missing Text")?;
    assert_eq!(text.read().retained_from(), before);
    assert!(
        text.read()
            .score_document_visible(node, "ancient", before, TransactionId::INVALID, None, false)?
            .is_some()
    );
    reader.rollback()?;
    drop(reader);
    db.gc()?;
    let score_bits = text
        .read()
        .score_document_visible(node, "modern", floor, TransactionId::INVALID, None, false)?
        .ok_or("missing retained score")?
        .to_bits();
    check_retained_copy(&db, node, before, floor, score_bits)?;

    let snapshot = db.export_snapshot()?;
    let imported = GrafeoDB::import_snapshot(&snapshot)?;
    check_retained_copy(&imported, node, before, floor, score_bits)?;
    check_retained_copy(&db.to_memory()?, node, before, floor, score_bits)?;

    let directory = tempfile::tempdir()?;
    let path = directory.path().join("retained.grafeo");
    db.save(&path)?;
    let reopened = GrafeoDB::open(&path)?;
    check_retained_copy(&reopened, node, before, floor, score_bits)?;
    Ok(())
}

#[cfg(feature = "compact-store")]
#[test]
fn compact_hydration_preserves_text_history_through_real_database_mutation() -> TestResult {
    use grafeo_common::storage::section::Section;
    use grafeo_core::graph::lpg::PhysicalIndexKey;
    use grafeo_core::index::text::TextIndexSection;

    let mut db = GrafeoDB::with_config(Config::in_memory().with_gc_interval(0))?;
    db.create_index(CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some("Doc".into()),
        property: "body".into(),
        kind: IndexCreateKind::Text {
            min_token_length: None,
        },
    })?;
    let node = db.create_node_with_props(&["Doc"], [("body", Value::from("original text"))]);
    assert!(node.is_valid());
    let historical = db.current_epoch();
    db.set_node_property(node, "body", Value::from("current text"))?;
    db.compact()?;
    let text = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_text_index("Doc", "body")
        .ok_or("missing Text")?;
    let section = TextIndexSection::from_views(vec![(
        PhysicalIndexKey::text(Default::default(), "Doc", "body"),
        text,
    )]);
    let before = section.serialize()?;
    let scores = db.graph_store().text_search_visible(
        "Doc",
        "body",
        "original",
        10,
        historical,
        TransactionId::INVALID,
    )?;
    assert_eq!(scores.first().map(|(id, _)| *id), Some(node));
    db.set_node_property(node, "unrelated", Value::Bool(true))?;
    assert_eq!(
        section.serialize()?,
        before,
        "an unrelated write must not rewrite Text history during hydration"
    );
    assert_eq!(
        db.graph_store().text_search_visible(
            "Doc",
            "body",
            "original",
            10,
            historical,
            TransactionId::INVALID,
        )?,
        scores
    );
    Ok(())
}
