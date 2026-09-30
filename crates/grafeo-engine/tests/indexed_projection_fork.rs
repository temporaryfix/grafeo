//! Logical forks discard projection rows, not unrelated exact index owners.
//! Vector reference roles are covered by core wire-scan tests; canonical
//! projected rows cannot carry vectors.
#![cfg(all(feature = "lpg", feature = "triple-store"))]

#[cfg(any(feature = "text-index", feature = "vector-index"))]
use std::sync::Arc;

#[cfg(any(feature = "text-index", feature = "vector-index"))]
use grafeo_common::storage::Section;
#[cfg(feature = "text-index")]
use grafeo_common::types::TransactionId;
use grafeo_common::types::{EpochId, GraphPath, IndexId, NodeId, Value};
#[cfg(any(feature = "text-index", feature = "vector-index"))]
use grafeo_core::graph::lpg::{LpgStoreSection, PhysicalIndexKey, decode_index_key};
use grafeo_core::graph::rdf::{Term, Triple};
#[cfg(feature = "text-index")]
use grafeo_core::index::text::TextIndexSection;
#[cfg(feature = "vector-index")]
use grafeo_core::index::vector::VectorStoreSection;
use grafeo_engine::{Config, CreateIndexRequest, GrafeoDB, GraphModel, IndexCreateKind};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

const CLASS: &str = "http://fork.test/Person";
const SUBJECT: &str = "http://fork.test/alix";

fn projected_source() -> TestResult<(GrafeoDB, u64, NodeId, EpochId)> {
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Both)
            .with_gc_interval(0),
    )?;
    assert_eq!(
        db.batch_insert_rdf([Triple::new(
            Term::iri(SUBJECT),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri(CLASS),
        )])?,
        1
    );
    let projection = db.declare_rdf_lpg_projection(CLASS, "Projected")?;
    assert_eq!(db.rebuild_rdf_lpg_projection(projection)?, 1);
    let rows = db.find_nodes_by_property("iri", &Value::from(SUBJECT));
    assert_eq!(rows, vec![NodeId::new(0)]);
    let epoch = db.current_epoch();
    Ok((db, projection, rows[0], epoch))
}

fn property_request(property: &str, kind: IndexCreateKind) -> CreateIndexRequest {
    CreateIndexRequest {
        graph: GraphPath::root(),
        name: Some(format!("owner_{property}")),
        label: None,
        property: property.into(),
        kind,
    }
}

fn assert_pending_fork(
    source: &GrafeoDB,
    fork: &GrafeoDB,
    projection: u64,
    removed: NodeId,
    source_epoch: EpochId,
) -> TestResult {
    assert_ne!(fork.store_id(), source.store_id());
    assert_eq!(fork.rdf_store().store_id(), fork.store_id());
    let original = source
        .rdf_lpg_projection(projection)
        .ok_or("source mapping missing")?;
    let pending = fork
        .rdf_lpg_projection(projection)
        .ok_or("fork mapping missing")?;
    assert_eq!(pending.mapping_digest(), original.mapping_digest());
    assert_eq!(pending.generation(), 0);
    assert_eq!(pending.row_count(), 0);
    assert_eq!(pending.last_source_epoch(), None);
    assert_eq!(pending.last_target_epoch(), None);
    assert_eq!(pending.receipt(), None);
    assert!(fork.get_node(removed).is_none());
    assert!(fork.get_node_at_epoch(removed, source_epoch).is_none());
    assert!(
        fork.find_nodes_by_property("iri", &Value::from(SUBJECT))
            .is_empty()
    );
    Ok(())
}

#[test]
fn property_and_btree_owners_rebuild_without_foreign_projection_rows() -> TestResult {
    for kind in [IndexCreateKind::Property, IndexCreateKind::BTree] {
        let (source, projection, row, epoch) = projected_source()?;
        let owner = source.create_index(property_request("iri", kind))?;
        assert_eq!(owner, IndexId::new(0));
        let retired =
            source.create_index(property_request("retired", IndexCreateKind::Property))?;
        assert_eq!(retired, IndexId::new(1));
        assert!(source.drop_index(retired)?);
        assert_eq!(
            source.find_nodes_by_property("iri", &Value::from(SUBJECT)),
            vec![row]
        );
        let before = source.export_snapshot()?;
        let cut = source.world_cut()?;
        for fork in [
            source.to_memory()?,
            GrafeoDB::import_snapshot_as_fork(&before)?,
        ] {
            assert_pending_fork(&source, &fork, projection, row, epoch)?;
            assert_eq!(fork.node_count(), 0);
            assert!(fork.has_property_index("iri"));
            assert_eq!(
                fork.create_index(property_request("after_fork", IndexCreateKind::Property))?,
                IndexId::new(2),
                "the dropped owner's allocator floor survives"
            );
            assert_eq!(fork.rebuild_rdf_lpg_projection(projection)?, 1);
            let rebuilt = fork.find_nodes_by_property("iri", &Value::from(SUBJECT));
            assert_eq!(
                rebuilt,
                vec![NodeId::new(1)],
                "removed row IDs are not reused"
            );
            assert!(fork.has_property_index("iri"));
            assert_eq!(
                fork.rdf_lpg_projection(projection)
                    .ok_or("rebuilt mapping missing")?
                    .receipt()
                    .ok_or("rebuilt receipt missing")?
                    .store_id(),
                fork.store_id()
            );
            assert!(fork.drop_index(owner)?);
            assert!(!fork.has_property_index("iri"));
        }
        assert_eq!(source.export_snapshot()?, before);
        assert_eq!(source.world_cut()?, cut);
    }
    Ok(())
}

#[cfg(feature = "text-index")]
fn text_image(db: &GrafeoDB) -> TestResult<Vec<u8>> {
    let mut entries = Vec::new();
    for (path, store) in LpgStoreSection::new(Arc::clone(
        grafeo_engine::database::testing::root_lpg_store(db),
    ))
    .capture_graphs()?
    {
        for (key, index) in store.text_index_entries() {
            let (label, property) = decode_index_key(&key).ok_or("invalid Text key")?;
            entries.push((PhysicalIndexKey::text(path.clone(), label, property), index));
        }
    }
    Ok(TextIndexSection::from_views(entries).serialize()?)
}

#[cfg(feature = "vector-index")]
fn vector_image(db: &GrafeoDB) -> TestResult<Vec<u8>> {
    let mut entries = Vec::new();
    for (path, store) in LpgStoreSection::new(Arc::clone(
        grafeo_engine::database::testing::root_lpg_store(db),
    ))
    .capture_graphs()?
    {
        for (key, index) in store.vector_index_entries() {
            let (label, property) = decode_index_key(&key).ok_or("invalid Vector key")?;
            entries.push((
                PhysicalIndexKey::vector(path.clone(), label, property),
                index,
            ));
        }
    }
    Ok(VectorStoreSection::from_views(entries).serialize()?)
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
fn unrelated_index_survives(
    kind: IndexCreateKind,
    property: &str,
    image: fn(&GrafeoDB) -> TestResult<Vec<u8>>,
) -> TestResult {
    for named in [false, true] {
        let (source, projection, row, epoch) = projected_source()?;
        let path = if named {
            assert!(source.create_graph("documents")?);
            GraphPath::from_components(&["documents"])?
        } else {
            GraphPath::root()
        };
        let session = source.session();
        session.use_graph_path(&path)?;
        let document = session.create_node_with_props(
            &["Doc"],
            [
                ("body", Value::from("unrelated exact document")),
                ("embedding", Value::Vector(vec![1.0, 0.0, 0.0].into())),
                ("rank", Value::Int64(7)),
            ],
        )?;
        drop(session);
        if named {
            assert_eq!(
                document, row,
                "equal local IDs in different graphs do not overlap"
            );
        } else {
            assert_ne!(document, row);
        }
        let property_owner =
            source.create_index(property_request("iri", IndexCreateKind::Property))?;
        let btree_owner = source.create_index(property_request("rank", IndexCreateKind::BTree))?;
        let auxiliary_owner = source.create_index(CreateIndexRequest {
            graph: path.clone(),
            name: Some("exact_document_owner".into()),
            label: Some("Doc".into()),
            property: property.into(),
            kind: kind.clone(),
        })?;
        assert_eq!(
            (property_owner, btree_owner, auxiliary_owner),
            (IndexId::new(0), IndexId::new(1), IndexId::new(2))
        );
        let retired =
            source.create_index(property_request("retired", IndexCreateKind::Property))?;
        assert_eq!(retired, IndexId::new(3));
        assert!(source.drop_index(retired)?);
        let source_store = if named {
            grafeo_engine::database::testing::root_lpg_store(&source)
                .graph("documents")
                .ok_or("source graph missing")?
        } else {
            Arc::clone(grafeo_engine::database::testing::root_lpg_store(&source))
        };
        match kind {
            #[cfg(feature = "text-index")]
            IndexCreateKind::Text { .. } => assert_eq!(
                source_store
                    .get_text_index("Doc", property)
                    .ok_or("Text not installed")?
                    .read()
                    .doc_count_at(source.current_epoch(), TransactionId::INVALID)?,
                1
            ),
            #[cfg(feature = "vector-index")]
            IndexCreateKind::Vector { .. } => assert_eq!(
                source_store
                    .get_vector_index("Doc", property)
                    .ok_or("Vector not installed")?
                    .len(),
                1
            ),
            _ => return Err("expected an auxiliary index family".into()),
        }
        let exact = image(&source)?;
        let before = source.export_snapshot()?;
        let cut = source.world_cut()?;
        for fork in [
            source.to_memory()?,
            GrafeoDB::import_snapshot_as_fork(&before)?,
        ] {
            assert_pending_fork(&source, &fork, projection, row, epoch)?;
            assert_eq!(
                image(&fork)?,
                exact,
                "full exact state survives, not a rebuilt index"
            );
            let target = if named {
                grafeo_engine::database::testing::root_lpg_store(&fork)
                    .graph("documents")
                    .ok_or("fork graph missing")?
            } else {
                Arc::clone(grafeo_engine::database::testing::root_lpg_store(&fork))
            };
            assert!(target.get_node(document).is_some());
            assert!(fork.has_property_index("iri"));
            assert!(fork.has_property_index("rank"));
            assert_eq!(
                fork.find_nodes_by_property("rank", &Value::Int64(7)),
                if named { vec![] } else { vec![document] }
            );
            assert_eq!(
                fork.create_index(property_request("after_fork", IndexCreateKind::Property))?,
                IndexId::new(4)
            );
            assert!(fork.drop_index(property_owner)?);
            assert!(fork.drop_index(btree_owner)?);
            assert!(fork.drop_index(auxiliary_owner)?);
            assert!(!fork.has_property_index("iri"));
            assert!(!fork.has_property_index("rank"));
            assert_ne!(
                image(&fork)?,
                exact,
                "the original auxiliary owner still addresses its image"
            );
        }
        assert_eq!(image(&source)?, exact);
        assert_eq!(source.export_snapshot()?, before);
        assert_eq!(source.world_cut()?, cut);
    }
    Ok(())
}

#[cfg(feature = "text-index")]
#[test]
fn unrelated_root_and_same_local_id_named_text_images_survive_forks() -> TestResult {
    unrelated_index_survives(
        IndexCreateKind::Text {
            min_token_length: None,
        },
        "body",
        text_image,
    )
}

#[cfg(feature = "vector-index")]
#[test]
fn unrelated_root_and_same_local_id_named_vector_images_survive_forks() -> TestResult {
    unrelated_index_survives(
        IndexCreateKind::Vector {
            dimensions: Some(3),
            metric: Some("cosine".into()),
            m: Some(4),
            ef_construction: Some(32),
            ef: None,
            quantization: None,
        },
        "embedding",
        vector_image,
    )
}

#[cfg(feature = "text-index")]
fn index_projection_text(source: &GrafeoDB) -> TestResult {
    let owner = source.create_index(CreateIndexRequest {
        graph: GraphPath::root(),
        name: Some("projection_text".into()),
        label: Some("Projected".into()),
        property: "iri".into(),
        kind: IndexCreateKind::Text {
            min_token_length: None,
        },
    })?;
    assert_eq!(owner, IndexId::new(0));
    assert_eq!(
        grafeo_engine::database::testing::root_lpg_store(source)
            .get_text_index("Projected", "iri")
            .ok_or("Text missing")?
            .read()
            .doc_count_at(source.current_epoch(), TransactionId::INVALID)?,
        1
    );
    Ok(())
}

#[cfg(feature = "text-index")]
fn assert_text_overlap_refused(source: &GrafeoDB) -> TestResult {
    let before = source.export_snapshot()?;
    let exact = text_image(source)?;
    let cut = source.world_cut()?;
    // The source is a valid exact replica; only changing its identity is refused.
    assert_eq!(
        GrafeoDB::import_snapshot(&before)?.export_snapshot()?,
        before
    );
    for result in [
        source.to_memory(),
        GrafeoDB::import_snapshot_as_fork(&before),
    ] {
        let error = result
            .err()
            .ok_or("projection Text overlap unexpectedly forked")?;
        let message = error.to_string();
        assert!(
            message.contains("exact Text index state") && message.contains("projection rows"),
            "wrong refusal: {message}"
        );
        assert_eq!(source.export_snapshot()?, before);
        assert_eq!(text_image(source)?, exact);
        assert_eq!(source.world_cut()?, cut);
    }
    Ok(())
}

#[cfg(feature = "text-index")]
#[test]
fn current_projection_text_overlap_refuses_both_fork_callers_without_mutation() -> TestResult {
    let (source, _, _, _) = projected_source()?;
    index_projection_text(&source)?;
    assert_text_overlap_refused(&source)
}

#[cfg(all(feature = "text-index", feature = "sparql"))]
#[test]
fn historical_only_projection_text_overlap_refuses_both_fork_callers() -> TestResult {
    let (source, projection, old_row, _) = projected_source()?;
    index_projection_text(&source)?;
    let indexed_epoch = source.current_epoch();
    source.execute_sparql(&format!("DELETE DATA {{ <{SUBJECT}> a <{CLASS}> . }}"))?;
    assert_eq!(source.rebuild_rdf_lpg_projection(projection)?, 0);
    assert!(source.get_node(old_row).is_none());
    assert!(source.get_node_at_epoch(old_row, indexed_epoch).is_some());
    assert_eq!(source.node_count(), 0);
    let index = grafeo_engine::database::testing::root_lpg_store(&source)
        .get_text_index("Projected", "iri")
        .ok_or("Text missing")?;
    {
        let index = index.read();
        assert_eq!(
            index.doc_count_at(source.current_epoch(), TransactionId::INVALID)?,
            0
        );
        assert_eq!(
            index.doc_count_at(indexed_epoch, TransactionId::INVALID)?,
            1
        );
        assert!(
            index
                .score_document_visible(
                    old_row,
                    "alix",
                    indexed_epoch,
                    TransactionId::INVALID,
                    None,
                    false
                )?
                .is_some()
        );
    }
    assert_text_overlap_refused(&source)
}
