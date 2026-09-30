//! Exact recursive index persistence through the real engine save/open path.
//! Raw fixture topology and deliberate mismatch injection use the private owner
//! scope; data and catalog owners use the public Session and index APIs.

#![cfg(all(
    feature = "lpg",
    feature = "wal",
    feature = "grafeo-file",
    feature = "vector-index",
    feature = "text-index"
))]

use super::GrafeoDB;
use crate::{Config, GraphModel};
use grafeo_common::storage::{Section, SectionType};
use grafeo_common::types::{EpochId, Value};
use grafeo_core::graph::lpg::{LpgStoreSection, PhysicalIndexKey, decode_index_key};
use grafeo_core::graph::{GraphStoreSearch, PropertyIndexPredicate, PropertyIndexRequest};
use grafeo_storage::file::GrafeoFileManager;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

#[cfg(all(feature = "vector-index", feature = "text-index"))]
fn recursive_auxiliary_images(db: &GrafeoDB) -> TestResult<(Vec<u8>, Vec<u8>)> {
    // These fixtures are quiescent. Production capture additionally retains
    // the engine publication barrier across all of these section reads.
    let graphs = LpgStoreSection::new(std::sync::Arc::clone(
        crate::database::testing::root_lpg_store(db),
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

#[cfg(all(feature = "vector-index", feature = "text-index"))]
#[test]
fn recursive_owned_indexes_preserve_exact_images_and_literal_slash_identity() -> TestResult {
    qualify_recursive_owned_indexes(false, |_| Ok(()))
}

#[cfg(feature = "compact-store")]
#[test]
fn recursive_owned_indexes_preserve_exact_images_through_compact_reopen_recompact() -> TestResult {
    qualify_recursive_owned_indexes(true, |db| {
        let expected_nodes = db.node_count();
        db.compact()?;
        assert!(db.layered_store().is_some());
        assert_eq!(
            crate::database::testing::root_lpg_store(db).node_count(),
            0,
            "root rows must live in the compact base"
        );
        assert_eq!(db.node_count(), expected_nodes);
        Ok(())
    })
}

fn qualify_recursive_owned_indexes(
    include_root: bool,
    maintenance: impl Fn(&mut GrafeoDB) -> TestResult,
) -> TestResult {
    use crate::{CreateIndexRequest, IndexCreateKind};
    use grafeo_common::types::{GraphPath, TransactionId};
    use grafeo_core::index::vector::PropertyVectorAccessor;
    use std::sync::Arc;

    let dir = tempfile::TempDir::new()?;
    let path = dir.path().join("recursive-owned-indexes.grafeo");
    let resaved = dir.path().join("recursive-owned-indexes-resaved.grafeo");
    let deep_path = GraphPath::from_components(&["nested", "", "deep"])?;
    let literal_path = GraphPath::from_components(&["nested//deep"])?;
    let root_path = GraphPath::root();
    assert_ne!(deep_path, literal_path);
    let mut db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Lpg)
            .with_gc_interval(0),
    )?;
    // In-memory databases do not automatically seal raw stores. Bind this
    // fixture to its real manager before qualifying retained-handle denial.
    assert!(
        crate::database::testing::root_lpg_store(&db)
            .seal_unframed_writes(db.transaction_manager.write_authority())
    );
    db.transaction_manager
        .with_write_authority(|| -> TestResult {
            assert!(
                crate::database::testing::root_lpg_store(&db)
                    .graph_or_create("nested")?
                    .graph_or_create("")?
                    .create_graph("deep")?
            );
            assert!(crate::database::testing::root_lpg_store(&db).create_graph("nested//deep")?);
            Ok(())
        })?;

    let mut nodes = Vec::new();
    let mut tombstone_nodes = Vec::new();
    let mut owners = Vec::new();
    let mut retained = Vec::new();
    let mut cases = vec![
        (&deep_path, "deepneedle", [1.0, 0.0, 0.0]),
        (&literal_path, "literalneedle", [0.0, 1.0, 0.0]),
    ];
    if include_root {
        cases.push((&root_path, "rootneedle", [1.0, 0.0, 0.0]));
    }
    for &(graph, name, direction) in &cases {
        let session = db.session();
        session.use_graph_path(graph)?;
        let node = session.create_node_with_props(
            &["Doc"],
            [
                ("email", Value::from(name)),
                ("body", Value::from(format!("{name} initial revision"))),
                ("embedding", Value::Vector(direction.to_vec().into())),
            ],
        )?;
        session.create_node_with_props(
            &["Doc"],
            [
                ("email", Value::from(format!("{name}-other"))),
                ("body", Value::from("unrelated document")),
                ("embedding", Value::Vector(vec![0.0, 0.0, 1.0].into())),
            ],
        )?;
        let tombstone = session.create_node_with_props(
            &["Doc"],
            [("email", Value::from(format!("{name}-deleted")))],
        )?;
        nodes.push(node);
        tombstone_nodes.push(tombstone);
        for (property, label, suffix, kind) in [
            ("email", None, "property", IndexCreateKind::Property),
            (
                "body",
                Some("Doc"),
                "text",
                IndexCreateKind::Text {
                    min_token_length: None,
                },
            ),
            (
                "embedding",
                Some("Doc"),
                "vector",
                IndexCreateKind::Vector {
                    dimensions: Some(3),
                    metric: Some("cosine".into()),
                    m: Some(8),
                    ef_construction: Some(64),
                    ef: None,
                    quantization: if graph == &deep_path {
                        Some("scalar".into())
                    } else {
                        None
                    },
                },
            ),
        ] {
            let owner_name = format!("{name}-{suffix}");
            let owner = db.create_index(CreateIndexRequest {
                graph: graph.clone(),
                name: Some(owner_name.clone()),
                label: label.map(str::to_owned),
                property: property.into(),
                kind,
            })?;
            owners.push((owner, owner_name));
        }
        let graph_store =
            LpgStoreSection::new(Arc::clone(crate::database::testing::root_lpg_store(&db)))
                .capture_graphs()?
                .into_iter()
                .find(|(path, _)| path == graph)
                .ok_or("source graph missing")?
                .1;
        let text = graph_store
            .get_text_index("Doc", "body")
            .ok_or("source Text missing")?;
        let vector = graph_store
            .get_vector_index("Doc", "embedding")
            .ok_or("source Vector missing")?;
        let recorded = db.current_epoch();
        let score = text
            .read()
            .score_document_visible(
                node,
                "initial",
                recorded,
                TransactionId::INVALID,
                None,
                false,
            )?
            .ok_or("source historical Text score missing")?;
        assert!(score > 0.0);
        retained.push((graph_store, text, vector, recorded, score.to_bits()));
        // Preserve actual Text revision history, not merely current documents.
        session.set_node_property(node, "body", Value::from(format!("{name} final revision")))?;
    }
    assert_eq!(nodes[0], nodes[1], "graph-local IDs deliberately collide");
    let owner_ids: std::collections::HashSet<_> = owners.iter().map(|(id, _)| *id).collect();
    assert_eq!(owner_ids.len(), cases.len() * 3);
    let owner_names: std::collections::HashSet<_> = owners.iter().map(|(_, name)| name).collect();
    assert_eq!(owner_names.len(), cases.len() * 3);

    // Leave the owner allocator strictly beyond every remaining owner. Exact
    // Catalog7 equality below also checks this otherwise-unobservable floor.
    let retired_owner = db.create_index(CreateIndexRequest {
        graph: deep_path.clone(),
        name: Some("retired-owner-floor".into()),
        label: None,
        property: "retired".into(),
        kind: IndexCreateKind::Property,
    })?;
    assert!(owners.iter().all(|(owner, _)| owner.0 < retired_owner.0));
    assert!(db.drop_index(retired_owner)?);

    let mut owners_before = db.catalog.all_indexes();
    owners_before.sort_unstable_by_key(|owner| owner.id);
    let allocator_before = db.catalog.index_allocator_high_water();
    assert_eq!(allocator_before, retired_owner.as_u32() + 1);
    let auxiliary_before = recursive_auxiliary_images(&db)?;
    maintenance(&mut db)?;
    assert_eq!(recursive_auxiliary_images(&db)?, auxiliary_before);
    for ((graph, text, vector, recorded, score), node) in retained.iter().zip(&nodes) {
        assert!(!graph.remove_text_index("Doc", "body"));
        assert!(!graph.remove_vector_index("Doc", "embedding"));
        assert!(!graph.delete_node(*node));
        assert_eq!(
            text.read()
                .score_document_visible(
                    *node,
                    "initial",
                    *recorded,
                    TransactionId::INVALID,
                    None,
                    false,
                )?
                .map(f64::to_bits),
            Some(*score)
        );
        assert!(vector.contains(*node));
    }
    assert_eq!(recursive_auxiliary_images(&db)?, auxiliary_before);

    // Keep one base-only row deleted in the overlay.  The persisted
    // OverlayDeletions record must reach the index-image rebuild with its
    // delete epoch intact: the row is absent now but remains an indexed hit
    // at the pre-delete retained snapshot.
    #[cfg(feature = "compact-store")]
    let deleted_root = if include_root {
        let root_index = cases
            .iter()
            .position(|(path, _, _)| *path == &root_path)
            .ok_or("compact tombstone fixture has no root graph")?;
        let before_delete = db.current_epoch();
        let deleted = tombstone_nodes[root_index];
        let deleted_value = Value::from(format!("{}-deleted", cases[root_index].1));
        assert!(db.delete_node(deleted));
        Some((deleted, deleted_value, before_delete))
    } else {
        None
    };
    db.save(&path)?;
    db.close()?;
    let sections_before = persisted_catalog_and_index_images(&path)?;
    assert_eq!(sections_before.len(), 3);
    for (kind, version, _) in &sections_before {
        let expected = match kind {
            SectionType::Catalog => 7,
            SectionType::TextIndex => 5,
            _ => 4,
        };
        assert_eq!(*version, expected);
    }

    for (pass, source) in [&path, &resaved].into_iter().enumerate() {
        let mut reopened = GrafeoDB::open(source)?;
        assert_eq!(recursive_auxiliary_images(&reopened)?, auxiliary_before);
        let mut owners_after = reopened.catalog.all_indexes();
        owners_after.sort_unstable_by_key(|owner| owner.id);
        assert_eq!(owners_after, owners_before);
        assert_eq!(
            reopened.catalog.index_allocator_high_water(),
            allocator_before
        );
        let restored_graphs = LpgStoreSection::new(Arc::clone(
            crate::database::testing::root_lpg_store(&reopened),
        ))
        .capture_graphs()?;
        for (index, &(graph_path, token, query)) in cases.iter().enumerate() {
            let node = nodes[index];
            let other_token = if token == "deepneedle" {
                "literalneedle"
            } else {
                "deepneedle"
            };
            let (_, graph) = restored_graphs
                .iter()
                .find(|(path, _)| path == graph_path)
                .ok_or("recursive graph missing after reopen")?;
            let reader: Arc<dyn GraphStoreSearch> = if graph_path == &root_path {
                reopened.graph_store()
            } else {
                Arc::clone(graph) as Arc<dyn GraphStoreSearch>
            };
            assert!(graph.has_property_index("email"));
            assert_eq!(
                reader.find_nodes_by_property("email", &Value::from(token)),
                vec![node]
            );
            assert!(
                reader
                    .find_nodes_by_property("email", &Value::from(other_token))
                    .is_empty()
            );

            #[cfg(feature = "compact-store")]
            if graph_path == &root_path
                && let Some((deleted, deleted_value, before_delete)) = deleted_root.as_ref()
            {
                let current = reader
                    .lookup_nodes_indexed(PropertyIndexRequest {
                        property: "email",
                        predicate: PropertyIndexPredicate::Equal(deleted_value),
                        epoch: reopened.current_epoch(),
                        transaction_id: None,
                    })?
                    .ok_or("deleted cold row lost the property index registration")?;
                assert!(
                    current.is_empty(),
                    "deleted cold row reappeared in current indexed results: {current:?}"
                );
                let floor = graph.retained_history_floor();
                if *before_delete >= floor {
                    let old = reader
                        .lookup_nodes_indexed(PropertyIndexRequest {
                            property: "email",
                            predicate: PropertyIndexPredicate::Equal(deleted_value),
                            epoch: *before_delete,
                            transaction_id: None,
                        })?
                        .ok_or("retained indexed lookup was not admitted")?;
                    assert_eq!(old, vec![*deleted]);
                }
            }

            // Exercise the actual indexed reader after a compact reopen. In
            // the root graph this hit exists only in the cold base, so a
            // generic result-equality query would not detect a lost rebuilt
            // property index. Measure the overlay counters because the
            // LayeredStore delegates its index probe to that representation.
            let epoch = reopened.current_epoch();
            let expected = vec![node];
            let needle = Value::from(token);
            let before = graph.work_snapshot();
            let eq = reader
                .lookup_nodes_indexed(PropertyIndexRequest {
                    property: "email",
                    predicate: PropertyIndexPredicate::Equal(&needle),
                    epoch,
                    transaction_id: None,
                })?
                .ok_or("reopened property index did not admit equality")?;
            let in_values = [needle.clone()];
            let indexed_in = reader
                .lookup_nodes_indexed(PropertyIndexRequest {
                    property: "email",
                    predicate: PropertyIndexPredicate::In(&in_values),
                    epoch,
                    transaction_id: None,
                })?
                .ok_or("reopened property index did not admit IN")?;
            let indexed_range = reader
                .lookup_nodes_indexed(PropertyIndexRequest {
                    property: "email",
                    predicate: PropertyIndexPredicate::Range {
                        min: Some(&needle),
                        max: Some(&needle),
                        min_inclusive: true,
                        max_inclusive: true,
                    },
                    epoch,
                    transaction_id: None,
                })?
                .ok_or("reopened property index did not admit range")?;
            assert_eq!(eq, expected, "reopened indexed equality diverged");
            assert_eq!(indexed_in, expected, "reopened indexed IN diverged");
            assert_eq!(indexed_range, expected, "reopened indexed range diverged");
            let work = graph.work_snapshot().since(before);
            assert!(
                !work.scanned_any(),
                "reopened indexed lookup scanned: {work:?}"
            );

            // If the compacted image still retains this earlier snapshot,
            // verify the same three predicates against that floor as well.
            let retained_epoch = graph.retained_history_floor();
            if retained_epoch > EpochId::INITIAL && retained_epoch < epoch {
                let before = graph.work_snapshot();
                let retained_in_values = [needle.clone()];
                for predicate in [
                    PropertyIndexPredicate::Equal(&needle),
                    PropertyIndexPredicate::In(&retained_in_values),
                    PropertyIndexPredicate::Range {
                        min: Some(&needle),
                        max: Some(&needle),
                        min_inclusive: true,
                        max_inclusive: true,
                    },
                ] {
                    let rows = reader
                        .lookup_nodes_indexed(PropertyIndexRequest {
                            property: "email",
                            predicate,
                            epoch: retained_epoch,
                            transaction_id: None,
                        })?
                        .ok_or("retained indexed lookup was not admitted")?;
                    assert_eq!(rows, expected);
                }
                let work = graph.work_snapshot().since(before);
                assert!(
                    !work.scanned_any(),
                    "retained indexed lookup scanned: {work:?}"
                );
            }
            let text = graph
                .get_text_index("Doc", "body")
                .ok_or("restored Text index missing")?;
            assert_eq!(
                text.read().search(token, 10).first().map(|hit| hit.0),
                Some(node)
            );
            assert!(text.read().search(other_token, 10).is_empty());
            let (_, _, _, recorded, score) = &retained[index];
            assert_eq!(
                text.read()
                    .score_document_visible(
                        node,
                        "initial",
                        *recorded,
                        TransactionId::INVALID,
                        None,
                        false,
                    )?
                    .map(f64::to_bits),
                Some(*score)
            );
            assert_eq!(text.read().score_document(node, "initial"), 0.0);
            let vector = graph
                .get_vector_index("Doc", "embedding")
                .ok_or("restored Vector index missing")?;
            assert_eq!(vector.config().dimensions, 3);
            assert_eq!(vector.config().m, 8);
            assert_eq!(vector.config().ef_construction, 64);
            let accessor = PropertyVectorAccessor::new(reader.as_ref(), "embedding");
            assert_eq!(
                vector.search(&query, 1, &accessor).first().map(|hit| hit.0),
                Some(node)
            );
            assert!(!graph.remove_text_index("Doc", "body"));
            assert!(!graph.remove_vector_index("Doc", "embedding"));
            assert!(!graph.delete_node(node));
        }
        assert_eq!(recursive_auxiliary_images(&reopened)?, auxiliary_before);
        // No mutation after reopen: these exact bytes cover owner IDs, names,
        // resolved configurations, dictionaries and the consumed owner-ID floor.
        if pass == 0 {
            maintenance(&mut reopened)?;
            assert_eq!(recursive_auxiliary_images(&reopened)?, auxiliary_before);
            reopened.save(&resaved)?;
            assert_eq!(
                persisted_catalog_and_index_images(&resaved)?,
                sections_before
            );
        }
        reopened.close()?;
    }
    Ok(())
}

#[cfg(all(feature = "vector-index", feature = "text-index"))]
#[test]
fn save_rejects_physical_tokenizer_mismatch_without_replacing_destination() -> TestResult {
    use crate::{CreateIndexRequest, IndexCreateKind};
    use grafeo_common::types::GraphPath;
    use grafeo_core::index::text::{BM25Config, InvertedIndex};
    use std::sync::Arc;

    let dir = tempfile::TempDir::new()?;
    let path = dir.path().join("owned-tokenizer.grafeo");
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg))?;
    let graph = db.transaction_manager.with_write_authority(|| {
        crate::database::testing::root_lpg_store(&db)
            .graph_or_create("parent")?
            .graph_or_create("child")
    })?;
    let graph_path = GraphPath::from_components(&["parent", "child"])?;
    let session = db.session();
    session.use_graph_path(&graph_path)?;
    session.create_node_with_props(&["Doc"], [("body", Value::from("indexed document"))])?;
    db.create_index(CreateIndexRequest {
        graph: graph_path,
        name: Some("owned-text".into()),
        label: Some("Doc".into()),
        property: "body".into(),
        kind: IndexCreateKind::Text {
            min_token_length: None,
        },
    })?;
    db.save(&path)?;
    let before = std::fs::read(&path)?;
    db.transaction_manager.with_write_authority(|| {
        graph.add_text_index(
            "Doc",
            "body",
            Arc::new(parking_lot::RwLock::new(
                InvertedIndex::with_simple_tokenizer(BM25Config::default(), 99),
            )),
        );
    });
    assert!(
        graph
            .get_text_index("Doc", "body")
            .ok_or("replacement Text index missing")?
            .read()
            .has_simple_tokenizer(99)
    );
    let result = db.save(&path);
    assert!(
        matches!(result, Err(grafeo_common::utils::error::Error::Serialization(ref message))
        if message.contains("physical Text owner/config mismatch")),
        "{result:?}"
    );
    assert_eq!(std::fs::read(&path)?, before);
    let reopened = GrafeoDB::open(&path)?;
    assert!(
        crate::database::testing::root_lpg_store(&reopened)
            .graph("parent")
            .and_then(|parent| parent.graph("child"))
            .and_then(|child| child.get_text_index("Doc", "body"))
            .is_some_and(|index| index.read().has_simple_tokenizer(2))
    );
    reopened.close()?;
    Ok(())
}
