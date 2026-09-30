//! Real database GC retains old-snapshot vector routing without resurrecting it.
#![cfg(all(feature = "lpg", feature = "vector-index"))]

use grafeo_common::storage::section::Section;
use grafeo_common::types::{GraphPath, PropertyKey, TransactionId, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::graph::lpg::PhysicalIndexKey;
use grafeo_core::index::vector::VectorStoreSection;
use grafeo_engine::{Config, CreateIndexRequest, GrafeoDB, IndexCreateKind};

#[test]
fn database_vector_gc_retains_snapshot_membership_and_collects_retired_membership() -> Result<()> {
    let compact_modes = &[
        false,
        #[cfg(feature = "compact-store")]
        true,
    ];
    for &compact in compact_modes {
        for quantization in [None, Some("scalar"), Some("binary")] {
            let db = GrafeoDB::with_config(Config::in_memory().with_gc_interval(0))?;
            #[cfg(feature = "compact-store")]
            let mut db = db;
            let owner = db.create_index(CreateIndexRequest {
                graph: GraphPath::root(),
                name: Some("retained_vectors".into()),
                label: Some("Doc".into()),
                property: "embedding".into(),
                kind: IndexCreateKind::Vector {
                    dimensions: Some(2),
                    metric: Some("euclidean".into()),
                    m: None,
                    ef_construction: None,
                    ef: None,
                    quantization: quantization.map(str::to_owned),
                },
            })?;
            let create = |db: &GrafeoDB| {
                db.create_node_with_props(
                    &["Doc"],
                    [("embedding", Value::Vector(vec![1.0, 0.0].into()))],
                )
            };
            let gone = create(&db);
            let old_property = create(&db);
            let old_label = create(&db);
            let retained = create(&db);
            let retained_property = create(&db);
            let retained_label = create(&db);
            let base_live = create(&db);
            for id in [
                gone,
                old_property,
                old_label,
                retained,
                retained_property,
                retained_label,
                base_live,
            ] {
                assert!(id.is_valid());
            }
            if compact {
                #[cfg(feature = "compact-store")]
                db.compact()?;
            }
            assert!(db.delete_node(gone));
            assert!(db.remove_node_property(old_property, "embedding"));
            assert!(db.remove_node_label(old_label, "Doc"));
            let horizon = db.current_epoch();
            let mut reader = db.session();
            reader.begin_transaction()?;
            assert!(db.delete_node(retained));
            assert!(db.remove_node_property(retained_property, "embedding"));
            assert!(db.remove_node_label(retained_label, "Doc"));
            let younger = create(&db);
            assert!(younger.is_valid());

            let graph = db.graph_store();
            let historical_before = graph.vector_search_visible(
                "Doc",
                "embedding",
                &[1.0, 0.0],
                20,
                horizon,
                TransactionId::INVALID,
            );
            for id in [retained, retained_property, retained_label] {
                assert!(
                    graph.is_node_visible_versioned(id, horizon, TransactionId::INVALID),
                    "pre-GC node visibility lost {id:?}, compact={compact}, quantization={quantization:?}"
                );
                assert_eq!(
                    graph.read_node_property_visible(
                        id,
                        &PropertyKey::new("embedding"),
                        horizon,
                        None
                    ),
                    Some(Value::Vector(vec![1.0, 0.0].into())),
                    "pre-GC vector lost {id:?}, compact={compact}, quantization={quantization:?}"
                );
                assert!(
                    graph
                        .read_node_labels_visible(id, horizon, None)
                        .iter()
                        .any(|label| label.as_str() == "Doc"),
                    "pre-GC label lost {id:?}, compact={compact}, quantization={quantization:?}"
                );
                assert!(
                    historical_before.iter().any(|(found, _)| *found == id),
                    "pre-GC search lost {id:?}, compact={compact}, quantization={quantization:?}, results={historical_before:?}"
                );
            }

            db.gc()?;
            let index = grafeo_engine::database::testing::root_lpg_store(&db)
                .get_vector_index("Doc", "embedding")
                .ok_or_else(|| Error::InvalidValue("GC lost the vector owner".into()))?;
            let topology: Vec<_> = index
                .snapshot_topology()
                .2
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            for id in [gone, old_property, old_label] {
                assert!(
                    !topology.contains(&id),
                    "retired membership {id:?}, compact={compact}, quantization={quantization:?}"
                );
            }
            for id in [retained, retained_property, retained_label] {
                assert!(topology.contains(&id), "retained membership {id:?}");
                assert!(!index.contains(id), "GC must not resurrect a tombstone");
            }
            assert!(index.contains(base_live));
            assert!(index.contains(younger));
            let historical = graph.vector_search_visible(
                "Doc",
                "embedding",
                &[1.0, 0.0],
                20,
                horizon,
                TransactionId::INVALID,
            );
            for id in [retained, retained_property, retained_label] {
                assert!(
                    historical.iter().any(|(found, _)| *found == id),
                    "post-GC snapshot lost {id:?}, compact={compact}, quantization={quantization:?}, results={historical:?}"
                );
            }
            let current = db.vector_search("Doc", "embedding", &[1.0, 0.0], 20, None, None)?;
            assert!(current.iter().any(|(id, _)| *id == younger));
            assert!(
                current
                    .iter()
                    .all(|(id, _)| [base_live, younger].contains(id))
            );
            let section = VectorStoreSection::from_views(vec![(
                PhysicalIndexKey::vector(GraphPath::root(), "Doc", "embedding"),
                index.clone(),
            )]);
            let before = section.serialize()?;
            db.gc()?;
            assert_eq!(
                section.serialize()?,
                before,
                "same horizon must be an exact no-op"
            );
            reader.rollback()?;
            db.gc()?;
            let topology = index.snapshot_topology().2;
            for id in [retained, retained_property, retained_label] {
                assert!(!topology.iter().any(|(found, _)| *found == id));
            }
            assert!(index.contains(base_live));
            assert!(index.contains(younger));
            assert!(db.drop_index(owner)?);
        }
    }
    Ok(())
}
