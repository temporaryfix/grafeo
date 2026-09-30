use super::{LpgRecoveryTarget, OwnerReplay};
use crate::catalog::{
    Catalog, IndexConfiguration, IndexOwnerBatch, IndexOwnerChange, IndexOwnerImage,
};
use crate::database::index_commit_wire::{IndexCommitBatch, VectorPostimage};
use grafeo_common::types::{EpochId, GraphPath, IndexId, NodeId, TransactionId, Value};
use grafeo_common::utils::error::Result;
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::index::vector::{
    DistanceMetric, HnswConfig, QuantizationType, QuantizedHnswIndex, VectorIndexKind,
};
use std::sync::Arc;

fn payload(owner: &IndexOwnerImage, bytes: Vec<u8>, rebuild: bool) -> Result<Vec<u8>> {
    IndexCommitBatch {
        frontier: if rebuild {
            EpochId::new(1)
        } else {
            EpochId::INITIAL
        },
        commit: if rebuild {
            EpochId::new(2)
        } else {
            EpochId::new(1)
        },
        owners: IndexOwnerBatch {
            expected_floor: u32::from(rebuild),
            next_floor: 1,
            changes: vec![if rebuild {
                IndexOwnerChange::Rebuild(owner.clone())
            } else {
                IndexOwnerChange::Create(owner.clone())
            }],
        },
        text: Vec::new(),
        vectors: vec![VectorPostimage {
            owner: owner.clone(),
            complete: true,
            payload: bytes,
        }],
    }
    .encode()
}

#[test]
fn full_vector_replay_qualifies_population_before_catalog_and_registration_publication()
-> Result<()> {
    for compact in [
        false,
        #[cfg(feature = "compact-store")]
        true,
    ] {
        let source = Arc::new(LpgStore::new()?);
        source.sync_epoch(EpochId::new(1));
        let node = source.create_node_with_props(
            &["Doc"],
            [("emb", Value::Vector(Arc::from([0.2_f32, 0.4])))],
        );
        let excluded = source.create_node_with_props(
            &["Other"],
            [("emb", Value::Vector(Arc::from([0.2_f32, 0.4])))],
        );
        source.create_node_with_props(&["Doc"], [("emb", Value::Int64(7))]);
        let default = if compact {
            #[cfg(feature = "compact-store")]
            {
                LpgRecoveryTarget::Layered(Arc::new(
                    grafeo_core::graph::compact::layered::LayeredStore::from_native_temporal(
                        source,
                    )?,
                ))
            }
            #[cfg(not(feature = "compact-store"))]
            {
                return Err(grafeo_common::Error::Internal(
                    "compact fixture unavailable".into(),
                ));
            }
        } else {
            LpgRecoveryTarget::Flat(source)
        };
        let root = match &default {
            LpgRecoveryTarget::Flat(store) => Arc::clone(store),
            #[cfg(feature = "compact-store")]
            LpgRecoveryTarget::Layered(store) => {
                let overlay = store.overlay_store();
                assert!(
                    overlay.get_node(node).is_none(),
                    "qualification must include cold rows"
                );
                overlay
            }
        };
        let config = HnswConfig::new(2, DistanceMetric::Euclidean).with_m(4);
        let owner = IndexOwnerImage {
            id: IndexId::new(0),
            name: "exact_vector".into(),
            graph: GraphPath::root(),
            label: "Doc".into(),
            property: "emb".into(),
            configuration: IndexConfiguration::Vector {
                config: config.clone(),
                quantization: QuantizationType::Scalar,
            },
        };
        let image = |id: Option<NodeId>, value: [f32; 2], deleted: bool| -> Result<Vec<u8>> {
            let index = VectorIndexKind::Quantized(QuantizedHnswIndex::with_seed(
                config.clone(),
                QuantizationType::Scalar,
                19,
            ));
            if let Some(id) = id {
                index.insert(id, &value, &|_| None);
                if deleted {
                    index.remove(id);
                }
            }
            index.encode_wal_birth()
        };
        let catalog = Catalog::new();
        for bytes in [
            image(Some(excluded), [0.2, 0.4], false)?,
            image(None, [0.2, 0.4], false)?,
        ] {
            let mut replay = OwnerReplay::new(&root, &default, &catalog)?;
            assert!(
                replay
                    .apply(
                        &root,
                        &default,
                        &catalog,
                        &payload(&owner, bytes, false)?,
                        TransactionId::new(2),
                        EpochId::new(1)
                    )
                    .is_err()
            );
            assert!(catalog.all_indexes().is_empty());
            assert_eq!(catalog.index_allocator_high_water(), 0);
            assert!(root.vector_index_entries().is_empty());
        }
        let mut replay = OwnerReplay::new(&root, &default, &catalog)?;
        replay.apply(
            &root,
            &default,
            &catalog,
            &payload(&owner, image(Some(node), [0.2, 0.4], false)?, false)?,
            TransactionId::new(2),
            EpochId::new(1),
        )?;
        let registration = root
            .observe_vector_index("Doc", "emb")
            .ok_or_else(|| grafeo_common::Error::Internal("missing Vector registration".into()))?;
        let definitions = catalog.all_indexes();
        let view = root
            .get_vector_index("Doc", "emb")
            .ok_or_else(|| grafeo_common::Error::Internal("missing Vector view".into()))?;
        let topology = view.snapshot_topology();
        root.sync_epoch(EpochId::new(2));
        for bytes in [
            image(Some(node), [0.2, 0.5], false)?,
            image(Some(node), [0.2, 0.4], true)?,
        ] {
            assert!(
                replay
                    .apply(
                        &root,
                        &default,
                        &catalog,
                        &payload(&owner, bytes, true)?,
                        TransactionId::new(3),
                        EpochId::new(2)
                    )
                    .is_err()
            );
            assert_eq!(catalog.all_indexes(), definitions);
            assert_eq!(catalog.index_allocator_high_water(), 1);
            root.validate_index_registration(&registration)?;
            assert_eq!(view.snapshot_topology(), topology);
        }
    }
    Ok(())
}
