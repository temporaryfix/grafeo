//! Cross-component ownership witness; not an engine/WAL acceptance control.

use super::*;
use crate::allocation_test as allocation;
use crate::graph::lpg::store::data_publication::StoreDataWorkspace;
use crate::graph::lpg::store::data_publication::slots::{
    StoreDataSlot, StoreDataSlots, prepare_store_data_slots,
};
use crate::index::vector::maintenance::{
    VectorMaintenanceSlot, VectorMaintenanceSlots, prepare_vector_maintenance_slots,
};
use crate::index::vector::{
    DistanceMetric, HnswConfig, HnswIndex, QuantizationType, QuantizedHnswIndex,
};
use grafeo_common::types::{EpochId, TransactionId};

fn exercise_shared_publication(reject_registry: bool) {
    let parent = LpgStore::new().unwrap();
    let child = parent.graph_or_create("child").unwrap();
    let stores = [&parent, child.as_ref()];
    let old_vector: Arc<[f32]> = Arc::from([1.0, 0.0]);
    let new_vector: Arc<[f32]> = Arc::from([0.0, 1.0]);
    let retired_vector: Arc<[f32]> = Arc::from([5.0, 5.0]);
    let tx = TransactionId::new(1190);
    let nodes = stores.map(|store| {
        let node = store.create_node(&["Doc"]);
        store.set_node_property(node, "counter", Value::Int64(1));
        store.set_node_property(node, "embedding", Value::from(vec![1.0_f64, 0.0]));
        store.set_node_property(node, "body", Value::from("oldtoken"));
        store.create_property_index("counter");
        node
    });
    let indexes = [
        Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
            2,
            DistanceMetric::Euclidean,
        )))),
        Arc::new(VectorIndexKind::Quantized(
            QuantizedHnswIndex::new(
                HnswConfig::new(2, DistanceMetric::Euclidean),
                QuantizationType::Scalar,
            )
            .with_training_threshold(1),
        )),
    ];
    let deleted_nodes = stores.map(|store| {
        let node = store.create_node(&["Doc"]);
        store.set_node_property(node, "embedding", Value::from(vec![5.0_f64, 5.0]));
        node
    });
    for (ordinal, (store, index)) in stores.iter().zip(&indexes).enumerate() {
        index.insert(nodes[ordinal], &old_vector, &|_| {
            Some(Arc::clone(&old_vector))
        });
        index.insert(deleted_nodes[ordinal], &retired_vector, &|id| {
            (id == nodes[ordinal]).then(|| Arc::clone(&old_vector))
        });
        store.add_vector_index("Doc", "embedding", Arc::clone(index));
    }
    let observed_vectors =
        stores.map(|store| store.observe_vector_index("Doc", "embedding").unwrap());
    let mut registry = IndexRegistryWorkspace::new(
        stores
            .iter()
            .enumerate()
            .map(|(ordinal, store)| {
                let edits = vec![IndexRegistryEdit::Maintain {
                    expected: store.observe_property_index("counter").unwrap(),
                    changes: IndexRegistryMaintenance::Property(vec![(
                        nodes[ordinal],
                        Some(Value::Int64(1)),
                        Some(Value::Int64(2)),
                    )]),
                }];
                #[cfg(feature = "text-index")]
                let edits = {
                    let mut edits = edits;
                    use crate::index::text::BM25Config;
                    let mut text = InvertedIndex::new(BM25Config::default());
                    text.insert(nodes[ordinal], "oldtoken");
                    store.add_text_index("Doc", "body", Arc::new(parking_lot::RwLock::new(text)));
                    edits.push(IndexRegistryEdit::Maintain {
                        expected: store.observe_text_index("Doc", "body").unwrap(),
                        changes: IndexRegistryMaintenance::Text {
                            rows: vec![(nodes[ordinal], Some("finaltoken".to_owned()))],
                            frontier: EpochId::INITIAL,
                            commit_epoch: EpochId::new(1),
                            transaction_id: tx,
                        },
                    });
                    edits
                };
                StoreIndexEdits { store, edits }
            })
            .collect(),
    );
    for (ordinal, (store, node)) in stores.iter().zip(nodes).enumerate() {
        store.set_node_property_buffered(node, "counter", Value::Int64(2), tx);
        store.set_node_property_buffered(node, "embedding", Value::from(vec![0.0_f64, 1.0]), tx);
        store.set_node_property_buffered(node, "body", Value::from("finaltoken"), tx);
        assert!(store.delete_node_transactional(deleted_nodes[ordinal], EpochId::INITIAL, tx));
        assert!(indexes[ordinal].contains(deleted_nodes[ordinal]));
    }
    let mut data_slots = StoreDataSlots::new(
        stores
            .iter()
            .rev()
            .map(|store| {
                StoreDataSlot::new(
                    store,
                    StoreDataWorkspace::new(tx, EpochId::INITIAL, EpochId::new(1)),
                )
            })
            .collect(),
    );
    let mut vector_slots = VectorMaintenanceSlots::new(
        indexes
            .iter()
            .enumerate()
            .map(|(ordinal, index)| {
                VectorMaintenanceSlot::new(
                    index,
                    vec![
                        (nodes[ordinal], Some(Arc::clone(&new_vector))),
                        (deleted_nodes[ordinal], None),
                    ],
                )
            })
            .collect(),
    );
    registry.prepare_inputs().unwrap();
    let authority =
        RegistryAuthority::acquire(&mut registry.registry.stores, &mut registry.authority).unwrap();
    // Fixed stack pins for this witness; no heap container borrows authority.
    let pins = [
        indexes[0].pin_maintenance().unwrap(),
        indexes[1].pin_maintenance().unwrap(),
    ];
    let data =
        prepare_store_data_slots(&mut data_slots, authority.workspace.retained_transitions())
            .unwrap();
    let physical = prepare_under_authority(&mut registry.registry, authority.workspace).unwrap();
    let vectors = prepare_vector_maintenance_slots(&mut vector_slots, &pins, &|ordinal, id| {
        if id == nodes[ordinal] {
            Some(Arc::clone(&old_vector))
        } else if id == deleted_nodes[ordinal] {
            Some(Arc::clone(&retired_vector))
        } else {
            None
        }
    })
    .unwrap()
    .exclude_readers()
    .unwrap();
    let held_registry = reject_registry.then(|| child.property_indexes.read());
    allocation::start();
    let data = data.rebind().unwrap();
    let vectors = vectors.rebind().unwrap();
    let published = match physical.rebind() {
        Ok(mut physical) => {
            // EVERY store/family is ready before the first install.
            let data = data.install();
            let vectors = vectors.install();
            physical.fences.install();
            drop(physical);
            drop(vectors);
            drop(data);
            true
        }
        Err(error) => {
            assert!(matches!(error, DataRebindError::Conflict(_)));
            drop(vectors);
            drop(data);
            false
        }
    };
    drop(held_registry);
    drop(pins);
    drop(authority);
    let traffic = allocation::stop();
    assert_eq!(traffic, allocation::Counts::default());
    assert_eq!(published, !reject_registry);
    // Both physical rows and every surviving registration agree after release.
    for (ordinal, store) in stores.iter().enumerate() {
        store
            .validate_index_registration(&observed_vectors[ordinal])
            .unwrap();
        let expected = if published { 2 } else { 1 };
        assert_eq!(
            store.get_node_property(nodes[ordinal], &PropertyKey::new("counter")),
            Some(Value::Int64(expected))
        );
        assert_eq!(
            store.find_nodes_by_property("counter", &Value::Int64(expected)),
            vec![nodes[ordinal]]
        );
        let vector = if published { &new_vector } else { &old_vector };
        let result = indexes[ordinal].search(&new_vector, 1, &|id| {
            if id == nodes[ordinal] {
                Some(Arc::clone(vector))
            } else if id == deleted_nodes[ordinal] {
                Some(Arc::clone(&retired_vector))
            } else {
                None
            }
        });
        assert_eq!(result[0].0, nodes[ordinal]);
        assert_eq!(result[0].1 == 0.0, published);
        // A topology-changing sentinel, independent of the search accessor,
        // catches accidental early HNSW or Quantized installation on rejection.
        assert_eq!(
            indexes[ordinal].contains(deleted_nodes[ordinal]),
            !published
        );
        #[cfg(feature = "text-index")]
        {
            let text = store.get_text_index("Doc", "body").unwrap();
            let token = if published { "finaltoken" } else { "oldtoken" };
            assert_eq!(text.read().search(token, 1)[0].0, nodes[ordinal]);
        }
        assert_eq!(
            store.tx_property_overlay.read().contains_key(&tx),
            !published
        );
    }
}

#[test]
fn data_registry_and_both_vector_kinds_install_and_release_without_allocator_traffic() {
    exercise_shared_publication(false);
}

#[test]
fn last_registry_rejection_preserves_all_earlier_data_and_vector_candidates() {
    exercise_shared_publication(true);
}
