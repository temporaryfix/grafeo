use super::*;
use crate::allocation_test as allocation;
use crate::graph::lpg::{IndexRegistryContents, IndexRegistryKey, IndexRegistryMaintenance};
use crate::graph::write_permit::with_authority;
#[cfg(feature = "vector-index")]
use crate::index::vector::VectorIndexKind;
use grafeo_common::types::{NodeId, PropertyKey, Value};
#[cfg(feature = "compact-store")]
use grafeo_common::utils::hash::FxHashSet;
#[cfg(any(feature = "compact-store", feature = "vector-index"))]
use std::sync::Arc;

const TX: TransactionId = TransactionId::new(1310);
const P: EpochId = EpochId::INITIAL;
const C: EpochId = EpochId::new(1);

#[test]
fn prepared_label_images_follow_exact_normalized_publication() {
    let store = LpgStore::new().unwrap();
    let changed = store.create_node(&["A"]);
    let noop = store.create_node(&["A"]);
    let deleted = store.create_node(&["A"]);
    let emptied = store.create_node(&["A"]);
    let born = store.create_node_versioned(&["Initial"], P, TX);
    let zero_width = store.create_node_versioned(&["Initial"], P, TX);
    store.add_label_buffered(changed, "B", TX);
    store.remove_label_buffered(changed, "A", TX);
    store.add_label_buffered(noop, "B", TX);
    store.remove_label_buffered(noop, "B", TX);
    store.add_label_buffered(deleted, "B", TX);
    assert!(store.delete_node_versioned(deleted, P, TX));
    store.remove_label_buffered(emptied, "A", TX);
    store.add_label_buffered(born, "B", TX);
    store.remove_label_buffered(born, "Initial", TX);
    store.add_label_buffered(zero_width, "B", TX);
    assert!(store.delete_node_versioned(zero_width, P, TX));
    let mut workspace = LpgCommitWorkspace::new(
        vec![StoreCommitInput {
            graph: None,
            store: &store,
            source: &store,
            publish_data: true,
            edits: vec![],
        }],
        #[cfg(feature = "vector-index")]
        vec![],
        TX,
        P,
        C,
    );
    workspace.capture_label_images().unwrap();
    let authority = WriteAuthority::new();
    with_authority(&authority, || {
        with_prepared_lpg_commit(&mut workspace, &authority, |released| {
            assert_eq!(released.label_images().count(), 1);
            let actual: Vec<_> = released
                .label_images()
                .flat_map(|(target, images)| {
                    assert!(std::ptr::eq(target, &raw const store));
                    images
                        .iter()
                        .map(|labels| (labels.id(), labels.birth(), labels.images().to_vec()))
                })
                .collect();
            let labels = |names: &[&str]| names.iter().map(|name| (*name).to_owned()).collect();
            assert_eq!(
                actual,
                vec![
                    (changed, false, vec![labels(&["B"])]),
                    (deleted, false, vec![labels(&["A", "B"])]),
                    (emptied, false, vec![labels(&[])]),
                    (born, true, vec![labels(&["B"])]),
                    (
                        zero_width,
                        true,
                        vec![labels(&["Initial"]), labels(&["B", "Initial"])],
                    ),
                ],
            );
            allocation::start();
            drop(released.rebind().unwrap().install());
            assert_eq!(allocation::stop(), allocation::Counts::default());
            Ok(())
        })
        .unwrap();
    });
    assert!(workspace.capture_label_images().is_err());
    assert_eq!(store.node_labels.read()[&noop].history().len(), 1);
    assert_eq!(store.node_labels.read()[&born].history().len(), 1);
    assert_eq!(store.node_labels.read()[&zero_width].history().len(), 2);
}

#[test]
fn prepared_label_images_are_opt_in_and_keep_abortable_live_state() {
    for capture in [false, true] {
        let store = LpgStore::new().unwrap();
        let node = store.create_node(&["Before"]);
        store.add_label_buffered(node, "After", TX);
        let mut workspace = LpgCommitWorkspace::new(
            vec![StoreCommitInput {
                graph: None,
                store: &store,
                source: &store,
                publish_data: true,
                edits: vec![],
            }],
            #[cfg(feature = "vector-index")]
            vec![],
            TX,
            P,
            C,
        );
        if capture {
            workspace.capture_label_images().unwrap();
        }
        let authority = WriteAuthority::new();
        with_authority(&authority, || {
            let result: Result<()> =
                with_prepared_lpg_commit(&mut workspace, &authority, |released| {
                    assert_eq!(released.label_images().count(), usize::from(capture));
                    Err(grafeo_common::utils::error::Error::Internal(
                        "abort before installation".to_owned(),
                    ))
                });
            assert!(result.is_err());
        });
        assert_eq!(store.node_labels.read()[&node].history().len(), 1);
        assert!(store.nodes_by_label("After").is_empty());
        store.drop_tx_overlay(TX);
        assert_eq!(store.nodes_by_label("Before"), vec![node]);
    }
}

#[test]
fn prepared_label_images_are_qualified_by_surviving_store_identity() {
    let first = LpgStore::new().unwrap();
    let second = LpgStore::new().unwrap();
    let dropped = LpgStore::new().unwrap();
    let node = first.create_node(&["Before"]);
    assert_eq!(second.create_node(&["Before"]), node);
    assert_eq!(dropped.create_node(&["Before"]), node);
    first.add_label_buffered(node, "First", TX);
    second.add_label_buffered(node, "Second", TX);
    dropped.add_label_buffered(node, "Dropped", TX);
    let mut workspace = LpgCommitWorkspace::new(
        vec![
            StoreCommitInput {
                graph: None,
                store: &first,
                source: &first,
                publish_data: true,
                edits: vec![],
            },
            StoreCommitInput {
                graph: None,
                store: &second,
                source: &second,
                publish_data: true,
                edits: vec![],
            },
            StoreCommitInput {
                graph: None,
                store: &dropped,
                source: &dropped,
                publish_data: false,
                edits: vec![],
            },
        ],
        #[cfg(feature = "vector-index")]
        vec![],
        TX,
        P,
        C,
    );
    workspace.capture_label_images().unwrap();
    let authority = WriteAuthority::new();
    with_authority(&authority, || {
        with_prepared_lpg_commit(&mut workspace, &authority, |released| {
            assert_eq!(released.label_images().count(), 2);
            for (target, images) in released.label_images() {
                assert_eq!(images.len(), 1);
                let labels = &images[0];
                assert_eq!(labels.id(), node);
                assert!(!labels.birth());
                let expected = if std::ptr::eq(target, &raw const first) {
                    "First"
                } else {
                    assert!(std::ptr::eq(target, &raw const second));
                    "Second"
                };
                assert_eq!(
                    labels.images(),
                    &[vec!["Before".to_owned(), expected.to_owned()]]
                );
            }
            drop(released.rebind().unwrap().install());
            Ok(())
        })
        .unwrap();
    });
    assert!(dropped.nodes_by_label("Dropped").is_empty());
    dropped.drop_tx_overlay(TX);
}

#[cfg(feature = "compact-store")]
#[test]
fn prepared_label_images_keep_layered_promotion_distinct_from_birth() {
    use crate::graph::compact::{from_graph_store_preserving_ids, layered::LayeredStore};

    let source = LpgStore::new().unwrap();
    let node = source.create_node(&["Cold"]);
    let base = from_graph_store_preserving_ids(&source).unwrap();
    let layered = LayeredStore::new(base, node.as_u64(), 0).unwrap();
    let overlay = layered.overlay_store();
    layered.add_label_buffered(node, "Changed", TX);
    layered.remove_label_buffered(node, "Cold", TX);
    let mut workspace = LpgCommitWorkspace::new(
        vec![StoreCommitInput {
            graph: None,
            store: &overlay,
            source: &layered,
            publish_data: true,
            edits: vec![],
        }],
        #[cfg(feature = "vector-index")]
        vec![],
        TX,
        P,
        C,
    );
    workspace.capture_label_images().unwrap();
    let authority = WriteAuthority::new();
    with_authority(&authority, || {
        with_prepared_lpg_commit(&mut workspace, &authority, |released| {
            let images: Vec<_> = released.label_images().collect();
            assert_eq!(images.len(), 1);
            let (target, facts) = images[0];
            assert_eq!(facts.len(), 1);
            let labels = &facts[0];
            assert!(std::ptr::eq(target, overlay.as_ref()));
            assert_eq!(labels.id(), node);
            assert!(!labels.birth());
            assert_eq!(labels.images(), &[vec!["Changed".to_owned()]]);
            drop(released.rebind().unwrap().install());
            Ok(())
        })
        .unwrap();
    });
}

#[cfg(feature = "compact-store")]
#[test]
fn layered_cold_property_removal_qualifies_exact_physical_membership() {
    use crate::graph::compact::{from_graph_store_preserving_ids, layered::LayeredStore};
    use crate::graph::traits::GraphStore;
    use grafeo_common::types::HashableValue;

    // Empty transferred definition, occupied bucket lacking this identity,
    // and a fresh index populated over the complete cold generation.
    for population in 0..3 {
        let source = LpgStore::new().unwrap();
        let removed = source.create_node(&["Cold"]);
        let foreign = source.create_node(&["Cold"]);
        source.set_node_property(removed, "counter", Value::Int64(1));
        source.set_node_property(foreign, "counter", Value::Int64(1));
        let base = from_graph_store_preserving_ids(&source).unwrap();
        let layered = LayeredStore::new(base, foreign.as_u64(), 0).unwrap();
        let overlay = layered.overlay_store();
        overlay.create_property_index("counter");
        let rows =
            Arc::clone(&overlay.property_indexes.read()[&PropertyKey::new("counter")].payload);
        // Registration now populates the full cold view. Deliberately replace
        // current memberships to retain the legacy partial-image cases here;
        // the authenticated historical payload stays complete.
        rows.clear();
        if population > 0 {
            let mut members = FxHashSet::default();
            members.insert(foreign);
            if population == 2 {
                members.insert(removed);
            }
            rows.insert(HashableValue::new(Value::Int64(1)), members);
        }
        let expected = overlay.observe_property_index("counter").unwrap();
        let foreign_tx = TransactionId::new(TX.as_u64() + 1);
        assert!(layered.delete_node_versioned(removed, P, TX));
        assert!(layered.delete_node_versioned(foreign, P, foreign_tx));
        assert!(!overlay.contains_node_identity(removed));
        let mut workspace = LpgCommitWorkspace::new(
            vec![StoreCommitInput {
                graph: None,
                store: &overlay,
                source: &layered,
                publish_data: true,
                edits: vec![IndexRegistryEdit::Maintain {
                    expected: overlay.observe_property_index("counter").unwrap(),
                    changes: IndexRegistryMaintenance::Property(vec![(
                        removed,
                        Some(Value::Int64(1)),
                        None,
                    )]),
                }],
            }],
            #[cfg(feature = "vector-index")]
            vec![],
            TX,
            P,
            C,
        );
        let authority = WriteAuthority::new();
        with_authority(&authority, || {
            with_prepared_lpg_commit(&mut workspace, &authority, |released| {
                allocation::start();
                drop(released.rebind().unwrap().install());
                Ok(())
            })
            .unwrap();
            assert_eq!(allocation::stop(), allocation::Counts::default());
            overlay.validate_index_registration(&expected).unwrap();
        });
        assert!(Arc::ptr_eq(
            &rows,
            &overlay.property_indexes.read()[&PropertyKey::new("counter")].payload,
        ));
        let key = HashableValue::new(Value::Int64(1));
        assert!(
            !rows
                .get(&key)
                .is_some_and(|members| members.contains(&removed))
        );
        assert_eq!(
            rows.get(&key)
                .is_some_and(|members| members.contains(&foreign)),
            population > 0,
        );
        assert!(!overlay.contains_node_identity(removed));
        assert!(layered.is_node_visible_versioned(removed, P, foreign_tx));
        assert!(!layered.is_node_visible_versioned(removed, C, foreign_tx));
        assert!(layered.is_node_visible_versioned(foreign, C, TX));
        assert_eq!(layered.pending_node_deletes_peek(foreign_tx), vec![foreign]);
    }
}

#[test]
fn native_missing_property_predecessor_still_rejects() {
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Doc"]);
    store.set_node_property(node, "counter", Value::Int64(1));
    store.create_property_index("counter");
    store.property_indexes.read()[&PropertyKey::new("counter")]
        .payload
        .clear();
    let expected = store.observe_property_index("counter").unwrap();
    store.set_node_property_buffered(node, "counter", Value::Int64(2), TX);
    let mut workspace = LpgCommitWorkspace::new(
        vec![StoreCommitInput {
            graph: None,
            store: &store,
            source: &store,
            publish_data: true,
            edits: vec![IndexRegistryEdit::Maintain {
                expected,
                changes: IndexRegistryMaintenance::Property(vec![(
                    node,
                    Some(Value::Int64(1)),
                    Some(Value::Int64(2)),
                )]),
            }],
        }],
        #[cfg(feature = "vector-index")]
        vec![],
        TX,
        P,
        C,
    );
    let authority = WriteAuthority::new();
    let result: Result<()> = with_authority(&authority, || {
        with_prepared_lpg_commit(&mut workspace, &authority, |_| {
            panic!("missing native predecessor reached publication")
        })
    });
    assert!(result.unwrap_err().to_string().contains("old membership"));
    assert_eq!(store.current_epoch(), P);
    assert!(store.tx_property_overlay.read().contains_key(&TX));
    assert!(store.mutation_scope_gate.try_write().is_some());
}

#[cfg(feature = "compact-store")]
#[test]
fn hydrated_layered_missing_property_predecessor_still_rejects() {
    use crate::graph::compact::{from_graph_store_preserving_ids, layered::LayeredStore};
    let source = LpgStore::new().unwrap();
    let node = source.create_node(&["Cold"]);
    source.set_node_property(node, "counter", Value::Int64(1));
    let base = from_graph_store_preserving_ids(&source).unwrap();
    let layered = LayeredStore::new(base, node.as_u64(), 0).unwrap();
    let overlay = layered.overlay_store();
    overlay.create_property_index("counter");
    layered.set_node_property_buffered(node, "counter", Value::Int64(2), TX);
    assert!(overlay.contains_node_identity(node));
    overlay.property_indexes.read()[&PropertyKey::new("counter")]
        .payload
        .clear();
    let expected = overlay.observe_property_index("counter").unwrap();
    let mut workspace = LpgCommitWorkspace::new(
        vec![StoreCommitInput {
            graph: None,
            store: &overlay,
            source: &layered,
            publish_data: true,
            edits: vec![IndexRegistryEdit::Maintain {
                expected,
                changes: IndexRegistryMaintenance::Property(vec![(
                    node,
                    Some(Value::Int64(1)),
                    Some(Value::Int64(2)),
                )]),
            }],
        }],
        #[cfg(feature = "vector-index")]
        vec![],
        TX,
        P,
        C,
    );
    let authority = WriteAuthority::new();
    let result: Result<()> = with_authority(&authority, || {
        with_prepared_lpg_commit(&mut workspace, &authority, |_| {
            panic!("missing hydrated predecessor reached publication")
        })
    });
    assert!(result.unwrap_err().to_string().contains("old membership"));
    assert_eq!(overlay.current_epoch(), P);
    assert!(overlay.tx_property_overlay.read().contains_key(&TX));
    assert!(overlay.mutation_scope_gate.try_write().is_some());
}

#[test]
fn scoped_driver_installs_native_data_and_property_with_no_final_allocator_traffic() {
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Doc"]);
    store.set_node_property(node, "counter", Value::Int64(1));
    store.create_property_index("counter");
    let expected = store.observe_property_index("counter").unwrap();
    store.set_node_property_buffered(node, "counter", Value::Int64(2), TX);
    let authority = WriteAuthority::new();
    let mut workspace = LpgCommitWorkspace::new(
        vec![StoreCommitInput {
            graph: None,
            store: &store,
            source: &store,
            publish_data: true,
            edits: vec![IndexRegistryEdit::Maintain {
                expected,
                changes: IndexRegistryMaintenance::Property(vec![(
                    node,
                    Some(Value::Int64(1)),
                    Some(Value::Int64(2)),
                )]),
            }],
        }],
        #[cfg(feature = "vector-index")]
        vec![],
        TX,
        P,
        C,
    );
    with_authority(&authority, || {
        with_prepared_lpg_commit(&mut workspace, &authority, |released| {
            allocation::start();
            let installed = released.rebind().unwrap().install();
            assert!(store.tx_property_overlay.try_read().is_none());
            assert!(store.property_indexes.try_read().is_none());
            drop(installed);
            Ok(())
        })
        .unwrap();
        // Includes the scoped driver's cross-family/pin/authority cleanup.
        assert_eq!(allocation::stop(), allocation::Counts::default());
    });
    assert_eq!(
        store.get_node_property(node, &PropertyKey::new("counter")),
        Some(Value::Int64(2))
    );
    assert_eq!(
        store.find_nodes_by_property("counter", &Value::Int64(2)),
        vec![node]
    );
    assert!(!store.tx_property_overlay.read().contains_key(&TX));
}

#[test]
fn scoped_driver_rejects_unheld_authority_and_mismatched_native_source() {
    let store = LpgStore::new().unwrap();
    let other = LpgStore::new().unwrap();
    let authority = WriteAuthority::new();
    for held in [false, true] {
        let mut workspace = LpgCommitWorkspace::new(
            vec![StoreCommitInput {
                graph: None,
                store: &store,
                source: if held { &other } else { &store },
                publish_data: true,
                edits: vec![],
            }],
            #[cfg(feature = "vector-index")]
            vec![],
            TX,
            P,
            C,
        );
        let mut run = || {
            with_prepared_lpg_commit(&mut workspace, &authority, |_| {
                panic!("invalid source reached callback")
            })
        };
        let result: Result<()> = if held {
            with_authority(&authority, run)
        } else {
            run()
        };
        assert!(result.is_err());
        assert!(store.mutation_scope_gate.try_write().is_some());
        assert!(store.property_indexes.try_write().is_some());
    }
}

#[test]
fn scoped_driver_registry_only_target_does_not_publish_its_buffered_data() {
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Doc"]);
    store.set_node_property(node, "counter", Value::Int64(1));
    store.set_node_property_buffered(node, "counter", Value::Int64(2), TX);
    let authority = WriteAuthority::new();
    let mut workspace = LpgCommitWorkspace::new(
        vec![StoreCommitInput {
            graph: None,
            store: &store,
            source: &store,
            publish_data: false,
            edits: vec![IndexRegistryEdit::Create {
                key: IndexRegistryKey::Property(PropertyKey::new("counter")),
                contents: IndexRegistryContents::Property(vec![(node, Value::Int64(1))]),
            }],
        }],
        #[cfg(feature = "vector-index")]
        vec![],
        TX,
        P,
        C,
    );
    with_authority(&authority, || {
        with_prepared_lpg_commit(&mut workspace, &authority, |released| {
            drop(released.rebind().unwrap().install());
            Ok(())
        })
    })
    .unwrap();
    assert_eq!(
        store.get_node_property(node, &PropertyKey::new("counter")),
        Some(Value::Int64(1))
    );
    assert_eq!(
        store.find_nodes_by_property("counter", &Value::Int64(1)),
        vec![node]
    );
    assert!(store.tx_property_overlay.read().contains_key(&TX));
}

#[cfg(feature = "vector-index")]
#[test]
fn scoped_driver_qualifies_every_vector_kind_and_cleans_forgotten_phases() {
    use crate::index::vector::{
        DistanceMetric, HnswConfig, HnswIndex, QuantizationType, QuantizedHnswIndex,
    };
    // Plain plus every quantized family; each phase gets an independent world.
    for quantization in [
        None,
        Some(QuantizationType::None),
        Some(QuantizationType::Scalar),
        Some(QuantizationType::Binary),
        Some(QuantizationType::Product { num_subvectors: 1 }),
    ] {
        for phase in 0..3 {
            let store = LpgStore::new().unwrap();
            let node = store.create_node(&["Doc"]);
            let deleted = store.create_node(&["Doc"]);
            let routing_only = store.create_node(&["Doc"]);
            store.set_node_property(routing_only, "embedding", Value::from(vec![1.0_f64, 0.0]));
            let old: Arc<[f32]> = Arc::from([1.0, 0.0]);
            let final_vector: Arc<[f32]> = Arc::from([0.0, 1.0]);
            let config = HnswConfig::new(2, DistanceMetric::Euclidean);
            let index = Arc::new(match quantization {
                None => VectorIndexKind::Hnsw(HnswIndex::new(config)),
                Some(kind) => VectorIndexKind::Quantized(QuantizedHnswIndex::new(config, kind)),
            });
            index.insert(node, &old, &|_| Some(Arc::clone(&old)));
            index.insert(deleted, &old, &|_| Some(Arc::clone(&old)));
            index.insert(routing_only, &old, &|_| Some(Arc::clone(&old)));
            store.add_vector_index("Doc", "embedding", Arc::clone(&index));
            let view = store.get_vector_index("Doc", "embedding").unwrap();
            let observation = store.observe_vector_index("Doc", "embedding").unwrap();
            let authority = WriteAuthority::new();
            let mut workspace = LpgCommitWorkspace::new(
                vec![StoreCommitInput {
                    graph: None,
                    store: &store,
                    source: &store,
                    publish_data: true,
                    edits: vec![],
                }],
                vec![VectorCommitInput {
                    store: &store,
                    view: &view,
                    expected: &observation,
                    property: PropertyKey::new("embedding"),
                    changes: VectorCommitChanges::Rows(vec![
                        (node, Some(Arc::clone(&final_vector))),
                        (deleted, None),
                    ]),
                    routing: [(node, Arc::clone(&old)), (deleted, Arc::clone(&old))]
                        .into_iter()
                        .collect(),
                }],
                TX,
                P,
                C,
            );
            with_authority(&authority, || {
                with_prepared_lpg_commit(&mut workspace, &authority, |released| {
                    allocation::start();
                    match phase {
                        0 => std::mem::forget(released),
                        1 => std::mem::forget(released.rebind().unwrap()),
                        _ => std::mem::forget(released.rebind().unwrap().install()),
                    }
                    Ok(())
                })
                .unwrap();
                assert_eq!(allocation::stop(), allocation::Counts::default());
                // Scope cleanup runs on callback return, not only workspace Drop.
                assert!(store.mutation_scope_gate.try_write().is_some());
                assert!(store.vector_indexes.try_write().is_some());
                drop(index.pin_maintenance().unwrap());
                store.validate_index_registration(&observation).unwrap();
            });
            assert_eq!(view.contains(deleted), phase != 2);
            assert!(view.contains(routing_only));
            drop(workspace);
        }
    }
}

#[test]
fn scoped_driver_unwind_after_forgetting_ready_drains_every_store() {
    let parent = LpgStore::new().unwrap();
    let child = parent.graph_or_create("child").unwrap();
    let authority = WriteAuthority::new();
    let mut workspace = LpgCommitWorkspace::new(
        [&parent, child.as_ref()]
            .into_iter()
            .map(|store| StoreCommitInput {
                graph: None,
                store,
                source: store,
                publish_data: true,
                edits: vec![],
            })
            .collect(),
        #[cfg(feature = "vector-index")]
        vec![],
        TX,
        P,
        C,
    );
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_authority(&authority, || {
            let _: Result<()> = with_prepared_lpg_commit(&mut workspace, &authority, |released| {
                std::mem::forget(released.rebind().unwrap());
                panic!("callback unwind after forgetting proof");
            });
        });
    }));
    assert!(result.is_err());
    for store in [&parent, child.as_ref()] {
        assert!(store.mutation_scope_gate.try_write().is_some());
        assert!(store.property_indexes.try_write().is_some());
    }
    // Parallel captures may own the process-wide gate after our unwind.
    // A bounded wait still detects a leaked guard without assuming exclusivity.
    assert!(
        crate::graph::lpg::store::NAMED_GRAPH_TOPOLOGY_GATE
            .try_lock_for(std::time::Duration::from_secs(1))
            .is_some()
    );
}

#[cfg(feature = "vector-index")]
#[test]
fn scoped_driver_rejects_vector_survivor_that_its_registry_drops_or_replaces() {
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex};
    for replace in [false, true] {
        let store = LpgStore::new().unwrap();
        let node = store.create_node(&["Doc"]);
        let vector: Arc<[f32]> = Arc::from([1.0, 0.0]);
        let make_index = || {
            VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
                2,
                DistanceMetric::Euclidean,
            )))
        };
        let index = Arc::new(make_index());
        index.insert(node, &vector, &|_| Some(Arc::clone(&vector)));
        store.add_vector_index("Doc", "embedding", Arc::clone(&index));
        let view = store.get_vector_index("Doc", "embedding").unwrap();
        let expected = store.observe_vector_index("Doc", "embedding").unwrap();
        let removal = store.observe_vector_index("Doc", "embedding").unwrap();
        let edit = if replace {
            IndexRegistryEdit::Replace {
                expected: removal,
                contents: IndexRegistryContents::Vector(make_index()),
            }
        } else {
            IndexRegistryEdit::Drop { expected: removal }
        };
        let mut workspace = LpgCommitWorkspace::new(
            vec![StoreCommitInput {
                graph: None,
                store: &store,
                source: &store,
                publish_data: true,
                edits: vec![edit],
            }],
            vec![VectorCommitInput {
                store: &store,
                view: &view,
                expected: &expected,
                property: PropertyKey::new("embedding"),
                changes: VectorCommitChanges::Rows(vec![(node, None)]),
                routing: [(node, Arc::clone(&vector))].into_iter().collect(),
            }],
            TX,
            P,
            C,
        );
        let authority = WriteAuthority::new();
        with_authority(&authority, || {
            with_prepared_lpg_commit(&mut workspace, &authority, |released| {
                allocation::start();
                assert!(matches!(
                    released.rebind(),
                    Err(DataRebindError::Conflict(_))
                ));
                Ok(())
            })
            .unwrap();
            assert_eq!(allocation::stop(), allocation::Counts::default());
            store.validate_index_registration(&expected).unwrap();
        });
        assert!(view.contains(node));
    }
}

#[test]
fn scoped_driver_sparse_preparation_matches_full_rows_without_mutating_pending_state() {
    #[cfg(feature = "compact-store")]
    use crate::graph::compact::{from_graph_store_preserving_ids, layered::LayeredStore};

    let store = LpgStore::new().unwrap();
    let ids: Vec<_> = (0..128)
        .map(|i| {
            let node = store.create_node(&["Doc"]);
            store.set_node_property(node, "counter", Value::Int64(i));
            node
        })
        .collect();
    #[cfg(feature = "compact-store")]
    let layered = LayeredStore::new(
        from_graph_store_preserving_ids(&store).unwrap(),
        ids[127].as_u64(),
        0,
    )
    .unwrap();
    let sources: Vec<&dyn GraphStoreMut> = vec![
        &store,
        #[cfg(feature = "compact-store")]
        &layered,
    ];
    for source in sources {
        source.set_node_property_buffered(ids[2], "counter", Value::Int64(400), TX);
        let born = source.create_node_versioned(&["Born"], P, TX);
        let foreign = TransactionId::new(1311);
        assert!(source.delete_node_versioned(ids[3], P, foreign));
        let selected = [born, ids[3], ids[2], ids[1], NodeId::new(999999)];
        let pending_before = source.overlay_touched_properties(TX);
        for transaction in [None, Some(TX)] {
            let sparse = source
                .prepare_index_node_rows_by_id(P, transaction, &selected)
                .unwrap();
            let mut full = source.prepare_index_node_rows(P, transaction).unwrap();
            full.retain(|node| selected.contains(&node.id));
            full.sort_unstable_by_key(|node| node.id);
            assert_eq!(sparse.len(), full.len());
            for (actual, expected) in sparse.iter().zip(&full) {
                assert_eq!(actual.id, expected.id);
                assert_eq!(actual.labels, expected.labels);
                assert_eq!(actual.properties, expected.properties);
            }
            assert!(sparse.windows(2).all(|pair| pair[0].id < pair[1].id));
            assert!(sparse.iter().any(|node| node.id == ids[3]));
            assert_eq!(
                sparse.iter().any(|node| node.id == born),
                transaction.is_some()
            );
        }
        assert_eq!(source.overlay_touched_properties(TX), pending_before);
    }
}

#[test]
fn scoped_driver_empty_companion_does_not_acquire_global_topology() {
    let authority = WriteAuthority::new();
    let mut workspace = LpgCommitWorkspace::new(
        vec![],
        #[cfg(feature = "vector-index")]
        vec![],
        TX,
        P,
        C,
    );
    let _topology = crate::graph::lpg::store::NAMED_GRAPH_TOPOLOGY_GATE.lock();
    with_authority(&authority, || {
        with_prepared_lpg_commit(&mut workspace, &authority, |released| {
            drop(released.rebind().unwrap().install());
            Ok(())
        })
    })
    .unwrap();
}
