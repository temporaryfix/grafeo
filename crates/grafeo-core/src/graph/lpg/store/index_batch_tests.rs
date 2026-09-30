//! Behavioral witnesses for the real prepared registry publication boundary.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
#[cfg(feature = "text-index")]
use crate::graph::lpg::encode_index_key;
use crate::graph::write_permit::{WriteAuthority, with_authority};
use std::sync::Arc;

fn property(name: &str, id: u64, value: &str) -> IndexRegistryEdit {
    IndexRegistryEdit::Create {
        key: IndexRegistryKey::Property(PropertyKey::new(name)),
        contents: IndexRegistryContents::Property(vec![(NodeId::new(id), Value::from(value))]),
    }
}

#[test]
fn final_registry_rebind_contention_is_static_and_drains_partial_writers() {
    let store = LpgStore::new().unwrap();
    macro_rules! rejects {
        ($reader:expr) => {{
            let mut workspace = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
                store: &store,
                edits: vec![property("private", 1, "candidate")],
            }]);
            workspace.prepare_inputs().unwrap();
            let authority = RegistryAuthority::acquire(
                &mut workspace.registry.stores,
                &mut workspace.authority,
            )
            .unwrap();
            let released =
                prepare_under_authority(&mut workspace.registry, authority.workspace).unwrap();
            let held = $reader;
            crate::allocation_test::start();
            let result = released.rebind();
            let conflict = matches!(result, Err(DataRebindError::Conflict(_)));
            drop(result);
            let traffic = crate::allocation_test::stop();
            drop(held);
            assert!(conflict, stringify!($reader));
            assert_eq!(traffic, crate::allocation_test::Counts::default());
            assert!(store.property_indexes.try_write().is_some());
            #[cfg(feature = "text-index")]
            assert!(store.text_indexes.try_write().is_some());
            #[cfg(feature = "vector-index")]
            assert!(store.vector_indexes.try_write().is_some());
            #[cfg(any(feature = "text-index", feature = "vector-index"))]
            assert!(store.index_slots.try_lock().is_some());
            assert!(!store.has_property_index("private"));
        }};
    }
    rejects!(store.property_indexes.read());
    #[cfg(feature = "text-index")]
    rejects!(store.text_indexes.read());
    #[cfg(feature = "vector-index")]
    rejects!(store.vector_indexes.read());
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    rejects!(store.index_slots.lock());
}

#[test]
fn surviving_property_maintenance_shares_data_publication_and_preserves_identity() {
    use super::super::data_publication::StoreDataWorkspace;
    use grafeo_common::types::{EpochId, TransactionId};
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Row"]);
    store.set_node_property(node, "value", Value::from("old"));
    store.create_property_index("value");
    let before = store.observe_property_index("value").unwrap();
    let registered = store.property_indexes.read()[&PropertyKey::new("value")].clone();
    let tx = TransactionId::new(914);
    store.set_node_property_buffered(node, "value", Value::from("final"), tx);
    let mut data_workspace = StoreDataWorkspace::new(tx, EpochId::INITIAL, EpochId::new(1));
    let mut workspace = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
        store: &store,
        edits: vec![IndexRegistryEdit::Maintain {
            expected: store.observe_property_index("value").unwrap(),
            changes: IndexRegistryMaintenance::Property(vec![(
                node,
                Some(Value::from("old")),
                Some(Value::from("final")),
            )]),
        }],
    }]);
    workspace.prepare_inputs().unwrap();
    {
        let authority =
            RegistryAuthority::acquire(&mut workspace.registry.stores, &mut workspace.authority)
                .unwrap();
        let data = store
            .prepare_buffered_commit_data(
                authority.transition(&store).unwrap(),
                &mut data_workspace,
            )
            .unwrap();
        let indexes =
            prepare_under_authority(&mut workspace.registry, authority.workspace).unwrap();
        assert!(
            registered
                .payload
                .contains_key(&HashableValue::new(Value::from("old")))
        );
        assert!(
            !registered
                .payload
                .contains_key(&HashableValue::new(Value::from("final")))
        );
        crate::allocation_test::start();
        let data = data.rebind().unwrap();
        let mut indexes = indexes.rebind().unwrap();
        let data = data.install();
        indexes.fences.install();
        drop(indexes);
        drop(data);
        let traffic = crate::allocation_test::stop();
        assert_eq!(traffic, crate::allocation_test::Counts::default());
    }
    store.validate_index_registration(&before).unwrap();
    let current = store.property_indexes.read()[&PropertyKey::new("value")].clone();
    assert!(Arc::ptr_eq(&registered.payload, &current.payload));
    assert!(Arc::ptr_eq(&registered.registration, &current.registration));
    assert_eq!(
        store.find_nodes_by_property("value", &Value::from("final")),
        vec![node]
    );
    assert!(
        store
            .find_nodes_by_property("value", &Value::from("old"))
            .is_empty()
    );
    assert_eq!(
        store.get_node_property(node, &PropertyKey::new("value")),
        Some(Value::from("final"))
    );
    assert_eq!(
        store
            .node_properties
            .get_at(node, &PropertyKey::new("value"), EpochId::INITIAL),
        Some(Value::from("old"))
    );
}

#[test]
fn public_property_maintenance_abandon_and_late_ddl_conflict_publish_nothing() {
    for late_conflict in [false, true] {
        let store = LpgStore::new().unwrap();
        let node = store.create_node(&[]);
        store.set_node_property(node, "value", Value::from("old"));
        store.create_property_index("value");
        if late_conflict {
            store.create_property_index("occupied");
        }
        let before = store.observe_property_index("value").unwrap();
        let mut edits = vec![IndexRegistryEdit::Maintain {
            expected: store.observe_property_index("value").unwrap(),
            changes: IndexRegistryMaintenance::Property(vec![(
                node,
                Some(Value::from("old")),
                Some(Value::from("final")),
            )]),
        }];
        if late_conflict {
            edits.push(property("occupied", 999, "invalid"));
        }
        let mut workspace = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits,
        }]);
        let result = prepare_index_registry_batch(&mut workspace);
        assert_eq!(result.is_err(), late_conflict);
        drop(result);
        store.validate_index_registration(&before).unwrap();
        assert_eq!(
            store.find_nodes_by_property("value", &Value::from("old")),
            vec![node]
        );
        assert!(
            store
                .find_nodes_by_property("value", &Value::from("final"))
                .is_empty()
        );
    }
}

#[test]
fn index_batch_shares_exact_transition_with_buffered_data_and_keeps_final_phase_allocation_free() {
    use super::super::data_publication::StoreDataWorkspace;
    use grafeo_common::types::{EpochId, TransactionId};

    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Row"]);
    store.set_node_property(node, "value", Value::from("old"));
    let tx = TransactionId::new(910);
    store.set_node_property_buffered(node, "value", Value::from("final"), tx);
    let mut data_workspace = StoreDataWorkspace::new(tx, EpochId::INITIAL, EpochId::new(1));
    let mut registry_workspace = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
        store: &store,
        edits: vec![property("value", node.as_u64(), "final")],
    }]);
    registry_workspace.prepare_inputs().unwrap();
    {
        let authority = RegistryAuthority::acquire(
            &mut registry_workspace.registry.stores,
            &mut registry_workspace.authority,
        )
        .unwrap();
        let transition = authority.transition(&store).unwrap();
        let data = store
            .prepare_buffered_commit_data(transition, &mut data_workspace)
            .unwrap();
        let indexes =
            prepare_under_authority(&mut registry_workspace.registry, authority.workspace).unwrap();
        assert!(store.property_indexes.try_read().is_some());
        assert!(store.mutation_scope_gate.try_read().is_none());
        crate::allocation_test::start();
        let data = data.rebind().unwrap();
        let mut indexes = indexes.rebind().unwrap();
        let data = data.install();
        indexes.fences.install();
        let index_readers_excluded = store.property_indexes.try_read().is_none();
        let label_readers_excluded = store.node_labels.try_read().is_none();
        drop(indexes);
        drop(data);
        let traffic = crate::allocation_test::stop();
        assert_eq!(traffic, crate::allocation_test::Counts::default());
        assert!(index_readers_excluded && label_readers_excluded);
        assert!(store.mutation_scope_gate.try_read().is_none());
    }
    assert_eq!(store.current_epoch(), EpochId::new(1));
    assert_eq!(
        store.get_node_property(node, &PropertyKey::new("value")),
        Some(Value::from("final"))
    );
    assert_eq!(
        store
            .node_properties
            .get_at(node, &PropertyKey::new("value"), EpochId::INITIAL),
        Some(Value::from("old"))
    );
    assert_eq!(
        store.find_nodes_by_property("value", &Value::from("final")),
        vec![node]
    );
    assert!(
        store
            .find_nodes_by_property("value", &Value::from("old"))
            .is_empty()
    );
    assert!(!store.tx_property_overlay.read().contains_key(&tx));
    crate::allocation_test::start();
    drop(registry_workspace);
    drop(data_workspace);
    let retirement = crate::allocation_test::stop();
    assert!(
        retirement.dealloc > 0,
        "outer workspaces must own the allocations"
    );
}

#[test]
fn index_batch_late_registration_conflict_releases_no_payloads_under_data_writers() {
    use super::super::data_publication::StoreDataWorkspace;
    use grafeo_common::types::{EpochId, TransactionId};

    let store = LpgStore::new().unwrap();
    let node = store.create_node(&[]);
    store.set_node_property(node, "value", Value::from("old"));
    let tx = TransactionId::new(911);
    store.set_node_property_buffered(node, "value", Value::from("final"), tx);
    let mut data_workspace = StoreDataWorkspace::new(tx, EpochId::INITIAL, EpochId::new(1));
    let mut registry_workspace = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
        store: &store,
        edits: vec![property("value", node.as_u64(), "final")],
    }]);
    registry_workspace.prepare_inputs().unwrap();
    {
        let authority = RegistryAuthority::acquire(
            &mut registry_workspace.registry.stores,
            &mut registry_workspace.authority,
        )
        .unwrap();
        let data = store
            .prepare_buffered_commit_data(
                authority.transition(&store).unwrap(),
                &mut data_workspace,
            )
            .unwrap();
        let indexes =
            prepare_under_authority(&mut registry_workspace.registry, authority.workspace).unwrap();
        // Deliberate private fault injection, not a permitted mutation path:
        // normal writers cannot pass the continuously held store transition.
        store.property_indexes.write().insert(
            PropertyKey::new("value"),
            RegisteredIndex::new(Arc::new(PropertyIndexRows::new())),
        );
        let data = data.rebind().unwrap();
        crate::allocation_test::start();
        let rejected = indexes.rebind();
        let failed = rejected.is_err();
        drop(rejected);
        drop(data);
        let traffic = crate::allocation_test::stop();
        assert!(failed);
        assert_eq!(traffic, crate::allocation_test::Counts::default());
    }
    assert_eq!(store.current_epoch(), EpochId::INITIAL);
    assert_eq!(
        store.get_node_property(node, &PropertyKey::new("value")),
        Some(Value::from("old"))
    );
    assert!(store.tx_property_overlay.read().contains_key(&tx));
    assert!(
        store
            .find_nodes_by_property("value", &Value::from("final"))
            .is_empty()
    );
}

#[cfg(feature = "text-index")]
#[test]
fn index_batch_text_contention_under_data_writers_is_allocation_free_and_retryable() {
    use super::super::data_publication::StoreDataWorkspace;
    use grafeo_common::types::{EpochId, TransactionId};

    for held_target in [false, true] {
        let store = LpgStore::new().unwrap();
        let node = store.create_node(&["Doc"]);
        store.set_node_property(node, "body", Value::from("old"));
        let IndexRegistryContents::Text(index) = text_contents(node.as_u64(), "old") else {
            panic!("Text fixture")
        };
        let caller = Arc::new(parking_lot::RwLock::new(index));
        store.add_text_index("Doc", "body", Arc::clone(&caller));
        let target = store.text_indexes.read()[&encode_index_key("Doc", "body")]
            .target_identity()
            .upgrade()
            .unwrap();
        let expected = store.observe_text_index("Doc", "body").unwrap();
        let tx = TransactionId::new(912);
        store.set_node_property_buffered(node, "body", Value::from("final"), tx);
        let mut data_workspace = StoreDataWorkspace::new(tx, EpochId::INITIAL, EpochId::new(1));
        let mut registry_workspace = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![IndexRegistryEdit::Drop { expected }],
        }]);
        registry_workspace.prepare_inputs().unwrap();
        {
            let authority = RegistryAuthority::acquire(
                &mut registry_workspace.registry.stores,
                &mut registry_workspace.authority,
            )
            .unwrap();
            let data = store
                .prepare_buffered_commit_data(
                    authority.transition(&store).unwrap(),
                    &mut data_workspace,
                )
                .unwrap();
            let indexes =
                prepare_under_authority(&mut registry_workspace.registry, authority.workspace)
                    .unwrap();
            let held = if held_target {
                target.read()
            } else {
                caller.read()
            };
            let data = data.rebind().unwrap();
            crate::allocation_test::start();
            let rejected = indexes.rebind();
            let conflict = match rejected {
                Err(error) => Some(error),
                Ok(ready) => {
                    drop(ready);
                    None
                }
            };
            let traffic = crate::allocation_test::stop();
            assert_eq!(traffic, crate::allocation_test::Counts::default());
            assert!(matches!(conflict, Some(DataRebindError::Conflict(_))));
            drop(data);
            drop(held);
            let error = conflict.unwrap().into_error();
            assert!(matches!(
                error,
                Error::Transaction(TransactionError::WriteConflict(_))
            ));
            assert!(store.property_indexes.try_write().is_some());
            assert!(caller.try_write().is_some());
            assert!(target.try_write().is_some());
        }
        assert_eq!(store.current_epoch(), EpochId::INITIAL);
        assert!(store.get_text_index("Doc", "body").is_some());
        assert_eq!(
            store.get_node_property(node, &PropertyKey::new("body")),
            Some(Value::from("old"))
        );
        assert!(store.tx_property_overlay.read().contains_key(&tx));
        // A fresh workspace can retry the still-live exact registration.
        let expected = store.observe_text_index("Doc", "body").unwrap();
        let mut retry = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![IndexRegistryEdit::Drop { expected }],
        }]);
        drop(prepare_index_registry_batch(&mut retry).unwrap().install());
        assert!(store.get_text_index("Doc", "body").is_none());
    }
}

#[test]
fn index_batch_workspace_releases_all_writers_after_forgotten_borrowed_fence() {
    let first = LpgStore::new().unwrap();
    let second = LpgStore::new().unwrap();
    let mut workspace = IndexRegistryWorkspace::new(vec![
        StoreIndexEdits {
            store: &first,
            edits: vec![property("first", 1, "one")],
        },
        StoreIndexEdits {
            store: &second,
            edits: vec![property("second", 2, "two")],
        },
    ]);
    let ready = prepare_index_registry_batch(&mut workspace).unwrap();
    std::mem::forget(ready);
    assert!(first.property_indexes.try_write().is_none());
    assert!(second.property_indexes.try_write().is_none());
    drop(workspace);
    assert!(first.property_indexes.try_write().is_some());
    assert!(second.property_indexes.try_write().is_some());
    assert!(first.mutation_scope_gate.try_write().is_some());
    assert!(second.mutation_scope_gate.try_write().is_some());
    assert!(!first.has_property_index("first"));
    assert!(!second.has_property_index("second"));
}

#[cfg(feature = "text-index")]
#[test]
fn index_batch_forgotten_fence_cleans_text_writers_before_candidate_destructor() {
    use crate::index::text::{BM25Config, Tokenizer};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Probe {
        stores: [Arc<LpgStore>; 2],
        caller: Arc<parking_lot::RwLock<InvertedIndex>>,
        target: Arc<parking_lot::RwLock<InvertedIndex>>,
        observed: Arc<AtomicUsize>,
    }
    impl Tokenizer for Probe {
        fn tokenize(&self, _text: &str) -> Vec<String> {
            Vec::new()
        }
    }
    impl Drop for Probe {
        fn drop(&mut self) {
            let all_released = self.stores.iter().all(|store| {
                store.property_indexes.try_read().is_some()
                    && store.mutation_scope_gate.try_read().is_some()
            }) && self.caller.try_read().is_some()
                && self.target.try_read().is_some();
            self.observed
                .store(if all_released { 1 } else { 2 }, Ordering::Relaxed);
        }
    }

    let first = Arc::new(LpgStore::new().unwrap());
    let second = Arc::new(LpgStore::new().unwrap());
    let caller = Arc::new(parking_lot::RwLock::new(InvertedIndex::new(
        BM25Config::default(),
    )));
    second.add_text_index("Doc", "old", Arc::clone(&caller));
    let target = second.text_indexes.read()[&encode_index_key("Doc", "old")]
        .target_identity()
        .upgrade()
        .unwrap();
    let observed = Arc::new(AtomicUsize::new(0));
    let candidate = InvertedIndex::with_tokenizer(
        BM25Config::default(),
        Box::new(Probe {
            stores: [Arc::clone(&first), Arc::clone(&second)],
            caller: Arc::clone(&caller),
            target: Arc::clone(&target),
            observed: Arc::clone(&observed),
        }),
    );
    let expected = second.observe_text_index("Doc", "old").unwrap();
    let mut workspace = IndexRegistryWorkspace::new(vec![
        StoreIndexEdits {
            store: &first,
            edits: vec![IndexRegistryEdit::Create {
                key: IndexRegistryKey::Text {
                    label: "Doc".into(),
                    property: "private".into(),
                },
                contents: IndexRegistryContents::Text(candidate),
            }],
        },
        StoreIndexEdits {
            store: &second,
            edits: vec![IndexRegistryEdit::Drop { expected }],
        },
    ]);
    let ready = prepare_index_registry_batch(&mut workspace).unwrap();
    std::mem::forget(ready);
    assert!(caller.try_read().is_none());
    assert!(target.try_read().is_none());
    assert_eq!(observed.load(Ordering::Relaxed), 0);
    drop(workspace);
    // Observed IN the candidate's destructor, not merely after workspace Drop.
    // Default field destruction would leave the old Text writers held here.
    assert_eq!(observed.load(Ordering::Relaxed), 1);
    assert!(first.get_text_index("Doc", "private").is_none());
    assert!(second.get_text_index("Doc", "old").is_some());
}

#[test]
fn index_batch_property_multiple_stores_publish_and_replace() {
    let a = LpgStore::new().unwrap();
    let b = LpgStore::new().unwrap();
    a.create_property_index("unchanged");
    let unchanged = a.property_indexes.read()[&PropertyKey::new("unchanged")].clone();
    let mut workspace = IndexRegistryWorkspace::new(vec![
        StoreIndexEdits {
            store: &a,
            edits: vec![
                property("code", 11, "first"),
                property("other", 12, "second"),
            ],
        },
        StoreIndexEdits {
            store: &b,
            edits: vec![property("code", 21, "third")],
        },
    ]);
    let ready = prepare_index_registry_batch(&mut workspace)
        .expect("a real multi-store batch must prepare");
    drop(ready.install());
    assert_eq!(
        a.find_nodes_by_property("code", &Value::from("first")),
        vec![NodeId::new(11)]
    );
    assert_eq!(
        a.find_nodes_by_property("other", &Value::from("second")),
        vec![NodeId::new(12)]
    );
    assert_eq!(
        b.find_nodes_by_property("code", &Value::from("third")),
        vec![NodeId::new(21)]
    );
    assert!(Arc::ptr_eq(
        &unchanged.payload,
        &a.property_indexes.read()[&PropertyKey::new("unchanged")].payload
    ));
    assert!(Arc::ptr_eq(
        &unchanged.registration,
        &a.property_indexes.read()[&PropertyKey::new("unchanged")].registration
    ));
    let stale = a.observe_property_index("code").unwrap();
    let replaced = a.observe_property_index("code").unwrap();
    let removed = b.observe_property_index("code").unwrap();
    drop(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![
            StoreIndexEdits {
                store: &a,
                edits: vec![IndexRegistryEdit::Replace {
                    expected: replaced,
                    contents: IndexRegistryContents::Property(vec![(
                        NodeId::new(13),
                        Value::from("replacement"),
                    )]),
                }],
            },
            StoreIndexEdits {
                store: &b,
                edits: vec![IndexRegistryEdit::Drop { expected: removed }],
            },
        ]))
        .unwrap()
        .install(),
    );
    assert!(a.validate_index_registration(&stale).is_err());
    assert_eq!(
        a.find_nodes_by_property("code", &Value::from("replacement")),
        vec![NodeId::new(13)]
    );
    assert!(
        a.find_nodes_by_property("code", &Value::from("first"))
            .is_empty()
    );
    assert!(!b.has_property_index("code"));
}

#[test]
fn index_batch_abandon_and_late_conflict_preserve_live_registration() {
    let store = LpgStore::new().unwrap();
    store.create_property_index("live");
    let before = store.observe_property_index("live").unwrap();
    let mut workspace = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
        store: &store,
        edits: vec![property("private", 9, "x")],
    }]);
    let ready = prepare_index_registry_batch(&mut workspace).unwrap();
    drop(ready);
    assert!(!store.has_property_index("private"));
    assert!(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![property("private", 9, "x"), property("live", 10, "bad")]
        }]))
        .is_err()
    );
    assert!(!store.has_property_index("private"));
    store.validate_index_registration(&before).unwrap();
}

#[test]
fn index_batch_rejects_duplicate_stores_and_repeated_keys() {
    let store = LpgStore::new().unwrap();
    assert!(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![
            StoreIndexEdits {
                store: &store,
                edits: vec![property("a", 1, "a")]
            },
            StoreIndexEdits {
                store: &store,
                edits: vec![property("b", 2, "b")]
            },
        ]))
        .is_err()
    );
    assert!(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![property("a", 1, "a"), property("a", 2, "b")]
        }]))
        .is_err()
    );
    assert!(!store.has_property_index("a"));
    drop(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![property("a", 1, "a")],
        }]))
        .unwrap()
        .install(),
    );
    assert!(store.has_property_index("a"));
}

#[test]
fn index_batch_rejects_foreign_and_stale_observations() {
    let a = LpgStore::new().unwrap();
    let b = LpgStore::new().unwrap();
    a.create_property_index("live");
    b.create_property_index("live");
    let foreign = a.observe_property_index("live").unwrap();
    assert!(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &b,
            edits: vec![IndexRegistryEdit::Drop { expected: foreign }]
        }]))
        .is_err()
    );
    let stale = b.observe_property_index("live").unwrap();
    assert!(b.drop_property_index("live"));
    b.create_property_index("live");
    assert!(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &b,
            edits: vec![IndexRegistryEdit::Drop { expected: stale }]
        }]))
        .is_err()
    );
    let valid = b.observe_property_index("live").unwrap();
    drop(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &b,
            edits: vec![IndexRegistryEdit::Drop { expected: valid }],
        }]))
        .unwrap()
        .install(),
    );
    assert!(a.has_property_index("live"));
    assert!(!b.has_property_index("live"));
}

#[test]
fn index_batch_observation_does_not_supply_mutation_authority() {
    let store = LpgStore::new().unwrap();
    store.create_property_index("live");
    let authority = WriteAuthority::new();
    assert!(store.seal_unframed_writes(&authority));
    let denied = store.observe_property_index("live").unwrap();
    assert!(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![IndexRegistryEdit::Drop { expected: denied }]
        }]))
        .is_err()
    );
    with_authority(&authority, || {
        let allowed = store.observe_property_index("live").unwrap();
        drop(
            prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
                store: &store,
                edits: vec![IndexRegistryEdit::Drop { expected: allowed }],
            }]))
            .unwrap()
            .install(),
        );
    });
    assert!(!store.has_property_index("live"));
}

#[test]
fn index_batch_retains_publication_fence_until_installed_release() {
    use std::sync::mpsc;
    use std::time::Duration;
    let store = Arc::new(LpgStore::new().unwrap());
    let mut workspace = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
        store: &store,
        edits: vec![property("prepared", 1, "ready")],
    }]);
    let ready = prepare_index_registry_batch(&mut workspace).unwrap();
    let prepared_store_locked = store.mutation_scope_gate.try_read().is_none();
    let prepared_registry_locked = store.property_indexes.try_write().is_none();
    let worker_store = Arc::clone(&store);
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        worker_store.create_property_index("contender");
        done_tx.send(()).unwrap();
    });
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let before_install = done_rx.recv_timeout(Duration::from_millis(30));
    let installed = ready.install();
    let installed_store_locked = store.mutation_scope_gate.try_read().is_none();
    let installed_registry_locked = store.property_indexes.try_write().is_none();
    let before_release = done_rx.recv_timeout(Duration::from_millis(30));
    drop(installed);
    let completed = done_rx.recv_timeout(Duration::from_secs(2));
    worker.join().unwrap();
    assert!(prepared_store_locked && prepared_registry_locked);
    assert!(installed_store_locked && installed_registry_locked);
    assert!(matches!(
        before_install,
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    assert!(matches!(
        before_release,
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    completed.unwrap();
    assert!(store.has_property_index("prepared"));
    assert!(store.has_property_index("contender"));
    assert!(store.mutation_scope_gate.try_write().is_some());
    assert!(store.property_indexes.try_write().is_some());
}

#[cfg(feature = "text-index")]
fn text_contents(id: u64, text: &str) -> IndexRegistryContents {
    let mut index = InvertedIndex::new(crate::index::text::BM25Config::default());
    index.insert(NodeId::new(id), text);
    IndexRegistryContents::Text(index)
}

#[cfg(feature = "vector-index")]
fn vector_contents() -> IndexRegistryContents {
    let index = VectorIndexKind::Hnsw(crate::index::vector::HnswIndex::new(
        crate::index::vector::HnswConfig::new(2, crate::index::vector::DistanceMetric::Euclidean),
    ));
    index.insert(NodeId::new(7), &[1.0, 0.0], &|_| None);
    IndexRegistryContents::Vector(index)
}

#[cfg(all(feature = "text-index", feature = "vector-index"))]
#[test]
fn index_batch_all_families_share_multiple_slots_and_keep_old_views() {
    let store = LpgStore::new().unwrap();
    drop(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![
                property("code", 1, "first"),
                IndexRegistryEdit::Create {
                    key: IndexRegistryKey::Text {
                        label: "Doc".into(),
                        property: "body".into(),
                    },
                    contents: text_contents(2, "oldtoken"),
                },
                IndexRegistryEdit::Create {
                    key: IndexRegistryKey::Vector {
                        label: "Doc".into(),
                        property: "embedding".into(),
                    },
                    contents: vector_contents(),
                },
            ],
        }]))
        .unwrap()
        .install(),
    );
    assert_eq!(store.index_slots.lock().len(), 2);
    let old = store.get_text_index("Doc", "body").unwrap();
    let text = store.observe_text_index("Doc", "body").unwrap();
    let vector = store.observe_vector_index("Doc", "embedding").unwrap();
    let stale = store.observe_text_index("Doc", "body").unwrap();
    drop(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![
                IndexRegistryEdit::Replace {
                    expected: text,
                    contents: text_contents(3, "newtoken"),
                },
                IndexRegistryEdit::Drop { expected: vector },
            ],
        }]))
        .unwrap()
        .install(),
    );
    assert!(store.validate_index_registration(&stale).is_err());
    assert!(store.get_vector_index("Doc", "embedding").is_none());
    assert!(old.read().contains(NodeId::new(2)));
    assert!(!old.read().contains(NodeId::new(3)));
    let new = store.get_text_index("Doc", "body").unwrap();
    assert!(new.read().contains(NodeId::new(3)));
    assert!(!new.read().contains(NodeId::new(2)));
}

#[cfg(feature = "text-index")]
#[test]
fn index_batch_owned_text_forwarder_is_not_a_private_replacement() {
    let store = LpgStore::new().unwrap();
    let mut index = InvertedIndex::new(crate::index::text::BM25Config::default());
    index.insert(NodeId::new(5), "kepttoken");
    let caller = Arc::new(parking_lot::RwLock::new(index));
    store.add_text_index("Doc", "body", Arc::clone(&caller));
    let view = store.get_text_index("Doc", "body").unwrap();
    let valid = store.observe_text_index("Doc", "body").unwrap();
    let expected = store.observe_text_index("Doc", "body").unwrap();
    let forwarded = std::mem::replace(
        &mut *caller.write(),
        InvertedIndex::new(crate::index::text::BM25Config::default()),
    );
    assert!(
        forwarded.contains(NodeId::new(5)),
        "fixture owns a live forwarding shell"
    );
    assert!(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![
                property("private", 1, "unpublished"),
                IndexRegistryEdit::Replace {
                    expected,
                    contents: IndexRegistryContents::Text(forwarded)
                },
            ]
        }]))
        .is_err()
    );
    assert!(!store.has_property_index("private"));
    store.validate_index_registration(&valid).unwrap();
    assert!(view.read().contains(NodeId::new(5)));
    drop(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![IndexRegistryEdit::Replace {
                expected: valid,
                contents: text_contents(6, "fresh token"),
            }],
        }]))
        .unwrap()
        .install(),
    );
    assert!(view.read().contains(NodeId::new(5)));
    assert!(
        store
            .get_text_index("Doc", "body")
            .unwrap()
            .read()
            .contains(NodeId::new(6))
    );
}

#[cfg(feature = "vector-index")]
#[test]
fn index_batch_rejects_same_vector_arc_aba_and_foreign_owned_payload() {
    let a = LpgStore::new().unwrap();
    let b = LpgStore::new().unwrap();
    let IndexRegistryContents::Vector(index) = vector_contents() else {
        panic!("vector fixture")
    };
    let index = Arc::new(index);
    a.add_vector_index("Doc", "embedding", Arc::clone(&index));
    let stale = a.observe_vector_index("Doc", "embedding").unwrap();
    assert!(a.remove_vector_index("Doc", "embedding"));
    a.add_vector_index("Doc", "embedding", Arc::clone(&index));
    assert!(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &a,
            edits: vec![IndexRegistryEdit::Drop { expected: stale }]
        }]))
        .is_err()
    );
    assert_eq!(a.get_vector_index("Doc", "embedding").unwrap().len(), 1);
    assert!(a.remove_vector_index("Doc", "embedding"));
    let owned = Arc::try_unwrap(index).ok().expect("no live vector aliases");
    assert!(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &b,
            edits: vec![
                property("private", 1, "no"),
                IndexRegistryEdit::Create {
                    key: IndexRegistryKey::Vector {
                        label: "Doc".into(),
                        property: "embedding".into()
                    },
                    contents: IndexRegistryContents::Vector(owned)
                },
            ]
        }]))
        .is_err()
    );
    assert!(!b.has_property_index("private"));
    assert!(b.get_vector_index("Doc", "embedding").is_none());
    drop(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &b,
            edits: vec![IndexRegistryEdit::Create {
                key: IndexRegistryKey::Vector {
                    label: "Doc".into(),
                    property: "embedding".into(),
                },
                contents: vector_contents(),
            }],
        }]))
        .unwrap()
        .install(),
    );
    assert_eq!(b.get_vector_index("Doc", "embedding").unwrap().len(), 1);
}

#[test]
fn index_batch_reversed_parent_child_input_publishes_both() {
    let parent = LpgStore::new().unwrap();
    let child = Arc::new(parent.new_named_graph_candidate().unwrap());
    parent.install_named_graphs(grafeo_common::utils::hash::FxHashMap::from_iter([(
        "child".into(),
        Arc::clone(&child),
    )]));
    drop(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![
            StoreIndexEdits {
                store: &child,
                edits: vec![property("child", 2, "c")],
            },
            StoreIndexEdits {
                store: &parent,
                edits: vec![property("parent", 1, "p")],
            },
        ]))
        .unwrap()
        .install(),
    );
    assert!(parent.has_property_index("parent"));
    assert!(child.has_property_index("child"));
}

#[cfg(feature = "text-index")]
struct ReentrantTokenizer {
    store: Arc<LpgStore>,
    outer: Arc<parking_lot::Mutex<()>>,
    releases: Arc<parking_lot::Mutex<Vec<bool>>>,
    result: std::sync::mpsc::Sender<(bool, std::thread::JoinHandle<()>)>,
}

#[cfg(feature = "text-index")]
impl crate::index::text::Tokenizer for ReentrantTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        vec![text.to_owned()]
    }
}

#[cfg(feature = "text-index")]
impl Drop for ReentrantTokenizer {
    fn drop(&mut self) {
        self.releases.lock().push(self.outer.try_lock().is_some());
        let store = Arc::clone(&self.store);
        let outer = Arc::clone(&self.outer);
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            // Reenters the modeled engine gate, a global Text transition and
            // the store registry. A batch-local unlock alone is insufficient.
            let _outer = outer.lock();
            store.add_text_index(
                "Reentry",
                "body",
                Arc::new(parking_lot::RwLock::new(InvertedIndex::new(
                    crate::index::text::BM25Config::default(),
                ))),
            );
            let _ = tx.send(());
        });
        let completed = rx.recv_timeout(std::time::Duration::from_secs(2)).is_ok();
        let _ = self.result.send((completed, worker));
    }
}

#[cfg(feature = "text-index")]
#[test]
fn index_batch_text_destructors_reenter_after_late_error_abandon_and_retirement() {
    struct ResetReservations;
    impl Drop for ResetReservations {
        fn drop(&mut self) {
            RESERVATION_FAILURE.with(|counter| counter.set(None));
        }
    }

    for mode in [
        "duplicate_store",
        "duplicate_key",
        "wrong_family",
        "forwarder",
        "mid_property",
        "late_error",
        "abandon",
        "retire",
        "unwind_ready",
        "unwind_installed",
    ] {
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_property_index("occupied");
        let outer = Arc::new(parking_lot::Mutex::new(()));
        let releases = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let (tx, rx) = std::sync::mpsc::channel();
        let mut index = InvertedIndex::with_tokenizer(
            crate::index::text::BM25Config::default(),
            Box::new(ReentrantTokenizer {
                store: Arc::clone(&store),
                outer: Arc::clone(&outer),
                releases: Arc::clone(&releases),
                result: tx,
            }),
        );
        index.insert(NodeId::new(8), "customtoken");
        if mode == "forwarder" {
            drop(index.pin_registry_target());
            assert!(!index.is_concrete_registry_candidate());
        }

        let edits = if matches!(mode, "retire" | "unwind_installed") {
            store.add_text_index("Doc", "body", Arc::new(parking_lot::RwLock::new(index)));
            vec![IndexRegistryEdit::Drop {
                expected: store.observe_text_index("Doc", "body").unwrap(),
            }]
        } else {
            let key = if matches!(mode, "wrong_family" | "duplicate_key") {
                IndexRegistryKey::Property(PropertyKey::new("duplicate"))
            } else {
                IndexRegistryKey::Text {
                    label: "Doc".into(),
                    property: "body".into(),
                }
            };
            let mut edits = Vec::new();
            if mode == "duplicate_key" {
                edits.push(property("duplicate", 1, "first"));
            }
            edits.push(IndexRegistryEdit::Create {
                key,
                contents: IndexRegistryContents::Text(index),
            });
            if mode == "late_error" {
                edits.push(property("occupied", 1, "bad"));
            }
            if mode == "mid_property" {
                edits.push(IndexRegistryEdit::Create {
                    key: IndexRegistryKey::Property(PropertyKey::new("partial")),
                    contents: IndexRegistryContents::Property(vec![
                        (NodeId::new(11), Value::from("first")),
                        (NodeId::new(12), Value::from("second")),
                    ]),
                });
            }
            edits
        };
        let mut inputs = Vec::new();
        if mode == "duplicate_store" {
            inputs.push(StoreIndexEdits {
                store: &store,
                edits: vec![property("first_store", 1, "private")],
            });
        }
        inputs.push(StoreIndexEdits {
            store: &store,
            edits,
        });
        // This owner encloses the modeled engine gate, including unwinding.
        let mut workspace = IndexRegistryWorkspace::new(inputs);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _outer = outer.lock();
            let _reset = ResetReservations;
            if mode == "mid_property" {
                // Seen set, key set, Property map, and first membership pass;
                // fail the second membership after the partial map is populated.
                RESERVATION_FAILURE.with(|counter| counter.set(Some(4)));
            }
            let result = prepare_index_registry_batch(&mut workspace);
            match mode {
                "abandon" => drop(result.unwrap()),
                "retire" => drop(result.unwrap().install()),
                "unwind_ready" => {
                    let _ready = result.unwrap();
                    std::panic::resume_unwind(Box::new("ready retirement probe"));
                }
                "unwind_installed" => {
                    let _installed = result.unwrap().install();
                    std::panic::resume_unwind(Box::new("installed retirement probe"));
                }
                _ => assert!(result.is_err(), "{mode} must reject preparation"),
            }
            assert!(
                releases.lock().is_empty(),
                "{mode}: retired under outer gate"
            );
        }));
        assert_eq!(
            outcome.is_err(),
            matches!(mode, "unwind_ready" | "unwind_installed"),
            "{mode}: unexpected unwind outcome",
        );
        assert!(
            releases.lock().is_empty(),
            "{mode}: proof destruction must not retire workspace payloads",
        );
        assert!(store.mutation_scope_gate.try_write().is_some());
        assert!(store.property_indexes.try_write().is_some());
        assert!(store.text_indexes.try_write().is_some());
        if mode == "mid_property" {
            let contents = workspace.registry.stores[0].edits[1]
                .contents
                .as_ref()
                .unwrap();
            let PrivateContents::Property { rows, prepared, .. } = contents else {
                panic!("partial Property fixture");
            };
            assert_eq!(
                rows.len(),
                2,
                "raw rows remain owned after conversion failure"
            );
            assert_eq!(
                prepared
                    .payload
                    .get(&HashableValue::new(Value::from("first")))
                    .unwrap()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
                vec![NodeId::new(11)],
            );
            assert!(
                prepared
                    .payload
                    .get(&HashableValue::new(Value::from("second")))
                    .unwrap()
                    .is_empty(),
                "the exact mid-membership failure must leave the partial map anchored",
            );
        }
        drop(workspace);
        assert_eq!(
            *releases.lock(),
            vec![true],
            "{mode}: outer retirement order"
        );
        let (completed, worker) = rx.recv_timeout(std::time::Duration::from_secs(3)).unwrap();
        worker.join().unwrap();
        assert!(
            completed,
            "{mode}: payload destructor ran under a retained publication gate",
        );
        assert!(store.get_text_index("Reentry", "body").is_some());
        assert!(store.get_text_index("Doc", "body").is_none());
    }
}

#[test]
fn index_batch_workspace_cannot_be_prepared_twice() {
    let store = LpgStore::new().unwrap();
    let mut workspace = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
        store: &store,
        edits: vec![property("private", 1, "retained")],
    }]);
    drop(prepare_index_registry_batch(&mut workspace).unwrap());
    assert!(prepare_index_registry_batch(&mut workspace).is_err());
    assert!(!store.has_property_index("private"));
}

#[test]
fn index_batch_displaced_property_rows_outlive_the_installed_fence() {
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Doc"]);
    store.set_node_property(node, "code", Value::from("old"));
    store.create_property_index("code");
    let previous = {
        let registry = store.property_indexes.read();
        Arc::downgrade(&registry[&PropertyKey::new("code")].payload)
    };
    let expected = store.observe_property_index("code").unwrap();
    let mut workspace = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
        store: &store,
        edits: vec![IndexRegistryEdit::Replace {
            expected,
            contents: IndexRegistryContents::Property(vec![(node, Value::from("new"))]),
        }],
    }]);
    let outer = parking_lot::Mutex::new(());
    {
        let _outer = outer.lock();
        let ready = prepare_index_registry_batch(&mut workspace).unwrap();
        drop(ready.install());
        assert!(store.property_indexes.try_write().is_some());
        let retired = previous
            .upgrade()
            .expect("workspace retains displaced rows");
        assert!(
            retired
                .get(&HashableValue::new(Value::from("old")))
                .unwrap()
                .contains(&node)
        );
    }
    assert_eq!(
        store.find_nodes_by_property("code", &Value::from("new")),
        vec![node]
    );
    assert!(previous.upgrade().is_some());
    drop(workspace);
    assert!(previous.upgrade().is_none());
}

#[cfg(all(feature = "text-index", feature = "vector-index"))]
#[test]
fn index_batch_unchanged_handles_keep_exact_vector_and_text_history()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    use grafeo_common::storage::section::Section;
    use grafeo_common::types::{EpochId, TransactionId};
    let store = LpgStore::new().unwrap();
    let IndexRegistryContents::Vector(vector) = vector_contents() else {
        panic!("vector fixture")
    };
    let vector = Arc::new(vector);
    store.add_vector_index("Doc", "embedding", Arc::clone(&vector));
    let section = crate::index::vector::VectorStoreSection::new(vec![(
        crate::graph::lpg::PhysicalIndexKey::vector(
            grafeo_common::types::GraphPath::root(),
            "Doc",
            "embedding",
        ),
        Arc::clone(&vector),
    )]);
    let bytes = section.serialize().unwrap();
    let vector_observation = store.observe_vector_index("Doc", "embedding").unwrap();
    let config = crate::index::text::BM25Config { k1: 1.7, b: 0.3 };
    let mut text = InvertedIndex::new(config);
    text.insert_versioned(
        NodeId::new(10),
        "first historical row",
        EpochId::new(4),
        None,
    );
    text.insert_versioned(
        NodeId::new(11),
        "second retained row",
        EpochId::new(8),
        None,
    );
    let text = Arc::new(parking_lot::RwLock::new(text));
    store.add_text_index("Doc", "body", Arc::clone(&text));
    let target = store.text_indexes.read()[&encode_index_key("Doc", "body")].target_identity();
    let observation = store.observe_text_index("Doc", "body").unwrap();
    drop(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![property("new", 12, "new")],
        }]))
        .unwrap()
        .install(),
    );
    store.validate_index_registration(&observation).unwrap();
    store
        .validate_index_registration(&vector_observation)
        .unwrap();
    assert_eq!(section.serialize().unwrap(), bytes);
    assert!(std::sync::Weak::ptr_eq(
        &target,
        &store.text_indexes.read()[&encode_index_key("Doc", "body")].target_identity()
    ));
    let retained = store.get_text_index("Doc", "body").unwrap();
    assert_eq!(retained.read().config().k1.to_bits(), 1.7_f64.to_bits());
    assert_eq!(retained.read().config().b.to_bits(), 0.3_f64.to_bits());
    assert_eq!(
        retained
            .read()
            .doc_count_at(EpochId::new(6), TransactionId::SYSTEM)?,
        1
    );
    assert_eq!(
        retained
            .read()
            .doc_count_at(EpochId::new(9), TransactionId::SYSTEM)?,
        2
    );
    Ok(())
}

#[cfg(feature = "compact-store")]
#[test]
fn index_batch_stale_compact_representation_cannot_publish() {
    let source = Arc::new(LpgStore::new().unwrap());
    source.create_property_index("live");
    let old = source.observe_property_index("live").unwrap();
    let topology = source.pin_named_graph_topology();
    let transition = source.pin_exclusive_unframed_transition().unwrap();
    let target = Arc::new(
        transition
            .prepare_same_incarnation_empty_successor()
            .unwrap(),
    );
    let transfer = transition
        .prepare_same_incarnation_representation_transfer(
            &topology,
            Arc::clone(&source),
            Arc::clone(&target),
        )
        .unwrap();
    transfer.validate_unpublished_target().unwrap();
    transfer.publish().commit();
    drop(transition);
    drop(topology);
    assert!(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &source,
            edits: vec![property("forbidden", 1, "x")]
        }]))
        .is_err()
    );
    assert!(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &target,
            edits: vec![IndexRegistryEdit::Drop { expected: old }]
        }]))
        .is_err()
    );
    let valid = target.observe_property_index("live").unwrap();
    drop(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &target,
            edits: vec![IndexRegistryEdit::Drop { expected: valid }],
        }]))
        .unwrap()
        .install(),
    );
    assert!(!source.has_property_index("forbidden"));
    assert!(source.has_property_index("live"));
    assert!(!target.has_property_index("live"));
}

#[test]
fn index_batch_reservation_denial_never_publishes_partial_postimages() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            RESERVATION_FAILURE.with(|counter| counter.set(None));
        }
    }
    let mut rejected = 0;
    let mut admitted = false;
    for failure_at in 0..128 {
        let store = LpgStore::new().unwrap();
        let node = store.create_node(&["Doc"]);
        store.set_node_property(node, "live", Value::from("original"));
        store.create_property_index("live");
        let exact = store.observe_property_index("live").unwrap();
        let expected = store.observe_property_index("live").unwrap();
        let mut workspace = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![
                property("new", 9, "private"),
                IndexRegistryEdit::Replace {
                    expected,
                    contents: IndexRegistryContents::Property(vec![(
                        node,
                        Value::from("replacement"),
                    )]),
                },
            ],
        }]);
        let result = {
            let _reset = Reset;
            RESERVATION_FAILURE.with(|counter| counter.set(Some(failure_at)));
            prepare_index_registry_batch(&mut workspace)
        };
        match result {
            Ok(prepared) => {
                drop(prepared);
                admitted = true;
            }
            Err(error) => {
                assert!(matches!(
                    error,
                    Error::Storage(grafeo_common::utils::error::StorageError::Full)
                ));
                rejected += 1;
            }
        }
        store.validate_index_registration(&exact).unwrap();
        assert!(!store.has_property_index("new"));
        assert_eq!(
            store.find_nodes_by_property("live", &Value::from("original")),
            vec![node]
        );
        assert!(
            store
                .find_nodes_by_property("live", &Value::from("replacement"))
                .is_empty()
        );
        if admitted {
            break;
        }
    }
    assert!(
        rejected > 10,
        "fixture reaches early and late fallible reservation sites"
    );
    assert!(
        admitted,
        "same operation must admit once every reservation is allowed"
    );
}

#[cfg(feature = "vector-index")]
#[test]
fn index_batch_checked_slot_exhaustion_does_not_publish_first_binding() {
    use std::sync::atomic::{AtomicU64, Ordering};
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            SLOT_SOURCE.with_borrow_mut(|source| *source = None);
        }
    }
    let store = LpgStore::new().unwrap();
    let source = Arc::new(AtomicU64::new(u64::MAX - 1));
    let mut workspace = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
        store: &store,
        edits: vec![
            IndexRegistryEdit::Create {
                key: IndexRegistryKey::Vector {
                    label: "Doc".into(),
                    property: "first".into(),
                },
                contents: vector_contents(),
            },
            IndexRegistryEdit::Create {
                key: IndexRegistryKey::Vector {
                    label: "Doc".into(),
                    property: "second".into(),
                },
                contents: vector_contents(),
            },
        ],
    }]);
    let result = {
        let _reset = Reset;
        SLOT_SOURCE.with_borrow_mut(|slot| *slot = Some(Arc::clone(&source)));
        prepare_index_registry_batch(&mut workspace)
    };
    assert!(result.is_err());
    assert_eq!(source.load(Ordering::Acquire), u64::MAX);
    assert!(store.index_slots.lock().is_empty());
    assert!(store.get_vector_index("Doc", "first").is_none());
    assert!(store.get_vector_index("Doc", "second").is_none());
    drop(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![
                IndexRegistryEdit::Create {
                    key: IndexRegistryKey::Vector {
                        label: "Doc".into(),
                        property: "first".into(),
                    },
                    contents: vector_contents(),
                },
                IndexRegistryEdit::Create {
                    key: IndexRegistryKey::Vector {
                        label: "Doc".into(),
                        property: "second".into(),
                    },
                    contents: vector_contents(),
                },
            ],
        }]))
        .unwrap()
        .install(),
    );
    assert_eq!(store.index_slots.lock().len(), 2);
    assert_eq!(store.get_vector_index("Doc", "first").unwrap().len(), 1);
    assert_eq!(store.get_vector_index("Doc", "second").unwrap().len(), 1);
}

#[cfg(feature = "text-index")]
#[test]
fn index_batch_same_text_arc_aba_rejects_without_losing_live_target() {
    let store = LpgStore::new().unwrap();
    let IndexRegistryContents::Text(index) = text_contents(7, "retained") else {
        panic!("Text fixture")
    };
    let caller = Arc::new(parking_lot::RwLock::new(index));
    store.add_text_index("Doc", "body", Arc::clone(&caller));
    let stale = store.observe_text_index("Doc", "body").unwrap();
    assert!(store.remove_text_index("Doc", "body"));
    store.add_text_index("Doc", "body", Arc::clone(&caller));
    let valid = store.observe_text_index("Doc", "body").unwrap();
    assert!(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![IndexRegistryEdit::Drop { expected: stale }]
        }]))
        .is_err()
    );
    assert!(
        store
            .get_text_index("Doc", "body")
            .unwrap()
            .read()
            .contains(NodeId::new(7))
    );
    store.validate_index_registration(&valid).unwrap();
    drop(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![IndexRegistryEdit::Drop { expected: valid }],
        }]))
        .unwrap()
        .install(),
    );
    assert!(store.get_text_index("Doc", "body").is_none());
    assert!(caller.read().contains(NodeId::new(7)));
}

#[cfg(feature = "text-index")]
#[test]
fn index_batch_wrong_replacement_family_keeps_old_property() {
    let store = LpgStore::new().unwrap();
    store.create_property_index("code");
    let observation = store.observe_property_index("code").unwrap();
    let valid = store.observe_property_index("code").unwrap();
    assert!(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![IndexRegistryEdit::Replace {
                expected: observation,
                contents: text_contents(1, "wrong family")
            }]
        }]))
        .is_err()
    );
    store.validate_index_registration(&valid).unwrap();
    drop(
        prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &store,
            edits: vec![IndexRegistryEdit::Replace {
                expected: valid,
                contents: IndexRegistryContents::Property(vec![(
                    NodeId::new(2),
                    Value::from("right"),
                )]),
            }],
        }]))
        .unwrap()
        .install(),
    );
    assert_eq!(
        store.find_nodes_by_property("code", &Value::from("right")),
        vec![NodeId::new(2)]
    );
}

#[cfg(feature = "text-index")]
#[test]
fn index_batch_shared_text_caller_gates_are_fenced_once_across_stores() {
    use std::io::{BufRead, Write};
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};
    const CHILD: &str = "GRAFEO_INDEX_BATCH_SHARED_TEXT_GATE_CHILD";
    const READY: &str = "shared-text-gate-fixture-ready";
    if std::env::var_os(CHILD).is_some() {
        for same_store in [true, false] {
            let first = LpgStore::new().unwrap();
            let other = LpgStore::new().unwrap();
            let second = if same_store { &first } else { &other };
            let IndexRegistryContents::Text(index) = text_contents(11, "firsttarget") else {
                panic!("Text fixture")
            };
            let caller = Arc::new(parking_lot::RwLock::new(index));
            first.add_text_index("Doc", "first", Arc::clone(&caller));
            let first_view = first.get_text_index("Doc", "first").unwrap();
            let first_target = first.text_indexes.read()[&encode_index_key("Doc", "first")]
                .target_identity()
                .upgrade()
                .unwrap();
            // Existing public operations admit a second concrete target behind
            // the same caller Arc, without changing the first registry target.
            let IndexRegistryContents::Text(fresh) = text_contents(22, "secondtarget") else {
                panic!("Text fixture")
            };
            let moved_forwarder = std::mem::replace(&mut *caller.write(), fresh);
            second.add_text_index("Doc", "second", Arc::clone(&caller));
            let second_view = second.get_text_index("Doc", "second").unwrap();
            let second_target = second.text_indexes.read()[&encode_index_key("Doc", "second")]
                .target_identity()
                .upgrade()
                .unwrap();
            assert!(!Arc::ptr_eq(&first_target, &second_target));
            assert!(first_view.read().contains(NodeId::new(11)));
            assert!(second_view.read().contains(NodeId::new(22)));
            assert!(moved_forwarder.contains(NodeId::new(11)));
            assert!(caller.try_write().is_some());
            assert!(first_target.try_write().is_some());
            assert!(second_target.try_write().is_some());
            let first_drop = IndexRegistryEdit::Drop {
                expected: first.observe_text_index("Doc", "first").unwrap(),
            };
            let second_drop = IndexRegistryEdit::Drop {
                expected: second.observe_text_index("Doc", "second").unwrap(),
            };
            let edits = if same_store {
                vec![StoreIndexEdits {
                    store: &first,
                    edits: vec![first_drop, second_drop],
                }]
            } else {
                vec![
                    StoreIndexEdits {
                        store: &first,
                        edits: vec![first_drop],
                    },
                    StoreIndexEdits {
                        store: second,
                        edits: vec![second_drop],
                    },
                ]
            };
            println!("{READY}");
            std::io::stdout().flush().unwrap();
            let mut workspace = IndexRegistryWorkspace::new(edits);
            let prepared = prepare_index_registry_batch(&mut workspace).unwrap();
            assert!(caller.try_write().is_none());
            assert!(first_target.try_write().is_none());
            assert!(
                second_target.try_write().is_none(),
                "moved forwarders also require the second concrete target fence"
            );
            let installed = prepared.install();
            assert!(caller.try_write().is_none());
            assert!(first_target.try_write().is_none());
            assert!(second_target.try_write().is_none());
            drop(installed);
            assert!(caller.try_write().is_some());
            assert!(first_target.try_write().is_some());
            assert!(second_target.try_write().is_some());
            assert!(first.get_text_index("Doc", "first").is_none());
            assert!(second.get_text_index("Doc", "second").is_none());
            assert!(first_view.read().contains(NodeId::new(11)));
            assert!(second_view.read().contains(NodeId::new(22)));
        }
        for held_target in [false, true] {
            let store = LpgStore::new().unwrap();
            let IndexRegistryContents::Text(index) = text_contents(33, "contentiontarget") else {
                panic!("Text fixture")
            };
            let caller = Arc::new(parking_lot::RwLock::new(index));
            store.add_text_index("Doc", "held", Arc::clone(&caller));
            let target = store.text_indexes.read()[&encode_index_key("Doc", "held")]
                .target_identity()
                .upgrade()
                .unwrap();
            let valid = store.observe_text_index("Doc", "held").unwrap();
            let expected = store.observe_text_index("Doc", "held").unwrap();
            let lock = if held_target { &target } else { &caller };
            let held = lock.write();
            assert!(
                prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![
                    StoreIndexEdits {
                        store: &store,
                        edits: vec![
                            property("private", 1, "must not publish"),
                            IndexRegistryEdit::Drop { expected },
                        ]
                    }
                ]))
                .is_err()
            );
            assert!(!store.has_property_index("private"));
            store.validate_index_registration(&valid).unwrap();
            if held_target {
                assert!(
                    caller.try_write().is_some(),
                    "partial gate acquisition must roll back on target contention"
                );
            }
            drop(held);
            assert!(
                store
                    .get_text_index("Doc", "held")
                    .unwrap()
                    .read()
                    .contains(NodeId::new(33))
            );
            drop(
                prepare_index_registry_batch(&mut IndexRegistryWorkspace::new(vec![
                    StoreIndexEdits {
                        store: &store,
                        edits: vec![IndexRegistryEdit::Drop { expected: valid }],
                    },
                ]))
                .unwrap()
                .install(),
            );
            assert!(store.get_text_index("Doc", "held").is_none());
        }
        return;
    }

    // A defective implementation self-deadlocks under global gates. Isolate
    // that witness in this exact child test, and always reap only our child.
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "graph::lpg::store::index_batch::tests::index_batch_shared_text_caller_gates_are_fenced_once_across_stores", "--nocapture", "--test-threads=1"])
        .env(CHILD, "1").stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().unwrap();
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut output = Vec::new();
        for line in std::io::BufReader::new(stdout).lines() {
            let line = line.unwrap();
            if line.contains(READY) {
                let _ = tx.send(());
            }
            output.push(line);
        }
        output
    });
    let fixture_ready = rx.recv_timeout(Duration::from_secs(2)).is_ok();
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut exited = None;
    while fixture_ready && Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(status)) => {
                exited = Some(status);
                break;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(5)),
            Err(_) => break,
        }
    }
    if exited.is_none() {
        let _ = child.kill();
    }
    let reaped = child.wait().unwrap();
    let output = reader.join().unwrap();
    assert!(
        fixture_ready,
        "child did not construct its admitted fixture: {output:?}"
    );
    assert!(
        exited.is_some(),
        "shared Text gate batch self-deadlocked; bounded child reaped: {output:?}"
    );
    assert!(
        reaped.success(),
        "shared gate proof assertion failed: {output:?}"
    );
}
