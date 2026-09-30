use super::*;
use crate::allocation_test as allocation;
use grafeo_common::types::{PropertyKey, Value};
use std::cell::Cell;

fn tx() -> TransactionId {
    TransactionId::new(710)
}

fn epoch(value: u64) -> EpochId {
    EpochId::new(value)
}

fn workspace() -> StoreDataWorkspace {
    StoreDataWorkspace::new(tx(), epoch(0), epoch(1))
}

fn stage(store: &LpgStore, value: i64) -> (NodeId, NodeId, EdgeId) {
    let changed = store.create_node(&["Before"]);
    let deleted = store.create_node(&["Deleted"]);
    let edge = store.create_edge(changed, deleted, "LINK");
    store.set_node_property(changed, "value", Value::Int64(value));
    store.set_node_property(deleted, "value", Value::Int64(value));
    store.set_edge_property(edge, "value", Value::Int64(value));
    store.set_node_property_buffered(changed, "value", Value::Int64(value + 1), tx());
    store.set_node_property_buffered(changed, "new", Value::from("retained payload"), tx());
    store.remove_label_buffered(changed, "Before", tx());
    store.add_label_buffered(changed, "After", tx());
    assert!(store.delete_edge_transactional(edge, epoch(0), tx()));
    assert!(store.delete_node_transactional(deleted, epoch(0), tx()));
    (changed, deleted, edge)
}

fn assert_writers_released(store: &LpgStore) {
    assert!(store.label_index.try_write().is_some());
    assert!(store.node_labels.try_write().is_some());
    assert!(store.edge_type_live_counts.try_write().is_some());
    assert!(store.property_undo_log.try_write().is_some());
    assert!(store.pending_tx_creates.try_write().is_some());
    assert!(store.tx_property_overlay.try_write().is_some());
    assert!(store.pending_tx_deletes.try_write().is_some());
    assert!(store.pending_tx_edge_deletes.try_write().is_some());
    #[cfg(feature = "text-index")]
    assert!(store.text_index_overlay.try_write().is_some());
    #[cfg(not(feature = "tiered-storage"))]
    {
        assert!(store.nodes.try_write().is_some());
        assert!(store.edges.try_write().is_some());
    }
    #[cfg(feature = "tiered-storage")]
    {
        assert!(store.node_versions.try_write().is_some());
        assert!(store.edge_versions.try_write().is_some());
    }
}

#[test]
fn paired_slots_publish_all_targets_without_final_allocator_traffic() {
    let parent = LpgStore::new().unwrap();
    parent.create_graph("child").unwrap();
    let child = parent.graph("child").unwrap();
    let parent_rows = stage(&parent, 10);
    let child_rows = stage(&child, 20);
    // Intentionally opposite the authority's parent-first order.
    let mut slots = StoreDataSlots::new(vec![
        StoreDataSlot::new(&child, workspace()),
        StoreDataSlot::new(&parent, workspace()),
    ]);
    {
        let transitions = [
            parent.pin_exclusive_unframed_transition().unwrap(),
            child.pin_exclusive_unframed_transition().unwrap(),
        ];
        let released = prepare_store_data_slots(&mut slots, &transitions).unwrap();
        assert_writers_released(&parent);
        assert_writers_released(&child);
        allocation::start();
        let ready = released.rebind();
        let bind_counts = allocation::stop();
        let ready = ready.unwrap();
        assert_eq!(ready.batch.slots.slots[0].transition_index, 0);
        assert_eq!(ready.batch.slots.slots[1].transition_index, 1);
        assert!(parent.node_labels.try_read().is_none());
        assert!(child.node_labels.try_read().is_none());
        assert_eq!(parent.current_epoch(), epoch(0));
        assert_eq!(child.current_epoch(), epoch(0));
        allocation::start();
        let installed = ready.install();
        let install_counts = allocation::stop();
        assert!(parent.node_labels.try_read().is_none());
        assert!(child.node_labels.try_read().is_none());
        assert!(parent.tx_property_overlay.try_read().is_none());
        assert!(child.tx_property_overlay.try_read().is_none());
        allocation::start();
        drop(installed);
        let release_counts = allocation::stop();
        assert_eq!(bind_counts, allocation::Counts::default());
        assert_eq!(install_counts, allocation::Counts::default());
        assert_eq!(release_counts, allocation::Counts::default());
        assert_writers_released(&parent);
        assert_writers_released(&child);
        for slot in &slots.slots {
            assert!(slot.guards.is_none());
            assert!(slot.workspace.retired.delta.is_some());
            assert!(slot.workspace.retired.node_deletes.is_some());
            assert!(slot.workspace.retired.edge_deletes.is_some());
        }
    }
    for (store, (changed, deleted, edge), before) in
        [(&parent, parent_rows, 10), (child.as_ref(), child_rows, 20)]
    {
        assert_eq!(store.current_epoch(), epoch(1));
        assert_eq!(store.live_node_count.load(Ordering::Relaxed), 1);
        assert_eq!(store.live_edge_count.load(Ordering::Relaxed), 0);
        assert_eq!(store.edge_type_live_counts.read()[0], 0);
        assert_eq!(store.forward_adj.active_edge_count(), 0);
        assert_eq!(store.forward_adj.total_edge_count(), 1);
        assert_eq!(
            store
                .forward_adj
                .edges_from_including_deleted(changed)
                .len(),
            1
        );
        assert!(store.get_node(deleted).is_none());
        assert!(store.get_node_at_epoch(deleted, epoch(0)).is_some());
        assert!(store.get_edge(edge).is_none());
        assert!(store.get_edge_at_epoch(edge, epoch(0)).is_some());
        assert_eq!(
            store
                .node_properties
                .get_at(changed, &PropertyKey::new("value"), epoch(0)),
            Some(Value::Int64(before))
        );
        assert_eq!(
            store
                .node_properties
                .get_at(changed, &PropertyKey::new("value"), epoch(1)),
            Some(Value::Int64(before + 1))
        );
        assert!(!store.tx_property_overlay.read().contains_key(&tx()));
    }
    allocation::start();
    drop(slots);
    let retired_counts = allocation::stop();
    assert!(retired_counts.dealloc > 0);
}

#[test]
fn paired_slots_later_bookkeeping_contention_releases_the_whole_prefix() {
    let first = LpgStore::new().unwrap();
    let second = LpgStore::new().unwrap();
    let first_rows = stage(&first, 10);
    let second_rows = stage(&second, 20);
    let mut slots = StoreDataSlots::new(vec![
        StoreDataSlot::new(&first, workspace()),
        StoreDataSlot::new(&second, workspace()),
    ]);
    let transitions = [
        first.pin_exclusive_unframed_transition().unwrap(),
        second.pin_exclusive_unframed_transition().unwrap(),
    ];
    let released = prepare_store_data_slots(&mut slots, &transitions).unwrap();
    let blocked = second.pending_tx_edge_deletes.read();
    allocation::start();
    let result = released.rebind();
    let conflict = matches!(result, Err(DataRebindError::Conflict(_)));
    drop(result);
    let counts = allocation::stop();
    drop(blocked);
    assert!(conflict);
    assert_eq!(counts, allocation::Counts::default());
    for (store, changed, value) in [(&first, first_rows.0, 10), (&second, second_rows.0, 20)] {
        assert_writers_released(store);
        assert_eq!(store.current_epoch(), epoch(0));
        assert_eq!(
            store
                .node_properties
                .get(changed, &PropertyKey::new("value")),
            Some(Value::Int64(value))
        );
        assert!(store.tx_property_overlay.read().contains_key(&tx()));
        assert_eq!(store.forward_adj.active_edge_count(), 1);
    }
    assert!(
        slots
            .slots
            .iter()
            .all(|slot| slot.guards.is_none() && slot.workspace.retired.delta.is_none())
    );
}

#[test]
fn paired_slots_later_stale_history_rejection_releases_earlier_store() {
    let first = LpgStore::new().unwrap();
    let second = LpgStore::new().unwrap();
    stage(&first, 10);
    let second_rows = stage(&second, 20);
    let mut slots = StoreDataSlots::new(vec![
        StoreDataSlot::new(&first, workspace()),
        StoreDataSlot::new(&second, workspace()),
    ]);
    let transitions = [
        first.pin_exclusive_unframed_transition().unwrap(),
        second.pin_exclusive_unframed_transition().unwrap(),
    ];
    let released = prepare_store_data_slots(&mut slots, &transitions).unwrap();
    // Test-only violation of a prepared observation. Normal mutation cannot
    // cross either retained transition. Keep the removed payload outside bind.
    let history = second.node_labels.write().remove(&second_rows.0).unwrap();
    allocation::start();
    let result = released.rebind();
    let invalid = matches!(result, Err(DataRebindError::Invalid(_)));
    drop(result);
    let counts = allocation::stop();
    assert!(invalid);
    assert_eq!(counts, allocation::Counts::default());
    assert_writers_released(&first);
    assert_writers_released(&second);
    second.node_labels.write().insert(second_rows.0, history);
    for store in [&first, &second] {
        assert_eq!(store.current_epoch(), epoch(0));
        assert!(store.tx_property_overlay.read().contains_key(&tx()));
    }
}

#[test]
fn paired_slots_late_preparation_failure_retains_earlier_candidates_without_publication() {
    let first = LpgStore::new().unwrap();
    let second = LpgStore::new().unwrap();
    stage(&first, 10);
    let rows = stage(&second, 20);
    second
        .tx_property_overlay
        .write()
        .entry(tx())
        .or_default()
        .node_labels
        .insert((rows.0, u32::MAX), super::super::super::LabelOp::Add);
    let mut slots = StoreDataSlots::new(vec![
        StoreDataSlot::new(&first, workspace()),
        StoreDataSlot::new(&second, workspace()),
    ]);
    let transitions = [
        first.pin_exclusive_unframed_transition().unwrap(),
        second.pin_exclusive_unframed_transition().unwrap(),
    ];
    assert!(prepare_store_data_slots(&mut slots, &transitions).is_err());
    assert_eq!(slots.slots[0].workspace.captured.nodes.len(), 2);
    assert!(
        !slots.slots[0]
            .workspace
            .captured
            .delta
            .node_props
            .is_empty()
    );
    for store in [&first, &second] {
        assert_writers_released(store);
        assert_eq!(store.current_epoch(), epoch(0));
        assert!(store.tx_property_overlay.read().contains_key(&tx()));
    }
    assert!(
        slots
            .slots
            .iter()
            .all(|slot| slot.guards.is_none() && slot.workspace.retired.delta.is_none())
    );
    assert!(prepare_store_data_slots(&mut slots, &transitions).is_err());
}

#[test]
fn paired_slots_abandon_and_forget_drain_all_guards_before_any_slot_retirement() {
    for phase in 0..3 {
        for forgotten in [false, true] {
            let first = LpgStore::new().unwrap();
            let second = LpgStore::new().unwrap();
            stage(&first, 10);
            stage(&second, 20);
            let probes = Cell::new(0usize);
            let mut slots = StoreDataSlots::new(vec![
                StoreDataSlot::new(&first, workspace()),
                StoreDataSlot::new(&second, workspace()),
            ]);
            for slot in &mut slots.slots {
                slot.retirement_probe = Some(RetirementProbe(Box::new(|| {
                    assert_writers_released(&first);
                    assert_writers_released(&second);
                    probes.set(probes.get() + 1);
                })));
            }
            let transitions = [
                first.pin_exclusive_unframed_transition().unwrap(),
                second.pin_exclusive_unframed_transition().unwrap(),
            ];
            let released = prepare_store_data_slots(&mut slots, &transitions).unwrap();
            allocation::start();
            match phase {
                0 => {
                    if forgotten {
                        std::mem::forget(released);
                    } else {
                        drop(released);
                    }
                }
                1 => {
                    let ready = released.rebind().unwrap();
                    if forgotten {
                        std::mem::forget(ready);
                    } else {
                        drop(ready);
                    }
                }
                _ => {
                    let installed = released.rebind().unwrap().install();
                    if forgotten {
                        std::mem::forget(installed);
                    } else {
                        drop(installed);
                    }
                }
            }
            let counts = allocation::stop();
            assert_eq!(counts, allocation::Counts::default());
            assert_eq!(probes.get(), 0);
            for store in [&first, &second] {
                assert_eq!(
                    store.node_labels.try_read().is_none(),
                    forgotten && phase != 0
                );
            }
            drop(transitions);
            drop(slots);
            assert_eq!(probes.get(), 2);
            for store in [&first, &second] {
                assert_writers_released(store);
                assert_eq!(store.current_epoch(), epoch(u64::from(phase == 2)));
                assert_eq!(
                    store.tx_property_overlay.read().contains_key(&tx()),
                    phase != 2
                );
            }
        }
    }
}

#[test]
fn paired_slots_require_exact_nonrepeated_authority() {
    let first = LpgStore::new().unwrap();
    let second = LpgStore::new().unwrap();
    stage(&first, 10);
    let mut missing = StoreDataSlots::new(vec![StoreDataSlot::new(&first, workspace())]);
    let mut duplicate = StoreDataSlots::new(vec![
        StoreDataSlot::new(&first, workspace()),
        StoreDataSlot::new(&first, workspace()),
    ]);
    let transitions = [
        first.pin_exclusive_unframed_transition().unwrap(),
        second.pin_exclusive_unframed_transition().unwrap(),
    ];
    assert!(prepare_store_data_slots(&mut missing, &transitions[1..]).is_err());
    assert!(prepare_store_data_slots(&mut duplicate, &transitions).is_err());
    assert!(!missing.slots[0].workspace.attempted);
    assert!(duplicate.slots.iter().all(|slot| !slot.workspace.attempted));
    assert_writers_released(&first);
    assert_writers_released(&second);
    assert_eq!(first.current_epoch(), epoch(0));
}
