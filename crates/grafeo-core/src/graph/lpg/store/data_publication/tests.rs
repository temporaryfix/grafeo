use super::*;
use crate::graph::lpg::store::LabelOp;
use grafeo_common::types::{PropertyKey, Value};

fn epoch(value: u64) -> EpochId {
    EpochId::new(value)
}
fn tx(value: u64) -> TransactionId {
    TransactionId::new(value)
}

#[test]
fn final_data_rebind_contention_is_static_and_releases_every_acquired_prefix() {
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Row"]);
    store.set_node_property(node, "value", Value::Int64(1));
    store.set_node_property_buffered(node, "value", Value::Int64(2), tx(83));
    macro_rules! rejects {
        ($reader:expr) => {{
            let mut workspace = StoreDataWorkspace::new(tx(83), epoch(0), epoch(1));
            let transition = store.pin_exclusive_unframed_transition().unwrap();
            let released = store
                .prepare_buffered_commit_data(&transition, &mut workspace)
                .unwrap();
            let held = $reader;
            crate::allocation_test::start();
            let result = released.rebind();
            let conflict = matches!(result, Err(DataRebindError::Conflict(_)));
            drop(result);
            let traffic = crate::allocation_test::stop();
            drop(held);
            assert!(conflict, stringify!($reader));
            assert_eq!(
                traffic,
                crate::allocation_test::Counts::default(),
                stringify!($reader)
            );
            assert!(store.node_labels.try_write().is_some());
            assert!(store.label_index.try_write().is_some());
            assert!(store.edge_type_live_counts.try_write().is_some());
            assert!(store.property_undo_log.try_write().is_some());
            assert!(store.pending_tx_creates.try_write().is_some());
            assert!(store.tx_property_overlay.try_write().is_some());
            assert!(store.pending_tx_deletes.try_write().is_some());
            assert!(store.pending_tx_edge_deletes.try_write().is_some());
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
            assert_eq!(store.current_epoch(), epoch(0));
            assert_eq!(
                store.get_node_property(node, &PropertyKey::new("value")),
                Some(Value::Int64(1))
            );
            assert!(store.tx_property_overlay.read().contains_key(&tx(83)));
            assert!(workspace.retired.delta.is_none());
        }};
    }
    #[cfg(not(feature = "tiered-storage"))]
    {
        rejects!(store.nodes.read());
        rejects!(store.edges.read());
    }
    #[cfg(feature = "tiered-storage")]
    {
        rejects!(store.node_versions.read());
        rejects!(store.edge_versions.read());
    }
    rejects!(store.label_index.read());
    rejects!(store.node_labels.read());
    rejects!(store.edge_type_live_counts.read());
    rejects!(store.property_undo_log.read());
    rejects!(store.pending_tx_creates.read());
    rejects!(store.tx_property_overlay.read());
    #[cfg(feature = "text-index")]
    rejects!(store.text_index_overlay.read());
    rejects!(store.pending_tx_deletes.read());
    rejects!(store.pending_tx_edge_deletes.read());
}

#[test]
fn buffered_data_prepares_without_draining_then_publishes_only_own_final_rows() {
    let store = LpgStore::new().unwrap();
    let existing = store.create_node(&["Old"]);
    store.set_node_property(existing, "value", Value::Int64(1));
    let foreign = store.create_node_versioned(&["Foreign"], epoch(0), tx(20));
    store.set_node_property_buffered(foreign, "value", Value::Int64(99), tx(20));
    let created = store.create_node_versioned(&["New"], epoch(0), tx(10));
    store.set_node_property_buffered(created, "value", Value::Int64(3), tx(10));
    store.set_node_property_buffered(existing, "value", Value::Int64(2), tx(10));
    store.remove_label_buffered(existing, "Old", tx(10));
    store.add_label_buffered(existing, "Final", tx(10));
    let foreign_labels = store.node_labels.read().get(&foreign).cloned().unwrap();
    let mut workspace = StoreDataWorkspace::new(tx(10), epoch(0), epoch(1));
    {
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        let released = store
            .prepare_buffered_commit_data(&transition, &mut workspace)
            .unwrap();
        assert_eq!(
            store
                .node_properties
                .get(existing, &PropertyKey::new("value")),
            Some(Value::Int64(1))
        );
        assert!(store.tx_property_overlay.read().contains_key(&tx(10)));
        assert!(store.pending_tx_creates.read().contains_key(&tx(10)));
        let ready = released.rebind().unwrap();
        assert!(store.node_labels.try_read().is_none());
        let installed = ready.install();
        assert!(store.node_labels.try_read().is_none());
        assert!(store.tx_property_overlay.try_read().is_none());
        drop(installed);
    }
    assert_eq!(
        store
            .node_properties
            .get_at(existing, &PropertyKey::new("value"), epoch(0)),
        Some(Value::Int64(1))
    );
    assert_eq!(
        store
            .node_properties
            .get_at(existing, &PropertyKey::new("value"), epoch(1)),
        Some(Value::Int64(2))
    );
    assert_eq!(
        store
            .get_node(created)
            .unwrap()
            .properties
            .get(&PropertyKey::new("value")),
        Some(&Value::Int64(3))
    );
    assert!(store.get_node(foreign).is_none());
    assert_eq!(
        store.node_labels.read().get(&foreign).unwrap().history(),
        foreign_labels.history()
    );
    assert_eq!(store.pending_node_creates(tx(20)), vec![foreign]);
    assert!(store.tx_property_overlay.read().contains_key(&tx(20)));
    assert!(!store.tx_property_overlay.read().contains_key(&tx(10)));
    assert!(workspace.retired.delta.is_some());
    assert!(workspace.retired.creates.is_some());
    let labels = &store.get_node(existing).unwrap().labels;
    assert_eq!(labels.len(), 1);
    assert_eq!(labels[0].as_str(), "Final");
}

#[test]
fn buffered_zero_width_creation_retains_final_history_without_live_membership() {
    let store = LpgStore::new().unwrap();
    let anchor = store.create_node(&["Anchor"]);
    store.create_property_index("value");
    let foreign = store.create_node_versioned(&["Foreign"], epoch(0), tx(20));
    let foreign_edge = store.create_edge_versioned(foreign, anchor, "FOREIGN", epoch(0), tx(20));
    assert!(foreign_edge.is_valid());
    let value = PropertyKey::new("value");
    store
        .node_properties
        .set(foreign, value.clone(), Value::Int64(90), EpochId::PENDING);
    store.edge_properties.set(
        foreign_edge,
        value.clone(),
        Value::Int64(91),
        EpochId::PENDING,
    );
    store.set_node_property_buffered(foreign, "value", Value::Int64(99), tx(20));
    store.set_edge_property_buffered(foreign_edge, "value", Value::Int64(100), tx(20));
    store.add_label_buffered(foreign, "ForeignPending", tx(20));
    let foreign_labels = store.node_label_history(foreign);
    let foreign_node_history = store.node_property_history(foreign);
    let foreign_edge_history = store.edge_property_history(foreign_edge);
    let before_nodes = store.live_node_count.load(Ordering::Relaxed);
    let before_edges = store.live_edge_count.load(Ordering::Relaxed);

    let ephemeral = store.create_node_versioned(&["Draft"], epoch(0), tx(10));
    let edge = store.create_edge_versioned(ephemeral, anchor, "ZERO", epoch(0), tx(10));
    assert!(edge.is_valid());
    store.add_label_buffered(ephemeral, "Reviewed", tx(10));
    store.add_label_buffered(ephemeral, "Removed", tx(10));
    store.remove_label_buffered(ephemeral, "Removed", tx(10));
    store.set_node_property_buffered(ephemeral, "value", Value::Int64(1), tx(10));
    store.set_node_property_buffered(ephemeral, "value", Value::Int64(3), tx(10));
    store.set_edge_property_buffered(edge, "value", Value::Int64(5), tx(10));
    store.set_edge_property_buffered(edge, "value", Value::Int64(7), tx(10));
    for (key, final_null) in [("removed", false), ("set_then_null", true)] {
        store.set_node_property_buffered(ephemeral, key, Value::Int64(8), tx(10));
        store.set_edge_property_buffered(edge, key, Value::Int64(8), tx(10));
        if final_null {
            store.set_node_property_buffered(ephemeral, key, Value::Null, tx(10));
            store.set_edge_property_buffered(edge, key, Value::Null, tx(10));
        } else {
            store.remove_node_property_buffered(ephemeral, key, tx(10));
            store.remove_edge_property_buffered(edge, key, tx(10));
        }
    }
    store.set_node_property_buffered(ephemeral, "null_only", Value::Null, tx(10));
    store.set_edge_property_buffered(edge, "null_only", Value::Null, tx(10));
    assert!(store.delete_edge_transactional(edge, epoch(0), tx(10)));
    assert!(store.delete_node_transactional(ephemeral, epoch(0), tx(10)));
    let empty = store.create_node_versioned(&["Unchanged"], epoch(0), tx(10));
    assert!(store.delete_node_transactional(empty, epoch(0), tx(10)));
    let mut workspace = StoreDataWorkspace::new(tx(10), epoch(0), epoch(1));
    {
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        let released = store
            .prepare_buffered_commit_data(&transition, &mut workspace)
            .unwrap();
        assert!(store.node_property_history(ephemeral).is_empty());
        assert!(store.edge_property_history(edge).is_empty());
        assert_eq!(store.current_epoch(), epoch(0));
        drop(released.rebind().unwrap().install());
    }

    assert_eq!(
        store.node_property_history(ephemeral),
        vec![(
            value.clone(),
            vec![(epoch(1), Value::Int64(3)), (epoch(1), Value::Null)]
        )]
    );
    assert_eq!(
        store.edge_property_history(edge),
        vec![(
            value.clone(),
            vec![(epoch(1), Value::Int64(7)), (epoch(1), Value::Null)]
        )]
    );
    assert_eq!(
        store.node_label_history(ephemeral),
        vec![
            (epoch(1), vec!["Draft".into()]),
            (epoch(1), vec!["Draft".into(), "Reviewed".into()]),
        ]
    );
    assert_eq!(
        store.node_label_history(empty),
        vec![(epoch(1), vec![arcstr::ArcStr::from("Unchanged")])]
    );
    assert!(store.node_property_history(empty).is_empty());
    assert!(store.nodes_by_label("Unchanged").is_empty());
    for cut in [epoch(0), epoch(1), epoch(2)] {
        assert!(store.get_node_at_epoch(ephemeral, cut).is_none());
        assert!(store.get_edge_at_epoch(edge, cut).is_none());
        assert_eq!(store.node_properties.get_at(ephemeral, &value, cut), None);
        assert_eq!(store.edge_properties.get_at(edge, &value, cut), None);
    }
    assert_eq!(
        store
            .get_node_history(ephemeral)
            .into_iter()
            .map(|(created, deleted, _)| (created, deleted))
            .collect::<Vec<_>>(),
        vec![(epoch(1), Some(epoch(1)))]
    );
    assert_eq!(
        store
            .get_edge_history(edge)
            .into_iter()
            .map(|(created, deleted, _)| (created, deleted))
            .collect::<Vec<_>>(),
        vec![(epoch(1), Some(epoch(1)))]
    );
    for label in ["Draft", "Reviewed", "Removed"] {
        assert!(store.nodes_by_label(label).is_empty());
    }
    assert!(
        store
            .find_nodes_by_property("value", &Value::Int64(3))
            .is_empty()
    );
    assert_eq!(store.node_count(), 1);
    assert_eq!(store.edge_count(), 0);
    assert_eq!(store.live_node_count.load(Ordering::Relaxed), before_nodes);
    assert_eq!(store.live_edge_count.load(Ordering::Relaxed), before_edges);
    assert_eq!(store.edge_type_live_counts.read().last(), Some(&0));
    assert_eq!(store.node_label_history(foreign), foreign_labels);
    assert_eq!(store.node_property_history(foreign), foreign_node_history);
    assert_eq!(
        store.edge_property_history(foreign_edge),
        foreign_edge_history
    );
    assert_eq!(store.pending_node_creates(tx(20)), vec![foreign]);
    assert_eq!(store.pending_edge_creates(tx(20)), vec![foreign_edge]);
    assert!(store.get_node(foreign).is_none());
    assert!(store.get_edge(foreign_edge).is_none());
    let pending_label = store.get_or_create_label_id("ForeignPending");
    let overlays = store.tx_property_overlay.read();
    let foreign_delta = &overlays[&tx(20)];
    assert!(matches!(
        foreign_delta.node_props.get(&(foreign, value.clone())),
        Some(PropOp::Set(Value::Int64(99)))
    ));
    assert!(matches!(
        foreign_delta.edge_props.get(&(foreign_edge, value)),
        Some(PropOp::Set(Value::Int64(100)))
    ));
    assert_eq!(
        foreign_delta.node_labels.get(&(foreign, pending_label)),
        Some(&LabelOp::Add)
    );
    assert!(!overlays.contains_key(&tx(10)));
}

#[test]
fn buffered_data_deletion_dominates_sets_and_deduplicates_adjacency_counts() {
    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    let edge = store.create_edge(src, dst, "LINK");
    store.set_node_property(src, "value", Value::Int64(1));
    store.set_edge_property(edge, "value", Value::Int64(2));
    store.set_node_property_buffered(src, "value", Value::Int64(8), tx(10));
    store.set_edge_property_buffered(edge, "value", Value::Int64(9), tx(10));
    assert!(store.delete_edge_transactional(edge, epoch(0), tx(10)));
    assert!(store.delete_node_transactional(src, epoch(0), tx(10)));
    store
        .pending_tx_deletes
        .write()
        .get_mut(&tx(10))
        .unwrap()
        .push(src);
    store
        .pending_tx_edge_deletes
        .write()
        .get_mut(&tx(10))
        .unwrap()
        .push((src, edge, dst));
    let mut workspace = StoreDataWorkspace::new(tx(10), epoch(0), epoch(1));
    {
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        let ready = store
            .prepare_buffered_commit_data(&transition, &mut workspace)
            .unwrap()
            .rebind()
            .unwrap();
        drop(ready.install());
    }
    assert!(store.get_node(src).is_none());
    assert!(store.get_edge(edge).is_none());
    assert!(store.get_node_at_epoch(src, epoch(0)).is_some());
    assert!(store.get_edge_at_epoch(edge, epoch(0)).is_some());
    assert_eq!(
        store.node_properties.get(src, &PropertyKey::new("value")),
        None
    );
    assert_eq!(
        store.edge_properties.get(edge, &PropertyKey::new("value")),
        None
    );
    assert_eq!(store.live_node_count.load(Ordering::Relaxed), 1);
    assert_eq!(store.live_edge_count.load(Ordering::Relaxed), 0);
    assert_eq!(store.edge_type_live_counts.read()[0], 0);
    assert_eq!(workspace.captured.deleted_nodes, vec![src]);
    assert_eq!(workspace.captured.deleted_edges, vec![(src, edge, dst)]);
}

#[test]
fn buffered_data_abandon_preserves_every_pending_input_and_committed_row() {
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Old"]);
    store.set_node_property(node, "value", Value::Int64(1));
    store.set_node_property_buffered(node, "value", Value::Int64(2), tx(10));
    store.add_label_buffered(node, "New", tx(10));
    let mut workspace = StoreDataWorkspace::new(tx(10), epoch(0), epoch(1));
    {
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        drop(
            store
                .prepare_buffered_commit_data(&transition, &mut workspace)
                .unwrap()
                .rebind()
                .unwrap(),
        );
        assert!(
            store
                .prepare_buffered_commit_data(&transition, &mut workspace)
                .is_err()
        );
    }
    assert_eq!(store.current_epoch(), epoch(0));
    assert_eq!(
        store.get_node_property(node, &PropertyKey::new("value")),
        Some(Value::Int64(1))
    );
    assert!(store.tx_property_overlay.read().contains_key(&tx(10)));
    assert!(workspace.retired.delta.is_none());
}

#[test]
fn buffered_data_rejects_ambiguous_write_through_before_finalization() {
    let store = LpgStore::new().unwrap();
    let node = store.create_node_with_props_versioned(
        &["New"],
        [("value", Value::Int64(1))],
        epoch(0),
        tx(10),
    );
    let mut workspace = StoreDataWorkspace::new(tx(10), epoch(0), epoch(1));
    {
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        assert!(
            store
                .prepare_buffered_commit_data(&transition, &mut workspace)
                .is_err()
        );
    }
    assert!(store.get_node(node).is_none());
    assert_eq!(store.pending_node_creates(tx(10)), vec![node]);
    assert_eq!(store.current_epoch(), epoch(0));
}

#[test]
fn buffered_data_preserves_foreign_pending_property_suffix_in_same_cell() {
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Node"]);
    let key = PropertyKey::new("value");
    store.set_node_property(node, "value", Value::Int64(1));
    store
        .node_properties
        .set(node, key.clone(), Value::Int64(99), EpochId::PENDING);
    store.set_node_property_buffered(node, "value", Value::Int64(2), tx(10));
    let mut workspace = StoreDataWorkspace::new(tx(10), epoch(0), epoch(1));
    {
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        drop(
            store
                .prepare_buffered_commit_data(&transition, &mut workspace)
                .unwrap()
                .rebind()
                .unwrap()
                .install(),
        );
    }
    assert_eq!(
        store.node_properties.get_history(node, &key),
        vec![
            (epoch(0), Value::Int64(1)),
            (epoch(1), Value::Int64(2)),
            (EpochId::PENDING, Value::Int64(99)),
        ]
    );
}

#[test]
fn buffered_data_rejects_foreign_transition_and_invalid_context() {
    let store = LpgStore::new().unwrap();
    let foreign = LpgStore::new().unwrap();
    let mut workspace = StoreDataWorkspace::new(tx(10), epoch(0), epoch(1));
    {
        let transition = foreign.pin_exclusive_unframed_transition().unwrap();
        assert!(
            store
                .prepare_buffered_commit_data(&transition, &mut workspace)
                .is_err()
        );
    }
    for (transaction, publication, commit) in [
        (TransactionId::SYSTEM, epoch(0), epoch(1)),
        (TransactionId::INVALID, epoch(0), epoch(1)),
        (tx(10), epoch(1), epoch(1)),
        (tx(10), epoch(0), EpochId::PENDING),
    ] {
        let mut workspace = StoreDataWorkspace::new(transaction, publication, commit);
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        assert!(
            store
                .prepare_buffered_commit_data(&transition, &mut workspace)
                .is_err()
        );
    }
}

#[test]
fn buffered_data_rebind_and_install_have_zero_allocator_traffic() {
    use crate::allocation_test as allocation;
    allocation::start();
    let mut control = Vec::<u8>::with_capacity(17);
    control.extend_from_slice(&[1; 17]);
    control.reserve(1024);
    std::hint::black_box(&control);
    let zeroed = vec![0_u8; std::hint::black_box(8192)];
    std::hint::black_box(&zeroed);
    drop(control);
    drop(zeroed);
    let positive = allocation::stop();
    assert!(
        positive.alloc > 0 && positive.zeroed > 0 && positive.realloc > 0 && positive.dealloc > 0
    );

    let store = LpgStore::new().unwrap();
    let changed = store.create_node(&["Before"]);
    let deleted = store.create_node(&["Deleted"]);
    let edge = store.create_edge(changed, deleted, "LINK");
    store.set_node_property(changed, "existing", Value::Int64(1));
    store.set_node_property(deleted, "existing", Value::Int64(2));
    store.set_edge_property(edge, "value", Value::Int64(3));
    store.set_node_property_buffered(changed, "existing", Value::Int64(4), tx(10));
    store.set_node_property_buffered(changed, "new", Value::Int64(5), tx(10));
    store.remove_label_buffered(changed, "Before", tx(10));
    store.add_label_buffered(changed, "After", tx(10));
    assert!(store.delete_edge_transactional(edge, epoch(0), tx(10)));
    assert!(store.delete_node_transactional(deleted, epoch(0), tx(10)));
    let created = store.create_node_versioned(&["Created"], epoch(0), tx(10));
    store.set_node_property_buffered(created, "new", Value::Int64(6), tx(10));
    let ephemeral = store.create_node_versioned(&["Draft"], epoch(0), tx(10));
    let ephemeral_edge = store.create_edge_versioned(ephemeral, created, "ZERO", epoch(0), tx(10));
    assert!(ephemeral_edge.is_valid());
    store.set_node_property_buffered(ephemeral, "new", Value::Int64(7), tx(10));
    store.set_edge_property_buffered(ephemeral_edge, "value", Value::Int64(8), tx(10));
    store.add_label_buffered(ephemeral, "Reviewed", tx(10));
    assert!(store.delete_edge_transactional(ephemeral_edge, epoch(0), tx(10)));
    assert!(store.delete_node_transactional(ephemeral, epoch(0), tx(10)));
    let mut workspace = StoreDataWorkspace::new(tx(10), epoch(0), epoch(1));
    let (rebound, installed_counts, released_fences) = {
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        let released = store
            .prepare_buffered_commit_data(&transition, &mut workspace)
            .unwrap();
        allocation::start();
        let ready = std::hint::black_box(released).rebind();
        let rebound = allocation::stop();
        let ready = ready.unwrap();
        allocation::start();
        let installed = std::hint::black_box(ready).install();
        let installed_counts = allocation::stop();
        allocation::start();
        drop(installed);
        let released_fences = allocation::stop();
        (rebound, installed_counts, released_fences)
    };
    assert_eq!(rebound, allocation::Counts::default());
    assert_eq!(installed_counts, allocation::Counts::default());
    assert_eq!(released_fences, allocation::Counts::default());
    // The transition is now gone. Private/displaced allocations retire with
    // the workspace, not when install returns or its reader fences drain.
    allocation::start();
    drop(workspace);
    let retired = allocation::stop();
    assert!(retired.dealloc > 0);
    assert!(store.get_node(created).is_some());
    assert!(store.get_node(deleted).is_none());
    assert_eq!(
        store.get_node_property(changed, &PropertyKey::new("new")),
        Some(Value::Int64(5))
    );
    assert_eq!(
        store.node_property_history_for_key(ephemeral, "new"),
        vec![(epoch(1), Value::Int64(7)), (epoch(1), Value::Null)]
    );
    assert_eq!(
        store.edge_property_history(ephemeral_edge),
        vec![(
            PropertyKey::new("value"),
            vec![(epoch(1), Value::Int64(8)), (epoch(1), Value::Null)]
        )]
    );
    assert_eq!(
        store.node_label_history(ephemeral),
        vec![
            (epoch(1), vec!["Draft".into()]),
            (epoch(1), vec!["Draft".into(), "Reviewed".into()]),
        ]
    );
}

#[test]
fn buffered_data_final_rebind_failure_does_not_allocate_under_entity_writers() {
    use crate::allocation_test as allocation;
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Before"]);
    store.add_label_buffered(node, "After", tx(10));
    let mut workspace = StoreDataWorkspace::new(tx(10), epoch(0), epoch(1));
    let observed = {
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        let released = store
            .prepare_buffered_commit_data(&transition, &mut workspace)
            .unwrap();
        // Test-only corruption of a prequalified input: normal writers cannot
        // cross the retained transition. This rejects after entity acquisition.
        store.node_labels.write().remove(&node);
        allocation::start();
        let result = released.rebind();
        let observed = allocation::stop();
        assert!(result.is_err());
        drop(result);
        observed
    };
    assert_eq!(observed, allocation::Counts::default());
    assert!(store.tx_property_overlay.read().contains_key(&tx(10)));
    assert_eq!(store.current_epoch(), epoch(0));
}

#[test]
fn buffered_data_late_store_rebind_keeps_errors_unallocated_under_earlier_fences() {
    use crate::allocation_test as allocation;
    let first = LpgStore::new().unwrap();
    let second = LpgStore::new().unwrap();
    let a = first.create_node(&["Before"]);
    let b = second.create_node(&["Before"]);
    first.add_label_buffered(a, "After", tx(10));
    second.add_label_buffered(b, "After", tx(10));
    let mut first_workspace = StoreDataWorkspace::new(tx(10), epoch(0), epoch(1));
    let mut second_workspace = StoreDataWorkspace::new(tx(10), epoch(0), epoch(1));
    let observed = {
        let first_transition = first.pin_exclusive_unframed_transition().unwrap();
        let second_transition = second.pin_exclusive_unframed_transition().unwrap();
        let first_released = first
            .prepare_buffered_commit_data(&first_transition, &mut first_workspace)
            .unwrap();
        let second_released = second
            .prepare_buffered_commit_data(&second_transition, &mut second_workspace)
            .unwrap();
        // Deliberately violate only the second store's private prepared input.
        second.node_labels.write().remove(&b);
        let first_ready = first_released.rebind().unwrap();
        allocation::start();
        let result = second_released.rebind();
        let observed = allocation::stop();
        assert!(result.is_err());
        assert!(first.node_labels.try_read().is_none());
        drop(result);
        drop(first_ready);
        observed
    };
    assert_eq!(observed, allocation::Counts::default());
    assert_eq!(first.current_epoch(), epoch(0));
    assert!(first.tx_property_overlay.read().contains_key(&tx(10)));
    assert!(second.tx_property_overlay.read().contains_key(&tx(10)));
}

#[test]
fn buffered_data_property_limit_checks_final_count_not_operation_order() {
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Node"]);
    for index in 0..u16::MAX {
        store.node_properties.set(
            node,
            PropertyKey::new(format!("key-{index}")),
            Value::Int64(1),
            epoch(0),
        );
    }
    store.remove_node_property_buffered(node, "key-0", tx(10));
    store.set_node_property_buffered(node, "replacement", Value::Int64(2), tx(10));
    let mut workspace = StoreDataWorkspace::new(tx(10), epoch(0), epoch(1));
    {
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        drop(
            store
                .prepare_buffered_commit_data(&transition, &mut workspace)
                .unwrap()
                .rebind()
                .unwrap()
                .install(),
        );
    }
    assert_eq!(
        store
            .node_properties
            .commit_property_count(node, epoch(1))
            .unwrap(),
        usize::from(u16::MAX)
    );
    assert_eq!(
        store.get_node_property(node, &PropertyKey::new("key-0")),
        None
    );
    assert_eq!(
        store.get_node_property(node, &PropertyKey::new("replacement")),
        Some(Value::Int64(2))
    );
}

#[cfg(feature = "tiered-storage")]
#[test]
fn buffered_data_rejects_frozen_pending_creates_that_hot_finalization_cannot_stamp() {
    let store = LpgStore::new().unwrap();
    let node = store.create_node_versioned(&["Pending"], epoch(0), tx(10));
    assert_eq!(store.freeze_epoch(EpochId::PENDING), 1);
    let mut workspace = StoreDataWorkspace::new(tx(10), epoch(0), epoch(1));
    {
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        assert!(
            store
                .prepare_buffered_commit_data(&transition, &mut workspace)
                .is_err()
        );
    }
    assert_eq!(store.pending_node_creates(tx(10)), vec![node]);
    assert_eq!(
        store.node_labels.read()[&node].latest_epoch(),
        Some(EpochId::PENDING)
    );
    assert_eq!(store.current_epoch(), epoch(0));
}
