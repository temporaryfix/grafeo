use super::*;
use crate::graph::lpg::LpgStore;
use crate::graph::lpg::property::{CompareOp, CompressionMode};
use grafeo_common::types::{EdgeId, NodeId};

fn id(value: u64) -> NodeId {
    NodeId::new(value)
}
fn epoch(value: u64) -> EpochId {
    EpochId::new(value)
}
fn key(value: &str) -> PropertyKey {
    PropertyKey::new(value)
}
fn set(node: u64, name: &str, value: i64) -> (NodeId, PropertyKey, Option<Value>) {
    (id(node), key(name), Some(Value::Int64(value)))
}
fn remove(node: u64, name: &str) -> (NodeId, PropertyKey, Option<Value>) {
    (id(node), key(name), None)
}

#[test]
fn sparse_zero_width_commit_rejects_existing_history_and_invalid_intersections() {
    for prior_epoch in [epoch(1), EpochId::PENDING] {
        let storage = PropertyStorage::<NodeId>::new();
        storage.set(id(1), key("value"), Value::Int64(5), prior_epoch);
        let before = storage.get_history(id(1), &key("value"));
        let mut workspace =
            PropertyCommitWorkspace::new(vec![id(1)], vec![set(1, "value", 10)], vec![id(1)]);
        let result = storage.prepare_commit_data(epoch(2), epoch(3), &mut workspace);
        assert!(matches!(
            result,
            Err(Error::Transaction(TransactionError::InvalidState(ref message)))
                if message.contains("zero-width creation already has property history")
        ));
        drop(result);
        assert_eq!(storage.get_history(id(1), &key("value")), before);
    }

    for (created_and_deleted, deleted) in [
        (vec![id(2), id(1)], vec![id(1), id(2)]),
        (vec![id(1), id(1)], vec![id(1)]),
        (vec![id(1)], vec![]),
    ] {
        let storage = PropertyStorage::<NodeId>::new();
        let mut workspace =
            PropertyCommitWorkspace::new(created_and_deleted, vec![set(1, "value", 10)], deleted);
        let result = storage.prepare_commit_data(epoch(2), epoch(3), &mut workspace);
        assert!(matches!(
            result,
            Err(Error::Transaction(TransactionError::InvalidState(ref message)))
                if message.contains("sorted unique deletion members")
        ));
        drop(result);
        assert!(storage.keys().is_empty());
    }
}

#[test]
fn sparse_commit_merges_new_and_existing_columns_and_cells() {
    let storage = PropertyStorage::<NodeId>::new();
    storage.set(id(1), key("existing"), Value::Int64(10), epoch(1));
    storage.set(id(2), key("existing"), Value::Int64(20), epoch(1));
    storage.set(id(99), key("unrelated"), Value::Int64(99), epoch(1));
    let unrelated_history = storage.get_history(id(99), &key("unrelated"));
    let mut workspace = PropertyCommitWorkspace::new(
        Vec::new(),
        vec![
            set(1, "existing", 11),
            set(3, "existing", 30),
            set(4, "new", 40),
        ],
        vec![],
    );
    let ready = storage
        .prepare_commit_data(epoch(2), epoch(3), &mut workspace)
        .unwrap();
    assert!(storage.columns.try_read().is_none());
    let installed = ready.install();
    assert!(storage.columns.try_read().is_none());
    drop(installed);
    assert_eq!(
        storage.get_at(id(1), &key("existing"), epoch(2)),
        Some(Value::Int64(10))
    );
    assert_eq!(storage.get(id(1), &key("existing")), Some(Value::Int64(11)));
    assert_eq!(storage.get(id(2), &key("existing")), Some(Value::Int64(20)));
    assert_eq!(storage.get(id(3), &key("existing")), Some(Value::Int64(30)));
    assert_eq!(storage.get(id(4), &key("new")), Some(Value::Int64(40)));
    assert_eq!(
        storage.get_history(id(99), &key("unrelated")),
        unrelated_history
    );
    assert_eq!(workspace.retired_logs.len(), 1);
    let existing = workspace
        .fragments
        .iter()
        .find(|fragment| fragment.key == key("existing"))
        .unwrap();
    let emptied = existing
        .candidate
        .as_ref()
        .expect("occupied fragment stays allocated");
    assert!(emptied.values.is_empty());
    assert!(emptied.values.capacity() >= 2);
    let added = workspace
        .fragments
        .iter()
        .find(|fragment| fragment.key == key("new"))
        .unwrap();
    assert!(
        added.candidate.is_none(),
        "new column moves its complete fragment"
    );
}

#[test]
fn sparse_commit_does_not_clone_other_histories_in_the_touched_column() {
    let storage = PropertyStorage::<NodeId>::new();
    let untouched: std::sync::Arc<[u8]> = std::sync::Arc::from([1u8, 2, 3]);
    storage.set(id(1), key("value"), Value::Int64(10), epoch(1));
    storage.set(
        id(99),
        key("value"),
        Value::Bytes(std::sync::Arc::clone(&untouched)),
        epoch(1),
    );
    let mut workspace = PropertyCommitWorkspace::new(Vec::new(), vec![set(1, "value", 20)], vec![]);
    assert_eq!(std::sync::Arc::strong_count(&untouched), 2);
    let ready = storage
        .prepare_commit_data(epoch(2), epoch(3), &mut workspace)
        .unwrap();
    assert_eq!(
        std::sync::Arc::strong_count(&untouched),
        2,
        "no clone of an untouched row's history"
    );
    drop(ready.install());
    assert_eq!(std::sync::Arc::strong_count(&untouched), 2);
    assert_eq!(workspace.fragments.len(), 1);
    assert_eq!(workspace.retired_logs.len(), 1);
}

#[test]
fn sparse_commit_normalizes_final_values_and_deletion_dominates() {
    let storage = PropertyStorage::<NodeId>::new();
    for node in [1, 2, 3, 4] {
        storage.set(id(node), key("value"), Value::Int64(5), epoch(1));
    }
    storage.set(id(3), key("other"), Value::Int64(8), epoch(1));
    let mut workspace = PropertyCommitWorkspace::new(
        Vec::new(),
        vec![
            set(1, "value", 8),
            remove(1, "value"),
            remove(2, "value"),
            set(2, "value", 9),
            set(3, "value", 100),
            set(3, "never_published", 101),
            set(4, "value", 6),
            set(4, "value", 7),
        ],
        vec![id(3), id(3)],
    );
    drop(
        storage
            .prepare_commit_data(epoch(2), epoch(3), &mut workspace)
            .unwrap()
            .install(),
    );
    assert_eq!(storage.get(id(1), &key("value")), None);
    assert_eq!(storage.get(id(2), &key("value")), Some(Value::Int64(9)));
    assert_eq!(storage.get(id(3), &key("value")), None);
    assert_eq!(storage.get(id(3), &key("other")), None);
    assert!(!storage.keys().contains(&key("never_published")));
    assert_eq!(
        storage.get_history(id(4), &key("value")),
        vec![(epoch(1), Value::Int64(5)), (epoch(3), Value::Int64(7)),]
    );
    assert_eq!(storage.get_history(id(1), &key("value")).len(), 2);
    assert_eq!(storage.get_history(id(3), &key("other")).len(), 2);
}

#[test]
fn sparse_commit_empty_removals_do_not_create_columns_or_histories() {
    let storage = PropertyStorage::<NodeId>::new();
    storage.set(id(1), key("tombstone"), Value::Null, epoch(1));
    storage.set(
        id(2),
        key("pending_only"),
        Value::Int64(80),
        EpochId::PENDING,
    );
    let before_pending = storage.get_history(id(2), &key("pending_only"));
    let mut workspace = PropertyCommitWorkspace::new(
        Vec::new(),
        vec![
            remove(1, "tombstone"),
            remove(2, "pending_only"),
            remove(3, "absent"),
            (id(4), key("explicit_null"), Some(Value::Null)),
        ],
        vec![id(2)],
    );
    drop(
        storage
            .prepare_commit_data(epoch(2), epoch(3), &mut workspace)
            .unwrap()
            .install(),
    );
    assert_eq!(
        storage.get_history(id(1), &key("tombstone")),
        vec![(epoch(1), Value::Null)]
    );
    assert_eq!(
        storage.get_history(id(2), &key("pending_only")),
        before_pending
    );
    assert!(!storage.keys().contains(&key("absent")));
    assert_eq!(
        storage.get_history(id(4), &key("explicit_null")),
        vec![(epoch(3), Value::Null)]
    );
    assert!(workspace.retired_logs.is_empty());
}

#[test]
fn sparse_commit_preserves_foreign_pending_tails_even_on_the_same_cell() {
    let storage = PropertyStorage::<NodeId>::new();
    storage.set(id(1), key("set"), Value::Int64(1), epoch(1));
    storage.set(id(1), key("set"), Value::Int64(80), EpochId::PENDING);
    storage.set(id(1), key("set"), Value::Int64(81), EpochId::PENDING);
    storage.set(id(2), key("remove"), Value::Int64(2), epoch(1));
    storage.set(id(2), key("remove"), Value::Null, EpochId::PENDING);
    storage.set(id(3), key("untouched"), Value::Int64(30), EpochId::PENDING);
    let untouched = storage.get_history(id(3), &key("untouched"));
    let mut workspace = PropertyCommitWorkspace::new(
        Vec::new(),
        vec![set(1, "set", 10), remove(2, "remove")],
        vec![],
    );
    drop(
        storage
            .prepare_commit_data(epoch(2), epoch(3), &mut workspace)
            .unwrap()
            .install(),
    );
    assert_eq!(
        storage.get_history(id(1), &key("set")),
        vec![
            (epoch(1), Value::Int64(1)),
            (epoch(3), Value::Int64(10)),
            (EpochId::PENDING, Value::Int64(80)),
            (EpochId::PENDING, Value::Int64(81)),
        ]
    );
    assert_eq!(
        storage.get_history(id(2), &key("remove")),
        vec![
            (epoch(1), Value::Int64(2)),
            (epoch(3), Value::Null),
            (EpochId::PENDING, Value::Null),
        ]
    );
    assert_eq!(storage.get_history(id(3), &key("untouched")), untouched);
    assert_eq!(
        storage.get_at(id(1), &key("set"), epoch(3)),
        Some(Value::Int64(10))
    );
    assert_eq!(storage.get_at(id(2), &key("remove"), epoch(3)), None);
}

#[test]
fn sparse_commit_rejects_invalid_epochs_future_history_and_workspace_reuse() {
    let storage = PropertyStorage::<NodeId>::new();
    for (publication, commit) in [
        (EpochId::PENDING, epoch(3)),
        (epoch(2), EpochId::PENDING),
        (epoch(2), epoch(2)),
        (epoch(3), epoch(2)),
    ] {
        let mut workspace =
            PropertyCommitWorkspace::new(Vec::new(), vec![set(1, "value", 8)], vec![]);
        assert!(matches!(
            storage.prepare_commit_data(publication, commit, &mut workspace),
            Err(Error::Transaction(TransactionError::InvalidState(_)))
        ));
        assert!(
            storage
                .prepare_commit_data(epoch(2), epoch(3), &mut workspace)
                .is_err()
        );
    }
    storage.set(id(1), key("future"), Value::Int64(9), epoch(4));
    let before = storage.get_history(id(1), &key("future"));
    for deleting in [false, true] {
        let mut workspace = PropertyCommitWorkspace::new(
            Vec::new(),
            if deleting {
                vec![]
            } else {
                vec![set(1, "future", 10)]
            },
            if deleting { vec![id(1)] } else { vec![] },
        );
        assert!(
            storage
                .prepare_commit_data(epoch(2), epoch(3), &mut workspace)
                .is_err()
        );
        assert_eq!(storage.get_history(id(1), &key("future")), before);
    }
    let mut workspace = PropertyCommitWorkspace::new(Vec::new(), vec![set(2, "safe", 10)], vec![]);
    drop(
        storage
            .prepare_commit_data(epoch(4), epoch(5), &mut workspace)
            .unwrap(),
    );
    assert!(
        storage
            .prepare_commit_data(epoch(4), epoch(5), &mut workspace)
            .is_err()
    );
    assert_eq!(storage.get(id(2), &key("safe")), None);
}

#[test]
fn sparse_commit_keeps_compression_metadata_and_retires_block_cache() {
    let storage = PropertyStorage::<NodeId>::new();
    storage.set(id(1), key("value"), Value::Int64(1), epoch(1));
    {
        let mut columns = storage.columns.write();
        let column = columns.get_mut(&key("value")).unwrap();
        column.compression_mode = CompressionMode::Auto;
        column.block_zone_maps.push(ZoneMapEntry::new());
    }
    let mut workspace =
        PropertyCommitWorkspace::new(Vec::new(), vec![set(1, "value", 1000)], vec![]);
    drop(
        storage
            .prepare_commit_data(epoch(2), epoch(3), &mut workspace)
            .unwrap()
            .install(),
    );
    let columns = storage.columns.read();
    let column = columns.get(&key("value")).unwrap();
    assert_eq!(column.compression_mode, CompressionMode::Auto);
    assert!(column.zone_map_dirty);
    assert!(column.block_zone_maps.is_empty());
    assert!(column.might_match(CompareOp::Eq, &Value::Int64(1000)));
    assert_eq!(workspace.retired_metadata.len(), 1);
    assert_eq!(workspace.retired_metadata[0].len(), 1);
}

#[test]
fn sparse_commit_range_scan_keeps_new_column_values() {
    let store = LpgStore::new().unwrap();
    let nodes = [
        store.create_node(&["Person"]),
        store.create_node(&["Person"]),
        store.create_node(&["Person"]),
    ];
    let age = key("age");
    let mut workspace = PropertyCommitWorkspace::new(
        Vec::new(),
        nodes
            .into_iter()
            .zip([25, 35, 45])
            .map(|(node, value)| (node, age.clone(), Some(Value::Int64(value))))
            .collect(),
        vec![],
    );
    drop(
        store
            .node_properties
            .prepare_commit_data(epoch(0), epoch(1), &mut workspace)
            .unwrap()
            .install(),
    );
    {
        let columns = store.node_properties.columns.read();
        let column = columns.get(&age).unwrap();
        assert!(column.zone_map_dirty);
        assert_eq!(column.zone_map.row_count, 0);
    }
    let mut matches = store.find_nodes_in_range("age", Some(&Value::Int64(30)), None, false, false);
    matches.sort_unstable();
    assert_eq!(matches, nodes[1..]);
    assert_eq!(
        store.find_nodes_in_range("age", None, Some(&Value::Int64(30)), false, false),
        vec![nodes[0]]
    );
}

#[test]
fn sparse_commit_range_scan_ignores_stale_occupied_column_bounds() {
    let store = LpgStore::new().unwrap();
    let nodes = [
        store.create_node(&["Person"]),
        store.create_node(&["Person"]),
        store.create_node(&["Person"]),
    ];
    let age = key("age");
    for (node, value) in nodes.into_iter().zip([100, 200, 300]) {
        store.set_node_property(node, "age", Value::Int64(value));
    }
    let lower = Value::Int64(30);
    let upper = Value::Int64(400);
    // Clean bounds remain useful negative controls before and after a rebuild.
    assert!(
        !store
            .node_properties
            .might_match_range(&age, None, Some(&lower), false, false)
    );
    assert!(
        !store
            .node_properties
            .might_match_range(&age, Some(&upper), None, false, false)
    );
    let mut workspace = PropertyCommitWorkspace::new(
        Vec::new(),
        nodes
            .into_iter()
            .zip([25, 35, 450])
            .map(|(node, value)| (node, age.clone(), Some(Value::Int64(value))))
            .collect(),
        vec![],
    );
    drop(
        store
            .node_properties
            .prepare_commit_data(epoch(0), epoch(1), &mut workspace)
            .unwrap()
            .install(),
    );
    assert_eq!(
        store.find_nodes_in_range("age", None, Some(&lower), false, false),
        vec![nodes[0]]
    );
    assert_eq!(
        store.find_nodes_in_range("age", Some(&upper), None, false, false),
        vec![nodes[2]]
    );
    let mut matches = store.find_nodes_in_range("age", Some(&lower), Some(&upper), true, true);
    matches.sort_unstable();
    assert_eq!(matches, vec![nodes[1]]);
    store.node_properties.rebuild_zone_maps();
    assert!(!store.node_properties.might_match_range(
        &age,
        None,
        Some(&Value::Int64(20)),
        false,
        false
    ));
    assert!(!store.node_properties.might_match_range(
        &age,
        Some(&Value::Int64(500)),
        None,
        false,
        false
    ));
    assert_eq!(
        store.find_nodes_in_range("age", Some(&upper), None, false, false),
        vec![nodes[2]]
    );
}

#[test]
fn sparse_commit_all_reservation_failures_leave_logical_preimage_unchanged() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            RESERVATION_FAILURE.with(|counter| counter.set(None));
        }
    }
    let mut saw_complete_fragments_failure = false;
    let mut reached_success = false;
    for at in 0..100 {
        let storage = PropertyStorage::<NodeId>::new();
        storage.set(id(1), key("a"), Value::Int64(1), epoch(1));
        storage.set(id(1), key("a"), Value::Int64(81), EpochId::PENDING);
        storage.set(id(2), key("b"), Value::Int64(2), epoch(1));
        let a = storage.get_history(id(1), &key("a"));
        let b = storage.get_history(id(2), &key("b"));
        let mut workspace = PropertyCommitWorkspace::new(
            Vec::new(),
            vec![set(1, "a", 10), set(2, "b", 20), set(3, "c", 30)],
            vec![],
        );
        let failed = {
            let _reset = Reset;
            RESERVATION_FAILURE.with(|counter| counter.set(Some(at)));
            match storage.prepare_commit_data(epoch(2), epoch(3), &mut workspace) {
                Ok(prepared) => {
                    drop(prepared);
                    false
                }
                Err(_) => true,
            }
        };
        assert_eq!(storage.get_history(id(1), &key("a")), a);
        assert_eq!(storage.get_history(id(2), &key("b")), b);
        assert!(!storage.keys().contains(&key("c")));
        assert_eq!(workspace.raw_ops.len(), 3);
        if failed {
            saw_complete_fragments_failure |= workspace
                .fragments
                .iter()
                .filter(|fragment| {
                    fragment
                        .candidate
                        .as_ref()
                        .is_some_and(|column| !column.values.is_empty())
                })
                .count()
                == 3;
        } else {
            reached_success = true;
            break;
        }
    }
    assert!(
        saw_complete_fragments_failure,
        "late failure must retain every completed candidate"
    );
    assert!(reached_success);
}

#[test]
fn sparse_commit_release_rebind_pins_exact_store_and_preserves_reservations() {
    let store = LpgStore::new().unwrap();
    let other = LpgStore::new().unwrap();
    let mut workspace = PropertyCommitWorkspace::new(Vec::new(), vec![set(1, "value", 5)], vec![]);
    let transition = store.pin_exclusive_unframed_transition().unwrap();
    let ready = store
        .node_properties
        .prepare_commit_data(epoch(2), epoch(3), &mut workspace)
        .unwrap();
    let released = ready.release(&transition).unwrap();
    assert!(store.node_properties.columns.try_write().is_some());
    assert!(transition.pins_property_storage(&store.node_properties));
    let ready = released.rebind().unwrap();
    assert!(store.node_properties.columns.try_read().is_none());
    drop(ready.install());
    assert!(store.node_properties.columns.try_read().is_some());
    drop(transition);
    assert_eq!(
        store.node_properties.get(id(1), &key("value")),
        Some(Value::Int64(5))
    );

    let wrong_transition = other.pin_exclusive_unframed_transition().unwrap();
    let mut rejected =
        PropertyCommitWorkspace::new(Vec::new(), vec![set(2, "rejected", 6)], vec![]);
    let ready = store
        .node_properties
        .prepare_commit_data(epoch(3), epoch(4), &mut rejected)
        .unwrap();
    assert!(ready.release(&wrong_transition).is_err());
    assert_eq!(store.node_properties.get(id(2), &key("rejected")), None);

    let mut edge_workspace = PropertyCommitWorkspace::new(
        Vec::new(),
        vec![(EdgeId::new(1), key("edge"), Some(Value::Int64(7)))],
        vec![],
    );
    let transition = store.pin_exclusive_unframed_transition().unwrap();
    let ready = store
        .edge_properties
        .prepare_commit_data(epoch(3), epoch(4), &mut edge_workspace)
        .unwrap();
    drop(
        ready
            .release(&transition)
            .unwrap()
            .rebind()
            .unwrap()
            .install(),
    );
    assert_eq!(
        store.edge_properties.get(EdgeId::new(1), &key("edge")),
        Some(Value::Int64(7))
    );
}

#[test]
fn sparse_property_final_column_contention_rejects_without_allocator_traffic() {
    let store = LpgStore::new().unwrap();
    store
        .node_properties
        .set(id(1), key("value"), Value::Int64(5), epoch(1));
    let before = store.node_properties.get_history(id(1), &key("value"));
    let mut workspace = PropertyCommitWorkspace::new(Vec::new(), vec![set(1, "value", 9)], vec![]);
    let transition = store.pin_exclusive_unframed_transition().unwrap();
    let released = store
        .node_properties
        .prepare_commit_data(epoch(2), epoch(3), &mut workspace)
        .unwrap()
        .release(&transition)
        .unwrap();
    let blocker = store.node_properties.columns.read();
    crate::allocation_test::start();
    let result = released.rebind();
    let conflict = matches!(result, Err(DataRebindError::Conflict(_)));
    drop(result);
    let traffic = crate::allocation_test::stop();
    assert!(conflict);
    assert_eq!(traffic, crate::allocation_test::Counts::default());
    drop(blocker);
    assert!(store.node_properties.columns.try_write().is_some());
    assert_eq!(
        store.node_properties.get_history(id(1), &key("value")),
        before
    );
    assert!(workspace.retired_logs.is_empty());
    let mut retry = PropertyCommitWorkspace::new(Vec::new(), vec![set(1, "value", 9)], vec![]);
    let released = store
        .node_properties
        .prepare_commit_data(epoch(2), epoch(3), &mut retry)
        .unwrap()
        .release(&transition)
        .unwrap();
    crate::allocation_test::start();
    drop(released.rebind().unwrap());
    assert_eq!(
        crate::allocation_test::stop(),
        crate::allocation_test::Counts::default()
    );
    assert_eq!(
        store.node_properties.get_history(id(1), &key("value")),
        before
    );
}

#[test]
fn sparse_commit_counts_borrowed_cells_and_rejects_write_through_creations() {
    let storage = PropertyStorage::<NodeId>::new();
    storage.set(id(1), key("live"), Value::Int64(10), epoch(1));
    storage.set(id(1), key("removed"), Value::Null, epoch(1));
    storage.set(id(1), key("foreign"), Value::Int64(80), EpochId::PENDING);
    storage.set(id(2), key("untouched_future"), Value::Int64(90), epoch(9));
    assert_eq!(storage.commit_property_count(id(1), epoch(2)).unwrap(), 1);
    assert_eq!(storage.commit_property_count(id(3), epoch(2)).unwrap(), 0);
    assert!(
        storage
            .commit_property_count(id(1), EpochId::PENDING)
            .is_err()
    );
    assert!(storage.commit_property_count(id(2), epoch(2)).is_err());
    assert!(storage.validate_buffered_creation(id(1)).is_err());
    assert!(storage.validate_buffered_creation(id(2)).is_err());
    assert!(storage.validate_buffered_creation(id(3)).is_ok());
}

#[test]
fn sparse_property_rebind_and_install_have_zero_allocator_traffic() {
    use crate::allocation_test as allocation;

    for full_capacity_replacement in [false, true] {
        let store = LpgStore::new().unwrap();
        let storage = &store.node_properties;
        for node in 1..=3 {
            storage.set(id(node), key("existing"), Value::Int64(10), epoch(0));
        }
        // The pinned hashbrown table has three usable buckets at this size.
        // Pure replacement must not reserve an extra slot to hide its behavior.
        let full_capacity = {
            let columns = storage.columns.read();
            let column = columns.get(&key("existing")).unwrap();
            assert_eq!(column.values.len(), column.values.capacity());
            column.values.capacity()
        };
        let ops = if full_capacity_replacement {
            vec![set(1, "existing", 20)]
        } else {
            vec![
                set(1, "existing", 20),
                set(4, "existing", 40),
                set(1, "new", 50),
                set(5, "new", 60),
            ]
        };
        let deleted = if full_capacity_replacement {
            vec![]
        } else {
            vec![id(2)]
        };
        let mut workspace = PropertyCommitWorkspace::new(Vec::new(), ops, deleted);
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        let ready = storage
            .prepare_commit_data(epoch(0), epoch(1), &mut workspace)
            .unwrap();
        if full_capacity_replacement {
            let column = ready.guards.columns.get(&key("existing")).unwrap();
            assert_eq!(column.values.capacity(), full_capacity);
            assert_eq!(column.values.len(), full_capacity);
        }
        let released = ready.release(&transition).unwrap();
        allocation::start();
        let result = std::hint::black_box(released).rebind();
        let rebound = allocation::stop();
        let ready = result.unwrap();
        allocation::start();
        let installed = std::hint::black_box(ready).install();
        let installed_counts = allocation::stop();
        drop(installed);
        assert_eq!(rebound, allocation::Counts::default());
        assert_eq!(
            installed_counts,
            allocation::Counts::default(),
            "full_capacity_replacement={full_capacity_replacement}"
        );
        assert_eq!(
            storage.get_at(id(1), &key("existing"), epoch(1)),
            Some(Value::Int64(20))
        );
        if !full_capacity_replacement {
            assert_eq!(storage.get_at(id(2), &key("existing"), epoch(1)), None);
            assert_eq!(
                storage.get_at(id(4), &key("existing"), epoch(1)),
                Some(Value::Int64(40))
            );
            assert_eq!(
                storage.get_at(id(1), &key("new"), epoch(1)),
                Some(Value::Int64(50))
            );
            assert_eq!(
                storage.get_at(id(5), &key("new"), epoch(1)),
                Some(Value::Int64(60))
            );
        }
    }
}
