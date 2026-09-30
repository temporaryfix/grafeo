use super::*;
use crate::graph::Direction;
use crate::graph::lpg::property::CompareOp;
use grafeo_common::types::{EpochId, TransactionId};

fn assert_tx_overlay_entity_work(edge: bool) {
    for count in [64, 128, 256] {
        let store = LpgStore::new().unwrap();
        let tx = TransactionId::new(701);
        let epoch = store.current_epoch();
        let nodes: Vec<_> = (0..count)
            .map(|_| store.create_node(&["Overlay"]))
            .collect();
        let edges: Vec<_> = nodes
            .iter()
            .map(|&node| store.create_edge(node, node, "SELF"))
            .collect();
        for index in 0..count {
            let value = Value::Int64(i64::try_from(index).unwrap());
            if edge {
                store.set_edge_property_buffered(edges[index], "value", value, tx);
                store.set_edge_property_buffered(edges[index], "other", Value::Int64(9), tx);
            } else {
                store.set_node_property_buffered(nodes[index], "value", value, tx);
                store.set_node_property_buffered(nodes[index], "other", Value::Int64(9), tx);
            }
        }
        super::property_ops::take_overlay_property_visits();
        for index in 0..count {
            let props = if edge {
                store.read_edge_properties_visible(edges[index], epoch, Some(tx))
            } else {
                store.read_node_properties_visible(nodes[index], epoch, Some(tx))
            };
            assert_eq!(props.len(), 2);
            assert_eq!(
                props[&PropertyKey::new("value")],
                Value::Int64(i64::try_from(index).unwrap())
            );
            assert_eq!(props[&PropertyKey::new("other")], Value::Int64(9));
        }
        let visits = super::property_ops::take_overlay_property_visits();
        assert_eq!(visits[usize::from(!edge)], 0);
        assert_eq!(
            visits[usize::from(edge)],
            2 * count,
            "{count} whole-entity reads must visit only their own two property operations"
        );

        // The older owned-entity materialization callers must use the same
        // bucket, rather than silently retaining the transaction-wide scan.
        for index in 0..count {
            let props = if edge {
                let mut entity = store.get_edge(edges[index]).unwrap();
                store.apply_edge_tx_delta(&mut entity, tx);
                entity.properties
            } else {
                let mut entity = store.get_node(nodes[index]).unwrap();
                store.apply_node_tx_delta(&mut entity, tx);
                entity.properties
            };
            assert_eq!(props.len(), 2);
            assert_eq!(
                props.get(&PropertyKey::new("value")),
                Some(&Value::Int64(i64::try_from(index).unwrap()))
            );
            assert_eq!(
                props.get(&PropertyKey::new("other")),
                Some(&Value::Int64(9))
            );
        }
        let visits = super::property_ops::take_overlay_property_visits();
        assert_eq!(visits[usize::from(!edge)], 0);
        assert_eq!(visits[usize::from(edge)], 2 * count);

        let untouched = store.create_node(&["Overlay"]);
        let untouched_edge = store.create_edge(untouched, untouched, "SELF");
        assert!(
            store
                .read_node_properties_visible(untouched, epoch, Some(tx))
                .is_empty()
        );
        assert!(
            store
                .read_edge_properties_visible(untouched_edge, epoch, Some(tx))
                .is_empty()
        );
        assert_eq!(super::property_ops::take_overlay_property_visits(), [0, 0]);
    }
}

#[test]
fn tx_overlay_entity_work_nodes() {
    assert_tx_overlay_entity_work(false);
}

#[test]
fn tx_overlay_entity_work_edges() {
    assert_tx_overlay_entity_work(true);
}

#[test]
fn tx_overlay_entity_overwrite_remove_snapshot_commit_and_drop() {
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Overlay"]);
    let other = store.create_node(&["Overlay"]);
    let edge = store.create_edge(node, other, "LINK");
    let other_edge = store.create_edge(other, node, "LINK");
    let epoch = store.current_epoch();
    let tx = TransactionId::new(702);
    for key in ["kept", "removed"] {
        store.set_node_property(node, key, Value::Int64(1));
        store.set_edge_property(edge, key, Value::Int64(1));
    }
    let set = |key, value| {
        store.set_node_property_buffered(node, key, Value::Int64(value), tx);
        store.set_edge_property_buffered(edge, key, Value::Int64(value), tx);
    };
    set("kept", 10);
    set("kept", 20);
    set("fresh", 30);
    store.remove_node_property_buffered(node, "removed", tx);
    store.remove_edge_property_buffered(edge, "removed", tx);
    store.set_node_property_buffered(other, "unrelated", Value::Int64(40), tx);
    store.set_edge_property_buffered(other_edge, "unrelated", Value::Int64(40), tx);
    let snapshot = store.tx_overlay_snapshot(tx);
    // Overwrite replaces one operation; a removal remains one tombstone.
    assert_eq!(snapshot.node_props.len(), 4);
    assert_eq!(snapshot.edge_props.len(), 4);
    assert_eq!(snapshot.node_props.keys().count(), 4);
    assert_eq!(snapshot.edge_props.iter().count(), 4);
    let node_ops: FxHashMap<_, _> = snapshot.node_props.clone().into_iter().collect();
    let edge_ops: FxHashMap<_, _> = snapshot.edge_props.clone().into_iter().collect();
    assert_eq!(node_ops.len(), 4);
    assert_eq!(edge_ops.len(), 4);
    assert!(matches!(
        node_ops[&(node, PropertyKey::new("removed"))],
        PropOp::Remove
    ));
    assert!(matches!(
        edge_ops[&(edge, PropertyKey::new("removed"))],
        PropOp::Remove
    ));
    let expected = FxHashMap::from_iter([
        (PropertyKey::new("kept"), Value::Int64(20)),
        (PropertyKey::new("fresh"), Value::Int64(30)),
    ]);
    assert_eq!(
        store.read_node_properties_visible(node, epoch, Some(tx)),
        expected
    );
    assert_eq!(
        store.read_edge_properties_visible(edge, epoch, Some(tx)),
        expected
    );
    assert_eq!(
        store.read_node_property_visible(node, &PropertyKey::new("kept"), epoch, Some(tx)),
        Some(Value::Int64(20))
    );
    assert_eq!(
        store.read_edge_property_visible(edge, &PropertyKey::new("removed"), epoch, Some(tx)),
        None
    );
    let committed = FxHashMap::from_iter([
        (PropertyKey::new("kept"), Value::Int64(1)),
        (PropertyKey::new("removed"), Value::Int64(1)),
    ]);
    for reader in [None, Some(TransactionId::new(703))] {
        assert_eq!(
            store.read_node_properties_visible(node, epoch, reader),
            committed
        );
        assert_eq!(
            store.read_edge_properties_visible(edge, epoch, reader),
            committed
        );
    }
    store.remove_node_property_buffered(node, "kept", tx);
    store.remove_edge_property_buffered(edge, "kept", tx);
    set("removed", 99);
    set("after_snapshot", 100);
    store.tx_overlay_restore(tx, snapshot);
    assert_eq!(
        store.read_node_properties_visible(node, epoch, Some(tx)),
        expected
    );
    assert_eq!(
        store.read_edge_properties_visible(edge, epoch, Some(tx)),
        expected
    );
    store.apply_tx_overlay(tx);
    assert_eq!(
        store.read_node_property_visible(other, &PropertyKey::new("unrelated"), epoch, None),
        Some(Value::Int64(40))
    );
    assert_eq!(
        store.read_edge_property_visible(other_edge, &PropertyKey::new("unrelated"), epoch, None),
        Some(Value::Int64(40))
    );
    assert_eq!(
        store.read_node_properties_visible(node, epoch, None),
        expected
    );
    assert_eq!(
        store.read_edge_properties_visible(edge, epoch, None),
        expected
    );
    assert!(store.tx_overlay_snapshot(tx).node_props.is_empty());
    assert!(store.tx_overlay_snapshot(tx).edge_props.is_empty());
    set("kept", 200);
    store.drop_tx_overlay(tx);
    assert_eq!(
        store.read_node_properties_visible(node, epoch, Some(tx)),
        expected
    );
    assert_eq!(
        store.read_edge_properties_visible(edge, epoch, Some(tx)),
        expected
    );
    set("kept", 300);
    store.tx_overlay_restore(tx, TxDelta::default());
    assert_eq!(
        store.read_node_properties_visible(node, epoch, Some(tx)),
        expected
    );
    assert_eq!(
        store.read_edge_properties_visible(edge, epoch, Some(tx)),
        expected
    );
    assert!(!store.tx_property_overlay.read().contains_key(&tx));
}

#[cfg(feature = "compact-store")]
#[test]
fn generation_totality_pinned_owned_tokens_move_once_with_authority() {
    use crate::graph::write_permit::{WriteAuthority, with_authority};
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Prepared(String);
    struct Published(String);
    let store = LpgStore::new().unwrap();
    let owner = WriteAuthority::new();
    let foreign = WriteAuthority::new();
    assert!(store.seal_unframed_writes(&owner));
    assert!(store.pin_exclusive_unframed_transition().is_none());
    assert!(with_authority(&foreign, || store
        .pin_exclusive_unframed_transition()
        .is_none()));
    let prepares = AtomicUsize::new(0);
    let publishes = AtomicUsize::new(0);
    let rollbacks = AtomicUsize::new(0);
    let value = with_authority(&owner, || {
        let pinned = store.pin_exclusive_unframed_transition().unwrap();
        pinned
            .publish_empty_generation_after_prepare_and_publish_with_rollback(
                |_| {
                    prepares.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, String>(Prepared("owned".into()))
                },
                |Prepared(value)| {
                    publishes.fetch_add(1, Ordering::SeqCst);
                    Published(value)
                },
                |_published| {
                    rollbacks.fetch_add(1, Ordering::SeqCst);
                },
            )
            .unwrap()
    });
    assert_eq!(value.0, "owned");
    assert_eq!(prepares.load(Ordering::SeqCst), 1);
    assert_eq!(publishes.load(Ordering::SeqCst), 1);
    assert_eq!(rollbacks.load(Ordering::SeqCst), 0);
    assert!(with_authority(&owner, || store
        .pin_exclusive_unframed_transition()
        .is_some()));
}

#[cfg(feature = "compact-store")]
#[test]
fn generation_totality_pinned_error_and_hostile_rollback_consume_exactly_once() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Prepared(String);
    struct Published(String);
    let store = LpgStore::new().unwrap();
    let publishes = AtomicUsize::new(0);
    let rollbacks = AtomicUsize::new(0);
    let retired = std::cell::RefCell::new(None);
    {
        let pinned = store.pin_exclusive_unframed_transition().unwrap();
        let failed = pinned.publish_empty_generation_after_prepare_and_publish_with_rollback(
            |_| Err::<Prepared, _>("chosen prepare error"),
            |Prepared(value)| {
                publishes.fetch_add(1, Ordering::SeqCst);
                Published(value)
            },
            |_value| {
                rollbacks.fetch_add(1, Ordering::SeqCst);
            },
        );
        assert!(matches!(failed, Err("chosen prepare error")));
    }
    assert_eq!(publishes.load(Ordering::SeqCst), 0);
    assert_eq!(rollbacks.load(Ordering::SeqCst), 0);
    store.panic_after_transport_publication_once_for_test();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let pinned = store.pin_exclusive_unframed_transition().unwrap();
        pinned.publish_empty_generation_after_prepare_and_publish_with_rollback(
            |_| Ok::<_, ()>(Prepared("rollback token".into())),
            |Prepared(value)| {
                publishes.fetch_add(1, Ordering::SeqCst);
                Published(value)
            },
            |value| {
                rollbacks.fetch_add(1, Ordering::SeqCst);
                // Caller owns this pinned transition: rollback does not release it.
                assert!(store.mutation_scope_gate.try_write().is_none());
                *retired.borrow_mut() = Some(value);
            },
        )
    }));
    assert_eq!(
        panic.err().unwrap().downcast_ref::<&str>(),
        Some(&"hostile post-publication LPG generation failpoint")
    );
    assert_eq!(publishes.load(Ordering::SeqCst), 1);
    assert_eq!(rollbacks.load(Ordering::SeqCst), 1);
    assert_eq!(retired.into_inner().unwrap().0, "rollback token");
    assert!(store.pin_exclusive_unframed_transition().is_some());
}

#[test]
fn test_create_node() {
    let store = LpgStore::new().unwrap();

    let id = store.create_node(&["Person"]);
    assert!(id.is_valid());

    let node = store.get_node(id).unwrap();
    assert!(node.has_label("Person"));
    assert!(!node.has_label("Animal"));
}

#[test]
fn test_create_node_with_props() {
    let store = LpgStore::new().unwrap();

    let id = store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Alix")), ("age", Value::from(30i64))],
    );

    let node = store.get_node(id).unwrap();
    assert_eq!(
        node.get_property("name").and_then(|v| v.as_str()),
        Some("Alix")
    );
    assert_eq!(
        node.get_property("age").and_then(|v| v.as_int64()),
        Some(30)
    );
}

#[test]
fn test_delete_node() {
    let store = LpgStore::new().unwrap();

    let id = store.create_node(&["Person"]);
    assert_eq!(store.node_count(), 1);

    assert!(store.delete_node(id));
    assert_eq!(store.node_count(), 0);
    assert!(store.get_node(id).is_none());

    // Double delete should return false
    assert!(!store.delete_node(id));
}

#[test]
fn eager_delete_retains_historical_labels_and_properties() {
    let store = LpgStore::new().unwrap();
    let created = EpochId::new(10);
    let deleted = EpochId::new(20);
    store.set_epoch(created);
    let id = store.create_node(&["Person", "Researcher"]);
    store.set_node_property_at_epoch(id, "name", Value::from("Alix"), created);

    store.set_epoch(deleted);
    assert!(store.delete_node_at_epoch(id, deleted));
    assert!(store.get_node(id).is_none());
    assert!(store.nodes_by_label("Person").is_empty());
    assert!(store.nodes_by_label("Researcher").is_empty());

    let history = store.get_node_history(id);
    assert_eq!(history.len(), 1);
    let (actual_created, actual_deleted, historical) = &history[0];
    assert_eq!(*actual_created, created);
    assert_eq!(*actual_deleted, Some(deleted));
    assert!(historical.has_label("Person"));
    assert!(historical.has_label("Researcher"));
    assert_eq!(historical.get_property("name"), Some(&Value::from("Alix")));

    let name_history = store
        .node_property_history(id)
        .into_iter()
        .find(|(key, _)| key.as_str() == "name")
        .expect("deleted node property log");
    assert_eq!(
        name_history.1,
        vec![(created, Value::from("Alix")), (deleted, Value::Null)]
    );
}

#[test]
fn finalized_transactional_delete_retains_historical_labels_and_properties() {
    let store = LpgStore::new().unwrap();
    let created = EpochId::new(30);
    let deleted = EpochId::new(40);
    let transaction = TransactionId::new(4040);
    store.set_epoch(created);
    let id = store.create_node(&["Person"]);
    store.set_node_property_at_epoch(id, "name", Value::from("Gus"), created);

    assert!(store.delete_node_transactional(id, created, transaction));
    assert_eq!(store.nodes_by_label("Person"), vec![id]);
    store.sync_epoch(deleted);
    store.finalize_deletes_by_id(transaction, deleted, &[id]);

    assert!(store.get_node(id).is_none());
    assert!(store.nodes_by_label("Person").is_empty());
    let history = store.get_node_history(id);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].0, created);
    assert_eq!(history[0].1, Some(deleted));
    assert!(history[0].2.has_label("Person"));
    assert_eq!(history[0].2.get_property("name"), Some(&Value::from("Gus")));

    let name_history = store
        .node_property_history(id)
        .into_iter()
        .find(|(key, _)| key.as_str() == "name")
        .expect("transactionally deleted node property log");
    assert_eq!(name_history.1.last(), Some(&(deleted, Value::Null)));
}

fn commit_transport_edge_delete(
    store: &LpgStore,
    edge: EdgeId,
    transaction: TransactionId,
    commit_epoch: EpochId,
) {
    assert!(store.delete_edge_transactional(edge, store.current_epoch(), transaction));
    let pending = store.take_pending_edge_deletes(transaction);
    assert_eq!(pending.len(), 1);
    store.sync_epoch(commit_epoch);
    store.finalize_edge_deletes_by_id(transaction, commit_epoch, &pending);
}

#[test]
fn transport_edge_purge_consumes_receipt_but_preserves_allocator_high_water() {
    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    let edge = EdgeId::new(41);
    let receipt = store
        .create_transport_edge_with_id(edge, src, dst, "CARRIED")
        .unwrap()
        .expect("fresh transport identity");
    store.set_edge_property(edge, "weight", Value::from(7i64));
    assert_eq!(store.next_edge_id(), 42);

    commit_transport_edge_delete(&store, edge, TransactionId::new(9001), EpochId::new(1));
    assert_eq!(store.get_edge_history(edge).len(), 1);
    assert!(!store.edge_property_history(edge).is_empty());

    assert!(store.purge_transport_extract_edges(&[&receipt]));
    assert!(store.get_edge_history(edge).is_empty());
    assert!(store.edge_property_history(edge).is_empty());
    assert_eq!(
        store.next_edge_id(),
        42,
        "purge must not rewind ID high-water"
    );
    assert_eq!(
        store.create_edge(src, dst, "NEXT"),
        EdgeId::new(42),
        "a purged transport identity must never be reused"
    );
}

#[test]
fn transport_purged_receipt_cannot_authorize_an_exactly_recreated_identity() {
    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    let edge = EdgeId::new(42);
    let receipt = store
        .create_transport_edge_with_id(edge, src, dst, "CARRIED")
        .unwrap()
        .expect("fresh transport identity");
    commit_transport_edge_delete(&store, edge, TransactionId::new(4201), EpochId::new(1));
    assert!(store.purge_transport_extract_edges(&[&receipt]));

    store
        .restore_edge_history_exact(edge, src, dst, "CARRIED", &[(EpochId::new(2), None)])
        .expect("hostile exact same-shape recreation");
    commit_transport_edge_delete(&store, edge, TransactionId::new(4202), EpochId::new(3));
    store.set_edge_property_at_epoch(edge, "proof", Value::from("ordinary"), EpochId::new(2));

    assert!(!store.is_transport_extract_edge(&receipt));
    assert!(!store.purge_transport_extract_edges(&[&receipt]));
    assert_eq!(store.get_edge_history(edge).len(), 1);
    assert_eq!(
        store
            .get_edge_at_epoch(edge, EpochId::new(2))
            .and_then(|edge| edge.get_property("proof").cloned()),
        Some(Value::from("ordinary"))
    );
}

#[test]
fn ordinary_deleted_edge_history_is_not_transport_purgeable() {
    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    let edge = store.create_edge(src, dst, "ORDINARY");
    store.set_edge_property(edge, "evidence", Value::from("retained"));
    let created = store.current_epoch();
    store.sync_epoch(EpochId::new(5));
    assert!(store.delete_edge(edge));

    let foreign = LpgStore::new().unwrap();
    let foreign_src = foreign.create_node(&["Source"]);
    let foreign_dst = foreign.create_node(&["Destination"]);
    let foreign_receipt = foreign
        .create_transport_edge_with_id(edge, foreign_src, foreign_dst, "ORDINARY")
        .unwrap()
        .expect("same-numbered foreign transport identity");
    assert!(!store.purge_transport_extract_edges(&[&foreign_receipt]));
    let history = store.get_edge_history(edge);
    assert_eq!(history.len(), 1);
    assert!(history[0].1.is_some());
    assert_eq!(
        store
            .get_edge_at_epoch(edge, created)
            .and_then(|edge| edge.get_property("evidence").cloned()),
        Some(Value::from("retained")),
        "ordinary delete must retain temporal structure and properties"
    );
}

#[test]
fn closed_cross_shard_restore_never_mints_or_acquires_purge_authority() {
    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["Source"]);
    let missing_dst = NodeId::new(500);
    let edge = EdgeId::new(77);
    store
        .restore_closed_cross_shard_edge_history_exact(
            edge,
            src,
            missing_dst,
            "HISTORICAL",
            &[(EpochId::new(2), Some(EpochId::new(3)))],
        )
        .expect("restore non-authorizing closed history");
    store.set_edge_property_at_epoch(edge, "proof", Value::from("ordinary"), EpochId::new(2));

    assert!(
        store
            .create_transport_edge_with_id(edge, src, missing_dst, "HISTORICAL")
            .expect("duplicate refusal is not allocation failure")
            .is_none(),
        "a pre-existing historical identity can never be blessed after restore"
    );
    let foreign = LpgStore::new().unwrap();
    let foreign_src = foreign.create_node(&["Source"]);
    let foreign_receipt = foreign
        .restore_transport_edge_history_exact(
            edge,
            foreign_src,
            missing_dst,
            "HISTORICAL",
            &[(EpochId::new(2), Some(EpochId::new(3)))],
        )
        .expect("foreign receipt with the same public shape");
    assert!(!store.purge_transport_extract_edges(&[&foreign_receipt]));
    assert_eq!(
        store
            .get_edge_at_epoch(edge, EpochId::new(2))
            .and_then(|edge| edge.get_property("proof").cloned()),
        Some(Value::from("ordinary"))
    );
    assert!(store.get_edge_at_epoch(edge, EpochId::new(3)).is_none());
    assert!(store.peek_next_node_id() > missing_dst.as_u64());
}

#[test]
fn transport_edge_purge_requires_exact_store_authority() {
    use crate::graph::write_permit::{WriteAuthority, with_authority};

    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    let edge = EdgeId::new(73);
    let receipt = store
        .create_transport_edge_with_id(edge, src, dst, "CARRIED")
        .unwrap()
        .expect("fresh transport identity");
    commit_transport_edge_delete(&store, edge, TransactionId::new(9002), EpochId::new(2));

    let owner = WriteAuthority::new();
    let foreign = WriteAuthority::new();
    assert!(store.seal_unframed_writes(&owner));
    assert!(!store.purge_transport_extract_edges(&[&receipt]));
    assert!(!with_authority(&foreign, || {
        store.purge_transport_extract_edges(&[&receipt])
    }));
    assert_eq!(store.get_edge_history(edge).len(), 1);

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_authority(&owner, || panic!("hostile caught panic"));
    }));
    assert!(panic.is_err());
    assert!(
        !store.purge_transport_extract_edges(&[&receipt]),
        "caught panic must not leak the owner's write authority"
    );

    assert!(with_authority(&owner, || {
        store.purge_transport_extract_edges(&[&receipt])
    }));
    assert!(store.get_edge_history(edge).is_empty());
}

#[test]
fn transport_publish_unwind_leaves_lpg_receipt_and_history_retryable() {
    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    let edge = EdgeId::new(7_301);
    let receipt = store
        .create_transport_edge_with_id(edge, src, dst, "CARRIED")
        .unwrap()
        .expect("fresh transport identity");
    commit_transport_edge_delete(&store, edge, TransactionId::new(7_302), EpochId::new(1));

    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = store.purge_transport_extract_edges_after_prepare_and_publish(
            &[&receipt],
            || Ok::<(), ()>(()),
            || panic!("hostile representation publisher"),
        );
    }));
    assert!(unwind.is_err());
    assert_eq!(store.get_edge_history(edge).len(), 1);
    assert!(
        store.is_transport_extract_closed_edge_unchanged_since(&receipt, store.current_epoch(),)
    );

    assert!(
        store.purge_transport_extract_edges(&[&receipt]),
        "a caught publisher panic must leave the exact physical purge retryable"
    );
    assert!(store.get_edge_history(edge).is_empty());
}

#[test]
fn transport_post_publish_unwind_rolls_back_representation_and_lpg_state() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    let edge = EdgeId::new(7_311);
    let receipt = store
        .create_transport_edge_with_id(edge, src, dst, "CARRIED")
        .unwrap()
        .expect("fresh transport identity");
    store.set_edge_property(edge, "proof", Value::from("retained"));
    commit_transport_edge_delete(&store, edge, TransactionId::new(7_312), EpochId::new(1));
    let property_history = store.edge_property_history(edge);

    let representation_published = AtomicBool::new(false);
    store.panic_after_transport_publication_once_for_test();
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = store.purge_transport_extract_edges_after_prepare_and_publish_with_rollback(
            &[&receipt],
            &[],
            |_| Ok::<(), ()>(()),
            |()| representation_published.swap(true, Ordering::SeqCst),
            |previous| representation_published.store(previous, Ordering::SeqCst),
        );
    }));
    assert!(unwind.is_err());
    assert!(
        !representation_published.load(Ordering::SeqCst),
        "the external representation must roll back with the hostile boundary unwind"
    );
    assert_eq!(store.get_edge_history(edge).len(), 1);
    assert_eq!(store.edge_property_history(edge), property_history);
    assert!(
        store.is_transport_extract_closed_edge_unchanged_since(&receipt, store.current_epoch(),)
    );

    assert!(
        store.purge_transport_extract_edges(&[&receipt]),
        "the exact same receipt must remain retryable after rollback"
    );
}

#[test]
fn empty_generation_post_publish_unwind_rolls_back_representation() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let store = LpgStore::new().unwrap();
    let representation_published = AtomicBool::new(false);
    store.panic_after_transport_publication_once_for_test();
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = store.publish_empty_generation_after_prepare_and_publish_with_rollback(
            |_| Ok::<(), ()>(()),
            || representation_published.swap(true, Ordering::SeqCst),
            |previous| representation_published.store(previous, Ordering::SeqCst),
        );
    }));
    assert!(unwind.is_err());
    assert!(!representation_published.load(Ordering::SeqCst));
    assert!(!store.purge_transport_extract_edges(&[]));
}

#[test]
fn transport_creation_and_missing_destination_grants_require_a_source_identity() {
    use crate::graph::write_permit::{WriteAuthority, with_authority};

    let store = LpgStore::new().unwrap();
    let present_destination = store.create_node(&["Destination"]);
    let absent_source = NodeId::new(8_001);
    let rejected_edge = EdgeId::new(8_002);
    let edge_high_water = store.peek_next_edge_id();
    assert!(
        store
            .create_transport_edge_with_id(
                rejected_edge,
                absent_source,
                present_destination,
                "CARRIED",
            )
            .unwrap()
            .is_none()
    );
    assert!(!store.all_known_edge_ids().contains(&rejected_edge));
    assert_eq!(store.peek_next_edge_id(), edge_high_water);
    assert!(
        !store
            .all_edge_types()
            .iter()
            .any(|edge_type| edge_type == "CARRIED"),
        "a rejected vacant identity must not mutate the type catalog"
    );
    let rejected_restore = EdgeId::new(8_005);
    assert!(
        store
            .restore_transport_edge_history_exact(
                rejected_restore,
                absent_source,
                present_destination,
                "RESTORED",
                &[(EpochId::new(1), None)],
            )
            .is_err()
    );
    assert!(!store.all_known_edge_ids().contains(&rejected_restore));
    assert!(
        !store
            .all_edge_types()
            .iter()
            .any(|edge_type| edge_type == "RESTORED")
    );

    let source_transaction = TransactionId::new(8_006);
    let source =
        store.create_node_versioned(&["Source"], store.current_epoch(), source_transaction);
    let missing_destination = NodeId::new(8_003);
    let edge = EdgeId::new(8_004);
    let receipt = store
        .create_transport_edge_with_id(edge, source, missing_destination, "CARRIED")
        .unwrap()
        .expect("present source authorizes transport creation");
    let owner = WriteAuthority::new();
    let grant = with_authority(&owner, || {
        store
            .grant_transport_edge_mutation(&receipt, &owner)
            .expect("held owner can mint a grant")
    });
    store.discard_entities_by_id(source_transaction, &[source], &[]);
    assert_eq!(store.get_edge_history(edge).len(), 1);
    assert!(!with_authority(&owner, || {
        store.accepts_transport_edge_missing_destination(&grant, missing_destination, &owner)
    }));
}

#[test]
fn in_flight_edge_reservation_survives_discard_and_gc() {
    use std::sync::Barrier;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    let store = Arc::new(LpgStore::new().unwrap());
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    let transaction = TransactionId::new(8_101);
    let reserved_id = EdgeId::new(8_102);
    store.set_next_edge_id(reserved_id.as_u64());
    let barrier = Arc::new(Barrier::new(2));
    *store.edge_publication_barrier.write() = Some(Arc::clone(&barrier));

    let creator_store = Arc::clone(&store);
    let (created_tx, created_rx) = mpsc::channel();
    let creator = thread::spawn(move || {
        let id =
            creator_store.create_edge_versioned(src, dst, "PENDING", EpochId::new(1), transaction);
        created_tx.send(id).unwrap();
    });
    barrier.wait();
    *store.edge_publication_barrier.write() = None;
    store.discard_entities_by_id(transaction, &[], &[reserved_id]);
    store.discard_uncommitted_versions(transaction);

    // GC requires exclusive admission, which the paused creator's mutation
    // pin excludes. Probe the actual gate before starting the GC worker;
    // lack of worker completion alone would not prove that exclusion.
    let exclusive_blocked = store.mutation_scope_gate.try_write().is_none();
    let gc_store = Arc::clone(&store);
    let (gc_started_tx, gc_started_rx) = mpsc::channel();
    let (gc_done_tx, gc_done_rx) = mpsc::channel();
    let collector = thread::spawn(move || {
        gc_started_tx.send(()).unwrap();
        gc_store.gc_versions(EpochId::new(100));
        gc_done_tx.send(()).unwrap();
    });
    let started = gc_started_rx.recv_timeout(Duration::from_secs(5));
    let before_release = gc_done_rx.recv_timeout(Duration::from_millis(100));
    // Release before any assertion so a failed observation cannot strand the
    // creator at the test hook while unwinding this test.
    barrier.wait();

    let created = created_rx.recv_timeout(Duration::from_secs(5));
    let gc_was_blocked = matches!(before_release, Err(mpsc::RecvTimeoutError::Timeout));
    let completed = if before_release.is_ok() {
        Ok(())
    } else {
        gc_done_rx.recv_timeout(Duration::from_secs(5))
    };
    assert_eq!(
        created.expect("creator must finish after release"),
        reserved_id
    );
    completed.expect("GC must finish after the creation pin is released");
    creator.join().unwrap();
    collector.join().unwrap();
    started.expect("GC worker must start");
    assert!(
        exclusive_blocked,
        "in-flight creation must exclude GC admission"
    );
    assert!(gc_was_blocked, "GC must wait for the creation pin");
    let edge = store
        .get_edge_versioned(reserved_id, EpochId::new(1), transaction)
        .expect("GC/discard must not erase an out-of-map in-flight publication");
    assert_eq!(edge.id, reserved_id);
    assert_eq!((edge.src, edge.dst), (src, dst));
    assert_eq!(edge.edge_type.as_str(), "PENDING");
}

#[test]
fn exact_restore_reservation_cannot_collide_with_generated_batch_ids() {
    use std::sync::Barrier;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    let store = Arc::new(LpgStore::new().unwrap());
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    let exact = EdgeId::new(8_201);
    let barrier = Arc::new(Barrier::new(2));
    *store.edge_publication_barrier.write() = Some(Arc::clone(&barrier));

    let restore_store = Arc::clone(&store);
    let restore = thread::spawn(move || {
        restore_store.restore_edge_history_exact(
            exact,
            src,
            dst,
            "EXACT",
            &[(EpochId::new(1), None)],
        )
    });
    barrier.wait();
    *store.edge_publication_barrier.write() = None;
    store.set_next_edge_id(exact.as_u64());
    let batch_store = Arc::clone(&store);
    let (batch_tx, batch_rx) = mpsc::channel();
    let batcher = thread::spawn(move || {
        batch_tx
            .send(batch_store.batch_create_edges(&[(src, dst, "BATCH")]))
            .unwrap();
    });
    let batch_before_release = batch_rx.recv_timeout(Duration::from_secs(1));
    barrier.wait();

    restore.join().unwrap().expect("exact restore succeeds");
    batcher.join().unwrap();
    let batch = batch_before_release
        .expect("generated allocation must skip, not wait for, a reserved exact identity");
    assert_eq!(batch, vec![EdgeId::new(exact.as_u64() + 1)]);
    assert_eq!(store.get_edge_history(exact).len(), 1);
    assert_eq!(store.get_edge_history(batch[0]).len(), 1);
}

#[test]
fn system_discard_cannot_destroy_committed_structure_or_provenance() {
    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    store.set_node_property(src, "node-proof", Value::from("retained"));
    let edge = EdgeId::new(8_301);
    let receipt = store
        .create_transport_edge_with_id(edge, src, dst, "CARRIED")
        .unwrap()
        .expect("fresh transport identity");
    store.set_edge_property(edge, "edge-proof", Value::from("retained"));

    store.discard_uncommitted_versions(TransactionId::SYSTEM);
    store.discard_entities_by_id(TransactionId::SYSTEM, &[src, dst], &[edge]);

    assert_eq!(store.node_count(), 2);
    assert_eq!(store.edge_count(), 1);
    assert!(store.nodes_by_label("Source").contains(&src));
    assert_eq!(
        store.get_node(src).unwrap().get_property("node-proof"),
        Some(&Value::from("retained"))
    );
    assert_eq!(
        store.get_edge(edge).unwrap().get_property("edge-proof"),
        Some(&Value::from("retained"))
    );
    assert_eq!(
        store
            .edges_from(src, Direction::Outgoing)
            .collect::<Vec<_>>(),
        vec![(dst, edge)]
    );
    assert!(store.is_transport_extract_edge(&receipt));
}

#[test]
fn transport_high_degree_batch_qualifies_and_purges_with_constant_list_work() {
    const EDGE_COUNT: usize = 1_024;

    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    store.sync_epoch(EpochId::new(1));
    let mut receipts = Vec::with_capacity(EDGE_COUNT);
    for offset in 0..EDGE_COUNT {
        let edge = EdgeId::new(10_000 + u64::try_from(offset).unwrap());
        receipts.push(
            store
                .create_transport_edge_with_id(edge, src, dst, "CARRIED")
                .unwrap()
                .expect("fresh transport identity"),
        );
    }

    let transaction = TransactionId::new(10_024);
    for receipt in &receipts {
        assert!(store.delete_edge_transactional(
            receipt.edge_id(),
            store.current_epoch(),
            transaction,
        ));
    }
    let pending = store.take_pending_edge_deletes(transaction);
    assert_eq!(pending.len(), EDGE_COUNT);
    store.sync_epoch(EpochId::new(2));
    store.finalize_edge_deletes_by_id(transaction, EpochId::new(2), &pending);

    // A repeated receipt is rejected without turning the one high-degree list
    // into receipt × adjacency work. The remaining exact identities still
    // classify CLOSED in the same single traversal.
    let mut hostile_refs: Vec<_> = receipts.iter().collect();
    hostile_refs.push(&receipts[0]);
    store.forward_adj.reset_transport_work_for_test();
    let states =
        store.classify_transport_extract_edges(&hostile_refs, EpochId::new(1), EpochId::new(2));
    assert_eq!(states[0], TransportEdgeState::Invalid);
    assert_eq!(states[EDGE_COUNT], TransportEdgeState::Invalid);
    assert!(
        states[1..EDGE_COUNT]
            .iter()
            .all(|state| *state == TransportEdgeState::Closed)
    );
    assert_eq!(store.forward_adj.transport_work_for_test(), (1, 0));

    let refs: Vec<_> = receipts.iter().collect();
    store.forward_adj.reset_transport_work_for_test();
    assert!(store.purge_transport_extract_edges(&refs));
    assert_eq!(
        store.forward_adj.transport_work_for_test(),
        (3, 1),
        "preflight, exact revalidation, and prepared rebuild stay constant per source list"
    );
    assert_eq!(store.edge_count(), 0);
}

#[test]
fn transport_purge_preserves_small_retained_adjacency_in_delta_storage() {
    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["Source"]);
    let transport_dst = store.create_node(&["TransportDestination"]);
    let retained_dst = store.create_node(&["RetainedDestination"]);
    let transport = EdgeId::new(20_000);
    let receipt = store
        .create_transport_edge_with_id(transport, src, transport_dst, "CARRIED")
        .unwrap()
        .expect("fresh transport identity");
    let retained = store.create_edge(src, retained_dst, "RETAINED");

    commit_transport_edge_delete(
        &store,
        transport,
        TransactionId::new(20_001),
        EpochId::new(1),
    );
    assert!(store.purge_transport_extract_edges(&[&receipt]));

    assert!(store.get_edge(transport).is_none());
    assert!(store.get_edge(retained).is_some());
    assert_eq!(store.edge_count(), 1);
    assert_eq!(
        store
            .edges_from(src, Direction::Outgoing)
            .collect::<Vec<_>>(),
        vec![(retained_dst, retained)],
        "rebuilding a low-degree list places the retained edge in the delta buffer; \
         the containing adjacency list must remain installed",
    );
}

#[test]
fn transport_property_grant_is_exact_incarnation_scoped_and_destination_only() {
    use crate::graph::write_permit::{WriteAuthority, with_authority};

    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["Source"]);
    let missing_dst = NodeId::new(400);
    let edge = EdgeId::new(74);
    let receipt = store
        .create_transport_edge_with_id(edge, src, missing_dst, "CARRIED")
        .unwrap()
        .expect("fresh transport identity");
    let owner = WriteAuthority::new();
    let foreign = WriteAuthority::new();
    assert!(store.seal_unframed_writes(&owner));

    let grant = with_authority(&owner, || {
        store
            .grant_transport_edge_mutation(&receipt, &owner)
            .expect("exact held owner may mint a property-only grant")
    });
    assert!(with_authority(&owner, || {
        store.accepts_transport_edge_missing_destination(&grant, missing_dst, &owner)
    }));
    assert!(!with_authority(&foreign, || {
        store.accepts_transport_edge_missing_destination(&grant, missing_dst, &foreign)
    }));
    assert!(!with_authority(&owner, || {
        store.accepts_transport_edge_missing_destination(&grant, src, &owner)
    }));

    let wrong_type = super::TransportEdgeMutationGrant {
        authority: Arc::clone(&grant.authority),
        id: grant.id,
        src: grant.src,
        dst: grant.dst,
        edge_type: "OTHER".into(),
        destination_labels: Arc::clone(&grant.destination_labels),
        nonce: grant.nonce,
        lifetimes: Arc::clone(&grant.lifetimes),
    };
    assert!(!with_authority(&owner, || {
        store.accepts_transport_edge_missing_destination(&wrong_type, missing_dst, &owner)
    }));

    with_authority(&owner, || {
        store
            .create_node_with_id(missing_dst, &["Reused"])
            .expect("install destination identity");
    });
    assert!(!with_authority(&owner, || {
        store.accepts_transport_edge_missing_destination(&grant, missing_dst, &owner)
    }));

    with_authority(&owner, || store.clear());
    assert!(!with_authority(&owner, || {
        store.accepts_transport_edge_missing_destination(&grant, missing_dst, &owner)
    }));

    let closing = LpgStore::new().unwrap();
    let closing_src = closing.create_node(&["Source"]);
    let closing_dst = NodeId::new(401);
    let closing_edge = EdgeId::new(75);
    let closing_receipt = closing
        .create_transport_edge_with_id(closing_edge, closing_src, closing_dst, "CARRIED")
        .unwrap()
        .expect("fresh closing identity");
    let closing_owner = WriteAuthority::new();
    assert!(closing.seal_unframed_writes(&closing_owner));
    let transaction = TransactionId::new(7474);
    with_authority(&closing_owner, || {
        assert!(closing.delete_edge_transactional(
            closing_edge,
            closing.current_epoch(),
            transaction,
        ));
    });
    assert!(with_authority(&closing_owner, || {
        closing
            .grant_transport_edge_mutation(&closing_receipt, &closing_owner)
            .is_none()
    }));
    let pending = with_authority(&closing_owner, || {
        closing.take_pending_edge_deletes(transaction)
    });
    with_authority(&closing_owner, || {
        closing.sync_epoch(EpochId::new(1));
        closing.finalize_edge_deletes_by_id(transaction, EpochId::new(1), &pending);
    });
    assert!(with_authority(&closing_owner, || {
        closing
            .grant_transport_edge_mutation(&closing_receipt, &closing_owner)
            .is_none()
    }));
}

#[test]
fn transport_creation_never_authenticates_or_overwrites_an_existing_identity() {
    let store = LpgStore::new().unwrap();
    let ordinary_src = store.create_node(&["OrdinarySource"]);
    let ordinary_dst = store.create_node(&["OrdinaryDestination"]);
    let attacker_src = store.create_node(&["AttackerSource"]);
    let attacker_dst = store.create_node(&["AttackerDestination"]);
    let edge = EdgeId::new(91);
    store
        .create_edge_with_id(edge, ordinary_src, ordinary_dst, "ORDINARY")
        .unwrap();

    let receipt = store
        .create_transport_edge_with_id(edge, attacker_src, attacker_dst, "CARRIED")
        .unwrap();
    assert!(receipt.is_none(), "pre-existing ids cannot gain provenance");
    let preserved = store.get_edge(edge).expect("ordinary edge preserved");
    assert_eq!((preserved.src, preserved.dst), (ordinary_src, ordinary_dst));
    assert_eq!(preserved.edge_type.as_str(), "ORDINARY");
    assert_eq!(store.get_edge_history(edge).len(), 1);
}

#[test]
fn clear_rotates_transport_authority_and_discards_pending_delete_state() {
    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    let pending_node = store.create_node(&["Pending"]);
    let edge = EdgeId::new(91);
    let receipt = store
        .create_transport_edge_with_id(edge, src, dst, "CARRIED")
        .unwrap()
        .expect("fresh transport identity");
    let transaction = TransactionId::new(9191);
    assert!(store.delete_edge_transactional(edge, store.current_epoch(), transaction));
    assert!(store.delete_node_transactional(pending_node, store.current_epoch(), transaction));

    store.clear();
    assert!(store.take_pending_edge_deletes(transaction).is_empty());
    assert!(store.take_pending_deletes(transaction).is_empty());

    store.create_node_with_id(src, &["Source"]).unwrap();
    store.create_node_with_id(dst, &["Destination"]).unwrap();
    store
        .create_edge_with_id(edge, src, dst, "REINCARNATED")
        .unwrap();
    store.set_edge_property(edge, "proof", Value::from("preserved"));
    commit_transport_edge_delete(&store, edge, TransactionId::new(9192), EpochId::new(1));
    assert!(!store.is_transport_extract_edge(&receipt));
    assert!(!store.purge_transport_extract_edges(&[&receipt]));
    assert_eq!(store.get_edge_history(edge).len(), 1);
    assert_eq!(
        store
            .get_edge_at_epoch(edge, EpochId::INITIAL)
            .and_then(|edge| edge.get_property("proof").cloned()),
        Some(Value::from("preserved"))
    );
}

#[test]
fn transport_purge_rejects_closed_history_with_open_adjacency() {
    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    let edge = EdgeId::new(92);
    let receipt = store
        .create_transport_edge_with_id(edge, src, dst, "CARRIED")
        .unwrap()
        .expect("fresh transport identity");
    commit_transport_edge_delete(&store, edge, TransactionId::new(9292), EpochId::new(1));

    store.forward_adj.unmark_deleted(src, edge);
    store
        .backward_adj
        .as_ref()
        .expect("default backward adjacency")
        .unmark_deleted(dst, edge);
    assert!(!store.purge_transport_extract_edges(&[&receipt]));
    assert_eq!(store.get_edge_history(edge).len(), 1);
    assert_eq!(store.forward_adj.active_edge_count(), 1);
}

#[test]
fn clear_waits_across_transport_creation_and_receipt_mint() {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    let store = Arc::new(LpgStore::new().unwrap());
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    let edge = EdgeId::new(93);
    let barrier = Arc::new(std::sync::Barrier::new(2));
    *store.transport_create_barrier.write() = Some(Arc::clone(&barrier));

    let creator_store = Arc::clone(&store);
    let creator = thread::spawn(move || {
        creator_store
            .create_transport_edge_with_id(edge, src, dst, "CARRIED")
            .unwrap()
            .expect("fresh transport identity")
    });
    barrier.wait();

    let (attempted_tx, attempted_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    let clear_store = Arc::clone(&store);
    let clearer = thread::spawn(move || {
        attempted_tx.send(()).unwrap();
        clear_store.clear();
        finished_tx.send(()).unwrap();
    });
    attempted_rx.recv().unwrap();
    assert!(
        finished_rx.recv_timeout(Duration::from_millis(50)).is_err(),
        "clear must wait for creation's incarnation read guard"
    );
    barrier.wait();
    let receipt = creator.join().unwrap();
    clearer.join().unwrap();
    *store.transport_create_barrier.write() = None;

    store.create_node_with_id(src, &["Source"]).unwrap();
    store.create_node_with_id(dst, &["Destination"]).unwrap();
    store
        .create_edge_with_id(edge, src, dst, "REINCARNATED")
        .unwrap();
    commit_transport_edge_delete(&store, edge, TransactionId::new(9393), EpochId::new(1));
    assert!(!store.purge_transport_extract_edges(&[&receipt]));
}

#[test]
fn clear_waits_across_transport_purge_validation_and_removal() {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    let store = Arc::new(LpgStore::new().unwrap());
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    let edge = EdgeId::new(94);
    let receipt = Arc::new(
        store
            .create_transport_edge_with_id(edge, src, dst, "CARRIED")
            .unwrap()
            .expect("fresh transport identity"),
    );
    commit_transport_edge_delete(&store, edge, TransactionId::new(9494), EpochId::new(1));
    let barrier = Arc::new(std::sync::Barrier::new(2));
    *store.transport_purge_barrier.write() = Some(Arc::clone(&barrier));

    let purge_store = Arc::clone(&store);
    let purge_receipt = Arc::clone(&receipt);
    let purger =
        thread::spawn(move || purge_store.purge_transport_extract_edges(&[purge_receipt.as_ref()]));
    barrier.wait();

    let (attempted_tx, attempted_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    let clear_store = Arc::clone(&store);
    let clearer = thread::spawn(move || {
        attempted_tx.send(()).unwrap();
        clear_store.clear();
        finished_tx.send(()).unwrap();
    });
    attempted_rx.recv().unwrap();
    assert!(
        finished_rx.recv_timeout(Duration::from_millis(50)).is_err(),
        "clear must wait for purge's incarnation read guard"
    );
    barrier.wait();
    assert!(purger.join().unwrap());
    clearer.join().unwrap();
    *store.transport_purge_barrier.write() = None;

    store.create_node_with_id(src, &["Source"]).unwrap();
    store.create_node_with_id(dst, &["Destination"]).unwrap();
    store
        .create_edge_with_id(edge, src, dst, "REINCARNATED")
        .unwrap();
    commit_transport_edge_delete(&store, edge, TransactionId::new(9495), EpochId::new(1));
    assert!(!store.purge_transport_extract_edges(&[receipt.as_ref()]));
}

#[test]
fn base_only_transport_publication_pins_scope_against_concurrent_seal() {
    use crate::graph::write_permit::WriteAuthority;
    use std::sync::Barrier;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    let store = Arc::new(LpgStore::new().unwrap());
    let barrier = Arc::new(Barrier::new(2));
    *store.transport_purge_barrier.write() = Some(Arc::clone(&barrier));

    let purge_store = Arc::clone(&store);
    let purger = thread::spawn(move || {
        purge_store
            .publish_empty_generation_after_prepare_and_publish_with_rollback(
                |_| Ok::<(), ()>(()),
                || (),
                |()| {},
            )
            .unwrap()
    });
    barrier.wait();

    let (attempted_tx, attempted_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    let seal_store = Arc::clone(&store);
    let sealer = thread::spawn(move || {
        let owner = WriteAuthority::new();
        attempted_tx.send(()).unwrap();
        let sealed = seal_store.seal_unframed_writes(&owner);
        finished_tx.send(sealed).unwrap();
    });
    attempted_rx.recv().unwrap();
    assert!(
        finished_rx.recv_timeout(Duration::from_millis(50)).is_err(),
        "sealing must wait for base-only publication's pinned scope"
    );

    barrier.wait();
    assert!(purger.join().unwrap().is_some());
    assert!(finished_rx.recv().unwrap());
    sealer.join().unwrap();
    *store.transport_purge_barrier.write() = None;
}

#[test]
fn base_only_transport_publication_pins_incarnation_against_clear() {
    use std::sync::Barrier;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    let store = Arc::new(LpgStore::new().unwrap());
    store.create_node(&["Before"]);
    let barrier = Arc::new(Barrier::new(2));
    *store.transport_purge_barrier.write() = Some(Arc::clone(&barrier));

    let purge_store = Arc::clone(&store);
    let purger = thread::spawn(move || {
        purge_store
            .publish_empty_generation_after_prepare_and_publish_with_rollback(
                |_| Ok::<(), ()>(()),
                || (),
                |()| {},
            )
            .unwrap()
    });
    barrier.wait();

    let (attempted_tx, attempted_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    let clear_store = Arc::clone(&store);
    let clearer = thread::spawn(move || {
        attempted_tx.send(()).unwrap();
        clear_store.clear();
        finished_tx.send(()).unwrap();
    });
    attempted_rx.recv().unwrap();
    assert!(
        finished_rx.recv_timeout(Duration::from_millis(50)).is_err(),
        "clear must wait for base-only publication's pinned incarnation"
    );

    barrier.wait();
    assert!(purger.join().unwrap().is_some());
    finished_rx.recv().unwrap();
    clearer.join().unwrap();
    assert_eq!(store.node_count(), 0);
    *store.transport_purge_barrier.write() = None;
}

#[test]
#[cfg(feature = "compact-store")]
fn empty_successor_preserves_representation_scope_and_optional_incarnation() {
    use crate::graph::write_permit::{WriteAuthority, with_authority};

    #[cfg(feature = "text-index")]
    use crate::index::text::{BM25Config, InvertedIndex};
    #[cfg(feature = "vector-index")]
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
    #[cfg(feature = "text-index")]
    use parking_lot::RwLock;

    let config = LpgStoreConfig {
        backward_edges: false,
        initial_node_capacity: 17,
        initial_edge_capacity: 31,
    };
    let store = LpgStore::with_config(config.clone()).unwrap();
    #[cfg(feature = "vector-index")]
    let retained_vector = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
        2,
        DistanceMetric::Euclidean,
    ))));
    #[cfg(feature = "vector-index")]
    store.add_vector_index("Document", "embedding", Arc::clone(&retained_vector));
    #[cfg(feature = "text-index")]
    let retained_text = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    #[cfg(feature = "text-index")]
    store.add_text_index("Document", "body", Arc::clone(&retained_text));
    store.sync_epoch(EpochId::new(91));
    store.set_next_node_id(92);
    store.set_next_edge_id(93);
    let original_authority = Arc::clone(&store.transport_extract_authority.read());
    let owner = WriteAuthority::new();
    assert!(store.seal_unframed_writes(&owner));

    let mut preserved = None;
    let mut rotated = None;
    assert!(with_authority(&owner, || {
        store
            .publish_empty_generation_after_prepare_and_publish_with_rollback(
                |scope| {
                    preserved = Some(scope.prepare_same_incarnation_empty_successor()?);
                    rotated = Some(scope.prepare_fresh_incarnation_empty_successor()?);
                    Ok::<(), grafeo_common::memory::arena::AllocError>(())
                },
                || (),
                |()| {},
            )
            .expect("successor allocation")
            .is_some()
    }));
    let preserved = preserved.expect("representation-preserving successor");
    let rotated = rotated.expect("fresh-incarnation successor");

    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    {
        assert_eq!(preserved.index_owner_id, store.index_owner_id);
        assert_eq!(*preserved.index_slots.lock(), *store.index_slots.lock());
        assert_ne!(rotated.index_owner_id, store.index_owner_id);
        assert!(rotated.index_slots.lock().is_empty());
    }

    for successor in [&preserved, &rotated] {
        assert!(!successor.has_backward_adjacency());
        assert_eq!(
            successor.representation_config.backward_edges,
            config.backward_edges
        );
        assert_eq!(
            successor.representation_config.initial_node_capacity,
            config.initial_node_capacity,
        );
        assert_eq!(
            successor.representation_config.initial_edge_capacity,
            config.initial_edge_capacity,
        );
        assert_eq!(successor.current_epoch(), EpochId::new(91));
        assert_eq!(successor.next_node_id(), 92);
        assert_eq!(successor.next_edge_id(), 93);
        assert!(
            !successor.create_node(&["unauthorized"]).is_valid(),
            "the exact mutation seal must survive representation replacement"
        );
    }
    assert!(Arc::ptr_eq(
        &original_authority,
        &preserved.transport_extract_authority.read(),
    ));
    assert!(!Arc::ptr_eq(
        &original_authority,
        &rotated.transport_extract_authority.read(),
    ));
    assert!(with_authority(&owner, || {
        preserved.create_node(&["authorized"]).is_valid()
    }));
    #[cfg(feature = "vector-index")]
    with_authority(&owner, || {
        preserved.add_vector_index("Document", "embedding", Arc::clone(&retained_vector));
        rotated.add_vector_index("Document", "embedding", Arc::clone(&retained_vector));
    });
    #[cfg(feature = "vector-index")]
    {
        assert!(
            preserved
                .get_vector_index("Document", "embedding")
                .is_some()
        );
        assert!(rotated.get_vector_index("Document", "embedding").is_none());
    }
    #[cfg(feature = "text-index")]
    with_authority(&owner, || {
        preserved.add_text_index("Document", "body", Arc::clone(&retained_text));
        rotated.add_text_index("Document", "body", Arc::clone(&retained_text));
    });
    #[cfg(feature = "text-index")]
    {
        assert!(preserved.get_text_index("Document", "body").is_some());
        assert!(rotated.get_text_index("Document", "body").is_none());
    }

    with_authority(&owner, || store.clear());
    assert!(!Arc::ptr_eq(
        &original_authority,
        &store.transport_extract_authority.read(),
    ));
    assert!(Arc::ptr_eq(
        &original_authority,
        &preserved.transport_extract_authority.read(),
    ));
}

#[test]
#[cfg(feature = "compact-store")]
fn prepared_node_labels_rollback_restores_exact_catalog_cardinality() {
    use arcstr::ArcStr;

    let store = LpgStore::new().unwrap();
    store.create_node(&["Existing"]);
    let registry_len = store.label_registry.read().len();
    let existing_id = store.label_registry.read().get_id("Existing").unwrap();
    let index_len = store.label_index.read().len();
    let versions = vec![
        (
            EpochId::new(1),
            vec![ArcStr::from("Existing"), ArcStr::from("Transient")],
        ),
        (
            EpochId::new(2),
            vec![ArcStr::from("Transient"), ArcStr::from("Second")],
        ),
    ];

    {
        let prepared = store.prepare_node_labels(&versions).unwrap();
        assert_eq!(prepared.id("Existing"), Some(existing_id));
        assert!(prepared.id("Transient").is_some());
        assert!(prepared.id("Second").is_some());
        assert_eq!(store.label_registry.read().len(), registry_len + 2);
        assert_eq!(store.label_index.read().len(), registry_len + 2);
    }
    assert_eq!(store.label_registry.read().len(), registry_len);
    assert_eq!(store.label_index.read().len(), index_len);
    assert!(store.label_registry.read().get_id("Transient").is_none());
    assert!(store.label_registry.read().get_id("Second").is_none());

    let prepared = store.prepare_node_labels(&versions).unwrap();
    prepared.commit();
    assert_eq!(store.label_registry.read().len(), registry_len + 2);
    assert!(store.label_registry.read().get_id("Transient").is_some());
    assert!(store.label_registry.read().get_id("Second").is_some());
}

#[test]
#[cfg(feature = "compact-store")]
fn prepared_promotion_edge_type_rollback_restores_exact_catalog_cardinality() {
    let store = LpgStore::new().unwrap();
    {
        let prepared = store.prepare_edge_type_for_promotion("TRANSIENT").unwrap();
        assert_eq!(prepared.id(), 0);
        assert_eq!(store.edge_type_to_id.read().get("TRANSIENT"), Some(&0));
        assert_eq!(store.id_to_edge_type.read().len(), 1);
        assert_eq!(&*store.edge_type_live_counts.read(), &[0]);
    }
    assert!(store.edge_type_to_id.read().is_empty());
    assert!(store.id_to_edge_type.read().is_empty());
    assert!(store.edge_type_live_counts.read().is_empty());

    store
        .prepare_edge_type_for_promotion("PERSISTENT")
        .unwrap()
        .commit();
    assert_eq!(store.edge_type_to_id.read().get("PERSISTENT"), Some(&0));
    assert_eq!(store.id_to_edge_type.read().len(), 1);
    assert_eq!(&*store.edge_type_live_counts.read(), &[0]);

    // An existing row is never owned by the preparation and therefore remains
    // intact when a later unpublished promotion aborts.
    drop(store.prepare_edge_type_for_promotion("PERSISTENT").unwrap());
    assert_eq!(store.edge_type_to_id.read().get("PERSISTENT"), Some(&0));
}

#[test]
fn transactional_delete_can_close_a_same_transaction_node_at_commit() {
    let store = LpgStore::new().unwrap();
    let transaction = TransactionId::new(4041);
    let commit_epoch = EpochId::new(41);
    let id = store.create_node_versioned(&["Draft"], EpochId::PENDING, transaction);
    store.set_node_property_buffered(id, "title", Value::from("ephemeral"), transaction);
    store.add_label_buffered(id, "Reviewed", transaction);

    assert!(
        store.delete_node_transactional(id, EpochId::INITIAL, transaction),
        "a transaction must be able to delete the node it just created"
    );
    assert!(
        store
            .get_node_versioned(id, EpochId::INITIAL, transaction)
            .is_none(),
        "the creating transaction must read its delete"
    );

    let (pending_nodes, pending_edges) = store.take_pending_creates(transaction);
    assert_eq!(pending_nodes, vec![id]);
    assert!(pending_edges.is_empty());
    store.finalize_entities_by_id(transaction, commit_epoch, &pending_nodes, &pending_edges);
    store.apply_tx_overlay(transaction);
    let pending_deletes = store.take_pending_deletes(transaction);
    assert_eq!(pending_deletes, vec![id]);
    store.finalize_deletes_by_id(transaction, commit_epoch, &pending_deletes);

    assert!(store.get_node(id).is_none());
    assert!(store.get_node_at_epoch(id, commit_epoch).is_none());
    let history = store.get_node_history(id);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].0, commit_epoch);
    assert_eq!(history[0].1, Some(commit_epoch));
    assert_eq!(
        store.node_property_history_for_key(id, "title"),
        vec![
            (commit_epoch, Value::from("ephemeral")),
            (commit_epoch, Value::Null),
        ]
    );
    assert_eq!(
        store
            .node_label_history(id)
            .into_iter()
            .map(|(epoch, labels)| {
                (
                    epoch,
                    labels
                        .into_iter()
                        .map(|label| label.to_string())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (commit_epoch, vec![String::from("Draft")]),
            (
                commit_epoch,
                vec![String::from("Draft"), String::from("Reviewed")],
            ),
        ]
    );
}

#[test]
fn exact_history_recovery_restores_closed_and_live_identities_atomically() {
    use arcstr::ArcStr;

    let store = LpgStore::new().unwrap();
    let e5 = EpochId::new(5);
    let e6 = EpochId::new(6);
    let e8 = EpochId::new(8);
    let e10 = EpochId::new(10);
    let e12 = EpochId::new(12);
    let e14 = EpochId::new(14);
    let e20 = EpochId::new(20);
    let e30 = EpochId::new(30);
    let gone = NodeId::new(42);
    let anchor = NodeId::new(43);
    let contiguous = NodeId::new(46);
    let live = NodeId::new(50);
    let closed_edge = EdgeId::new(99);
    let live_edge = EdgeId::new(100);

    store
        .restore_node_history_exact(
            gone,
            &[(e5, Some(e8)), (e10, Some(e20))],
            &[
                (e5, vec![ArcStr::from("Prototype")]),
                (e8, vec![ArcStr::from("InvisibleAtDelete")]),
                (e10, vec![ArcStr::from("Person")]),
                (
                    e12,
                    vec![ArcStr::from("Person"), ArcStr::from("Researcher")],
                ),
                (e14, vec![ArcStr::from("Researcher")]),
            ],
        )
        .expect("restore deleted node history");
    store
        .restore_node_history_exact(anchor, &[(e5, None)], &[(e5, vec![ArcStr::from("Anchor")])])
        .expect("restore live anchor");
    store
        .restore_node_history_exact(
            contiguous,
            &[(e5, Some(e8)), (e8, Some(e20))],
            &[
                (e5, vec![ArcStr::from("FirstLife")]),
                (e8, vec![ArcStr::from("InvisibleAtDelete")]),
                (e8, vec![ArcStr::from("SecondLife")]),
            ],
        )
        .expect("restore contiguous lifetimes with an ordered boundary state");
    store.set_node_property_at_epoch(gone, "name", Value::from("Draft"), e5);
    store.set_node_property_at_epoch(gone, "name", Value::Null, e8);
    store.set_node_property_at_epoch(gone, "name", Value::from("Alix"), e10);
    store.set_node_property_at_epoch(gone, "name", Value::Null, e20);

    store
        .restore_edge_history_exact(
            closed_edge,
            gone,
            anchor,
            "KNOWS",
            &[(e6, Some(e8)), (e10, Some(e20))],
        )
        .expect("restore deleted edge history");
    store.set_edge_property_at_epoch(closed_edge, "weight", Value::Int64(1), e6);
    store.set_edge_property_at_epoch(closed_edge, "weight", Value::Null, e8);
    store.set_edge_property_at_epoch(closed_edge, "weight", Value::Int64(2), e10);
    store.set_edge_property_at_epoch(closed_edge, "weight", Value::Null, e20);

    assert!(store.get_node(gone).is_none());
    assert!(
        store
            .get_node_at_epoch(gone, e6)
            .unwrap()
            .has_label("Prototype")
    );
    assert!(store.get_node_at_epoch(gone, EpochId::new(9)).is_none());
    assert!(
        store
            .get_node_at_epoch(gone, e10)
            .unwrap()
            .has_label("Person")
    );
    let at12 = store.get_node_at_epoch(gone, e12).unwrap();
    assert!(at12.has_label("Person") && at12.has_label("Researcher"));
    let at14 = store.get_node_at_epoch(gone, e14).unwrap();
    assert!(!at14.has_label("Person") && at14.has_label("Researcher"));
    assert!(store.get_node_at_epoch(gone, e20).is_none());
    let contiguous_at_boundary = store.get_node_at_epoch(contiguous, e8).unwrap();
    assert!(contiguous_at_boundary.has_label("SecondLife"));
    assert!(!contiguous_at_boundary.has_label("InvisibleAtDelete"));
    assert_eq!(
        store.node_label_history(contiguous),
        vec![
            (e5, vec![ArcStr::from("FirstLife")]),
            (e8, vec![ArcStr::from("InvisibleAtDelete")]),
            (e8, vec![ArcStr::from("SecondLife")]),
        ]
    );
    assert!(store.nodes_by_label("Prototype").is_empty());
    assert!(store.nodes_by_label("Researcher").is_empty());
    assert_eq!(store.nodes_by_label("Anchor"), vec![anchor]);
    assert_eq!(
        store
            .node_label_history(gone)
            .into_iter()
            .map(|(epoch, labels)| {
                (
                    epoch,
                    labels.iter().map(ToString::to_string).collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (e5, vec![String::from("Prototype")]),
            (e8, vec![String::from("InvisibleAtDelete")]),
            (e10, vec![String::from("Person")]),
            (
                e12,
                vec![String::from("Person"), String::from("Researcher")],
            ),
            (e14, vec![String::from("Researcher")]),
        ]
    );
    let node_history = store.get_node_history(gone);
    assert_eq!(node_history.len(), 2);
    assert_eq!((node_history[0].0, node_history[0].1), (e10, Some(e20)));
    assert_eq!((node_history[1].0, node_history[1].1), (e5, Some(e8)));

    assert!(store.get_edge(closed_edge).is_none());
    assert!(
        store
            .get_edge_at_epoch(closed_edge, EpochId::new(7))
            .is_some()
    );
    assert!(
        store
            .get_edge_at_epoch(closed_edge, EpochId::new(9))
            .is_none()
    );
    assert!(store.get_edge_at_epoch(closed_edge, e10).is_some());
    assert!(store.get_edge_at_epoch(closed_edge, e20).is_none());
    let edge_history = store.get_edge_history(closed_edge);
    assert_eq!(edge_history.len(), 2);
    assert_eq!((edge_history[0].0, edge_history[0].1), (e10, Some(e20)));
    assert_eq!((edge_history[1].0, edge_history[1].1), (e6, Some(e8)));

    // Invalid structure and labels are rejected before installing an identity.
    let invalid_node = NodeId::new(44);
    let overlap = store.restore_node_history_exact(
        invalid_node,
        &[(e5, Some(e10)), (e8, None)],
        &[
            (e5, vec![ArcStr::from("Invalid")]),
            (e8, vec![ArcStr::from("Invalid")]),
        ],
    );
    assert!(overlap.is_err());
    assert!(!store.all_node_ids().contains(&invalid_node));
    let outside = store.restore_node_history_exact(
        NodeId::new(45),
        &[(e5, Some(e10))],
        &[(e5, vec![ArcStr::from("Invalid")]), (e12, Vec::new())],
    );
    assert!(outside.is_err());
    assert!(!store.all_node_ids().contains(&NodeId::new(45)));
    assert!(
        store
            .restore_edge_history_exact(
                EdgeId::new(98),
                NodeId::new(999),
                anchor,
                "MISSING",
                &[(e5, None)],
            )
            .is_err()
    );
    assert!(!store.all_known_edge_ids().contains(&EdgeId::new(98)));
    assert!(
        store
            .restore_node_history_exact(
                gone,
                &[(e5, Some(e8))],
                &[(e5, vec![ArcStr::from("Duplicate")])],
            )
            .is_err()
    );
    assert_eq!(store.get_node_history(gone).len(), 2);

    // Open restores publish current label/adjacency projections and counters.
    store
        .restore_node_history_exact(live, &[(e30, None)], &[(e30, vec![ArcStr::from("Live")])])
        .expect("restore current node");
    store
        .restore_edge_history_exact(live_edge, anchor, live, "LINK", &[(e30, None)])
        .expect("restore current edge");
    assert_eq!(store.node_count(), 2);
    assert_eq!(store.edge_count(), 1);
    assert_eq!(store.nodes_by_label("Live"), vec![live]);
    assert_eq!(
        store
            .edges_from(anchor, Direction::Outgoing)
            .collect::<Vec<_>>(),
        vec![(live, live_edge)]
    );
    assert!(store.next_node_id() > live.as_u64());
    assert!(store.next_edge_id() > live_edge.as_u64());
}

#[test]
fn exact_history_restore_retains_zero_width_lifetimes_and_rejects_inversion_atomically() {
    use arcstr::ArcStr;

    let store = LpgStore::new().unwrap();
    let committed = EpochId::new(70);
    let before = EpochId::new(69);
    let after = EpochId::new(71);
    let ephemeral = NodeId::new(70);
    let anchor = NodeId::new(71);
    let ephemeral_edge = EdgeId::new(170);

    store
        .restore_node_history_exact(
            ephemeral,
            &[(committed, Some(committed))],
            &[
                (committed, vec![ArcStr::from("Draft")]),
                (
                    committed,
                    vec![ArcStr::from("Draft"), ArcStr::from("Reviewed")],
                ),
            ],
        )
        .expect("restore an exact zero-width node lifetime");
    store
        .restore_node_history_exact(
            anchor,
            &[(committed, None)],
            &[(committed, vec![ArcStr::from("Anchor")])],
        )
        .expect("restore the live edge endpoint");
    store.set_node_property_at_epoch(ephemeral, "title", Value::from("ephemeral"), committed);
    store.set_node_property_at_epoch(ephemeral, "title", Value::Null, committed);
    store
        .restore_edge_history_exact(
            ephemeral_edge,
            ephemeral,
            anchor,
            "TEMPORARY",
            &[(committed, Some(committed))],
        )
        .expect("restore an exact zero-width edge lifetime");
    store.set_edge_property_at_epoch(ephemeral_edge, "weight", Value::Int64(7), committed);
    store.set_edge_property_at_epoch(ephemeral_edge, "weight", Value::Null, committed);

    for epoch in [before, committed, after] {
        assert!(
            store.get_node_at_epoch(ephemeral, epoch).is_none(),
            "[C,C) node must be invisible at epoch {}",
            epoch.as_u64()
        );
        assert!(
            store.get_edge_at_epoch(ephemeral_edge, epoch).is_none(),
            "[C,C) edge must be invisible at epoch {}",
            epoch.as_u64()
        );
    }
    assert!(store.get_node(ephemeral).is_none());
    assert!(store.get_edge(ephemeral_edge).is_none());
    assert!(store.all_node_ids().contains(&ephemeral));
    assert!(store.all_known_edge_ids().contains(&ephemeral_edge));
    assert_eq!(
        store
            .get_node_history(ephemeral)
            .into_iter()
            .map(|(created, deleted, _)| (created, deleted))
            .collect::<Vec<_>>(),
        vec![(committed, Some(committed))]
    );
    let edge_history = store.get_edge_history(ephemeral_edge);
    assert_eq!(edge_history.len(), 1);
    assert_eq!(
        (edge_history[0].0, edge_history[0].1),
        (committed, Some(committed))
    );
    assert_eq!(edge_history[0].2.src, ephemeral);
    assert_eq!(edge_history[0].2.dst, anchor);
    assert_eq!(edge_history[0].2.edge_type.as_str(), "TEMPORARY");
    assert_eq!(
        store.node_label_history(ephemeral),
        vec![
            (committed, vec![ArcStr::from("Draft")]),
            (
                committed,
                vec![ArcStr::from("Draft"), ArcStr::from("Reviewed")],
            ),
        ]
    );
    assert_eq!(
        store.node_property_history_for_key(ephemeral, "title"),
        vec![
            (committed, Value::from("ephemeral")),
            (committed, Value::Null),
        ]
    );
    assert_eq!(
        store.edge_property_history(ephemeral_edge),
        vec![(
            "weight".into(),
            vec![(committed, Value::Int64(7)), (committed, Value::Null)],
        )]
    );

    let nodes_before = store.all_node_ids();
    let edges_before = store.all_known_edge_ids();
    let node_high_water_before = store.peek_next_node_id();
    let edge_high_water_before = store.peek_next_edge_id();
    let epoch_before = store.current_epoch();
    let inverted_node = NodeId::new(72);
    let inverted_edge = EdgeId::new(171);

    assert!(
        store
            .restore_node_history_exact(
                inverted_node,
                &[(committed, Some(before))],
                &[(committed, vec![ArcStr::from("Malformed")])],
            )
            .is_err()
    );
    assert!(
        store
            .restore_edge_history_exact(
                inverted_edge,
                ephemeral,
                anchor,
                "MALFORMED",
                &[(committed, Some(before))],
            )
            .is_err()
    );
    assert_eq!(store.all_node_ids(), nodes_before);
    assert_eq!(store.all_known_edge_ids(), edges_before);
    assert_eq!(store.peek_next_node_id(), node_high_water_before);
    assert_eq!(store.peek_next_edge_id(), edge_high_water_before);
    assert_eq!(store.current_epoch(), epoch_before);
    assert!(store.get_node_history(inverted_node).is_empty());
    assert!(store.get_edge_history(inverted_edge).is_empty());
}

#[test]
fn delete_node_cleans_property_index_membership() {
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Person"]);
    let name = Value::from("Alix");
    store.set_node_property(node, "name", name.clone());
    store.create_property_index("name");
    assert_eq!(store.find_nodes_by_property("name", &name), vec![node]);

    assert!(store.delete_node(node));
    assert!(
        store.find_nodes_by_property("name", &name).is_empty(),
        "deleting a node must remove its ordinary property-index entry"
    );
}

#[cfg(feature = "vector-index")]
#[test]
fn delete_node_cleans_vector_index_membership() {
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
    use std::sync::Arc;

    let store = LpgStore::new().unwrap();
    let index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
        3,
        DistanceMetric::Cosine,
    ))));
    store.add_vector_index("Doc", "embedding", Arc::clone(&index));
    let node = store.create_node(&["Doc"]);
    store.set_node_property(
        node,
        "embedding",
        Value::Vector(vec![1.0_f32, 0.0, 0.0].into()),
    );
    assert!(index.contains(node));

    assert!(store.delete_node(node));
    assert!(
        !index.contains(node),
        "deleting a node must remove its HNSW entry"
    );
}

#[cfg(feature = "vector-index")]
#[test]
fn finalized_transactional_delete_cleans_indexes_but_rollback_preserves_them() {
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
    use grafeo_common::types::EpochId;
    use std::sync::Arc;

    let store = LpgStore::new().unwrap();
    let index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
        3,
        DistanceMetric::Cosine,
    ))));
    store.add_vector_index("Doc", "embedding", Arc::clone(&index));
    let node = store.create_node(&["Doc"]);
    let name = Value::from("transactional-delete");
    store.set_node_property(node, "name", name.clone());
    store.set_node_property(
        node,
        "embedding",
        Value::Vector(vec![1.0_f32, 0.0, 0.0].into()),
    );
    store.create_property_index("name");

    let rolled_back = TransactionId::new(80);
    assert!(store.delete_node_transactional(node, store.current_epoch(), rolled_back));
    assert!(index.contains(node));
    assert_eq!(store.find_nodes_by_property("name", &name), vec![node]);
    store.rollback_pending_deletes(rolled_back, &[node]);
    assert!(index.contains(node));
    assert_eq!(store.find_nodes_by_property("name", &name), vec![node]);

    let committed = TransactionId::new(81);
    assert!(store.delete_node_transactional(node, store.current_epoch(), committed));
    let commit_epoch = EpochId::new(1);
    store.sync_epoch(commit_epoch);
    store.finalize_deletes_by_id(committed, commit_epoch, &[node]);
    assert!(!index.contains(node));
    assert!(store.find_nodes_by_property("name", &name).is_empty());
}

#[cfg(feature = "text-index")]
#[test]
fn label_and_delete_centrally_maintain_text_index() {
    use crate::index::text::{BM25Config, InvertedIndex};
    use parking_lot::RwLock;
    use std::sync::Arc;

    let store = LpgStore::new().unwrap();
    let index = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    store.add_text_index("Article", "content", Arc::clone(&index));
    let node = store.create_node(&["Other"]);
    store.set_node_property(node, "content", Value::from("central text maintenance"));
    assert!(index.read().search("central", 10).is_empty());

    assert!(store.add_label(node, "Article"));
    assert_eq!(index.read().search("central", 10)[0].0, node);

    assert!(store.remove_label(node, "Article"));
    assert!(index.read().search("central", 10).is_empty());

    assert!(store.add_label(node, "Article"));
    assert!(store.delete_node(node));
    assert!(index.read().search("central", 10).is_empty());
}

#[cfg(all(feature = "vector-index", feature = "text-index"))]
#[test]
fn compound_create_publishes_indexes_only_for_system_writes() {
    use crate::index::text::{BM25Config, InvertedIndex};
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
    use parking_lot::RwLock;
    use std::sync::Arc;

    let store = LpgStore::new().unwrap();
    store.create_property_index("name");
    let vector_index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
        3,
        DistanceMetric::Cosine,
    ))));
    store.add_vector_index("Doc", "embedding", Arc::clone(&vector_index));
    let text_index = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    store.add_text_index("Doc", "content", Arc::clone(&text_index));

    let committed_name = Value::from("committed-compound");
    let committed = store.create_node_with_props(
        &["Doc"],
        [
            ("name", committed_name.clone()),
            ("embedding", Value::Vector(vec![1.0_f32, 0.0, 0.0].into())),
            ("content", Value::from("committed compound text")),
        ],
    );
    assert_eq!(
        store.find_nodes_by_property("name", &committed_name),
        vec![committed]
    );
    assert!(vector_index.contains(committed));
    assert_eq!(text_index.read().search("compound", 10)[0].0, committed);

    let pending_name = Value::from("pending-compound");
    let pending = store.create_node_with_props_versioned(
        &["Doc"],
        [
            ("name", pending_name.clone()),
            ("embedding", Value::Vector(vec![0.0_f32, 1.0, 0.0].into())),
            ("content", Value::from("uncommitted private words")),
        ],
        store.current_epoch(),
        TransactionId::new(91),
    );
    assert!(
        !store
            .find_nodes_by_property("name", &pending_name)
            .contains(&pending)
    );
    assert!(!vector_index.contains(pending));
    assert!(text_index.read().search("private", 10).is_empty());
}

#[test]
fn test_create_edge() {
    let store = LpgStore::new().unwrap();

    let alix = store.create_node(&["Person"]);
    let gus = store.create_node(&["Person"]);

    let edge_id = store.create_edge(alix, gus, "KNOWS");
    assert!(edge_id.is_valid());

    let edge = store.get_edge(edge_id).unwrap();
    assert_eq!(edge.src, alix);
    assert_eq!(edge.dst, gus);
    assert_eq!(edge.edge_type.as_str(), "KNOWS");
}

#[test]
fn test_neighbors() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "KNOWS");

    let outgoing: Vec<_> = store.neighbors(a, Direction::Outgoing).collect();
    assert_eq!(outgoing.len(), 2);
    assert!(outgoing.contains(&b));
    assert!(outgoing.contains(&c));

    let incoming: Vec<_> = store.neighbors(b, Direction::Incoming).collect();
    assert_eq!(incoming.len(), 1);
    assert!(incoming.contains(&a));
}

#[test]
fn test_nodes_by_label() {
    let store = LpgStore::new().unwrap();

    let p1 = store.create_node(&["Person"]);
    let p2 = store.create_node(&["Person"]);
    let _a = store.create_node(&["Animal"]);

    let persons = store.nodes_by_label("Person");
    assert_eq!(persons.len(), 2);
    assert!(persons.contains(&p1));
    assert!(persons.contains(&p2));

    let animals = store.nodes_by_label("Animal");
    assert_eq!(animals.len(), 1);
}

#[test]
fn test_delete_edge() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let edge_id = store.create_edge(a, b, "KNOWS");

    assert_eq!(store.edge_count(), 1);

    assert!(store.delete_edge(edge_id));
    assert_eq!(store.edge_count(), 0);
    assert!(store.get_edge(edge_id).is_none());
}

// === New tests for improved coverage ===

#[test]
fn test_lpg_store_config() {
    // Test with_config
    let config = LpgStoreConfig {
        backward_edges: false,
        initial_node_capacity: 100,
        initial_edge_capacity: 200,
    };
    let store = LpgStore::with_config(config).unwrap();

    // Store should work but without backward adjacency
    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    store.create_edge(a, b, "KNOWS");

    // Outgoing should work
    let outgoing: Vec<_> = store.neighbors(a, Direction::Outgoing).collect();
    assert_eq!(outgoing.len(), 1);

    // Incoming should be empty (no backward adjacency)
    let incoming: Vec<_> = store.neighbors(b, Direction::Incoming).collect();
    assert_eq!(incoming.len(), 0);
}

#[test]
fn test_epoch_management() {
    let store = LpgStore::new().unwrap();

    let epoch0 = store.current_epoch();
    assert_eq!(epoch0.as_u64(), 0);

    let epoch1 = store.new_epoch();
    assert_eq!(epoch1.as_u64(), 1);

    let current = store.current_epoch();
    assert_eq!(current.as_u64(), 1);
}

#[test]
fn test_node_properties() {
    let store = LpgStore::new().unwrap();
    let id = store.create_node(&["Person"]);

    // Set and get property
    store.set_node_property(id, "name", Value::from("Alix"));
    let name = store.get_node_property(id, &"name".into());
    assert!(matches!(name, Some(Value::String(s)) if s.as_str() == "Alix"));

    // Update property
    store.set_node_property(id, "name", Value::from("Gus"));
    let name = store.get_node_property(id, &"name".into());
    assert!(matches!(name, Some(Value::String(s)) if s.as_str() == "Gus"));

    // Remove property
    let old = store.remove_node_property(id, "name");
    assert!(matches!(old, Some(Value::String(s)) if s.as_str() == "Gus"));

    // Property should be gone
    let name = store.get_node_property(id, &"name".into());
    assert!(name.is_none());

    // Remove non-existent property
    let none = store.remove_node_property(id, "nonexistent");
    assert!(none.is_none());
}

#[test]
fn test_edge_properties() {
    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let edge_id = store.create_edge(a, b, "KNOWS");

    // Set and get property
    store.set_edge_property(edge_id, "since", Value::from(2020i64));
    let since = store.get_edge_property(edge_id, &"since".into());
    assert_eq!(since.and_then(|v| v.as_int64()), Some(2020));

    // Remove property
    let old = store.remove_edge_property(edge_id, "since");
    assert_eq!(old.and_then(|v| v.as_int64()), Some(2020));

    let since = store.get_edge_property(edge_id, &"since".into());
    assert!(since.is_none());
}

#[test]
fn test_add_remove_label() {
    let store = LpgStore::new().unwrap();
    let id = store.create_node(&["Person"]);

    // Add new label
    assert!(store.add_label(id, "Employee"));

    let node = store.get_node(id).unwrap();
    assert!(node.has_label("Person"));
    assert!(node.has_label("Employee"));

    // Adding same label again should fail
    assert!(!store.add_label(id, "Employee"));

    // Remove label
    assert!(store.remove_label(id, "Employee"));

    let node = store.get_node(id).unwrap();
    assert!(node.has_label("Person"));
    assert!(!node.has_label("Employee"));

    // Removing non-existent label should fail
    assert!(!store.remove_label(id, "Employee"));
    assert!(!store.remove_label(id, "NonExistent"));
}

#[test]
fn test_add_label_to_nonexistent_node() {
    let store = LpgStore::new().unwrap();
    let fake_id = NodeId::new(999);
    assert!(!store.add_label(fake_id, "Label"));
}

#[test]
fn test_remove_label_from_nonexistent_node() {
    let store = LpgStore::new().unwrap();
    let fake_id = NodeId::new(999);
    assert!(!store.remove_label(fake_id, "Label"));
}

#[test]
fn test_node_ids() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    let n3 = store.create_node(&["Person"]);

    let ids = store.node_ids();
    assert_eq!(ids.len(), 3);
    assert!(ids.contains(&n1));
    assert!(ids.contains(&n2));
    assert!(ids.contains(&n3));

    // Delete one
    store.delete_node(n2);
    let ids = store.node_ids();
    assert_eq!(ids.len(), 2);
    assert!(!ids.contains(&n2));
}

#[test]
fn test_delete_node_nonexistent() {
    let store = LpgStore::new().unwrap();
    let fake_id = NodeId::new(999);
    assert!(!store.delete_node(fake_id));
}

#[test]
fn test_delete_edge_nonexistent() {
    let store = LpgStore::new().unwrap();
    let fake_id = EdgeId::new(999);
    assert!(!store.delete_edge(fake_id));
}

#[test]
fn test_delete_edge_double() {
    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let edge_id = store.create_edge(a, b, "KNOWS");

    assert!(store.delete_edge(edge_id));
    assert!(!store.delete_edge(edge_id)); // Double delete
}

#[test]
fn test_create_edge_with_props() {
    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);

    let edge_id = store.create_edge_with_props(
        a,
        b,
        "KNOWS",
        [
            ("since", Value::from(2020i64)),
            ("weight", Value::from(1.0)),
        ],
    );

    let edge = store.get_edge(edge_id).unwrap();
    assert_eq!(
        edge.get_property("since").and_then(|v| v.as_int64()),
        Some(2020)
    );
    assert_eq!(
        edge.get_property("weight").and_then(|v| v.as_float64()),
        Some(1.0)
    );
}

#[test]
fn test_delete_node_edges() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS"); // a -> b
    store.create_edge(c, a, "KNOWS"); // c -> a

    assert_eq!(store.edge_count(), 2);

    // Delete all edges connected to a
    store.delete_node_edges(a);

    assert_eq!(store.edge_count(), 0);
}

#[test]
fn test_delete_node_edges_self_loop() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let _e = store.create_edge(a, a, "SELF"); // self-loop

    assert_eq!(store.edge_count(), 1);

    // Self-loop appears in both outgoing and incoming scans.
    // The fix deduplicates via HashSet, so only one delete happens.
    store.delete_node_edges(a);

    assert_eq!(store.edge_count(), 0);
}

#[test]
fn test_delete_node_edges_self_loop_plus_others() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, a, "SELF"); // self-loop on a
    store.create_edge(a, b, "KNOWS"); // outgoing from a
    store.create_edge(c, a, "KNOWS"); // incoming to a
    store.create_edge(b, c, "KNOWS"); // unrelated

    assert_eq!(store.edge_count(), 4);

    store.delete_node_edges(a);

    // Only the b->c edge should remain
    assert_eq!(store.edge_count(), 1);
}

#[test]
fn test_delete_node_edges_atomic_batch() {
    use std::sync::Arc;

    let store = Arc::new(LpgStore::new().unwrap());

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);
    let d = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "KNOWS");
    store.create_edge(d, a, "KNOWS");

    assert_eq!(store.edge_count(), 3);

    // Spawn a reader thread that checks edge count.
    // With batch locking, the reader should never see 1 or 2
    // (partially deleted): it should see either 3 (before) or 0 (after).
    //
    // A barrier ensures both threads start at the same time, and an
    // AtomicBool keeps the reader spinning until deletion finishes,
    // so the two threads are guaranteed to overlap.
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, Ordering};

    let barrier = Arc::new(Barrier::new(2));
    let done = Arc::new(AtomicBool::new(false));

    let reader = Arc::clone(&store);
    let reader_barrier = Arc::clone(&barrier);
    let reader_done = Arc::clone(&done);

    let handle = std::thread::spawn(move || {
        let mut saw_partial = false;
        reader_barrier.wait();
        while !reader_done.load(Ordering::Acquire) {
            let count = reader.edge_count();
            if count != 0 && count != 3 {
                saw_partial = true;
                break;
            }
        }
        saw_partial
    });

    barrier.wait();
    store.delete_node_edges(a);
    done.store(true, Ordering::Release);

    let saw_partial = handle.join().unwrap();
    assert!(
        !saw_partial,
        "concurrent reader observed partially deleted edges"
    );
    assert_eq!(store.edge_count(), 0);
}

#[test]
fn test_neighbors_both_directions() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS"); // a -> b
    store.create_edge(c, a, "KNOWS"); // c -> a

    // Direction::Both for node a
    let neighbors: Vec<_> = store.neighbors(a, Direction::Both).collect();
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&b)); // outgoing
    assert!(neighbors.contains(&c)); // incoming
}

#[test]
fn test_edges_from() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    let e1 = store.create_edge(a, b, "KNOWS");
    let e2 = store.create_edge(a, c, "KNOWS");

    let edges: Vec<_> = store.edges_from(a, Direction::Outgoing).collect();
    assert_eq!(edges.len(), 2);
    assert!(edges.iter().any(|(_, e)| *e == e1));
    assert!(edges.iter().any(|(_, e)| *e == e2));

    // Incoming edges to b
    let incoming: Vec<_> = store.edges_from(b, Direction::Incoming).collect();
    assert_eq!(incoming.len(), 1);
    assert_eq!(incoming[0].1, e1);
}

#[test]
fn test_edges_to() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    let e1 = store.create_edge(a, b, "KNOWS");
    let e2 = store.create_edge(c, b, "KNOWS");

    // Edges pointing TO b
    let to_b = store.edges_to(b);
    assert_eq!(to_b.len(), 2);
    assert!(to_b.iter().any(|(src, e)| *src == a && *e == e1));
    assert!(to_b.iter().any(|(src, e)| *src == c && *e == e2));
}

#[test]
fn test_out_degree_in_degree() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "KNOWS");
    store.create_edge(c, b, "KNOWS");

    assert_eq!(store.out_degree(a), 2);
    assert_eq!(store.out_degree(b), 0);
    assert_eq!(store.out_degree(c), 1);

    assert_eq!(store.in_degree(a), 0);
    assert_eq!(store.in_degree(b), 2);
    assert_eq!(store.in_degree(c), 1);
}

#[test]
fn test_edge_type() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let edge_id = store.create_edge(a, b, "KNOWS");

    let edge_type = store.edge_type(edge_id);
    assert_eq!(edge_type.as_deref(), Some("KNOWS"));

    // Non-existent edge
    let fake_id = EdgeId::new(999);
    assert!(store.edge_type(fake_id).is_none());
}

#[test]
fn test_count_methods() {
    let store = LpgStore::new().unwrap();

    assert_eq!(store.label_count(), 0);
    assert_eq!(store.edge_type_count(), 0);
    assert_eq!(store.property_key_count(), 0);

    let a = store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
    let b = store.create_node(&["Company"]);
    store.create_edge_with_props(a, b, "WORKS_AT", [("since", Value::from(2020i64))]);

    assert_eq!(store.label_count(), 2); // Person, Company
    assert_eq!(store.edge_type_count(), 1); // WORKS_AT
    assert_eq!(store.property_key_count(), 2); // age, since
}

#[test]
fn test_all_nodes_and_edges() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    store.create_edge(a, b, "KNOWS");

    let nodes: Vec<_> = store.all_nodes().collect();
    assert_eq!(nodes.len(), 2);

    let edges: Vec<_> = store.all_edges().collect();
    assert_eq!(edges.len(), 1);
}

#[test]
fn test_all_labels_and_edge_types() {
    let store = LpgStore::new().unwrap();

    store.create_node(&["Person"]);
    store.create_node(&["Company"]);
    let a = store.create_node(&["Animal"]);
    let b = store.create_node(&["Animal"]);
    store.create_edge(a, b, "EATS");

    let labels = store.all_labels();
    assert_eq!(labels.len(), 3);
    assert!(labels.contains(&"Person".to_string()));
    assert!(labels.contains(&"Company".to_string()));
    assert!(labels.contains(&"Animal".to_string()));

    let edge_types = store.all_edge_types();
    assert_eq!(edge_types.len(), 1);
    assert!(edge_types.contains(&"EATS".to_string()));
}

#[test]
fn test_all_property_keys() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node_with_props(&["Person"], [("name", Value::from("Alix"))]);
    let b = store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
    store.create_edge_with_props(a, b, "KNOWS", [("since", Value::from(2020i64))]);

    let keys = store.all_property_keys();
    assert!(keys.contains(&"name".to_string()));
    assert!(keys.contains(&"age".to_string()));
    assert!(keys.contains(&"since".to_string()));
}

#[test]
fn test_nodes_with_label() {
    let store = LpgStore::new().unwrap();

    store.create_node(&["Person"]);
    store.create_node(&["Person"]);
    store.create_node(&["Company"]);

    let persons: Vec<_> = store.nodes_with_label("Person").collect();
    assert_eq!(persons.len(), 2);

    let companies: Vec<_> = store.nodes_with_label("Company").collect();
    assert_eq!(companies.len(), 1);

    let none: Vec<_> = store.nodes_with_label("NonExistent").collect();
    assert_eq!(none.len(), 0);
}

#[test]
fn test_edges_with_type() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Company"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "WORKS_AT");

    let knows: Vec<_> = store.edges_with_type("KNOWS").collect();
    assert_eq!(knows.len(), 1);

    let works_at: Vec<_> = store.edges_with_type("WORKS_AT").collect();
    assert_eq!(works_at.len(), 1);

    let none: Vec<_> = store.edges_with_type("NonExistent").collect();
    assert_eq!(none.len(), 0);
}

#[test]
fn test_nodes_by_label_nonexistent() {
    let store = LpgStore::new().unwrap();
    store.create_node(&["Person"]);

    let empty = store.nodes_by_label("NonExistent");
    assert!(empty.is_empty());
}

#[test]
fn test_nodes_by_label_count_matches_vec_len() {
    // nodes_by_label_count is the O(1) fast path used by the planner to
    // bound unbounded VectorScan k; it must agree with the Vec-returning
    // variant for every label, including ones that don't exist.
    let store = LpgStore::new().unwrap();
    store.create_node(&["Person"]);
    store.create_node(&["Person"]);
    store.create_node(&["Animal"]);

    for label in ["Person", "Animal", "NonExistent"] {
        assert_eq!(
            store.nodes_by_label_count(label),
            store.nodes_by_label(label).len(),
            "count mismatch for label {label:?}"
        );
    }
}

#[test]
fn test_statistics() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Company"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "WORKS_AT");

    store.compute_statistics();
    let stats = store.statistics();

    assert_eq!(stats.total_nodes, 3);
    assert_eq!(stats.total_edges, 2);

    // Estimates
    let person_card = store.estimate_label_cardinality("Person");
    assert!(person_card > 0.0);

    let avg_degree = store.estimate_avg_degree("KNOWS", true);
    assert!(avg_degree >= 0.0);
}

#[test]
fn test_zone_maps() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("age", Value::from(25i64))]);
    store.create_node_with_props(&["Person"], [("age", Value::from(35i64))]);

    // Zone map should indicate possible matches (30 is within [25, 35] range)
    let might_match =
        store.node_property_might_match(&"age".into(), CompareOp::Eq, &Value::from(30i64));
    // Zone maps return true conservatively when value is within min/max range
    assert!(might_match);

    let zone = store.node_property_zone_map(&"age".into());
    assert!(zone.is_some());

    // Non-existent property
    let no_zone = store.node_property_zone_map(&"nonexistent".into());
    assert!(no_zone.is_none());

    // Edge zone maps
    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    store.create_edge_with_props(a, b, "REL", [("weight", Value::from(1.0))]);

    let edge_zone = store.edge_property_zone_map(&"weight".into());
    assert!(edge_zone.is_some());
}

#[test]
fn test_rebuild_zone_maps() {
    let store = LpgStore::new().unwrap();
    store.create_node_with_props(&["Person"], [("age", Value::from(25i64))]);

    // Should not panic
    store.rebuild_zone_maps();
}

#[test]
fn test_create_node_with_id() {
    let store = LpgStore::new().unwrap();

    let specific_id = NodeId::new(100);
    store
        .create_node_with_id(specific_id, &["Person", "Employee"])
        .unwrap();

    let node = store.get_node(specific_id).unwrap();
    assert!(node.has_label("Person"));
    assert!(node.has_label("Employee"));

    // Next auto-generated ID should be > 100
    let next = store.create_node(&["Other"]);
    assert!(next.as_u64() > 100);
}

#[test]
fn test_create_edge_with_id() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);

    let specific_id = EdgeId::new(500);
    store.create_edge_with_id(specific_id, a, b, "REL").unwrap();

    let edge = store.get_edge(specific_id).unwrap();
    assert_eq!(edge.src, a);
    assert_eq!(edge.dst, b);
    assert_eq!(edge.edge_type.as_str(), "REL");

    // Next auto-generated ID should be > 500
    let next = store.create_edge(a, b, "OTHER");
    assert!(next.as_u64() > 500);
}

#[test]
fn test_set_epoch() {
    let store = LpgStore::new().unwrap();

    assert_eq!(store.current_epoch().as_u64(), 0);

    store.set_epoch(EpochId::new(42));
    assert_eq!(store.current_epoch().as_u64(), 42);
}

#[test]
fn test_get_node_nonexistent() {
    let store = LpgStore::new().unwrap();
    let fake_id = NodeId::new(999);
    assert!(store.get_node(fake_id).is_none());
}

#[test]
fn test_get_edge_nonexistent() {
    let store = LpgStore::new().unwrap();
    let fake_id = EdgeId::new(999);
    assert!(store.get_edge(fake_id).is_none());
}

#[test]
fn test_multiple_labels() {
    let store = LpgStore::new().unwrap();

    let id = store.create_node(&["Person", "Employee", "Manager"]);
    let node = store.get_node(id).unwrap();

    assert!(node.has_label("Person"));
    assert!(node.has_label("Employee"));
    assert!(node.has_label("Manager"));
    assert!(!node.has_label("Other"));
}

#[test]
fn test_new_store_is_empty() {
    let store = LpgStore::new().unwrap();
    assert_eq!(store.node_count(), 0);
    assert_eq!(store.edge_count(), 0);
}

#[test]
fn test_edges_from_both_directions() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    let c = store.create_node(&["C"]);

    let e1 = store.create_edge(a, b, "R1"); // a -> b
    let e2 = store.create_edge(c, a, "R2"); // c -> a

    // Both directions from a
    let edges: Vec<_> = store.edges_from(a, Direction::Both).collect();
    assert_eq!(edges.len(), 2);
    assert!(edges.iter().any(|(_, e)| *e == e1)); // outgoing
    assert!(edges.iter().any(|(_, e)| *e == e2)); // incoming
}

#[test]
fn test_no_backward_adj_in_degree() {
    let config = LpgStoreConfig {
        backward_edges: false,
        initial_node_capacity: 10,
        initial_edge_capacity: 10,
    };
    let store = LpgStore::with_config(config).unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    store.create_edge(a, b, "R");

    // in_degree should still work (falls back to scanning)
    let degree = store.in_degree(b);
    assert_eq!(degree, 1);
}

#[test]
fn test_no_backward_adj_edges_to() {
    let config = LpgStoreConfig {
        backward_edges: false,
        initial_node_capacity: 10,
        initial_edge_capacity: 10,
    };
    let store = LpgStore::with_config(config).unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    let e = store.create_edge(a, b, "R");

    // edges_to should still work (falls back to scanning)
    let edges = store.edges_to(b);
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].1, e);
}

#[test]
fn test_node_versioned_creation() {
    let store = LpgStore::new().unwrap();

    let epoch = store.new_epoch();
    let transaction_id = TransactionId::new(1);

    let id = store.create_node_versioned(&["Person"], epoch, transaction_id);
    assert!(store.get_node(id).is_some());
}

#[test]
fn test_edge_versioned_creation() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);

    let epoch = store.new_epoch();
    let transaction_id = TransactionId::new(1);

    let edge_id = store.create_edge_versioned(a, b, "REL", epoch, transaction_id);
    assert!(store.get_edge(edge_id).is_some());
}

#[test]
fn test_node_with_props_versioned() {
    let store = LpgStore::new().unwrap();

    let epoch = store.new_epoch();
    let transaction_id = TransactionId::new(1);

    let id = store.create_node_with_props_versioned(
        &["Person"],
        [("name", Value::from("Alix"))],
        epoch,
        transaction_id,
    );

    let node = store.get_node(id).unwrap();
    assert_eq!(
        node.get_property("name").and_then(|v| v.as_str()),
        Some("Alix")
    );
}

#[test]
fn test_discard_uncommitted_versions() {
    let store = LpgStore::new().unwrap();

    let epoch = store.new_epoch();
    let transaction_id = TransactionId::new(42);
    store.create_property_index("proof");

    // Create a complete pending subgraph. Rollback must erase the identities,
    // not merely their structural chains: labels, properties, indexes,
    // adjacency, counters, and same-ID collision state are part of the cut.
    let node_id = store.create_node_versioned(&["Person"], epoch, transaction_id);
    let destination = store.create_node_versioned(&["Destination"], epoch, transaction_id);
    let edge_id =
        store.create_edge_versioned(node_id, destination, "PENDING", epoch, transaction_id);
    store.set_node_property_versioned(
        node_id,
        "proof",
        Value::from("pending-node"),
        transaction_id,
    );
    store.set_edge_property_versioned(
        edge_id,
        "proof",
        Value::from("pending-edge"),
        transaction_id,
    );
    // Verify the node exists via versioned lookup (own tx can see its PENDING writes)
    assert!(
        store
            .get_node_versioned(node_id, epoch, transaction_id)
            .is_some(),
        "Node should be visible to its own transaction"
    );

    // Discard uncommitted versions for this tx
    store.discard_uncommitted_versions(transaction_id);

    // Every trace of the never-committed identities is physically gone.
    assert!(
        store
            .get_node_versioned(node_id, epoch, transaction_id)
            .is_none(),
        "Node should be gone after discard"
    );
    assert!(store.all_node_ids().is_empty());
    assert!(store.all_known_edge_ids().is_empty());
    assert!(store.nodes_by_label("Person").is_empty());
    assert!(store.nodes_by_label("Destination").is_empty());
    assert!(store.node_label_history(node_id).is_empty());
    assert!(store.node_property_history(node_id).is_empty());
    assert!(store.edge_property_history(edge_id).is_empty());
    assert!(
        store
            .find_nodes_by_property("proof", &Value::from("pending-node"))
            .is_empty()
    );
    assert!(
        store
            .edges_from(node_id, Direction::Outgoing)
            .next()
            .is_none()
    );
    assert_eq!(store.node_count(), 0);
    assert_eq!(store.edge_count(), 0);

    // Recovery may deliberately reuse the exact IDs. No stale derived state
    // from the rolled-back incarnation may bleed into the replacement.
    let restored = EpochId::new(epoch.as_u64().saturating_add(1));
    store
        .restore_node_history_exact(
            node_id,
            &[(restored, None)],
            &[(restored, vec![arcstr::ArcStr::from("Restored")])],
        )
        .expect("same node ID is vacant after physical rollback");
    store
        .restore_node_history_exact(
            destination,
            &[(restored, None)],
            &[(restored, vec![arcstr::ArcStr::from("Destination")])],
        )
        .expect("same destination ID is vacant after physical rollback");
    store
        .restore_edge_history_exact(
            edge_id,
            node_id,
            destination,
            "RESTORED",
            &[(restored, None)],
        )
        .expect("same edge ID is vacant after physical rollback");
    assert_eq!(store.nodes_by_label("Restored"), vec![node_id]);
    assert!(store.nodes_by_label("Person").is_empty());
    assert!(store.node_property_history(node_id).is_empty());
    assert!(store.edge_property_history(edge_id).is_empty());
    assert_eq!(
        store
            .edges_from(node_id, Direction::Outgoing)
            .collect::<Vec<_>>(),
        vec![(destination, edge_id)]
    );
}

// === Property Index Tests ===

#[test]
fn test_property_index_create_and_lookup() {
    let store = LpgStore::new().unwrap();

    // Create nodes with properties
    let alix = store.create_node(&["Person"]);
    let gus = store.create_node(&["Person"]);
    let vincent = store.create_node(&["Person"]);

    store.set_node_property(alix, "city", Value::from("NYC"));
    store.set_node_property(gus, "city", Value::from("NYC"));
    store.set_node_property(vincent, "city", Value::from("LA"));

    // Before indexing, lookup still works (via scan)
    let nyc_people = store.find_nodes_by_property("city", &Value::from("NYC"));
    assert_eq!(nyc_people.len(), 2);

    // Create index
    store.create_property_index("city");
    assert!(store.has_property_index("city"));

    // Indexed lookup should return same results
    let nyc_people = store.find_nodes_by_property("city", &Value::from("NYC"));
    assert_eq!(nyc_people.len(), 2);
    assert!(nyc_people.contains(&alix));
    assert!(nyc_people.contains(&gus));

    let la_people = store.find_nodes_by_property("city", &Value::from("LA"));
    assert_eq!(la_people.len(), 1);
    assert!(la_people.contains(&vincent));
}

#[test]
fn test_property_index_maintained_on_update() {
    let store = LpgStore::new().unwrap();

    // Create index first
    store.create_property_index("status");

    let node = store.create_node(&["Task"]);
    store.set_node_property(node, "status", Value::from("pending"));

    // Should find by initial value
    let pending = store.find_nodes_by_property("status", &Value::from("pending"));
    assert_eq!(pending.len(), 1);
    assert!(pending.contains(&node));

    // Update the property
    store.set_node_property(node, "status", Value::from("done"));

    // Old value should not find it
    let pending = store.find_nodes_by_property("status", &Value::from("pending"));
    assert!(pending.is_empty());

    // New value should find it
    let done = store.find_nodes_by_property("status", &Value::from("done"));
    assert_eq!(done.len(), 1);
    assert!(done.contains(&node));
}

#[test]
fn test_property_index_maintained_on_remove() {
    let store = LpgStore::new().unwrap();

    store.create_property_index("tag");

    let node = store.create_node(&["Item"]);
    store.set_node_property(node, "tag", Value::from("important"));

    // Should find it
    let found = store.find_nodes_by_property("tag", &Value::from("important"));
    assert_eq!(found.len(), 1);

    // Remove the property
    store.remove_node_property(node, "tag");

    // Should no longer find it
    let found = store.find_nodes_by_property("tag", &Value::from("important"));
    assert!(found.is_empty());
}

#[test]
fn test_property_index_drop() {
    let store = LpgStore::new().unwrap();

    store.create_property_index("key");
    assert!(store.has_property_index("key"));

    assert!(store.drop_property_index("key"));
    assert!(!store.has_property_index("key"));

    // Dropping non-existent index returns false
    assert!(!store.drop_property_index("key"));
}

#[test]
fn test_property_index_multiple_values() {
    let store = LpgStore::new().unwrap();

    store.create_property_index("age");

    // Create multiple nodes with same and different ages
    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    let n3 = store.create_node(&["Person"]);
    let n4 = store.create_node(&["Person"]);

    store.set_node_property(n1, "age", Value::from(25i64));
    store.set_node_property(n2, "age", Value::from(25i64));
    store.set_node_property(n3, "age", Value::from(30i64));
    store.set_node_property(n4, "age", Value::from(25i64));

    let age_25 = store.find_nodes_by_property("age", &Value::from(25i64));
    assert_eq!(age_25.len(), 3);

    let age_30 = store.find_nodes_by_property("age", &Value::from(30i64));
    assert_eq!(age_30.len(), 1);

    let age_40 = store.find_nodes_by_property("age", &Value::from(40i64));
    assert!(age_40.is_empty());
}

#[test]
fn test_property_index_builds_from_existing_data() {
    let store = LpgStore::new().unwrap();

    // Create nodes first
    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    store.set_node_property(n1, "email", Value::from("alix@example.com"));
    store.set_node_property(n2, "email", Value::from("gus@example.com"));

    // Create index after data exists
    store.create_property_index("email");

    // Index should include existing data
    let alix = store.find_nodes_by_property("email", &Value::from("alix@example.com"));
    assert_eq!(alix.len(), 1);
    assert!(alix.contains(&n1));

    let gus = store.find_nodes_by_property("email", &Value::from("gus@example.com"));
    assert_eq!(gus.len(), 1);
    assert!(gus.contains(&n2));
}

#[test]
fn test_get_node_property_batch() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    let n3 = store.create_node(&["Person"]);

    store.set_node_property(n1, "age", Value::from(25i64));
    store.set_node_property(n2, "age", Value::from(30i64));
    // n3 has no age property

    let age_key = PropertyKey::new("age");
    let values = store.get_node_property_batch(&[n1, n2, n3], &age_key);

    assert_eq!(values.len(), 3);
    assert_eq!(values[0], Some(Value::from(25i64)));
    assert_eq!(values[1], Some(Value::from(30i64)));
    assert_eq!(values[2], None);
}

#[test]
fn test_get_node_property_batch_empty() {
    let store = LpgStore::new().unwrap();
    let key = PropertyKey::new("any");

    let values = store.get_node_property_batch(&[], &key);
    assert!(values.is_empty());
}

#[test]
fn test_get_nodes_properties_batch() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    let n3 = store.create_node(&["Person"]);

    store.set_node_property(n1, "name", Value::from("Alix"));
    store.set_node_property(n1, "age", Value::from(25i64));
    store.set_node_property(n2, "name", Value::from("Gus"));
    // n3 has no properties

    let all_props = store.get_nodes_properties_batch(&[n1, n2, n3]);

    assert_eq!(all_props.len(), 3);
    assert_eq!(all_props[0].len(), 2); // name and age
    assert_eq!(all_props[1].len(), 1); // name only
    assert_eq!(all_props[2].len(), 0); // no properties

    assert_eq!(
        all_props[0].get(&PropertyKey::new("name")),
        Some(&Value::from("Alix"))
    );
    assert_eq!(
        all_props[1].get(&PropertyKey::new("name")),
        Some(&Value::from("Gus"))
    );
}

#[test]
fn test_get_nodes_properties_batch_empty() {
    let store = LpgStore::new().unwrap();

    let all_props = store.get_nodes_properties_batch(&[]);
    assert!(all_props.is_empty());
}

#[test]
fn test_get_nodes_properties_selective_batch() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);

    // Set multiple properties
    store.set_node_property(n1, "name", Value::from("Alix"));
    store.set_node_property(n1, "age", Value::from(25i64));
    store.set_node_property(n1, "email", Value::from("alix@example.com"));
    store.set_node_property(n2, "name", Value::from("Gus"));
    store.set_node_property(n2, "age", Value::from(30i64));
    store.set_node_property(n2, "city", Value::from("NYC"));

    // Request only name and age (not email or city)
    let keys = vec![PropertyKey::new("name"), PropertyKey::new("age")];
    let props = store.get_nodes_properties_selective_batch(&[n1, n2], &keys);

    assert_eq!(props.len(), 2);

    // n1: should have name and age, but NOT email
    assert_eq!(props[0].len(), 2);
    assert_eq!(
        props[0].get(&PropertyKey::new("name")),
        Some(&Value::from("Alix"))
    );
    assert_eq!(
        props[0].get(&PropertyKey::new("age")),
        Some(&Value::from(25i64))
    );
    assert_eq!(props[0].get(&PropertyKey::new("email")), None);

    // n2: should have name and age, but NOT city
    assert_eq!(props[1].len(), 2);
    assert_eq!(
        props[1].get(&PropertyKey::new("name")),
        Some(&Value::from("Gus"))
    );
    assert_eq!(
        props[1].get(&PropertyKey::new("age")),
        Some(&Value::from(30i64))
    );
    assert_eq!(props[1].get(&PropertyKey::new("city")), None);
}

#[test]
fn test_get_nodes_properties_selective_batch_empty_keys() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    store.set_node_property(n1, "name", Value::from("Alix"));

    // Request no properties
    let props = store.get_nodes_properties_selective_batch(&[n1], &[]);

    assert_eq!(props.len(), 1);
    assert!(props[0].is_empty()); // Empty map when no keys requested
}

#[test]
fn test_get_nodes_properties_selective_batch_missing_keys() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    store.set_node_property(n1, "name", Value::from("Alix"));

    // Request a property that doesn't exist
    let keys = vec![PropertyKey::new("nonexistent"), PropertyKey::new("name")];
    let props = store.get_nodes_properties_selective_batch(&[n1], &keys);

    assert_eq!(props.len(), 1);
    assert_eq!(props[0].len(), 1); // Only name exists
    assert_eq!(
        props[0].get(&PropertyKey::new("name")),
        Some(&Value::from("Alix"))
    );
}

// === Range Query Tests ===

#[test]
fn test_find_nodes_in_range_inclusive() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);
    let n2 = store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
    let n3 = store.create_node_with_props(&["Person"], [("age", Value::from(40i64))]);
    let _n4 = store.create_node_with_props(&["Person"], [("age", Value::from(50i64))]);

    // age >= 20 AND age <= 40 (inclusive both sides)
    let result = store.find_nodes_in_range(
        "age",
        Some(&Value::from(20i64)),
        Some(&Value::from(40i64)),
        true,
        true,
    );
    assert_eq!(result.len(), 3);
    assert!(result.contains(&n1));
    assert!(result.contains(&n2));
    assert!(result.contains(&n3));
}

#[test]
fn test_find_nodes_in_range_exclusive() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);
    let n2 = store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
    store.create_node_with_props(&["Person"], [("age", Value::from(40i64))]);

    // age > 20 AND age < 40 (exclusive both sides)
    let result = store.find_nodes_in_range(
        "age",
        Some(&Value::from(20i64)),
        Some(&Value::from(40i64)),
        false,
        false,
    );
    assert_eq!(result.len(), 1);
    assert!(result.contains(&n2));
}

#[test]
fn test_find_nodes_in_range_open_ended() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);
    store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
    let n3 = store.create_node_with_props(&["Person"], [("age", Value::from(40i64))]);
    let n4 = store.create_node_with_props(&["Person"], [("age", Value::from(50i64))]);

    // age >= 35 (no upper bound)
    let result = store.find_nodes_in_range("age", Some(&Value::from(35i64)), None, true, true);
    assert_eq!(result.len(), 2);
    assert!(result.contains(&n3));
    assert!(result.contains(&n4));

    // age <= 25 (no lower bound)
    let result = store.find_nodes_in_range("age", None, Some(&Value::from(25i64)), true, true);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_find_nodes_in_range_empty_result() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);

    // Range that doesn't match anything
    let result = store.find_nodes_in_range(
        "age",
        Some(&Value::from(100i64)),
        Some(&Value::from(200i64)),
        true,
        true,
    );
    assert!(result.is_empty());
}

#[test]
fn test_find_nodes_in_range_nonexistent_property() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);

    let result = store.find_nodes_in_range(
        "weight",
        Some(&Value::from(50i64)),
        Some(&Value::from(100i64)),
        true,
        true,
    );
    assert!(result.is_empty());
}

// === Multi-Property Query Tests ===

#[test]
fn test_find_nodes_by_properties_multiple_conditions() {
    let store = LpgStore::new().unwrap();

    let alix = store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Alix")), ("city", Value::from("NYC"))],
    );
    store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Gus")), ("city", Value::from("NYC"))],
    );
    store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Alix")), ("city", Value::from("LA"))],
    );

    // Match name="Alix" AND city="NYC"
    let result = store
        .find_nodes_by_properties(&[("name", Value::from("Alix")), ("city", Value::from("NYC"))]);
    assert_eq!(result.len(), 1);
    assert!(result.contains(&alix));
}

#[test]
fn test_find_nodes_by_properties_empty_conditions() {
    let store = LpgStore::new().unwrap();

    store.create_node(&["Person"]);
    store.create_node(&["Person"]);

    // Empty conditions should return all nodes
    let result = store.find_nodes_by_properties(&[]);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_find_nodes_by_properties_no_match() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("name", Value::from("Alix"))]);

    let result = store.find_nodes_by_properties(&[("name", Value::from("Nobody"))]);
    assert!(result.is_empty());
}

#[test]
fn test_find_nodes_by_properties_with_index() {
    let store = LpgStore::new().unwrap();

    // Create index on name
    store.create_property_index("name");

    let alix = store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Alix")), ("age", Value::from(30i64))],
    );
    store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Gus")), ("age", Value::from(30i64))],
    );

    // Index should accelerate the lookup
    let result = store
        .find_nodes_by_properties(&[("name", Value::from("Alix")), ("age", Value::from(30i64))]);
    assert_eq!(result.len(), 1);
    assert!(result.contains(&alix));
}

// === Cardinality Estimation Tests ===

#[test]
fn test_estimate_label_cardinality() {
    let store = LpgStore::new().unwrap();

    store.create_node(&["Person"]);
    store.create_node(&["Person"]);
    store.create_node(&["Animal"]);

    store.ensure_statistics_fresh();

    let person_est = store.estimate_label_cardinality("Person");
    let animal_est = store.estimate_label_cardinality("Animal");
    let unknown_est = store.estimate_label_cardinality("Unknown");

    assert!(
        person_est >= 1.0,
        "Person should have cardinality >= 1, got {person_est}"
    );
    assert!(
        animal_est >= 1.0,
        "Animal should have cardinality >= 1, got {animal_est}"
    );
    // Unknown label should return some default (not panic)
    assert!(unknown_est >= 0.0);
}

#[test]
fn test_estimate_avg_degree() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "KNOWS");
    store.create_edge(b, c, "KNOWS");

    store.ensure_statistics_fresh();

    let outgoing = store.estimate_avg_degree("KNOWS", true);
    let incoming = store.estimate_avg_degree("KNOWS", false);

    assert!(
        outgoing > 0.0,
        "Outgoing degree should be > 0, got {outgoing}"
    );
    assert!(
        incoming > 0.0,
        "Incoming degree should be > 0, got {incoming}"
    );
}

// === Delete operations ===

#[test]
fn test_delete_node_does_not_cascade() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    let e = store.create_edge(a, b, "KNOWS");

    assert!(store.delete_node(a));
    assert!(store.get_node(a).is_none());

    // Edges are NOT automatically deleted (non-detach delete)
    assert!(
        store.get_edge(e).is_some(),
        "Edge should survive non-detach node delete"
    );
}

#[test]
fn test_delete_already_deleted_node() {
    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["A"]);

    assert!(store.delete_node(a));
    // Second delete should return false (already deleted)
    assert!(!store.delete_node(a));
}

#[test]
fn test_delete_nonexistent_node() {
    let store = LpgStore::new().unwrap();
    assert!(!store.delete_node(NodeId::new(999)));
}

// === GraphStore / GraphStoreMut Trait Compliance ===

/// Verifies that LpgStore's trait implementations are object-safe and
/// produce identical results to the concrete methods.
mod graph_store_traits {
    use super::*;
    use crate::graph::Direction;
    use crate::graph::traits::{GraphStore, GraphStoreMut};

    #[test]
    fn trait_object_safety() {
        // Must compile: Arc<dyn GraphStoreMut> proves object safety
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());
        let _read: &dyn GraphStore = &*store;
    }

    #[test]
    fn trait_round_trip() {
        let store = LpgStore::new().unwrap();
        let store: &dyn GraphStoreMut = &store;

        // Create nodes via trait
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person", "Developer"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::from(30i64));
        store.set_node_property(gus, "name", Value::from("Gus"));

        // Create edge via trait
        let edge = store.create_edge(alix, gus, "KNOWS");
        store.set_edge_property(edge, "since", Value::from(2020i64));

        // Read back via GraphStore trait
        let read: &dyn GraphStore = store;

        // Point lookups
        let alice_node = read.get_node(alix).expect("alix should exist");
        assert!(alice_node.labels.contains(&arcstr::literal!("Person")));

        let edge_data = read.get_edge(edge).expect("edge should exist");
        assert_eq!(edge_data.src, alix);
        assert_eq!(edge_data.dst, gus);

        // Properties
        assert_eq!(
            read.get_node_property(alix, &PropertyKey::new("name")),
            Some(Value::from("Alix"))
        );
        assert_eq!(
            read.get_edge_property(edge, &PropertyKey::new("since")),
            Some(Value::from(2020i64))
        );

        // Traversal
        let neighbors = read.neighbors(alix, Direction::Outgoing);
        assert_eq!(neighbors, vec![gus]);

        let edges = read.edges_from(alix, Direction::Outgoing);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0], (gus, edge));

        assert_eq!(read.out_degree(alix), 1);
        assert_eq!(read.in_degree(gus), 1);

        // Scans
        assert_eq!(read.node_count(), 2);
        assert_eq!(read.edge_count(), 1);
        assert_eq!(read.nodes_by_label("Person").len(), 2);
        assert_eq!(read.node_ids().len(), 2);

        // Edge type
        assert_eq!(read.edge_type(edge), Some(arcstr::literal!("KNOWS")));

        // Search
        let found = read.find_nodes_by_property("name", &Value::from("Alix"));
        assert_eq!(found, vec![alix]);
    }

    #[test]
    fn trait_mutation_operations() {
        let store = LpgStore::new().unwrap();
        let store: &dyn GraphStoreMut = &store;

        let node = store.create_node(&["A"]);

        // Label mutation
        assert!(store.add_label(node, "B"));
        assert!(store.remove_label(node, "B"));

        // Property mutation
        store.set_node_property(node, "key", Value::from("val"));
        let removed = store.remove_node_property(node, "key");
        assert_eq!(removed, Some(Value::from("val")));

        // Deletion
        assert!(store.delete_node(node));
        assert!(store.get_node(node).is_none());
    }

    #[test]
    fn trait_batch_edges() {
        let store = LpgStore::new().unwrap();
        let store: &dyn GraphStoreMut = &store;

        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let c = store.create_node(&["N"]);

        let ids = store.batch_create_edges(&[(a, b, "E"), (a, c, "E"), (b, c, "E")]);
        assert_eq!(ids.len(), 3);
        assert_eq!(store.edge_count(), 3);
    }
}

#[test]
fn test_clear() {
    let store = LpgStore::new().unwrap();
    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    store.set_node_property(n1, "name", "Alix".into());
    let _e = store.create_edge(n1, n2, "KNOWS");
    store.set_edge_property(_e, "since", 2024.into());

    assert_eq!(store.node_count(), 2);
    assert_eq!(store.edge_count(), 1);

    store.clear();

    assert_eq!(store.node_count(), 0);
    assert_eq!(store.edge_count(), 0);

    // Should be able to add new data after clear
    let n3 = store.create_node(&["Animal"]);
    assert_eq!(store.node_count(), 1);
    assert!(store.get_node(n3).is_some());
}

#[test]
fn copy_graph_deep_copies_nodes_edges_props_labels_and_index() {
    let store = LpgStore::new().expect("arena");
    let src = store.graph_or_create("src").expect("src graph");
    let a = src.create_node_with_props(
        &["Person"],
        [("name", Value::from("Ann")), ("age", Value::from(30i64))],
    );
    let b = src.create_node_with_props(&["Person"], [("name", Value::from("Bob"))]);
    src.create_edge_with_props(a, b, "KNOWS", [("since", Value::from(2020i64))]);
    src.create_property_index("name");

    store.copy_graph(Some("src"), Some("dst")).expect("copy");
    let dst = store.graph("dst").expect("dst graph created");

    // Counts, labels, props.
    assert_eq!(dst.node_count(), 2);
    assert_eq!(dst.edge_count(), 1);
    let ann_ids = dst.find_nodes_by_property("name", &Value::from("Ann"));
    assert_eq!(ann_ids.len(), 1, "property index must work on the copy");
    let ann = dst.get_node(ann_ids[0]).unwrap();
    assert!(ann.labels.iter().any(|l| l.as_str() == "Person"));
    assert_eq!(
        ann.properties.get(&PropertyKey::new("age")),
        Some(&Value::Int64(30))
    );

    // Edge: type, remapped endpoints, and property carried.
    let edges: Vec<_> = dst.all_edges().collect();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].edge_type.as_str(), "KNOWS");
    assert_eq!(
        edges[0].properties.get(&PropertyKey::new("since")),
        Some(&Value::Int64(2020))
    );

    // Deep copy: mutating the copy does not affect the source.
    dst.set_node_property(ann_ids[0], "age", Value::from(99i64));
    let src_ann = src.find_nodes_by_property("name", &Value::from("Ann"));
    assert_eq!(
        src.get_node(src_ann[0])
            .unwrap()
            .properties
            .get(&PropertyKey::new("age")),
        Some(&Value::Int64(30)),
        "source must be unchanged by mutations to the copy"
    );
}

#[test]
fn copy_graph_self_copy_is_a_noop() {
    let store = LpgStore::new().expect("arena");
    let g = store.graph_or_create("g").expect("g");
    g.create_node(&["X"]);
    store
        .copy_graph(Some("g"), Some("g"))
        .expect("self-copy ok");
    assert_eq!(
        store.graph("g").unwrap().node_count(),
        1,
        "self-copy must not duplicate"
    );
}

#[test]
fn finalize_entities_by_id_scopes_to_named_entities() {
    use grafeo_common::types::EpochId;
    let store = LpgStore::new().unwrap();
    let tx = TransactionId::new(2);
    // Two nodes created by the same transaction at PENDING (invisible until finalized).
    let n1 = store.create_node_versioned(&["A"], EpochId::new(0), tx);
    let n2 = store.create_node_versioned(&["A"], EpochId::new(0), tx);
    let commit_epoch = EpochId::new(5);

    // Finalize only n1.
    store.finalize_entities_by_id(tx, commit_epoch, &[n1], &[]);

    assert!(
        store.is_node_visible_at_epoch(n1, commit_epoch),
        "finalized node must be visible at the commit epoch"
    );
    assert!(
        !store.is_node_visible_at_epoch(n2, commit_epoch),
        "an un-finalized node must remain PENDING (invisible)"
    );
}

#[test]
fn tx_property_overlay_isolates_uncommitted_writes() {
    use grafeo_common::types::PropertyKey;
    let store = LpgStore::new().unwrap();
    let n = store.create_node(&["P"]);
    store.set_node_property(n, "age", Value::from(30i64)); // committed
    let tx = TransactionId::new(2);
    let key = PropertyKey::new("age");
    let epoch = store.current_epoch();

    // Uncommitted write goes to the transaction's delta, not the committed column.
    store.set_node_property_buffered(n, "age", Value::from(99i64), tx);

    // The writing transaction sees its own uncommitted value (read-your-writes).
    assert_eq!(
        store.read_node_property_visible(n, &key, epoch, Some(tx)),
        Some(Value::Int64(99))
    );
    // Any other reader (no transaction) sees only the committed value (no dirty read).
    assert_eq!(
        store.read_node_property_visible(n, &key, epoch, None),
        Some(Value::Int64(30))
    );
    // The committed column is untouched while the write is buffered.
    assert_eq!(store.get_node_property(n, &key), Some(Value::Int64(30)));

    // Apply (commit) promotes the value to the committed column.
    store.apply_tx_overlay(tx);
    assert_eq!(store.get_node_property(n, &key), Some(Value::Int64(99)));

    // A buffered remove tombstones for own-reads; drop (rollback) discards it.
    let tx2 = TransactionId::new(3);
    store.remove_node_property_buffered(n, "age", tx2);
    assert_eq!(
        store.read_node_property_visible(n, &key, epoch, Some(tx2)),
        None
    );
    assert_eq!(
        store.read_node_property_visible(n, &key, epoch, None),
        Some(Value::Int64(99))
    );
    store.drop_tx_overlay(tx2);
    assert_eq!(store.get_node_property(n, &key), Some(Value::Int64(99)));
}

#[test]
fn trait_accessor_isolates_buffered_writes() {
    use crate::graph::traits::GraphStoreMut;
    use grafeo_common::types::{PropertyKey, TransactionId};

    let store = LpgStore::new().unwrap();
    let n = store.create_node(&["Person"]);
    store.set_node_property(n, "age", Value::Int64(30));

    let tx = TransactionId::new(7);
    let key = PropertyKey::new("age");

    // Buffer an uncommitted write via the trait object.
    let s: &dyn GraphStoreMut = &store;
    s.set_node_property_buffered(n, "age", Value::Int64(99), tx);

    // Writer (Some(tx)) sees its own write; everyone else (None) sees committed.
    let epoch = store.current_epoch();
    assert_eq!(
        s.read_node_property_visible(n, &key, epoch, Some(tx)),
        Some(Value::Int64(99))
    );
    assert_eq!(
        s.read_node_property_visible(n, &key, epoch, None),
        Some(Value::Int64(30))
    );

    // Apply promotes the delta to the committed store.
    s.apply_tx_overlay(tx);
    assert_eq!(
        s.read_node_property_visible(n, &key, epoch, None),
        Some(Value::Int64(99))
    );
}

#[test]
fn whole_entity_accessor_merges_delta_for_writer() {
    use grafeo_common::types::{PropertyKey, TransactionId};

    let store = LpgStore::new().unwrap();
    let n = store.create_node(&["Thing"]);
    store.set_node_property(n, "keep", Value::Int64(1));
    store.set_node_property(n, "change", Value::Int64(2));
    store.set_node_property(n, "remove_me", Value::Int64(3));

    let tx = TransactionId::new(42);
    let epoch = store.current_epoch();

    // Buffer: change one key, remove another.
    store.set_node_property_buffered(n, "change", Value::Int64(99), tx);
    store.remove_node_property_buffered(n, "remove_me", tx);

    // Writer sees merged map: keep=1, change=99; remove_me absent.
    let writer_map = store.read_node_properties_visible(n, epoch, Some(tx));
    assert_eq!(
        writer_map.get(&PropertyKey::new("keep")),
        Some(&Value::Int64(1))
    );
    assert_eq!(
        writer_map.get(&PropertyKey::new("change")),
        Some(&Value::Int64(99))
    );
    assert!(
        !writer_map.contains_key(&PropertyKey::new("remove_me")),
        "remove_me must be absent for writer"
    );

    // Reader (tx=None) sees committed map: keep=1, change=2, remove_me=3.
    let reader_map = store.read_node_properties_visible(n, epoch, None);
    assert_eq!(
        reader_map.get(&PropertyKey::new("keep")),
        Some(&Value::Int64(1))
    );
    assert_eq!(
        reader_map.get(&PropertyKey::new("change")),
        Some(&Value::Int64(2))
    );
    assert_eq!(
        reader_map.get(&PropertyKey::new("remove_me")),
        Some(&Value::Int64(3))
    );
}

#[test]
fn label_delta_isolates_buffered_label_ops() {
    let store = LpgStore::new().unwrap();
    let n = store.create_node(&["Person"]);
    let tx = TransactionId::new(7);

    // Buffer add :Secret and remove :Person for tx.
    store.add_label_buffered(n, "Secret", tx);
    store.remove_label_buffered(n, "Person", tx);

    let writer_view =
        store.read_node_labels_visible(n, grafeo_common::types::EpochId::new(0), Some(tx));
    assert!(
        writer_view.contains(&arcstr::ArcStr::from("Secret")),
        "writer sees buffered add"
    );
    assert!(
        !writer_view.contains(&arcstr::ArcStr::from("Person")),
        "writer sees buffered remove"
    );

    // Other readers (None) see committed labels unchanged.
    let committed_view =
        store.read_node_labels_visible(n, grafeo_common::types::EpochId::new(0), None);
    assert!(
        committed_view.contains(&arcstr::ArcStr::from("Person")),
        "others see committed :Person"
    );
    assert!(
        !committed_view.contains(&arcstr::ArcStr::from("Secret")),
        "others do NOT see uncommitted :Secret"
    );

    // Apply promotes.
    store.apply_tx_overlay(tx);
    let after = store.read_node_labels_visible(n, grafeo_common::types::EpochId::new(0), None);
    assert!(
        after.contains(&arcstr::ArcStr::from("Secret"))
            && !after.contains(&arcstr::ArcStr::from("Person")),
        "commit applied label ops"
    );
}

#[test]
fn label_delta_rollback_and_savepoint() {
    let store = LpgStore::new().unwrap();
    let n = store.create_node(&["Person"]);
    let epoch = grafeo_common::types::EpochId::new(0);

    // --- Rollback path ---
    // tx A buffers :Secret but is then dropped (rolled back).
    let tx_a = TransactionId::new(10);
    store.add_label_buffered(n, "Secret", tx_a);
    store.drop_tx_overlay(tx_a);

    // After rollback the committed base must be unchanged — no :Secret visible.
    let after_rollback = store.read_node_labels_visible(n, epoch, None);
    assert!(
        after_rollback.contains(&arcstr::ArcStr::from("Person")),
        "committed :Person must survive rollback of tx A"
    );
    // :Secret must not have leaked into committed view.
    assert!(
        !after_rollback.contains(&arcstr::ArcStr::from("Secret")),
        "rolled-back :Secret must NOT be visible in committed view"
    );

    // --- Savepoint path ---
    let tx_b = TransactionId::new(11);

    // Buffer :Vip, take a savepoint, then buffer :Temp.
    store.add_label_buffered(n, "Vip", tx_b);
    let snap = store.tx_overlay_snapshot(tx_b);
    store.add_label_buffered(n, "Temp", tx_b);

    // Before restore: writer sees both Vip and Temp.
    let before_restore = store.read_node_labels_visible(n, epoch, Some(tx_b));
    assert!(
        before_restore.contains(&arcstr::ArcStr::from("Vip")),
        "writer must see buffered :Vip before restore"
    );
    assert!(
        before_restore.contains(&arcstr::ArcStr::from("Temp")),
        "writer must see buffered :Temp before restore"
    );

    // Restore to snapshot (Vip only, no Temp).
    store.tx_overlay_restore(tx_b, snap);

    let after_restore = store.read_node_labels_visible(n, epoch, Some(tx_b));
    assert!(
        after_restore.contains(&arcstr::ArcStr::from("Vip")),
        "writer must still see :Vip after savepoint restore"
    );
    assert!(
        !after_restore.contains(&arcstr::ArcStr::from("Temp")),
        "savepoint restore must discard :Temp — if this fails it is a real bug"
    );
    // The committed :Person must also be visible to the writer.
    assert!(
        after_restore.contains(&arcstr::ArcStr::from("Person")),
        "committed :Person must be visible to writer after restore"
    );
}

/// Mirror of `label_delta_isolates_buffered_label_ops` but exercised through
/// `&dyn GraphStoreMut` so the trait surface is tested at the trait level.
#[test]
fn trait_label_accessor_isolates() {
    use crate::graph::traits::GraphStoreMut;
    use grafeo_common::types::EpochId;

    let store = LpgStore::new().unwrap();
    // Exercise through a trait-object pointer.
    let dyn_store: &dyn GraphStoreMut = &store;

    let n = dyn_store.create_node(&["Person"]);
    let tx = TransactionId::new(99);

    // Buffer add :Secret and remove :Person for tx — via the trait object.
    dyn_store.add_label_buffered(n, "Secret", tx);
    dyn_store.remove_label_buffered(n, "Person", tx);

    // Writer view (Some(tx)) through the trait object.
    let writer_view = dyn_store.read_node_labels_visible(n, EpochId::new(0), Some(tx));
    assert!(
        writer_view.contains(&arcstr::ArcStr::from("Secret")),
        "writer sees buffered add via trait"
    );
    assert!(
        !writer_view.contains(&arcstr::ArcStr::from("Person")),
        "writer sees buffered remove via trait"
    );

    // Other reader (None) sees only committed labels.
    let committed_view = dyn_store.read_node_labels_visible(n, EpochId::new(0), None);
    assert!(
        committed_view.contains(&arcstr::ArcStr::from("Person")),
        "others see committed :Person via trait"
    );
    assert!(
        !committed_view.contains(&arcstr::ArcStr::from("Secret")),
        "others do NOT see uncommitted :Secret via trait"
    );

    // Apply promotes delta: committed view now reflects the writes.
    dyn_store.apply_tx_overlay(tx);
    let after = dyn_store.read_node_labels_visible(n, EpochId::new(0), None);
    assert!(
        after.contains(&arcstr::ArcStr::from("Secret"))
            && !after.contains(&arcstr::ArcStr::from("Person")),
        "commit promoted label ops via trait"
    );
}

/// Regression: a node inline-created **within a transaction** (e.g.
/// `MERGE (:Item)` / `CREATE (:Item)`) registers its labels directly into
/// `node_labels` at `EpochId::PENDING`. The writing transaction must see that
/// label through `read_node_labels_visible` so a later UNWIND row's MERGE can
/// dedupe against it. Under the `temporal` feature the committed-base read used
/// `VersionLog::at(real_epoch)`, which skips the PENDING entry and returned an
/// empty set — dropping the inline-create label and breaking MERGE-in-UNWIND
/// dedup (regression_external::unwind_merge_*).
#[test]
fn writer_sees_inline_create_label_for_own_pending_node() {
    let store = LpgStore::new().unwrap();
    let tx = TransactionId::new(42);
    let epoch = store.current_epoch();

    // Transactional inline create: labels land in `node_labels` at PENDING
    // (version_epoch = PENDING for a non-SYSTEM transaction).
    let n = store.create_node_versioned(&["Item"], epoch, tx);

    // The writing transaction must see its own just-created label.
    let writer_view = store.read_node_labels_visible(n, epoch, Some(tx));
    assert!(
        writer_view.contains(&arcstr::ArcStr::from("Item")),
        "writer must see its own inline-created :Item (got {writer_view:?})"
    );
}

/// Edge-delete isolation (MVCC increment 2b, Task 1).
///
/// A `delete_edge_transactional` must isolate the delete to the writing
/// transaction until commit, mirroring the node-delete deferral model:
///  - the writer sees the edge gone (read-your-writes via the version chain),
///  - every other session still sees it (PENDING `deleted_epoch` > any real epoch),
///  - the candidate adjacency index stays populated (it is NON-MVCC by design;
///    visibility is post-filtered one layer up in the expand operators), and
///  - edge properties and the live/edge-type counts are NOT touched at delete —
///    they are deferred to `finalize_edge_deletes_by_id` at commit, so an
///    uncommitted/rolled-back delete neither drops another session's property
///    read nor under-counts.
#[test]
fn edge_delete_pending_isolates() {
    use grafeo_common::types::{EpochId, PropertyKey};

    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    // Edge with a property so we can assert property removal is deferred.
    let eid = store.create_edge_with_props(a, b, "R", [("prop", Value::from(7i64))]);

    let key = PropertyKey::new("prop");
    let epoch = store.current_epoch();
    let tx = TransactionId::new(2);
    let other_tx = TransactionId::new(3);

    // Pre-conditions: edge live and readable by everyone.
    assert!(store.is_edge_visible_versioned(eid, epoch, tx));
    assert_eq!(store.edge_properties.get(eid, &key), Some(Value::Int64(7)));
    let live_before = store.live_edge_count.load(Ordering::Relaxed);

    // Transactional delete: stamps PENDING deleted_epoch by `tx`.
    assert!(store.delete_edge_transactional(eid, epoch, tx));

    // Writer sees it gone (via the chain).
    assert!(
        store.get_edge_versioned(eid, epoch, tx).is_none(),
        "writer must not see its own deleted edge"
    );
    assert!(
        !store.is_edge_visible_versioned(eid, epoch, tx),
        "writer visibility check must report the edge gone"
    );

    // Other transactions still see it (via the chain) — no dirty write.
    assert!(
        store.get_edge_versioned(eid, epoch, other_tx).is_some(),
        "other session must still see the not-yet-committed edge"
    );
    assert!(
        store.is_edge_visible_versioned(eid, epoch, other_tx),
        "other session visibility check must still report the edge present"
    );

    // Candidate adjacency is NOT tombstoned (non-MVCC index, by design — true for
    // everyone; visibility is enforced by the expand operators' post-filter).
    assert!(
        store
            .forward_adj
            .edges_from(a)
            .iter()
            .any(|(_, e)| *e == eid),
        "forward adjacency must still contain the candidate edge before commit"
    );
    assert!(
        store
            .backward_adj
            .as_ref()
            .expect("default config enables backward adjacency")
            .edges_from(b)
            .iter()
            .any(|(_, e)| *e == eid),
        "backward adjacency must still contain the candidate edge before commit"
    );

    // Property removal is deferred: the stored property is still readable, and an
    // other-session read of the edge still carries it.
    assert_eq!(
        store.edge_properties.get(eid, &key),
        Some(Value::Int64(7)),
        "edge property must not be removed at delete time"
    );
    let other_view = store
        .get_edge_versioned(eid, epoch, other_tx)
        .expect("edge still visible to other session");
    assert_eq!(
        other_view.get_property("prop").and_then(|v| v.as_int64()),
        Some(7),
        "other session must still read the edge's property"
    );

    // Count decrement is deferred too.
    assert_eq!(
        store.live_edge_count.load(Ordering::Relaxed),
        live_before,
        "live-edge count must not change at delete time"
    );

    // Commit: finalize the deferred delete at a real commit epoch.
    let commit_epoch = EpochId::new(5);
    store.finalize_edge_deletes_by_id(tx, commit_epoch, &[(a, eid, b)]);

    // Gone for everyone at the commit epoch.
    assert!(
        !store.is_edge_visible_versioned(eid, commit_epoch, other_tx),
        "after finalize the edge must be invisible to all sessions"
    );
    // Adjacency tombstone is now applied.
    assert!(
        !store
            .forward_adj
            .edges_from(a)
            .iter()
            .any(|(_, e)| *e == eid),
        "forward adjacency tombstone must be applied at finalize"
    );
    assert!(
        !store
            .backward_adj
            .as_ref()
            .expect("default config enables backward adjacency")
            .edges_from(b)
            .iter()
            .any(|(_, e)| *e == eid),
        "backward adjacency tombstone must be applied at finalize"
    );
    // Property removal moved to finalize too.
    assert_eq!(
        store.edge_properties.get(eid, &key),
        None,
        "edge property must be removed at finalize"
    );
    // The live-edge count decrement moved to finalize.
    assert_eq!(
        store.live_edge_count.load(Ordering::Relaxed),
        live_before - 1,
        "live-edge count must decrease by one at finalize"
    );
}

/// Edge-delete rollback cleanliness (MVCC increment 2b, Task 3).
///
/// After a `delete_edge_transactional`, draining the pending set with
/// `take_pending_edge_deletes` and replaying it through
/// `rollback_pending_edge_deletes` must FULLY restore the edge — the chain is
/// unmarked (not left PENDING), so the edge is visible again to the writer.
/// Because the deferred model never touched adjacency, properties, or counts at
/// delete time, those are untouched throughout. Finally the pending set must be
/// drained (the `take` already emptied it). This is the only probe that
/// distinguishes a real restore from leaked-PENDING state.
#[test]
fn edge_delete_rollback_restores() {
    use grafeo_common::types::PropertyKey;

    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    // Edge with a property so we can assert it survives the rollback.
    let eid = store.create_edge_with_props(a, b, "R", [("prop", Value::from(7i64))]);

    let key = PropertyKey::new("prop");
    let epoch = store.current_epoch();
    let tx = TransactionId::new(2);

    let live_before = store.live_edge_count.load(Ordering::Relaxed);

    // Transactional delete: stamps PENDING deleted_epoch by `tx`.
    assert!(store.delete_edge_transactional(eid, epoch, tx));
    assert!(
        !store.is_edge_visible_versioned(eid, epoch, tx),
        "writer must not see its own deleted edge before rollback"
    );

    // Rollback: drain the pending set and unmark the chain.
    let ed = store.take_pending_edge_deletes(tx);
    assert_eq!(
        ed,
        vec![(a, eid, b)],
        "pending edge-delete set must carry the (src, edge, dst) tuple"
    );
    store.rollback_pending_edge_deletes(tx, &ed);

    // FULLY restored: chain unmarked, edge visible to the writer again (NOT a
    // leaked-PENDING state — a leak would leave it invisible to `tx`).
    assert!(
        store.is_edge_visible_versioned(eid, epoch, tx),
        "edge must be visible to the writer again after rollback (chain unmarked)"
    );
    assert!(
        store.get_edge_versioned(eid, epoch, tx).is_some(),
        "writer must read the restored edge after rollback"
    );

    // Adjacency was never touched (deferred path) — still present both directions.
    assert!(
        store
            .forward_adj
            .edges_from(a)
            .iter()
            .any(|(_, e)| *e == eid),
        "forward adjacency must still contain the edge after rollback"
    );
    assert!(
        store
            .backward_adj
            .as_ref()
            .expect("default config enables backward adjacency")
            .edges_from(b)
            .iter()
            .any(|(_, e)| *e == eid),
        "backward adjacency must still contain the edge after rollback"
    );

    // Live-edge count unchanged (decrement was deferred, never applied).
    assert_eq!(
        store.live_edge_count.load(Ordering::Relaxed),
        live_before,
        "live-edge count must be unchanged after rollback"
    );

    // Property survives the rollback (removal was deferred, never applied).
    assert_eq!(
        store.edge_properties.get(eid, &key),
        Some(Value::Int64(7)),
        "edge property must survive the rollback"
    );

    // The pending set is drained — `take` already emptied it.
    assert!(
        store.take_pending_edge_deletes(tx).is_empty(),
        "pending edge-delete set must be drained after take"
    );
}

/// Re-deleting an edge already deleted by the SAME transaction must be an
/// idempotent no-op (MVCC increment 2b). The PENDING `deleted_epoch` is
/// `u64::MAX`, so a naive `visible_at(epoch)` re-delete guard still sees the
/// record and re-stamps it, pushing a DUPLICATE `(src, edge, dst)` into the
/// pending set; `finalize_edge_deletes_by_id` would then decrement counts twice.
/// The tx-aware guard (`visible_to`, which hides an edge from the tx that
/// deleted it) makes the second call return `false` with no second push.
#[test]
fn edge_delete_transactional_is_idempotent_per_tx() {
    use grafeo_common::types::EpochId;

    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    let eid = store.create_edge(a, b, "R");

    let epoch = store.current_epoch();
    let tx = TransactionId::new(2);
    let live_before = store.live_edge_count.load(Ordering::Relaxed);

    // First delete succeeds and records the edge once.
    assert!(store.delete_edge_transactional(eid, epoch, tx));
    // Second delete by the SAME tx is a no-op: the edge is already gone for `tx`.
    assert!(
        !store.delete_edge_transactional(eid, epoch, tx),
        "re-delete by the same tx must be an idempotent no-op (return false)"
    );

    // The pending set must carry the edge exactly ONCE (not twice).
    {
        let pending = store.pending_tx_edge_deletes.read();
        let entries = pending.get(&tx).map_or(0, |v| v.len());
        assert_eq!(
            entries, 1,
            "pending edge-delete set must carry the edge exactly once after a re-delete"
        );
    }

    // Finalize must decrement the live-edge count by exactly ONE (not two).
    let commit_epoch = EpochId::new(5);
    let pending = store.take_pending_edge_deletes(tx);
    store.finalize_edge_deletes_by_id(tx, commit_epoch, &pending);
    assert_eq!(
        store.live_edge_count.load(Ordering::Relaxed),
        live_before - 1,
        "live-edge count must decrease by exactly one despite the duplicate delete attempt"
    );
}

// ── Read-tracker registry ────────────────────────────────────────────────────

#[test]
fn test_read_tracker_registered_records_node() {
    use crate::execution::operators::{ReadTracker, SharedReadTracker};
    use grafeo_common::types::{EdgeId, NodeId};
    use parking_lot::Mutex;
    use std::sync::Arc;

    struct SpyTracker {
        nodes: Mutex<Vec<NodeId>>,
        edges: Mutex<Vec<EdgeId>>,
    }
    impl ReadTracker for SpyTracker {
        fn record_node_read(&self, _tx: TransactionId, id: NodeId) {
            self.nodes.lock().push(id);
        }
        fn record_edge_read(&self, _tx: TransactionId, id: EdgeId) {
            self.edges.lock().push(id);
        }
    }

    let store = LpgStore::new().unwrap();
    let tx = TransactionId::new(42);
    let spy = Arc::new(SpyTracker {
        nodes: Mutex::new(Vec::new()),
        edges: Mutex::new(Vec::new()),
    });
    let tracker: SharedReadTracker = spy.clone();

    // Before registration: no-op.
    store.record_read_node(tx, NodeId::new(1));
    store.record_read_edge(tx, EdgeId::new(1));
    assert!(
        spy.nodes.lock().is_empty(),
        "no recording before registration"
    );
    assert!(
        spy.edges.lock().is_empty(),
        "no recording before registration"
    );

    // After registration: reads are recorded.
    store.register_read_tracker(tx, tracker);
    store.record_read_node(tx, NodeId::new(5));
    store.record_read_edge(tx, EdgeId::new(7));
    assert_eq!(
        *spy.nodes.lock(),
        vec![NodeId::new(5)],
        "node read recorded"
    );
    assert_eq!(
        *spy.edges.lock(),
        vec![EdgeId::new(7)],
        "edge read recorded"
    );

    // After unregistration: no further recording.
    store.unregister_read_tracker(tx);
    store.record_read_node(tx, NodeId::new(99));
    store.record_read_edge(tx, EdgeId::new(99));
    assert_eq!(
        spy.nodes.lock().len(),
        1,
        "no extra node read after unregistration"
    );
    assert_eq!(
        spy.edges.lock().len(),
        1,
        "no extra edge read after unregistration"
    );
}

#[test]
fn test_read_tracker_unregistered_tx_is_noop() {
    // record_read_* for a tx that was never registered must be a silent no-op
    // (no panic, nothing collected anywhere).
    let store = LpgStore::new().unwrap();
    let tx = TransactionId::new(999);
    // These must not panic.
    store.record_read_node(tx, NodeId::new(1));
    store.record_read_edge(tx, EdgeId::new(1));
}

#[test]
fn test_read_tracker_cleared_by_store_clear() {
    use crate::execution::operators::{ReadTracker, SharedReadTracker};
    use grafeo_common::types::{EdgeId, NodeId};
    use parking_lot::Mutex;
    use std::sync::Arc;

    struct SpyTracker {
        nodes: Mutex<Vec<NodeId>>,
        edges: Mutex<Vec<EdgeId>>,
    }
    impl ReadTracker for SpyTracker {
        fn record_node_read(&self, _tx: TransactionId, id: NodeId) {
            self.nodes.lock().push(id);
        }
        fn record_edge_read(&self, _tx: TransactionId, id: EdgeId) {
            self.edges.lock().push(id);
        }
    }

    let store = LpgStore::new().unwrap();
    let tx = TransactionId::new(1);
    let spy = Arc::new(SpyTracker {
        nodes: Mutex::new(Vec::new()),
        edges: Mutex::new(Vec::new()),
    });
    let tracker: SharedReadTracker = spy.clone();

    store.register_read_tracker(tx, tracker);
    // Sanity: records before clear.
    store.record_read_node(tx, NodeId::new(3));
    assert_eq!(spy.nodes.lock().len(), 1);

    // clear() must drop the tracker entry.
    store.clear();
    store.record_read_node(tx, NodeId::new(9));
    assert_eq!(
        spy.nodes.lock().len(),
        1,
        "read tracker must be cleared by store.clear()"
    );
}

// ── Store-level visible-read chokepoints record into the tracker ─────────────

/// Builds a reusable spy tracker + helper to avoid duplication across tests.
#[cfg(test)]
mod visible_read_recording {
    use super::*;
    use crate::execution::operators::{ReadTracker, SharedReadTracker};
    use grafeo_common::types::{EdgeId, NodeId};
    use parking_lot::Mutex;
    use std::sync::Arc;

    pub struct SpyTracker {
        pub nodes: Mutex<Vec<NodeId>>,
        pub edges: Mutex<Vec<EdgeId>>,
    }

    impl SpyTracker {
        pub fn new() -> Arc<Self> {
            Arc::new(Self {
                nodes: Mutex::new(Vec::new()),
                edges: Mutex::new(Vec::new()),
            })
        }
    }

    impl ReadTracker for SpyTracker {
        fn record_node_read(&self, _tx: TransactionId, id: NodeId) {
            self.nodes.lock().push(id);
        }
        fn record_edge_read(&self, _tx: TransactionId, id: EdgeId) {
            self.edges.lock().push(id);
        }
    }

    /// Creates a store with:
    ///   - two committed nodes (n_visible, n_deleted) with properties
    ///   - one committed edge between them (e_visible)
    ///   - n_deleted is then deleted
    ///   - a Serializable-tx read tracker registered for `tx`
    pub fn fixture() -> (
        LpgStore,
        TransactionId,
        NodeId,
        NodeId,
        EdgeId,
        Arc<SpyTracker>,
    ) {
        let store = LpgStore::new().unwrap();
        let epoch = store.current_epoch();

        let n_visible = store.create_node_versioned(&["Person"], epoch, TransactionId::SYSTEM);
        let n_deleted = store.create_node_versioned(&["Person"], epoch, TransactionId::SYSTEM);
        let e_visible = store.create_edge_versioned(
            n_visible,
            n_deleted,
            "KNOWS",
            epoch,
            TransactionId::SYSTEM,
        );
        store.set_node_property(n_visible, "name", Value::from("Alix"));
        store.set_node_property(n_deleted, "name", Value::from("ghost"));
        store.set_edge_property(e_visible, "since", Value::from(2020i64));
        // Now delete n_deleted so visibility checks return false for it
        store.delete_node(n_deleted);
        // n_deleted is gone — but e_visible still exists (endpoints: n_visible→n_deleted)
        // (edge endpoints survive node deletes unless DETACH DELETE is used)

        let tx = TransactionId::new(77);
        let spy = SpyTracker::new();
        let tracker: SharedReadTracker = Arc::clone(&spy) as SharedReadTracker;
        store.register_read_tracker(tx, tracker);

        (store, tx, n_visible, n_deleted, e_visible, spy)
    }
}

#[test]
fn test_get_node_versioned_records_visible_node() {
    use visible_read_recording::fixture;
    let (store, tx, n_visible, n_deleted, _e, spy) = fixture();
    let epoch = store.current_epoch();

    // Visible node → recorded
    let node = store.get_node_versioned(n_visible, epoch, tx);
    assert!(node.is_some(), "expected node to be visible");
    assert!(
        spy.nodes.lock().contains(&n_visible),
        "visible node must be recorded"
    );

    // Deleted node → not recorded
    let deleted = store.get_node_versioned(n_deleted, epoch, tx);
    assert!(deleted.is_none(), "deleted node should not be visible");
    assert!(
        !spy.nodes.lock().contains(&n_deleted),
        "non-visible (deleted) node must NOT be recorded"
    );

    // tx=None → no recording
    spy.nodes.lock().clear();
    let _ = store.get_node_versioned(n_visible, epoch, TransactionId::new(999));
    assert!(
        spy.nodes.lock().is_empty(),
        "unregistered tx must not record anything"
    );
}

#[test]
fn test_get_edge_versioned_records_visible_edge() {
    use visible_read_recording::fixture;
    let (store, tx, _n, _nd, e_visible, spy) = fixture();
    let epoch = store.current_epoch();

    // Visible edge → recorded
    let edge = store.get_edge_versioned(e_visible, epoch, tx);
    assert!(edge.is_some(), "expected edge to be visible");
    assert!(
        spy.edges.lock().contains(&e_visible),
        "visible edge must be recorded"
    );
}

#[test]
fn test_is_node_visible_versioned_records_on_true() {
    use visible_read_recording::fixture;
    let (store, tx, n_visible, n_deleted, _e, spy) = fixture();
    let epoch = store.current_epoch();

    // True → recorded
    assert!(store.is_node_visible_versioned(n_visible, epoch, tx));
    assert!(
        spy.nodes.lock().contains(&n_visible),
        "visible node must be recorded on true return"
    );

    // False (deleted) → not recorded
    spy.nodes.lock().clear();
    assert!(!store.is_node_visible_versioned(n_deleted, epoch, tx));
    assert!(
        spy.nodes.lock().is_empty(),
        "non-visible (deleted) node must NOT be recorded"
    );
}

#[test]
fn test_is_edge_visible_versioned_records_on_true() {
    use visible_read_recording::fixture;
    let (store, tx, _n, _nd, e_visible, spy) = fixture();
    let epoch = store.current_epoch();

    assert!(store.is_edge_visible_versioned(e_visible, epoch, tx));
    assert!(
        spy.edges.lock().contains(&e_visible),
        "visible edge must be recorded"
    );
}

#[test]
fn test_filter_visible_node_ids_versioned_records_each_visible_node() {
    use visible_read_recording::fixture;
    let (store, tx, n_visible, n_deleted, _e, spy) = fixture();
    let epoch = store.current_epoch();

    let visible = store.filter_visible_node_ids_versioned(&[n_visible, n_deleted], epoch, tx);
    assert_eq!(
        visible,
        vec![n_visible],
        "only n_visible should pass filter"
    );
    assert!(
        spy.nodes.lock().contains(&n_visible),
        "visible node must be in recorded set"
    );
    assert!(
        !spy.nodes.lock().contains(&n_deleted),
        "non-visible node must NOT be recorded"
    );
}

#[test]
fn test_read_node_property_visible_records_node() {
    use grafeo_common::types::PropertyKey;
    use visible_read_recording::fixture;
    let (store, tx, n_visible, _nd, _e, spy) = fixture();
    let epoch = store.current_epoch();
    let key = PropertyKey::new("name");

    // With Some(tx) → records the node
    let val = store.read_node_property_visible(n_visible, &key, epoch, Some(tx));
    assert!(val.is_some(), "property should exist");
    assert!(
        spy.nodes.lock().contains(&n_visible),
        "node must be recorded on property read"
    );

    // With None (no tx) → no recording
    spy.nodes.lock().clear();
    let _ = store.read_node_property_visible(n_visible, &key, epoch, None);
    assert!(spy.nodes.lock().is_empty(), "tx=None must not record");
}

#[test]
fn test_read_edge_property_visible_records_edge() {
    use grafeo_common::types::PropertyKey;
    use visible_read_recording::fixture;
    let (store, tx, _n, _nd, e_visible, spy) = fixture();
    let epoch = store.current_epoch();
    let key = PropertyKey::new("since");

    let val = store.read_edge_property_visible(e_visible, &key, epoch, Some(tx));
    assert!(val.is_some(), "property should exist");
    assert!(
        spy.edges.lock().contains(&e_visible),
        "edge must be recorded on property read"
    );

    // With None → no recording
    spy.edges.lock().clear();
    let _ = store.read_edge_property_visible(e_visible, &key, epoch, None);
    assert!(spy.edges.lock().is_empty(), "tx=None must not record");
}

#[test]
fn test_read_node_properties_visible_records_node() {
    use visible_read_recording::fixture;
    let (store, tx, n_visible, _nd, _e, spy) = fixture();
    let epoch = store.current_epoch();

    let props = store.read_node_properties_visible(n_visible, epoch, Some(tx));
    assert!(!props.is_empty(), "properties should exist");
    assert!(
        spy.nodes.lock().contains(&n_visible),
        "node must be recorded on whole-entity property read"
    );

    spy.nodes.lock().clear();
    let _ = store.read_node_properties_visible(n_visible, epoch, None);
    assert!(spy.nodes.lock().is_empty(), "tx=None must not record");
}

#[test]
fn test_read_edge_properties_visible_records_edge() {
    use visible_read_recording::fixture;
    let (store, tx, _n, _nd, e_visible, spy) = fixture();
    let epoch = store.current_epoch();

    let props = store.read_edge_properties_visible(e_visible, epoch, Some(tx));
    assert!(!props.is_empty(), "properties should exist");
    assert!(
        spy.edges.lock().contains(&e_visible),
        "edge must be recorded on whole-entity property read"
    );

    spy.edges.lock().clear();
    let _ = store.read_edge_properties_visible(e_visible, epoch, None);
    assert!(spy.edges.lock().is_empty(), "tx=None must not record");
}

#[test]
fn test_read_node_labels_visible_records_node() {
    use visible_read_recording::fixture;
    let (store, tx, n_visible, _nd, _e, spy) = fixture();
    let epoch = store.current_epoch();

    let labels = store.read_node_labels_visible(n_visible, epoch, Some(tx));
    assert!(!labels.is_empty(), "labels should exist");
    assert!(
        spy.nodes.lock().contains(&n_visible),
        "node must be recorded on label read"
    );

    spy.nodes.lock().clear();
    let _ = store.read_node_labels_visible(n_visible, epoch, None);
    assert!(spy.nodes.lock().is_empty(), "tx=None must not record");
}

#[test]
fn test_nodes_by_label_visible_records_each_returned_node() {
    use grafeo_common::types::LabelId;
    use visible_read_recording::fixture;
    let (store, tx, n_visible, n_deleted, _e, spy) = fixture();
    let epoch = store.current_epoch();

    // GE3: recording happens in filter_visible_node_ids_in_label_versioned,
    // not in nodes_by_label_visible (nodes_by_label_visible is now a pure
    // ID-merge function; the MVCC filter records with label context).
    let label_id = store.label_id("Person").expect("Person label must exist");
    let all_ids = store.nodes_by_label_visible("Person", Some(tx));
    let ids =
        store.filter_visible_node_ids_in_label_versioned(&all_ids, epoch, tx, LabelId(label_id));

    // Only n_visible should survive the MVCC filter (n_deleted is deleted).
    assert!(ids.contains(&n_visible), "n_visible must be in the result");
    assert!(
        !ids.contains(&n_deleted),
        "deleted node must be filtered out by MVCC"
    );
    // The spy must have recorded n_visible (via label-context recording).
    assert!(
        spy.nodes.lock().contains(&n_visible),
        "n_visible must be recorded by filter_visible_node_ids_in_label_versioned"
    );
    assert!(
        !spy.nodes.lock().contains(&n_deleted),
        "deleted node must not be recorded"
    );
}

// ── edges_from_versioned / neighbors_versioned ───────────────────────────────

/// Tx A snapshots at E0. Another tx creates edge F (src→dst) and commits at
/// E1 > E0. `edges_from_versioned(src, Outgoing, E0, tx_a)` must NOT include F
/// because F was created after A's snapshot.
#[test]
fn edges_from_versioned_excludes_concurrent_committed_edge() {
    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["N"]);
    let dst = store.create_node(&["N"]);

    // Tx A's snapshot epoch — taken before the concurrent tx runs.
    let e0 = store.new_epoch();
    let tx_a = TransactionId::new(10);

    // Another tx creates edge F and commits at E1.
    let tx_other = TransactionId::new(11);
    let e1_start = store.new_epoch();
    let edge_f = store.create_edge_versioned(src, dst, "F", e1_start, tx_other);
    let e1 = store.new_epoch();
    store.finalize_entities_by_id(tx_other, e1, &[], &[edge_f]);

    // A's versioned traversal at E0 must not see F.
    let edges = store.edges_from_versioned(src, Direction::Outgoing, e0, tx_a);
    assert!(
        !edges.iter().any(|&(_, eid)| eid == edge_f),
        "edge F created after snapshot E0 must not appear in edges_from_versioned"
    );
}

/// Edge G exists at E0. Tx A snapshots at E0. Another tx deletes G and commits
/// at E1 > E0. `edges_from_versioned(src, Outgoing, E0, tx_a)` MUST still
/// include G — it was visible at A's snapshot epoch. This test fails if the
/// raw adjacency (including soft-deleted entries) is not used.
#[test]
fn edges_from_versioned_includes_edge_deleted_after_my_start() {
    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["N"]);
    let dst = store.create_node(&["N"]);

    // Edge G exists before both snapshots.
    let e0_pre = store.new_epoch();
    let edge_g = store.create_edge_versioned(src, dst, "G", e0_pre, TransactionId::SYSTEM);
    let e0 = store.new_epoch();
    store.finalize_entities_by_id(TransactionId::SYSTEM, e0, &[], &[edge_g]);

    // Tx A snapshots at E0 (edge G is committed and visible).
    let tx_a = TransactionId::new(20);
    let snapshot_e0 = e0;

    // Another tx deletes G and commits at E1.
    let tx_del = TransactionId::new(21);
    let e1_start = store.new_epoch();
    store.delete_edge_transactional(edge_g, e1_start, tx_del);
    let e1 = store.new_epoch();
    // finalize_edge_deletes_by_id applies the adjacency tombstone (mark_deleted).
    let pending = store.take_pending_edge_deletes(tx_del);
    store.finalize_edge_deletes_by_id(tx_del, e1, &pending);

    // Sanity: the non-versioned path no longer sees G (tombstone applied).
    let non_versioned = store
        .edges_from(src, Direction::Outgoing)
        .collect::<Vec<_>>();
    assert!(
        !non_versioned.iter().any(|&(_, eid)| eid == edge_g),
        "non-versioned path should not see deleted edge G"
    );

    // Versioned path at E0 MUST still see G (deleted after snapshot).
    let edges = store.edges_from_versioned(src, Direction::Outgoing, snapshot_e0, tx_a);
    assert!(
        edges.iter().any(|&(_, eid)| eid == edge_g),
        "edge G deleted after snapshot E0 must still appear in edges_from_versioned \
         — this fails if soft-deleted entries are pre-filtered from raw adjacency"
    );
}

/// Tx A creates edge H (uncommitted / PENDING). `edges_from_versioned` for tx A
/// must include H (read-your-writes).
#[test]
fn edges_from_versioned_includes_own_pending_edge() {
    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["N"]);
    let dst = store.create_node(&["N"]);

    let tx_a = TransactionId::new(30);
    let epoch = store.new_epoch();

    // Tx A creates edge H — PENDING, not yet committed.
    let edge_h = store.create_edge_versioned(src, dst, "H", epoch, tx_a);

    // Tx A must see its own uncommitted edge.
    let edges = store.edges_from_versioned(src, Direction::Outgoing, epoch, tx_a);
    assert!(
        edges.iter().any(|&(_, eid)| eid == edge_h),
        "tx A must see its own pending edge H in edges_from_versioned (read-your-writes)"
    );

    // Another tx at the same epoch must NOT see H (not yet committed).
    let tx_other = TransactionId::new(31);
    let edges_other = store.edges_from_versioned(src, Direction::Outgoing, epoch, tx_other);
    assert!(
        !edges_other.iter().any(|&(_, eid)| eid == edge_h),
        "other tx must not see tx A's pending edge H"
    );
}

/// Under a Serializable tx with a registered read-tracker,
/// `edges_from_versioned` populates the read-set with visible edge IDs.
/// The recording rides on `is_edge_visible_versioned`.
#[test]
fn edges_from_versioned_records_reads() {
    use crate::execution::operators::{ReadTracker, SharedReadTracker};
    use parking_lot::Mutex;
    use std::sync::Arc;

    struct EdgeSpy {
        edges: Mutex<Vec<EdgeId>>,
    }
    impl ReadTracker for EdgeSpy {
        fn record_node_read(&self, _tx: TransactionId, _id: NodeId) {}
        fn record_edge_read(&self, _tx: TransactionId, id: EdgeId) {
            self.edges.lock().push(id);
        }
    }

    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["N"]);
    let dst = store.create_node(&["N"]);

    // Two committed edges.
    let e0 = store.new_epoch();
    let edge1 = store.create_edge_versioned(src, dst, "R", e0, TransactionId::SYSTEM);
    let edge2 = store.create_edge_versioned(src, dst, "R", e0, TransactionId::SYSTEM);
    let e1 = store.new_epoch();
    store.finalize_entities_by_id(TransactionId::SYSTEM, e1, &[], &[edge1, edge2]);

    // Register a spy tracker for the Serializable tx.
    let tx = TransactionId::new(40);
    let spy = Arc::new(EdgeSpy {
        edges: Mutex::new(Vec::new()),
    });
    store.register_read_tracker(tx, Arc::clone(&spy) as SharedReadTracker);

    // Traverse: both visible edges must be recorded.
    let edges = store.edges_from_versioned(src, Direction::Outgoing, e1, tx);
    assert_eq!(edges.len(), 2, "both edges should be visible");

    let recorded = spy.edges.lock().clone();
    assert!(
        recorded.contains(&edge1),
        "edge1 must be recorded in the SSI read-set"
    );
    assert!(
        recorded.contains(&edge2),
        "edge2 must be recorded in the SSI read-set"
    );
}

/// A >T edge traversal of type `R` escalates the edge reads to the coarse
/// `RelType(R)` predicate via the store chokepoint: the spy sees
/// `record_read_edge_in_rel_type`, not bare `record_read_edge`.
/// Covers all three tx-visible edge read paths:
///   - `is_edge_visible_versioned` (traversal filter)
///   - `get_edge_versioned` (materialization)
///   - `edge_type_versioned` (Expand's per-candidate type-filter call)
#[test]
fn edge_reads_escalate_by_intrinsic_type_at_chokepoint() {
    use crate::execution::operators::{ReadTracker, SharedReadTracker};
    use grafeo_common::types::EdgeTypeId;
    use parking_lot::Mutex;
    use std::sync::Arc;

    #[derive(Default)]
    struct Spy {
        fine: Mutex<Vec<EdgeId>>,
        coarse: Mutex<Vec<(EdgeId, EdgeTypeId)>>,
    }
    impl ReadTracker for Spy {
        fn record_node_read(&self, _t: TransactionId, _i: NodeId) {}
        fn record_edge_read(&self, _t: TransactionId, id: EdgeId) {
            self.fine.lock().push(id);
        }
        fn record_read_edge_in_rel_type(&self, _t: TransactionId, id: EdgeId, rt: EdgeTypeId) {
            self.coarse.lock().push((id, rt));
        }
    }

    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["N"]);
    let b = store.create_node(&["N"]);
    let e0 = store.new_epoch();
    let edge = store.create_edge_versioned(a, b, "R", e0, TransactionId::SYSTEM);
    let e1 = store.new_epoch();
    store.finalize_entities_by_id(TransactionId::SYSTEM, e1, &[], &[edge]);

    // Capture the expected EdgeTypeId for "R" — for a fresh store with only "R"
    // this is EdgeTypeId::from(0), confirmed via committed_edge_type_id.
    let expected_rel_type = store
        .committed_edge_type_id(edge)
        .expect("edge must have a committed type id");

    let tx = TransactionId::new(40);
    let spy = Arc::new(Spy::default());
    store.register_read_tracker(tx, Arc::clone(&spy) as SharedReadTracker);

    // Visibility check (the traversal filter path) records via the coarse path.
    assert!(store.is_edge_visible_versioned(edge, e1, tx));
    // Materialization path too.
    let _ = store.get_edge_versioned(edge, e1, tx);
    // Type-resolver path (Expand calls this first for every candidate edge to
    // filter by relationship type — expand.rs:172).
    let _ = store.edge_type_versioned(edge, e1, tx);

    let coarse = spy.coarse.lock().clone();
    // Every coarse entry for `edge` must carry the correct RelType id.
    assert!(
        coarse.iter().any(|(id, _)| *id == edge),
        "edge read must route through record_read_edge_in_rel_type, got coarse={coarse:?}"
    );
    for (id, rt) in coarse.iter().filter(|(id, _)| *id == edge) {
        assert_eq!(
            *rt, expected_rel_type,
            "coarse entry for edge {id:?} must carry RelType {expected_rel_type:?}, got {rt:?}"
        );
    }
    // No path must fall back to bare fine record_read_edge.
    assert!(
        spy.fine.lock().is_empty(),
        "edge reads must NOT use bare record_read_edge on any visibility/type/materialization path"
    );
}

/// `get_node_versioned` (node materialization, e.g. `RETURN n`) must route its
/// read through [`ReadTracker::record_node_read_in_labels`], NOT the bare
/// [`record_node_read`]. Under an already-escalated label the engine bridge
/// short-circuits the re-add; here we verify the wiring by asserting the
/// `labeled` spy path fires and the bare `fine` path is empty.
#[test]
fn node_materialization_routes_through_label_carrying_read() {
    use crate::execution::operators::{ReadTracker, SharedReadTracker};
    use grafeo_common::types::LabelId;
    use parking_lot::Mutex;
    use std::sync::Arc;

    struct Spy {
        fine: Mutex<Vec<NodeId>>,
        labeled: Mutex<Vec<(NodeId, Vec<LabelId>)>>,
    }
    impl ReadTracker for Spy {
        fn record_node_read(&self, _tx: TransactionId, id: NodeId) {
            self.fine.lock().push(id);
        }
        fn record_edge_read(&self, _tx: TransactionId, _id: EdgeId) {}
        fn record_node_read_in_labels(&self, _tx: TransactionId, id: NodeId, labels: &[LabelId]) {
            self.labeled.lock().push((id, labels.to_vec()));
        }
    }

    let store = LpgStore::new().unwrap();
    // Create a committed node with a label.
    let epoch = store.new_epoch();
    let node = store.create_node_versioned(&["N"], epoch, TransactionId::SYSTEM);
    let commit_epoch = store.new_epoch();
    store.finalize_entities_by_id(TransactionId::SYSTEM, commit_epoch, &[node], &[]);

    let tx = TransactionId::new(55);
    let spy = Arc::new(Spy {
        fine: Mutex::new(Vec::new()),
        labeled: Mutex::new(Vec::new()),
    });
    store.register_read_tracker(tx, Arc::clone(&spy) as SharedReadTracker);

    // Call get_node_versioned (node materialization).
    let result = store.get_node_versioned(node, commit_epoch, tx);
    assert!(result.is_some(), "node must be visible");

    // The label-carrying path must have fired.
    let labeled = spy.labeled.lock().clone();
    assert!(
        labeled.iter().any(|(id, _)| *id == node),
        "get_node_versioned must route through record_node_read_in_labels, got labeled={labeled:?}"
    );
    // Labels must be non-empty (the node has label "N").
    for (id, labels) in labeled.iter().filter(|(id, _)| *id == node) {
        assert!(
            !labels.is_empty(),
            "labels passed to record_node_read_in_labels for node {id:?} must be non-empty"
        );
    }
    // The bare fine path must NOT have fired.
    assert!(
        spy.fine.lock().is_empty(),
        "node materialization must NOT use bare record_node_read, got fine={:?}",
        spy.fine.lock()
    );
}

/// `read_edge_properties_visible` (edge property materialization, e.g. `RETURN e`)
/// must route its read through the Task 1 intrinsic-RelType chokepoint
/// ([`ReadTracker::record_read_edge_in_rel_type`]), NOT the bare
/// [`record_edge_read`]. Under an already-escalated RelType the engine bridge
/// short-circuits the re-add; here we verify the wiring.
#[test]
fn edge_property_materialization_routes_through_rel_type() {
    use crate::execution::operators::{ReadTracker, SharedReadTracker};
    use grafeo_common::types::EdgeTypeId;
    use parking_lot::Mutex;
    use std::sync::Arc;

    #[derive(Default)]
    struct Spy {
        fine: Mutex<Vec<EdgeId>>,
        coarse: Mutex<Vec<(EdgeId, EdgeTypeId)>>,
    }
    impl ReadTracker for Spy {
        fn record_node_read(&self, _tx: TransactionId, _id: NodeId) {}
        fn record_edge_read(&self, _tx: TransactionId, id: EdgeId) {
            self.fine.lock().push(id);
        }
        fn record_read_edge_in_rel_type(&self, _tx: TransactionId, id: EdgeId, rt: EdgeTypeId) {
            self.coarse.lock().push((id, rt));
        }
    }

    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["N"]);
    let b = store.create_node(&["N"]);
    let e0 = store.new_epoch();
    let edge = store.create_edge_versioned(a, b, "R", e0, TransactionId::SYSTEM);
    let e1 = store.new_epoch();
    store.finalize_entities_by_id(TransactionId::SYSTEM, e1, &[], &[edge]);
    // Give the edge a property so the map is non-empty.
    store.set_edge_property(edge, "weight", Value::from(1i64));

    let expected_rel_type = store
        .committed_edge_type_id(edge)
        .expect("edge must have a committed rel type");

    let tx = TransactionId::new(60);
    let spy = Arc::new(Spy::default());
    store.register_read_tracker(tx, Arc::clone(&spy) as SharedReadTracker);

    // Call read_edge_properties_visible (edge property materialization).
    let props = store.read_edge_properties_visible(edge, e1, Some(tx));
    assert!(!props.is_empty(), "edge properties must be non-empty");

    // The coarse (rel-type) path must have fired.
    let coarse = spy.coarse.lock().clone();
    assert!(
        coarse.iter().any(|(id, _)| *id == edge),
        "read_edge_properties_visible must route through record_read_edge_in_rel_type, \
         got coarse={coarse:?}"
    );
    for (id, rt) in coarse.iter().filter(|(id, _)| *id == edge) {
        assert_eq!(
            *rt, expected_rel_type,
            "coarse entry for edge {id:?} must carry RelType {expected_rel_type:?}, got {rt:?}"
        );
    }
    // The bare fine path must NOT have fired.
    assert!(
        spy.fine.lock().is_empty(),
        "edge property materialization must NOT use bare record_edge_read, got fine={:?}",
        spy.fine.lock()
    );
}

// ── TI3: per-tx text-index delta ─────────────────────────────────────────────

/// Under a transaction, setting an indexed text property buffers the change into
/// `text_index_overlay` and does NOT mutate the committed `InvertedIndex`.
#[cfg(feature = "text-index")]
#[test]
fn tx_text_index_delta_buffers_set_without_mutating_committed_index() {
    use crate::index::text::{BM25Config, InvertedIndex};
    use parking_lot::RwLock;
    use std::sync::Arc;

    let store = LpgStore::new().unwrap();

    // Create a committed text index on :Doc(content).
    let committed_idx = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    store.add_text_index("Doc", "content", Arc::clone(&committed_idx));

    // Create a node with the Doc label (committed).
    let node = store.create_node(&["Doc"]);

    // Transactional buffered set.
    let tx = TransactionId::new(1);
    store.set_node_property_buffered(
        node,
        "content",
        Value::String("transactional hello world".into()),
        tx,
    );

    // The committed InvertedIndex must NOT see the new text.
    let results = committed_idx.read().search("hello world", 10);
    assert!(
        results.is_empty(),
        "committed index must be untouched by a transactional buffered write"
    );

    // The text_index_overlay must have the buffered change.
    {
        let overlay = store.text_index_overlay.read();
        let delta = overlay
            .get(&tx)
            .expect("text_index_overlay must have an entry for tx");
        let entry = delta.get("Doc:content", node);
        assert_eq!(
            entry,
            Some(&Some("transactional hello world".to_owned())),
            "text_index_overlay must hold the buffered text"
        );
    }
}

/// Under a transaction, removing an indexed text property buffers a tombstone
/// into `text_index_overlay` and does NOT touch the committed `InvertedIndex`.
#[cfg(feature = "text-index")]
#[test]
fn tx_text_index_delta_buffers_remove_tombstone() {
    use crate::index::text::{BM25Config, InvertedIndex};
    use parking_lot::RwLock;
    use std::sync::Arc;

    let store = LpgStore::new().unwrap();
    let committed_idx = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    store.add_text_index("Doc", "content", Arc::clone(&committed_idx));

    // Committed node + committed text (via auto-commit path).
    let node = store.create_node(&["Doc"]);
    store.set_node_property(node, "content", Value::String("committed text".into()));

    // Verify committed index has the entry.
    let results_before = committed_idx.read().search("committed text", 10);
    assert_eq!(
        results_before.len(),
        1,
        "committed index should have the node before transactional remove"
    );

    // Transactional buffered remove.
    let tx = TransactionId::new(2);
    store.remove_node_property_buffered(node, "content", tx);

    // Committed index must still have the entry (remove is only buffered).
    let results_after = committed_idx.read().search("committed text", 10);
    assert_eq!(
        results_after.len(),
        1,
        "committed index must be untouched by a transactional buffered remove"
    );

    // The overlay must have a None tombstone.
    {
        let overlay = store.text_index_overlay.read();
        let delta = overlay
            .get(&tx)
            .expect("text_index_overlay must have an entry for tx");
        let entry = delta.get("Doc:content", node);
        assert_eq!(
            entry,
            Some(&None),
            "text_index_overlay must hold a removal tombstone"
        );
    }
}

/// Rolling back a transaction drops its `text_index_overlay` entry.
#[cfg(feature = "text-index")]
#[test]
fn tx_text_index_delta_rollback_clears_overlay() {
    use crate::index::text::{BM25Config, InvertedIndex};
    use parking_lot::RwLock;
    use std::sync::Arc;

    let store = LpgStore::new().unwrap();
    let committed_idx = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    store.add_text_index("Doc", "content", Arc::clone(&committed_idx));

    let node = store.create_node(&["Doc"]);
    let tx = TransactionId::new(3);

    store.set_node_property_buffered(
        node,
        "content",
        Value::String("will be rolled back".into()),
        tx,
    );

    // Verify it's buffered.
    {
        let overlay = store.text_index_overlay.read();
        assert!(
            overlay.get(&tx).is_some(),
            "overlay must exist before rollback"
        );
    }

    // Rollback: drop_tx_overlay must clear both property delta and text-index delta.
    store.drop_tx_overlay(tx);

    {
        let overlay = store.text_index_overlay.read();
        assert!(
            overlay.get(&tx).is_none(),
            "text_index_overlay entry must be gone after rollback"
        );
    }

    // The committed index must still be clean.
    let results = committed_idx.read().search("rolled back", 10);
    assert!(
        results.is_empty(),
        "committed index must remain clean after rollback"
    );
}

// ── TI4: search_visible + search_text_visible ──────────────────────────────

/// A transaction that buffers `SET n.body='quantum rust'` via the text-index
/// delta must see that node returned by `search_text_visible` (read-your-writes),
/// while the committed-latest `search` does not.
#[cfg(feature = "text-index")]
#[test]
fn serializable_text_search_sees_own_uncommitted_insert() -> Result<(), Box<dyn std::error::Error>>
{
    use crate::index::text::{BM25Config, InvertedIndex};
    use parking_lot::RwLock;
    use std::sync::Arc;

    let store = LpgStore::new().unwrap();
    let committed_idx = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    store.add_text_index("Doc", "body", Arc::clone(&committed_idx));

    let node = store.create_node(&["Doc"]);
    let tx = TransactionId::new(42);
    let epoch = store.current_epoch();

    // Buffer a text-index set for the tx (does NOT touch committed index).
    store.set_node_property_buffered(node, "body", Value::String("quantum rust".into()), tx);

    // search_text_visible at (epoch, tx) must return the node (read-your-writes).
    let results = store.search_text_visible("Doc:body", "quantum", 10, epoch, tx)?;
    assert_eq!(results.len(), 1, "tx must see its own buffered insert");
    assert_eq!(results[0].0, node);

    // The committed-latest search must NOT see it.
    let committed_results = committed_idx.read().search("quantum", 10);
    assert!(
        committed_results.is_empty(),
        "committed search must not see uncommitted insert"
    );
    Ok(())
}

/// A document committed at epoch E2 must NOT be returned by `search_visible`
/// at a snapshot epoch E1 < E2.
#[cfg(feature = "text-index")]
#[test]
fn search_visible_excludes_committed_after_epoch() -> Result<(), Box<dyn std::error::Error>> {
    use crate::index::text::{BM25Config, InvertedIndex};
    use grafeo_common::types::EpochId;
    use parking_lot::RwLock;
    use std::sync::Arc;

    let store = LpgStore::new().unwrap();
    let idx = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    store.add_text_index("Doc", "body", Arc::clone(&idx));

    // Manually insert a versioned posting at epoch 5.
    idx.write().insert_versioned(
        grafeo_common::types::NodeId::new(1),
        "future document",
        EpochId::new(5),
        None,
    );

    // Searching at epoch 3 (before commit epoch 5) should find nothing.
    let results_e3 = store.search_text_visible(
        "Doc:body",
        "future document",
        10,
        EpochId::new(3),
        TransactionId::INVALID,
    )?;
    assert!(
        results_e3.is_empty(),
        "doc committed at E5 must be invisible at E3"
    );

    // Searching at epoch 5 should find it.
    let results_e5 = store.search_text_visible(
        "Doc:body",
        "future document",
        10,
        EpochId::new(5),
        TransactionId::INVALID,
    )?;
    assert_eq!(
        results_e5.len(),
        1,
        "doc committed at E5 must be visible at E5"
    );
    Ok(())
}

/// A document deleted at epoch E2 must still be returned by `search_visible`
/// at a snapshot E1 < E2.
#[cfg(feature = "text-index")]
#[test]
fn search_visible_includes_deleted_after_epoch() -> Result<(), Box<dyn std::error::Error>> {
    use crate::index::text::{BM25Config, InvertedIndex};
    use grafeo_common::types::EpochId;
    use parking_lot::RwLock;
    use std::sync::Arc;

    let store = LpgStore::new().unwrap();
    let idx = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    store.add_text_index("Doc", "body", Arc::clone(&idx));

    let nid = grafeo_common::types::NodeId::new(10);

    // Insert at epoch 1, delete at epoch 5.
    idx.write()
        .insert_versioned(nid, "ancient scroll", EpochId::new(1), None);
    idx.write().remove_versioned(nid, EpochId::new(5), None);

    // At epoch 3 the doc is still alive.
    let results_e3 = store.search_text_visible(
        "Doc:body",
        "ancient scroll",
        10,
        EpochId::new(3),
        TransactionId::INVALID,
    )?;
    assert_eq!(
        results_e3.len(),
        1,
        "doc deleted at E5 must be visible at E3"
    );

    // At epoch 5 the doc is gone.
    let results_e5 = store.search_text_visible(
        "Doc:body",
        "ancient scroll",
        10,
        EpochId::new(5),
        TransactionId::INVALID,
    )?;
    assert!(
        results_e5.is_empty(),
        "doc deleted at E5 must be invisible at E5"
    );
    Ok(())
}

/// A committed document that the current tx buffered a removal for must NOT
/// appear in `search_text_visible` for that tx.
#[cfg(feature = "text-index")]
#[test]
fn search_visible_excludes_tx_removed() -> Result<(), Box<dyn std::error::Error>> {
    use crate::index::text::{BM25Config, InvertedIndex};
    use parking_lot::RwLock;
    use std::sync::Arc;

    let store = LpgStore::new().unwrap();
    let committed_idx = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    store.add_text_index("Doc", "body", Arc::clone(&committed_idx));

    // Committed doc.
    let node = store.create_node(&["Doc"]);
    store.set_node_property(
        node,
        "body",
        Value::String("committed knowledge graph".into()),
    );

    // Verify committed search sees it.
    let committed = committed_idx.read().search("knowledge graph", 10);
    assert_eq!(committed.len(), 1);

    let tx = TransactionId::new(77);
    let epoch = store.current_epoch();

    // Buffer a removal of the committed doc.
    store.remove_node_property_buffered(node, "body", tx);

    // search_text_visible for tx must exclude the tx-removed doc.
    let results = store.search_text_visible("Doc:body", "knowledge graph", 10, epoch, tx)?;
    assert!(
        results.is_empty(),
        "tx-removed doc must be excluded from search_text_visible for that tx"
    );

    // But other transactions still see it via the committed search.
    let other_tx_results = store.search_text_visible(
        "Doc:body",
        "knowledge graph",
        10,
        epoch,
        TransactionId::new(99),
    )?;
    assert_eq!(
        other_tx_results.len(),
        1,
        "other transactions must still see the committed doc"
    );
    Ok(())
}

// ── TI5: single-source commit-promote (index epoch == property commit epoch) ──

/// The headline soundness invariant (spec §4a / §9.2).
///
/// For every node `N` and epoch `E`, the index's as-of-`E` term set for `N`
/// MUST equal `tokenize(committed text of N's indexed property as-of E)`.
/// The index is a consistent-by-construction projection of the property's
/// version chain — one commit stamps both.
///
/// This test drives the **real** commit flow at store level: buffer the write
/// into `text_index_overlay` / `tx_property_overlay`, advance the store epoch
/// to the commit epoch `C` (as `finalize_entities_by_id` does via
/// `sync_epoch(C)` BEFORE `apply_tx_overlay` runs), then `apply_tx_overlay`
/// promotes the property AND the index in the same `set_node_property` call —
/// both stamped at `current_epoch() == C`.
///
/// The property's own version chain (`read_node_property_visible(.., E, None)`)
/// is the oracle. With the legacy epoch-0 stamping this test FAILS: every
/// posting is created at epoch 0 so the index reports the latest text at every
/// old epoch, disagreeing with the property's as-of-E value.
#[cfg(feature = "text-index")]
#[test]
fn single_source_index_matches_property_history() -> Result<(), Box<dyn std::error::Error>> {
    use crate::index::text::{BM25Config, InvertedIndex};
    use grafeo_common::types::{EpochId, PropertyKey};
    use parking_lot::RwLock;
    use std::sync::Arc;

    let store = LpgStore::new().unwrap();
    let idx = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    store.add_text_index("Doc", "body", Arc::clone(&idx));

    let node = store.create_node(&["Doc"]);
    let key = PropertyKey::new("body");

    // Commit a buffered node-property write at an explicit commit epoch `C`,
    // reproducing the engine commit ordering: the store epoch is advanced to
    // `C` (finalize_entities_by_id -> sync_epoch(C)) BEFORE apply_tx_overlay
    // promotes the overlay (which re-applies the property + index at
    // current_epoch() == C).
    let commit_set = |tx: TransactionId, text: &str, c: u64| {
        store.set_node_property_buffered(node, "body", Value::String(text.into()), tx);
        store.sync_epoch(EpochId::new(c)); // finalize step
        store.apply_tx_overlay(tx); // promote: stamps property + index at C
    };
    let commit_remove = |tx: TransactionId, c: u64| {
        store.remove_node_property_buffered(node, "body", tx);
        store.sync_epoch(EpochId::new(c));
        store.apply_tx_overlay(tx);
    };

    // A short committed history of n.body across distinct commit epochs.
    //   E1: "alpha beta"
    //   E2: "beta gamma"   (alpha gone, gamma new)
    //   E3: "delta"        (beta & gamma gone)
    //   E4: property removed entirely
    commit_set(TransactionId::new(101), "alpha beta", 1);
    commit_set(TransactionId::new(102), "beta gamma", 2);
    commit_set(TransactionId::new(103), "delta", 3);
    commit_remove(TransactionId::new(104), 4);

    // The full vocabulary that ever appeared.
    let vocab = ["alpha", "beta", "gamma", "delta"];

    // For every committed epoch E in [0, 4] and every term, the index's
    // as-of-E membership MUST equal the property's as-of-E tokenization.
    for e in 0..=4u64 {
        let epoch = EpochId::new(e);

        // Oracle: tokenize the property's committed value as-of E.
        let oracle_terms: std::collections::HashSet<String> = store
            .read_node_property_visible(node, &key, epoch, None)
            .and_then(|v| match v {
                Value::String(s) => Some(s),
                _ => None,
            })
            .map(|s| {
                s.split_whitespace()
                    .map(str::to_string)
                    .collect::<std::collections::HashSet<_>>()
            })
            .unwrap_or_default();

        for term in vocab {
            let in_index = !store
                .search_text_visible("Doc:body", term, 10, epoch, TransactionId::INVALID)?
                .is_empty();
            let in_oracle = oracle_terms.contains(term);
            assert_eq!(
                in_index, in_oracle,
                "index/property divergence at epoch {e} for term '{term}': \
                 index_has={in_index} property_has={in_oracle} (oracle={oracle_terms:?})"
            );
        }
    }
    Ok(())
}

/// After a commit at epoch `C`, the promoted index postings carry
/// `created_epoch == C` (not the legacy epoch 0): a snapshot taken *before*
/// `C` does not see the post-commit text, and a snapshot at/after `C` does.
#[cfg(feature = "text-index")]
#[test]
fn single_source_post_commit_snapshot_boundary() -> Result<(), Box<dyn std::error::Error>> {
    use crate::index::text::{BM25Config, InvertedIndex};
    use grafeo_common::types::EpochId;
    use parking_lot::RwLock;
    use std::sync::Arc;

    let store = LpgStore::new().unwrap();
    let idx = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    store.add_text_index("Doc", "body", Arc::clone(&idx));

    let node = store.create_node(&["Doc"]);

    // Commit n.body = "quantum entanglement" at commit epoch C = 7.
    let tx = TransactionId::new(201);
    store.set_node_property_buffered(
        node,
        "body",
        Value::String("quantum entanglement".into()),
        tx,
    );
    store.sync_epoch(EpochId::new(7)); // finalize step advances store epoch to C
    store.apply_tx_overlay(tx); // promote at current_epoch() == 7

    // A pre-commit snapshot (E = 6 < 7) must NOT see the text.
    let before = store.search_text_visible(
        "Doc:body",
        "quantum",
        10,
        EpochId::new(6),
        TransactionId::INVALID,
    )?;
    assert!(
        before.is_empty(),
        "a snapshot before the commit epoch must not see the post-commit text"
    );

    // A snapshot at/after the commit epoch (E = 7) must see it.
    let at = store.search_text_visible(
        "Doc:body",
        "quantum",
        10,
        EpochId::new(7),
        TransactionId::INVALID,
    )?;
    assert_eq!(
        at.len(),
        1,
        "a snapshot at the commit epoch must see the post-commit text"
    );
    assert_eq!(at[0].0, node);

    // And the committed-latest search (COMMITTED_EPOCH filter) still sees it —
    // commit epochs are all <= COMMITTED_EPOCH = MAX-1.
    let latest = idx.read().search("quantum", 10);
    assert_eq!(
        latest.len(),
        1,
        "committed-latest search must still see the promoted posting"
    );
    Ok(())
}

// ── Task 7: IndexId predicate-read / index-write recording ──────────────────

/// `search_text_visible` must call `record_read_index` on the registered
/// read tracker, so a Serializable text search records the index read.
#[cfg(feature = "text-index")]
#[test]
fn search_text_visible_calls_record_index_read() -> Result<(), Box<dyn std::error::Error>> {
    use crate::execution::operators::{ReadTracker, SharedReadTracker};
    use crate::index::text::{BM25Config, InvertedIndex};
    use parking_lot::RwLock;
    use std::sync::{Arc, Mutex};

    // A spy ReadTracker that records every `record_index_read` call.
    struct IndexReadSpy {
        calls: Mutex<Vec<(TransactionId, String)>>,
    }
    impl ReadTracker for IndexReadSpy {
        fn record_node_read(&self, _tx: TransactionId, _id: grafeo_common::types::NodeId) {}
        fn record_edge_read(&self, _tx: TransactionId, _id: grafeo_common::types::EdgeId) {}
        fn record_index_read(&self, tx: TransactionId, key: &str) {
            self.calls.lock().unwrap().push((tx, key.to_string()));
        }
    }

    let store = LpgStore::new().unwrap();
    let idx = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    store.add_text_index("Doc", "body", Arc::clone(&idx));

    let tx = TransactionId::new(7);
    let epoch = store.current_epoch();

    let typed_spy = Arc::new(IndexReadSpy {
        calls: Mutex::new(Vec::new()),
    });
    store.register_read_tracker(tx, Arc::clone(&typed_spy) as SharedReadTracker);

    // Execute text search — should trigger record_index_read.
    let _ = store.search_text_visible("Doc:body", "rust", 10, epoch, tx)?;

    let calls = typed_spy.calls.lock().unwrap().clone();
    assert_eq!(
        calls.len(),
        1,
        "search_text_visible must call record_index_read exactly once"
    );
    assert_eq!(
        calls[0],
        (tx, "Doc:body".to_string()),
        "record_index_read must be called with the correct (tx, index_key)"
    );
    // Cleanup.
    store.unregister_read_tracker(tx);
    Ok(())
}

/// `buffer_text_index_set` must call `record_write_index` on the registered
/// write tracker, so a transactional indexed SET records the index write.
#[cfg(feature = "text-index")]
#[test]
fn buffer_text_index_set_calls_record_index_write() {
    use crate::execution::operators::{SharedWriteTracker, WriteTracker};
    use crate::index::text::{BM25Config, InvertedIndex};
    use parking_lot::RwLock;
    use std::sync::{Arc, Mutex};

    struct IndexWriteSpy {
        calls: Mutex<Vec<(TransactionId, String)>>,
    }
    impl WriteTracker for IndexWriteSpy {
        fn record_node_write(
            &self,
            _tx: TransactionId,
            _id: grafeo_common::types::NodeId,
        ) -> Result<(), crate::execution::operators::OperatorError> {
            Ok(())
        }
        fn record_edge_write(
            &self,
            _tx: TransactionId,
            _id: grafeo_common::types::EdgeId,
        ) -> Result<(), crate::execution::operators::OperatorError> {
            Ok(())
        }
        fn record_index_write(&self, tx: TransactionId, key: &str) {
            self.calls.lock().unwrap().push((tx, key.to_string()));
        }
    }

    let store = LpgStore::new().unwrap();
    let idx = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    store.add_text_index("Doc", "body", Arc::clone(&idx));

    let node = store.create_node(&["Doc"]);
    let tx = TransactionId::new(11);

    let spy = Arc::new(IndexWriteSpy {
        calls: Mutex::new(Vec::new()),
    });
    store.register_write_tracker(tx, Arc::clone(&spy) as SharedWriteTracker);

    // Buffer a SET on an indexed property.
    store.set_node_property_buffered(node, "body", Value::String("graphs are cool".into()), tx);

    let calls = spy.calls.lock().unwrap().clone();
    assert_eq!(
        calls.len(),
        1,
        "buffer_text_index_set must call record_index_write exactly once"
    );
    assert_eq!(
        calls[0],
        (tx, "Doc:body".to_string()),
        "record_index_write must be called with the correct (tx, index_key)"
    );

    store.unregister_write_tracker(tx);
}

/// `buffer_text_index_remove` must also call `record_write_index`.
#[cfg(feature = "text-index")]
#[test]
fn buffer_text_index_remove_calls_record_index_write() {
    use crate::execution::operators::{SharedWriteTracker, WriteTracker};
    use crate::index::text::{BM25Config, InvertedIndex};
    use parking_lot::RwLock;
    use std::sync::{Arc, Mutex};

    struct IndexWriteSpy {
        calls: Mutex<Vec<(TransactionId, String)>>,
    }
    impl WriteTracker for IndexWriteSpy {
        fn record_node_write(
            &self,
            _tx: TransactionId,
            _id: grafeo_common::types::NodeId,
        ) -> Result<(), crate::execution::operators::OperatorError> {
            Ok(())
        }
        fn record_edge_write(
            &self,
            _tx: TransactionId,
            _id: grafeo_common::types::EdgeId,
        ) -> Result<(), crate::execution::operators::OperatorError> {
            Ok(())
        }
        fn record_index_write(&self, tx: TransactionId, key: &str) {
            self.calls.lock().unwrap().push((tx, key.to_string()));
        }
    }

    let store = LpgStore::new().unwrap();
    let idx = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    store.add_text_index("Doc", "body", Arc::clone(&idx));

    let node = store.create_node(&["Doc"]);
    // First buffer a value so the node has a label entry, then test the remove path.
    let tx = TransactionId::new(13);

    let spy = Arc::new(IndexWriteSpy {
        calls: Mutex::new(Vec::new()),
    });
    store.register_write_tracker(tx, Arc::clone(&spy) as SharedWriteTracker);

    // Buffer a REMOVE on an indexed property.
    store.remove_node_property_buffered(node, "body", tx);

    let calls = spy.calls.lock().unwrap().clone();
    assert_eq!(
        calls.len(),
        1,
        "buffer_text_index_remove must call record_index_write exactly once"
    );
    assert_eq!(
        calls[0],
        (tx, "Doc:body".to_string()),
        "record_index_write must be called with the correct (tx, index_key)"
    );

    store.unregister_write_tracker(tx);
}

// ── VI4: vector_search_visible (snapshot visibility + as-of-E scoring + read-your-writes) ──

/// A transaction that buffers `SET n.embedding = <vector near query>` via
/// `set_node_property_buffered` must have that node returned by
/// `search_vector_visible` (read-your-writes), while the committed-latest
/// `vector_search` does not see it.
#[cfg(feature = "vector-index")]
#[test]
fn vector_search_visible_reads_own_writes() {
    use crate::graph::GraphStoreSearch;
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
    use std::sync::Arc;

    let store = LpgStore::new().unwrap();

    // Build a small HNSW index for (Doc, embedding).
    let config = HnswConfig::new(3, DistanceMetric::Euclidean);
    let idx = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config)));
    store.add_vector_index("Doc", "embedding", Arc::clone(&idx));

    // Commit node A with a vector far from query.
    let node_a = store.create_node(&["Doc"]);
    let far_vec: Arc<[f32]> = vec![10.0_f32, 10.0, 10.0].into();
    store.set_node_property(node_a, "embedding", Value::Vector(Arc::clone(&far_vec)));
    // Insert A into the HNSW index.
    let accessor_a = crate::graph::lpg::store::vector_accessor::SnapshotVectorAccessor {
        store: &store,
        property: grafeo_common::types::PropertyKey::new("embedding"),
        epoch: store.current_epoch(),
        tx: None,
    };
    idx.insert(node_a, &far_vec, &accessor_a);

    let epoch = store.current_epoch();
    let tx = TransactionId::new(42);

    // Node B: created in this tx (PENDING), buffered with a vector near query.
    let node_b = store.create_node_versioned(&["Doc"], epoch, tx);
    let near_vec: Arc<[f32]> = vec![0.1_f32, 0.1, 0.1].into();
    store.set_node_property_buffered(
        node_b,
        "embedding",
        Value::Vector(Arc::clone(&near_vec)),
        tx,
    );

    // search_vector_visible at (epoch, tx) must return B (read-your-writes).
    let results = store.search_vector_visible("Doc:embedding", &[0.0, 0.0, 0.0], 2, epoch, tx);
    let ids: Vec<_> = results.iter().map(|(id, _)| *id).collect();
    assert!(
        ids.contains(&node_b),
        "tx must see its own buffered vector insert B; got {ids:?}"
    );

    // The committed-latest vector_search must NOT see B.
    let committed_results = GraphStoreSearch::vector_search(
        &store,
        Some("Doc"),
        "embedding",
        &[0.0, 0.0, 0.0],
        10,
        DistanceMetric::Euclidean,
    );
    let committed_ids: Vec<_> = committed_results.iter().map(|(id, _)| *id).collect();
    assert!(
        !committed_ids.contains(&node_b),
        "committed search must not see uncommitted insert B; got {committed_ids:?}"
    );
}

/// A node committed after the snapshot `epoch` must not appear in
/// `search_vector_visible` at that epoch.
#[cfg(feature = "vector-index")]
#[test]
fn vector_search_visible_excludes_committed_after_epoch() {
    use grafeo_common::types::EpochId;

    let store = LpgStore::new().unwrap();

    // No HNSW index — brute-force-only path.
    // Node A committed at epoch 1.  Must call finalize_entities_by_id to promote
    // the PENDING version to the real commit epoch (mirrors the engine path).
    let tx1 = TransactionId::new(1);
    let node_a = store.create_node_versioned(&["Doc"], EpochId::new(1), tx1);
    store.set_node_property_buffered(
        node_a,
        "embedding",
        Value::Vector(vec![0.1_f32, 0.0, 0.0].into()),
        tx1,
    );
    store.finalize_entities_by_id(tx1, EpochId::new(1), &[node_a], &[]);
    store.apply_tx_overlay(tx1);

    // Node B committed at epoch 5.
    let tx2 = TransactionId::new(2);
    let node_b = store.create_node_versioned(&["Doc"], EpochId::new(5), tx2);
    store.set_node_property_buffered(
        node_b,
        "embedding",
        Value::Vector(vec![0.2_f32, 0.0, 0.0].into()),
        tx2,
    );
    store.finalize_entities_by_id(tx2, EpochId::new(5), &[node_b], &[]);
    store.apply_tx_overlay(tx2);

    // At epoch 3 (between the two commits) only A is visible.
    let results_e3 = store.search_vector_visible(
        "Doc:embedding",
        &[0.0, 0.0, 0.0],
        10,
        EpochId::new(3),
        TransactionId::INVALID,
    );
    let ids_e3: Vec<_> = results_e3.iter().map(|(id, _)| *id).collect();
    assert!(
        ids_e3.contains(&node_a),
        "node A committed at E1 must be visible at E3; got {ids_e3:?}"
    );
    assert!(
        !ids_e3.contains(&node_b),
        "node B committed at E5 must not be visible at E3; got {ids_e3:?}"
    );
}

/// The no-index fallback must preserve the index key's label predicate. This
/// also exercises a historical snapshot after the matching node was deleted:
/// retained label logs must make the pre-delete label visible without allowing
/// a different label with the same property to leak into the results.
#[cfg(feature = "vector-index")]
#[test]
fn vector_search_visible_brute_force_filters_label_at_historical_epoch() {
    use crate::graph::GraphStoreMut;
    use grafeo_common::types::EpochId;

    let store = LpgStore::new().unwrap();
    let created = EpochId::new(1);

    let tx_doc = TransactionId::new(101);
    let doc = store.create_node_versioned(&["Doc"], created, tx_doc);
    store.set_node_property_buffered(
        doc,
        "embedding",
        Value::Vector(vec![0.2_f32, 0.0, 0.0].into()),
        tx_doc,
    );
    store.finalize_entities_by_id(tx_doc, created, &[doc], &[]);
    store.apply_tx_overlay(tx_doc);

    let tx_image = TransactionId::new(102);
    let image = store.create_node_versioned(&["Image"], created, tx_image);
    store.set_node_property_buffered(
        image,
        "embedding",
        Value::Vector(vec![0.1_f32, 0.0, 0.0].into()),
        tx_image,
    );
    store.finalize_entities_by_id(tx_image, created, &[image], &[]);
    store.apply_tx_overlay(tx_image);

    let deleted = EpochId::new(5);
    let tx_delete = TransactionId::new(103);
    assert!(GraphStoreMut::delete_node_versioned(
        &store, doc, deleted, tx_delete
    ));
    store.sync_epoch(deleted);
    store.finalize_deletes_by_id(tx_delete, deleted, &[doc]);

    let results = store.search_vector_visible(
        "Doc:embedding",
        &[0.0, 0.0, 0.0],
        10,
        EpochId::new(3),
        TransactionId::INVALID,
    );
    let ids: Vec<_> = results.into_iter().map(|(id, _)| id).collect();
    assert_eq!(ids, vec![doc]);
    assert!(!ids.contains(&image));
}

/// `get_node_versioned(id, epoch, tx)` must materialize property values AS OF
/// `epoch` (snapshot isolation), not the current committed values. Before the
/// fix it checked existence-visibility at `epoch` but built the node from
/// `get_all` (current), so a transaction could observe another transaction's
/// commit made after its snapshot.
#[test]
fn get_node_versioned_reads_as_of_snapshot_not_current() {
    use grafeo_common::types::EpochId;

    let store = LpgStore::new().unwrap();
    // Node committed at epoch 1 with v = 1, then v = 2 committed at epoch 5.
    let n = store.create_node_versioned(&["N"], EpochId::new(1), TransactionId::SYSTEM);
    store.set_node_property_at_epoch(n, "v", Value::Int64(1), EpochId::new(1));
    store.set_node_property_at_epoch(n, "v", Value::Int64(2), EpochId::new(5));

    let key = PropertyKey::new("v");

    // A reader snapshotted at epoch 1 must see v = 1, NOT the current v = 2.
    let at_e1 = store
        .get_node_versioned(n, EpochId::new(1), TransactionId::SYSTEM)
        .expect("node visible at epoch 1");
    assert_eq!(
        at_e1.properties.get(&key),
        Some(&Value::Int64(1)),
        "get_node_versioned at epoch 1 must read the as-of-snapshot value, got {:?}",
        at_e1.properties.get(&key)
    );

    // A reader at epoch 5 sees the newer value.
    let at_e5 = store
        .get_node_versioned(n, EpochId::new(5), TransactionId::SYSTEM)
        .expect("node visible at epoch 5");
    assert_eq!(at_e5.properties.get(&key), Some(&Value::Int64(2)));
}

/// Edge counterpart: `get_edge_versioned` must materialize values AS OF `epoch`,
/// not the current committed values.
#[test]
fn get_edge_versioned_reads_as_of_snapshot_not_current() {
    use grafeo_common::types::EpochId;

    let store = LpgStore::new().unwrap();
    let a = store.create_node_versioned(&["N"], EpochId::new(1), TransactionId::SYSTEM);
    let b = store.create_node_versioned(&["N"], EpochId::new(1), TransactionId::SYSTEM);
    let e = store.create_edge_versioned(a, b, "T", EpochId::new(1), TransactionId::SYSTEM);
    store.set_edge_property_at_epoch(e, "w", Value::Int64(1), EpochId::new(1));
    store.set_edge_property_at_epoch(e, "w", Value::Int64(2), EpochId::new(5));

    let key = PropertyKey::new("w");
    let at_e1 = store
        .get_edge_versioned(e, EpochId::new(1), TransactionId::SYSTEM)
        .expect("edge visible at epoch 1");
    assert_eq!(
        at_e1.properties.get(&key),
        Some(&Value::Int64(1)),
        "get_edge_versioned at epoch 1 must read the as-of-snapshot value, got {:?}",
        at_e1.properties.get(&key)
    );
    let at_e5 = store
        .get_edge_versioned(e, EpochId::new(5), TransactionId::SYSTEM)
        .expect("edge visible at epoch 5");
    assert_eq!(at_e5.properties.get(&key), Some(&Value::Int64(2)));
}

// ── Task 5: Vector-index read/write recording for anti-phantom SSI ──────────

/// `search_vector_visible` must call `record_read_index` on the registered
/// read tracker, so a Serializable vector search records the index read.
/// Mirrors `search_text_visible_calls_record_index_read` for the vector path.
#[cfg(feature = "vector-index")]
#[test]
fn search_vector_visible_calls_record_index_read() {
    use crate::execution::operators::{ReadTracker, SharedReadTracker};
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
    use std::sync::{Arc, Mutex};

    // A spy ReadTracker that records every `record_index_read` call.
    struct IndexReadSpy {
        calls: Mutex<Vec<(TransactionId, String)>>,
    }
    impl ReadTracker for IndexReadSpy {
        fn record_node_read(&self, _tx: TransactionId, _id: grafeo_common::types::NodeId) {}
        fn record_edge_read(&self, _tx: TransactionId, _id: grafeo_common::types::EdgeId) {}
        fn record_index_read(&self, tx: TransactionId, key: &str) {
            self.calls.lock().unwrap().push((tx, key.to_string()));
        }
    }

    let store = LpgStore::new().unwrap();
    // Add a vector index so the index_key is known.
    let config = HnswConfig::new(3, DistanceMetric::Cosine);
    let idx = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config)));
    store.add_vector_index("Doc", "embedding", Arc::clone(&idx));

    let tx = TransactionId::new(17);
    let epoch = store.current_epoch();

    let spy = Arc::new(IndexReadSpy {
        calls: Mutex::new(Vec::new()),
    });
    store.register_read_tracker(tx, Arc::clone(&spy) as SharedReadTracker);

    // Execute a vector search — should trigger record_index_read as FIRST action.
    let _ = store.search_vector_visible("Doc:embedding", &[0.0, 1.0, 0.0], 5, epoch, tx);

    let calls = spy.calls.lock().unwrap().clone();
    assert_eq!(
        calls.len(),
        1,
        "search_vector_visible must call record_index_read exactly once; got {calls:?}"
    );
    assert_eq!(
        calls[0],
        (tx, "Doc:embedding".to_string()),
        "record_index_read must be called with the correct (tx, index_key)"
    );

    store.unregister_read_tracker(tx);
}

/// A LABEL-LESS vector search (empty label = "any label") must record a read for
/// EVERY vector index on the property, so a concurrent indexed SET on any label
/// forms the anti-phantom rw-edge — matching the per-(label,property) write-side
/// recording. Regression for the gap where a label-less scan under Serializable
/// recorded nothing.
#[cfg(feature = "vector-index")]
#[test]
fn label_less_vector_search_records_all_matching_indexes() {
    use crate::execution::operators::{ReadTracker, SharedReadTracker};
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
    use std::sync::{Arc, Mutex};

    struct IndexReadSpy {
        calls: Mutex<Vec<String>>,
    }
    impl ReadTracker for IndexReadSpy {
        fn record_node_read(&self, _tx: TransactionId, _id: grafeo_common::types::NodeId) {}
        fn record_edge_read(&self, _tx: TransactionId, _id: grafeo_common::types::EdgeId) {}
        fn record_index_read(&self, _tx: TransactionId, key: &str) {
            self.calls.lock().unwrap().push(key.to_string());
        }
    }

    let store = LpgStore::new().unwrap();
    let mk = || {
        Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
            3,
            DistanceMetric::Cosine,
        ))))
    };
    // Two vector indexes on the SAME property `embedding`, different labels.
    store.add_vector_index("Doc", "embedding", mk());
    store.add_vector_index("Article", "embedding", mk());
    // An index on a DIFFERENT property must NOT be recorded.
    store.add_vector_index("Doc", "other", mk());

    let tx = TransactionId::new(71);
    let epoch = store.current_epoch();
    let spy = Arc::new(IndexReadSpy {
        calls: Mutex::new(Vec::new()),
    });
    store.register_read_tracker(tx, Arc::clone(&spy) as SharedReadTracker);

    // Empty label = any label.
    let _ = store.search_vector_visible(":embedding", &[0.0, 1.0, 0.0], 5, epoch, tx);

    let calls = spy.calls.lock().unwrap().clone();
    assert!(
        calls.iter().any(|k| k == "Doc:embedding"),
        "must record the Doc:embedding index read; got {calls:?}"
    );
    assert!(
        calls.iter().any(|k| k == "Article:embedding"),
        "must record the Article:embedding index read; got {calls:?}"
    );
    assert!(
        !calls.iter().any(|k| k == "Doc:other"),
        "must NOT record an index on a different property; got {calls:?}"
    );

    store.unregister_read_tracker(tx);
}

/// A LABEL-LESS `search_vector_visible` must still apply read-your-writes — the
/// per-label filter in the overlay merge must be skipped for the any-label case.
#[cfg(feature = "vector-index")]
#[test]
fn label_less_vector_search_reads_own_writes() {
    use std::sync::Arc;
    let store = LpgStore::new().unwrap();

    let epoch = store.current_epoch();
    let tx = TransactionId::new(73);

    // Node created + vector buffered in this tx (no committed index needed —
    // brute-force-by-property path).
    let node_b = store.create_node_versioned(&["Doc"], epoch, tx);
    let near_vec: Arc<[f32]> = vec![0.1_f32, 0.1, 0.1].into();
    store.set_node_property_buffered(node_b, "embedding", Value::Vector(near_vec), tx);

    // Label-less visible search must see the tx's own buffered vector.
    let results = store.search_vector_visible(":embedding", &[0.0, 0.0, 0.0], 5, epoch, tx);
    let ids: Vec<_> = results.iter().map(|(id, _)| *id).collect();
    assert!(
        ids.contains(&node_b),
        "label-less search must read its own buffered write; got {ids:?}"
    );
}

/// `search_vector_visible` must record the index read even when there are no
/// results (zero-result search must still close the anti-phantom rw-edge).
#[cfg(feature = "vector-index")]
#[test]
fn search_vector_visible_records_index_read_when_no_results() {
    use crate::execution::operators::{ReadTracker, SharedReadTracker};
    use std::sync::{Arc, Mutex};

    struct IndexReadSpy {
        calls: Mutex<Vec<(TransactionId, String)>>,
    }
    impl ReadTracker for IndexReadSpy {
        fn record_node_read(&self, _tx: TransactionId, _id: grafeo_common::types::NodeId) {}
        fn record_edge_read(&self, _tx: TransactionId, _id: grafeo_common::types::EdgeId) {}
        fn record_index_read(&self, tx: TransactionId, key: &str) {
            self.calls.lock().unwrap().push((tx, key.to_string()));
        }
    }

    // No index registered — brute-force path with no nodes.
    let store = LpgStore::new().unwrap();
    let tx = TransactionId::new(19);
    let epoch = store.current_epoch();

    let spy = Arc::new(IndexReadSpy {
        calls: Mutex::new(Vec::new()),
    });
    store.register_read_tracker(tx, Arc::clone(&spy) as SharedReadTracker);

    let results = store.search_vector_visible("Doc:embedding", &[0.0, 0.0, 1.0], 5, epoch, tx);
    assert!(results.is_empty(), "no nodes → empty results");

    let calls = spy.calls.lock().unwrap().clone();
    assert_eq!(
        calls.len(),
        1,
        "zero-result vector search must still call record_index_read; got {calls:?}"
    );

    store.unregister_read_tracker(tx);
}

/// A buffered `SET` on a vector-indexed property must call `record_write_index`
/// on the registered write tracker. Mirrors `buffer_text_index_set_calls_record_index_write`.
#[cfg(feature = "vector-index")]
#[test]
fn set_node_property_buffered_calls_record_index_write_for_vector() {
    use crate::execution::operators::{SharedWriteTracker, WriteTracker};
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
    use std::sync::{Arc, Mutex};

    struct IndexWriteSpy {
        calls: Mutex<Vec<(TransactionId, String)>>,
    }
    impl WriteTracker for IndexWriteSpy {
        fn record_node_write(
            &self,
            _tx: TransactionId,
            _id: grafeo_common::types::NodeId,
        ) -> Result<(), crate::execution::operators::OperatorError> {
            Ok(())
        }
        fn record_edge_write(
            &self,
            _tx: TransactionId,
            _id: grafeo_common::types::EdgeId,
        ) -> Result<(), crate::execution::operators::OperatorError> {
            Ok(())
        }
        fn record_index_write(&self, tx: TransactionId, key: &str) {
            self.calls.lock().unwrap().push((tx, key.to_string()));
        }
    }

    let store = LpgStore::new().unwrap();
    let config = HnswConfig::new(3, DistanceMetric::Cosine);
    let idx = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config)));
    store.add_vector_index("Doc", "embedding", Arc::clone(&idx));

    let node = store.create_node(&["Doc"]);
    let tx = TransactionId::new(23);

    let spy = Arc::new(IndexWriteSpy {
        calls: Mutex::new(Vec::new()),
    });
    store.register_write_tracker(tx, Arc::clone(&spy) as SharedWriteTracker);

    // Buffer a SET on a vector-indexed property.
    store.set_node_property_buffered(
        node,
        "embedding",
        Value::Vector(vec![1.0_f32, 0.0, 0.0].into()),
        tx,
    );

    let calls = spy.calls.lock().unwrap().clone();
    assert_eq!(
        calls.len(),
        1,
        "set_node_property_buffered must call record_index_write exactly once for a vector-indexed property; got {calls:?}"
    );
    assert_eq!(
        calls[0],
        (tx, "Doc:embedding".to_string()),
        "record_index_write must be called with the correct (tx, index_key)"
    );

    store.unregister_write_tracker(tx);
}

/// A buffered `REMOVE` on a vector-indexed property must also call `record_write_index`.
#[cfg(feature = "vector-index")]
#[test]
fn remove_node_property_buffered_calls_record_index_write_for_vector() {
    use crate::execution::operators::{SharedWriteTracker, WriteTracker};
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
    use std::sync::{Arc, Mutex};

    struct IndexWriteSpy {
        calls: Mutex<Vec<(TransactionId, String)>>,
    }
    impl WriteTracker for IndexWriteSpy {
        fn record_node_write(
            &self,
            _tx: TransactionId,
            _id: grafeo_common::types::NodeId,
        ) -> Result<(), crate::execution::operators::OperatorError> {
            Ok(())
        }
        fn record_edge_write(
            &self,
            _tx: TransactionId,
            _id: grafeo_common::types::EdgeId,
        ) -> Result<(), crate::execution::operators::OperatorError> {
            Ok(())
        }
        fn record_index_write(&self, tx: TransactionId, key: &str) {
            self.calls.lock().unwrap().push((tx, key.to_string()));
        }
    }

    let store = LpgStore::new().unwrap();
    let config = HnswConfig::new(3, DistanceMetric::Cosine);
    let idx = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config)));
    store.add_vector_index("Doc", "embedding", Arc::clone(&idx));

    let node = store.create_node(&["Doc"]);
    let tx = TransactionId::new(29);

    let spy = Arc::new(IndexWriteSpy {
        calls: Mutex::new(Vec::new()),
    });
    store.register_write_tracker(tx, Arc::clone(&spy) as SharedWriteTracker);

    // Buffer a REMOVE on a vector-indexed property.
    store.remove_node_property_buffered(node, "embedding", tx);

    let calls = spy.calls.lock().unwrap().clone();
    assert_eq!(
        calls.len(),
        1,
        "remove_node_property_buffered must call record_index_write exactly once for a vector-indexed property; got {calls:?}"
    );
    assert_eq!(
        calls[0],
        (tx, "Doc:embedding".to_string()),
        "record_index_write must be called with the correct (tx, index_key)"
    );

    store.unregister_write_tracker(tx);
}

/// A buffered SET on a property that is NOT covered by any vector index must NOT
/// call `record_write_index` (no spurious index-write recording).
#[cfg(feature = "vector-index")]
#[test]
fn set_node_property_buffered_no_record_when_not_vector_indexed() {
    use crate::execution::operators::{SharedWriteTracker, WriteTracker};
    use std::sync::{Arc, Mutex};

    struct IndexWriteSpy {
        calls: Mutex<Vec<(TransactionId, String)>>,
    }
    impl WriteTracker for IndexWriteSpy {
        fn record_node_write(
            &self,
            _tx: TransactionId,
            _id: grafeo_common::types::NodeId,
        ) -> Result<(), crate::execution::operators::OperatorError> {
            Ok(())
        }
        fn record_edge_write(
            &self,
            _tx: TransactionId,
            _id: grafeo_common::types::EdgeId,
        ) -> Result<(), crate::execution::operators::OperatorError> {
            Ok(())
        }
        fn record_index_write(&self, tx: TransactionId, key: &str) {
            self.calls.lock().unwrap().push((tx, key.to_string()));
        }
    }

    // No vector index registered at all.
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Doc"]);
    let tx = TransactionId::new(31);

    let spy = Arc::new(IndexWriteSpy {
        calls: Mutex::new(Vec::new()),
    });
    store.register_write_tracker(tx, Arc::clone(&spy) as SharedWriteTracker);

    store.set_node_property_buffered(node, "embedding", Value::from("not a vector"), tx);

    let calls = spy.calls.lock().unwrap().clone();
    assert!(
        calls.is_empty(),
        "no index registered → record_write_index must not be called; got {calls:?}"
    );

    store.unregister_write_tracker(tx);
}

/// A node whose delete commits AFTER the snapshot `epoch` must still appear in
/// `search_vector_visible` at that epoch (delete-after-epoch stays visible).
#[cfg(feature = "vector-index")]
#[test]
fn vector_search_visible_includes_deleted_after_epoch() {
    use crate::graph::GraphStoreMut;
    use grafeo_common::types::EpochId;

    let store = LpgStore::new().unwrap();

    // Commit node A at epoch 1.  Must call finalize_entities_by_id to promote
    // the PENDING version to the real commit epoch (matches the engine path).
    let tx1 = TransactionId::new(1);
    let node_a = store.create_node_versioned(&["Doc"], EpochId::new(1), tx1);
    store.set_node_property_buffered(
        node_a,
        "embedding",
        Value::Vector(vec![0.1_f32, 0.0, 0.0].into()),
        tx1,
    );
    // Finalize the node version chain (PENDING → E1) and promote properties.
    store.finalize_entities_by_id(tx1, EpochId::new(1), &[node_a], &[]);
    store.apply_tx_overlay(tx1);

    // Delete A at epoch 5 via the trait (which calls delete_node_transactional).
    let tx_del = TransactionId::new(3);
    GraphStoreMut::delete_node_versioned(&store, node_a, EpochId::new(5), tx_del);
    store.sync_epoch(EpochId::new(5));
    store.finalize_deletes_by_id(tx_del, EpochId::new(5), &[node_a]);

    // At epoch 3 (before delete) node A should be visible.
    let results_e3 = store.search_vector_visible(
        "Doc:embedding",
        &[0.0, 0.0, 0.0],
        10,
        EpochId::new(3),
        TransactionId::INVALID,
    );
    let ids_e3: Vec<_> = results_e3.iter().map(|(id, _)| *id).collect();
    assert!(
        ids_e3.contains(&node_a),
        "node deleted at E5 must still be visible at E3; got {ids_e3:?}"
    );

    // At epoch 5 (at delete epoch) node A should be gone.
    let results_e5 = store.search_vector_visible(
        "Doc:embedding",
        &[0.0, 0.0, 0.0],
        10,
        EpochId::new(5),
        TransactionId::INVALID,
    );
    let ids_e5: Vec<_> = results_e5.iter().map(|(id, _)| *id).collect();
    assert!(
        !ids_e5.contains(&node_a),
        "node deleted at E5 must not be visible at E5; got {ids_e5:?}"
    );
}

// ============================================================================
// Vector-index GC horizon tests (tiered-storage path)
// ============================================================================

/// `node_deleted_at_or_below` must return `false` for a node committed AFTER
/// the GC horizon that has never been deleted.  The tiered branch used to
/// return `true` here (visible_at(horizon).is_none() is true for a live node
/// created after horizon), which would incorrectly GC the node from the HNSW.
#[cfg(all(feature = "vector-index", feature = "tiered-storage"))]
#[test]
fn tiered_gc_horizon_live_node_created_after_horizon_not_gc_able() {
    use grafeo_common::types::EpochId;

    let store = LpgStore::new().unwrap();
    let horizon = EpochId::new(5);

    // Commit node_b at epoch 10 (after horizon) — never deleted.
    let tx_b = TransactionId::new(10);
    let node_b = store.create_node_versioned(&["Item"], EpochId::new(10), tx_b);
    store.finalize_entities_by_id(tx_b, EpochId::new(10), &[node_b], &[]);

    // The node is live and was created AFTER the GC horizon — must NOT be GC-able.
    assert!(
        !store.node_deleted_at_or_below(node_b, horizon),
        "live node created after horizon must not be GC-able (tiered path)"
    );
}

/// `node_deleted_at_or_below` must return `true` for a node whose committed
/// `deleted_epoch` is at or below the GC horizon.
#[cfg(all(feature = "vector-index", feature = "tiered-storage"))]
#[test]
fn tiered_gc_horizon_deleted_at_or_below_horizon_is_gc_able() {
    use grafeo_common::types::EpochId;

    let store = LpgStore::new().unwrap();
    let horizon = EpochId::new(5);

    // Commit node_c at epoch 2, delete it at epoch 4 (deleted_epoch <= horizon).
    let tx_c = TransactionId::new(2);
    let node_c = store.create_node_versioned(&["Item"], EpochId::new(2), tx_c);
    store.finalize_entities_by_id(tx_c, EpochId::new(2), &[node_c], &[]);

    let tx_del = TransactionId::new(3);
    store.delete_node_transactional(node_c, EpochId::new(4), tx_del);
    store.finalize_deletes_by_id(tx_del, EpochId::new(4), &[node_c]);

    // deleted_epoch == 4 <= horizon 5 → GC-able.
    assert!(
        store.node_deleted_at_or_below(node_c, horizon),
        "node deleted at E4 with horizon E5 must be GC-able (tiered path)"
    );
}

/// `node_deleted_at_or_below` must return `false` for a node whose committed
/// `deleted_epoch` is strictly above the GC horizon.
#[cfg(all(feature = "vector-index", feature = "tiered-storage"))]
#[test]
fn tiered_gc_horizon_deleted_above_horizon_not_gc_able() {
    use grafeo_common::types::EpochId;

    let store = LpgStore::new().unwrap();
    let horizon = EpochId::new(5);

    // Commit node_d at epoch 2, delete it at epoch 6 (deleted_epoch > horizon).
    let tx_d = TransactionId::new(2);
    let node_d = store.create_node_versioned(&["Item"], EpochId::new(2), tx_d);
    store.finalize_entities_by_id(tx_d, EpochId::new(2), &[node_d], &[]);

    let tx_del = TransactionId::new(4);
    store.delete_node_transactional(node_d, EpochId::new(6), tx_del);
    store.finalize_deletes_by_id(tx_del, EpochId::new(6), &[node_d]);

    // deleted_epoch == 6 > horizon 5 → not GC-able yet.
    assert!(
        !store.node_deleted_at_or_below(node_d, horizon),
        "node deleted at E6 with horizon E5 must not be GC-able (tiered path)"
    );
}

/// `gc_vector_indexes` must keep a node committed after the GC horizon
/// searchable even after GC runs.
///
/// This is the end-to-end regression: the buggy tiered branch would call
/// `is_live(node_b) = false` and drop node_b from the HNSW, making vector
/// search return no results for a perfectly live node.
#[cfg(all(feature = "vector-index", feature = "tiered-storage"))]
#[test]
fn tiered_gc_vector_indexes_keeps_created_after_horizon_live_node_searchable()
-> grafeo_common::utils::error::Result<()> {
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
    use grafeo_common::types::EpochId;
    use std::sync::Arc;

    let store = LpgStore::new().unwrap();

    // Set up a vector index.
    let config = HnswConfig::new(3, DistanceMetric::Cosine);
    let idx = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config)));
    store.add_vector_index("Widget", "embedding", Arc::clone(&idx));

    let horizon = EpochId::new(5);

    // node_live: committed at epoch 10 (after horizon), never deleted, has a vector.
    let tx_live = TransactionId::new(10);
    let node_live = store.create_node_versioned(&["Widget"], EpochId::new(10), tx_live);
    store.finalize_entities_by_id(tx_live, EpochId::new(10), &[node_live], &[]);
    store.set_node_property(
        node_live,
        "embedding",
        Value::Vector(vec![1.0_f32, 0.0, 0.0].into()),
    );

    // Insert node_live into the HNSW manually (mirrors what the engine does on
    // commit when a vector property is set).
    {
        let accessor = super::vector_accessor::SnapshotVectorAccessor {
            store: &store,
            property: grafeo_common::types::PropertyKey::new("embedding"),
            epoch: EpochId::new(10),
            tx: None,
        };
        use crate::index::vector::VectorAccessor as _;
        if let Some(vec) = accessor.get_vector(node_live) {
            idx.insert(node_live, &vec, &accessor);
        }
    }

    // Run GC with horizon = 5 (node_live is at epoch 10, above horizon).
    store.gc_vector_indexes(horizon)?;

    // node_live must still be searchable after GC.
    let results = store.search_vector_visible(
        "Widget:embedding",
        &[1.0, 0.0, 0.0],
        10,
        EpochId::new(10),
        TransactionId::INVALID,
    );
    let ids: Vec<_> = results.iter().map(|(id, _)| *id).collect();
    assert!(
        ids.contains(&node_live),
        "live node created after GC horizon must remain searchable after gc_vector_indexes; got {ids:?}"
    );
    Ok(())
}

#[cfg(feature = "vector-index")]
#[test]
fn vector_gc_retains_deleted_backing_and_newer_incarnations()
-> grafeo_common::utils::error::Result<()> {
    use crate::graph::lpg::PhysicalIndexKey;
    use crate::index::vector::{
        DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind, VectorStoreSection,
    };
    use grafeo_common::storage::section::Section;
    use grafeo_common::types::GraphPath;
    use grafeo_common::utils::error::Error;

    let store = LpgStore::new()?;
    let index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
        HnswConfig::new(2, DistanceMetric::Euclidean),
        17,
    )));
    let eligible = NodeId::new(1);
    let retained = NodeId::new(2);
    let younger = NodeId::new(3);
    let successor = NodeId::new(4);
    for (id, lifetimes) in [
        (eligible, vec![(EpochId::new(1), Some(EpochId::new(4)))]),
        (retained, vec![(EpochId::new(1), Some(EpochId::new(8)))]),
        (younger, vec![(EpochId::new(10), None)]),
        (
            successor,
            vec![
                (EpochId::new(1), Some(EpochId::new(4))),
                (EpochId::new(10), None),
            ],
        ),
    ] {
        let labels: Vec<_> = lifetimes
            .iter()
            .map(|(created, _)| (*created, vec![arcstr::ArcStr::from("Doc")]))
            .collect();
        store
            .restore_node_history_exact(id, &lifetimes, &labels)
            .map_err(Error::InvalidValue)?;
        for (created, deleted) in lifetimes {
            store.set_node_property_at_epoch(
                id,
                "embedding",
                Value::Vector(vec![id.as_u64() as f32, 1.0].into()),
                created,
            );
            if let Some(deleted) = deleted {
                store.set_node_property_at_epoch(id, "embedding", Value::Null, deleted);
            }
        }
    }
    store.sync_epoch(EpochId::new(10));
    let accessor = |id: NodeId| Some(Arc::<[f32]>::from([id.as_u64() as f32, 1.0]));
    for id in [eligible, retained, younger, successor] {
        index.insert(id, &[id.as_u64() as f32, 1.0], &accessor);
    }
    assert!(index.remove(eligible));
    assert!(index.remove(retained));
    store.add_vector_index("Doc", "embedding", Arc::clone(&index));
    let horizon = EpochId::new(5);
    assert!(!store.node_deleted_at_or_below(successor, horizon));
    store.gc_versions(horizon);
    store.gc_vector_indexes(horizon)?;
    let ids: Vec<_> = index
        .snapshot_topology()
        .2
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert!(!ids.contains(&eligible));
    assert!(ids.contains(&retained));
    assert!(!index.contains(retained));
    assert!(index.contains(younger));
    assert!(index.contains(successor));
    assert!(
        store
            .search_vector_visible(
                "Doc:embedding",
                &[2.0, 1.0],
                10,
                horizon,
                TransactionId::INVALID
            )
            .iter()
            .any(|(id, _)| *id == retained)
    );
    let section = VectorStoreSection::new(vec![(
        PhysicalIndexKey::vector(GraphPath::root(), "Doc", "embedding"),
        index,
    )]);
    let before = section.serialize()?;
    store.gc_vector_indexes(horizon)?;
    assert_eq!(
        section.serialize()?,
        before,
        "no eligible deletion must preserve exact bytes"
    );
    Ok(())
}

#[cfg(feature = "vector-index")]
#[test]
fn vector_gc_refuses_denied_authority_and_cross_incarnation_backing()
-> grafeo_common::utils::error::Result<()> {
    use crate::graph::lpg::PhysicalIndexKey;
    use crate::graph::write_permit::{WriteAuthority, with_authority};
    use crate::index::vector::{
        DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind, VectorStoreSection,
    };
    use grafeo_common::storage::section::Section;
    use grafeo_common::types::GraphPath;
    use grafeo_common::utils::error::{Error, TransactionError};

    let store = LpgStore::new()?;
    let index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
        HnswConfig::new(2, DistanceMetric::Euclidean),
        18,
    )));
    for (id, lifetimes) in [
        (
            NodeId::new(1),
            vec![(EpochId::new(1), Some(EpochId::new(4)))],
        ),
        (
            NodeId::new(2),
            vec![
                (EpochId::new(1), Some(EpochId::new(4))),
                (EpochId::new(10), None),
            ],
        ),
    ] {
        let labels: Vec<_> = lifetimes
            .iter()
            .map(|(created, _)| (*created, vec![arcstr::ArcStr::from("Doc")]))
            .collect();
        store
            .restore_node_history_exact(id, &lifetimes, &labels)
            .map_err(Error::InvalidValue)?;
        store.set_node_property_at_epoch(
            id,
            "embedding",
            Value::Vector(vec![1.0, 0.0].into()),
            EpochId::new(1),
        );
        store.set_node_property_at_epoch(id, "embedding", Value::Null, EpochId::new(4));
        index.insert(id, &[1.0, 0.0], &|_| Some(Arc::<[f32]>::from([1.0, 0.0])));
    }
    assert!(index.remove(NodeId::new(1)));
    store.add_vector_index("Doc", "embedding", Arc::clone(&index));
    let section = VectorStoreSection::new(vec![(
        PhysicalIndexKey::vector(GraphPath::root(), "Doc", "embedding"),
        index,
    )]);
    let before = section.serialize()?;
    let owner = WriteAuthority::new();
    assert!(store.seal_unframed_writes(&owner));
    assert!(matches!(
        store.gc_vector_indexes(EpochId::new(5)),
        Err(Error::Transaction(TransactionError::ReadOnly))
    ));
    assert!(matches!(
        with_authority(&owner, || store.gc_vector_indexes(EpochId::PENDING)),
        Err(Error::InvalidValue(_))
    ));
    let missing = with_authority(&owner, || store.gc_vector_indexes(EpochId::new(5)));
    assert!(matches!(missing, Err(Error::InvalidValue(message)) if message.contains("vector")));
    assert_eq!(
        section.serialize()?,
        before,
        "failed preparation must preserve the complete index image"
    );
    Ok(())
}

#[cfg(feature = "vector-index")]
#[test]
fn vector_indexes_distinguish_colons_in_label_and_property() {
    use std::sync::Arc;

    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};

    let store = LpgStore::new().unwrap();
    let label_colon = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
        2,
        DistanceMetric::Cosine,
    ))));
    let property_colon = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
        2,
        DistanceMetric::Cosine,
    ))));
    store.add_vector_index("tenant:Doc", "embedding", Arc::clone(&label_colon));
    store.add_vector_index("tenant", "Doc:embedding", Arc::clone(&property_colon));

    let first = store.create_node(&["tenant:Doc"]);
    store.set_node_property(first, "embedding", Value::Vector(vec![1.0, 0.0].into()));
    let second = store.create_node(&["tenant"]);
    store.set_node_property(
        second,
        "Doc:embedding",
        Value::Vector(vec![0.0, 1.0].into()),
    );

    assert_eq!(label_colon.len(), 1);
    assert_eq!(property_colon.len(), 1);
    assert_eq!(
        store
            .get_vector_index("tenant:Doc", "embedding")
            .unwrap()
            .len(),
        label_colon.len()
    );
    assert_eq!(
        store
            .get_vector_index("tenant", "Doc:embedding")
            .unwrap()
            .len(),
        property_colon.len()
    );
}

#[cfg(feature = "text-index")]
#[test]
fn text_indexes_distinguish_colons_in_label_and_property() {
    use std::sync::Arc;

    use parking_lot::RwLock;

    use crate::index::text::{BM25Config, InvertedIndex};

    let store = LpgStore::new().unwrap();
    let label_colon = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    let property_colon = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    store.add_text_index("tenant:Doc", "body", Arc::clone(&label_colon));
    store.add_text_index("tenant", "Doc:body", Arc::clone(&property_colon));

    let first = store.create_node(&["tenant:Doc"]);
    store.set_node_property(first, "body", Value::from("label colon token"));
    let second = store.create_node(&["tenant"]);
    store.set_node_property(second, "Doc:body", Value::from("property colon token"));

    assert_eq!(label_colon.read().search("label", 10)[0].0, first);
    assert_eq!(property_colon.read().search("property", 10)[0].0, second);
    assert!(label_colon.read().search("property", 10).is_empty());
    assert!(property_colon.read().search("label", 10).is_empty());
}

#[cfg(feature = "vector-index")]
#[test]
fn sealed_store_rejects_retained_vector_index_alias_mutation() {
    use std::sync::Arc;

    use crate::graph::write_permit::{WriteAuthority, with_authority};
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};

    let store = LpgStore::new().unwrap();
    let index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
        2,
        DistanceMetric::Cosine,
    ))));
    store.add_vector_index("Doc", "embedding", Arc::clone(&index));

    let owner = WriteAuthority::new();
    let foreign = WriteAuthority::new();
    assert!(store.seal_unframed_writes(&owner));

    let topology = || vec![(NodeId::new(41), vec![Vec::<NodeId>::new()])];
    index.restore_topology(Some(NodeId::new(41)), 0, topology());
    assert_eq!(
        store.get_vector_index("Doc", "embedding").unwrap().len(),
        0,
        "a retained pre-seal Arc must not mutate the live index"
    );

    with_authority(&foreign, || {
        index.restore_topology(Some(NodeId::new(41)), 0, topology());
    });
    assert_eq!(
        store.get_vector_index("Doc", "embedding").unwrap().len(),
        0,
        "foreign authority must not authorize a retained index alias"
    );

    with_authority(&owner, || {
        index.restore_topology(Some(NodeId::new(41)), 0, topology());
    });
    assert_eq!(store.get_vector_index("Doc", "embedding").unwrap().len(), 1);

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_authority(&owner, || panic!("injected panic"));
    }));
    assert!(!index.remove(NodeId::new(41)));
    assert_eq!(
        store.get_vector_index("Doc", "embedding").unwrap().len(),
        1,
        "caught panic must not leave the retained alias authorized"
    );
}

#[cfg(feature = "vector-index")]
#[test]
fn vector_index_binding_is_exact_and_failed_recursive_seal_is_atomic() {
    use std::sync::Arc;

    use crate::graph::write_permit::WriteAuthority;
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};

    let shared = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
        2,
        DistanceMetric::Cosine,
    ))));
    let first = LpgStore::new().unwrap();
    first.add_vector_index("Doc", "embedding", Arc::clone(&shared));
    assert!(first.get_vector_index("Doc", "embedding").is_some());

    let second = LpgStore::new().unwrap();
    second.add_vector_index("Doc", "embedding", Arc::clone(&shared));
    assert!(
        second.get_vector_index("Doc", "embedding").is_none(),
        "one retained Arc cannot bind two unsealed stores"
    );
    first.add_vector_index("Other", "embedding", Arc::clone(&shared));
    assert!(
        first.get_vector_index("Other", "embedding").is_none(),
        "one retained Arc cannot combine two encoded index slots"
    );

    let transition = VectorIndexKind::pin_scope_transition();
    assert!(!shared.scope_is_compatible(0, &transition));
    assert!(!shared.seal_with_scope_under_transition(0, &transition));
    drop(transition);

    let foreign_index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
        2,
        DistanceMetric::Cosine,
    ))));
    let foreign_store = LpgStore::new().unwrap();
    foreign_store.add_vector_index("Foreign", "embedding", Arc::clone(&foreign_index));
    let foreign_owner = WriteAuthority::new();
    assert!(foreign_store.seal_unframed_writes(&foreign_owner));
    let unsealed_target = LpgStore::new().unwrap();
    unsealed_target.add_vector_index("Foreign", "embedding", Arc::clone(&foreign_index));
    assert!(
        unsealed_target
            .get_vector_index("Foreign", "embedding")
            .is_none()
    );

    // Simulate a hostile internally-corrupted derived scope to exercise the
    // multi-index preflight itself. No good sibling may be captured before the
    // later incompatible sibling rejects the recursive seal.
    let root = LpgStore::new().unwrap();
    let good = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
        2,
        DistanceMetric::Cosine,
    ))));
    let bad = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
        2,
        DistanceMetric::Cosine,
    ))));
    root.add_vector_index("Good", "embedding", Arc::clone(&good));
    root.add_vector_index("Bad", "embedding", Arc::clone(&bad));
    let transition = VectorIndexKind::pin_scope_transition();
    assert!(bad.seal_with_scope_under_transition(foreign_owner.scope().get(), &transition));
    drop(transition);

    let root_owner = WriteAuthority::new();
    assert!(!root.seal_unframed_writes(&root_owner));
    let transition = VectorIndexKind::pin_scope_transition();
    assert!(good.scope_is_unsealed(&transition));
    drop(transition);
    assert!(root.remove_vector_index("Bad", "embedding"));
    assert!(root.seal_unframed_writes(&root_owner));
}

#[cfg(feature = "vector-index")]
#[test]
fn sealed_quantized_alias_rejects_every_auxiliary_mutation_path() {
    use std::sync::Arc;

    use crate::graph::write_permit::{WriteAuthority, with_authority};
    use crate::index::vector::{
        DistanceMetric, HnswConfig, QuantizationType, QuantizedHnswIndex, VectorIndexKind,
    };

    let store = LpgStore::new().unwrap();
    let index = Arc::new(VectorIndexKind::Quantized(
        QuantizedHnswIndex::with_seed(
            HnswConfig::new(2, DistanceMetric::Euclidean),
            QuantizationType::Scalar,
            7,
        )
        .with_training_threshold(10),
    ));
    store.add_vector_index("Doc", "embedding", Arc::clone(&index));
    let owner = WriteAuthority::new();
    let foreign = WriteAuthority::new();
    assert!(store.seal_unframed_writes(&owner));

    let quantized = index.as_quantized().unwrap();
    with_authority(&owner, || {
        quantized.insert(NodeId::new(1), &[1.0, 0.0]);
    });
    let topology = quantized.snapshot_topology();
    let heap = quantized.heap_memory_bytes();
    let state = quantized.state_fingerprint();

    quantized.insert(NodeId::new(2), &[0.0, 1.0]);
    let batch_vector = [0.5, 0.5];
    quantized.batch_insert([(NodeId::new(3), batch_vector.as_slice())]);
    assert!(!quantized.remove(NodeId::new(1)));
    assert!(quantized.gc(&|_| false).is_err());
    quantized.restore_topology(
        Some(NodeId::new(99)),
        0,
        vec![(NodeId::new(99), vec![Vec::new()])],
    );
    with_authority(&foreign, || {
        quantized.insert(NodeId::new(4), &[0.25, 0.75]);
        assert!(quantized.gc(&|_| false).is_err());
    });
    assert_eq!(quantized.snapshot_topology(), topology);
    assert_eq!(quantized.heap_memory_bytes(), heap);
    assert_eq!(quantized.state_fingerprint(), state);
    assert!(quantized.contains(NodeId::new(1)));
    assert!(!quantized.contains(NodeId::new(2)));
    assert!(!quantized.contains(NodeId::new(3)));
    assert!(!quantized.contains(NodeId::new(4)));
    assert_eq!(quantized.search(&[1.0, 0.0], 1)[0].0, NodeId::new(1));

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_authority(&owner, || panic!("injected owner panic"));
    }));
    quantized.insert(NodeId::new(5), &[0.0, 1.0]);
    assert!(!quantized.contains(NodeId::new(5)));
    assert_eq!(quantized.state_fingerprint(), state);

    with_authority(&owner, || {
        quantized.batch_insert([(NodeId::new(6), batch_vector.as_slice())]);
    });
    assert!(quantized.contains(NodeId::new(6)));
}

#[cfg(feature = "text-index")]
#[test]
fn sealed_named_store_rejects_retained_text_index_alias_mutation() {
    use std::sync::Arc;

    use parking_lot::RwLock;

    use crate::graph::write_permit::{WriteAuthority, with_authority};
    use crate::index::text::{BM25Config, InvertedIndex};

    let root = LpgStore::new().unwrap();
    assert!(root.create_graph("notes").unwrap());
    let named = root.graph("notes").unwrap();
    let index = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    named.add_text_index("Note", "body", Arc::clone(&index));

    let owner = WriteAuthority::new();
    let foreign = WriteAuthority::new();
    assert!(root.seal_unframed_writes(&owner));

    index.write().insert(NodeId::new(7), "unauthorized token");
    assert!(
        named
            .get_text_index("Note", "body")
            .unwrap()
            .read()
            .search("unauthorized", 10)
            .is_empty(),
        "a retained pre-seal text-index Arc must not mutate a named store"
    );

    with_authority(&foreign, || {
        index.write().insert(NodeId::new(7), "foreign token");
    });
    assert!(
        named
            .get_text_index("Note", "body")
            .unwrap()
            .read()
            .search("foreign", 10)
            .is_empty(),
        "foreign authority must not authorize a retained text-index alias"
    );

    with_authority(&owner, || {
        index.write().insert(NodeId::new(7), "owned token");
    });
    assert_eq!(
        named
            .get_text_index("Note", "body")
            .unwrap()
            .read()
            .search("owned", 10)[0]
            .0,
        NodeId::new(7)
    );

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_authority(&owner, || panic!("injected panic"));
    }));
    assert!(!index.write().remove(NodeId::new(7)));
    assert!(
        !named
            .get_text_index("Note", "body")
            .unwrap()
            .read()
            .search("owned", 10)
            .is_empty(),
        "caught panic must not leave the retained text-index alias authorized"
    );
}

#[cfg(feature = "text-index")]
#[test]
fn recursive_store_seal_waits_for_retained_text_index_mutation() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    use parking_lot::RwLock;

    use crate::graph::write_permit::WriteAuthority;
    use crate::index::text::{BM25Config, InvertedIndex, Tokenizer};

    struct BlockingTokenizer {
        barrier: Arc<Barrier>,
        block_once: AtomicBool,
    }

    impl Tokenizer for BlockingTokenizer {
        fn tokenize(&self, text: &str) -> Vec<String> {
            if self.block_once.swap(false, Ordering::SeqCst) {
                self.barrier.wait();
                self.barrier.wait();
            }
            vec![text.to_owned()]
        }
    }

    let barrier = Arc::new(Barrier::new(2));
    let store = Arc::new(LpgStore::new().unwrap());
    let index = Arc::new(RwLock::new(InvertedIndex::with_tokenizer(
        BM25Config::default(),
        Box::new(BlockingTokenizer {
            barrier: Arc::clone(&barrier),
            block_once: AtomicBool::new(true),
        }),
    )));
    store.add_text_index("Note", "body", Arc::clone(&index));

    let mutation_index = Arc::clone(&index);
    let mutator = thread::spawn(move || {
        mutation_index.write().insert(NodeId::new(1), "before-seal");
    });
    barrier.wait();

    let (sealed_tx, sealed_rx) = mpsc::channel();
    let seal_store = Arc::clone(&store);
    let sealer = thread::spawn(move || {
        let owner = WriteAuthority::new();
        sealed_tx
            .send(seal_store.seal_unframed_writes(&owner))
            .unwrap();
        owner
    });
    assert!(
        sealed_rx.recv_timeout(Duration::from_millis(50)).is_err(),
        "recursive seal must wait for a retained alias's complete text mutation"
    );
    barrier.wait();
    mutator.join().unwrap();
    assert!(sealed_rx.recv().unwrap());
    let owner = sealer.join().unwrap();
    assert_eq!(index.read().search("before-seal", 10)[0].0, NodeId::new(1));

    index
        .write()
        .insert(NodeId::new(2), "after-seal-unauthorized");
    assert!(
        index
            .read()
            .search("after-seal-unauthorized", 10)
            .is_empty()
    );
    crate::graph::write_permit::with_authority(&owner, || {
        index.write().insert(NodeId::new(2), "after-seal-owned");
    });
    assert_eq!(
        index.read().search("after-seal-owned", 10)[0].0,
        NodeId::new(2)
    );
}

#[cfg(feature = "text-index")]
#[test]
fn text_index_binding_is_exact_and_failed_recursive_seal_is_atomic() {
    use std::sync::Arc;

    use parking_lot::RwLock;

    use crate::graph::write_permit::WriteAuthority;
    use crate::index::text::{BM25Config, InvertedIndex};

    let shared = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    let first = LpgStore::new().unwrap();
    first.add_text_index("Doc", "body", Arc::clone(&shared));
    assert!(first.get_text_index("Doc", "body").is_some());

    let second = LpgStore::new().unwrap();
    second.add_text_index("Doc", "body", Arc::clone(&shared));
    assert!(second.get_text_index("Doc", "body").is_none());
    first.add_text_index("Other", "body", Arc::clone(&shared));
    assert!(first.get_text_index("Other", "body").is_none());

    let transition = InvertedIndex::pin_scope_transition();
    assert!(!shared.read().scope_is_compatible(0, &transition));
    assert!(
        !shared
            .read()
            .seal_with_scope_under_transition(0, &transition)
    );
    drop(transition);

    let foreign_index = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    let foreign_store = LpgStore::new().unwrap();
    foreign_store.add_text_index("Foreign", "body", Arc::clone(&foreign_index));
    let foreign_owner = WriteAuthority::new();
    assert!(foreign_store.seal_unframed_writes(&foreign_owner));
    let unsealed_target = LpgStore::new().unwrap();
    unsealed_target.add_text_index("Foreign", "body", Arc::clone(&foreign_index));
    assert!(unsealed_target.get_text_index("Foreign", "body").is_none());

    let root = LpgStore::new().unwrap();
    let good = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    let bad = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    root.add_text_index("Good", "body", Arc::clone(&good));
    root.add_text_index("Bad", "body", Arc::clone(&bad));
    let transition = InvertedIndex::pin_scope_transition();
    assert!(
        bad.read()
            .seal_with_scope_under_transition(foreign_owner.scope().get(), &transition)
    );
    drop(transition);

    let root_owner = WriteAuthority::new();
    assert!(!root.seal_unframed_writes(&root_owner));
    let transition = InvertedIndex::pin_scope_transition();
    assert!(good.read().scope_is_unsealed(&transition));
    drop(transition);
    assert!(root.remove_text_index("Bad", "body"));
    assert!(root.seal_unframed_writes(&root_owner));
}

#[test]
fn generated_identity_and_epoch_exhaustion_fail_closed_without_wrap() {
    use std::sync::atomic::Ordering;

    let store = LpgStore::new().unwrap();
    store.set_next_node_id(u64::MAX - 1);
    let last_node = store.create_node_with_props(&["Last"], [("proof", Value::from(1i64))]);
    assert_eq!(last_node, NodeId::new(u64::MAX - 1));
    assert_eq!(store.next_node_id(), u64::MAX);
    assert_eq!(
        store.create_node_with_props(&["Never"], [("proof", Value::from(2i64))]),
        NodeId::INVALID
    );
    assert!(store.get_node(NodeId::INVALID).is_none());

    // Use exact, low endpoint IDs so node exhaustion does not affect edge
    // allocation coverage.
    let edges = LpgStore::new().unwrap();
    edges
        .create_node_with_id(NodeId::new(1), &["Source"])
        .unwrap();
    edges
        .create_node_with_id(NodeId::new(2), &["Destination"])
        .unwrap();
    edges.set_next_edge_id(u64::MAX - 1);
    assert_eq!(
        edges.create_edge(NodeId::new(1), NodeId::new(2), "LAST"),
        EdgeId::new(u64::MAX - 1)
    );
    assert_eq!(edges.next_edge_id(), u64::MAX);
    assert_eq!(
        edges.create_edge(NodeId::new(1), NodeId::new(2), "NEVER"),
        EdgeId::INVALID
    );
    assert!(
        edges
            .batch_create_edges(&[(NodeId::new(1), NodeId::new(2), "NEVER")])
            .is_empty()
    );
    assert!(!edges.all_known_edge_ids().contains(&EdgeId::INVALID));

    edges
        .current_epoch
        .store(EpochId::PENDING.as_u64() - 1, Ordering::Release);
    let exhausted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        edges.new_epoch();
    }));
    assert!(exhausted.is_err());
    assert_eq!(edges.current_epoch(), EpochId::new(u64::MAX - 1));
    edges.sync_epoch(EpochId::PENDING);
    assert_eq!(edges.current_epoch(), EpochId::new(u64::MAX - 1));
}

#[test]
fn first_seal_and_clear_wait_for_whole_ordinary_edge_publication() {
    use crate::graph::write_permit::WriteAuthority;
    use std::sync::Barrier;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    for clear in [false, true] {
        let store = Arc::new(LpgStore::new().unwrap());
        let src = store.create_node(&["Source"]);
        let dst = store.create_node(&["Destination"]);
        let barrier = Arc::new(Barrier::new(2));
        *store.edge_publication_barrier.write() = Some(Arc::clone(&barrier));
        let creator_store = Arc::clone(&store);
        let creator = thread::spawn(move || creator_store.create_edge(src, dst, "PINNED"));
        barrier.wait();

        let (finished_tx, finished_rx) = mpsc::channel();
        let transition_store = Arc::clone(&store);
        let transition = thread::spawn(move || {
            if clear {
                transition_store.clear();
                finished_tx.send(true).unwrap();
            } else {
                let owner = WriteAuthority::new();
                finished_tx
                    .send(transition_store.seal_unframed_writes(&owner))
                    .unwrap();
            }
        });
        assert!(
            finished_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "exclusive transition must wait for the whole ordinary mutation"
        );
        barrier.wait();
        assert!(creator.join().unwrap().is_valid());
        assert!(finished_rx.recv().unwrap());
        transition.join().unwrap();
        if clear {
            assert_eq!(store.edge_count(), 0);
            assert_eq!(store.node_count(), 0);
        } else {
            assert_eq!(store.edge_count(), 1);
        }
    }
}

#[test]
fn named_graph_topology_rejects_cycles_aliases_and_failure_atomic_seal() {
    use crate::graph::write_permit::WriteAuthority;
    use std::sync::atomic::Ordering;

    let root = Arc::new(LpgStore::new().unwrap());
    let child = Arc::new(root.new_named_graph_candidate().unwrap());
    assert!(root.install_graph_if_absent("child", Arc::clone(&child)));
    assert!(!root.install_graph_if_absent("alias", Arc::clone(&child)));
    assert!(!child.install_graph_if_absent("cycle", Arc::clone(&root)));

    let foreign_owner = WriteAuthority::new();
    assert!(child.seal_unframed_writes(&foreign_owner));
    let root_owner = WriteAuthority::new();
    assert!(!root.seal_unframed_writes(&root_owner));
    assert_eq!(root.mutation_scope.load(Ordering::Acquire), 0);
    assert_eq!(
        child.mutation_scope.load(Ordering::Acquire),
        foreign_owner.scope().get()
    );
}

#[test]
fn transport_prepare_can_read_overlay_and_invalid_input_never_prepares() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let store = LpgStore::new().unwrap();
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    let closed = store
        .create_transport_edge_with_id(EdgeId::new(90_001), src, dst, "CARRIED")
        .unwrap()
        .unwrap();
    commit_transport_edge_delete(
        &store,
        closed.edge_id(),
        TransactionId::new(90_002),
        EpochId::new(1),
    );
    let prepared = AtomicBool::new(false);
    let result = store
        .purge_transport_extract_edges_after_prepare_and_publish_with_rollback(
            &[&closed],
            &[],
            |_| {
                assert_eq!(store.get_edge_history(closed.edge_id()).len(), 1);
                assert_eq!(
                    store.edges_from(src, Direction::Outgoing).count(),
                    0,
                    "the exact adjacency is tombstoned but remains physically readable"
                );
                prepared.store(true, Ordering::SeqCst);
                Ok::<(), ()>(())
            },
            |()| (),
            |()| {},
        )
        .unwrap();
    assert!(matches!(result, PreparedPurgeOutcome::Published(())));
    assert!(prepared.load(Ordering::SeqCst));

    let open = store
        .create_transport_edge_with_id(EdgeId::new(90_003), src, dst, "OPEN")
        .unwrap()
        .unwrap();
    prepared.store(false, Ordering::SeqCst);
    let rejected = store
        .purge_transport_extract_edges_after_prepare_and_publish_with_rollback(
            &[&open],
            &[],
            |_| {
                prepared.store(true, Ordering::SeqCst);
                Ok::<(), ()>(())
            },
            |()| (),
            |()| {},
        )
        .unwrap();
    assert!(matches!(rejected, PreparedPurgeOutcome::Rejected));
    assert!(!prepared.load(Ordering::SeqCst));
}

#[test]
fn transport_success_token_retires_only_after_every_lpg_guard_is_released() {
    use std::sync::atomic::{AtomicBool, Ordering};

    struct DropProbe {
        store: Arc<LpgStore>,
        dropped: Arc<AtomicBool>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            assert!(self.store.mutation_scope_gate.try_write().is_some());
            assert!(self.store.transport_extract_authority.try_write().is_some());
            assert!(
                self.store
                    .edge_identity_reservations
                    .identities
                    .try_lock()
                    .is_some()
            );
            #[cfg(not(feature = "tiered-storage"))]
            assert!(self.store.edges.try_write().is_some());
            #[cfg(feature = "tiered-storage")]
            assert!(self.store.edge_versions.try_write().is_some());
            assert!(self.store.pending_tx_creates.try_write().is_some());
            assert!(self.store.pending_tx_edge_deletes.try_write().is_some());
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    let store = Arc::new(LpgStore::new().unwrap());
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    let receipt = store
        .create_transport_edge_with_id(EdgeId::new(90_101), src, dst, "CARRIED")
        .unwrap()
        .unwrap();
    commit_transport_edge_delete(
        &store,
        receipt.edge_id(),
        TransactionId::new(90_102),
        EpochId::new(1),
    );
    let dropped = Arc::new(AtomicBool::new(false));
    let token = store
        .purge_transport_extract_edges_after_prepare_and_publish_with_rollback(
            &[&receipt],
            &[],
            |_| Ok::<(), ()>(()),
            |()| DropProbe {
                store: Arc::clone(&store),
                dropped: Arc::clone(&dropped),
            },
            drop,
        )
        .unwrap();
    let PreparedPurgeOutcome::Published(token) = token else {
        panic!("qualified purge publishes one token");
    };
    assert!(!dropped.load(Ordering::SeqCst));
    drop(token);
    assert!(dropped.load(Ordering::SeqCst));
}

#[test]
fn transport_multiple_lifetimes_preserve_grant_and_allow_final_delete_purge() {
    use crate::graph::write_permit::{WriteAuthority, with_authority};
    let store = LpgStore::new().unwrap();
    let source = store.create_node(&["Source"]);
    let missing = NodeId::new(99);
    let edge = EdgeId::new(71);
    let receipt = store
        .restore_transport_edge_history_exact(
            edge,
            source,
            missing,
            "CARRIED",
            &[
                (EpochId::new(1), Some(EpochId::new(2))),
                (EpochId::new(3), None),
            ],
        )
        .unwrap();
    store.sync_epoch(EpochId::new(3));
    assert!(
        store
            .transport_edges_may_be_unresolved
            .load(Ordering::Acquire)
    );
    assert_eq!(store.get_edge_history(edge).len(), 2);
    assert!(store.is_transport_extract_edge_unchanged_since(&receipt, EpochId::new(3)));
    let authority = WriteAuthority::new();
    with_authority(&authority, || {
        let grant = store
            .grant_transport_edge_mutation(&receipt, &authority)
            .unwrap();
        assert!(store.accepts_transport_edge_missing_destination(&grant, missing, &authority));
        commit_transport_edge_delete(&store, edge, TransactionId::new(710), EpochId::new(4));
        assert!(!store.accepts_transport_edge_missing_destination(&grant, missing, &authority));
    });
    assert_eq!(
        store.classify_transport_extract_edges(&[&receipt], EpochId::new(3), EpochId::new(4)),
        [TransportEdgeState::Closed]
    );
    assert!(store.is_transport_extract_closed_edge_unchanged_since(&receipt, EpochId::new(4)));
    assert!(store.purge_transport_extract_edges(&[&receipt]));
    assert!(store.get_edge_history(edge).is_empty());
    // Even identical structural history cannot recover a consumed nonce.
    let replacement = store
        .restore_transport_edge_history_exact(
            edge,
            source,
            missing,
            "CARRIED",
            &[
                (EpochId::new(1), Some(EpochId::new(2))),
                (EpochId::new(3), Some(EpochId::new(4))),
            ],
        )
        .unwrap();
    assert!(!store.purge_transport_extract_edges(&[&receipt]));
    assert!(store.purge_transport_extract_edges(&[&replacement]));
}

#[test]
fn transport_multiple_lifetimes_reject_altered_history_and_recreation() {
    let store = LpgStore::new().unwrap();
    let source = store.create_node(&["Source"]);
    let destination = store.create_node(&["Destination"]);
    let edge = EdgeId::new(72);
    let receipt = store
        .restore_transport_edge_history_exact(
            edge,
            source,
            destination,
            "CARRIED",
            &[
                (EpochId::new(1), Some(EpochId::new(2))),
                (EpochId::new(3), None),
            ],
        )
        .unwrap();
    for history in [
        vec![
            (EpochId::new(1), Some(EpochId::new(3))),
            (EpochId::new(3), None),
        ],
        vec![
            (EpochId::new(1), Some(EpochId::new(2))),
            (EpochId::new(3), Some(EpochId::new(4))),
            (EpochId::new(5), None),
        ],
    ] {
        assert_eq!(
            receipt.history_state(
                history.iter().rev().copied(),
                EpochId::new(6),
                EpochId::new(6)
            ),
            TransportEdgeState::Invalid
        );
    }
    store
        .restore_edge_history_exact(
            edge,
            source,
            destination,
            "CARRIED",
            &[
                (EpochId::new(1), Some(EpochId::new(2))),
                (EpochId::new(3), Some(EpochId::new(4))),
                (EpochId::new(5), None),
            ],
        )
        .expect_err("exact restore cannot replace a receipt-owned identity");
    store.sync_epoch(EpochId::new(6));
    assert!(store.is_transport_extract_edge(&receipt));
    assert!(!store.purge_transport_extract_edges(&[&receipt]));
}
