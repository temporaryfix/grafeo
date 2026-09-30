use super::LayeredStore;
use crate::graph::compact::{CompactStore, CompactStoreBuilder};
use crate::graph::lpg::LpgStore;
use crate::graph::traits::{GraphStore, GraphStoreMut};
use crate::graph::write_permit::{WriteAuthority, with_authority};
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, Value};
use grafeo_common::utils::error::{Error, TransactionError};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn empty_base() -> Arc<CompactStore> {
    Arc::new(CompactStoreBuilder::new().build().unwrap())
}

fn assert_native(source: &LpgStore) {
    let transition = source
        .pin_exclusive_unframed_transition()
        .expect("native representation remains active");
    assert!(transition.require_native_compact_source().is_ok());
}

#[test]
fn native_temporal_preparation_failure_preserves_unbound_source_and_allows_retry() {
    let source = Arc::new(LpgStore::new().unwrap());
    source.sync_epoch(EpochId::new(1));
    let node = source.create_node(&["Document"]);
    source.set_node_property(node, "title", Value::from("before"));
    source.create_property_index("title");
    let before = source.node_property_history(node);
    let node_floor = source.next_node_id();
    let edge_floor = source.next_edge_id();
    let prepares = AtomicUsize::new(0);
    let failed = LayeredStore::from_native_temporal_with_prepare(Arc::clone(&source), |base| {
        prepares.fetch_add(1, Ordering::SeqCst);
        assert_eq!(
            base.get_node_property(node, &PropertyKey::new("title")),
            Some(Value::from("before"))
        );
        Err("reject completed native temporal candidate".to_owned())
    });
    assert!(
        matches!(failed, Err(Error::Internal(reason)) if reason == "reject completed native temporal candidate")
    );
    assert_eq!(prepares.load(Ordering::SeqCst), 1);
    assert_native(&source);
    assert_eq!(source.next_node_id(), node_floor);
    assert_eq!(source.next_edge_id(), edge_floor);
    assert_eq!(source.node_property_history(node), before);
    assert!(source.has_property_index("title"));
    assert_eq!(
        source.find_nodes_by_property("title", &Value::from("before")),
        vec![node]
    );

    source.sync_epoch(EpochId::new(2));
    source.set_node_property(node, "title", Value::from("after"));
    let expected = source.node_property_history(node);
    let layered = LayeredStore::from_native_temporal(Arc::clone(&source)).unwrap();
    assert_eq!(layered.node_property_full_history(node), expected);
    assert_eq!(
        layered.find_nodes_by_property("title", &Value::from("after")),
        vec![node]
    );
    assert!(source.pin_exclusive_unframed_transition().is_none());
    let successor = layered.overlay_store();
    let transition = successor.pin_exclusive_unframed_transition().unwrap();
    assert!(transition.require_native_compact_source().is_err());
}

#[test]
fn native_temporal_preparation_unwind_preserves_unbound_source_and_allows_retry() {
    let source = Arc::new(LpgStore::new().unwrap());
    let node = source.create_node(&["Document"]);
    source.set_node_property(node, "title", Value::from("retained"));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        LayeredStore::from_native_temporal_with_prepare(Arc::clone(&source), |_| {
            panic!("native temporal preparation unwind")
        })
    }));
    assert!(result.is_err());
    assert_native(&source);
    assert!(source.create_node(&["AfterUnwind"]).is_valid());
    let layered = LayeredStore::from_native_temporal(source).unwrap();
    assert_eq!(layered.node_count(), 2);
    assert_eq!(
        layered.get_node_property(node, &PropertyKey::new("title")),
        Some(Value::from("retained"))
    );
}

#[test]
fn temporal_property_seeks_return_current_original_node_ids() {
    let source = Arc::new(LpgStore::new().unwrap());
    source.sync_epoch(EpochId::new(1));
    let a = source.create_node(&["Document"]);
    let b = source.create_node(&["Document"]);
    source.set_node_property(a, "value", Value::Int64(1));
    source.set_node_property(b, "value", Value::Int64(2));
    source.sync_epoch(EpochId::new(2));
    source.set_node_property(a, "value", Value::Int64(20));
    source.set_node_property(b, "value", Value::Int64(5));
    let layered = LayeredStore::from_native_temporal(source).unwrap();
    assert_eq!(
        layered.find_nodes_by_property("value", &Value::Int64(20)),
        vec![a]
    );
    assert_eq!(
        layered.find_nodes_by_property("value", &Value::Int64(5)),
        vec![b]
    );
    assert!(
        layered
            .find_nodes_by_property("value", &Value::Int64(2))
            .is_empty()
    );
    assert_eq!(
        layered.find_nodes_in_range(
            "value",
            Some(&Value::Int64(6)),
            Some(&Value::Int64(30)),
            true,
            true
        ),
        vec![a]
    );
}

#[test]
fn temporal_property_seeks_exclude_closed_identity_rows() {
    let source = Arc::new(LpgStore::new().unwrap());
    source.sync_epoch(EpochId::new(1));
    let removed = source.create_node(&["Document"]);
    let retained = source.create_node(&["Document"]);
    source.set_node_property(removed, "score", Value::Int64(2));
    source.set_node_property(retained, "score", Value::Int64(2));
    source.sync_epoch(EpochId::new(2));
    assert_eq!(
        source.remove_node_property(removed, "score"),
        Some(Value::Int64(2))
    );
    let layered = LayeredStore::from_native_temporal(source).unwrap();
    assert!(layered.get_node(removed).is_some());
    assert_eq!(
        layered.get_node_property(removed, &PropertyKey::new("score")),
        None
    );
    assert_eq!(
        layered.find_nodes_by_property("score", &Value::Int64(2)),
        vec![retained]
    );
    assert_eq!(
        layered.find_nodes_in_range(
            "score",
            Some(&Value::Int64(2)),
            Some(&Value::Int64(2)),
            true,
            true
        ),
        vec![retained]
    );
}

#[test]
fn compact_adoption_rejects_second_generation_in_every_profile() {
    let source = Arc::new(LpgStore::new().unwrap());
    let node = source.create_node(&["Document"]);
    let base = empty_base();
    let layered = LayeredStore::with_overlay(Arc::clone(&base), Arc::clone(&source)).unwrap();
    let floors = (source.next_node_id(), source.next_edge_id());
    for proposed in [base, empty_base()] {
        assert!(matches!(
            LayeredStore::with_overlay(proposed, Arc::clone(&source)),
            Err(Error::Transaction(TransactionError::InvalidState(_)))
        ));
    }
    assert!(matches!(
        LayeredStore::from_native_temporal(Arc::clone(&source)),
        Err(Error::Transaction(TransactionError::InvalidState(_)))
    ));
    assert_eq!((source.next_node_id(), source.next_edge_id()), floors);
    assert!(layered.get_node(node).is_some());

    layered.reset_overlay();
    assert!(matches!(
        LayeredStore::with_overlay(empty_base(), layered.overlay_store()),
        Err(Error::Transaction(TransactionError::InvalidState(_)))
    ));
}

#[cfg(not(feature = "vector-index"))]
#[test]
fn non_vector_adoption_does_not_retain_displaced_base_memory() {
    let source = Arc::new(LpgStore::new().unwrap());
    let base = empty_base();
    let lifetime = Arc::downgrade(&base);
    let layered = LayeredStore::with_overlay(base, Arc::clone(&source)).unwrap();
    drop(layered.swap_base(empty_base()));
    assert!(lifetime.upgrade().is_none());
    assert!(LayeredStore::with_overlay(empty_base(), source).is_err());
}

#[test]
fn compact_adoption_requires_exact_authority_without_binding_on_rejection() {
    let source = Arc::new(LpgStore::new().unwrap());
    let owner = WriteAuthority::new();
    let foreign = WriteAuthority::new();
    assert!(source.seal_unframed_writes(&owner));
    let floors = (source.next_node_id(), source.next_edge_id());
    for denied in [
        LayeredStore::with_overlay(empty_base(), Arc::clone(&source)),
        with_authority(&foreign, || {
            LayeredStore::with_overlay(empty_base(), Arc::clone(&source))
        }),
        LayeredStore::from_native_temporal(Arc::clone(&source)),
        with_authority(&foreign, || {
            LayeredStore::from_native_temporal(Arc::clone(&source))
        }),
    ] {
        assert!(matches!(
            denied,
            Err(Error::Transaction(TransactionError::InvalidState(_)))
        ));
    }
    assert_eq!((source.next_node_id(), source.next_edge_id()), floors);
    with_authority(&owner, || assert_native(&source));
    let layered = with_authority(&owner, || {
        LayeredStore::with_overlay(empty_base(), Arc::clone(&source))
    })
    .unwrap();
    assert!(Arc::ptr_eq(&source, &layered.overlay_store()));
}

#[test]
fn compact_adoption_rejects_retired_native_source_without_affecting_successor() {
    let source = Arc::new(LpgStore::new().unwrap());
    let node = source.create_node(&["Document"]);
    let layered = LayeredStore::from_native_temporal(Arc::clone(&source)).unwrap();
    let successor = layered.overlay_store();
    assert!(matches!(
        LayeredStore::with_overlay(empty_base(), Arc::clone(&source)),
        Err(Error::Transaction(TransactionError::InvalidState(_)))
    ));
    assert!(matches!(
        LayeredStore::from_native_temporal(source),
        Err(Error::Transaction(TransactionError::InvalidState(_)))
    ));
    assert!(Arc::ptr_eq(&successor, &layered.overlay_store()));
    assert!(layered.get_node(node).is_some());
    assert!(layered.create_node(&["AfterRejection"]).is_valid());
}

#[test]
fn compact_constructor_rejects_invalid_maxima_and_preserves_exhausted_floors() {
    for (node_max, edge_max) in [(u64::MAX, 0), (0, u64::MAX)] {
        assert!(matches!(
            LayeredStore::new(
                CompactStoreBuilder::new().build().unwrap(),
                node_max,
                edge_max
            ),
            Err(Error::InvalidValue(_))
        ));
    }
    let layered = LayeredStore::new(
        CompactStoreBuilder::new().build().unwrap(),
        u64::MAX - 1,
        u64::MAX - 1,
    )
    .unwrap();
    assert_eq!(layered.overlay_store().next_node_id(), u64::MAX);
    assert_eq!(layered.overlay_store().next_edge_id(), u64::MAX);
}

#[test]
fn compact_adoption_preserves_highwaters_and_raises_closed_identity_floors() {
    let source = Arc::new(LpgStore::new().unwrap());
    for id in [NodeId::new(7), NodeId::new(90)] {
        source
            .restore_node_history_exact(
                id,
                &[(EpochId::new(1), None)],
                &[(EpochId::new(1), vec!["Document".into()])],
            )
            .unwrap();
    }
    source
        .restore_edge_history_exact(
            EdgeId::new(900),
            NodeId::new(7),
            NodeId::new(90),
            "CLOSED",
            &[(EpochId::new(1), Some(EpochId::new(2)))],
        )
        .unwrap();
    source.sync_epoch(EpochId::new(2));
    let base = LayeredStore::from_native_temporal(source)
        .unwrap()
        .base_store_arc();
    let overlay = Arc::new(LpgStore::new().unwrap());
    overlay.set_next_node_id(1_234);
    overlay.set_next_edge_id(5);
    let owner = WriteAuthority::new();
    assert!(overlay.seal_unframed_writes(&owner));
    let layered = with_authority(&owner, || LayeredStore::with_overlay(base, overlay)).unwrap();
    let overlay = layered.overlay_store();
    assert_eq!(overlay.next_node_id(), 1_234);
    assert_eq!(overlay.next_edge_id(), 901);
    assert_eq!(layered.node_count(), 2);
    assert_eq!(layered.edge_count(), 0);
}
