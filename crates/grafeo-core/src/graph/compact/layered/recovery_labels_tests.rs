use super::LayeredStore;
use crate::graph::lpg::LpgStore;
use crate::graph::traits::{GraphStore, GraphStoreMut};
use crate::graph::write_permit::{WriteAuthority, with_authority};
use arcstr::ArcStr;
use grafeo_common::types::{EpochId, NodeId, Value};
use std::cell::Cell;
use std::sync::Arc;

fn labels(names: &[&str]) -> Vec<ArcStr> {
    names.iter().map(|name| ArcStr::from(*name)).collect()
}

#[test]
fn ambiguous_label_boundary_rejects_native_conversion_without_retiring_source() {
    let source = Arc::new(LpgStore::new().unwrap());
    let id = NodeId::new(7);
    let epoch = EpochId::new(5);
    let lifetimes = [(epoch, Some(epoch)), (epoch, None)];
    let image = labels(&["Shared"]);
    let history = vec![(epoch, image.clone())];
    source
        .restore_node_history_exact(id, &lifetimes, &history)
        .unwrap();
    source.set_node_property(id, "retained", Value::Int64(7));
    let properties = source.node_property_history(id);
    let floors = (source.next_node_id(), source.next_edge_id());
    let prepared = Cell::new(false);
    let result = LayeredStore::from_native_temporal_with_prepare(Arc::clone(&source), |base| {
        prepared.set(true);
        Ok(base)
    });
    let error = result.err().unwrap().to_string();
    assert!(error.contains("ambiguous label-image boundary"), "{error}");
    assert!(
        !prepared.get(),
        "ambiguous rows must not reach external preparation"
    );
    assert_eq!(source.node_label_history(id), history);
    assert_eq!(source.node_property_history(id), properties);
    assert_eq!(
        source
            .get_node_history(id)
            .iter()
            .rev()
            .map(|(created, deleted, _)| (*created, *deleted))
            .collect::<Vec<_>>(),
        lifetimes
    );
    assert_eq!((source.next_node_id(), source.next_edge_id()), floors);
    assert_eq!(source.node_count(), 1);
    assert_eq!(source.nodes_by_label("Shared"), vec![id]);
    {
        let transition = source.pin_exclusive_unframed_transition().unwrap();
        assert!(transition.require_native_compact_source().is_ok());
    }
    // Supplying the missing explicit birth image makes the same source
    // representable; the failed conversion must not have retired its authority.
    source
        .replay_node_labels_at_epoch(id, epoch, &image)
        .unwrap();
    let expected = source.node_label_history(id);
    let layered = LayeredStore::from_native_temporal(source).unwrap();
    assert_eq!(layered.node_structural_history(id).label_versions, expected);
}

#[test]
fn ambiguous_label_boundary_rejects_merge_without_changing_generation() {
    let source = Arc::new(LpgStore::new().unwrap());
    let cold = source.create_node(&["Cold"]);
    let layered = LayeredStore::from_native_temporal(source).unwrap();
    let base = layered.base_store_arc();
    let overlay = layered.overlay_store();
    let cold_history = layered.node_structural_history(cold).label_versions;
    let id = NodeId::new(7);
    let epoch = EpochId::new(5);
    let image = labels(&["Shared"]);
    let history = vec![(epoch, image.clone())];
    overlay
        .restore_node_history_exact(id, &[(epoch, Some(epoch)), (epoch, None)], &history)
        .unwrap();
    overlay.set_node_property(id, "retained", Value::Int64(7));
    let properties = layered.node_property_full_history(id);
    let floors = (overlay.next_node_id(), overlay.next_edge_id());
    let prepared = Cell::new(false);
    let published = Cell::new(false);
    let result = layered.merge_overlay_temporal_with_publication(
        |candidate| {
            prepared.set(true);
            Ok((candidate, ()))
        },
        |()| published.set(true),
        |()| (),
    );
    let error = result.unwrap_err();
    assert!(error.contains("ambiguous label-image boundary"), "{error}");
    assert!(!prepared.get());
    assert!(!published.get());
    assert!(Arc::ptr_eq(&base, &layered.base_store_arc()));
    assert!(Arc::ptr_eq(&overlay, &layered.overlay_store()));
    assert_eq!(layered.node_structural_history(id).label_versions, history);
    assert_eq!(
        layered.node_structural_history(cold).label_versions,
        cold_history
    );
    assert_eq!(layered.node_property_full_history(id), properties);
    assert_eq!((overlay.next_node_id(), overlay.next_edge_id()), floors);
    assert_eq!(layered.node_count(), 2);
    assert_eq!(layered.nodes_by_label("Shared"), vec![id]);
    overlay
        .replay_node_labels_at_epoch(id, epoch, &image)
        .unwrap();
    let expected = overlay.node_label_history(id);
    layered.merge_overlay_temporal().unwrap();
    assert_eq!(layered.node_structural_history(id).label_versions, expected);
    assert_eq!(layered.node_count(), 2);
}

#[test]
fn exact_label_images_survive_cold_hydration_and_repeated_temporal_folds() {
    let source = Arc::new(LpgStore::new().unwrap());
    let birth = EpochId::new(2);
    source.sync_epoch(birth);
    let id = source.create_node(&["Initial"]);
    source.set_node_property(id, "untouched", Value::from("retained"));
    let first_image = labels(&["First"]);
    source
        .replay_node_labels_at_epoch(id, birth, &first_image)
        .unwrap();
    source
        .replay_node_labels_at_epoch(id, birth, &first_image)
        .unwrap();
    let mut expected = source.node_label_history(id);
    let properties = source.node_property_history(id);
    let layered = LayeredStore::from_native_temporal(source).unwrap();
    assert_eq!(layered.node_structural_history(id).label_versions, expected);
    assert!(!layered.overlay_store().contains_node_identity(id));
    let next = EpochId::new(4);
    layered.overlay_store().sync_epoch(next);
    let duplicate = labels(&["Never", "Never"]);
    assert!(
        layered
            .replay_node_labels_at_epoch(id, next, &duplicate)
            .is_err()
    );
    assert!(!layered.overlay_store().contains_node_identity(id));
    assert!(layered.overlay_store().label_id("Never").is_none());
    assert!(
        layered
            .replay_node_labels_at_epoch(id, birth, &labels(&["Never"]))
            .is_err()
    );
    assert!(!layered.overlay_store().contains_node_identity(id));

    let final_image = labels(&["Final", "Shared"]);
    for image in [&final_image, &final_image] {
        layered
            .replay_node_labels_at_epoch(id, next, image)
            .unwrap();
        expected.push((next, image.clone()));
    }
    assert_eq!(layered.node_structural_history(id).label_versions, expected);
    assert_eq!(layered.overlay_store().node_label_history(id), expected);
    assert_eq!(layered.node_property_full_history(id), properties);
    assert!(layered.nodes_by_label("First").is_empty());
    assert_eq!(layered.nodes_by_label("Final"), vec![id]);
    assert_eq!(layered.node_count(), 1);
    for _ in 0..2 {
        layered.merge_overlay_temporal().unwrap();
        assert_eq!(layered.node_structural_history(id).label_versions, expected);
        assert_eq!(layered.node_property_full_history(id), properties);
        assert_eq!(layered.nodes_by_label("Final"), vec![id]);
        assert_eq!(layered.node_count(), 1);
    }
    layered
        .replay_node_labels_at_epoch(id, next, &final_image)
        .unwrap();
    expected.push((next, final_image));
    assert_eq!(layered.node_structural_history(id).label_versions, expected);
    assert!(layered.delete_node(id));
    assert!(
        layered
            .replay_node_labels_at_epoch(id, next, &labels(&["Never"]))
            .is_err()
    );
    assert_eq!(layered.node_structural_history(id).label_versions, expected);
    assert!(layered.get_node(id).is_none());
    assert_eq!(layered.node_count(), 0);
    layered.merge_overlay_temporal().unwrap();
    assert_eq!(layered.node_structural_history(id).label_versions, expected);
}

#[test]
fn layered_exact_label_replay_requires_owner_before_hydration() {
    let source = Arc::new(LpgStore::new().unwrap());
    let id = source.create_node(&["Before"]);
    let owner = WriteAuthority::new();
    let foreign = WriteAuthority::new();
    assert!(source.seal_unframed_writes(&owner));
    let layered = with_authority(&owner, || LayeredStore::from_native_temporal(source)).unwrap();
    let overlay = layered.overlay_store();
    let epoch = overlay.current_epoch();
    let image = labels(&["After"]);
    let history = layered.node_structural_history(id).label_versions;
    assert!(
        layered
            .replay_node_labels_at_epoch(id, epoch, &image)
            .is_err()
    );
    assert!(
        with_authority(&foreign, || layered
            .replay_node_labels_at_epoch(id, epoch, &image))
        .is_err()
    );
    assert!(!overlay.contains_node_identity(id));
    assert!(overlay.label_id("After").is_none());
    assert_eq!(layered.node_structural_history(id).label_versions, history);
    with_authority(&owner, || {
        layered.replay_node_labels_at_epoch(id, epoch, &image)
    })
    .unwrap();
    assert!(overlay.contains_node_identity(id));
    assert_eq!(layered.nodes_by_label("After"), vec![id]);
}
