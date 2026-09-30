use super::*;
use crate::graph::compact::from_graph_store_preserving_ids;
use crate::graph::traits::{GraphStore, GraphStoreMut, LpgCommitRepresentation};

fn fixture() -> (LayeredStore, NodeId, NodeId, EdgeId) {
    let source = LpgStore::new().unwrap();
    let node = source.create_node(&["Cold"]);
    let foreign = source.create_node(&["Cold"]);
    let other = source.create_node(&["Other"]);
    let edge = source.create_edge(foreign, other, "LINK");
    let base = from_graph_store_preserving_ids(&source).unwrap();
    (
        LayeredStore::new(base, other.as_u64(), edge.as_u64()).unwrap(),
        node,
        foreign,
        edge,
    )
}

fn workspace() -> LayeredCommitWorkspace {
    LayeredCommitWorkspace::new(TransactionId::new(12), EpochId::new(5), EpochId::new(6))
}

fn pending(layered: &LayeredStore, node: NodeId, edge: EdgeId) {
    assert!(layered.delete_node_versioned(node, EpochId::new(5), TransactionId::new(12)));
    assert!(layered.delete_edge_versioned(edge, EpochId::new(5), TransactionId::new(12)));
}

#[test]
fn cold_only_deletes_publish_at_commit_and_retain_foreign_ownership_without_promotion() {
    let (layered, node, foreign, edge) = fixture();
    pending(&layered, node, edge);
    let foreign_tx = TransactionId::new(13);
    assert!(layered.delete_node_versioned(foreign, EpochId::new(5), foreign_tx));
    let overlay = layered.overlay_store();
    assert!(
        overlay
            .take_pending_deletes(TransactionId::new(12))
            .is_empty()
    );
    assert!(
        overlay
            .take_pending_edge_deletes(TransactionId::new(12))
            .is_empty()
    );
    assert_eq!(overlay.node_count(), 0);
    layered.deletions_dirty.store(false, Ordering::Release);
    let mut workspace = workspace();
    {
        let pin = layered.pin_commit(&overlay).unwrap();
        let transition = overlay.pin_exclusive_unframed_transition().unwrap();
        let released = pin.prepare(&transition, &mut workspace).unwrap();
        assert_eq!(
            layered.deleted_from_base_nodes.read()[&node].epoch,
            EpochId::PENDING
        );
        assert!(layered.is_node_visible_versioned(node, EpochId::new(6), foreign_tx));
        crate::allocation_test::start();
        let publication = pin.exclude_readers(&transition).unwrap();
        let ready = released.rebind(&publication).unwrap();
        let released = ready.release();
        let ready = released.rebind(&publication).unwrap();
        let installed = ready.install();
        let retained = layered.deleted_from_base_nodes.try_read().is_none()
            && layered.pending_base_edge_deletes.try_read().is_none()
            && layered.publication_guard.try_read().is_none();
        drop(installed);
        drop(publication);
        let traffic = crate::allocation_test::stop();
        assert!(retained);
        assert_eq!(traffic, crate::allocation_test::Counts::default());
        assert!(layered.merge_guard.try_write().is_none());
    }
    assert!(workspace.retired_nodes.is_some());
    assert!(workspace.retired_edges.is_some());
    assert!(layered.deletions_dirty.load(Ordering::Acquire));
    assert!(
        layered
            .pending_base_node_deletes
            .read()
            .contains_key(&foreign_tx)
    );
    assert_eq!(
        layered.deleted_from_base_nodes.read()[&foreign].epoch,
        EpochId::PENDING
    );
    assert!(layered.is_node_visible_versioned(node, EpochId::new(5), foreign_tx));
    assert!(!layered.is_node_visible_versioned(node, EpochId::new(6), foreign_tx));
    assert!(layered.is_edge_visible_versioned(edge, EpochId::new(5), foreign_tx));
    assert!(!layered.is_edge_visible_versioned(edge, EpochId::new(6), foreign_tx));
    assert!(layered.dirty_node_ids.read().is_empty());
    assert!(layered.dirty_edge_ids.read().is_empty());
    assert_eq!(overlay.node_count(), 0);
    assert_eq!(overlay.edge_count(), 0);
}

#[test]
fn every_final_contention_path_releases_its_prefix_without_allocator_traffic() {
    let (layered, node, _, edge) = fixture();
    pending(&layered, node, edge);
    let overlay = layered.overlay_store();
    macro_rules! rejects {
        ($held:expr) => {{
            let mut workspace = workspace();
            let pin = layered.pin_commit(&overlay).unwrap();
            let transition = overlay.pin_exclusive_unframed_transition().unwrap();
            let released = pin.prepare(&transition, &mut workspace).unwrap();
            let publication = pin.exclude_readers(&transition).unwrap();
            let held = $held;
            crate::allocation_test::start();
            let result = released.rebind(&publication);
            let conflict = matches!(result, Err(DataRebindError::Conflict(_)));
            drop(result);
            let traffic = crate::allocation_test::stop();
            drop(held);
            assert!(conflict, stringify!($held));
            assert_eq!(
                traffic,
                crate::allocation_test::Counts::default(),
                stringify!($held)
            );
            assert!(layered.pending_base_node_deletes.try_write().is_some());
            assert!(layered.deleted_from_base_nodes.try_write().is_some());
            assert!(layered.pending_base_edge_deletes.try_write().is_some());
            assert!(layered.deleted_from_base_edges.try_write().is_some());
            assert_eq!(
                layered.deleted_from_base_nodes.read()[&node].epoch,
                EpochId::PENDING
            );
            assert!(workspace.retired_nodes.is_none());
        }};
    }
    rejects!(layered.pending_base_node_deletes.read());
    rejects!(layered.deleted_from_base_nodes.read());
    rejects!(layered.pending_base_edge_deletes.read());
    rejects!(layered.deleted_from_base_edges.read());
    let mut workspace = workspace();
    let pin = layered.pin_commit(&overlay).unwrap();
    let transition = overlay.pin_exclusive_unframed_transition().unwrap();
    let (conflict, traffic) = {
        let _released = pin.prepare(&transition, &mut workspace).unwrap();
        let reader = super::super::GenerationReadScope::enter(&layered);
        crate::allocation_test::start();
        let result = pin.exclude_readers(&transition);
        let conflict = matches!(result, Err(DataRebindError::Conflict(_)));
        drop(result);
        let traffic = crate::allocation_test::stop();
        drop(reader);
        (conflict, traffic)
    };
    assert!(conflict);
    assert_eq!(traffic, crate::allocation_test::Counts::default());
    assert!(layered.publication_guard.try_write().is_some());
}

#[test]
fn wrong_representation_or_transition_is_rejected_and_pins_are_one_shot() {
    let (layered, node, _, edge) = fixture();
    pending(&layered, node, edge);
    let overlay = layered.overlay_store();
    let wrong = LpgStore::new().unwrap();
    assert!(matches!(
        layered.pin_commit(&wrong),
        Err(DataRebindError::Invalid(_))
    ));
    assert!(layered.merge_guard.try_write().is_some());
    let mut first = workspace();
    let mut second = workspace();
    let pin = layered.pin_commit(&overlay).unwrap();
    let wrong_transition = wrong.pin_exclusive_unframed_transition().unwrap();
    assert!(matches!(
        pin.prepare(&wrong_transition, &mut first),
        Err(DataRebindError::Invalid(_))
    ));
    assert!(!first.attempted);
    let transition = overlay.pin_exclusive_unframed_transition().unwrap();
    {
        let _released = pin.prepare(&transition, &mut first).unwrap();
        assert!(matches!(
            pin.prepare(&transition, &mut second),
            Err(DataRebindError::Invalid(_))
        ));
        assert!(!second.attempted);
    }
    assert_eq!(
        layered.deleted_from_base_nodes.read()[&node].epoch,
        EpochId::PENDING
    );
}

#[test]
fn changed_queue_or_foreign_stamp_rejects_final_binding_without_partial_publication() {
    for change_stamp in [false, true] {
        let (layered, node, _, edge) = fixture();
        pending(&layered, node, edge);
        let overlay = layered.overlay_store();
        let mut workspace = workspace();
        let pin = layered.pin_commit(&overlay).unwrap();
        let transition = overlay.pin_exclusive_unframed_transition().unwrap();
        let released = pin.prepare(&transition, &mut workspace).unwrap();
        // Hostile private mutation witnesses requalification; normal mutators
        // cannot alter either map while this exact transition is retained.
        if change_stamp {
            layered
                .deleted_from_base_edges
                .write()
                .get_mut(&edge)
                .unwrap()
                .deleter = Some(TransactionId::new(33));
        } else {
            layered
                .pending_base_edge_deletes
                .write()
                .get_mut(&TransactionId::new(12))
                .unwrap()
                .push(edge);
        }
        let publication = pin.exclude_readers(&transition).unwrap();
        crate::allocation_test::start();
        let result = released.rebind(&publication);
        let invalid = matches!(result, Err(DataRebindError::Invalid(_)));
        drop(result);
        let traffic = crate::allocation_test::stop();
        assert!(invalid);
        assert_eq!(traffic, crate::allocation_test::Counts::default());
        assert!(layered.pending_base_node_deletes.try_write().is_some());
        assert_eq!(
            layered.deleted_from_base_nodes.read()[&node].epoch,
            EpochId::PENDING
        );
        assert!(workspace.retired_nodes.is_none());
    }
}

#[test]
fn dropped_ready_proof_abandons_without_tombstone_or_queue_changes() {
    let (layered, node, _, edge) = fixture();
    pending(&layered, node, edge);
    let overlay = layered.overlay_store();
    let mut workspace = workspace();
    {
        let pin = layered.pin_commit(&overlay).unwrap();
        let transition = overlay.pin_exclusive_unframed_transition().unwrap();
        let released = pin.prepare(&transition, &mut workspace).unwrap();
        let publication = pin.exclude_readers(&transition).unwrap();
        drop(released.rebind(&publication).unwrap());
    }
    assert!(layered.publication_guard.try_write().is_some());
    assert!(layered.merge_guard.try_write().is_some());
    assert!(
        layered
            .pending_base_node_deletes
            .read()
            .contains_key(&TransactionId::new(12))
    );
    assert_eq!(
        layered.deleted_from_base_nodes.read()[&node].epoch,
        EpochId::PENDING
    );
    assert_eq!(
        layered.deleted_from_base_edges.read()[&edge].epoch,
        EpochId::PENDING
    );
    assert!(workspace.retired_nodes.is_none());
}

#[test]
fn commit_target_identifies_exact_representation_but_not_read_only_views() {
    let (layered, _, _, _) = fixture();
    let overlay = layered.overlay_store();
    assert!(
        matches!(layered.lpg_commit_target().unwrap().representation, LpgCommitRepresentation::Layered(target)
        if std::ptr::eq(target, &raw const layered))
    );
    assert!(
        matches!(overlay.lpg_commit_target().unwrap().representation, LpgCommitRepresentation::Native(target)
        if std::ptr::eq(target, overlay.as_ref()))
    );
    assert!(
        crate::graph::traits::NullGraphStore
            .lpg_commit_target()
            .is_err()
    );
}
