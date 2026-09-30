use super::*;
use crate::graph::compact::from_graph_store_preserving_ids;
use crate::graph::traits::{GraphStore, GraphStoreMut};

const TX: TransactionId = TransactionId::new(12);
const FOREIGN: TransactionId = TransactionId::new(13);
const P: EpochId = EpochId::new(5);
const C: EpochId = EpochId::new(6);

struct Fixture {
    store: LayeredStore,
    node: NodeId,
    foreign: NodeId,
    edge: EdgeId,
}

fn fixture() -> Fixture {
    let source = LpgStore::new().unwrap();
    let node = source.create_node(&["Cold"]);
    let foreign = source.create_node(&["Cold"]);
    let other = source.create_node(&["Other"]);
    let edge = source.create_edge(foreign, other, "LINK");
    let base = from_graph_store_preserving_ids(&source).unwrap();
    let store = LayeredStore::new(base, other.as_u64(), edge.as_u64()).unwrap();
    assert!(store.delete_node_versioned(node, P, TX));
    assert!(store.delete_edge_versioned(edge, P, TX));
    assert!(store.delete_node_versioned(foreign, P, FOREIGN));
    store.deletions_dirty.store(false, Ordering::Release);
    Fixture {
        store,
        node,
        foreign,
        edge,
    }
}

fn slot(fixture: &Fixture) -> LayeredCommitSlot<'_> {
    LayeredCommitSlot::new(&fixture.store, LayeredCommitWorkspace::new(TX, P, C))
}

fn assert_unbound(store: &LayeredStore) {
    assert!(store.pending_base_node_deletes.try_write().is_some());
    assert!(store.deleted_from_base_nodes.try_write().is_some());
    assert!(store.pending_base_edge_deletes.try_write().is_some());
    assert!(store.deleted_from_base_edges.try_write().is_some());
    assert!(store.publication_guard.try_write().is_some());
}

#[test]
fn two_cold_targets_install_without_final_allocator_traffic_or_foreign_changes() {
    let fixtures = [fixture(), fixture()];
    let overlays: Vec<_> = fixtures.iter().map(|f| f.store.overlay_store()).collect();
    let unrelated = LpgStore::new().unwrap();
    let mut slots = LayeredCommitSlots::new(fixtures.iter().map(slot).collect());
    let pins: Vec<_> = fixtures
        .iter()
        .zip(&overlays)
        .map(|(f, overlay)| f.store.pin_commit(overlay).unwrap())
        .collect();
    // Deliberately not slot order, with an unrelated native transition.
    let transitions = [
        unrelated.pin_exclusive_unframed_transition().unwrap(),
        overlays[1].pin_exclusive_unframed_transition().unwrap(),
        overlays[0].pin_exclusive_unframed_transition().unwrap(),
    ];
    let released = prepare_layered_commit_slots(&mut slots, &pins, &transitions).unwrap();
    crate::allocation_test::start();
    let readers = released.exclude_readers().unwrap();
    for fixture in &fixtures {
        assert!(fixture.store.publication_guard.try_read().is_none());
        assert!(fixture.store.deleted_from_base_nodes.try_write().is_some());
    }
    let ready = readers.rebind().unwrap();
    let readers = ready.release();
    for fixture in &fixtures {
        assert!(fixture.store.publication_guard.try_read().is_none());
        assert!(fixture.store.deleted_from_base_nodes.try_write().is_some());
    }
    drop(readers.rebind().unwrap().install());
    let counts = crate::allocation_test::stop();
    assert_eq!(counts, crate::allocation_test::Counts::default());
    for (fixture, overlay) in fixtures.iter().zip(&overlays) {
        assert_unbound(&fixture.store);
        assert_eq!(
            fixture.store.deleted_from_base_nodes.read()[&fixture.node].epoch,
            C
        );
        assert_eq!(
            fixture.store.deleted_from_base_edges.read()[&fixture.edge].epoch,
            C
        );
        assert_eq!(
            fixture.store.deleted_from_base_nodes.read()[&fixture.foreign].epoch,
            EpochId::PENDING
        );
        assert!(
            fixture
                .store
                .pending_base_node_deletes
                .read()
                .contains_key(&FOREIGN)
        );
        assert!(
            !fixture
                .store
                .pending_base_node_deletes
                .read()
                .contains_key(&TX)
        );
        assert!(
            !fixture
                .store
                .pending_base_edge_deletes
                .read()
                .contains_key(&TX)
        );
        assert!(
            fixture
                .store
                .is_node_visible_versioned(fixture.node, P, FOREIGN)
        );
        assert!(
            !fixture
                .store
                .is_node_visible_versioned(fixture.node, C, FOREIGN)
        );
        assert!(
            fixture
                .store
                .is_edge_visible_versioned(fixture.edge, P, FOREIGN)
        );
        assert!(
            !fixture
                .store
                .is_edge_visible_versioned(fixture.edge, C, FOREIGN)
        );
        assert!(fixture.store.deletions_dirty.load(Ordering::Acquire));
        assert_eq!(overlay.node_count(), 0);
        assert_eq!(overlay.edge_count(), 0);
    }
    assert!(slots.slots.iter().all(
        |slot| slot.workspace.retired_nodes.is_some() && slot.workspace.retired_edges.is_some()
    ));
}

#[test]
fn late_generation_map_and_stamp_failures_drain_every_target_without_allocation() {
    for contended in 0..6 {
        let fixtures = [fixture(), fixture()];
        let overlays: Vec<_> = fixtures.iter().map(|f| f.store.overlay_store()).collect();
        let mut slots = LayeredCommitSlots::new(fixtures.iter().map(slot).collect());
        let pins: Vec<_> = fixtures
            .iter()
            .zip(&overlays)
            .map(|(f, overlay)| f.store.pin_commit(overlay).unwrap())
            .collect();
        let transitions: Vec<_> = overlays
            .iter()
            .map(|overlay| overlay.pin_exclusive_unframed_transition().unwrap())
            .collect();
        let released = prepare_layered_commit_slots(&mut slots, &pins, &transitions).unwrap();
        let second = &fixtures[1].store;
        if contended == 5 {
            second
                .deleted_from_base_edges
                .write()
                .get_mut(&fixtures[1].edge)
                .unwrap()
                .deleter = Some(FOREIGN);
        }
        let generation = (contended == 0).then(|| second.publication_guard.read());
        let pending_nodes = (contended == 1).then(|| second.pending_base_node_deletes.read());
        let nodes = (contended == 2).then(|| second.deleted_from_base_nodes.read());
        let pending_edges = (contended == 3).then(|| second.pending_base_edge_deletes.read());
        let edges = (contended == 4).then(|| second.deleted_from_base_edges.read());
        crate::allocation_test::start();
        let result = released
            .exclude_readers()
            .and_then(|readers| readers.rebind());
        let counts = crate::allocation_test::stop();
        assert_eq!(counts, crate::allocation_test::Counts::default());
        if contended == 5 {
            assert!(matches!(result, Err(DataRebindError::Invalid(_))));
        } else {
            assert!(matches!(result, Err(DataRebindError::Conflict(_))));
        }
        drop(result);
        drop((generation, pending_nodes, nodes, pending_edges, edges));
        for fixture in &fixtures {
            assert_unbound(&fixture.store);
            assert_eq!(
                fixture.store.deleted_from_base_nodes.read()[&fixture.node].epoch,
                EpochId::PENDING
            );
            assert!(
                fixture
                    .store
                    .pending_base_node_deletes
                    .read()
                    .contains_key(&TX)
            );
            assert!(!fixture.store.deletions_dirty.load(Ordering::Acquire));
        }
        assert!(
            slots
                .slots
                .iter()
                .all(|slot| slot.workspace.retired_nodes.is_none()
                    && slot.workspace.retired_edges.is_none())
        );
    }
}

#[test]
fn exact_sequence_and_one_shot_pin_rules_survive_batch_adaptation() {
    let fixtures = [fixture(), fixture()];
    let overlays: Vec<_> = fixtures.iter().map(|f| f.store.overlay_store()).collect();
    let mut wrong = LayeredCommitSlots::new(vec![slot(&fixtures[1]), slot(&fixtures[0])]);
    let mut first = LayeredCommitSlots::new(fixtures.iter().map(slot).collect());
    let mut second = LayeredCommitSlots::new(fixtures.iter().map(slot).collect());
    let pins: Vec<_> = fixtures
        .iter()
        .zip(&overlays)
        .map(|(f, overlay)| f.store.pin_commit(overlay).unwrap())
        .collect();
    let transitions: Vec<_> = overlays
        .iter()
        .map(|overlay| overlay.pin_exclusive_unframed_transition().unwrap())
        .collect();
    assert!(matches!(
        prepare_layered_commit_slots(&mut wrong, &pins, &transitions),
        Err(DataRebindError::Invalid(_))
    ));
    assert!(pins.iter().all(|pin| !pin.prepared.get()));
    let released = prepare_layered_commit_slots(&mut first, &pins, &transitions).unwrap();
    assert!(matches!(
        prepare_layered_commit_slots(&mut second, &pins, &transitions),
        Err(DataRebindError::Invalid(_))
    ));
    assert!(second.slots.iter().all(|slot| !slot.workspace.attempted));
    drop(released);
    for fixture in &fixtures {
        assert_unbound(&fixture.store);
    }
}

#[test]
fn forgotten_all_phases_and_outer_cleanup_retire_after_every_authority() {
    for (stage, explicit_cleanup) in [
        (0, false),
        (1, false),
        (2, false),
        (0, true),
        (1, true),
        (2, true),
    ] {
        let fixtures = [fixture(), fixture()];
        let overlays: Vec<_> = fixtures.iter().map(|f| f.store.overlay_store()).collect();
        let gate = parking_lot::Mutex::new(());
        let probes = std::sync::atomic::AtomicUsize::new(0);
        let mut slots = LayeredCommitSlots::new(
            fixtures
                .iter()
                .map(|fixture| {
                    let mut slot = slot(fixture);
                    slot.before_retire = Some(Box::new(|| {
                        assert!(gate.try_lock().is_some());
                        for fixture in &fixtures {
                            assert_unbound(&fixture.store);
                            assert!(fixture.store.merge_guard.try_write().is_some());
                        }
                        for overlay in &overlays {
                            assert!(overlay.pin_exclusive_unframed_transition().is_some());
                        }
                        probes.fetch_add(1, Ordering::Relaxed);
                    }));
                    slot
                })
                .collect(),
        );
        let mut pins = Vec::with_capacity(fixtures.len());
        let mut transitions = Vec::with_capacity(fixtures.len());
        let outer = gate.lock();
        pins.extend(
            fixtures
                .iter()
                .zip(&overlays)
                .map(|(f, overlay)| f.store.pin_commit(overlay).unwrap()),
        );
        transitions.extend(
            overlays
                .iter()
                .map(|overlay| overlay.pin_exclusive_unframed_transition().unwrap()),
        );
        let readers = prepare_layered_commit_slots(&mut slots, &pins, &transitions)
            .unwrap()
            .exclude_readers()
            .unwrap();
        match stage {
            0 => std::mem::forget(readers),
            1 => std::mem::forget(readers.rebind().unwrap()),
            _ => std::mem::forget(readers.rebind().unwrap().install()),
        }
        // Models the aggregate's Drop: drain every family, then authority,
        // then enclosing gates, and only then payloads and backing buffers.
        if explicit_cleanup {
            crate::allocation_test::start();
            slots.release_guards();
            let counts = crate::allocation_test::stop();
            assert_eq!(counts, crate::allocation_test::Counts::default());
        }
        transitions.clear();
        pins.clear();
        drop(outer);
        drop(transitions);
        drop(pins);
        drop(slots);
        assert_eq!(probes.load(Ordering::Relaxed), 2);
        for fixture in &fixtures {
            assert_eq!(
                fixture.store.deleted_from_base_nodes.read()[&fixture.node].epoch,
                if stage == 2 { C } else { EpochId::PENDING }
            );
        }
    }
}
