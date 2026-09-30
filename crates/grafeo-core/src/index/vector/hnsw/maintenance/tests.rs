use super::*;
use crate::index::vector::DistanceMetric;
use crate::index::vector::paged_topology::serialize_topology;
use bytes::Bytes;
use std::sync::{Condvar, Mutex, mpsc};
use std::time::Duration;

fn fixture(count: u64) -> (HnswIndex, HashMap<NodeId, Arc<[f32]>>) {
    let index = HnswIndex::with_seed(HnswConfig::new(4, DistanceMetric::Euclidean).with_m(4), 41);
    let vectors: HashMap<_, Arc<[f32]>> = (0..count)
        .map(|id| {
            (
                NodeId::new(id),
                Arc::from([id as f32, (id % 7) as f32, 1.0, 0.5]),
            )
        })
        .collect();
    let accessor = |id| vectors.get(&id).cloned();
    for id in 0..count {
        index.insert(NodeId::new(id), &vectors[&NodeId::new(id)], &accessor);
    }
    (index, vectors)
}

fn assert_exact(actual: HnswExactState, expected: HnswExactState) {
    assert_eq!(actual.entry_point, expected.entry_point);
    assert_eq!(actual.max_level, expected.max_level);
    assert_eq!(actual.nodes, expected.nodes);
    assert_eq!(actual.deleted, expected.deleted);
    assert_eq!(actual.rng_state, expected.rng_state);
}

#[test]
fn sparse_postimage_matches_real_insertion_and_preserves_mmap_base() {
    for mmap in [false, true] {
        let (index, mut vectors) = fixture(96);
        let (reference, _) = fixture(96);
        index.remove(NodeId::new(7));
        reference.remove(NodeId::new(7));
        if mmap {
            let (entry, level, nodes) = index.snapshot_topology();
            let topology =
                MmapTopology::from_bytes(Bytes::from(serialize_topology(entry, level, &nodes)))
                    .unwrap();
            index.adopt_mmap_topology(topology);
        }
        let before = index.snapshot_exact().unwrap();
        // Re-embed the actual entry point, resurrect a soft delete, insert a
        // new identity, and remove a live identity. Duplicate final absence
        // dominates an otherwise valid upsert.
        let entry = before.entry_point.unwrap();
        let changed: Arc<[f32]> = Arc::from([0.25, 1.5, 3.0, 5.0]);
        let created: Arc<[f32]> = Arc::from([99.0, 4.0, 1.0, 0.5]);
        vectors.insert(entry, Arc::clone(&changed));
        vectors.insert(NodeId::new(100), Arc::clone(&created));
        let mut operations = vec![
            (entry, Some(changed)),
            (NodeId::new(7), Some(Arc::clone(&vectors[&NodeId::new(7)]))),
            (NodeId::new(100), Some(created)),
            (NodeId::new(95), None),
            (
                NodeId::new(95),
                Some(Arc::clone(&vectors[&NodeId::new(95)])),
            ),
        ];
        // Avoid accidental overlap between the sampled entry point and the
        // fixture's other operations while retaining genuine entry replacement.
        operations.retain(|(id, _)| *id != entry);
        operations.push((entry, Some(Arc::clone(&vectors[&entry]))));
        let accessor = |id| vectors.get(&id).cloned();
        let mut workspace = HnswMaintenanceWorkspace::new(operations);
        let pin = index.pin_maintenance().unwrap();
        let released = pin.prepare(&mut workspace, &accessor).unwrap().release();
        assert_exact(index.snapshot_exact().unwrap(), before.clone());
        for (id, vector) in &released.workspace.operations {
            reference.remove(*id);
            if let Some(vector) = vector {
                reference.insert(*id, vector, &accessor);
            }
        }
        assert!(
            released.workspace.nodes.len() < 96,
            "must not clone every topology node"
        );
        let readers = pin.exclude_readers();
        crate::allocation_test::start();
        let ready = released.rebind(&readers).unwrap();
        let rebind = crate::allocation_test::stop();
        assert_eq!(rebind, crate::allocation_test::Counts::default());
        crate::allocation_test::start();
        let installed = ready.install();
        let install = crate::allocation_test::stop();
        assert_eq!(install, crate::allocation_test::Counts::default());
        drop(installed);
        assert_exact(
            index.snapshot_exact().unwrap(),
            reference.snapshot_exact().unwrap(),
        );
        if mmap {
            let nodes = index.nodes.read();
            let TopologyBackend::Mmap {
                base,
                overrides,
                additional_nodes,
            } = &*nodes
            else {
                panic!("sparse maintenance must retain the mmap backend");
            };
            assert_eq!(base.len(), 96);
            assert_eq!(*additional_nodes, 1);
            assert!(overrides.len() < base.len());
        }
        drop(readers);
        drop(pin);
        // The moved-out map's allocation and displaced nodes remain outer-owned.
        assert!(workspace.nodes.is_empty());
        assert!(workspace.nodes.capacity() > 0);
        if !mmap {
            assert!(!workspace.retired_nodes.is_empty());
        }
    }
}

#[test]
fn empty_and_full_capacity_replacements_install_without_allocator_traffic() {
    for count in [0, 3] {
        let (index, mut vectors) = fixture(count);
        if count != 0 {
            let mut nodes = index.nodes.write();
            let TopologyBackend::Heap(nodes) = &mut *nodes else {
                panic!("heap fixture");
            };
            nodes.shrink_to_fit();
            assert_eq!(nodes.len(), nodes.capacity());
        }
        let vector: Arc<[f32]> = Arc::from([0.1, 0.2, 0.3, 0.4]);
        vectors.insert(NodeId::new(0), Arc::clone(&vector));
        let accessor = |id| vectors.get(&id).cloned();
        let mut workspace = HnswMaintenanceWorkspace::new(vec![(NodeId::new(0), Some(vector))]);
        let pin = index.pin_maintenance().unwrap();
        let released = pin.prepare(&mut workspace, &accessor).unwrap().release();
        let readers = pin.exclude_readers();
        let ready = released.rebind(&readers).unwrap();
        crate::allocation_test::start();
        let installed = ready.install();
        let counts = crate::allocation_test::stop();
        assert_eq!(counts, crate::allocation_test::Counts::default());
        drop(installed);
        assert!(index.contains(NodeId::new(0)));
    }
}

#[test]
fn late_failure_and_abandonment_leave_live_state_exact() {
    let (mut index, vectors) = fixture(8);
    index.config.max_elements = Some(9);
    let before = index.snapshot_exact().unwrap();
    let vector: Arc<[f32]> = Arc::from([0.1, 0.2, 0.3, 0.4]);
    let mut workspace = HnswMaintenanceWorkspace::new(vec![
        (NodeId::new(10), Some(Arc::clone(&vector))),
        (NodeId::new(11), Some(vector)),
    ]);
    let pin = index.pin_maintenance().unwrap();
    let accessor = |id| vectors.get(&id).cloned();
    assert!(pin.prepare(&mut workspace, &accessor).is_err());
    assert!(
        !workspace.nodes.is_empty(),
        "retain the completed prefix after late rejection"
    );
    assert_exact(index.snapshot_exact().unwrap(), before.clone());
    assert!(pin.prepare(&mut workspace, &accessor).is_err());
    drop(pin);
    let pin = index.pin_maintenance().unwrap();
    let mut removal = HnswMaintenanceWorkspace::new(vec![(NodeId::new(0), None)]);
    {
        let _consumed_proof = pin.prepare(&mut removal, &accessor).unwrap().release();
    }
    assert_exact(index.snapshot_exact().unwrap(), before);
}

#[test]
fn all_candidate_and_routing_dimensions_fail_before_live_mutation() {
    let (index, mut vectors) = fixture(4);
    let before = index.snapshot_exact().unwrap();
    let pin = index.pin_maintenance().unwrap();
    let mut wrong_candidate =
        HnswMaintenanceWorkspace::new(vec![(NodeId::new(8), Some(Arc::from([1.0])))]);
    let error = pin
        .prepare(&mut wrong_candidate, &|id| vectors.get(&id).cloned())
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("final-row vector has wrong dimensions")
    );
    drop(pin);
    let pin = index.pin_maintenance().unwrap();
    let entry = before.entry_point.unwrap();
    vectors.insert(entry, Arc::from([1.0]));
    let mut wrong_routing = HnswMaintenanceWorkspace::new(vec![(
        NodeId::new(8),
        Some(Arc::from([1.0, 2.0, 3.0, 4.0])),
    )]);
    let error = pin
        .prepare(&mut wrong_routing, &|id| vectors.get(&id).cloned())
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("final-row routing vector has wrong dimensions")
    );
    assert_exact(index.snapshot_exact().unwrap(), before);
}

#[test]
fn rebind_rejects_foreign_reader_fence_without_allocator_traffic() {
    let (index, vectors) = fixture(2);
    let (foreign, _) = fixture(2);
    let mut workspace = HnswMaintenanceWorkspace::new(vec![(NodeId::new(0), None)]);
    let pin = index.pin_maintenance().unwrap();
    let foreign_pin = foreign.pin_maintenance().unwrap();
    let released = pin
        .prepare(&mut workspace, &|id| vectors.get(&id).cloned())
        .unwrap()
        .release();
    let wrong_readers = foreign_pin.exclude_readers();
    crate::allocation_test::start();
    let result = released.rebind(&wrong_readers);
    let counts = crate::allocation_test::stop();
    assert_eq!(counts, crate::allocation_test::Counts::default());
    assert!(result.is_err());
}

#[test]
fn final_rebind_contention_is_allocation_free_and_drains_partial_guards() {
    for contended in 0..5 {
        let (index, vectors) = fixture(3);
        let before = index.snapshot_exact().unwrap();
        let mut workspace = HnswMaintenanceWorkspace::new(vec![(NodeId::new(0), None)]);
        let pin = index.pin_maintenance().unwrap();
        let released = pin
            .prepare(&mut workspace, &|id| vectors.get(&id).cloned())
            .unwrap()
            .release();
        let readers = pin.exclude_readers();
        // Hold each independent state reader in turn. This also exercises
        // cleanup after every possible prefix of acquired final writers.
        let nodes = (contended == 0).then(|| index.nodes.read());
        let entry = (contended == 1).then(|| index.entry_point.read());
        let level = (contended == 2).then(|| index.max_level.read());
        let rng = (contended == 3).then(|| index.rng.read());
        let deleted = (contended == 4).then(|| index.deleted.read());
        crate::allocation_test::start();
        let result = released.rebind(&readers);
        let counts = crate::allocation_test::stop();
        assert_eq!(counts, crate::allocation_test::Counts::default());
        assert!(matches!(result, Err(DataRebindError::Conflict(_))));
        if contended != 0 {
            assert!(index.nodes.try_write().is_some());
        }
        if contended != 1 {
            assert!(index.entry_point.try_write().is_some());
        }
        if contended != 2 {
            assert!(index.max_level.try_write().is_some());
        }
        if contended != 3 {
            assert!(index.rng.try_write().is_some());
        }
        if contended != 4 {
            assert!(index.deleted.try_write().is_some());
        }
        drop(deleted);
        drop(rng);
        drop(level);
        drop(entry);
        drop(nodes);
        assert_exact(index.snapshot_exact().unwrap(), before);
    }
}

#[test]
fn same_thread_aliases_cannot_mutate_released_postimage_and_pin_recovers_on_drop() {
    let (index, vectors) = fixture(3);
    let before = index.snapshot_exact().unwrap();
    let mut workspace = HnswMaintenanceWorkspace::new(vec![(NodeId::new(0), None)]);
    let accessor = |id| vectors.get(&id).cloned();
    let pin = index.pin_maintenance().unwrap();
    let released = pin.prepare(&mut workspace, &accessor).unwrap().release();
    index.insert(NodeId::new(9), &[1.0, 2.0, 3.0, 4.0], &accessor);
    assert!(!index.remove(NodeId::new(1)));
    assert!(index.pin_maintenance().is_err());
    assert_exact(index.snapshot_exact().unwrap(), before);
    let readers = pin.exclude_readers();
    let ready = released.rebind(&readers).unwrap();
    crate::allocation_test::start();
    let released_again = ready.release();
    let ready_again = released_again.rebind(&readers).unwrap();
    let counts = crate::allocation_test::stop();
    assert_eq!(counts, crate::allocation_test::Counts::default());
    drop(ready_again.install());
    drop(readers);
    drop(pin);
    assert!(index.remove(NodeId::new(1)));
}

#[test]
fn one_pin_cannot_prepare_two_independently_installable_postimages() {
    let (index, vectors) = fixture(3);
    let before = index.snapshot_exact().unwrap();
    let accessor = |id| vectors.get(&id).cloned();
    let mut first = HnswMaintenanceWorkspace::new(vec![(NodeId::new(0), None)]);
    let mut second = HnswMaintenanceWorkspace::new(vec![(NodeId::new(1), None)]);
    let pin = index.pin_maintenance().unwrap();
    let second_rejected = {
        let _released_first = pin.prepare(&mut first, &accessor).unwrap().release();
        // Never install either potentially stale postimage. Admission itself must
        // reject a second preparation while the first proof remains in scope.
        pin.prepare(&mut second, &accessor).is_err()
    };
    drop(pin);
    assert!(
        second_rejected,
        "one retained alias pin must admit only one preparation"
    );
    assert_exact(index.snapshot_exact().unwrap(), before);

    // Rejection occurs before claiming the second workspace. After every old
    // proof drains, a fresh pin may safely prepare and publish that workspace.
    let fresh_pin = index.pin_maintenance().unwrap();
    let released_second = fresh_pin.prepare(&mut second, &accessor).unwrap().release();
    let readers = fresh_pin.exclude_readers();
    drop(released_second.rebind(&readers).unwrap().install());
    assert!(index.contains(NodeId::new(0)));
    assert!(!index.contains(NodeId::new(1)));
}

#[test]
fn searches_are_parallel_and_maintenance_drains_graph_accessor_readers() {
    let (index, vectors) = fixture(4);
    let index = Arc::new(index);
    let vectors = Arc::new(vectors);
    let released = Arc::new((Mutex::new(false), Condvar::new()));
    let (entered_tx, entered_rx) = mpsc::channel();
    let mut workers = Vec::new();
    for _ in 0..2 {
        let index = Arc::clone(&index);
        let vectors = Arc::clone(&vectors);
        let released = Arc::clone(&released);
        let entered = entered_tx.clone();
        workers.push(std::thread::spawn(move || {
            let first = std::sync::atomic::AtomicBool::new(true);
            let accessor = |id| {
                if first.swap(false, Ordering::Relaxed) {
                    entered.send(()).unwrap();
                    let (lock, signal) = &*released;
                    let guard = lock.lock().unwrap();
                    let _wait = signal
                        .wait_timeout_while(guard, Duration::from_secs(5), |released| !*released)
                        .unwrap();
                }
                vectors.get(&id).cloned()
            };
            index.search(&[0.0, 0.0, 1.0, 0.5], 2, &accessor)
        }));
    }
    let first = entered_rx.recv_timeout(Duration::from_secs(3)).is_ok();
    let second = entered_rx.recv_timeout(Duration::from_secs(3)).is_ok();
    let (waiting_tx, waiting_rx) = mpsc::channel();
    let (exclusive_tx, exclusive_rx) = mpsc::channel();
    let maintenance_index = Arc::clone(&index);
    let maintenance = std::thread::spawn(move || {
        let pin = maintenance_index.pin_maintenance().unwrap();
        waiting_tx.send(()).unwrap();
        let _readers = pin.exclude_readers();
        exclusive_tx.send(()).unwrap();
    });
    let waiting = waiting_rx.recv_timeout(Duration::from_secs(3)).is_ok();
    let excluded_while_reading = exclusive_rx.try_recv().is_ok();
    let (lock, signal) = &*released;
    *lock.lock().unwrap() = true;
    signal.notify_all();
    for worker in workers {
        worker.join().unwrap();
    }
    maintenance.join().unwrap();
    assert!(first && second, "two graph accessors must run concurrently");
    assert!(waiting);
    assert!(
        !excluded_while_reading,
        "maintenance must drain admitted readers"
    );
    assert!(exclusive_rx.recv_timeout(Duration::from_secs(3)).is_ok());
}

#[test]
fn outer_slots_match_single_target_install_and_keep_final_buffer_traffic_zero() {
    use crate::index::vector::VectorIndexKind;
    use crate::index::vector::maintenance::{
        VectorMaintenanceSlot, VectorMaintenanceSlots, prepare_vector_maintenance_slots,
    };

    for mmap in [false, true] {
        let mut indexes = Vec::new();
        let mut references = Vec::new();
        let mut images = Vec::new();
        let mut changes = Vec::new();
        for _ in 0..2 {
            let (index, mut image) = fixture(3);
            let (reference, _) = fixture(3);
            if mmap {
                let (entry, level, nodes) = index.snapshot_topology();
                index.adopt_mmap_topology(
                    MmapTopology::from_bytes(Bytes::from(serialize_topology(entry, level, &nodes)))
                        .unwrap(),
                );
            } else {
                let mut nodes = index.nodes.write();
                let TopologyBackend::Heap(nodes) = &mut *nodes else {
                    panic!("heap fixture");
                };
                nodes.shrink_to_fit();
                assert_eq!(nodes.len(), nodes.capacity());
            }
            let replacement: Arc<[f32]> = Arc::from([7.0, 2.0, 3.0, 1.0]);
            image.insert(NodeId::new(0), Arc::clone(&replacement));
            let rows = vec![(NodeId::new(0), Some(replacement)), (NodeId::new(2), None)];
            let mut reference_workspace = HnswMaintenanceWorkspace::new(rows.clone());
            let pin = reference.pin_maintenance().unwrap();
            let released = pin
                .prepare(&mut reference_workspace, &|id| image.get(&id).cloned())
                .unwrap()
                .release();
            let readers = pin.exclude_readers();
            drop(released.rebind(&readers).unwrap().install());
            drop(readers);
            drop(pin);
            references.push(reference);
            indexes.push(VectorIndexKind::Hnsw(index));
            images.push(image);
            changes.push(rows);
        }
        let mut slots = VectorMaintenanceSlots::new(
            indexes
                .iter()
                .zip(changes)
                .map(|(index, rows)| VectorMaintenanceSlot::new(index, rows))
                .collect(),
        );
        // The pin buffer, like slot storage, is declared before the modeled
        // enclosing gate. Clearing releases pins without freeing that buffer.
        let mut pins = Vec::with_capacity(indexes.len());
        let gate = parking_lot::Mutex::new(());
        let outer = gate.lock();
        pins.extend(indexes.iter().map(|index| index.pin_maintenance().unwrap()));
        let released = prepare_vector_maintenance_slots(&mut slots, &pins, &|slot, id| {
            images[slot].get(&id).cloned()
        })
        .unwrap();
        let readers = released.exclude_readers().unwrap();
        crate::allocation_test::start();
        let ready = readers.rebind().unwrap();
        let readers = ready.release();
        for index in &indexes {
            let VectorIndexKind::Hnsw(index) = index else {
                panic!("HNSW fixture");
            };
            assert!(index.nodes.try_write().is_some());
            assert!(index.reader_admission.try_read().is_none());
        }
        let ready = readers.rebind().unwrap();
        let installed = ready.install();
        drop(installed);
        let counts = crate::allocation_test::stop();
        assert_eq!(counts, crate::allocation_test::Counts::default());
        for (index, reference) in indexes.iter().zip(&references) {
            let VectorIndexKind::Hnsw(index) = index else {
                panic!("HNSW fixture");
            };
            assert_exact(
                index.snapshot_exact().unwrap(),
                reference.snapshot_exact().unwrap(),
            );
        }
        pins.clear();
        drop(outer);
        crate::allocation_test::start();
        drop(slots);
        let retirement = crate::allocation_test::stop();
        assert!(retirement.dealloc > 0);
    }
}

#[test]
fn slots_reject_wrong_pin_sequence_and_preserve_one_shot_preparation() {
    use crate::index::vector::VectorIndexKind;
    use crate::index::vector::maintenance::{
        VectorMaintenanceSlot, VectorMaintenanceSlots, prepare_vector_maintenance_slots,
    };

    let (first, image) = fixture(3);
    let (second, _) = fixture(3);
    let indexes = [VectorIndexKind::Hnsw(first), VectorIndexKind::Hnsw(second)];
    let mut first_slots = VectorMaintenanceSlots::new(vec![VectorMaintenanceSlot::new(
        &indexes[0],
        vec![(NodeId::new(0), None)],
    )]);
    let mut second_slots = VectorMaintenanceSlots::new(vec![VectorMaintenanceSlot::new(
        &indexes[0],
        vec![(NodeId::new(1), None)],
    )]);
    let accessor = |_, id| image.get(&id).cloned();
    let wrong = [indexes[1].pin_maintenance().unwrap()];
    assert!(prepare_vector_maintenance_slots(&mut first_slots, &wrong, &accessor).is_err());
    drop(wrong);
    let pins = [indexes[0].pin_maintenance().unwrap()];
    let first = prepare_vector_maintenance_slots(&mut first_slots, &pins, &accessor).unwrap();
    assert!(prepare_vector_maintenance_slots(&mut second_slots, &pins, &accessor).is_err());
    drop(first);
    drop(pins);
    let fresh = [indexes[0].pin_maintenance().unwrap()];
    let second = prepare_vector_maintenance_slots(&mut second_slots, &fresh, &accessor).unwrap();
    drop(
        second
            .exclude_readers()
            .unwrap()
            .rebind()
            .unwrap()
            .install(),
    );
    let VectorIndexKind::Hnsw(index) = &indexes[0] else {
        panic!("HNSW fixture");
    };
    assert!(index.contains(NodeId::new(0)));
    assert!(!index.contains(NodeId::new(1)));
}

#[test]
fn late_slot_contention_releases_every_prior_target_without_allocator_traffic() {
    use crate::index::vector::VectorIndexKind;
    use crate::index::vector::maintenance::{
        VectorMaintenanceSlot, VectorMaintenanceSlots, prepare_vector_maintenance_slots,
    };

    for contended in 0..5 {
        let (first, image) = fixture(3);
        let (second, _) = fixture(3);
        let before = [
            first.snapshot_exact().unwrap(),
            second.snapshot_exact().unwrap(),
        ];
        let indexes = [VectorIndexKind::Hnsw(first), VectorIndexKind::Hnsw(second)];
        let mut slots = VectorMaintenanceSlots::new(
            indexes
                .iter()
                .map(|index| VectorMaintenanceSlot::new(index, vec![(NodeId::new(0), None)]))
                .collect(),
        );
        let pins: Vec<_> = indexes
            .iter()
            .map(|index| index.pin_maintenance().unwrap())
            .collect();
        let readers =
            prepare_vector_maintenance_slots(&mut slots, &pins, &|_, id| image.get(&id).cloned())
                .unwrap()
                .exclude_readers()
                .unwrap();
        let VectorIndexKind::Hnsw(second) = &indexes[1] else {
            panic!("HNSW fixture");
        };
        let nodes = (contended == 0).then(|| second.nodes.read());
        let entry = (contended == 1).then(|| second.entry_point.read());
        let level = (contended == 2).then(|| second.max_level.read());
        let rng = (contended == 3).then(|| second.rng.read());
        let deleted = (contended == 4).then(|| second.deleted.read());
        crate::allocation_test::start();
        let result = readers.rebind();
        let counts = crate::allocation_test::stop();
        assert_eq!(counts, crate::allocation_test::Counts::default());
        assert!(matches!(result, Err(DataRebindError::Conflict(_))));
        drop(result);
        drop((nodes, entry, level, rng, deleted));
        for (index, expected) in indexes.iter().zip(before) {
            let VectorIndexKind::Hnsw(index) = index else {
                panic!("HNSW fixture")
            };
            assert!(index.nodes.try_write().is_some());
            assert!(index.entry_point.try_write().is_some());
            assert!(index.max_level.try_write().is_some());
            assert!(index.rng.try_write().is_some());
            assert!(index.deleted.try_write().is_some());
            assert!(index.reader_admission.try_write().is_some());
            assert_exact(index.snapshot_exact().unwrap(), expected);
        }
    }
}

#[test]
fn forgotten_batch_drains_all_slots_before_the_first_payload_retires() {
    use crate::index::vector::VectorIndexKind;
    use crate::index::vector::maintenance::{
        VectorMaintenanceSlot, VectorMaintenanceSlots, prepare_vector_maintenance_slots,
    };

    let (first, image) = fixture(3);
    let (second, _) = fixture(3);
    let indexes = [VectorIndexKind::Hnsw(first), VectorIndexKind::Hnsw(second)];
    let probes = std::sync::atomic::AtomicUsize::new(0);
    let mut slots = VectorMaintenanceSlots::new(
        indexes
            .iter()
            .map(|index| {
                let mut slot = VectorMaintenanceSlot::new(index, vec![(NodeId::new(0), None)]);
                slot.before_retire_for_test(|| {
                    for index in &indexes {
                        let VectorIndexKind::Hnsw(index) = index else {
                            panic!("HNSW fixture")
                        };
                        assert!(index.nodes.try_write().is_some());
                        assert!(index.reader_admission.try_write().is_some());
                        assert!(!index.maintenance_active.load(Ordering::Acquire));
                    }
                    probes.fetch_add(1, Ordering::Relaxed);
                });
                slot
            })
            .collect(),
    );
    let pins: Vec<_> = indexes
        .iter()
        .map(|index| index.pin_maintenance().unwrap())
        .collect();
    let ready =
        prepare_vector_maintenance_slots(&mut slots, &pins, &|_, id| image.get(&id).cloned())
            .unwrap()
            .exclude_readers()
            .unwrap()
            .rebind()
            .unwrap();
    std::mem::forget(ready);
    drop(pins);
    drop(slots);
    assert_eq!(probes.load(Ordering::Relaxed), 2);
}
