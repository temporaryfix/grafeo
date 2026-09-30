use super::*;
use crate::index::vector::DistanceMetric;
use crate::index::vector::paged_topology::{MmapTopology, serialize_topology};
use bytes::Bytes;
use std::collections::BTreeMap;
use std::sync::{Condvar, Mutex, mpsc};
use std::time::Duration;

thread_local! {
    static BINARY_READ_SEAM: std::cell::RefCell<Option<Box<dyn Fn()>>> = const { std::cell::RefCell::new(None) };
}

pub(super) fn binary_read_seam() {
    BINARY_READ_SEAM.with(|seam| {
        if let Some(seam) = seam.borrow().as_ref() {
            seam();
        }
    });
}

fn kinds() -> [QuantizationType; 4] {
    [
        QuantizationType::None,
        QuantizationType::Scalar,
        QuantizationType::Binary,
        QuantizationType::Product { num_subvectors: 2 },
    ]
}

fn vector(id: u64) -> Arc<[f32]> {
    Arc::from([id as f32, (id % 7) as f32, 1.0, 0.5])
}

fn fixture(kind: QuantizationType, count: u64) -> QuantizedHnswIndex {
    let index = QuantizedHnswIndex::with_seed(
        HnswConfig::new(4, DistanceMetric::Euclidean).with_m(4),
        kind,
        41,
    )
    .with_training_threshold(10)
    .with_rescore_factor(3);
    for id in 0..count {
        index.insert(NodeId::new(id), &vector(id));
    }
    index
}

fn assert_exact(actual: &QuantizedExactState, expected: &QuantizedExactState) {
    assert_eq!(
        format!("{:?}", actual.hnsw.config),
        format!("{:?}", expected.hnsw.config)
    );
    assert_eq!(actual.hnsw.entry_point, expected.hnsw.entry_point);
    assert_eq!(actual.hnsw.max_level, expected.hnsw.max_level);
    assert_eq!(actual.hnsw.nodes, expected.hnsw.nodes);
    assert_eq!(actual.hnsw.deleted, expected.hnsw.deleted);
    assert_eq!(actual.hnsw.rng_state, expected.hnsw.rng_state);
    assert_eq!(actual.quantization_type, expected.quantization_type);
    assert_eq!(actual.vectors, expected.vectors);
    assert_eq!(actual.scalar_vectors, expected.scalar_vectors);
    assert_eq!(actual.binary_vectors, expected.binary_vectors);
    assert_eq!(actual.product_codes, expected.product_codes);
    assert_eq!(actual.rescore, expected.rescore);
    assert_eq!(actual.rescore_factor, expected.rescore_factor);
    assert_eq!(actual.training_threshold, expected.training_threshold);
    assert_eq!(actual.training_samples, expected.training_samples);
    assert_eq!(actual.quantizer_trained, expected.quantizer_trained);
    assert_eq!(
        format!("{:?}", actual.scalar_quantizer),
        format!("{:?}", expected.scalar_quantizer)
    );
    assert_eq!(
        format!("{:?}", actual.product_quantizer),
        format!("{:?}", expected.product_quantizer)
    );
}

/// Ordinary insertion/quantization is the reference. Its topology accessor is
/// the same immutable final-row image specified for a single atomic commit;
/// auxiliary mutation still proceeds in order, exercising real threshold code.
fn apply_reference(index: &QuantizedHnswIndex, operations: &[(NodeId, Option<Arc<[f32]>>)]) {
    let mut ordered = BTreeMap::new();
    for (id, value) in operations {
        let entry = ordered.entry(*id).or_insert_with(|| value.clone());
        if entry.is_some() {
            *entry = value.clone();
        }
    }
    let mut final_vectors = index.vectors.read().clone();
    for (id, value) in &ordered {
        if let Some(value) = value {
            final_vectors.insert(*id, Arc::clone(value));
        }
    }
    let accessor = |id| final_vectors.get(&id).cloned();
    for (id, value) in ordered {
        if let Some(value) = value {
            index.vectors.write().insert(id, Arc::clone(&value));
            index.hnsw.insert(id, &value, &accessor);
            match index.quantization_type {
                QuantizationType::None => {}
                QuantizationType::Scalar => index.insert_scalar_quantized(id, &value),
                QuantizationType::Binary => index.insert_binary_quantized(id, &value),
                QuantizationType::Product { num_subvectors } => {
                    index.insert_product_quantized(id, &value, num_subvectors);
                }
            }
        } else {
            index.remove(id);
        }
    }
}

fn install_checked(index: &QuantizedHnswIndex, workspace: &mut QuantizedMaintenanceWorkspace) {
    let before = index.snapshot_exact().unwrap();
    let pin = index.pin_maintenance().unwrap();
    let released = pin.prepare(workspace).unwrap();
    assert_exact(&index.snapshot_exact().unwrap(), &before);
    let readers = pin.exclude_readers();
    crate::allocation_test::start();
    let ready = released.rebind(&readers).unwrap();
    let counts = crate::allocation_test::stop();
    assert_eq!(counts, crate::allocation_test::Counts::default());
    crate::allocation_test::start();
    let released = ready.release();
    let ready = released.rebind(&readers).unwrap();
    let counts = crate::allocation_test::stop();
    assert_eq!(counts, crate::allocation_test::Counts::default());
    crate::allocation_test::start();
    let installed = ready.install();
    let counts = crate::allocation_test::stop();
    assert_eq!(counts, crate::allocation_test::Counts::default());
    crate::allocation_test::start();
    drop(installed);
    let counts = crate::allocation_test::stop();
    assert_eq!(counts, crate::allocation_test::Counts::default());
}

#[test]
fn quantized_outer_slots_preserve_all_families_training_and_mmap_with_zero_final_traffic() {
    use crate::index::vector::VectorIndexKind;
    use crate::index::vector::maintenance::{
        VectorMaintenancePin, VectorMaintenanceSlot, VectorMaintenanceSlots,
        prepare_vector_maintenance_slots,
    };

    for kind in kinds() {
        for (count, upserts) in [(6, 1), (9, 1), (8, 4), (12, 4)] {
            for mmap in [false, true] {
                let mut indexes = Vec::new();
                let mut references = Vec::new();
                let mut changes = Vec::new();
                for family in [QuantizationType::None, kind] {
                    let index = fixture(family, count);
                    let reference = fixture(family, count);
                    if mmap {
                        let (entry, level, nodes) = index.hnsw.snapshot_topology();
                        index.hnsw.adopt_mmap_topology(
                            MmapTopology::from_bytes(Bytes::from(serialize_topology(
                                entry, level, &nodes,
                            )))
                            .unwrap(),
                        );
                    }
                    let mut rows: Vec<_> = (20..20 + upserts)
                        .map(|id| (NodeId::new(id), Some(vector(id))))
                        .collect();
                    rows.push((NodeId::new(2), None));
                    apply_reference(&reference, &rows);
                    references.push(reference);
                    indexes.push(VectorIndexKind::Quantized(index));
                    changes.push(rows);
                }
                let mut slots = VectorMaintenanceSlots::new(
                    indexes
                        .iter()
                        .zip(changes)
                        .map(|(index, rows)| VectorMaintenanceSlot::new(index, rows))
                        .collect(),
                );
                let pins: Vec<_> = indexes
                    .iter()
                    .map(|index| index.pin_maintenance().unwrap())
                    .collect();
                let released = prepare_vector_maintenance_slots(&mut slots, &pins, &|_, _| {
                    panic!("Quantized maintenance must use its retained vector directory")
                })
                .unwrap();
                let readers = released.exclude_readers().unwrap();
                crate::allocation_test::start();
                let ready = readers.rebind().unwrap();
                let readers = ready.release();
                for pin in &pins {
                    let VectorMaintenancePin::Quantized(pin) = pin else {
                        panic!("Quantized pin");
                    };
                    assert!(pin.topology.state_guards_available_for_test());
                    assert!(pin.topology.try_exclude_readers().is_none());
                    assert!(pin.index.vectors.try_write().is_some());
                }
                let ready = readers.rebind().unwrap();
                drop(ready.install());
                let counts = crate::allocation_test::stop();
                assert_eq!(counts, crate::allocation_test::Counts::default());
                for (index, reference) in indexes.iter().zip(&references) {
                    let VectorIndexKind::Quantized(index) = index else {
                        panic!("Quantized fixture")
                    };
                    assert_exact(
                        &index.snapshot_exact().unwrap(),
                        &reference.snapshot_exact().unwrap(),
                    );
                }
            }
        }
    }
}

#[test]
fn late_quantized_slot_conflicts_drain_prior_target_and_all_partial_guards_without_allocation() {
    use crate::index::vector::VectorIndexKind;
    use crate::index::vector::maintenance::{
        VectorMaintenancePin, VectorMaintenanceSlot, VectorMaintenanceSlots,
        prepare_vector_maintenance_slots,
    };

    for contended in 0..8 {
        let indexes = [
            VectorIndexKind::Quantized(fixture(QuantizationType::Scalar, 9)),
            VectorIndexKind::Quantized(fixture(QuantizationType::Binary, 3)),
        ];
        let before: Vec<_> = indexes
            .iter()
            .map(|index| {
                let VectorIndexKind::Quantized(index) = index else {
                    panic!("Quantized fixture")
                };
                index.snapshot_exact().unwrap()
            })
            .collect();
        let mut slots = VectorMaintenanceSlots::new(
            indexes
                .iter()
                .map(|index| {
                    VectorMaintenanceSlot::new(index, vec![(NodeId::new(0), Some(vector(8)))])
                })
                .collect(),
        );
        let pins: Vec<_> = indexes
            .iter()
            .map(|index| index.pin_maintenance().unwrap())
            .collect();
        let readers = prepare_vector_maintenance_slots(&mut slots, &pins, &|_, _| None)
            .unwrap()
            .exclude_readers()
            .unwrap();
        let VectorIndexKind::Quantized(index) = &indexes[1] else {
            panic!("Quantized fixture")
        };
        let vectors = (contended == 0).then(|| index.vectors.read());
        let scalar = (contended == 1).then(|| index.scalar_quantizer.read());
        let product = (contended == 2).then(|| index.product_quantizer.read());
        let scalar_codes = (contended == 3).then(|| index.scalar_vectors.read());
        let binary_codes = (contended == 4).then(|| index.binary_vectors.read());
        let product_codes = (contended == 5).then(|| index.product_codes.read());
        let samples = (contended == 6).then(|| index.training_samples.read());
        let trained = (contended == 7).then(|| index.quantizer_trained.read());
        crate::allocation_test::start();
        let result = readers.rebind();
        let counts = crate::allocation_test::stop();
        assert_eq!(counts, crate::allocation_test::Counts::default());
        assert!(matches!(result, Err(DataRebindError::Conflict(_))));
        drop(result);
        drop((
            vectors,
            scalar,
            product,
            scalar_codes,
            binary_codes,
            product_codes,
            samples,
            trained,
        ));
        for ((index, pin), expected) in indexes.iter().zip(&pins).zip(before) {
            let VectorIndexKind::Quantized(index) = index else {
                panic!("Quantized fixture")
            };
            let VectorMaintenancePin::Quantized(pin) = pin else {
                panic!("Quantized pin")
            };
            assert!(index.vectors.try_write().is_some());
            assert!(index.scalar_quantizer.try_write().is_some());
            assert!(index.product_quantizer.try_write().is_some());
            assert!(index.scalar_vectors.try_write().is_some());
            assert!(index.binary_vectors.try_write().is_some());
            assert!(index.product_codes.try_write().is_some());
            assert!(index.training_samples.try_write().is_some());
            assert!(index.quantizer_trained.try_write().is_some());
            assert!(pin.topology.try_exclude_readers().is_some());
            assert_exact(&index.snapshot_exact().unwrap(), &expected);
        }
    }
}

#[test]
fn mixed_forgotten_reader_ready_and_installed_fences_retire_after_pins_and_outer_gate() {
    use crate::index::vector::maintenance::{
        VectorMaintenanceSlot, VectorMaintenanceSlots, prepare_vector_maintenance_slots,
    };
    use crate::index::vector::{HnswIndex, VectorIndexKind};

    for kind in kinds() {
        for stage in 0..3 {
            let plain =
                HnswIndex::with_seed(HnswConfig::new(4, DistanceMetric::Euclidean).with_m(4), 41);
            let image: HashMap<_, _> = (0..3).map(|id| (NodeId::new(id), vector(id))).collect();
            for id in 0..3 {
                plain.insert(NodeId::new(id), &vector(id), &|id| image.get(&id).cloned());
            }
            let indexes = [
                VectorIndexKind::Hnsw(plain),
                VectorIndexKind::Quantized(fixture(kind, 3)),
            ];
            let probes = std::sync::atomic::AtomicUsize::new(0);
            let gate = parking_lot::Mutex::new(());
            let mut slots = VectorMaintenanceSlots::new(
                indexes
                    .iter()
                    .map(|index| {
                        let mut slot =
                            VectorMaintenanceSlot::new(index, vec![(NodeId::new(0), None)]);
                        slot.before_retire_for_test(|| {
                            assert!(gate.try_lock().is_some());
                            for target in &indexes {
                                // Acquiring a fresh pin also proves all old alias pins
                                // retired before the first candidate payload does.
                                match target {
                                    VectorIndexKind::Hnsw(index) => {
                                        let pin = index.pin_maintenance().unwrap();
                                        assert!(pin.state_guards_available_for_test());
                                        assert!(pin.try_exclude_readers().is_some());
                                    }
                                    VectorIndexKind::Quantized(index) => {
                                        let pin = index.pin_maintenance().unwrap();
                                        assert!(pin.topology.state_guards_available_for_test());
                                        assert!(pin.topology.try_exclude_readers().is_some());
                                        assert!(index.vectors.try_write().is_some());
                                        assert!(index.scalar_quantizer.try_write().is_some());
                                        assert!(index.product_quantizer.try_write().is_some());
                                        assert!(index.scalar_vectors.try_write().is_some());
                                        assert!(index.binary_vectors.try_write().is_some());
                                        assert!(index.product_codes.try_write().is_some());
                                        assert!(index.training_samples.try_write().is_some());
                                        assert!(index.quantizer_trained.try_write().is_some());
                                    }
                                }
                            }
                            probes.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        });
                        slot
                    })
                    .collect(),
            );
            let mut pins = Vec::with_capacity(indexes.len());
            let outer = gate.lock();
            pins.extend(indexes.iter().map(|index| index.pin_maintenance().unwrap()));
            let readers = prepare_vector_maintenance_slots(&mut slots, &pins, &|_, id| {
                image.get(&id).cloned()
            })
            .unwrap()
            .exclude_readers()
            .unwrap();
            match stage {
                0 => std::mem::forget(readers),
                1 => std::mem::forget(readers.rebind().unwrap()),
                _ => std::mem::forget(readers.rebind().unwrap().install()),
            }
            pins.clear();
            drop(outer);
            drop(pins);
            drop(slots);
            assert_eq!(probes.load(std::sync::atomic::Ordering::Relaxed), 2);
            for index in &indexes {
                assert_eq!(index.contains(NodeId::new(0)), stage != 2);
            }
        }
    }
}

#[test]
fn all_families_match_ordinary_reference_below_at_and_across_training() {
    for kind in kinds() {
        for (count, upserts) in [(6, 1), (9, 1), (8, 4), (12, 4)] {
            for mmap in [false, true] {
                let index = if mmap {
                    fixture(kind, count)
                } else {
                    fixture(kind, count).without_rescore()
                };
                let reference = if mmap {
                    fixture(kind, count)
                } else {
                    fixture(kind, count).without_rescore()
                };
                index.remove(NodeId::new(2));
                reference.remove(NodeId::new(2));
                if mmap {
                    let (entry, level, nodes) = index.snapshot_topology();
                    index.hnsw.adopt_mmap_topology(
                        MmapTopology::from_bytes(Bytes::from(serialize_topology(
                            entry, level, &nodes,
                        )))
                        .unwrap(),
                    );
                }
                let mut operations = vec![(NodeId::new(0), Some(vector(3)))];
                if upserts == 4 {
                    operations.extend([
                        (NodeId::new(2), Some(vector(7))),
                        (NodeId::new(20), Some(vector(20))),
                        (NodeId::new(21), Some(vector(1000))),
                    ]);
                }
                operations.extend([
                    (NodeId::new(1), Some(vector(100))),
                    (NodeId::new(1), None),
                    (NodeId::new(1), Some(vector(200))),
                ]);
                apply_reference(&reference, &operations);
                let mut workspace = QuantizedMaintenanceWorkspace::new(operations);
                install_checked(&index, &mut workspace);
                assert_exact(
                    &index.snapshot_exact().unwrap(),
                    &reference.snapshot_exact().unwrap(),
                );
                assert!(!index.contains(NodeId::new(1)));
                assert!(
                    index.get(NodeId::new(1)).is_some(),
                    "soft delete retains routing vector"
                );
                assert_eq!(
                    index.get(NodeId::new(1)).unwrap().as_ref(),
                    vector(1).as_ref()
                );
                assert!(workspace.auxiliary.vectors.is_empty());
                assert!(workspace.auxiliary.vectors.capacity() > 0);
                assert!(!workspace.auxiliary.retired_vectors.is_empty());
                if workspace.auxiliary.first_training {
                    assert_eq!(
                        workspace.auxiliary.retired_samples.len(),
                        usize::try_from(count).unwrap()
                    );
                    assert!(index.training_samples.read().is_empty());
                }
            }
        }
    }
}

#[test]
fn repeated_sample_history_uses_only_crossing_prefix_and_trained_outliers_do_not_recalibrate() {
    for kind in [
        QuantizationType::Scalar,
        QuantizationType::Product { num_subvectors: 2 },
    ] {
        let index = fixture(kind, 6);
        let reference = fixture(kind, 6);
        for value in [50, 70, 90] {
            index.insert(NodeId::new(0), &vector(value));
            reference.insert(NodeId::new(0), &vector(value));
        }
        assert_eq!(index.training_samples.read().len(), 9);
        assert_eq!(index.vectors.read().len(), 6);
        let historical = index.training_samples.read().clone();
        let operations = vec![
            (NodeId::new(20), Some(vector(10000))),
            (NodeId::new(0), Some(vector(17))),
            (NodeId::new(0), Some(vector(18))),
        ];
        let mut workspace = QuantizedMaintenanceWorkspace::new(operations.clone());
        apply_reference(&reference, &operations);
        install_checked(&index, &mut workspace);
        assert_eq!(workspace.auxiliary.samples.len(), 1);
        assert_eq!(workspace.auxiliary.samples[0].as_ref(), vector(18).as_ref());
        assert_eq!(workspace.auxiliary.retired_samples, historical);
        assert_exact(
            &index.snapshot_exact().unwrap(),
            &reference.snapshot_exact().unwrap(),
        );
        let trained = index.snapshot_exact().unwrap();
        let operations = vec![(NodeId::new(21), Some(vector(100000)))];
        let mut outlier = QuantizedMaintenanceWorkspace::new(operations.clone());
        apply_reference(&reference, &operations);
        install_checked(&index, &mut outlier);
        let after = index.snapshot_exact().unwrap();
        assert_exact(&after, &reference.snapshot_exact().unwrap());
        assert_eq!(
            format!("{:?}", trained.scalar_quantizer),
            format!("{:?}", after.scalar_quantizer)
        );
        assert_eq!(
            format!("{:?}", trained.product_quantizer),
            format!("{:?}", after.product_quantizer)
        );
        assert_eq!(outlier.auxiliary.retired_vectors.len(), 0);
        assert_eq!(outlier.auxiliary.retired_scalar_vectors.len(), 0);
        assert_eq!(outlier.auxiliary.retired_product_codes.len(), 0);
    }
}

#[test]
fn sparse_full_capacity_replacement_and_soft_delete_resurrection_have_zero_final_traffic() {
    for kind in kinds() {
        let index = fixture(kind, 3);
        if matches!(
            kind,
            QuantizationType::Scalar | QuantizationType::Product { .. }
        ) {
            for value in 3..10 {
                index.insert(NodeId::new(0), &vector(value));
            }
            assert!(*index.quantizer_trained.read());
        }
        index.vectors.write().shrink_to_fit();
        assert_eq!(index.vectors.read().len(), index.vectors.read().capacity());
        index.binary_vectors.write().shrink_to_fit();
        index.scalar_vectors.write().shrink_to_fit();
        index.product_codes.write().shrink_to_fit();
        for (len, capacity) in [
            {
                let codes = index.binary_vectors.read();
                (codes.len(), codes.capacity())
            },
            {
                let codes = index.scalar_vectors.read();
                (codes.len(), codes.capacity())
            },
            {
                let codes = index.product_codes.read();
                (codes.len(), codes.capacity())
            },
        ] {
            assert_eq!(len, capacity);
        }
        let mut replacement =
            QuantizedMaintenanceWorkspace::new(vec![(NodeId::new(0), Some(vector(5)))]);
        install_checked(&index, &mut replacement);
        assert_eq!(replacement.auxiliary.retired_vectors.len(), 1);
        let before = index.get(NodeId::new(0)).unwrap();
        let mut deletion = QuantizedMaintenanceWorkspace::new(vec![(NodeId::new(0), None)]);
        install_checked(&index, &mut deletion);
        assert!(!index.contains(NodeId::new(0)));
        assert!(Arc::ptr_eq(&before, &index.get(NodeId::new(0)).unwrap()));
        let mut resurrection =
            QuantizedMaintenanceWorkspace::new(vec![(NodeId::new(0), Some(vector(8)))]);
        install_checked(&index, &mut resurrection);
        assert!(index.contains(NodeId::new(0)));
        assert_eq!(
            index.get(NodeId::new(0)).unwrap().as_ref(),
            vector(8).as_ref()
        );
    }
}

#[test]
fn empty_indexes_install_first_vectors_without_final_allocator_traffic() {
    for kind in kinds() {
        let index = fixture(kind, 0);
        let reference = fixture(kind, 0);
        let operations = vec![(NodeId::new(0), Some(vector(1)))];
        let mut workspace = QuantizedMaintenanceWorkspace::new(operations.clone());
        apply_reference(&reference, &operations);
        install_checked(&index, &mut workspace);
        assert_exact(
            &index.snapshot_exact().unwrap(),
            &reference.snapshot_exact().unwrap(),
        );
    }
}

#[test]
fn one_pin_claims_complete_preparation_and_blocks_same_thread_aliases() {
    for kind in kinds() {
        let index = fixture(kind, 8);
        let before = index.snapshot_exact().unwrap();
        let mut first = QuantizedMaintenanceWorkspace::new(vec![(NodeId::new(0), None)]);
        let mut second = QuantizedMaintenanceWorkspace::new(vec![(NodeId::new(1), None)]);
        let pin = index.pin_maintenance().unwrap();
        {
            let _released = pin.prepare(&mut first).unwrap();
            assert!(pin.prepare(&mut second).is_err());
            index.insert(NodeId::new(20), &vector(20));
            assert!(!index.remove(NodeId::new(1)));
            assert!(index.pin_maintenance().is_err());
            assert_exact(&index.snapshot_exact().unwrap(), &before);
        }
        drop(pin);
        install_checked(&index, &mut second);
        assert!(index.contains(NodeId::new(0)));
        assert!(!index.contains(NodeId::new(1)));
    }
}

#[test]
fn late_topology_failure_retains_calibration_candidates_without_mutating_live_state() {
    for kind in [
        QuantizationType::Scalar,
        QuantizationType::Product { num_subvectors: 2 },
    ] {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean)
            .with_m(4)
            .with_max_elements(10);
        let index = QuantizedHnswIndex::with_seed(config, kind, 41).with_training_threshold(10);
        for id in 0..9 {
            index.insert(NodeId::new(id), &vector(id));
        }
        let before = index.snapshot_exact().unwrap();
        let mut workspace = QuantizedMaintenanceWorkspace::new(vec![
            (NodeId::new(20), Some(vector(20))),
            (NodeId::new(21), Some(vector(21))),
        ]);
        let pin = index.pin_maintenance().unwrap();
        let error = pin.prepare(&mut workspace).err().unwrap();
        assert!(error.to_string().contains("maximum element count"));
        assert!(workspace.auxiliary.first_training);
        assert_eq!(
            workspace.auxiliary.scalar_vectors.len() + workspace.auxiliary.product_codes.len(),
            11
        );
        assert!(
            workspace.auxiliary.scalar_quantizer.is_some()
                || workspace.auxiliary.product_quantizer.is_some()
        );
        assert_exact(&index.snapshot_exact().unwrap(), &before);
    }
}

#[test]
fn every_auxiliary_rebind_conflict_is_static_and_allocation_free_with_partial_cleanup() {
    for contended in 0..8 {
        let index = fixture(QuantizationType::Binary, 3);
        let before = index.snapshot_exact().unwrap();
        let mut workspace =
            QuantizedMaintenanceWorkspace::new(vec![(NodeId::new(0), Some(vector(8)))]);
        let pin = index.pin_maintenance().unwrap();
        let released = pin.prepare(&mut workspace).unwrap();
        let readers = pin.exclude_readers();
        let vectors = (contended == 0).then(|| index.vectors.read());
        let scalar = (contended == 1).then(|| index.scalar_quantizer.read());
        let product = (contended == 2).then(|| index.product_quantizer.read());
        let scalar_codes = (contended == 3).then(|| index.scalar_vectors.read());
        let binary_codes = (contended == 4).then(|| index.binary_vectors.read());
        let product_codes = (contended == 5).then(|| index.product_codes.read());
        let samples = (contended == 6).then(|| index.training_samples.read());
        let trained = (contended == 7).then(|| index.quantizer_trained.read());
        crate::allocation_test::start();
        let result = released.rebind(&readers);
        let counts = crate::allocation_test::stop();
        assert_eq!(counts, crate::allocation_test::Counts::default());
        assert!(matches!(result, Err(DataRebindError::Conflict(_))));
        drop(result);
        if contended != 0 {
            assert!(index.vectors.try_write().is_some());
        }
        if contended != 1 {
            assert!(index.scalar_quantizer.try_write().is_some());
        }
        if contended != 2 {
            assert!(index.product_quantizer.try_write().is_some());
        }
        if contended != 3 {
            assert!(index.scalar_vectors.try_write().is_some());
        }
        if contended != 4 {
            assert!(index.binary_vectors.try_write().is_some());
        }
        if contended != 5 {
            assert!(index.product_codes.try_write().is_some());
        }
        if contended != 6 {
            assert!(index.training_samples.try_write().is_some());
        }
        if contended != 7 {
            assert!(index.quantizer_trained.try_write().is_some());
        }
        drop((
            vectors,
            scalar,
            product,
            scalar_codes,
            binary_codes,
            product_codes,
            samples,
            trained,
        ));
        // Also demonstrates that an auxiliary failure released inner topology.
        assert_exact(&index.snapshot_exact().unwrap(), &before);
    }
}

#[test]
fn invalid_dimensions_layout_and_calibration_numeric_range_are_premarker_errors() {
    for kind in kinds() {
        let index = fixture(kind, 3);
        let before = index.snapshot_exact().unwrap();
        for value in [
            Arc::from([1.0_f32].as_slice()),
            Arc::from([f32::NAN, 0.0, 0.0, 0.0].as_slice()),
        ] {
            let mut workspace =
                QuantizedMaintenanceWorkspace::new(vec![(NodeId::new(4), Some(value))]);
            let pin = index.pin_maintenance().unwrap();
            assert!(pin.prepare(&mut workspace).is_err());
            assert_exact(&index.snapshot_exact().unwrap(), &before);
        }
    }
    for partitions in [0, 3] {
        let index = QuantizedHnswIndex::new(
            HnswConfig::new(4, DistanceMetric::Euclidean),
            QuantizationType::Product {
                num_subvectors: partitions,
            },
        );
        let mut workspace =
            QuantizedMaintenanceWorkspace::new(vec![(NodeId::new(0), Some(vector(0)))]);
        let pin = index.pin_maintenance().unwrap();
        assert!(
            pin.prepare(&mut workspace)
                .err()
                .unwrap()
                .to_string()
                .contains("partitions")
        );
        assert!(index.is_empty());
        assert!(index.vectors.read().is_empty());
    }
    for kind in [
        QuantizationType::Scalar,
        QuantizationType::Product { num_subvectors: 2 },
    ] {
        let index = fixture(kind, 9);
        // Scalar min/max subtraction and PQ centroid accumulation must not
        // publish a non-finite trained model from individually finite samples.
        index
            .training_samples
            .write()
            .iter_mut()
            .enumerate()
            .for_each(|(position, sample)| {
                *sample = Arc::from(
                    [if kind == QuantizationType::Scalar && position % 2 == 0 {
                        -f32::MAX
                    } else {
                        f32::MAX
                    }; 4],
                );
            });
        let before = index.snapshot_exact().unwrap();
        let mut workspace = QuantizedMaintenanceWorkspace::new(vec![(
            NodeId::new(20),
            Some(Arc::from([f32::MAX; 4])),
        )]);
        let pin = index.pin_maintenance().unwrap();
        let error = pin.prepare(&mut workspace).err().unwrap();
        assert!(error.to_string().contains("numeric range"));
        assert_exact(&index.snapshot_exact().unwrap(), &before);
    }
}

#[test]
fn external_graph_callbacks_are_parallel_and_drained_by_quantized_maintenance() {
    let index = Arc::new(fixture(QuantizationType::Binary, 4));
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let (entered_tx, entered_rx) = mpsc::channel();
    let mut workers = Vec::new();
    for _ in 0..2 {
        let index = Arc::clone(&index);
        let release = Arc::clone(&release);
        let entered = entered_tx.clone();
        workers.push(std::thread::spawn(move || {
            let first = std::sync::atomic::AtomicBool::new(true);
            let accessor = |id| {
                if first.swap(false, std::sync::atomic::Ordering::Relaxed) {
                    entered.send(()).unwrap();
                    let (lock, signal) = &*release;
                    let guard = lock.lock().unwrap();
                    let _wait = signal
                        .wait_timeout_while(guard, Duration::from_secs(5), |released| !*released)
                        .unwrap();
                }
                index.get(id)
            };
            index.search_visible(&[0.0, 0.0, 1.0, 0.5], 2, 8, &|_| true, &accessor)
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
    let (lock, signal) = &*release;
    *lock.lock().unwrap() = true;
    signal.notify_all();
    for worker in workers {
        assert!(!worker.join().unwrap().is_empty());
    }
    maintenance.join().unwrap();
    assert!(first && second && waiting);
    assert!(
        !excluded_while_reading,
        "external graph callbacks remain admitted after inner HNSW search"
    );
    assert!(exclusive_rx.recv_timeout(Duration::from_secs(3)).is_ok());
}

#[test]
fn binary_search_is_admitted_before_auxiliary_codes_and_inner_topology() {
    let index = Arc::new(fixture(QuantizationType::Binary, 4));
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let (entered_tx, entered_rx) = mpsc::channel();
    let worker_index = Arc::clone(&index);
    let worker_release = Arc::clone(&release);
    let worker = std::thread::spawn(move || {
        BINARY_READ_SEAM.with(|seam| {
            *seam.borrow_mut() = Some(Box::new(move || {
                entered_tx.send(()).unwrap();
                let (lock, signal) = &*worker_release;
                let guard = lock.lock().unwrap();
                let _wait = signal
                    .wait_timeout_while(guard, Duration::from_secs(5), |released| !*released)
                    .unwrap();
            }));
        });
        let results = worker_index.search(&[0.0, 0.0, 1.0, 0.5], 2);
        BINARY_READ_SEAM.with(|seam| {
            *seam.borrow_mut() = None;
        });
        results
    });
    let entered = entered_rx.recv_timeout(Duration::from_secs(3)).is_ok();
    let pin = index.pin_maintenance().unwrap();
    // Nonblocking probe at an explicit public-query seam: no scheduler timing
    // inference, and inner HNSW admission has not been entered yet.
    let wrongly_excluded = pin.topology.try_exclude_readers().is_some();
    let (lock, signal) = &*release;
    *lock.lock().unwrap() = true;
    signal.notify_all();
    assert!(!worker.join().unwrap().is_empty());
    assert!(entered);
    assert!(
        !wrongly_excluded,
        "quantized code access must already hold outer shared admission"
    );
}
