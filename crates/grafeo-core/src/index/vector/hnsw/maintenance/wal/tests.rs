use super::{LIMIT, MAGIC, Wire, decode};
use crate::index::vector::hnsw::maintenance::HnswMaintenanceWorkspace;
use crate::index::vector::hnsw::{HnswExactState, TopologyBackend};
use crate::index::vector::paged_topology::{MmapTopology, serialize_topology};
use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorAccessor};
use bytes::Bytes;
use grafeo_common::types::NodeId;
use grafeo_common::utils::error::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

fn fixture(count: u16) -> (HnswIndex, HashMap<NodeId, Arc<[f32]>>) {
    let index = HnswIndex::with_seed(HnswConfig::new(4, DistanceMetric::Euclidean).with_m(4), 41);
    let vectors: HashMap<NodeId, Arc<[f32]>> = (0..count)
        .map(|id| {
            (
                NodeId::new(u64::from(id)),
                Arc::from([f32::from(id), f32::from(id % 7), 1.0, 0.5]),
            )
        })
        .collect();
    let accessor = |id| vectors.get(&id).cloned();
    for id in 0..count {
        let id = NodeId::new(u64::from(id));
        if let Some(vector) = vectors.get(&id) {
            index.insert(id, vector, &accessor);
        }
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

fn install(
    index: &HnswIndex,
    workspace: &mut HnswMaintenanceWorkspace,
    accessor: &impl VectorAccessor,
) -> Result<Vec<u8>> {
    workspace.capture_wal()?;
    let pin = index.pin_maintenance()?;
    let released = pin.prepare(workspace, accessor)?.release();
    let bytes = released.workspace.encode_wal_postimage()?;
    let readers = pin.exclude_readers();
    let ready = released
        .rebind(&readers)
        .map_err(|error| error.into_error())?;
    crate::allocation_test::start();
    drop(ready.install());
    let allocations = crate::allocation_test::stop();
    assert_eq!(allocations, crate::allocation_test::Counts::default());
    assert!(workspace.encode_wal_postimage().is_err());
    Ok(bytes)
}

fn raw(wire: &Wire) -> std::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = MAGIC.to_vec();
    bytes.extend(bincode::serde::encode_to_vec(
        wire,
        bincode::config::standard(),
    )?);
    Ok(bytes)
}

#[test]
fn recorded_heap_and_mmap_match_exact_topology_and_next_rng_without_accessor_calls() -> TestResult {
    for mmap in [false, true] {
        let (live, mut vectors) = fixture(64);
        let (recovered, _) = fixture(64);
        assert!(live.remove(NodeId::new(7)));
        assert!(recovered.remove(NodeId::new(7)));
        if mmap {
            let (entry, level, nodes) = recovered.snapshot_topology();
            let topology =
                MmapTopology::from_bytes(Bytes::from(serialize_topology(entry, level, &nodes)))?;
            recovered.adopt_mmap_topology(topology);
        }
        let replacement: Arc<[f32]> = Arc::from([0.25, 1.5, 3.0, 5.0]);
        vectors.insert(NodeId::new(1), Arc::clone(&replacement));
        let created: Arc<[f32]> = Arc::from([99.0, 4.0, 1.0, 0.5]);
        vectors.insert(NodeId::new(70), Arc::clone(&created));
        let resurrected = vectors
            .get(&NodeId::new(7))
            .cloned()
            .ok_or("fixture vector missing")?;
        let mut workspace = HnswMaintenanceWorkspace::new(vec![
            (NodeId::new(1), Some(replacement)),
            (NodeId::new(7), Some(resurrected)),
            (NodeId::new(70), Some(created)),
            (NodeId::new(63), None),
        ]);
        let accessor = |id| vectors.get(&id).cloned();
        let bytes = install(&live, &mut workspace, &accessor)?;
        let wire = decode(&bytes)?;
        assert!(
            wire.nodes.len() < 64,
            "only touched topology may be recorded"
        );
        let calls = AtomicUsize::new(0);
        let no_vectors = |_id| -> Option<Arc<[f32]>> {
            calls.fetch_add(1, Ordering::Relaxed);
            None
        };
        let mut recorded = HnswMaintenanceWorkspace::from_recorded(bytes.clone());
        assert_eq!(install(&recovered, &mut recorded, &no_vectors)?, bytes);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_exact(recovered.snapshot_exact()?, live.snapshot_exact()?);
        if mmap {
            let topology = recovered.nodes.read();
            match &*topology {
                TopologyBackend::Mmap {
                    base,
                    overrides,
                    additional_nodes,
                } => {
                    assert_eq!(base.len(), 64);
                    assert_eq!(*additional_nodes, 1);
                    assert!(overrides.len() < base.len());
                }
                TopologyBackend::Heap(_) => {
                    return Err("recorded install replaced mmap backing".into());
                }
            }
        }
        let next: Arc<[f32]> = Arc::from([0.3, 2.1, 7.0, 1.0]);
        vectors.insert(NodeId::new(80), Arc::clone(&next));
        let accessor = |id| vectors.get(&id).cloned();
        let mut left =
            HnswMaintenanceWorkspace::new(vec![(NodeId::new(80), Some(Arc::clone(&next)))]);
        let mut right = HnswMaintenanceWorkspace::new(vec![(NodeId::new(80), Some(next))]);
        assert_eq!(
            install(&live, &mut left, &accessor)?,
            install(&recovered, &mut right, &accessor)?
        );
        assert_exact(recovered.snapshot_exact()?, live.snapshot_exact()?);
    }
    Ok(())
}

#[test]
fn recorded_forged_and_stale_preimages_refuse_before_installation() -> TestResult {
    let (live, mut vectors) = fixture(16);
    let next: Arc<[f32]> = Arc::from([99.0, 1.5, 2.0, 4.0]);
    vectors.insert(NodeId::new(20), Arc::clone(&next));
    let accessor = |id| vectors.get(&id).cloned();
    let mut source =
        HnswMaintenanceWorkspace::new(vec![(NodeId::new(20), Some(next)), (NodeId::new(3), None)]);
    let bytes = install(&live, &mut source, &accessor)?;
    for corruption in 0..10 {
        let mut wire = decode(&bytes)?;
        match corruption {
            0 => wire.configuration.ml_bits ^= 1,
            1 => wire.before.rng ^= 1,
            2 => wire.before.nodes += 1,
            3 => wire.after.rng ^= 1,
            4 => wire.after.entry_point = Some(NodeId::new(999)),
            5 => {
                let row = wire
                    .nodes
                    .iter_mut()
                    .find(|node| node.before.is_some())
                    .ok_or("no baseline row")?;
                row.before
                    .as_mut()
                    .and_then(|layers| layers.first_mut())
                    .ok_or("no baseline layer")?
                    .push(NodeId::new(999));
            }
            6 => {
                let row = wire.nodes.first_mut().ok_or("no candidate row")?;
                row.after
                    .first_mut()
                    .ok_or("no candidate layer")?
                    .push(row.id);
            }
            7 => wire.deleted.first_mut().ok_or("no deletion row")?.before = true,
            8 => {
                let operation = wire.operations.first().copied().ok_or("no operation")?;
                wire.operations.push(operation);
            }
            _ => wire.after.deleted += 1,
        }
        let (target, _) = fixture(16);
        let before = target.snapshot_exact()?;
        let mut workspace = HnswMaintenanceWorkspace::from_recorded(raw(&wire)?);
        let calls = AtomicUsize::new(0);
        let no_vectors = |_id| -> Option<Arc<[f32]>> {
            calls.fetch_add(1, Ordering::Relaxed);
            None
        };
        let pin = target.pin_maintenance()?;
        assert!(
            pin.prepare_workspace(&mut workspace, &no_vectors).is_err(),
            "corruption {corruption}"
        );
        drop(pin);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_exact(target.snapshot_exact()?, before);
    }
    let before = live.snapshot_exact()?;
    let mut replay = HnswMaintenanceWorkspace::from_recorded(bytes);
    let pin = live.pin_maintenance()?;
    assert!(
        pin.prepare_workspace(&mut replay, &accessor).is_err(),
        "duplicate replay must reject its stale baseline"
    );
    drop(pin);
    assert_exact(live.snapshot_exact()?, before);
    Ok(())
}

#[test]
fn recorded_codec_bounds_declared_allocations_and_rejects_versions_and_truncation() -> TestResult {
    let (index, vectors) = fixture(4);
    let before = index.snapshot_exact()?;
    let accessor = |id| vectors.get(&id).cloned();
    let mut workspace = HnswMaintenanceWorkspace::new(vec![(NodeId::new(1), None)]);
    workspace.capture_wal()?;
    let pin = index.pin_maintenance()?;
    pin.prepare_workspace(&mut workspace, &accessor)?;
    let bytes = workspace.encode_wal_postimage()?;
    for end in 0..bytes.len() {
        assert!(decode(&bytes[..end]).is_err());
    }
    let mut old = bytes.clone();
    old[3] = b'0';
    assert!(decode(&old).is_err());
    let mut trailing = bytes;
    trailing.push(0);
    assert!(decode(&trailing).is_err());
    assert!(decode(&vec![0; LIMIT + 1]).is_err());
    // Two wire bytes per operation fit comfortably while their declared
    // in-memory tuple allocation exceeds the same decoder's 16 MiB budget.
    workspace.final_presence.resize(
        LIMIT / std::mem::size_of::<(NodeId, bool)>() + 1,
        (NodeId::new(1), false),
    );
    assert!(
        workspace.encode_wal_postimage().is_err(),
        "encoder must reject self-unreadable allocation claims"
    );
    drop(pin);
    assert_exact(index.snapshot_exact()?, before);
    Ok(())
}

#[test]
fn surviving_delete_payload_is_independent_of_unrelated_topology_size() -> TestResult {
    let mut sizes = Vec::new();
    for count in [32, 2_048] {
        let (index, vectors) = fixture(count);
        let accessor = |id| vectors.get(&id).cloned();
        let mut workspace = HnswMaintenanceWorkspace::new(vec![(NodeId::new(1), None)]);
        let bytes = install(&index, &mut workspace, &accessor)?;
        let wire = decode(&bytes)?;
        assert_eq!(wire.operations.len(), 1);
        assert_eq!(wire.deleted.len(), 1);
        assert!(wire.nodes.is_empty());
        assert!(bytes.len() < 256);
        sizes.push(bytes.len());
    }
    assert!(
        sizes[0].abs_diff(sizes[1]) < 32,
        "only baseline counters grow: {sizes:?}"
    );
    Ok(())
}

#[test]
fn memory_maintenance_does_not_allocate_wal_baseline_or_presence() -> TestResult {
    let (index, mut vectors) = fixture(16);
    let replacement: Arc<[f32]> = Arc::from([0.2, 1.5, 3.0, 8.0]);
    vectors.insert(NodeId::new(1), Arc::clone(&replacement));
    let accessor = |id| vectors.get(&id).cloned();
    let before = index.snapshot_exact()?;
    let mut workspace = HnswMaintenanceWorkspace::new(vec![(NodeId::new(1), Some(replacement))]);
    let pin = index.pin_maintenance()?;
    let released = pin.prepare(&mut workspace, &accessor)?.release();
    assert!(
        !released.workspace.nodes.is_empty(),
        "real sparse maintenance was prepared"
    );
    assert!(released.workspace.wal_baseline.is_none());
    assert_eq!(released.workspace.final_presence.capacity(), 0);
    assert_eq!(released.workspace.prepared_final_presence().count(), 0);
    assert!(released.workspace.encode_wal_postimage().is_err());
    assert!(
        released.workspace.capture_wal().is_err(),
        "capture cannot be enabled after preparation"
    );
    let readers = pin.exclude_readers();
    drop(
        released
            .rebind(&readers)
            .map_err(|error| error.into_error())?
            .install(),
    );
    drop(readers);
    drop(pin);
    assert!(workspace.wal_baseline.is_none());
    assert_eq!(workspace.final_presence.capacity(), 0);
    assert_ne!(
        index.snapshot_exact()?.rng_state,
        before.rng_state,
        "ordinary maintenance still installs"
    );
    Ok(())
}
