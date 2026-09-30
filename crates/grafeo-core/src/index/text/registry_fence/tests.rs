use super::*;
use crate::allocation_test as allocation;
use crate::index::text::{BM25Config, Tokenizer};
use std::sync::atomic::{AtomicUsize, Ordering};

fn anchor() -> IndexAnchor {
    Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())))
}

fn registrations(count: usize) -> Vec<RegisteredTextIndex> {
    (0..count)
        .map(|_| RegisteredTextIndex::new(anchor(), anchor()))
        .collect()
}

fn acquire_and_drop_without_traffic(workspace: &mut TextRegistryFenceWorkspace) {
    allocation::start();
    let result = TextRegistryBatchFence::try_acquire(std::hint::black_box(workspace));
    let acquired = result.is_ok();
    drop(result);
    let observed = allocation::stop();
    assert!(acquired);
    assert_eq!(observed, allocation::Counts::default());
}

#[test]
fn text_collective_fence_deduplicates_identities_and_has_zero_acquire_drop_traffic() {
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

    let first = anchor();
    let shared = anchor();
    let last = anchor();
    let entry = RegisteredTextIndex::new(Arc::clone(&first), Arc::clone(&shared));
    let entries = vec![
        entry.clone(),
        entry,
        RegisteredTextIndex::new(Arc::clone(&shared), Arc::clone(&last)),
        RegisteredTextIndex::new(Arc::clone(&shared), Arc::clone(&shared)),
    ];
    let mut workspace = TextRegistryFenceWorkspace::new();
    workspace.prepare(&entries).unwrap();
    assert_eq!(workspace.gates.len(), 2);
    assert_eq!(workspace.targets.len(), 1);
    let capacities = (
        workspace.gate_guards.capacity(),
        workspace.target_guards.capacity(),
    );
    {
        let fence = TextRegistryBatchFence::try_acquire(&mut workspace).unwrap();
        assert!(first.try_read().is_none());
        assert!(shared.try_read().is_none());
        assert!(last.try_read().is_none());
        drop(fence);
    }
    for _ in 0..3 {
        acquire_and_drop_without_traffic(&mut workspace);
        assert!(workspace.gate_guards.is_empty());
        assert!(workspace.target_guards.is_empty());
        assert_eq!(
            (
                workspace.gate_guards.capacity(),
                workspace.target_guards.capacity()
            ),
            capacities
        );
    }
    assert!(first.try_write().is_some());
    assert!(shared.try_write().is_some());
    assert!(last.try_write().is_some());
}

#[test]
fn text_collective_fence_partial_gate_and_target_conflicts_drain_without_traffic() {
    for target_conflict in [false, true] {
        let entries = registrations(3);
        let mut workspace = TextRegistryFenceWorkspace::new();
        workspace.prepare(&entries).unwrap();
        let blocked = Arc::clone(if target_conflict {
            workspace.targets.last().unwrap()
        } else {
            workspace.gates.last().unwrap()
        });
        let blocker = blocked.read_arc();
        let capacities = (
            workspace.gate_guards.capacity(),
            workspace.target_guards.capacity(),
        );
        allocation::start();
        let failure = TextRegistryBatchFence::try_acquire(&mut workspace).err();
        let expected = if target_conflict {
            matches!(
                failure,
                Some(DataRebindError::Conflict(
                    "Text registry batch concrete target is in use"
                ))
            )
        } else {
            matches!(
                failure,
                Some(DataRebindError::Conflict(
                    "Text registry batch caller gate is in use"
                ))
            )
        };
        let observed = allocation::stop();
        assert!(expected);
        assert_eq!(observed, allocation::Counts::default());
        assert!(matches!(
            failure.unwrap().into_error(),
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::WriteConflict(_)
            )
        ));
        assert!(workspace.gate_guards.is_empty());
        assert!(workspace.target_guards.is_empty());
        assert_eq!(
            (
                workspace.gate_guards.capacity(),
                workspace.target_guards.capacity()
            ),
            capacities
        );
        for gate in &workspace.gates {
            assert!(gate.try_read().is_some());
        }
        for target in &workspace.targets {
            assert!(target.try_read().is_some());
        }
        drop(blocker);
        acquire_and_drop_without_traffic(&mut workspace);
    }
}

struct DropProbe {
    outer: Arc<RwLock<()>>,
    drops: Arc<AtomicUsize>,
    premature: Arc<AtomicUsize>,
}

impl Tokenizer for DropProbe {
    fn tokenize(&self, _text: &str) -> Vec<String> {
        Vec::new()
    }
}

impl Drop for DropProbe {
    fn drop(&mut self) {
        if self.outer.try_read().is_none() {
            self.premature.fetch_add(1, Ordering::Relaxed);
        }
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

fn tracked_anchor(
    outer: &Arc<RwLock<()>>,
    drops: &Arc<AtomicUsize>,
    premature: &Arc<AtomicUsize>,
) -> IndexAnchor {
    Arc::new(RwLock::new(InvertedIndex::with_tokenizer(
        BM25Config::default(),
        Box::new(DropProbe {
            outer: Arc::clone(outer),
            drops: Arc::clone(drops),
            premature: Arc::clone(premature),
        }),
    )))
}

#[test]
fn text_collective_fence_retains_payloads_beyond_drop_conflict_and_unwind() {
    for path in 0..3 {
        let outer = Arc::new(RwLock::new(()));
        let drops = Arc::new(AtomicUsize::new(0));
        let premature = Arc::new(AtomicUsize::new(0));
        let mut workspace = TextRegistryFenceWorkspace::new();
        {
            let _outer = outer.write();
            let gate = tracked_anchor(&outer, &drops, &premature);
            let target = tracked_anchor(&outer, &drops, &premature);
            let entries = [RegisteredTextIndex::new(gate, target)];
            workspace.prepare(&entries).unwrap();
            drop(entries); // Only workspace anchors retain either payload now.
            match path {
                0 => acquire_and_drop_without_traffic(&mut workspace),
                1 => {
                    let blocker = workspace.targets[0].read_arc();
                    assert!(TextRegistryBatchFence::try_acquire(&mut workspace).is_err());
                    drop(blocker);
                }
                _ => {
                    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _fence = TextRegistryBatchFence::try_acquire(&mut workspace).unwrap();
                        panic!("test unwind while collective fence is retained");
                    }));
                    assert!(unwound.is_err());
                }
            }
            assert_eq!(drops.load(Ordering::Relaxed), 0);
            assert_eq!(premature.load(Ordering::Relaxed), 0);
            assert!(workspace.gate_guards.is_empty());
            assert!(workspace.target_guards.is_empty());
        }
        drop(workspace);
        assert_eq!(drops.load(Ordering::Relaxed), 2);
        assert_eq!(premature.load(Ordering::Relaxed), 0);
    }
}

struct ResetPreparationFailure;

impl Drop for ResetPreparationFailure {
    fn drop(&mut self) {
        FAIL_AFTER_ANCHORS.with(|fail| fail.set(false));
    }
}

#[test]
fn text_collective_fence_failed_preparation_keeps_anchors_outside_outer_gates() {
    let outer = Arc::new(RwLock::new(()));
    let drops = Arc::new(AtomicUsize::new(0));
    let premature = Arc::new(AtomicUsize::new(0));
    let mut workspace = TextRegistryFenceWorkspace::new();
    {
        let _outer = outer.write();
        let entries = [RegisteredTextIndex::new(
            tracked_anchor(&outer, &drops, &premature),
            tracked_anchor(&outer, &drops, &premature),
        )];
        {
            let _reset = ResetPreparationFailure;
            FAIL_AFTER_ANCHORS.with(|fail| fail.set(true));
            assert!(workspace.prepare(&entries).is_err());
        }
        drop(entries);
        allocation::start();
        let result = TextRegistryBatchFence::try_acquire(&mut workspace);
        let unprepared = matches!(
            result,
            Err(DataRebindError::Invalid(
                "Text fence workspace is not prepared"
            ))
        );
        drop(result);
        let observed = allocation::stop();
        assert!(unprepared);
        assert_eq!(observed, allocation::Counts::default());
        assert!(workspace.prepare(&[]).is_err());
        assert_eq!(drops.load(Ordering::Relaxed), 0);
    }
    drop(workspace);
    assert_eq!(drops.load(Ordering::Relaxed), 2);
    assert_eq!(premature.load(Ordering::Relaxed), 0);
}

#[test]
fn text_collective_fence_capacity_rejection_has_zero_traffic() {
    let entries = registrations(1);
    let mut workspace = TextRegistryFenceWorkspace::new();
    workspace.prepare(&entries).unwrap();
    // Test-only corruption: production preparation fixes these buffers before
    // any final writers exist and exposes no mutation of their capacities.
    workspace.target_guards = Vec::new();
    allocation::start();
    let result = TextRegistryBatchFence::try_acquire(&mut workspace);
    let capacity_lost = matches!(
        result,
        Err(DataRebindError::Invalid(
            "Text fence reserved guard capacity was lost"
        ))
    );
    drop(result);
    let observed = allocation::stop();
    assert!(capacity_lost);
    assert_eq!(observed, allocation::Counts::default());
    assert!(workspace.gate_guards.is_empty());
    assert!(workspace.target_guards.is_empty());
}

#[test]
fn text_collective_fence_empty_workspace_can_be_prepared_and_reacquired() {
    let mut workspace = TextRegistryFenceWorkspace::new();
    workspace.prepare(&[]).unwrap();
    acquire_and_drop_without_traffic(&mut workspace);
    acquire_and_drop_without_traffic(&mut workspace);
    assert!(workspace.prepare(&[]).is_err());
}

#[test]
fn text_collective_fence_forgotten_guard_cleanup_retains_all_buffers_and_anchors() {
    let entries = registrations(2);
    let mut workspace = TextRegistryFenceWorkspace::new();
    workspace.prepare(&entries).unwrap();
    drop(entries);
    let capacities = (
        workspace.gate_guards.capacity(),
        workspace.target_guards.capacity(),
    );
    let fence = TextRegistryBatchFence::try_acquire(&mut workspace).unwrap();
    std::mem::forget(fence);
    assert_eq!(workspace.gate_guards.len(), 2);
    assert_eq!(workspace.target_guards.len(), 2);
    allocation::start();
    workspace.release_guards();
    let observed = allocation::stop();
    assert_eq!(observed, allocation::Counts::default());
    assert!(workspace.gate_guards.is_empty());
    assert!(workspace.target_guards.is_empty());
    assert_eq!(workspace.gates.len(), 2);
    assert_eq!(workspace.targets.len(), 2);
    assert_eq!(
        (
            workspace.gate_guards.capacity(),
            workspace.target_guards.capacity()
        ),
        capacities
    );
    acquire_and_drop_without_traffic(&mut workspace);
}
