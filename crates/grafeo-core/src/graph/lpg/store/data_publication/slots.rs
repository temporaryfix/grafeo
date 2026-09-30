//! Outer-owned paired data slots for a continuously authorized batch.
//!
//! Only stack proofs borrow authority. Heap storage contains target-only raw
//! writers paired with the exact workspace that produced their postimages.

use super::PreparedNodeLabelImages;
use super::{
    DataCommitScope, DataRebindError, LpgStore, PinnedLpgTransition, Result, StoreDataGuards,
    StoreDataWorkspace, invalid,
};
#[cfg(test)]
use grafeo_common::types::{EdgeId, EpochId, NodeId, TransactionId};
#[cfg(test)]
use std::sync::atomic::Ordering;

pub(in crate::graph::lpg::store) struct StoreDataSlot<'store> {
    store: &'store LpgStore,
    #[cfg(test)]
    retirement_probe: Option<RetirementProbe<'store>>,
    workspace: StoreDataWorkspace,
    guards: Option<StoreDataGuards<'store>>,
    transition_index: usize,
}

impl<'store> StoreDataSlot<'store> {
    pub(in crate::graph::lpg::store) fn new(
        store: &'store LpgStore,
        workspace: StoreDataWorkspace,
    ) -> Self {
        Self {
            store,
            #[cfg(test)]
            retirement_probe: None,
            workspace,
            guards: None,
            transition_index: 0,
        }
    }
}

#[cfg(test)]
struct RetirementProbe<'store>(Box<dyn Fn() + 'store>);

#[cfg(test)]
impl Drop for RetirementProbe<'_> {
    fn drop(&mut self) {
        (self.0)();
    }
}

/// Declare before every enclosing publication and transition owner. Drop drains
/// ALL targets' guards before the first slot's candidates or retirees can drop,
/// including when a borrowed batch proof has been deliberately forgotten.
pub(in crate::graph::lpg::store) struct StoreDataSlots<'store> {
    slots: Vec<StoreDataSlot<'store>>,
    phase: Phase,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Fresh,
    Preparing,
    Released,
    Ready,
    Installed,
}

impl<'store> StoreDataSlots<'store> {
    pub(in crate::graph::lpg::store) fn new(slots: Vec<StoreDataSlot<'store>>) -> Self {
        Self {
            slots,
            phase: Phase::Fresh,
        }
    }

    pub(in crate::graph::lpg::store) fn release_guards(&mut self) {
        for slot in self.slots.iter_mut().rev() {
            drop(slot.guards.take());
        }
    }

    pub(in crate::graph::lpg::store) fn capture_label_images(&mut self) {
        for slot in &mut self.slots {
            slot.workspace.capture_label_images = true;
        }
    }
}

impl Drop for StoreDataSlots<'_> {
    fn drop(&mut self) {
        self.release_guards();
    }
}

struct Batch<'store, 'workspace, 'authority> {
    slots: &'workspace mut StoreDataSlots<'store>,
    transitions: &'authority [PinnedLpgTransition<'store>],
}

impl Drop for Batch<'_, '_, '_> {
    fn drop(&mut self) {
        self.slots.release_guards();
    }
}

#[must_use]
pub(in crate::graph::lpg::store) struct ReleasedStoreDataSlots<'store, 'workspace, 'authority> {
    batch: Batch<'store, 'workspace, 'authority>,
}

#[must_use]
pub(in crate::graph::lpg::store) struct PreparedStoreDataSlots<'store, 'workspace, 'authority> {
    batch: Batch<'store, 'workspace, 'authority>,
}

#[must_use]
pub(in crate::graph::lpg::store) struct InstalledStoreDataSlots<'store, 'workspace, 'authority> {
    _batch: Batch<'store, 'workspace, 'authority>,
}

/// Prepares every target before any final data writers are acquired. Transitions
/// must be the aggregate's continuously retained, parent-first authority slice.
/// Unrelated transitions are allowed, repeated slot or transition targets are
/// rejected. Slot ordering is normalized in place without auxiliary allocation.
pub(in crate::graph::lpg::store) fn prepare_store_data_slots<'store, 'workspace, 'authority>(
    slots: &'workspace mut StoreDataSlots<'store>,
    transitions: &'authority [PinnedLpgTransition<'store>],
) -> Result<ReleasedStoreDataSlots<'store, 'workspace, 'authority>> {
    if slots.phase != Phase::Fresh {
        return Err(invalid("data slot batch preparation was already attempted"));
    }
    slots.phase = Phase::Preparing;
    let batch = Batch { slots, transitions };
    batch
        .slots
        .slots
        .sort_unstable_by_key(|slot| std::ptr::from_ref(slot.store));
    if batch
        .slots
        .slots
        .windows(2)
        .any(|pair| std::ptr::eq(pair[0].store, pair[1].store))
    {
        return Err(invalid("data slot batch repeats a target store"));
    }
    // The sentinel cannot be a valid slice index, including the empty case.
    for slot in &mut batch.slots.slots {
        slot.transition_index = transitions.len();
    }
    for (index, transition) in transitions.iter().enumerate() {
        if let Ok(slot_index) = batch
            .slots
            .slots
            .binary_search_by_key(&std::ptr::from_ref(transition.store), |slot| {
                std::ptr::from_ref(slot.store)
            })
        {
            let slot = &mut batch.slots.slots[slot_index];
            if slot.transition_index != transitions.len() {
                return Err(invalid("data target has repeated transition authority"));
            }
            slot.transition_index = index;
        }
    }
    if batch
        .slots
        .slots
        .iter()
        .any(|slot| slot.transition_index == transitions.len())
    {
        return Err(invalid("data slot lacks its exact store transition"));
    }
    batch
        .slots
        .slots
        .sort_unstable_by_key(|slot| slot.transition_index);
    for slot in &mut batch.slots.slots {
        slot.store
            .prepare_buffered_data(&transitions[slot.transition_index], &mut slot.workspace)?;
    }
    batch.slots.phase = Phase::Released;
    Ok(ReleasedStoreDataSlots { batch })
}

impl<'store, 'workspace, 'authority> ReleasedStoreDataSlots<'store, 'workspace, 'authority> {
    pub(in crate::graph::lpg::store) fn label_images(
        &self,
    ) -> impl Iterator<Item = (&'store LpgStore, &[PreparedNodeLabelImages])> {
        self.batch.slots.slots.iter().filter_map(|slot| {
            let images = slot.workspace.label_images();
            (!images.is_empty()).then_some((slot.store, images))
        })
    }

    /// All-or-nothing final acquisition. On a later rejection the same batch
    /// owner drains every earlier target writer before the typed error returns.
    pub(in crate::graph::lpg::store) fn rebind(
        self,
    ) -> std::result::Result<PreparedStoreDataSlots<'store, 'workspace, 'authority>, DataRebindError>
    {
        for slot in &mut self.batch.slots.slots {
            let scope = DataCommitScope {
                transition: &self.batch.transitions[slot.transition_index],
            };
            slot.guards = Some(StoreDataGuards::rebind(&mut slot.workspace, &scope)?);
        }
        self.batch.slots.phase = Phase::Ready;
        Ok(PreparedStoreDataSlots { batch: self.batch })
    }
}

impl<'store, 'workspace, 'authority> PreparedStoreDataSlots<'store, 'workspace, 'authority> {
    /// Every slot is ready before the first installation. No proof-vector
    /// conversion, allocation, target discovery or guard release occurs here.
    pub(in crate::graph::lpg::store) fn install(
        self,
    ) -> InstalledStoreDataSlots<'store, 'workspace, 'authority> {
        for slot in &mut self.batch.slots.slots {
            let scope = DataCommitScope {
                transition: &self.batch.transitions[slot.transition_index],
            };
            // Only successful all-slot rebind constructs this proof; slots
            // and their private guard Options cannot be accessed between phases.
            if let Some(guards) = &mut slot.guards {
                guards.install(&mut slot.workspace, &scope);
            }
        }
        self.batch.slots.phase = Phase::Installed;
        InstalledStoreDataSlots { _batch: self.batch }
    }
}

#[cfg(test)]
mod tests;
