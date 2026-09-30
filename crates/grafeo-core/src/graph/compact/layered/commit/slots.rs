//! Outer-owned Layered guard slots for the aggregate's cold-delete companion.

use super::{
    DataRebindError, LayeredCommitPin, LayeredCommitWorkspace, LayeredStore, PinnedLpgTransition,
    RwLockWriteGuard, TombstoneGuards,
};
#[cfg(test)]
use crate::graph::lpg::LpgStore;
#[cfg(test)]
use grafeo_common::types::{EdgeId, EpochId, NodeId, TransactionId};
#[cfg(test)]
use std::sync::atomic::Ordering;

/// The target is independently anchored. Neither raw guard borrows a local
/// merge pin, LPG transition, or another field of this slot.
pub(crate) struct LayeredCommitSlot<'store> {
    guards: Option<TombstoneGuards<'store>>,
    publication: Option<RwLockWriteGuard<'store, ()>>,
    store: &'store LayeredStore,
    workspace: LayeredCommitWorkspace,
    transition_index: usize,
    #[cfg(test)]
    before_retire: Option<Box<dyn FnOnce() + 'store>>,
}

impl<'store> LayeredCommitSlot<'store> {
    pub(crate) fn new(store: &'store LayeredStore, workspace: LayeredCommitWorkspace) -> Self {
        Self {
            guards: None,
            publication: None,
            store,
            workspace,
            transition_index: 0,
            #[cfg(test)]
            before_retire: None,
        }
    }
}

impl Drop for LayeredCommitSlot<'_> {
    fn drop(&mut self) {
        drop(self.guards.take());
        drop(self.publication.take());
        #[cfg(test)]
        if let Some(probe) = self.before_retire.take() {
            probe();
        }
    }
}

/// Declare before all enclosing publication/authority owners. Its destructor
/// drains ALL map writers, then ALL generation writers, before ANY slot payload
/// or buffer retires, including after a deliberately forgotten batch proof.
pub(crate) struct LayeredCommitSlots<'store> {
    slots: Vec<LayeredCommitSlot<'store>>,
    phase: Phase,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Fresh,
    Preparing,
    Released,
    Readers,
    Ready,
    Installed,
}

impl<'store> LayeredCommitSlots<'store> {
    pub(crate) fn new(slots: Vec<LayeredCommitSlot<'store>>) -> Self {
        Self {
            slots,
            phase: Phase::Fresh,
        }
    }

    /// The aggregate's outer Drop calls this for every family before allowing
    /// any family payload to retire. It releases guards only, never authority,
    /// candidates, retired queues, or the backing slot allocation.
    pub(crate) fn release_guards(&mut self) {
        self.release_maps();
        for slot in self.slots.iter_mut().rev() {
            drop(slot.publication.take());
        }
    }

    fn release_maps(&mut self) {
        for slot in self.slots.iter_mut().rev() {
            drop(slot.guards.take());
        }
    }
}

impl Drop for LayeredCommitSlots<'_> {
    fn drop(&mut self) {
        self.release_guards();
    }
}

struct Batch<'store, 'workspace, 'authority> {
    slots: &'workspace mut LayeredCommitSlots<'store>,
    pins: &'authority [LayeredCommitPin<'store>],
    transitions: &'authority [PinnedLpgTransition<'store>],
}

impl Drop for Batch<'_, '_, '_> {
    fn drop(&mut self) {
        self.slots.release_guards();
    }
}

#[must_use]
pub(crate) struct ReleasedLayeredCommitSlots<'store, 'workspace, 'authority> {
    batch: Batch<'store, 'workspace, 'authority>,
}

#[must_use]
pub(crate) struct LayeredCommitReaderFence<'store, 'workspace, 'authority> {
    batch: Batch<'store, 'workspace, 'authority>,
}

#[must_use]
pub(crate) struct PreparedLayeredCommitSlots<'store, 'workspace, 'authority> {
    batch: Batch<'store, 'workspace, 'authority>,
}

#[must_use]
pub(crate) struct InstalledLayeredCommitSlots<'store, 'workspace, 'authority> {
    _batch: Batch<'store, 'workspace, 'authority>,
}

/// Pins must correspond to the frozen Layered slot sequence and have been
/// acquired before topology/LPG authority. The full transition slice may also
/// contain unrelated native targets. Scalar transition coordinates are resolved
/// during preparation, never by allocating a vector of borrowed proofs.
pub(crate) fn prepare_layered_commit_slots<'store, 'workspace, 'authority>(
    slots: &'workspace mut LayeredCommitSlots<'store>,
    pins: &'authority [LayeredCommitPin<'store>],
    transitions: &'authority [PinnedLpgTransition<'store>],
) -> Result<ReleasedLayeredCommitSlots<'store, 'workspace, 'authority>, DataRebindError> {
    if slots.phase != Phase::Fresh {
        return Err(DataRebindError::new("Layered slot preparation is one-shot"));
    }
    slots.phase = Phase::Preparing;
    let batch = Batch {
        slots,
        pins,
        transitions,
    };
    if batch.slots.slots.len() != pins.len() {
        return Err(DataRebindError::new(
            "Layered slots and merge pins differ in length",
        ));
    }
    // Validate the complete pairing before claiming any pin or workspace.
    for (index, (slot, pin)) in batch.slots.slots.iter().zip(pins).enumerate() {
        if !std::ptr::eq(slot.store, pin.store)
            || batch.slots.slots[..index]
                .iter()
                .any(|previous| std::ptr::eq(previous.store, slot.store))
        {
            return Err(DataRebindError::new(
                "Layered slots repeat or mismatch a merge-pin target",
            ));
        }
    }
    for (slot, pin) in batch.slots.slots.iter_mut().zip(pins) {
        let mut coordinate = None;
        for (index, transition) in transitions.iter().enumerate() {
            if transition.pins_store(pin.overlay) {
                if coordinate.is_some() {
                    return Err(DataRebindError::new(
                        "Layered overlay has repeated transition authority",
                    ));
                }
                coordinate = Some(index);
            }
        }
        slot.transition_index = coordinate
            .ok_or_else(|| DataRebindError::new("Layered overlay lacks its exact transition"))?;
    }
    for (slot, pin) in batch.slots.slots.iter_mut().zip(pins) {
        pin.prepare_workspace(&transitions[slot.transition_index], &mut slot.workspace)?;
    }
    batch.slots.phase = Phase::Released;
    Ok(ReleasedLayeredCommitSlots { batch })
}

impl<'store, 'workspace, 'authority> ReleasedLayeredCommitSlots<'store, 'workspace, 'authority> {
    /// Complete every generation exclusion after all Vector reader exclusions,
    /// before any component's final map/entity writers. Acquisition is try-only.
    pub(crate) fn exclude_readers(
        self,
    ) -> Result<LayeredCommitReaderFence<'store, 'workspace, 'authority>, DataRebindError> {
        for (slot, pin) in self.batch.slots.slots.iter_mut().zip(self.batch.pins) {
            slot.publication =
                Some(pin.publication_writer(&self.batch.transitions[slot.transition_index])?);
        }
        self.batch.slots.phase = Phase::Readers;
        Ok(LayeredCommitReaderFence { batch: self.batch })
    }
}

impl<'store, 'workspace, 'authority> LayeredCommitReaderFence<'store, 'workspace, 'authority> {
    /// A late rejection drains every previously acquired target's guards. The
    /// complete aggregate must drain its other final fences before owning Error.
    pub(crate) fn rebind(
        self,
    ) -> Result<PreparedLayeredCommitSlots<'store, 'workspace, 'authority>, DataRebindError> {
        for slot in &mut self.batch.slots.slots {
            let guards = TombstoneGuards::try_acquire(slot.store)?;
            guards.validate(&slot.workspace)?;
            slot.guards = Some(guards);
        }
        self.batch.slots.phase = Phase::Ready;
        Ok(PreparedLayeredCommitSlots { batch: self.batch })
    }
}

impl<'store, 'workspace, 'authority> PreparedLayeredCommitSlots<'store, 'workspace, 'authority> {
    /// Installation uses each intrinsically paired target/guard/workspace. No
    /// authority resolution, allocation, payload retirement, or callback remains.
    pub(crate) fn install(self) -> InstalledLayeredCommitSlots<'store, 'workspace, 'authority> {
        for slot in &mut self.batch.slots.slots {
            if let Some(guards) = &mut slot.guards {
                guards.install(slot.store, &mut slot.workspace);
            }
        }
        self.batch.slots.phase = Phase::Installed;
        InstalledLayeredCommitSlots { _batch: self.batch }
    }

    /// Premarker map-writer release retains every generation writer and the
    /// original pin/transition loans for another final binding attempt.
    #[cfg(test)]
    pub(crate) fn release(self) -> LayeredCommitReaderFence<'store, 'workspace, 'authority> {
        self.batch.slots.release_maps();
        self.batch.slots.phase = Phase::Readers;
        LayeredCommitReaderFence { batch: self.batch }
    }
}

#[cfg(test)]
mod tests;
