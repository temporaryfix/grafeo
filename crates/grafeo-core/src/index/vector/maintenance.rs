//! Typed surviving-index dispatch, preserving each concrete registered kind.
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use super::hnsw::maintenance::{HnswMaintenancePin, HnswMaintenanceSlot};
use super::quantized_hnsw::maintenance::{QuantizedMaintenancePin, QuantizedMaintenanceSlot};
use super::{VectorAccessor, VectorIndexKind};
use crate::graph::lpg::DataRebindError;
use grafeo_common::types::NodeId;
use grafeo_common::utils::error::Result;
use std::sync::Arc;

/// Borrows a captured immutable target anchor, not an Arc stored in this pin.
/// The aggregate keeps pins separate from the workspaces/proofs borrowing them.
pub(crate) enum VectorMaintenancePin<'index> {
    Hnsw(HnswMaintenancePin<'index>),
    Quantized(QuantizedMaintenancePin<'index>),
}

/// Sealed entry to raw slot helpers. Only this coordinator constructs a scope,
/// from one previously qualified slot/pin pair retained by the batch loan.
/// Sibling modules can consume its proof but cannot manufacture or retain one.
pub(in crate::index::vector) struct VectorCommitScope<'index, 'pin> {
    pin: &'pin VectorMaintenancePin<'index>,
}

impl VectorCommitScope<'_, '_> {
    pub(in crate::index::vector) fn pins_hnsw(&self, pin: &HnswMaintenancePin<'_>) -> bool {
        match self.pin {
            VectorMaintenancePin::Hnsw(expected) => std::ptr::eq(expected, pin),
            VectorMaintenancePin::Quantized(expected) => expected.pins_topology(pin),
        }
    }

    pub(in crate::index::vector) fn pins_quantized(
        &self,
        pin: &QuantizedMaintenancePin<'_>,
    ) -> bool {
        match self.pin {
            VectorMaintenancePin::Hnsw(_) => false,
            VectorMaintenancePin::Quantized(expected) => std::ptr::eq(expected, pin),
        }
    }
}

impl VectorIndexKind {
    pub(crate) fn pin_maintenance(&self) -> Result<VectorMaintenancePin<'_>> {
        match self {
            Self::Hnsw(index) => index.pin_maintenance().map(VectorMaintenancePin::Hnsw),
            Self::Quantized(index) => index.pin_maintenance().map(VectorMaintenancePin::Quantized),
        }
    }
}

/// One fixed target and its outer-owned sparse candidate/retirement state.
/// Raw guard fields borrow only the separately retained concrete index.
pub(crate) struct VectorMaintenanceSlot<'index> {
    kind: MaintenanceSlotKind<'index>,
    #[cfg(test)]
    before_retire: Option<Box<dyn FnOnce() + 'index>>,
}

enum MaintenanceSlotKind<'index> {
    Hnsw(HnswMaintenanceSlot<'index>),
    Quantized(QuantizedMaintenanceSlot<'index>),
}

impl<'index> VectorMaintenanceSlot<'index> {
    pub(crate) fn new(
        index: &'index VectorIndexKind,
        rows: Vec<(NodeId, Option<Arc<[f32]>>)>,
    ) -> Self {
        let kind = match index {
            VectorIndexKind::Hnsw(index) => {
                MaintenanceSlotKind::Hnsw(HnswMaintenanceSlot::new(index, rows))
            }
            VectorIndexKind::Quantized(index) => {
                MaintenanceSlotKind::Quantized(QuantizedMaintenanceSlot::new(index, rows))
            }
        };
        Self {
            kind,
            #[cfg(test)]
            before_retire: None,
        }
    }

    #[cfg(test)]
    pub(in crate::index::vector) fn before_retire_for_test(
        &mut self,
        probe: impl FnOnce() + 'index,
    ) {
        self.before_retire = Some(Box::new(probe));
    }

    pub(crate) fn from_recorded(index: &'index VectorIndexKind, payload: Vec<u8>) -> Self {
        let kind = match index {
            VectorIndexKind::Hnsw(index) => {
                MaintenanceSlotKind::Hnsw(HnswMaintenanceSlot::from_recorded(index, payload))
            }
            VectorIndexKind::Quantized(index) => MaintenanceSlotKind::Quantized(
                QuantizedMaintenanceSlot::from_recorded(index, payload),
            ),
        };
        Self {
            kind,
            #[cfg(test)]
            before_retire: None,
        }
    }

    fn encode_wal_postimage(&self) -> Result<Vec<u8>> {
        match &self.kind {
            MaintenanceSlotKind::Hnsw(slot) => slot.encode_wal_postimage(),
            MaintenanceSlotKind::Quantized(slot) => slot.encode_wal_postimage(),
        }
    }

    fn matches(&self, pin: &VectorMaintenancePin<'_>) -> bool {
        match (&self.kind, pin) {
            (MaintenanceSlotKind::Hnsw(slot), VectorMaintenancePin::Hnsw(pin)) => slot.matches(pin),
            (MaintenanceSlotKind::Quantized(slot), VectorMaintenancePin::Quantized(pin)) => {
                slot.matches(pin)
            }
            _ => false,
        }
    }

    fn prepare(
        &mut self,
        pin: &VectorMaintenancePin<'index>,
        vectors: &impl VectorAccessor,
        capture_wal: bool,
    ) -> Result<()> {
        match (&mut self.kind, pin) {
            (MaintenanceSlotKind::Hnsw(slot), VectorMaintenancePin::Hnsw(pin)) => {
                if capture_wal {
                    slot.capture_wal()?;
                }
                slot.prepare(pin, vectors)
            }
            (MaintenanceSlotKind::Quantized(slot), VectorMaintenancePin::Quantized(pin)) => {
                if capture_wal {
                    slot.capture_wal()?;
                }
                slot.prepare(pin)
            }
            _ => Err(DataRebindError::new("Vector slot concrete kind changed").into_error()),
        }
    }

    fn exclude_readers(
        &mut self,
        pin: &VectorMaintenancePin<'index>,
    ) -> std::result::Result<(), DataRebindError> {
        let scope = VectorCommitScope { pin };
        match (&mut self.kind, pin) {
            (MaintenanceSlotKind::Hnsw(slot), VectorMaintenancePin::Hnsw(pin)) => {
                slot.exclude_readers(pin, &scope)
            }
            (MaintenanceSlotKind::Quantized(slot), VectorMaintenancePin::Quantized(pin)) => {
                slot.exclude_readers(pin, &scope)
            }
            _ => Err(DataRebindError::new(
                "Vector reader slot concrete kind changed",
            )),
        }
    }

    fn rebind(
        &mut self,
        pin: &VectorMaintenancePin<'index>,
    ) -> std::result::Result<(), DataRebindError> {
        let scope = VectorCommitScope { pin };
        match (&mut self.kind, pin) {
            (MaintenanceSlotKind::Hnsw(slot), VectorMaintenancePin::Hnsw(pin)) => {
                slot.rebind(pin, &scope)
            }
            (MaintenanceSlotKind::Quantized(slot), VectorMaintenancePin::Quantized(pin)) => {
                slot.rebind(pin, &scope)
            }
            _ => Err(DataRebindError::new(
                "Vector state slot concrete kind changed",
            )),
        }
    }

    fn install(&mut self, pin: &VectorMaintenancePin<'index>) {
        let scope = VectorCommitScope { pin };
        match &mut self.kind {
            MaintenanceSlotKind::Hnsw(slot) => slot.install(&scope),
            MaintenanceSlotKind::Quantized(slot) => slot.install(&scope),
        }
    }

    fn release_state(&mut self) {
        match &mut self.kind {
            MaintenanceSlotKind::Hnsw(slot) => slot.release_state(),
            MaintenanceSlotKind::Quantized(slot) => slot.release_state(),
        }
    }

    fn release_readers(&mut self) {
        match &mut self.kind {
            MaintenanceSlotKind::Hnsw(slot) => slot.release_readers(),
            MaintenanceSlotKind::Quantized(slot) => slot.release_readers(),
        }
    }
}

impl Drop for VectorMaintenanceSlot<'_> {
    fn drop(&mut self) {
        self.release_state();
        self.release_readers();
        #[cfg(test)]
        if let Some(probe) = self.before_retire.take() {
            probe();
        }
    }
}

/// The buffer owner must precede all enclosing publication/authority guards.
/// A separate owner (rather than a bare Vec) also drains every target's guards
/// before any payload is destroyed if a borrowed batch fence was forgotten.
pub(crate) struct VectorMaintenanceSlots<'index> {
    slots: Vec<VectorMaintenanceSlot<'index>>,
    capture_wal: bool,
    postimages: Vec<Vec<u8>>,
}

impl<'index> VectorMaintenanceSlots<'index> {
    pub(crate) fn new(slots: Vec<VectorMaintenanceSlot<'index>>) -> Self {
        Self {
            slots,
            capture_wal: false,
            postimages: Vec::new(),
        }
    }

    pub(crate) fn capture_wal(&mut self) {
        self.capture_wal = true;
    }

    pub(crate) fn release_guards(&mut self) {
        release_slot_guards(&mut self.slots);
    }
}

impl Drop for VectorMaintenanceSlots<'_> {
    fn drop(&mut self) {
        self.release_guards();
    }
}

fn release_slot_guards(slots: &mut [VectorMaintenanceSlot<'_>]) {
    for slot in slots.iter_mut().rev() {
        slot.release_state();
    }
    for slot in slots.iter_mut().rev() {
        slot.release_readers();
    }
}

/// The only aggregate loan containing local authority references. Its heap
/// storage contains no such references and therefore survives this loan.
struct SlotFence<'index, 'workspace, 'pin> {
    slots: &'workspace mut [VectorMaintenanceSlot<'index>],
    pins: &'pin [VectorMaintenancePin<'index>],
    postimages: &'workspace mut Vec<Vec<u8>>,
}

impl Drop for SlotFence<'_, '_, '_> {
    fn drop(&mut self) {
        release_slot_guards(self.slots);
    }
}

#[must_use]
pub(crate) struct ReleasedVectorMaintenanceSlots<'index, 'workspace, 'pin> {
    fence: SlotFence<'index, 'workspace, 'pin>,
}

/// Every participating Vector search has drained, before any final graph
/// writer. Keep this admission through all companion installations.
#[must_use]
pub(crate) struct VectorMaintenanceReaderFence<'index, 'workspace, 'pin> {
    fence: SlotFence<'index, 'workspace, 'pin>,
}

#[must_use]
pub(crate) struct PreparedVectorMaintenanceSlots<'index, 'workspace, 'pin> {
    fence: SlotFence<'index, 'workspace, 'pin>,
}

#[must_use]
pub(crate) struct InstalledVectorMaintenanceSlots<'index, 'workspace, 'pin> {
    _fence: SlotFence<'index, 'workspace, 'pin>,
}

/// Completes all sparse candidates under the exact, continuously retained pin
/// slice. Slot order is frozen for this loan. `vectors(slot, id)` supplies the
/// immutable non-recording final-row routing vector for a plain HNSW target;
/// Quantized targets use their own retained full-vector directory.
pub(crate) fn prepare_vector_maintenance_slots<'index, 'workspace, 'pin>(
    workspace: &'workspace mut VectorMaintenanceSlots<'index>,
    pins: &'pin [VectorMaintenancePin<'index>],
    vectors: &(impl Fn(usize, NodeId) -> Option<Arc<[f32]>> + Sync),
) -> Result<ReleasedVectorMaintenanceSlots<'index, 'workspace, 'pin>> {
    let capture_wal = workspace.capture_wal;
    let fence = SlotFence {
        slots: &mut workspace.slots,
        pins,
        postimages: &mut workspace.postimages,
    };
    if fence.slots.len() != pins.len()
        || fence
            .slots
            .iter()
            .zip(pins)
            .any(|(slot, pin)| !slot.matches(pin))
    {
        return Err(
            DataRebindError::new("Vector slots do not match the exact pin sequence").into_error(),
        );
    }
    if capture_wal {
        fence
            .postimages
            .try_reserve(fence.slots.len())
            .map_err(|_| grafeo_common::memory::AllocError::OutOfMemory)?;
    }
    for (ordinal, (slot, pin)) in fence.slots.iter_mut().zip(pins).enumerate() {
        slot.prepare(pin, &|id| vectors(ordinal, id), capture_wal)?;
        if capture_wal {
            fence.postimages.push(slot.encode_wal_postimage()?);
        }
    }
    Ok(ReleasedVectorMaintenanceSlots { fence })
}

impl<'index, 'workspace, 'pin> ReleasedVectorMaintenanceSlots<'index, 'workspace, 'pin> {
    /// This is the blocking drainage phase, not final state binding. Call it
    /// for all indexes before acquiring any final graph/property writer.
    pub(crate) fn exclude_readers(
        self,
    ) -> std::result::Result<VectorMaintenanceReaderFence<'index, 'workspace, 'pin>, DataRebindError>
    {
        for (slot, pin) in self.fence.slots.iter_mut().zip(self.fence.pins) {
            slot.exclude_readers(pin)?;
        }
        Ok(VectorMaintenanceReaderFence { fence: self.fence })
    }
}

impl<'index, 'workspace, 'pin> VectorMaintenanceReaderFence<'index, 'workspace, 'pin> {
    pub(crate) fn wal_postimages(&self) -> impl Iterator<Item = (usize, &[u8])> {
        self.fence
            .postimages
            .iter()
            .enumerate()
            .map(|(index, bytes)| (index, bytes.as_slice()))
    }

    /// Try-only final binding. Rejection releases every acquired state/reader
    /// guard but retains every candidate, retiree and outer buffer allocation.
    /// The aggregate must drain its OTHER final fences before owning the error.
    pub(crate) fn rebind(
        self,
    ) -> std::result::Result<
        PreparedVectorMaintenanceSlots<'index, 'workspace, 'pin>,
        DataRebindError,
    > {
        for (slot, pin) in self.fence.slots.iter_mut().zip(self.fence.pins) {
            slot.rebind(pin)?;
        }
        Ok(PreparedVectorMaintenanceSlots { fence: self.fence })
    }
}

impl<'index, 'workspace, 'pin> PreparedVectorMaintenanceSlots<'index, 'workspace, 'pin> {
    /// Installs in place with all the same slot guards retained. This consumes
    /// the only ready loan; callers cannot repeat installation or swap targets.
    pub(crate) fn install(self) -> InstalledVectorMaintenanceSlots<'index, 'workspace, 'pin> {
        // Preparation checked this exact, fixed-length slot/pin sequence.
        // Both slices remain borrowed unchanged through the ready loan.
        for (slot, pin) in self.fence.slots.iter_mut().zip(self.fence.pins) {
            slot.install(pin);
        }
        InstalledVectorMaintenanceSlots { _fence: self.fence }
    }

    /// Premarker release of inner state writers, retaining reader exclusion
    /// and the exact pin sequence for a fresh final binding attempt.
    #[cfg(test)]
    pub(crate) fn release(self) -> VectorMaintenanceReaderFence<'index, 'workspace, 'pin> {
        for slot in self.fence.slots.iter_mut().rev() {
            slot.release_state();
        }
        VectorMaintenanceReaderFence { fence: self.fence }
    }
}
