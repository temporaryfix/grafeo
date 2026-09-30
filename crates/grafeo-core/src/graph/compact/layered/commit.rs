//! The cold-delete companion of aggregate LPG publication.
//!
//! Merge admission precedes topology and LPG transition admission. The final
//! generation writer is try-only and follows Vector reader exclusion, before
//! any raw data writers. This is not an independent commit API: the driver must
//! bind and retain every hot-data, index, catalog and transaction companion.
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use super::{BaseEdgeDelete, BaseNodeDelete, LayeredStore};
use crate::graph::lpg::{DataRebindError, LpgStore, PinnedLpgTransition};
use grafeo_common::memory::AllocError;
use grafeo_common::types::{EdgeId, EpochId, NodeId, TransactionId};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::{RwLockReadGuard, RwLockWriteGuard};
use std::cell::Cell;
use std::sync::atomic::Ordering;

pub(crate) mod slots;

/// The exact Layered generation, pinned before any topology/store transition.
pub(crate) struct LayeredCommitPin<'store> {
    store: &'store LayeredStore,
    overlay: &'store LpgStore,
    _merge: RwLockReadGuard<'store, ()>,
    prepared: Cell<bool>,
}

impl LayeredStore {
    pub(crate) fn pin_commit<'store>(
        &'store self,
        expected_overlay: &'store LpgStore,
    ) -> Result<LayeredCommitPin<'store>, DataRebindError> {
        // Nonblocking admission also rejects same-thread generation callbacks
        // without the ordinary scope's assertion/TLS bookkeeping machinery.
        let merge = self
            .merge_guard
            .try_read()
            .ok_or(DataRebindError::Conflict(
                "Layered generation is changing during commit admission",
            ))?;
        if !std::ptr::eq(self.overlay.load().as_ref(), expected_overlay) {
            return Err(DataRebindError::new("Layered commit overlay changed"));
        }
        Ok(LayeredCommitPin {
            store: self,
            overlay: expected_overlay,
            _merge: merge,
            prepared: Cell::new(false),
        })
    }
}

/// Outer ownership of captured queues and eventual removed queue allocations.
pub(crate) struct LayeredCommitWorkspace {
    transaction: TransactionId,
    publication: EpochId,
    commit: EpochId,
    nodes: Option<Vec<NodeId>>,
    edges: Option<Vec<EdgeId>>,
    retired_nodes: Option<Vec<NodeId>>,
    retired_edges: Option<Vec<EdgeId>>,
    attempted: bool,
}

impl LayeredCommitWorkspace {
    pub(crate) fn new(transaction: TransactionId, publication: EpochId, commit: EpochId) -> Self {
        Self {
            transaction,
            publication,
            commit,
            nodes: None,
            edges: None,
            retired_nodes: None,
            retired_edges: None,
            attempted: false,
        }
    }
}

/// A loaned generation writer; no public Layered callback may run beneath it.
#[cfg(test)]
pub(crate) struct LayeredPublicationFence<'store, 'authority> {
    pin: &'authority LayeredCommitPin<'store>,
    transition: &'authority PinnedLpgTransition<'store>,
    _publication: RwLockWriteGuard<'store, ()>,
}

impl<'store> LayeredCommitPin<'store> {
    /// Identifies the independently anchored overlay of this retained generation.
    pub(crate) fn pins_overlay(&self, overlay: &LpgStore) -> bool {
        std::ptr::eq(self.overlay, overlay)
    }

    /// Property definitions transferred during compaction can have hot-only
    /// physical memberships. Only a cold identity with NO overlay lifetime is
    /// eligible for translating a logical old value to absent physical state.
    pub(crate) fn is_unhydrated_cold_node(
        &self,
        transition: &PinnedLpgTransition<'store>,
        id: NodeId,
    ) -> Result<bool, DataRebindError> {
        if !transition.pins_store(self.overlay) {
            return Err(DataRebindError::new("Layered cold-row transition mismatch"));
        }
        Ok(!self.overlay.contains_node_identity(id)
            && LayeredStore::base_has_node_identity(&self.store.base.load(), id))
    }

    #[cfg(test)]
    pub(crate) fn prepare<'workspace, 'authority>(
        &'authority self,
        transition: &'authority PinnedLpgTransition<'store>,
        workspace: &'workspace mut LayeredCommitWorkspace,
    ) -> Result<ReleasedLayeredCommit<'store, 'workspace, 'authority>, DataRebindError> {
        self.prepare_workspace(transition, workspace)?;
        Ok(ReleasedLayeredCommit {
            pin: self,
            transition,
            workspace,
        })
    }

    fn prepare_workspace(
        &self,
        transition: &PinnedLpgTransition<'store>,
        workspace: &mut LayeredCommitWorkspace,
    ) -> Result<(), DataRebindError> {
        if !transition.pins_store(self.overlay) {
            return Err(DataRebindError::new("Layered commit transition mismatch"));
        }
        if self.prepared.replace(true) || workspace.attempted {
            return Err(DataRebindError::new(
                "Layered commit preparation is one-shot",
            ));
        }
        workspace.attempted = true;
        if workspace.transaction == TransactionId::INVALID
            || workspace.transaction == TransactionId::SYSTEM
            || workspace.publication >= workspace.commit
            || workspace.commit == EpochId::PENDING
        {
            return Err(DataRebindError::new("invalid Layered commit frontier"));
        }
        // Only the owning transaction's queues are copied. No base/overlay,
        // routing directory, or unrelated tombstone history is reconstructed.
        copy_queue(
            self.store
                .pending_base_node_deletes
                .read()
                .get(&workspace.transaction),
            &mut workspace.nodes,
        )?;
        copy_queue(
            self.store
                .pending_base_edge_deletes
                .read()
                .get(&workspace.transaction),
            &mut workspace.edges,
        )?;
        let nodes = self.store.deleted_from_base_nodes.read();
        let edges = self.store.deleted_from_base_edges.read();
        validate_stamps(workspace, &nodes, &edges)?;
        Ok(())
    }

    /// Call after every Vector reader exclusion and before any raw data writer.
    /// Never park: a Layered reader may be waiting on those Vector exclusions.
    #[cfg(test)]
    pub(crate) fn exclude_readers<'authority>(
        &'authority self,
        transition: &'authority PinnedLpgTransition<'store>,
    ) -> Result<LayeredPublicationFence<'store, 'authority>, DataRebindError> {
        let publication = self.publication_writer(transition)?;
        Ok(LayeredPublicationFence {
            pin: self,
            transition,
            _publication: publication,
        })
    }

    fn publication_writer(
        &self,
        transition: &PinnedLpgTransition<'store>,
    ) -> Result<RwLockWriteGuard<'store, ()>, DataRebindError> {
        if !transition.pins_store(self.overlay) {
            return Err(DataRebindError::new(
                "Layered publication transition mismatch",
            ));
        }
        self.store
            .publication_guard
            .try_write()
            .ok_or(DataRebindError::Conflict(
                "Layered reader is active at final commit binding",
            ))
    }
}

fn copy_queue<Id: Copy>(
    source: Option<&Vec<Id>>,
    destination: &mut Option<Vec<Id>>,
) -> Result<(), DataRebindError> {
    if let Some(source) = source {
        // Install the owner before the fallible reserve, retaining partial
        // preparation in the caller's outer retirement scope on every error.
        let captured = destination.insert(Vec::new());
        captured
            .try_reserve_exact(source.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        captured.extend_from_slice(source);
    }
    Ok(())
}

fn validate_stamps(
    workspace: &LayeredCommitWorkspace,
    nodes: &FxHashMap<NodeId, BaseNodeDelete>,
    edges: &FxHashMap<EdgeId, BaseEdgeDelete>,
) -> Result<(), DataRebindError> {
    for id in workspace.nodes.iter().flatten() {
        if !id.is_valid()
            || !nodes.get(id).is_some_and(|stamp| {
                stamp.epoch == EpochId::PENDING && stamp.deleter == Some(workspace.transaction)
            })
        {
            return Err(DataRebindError::new(
                "unowned Layered pending node deletion",
            ));
        }
    }
    for id in workspace.edges.iter().flatten() {
        if !id.is_valid()
            || !edges.get(id).is_some_and(|stamp| {
                stamp.epoch == EpochId::PENDING && stamp.deleter == Some(workspace.transaction)
            })
        {
            return Err(DataRebindError::new(
                "unowned Layered pending edge deletion",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) struct ReleasedLayeredCommit<'store, 'workspace, 'authority> {
    pin: &'authority LayeredCommitPin<'store>,
    transition: &'authority PinnedLpgTransition<'store>,
    workspace: &'workspace mut LayeredCommitWorkspace,
}

struct TombstoneGuards<'store> {
    edges: RwLockWriteGuard<'store, FxHashMap<EdgeId, BaseEdgeDelete>>,
    pending_edges: RwLockWriteGuard<'store, FxHashMap<TransactionId, Vec<EdgeId>>>,
    nodes: RwLockWriteGuard<'store, FxHashMap<NodeId, BaseNodeDelete>>,
    pending_nodes: RwLockWriteGuard<'store, FxHashMap<TransactionId, Vec<NodeId>>>,
}

impl<'store> TombstoneGuards<'store> {
    fn try_acquire(store: &'store LayeredStore) -> Result<Self, DataRebindError> {
        let pending_nodes =
            store
                .pending_base_node_deletes
                .try_write()
                .ok_or(DataRebindError::Conflict(
                    "Layered pending node queue is borrowed",
                ))?;
        let nodes = store
            .deleted_from_base_nodes
            .try_write()
            .ok_or(DataRebindError::Conflict(
                "Layered node tombstones are borrowed",
            ))?;
        let pending_edges =
            store
                .pending_base_edge_deletes
                .try_write()
                .ok_or(DataRebindError::Conflict(
                    "Layered pending edge queue is borrowed",
                ))?;
        let edges = store
            .deleted_from_base_edges
            .try_write()
            .ok_or(DataRebindError::Conflict(
                "Layered edge tombstones are borrowed",
            ))?;
        Ok(Self {
            edges,
            pending_edges,
            nodes,
            pending_nodes,
        })
    }

    fn validate(&self, workspace: &LayeredCommitWorkspace) -> Result<(), DataRebindError> {
        if self.pending_nodes.get(&workspace.transaction) != workspace.nodes.as_ref()
            || self.pending_edges.get(&workspace.transaction) != workspace.edges.as_ref()
            || workspace.retired_nodes.is_some()
            || workspace.retired_edges.is_some()
        {
            return Err(DataRebindError::new(
                "Layered pending delete queues changed",
            ));
        }
        validate_stamps(workspace, &self.nodes, &self.edges)
    }

    fn install(&mut self, store: &LayeredStore, workspace: &mut LayeredCommitWorkspace) {
        for id in workspace.nodes.iter().flatten() {
            if let Some(stamp) = self.nodes.get_mut(id) {
                stamp.epoch = workspace.commit;
            }
        }
        for id in workspace.edges.iter().flatten() {
            if let Some(stamp) = self.edges.get_mut(id) {
                stamp.epoch = workspace.commit;
            }
        }
        workspace.retired_nodes = self.pending_nodes.remove(&workspace.transaction);
        workspace.retired_edges = self.pending_edges.remove(&workspace.transaction);
        if workspace.nodes.is_some() || workspace.edges.is_some() {
            store.deletions_dirty.store(true, Ordering::Release);
        }
    }
}

#[cfg(test)]
pub(crate) struct PreparedLayeredCommit<'store, 'workspace, 'authority, 'fence> {
    guards: TombstoneGuards<'store>,
    released: ReleasedLayeredCommit<'store, 'workspace, 'authority>,
    publication: &'fence LayeredPublicationFence<'store, 'authority>,
}

#[cfg(test)]
impl<'store, 'workspace, 'authority> ReleasedLayeredCommit<'store, 'workspace, 'authority> {
    pub(crate) fn rebind<'fence>(
        self,
        publication: &'fence LayeredPublicationFence<'store, 'authority>,
    ) -> Result<PreparedLayeredCommit<'store, 'workspace, 'authority, 'fence>, DataRebindError>
    {
        if !std::ptr::eq(self.pin, publication.pin)
            || !std::ptr::eq(self.transition, publication.transition)
        {
            return Err(DataRebindError::new("Layered publication fence mismatch"));
        }
        let guards = TombstoneGuards::try_acquire(self.pin.store)?;
        guards.validate(self.workspace)?;
        Ok(PreparedLayeredCommit {
            guards,
            released: self,
            publication,
        })
    }
}

#[cfg(test)]
pub(crate) struct InstalledLayeredCommit<'store, 'workspace, 'authority, 'fence> {
    _guards: TombstoneGuards<'store>,
    _workspace: &'workspace mut LayeredCommitWorkspace,
    _publication: &'fence LayeredPublicationFence<'store, 'authority>,
}

#[cfg(test)]
impl<'store, 'workspace, 'authority, 'fence>
    PreparedLayeredCommit<'store, 'workspace, 'authority, 'fence>
{
    pub(crate) fn release(self) -> ReleasedLayeredCommit<'store, 'workspace, 'authority> {
        self.released
    }

    pub(crate) fn install(self) -> InstalledLayeredCommit<'store, 'workspace, 'authority, 'fence> {
        let Self {
            mut guards,
            released,
            publication,
        } = self;
        let workspace = released.workspace;
        guards.install(released.pin.store, workspace);
        InstalledLayeredCommit {
            _guards: guards,
            _workspace: workspace,
            _publication: publication,
        }
    }
}

#[cfg(test)]
mod tests;
