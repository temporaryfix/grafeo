//! Preallocated collective Text reader exclusion, without publication authority.
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use super::{InvertedIndex, RegisteredTextIndex, TextCommitScope, TextCommitWorkspace};
use crate::graph::lpg::DataRebindError;
use grafeo_common::memory::AllocError;
use grafeo_common::utils::error::Result;
use parking_lot::{ArcRwLockWriteGuard, RawRwLock, RwLock};
use std::sync::Arc;

type IndexAnchor = Arc<RwLock<InvertedIndex>>;
type IndexWriter = ArcRwLockWriteGuard<RawRwLock, InvertedIndex>;

/// Owns exact synchronization identities and every guard-buffer allocation.
///
/// Declare outside all enclosing publication/transition guards. Preparation
/// may allocate; subsequent acquisition, rejection and fence drop cannot retire
/// these buffers or the anchored index payloads. This workspace is not authority.
pub(crate) struct TextRegistryFenceWorkspace {
    // Also safe if a caller forgets a fence: workspace destruction releases
    // concrete writers before caller gates, and both before the Arc anchors.
    target_guards: Vec<IndexWriter>,
    gate_guards: Vec<IndexWriter>,
    targets: Vec<IndexAnchor>,
    gates: Vec<IndexAnchor>,
    attempted: bool,
    prepared: bool,
}

impl TextRegistryFenceWorkspace {
    pub(crate) fn new() -> Self {
        Self {
            target_guards: Vec::new(),
            gate_guards: Vec::new(),
            targets: Vec::new(),
            gates: Vec::new(),
            attempted: false,
            prepared: false,
        }
    }

    /// Captures a fixed set of gates and concrete targets before entity writers.
    /// One-shot preparation retains partial anchors on failure; acquisition may
    /// subsequently be retried after a conflict or a completed fence lifetime.
    pub(crate) fn prepare(&mut self, entries: &[RegisteredTextIndex]) -> Result<()> {
        if self.attempted {
            return Err(
                DataRebindError::new("Text fence preparation was already attempted").into_error(),
            );
        }
        self.attempted = true;
        self.gates
            .try_reserve(entries.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        self.targets
            .try_reserve(entries.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        for entry in entries {
            self.gates.push(Arc::clone(&entry.gate));
            self.targets.push(Arc::clone(&entry.target));
        }
        self.gates.sort_unstable_by_key(identity);
        self.gates.dedup_by(|left, right| Arc::ptr_eq(left, right));
        self.targets.sort_unstable_by_key(identity);
        self.targets
            .dedup_by(|left, right| Arc::ptr_eq(left, right));
        // A lock serving both roles is held once, as a caller gate. The gate
        // anchor also retains the concrete payload after deduplication.
        self.targets.retain(|target| {
            self.gates
                .binary_search_by_key(&identity(target), identity)
                .is_err()
        });

        #[cfg(test)]
        if FAIL_AFTER_ANCHORS.with(std::cell::Cell::get) {
            return Err(AllocError::OutOfMemory.into());
        }

        self.gate_guards
            .try_reserve(self.gates.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        self.target_guards
            .try_reserve(self.targets.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        self.prepared = true;
        Ok(())
    }

    /// Releases concrete writers before caller gates, retaining every anchor
    /// and guard-buffer allocation. The exclusive workspace borrow also makes
    /// this usable for enclosing-workspace cleanup after a forgotten fence.
    pub(crate) fn release_guards(&mut self) {
        self.target_guards.clear();
        self.gate_guards.clear();
    }
}

/// All fixed caller gates and concrete targets for one registry publication.
///
/// Exclusion alone is not mutation authority. Exact-target maintenance also
/// requires its retained local scope and the aggregate's linear ready proof.
/// Nonblocking acquisition preserves the section restore semantics.
#[must_use]
pub(crate) struct TextRegistryBatchFence<'workspace> {
    workspace: &'workspace mut TextRegistryFenceWorkspace,
}

impl<'workspace> TextRegistryBatchFence<'workspace> {
    /// Acquires only prequalified identities into preallocated guard buffers.
    /// Errors stay allocation-free while other stores' writers may be retained.
    pub(crate) fn try_acquire(
        workspace: &'workspace mut TextRegistryFenceWorkspace,
    ) -> std::result::Result<Self, DataRebindError> {
        if !workspace.prepared {
            return Err(DataRebindError::new("Text fence workspace is not prepared"));
        }
        if !workspace.gate_guards.is_empty() || !workspace.target_guards.is_empty() {
            return Err(DataRebindError::new(
                "Text fence workspace still holds writers",
            ));
        }
        if workspace.gate_guards.capacity() < workspace.gates.len()
            || workspace.target_guards.capacity() < workspace.targets.len()
        {
            return Err(DataRebindError::new(
                "Text fence reserved guard capacity was lost",
            ));
        }
        // Establish RAII cleanup before the first lock attempt. Every partial
        // failure and unwind drains target guards, then caller gates, while all
        // Arc anchors and Vec allocations remain in the outer workspace.
        let fence = Self { workspace };
        for gate in &fence.workspace.gates {
            let guard = gate.try_write_arc().ok_or(DataRebindError::Conflict(
                "Text registry batch caller gate is in use",
            ))?;
            fence.workspace.gate_guards.push(guard);
        }
        for target in &fence.workspace.targets {
            let guard = target.try_write_arc().ok_or(DataRebindError::Conflict(
                "Text registry batch concrete target is in use",
            ))?;
            fence.workspace.target_guards.push(guard);
        }
        Ok(fence)
    }

    /// Borrows the already-locked exact target, qualified by its local scope.
    /// This never reacquires a concrete target or a caller gate.
    pub(crate) fn maintenance_target(
        &mut self,
        entry: &RegisteredTextIndex,
        scope: &TextCommitScope,
    ) -> Option<&mut InvertedIndex> {
        let target = identity(&entry.target);
        let guards = if self
            .workspace
            .gate_guards
            .binary_search_by_key(&target, |guard| {
                identity(ArcRwLockWriteGuard::rwlock(guard))
            })
            .is_ok()
        {
            &mut self.workspace.gate_guards
        } else {
            &mut self.workspace.target_guards
        };
        let position = guards
            .binary_search_by_key(&target, |guard| {
                identity(ArcRwLockWriteGuard::rwlock(guard))
            })
            .ok()?;
        let index = guards.get_mut(position)?;
        scope.matches(index).then_some(&mut **index)
    }

    /// Called only by the validated aggregate while every exact target/scope
    /// remains exclusively loaned to that proof. No late fallible bind.
    pub(crate) fn install_maintenance(
        &mut self,
        entry: &RegisteredTextIndex,
        workspace: &mut TextCommitWorkspace,
        scope: &TextCommitScope,
    ) {
        if let Some(index) = self.maintenance_target(entry, scope) {
            workspace.install_prequalified(index, scope);
        }
    }
}

impl Drop for TextRegistryBatchFence<'_> {
    fn drop(&mut self) {
        self.workspace.release_guards();
    }
}

fn identity(anchor: &IndexAnchor) -> usize {
    Arc::as_ptr(anchor).addr()
}

#[cfg(test)]
thread_local! {
    static FAIL_AFTER_ANCHORS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
mod tests;
