//! Prepared transaction-state publication for the engine's aggregate commit.
//!
//! This does not validate conflicts or reserve an epoch: `prepare_durable_commit`
//! must already have done both. Preparation reserves only the two epoch-map
//! entries. Final binding uses nonblocking writers and static diagnostics.
//!
//! SSI garbage collection is deliberately a separate, explicit tail. The old
//! finalizer also released its transaction/epoch writers before collecting
//! readers. Keeping those readers a little longer is conservative; pruning them
//! before the aggregate's durable outcome is not. The engine must drain ALL
//! aggregate final fences before running cleanup, while retaining its enclosing
//! publication authority. Neither a fence destructor nor a workspace destructor
//! performs garbage collection or rolls back an installed durable outcome.

use super::{IsolationLevel, TransactionInfo, TransactionManager, TransactionState};
use grafeo_common::memory::AllocError;
use grafeo_common::types::{EpochId, TransactionId};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::{MappedRwLockWriteGuard, RwLock, RwLockWriteGuard};
use std::sync::Arc;
use std::sync::atomic::Ordering;

#[cfg(test)]
thread_local! {
    static REJECT_NEXT_FINAL_BIND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Allocation-free rejection, including when another component already holds
/// final structural writers. Convert only outside that complete acquisition.
#[derive(Debug)]
pub(crate) enum TransactionFinalizationError {
    Invalid(&'static str),
    Conflict(&'static str),
    Allocation(AllocError),
}

impl TransactionFinalizationError {
    pub(crate) fn into_error(self) -> Error {
        match self {
            Self::Invalid(reason) => TransactionError::InvalidState(reason.to_owned()).into(),
            Self::Conflict(reason) => TransactionError::WriteConflict(reason.to_owned()).into(),
            Self::Allocation(error) => error.into(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Fresh,
    Preparing,
    Prepared,
    Installed,
    Cleaned,
}

/// Construct before lifecycle/publication/catalog guards. This independent
/// owner retains its exact manager identity and every displaced scalar until
/// those guards have drained; ready/installed proofs only borrow it.
pub(crate) struct TransactionFinalizationWorkspace {
    publication: Option<Arc<RwLock<()>>>,
    phase: Phase,
    displaced_committed: Option<EpochId>,
    displaced_retired: Option<EpochId>,
}

impl TransactionFinalizationWorkspace {
    pub(crate) fn new() -> Self {
        Self {
            publication: None,
            phase: Phase::Fresh,
            displaced_committed: None,
            displaced_retired: None,
        }
    }

    fn belongs_to(&self, manager: &TransactionManager) -> bool {
        self.publication
            .as_ref()
            .is_some_and(|publication| Arc::ptr_eq(publication, &manager.publication))
    }
}

#[derive(Clone, Copy)]
struct Coordinate {
    transaction: TransactionId,
    commit: EpochId,
    start: EpochId,
    isolation: IsolationLevel,
    committed_before: Option<EpochId>,
    retired_before: Option<EpochId>,
}

#[must_use]
pub(crate) struct ReleasedTransactionFinalization<'manager, 'workspace> {
    manager: &'manager TransactionManager,
    workspace: &'workspace mut TransactionFinalizationWorkspace,
    coordinate: Coordinate,
}

type EpochWriter<'manager> = RwLockWriteGuard<'manager, FxHashMap<TransactionId, EpochId>>;

/// The current record is mapped from the retained transaction-map writer: no
/// keyed transaction lookup or fallible qualification remains during install.
#[must_use]
pub(crate) struct PreparedTransactionFinalization<'manager, 'workspace> {
    retired: EpochWriter<'manager>,
    committed: EpochWriter<'manager>,
    transaction: MappedRwLockWriteGuard<'manager, TransactionInfo>,
    manager: &'manager TransactionManager,
    workspace: &'workspace mut TransactionFinalizationWorkspace,
    coordinate: Coordinate,
}

/// Retains the same complete writer set through data/catalog/TM companions.
#[must_use]
pub(crate) struct InstalledTransactionFinalizationFence<'manager, 'workspace> {
    retired: EpochWriter<'manager>,
    committed: EpochWriter<'manager>,
    transaction: MappedRwLockWriteGuard<'manager, TransactionInfo>,
    manager: &'manager TransactionManager,
    workspace: &'workspace mut TransactionFinalizationWorkspace,
}

/// Explicit post-fence housekeeping, not candidate-payload retirement. Read
/// registry GC can allocate and deallocate; never run it inside an aggregate
/// fence, or infer from this token that unrelated fences have been released.
#[must_use]
pub(crate) struct TransactionFinalizationCleanup<'manager, 'workspace> {
    manager: &'manager TransactionManager,
    workspace: &'workspace mut TransactionFinalizationWorkspace,
}

impl TransactionManager {
    /// Prepares finalization after conflict validation reserved this exact C.
    /// The caller retains database publication authority continuously through
    /// preparation, marker, installation and the explicit cleanup tail.
    pub(crate) fn prepare_finalization<'manager, 'workspace>(
        &'manager self,
        transaction: TransactionId,
        commit: EpochId,
        workspace: &'workspace mut TransactionFinalizationWorkspace,
    ) -> Result<ReleasedTransactionFinalization<'manager, 'workspace>> {
        if workspace.phase != Phase::Fresh {
            return Err(TransactionFinalizationError::Invalid(
                "transaction finalization workspace is not fresh",
            )
            .into_error());
        }
        workspace.publication = Some(Arc::clone(&self.publication));
        workspace.phase = Phase::Preparing;
        self.prepare_finalization_inner(transaction, commit, workspace)
            .map_err(TransactionFinalizationError::into_error)
    }

    fn prepare_finalization_inner<'manager, 'workspace>(
        &'manager self,
        transaction: TransactionId,
        commit: EpochId,
        workspace: &'workspace mut TransactionFinalizationWorkspace,
    ) -> std::result::Result<
        ReleasedTransactionFinalization<'manager, 'workspace>,
        TransactionFinalizationError,
    > {
        if !transaction.is_valid()
            || transaction == TransactionId::SYSTEM
            || commit == EpochId::PENDING
        {
            return Err(TransactionFinalizationError::Invalid(
                "transaction finalization requires a real reserved coordinate",
            ));
        }
        // Same order as ordinary finalize/GC. These are preparation writers,
        // acquired before the aggregate begins ANY final acquisition.
        let transactions = self.transactions.read();
        let info = transactions
            .get(&transaction)
            .ok_or(TransactionFinalizationError::Invalid(
                "transaction not found during preparation",
            ))?;
        qualify(info, commit)?;
        let mut committed = self.committed_epochs.write();
        let mut retired = self.retired_readers.write();
        let coordinate = Coordinate {
            transaction,
            commit,
            start: info.start_epoch,
            isolation: info.isolation_level,
            committed_before: committed.get(&transaction).copied(),
            retired_before: retired.get(&transaction).copied(),
        };
        committed
            .try_reserve(usize::from(coordinate.committed_before.is_none()))
            .map_err(|_| TransactionFinalizationError::Allocation(AllocError::OutOfMemory))?;
        retired
            .try_reserve(usize::from(coordinate.retired_before.is_none()))
            .map_err(|_| TransactionFinalizationError::Allocation(AllocError::OutOfMemory))?;
        drop(retired);
        drop(committed);
        drop(transactions);
        workspace.phase = Phase::Prepared;
        Ok(ReleasedTransactionFinalization {
            manager: self,
            workspace,
            coordinate,
        })
    }

    /// Completes an installed workspace whose cleanup token was dropped or
    /// unwound. Call only after ALL aggregate fences drain, before releasing
    /// the enclosing publication authority. An abandoned preparation is not a
    /// committed outcome and cannot run this tail.
    ///
    /// If GC unwinds, the workspace remains Installed and cleanup may be
    /// retried. Publication is never reversed. Displaced epoch values and the
    /// publication Arc remain in the outer workspace after successful cleanup.
    #[cfg(test)]
    pub(crate) fn finish_finalization(
        &self,
        workspace: &mut TransactionFinalizationWorkspace,
    ) -> Result<()> {
        if !workspace.belongs_to(self) || workspace.phase != Phase::Installed {
            return Err(TransactionFinalizationError::Invalid(
                "transaction finalization cleanup lacks this manager's installed outcome",
            )
            .into_error());
        }
        self.gc_retired_readers();
        workspace.phase = Phase::Cleaned;
        Ok(())
    }
}

impl<'manager, 'workspace> ReleasedTransactionFinalization<'manager, 'workspace> {
    pub(crate) fn rebind(
        self,
    ) -> std::result::Result<
        PreparedTransactionFinalization<'manager, 'workspace>,
        TransactionFinalizationError,
    > {
        if self.workspace.phase != Phase::Prepared || !self.workspace.belongs_to(self.manager) {
            return Err(TransactionFinalizationError::Invalid(
                "transaction finalization preparation is not retained",
            ));
        }
        let transactions =
            self.manager
                .transactions
                .try_write()
                .ok_or(TransactionFinalizationError::Conflict(
                    "transaction records are busy",
                ))?;
        let transaction = RwLockWriteGuard::try_map(transactions, |transactions| {
            transactions.get_mut(&self.coordinate.transaction)
        })
        .map_err(|_guard| {
            TransactionFinalizationError::Invalid("prepared transaction record disappeared")
        })?;
        qualify(&transaction, self.coordinate.commit)?;
        if transaction.start_epoch != self.coordinate.start
            || transaction.isolation_level != self.coordinate.isolation
            || self.manager.active_count.load(Ordering::Relaxed) == 0
        {
            return Err(TransactionFinalizationError::Invalid(
                "prepared transaction identity or active count changed",
            ));
        }
        let committed = self.manager.committed_epochs.try_write().ok_or(
            TransactionFinalizationError::Conflict("committed transaction epochs are busy"),
        )?;
        let retired = self.manager.retired_readers.try_write().ok_or(
            TransactionFinalizationError::Conflict("retained Serializable readers are busy"),
        )?;
        qualify_slot(
            &committed,
            self.coordinate.transaction,
            self.coordinate.committed_before,
        )?;
        qualify_slot(
            &retired,
            self.coordinate.transaction,
            self.coordinate.retired_before,
        )?;
        #[cfg(test)]
        if REJECT_NEXT_FINAL_BIND.with(|reject| reject.replace(false)) {
            return Err(TransactionFinalizationError::Conflict(
                "test-only transaction finalization rebind rejection",
            ));
        }
        Ok(PreparedTransactionFinalization {
            retired,
            committed,
            transaction,
            manager: self.manager,
            workspace: self.workspace,
            coordinate: self.coordinate,
        })
    }
}

impl<'manager, 'workspace> PreparedTransactionFinalization<'manager, 'workspace> {
    /// No allocation, retirement, callback, reacquisition or fallible work.
    /// The marker must already be durable (or this is an in-memory commit).
    pub(crate) fn install(mut self) -> InstalledTransactionFinalizationFence<'manager, 'workspace> {
        self.transaction.reserved_commit_epoch = None;
        self.transaction.state = TransactionState::Committed;
        // Preserve atomic read/modify/write: a previously admitted begin may
        // still publish its active-count increment outside the transaction lock.
        self.manager.active_count.fetch_sub(1, Ordering::Relaxed);
        self.workspace.displaced_committed = install_epoch(
            &mut self.committed,
            self.coordinate.transaction,
            self.coordinate.commit,
        );
        self.workspace.displaced_retired = install_epoch(
            &mut self.retired,
            self.coordinate.transaction,
            self.coordinate.commit,
        );
        self.manager.publish_reserved_epoch(self.coordinate.commit);
        self.workspace.phase = Phase::Installed;
        InstalledTransactionFinalizationFence {
            retired: self.retired,
            committed: self.committed,
            transaction: self.transaction,
            manager: self.manager,
            workspace: self.workspace,
        }
    }
}

impl<'manager, 'workspace> InstalledTransactionFinalizationFence<'manager, 'workspace> {
    /// Releases only this companion's writers. The caller must drain the other
    /// aggregate fences before consuming the returned cleanup token.
    pub(crate) fn release(self) -> TransactionFinalizationCleanup<'manager, 'workspace> {
        drop(self.retired);
        drop(self.committed);
        drop(self.transaction);
        TransactionFinalizationCleanup {
            manager: self.manager,
            workspace: self.workspace,
        }
    }
}

impl TransactionFinalizationCleanup<'_, '_> {
    /// Runs existing watermark-based SSI housekeeping only after all aggregate
    /// final fences are gone. Dropping this token performs no housekeeping;
    /// readers remain conservatively retained for a later normal GC sweep.
    pub(crate) fn finish(self) {
        self.manager.gc_retired_readers();
        self.workspace.phase = Phase::Cleaned;
    }
}

fn install_epoch(
    map: &mut FxHashMap<TransactionId, EpochId>,
    transaction: TransactionId,
    commit: EpochId,
) -> Option<EpochId> {
    if let Some(previous) = map.get_mut(&transaction) {
        // HashMap::insert may grow before discovering an occupied entry.
        // Replacing through the existing slot cannot touch the allocation.
        Some(std::mem::replace(previous, commit))
    } else {
        // Final binding proved spare capacity for this exact vacant key and
        // retained the writer; no intervening insertion can consume it.
        map.insert(transaction, commit)
    }
}

fn qualify(
    info: &TransactionInfo,
    commit: EpochId,
) -> std::result::Result<(), TransactionFinalizationError> {
    if info.state != TransactionState::Active || info.reserved_commit_epoch != Some(commit) {
        return Err(TransactionFinalizationError::Invalid(
            "transaction has no matching prepared commit",
        ));
    }
    Ok(())
}

fn qualify_slot(
    map: &FxHashMap<TransactionId, EpochId>,
    transaction: TransactionId,
    before: Option<EpochId>,
) -> std::result::Result<(), TransactionFinalizationError> {
    if map.get(&transaction).copied() != before {
        return Err(TransactionFinalizationError::Conflict(
            "prepared transaction epoch metadata changed",
        ));
    }
    if before.is_none() && map.len() >= map.capacity() {
        return Err(TransactionFinalizationError::Invalid(
            "prepared transaction epoch capacity was consumed",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
