//! Exact snapshot clocks, with transaction admission retained through publication.

use super::{TransactionInfo, TransactionManager, TransactionState};
use crate::transaction::read_registry::QuiescentReadRegistry;
use grafeo_common::types::{EpochId, TransactionId};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::RwLockWriteGuard;
use std::sync::atomic::Ordering;

/// Owns displaced conflict metadata outside every publication/transaction guard.
#[derive(Default)]
pub(crate) struct SnapshotClockWorkspace {
    transactions: FxHashMap<TransactionId, TransactionInfo>,
    committed: FxHashMap<TransactionId, EpochId>,
    retired: FxHashMap<TransactionId, EpochId>,
}

pub(crate) struct ReadySnapshotClock<'a> {
    manager: &'a TransactionManager,
    epoch: EpochId,
    workspace: &'a mut SnapshotClockWorkspace,
    transactions: RwLockWriteGuard<'a, FxHashMap<TransactionId, TransactionInfo>>,
    committed: RwLockWriteGuard<'a, FxHashMap<TransactionId, EpochId>>,
    retired: RwLockWriteGuard<'a, FxHashMap<TransactionId, EpochId>>,
    _readers: QuiescentReadRegistry<'a>,
    _publication: &'a RwLockWriteGuard<'a, ()>,
}

pub(crate) struct InstalledSnapshotClock<'a> {
    _ready: ReadySnapshotClock<'a>,
}

impl TransactionManager {
    pub(crate) fn prepare_snapshot_clock<'a>(
        &'a self,
        workspace: &'a mut SnapshotClockWorkspace,
        epoch: EpochId,
        publication: &'a RwLockWriteGuard<'a, ()>,
    ) -> Result<ReadySnapshotClock<'a>> {
        if epoch == EpochId::PENDING
            || !std::ptr::eq(RwLockWriteGuard::rwlock(publication), self.publication())
            || !workspace.transactions.is_empty()
            || !workspace.committed.is_empty()
            || !workspace.retired.is_empty()
        {
            return Err(Error::InvalidValue(
                "snapshot clock requires a real epoch and its exact publication guard".into(),
            ));
        }
        let transactions = self.transactions.write();
        if transactions
            .values()
            .any(|info| info.state == TransactionState::Active)
            || self.active_count.load(Ordering::Acquire) != 0
        {
            return Err(Error::Internal(
                "restore_snapshot requires quiescent transaction admission".into(),
            ));
        }
        let committed = self.committed_epochs.write();
        let retired = self.retired_readers.write();
        if !retired.is_empty() {
            return Err(Error::Internal(
                "restore_snapshot requires completed reader retirement".into(),
            ));
        }
        let readers = self.read_registry.pin_quiescent().ok_or_else(|| {
            Error::Internal("restore_snapshot requires an idle, empty reader registry".into())
        })?;
        Ok(ReadySnapshotClock {
            manager: self,
            epoch,
            workspace,
            transactions,
            committed,
            retired,
            _readers: readers,
            _publication: publication,
        })
    }
}

impl<'a> ReadySnapshotClock<'a> {
    pub(crate) fn install(mut self) -> InstalledSnapshotClock<'a> {
        std::mem::swap(&mut *self.transactions, &mut self.workspace.transactions);
        std::mem::swap(&mut *self.committed, &mut self.workspace.committed);
        std::mem::swap(&mut *self.retired, &mut self.workspace.retired);
        // Exact world replacement restores the source's continuation clock.
        // Admission is quiescent and the old target's history is displaced;
        // ordinary aborts never reset either clock.
        self.manager
            .reserved_epoch
            .store(self.epoch.as_u64(), Ordering::SeqCst);
        self.manager
            .current_epoch
            .store(self.epoch.as_u64(), Ordering::SeqCst);
        InstalledSnapshotClock { _ready: self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_clock_replaces_older_cut_without_reusing_transaction_ids() {
        let manager = TransactionManager::new();
        let first = manager.begin();
        manager.abort(first).unwrap();
        manager.try_sync_epoch(EpochId::new(90)).unwrap();
        let mut workspace = SnapshotClockWorkspace::default();
        {
            let publication = manager.publication().write();
            let ready = manager
                .prepare_snapshot_clock(&mut workspace, EpochId::new(3), &publication)
                .unwrap();
            let _installed = ready.install();
            assert_eq!(manager.current_epoch(), EpochId::new(3));
            assert!(manager.transactions.try_write().is_none());
        }
        let next = manager.begin();
        assert!(next > first);
        assert_eq!(manager.start_epoch(next), Some(EpochId::new(3)));
        manager.abort(next).unwrap();
        let publication = manager.publication().write();
        assert_eq!(
            manager.reserve_publication_epoch().unwrap(),
            EpochId::new(4)
        );
        assert_eq!(manager.current_epoch(), EpochId::new(3));
        drop(publication);
    }

    #[test]
    fn snapshot_clock_refuses_active_or_foreign_publication_without_change() {
        let manager = TransactionManager::new();
        let foreign = TransactionManager::new();
        let tx = manager.begin();
        let before = manager.current_epoch();
        let mut workspace = SnapshotClockWorkspace::default();
        {
            let publication = manager.publication().write();
            assert!(
                manager
                    .prepare_snapshot_clock(&mut workspace, EpochId::new(8), &publication)
                    .is_err()
            );
        }
        manager.abort(tx).unwrap();
        {
            let publication = foreign.publication().write();
            assert!(
                manager
                    .prepare_snapshot_clock(&mut workspace, EpochId::new(8), &publication)
                    .is_err()
            );
        }
        assert_eq!(manager.current_epoch(), before);
    }
}
