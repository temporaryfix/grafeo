//! Transaction manager.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use grafeo_common::types::{EdgeId, EpochId, NodeId, TransactionId};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::RwLock;

use super::ReadRegistry;

/// State of a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransactionState {
    /// Transaction is active.
    Active,
    /// Transaction is committed.
    Committed,
    /// Transaction is aborted.
    Aborted,
}

/// Transaction isolation level.
///
/// Controls the consistency guarantees and performance tradeoffs for transactions.
///
/// # Comparison
///
/// | Level | Dirty Reads | Non-Repeatable Reads | Phantom Reads | Write Skew |
/// |-------|-------------|----------------------|---------------|------------|
/// | ReadCommitted | No | Yes | Yes | Yes |
/// | SnapshotIsolation | No | No | No | Yes |
/// | Serializable | No | No | No | No |
///
/// # Performance
///
/// Higher isolation levels require more bookkeeping:
/// - `ReadCommitted`: Only tracks writes
/// - `SnapshotIsolation`: Tracks writes + snapshot versioning
/// - `Serializable`: Tracks writes + reads + SSI validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum IsolationLevel {
    /// Read Committed: sees only committed data, but may see different
    /// versions of the same row within a transaction.
    ///
    /// Lowest overhead, highest throughput, but weaker consistency.
    ReadCommitted,

    /// Snapshot Isolation (default): each transaction sees a consistent
    /// snapshot as of transaction start. Prevents non-repeatable reads
    /// and phantom reads.
    ///
    /// Vulnerable to write skew anomaly.
    #[default]
    SnapshotIsolation,

    /// Serializable Snapshot Isolation (SSI): provides full serializability
    /// by detecting read-write conflicts in addition to write-write conflicts.
    ///
    /// Prevents all anomalies including write skew, but may abort more
    /// transactions due to stricter conflict detection.
    Serializable,
}

/// Entity identifier for write tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EntityId {
    /// A node.
    Node(NodeId),
    /// An edge.
    Edge(EdgeId),
}

impl From<NodeId> for EntityId {
    fn from(id: NodeId) -> Self {
        Self::Node(id)
    }
}

impl From<EdgeId> for EntityId {
    fn from(id: EdgeId) -> Self {
        Self::Edge(id)
    }
}

/// Information about an active transaction.
pub struct TransactionInfo {
    /// Transaction state.
    pub state: TransactionState,
    /// Isolation level for this transaction.
    pub isolation_level: IsolationLevel,
    /// Start epoch (snapshot epoch for reads).
    pub start_epoch: EpochId,
    /// Set of entities written by this transaction.
    pub write_set: HashSet<EntityId>,
    /// Set of entities read by this transaction (for serializable isolation).
    pub read_set: HashSet<EntityId>,
    /// An rw-antidependency edge points INTO this transaction (this tx is the
    /// writer end of some `reader →rw self`). Used for F2 incremental SSI pivot
    /// detection.
    pub in_conflict: bool,
    /// An rw-antidependency edge points OUT of this transaction (this tx is
    /// the reader end of some `self →rw writer`). Used for F2 incremental SSI
    /// pivot detection.
    pub out_conflict: bool,
}

impl TransactionInfo {
    /// Creates a new transaction info with the given isolation level.
    fn new(start_epoch: EpochId, isolation_level: IsolationLevel) -> Self {
        Self {
            state: TransactionState::Active,
            isolation_level,
            start_epoch,
            write_set: HashSet::new(),
            read_set: HashSet::new(),
            in_conflict: false,
            out_conflict: false,
        }
    }
}

/// Manages transactions and MVCC versioning.
pub struct TransactionManager {
    /// Next transaction ID.
    next_transaction_id: AtomicU64,
    /// Current epoch.
    current_epoch: AtomicU64,
    /// Number of currently active transactions (for fast-path conflict skip).
    active_count: AtomicU64,
    /// Active transactions.
    transactions: RwLock<FxHashMap<TransactionId, TransactionInfo>>,
    /// Committed transaction epochs (for conflict detection).
    /// Maps TransactionId -> commit epoch.
    committed_epochs: RwLock<FxHashMap<TransactionId, EpochId>>,
    /// Sharded registry of active Serializable readers (SIREAD locks).
    /// Used for read-time rw-antidependency detection (F2 incremental SSI).
    read_registry: ReadRegistry,
}

impl TransactionManager {
    /// Creates a new transaction manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            // Start at 2 to avoid collision with TransactionId::SYSTEM (which is 1)
            // TransactionId::INVALID = u64::MAX, TransactionId::SYSTEM = 1, user transactions start at 2
            next_transaction_id: AtomicU64::new(2),
            current_epoch: AtomicU64::new(0),
            active_count: AtomicU64::new(0),
            transactions: RwLock::new(FxHashMap::default()),
            committed_epochs: RwLock::new(FxHashMap::default()),
            read_registry: ReadRegistry::new(),
        }
    }

    /// Begins a new transaction with the default isolation level (Snapshot Isolation).
    pub fn begin(&self) -> TransactionId {
        self.begin_with_isolation(IsolationLevel::default())
    }

    /// Begins a new transaction with the specified isolation level.
    pub fn begin_with_isolation(&self, isolation_level: IsolationLevel) -> TransactionId {
        let transaction_id =
            TransactionId::new(self.next_transaction_id.fetch_add(1, Ordering::Relaxed));
        let epoch = EpochId::new(self.current_epoch.load(Ordering::Acquire));

        let info = TransactionInfo::new(epoch, isolation_level);
        self.transactions.write().insert(transaction_id, info);
        self.active_count.fetch_add(1, Ordering::Relaxed);
        transaction_id
    }

    /// Returns the isolation level of a transaction.
    pub fn isolation_level(&self, transaction_id: TransactionId) -> Option<IsolationLevel> {
        self.transactions
            .read()
            .get(&transaction_id)
            .map(|info| info.isolation_level)
    }

    /// Records a write operation for the transaction.
    ///
    /// Uses first-writer-wins: if another active transaction has already
    /// written to the same entity, returns a write-write conflict error
    /// immediately (before the caller mutates the store).
    ///
    /// For Serializable transactions, also performs write-time
    /// rw-antidependency detection (the mirror of the read-time detection in
    /// [`record_read`](Self::record_read)): any concurrent Serializable
    /// transaction that has read `entity` (and is still active) is found via
    /// the `ReadRegistry` and a `reader →rw tx` edge is recorded for each.
    ///
    /// # Lock discipline
    ///
    /// The `transactions` write lock is held only for the W-W check and
    /// `write_set.insert`. It is dropped before consulting the `ReadRegistry`
    /// and before calling `set_rw_edge` (which also takes `transactions.write()`),
    /// avoiding re-entrant deadlock.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active or if another
    /// active transaction has already written to the same entity.
    pub fn record_write(
        &self,
        transaction_id: TransactionId,
        entity: impl Into<EntityId>,
    ) -> Result<()> {
        let entity = entity.into();

        // Perform the W-W check and write_set insert under the transactions lock,
        // then release the lock before any read_registry or set_rw_edge calls.
        let is_serializable: bool = {
            let mut txns = self.transactions.write();

            // First-writer-wins conflict detection. Skip the scan when only one
            // transaction is active (common case for auto-commit).
            if self.active_count.load(Ordering::Relaxed) > 1 {
                for (other_tx, other_info) in txns.iter() {
                    if *other_tx != transaction_id
                        && other_info.state == TransactionState::Active
                        && other_info.write_set.contains(&entity)
                    {
                        return Err(Error::Transaction(TransactionError::WriteConflict(
                            format!("Write-write conflict on entity {entity:?}"),
                        )));
                    }
                }
            }

            // Single lookup: get_mut for both state check and write_set insert
            let info = txns.get_mut(&transaction_id).ok_or_else(|| {
                Error::Transaction(TransactionError::InvalidState(
                    "Transaction not found".to_string(),
                ))
            })?;

            if info.state != TransactionState::Active {
                return Err(Error::Transaction(TransactionError::InvalidState(
                    "Transaction is not active".to_string(),
                )));
            }

            info.write_set.insert(entity);
            // Capture isolation level while we hold the lock; drop the lock at
            // the end of this block.
            info.isolation_level == IsolationLevel::Serializable
            // transactions write lock drops here
        };

        // Write-time rw-antidependency detection (Serializable writers only).
        //
        // An SI writer's overwrite does not participate in SSI cycle detection —
        // symmetric with the read-time choice in record_read, which only detects
        // when the READER is Serializable. SSI edges are formed only when the
        // acting side (reader at read-time, writer at write-time) is Serializable.
        if is_serializable {
            // readers_of uses its own sharded locks, independent of transactions.
            let readers = self.read_registry.readers_of(entity);
            for reader in readers {
                if reader != transaction_id {
                    // reader read a version this writer is now overwriting.
                    self.set_rw_edge(reader, transaction_id);
                }
            }
        }

        Ok(())
    }

    /// Records a touched entity in the transaction's write-set **without**
    /// conflict detection.
    ///
    /// Unlike [`record_write`](Self::record_write), this performs no
    /// first-writer-wins check: it simply inserts the entity so the write-set is
    /// a complete record of what the transaction touched (used by
    /// write-set-scoped commit/rollback). Used by session-direct mutators and for
    /// newly created entities, which allocate fresh ids and cannot
    /// write-write-conflict.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active.
    pub fn record_entity(
        &self,
        transaction_id: TransactionId,
        entity: impl Into<EntityId>,
    ) -> Result<()> {
        let entity = entity.into();
        let mut txns = self.transactions.write();
        let info = txns.get_mut(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;

        if info.state != TransactionState::Active {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Transaction is not active".to_string(),
            )));
        }

        info.write_set.insert(entity);
        Ok(())
    }

    /// Adds entities to the transaction's write-set without conflict detection
    /// (used to complete the set from store chokepoints before validation).
    ///
    /// Unlike [`record_write`](Self::record_write), this performs no
    /// first-writer-wins check — it simply bulk-inserts entities so the
    /// write-set is a complete record of what the transaction touched.
    /// Silently no-ops if the transaction is not active (to keep commit
    /// on the hot path allocation-free on failure).
    pub fn extend_write_set(
        &self,
        transaction_id: TransactionId,
        entities: impl IntoIterator<Item = EntityId>,
    ) {
        if let Some(info) = self.transactions.write().get_mut(&transaction_id)
            && info.state == TransactionState::Active
        {
            info.write_set.extend(entities);
        }
    }

    /// Records a read operation for the transaction (for serializable isolation).
    ///
    /// For Serializable transactions, also registers the reader in the
    /// `ReadRegistry` and performs read-time rw-antidependency detection:
    /// any concurrent transaction (active or committed-after-our-start) that
    /// has written `entity` is recorded as a writer end of a `tx →rw T_w` edge.
    ///
    /// # Lock discipline
    ///
    /// This method holds `transactions.write()` only long enough to (a) validate
    /// state, (b) insert into `read_set`, and (c) collect concurrent-writer IDs
    /// into a local `Vec`. It releases the lock before calling `set_rw_edge` (which
    /// also takes `transactions.write()`), avoiding re-entrant deadlock.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active.
    pub fn record_read(
        &self,
        transaction_id: TransactionId,
        entity: impl Into<EntityId>,
    ) -> Result<()> {
        let entity = entity.into();

        // Collect concurrent writers while holding the transactions lock, then
        // release it before calling set_rw_edge (which also takes the lock).
        let concurrent_writers: Vec<TransactionId> = {
            let mut txns = self.transactions.write();
            let info = txns.get_mut(&transaction_id).ok_or_else(|| {
                Error::Transaction(TransactionError::InvalidState(
                    "Transaction not found".to_string(),
                ))
            })?;

            if info.state != TransactionState::Active {
                return Err(Error::Transaction(TransactionError::InvalidState(
                    "Transaction is not active".to_string(),
                )));
            }

            info.read_set.insert(entity);

            // Only gather concurrent writers for Serializable transactions.
            if info.isolation_level != IsolationLevel::Serializable {
                return Ok(());
            }

            let our_start = info.start_epoch;

            // Collect active writers (uncommitted writes that our snapshot cannot see).
            let mut writers: Vec<TransactionId> = txns
                .iter()
                .filter(|(other_tx, other_info)| {
                    **other_tx != transaction_id
                        && other_info.state == TransactionState::Active
                        && other_info.write_set.contains(&entity)
                })
                .map(|(id, _)| *id)
                .collect();

            // Collect committed-after-our-start writers (they committed a newer
            // version that our snapshot doesn't see).
            //
            // We take committed_epochs as a *read* lock here. Lock ordering
            // is: transactions first, then committed_epochs — which we respect
            // (transactions write lock is already held above).
            let committed = self.committed_epochs.read();
            for (other_tx, commit_epoch) in committed.iter() {
                if *other_tx != transaction_id
                    && commit_epoch.as_u64() > our_start.as_u64()
                    && txns
                        .get(other_tx)
                        .is_some_and(|i| i.write_set.contains(&entity))
                {
                    writers.push(*other_tx);
                }
            }

            writers
            // transactions write lock and committed_epochs read lock drop here
        };

        // Register this reader in the SIREAD registry (uses its own sharded locks).
        self.read_registry.record_reader(entity, transaction_id);

        // Apply rw-antidependency edges now that the transactions lock is released.
        for writer in concurrent_writers {
            self.set_rw_edge(transaction_id, writer);
        }

        Ok(())
    }

    /// Commits a transaction with conflict detection.
    ///
    /// # Conflict Detection
    ///
    /// - **All isolation levels**: Write-write conflicts (two transactions writing
    ///   to the same entity) are always detected and cause the second committer to abort.
    ///
    /// - **Serializable only**: Read-write conflicts (SSI validation) are additionally
    ///   checked. If transaction T1 read an entity that another transaction T2 wrote,
    ///   and T2 committed after T1 started, T1 will abort. This prevents write skew.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The transaction is not active
    /// - There's a write-write conflict with another committed transaction
    /// - (Serializable only) There's a read-write conflict (SSI violation)
    pub fn commit(&self, transaction_id: TransactionId) -> Result<EpochId> {
        // Lock ordering: transactions first, then committed_epochs (matches gc()).
        // Both held as write locks to ensure state and epoch are updated atomically,
        // preventing a race where another thread sees state == Committed but the
        // epoch is not yet in committed_epochs.
        let mut txns = self.transactions.write();
        let mut committed = self.committed_epochs.write();

        // First, validate the transaction exists and is active
        let (our_isolation, our_start_epoch, our_write_set, our_read_set) = {
            let info = txns.get(&transaction_id).ok_or_else(|| {
                Error::Transaction(TransactionError::InvalidState(
                    "Transaction not found".to_string(),
                ))
            })?;

            if info.state != TransactionState::Active {
                return Err(Error::Transaction(TransactionError::InvalidState(
                    "Transaction is not active".to_string(),
                )));
            }

            (
                info.isolation_level,
                info.start_epoch,
                info.write_set.clone(),
                info.read_set.clone(),
            )
        };

        // Check for write-write conflicts with transactions that committed
        // after our snapshot (i.e., concurrent writers to the same entities).
        // Transactions committed before our start_epoch are part of our visible
        // snapshot, so overwriting their values is not a conflict.
        for (other_tx, commit_epoch) in committed.iter() {
            if *other_tx != transaction_id && commit_epoch.as_u64() > our_start_epoch.as_u64() {
                // Check if that transaction wrote to any of our entities
                if let Some(other_info) = txns.get(other_tx) {
                    for entity in &our_write_set {
                        if other_info.write_set.contains(entity) {
                            return Err(Error::Transaction(TransactionError::WriteConflict(
                                format!("Write-write conflict on entity {:?}", entity),
                            )));
                        }
                    }
                }
            }
        }

        // SSI validation for Serializable isolation level.
        // Check for read-write conflicts: if we read an entity that another
        // transaction (that committed after we started) wrote, we have a
        // "rw-antidependency" which can cause write skew.
        //
        // With both transactions.write() and committed_epochs.write() held,
        // no concurrent commit can insert into committed_epochs or change
        // transaction state during our validation window. A single pass over
        // committed_epochs is sufficient.
        if our_isolation == IsolationLevel::Serializable && !our_read_set.is_empty() {
            for (other_tx, commit_epoch) in committed.iter() {
                if *other_tx != transaction_id && commit_epoch.as_u64() > our_start_epoch.as_u64() {
                    // Check if that transaction wrote to any entity we read
                    if let Some(other_info) = txns.get(other_tx) {
                        for entity in &our_read_set {
                            if other_info.write_set.contains(entity) {
                                return Err(Error::Transaction(
                                    TransactionError::SerializationFailure(format!(
                                        "Read-write conflict on entity {:?}: \
                                         another transaction modified data we read",
                                        entity
                                    )),
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Commit successful: advance epoch atomically.
        // SeqCst ensures all threads see commits in a consistent total order.
        let commit_epoch = EpochId::new(self.current_epoch.fetch_add(1, Ordering::SeqCst) + 1);

        // Update state and record commit epoch atomically (both write locks held).
        if let Some(info) = txns.get_mut(&transaction_id) {
            info.state = TransactionState::Committed;
        }
        self.active_count.fetch_sub(1, Ordering::Relaxed);
        committed.insert(transaction_id, commit_epoch);

        Ok(commit_epoch)
    }

    /// Aborts a transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active.
    pub fn abort(&self, transaction_id: TransactionId) -> Result<()> {
        let mut txns = self.transactions.write();

        let info = txns.get_mut(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;

        if info.state != TransactionState::Active {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Transaction is not active".to_string(),
            )));
        }

        info.state = TransactionState::Aborted;
        self.active_count.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    /// Returns the write set of a transaction.
    ///
    /// This returns a copy of the entities written by this transaction,
    /// used for rollback to discard uncommitted versions.
    ///
    /// # Errors
    ///
    /// Returns a `TransactionError::InvalidState` if the transaction is not found.
    pub fn get_write_set(&self, transaction_id: TransactionId) -> Result<HashSet<EntityId>> {
        let txns = self.transactions.read();
        let info = txns.get(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;
        Ok(info.write_set.clone())
    }

    /// Returns a copy of the read-set of a transaction (serializable read tracking).
    pub fn read_set(&self, transaction_id: TransactionId) -> HashSet<EntityId> {
        self.transactions
            .read()
            .get(&transaction_id)
            .map(|i| i.read_set.clone())
            .unwrap_or_default()
    }

    /// Replaces the write set of a transaction (used for savepoint rollback).
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not found.
    pub fn reset_write_set(
        &self,
        transaction_id: TransactionId,
        write_set: HashSet<EntityId>,
    ) -> Result<()> {
        let mut txns = self.transactions.write();
        let info = txns.get_mut(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;
        info.write_set = write_set;
        Ok(())
    }

    /// Aborts all active transactions.
    ///
    /// Used during database shutdown.
    pub fn abort_all_active(&self) {
        let mut txns = self.transactions.write();
        for info in txns.values_mut() {
            if info.state == TransactionState::Active {
                info.state = TransactionState::Aborted;
                self.active_count.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    /// Returns the state of a transaction.
    pub fn state(&self, transaction_id: TransactionId) -> Option<TransactionState> {
        self.transactions
            .read()
            .get(&transaction_id)
            .map(|info| info.state)
    }

    /// Returns the start epoch of a transaction.
    pub fn start_epoch(&self, transaction_id: TransactionId) -> Option<EpochId> {
        self.transactions
            .read()
            .get(&transaction_id)
            .map(|info| info.start_epoch)
    }

    /// Returns the current epoch.
    #[must_use]
    pub fn current_epoch(&self) -> EpochId {
        EpochId::new(self.current_epoch.load(Ordering::Acquire))
    }

    /// Synchronizes the epoch counter to at least the given value.
    ///
    /// Used after snapshot import and WAL recovery to align the
    /// TransactionManager epoch with the store epoch.
    pub fn sync_epoch(&self, epoch: EpochId) {
        self.current_epoch
            .fetch_max(epoch.as_u64(), Ordering::SeqCst);
    }

    /// Returns the minimum epoch that must be preserved for active transactions.
    ///
    /// This is used for garbage collection - versions visible at this epoch
    /// must be preserved.
    #[must_use]
    pub fn min_active_epoch(&self) -> EpochId {
        let txns = self.transactions.read();
        txns.values()
            .filter(|info| info.state == TransactionState::Active)
            .map(|info| info.start_epoch)
            .min()
            .unwrap_or_else(|| self.current_epoch())
    }

    /// Returns the number of active transactions.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.transactions
            .read()
            .values()
            .filter(|info| info.state == TransactionState::Active)
            .count()
    }

    /// Cleans up completed transactions that are no longer needed for conflict detection.
    ///
    /// A committed transaction's write set must be preserved until all transactions
    /// that started before its commit have completed. This ensures write-write
    /// conflict detection works correctly.
    ///
    /// Returns the number of transactions cleaned up.
    pub fn gc(&self) -> usize {
        let mut txns = self.transactions.write();
        let mut committed = self.committed_epochs.write();

        // Find the minimum start epoch among active transactions
        let min_active_start = txns
            .values()
            .filter(|info| info.state == TransactionState::Active)
            .map(|info| info.start_epoch)
            .min();

        let initial_count = txns.len();

        // Collect transactions safe to remove
        let to_remove: Vec<TransactionId> = txns
            .iter()
            .filter(|(transaction_id, info)| {
                match info.state {
                    TransactionState::Active => false, // Never remove active transactions
                    TransactionState::Aborted => true, // Always safe to remove aborted transactions
                    TransactionState::Committed => {
                        // Only remove committed transactions if their commit epoch
                        // is older than all active transactions' start epochs
                        if let Some(min_start) = min_active_start {
                            if let Some(commit_epoch) = committed.get(*transaction_id) {
                                // Safe to remove if committed before all active txns started
                                commit_epoch.as_u64() < min_start.as_u64()
                            } else {
                                // No commit epoch recorded, keep it to be safe
                                false
                            }
                        } else {
                            // No active transactions, safe to remove all committed
                            true
                        }
                    }
                }
            })
            .map(|(id, _)| *id)
            .collect();

        for id in &to_remove {
            txns.remove(id);
            committed.remove(id);
        }

        initial_count - txns.len()
    }

    /// Marks a transaction as committed at a specific epoch.
    ///
    /// Used during recovery to restore transaction state.
    pub fn mark_committed(&self, transaction_id: TransactionId, epoch: EpochId) {
        self.committed_epochs.write().insert(transaction_id, epoch);
    }

    /// Returns the last assigned transaction ID.
    ///
    /// Returns `None` if no transactions have been started yet.
    #[must_use]
    pub fn last_assigned_transaction_id(&self) -> Option<TransactionId> {
        let next = self.next_transaction_id.load(Ordering::Relaxed);
        if next > 1 {
            Some(TransactionId::new(next - 1))
        } else {
            None
        }
    }

    /// Returns the commit epoch of a transaction, if committed.
    #[cfg(test)]
    pub fn committed_epoch(&self, transaction_id: TransactionId) -> Option<EpochId> {
        self.committed_epochs.read().get(&transaction_id).copied()
    }

    /// Record a read-write antidependency edge `reader →rw writer` (the reader read a
    /// version the writer overwrites). Sets each flag independently:
    ///
    /// - `reader.out_conflict` is set iff `reader` is Active + Serializable.
    /// - `writer.in_conflict` is set iff `writer` is Active + Serializable.
    ///
    /// This means a committed writer's flag is never set (it is done), but an
    /// active reader's `out_conflict` is still set even when writing to a
    /// committed writer — which is the correct pivot-detection signal.
    pub(crate) fn set_rw_edge(&self, reader: TransactionId, writer: TransactionId) {
        if reader == writer {
            return;
        }
        let mut txns = self.transactions.write();
        if let Some(i) = txns.get_mut(&reader)
            && i.state == TransactionState::Active
            && i.isolation_level == IsolationLevel::Serializable
        {
            i.out_conflict = true;
        }
        if let Some(i) = txns.get_mut(&writer)
            && i.state == TransactionState::Active
            && i.isolation_level == IsolationLevel::Serializable
        {
            i.in_conflict = true;
        }
    }

    /// Returns the `(in_conflict, out_conflict)` flags for a transaction.
    /// Returns `(false, false)` if the transaction is not found.
    #[cfg(test)]
    pub(crate) fn conflict_flags(&self, tx: TransactionId) -> (bool, bool) {
        self.transactions
            .read()
            .get(&tx)
            .map(|i| (i.in_conflict, i.out_conflict))
            .unwrap_or((false, false))
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_begin_commit() {
        let mgr = TransactionManager::new();

        let tx = mgr.begin();
        assert_eq!(mgr.state(tx), Some(TransactionState::Active));

        let commit_epoch = mgr.commit(tx).unwrap();
        assert_eq!(mgr.state(tx), Some(TransactionState::Committed));
        assert!(commit_epoch.as_u64() > 0);
    }

    #[test]
    fn test_begin_abort() {
        let mgr = TransactionManager::new();

        let tx = mgr.begin();
        mgr.abort(tx).unwrap();
        assert_eq!(mgr.state(tx), Some(TransactionState::Aborted));
    }

    #[test]
    fn test_epoch_advancement() {
        let mgr = TransactionManager::new();

        let initial_epoch = mgr.current_epoch();

        let tx = mgr.begin();
        let commit_epoch = mgr.commit(tx).unwrap();

        assert!(mgr.current_epoch().as_u64() > initial_epoch.as_u64());
        assert!(commit_epoch.as_u64() > initial_epoch.as_u64());
    }

    #[test]
    fn test_gc_preserves_needed_write_sets() {
        let mgr = TransactionManager::new();

        let tx1 = mgr.begin();
        let tx2 = mgr.begin();

        mgr.commit(tx1).unwrap();
        // tx2 still active - started before tx1 committed

        assert_eq!(mgr.active_count(), 1);

        // GC should NOT remove tx1 because tx2 might need its write set for conflict detection
        let cleaned = mgr.gc();
        assert_eq!(cleaned, 0);

        // Both transactions should remain
        assert_eq!(mgr.state(tx1), Some(TransactionState::Committed));
        assert_eq!(mgr.state(tx2), Some(TransactionState::Active));
    }

    #[test]
    fn test_gc_removes_old_commits() {
        let mgr = TransactionManager::new();

        // tx1 commits at epoch 1
        let tx1 = mgr.begin();
        mgr.commit(tx1).unwrap();

        // tx2 starts at epoch 1, commits at epoch 2
        let tx2 = mgr.begin();
        mgr.commit(tx2).unwrap();

        // tx3 starts at epoch 2
        let tx3 = mgr.begin();

        // At this point:
        // - tx1 committed at epoch 1, tx3 started at epoch 2 → tx1 commit < tx3 start → safe to GC
        // - tx2 committed at epoch 2, tx3 started at epoch 2 → tx2 commit >= tx3 start → NOT safe
        let cleaned = mgr.gc();
        assert_eq!(cleaned, 1); // Only tx1 removed

        assert_eq!(mgr.state(tx1), None);
        assert_eq!(mgr.state(tx2), Some(TransactionState::Committed)); // Preserved for conflict detection
        assert_eq!(mgr.state(tx3), Some(TransactionState::Active));

        // After tx3 commits, tx2 can be GC'd
        mgr.commit(tx3).unwrap();
        let cleaned = mgr.gc();
        assert_eq!(cleaned, 2); // tx2 and tx3 both cleaned (no active transactions)
    }

    #[test]
    fn test_gc_removes_aborted() {
        let mgr = TransactionManager::new();

        let tx1 = mgr.begin();
        let tx2 = mgr.begin();

        mgr.abort(tx1).unwrap();
        // tx2 still active

        // Aborted transactions are always safe to remove
        let cleaned = mgr.gc();
        assert_eq!(cleaned, 1);

        assert_eq!(mgr.state(tx1), None);
        assert_eq!(mgr.state(tx2), Some(TransactionState::Active));
    }

    #[test]
    fn test_write_tracking() {
        let mgr = TransactionManager::new();

        let tx = mgr.begin();

        // Record writes
        mgr.record_write(tx, NodeId::new(1)).unwrap();
        mgr.record_write(tx, NodeId::new(2)).unwrap();
        mgr.record_write(tx, EdgeId::new(100)).unwrap();

        // Should commit successfully (no conflicts)
        assert!(mgr.commit(tx).is_ok());
    }

    #[test]
    fn test_min_active_epoch() {
        let mgr = TransactionManager::new();

        // No active transactions - should return current epoch
        assert_eq!(mgr.min_active_epoch(), mgr.current_epoch());

        // Start some transactions
        let tx1 = mgr.begin();
        let epoch1 = mgr.start_epoch(tx1).unwrap();

        // Advance epoch
        let tx2 = mgr.begin();
        mgr.commit(tx2).unwrap();

        let _tx3 = mgr.begin();

        // min_active_epoch should be tx1's start epoch (earliest active)
        assert_eq!(mgr.min_active_epoch(), epoch1);
    }

    #[test]
    fn test_abort_all_active() {
        let mgr = TransactionManager::new();

        let tx1 = mgr.begin();
        let tx2 = mgr.begin();
        let tx3 = mgr.begin();

        mgr.commit(tx1).unwrap();
        // tx2 and tx3 still active

        mgr.abort_all_active();

        assert_eq!(mgr.state(tx1), Some(TransactionState::Committed)); // Already committed
        assert_eq!(mgr.state(tx2), Some(TransactionState::Aborted));
        assert_eq!(mgr.state(tx3), Some(TransactionState::Aborted));
    }

    #[test]
    fn test_start_epoch_snapshot() {
        let mgr = TransactionManager::new();

        // Start epoch for tx1
        let tx1 = mgr.begin();
        let start1 = mgr.start_epoch(tx1).unwrap();

        // Commit tx1, advancing epoch
        mgr.commit(tx1).unwrap();

        // Start tx2 after epoch advanced
        let tx2 = mgr.begin();
        let start2 = mgr.start_epoch(tx2).unwrap();

        // tx2 should have a later start epoch
        assert!(start2.as_u64() > start1.as_u64());
    }

    #[test]
    fn test_record_entity_no_conflict_and_in_write_set() {
        let mgr = TransactionManager::new();
        let tx1 = mgr.begin();
        let tx2 = mgr.begin();
        let entity = NodeId::new(7);

        // record_entity adds to the write-set WITHOUT conflict detection: both
        // transactions can record the same entity (record_write would reject the
        // second). This keeps the write-set a complete scoping record.
        mgr.record_entity(tx1, entity).unwrap();
        mgr.record_entity(tx2, entity).unwrap();

        assert!(
            mgr.get_write_set(tx1)
                .unwrap()
                .contains(&EntityId::Node(entity))
        );
        assert!(
            mgr.get_write_set(tx2)
                .unwrap()
                .contains(&EntityId::Node(entity))
        );
    }

    #[test]
    fn test_write_write_conflict_detection() {
        let mgr = TransactionManager::new();

        // Both transactions start at the same epoch
        let tx1 = mgr.begin();
        let tx2 = mgr.begin();

        // First writer succeeds
        let entity = NodeId::new(42);
        mgr.record_write(tx1, entity).unwrap();

        // Second writer is rejected immediately (first-writer-wins)
        let result = mgr.record_write(tx2, entity);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Write-write conflict"),
            "Expected write-write conflict error"
        );

        // First commit succeeds (no conflict at commit time either)
        let result1 = mgr.commit(tx1);
        assert!(result1.is_ok());
    }

    #[test]
    fn test_commit_epoch_monotonicity() {
        let mgr = TransactionManager::new();

        let mut epochs = Vec::new();

        // Commit multiple transactions and verify epochs are strictly increasing
        for _ in 0..10 {
            let tx = mgr.begin();
            let epoch = mgr.commit(tx).unwrap();
            epochs.push(epoch.as_u64());
        }

        // Verify strict monotonicity
        for i in 1..epochs.len() {
            assert!(
                epochs[i] > epochs[i - 1],
                "Epoch {} ({}) should be greater than epoch {} ({})",
                i,
                epochs[i],
                i - 1,
                epochs[i - 1]
            );
        }
    }

    #[test]
    fn test_concurrent_commits_via_threads() {
        use std::sync::Arc;
        use std::thread;

        let mgr = Arc::new(TransactionManager::new());
        let num_threads = 10;
        let commits_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let mgr = Arc::clone(&mgr);
                thread::spawn(move || {
                    let mut epochs = Vec::new();
                    for _ in 0..commits_per_thread {
                        let tx = mgr.begin();
                        let epoch = mgr.commit(tx).unwrap();
                        epochs.push(epoch.as_u64());
                    }
                    epochs
                })
            })
            .collect();

        let mut all_epochs: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();

        // All epochs should be unique (no duplicates)
        all_epochs.sort_unstable();
        let unique_count = all_epochs.len();
        all_epochs.dedup();
        assert_eq!(
            all_epochs.len(),
            unique_count,
            "All commit epochs should be unique"
        );

        // Final epoch should equal number of commits
        // num_threads=10, commits_per_thread=100, product is 1000
        // reason: value is non-negative by preceding validation
        #[allow(clippy::cast_sign_loss)]
        let expected_epoch = (num_threads * commits_per_thread) as u64;
        assert_eq!(
            mgr.current_epoch().as_u64(),
            expected_epoch,
            "Final epoch should equal total commits"
        );
    }

    #[test]
    fn test_isolation_level_default() {
        let mgr = TransactionManager::new();

        let tx = mgr.begin();
        assert_eq!(
            mgr.isolation_level(tx),
            Some(IsolationLevel::SnapshotIsolation)
        );
    }

    #[test]
    fn test_isolation_level_explicit() {
        let mgr = TransactionManager::new();

        let transaction_rc = mgr.begin_with_isolation(IsolationLevel::ReadCommitted);
        let transaction_si = mgr.begin_with_isolation(IsolationLevel::SnapshotIsolation);
        let transaction_ser = mgr.begin_with_isolation(IsolationLevel::Serializable);

        assert_eq!(
            mgr.isolation_level(transaction_rc),
            Some(IsolationLevel::ReadCommitted)
        );
        assert_eq!(
            mgr.isolation_level(transaction_si),
            Some(IsolationLevel::SnapshotIsolation)
        );
        assert_eq!(
            mgr.isolation_level(transaction_ser),
            Some(IsolationLevel::Serializable)
        );
    }

    #[test]
    fn test_ssi_read_write_conflict_detected() {
        let mgr = TransactionManager::new();

        // tx1 starts with Serializable isolation
        let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);

        // tx2 starts and will modify an entity
        let tx2 = mgr.begin();

        // tx1 reads entity 42
        let entity = NodeId::new(42);
        mgr.record_read(tx1, entity).unwrap();

        // tx2 writes to the same entity and commits
        mgr.record_write(tx2, entity).unwrap();
        mgr.commit(tx2).unwrap();

        // tx1 tries to commit - should fail due to SSI read-write conflict
        let result = mgr.commit(tx1);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Serialization failure"),
            "Expected serialization failure error"
        );
    }

    #[test]
    fn test_ssi_no_conflict_when_not_serializable() {
        let mgr = TransactionManager::new();

        // tx1 starts with default Snapshot Isolation
        let tx1 = mgr.begin();

        // tx2 starts and will modify an entity
        let tx2 = mgr.begin();

        // tx1 reads entity 42
        let entity = NodeId::new(42);
        mgr.record_read(tx1, entity).unwrap();

        // tx2 writes to the same entity and commits
        mgr.record_write(tx2, entity).unwrap();
        mgr.commit(tx2).unwrap();

        // tx1 should commit successfully (SI doesn't check read-write conflicts)
        let result = mgr.commit(tx1);
        assert!(
            result.is_ok(),
            "Snapshot Isolation should not detect read-write conflicts"
        );
    }

    #[test]
    fn test_ssi_no_conflict_when_write_before_read() {
        let mgr = TransactionManager::new();

        // tx1 writes and commits first
        let tx1 = mgr.begin();
        let entity = NodeId::new(42);
        mgr.record_write(tx1, entity).unwrap();
        mgr.commit(tx1).unwrap();

        // tx2 starts AFTER tx1 committed and reads the entity
        let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        mgr.record_read(tx2, entity).unwrap();

        // tx2 should commit successfully (tx1 committed before tx2 started)
        let result = mgr.commit(tx2);
        assert!(
            result.is_ok(),
            "Should not conflict when writer committed before reader started"
        );
    }

    #[test]
    fn test_write_skew_prevented_by_ssi() {
        // Classic write skew scenario:
        // Account A = 50, Account B = 50, constraint: A + B >= 0
        // T1 reads A, B, writes A = A - 100
        // T2 reads A, B, writes B = B - 100
        // Without SSI, both could commit violating the constraint.

        let mgr = TransactionManager::new();

        let account_a = NodeId::new(1);
        let account_b = NodeId::new(2);

        // T1 and T2 both start with Serializable isolation
        let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

        // Both read both accounts
        mgr.record_read(tx1, account_a).unwrap();
        mgr.record_read(tx1, account_b).unwrap();
        mgr.record_read(tx2, account_a).unwrap();
        mgr.record_read(tx2, account_b).unwrap();

        // T1 writes to A, T2 writes to B (no write-write conflict)
        mgr.record_write(tx1, account_a).unwrap();
        mgr.record_write(tx2, account_b).unwrap();

        // T1 commits first
        let result1 = mgr.commit(tx1);
        assert!(result1.is_ok(), "First commit should succeed");

        // T2 tries to commit - should fail because it read account_a which T1 wrote
        let result2 = mgr.commit(tx2);
        assert!(result2.is_err(), "Second commit should fail due to SSI");
        assert!(
            result2
                .unwrap_err()
                .to_string()
                .contains("Serialization failure"),
            "Expected serialization failure error for write skew prevention"
        );
    }

    #[test]
    fn test_read_committed_allows_non_repeatable_reads() {
        let mgr = TransactionManager::new();

        // tx1 starts with ReadCommitted isolation
        let tx1 = mgr.begin_with_isolation(IsolationLevel::ReadCommitted);
        let entity = NodeId::new(42);

        // tx1 reads entity
        mgr.record_read(tx1, entity).unwrap();

        // tx2 writes and commits
        let tx2 = mgr.begin();
        mgr.record_write(tx2, entity).unwrap();
        mgr.commit(tx2).unwrap();

        // tx1 can still commit (ReadCommitted allows non-repeatable reads)
        let result = mgr.commit(tx1);
        assert!(
            result.is_ok(),
            "ReadCommitted should allow non-repeatable reads"
        );
    }

    #[test]
    fn test_isolation_level_debug() {
        assert_eq!(
            format!("{:?}", IsolationLevel::ReadCommitted),
            "ReadCommitted"
        );
        assert_eq!(
            format!("{:?}", IsolationLevel::SnapshotIsolation),
            "SnapshotIsolation"
        );
        assert_eq!(
            format!("{:?}", IsolationLevel::Serializable),
            "Serializable"
        );
    }

    #[test]
    fn test_isolation_level_default_trait() {
        let default: IsolationLevel = Default::default();
        assert_eq!(default, IsolationLevel::SnapshotIsolation);
    }

    #[test]
    fn test_ssi_concurrent_reads_no_conflict() {
        let mgr = TransactionManager::new();

        let entity = NodeId::new(42);

        // Both transactions read the same entity
        let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

        mgr.record_read(tx1, entity).unwrap();
        mgr.record_read(tx2, entity).unwrap();

        // Both should commit successfully (read-read is not a conflict)
        assert!(mgr.commit(tx1).is_ok());
        assert!(mgr.commit(tx2).is_ok());
    }

    #[test]
    fn test_ssi_write_write_conflict() {
        let mgr = TransactionManager::new();

        let entity = NodeId::new(42);

        // Both transactions attempt to write the same entity
        let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

        // First writer succeeds
        mgr.record_write(tx1, entity).unwrap();

        // Second writer is rejected immediately (first-writer-wins)
        let result = mgr.record_write(tx2, entity);
        assert!(
            result.is_err(),
            "Second record_write should fail with write-write conflict"
        );

        // First commit succeeds
        assert!(mgr.commit(tx1).is_ok());
    }

    #[test]
    fn test_ssi_concurrent_commit_race() {
        // Regression test: with the old read-then-upgrade lock pattern,
        // two concurrent SSI commits could both succeed when one should
        // have been aborted due to a read-write conflict (write skew).
        use std::sync::Arc;

        let mgr = Arc::new(TransactionManager::new());

        // Run many iterations to exercise the race window
        for _ in 0..100 {
            let entity_a = NodeId::new(1);
            let entity_b = NodeId::new(2);

            // Classic write skew setup: both transactions read both entities,
            // then each writes to a different one.
            let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
            let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

            mgr.record_read(tx1, entity_a).unwrap();
            mgr.record_read(tx1, entity_b).unwrap();
            mgr.record_read(tx2, entity_a).unwrap();
            mgr.record_read(tx2, entity_b).unwrap();

            mgr.record_write(tx1, entity_a).unwrap();
            mgr.record_write(tx2, entity_b).unwrap();

            // Commit tx1 first so it's in committed_epochs
            mgr.commit(tx1).unwrap();

            // tx2 should be rejected: it read entity_a which tx1 wrote
            let result = mgr.commit(tx2);
            assert!(
                result.is_err(),
                "SSI should detect read-write conflict on entity_a"
            );

            // Abort the rejected transaction so its write set is cleared
            // before the next iteration.
            let _ = mgr.abort(tx2);
            mgr.gc();
        }
    }

    #[test]
    fn test_ssi_concurrent_commit_barrier() {
        // Stress test with barrier synchronization to maximize the chance
        // of concurrent commit() calls overlapping.
        use std::sync::{Arc, Barrier};
        use std::thread;

        let mgr = Arc::new(TransactionManager::new());
        let mut both_ok_count = 0;

        for _ in 0..50 {
            let entity_a = NodeId::new(1);
            let entity_b = NodeId::new(2);

            let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
            let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

            mgr.record_read(tx1, entity_a).unwrap();
            mgr.record_read(tx1, entity_b).unwrap();
            mgr.record_read(tx2, entity_a).unwrap();
            mgr.record_read(tx2, entity_b).unwrap();

            mgr.record_write(tx1, entity_a).unwrap();
            mgr.record_write(tx2, entity_b).unwrap();

            let mgr1 = Arc::clone(&mgr);
            let mgr2 = Arc::clone(&mgr);
            let barrier = Arc::new(Barrier::new(2));
            let b1 = Arc::clone(&barrier);
            let b2 = Arc::clone(&barrier);

            let h1 = thread::spawn(move || {
                b1.wait();
                mgr1.commit(tx1)
            });
            let h2 = thread::spawn(move || {
                b2.wait();
                mgr2.commit(tx2)
            });

            let r1 = h1.join().unwrap();
            let r2 = h2.join().unwrap();

            if r1.is_ok() && r2.is_ok() {
                both_ok_count += 1;
            }

            // Clean up
            if r1.is_err() {
                let _ = mgr.abort(tx1);
            }
            if r2.is_err() {
                let _ = mgr.abort(tx2);
            }
            mgr.gc();
        }

        // At most one should succeed per iteration (write skew prevention).
        // With the fix, both_ok_count should always be 0.
        assert_eq!(
            both_ok_count, 0,
            "SSI must prevent both concurrent write-skew commits from succeeding"
        );
    }

    #[test]
    fn test_committed_epoch_present_after_commit() {
        // Verify that after commit(), the committed_epochs entry is always
        // present (no window where state is Committed but epoch is missing).
        let mgr = TransactionManager::new();

        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(1)).unwrap();
        let epoch = mgr.commit(tx).unwrap();

        // committed_epoch must be available immediately after commit returns
        assert_eq!(
            mgr.committed_epoch(tx),
            Some(epoch),
            "committed_epochs must contain tx immediately after commit()"
        );
    }

    // --- F2 incremental SSI: rw-conflict flags ---

    #[test]
    fn test_rw_conflict_flags_initial_false() {
        // Both flags start as false for any new Serializable transaction.
        let mgr = TransactionManager::new();
        let t1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let t2 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        assert_eq!(mgr.conflict_flags(t1), (false, false));
        assert_eq!(mgr.conflict_flags(t2), (false, false));
    }

    #[test]
    fn test_set_rw_edge_sets_reader_out_and_writer_in() {
        // set_rw_edge(t1, t2): t1 is the reader, t2 is the writer.
        // t1.out_conflict must become true; t2.in_conflict must become true.
        let mgr = TransactionManager::new();
        let t1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let t2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

        mgr.set_rw_edge(t1, t2);

        // t1 is the reader end: out_conflict = true, in_conflict unchanged (false)
        assert_eq!(
            mgr.conflict_flags(t1),
            (false, true),
            "reader t1 must have out_conflict=true"
        );
        // t2 is the writer end: in_conflict = true, out_conflict unchanged (false)
        assert_eq!(
            mgr.conflict_flags(t2),
            (true, false),
            "writer t2 must have in_conflict=true"
        );
    }

    #[test]
    fn test_set_rw_edge_noop_for_non_serializable() {
        // Flags are set independently per transaction: an SI transaction never gets
        // a flag, but the Serializable peer's flag IS set if it is Active+Ser.
        let mgr = TransactionManager::new();
        let t_ser = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let t_si = mgr.begin_with_isolation(IsolationLevel::SnapshotIsolation);

        // t_ser reads, t_si writes:
        //   t_ser is Active+Ser → out_conflict set.
        //   t_si is SI → no in_conflict.
        mgr.set_rw_edge(t_ser, t_si);
        assert_eq!(
            mgr.conflict_flags(t_ser),
            (false, true),
            "Serializable reader gets out_conflict even when writer is SI"
        );
        assert_eq!(
            mgr.conflict_flags(t_si),
            (false, false),
            "SI writer never gets a flag"
        );

        // t_si reads, t_ser writes:
        //   t_si is SI → no out_conflict.
        //   t_ser is Active+Ser → in_conflict set (it was already out=true above).
        mgr.set_rw_edge(t_si, t_ser);
        assert_eq!(
            mgr.conflict_flags(t_ser),
            (true, true),
            "Serializable writer gets in_conflict; out_conflict already set above"
        );
        assert_eq!(
            mgr.conflict_flags(t_si),
            (false, false),
            "SI reader never gets a flag"
        );
    }

    #[test]
    fn test_set_rw_edge_self_loop_is_noop() {
        // set_rw_edge(t, t) must be silently ignored.
        let mgr = TransactionManager::new();
        let t = mgr.begin_with_isolation(IsolationLevel::Serializable);
        mgr.set_rw_edge(t, t);
        assert_eq!(mgr.conflict_flags(t), (false, false));
    }

    #[test]
    fn test_set_rw_edge_committed_reader_no_flag() {
        // set_rw_edge(committed_reader, active_writer): the committed reader
        // cannot receive out_conflict (it is done); the active+Ser writer gets
        // in_conflict independently.
        let mgr = TransactionManager::new();
        let t_reader = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let t_writer = mgr.begin_with_isolation(IsolationLevel::Serializable);

        // Commit the reader before recording the edge
        mgr.commit(t_reader).unwrap();

        // t_reader is committed → no out_conflict; t_writer is Active+Ser → in_conflict.
        mgr.set_rw_edge(t_reader, t_writer);
        assert_eq!(
            mgr.conflict_flags(t_reader),
            (false, false),
            "committed reader must never get out_conflict"
        );
        assert_eq!(
            mgr.conflict_flags(t_writer),
            (true, false),
            "active Serializable writer gets in_conflict independently"
        );
    }

    #[test]
    fn test_set_rw_edge_reader_active_writer_committed() {
        // The key case for read-time detection: T_reader is active+Ser and
        // T_writer already committed.  reader.out_conflict must be set; the
        // committed writer's in_conflict is moot and must stay false.
        let mgr = TransactionManager::new();
        let t_writer = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let t_reader = mgr.begin_with_isolation(IsolationLevel::Serializable);

        mgr.commit(t_writer).unwrap();

        // Edge: t_reader →rw t_writer (t_reader read something t_writer already wrote)
        mgr.set_rw_edge(t_reader, t_writer);
        assert_eq!(
            mgr.conflict_flags(t_reader),
            (false, true),
            "active Serializable reader gets out_conflict even for a committed writer"
        );
        assert_eq!(
            mgr.conflict_flags(t_writer),
            (false, false),
            "committed writer in_conflict stays false"
        );
    }

    // --- F2 Task 3: read-registry feed + read-time rw-edge detection ---

    #[test]
    fn test_read_after_concurrent_committed_write_sets_reader_out() {
        // t_w (Serializable) writes E and commits; t_r (Serializable) began BEFORE
        // t_w committed (lower start_epoch) then reads E → t_r.out_conflict=true,
        // t_w is done so its in_conflict stays false.
        let mgr = TransactionManager::new();

        // t_r begins first (start_epoch = 0)
        let t_r = mgr.begin_with_isolation(IsolationLevel::Serializable);

        // t_w begins, writes E, commits (commit_epoch > t_r.start_epoch = 0)
        let t_w = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let entity = NodeId::new(1);
        mgr.record_write(t_w, entity).unwrap();
        mgr.commit(t_w).unwrap();

        // t_r now reads E; t_w committed after t_r started → concurrent writer
        mgr.record_read(t_r, entity).unwrap();

        assert_eq!(
            mgr.conflict_flags(t_r),
            (false, true),
            "t_r must have out_conflict: it read a version t_w (concurrent committed) overwrote"
        );
        // t_w committed; its in_conflict was never set (and is moot)
        assert_eq!(
            mgr.conflict_flags(t_w),
            (false, false),
            "committed t_w in_conflict stays false"
        );
    }

    #[test]
    fn test_read_after_concurrent_active_write_sets_both() {
        // t_w (Serializable, active) record_write(E); t_r (Serializable, active)
        // record_read(E) → t_r.out_conflict=true AND t_w.in_conflict=true.
        let mgr = TransactionManager::new();

        let t_w = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let t_r = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let entity = NodeId::new(2);

        mgr.record_write(t_w, entity).unwrap();
        mgr.record_read(t_r, entity).unwrap();

        assert_eq!(
            mgr.conflict_flags(t_r),
            (false, true),
            "t_r (reader) must have out_conflict"
        );
        assert_eq!(
            mgr.conflict_flags(t_w),
            (true, false),
            "t_w (active writer) must have in_conflict"
        );
    }

    #[test]
    fn test_read_of_unwritten_entity_no_edge() {
        // t_r reads E that nobody has written → no flags; t_r IS registered in
        // read_registry.
        let mgr = TransactionManager::new();
        let t_r = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let entity = EntityId::Node(NodeId::new(42));

        mgr.record_read(t_r, entity).unwrap();

        assert_eq!(
            mgr.conflict_flags(t_r),
            (false, false),
            "no rw edge when nobody wrote the entity"
        );

        // The reader must be registered in the read_registry.
        let readers = mgr.read_registry.readers_of(entity);
        assert!(
            readers.contains(&t_r),
            "t_r must be registered in read_registry after record_read"
        );
    }

    #[test]
    fn test_non_serializable_read_no_registry_no_edges() {
        // An SI transaction reads E that a Serializable writer wrote → no flags,
        // not in registry.
        let mgr = TransactionManager::new();

        let t_w = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let entity = NodeId::new(7);
        mgr.record_write(t_w, entity).unwrap();

        let t_si = mgr.begin_with_isolation(IsolationLevel::SnapshotIsolation);
        mgr.record_read(t_si, entity).unwrap();

        assert_eq!(
            mgr.conflict_flags(t_si),
            (false, false),
            "SI reader must not get any conflict flags"
        );
        assert_eq!(
            mgr.conflict_flags(t_w),
            (false, false),
            "writer must not get in_conflict from a non-Serializable reader"
        );

        // SI reader must NOT be in the read_registry.
        let readers = mgr.read_registry.readers_of(EntityId::Node(entity));
        assert!(
            !readers.contains(&t_si),
            "SI reader must not be registered in read_registry"
        );
    }

    // --- F2 Task 4: write-time rw-edge detection via read-registry (mirror direction) ---

    #[test]
    fn test_write_after_concurrent_read_sets_reader_out_writer_in() {
        // t_r (Serializable, active) record_read(E) registers it in the
        // read_registry; t_w (Serializable, active) record_write(E) must detect
        // t_r as a concurrent reader and set the rw-edge: reader.out_conflict=true,
        // writer.in_conflict=true.
        let mgr = TransactionManager::new();
        let entity = NodeId::new(10);

        let t_r = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let t_w = mgr.begin_with_isolation(IsolationLevel::Serializable);

        // t_r reads E first → registered in read_registry
        mgr.record_read(t_r, entity).unwrap();
        // t_w writes E → discovers t_r as concurrent reader → sets rw-edge
        mgr.record_write(t_w, entity).unwrap();

        assert_eq!(
            mgr.conflict_flags(t_r),
            (false, true),
            "t_r (reader) must have out_conflict after write-time detection"
        );
        assert_eq!(
            mgr.conflict_flags(t_w),
            (true, false),
            "t_w (writer) must have in_conflict after write-time detection"
        );
    }

    #[test]
    fn test_write_with_no_concurrent_readers_no_edge() {
        // t_w writes E that nobody has read → no flags set.
        let mgr = TransactionManager::new();
        let entity = NodeId::new(20);

        let t_w = mgr.begin_with_isolation(IsolationLevel::Serializable);
        mgr.record_write(t_w, entity).unwrap();

        assert_eq!(
            mgr.conflict_flags(t_w),
            (false, false),
            "no rw edge when nobody read the entity"
        );
    }

    #[test]
    fn test_non_serializable_write_no_edges() {
        // An SI writer does not participate in SSI cycle detection; detection
        // fires only for Serializable actors on the acting side.
        //
        // Choice: we skip write-time detection when the WRITER is not Serializable
        // (symmetric with Task 3's choice to skip when the READER is not
        // Serializable). An SI writer's overwrite does not form an SSI rw-edge.
        let mgr = TransactionManager::new();
        let entity = NodeId::new(30);

        // t_r is Serializable and reads E → registered in read_registry
        let t_r = mgr.begin_with_isolation(IsolationLevel::Serializable);
        mgr.record_read(t_r, entity).unwrap();

        // t_si is SI and writes E → does NOT trigger write-time detection
        let t_si = mgr.begin_with_isolation(IsolationLevel::SnapshotIsolation);
        mgr.record_write(t_si, entity).unwrap();

        // The SI writer gets no flag
        assert_eq!(
            mgr.conflict_flags(t_si),
            (false, false),
            "SI writer must not get any conflict flags"
        );
        // The Serializable reader's flags are unchanged by the SI write
        assert_eq!(
            mgr.conflict_flags(t_r),
            (false, false),
            "Serializable reader must not get out_conflict from a non-Serializable writer"
        );
    }

    #[test]
    fn test_write_does_not_self_edge() {
        // A transaction that reads E then writes E is in the read_registry for E,
        // but the `reader != tx` guard must skip itself → no self-edge.
        let mgr = TransactionManager::new();
        let entity = NodeId::new(40);

        let tx = mgr.begin_with_isolation(IsolationLevel::Serializable);
        // Read first (registers in read_registry)
        mgr.record_read(tx, entity).unwrap();
        // Write same entity → readers_of returns `tx` itself, but guard skips it
        mgr.record_write(tx, entity).unwrap();

        assert_eq!(
            mgr.conflict_flags(tx),
            (false, false),
            "a tx that reads then writes the same entity must not self-edge"
        );
    }
}
