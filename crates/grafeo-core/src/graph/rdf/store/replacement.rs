//! Allocation-complete exact RDF replacement with retained publication writers.

use parking_lot::{MutexGuard, RwLockWriteGuard};

use super::super::{Term, Triple};
use super::{
    CanonicalTripleKey, NamedGraphHistory, QuadLife, RdfCommitGuard, RdfStore, TransactionBuffer,
};
use grafeo_common::types::{EpochId, GraphIncarnationId, HistoryCompleteness, StoreId};
use grafeo_common::utils::hash::FxHashSet;
use hashbrown::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

type TermIndex = hashbrown::HashMap<Term, Vec<Arc<Triple>>, foldhash::fast::RandomState>;
type PairIndex = hashbrown::HashMap<(Term, Term), Vec<Arc<Triple>>, foldhash::fast::RandomState>;

struct ReplacementState {
    triples: HashMap<CanonicalTripleKey, Arc<Triple>>,
    subject_index: TermIndex,
    predicate_index: TermIndex,
    object_index: Option<TermIndex>,
    sp_index: PairIndex,
    po_index: PairIndex,
    os_index: PairIndex,
    history: HashMap<Arc<Triple>, Vec<QuadLife>>,
    named_graphs: HashMap<String, Arc<RdfStore>>,
    named_graph_history: Vec<NamedGraphHistory>,
    store_id: StoreId,
    history_completeness: HistoryCompleteness,
    reserved: FxHashSet<GraphIncarnationId>,
    tx_buffer: TransactionBuffer,
    projections: HashMap<u64, (String, String, Option<EpochId>)>,
    statistics_cache: Option<Arc<crate::statistics::RdfStatistics>>,
    dictionary_cache: Option<Arc<super::super::dictionary::TermDictionary>>,
    #[cfg(feature = "ring-index")]
    ring: Option<Arc<crate::index::ring::TripleRing>>,
}

/// Exact private dataset state, retained after installation to retire old data
/// only after all companion publication guards have been released.
#[must_use = "prepare a ready replacement before publishing companion state"]
pub struct PreparedRdfDatasetReplacement<'store> {
    target: &'store RdfStore,
    state: ReplacementState,
    next_graph_incarnation: u64,
    commit_epoch: EpochId,
    installed: bool,
}

struct ReplacementWriters<'store> {
    triples: RwLockWriteGuard<'store, HashMap<CanonicalTripleKey, Arc<Triple>>>,
    subject_index: RwLockWriteGuard<'store, TermIndex>,
    predicate_index: RwLockWriteGuard<'store, TermIndex>,
    object_index: RwLockWriteGuard<'store, Option<TermIndex>>,
    sp_index: RwLockWriteGuard<'store, PairIndex>,
    po_index: RwLockWriteGuard<'store, PairIndex>,
    os_index: RwLockWriteGuard<'store, PairIndex>,
    history: RwLockWriteGuard<'store, HashMap<Arc<Triple>, Vec<QuadLife>>>,
    named_graphs: RwLockWriteGuard<'store, HashMap<String, Arc<RdfStore>>>,
    named_graph_history: RwLockWriteGuard<'store, Vec<NamedGraphHistory>>,
    store_id: RwLockWriteGuard<'store, StoreId>,
    history_completeness: RwLockWriteGuard<'store, HistoryCompleteness>,
    reserved: RwLockWriteGuard<'store, FxHashSet<GraphIncarnationId>>,
    tx_buffer: RwLockWriteGuard<'store, TransactionBuffer>,
    projections: RwLockWriteGuard<'store, HashMap<u64, (String, String, Option<EpochId>)>>,
    statistics_cache: RwLockWriteGuard<'store, Option<Arc<crate::statistics::RdfStatistics>>>,
    dictionary_cache:
        RwLockWriteGuard<'store, Option<Arc<super::super::dictionary::TermDictionary>>>,
    #[cfg(feature = "ring-index")]
    ring: RwLockWriteGuard<'store, Option<Arc<crate::index::ring::TripleRing>>>,
}

/// All RDF writers are held and every fallible replacement check is complete.
#[must_use = "install only after every companion is ready"]
pub struct ReadyRdfDatasetReplacement<'ready, 'store> {
    prepared: &'ready mut PreparedRdfDatasetReplacement<'store>,
    writers: ReplacementWriters<'store>,
    _lifecycle: MutexGuard<'store, ()>,
    _commit: &'ready RdfCommitGuard<'store>,
    next_revision: u64,
}

/// Installed replacement retaining all RDF writers through companion publication.
#[must_use = "release the RDF writers after all companion state is installed"]
pub struct InstalledRdfDatasetReplacement<'ready, 'store> {
    ready: ReadyRdfDatasetReplacement<'ready, 'store>,
}

impl<'store> PreparedRdfDatasetReplacement<'store> {
    pub(super) fn new(
        target: &'store RdfStore,
        replacement: RdfStore,
        reserved: FxHashSet<GraphIncarnationId>,
        next_graph_incarnation: u64,
        commit_epoch: EpochId,
    ) -> Self {
        Self {
            target,
            state: ReplacementState {
                triples: replacement.triples.into_inner(),
                subject_index: replacement.subject_index.into_inner(),
                predicate_index: replacement.predicate_index.into_inner(),
                object_index: replacement.object_index.into_inner(),
                sp_index: replacement.sp_index.into_inner(),
                po_index: replacement.po_index.into_inner(),
                os_index: replacement.os_index.into_inner(),
                history: replacement.history.into_inner(),
                named_graphs: replacement.named_graphs.into_inner(),
                named_graph_history: replacement.named_graph_history.into_inner(),
                store_id: replacement.store_id.into_inner(),
                history_completeness: replacement.history_completeness.into_inner(),
                reserved,
                tx_buffer: replacement.tx_buffer.into_inner(),
                projections: replacement.projections.into_inner(),
                statistics_cache: replacement.statistics_cache.into_inner(),
                dictionary_cache: replacement.dictionary_cache.into_inner(),
                #[cfg(feature = "ring-index")]
                ring: replacement.ring.into_inner(),
            },
            next_graph_incarnation,
            commit_epoch,
            installed: false,
        }
    }

    /// Acquires every RDF storage writer before any companion is published.
    ///
    /// The scoped commit gate must belong to the exact target. The workspace
    /// remains reusable after a failed or abandoned preparation.
    ///
    /// # Errors
    ///
    /// Returns an error for missing authority, a foreign commit guard, a busy
    /// storage writer, active RDF transaction state, incompatible descendant
    /// authority or revision exhaustion.
    pub fn ready<'ready>(
        &'ready mut self,
        commit: &'ready RdfCommitGuard<'store>,
    ) -> Result<ReadyRdfDatasetReplacement<'ready, 'store>, String> {
        if self.installed {
            return Err("RDF dataset replacement has already been installed".into());
        }
        if !std::ptr::eq(commit.store, self.target) {
            return Err("RDF commit guard belongs to a different store".into());
        }
        if !self.target.unframed_writes_allowed() {
            return Err("RDF dataset history replacement lacks mutation authority".into());
        }
        let lifecycle = self
            .target
            .history_lifecycle_lock
            .try_lock()
            .ok_or_else(|| "RDF history lifecycle is busy during replacement".to_string())?;
        let scope = self.target.mutation_scope.load(Ordering::Acquire);
        if scope != 0 {
            for archive in &self.state.named_graph_history {
                if !archive.store.seal_with_scope(scope) {
                    return Err("failed to apply RDF history mutation scope".into());
                }
            }
        }
        let next_revision = self
            .target
            .lifecycle_revision
            .load(Ordering::Acquire)
            .checked_add(1)
            .ok_or_else(|| "RDF replacement revision space exhausted".to_string())?;
        let busy = || "RDF storage is busy during replacement preparation".to_string();
        let writers = ReplacementWriters {
            triples: self.target.triples.try_write().ok_or_else(busy)?,
            subject_index: self.target.subject_index.try_write().ok_or_else(busy)?,
            predicate_index: self.target.predicate_index.try_write().ok_or_else(busy)?,
            object_index: self.target.object_index.try_write().ok_or_else(busy)?,
            sp_index: self.target.sp_index.try_write().ok_or_else(busy)?,
            po_index: self.target.po_index.try_write().ok_or_else(busy)?,
            os_index: self.target.os_index.try_write().ok_or_else(busy)?,
            history: self.target.history.try_write().ok_or_else(busy)?,
            named_graphs: self.target.named_graphs.try_write().ok_or_else(busy)?,
            named_graph_history: self
                .target
                .named_graph_history
                .try_write()
                .ok_or_else(busy)?,
            store_id: self.target.store_id.try_write().ok_or_else(busy)?,
            history_completeness: self
                .target
                .history_completeness
                .try_write()
                .ok_or_else(busy)?,
            reserved: self
                .target
                .reserved_graph_incarnations
                .try_write()
                .ok_or_else(busy)?,
            tx_buffer: self.target.tx_buffer.try_write().ok_or_else(busy)?,
            projections: self.target.projections.try_write().ok_or_else(busy)?,
            statistics_cache: self.target.statistics_cache.try_write().ok_or_else(busy)?,
            dictionary_cache: self.target.dictionary_cache.try_write().ok_or_else(busy)?,
            #[cfg(feature = "ring-index")]
            ring: self.target.ring.try_write().ok_or_else(busy)?,
        };
        let transactions = &*writers.tx_buffer;
        if !transactions.buffers.is_empty()
            || !transactions.snapshot_epochs.is_empty()
            || !transactions.created_graphs.is_empty()
            || !transactions.dropped_graphs.is_empty()
            || !transactions.touched_graphs.is_empty()
            || !transactions.read_graphs.is_empty()
            || !transactions.snapshotted_graph_catalogs.is_empty()
        {
            return Err("RDF dataset replacement requires no pending transaction state".into());
        }
        Ok(ReadyRdfDatasetReplacement {
            prepared: self,
            writers,
            _lifecycle: lifecycle,
            _commit: commit,
            next_revision,
        })
    }
}

impl<'ready, 'store> ReadyRdfDatasetReplacement<'ready, 'store> {
    /// Swaps the prepared state without allocation, lock acquisition or failure.
    /// Displaced data remains in the outer workspace while all writers stay held.
    pub fn install(mut self) -> InstalledRdfDatasetReplacement<'ready, 'store> {
        let state = &mut self.prepared.state;
        let writers = &mut self.writers;
        std::mem::swap(&mut *writers.triples, &mut state.triples);
        std::mem::swap(&mut *writers.subject_index, &mut state.subject_index);
        std::mem::swap(&mut *writers.predicate_index, &mut state.predicate_index);
        std::mem::swap(&mut *writers.object_index, &mut state.object_index);
        std::mem::swap(&mut *writers.sp_index, &mut state.sp_index);
        std::mem::swap(&mut *writers.po_index, &mut state.po_index);
        std::mem::swap(&mut *writers.os_index, &mut state.os_index);
        std::mem::swap(&mut *writers.history, &mut state.history);
        std::mem::swap(&mut *writers.named_graphs, &mut state.named_graphs);
        std::mem::swap(
            &mut *writers.named_graph_history,
            &mut state.named_graph_history,
        );
        std::mem::swap(&mut *writers.store_id, &mut state.store_id);
        std::mem::swap(
            &mut *writers.history_completeness,
            &mut state.history_completeness,
        );
        std::mem::swap(&mut *writers.reserved, &mut state.reserved);
        std::mem::swap(&mut *writers.tx_buffer, &mut state.tx_buffer);
        std::mem::swap(&mut *writers.projections, &mut state.projections);
        std::mem::swap(&mut *writers.statistics_cache, &mut state.statistics_cache);
        std::mem::swap(&mut *writers.dictionary_cache, &mut state.dictionary_cache);
        #[cfg(feature = "ring-index")]
        std::mem::swap(&mut *writers.ring, &mut state.ring);
        let target = self.prepared.target;
        target.durable_identity.store(true, Ordering::Release);
        target
            .next_graph_incarnation
            .store(self.prepared.next_graph_incarnation, Ordering::Release);
        target
            .commit_epoch
            .store(self.prepared.commit_epoch.as_u64(), Ordering::Release);
        target
            .lifecycle_revision
            .store(self.next_revision, Ordering::Release);
        self.prepared.installed = true;
        #[cfg(feature = "ring-index")]
        target.ring_stale.store(false, Ordering::Release);
        InstalledRdfDatasetReplacement { ready: self }
    }
}

impl InstalledRdfDatasetReplacement<'_, '_> {
    /// Releases RDF writers; the caller still owns the displaced-state workspace.
    pub fn release(self) {
        drop(self.ready);
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::RdfDatasetHistory;
    use super::super::RdfStoreConfig;
    use super::*;
    use grafeo_common::types::TransactionId;

    fn store(byte: u8) -> RdfStore {
        RdfStore::with_config_and_store_id(
            RdfStoreConfig::default(),
            StoreId::from_bytes([byte; 32]).unwrap(),
        )
    }

    fn triple(value: &str) -> Triple {
        Triple::new(
            Term::iri("http://example.org/subject"),
            Term::iri("http://example.org/value"),
            Term::literal(value),
        )
    }

    fn same_history(actual: &RdfDatasetHistory, expected: &RdfDatasetHistory) {
        assert_eq!(actual.store_id(), expected.store_id());
        assert_eq!(actual.completeness(), expected.completeness());
        assert_eq!(
            actual.next_graph_incarnation(),
            expected.next_graph_incarnation()
        );
        assert_eq!(actual.graph_lives(), expected.graph_lives());
        assert_eq!(actual.quad_versions(), expected.quad_versions());
    }

    #[test]
    fn prepared_rdf_replacement_foreign_gate_and_late_contention_leave_live_state_unchanged() {
        let target = store(1);
        target.try_set_commit_epoch(EpochId::new(7)).unwrap();
        assert!(target.insert(triple("old")));
        let before = target.dataset_history().unwrap();
        let source = store(2);
        assert!(source.insert(triple("replacement")));
        let mut prepared = target
            .prepare_dataset_history_replacement(
                source.dataset_history().unwrap(),
                EpochId::INITIAL,
            )
            .unwrap();
        let foreign = source.lock_commit_scoped();
        assert!(prepared.ready(&foreign).is_err());
        drop(foreign);
        let commit = target.lock_commit_scoped();
        let late_reader = target.dictionary_cache.read();
        assert!(prepared.ready(&commit).is_err());
        drop(late_reader);
        assert!(target.triples.try_write().is_some());
        assert!(target.subject_index.try_write().is_some());
        assert!(target.history.try_write().is_some());
        assert!(target.named_graphs.try_write().is_some());
        assert!(target.tx_buffer.try_write().is_some());
        assert!(target.reserved_graph_incarnations.try_write().is_some());
        same_history(
            &target.dataset_history_under_commit_gate().unwrap(),
            &before,
        );
        assert_eq!(target.commit_epoch(), EpochId::new(7));
        // A failed ready attempt leaves the private replacement reusable.
        drop(prepared.ready(&commit).unwrap());
        same_history(
            &target.dataset_history_under_commit_gate().unwrap(),
            &before,
        );
    }

    #[test]
    fn prepared_rdf_replacement_swaps_without_allocator_traffic_and_retires_outside_writers() {
        let target = store(1);
        target.try_set_commit_epoch(EpochId::new(9)).unwrap();
        assert!(target.insert(triple("old")));
        let retired = Arc::downgrade(target.triples.read().values().next().unwrap());
        let source = store(2);
        source.try_set_commit_epoch(EpochId::new(3)).unwrap();
        assert!(source.insert(triple("new")));
        assert!(source.create_graph("http://example.org/live"));
        assert!(
            source
                .graph("http://example.org/live")
                .unwrap()
                .insert(triple("named"))
        );
        let history = source.dataset_history().unwrap();
        let mut prepared = target
            .prepare_dataset_history_replacement(history.clone(), EpochId::new(3))
            .unwrap();
        let commit = target.lock_commit_scoped();
        let ready = prepared.ready(&commit).unwrap();
        #[cfg(feature = "lpg")]
        crate::allocation_test::start();
        let installed = ready.install();
        #[cfg(feature = "lpg")]
        assert_eq!(
            crate::allocation_test::stop(),
            crate::allocation_test::Counts::default()
        );
        assert!(target.triples.try_read().is_none());
        assert!(target.history.try_read().is_none());
        assert!(target.named_graphs.try_read().is_none());
        assert!(target.store_id.try_read().is_none());
        assert!(retired.upgrade().is_some());
        #[cfg(feature = "lpg")]
        crate::allocation_test::start();
        installed.release();
        #[cfg(feature = "lpg")]
        assert_eq!(
            crate::allocation_test::stop(),
            crate::allocation_test::Counts::default()
        );
        assert!(prepared.ready(&commit).is_err());
        drop(commit);
        same_history(&target.dataset_history().unwrap(), &history);
        assert_eq!(target.commit_epoch(), EpochId::new(3));
        let named = target.graph("http://example.org/live").unwrap();
        assert!(Arc::ptr_eq(
            &named.next_graph_incarnation,
            &target.next_graph_incarnation
        ));
        assert!(Arc::ptr_eq(
            &named.reserved_graph_incarnations,
            &target.reserved_graph_incarnations
        ));
        assert!(
            retired.upgrade().is_some(),
            "outer workspace retains displaced data"
        );
        drop(prepared);
        assert!(retired.upgrade().is_none());
    }

    #[test]
    fn exact_rdf_restore_apis_share_prepared_empty_identity_replacement() {
        for under_gate in [false, true] {
            let target = store(1);
            target.try_set_commit_epoch(EpochId::new(5)).unwrap();
            assert!(target.insert(triple("removed")));
            assert!(target.create_graph("http://example.org/removed"));
            let history = store(2).dataset_history().unwrap();
            if under_gate {
                let commit = target.lock_commit_scoped();
                target
                    .replace_dataset_history_exact_under_commit_gate(
                        &commit,
                        history.clone(),
                        EpochId::INITIAL,
                    )
                    .unwrap();
            } else {
                target
                    .replace_dataset_history_exact(history.clone(), EpochId::INITIAL)
                    .unwrap();
            }
            same_history(&target.dataset_history().unwrap(), &history);
            assert_eq!(target.commit_epoch(), EpochId::INITIAL);
            assert!(target.graph("http://example.org/removed").is_none());
        }
    }

    #[test]
    fn rdf_allocator_and_high_water_wait_for_exact_replacement_identity() {
        use std::sync::mpsc;
        use std::time::Duration;

        for allocate in [false, true] {
            let target = Arc::new(store(1));
            let old_identity = target.store_id();
            target
                .adopt_graph_incarnation_high_water(old_identity, GraphIncarnationId::new(50))
                .unwrap();
            let source = store(2);
            let new_identity = source.store_id();
            let mut prepared = target
                .prepare_dataset_history_replacement(
                    source.dataset_history().unwrap(),
                    EpochId::INITIAL,
                )
                .unwrap();
            let commit = target.lock_commit_scoped();
            let ready = prepared.ready(&commit).unwrap();
            let (entered_tx, entered_rx) = mpsc::channel();
            let (finished_tx, finished_rx) = mpsc::channel();
            let worker_target = Arc::clone(&target);
            let worker = std::thread::spawn(move || {
                entered_tx.send(()).unwrap();
                let result = if allocate {
                    worker_target
                        .new_detached_graph()
                        .map(|graph| Some((graph.store_id(), graph.graph_incarnation())))
                        .map_err(|error| error.to_string())
                } else {
                    worker_target
                        .adopt_graph_incarnation_high_water(
                            old_identity,
                            GraphIncarnationId::new(99),
                        )
                        .map(|()| None)
                };
                finished_tx.send(()).unwrap();
                result
            });
            entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            let blocked = matches!(
                finished_rx.recv_timeout(Duration::from_millis(50)),
                Err(mpsc::RecvTimeoutError::Timeout)
            );
            let before_install = target.next_graph_incarnation();
            let installed = ready.install();
            let still_blocked = matches!(finished_rx.try_recv(), Err(mpsc::TryRecvError::Empty));
            let installed_floor = target.next_graph_incarnation();
            installed.release();
            drop(commit);
            finished_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            let result = worker.join().unwrap();
            assert!(blocked && still_blocked);
            assert_eq!(
                before_install,
                GraphIncarnationId::new(50),
                "a blocked allocator cannot reserve from the old cut"
            );
            assert_eq!(installed_floor, GraphIncarnationId::FIRST_NAMED);
            if allocate {
                assert_eq!(
                    result.unwrap(),
                    Some((new_identity, GraphIncarnationId::FIRST_NAMED))
                );
                assert_eq!(target.next_graph_incarnation(), GraphIncarnationId::new(2));
            } else {
                assert!(
                    result.is_err(),
                    "old provenance must be rechecked after admission"
                );
                assert_eq!(
                    target.next_graph_incarnation(),
                    GraphIncarnationId::FIRST_NAMED
                );
            }
        }
    }

    #[test]
    fn graph_creation_retains_lifecycle_while_a_prebuilt_child_waits_for_snapshot_inheritance() {
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        for (get_or_create, transactional) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let target = Arc::new(store(1));
            let source = store(2);
            let mut prepared = target
                .prepare_dataset_history_replacement(
                    source.dataset_history().unwrap(),
                    EpochId::INITIAL,
                )
                .unwrap();
            let commit = target.lock_commit_scoped();
            let inheritance = target.tx_buffer.write();
            let (started_tx, started_rx) = mpsc::channel();
            let worker_target = Arc::clone(&target);
            let worker = std::thread::spawn(move || {
                started_tx.send(()).unwrap();
                let transaction = transactional.then_some(TransactionId::new(7001));
                if get_or_create {
                    worker_target
                        .graph_or_create_in_tx("http://example.org/created", transaction)
                        .is_ok()
                } else {
                    worker_target.create_graph_in_tx("http://example.org/created", transaction)
                }
            });
            started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            // Transactional callers block during initial buffer inspection;
            // nontransactional callers reach inheritance with a built child.
            let deadline = Instant::now() + Duration::from_secs(2);
            let mut lifecycle_held = false;
            while Instant::now() < deadline {
                if target.history_lifecycle_lock.try_lock().is_none() {
                    lifecycle_held = true;
                    break;
                }
                std::thread::yield_now();
            }
            if !transactional {
                while target.next_graph_incarnation() == GraphIncarnationId::FIRST_NAMED
                    && Instant::now() < deadline
                {
                    std::thread::yield_now();
                }
            }
            let allocated = target.next_graph_incarnation();
            let rejected = prepared
                .ready(&commit)
                .err()
                .is_some_and(|error| error.contains("lifecycle is busy"));
            drop(inheritance);
            let created = worker.join().unwrap();
            drop(commit);
            assert!(lifecycle_held && rejected && created);
            if !transactional {
                assert!(
                    allocated > GraphIncarnationId::FIRST_NAMED,
                    "the witness must reach the prebuilt-child boundary"
                );
            }
            assert_eq!(target.store_id(), StoreId::from_bytes([1; 32]).unwrap());
        }
    }

    #[test]
    fn prepared_rdf_replacement_refuses_every_pending_root_transaction_collection() {
        use super::super::RdfGraphPin;

        for collection in 0..7 {
            let target = store(1);
            assert!(target.create_graph("http://example.org/retained"));
            let retained = target.graph("http://example.org/retained").unwrap();
            let tx = TransactionId::new(7001);
            let source = store(2);
            let mut prepared = target
                .prepare_dataset_history_replacement(
                    source.dataset_history().unwrap(),
                    EpochId::INITIAL,
                )
                .unwrap();
            {
                // Isolate each collection: even an otherwise empty transaction
                // footprint is authority that exact replacement cannot discard.
                let mut buffer = target.tx_buffer.write();
                match collection {
                    0 => {
                        buffer.buffers.insert(tx, Vec::new());
                    }
                    1 => {
                        buffer.snapshot_epochs.insert(tx, EpochId::INITIAL);
                    }
                    2 => {
                        buffer
                            .created_graphs
                            .entry(tx)
                            .or_default()
                            .insert("pending".into(), Arc::clone(&retained));
                    }
                    3 => {
                        buffer.dropped_graphs.entry(tx).or_default().insert(
                            "retained".into(),
                            RdfGraphPin {
                                store: Arc::clone(&retained),
                                revision: 0,
                            },
                        );
                    }
                    4 => {
                        buffer.touched_graphs.entry(tx).or_default().insert(
                            "retained".into(),
                            RdfGraphPin {
                                store: Arc::clone(&retained),
                                revision: 0,
                            },
                        );
                    }
                    5 => {
                        buffer
                            .read_graphs
                            .entry(tx)
                            .or_default()
                            .insert("retained".into(), Some(Arc::clone(&retained)));
                    }
                    _ => {
                        buffer.snapshotted_graph_catalogs.insert(tx);
                    }
                }
            }
            let before = target.dataset_history().unwrap();
            let commit = target.lock_commit_scoped();
            let rejected = prepared
                .ready(&commit)
                .err()
                .is_some_and(|error| error.contains("pending transaction state"));
            assert!(
                rejected,
                "transaction collection {collection} must prevent replacement"
            );
            assert!(target.tx_buffer.try_read().is_some());
            same_history(
                &target.dataset_history_under_commit_gate().unwrap(),
                &before,
            );
            let buffer = target.tx_buffer.read();
            assert!(match collection {
                0 => buffer.buffers.contains_key(&tx),
                1 => buffer.snapshot_epochs.contains_key(&tx),
                2 => buffer.created_graphs.contains_key(&tx),
                3 => buffer.dropped_graphs.contains_key(&tx),
                4 => buffer.touched_graphs.contains_key(&tx),
                5 => buffer.read_graphs.contains_key(&tx),
                _ => buffer.snapshotted_graph_catalogs.contains(&tx),
            });
        }
    }
}
