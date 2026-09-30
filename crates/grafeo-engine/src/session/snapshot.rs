//! Mixed-model read snapshot: one publication instant for GQL and SPARQL.

use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::{Rc, Weak};
#[cfg(all(feature = "gql", feature = "lpg"))]
use std::time::Duration;

use grafeo_common::utils::error::{Error, Result, TransactionError};

thread_local! {
    /// Only an active, thread-bound owner authorizes a nested snapshot read.
    static SNAPSHOTS: RefCell<HashMap<usize, Weak<SnapshotActivation>>> = RefCell::new(HashMap::new());
}

fn tm_key(tm: &crate::transaction::TransactionManager) -> usize {
    // The activation retains this exact Arc-backed lock, preventing address
    // reuse for the full lifetime of every authorized nested reader.
    std::ptr::from_ref(tm.publication()) as usize
}

struct SnapshotActivation {
    _publication: parking_lot::ArcRwLockReadGuard<parking_lot::RawRwLock, ()>,
    key: usize,
}

impl Drop for SnapshotActivation {
    fn drop(&mut self) {
        // TLS teardown may already have removed the weak registry. No lock
        // ownership depends on it: the actual guard is a field of this owner.
        let _ = SNAPSHOTS.try_with(|snapshots| {
            let mut snapshots = snapshots.borrow_mut();
            if snapshots
                .get(&self.key)
                .is_some_and(|entry| std::ptr::eq(entry.as_ptr(), self))
            {
                snapshots.remove(&self.key);
            }
        });
    }
}

fn held_snapshot(tm: &crate::transaction::TransactionManager) -> Option<Rc<SnapshotActivation>> {
    SNAPSHOTS.with(|snapshots| snapshots.borrow().get(&tm_key(tm)).and_then(Weak::upgrade))
}

fn snapshot_holds(tm: &crate::transaction::TransactionManager) -> bool {
    held_snapshot(tm).is_some()
}

enum ReadOwnership<'a> {
    Borrowed {
        _guard: parking_lot::RwLockReadGuard<'a, ()>,
    },
    Snapshot {
        _activation: Rc<SnapshotActivation>,
    },
}

/// A read always retains a real lock owner, including inside a snapshot.
pub(crate) struct PublicationReadGuard<'a> {
    _ownership: ReadOwnership<'a>,
}

/// Acquire the publication read lock or retain this thread's snapshot owner.
/// Ordinary reads retain a borrowed guard without allocating; nested snapshot
/// reads retain the existing activation without reacquiring a non-reentrant lock.
pub(crate) fn acquire_publication_read(
    tm: &crate::transaction::TransactionManager,
) -> PublicationReadGuard<'_> {
    let ownership = if let Some(activation) = held_snapshot(tm) {
        ReadOwnership::Snapshot {
            _activation: activation,
        }
    } else {
        ReadOwnership::Borrowed {
            _guard: tm.publication().read(),
        }
    };
    PublicationReadGuard {
        _ownership: ownership,
    }
}

/// Cooperatively acquires the publication read barrier for a controlled query.
///
/// Unlike [`acquire_publication_read`], this never waits indefinitely behind a
/// writer: cancellation and deadlines are observed between bounded lock
/// attempts. Uncontrolled query paths retain their existing blocking
/// acquisition semantics.
#[cfg(all(feature = "gql", feature = "lpg"))]
pub(crate) fn acquire_publication_read_with_checkpoint<'a>(
    tm: &'a crate::transaction::TransactionManager,
    checkpoint: &grafeo_core::execution::QueryExecutionCheckpoint,
) -> std::result::Result<PublicationReadGuard<'a>, grafeo_core::execution::QueryCancellationError> {
    const POLL_INTERVAL: Duration = Duration::from_millis(1);

    checkpoint.check()?;
    if let Some(activation) = held_snapshot(tm) {
        return Ok(PublicationReadGuard {
            _ownership: ReadOwnership::Snapshot {
                _activation: activation,
            },
        });
    }

    loop {
        checkpoint.check()?;
        if let Some(guard) = tm.publication().try_read_for(POLL_INTERVAL) {
            checkpoint.check()?;
            return Ok(PublicationReadGuard {
                _ownership: ReadOwnership::Borrowed { _guard: guard },
            });
        }
    }
}

/// Holds the publication read barrier so GQL and SPARQL see one committed instant.
///
/// Bind the guard. Dropping the final snapshot or nested read owner releases
/// the barrier and lets mixed commits proceed.
///
/// Participating Session mutations and `commit` on this thread reject an
/// active snapshot before trying to acquire their publication write lock.
/// Nested snapshots retain the same lock; either guard may be dropped first.
/// A snapshot and its nested read scopes stay on their acquiring thread.
///
/// ```compile_fail
/// use grafeo_engine::MixedSnapshot;
/// fn require_send<T: Send>() {}
/// require_send::<MixedSnapshot<'static>>();
/// ```
///
/// ```compile_fail
/// use grafeo_engine::MixedSnapshot;
/// fn require_sync<T: Sync>() {}
/// require_sync::<MixedSnapshot<'static>>();
/// ```
#[must_use = "the snapshot holds the mixed-model read barrier; bind it until both languages have run"]
pub struct MixedSnapshot<'a> {
    _activation: Rc<SnapshotActivation>,
    _manager: PhantomData<&'a crate::transaction::TransactionManager>,
}

impl super::Session {
    /// Pin committed mixed-model visibility until the guard is dropped.
    ///
    /// Concurrent mixed commits block until this snapshot is dropped. Reads
    /// on this thread (`execute`, `execute_sparql`, `get_node`,
    /// `contains_rdf_quad`) share that instant and must not take the
    /// publication lock again.
    ///
    /// Rejects an explicit open transaction: LPG-at-tx-start and RDF-at-commit
    /// would not be one cut.
    ///
    /// # Errors
    ///
    /// Returns an error if a transaction is already active on this session.
    ///
    /// # Examples
    ///
    /// ```
    /// # use grafeo_engine::GrafeoDB;
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    /// let _snap: grafeo_engine::MixedSnapshot<'_> = session.snapshot().expect("no open transaction");
    /// let _ = session.get_node(grafeo_common::types::NodeId::new(1));
    /// ```
    pub fn snapshot(&self) -> Result<MixedSnapshot<'_>> {
        if self.in_transaction() {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "cannot Session::snapshot() while a transaction is open".into(),
            )));
        }
        let activation = match held_snapshot(&self.transaction_manager) {
            Some(activation) => activation,
            None => {
                let publication = self.transaction_manager.publication_arc().read_arc();
                let key = tm_key(&self.transaction_manager);
                let activation = Rc::new(SnapshotActivation {
                    _publication: publication,
                    key,
                });
                SNAPSHOTS.with(|snapshots| {
                    snapshots
                        .borrow_mut()
                        .insert(key, Rc::downgrade(&activation));
                });
                activation
            }
        };
        Ok(MixedSnapshot {
            _activation: activation,
            _manager: PhantomData,
        })
    }

    pub(crate) fn publication_read_guard(&self) -> Option<PublicationReadGuard<'_>> {
        Some(acquire_publication_read(&self.transaction_manager))
    }

    /// Acquires the publication barrier for a borrowed lazy stream.
    ///
    /// A stream cannot merely borrow an already-held `MixedSnapshot`: that
    /// snapshot may be dropped before the lazy operator is consumed. Reject
    /// that ambiguous nesting and require the stream to own its full lifetime.
    #[cfg(all(feature = "gql", feature = "lpg"))]
    pub(crate) fn streaming_publication_read_guard(
        &self,
        control: &grafeo_core::execution::QueryExecutionControl,
    ) -> Result<parking_lot::RwLockReadGuard<'_, ()>> {
        if snapshot_holds(&self.transaction_manager) {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "cannot create a lazy result stream while Session::snapshot() is held; drop the snapshot first"
                    .into(),
            )));
        }
        loop {
            control
                .check()
                .map_err(Self::map_query_cancellation_error)?;
            if let Some(guard) = self
                .transaction_manager
                .publication()
                .try_read_for(Duration::from_millis(1))
            {
                control
                    .check()
                    .map_err(Self::map_query_cancellation_error)?;
                return Ok(guard);
            }
        }
    }

    /// Acquires an Arc-backed publication barrier for an owned lazy stream.
    #[cfg(all(feature = "gql", feature = "lpg"))]
    pub(crate) fn streaming_owned_publication_read_guard(
        &self,
        control: &grafeo_core::execution::QueryExecutionControl,
    ) -> Result<parking_lot::ArcRwLockReadGuard<parking_lot::RawRwLock, ()>> {
        if snapshot_holds(&self.transaction_manager) {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "cannot create a lazy result stream while Session::snapshot() is held; drop the snapshot first"
                    .into(),
            )));
        }
        let publication = self.transaction_manager.publication_arc();
        loop {
            control
                .check()
                .map_err(Self::map_query_cancellation_error)?;
            if let Some(guard) = publication.try_read_arc_for(Duration::from_millis(1)) {
                control
                    .check()
                    .map_err(Self::map_query_cancellation_error)?;
                return Ok(guard);
            }
        }
    }

    pub(crate) fn check_not_in_mixed_snapshot(&self) -> Result<()> {
        if snapshot_holds(&self.transaction_manager) {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "cannot mutate or commit while Session::snapshot() is held".into(),
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, GrafeoDB, GraphModel};

    fn database() -> Result<GrafeoDB> {
        #[cfg(feature = "lpg")]
        let model = GraphModel::Lpg;
        #[cfg(not(feature = "lpg"))]
        let model = GraphModel::Rdf;
        GrafeoDB::with_config(Config::in_memory().with_graph_model(model))
    }

    #[test]
    fn nested_snapshot_retains_admission_after_outer_drop() -> Result<()> {
        let db = database()?;
        let session = db.session();
        let outer = session.snapshot()?;
        let inner = session.snapshot()?;
        drop(outer);

        assert!(snapshot_holds(&session.transaction_manager));
        assert!(session.check_not_in_mixed_snapshot().is_err());
        assert!(
            session
                .transaction_manager
                .publication()
                .try_write()
                .is_none()
        );
        drop(inner);
        assert!(!snapshot_holds(&session.transaction_manager));
        assert!(
            session
                .transaction_manager
                .publication()
                .try_write()
                .is_some()
        );
        Ok(())
    }

    #[test]
    fn nested_read_retains_real_lock_after_snapshot_drop() -> Result<()> {
        let db = database()?;
        let session = db.session();
        let snapshot = session.snapshot()?;
        let read = acquire_publication_read(&session.transaction_manager);
        drop(snapshot);

        assert!(
            session
                .transaction_manager
                .publication()
                .try_write()
                .is_none()
        );
        assert!(snapshot_holds(&session.transaction_manager));
        drop(read);
        assert!(
            session
                .transaction_manager
                .publication()
                .try_write()
                .is_some()
        );
        assert!(!snapshot_holds(&session.transaction_manager));
        Ok(())
    }

    #[test]
    fn nested_snapshot_retains_outer_when_inner_drops_first() -> Result<()> {
        let db = database()?;
        let session = db.session();
        let outer = session.snapshot()?;
        let inner = session.snapshot()?;
        drop(inner);

        assert!(snapshot_holds(&session.transaction_manager));
        assert!(session.check_not_in_mixed_snapshot().is_err());
        assert!(
            session
                .transaction_manager
                .publication()
                .try_write()
                .is_none()
        );
        drop(outer);
        assert!(!snapshot_holds(&session.transaction_manager));
        assert!(
            session
                .transaction_manager
                .publication()
                .try_write()
                .is_some()
        );
        Ok(())
    }

    #[test]
    fn queued_writer_does_not_block_nested_snapshot_or_read() -> Result<()> {
        use std::time::{Duration, Instant};

        let db = database()?;
        let session = db.session();
        let outer = session.snapshot()?;
        let tm = &session.transaction_manager;
        std::thread::scope(|threads| {
            // A bounded attempt ensures a regression fails instead of hanging
            // the suite: a recursive acquisition would wait until this expires.
            let writer = threads.spawn(|| {
                tm.publication()
                    .try_write_for(Duration::from_secs(5))
                    .is_some()
            });
            let deadline = Instant::now() + Duration::from_secs(2);
            while tm.publication().try_read().is_some() {
                assert!(Instant::now() < deadline, "writer did not queue");
                std::thread::yield_now();
            }
            // The failed raw read above proves that a writer is queued while
            // the outer read owns the lock. Sharing must not reacquire it.
            let inner = session.snapshot()?;
            let read = acquire_publication_read(tm);
            drop(outer);
            drop(inner);
            assert!(snapshot_holds(tm));
            assert!(tm.publication().try_write().is_none());
            drop(read);
            assert!(!snapshot_holds(tm));
            assert!(writer.join().expect("writer thread"), "writer timed out");
            Ok(())
        })
    }

    #[test]
    fn snapshot_unwind_removes_registration_and_releases_lock() {
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let db = database().expect("test database");
        let session = db.session();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _outer = session.snapshot().expect("outer snapshot");
            let _inner = session.snapshot().expect("inner snapshot");
            let _read = acquire_publication_read(&session.transaction_manager);
            panic!("test unwind after nested acquisition");
        }));
        assert!(result.is_err());
        assert!(!snapshot_holds(&session.transaction_manager));
        assert!(
            session
                .transaction_manager
                .publication()
                .try_write()
                .is_some()
        );
        // The weak registry does not accumulate entries for dead activations.
        SNAPSHOTS.with(|snapshots| {
            assert!(
                !snapshots
                    .borrow()
                    .contains_key(&tm_key(&session.transaction_manager))
            );
        });
    }

    #[test]
    fn snapshot_on_one_database_does_not_authorize_another() -> Result<()> {
        let a = database()?;
        let b = database()?;
        let session_a = a.session();
        let session_b = b.session();
        let snapshot_a = session_a.snapshot()?;
        let read_b = acquire_publication_read(&session_b.transaction_manager);
        assert!(snapshot_holds(&session_a.transaction_manager));
        assert!(!snapshot_holds(&session_b.transaction_manager));
        drop(snapshot_a);
        assert!(
            session_a
                .transaction_manager
                .publication()
                .try_write()
                .is_some()
        );
        assert!(
            session_b
                .transaction_manager
                .publication()
                .try_write()
                .is_none()
        );
        drop(read_b);
        assert!(
            session_b
                .transaction_manager
                .publication()
                .try_write()
                .is_some()
        );
        Ok(())
    }

    #[cfg(all(feature = "gql", feature = "lpg", not(target_arch = "wasm32")))]
    #[test]
    fn stream_publication_waits_observe_the_owner_deadline() -> Result<()> {
        use grafeo_common::utils::error::QueryErrorKind;
        use grafeo_core::execution::QueryExecutionControl;
        let db = database()?;
        let session = db.session();
        let _writer = session.transaction_manager.publication().write();
        // A blocking read would deadlock while this thread retains the writer.
        for owned in [false, true] {
            let control = QueryExecutionControl::with_timeout(Duration::from_millis(5))
                .expect("test deadline");
            let result = if owned {
                session
                    .streaming_owned_publication_read_guard(&control)
                    .map(drop)
            } else {
                session.streaming_publication_read_guard(&control).map(drop)
            };
            assert!(
                matches!(result, Err(Error::Query(error)) if error.kind == QueryErrorKind::Timeout)
            );
        }
        Ok(())
    }

    #[cfg(all(feature = "gql", feature = "lpg"))]
    #[test]
    fn controlled_nested_read_retains_lock_and_observes_cancellation() -> Result<()> {
        use grafeo_core::execution::{QueryCancellationError, QueryExecutionControl};

        let db = database()?;
        let session = db.session();
        let snapshot = session.snapshot()?;
        let control = QueryExecutionControl::new();
        let checkpoint = control.checkpoint();
        let read =
            acquire_publication_read_with_checkpoint(&session.transaction_manager, &checkpoint)
                .expect("uncancelled read");
        control.cancellation_handle().cancel();
        assert!(matches!(
            acquire_publication_read_with_checkpoint(&session.transaction_manager, &checkpoint),
            Err(QueryCancellationError::Cancelled)
        ));
        drop(snapshot);
        assert!(
            session
                .transaction_manager
                .publication()
                .try_write()
                .is_none()
        );
        drop(read);
        assert!(
            session
                .transaction_manager
                .publication()
                .try_write()
                .is_some()
        );
        Ok(())
    }

    #[cfg(all(feature = "gql", feature = "lpg"))]
    #[test]
    fn lazy_streams_remain_send_without_thread_local_activation() {
        fn require_send<T: Send>() {}
        require_send::<crate::query::executor::stream::OwnedResultStream>();
        require_send::<crate::query::executor::stream::ResultStream<'static>>();
    }
}
