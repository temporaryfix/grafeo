//! Bounded scheduling over a query's shared core spill authority.

use super::async_file::AsyncSpillFile;
use grafeo_common::memory::buffer::MemoryGrant;
use grafeo_core::execution::spill::{OwnedSpillError, OwnedSpillFile, SpillFileRole};
use grafeo_core::execution::{
    QueryCancellationError, QueryResourceContext, QueryResourceContextError,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Failure of admission, scheduling, or a shared framed operation.
#[derive(Debug, thiserror::Error)]
pub enum AsyncSpillError {
    /// The shared operation retains its physical state and accounting.
    #[error(transparent)]
    Operation(#[from] OwnedSpillError),
    /// Query admission failed before scheduling.
    #[error(transparent)]
    Resource(#[from] QueryResourceContextError),
    /// Cooperative cancellation at a bounded operation boundary.
    #[error(transparent)]
    Cancelled(#[from] QueryCancellationError),
    /// All bounded worker slots are occupied.
    #[error("async spill worker capacity exhausted")]
    Busy,
    /// A dropped waiter has already transferred this handle to a physical job.
    #[error("async spill handle is closed or belongs to an abandoned operation")]
    Closed,
    /// A runtime is unavailable or its physical worker failed.
    #[error("async spill worker unavailable")]
    WorkerUnavailable,
    /// An intrinsic manager cleanup error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// A worker failure retains its diagnostic allocation through destruction.
    #[error(transparent)]
    Worker(WorkerFailure),
}

#[derive(Debug)]
enum WorkerPrimary {
    Io(std::io::Error),
    Panic {
        _payload: Box<dyn std::any::Any + Send>,
    },
}

/// Scheduler diagnostic and the grant covering its escaping payload.
#[derive(Debug)]
pub struct WorkerFailure {
    primary: Option<WorkerPrimary>,
    grant: Option<MemoryGrant>,
}

impl std::fmt::Display for WorkerFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.primary {
            Some(WorkerPrimary::Io(error)) => std::fmt::Display::fmt(error, f),
            _ => f.write_str("async spill physical worker panicked"),
        }
    }
}
impl std::error::Error for WorkerFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.primary {
            Some(WorkerPrimary::Io(error)) => Some(error),
            _ => None,
        }
    }
}
impl Drop for WorkerFailure {
    fn drop(&mut self) {
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(self.primary.take()))) {
            std::mem::forget(payload);
            if let Some(grant) = self.grant.take() {
                std::mem::forget(grant);
            }
        }
    }
}

// Includes the pinned runtime's task envelope, notification and result slot;
// closure/result storage is charged separately below. The queue is also hard
// bounded, so scheduler bookkeeping cannot grow with rejected callers.
const JOB_OVERHEAD_BYTES: usize = 4096;
const MAX_JOBS: usize = 64;

pub(crate) struct Scheduler {
    pub(super) resources: QueryResourceContext,
    slots: Arc<Semaphore>,
    _grant: MemoryGrant,
}

struct Completed<T> {
    result: Result<T, AsyncSpillError>,
    // These remain attached to an undelivered response after physical I/O.
    _permit: OwnedSemaphorePermit,
    _grant: Option<MemoryGrant>,
    _scheduler: Arc<Scheduler>,
}

#[allow(
    clippy::result_large_err,
    reason = "operation failures retain their accounting owners"
)]
impl Scheduler {
    pub(crate) async fn run<T, F>(
        self: &Arc<Self>,
        cleanup: bool,
        operation: F,
    ) -> Result<T, AsyncSpillError>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T, AsyncSpillError> + Send + 'static,
    {
        if !cleanup {
            self.resources.check_cancelled()?;
        }
        let permit = Arc::clone(&self.slots)
            .try_acquire_owned()
            .map_err(|_| AsyncSpillError::Busy)?;
        let bytes = JOB_OVERHEAD_BYTES
            .checked_add(size_of::<F>())
            .and_then(|n| n.checked_add(size_of::<Completed<T>>()))
            .ok_or(AsyncSpillError::Busy)?;
        let grant = self.resources.try_allocate(bytes)?;
        self.run_admitted(cleanup, operation, permit, grant).await
    }

    /// Reserve terminal transport before any query job can consume its budget.
    #[cfg(all(feature = "gql", feature = "lpg", feature = "spill"))]
    pub(crate) fn reserve_cleanup<T, F>(
        self: &Arc<Self>,
        operation: F,
    ) -> Result<ReservedCleanup<T>, AsyncSpillError>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T, AsyncSpillError> + Send + 'static,
    {
        let permit = Arc::clone(&self.slots)
            .try_acquire_owned()
            .map_err(|_| AsyncSpillError::Busy)?;
        let bytes = JOB_OVERHEAD_BYTES
            .checked_add(size_of::<F>())
            .and_then(|n| n.checked_add(size_of::<Completed<T>>()))
            .and_then(|n| n.checked_add(size_of::<ReservedCleanup<T>>()))
            .ok_or(AsyncSpillError::Busy)?;
        let grant = self.resources.try_allocate(bytes)?;
        Ok(ReservedCleanup {
            scheduler: Arc::clone(self),
            permit,
            grant,
            operation: Box::new(operation),
        })
    }

    fn start_admitted<T, F>(
        self: &Arc<Self>,
        cleanup: bool,
        operation: F,
        permit: OwnedSemaphorePermit,
        grant: MemoryGrant,
        runtime: Option<&tokio::runtime::Handle>,
    ) -> Result<tokio::task::JoinHandle<Completed<T>>, AsyncSpillError>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T, AsyncSpillError> + Send + 'static,
    {
        let runtime = match runtime {
            Some(runtime) => runtime.clone(),
            None => tokio::runtime::Handle::try_current()
                .map_err(|_| AsyncSpillError::WorkerUnavailable)?,
        };
        let scheduler = Arc::clone(self);
        // The closure, permit and grant move into the runtime before awaiting.
        // Aborting the waiter drops only its JoinHandle, never the running job.
        Ok(runtime.spawn_blocking(move || {
            let mut grant = Some(grant);
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                let result = if !cleanup {
                    scheduler
                        .resources
                        .check_cancelled()
                        .map_err(AsyncSpillError::from)
                        .and_then(|()| operation())
                } else {
                    operation()
                };
                match result {
                    Ok(value) if !cleanup => scheduler
                        .resources
                        .check_cancelled()
                        .map(|()| value)
                        .map_err(Into::into),
                    result => result,
                }
            }));
            let result = match outcome {
                Ok(Err(AsyncSpillError::Io(error))) => {
                    Err(AsyncSpillError::Worker(WorkerFailure {
                        primary: Some(WorkerPrimary::Io(error)),
                        grant: grant.take(),
                    }))
                }
                Ok(result) => result,
                Err(payload) => Err(AsyncSpillError::Worker(WorkerFailure {
                    primary: Some(WorkerPrimary::Panic { _payload: payload }),
                    grant: grant.take(),
                })),
            };
            Completed {
                result,
                _permit: permit,
                _grant: grant,
                _scheduler: scheduler,
            }
        }))
    }

    async fn run_admitted<T, F>(
        self: &Arc<Self>,
        cleanup: bool,
        operation: F,
        permit: OwnedSemaphorePermit,
        grant: MemoryGrant,
    ) -> Result<T, AsyncSpillError>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T, AsyncSpillError> + Send + 'static,
    {
        let completed = self
            .start_admitted(cleanup, operation, permit, grant, None)?
            .await
            .map_err(|error| {
                if error.is_panic() {
                    std::mem::forget(error.into_panic());
                }
                AsyncSpillError::WorkerUnavailable
            })?;
        match completed.result {
            Ok(value) if !cleanup => {
                self.resources.check_cancelled()?;
                Ok(value)
            }
            result => result,
        }
    }
}

/// One terminal job whose capacity and memory never compete with query work.
#[cfg(all(feature = "gql", feature = "lpg", feature = "spill"))]
pub(crate) struct ReservedCleanup<T> {
    scheduler: Arc<Scheduler>,
    permit: OwnedSemaphorePermit,
    grant: MemoryGrant,
    operation: Box<dyn FnOnce() -> Result<T, AsyncSpillError> + Send>,
}

#[cfg(all(feature = "gql", feature = "lpg", feature = "spill"))]
#[allow(
    clippy::result_large_err,
    reason = "scheduling failures retain their accounting owners"
)]
impl<T: Send + 'static> ReservedCleanup<T> {
    /// Starts physical cleanup immediately, including from a cancelled waiter's Drop.
    pub(crate) fn start(
        self,
        runtime: Option<&tokio::runtime::Handle>,
    ) -> Result<(), AsyncSpillError> {
        let job = self.scheduler.start_admitted(
            true,
            self.operation,
            self.permit,
            self.grant,
            runtime,
        )?;
        drop(job);
        Ok(())
    }
}

/// Schedules owned framed operations on the existing query root and quota.
/// There are no raw path constructors or manually adjusted byte counters.
#[derive(Clone)]
pub struct AsyncSpillManager {
    pub(crate) scheduler: Arc<Scheduler>,
}

#[allow(
    clippy::result_large_err,
    reason = "operation failures retain their accounting owners"
)]
impl AsyncSpillManager {
    /// Creates a bounded adapter; query-leaf creation remains deferred to I/O.
    ///
    /// # Errors
    /// Rejects zero/excessive capacity, cancellation or memory exhaustion.
    pub fn new(resources: QueryResourceContext, max_jobs: usize) -> Result<Self, AsyncSpillError> {
        resources.check_cancelled()?;
        if !(1..=MAX_JOBS).contains(&max_jobs) {
            return Err(AsyncSpillError::Busy);
        }
        let grant = resources.try_allocate(
            size_of::<Scheduler>() + size_of::<Semaphore>() + 4 * size_of::<usize>(),
        )?;
        Ok(Self {
            scheduler: Arc::new(Scheduler {
                resources,
                slots: Arc::new(Semaphore::new(max_jobs)),
                _grant: grant,
            }),
        })
    }

    /// Creates a shared framed file on a bounded physical worker.
    ///
    /// # Errors
    /// Returns admission, cancellation, capacity or owned operation failures.
    pub async fn create_file(
        &self,
        role: SpillFileRole,
    ) -> Result<AsyncSpillFile, AsyncSpillError> {
        let resources = self.scheduler.resources.clone();
        let file = self
            .scheduler
            .run(false, move || {
                OwnedSpillFile::create(&resources, role).map_err(Into::into)
            })
            .await?;
        Ok(AsyncSpillFile::from_owned(
            file,
            Arc::clone(&self.scheduler),
        ))
    }

    /// Cleans tracked files through the shared authority. Live handles refuse deletion.
    /// Cleanup remains available after cancellation.
    ///
    /// # Errors
    /// Returns capacity, live-lease, identity or cleanup errors without quota credit.
    pub async fn cleanup(&self) -> Result<(), AsyncSpillError> {
        let resources = self.scheduler.resources.clone();
        self.scheduler
            .run(true, move || {
                if let Some(manager) = resources.spill_manager() {
                    manager.cleanup()?;
                }
                Ok(())
            })
            .await
    }

    /// Published framed-file bytes reported by the shared manager.
    #[must_use]
    pub fn spilled_bytes(&self) -> u64 {
        self.scheduler
            .resources
            .spill_manager()
            .map_or(0, |manager| manager.spilled_bytes())
    }

    /// Number of tracked files, including cleanup debt.
    #[must_use]
    pub fn active_file_count(&self) -> usize {
        self.scheduler
            .resources
            .spill_manager()
            .map_or(0, |manager| manager.active_file_count())
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests {
    use super::*;
    use grafeo_common::memory::buffer::BufferManager;
    use grafeo_common::types::StoreId;
    use grafeo_core::execution::QueryExecutionControl;
    use grafeo_core::execution::spill::{
        CleartextSpillRecordProvider, OwnedSpillBytes, OwnedSpillWrite, SpillDiskQuota,
        SpillFrameLimits, SpillIo, SpillIoOperation, SpillQueryIdentity, SpillRecordProvider,
        SpillRoot, SpillRootAuthority,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Condvar, Mutex};
    use std::time::Duration;

    fn resources(root: &Arc<SpillRoot>, control: &QueryExecutionControl) -> QueryResourceContext {
        QueryResourceContext::with_spill_root(
            BufferManager::with_budget(8 << 20),
            root,
            control.token(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn framed_roles_use_database_root_and_retain_output_grants() {
        for encrypted in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let config = crate::Config::in_memory().with_spill_path(directory.path());
            #[cfg(feature = "encryption")]
            let config = {
                let mut config = config;
                if encrypted {
                    config.encryption = Some(crate::config::EncryptionConfig {
                        key_chain: Arc::new(grafeo_common::encryption::KeyChain::new([37; 32])),
                    });
                }
                config
            };
            #[cfg(not(feature = "encryption"))]
            if encrypted {
                continue;
            }
            let control = QueryExecutionControl::new();
            let root = crate::spill_crypto::DatabaseSpillRoot::new(&config)
                .open(
                    directory.path(),
                    StoreId::from_bytes([17; 32]).unwrap(),
                    &control.token(),
                )
                .unwrap();
            let resources = resources(&root, &control);
            let manager = AsyncSpillManager::new(resources.clone(), 2).unwrap();
            assert!(resources.spill_manager().is_none());
            for role in [
                SpillFileRole::SortRun,
                SpillFileRole::NativePartition,
                SpillFileRole::RdfAggregateState,
            ] {
                let mut file = manager.create_file(role).await.unwrap();
                match role {
                    SpillFileRole::SortRun => file
                        .write(OwnedSpillWrite::SortStart {
                            columns: 1,
                            rows: 1,
                        })
                        .await
                        .unwrap(),
                    SpillFileRole::NativePartition => file
                        .write(OwnedSpillWrite::PartitionStart(1))
                        .await
                        .unwrap(),
                    SpillFileRole::RdfAggregateState => {}
                }
                let bytes = OwnedSpillBytes::copy_from(&resources, b"shared framed bytes").unwrap();
                let operation = match role {
                    SpillFileRole::SortRun => OwnedSpillWrite::SortRow(bytes),
                    SpillFileRole::NativePartition => OwnedSpillWrite::PartitionEntry(bytes),
                    SpillFileRole::RdfAggregateState => OwnedSpillWrite::AggregateState(bytes),
                };
                file.write(operation).await.unwrap();
                file.write(OwnedSpillWrite::Finish).await.unwrap();
                assert!(manager.spilled_bytes() > 0);
                let mut reader = file.reader().await.unwrap();
                match role {
                    SpillFileRole::SortRun => {
                        assert_eq!(reader.read_declaration().await.unwrap(), (1, 1));
                    }
                    SpillFileRole::NativePartition => {
                        assert_eq!(reader.read_declaration().await.unwrap(), (0, 1));
                    }
                    SpillFileRole::RdfAggregateState => {}
                }
                let record = reader.read_record().await.unwrap();
                assert_eq!(record.as_slice(), b"shared framed bytes");
                reader.finish().await.unwrap();
                file.close_and_delete().await.unwrap();
                let retained = resources.query_stats().allocated_bytes;
                drop(record);
                assert_eq!(
                    retained - resources.query_stats().allocated_bytes,
                    b"shared framed bytes".len()
                );
                assert_eq!(manager.active_file_count(), 0);
            }
            manager.cleanup().await.unwrap();
            drop(manager);
            assert_eq!(resources.query_stats().allocated_bytes, 0);
        }
    }

    struct TestAuthority;
    impl SpillRootAuthority for TestAuthority {
        fn store_id(&self) -> StoreId {
            StoreId::from_bytes([71; 32]).unwrap()
        }
        fn key_id(&self) -> [u8; 32] {
            [11; 32]
        }
        fn authenticate_marker(&self, bytes: &[u8]) -> std::io::Result<[u8; 32]> {
            Ok(*blake3::keyed_hash(&[19; 32], bytes).as_bytes())
        }
        fn verify_marker(&self, bytes: &[u8], auth: &[u8; 32]) -> std::io::Result<bool> {
            Ok(self.authenticate_marker(bytes)? == *auth)
        }
        fn record_provider(
            &self,
            _: SpillQueryIdentity,
        ) -> std::io::Result<Arc<dyn SpillRecordProvider>> {
            Ok(Arc::new(CleartextSpillRecordProvider))
        }
    }

    #[derive(Default)]
    struct PausedWrite {
        armed: AtomicBool,
        read_open: AtomicBool,
        fail: AtomicBool,
        entered: tokio::sync::Notify,
        released: Mutex<bool>,
        changed: Condvar,
    }
    impl PausedWrite {
        fn release(&self) {
            *self.released.lock().unwrap() = true;
            self.changed.notify_all();
        }
    }
    struct Release(Arc<PausedWrite>);
    impl Drop for Release {
        fn drop(&mut self) {
            self.0.release();
        }
    }
    impl SpillIo for PausedWrite {
        fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
            let target = if self.read_open.load(Ordering::SeqCst) {
                SpillIoOperation::ReadOpen
            } else {
                SpillIoOperation::WritePayload
            };
            if operation == target && self.armed.swap(false, Ordering::SeqCst) {
                self.entered.notify_one();
                let (released, _) = self
                    .changed
                    .wait_timeout_while(
                        self.released.lock().unwrap(),
                        Duration::from_secs(10),
                        |released| !*released,
                    )
                    .unwrap();
                if !*released {
                    return Err(std::io::ErrorKind::TimedOut.into());
                }
                if self.fail.load(Ordering::SeqCst) {
                    return Err(std::io::Error::other("physical write primary"));
                }
            }
            Ok(())
        }
        fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
            Some(0)
        }
    }

    fn paused_root(directory: &std::path::Path, io: Arc<PausedWrite>) -> Arc<SpillRoot> {
        SpillRoot::open(
            directory,
            Arc::new(TestAuthority),
            SpillFrameLimits::format_max(),
            io,
            SpillDiskQuota::new(1 << 20),
            Some(2 << 20),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn dropped_waiter_keeps_file_quota_memory_and_worker_slot_until_physical_completion() {
        let directory = tempfile::tempdir().unwrap();
        let io = Arc::new(PausedWrite::default());
        let _release = Release(Arc::clone(&io));
        let control = QueryExecutionControl::new();
        let resources = resources(&paused_root(directory.path(), Arc::clone(&io)), &control);
        let manager = AsyncSpillManager::new(resources.clone(), 1).unwrap();
        let base_bytes = resources.query_stats().allocated_bytes;
        let mut file = manager
            .create_file(SpillFileRole::RdfAggregateState)
            .await
            .unwrap();
        let payload = OwnedSpillBytes::copy_from(&resources, &[5; 1024]).unwrap();
        io.armed.store(true, Ordering::SeqCst);
        let mut waiting = Box::pin(file.write(OwnedSpillWrite::AggregateState(payload)));
        tokio::select! {
            () = io.entered.notified() => {},
            result = &mut waiting => panic!("write did not pause: {result:?}"),
            () = tokio::time::sleep(Duration::from_secs(10)) => panic!("write did not start"),
        }
        drop(waiting);
        assert!(matches!(
            file.write(OwnedSpillWrite::Finish).await,
            Err(AsyncSpillError::Closed)
        ));
        assert!(matches!(
            manager.create_file(SpillFileRole::SortRun).await,
            Err(AsyncSpillError::Busy)
        ));
        let shared = resources.spill_manager().unwrap();
        let reserved = shared.disk_stats().reserved_live_bytes;
        assert!(reserved > 1024);
        assert!(shared.cleanup().is_err());
        assert_eq!(shared.disk_stats().reserved_live_bytes, reserved);
        assert_eq!(shared.active_file_count(), 1);
        assert!(resources.query_stats().allocated_bytes > base_bytes + 1024);
        io.release();
        tokio::time::timeout(Duration::from_secs(10), async {
            while manager.scheduler.slots.available_permits() == 0
                || resources.query_stats().allocated_bytes != base_bytes
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(shared.active_file_count(), 0);
        assert_eq!(shared.disk_stats().reserved_live_bytes, 0);
        assert_eq!(resources.query_stats().allocated_bytes, base_bytes);
        manager.cleanup().await.unwrap();
        drop(file);
        drop(manager);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[tokio::test]
    async fn physical_error_survives_concurrent_cancellation() {
        let directory = tempfile::tempdir().unwrap();
        let io = Arc::new(PausedWrite::default());
        let _release = Release(Arc::clone(&io));
        let control = QueryExecutionControl::new();
        let resources = resources(&paused_root(directory.path(), Arc::clone(&io)), &control);
        let manager = AsyncSpillManager::new(resources.clone(), 1).unwrap();
        let mut file = manager
            .create_file(SpillFileRole::RdfAggregateState)
            .await
            .unwrap();
        let payload = OwnedSpillBytes::copy_from(&resources, b"failure").unwrap();
        io.fail.store(true, Ordering::SeqCst);
        io.armed.store(true, Ordering::SeqCst);
        let mut waiting = Box::pin(file.write(OwnedSpillWrite::AggregateState(payload)));
        tokio::select! {
            () = io.entered.notified() => {},
            result = &mut waiting => panic!("write did not pause: {result:?}"),
            () = tokio::time::sleep(Duration::from_secs(10)) => panic!("write did not start"),
        }
        control.cancellation_handle().cancel();
        io.release();
        let error = waiting.await.unwrap_err();
        assert!(matches!(error, AsyncSpillError::Operation(_)));
        assert!(error.to_string().contains("physical write primary"));
        assert!(
            manager.active_file_count() > 0,
            "error retains file authority"
        );
        drop(error);
        manager.cleanup().await.unwrap();
        assert_eq!(manager.active_file_count(), 0);
    }

    #[tokio::test]
    async fn cancellation_before_admission_creates_no_query_leaf() {
        let directory = tempfile::tempdir().unwrap();
        let control = QueryExecutionControl::new();
        let resources = resources(
            &paused_root(directory.path(), Arc::new(PausedWrite::default())),
            &control,
        );
        let manager = AsyncSpillManager::new(resources.clone(), 1).unwrap();
        control.cancellation_handle().cancel();
        assert!(matches!(
            manager.create_file(SpillFileRole::SortRun).await,
            Err(AsyncSpillError::Cancelled(_))
        ));
        assert!(resources.spill_manager().is_none());
        manager.cleanup().await.unwrap();
    }
    #[expect(
        clippy::result_large_err,
        reason = "worker fixtures transfer the same move-only error and grant owner as production"
    )]
    #[tokio::test]
    async fn physical_job_retains_scheduler_account_after_last_public_owner_drops() {
        let resources = QueryResourceContext::new(BufferManager::with_budget(1 << 20)).unwrap();
        let manager = AsyncSpillManager::new(resources.clone(), 1).unwrap();
        let weak = Arc::downgrade(&manager.scheduler);
        let scheduler = Arc::clone(&manager.scheduler);
        let io = Arc::new(PausedWrite::default());
        io.armed.store(true, Ordering::SeqCst);
        let _release = Release(Arc::clone(&io));
        let worker_io = Arc::clone(&io);
        let waiter = tokio::spawn(async move {
            scheduler
                .run(false, move || {
                    worker_io.check(SpillIoOperation::WritePayload)?;
                    Ok(())
                })
                .await
        });
        tokio::time::timeout(Duration::from_secs(10), io.entered.notified())
            .await
            .unwrap();
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        drop(manager);
        assert!(
            weak.upgrade().is_some(),
            "physical job lost the scheduler's allocation grant"
        );
        io.release();
        tokio::time::timeout(Duration::from_secs(10), async {
            while weak.upgrade().is_some() || resources.query_stats().allocated_bytes != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[expect(
        clippy::result_large_err,
        reason = "worker fixtures transfer the same move-only error and grant owner as production"
    )]
    #[tokio::test]
    async fn worker_panic_and_panicking_payload_destructor_are_contained() {
        struct HostilePanic;
        impl Drop for HostilePanic {
            fn drop(&mut self) {
                panic!("hostile panic-payload destructor");
            }
        }
        let resources = QueryResourceContext::new(BufferManager::with_budget(1 << 20)).unwrap();
        let manager = AsyncSpillManager::new(resources, 1).unwrap();
        let scheduler = Arc::clone(&manager.scheduler);
        let waiter = tokio::spawn(async move {
            let result = scheduler
                .run::<(), _>(false, || std::panic::panic_any(HostilePanic))
                .await;
            assert!(result.is_err());
            drop(result);
        });
        waiter
            .await
            .expect("worker and failure destruction must not unwind into the waiter");
    }

    #[tokio::test]
    async fn write_rejects_payload_charged_to_another_query() {
        let directory = tempfile::tempdir().unwrap();
        let control = QueryExecutionControl::new();
        let root = paused_root(directory.path(), Arc::new(PausedWrite::default()));
        let owner = resources(&root, &control);
        let unrelated = resources(&root, &QueryExecutionControl::new());
        let manager = AsyncSpillManager::new(owner, 1).unwrap();
        let mut file = manager
            .create_file(SpillFileRole::RdfAggregateState)
            .await
            .unwrap();
        let payload = OwnedSpillBytes::copy_from(&unrelated, b"wrong query grant").unwrap();
        assert!(
            file.write(OwnedSpillWrite::AggregateState(payload))
                .await
                .is_err()
        );
    }
    // Only this test provider fixes crypto context. Physical identities remain
    // random; the framing header contains no file/query identity. This lets
    // independent sync/async files use the same fixed logical crypto identity,
    // key and nonce schedule without a production entropy override.
    struct FixtureProvider;
    struct FixtureRecord {
        physical_identity: grafeo_core::execution::spill::SpillFileIdentity,
        cipher: grafeo_common::encryption::PageEncryptor,
    }
    impl FixtureRecord {
        fn nonce(sequence: u64) -> [u8; 12] {
            let mut nonce = [43; 12];
            nonce[4..].copy_from_slice(&sequence.to_be_bytes());
            nonce
        }
        fn validate(
            &self,
            meta: &grafeo_core::execution::spill::SpillRecordMeta,
        ) -> std::io::Result<()> {
            if meta.identity() == self.physical_identity && meta.is_sealed() {
                Ok(())
            } else {
                Err(std::io::ErrorKind::InvalidData.into())
            }
        }
    }
    impl SpillRecordProvider for FixtureProvider {
        fn seals(&self) -> bool {
            true
        }
        fn supports_qualified_exact_open(&self) -> bool {
            true
        }
        fn file_workspace_allocation_bound(&self) -> Option<usize> {
            Some(4096)
        }
        fn begin_file(
            &self,
            identity: grafeo_core::execution::spill::SpillFileIdentity,
        ) -> std::io::Result<Box<dyn grafeo_core::execution::spill::OpenSpillRecord>> {
            let mut logical_identity = [29; 32];
            logical_identity[16..].fill(31);
            Ok(Box::new(FixtureRecord {
                physical_identity: identity,
                cipher: grafeo_common::encryption::KeyChain::new([23; 32])
                    .encryptor_for("async-spill-test-fixture", &logical_identity),
            }))
        }
    }
    impl grafeo_core::execution::spill::OpenSpillRecord for FixtureRecord {
        fn stored_len(&self, len: usize) -> std::io::Result<usize> {
            len.checked_add(grafeo_common::encryption::ENCRYPTION_OVERHEAD)
                .ok_or_else(|| std::io::ErrorKind::InvalidInput.into())
        }
        fn seal_allocation_bound(&self, len: usize) -> Option<usize> {
            grafeo_common::encryption::PageEncryptor::encrypt_allocation_bound(len)
        }
        fn open_allocation_bound(&self, len: usize) -> Option<usize> {
            grafeo_common::encryption::PageEncryptor::decrypt_allocation_bound(len)
        }
        fn seal(
            &mut self,
            meta: &grafeo_core::execution::spill::SpillRecordMeta,
            aad: &[u8; 32],
            bytes: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            self.validate(meta)?;
            self.cipher
                .encrypt(bytes, &Self::nonce(meta.sequence()), aad)
                .map_err(|_| std::io::ErrorKind::InvalidData.into())
        }
        fn open(
            &mut self,
            meta: &grafeo_core::execution::spill::SpillRecordMeta,
            aad: &[u8; 32],
            bytes: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            self.validate(meta)?;
            if bytes.get(..12) != Some(Self::nonce(meta.sequence()).as_slice()) {
                return Err(std::io::ErrorKind::InvalidData.into());
            }
            self.cipher
                .decrypt(bytes, aad)
                .map_err(|_| std::io::ErrorKind::InvalidData.into())
        }
        fn open_qualified_into(
            &mut self,
            meta: &grafeo_core::execution::spill::SpillRecordMeta,
            aad: &[u8; 32],
            bytes: &[u8],
            plaintext: &mut [u8],
        ) -> Option<std::io::Result<()>> {
            Some((|| {
                self.validate(meta)?;
                if bytes.get(..12) != Some(Self::nonce(meta.sequence()).as_slice()) {
                    return Err(std::io::ErrorKind::InvalidData.into());
                }
                self.cipher
                    .decrypt_into(bytes, aad, plaintext)
                    .map_err(|_| std::io::ErrorKind::InvalidData.into())
            })())
        }
    }
    struct FixtureAuthority(bool);
    impl SpillRootAuthority for FixtureAuthority {
        fn store_id(&self) -> StoreId {
            TestAuthority.store_id()
        }
        fn key_id(&self) -> [u8; 32] {
            TestAuthority.key_id()
        }
        fn authenticate_marker(&self, bytes: &[u8]) -> std::io::Result<[u8; 32]> {
            TestAuthority.authenticate_marker(bytes)
        }
        fn verify_marker(&self, bytes: &[u8], auth: &[u8; 32]) -> std::io::Result<bool> {
            TestAuthority.verify_marker(bytes, auth)
        }
        fn record_provider(
            &self,
            _: SpillQueryIdentity,
        ) -> std::io::Result<Arc<dyn SpillRecordProvider>> {
            if self.0 {
                Ok(Arc::new(FixtureProvider))
            } else {
                Ok(Arc::new(CleartextSpillRecordProvider))
            }
        }
    }

    #[tokio::test]
    async fn sync_async_exact_bytes_and_cross_read_with_fixed_crypto_fixture() {
        for sealed in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let control = QueryExecutionControl::new();
            let root = SpillRoot::open(
                directory.path(),
                Arc::new(FixtureAuthority(sealed)),
                SpillFrameLimits::format_max(),
                Arc::new(grafeo_core::execution::spill::NoopSpillIo),
                SpillDiskQuota::new(1 << 20),
                Some(2 << 20),
            )
            .unwrap();
            let resources = resources(&root, &control);
            let asynchronous = AsyncSpillManager::new(resources.clone(), 2).unwrap();
            let shared = resources.ensure_spill_manager().unwrap().unwrap();
            for role in [
                SpillFileRole::SortRun,
                SpillFileRole::NativePartition,
                SpillFileRole::RdfAggregateState,
            ] {
                let mut sync_file = shared.create_file(role).unwrap();
                let mut async_file = asynchronous.create_file(role).await.unwrap();
                assert_ne!(
                    sync_file.identity(),
                    async_file.identity(),
                    "physical entropy remains enabled"
                );
                match role {
                    SpillFileRole::SortRun => {
                        sync_file.write_sort_run_start(2, 2).unwrap();
                        async_file
                            .write(OwnedSpillWrite::SortStart {
                                columns: 2,
                                rows: 2,
                            })
                            .await
                            .unwrap();
                    }
                    SpillFileRole::NativePartition => {
                        sync_file.write_partition_start(2).unwrap();
                        async_file
                            .write(OwnedSpillWrite::PartitionStart(2))
                            .await
                            .unwrap();
                    }
                    SpillFileRole::RdfAggregateState => {}
                }
                let records: [&[u8]; 2] = [b"first record", &[0, 255, 10, 128, 42]];
                for record in records {
                    let bytes = OwnedSpillBytes::copy_from(&resources, record).unwrap();
                    let operation = match role {
                        SpillFileRole::SortRun => {
                            sync_file.write_sort_row(record).unwrap();
                            OwnedSpillWrite::SortRow(bytes)
                        }
                        SpillFileRole::NativePartition => {
                            sync_file.write_partition_entry(record).unwrap();
                            OwnedSpillWrite::PartitionEntry(bytes)
                        }
                        SpillFileRole::RdfAggregateState => {
                            sync_file.write_aggregate_state(record).unwrap();
                            OwnedSpillWrite::AggregateState(bytes)
                        }
                    };
                    async_file.write(operation).await.unwrap();
                }
                sync_file.finish_write().unwrap();
                async_file.write(OwnedSpillWrite::Finish).await.unwrap();
                let mut identity = String::with_capacity(32);
                for byte in async_file.identity().as_bytes() {
                    use std::fmt::Write as _;
                    write!(&mut identity, "{byte:02x}").unwrap();
                }
                let async_path = std::fs::read_dir(shared.spill_dir())
                    .unwrap()
                    .map(|entry| entry.unwrap().path())
                    .find(|path| {
                        path.file_name()
                            .unwrap()
                            .to_string_lossy()
                            .contains(&identity)
                    })
                    .unwrap();
                let sync_bytes = std::fs::read(sync_file.path()).unwrap();
                let async_bytes = std::fs::read(&async_path).unwrap();
                assert_eq!(sync_bytes, async_bytes, "sealed={sealed}, role={role:?}");
                // Test-only byte substitution preserves each retained inode;
                // each reader now consumes the other writer's exact output.
                std::fs::write(sync_file.path(), &async_bytes).unwrap();
                std::fs::write(&async_path, &sync_bytes).unwrap();
                let mut sync_reader = sync_file.reader().unwrap();
                let mut async_reader = async_file.reader().await.unwrap();
                match role {
                    SpillFileRole::SortRun => assert_eq!(
                        sync_reader.read_sort_run_start().unwrap(),
                        async_reader.read_declaration().await.unwrap()
                    ),
                    SpillFileRole::NativePartition => assert_eq!(
                        sync_reader.read_partition_start().unwrap(),
                        async_reader.read_declaration().await.unwrap().1
                    ),
                    SpillFileRole::RdfAggregateState => {}
                }
                for expected in records {
                    let sync_record = match role {
                        SpillFileRole::SortRun => sync_reader.read_sort_row().unwrap(),
                        SpillFileRole::NativePartition => {
                            sync_reader.read_partition_entry().unwrap()
                        }
                        SpillFileRole::RdfAggregateState => {
                            sync_reader.read_aggregate_state().unwrap()
                        }
                    };
                    let async_record = async_reader.read_record().await.unwrap();
                    assert_eq!(sync_record, expected);
                    assert_eq!(async_record.as_slice(), expected);
                }
                sync_reader.finish().unwrap();
                drop(sync_reader);
                async_reader.finish().await.unwrap();
                sync_file.close_and_delete().unwrap();
                async_file.close_and_delete().await.unwrap();
            }
            asynchronous.cleanup().await.unwrap();
        }
    }
    #[tokio::test]
    async fn live_reader_cleanup_refusal_does_not_leak_retired_writer_memory() {
        let directory = tempfile::tempdir().unwrap();
        let control = QueryExecutionControl::new();
        let resources = resources(
            &paused_root(directory.path(), Arc::new(PausedWrite::default())),
            &control,
        );
        let manager = AsyncSpillManager::new(resources.clone(), 2).unwrap();
        let base = resources.query_stats().allocated_bytes;
        let mut file = manager
            .create_file(SpillFileRole::RdfAggregateState)
            .await
            .unwrap();
        file.write(OwnedSpillWrite::Finish).await.unwrap();
        let mut reader = file.reader().await.unwrap();
        let failure = file
            .close_and_delete()
            .await
            .expect_err("live reader owns the file");
        drop(failure);
        assert_eq!(manager.active_file_count(), 1);
        reader.close().await.unwrap();
        manager.cleanup().await.unwrap();
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(
            resources.query_stats().allocated_bytes,
            base,
            "retired writer grant leaked after reader cleanup"
        );
    }

    #[tokio::test]
    async fn abandoned_reader_open_closes_reader_before_deleting_its_file() {
        let directory = tempfile::tempdir().unwrap();
        let io = Arc::new(PausedWrite::default());
        let _release = Release(Arc::clone(&io));
        let control = QueryExecutionControl::new();
        let resources = resources(&paused_root(directory.path(), Arc::clone(&io)), &control);
        let manager = AsyncSpillManager::new(resources.clone(), 1).unwrap();
        let base = resources.query_stats().allocated_bytes;
        let mut file = manager
            .create_file(SpillFileRole::RdfAggregateState)
            .await
            .unwrap();
        file.write(OwnedSpillWrite::Finish).await.unwrap();
        io.read_open.store(true, Ordering::SeqCst);
        io.armed.store(true, Ordering::SeqCst);
        let mut waiting = Box::pin(file.reader());
        tokio::select! {
            () = io.entered.notified() => {},
            _ = &mut waiting => panic!("reader did not pause"),
            () = tokio::time::sleep(Duration::from_secs(10)) => panic!("reader did not start"),
        }
        drop(waiting);
        assert!(resources.spill_manager().unwrap().cleanup().is_err());
        io.release();
        tokio::time::timeout(Duration::from_secs(10), async {
            while manager.scheduler.slots.available_permits() == 0
                || resources.query_stats().allocated_bytes != base
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(
            manager.active_file_count(),
            0,
            "abandoned reader response stranded its file"
        );
        assert_eq!(resources.query_stats().allocated_bytes, base);
    }
    #[tokio::test]
    async fn construction_primary_retains_authority_for_undestroyable_cleanup_payload() {
        struct FailProvider;
        impl SpillRecordProvider for FailProvider {
            fn seals(&self) -> bool {
                false
            }
            fn file_workspace_allocation_bound(&self) -> Option<usize> {
                Some(32 << 10)
            }
            fn begin_file(
                &self,
                _: grafeo_core::execution::spill::SpillFileIdentity,
            ) -> std::io::Result<Box<dyn grafeo_core::execution::spill::OpenSpillRecord>>
            {
                Err(std::io::Error::other("initialization primary"))
            }
        }
        struct HostilePayload {
            _bytes: std::mem::ManuallyDrop<Vec<u8>>,
            destroyed: Arc<AtomicBool>,
        }
        impl Drop for HostilePayload {
            fn drop(&mut self) {
                self.destroyed.store(true, Ordering::SeqCst);
                panic!("cleanup payload destructor refused retirement");
            }
        }
        struct CleanupPanic {
            fired: AtomicBool,
            destroyed: Arc<AtomicBool>,
        }
        impl SpillIo for CleanupPanic {
            fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
                if operation == SpillIoOperation::Delete && !self.fired.swap(true, Ordering::SeqCst)
                {
                    std::panic::panic_any(HostilePayload {
                        _bytes: std::mem::ManuallyDrop::new(vec![7; 8192]),
                        destroyed: Arc::clone(&self.destroyed),
                    });
                }
                Ok(())
            }
        }
        let directory = tempfile::tempdir().unwrap();
        let destroyed = Arc::new(AtomicBool::new(false));
        let (resources, _shared) = crate::spill_crypto::admitted_spill_test_resources(
            directory.path(),
            BufferManager::with_budget(1 << 20),
            QueryExecutionControl::new().token(),
            Arc::new(FailProvider),
            SpillFrameLimits::format_max(),
            Arc::new(CleanupPanic {
                fired: AtomicBool::new(false),
                destroyed: Arc::clone(&destroyed),
            }),
            SpillDiskQuota::new(u64::MAX),
        );
        let manager = AsyncSpillManager::new(resources.clone(), 1).unwrap();
        let baseline = resources.query_stats().allocated_bytes;
        let error = manager
            .create_file(SpillFileRole::SortRun)
            .await
            .err()
            .expect("provider must fail");
        assert!(error.to_string().contains("initialization primary"));
        assert!(
            destroyed.load(Ordering::SeqCst),
            "cleanup payload was not retired behind the boundary"
        );
        drop(error);
        assert!(
            resources.query_stats().allocated_bytes >= baseline + 8192,
            "undestroyable provider heap lost its accounting authority"
        );
        manager.cleanup().await.unwrap();
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    fn queued_job_keeps_admission_until_physical_dispatch_even_without_waiter() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            for cancel in [false, true] {
                let directory = tempfile::tempdir().unwrap();
                let control = QueryExecutionControl::new();
                let root = paused_root(directory.path(), Arc::new(PausedWrite::default()));
                let resources = resources(&root, &control);
                let manager = AsyncSpillManager::new(resources.clone(), 1).unwrap();
                let baseline = resources.query_stats().allocated_bytes;
                let blocker = Arc::new(PausedWrite::default());
                blocker.armed.store(true, Ordering::SeqCst);
                let _release = Release(Arc::clone(&blocker));
                let worker_blocker = Arc::clone(&blocker);
                let physical = tokio::task::spawn_blocking(move || {
                    worker_blocker.check(SpillIoOperation::WritePayload)
                });
                tokio::time::timeout(Duration::from_secs(10), blocker.entered.notified())
                    .await
                    .unwrap();
                let mut waiting = Box::pin(manager.create_file(SpillFileRole::RdfAggregateState));
                std::future::poll_fn(|cx| {
                    assert!(std::future::Future::poll(waiting.as_mut(), cx).is_pending());
                    std::task::Poll::Ready(())
                })
                .await;
                drop(waiting);
                assert!(
                    resources.spill_manager().is_none(),
                    "queued create ran before dispatch"
                );
                assert_eq!(manager.scheduler.slots.available_permits(), 0);
                assert!(resources.query_stats().allocated_bytes > baseline);
                if cancel {
                    control.cancellation_handle().cancel();
                }
                blocker.release();
                physical.await.unwrap().unwrap();
                tokio::time::timeout(Duration::from_secs(10), async {
                    while manager.scheduler.slots.available_permits() == 0
                        || resources.query_stats().allocated_bytes != baseline
                    {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .unwrap();
                if cancel {
                    assert!(resources.spill_manager().is_none());
                }
                assert_eq!(manager.active_file_count(), 0);
                manager.cleanup().await.unwrap();
            }
        });
    }
}
