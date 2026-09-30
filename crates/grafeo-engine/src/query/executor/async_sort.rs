//! Owned scheduled execution of the existing native sort operator.

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod memory_tests;

use super::{ResultAccumulator, ResultAdmission, ResultLimits, convert_cancellation_error};
use crate::database::QueryResult;
use crate::execution::spill::async_manager::ReservedCleanup;
use crate::execution::spill::{AsyncSpillError, AsyncSpillManager};
use grafeo_common::memory::buffer::{AccountedErrorPublisher, MemoryGrant};
use grafeo_common::utils::error::RetainedErrorContext;
use grafeo_common::utils::error::{Error, ErrorCode, Result};
use grafeo_core::execution::operators::{
    AccountedFailureClassification, OperatorError, SortOperator,
};
use grafeo_core::execution::{
    QueryCancellationHandle, QueryExecutionControl, QueryResourceContext,
};
use parking_lot::Mutex;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// A fixed number of pulls/outputs amortizes dispatch without collecting an
// unbounded result in a job. Core continues to poll between records and rows.
const INPUT_CHUNKS_PER_JOB: usize = 32;
const OUTPUT_ROWS_PER_JOB: usize = 2048;
const OWNER_TASK_BYTES: usize = 4096;
type Publication = parking_lot::ArcRwLockReadGuard<parking_lot::RawRwLock, ()>;

/// Result of the binding's synchronous query-dispatch phase.
pub enum AsyncSortDispatch {
    /// The existing execution path completed this request.
    Completed(QueryResult),
    /// An admitted read-only sort now owns its publication cut and resources.
    Prepared(PreparedAsyncSort),
}

/// One owned query prepared for bounded sort jobs on the async spill scheduler.
/// Dropping either preparation or its awaiting caller cancels execution while
/// cleanup retains the publication cut through physical completion.
pub struct PreparedAsyncSort {
    owner: Option<QueryOwner>,
}

#[derive(Clone, Copy)]
enum Job {
    Input,
    FinishInput,
    Output,
    Cleanup,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Progress {
    More,
    InputDone,
    OutputDone,
    Failed,
    Cleaned,
}

struct SortState {
    sort: Option<SortOperator>,
    accumulator: Option<ResultAccumulator>,
    visible_width: usize,
    primary: Option<Error>,
    result: Option<QueryResult>,
    cleaned: bool,
    error_publishers: [Option<AccountedErrorPublisher<RetainedErrorContext>>; 6],
    operator_publishers: [Option<AccountedErrorPublisher<OperatorError>>; 2],
    panic_publishers: [Option<AccountedErrorPublisher<OwnedWorkerPanic>>; 4],
    panic_grants: [Option<MemoryGrant>; 4],
    scheduling_publishers: [Option<AccountedErrorPublisher<AsyncSpillError>>; 2],
    #[cfg(test)]
    jobs: usize,
}

// Foreign panic values never execute Display. Each gets its own pre-admitted
// diagnostic envelope; the common retained owner contains a destructor panic.
struct OwnedWorkerPanic {
    payload: Option<Box<dyn std::any::Any + Send>>,
    grant: Option<MemoryGrant>,
}
impl std::fmt::Display for OwnedWorkerPanic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("async sort physical operation panicked")
    }
}
impl std::fmt::Debug for OwnedWorkerPanic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, formatter)
    }
}
impl std::error::Error for OwnedWorkerPanic {}

impl Drop for OwnedWorkerPanic {
    fn drop(&mut self) {
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(self.payload.take()))) {
            // Keep the escaping opaque diagnostic envelope charged too; the
            // publisher will independently quarantine its physical block.
            std::mem::forget(self.grant.take());
            std::panic::resume_unwind(payload);
        }
    }
}

fn operator_code(error: &OperatorError) -> ErrorCode {
    fn cancelled(error: &grafeo_core::execution::QueryCancellationError) -> ErrorCode {
        match error {
            grafeo_core::execution::QueryCancellationError::Cancelled => ErrorCode::QueryCancelled,
            grafeo_core::execution::QueryCancellationError::DeadlineExceeded { .. } => {
                ErrorCode::QueryTimeout
            }
            _ => ErrorCode::Internal,
        }
    }
    fn memory(error: &grafeo_common::memory::buffer::MemoryGrantError) -> ErrorCode {
        use grafeo_common::memory::buffer::MemoryGrantError;
        match error {
            MemoryGrantError::LimitExceeded { .. }
            | MemoryGrantError::ArithmeticOverflow { .. }
            | MemoryGrantError::Denied { .. } => ErrorCode::StorageFull,
            _ => ErrorCode::Internal,
        }
    }
    match error {
        OperatorError::Context { source, .. } => operator_code(source),
        OperatorError::ClassifiedAccountedFailure { classification, .. } => match classification {
            AccountedFailureClassification::ResidentMemory(error) => memory(error),
            AccountedFailureClassification::ResidentAllocation
            | AccountedFailureClassification::ResidentExactVectorAllocation(_)
            | AccountedFailureClassification::StorageFull => ErrorCode::StorageFull,
            AccountedFailureClassification::QueryCancelled(error) => cancelled(error),
            AccountedFailureClassification::TypeMismatch => ErrorCode::TypeMismatch,
            AccountedFailureClassification::ColumnNotFound
            | AccountedFailureClassification::ConstraintViolation => ErrorCode::InvalidInput,
            AccountedFailureClassification::WriteConflict => ErrorCode::TransactionConflict,
            _ => ErrorCode::Internal,
        },
        OperatorError::TypeMismatch { .. } => ErrorCode::TypeMismatch,
        OperatorError::ColumnNotFound(_) | OperatorError::ConstraintViolation(_) => {
            ErrorCode::InvalidInput
        }
        OperatorError::WriteConflict(_) => ErrorCode::TransactionConflict,
        OperatorError::ResidentMemory(error) => memory(error),
        OperatorError::ResidentAllocation(_)
        | OperatorError::ResidentContainerAllocation { .. }
        | OperatorError::ResidentExactVectorAllocation(_)
        | OperatorError::ResidentNativeMapAllocation { .. }
        | OperatorError::ResidentNativeMapAllocationWithRollback { .. }
        | OperatorError::StorageFull(_) => ErrorCode::StorageFull,
        OperatorError::QueryCancelled(error) => cancelled(error),
        _ => ErrorCode::Internal,
    }
}

fn owned_operator_error(
    error: OperatorError,
    publishers: &mut [Option<AccountedErrorPublisher<OperatorError>>; 2],
) -> Error {
    let code = operator_code(&error);
    if let Some(publisher) = publishers.iter_mut().find_map(Option::take) {
        Error::RetainedContext {
            code,
            source: publisher.publish(error).into(),
        }
    } else {
        // Input/output stop at their first error; only cleanup can add another.
        std::mem::forget(error);
        Error::Internal("async sort exceeded its terminal error protocol".into())
    }
}

impl SortState {
    fn execute(&mut self, job: Job) -> Result<Progress> {
        #[cfg(test)]
        {
            self.jobs += 1;
        }
        if matches!(job, Job::Cleanup) {
            self.cleanup();
            return Ok(Progress::Cleaned);
        }
        let publishers = &mut self.operator_publishers;
        let sort = self
            .sort
            .as_mut()
            .ok_or_else(|| Error::Internal("async sort has no live operator".into()))?;
        match job {
            Job::Input => {
                for _ in 0..INPUT_CHUNKS_PER_JOB {
                    if !sort
                        .ingest_next_input_chunk()
                        .map_err(|error| owned_operator_error(error, publishers))?
                    {
                        return Ok(Progress::InputDone);
                    }
                }
                Ok(Progress::More)
            }
            Job::FinishInput => {
                sort.finish_input()
                    .map_err(|error| owned_operator_error(error, publishers))?;
                Ok(Progress::More)
            }
            Job::Output => {
                let accumulator = self
                    .accumulator
                    .as_mut()
                    .ok_or_else(|| Error::Internal("async sort lost its result account".into()))?;
                let mut rows = 0;
                while rows < OUTPUT_ROWS_PER_JOB {
                    let Some(chunk) = sort
                        .next_prepared_output()
                        .map_err(|error| owned_operator_error(error, publishers))?
                    else {
                        return Ok(Progress::OutputDone);
                    };
                    accumulator.consume_visible_sort_columns(&chunk, self.visible_width)?;
                    // Count even empty chunks, so a source cannot monopolize a job.
                    rows += chunk.row_count().max(1);
                    drop(chunk);
                }
                Ok(Progress::More)
            }
            Job::Cleanup => Ok(Progress::Cleaned),
        }
    }

    fn retain_error(&mut self, error: Error, phase: &'static str) {
        self.primary = Some(match self.primary.take() {
            Some(primary) => {
                let code = primary.error_code();
                // At most operation, post-job cancellation, operator cleanup,
                // leaf cleanup and terminal scheduler failures can coexist.
                if let Some(publisher) = self.error_publishers.iter_mut().find_map(Option::take) {
                    Error::RetainedContext {
                        code,
                        source: publisher
                            .publish(RetainedErrorContext::new(primary, error, phase))
                            .into(),
                    }
                } else {
                    // The fixed terminal protocol cannot exceed its reserved terminal phases;
                    // preserve opaque authority even if that invariant changes.
                    std::mem::forget(error);
                    primary
                }
            }
            None => {
                let code = error.error_code();
                if let Some(publisher) = self.error_publishers.iter_mut().find_map(Option::take) {
                    Error::RetainedContext {
                        code,
                        source: publisher
                            .publish(RetainedErrorContext::from_primary(error))
                            .into(),
                    }
                } else {
                    // The initial failure always has the first reserved slot.
                    std::mem::forget(error);
                    Error::Internal("async sort lost its initial error publisher".into())
                }
            }
        });
    }

    fn retain_panic(&mut self, payload: Box<dyn std::any::Any + Send>, phase: &'static str) {
        if let Some((publisher, grant)) = self
            .panic_publishers
            .iter_mut()
            .zip(&mut self.panic_grants)
            .find_map(|(publisher, grant)| publisher.take().zip(grant.take()))
        {
            self.retain_error(
                Error::RetainedContext {
                    code: ErrorCode::Internal,
                    source: publisher
                        .publish(OwnedWorkerPanic {
                            payload: Some(payload),
                            grant: Some(grant),
                        })
                        .into(),
                },
                phase,
            );
        } else {
            // Four disjoint physical phases are the only panic sources.
            std::mem::forget(payload);
        }
    }

    fn cleanup(&mut self) {
        if self.cleaned {
            return;
        }
        // Mark before destruction: a foreign unwind cannot retry execution.
        self.cleaned = true;
        if let Some(mut sort) = self.sort.take() {
            if let Err(error) = sort.finish_owned_cleanup(None) {
                let error = owned_operator_error(error, &mut self.operator_publishers);
                self.retain_error(error, "async sort cleanup");
            }
            drop(sort);
        }
        if self.primary.is_none() {
            self.result = self.accumulator.take().map(ResultAccumulator::finish);
        } else {
            drop(self.accumulator.take());
        }
    }
}

// A physical job retains this entire owner, not just its operator. Thus runtime
// shutdown cannot release publication while an abandoned blocking job still runs.
struct SharedQuery {
    state: Mutex<Option<SortState>>,
    resources: QueryResourceContext,
    leaf_finish_attempted: AtomicBool,
    publication: Mutex<Option<Publication>>,
    database_owner: Mutex<Option<Arc<dyn Send + Sync>>>,
    terminal_requested: AtomicBool,
    _grant: MemoryGrant,
}

impl SharedQuery {
    fn job(&self, job: Job) -> Progress {
        let mut slot = self.state.lock();
        if !matches!(job, Job::Cleanup) && self.terminal_requested.load(Ordering::Acquire) {
            return Progress::Failed;
        }
        let Some(mut state) = slot.take() else {
            return Progress::Failed;
        };
        let outcome = catch_unwind(AssertUnwindSafe(|| state.execute(job)));
        let progress = match outcome {
            Ok(Ok(progress)) => progress,
            Ok(Err(error)) => {
                state.retain_error(error, "async sort operation");
                Progress::Failed
            }
            Err(payload) => {
                state.retain_panic(payload, "async sort operation");
                Progress::Failed
            }
        };
        *slot = Some(state);
        progress
    }

    fn retain_error(&self, error: Error, phase: &'static str) {
        if let Some(state) = self.state.lock().as_mut() {
            state.retain_error(error, phase);
        }
    }

    fn retain_scheduling_error(&self, error: AsyncSpillError, phase: &'static str) {
        if let Some(state) = self.state.lock().as_mut() {
            let error = scheduling_error(error, &mut state.scheduling_publishers);
            state.retain_error(error, phase);
        }
    }

    fn finish_leaf(&self) {
        if self.leaf_finish_attempted.swap(true, Ordering::AcqRel) {
            return;
        }
        if let Some(manager) = self.resources.spill_manager() {
            match catch_unwind(AssertUnwindSafe(|| manager.finish_query())) {
                Ok(Ok(())) => {}
                Ok(Err(error)) => self.retain_error(error.into(), "spill query cleanup"),
                Err(payload) => {
                    if let Some(state) = self.state.lock().as_mut() {
                        state.retain_panic(payload, "spill query cleanup");
                    } else {
                        std::mem::forget(payload);
                    }
                }
            }
        }
        // Physical execution and terminal I/O are finished. Diagnostics and
        // waiter bookkeeping may outlive this publication cut.
        drop(self.publication.lock().take());
        // The binding may have transferred the last database reference before
        // its initial worker returns this preparation. Close only after the
        // publication cut is released, on this same physical terminal worker.
        let database_owner = self.database_owner.lock().take();
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(database_owner))) {
            if let Some(state) = self.state.lock().as_mut() {
                state.retain_panic(payload, "database owner destruction");
            } else {
                std::mem::forget(payload);
            }
        }
    }
}

impl Drop for SharedQuery {
    fn drop(&mut self) {
        // Last-owner backstop: no physical worker can still hold this owner.
        // Explicit execution performs both operations on a scheduled worker.
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let _ = self.job(Job::Cleanup);
            self.finish_leaf();
        }));
        if let Err(payload) = outcome {
            std::mem::forget(payload);
        }
    }
}

struct CleanupTicket {
    job: Mutex<Option<ReservedCleanup<()>>>,
    runtime: Mutex<Option<tokio::runtime::Handle>>,
}
impl CleanupTicket {
    fn start(&self, shared: &SharedQuery) {
        // Close admission before a new physical job can borrow the state.
        shared.terminal_requested.store(true, Ordering::Release);
        let job = self.job.lock().take();
        let runtime = self.runtime.lock().clone();
        if let Some(job) = job
            && let Err(error) = job.start(runtime.as_ref())
        {
            shared.retain_scheduling_error(error, "async cleanup scheduling");
        }
    }
}

struct QueryOwner {
    shared: Arc<SharedQuery>,
    scheduler: AsyncSpillManager,
    cleanup: Arc<CleanupTicket>,
    cleanup_complete: Option<tokio::sync::oneshot::Receiver<()>>,
    control: QueryExecutionControl,
    limits: ResultLimits,
    admission: Option<ResultAdmission>,
    started: std::time::Instant,
}

fn resource_error(error: grafeo_core::execution::QueryResourceContextError) -> Error {
    match error {
        grafeo_core::execution::QueryResourceContextError::Memory(error) => {
            Error::Storage(grafeo_common::utils::error::StorageError::Full)
                .with_context(error.to_string())
        }
        other => Error::Io(std::io::Error::other(other)),
    }
}

fn scheduling_error(
    error: AsyncSpillError,
    publishers: &mut [Option<AccountedErrorPublisher<AsyncSpillError>>; 2],
) -> Error {
    let code = match &error {
        AsyncSpillError::Cancelled(reason) => convert_cancellation_error(*reason).error_code(),
        AsyncSpillError::Resource(grafeo_core::execution::QueryResourceContextError::Memory(_)) => {
            ErrorCode::StorageFull
        }
        _ => ErrorCode::IoError,
    };
    if let Some(publisher) = publishers.iter_mut().find_map(Option::take) {
        Error::RetainedContext {
            code,
            source: publisher.publish(error).into(),
        }
    } else {
        std::mem::forget(error);
        Error::Internal("async sort exceeded its scheduling error protocol".into())
    }
}

impl QueryOwner {
    #[expect(
        clippy::result_large_err,
        reason = "scheduled closures transfer inline failures with their admitted accounting owners"
    )]
    async fn job(&mut self, job: Job) -> Progress {
        let shared = Arc::clone(&self.shared);
        let outcome = self
            .scheduler
            .scheduler
            .run(matches!(job, Job::Cleanup), move || {
                let progress = shared.job(job);
                if matches!(job, Job::Cleanup) {
                    shared.finish_leaf();
                }
                Ok(progress)
            })
            .await;
        match outcome {
            Ok(progress) => progress,
            Err(error) => {
                self.shared
                    .retain_scheduling_error(error, "async scheduling");
                Progress::Failed
            }
        }
    }

    async fn cleanup(&mut self) {
        self.cleanup.start(&self.shared);
        if let Some(completion) = self.cleanup_complete.take()
            && completion.await.is_err()
        {
            self.shared.retain_scheduling_error(
                AsyncSpillError::WorkerUnavailable,
                "async cleanup completion",
            );
        }
    }

    async fn run(mut self) -> Result<QueryResult> {
        let mut progress = self.job(Job::Input).await;
        while progress == Progress::More {
            progress = self.job(Job::Input).await;
        }
        if progress == Progress::InputDone {
            progress = self.job(Job::FinishInput).await;
            while progress == Progress::More {
                progress = self.job(Job::Output).await;
            }
        }
        self.cleanup().await;
        if self
            .shared
            .state
            .lock()
            .as_ref()
            .is_some_and(|state| !state.cleaned)
        {
            let primary = self
                .shared
                .state
                .lock()
                .as_mut()
                .and_then(|state| state.primary.take());
            // Keep the uncleaned operator in the last-owner backstop. A reserved
            // job cannot fail admission; this is runtime shutdown/worker loss.
            return Err(primary
                .unwrap_or_else(|| Error::Internal("async cleanup worker unavailable".into())));
        }
        let mut state = self
            .shared
            .state
            .lock()
            .take()
            .ok_or_else(|| Error::Internal("async sort lost terminal ownership".into()))?;
        if let Some(primary) = state.primary.take() {
            return Err(primary);
        }
        let mut result = state
            .result
            .take()
            .ok_or_else(|| Error::Internal("async sort did not finish its result".into()))?;
        if let Some(admit) = self.admission {
            admit(&result, self.limits)?;
        }
        self.control.complete().map_err(|error| match error {
            grafeo_core::execution::QueryLifecycleError::Cancelled(reason) => {
                convert_cancellation_error(reason)
            }
            other => Error::Internal(other.to_string()),
        })?;
        result.execution_time_ms = Some(self.started.elapsed().as_secs_f64() * 1000.0);
        Ok(result)
    }
}

struct CancelWaiter {
    handle: Option<QueryCancellationHandle>,
    shared: Arc<SharedQuery>,
    cleanup: Arc<CleanupTicket>,
}
impl Drop for CancelWaiter {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.cancel();
            // No async poll is needed to queue the already-admitted physical
            // cleanup, so synchronous DB close cannot starve its completion.
            self.cleanup.start(&self.shared);
        }
    }
}

impl PreparedAsyncSort {
    #[expect(
        clippy::result_large_err,
        reason = "reserved cleanup transfers inline failures with their admitted accounting owners"
    )]
    pub(crate) fn new(
        sort: SortOperator,
        columns: Vec<String>,
        resources: QueryResourceContext,
        control: QueryExecutionControl,
        publication: Publication,
        limits: ResultLimits,
        admission: Option<ResultAdmission>,
    ) -> Result<Self> {
        let grant = resources
            .try_allocate(
                size_of::<SharedQuery>()
                    + size_of::<SortState>()
                    + size_of::<QueryOwner>()
                    + 4 * size_of::<usize>()
                    + OWNER_TASK_BYTES
                    + size_of::<CleanupTicket>()
                    + 2 * size_of::<usize>()
                    + 1024,
            )
            .map_err(resource_error)?;
        let mut scheduling_publishers = [const { None }; 2];
        for publisher in &mut scheduling_publishers {
            let grant = resources.try_allocate(0).map_err(resource_error)?;
            *publisher = Some(
                AccountedErrorPublisher::try_new(grant)
                    .map_err(|error| Error::Io(std::io::Error::other(error)))?,
            );
        }
        let scheduler = AsyncSpillManager::new(resources.clone(), 2)
            .map_err(|error| scheduling_error(error, &mut scheduling_publishers))?;
        let mut error_publishers = [const { None }; 6];
        for publisher in &mut error_publishers {
            let grant = resources.try_allocate(0).map_err(resource_error)?;
            *publisher = Some(
                AccountedErrorPublisher::try_new(grant)
                    .map_err(|error| Error::Io(std::io::Error::other(error)))?,
            );
        }
        let mut operator_publishers = [const { None }; 2];
        for publisher in &mut operator_publishers {
            let grant = resources.try_allocate(0).map_err(resource_error)?;
            *publisher = Some(
                AccountedErrorPublisher::try_new(grant)
                    .map_err(|error| Error::Io(std::io::Error::other(error)))?,
            );
        }
        let mut panic_publishers = [const { None }; 4];
        let mut panic_grants = [const { None }; 4];
        for (publisher, envelope) in panic_publishers.iter_mut().zip(&mut panic_grants) {
            let grant = resources.try_allocate(0).map_err(resource_error)?;
            *publisher = Some(
                AccountedErrorPublisher::try_new(grant)
                    .map_err(|error| Error::Io(std::io::Error::other(error)))?,
            );
            *envelope = Some(
                resources
                    .try_allocate(OWNER_TASK_BYTES)
                    .map_err(resource_error)?,
            );
        }
        let visible_width = columns.len();
        let accumulator =
            ResultAccumulator::new_inner(&columns, &[], resources.clone(), limits, true)?;
        let shared = Arc::new(SharedQuery {
            state: Mutex::new(Some(SortState {
                sort: Some(sort),
                accumulator: Some(accumulator),
                visible_width,
                primary: None,
                result: None,
                cleaned: false,
                error_publishers,
                operator_publishers,
                panic_publishers,
                panic_grants,
                scheduling_publishers,
                #[cfg(test)]
                jobs: 0,
            })),
            resources,
            leaf_finish_attempted: AtomicBool::new(false),
            publication: Mutex::new(Some(publication)),
            database_owner: Mutex::new(None),
            terminal_requested: AtomicBool::new(false),
            _grant: grant,
        });
        let cleanup_owner = Arc::clone(&shared);
        let (cleanup_finished, cleanup_complete) = tokio::sync::oneshot::channel();
        let cleanup_job = scheduler
            .scheduler
            .reserve_cleanup(move || {
                let _ = cleanup_owner.job(Job::Cleanup);
                cleanup_owner.finish_leaf();
                let _ = cleanup_finished.send(());
                Ok(())
            })
            .map_err(|error| {
                let mut state = shared.state.lock();
                if let Some(state) = state.as_mut() {
                    scheduling_error(error, &mut state.scheduling_publishers)
                } else {
                    std::mem::forget(error);
                    Error::Internal("async preparation lost state".into())
                }
            })?;
        Ok(Self {
            owner: Some(QueryOwner {
                shared,
                scheduler,
                cleanup: Arc::new(CleanupTicket {
                    job: Mutex::new(Some(cleanup_job)),
                    runtime: Mutex::new(tokio::runtime::Handle::try_current().ok()),
                }),
                cleanup_complete: Some(cleanup_complete),
                control,
                limits,
                admission,
                started: std::time::Instant::now(),
            }),
        })
    }

    /// Retains the binding's database reference through physical query cleanup.
    ///
    /// Attach it before returning from the initial blocking preparation worker,
    /// so an abandoned Python waiter cannot close its last database owner while
    /// this prepared query still holds the publication cut.
    #[must_use]
    pub fn retain_database_owner(self, owner: Arc<dyn Send + Sync>) -> Self {
        if let Some(query) = &self.owner {
            *query.shared.database_owner.lock() = Some(owner);
        }
        self
    }

    /// Runs this query through input/output batches and sort-finalization work.
    /// Finalization may merge multiple runs; jobs have no fixed duration bound.
    /// The retained task completes cleanup even when the awaiting future drops.
    ///
    /// # Errors
    /// Returns query, resource, cancellation, scheduling or cleanup failures.
    pub async fn execute(mut self) -> Result<QueryResult> {
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_| Error::Internal("async sort requires a Tokio runtime".into()))?;
        let owner = self
            .owner
            .take()
            .ok_or_else(|| Error::Internal("async preparation already consumed".into()))?;
        *owner.cleanup.runtime.lock() = Some(runtime.clone());
        let mut waiter = CancelWaiter {
            handle: Some(owner.control.cancellation_handle()),
            shared: Arc::clone(&owner.shared),
            cleanup: Arc::clone(&owner.cleanup),
        };
        let outcome = runtime
            .spawn(owner.run())
            .await
            .map_err(|error| Error::Io(std::io::Error::other(error)))?;
        waiter.handle = None;
        outcome
    }
}

impl Drop for PreparedAsyncSort {
    fn drop(&mut self) {
        let Some(owner) = self.owner.take() else {
            return;
        };
        owner.control.cancellation_handle().cancel();
        owner.cleanup.start(&owner.shared);
        // Only the physical cleanup job retains the owner now. If no runtime
        // exists, SharedQuery's synchronous last-owner backstop handles it.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::executor::ExecutionOptions;
    use crate::{Config, GrafeoDB};
    use grafeo_common::types::Value;
    use grafeo_common::utils::error::ErrorCode;
    use std::collections::HashMap;
    use std::time::Duration;

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap()
    }

    fn prepare(database: &GrafeoDB, query: &str, options: ExecutionOptions) -> PreparedAsyncSort {
        match database
            .execute_or_prepare_async_sort(query, HashMap::new(), options)
            .unwrap()
        {
            AsyncSortDispatch::Prepared(prepared) => prepared,
            AsyncSortDispatch::Completed(_) => panic!("fixture must exercise scheduled sort"),
        }
    }

    fn block_next_input(
        prepared: &PreparedAsyncSort,
    ) -> (
        tokio::sync::oneshot::Receiver<()>,
        std::sync::mpsc::Sender<()>,
    ) {
        use grafeo_core::execution::operators::{Operator, OperatorResult};
        struct BlockingInput {
            entered: Option<tokio::sync::oneshot::Sender<()>>,
            release: Mutex<std::sync::mpsc::Receiver<()>>,
        }
        impl Operator for BlockingInput {
            fn next(&mut self) -> OperatorResult {
                if let Some(entered) = self.entered.take() {
                    entered.send(()).unwrap();
                    self.release.lock().recv().unwrap();
                }
                Ok(None)
            }
            fn reset(&mut self) {}
            fn name(&self) -> &'static str {
                "owned-blocked-input-fixture"
            }
            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }
        let shared = &prepared.owner.as_ref().unwrap().shared;
        let (entered, wait_entered) = tokio::sync::oneshot::channel();
        let (release, wait_release) = std::sync::mpsc::channel();
        let mut sort = SortOperator::new(
            Box::new(BlockingInput {
                entered: Some(entered),
                release: Mutex::new(wait_release),
            }),
            vec![],
            vec![],
        );
        sort.install_resource_context(&shared.resources).unwrap();
        shared.state.lock().as_mut().unwrap().sort = Some(sort);
        (wait_entered, release)
    }

    const FORCED_QUERY: &str = "MATCH (n:AsyncSort) RETURN n.value AS value ORDER BY n.key";
    fn forced_database(root: &std::path::Path, quota: Option<u64>) -> GrafeoDB {
        let mut config = Config::in_memory()
            .with_memory_limit(2 << 20)
            .with_spill_path(root);
        if let Some(quota) = quota {
            config = config.with_max_query_spill_bytes(quota);
        }
        let database = GrafeoDB::with_config(config).unwrap();
        for i in 0..4096_i64 {
            database.create_node_with_props(
                &["AsyncSort"],
                [
                    ("value", Value::Int64(i)),
                    (
                        "key",
                        Value::from(format!("{:04}-{}", 4095 - i, "x".repeat(512))),
                    ),
                ],
            );
        }
        database
    }

    fn assert_no_leaves(root: &std::path::Path, database: &GrafeoDB) {
        let namespace = root.join(format!("grafeo-store-{}", database.store_id()));
        for entry in std::fs::read_dir(namespace).unwrap() {
            assert!(
                !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with("grafeo-query-")
            );
        }
    }

    #[test]
    fn single_blocking_thread_hidden_keys_and_empty_types_match_sync() {
        runtime().block_on(async {
            let database = Arc::new(GrafeoDB::new_in_memory());
            database.execute("INSERT (:AsyncSort {value: 1, key: 3}), (:AsyncSort {value: 2, key: 1}), (:AsyncSort {value: 3, key: 2})").unwrap();
            for query in [FORCED_QUERY, "MATCH (n:Absent) RETURN n.value AS value ORDER BY n.key",
                "UNWIND [3, 1, 2] AS i RETURN i AS value ORDER BY -i"] {
                let expected = database.execute(query).unwrap();
                let database_copy = Arc::clone(&database);
                let dispatch = tokio::task::spawn_blocking(move || database_copy.execute_or_prepare_async_sort(query, HashMap::new(), ExecutionOptions::default())).await.unwrap().unwrap();
                let result = match dispatch {
                    AsyncSortDispatch::Completed(result) => result,
                    AsyncSortDispatch::Prepared(prepared) => tokio::time::timeout(Duration::from_secs(10), prepared.execute()).await.unwrap().unwrap(),
                };
                assert_eq!(result.columns, expected.columns);
                assert_eq!(result.column_types, expected.column_types);
                assert_eq!(result.rows(), expected.rows());
            }
        });
    }

    #[test]
    fn bounded_literal_sort_completes_cold_and_cached_with_exact_rows() {
        let database = GrafeoDB::new_in_memory();
        let query = "UNWIND range(0, 4095) AS i RETURN i AS value ORDER BY -i";
        // Use an independent ordinary caller so it cannot warm this dispatch's
        // caches; both callers must retain renamed variables in hidden keys.
        let ordinary_database = GrafeoDB::new_in_memory();
        let ordinary = ordinary_database.execute(query).unwrap();
        let ordinary_cached = ordinary_database.execute(query).unwrap();
        assert_eq!(ordinary.rows(), ordinary_cached.rows());
        assert_eq!(ordinary.column_types, ordinary_cached.column_types);
        let mut column_types = None;
        for _ in 0..2 {
            let AsyncSortDispatch::Completed(result) = database
                .execute_or_prepare_async_sort(query, HashMap::new(), ExecutionOptions::default())
                .unwrap()
            else {
                panic!("the complete 4096-row literal source fits the resident work bound");
            };
            assert_eq!(result.columns, ["value"]);
            assert_eq!(result.row_count(), 4096);
            assert_eq!(result.rows(), ordinary.rows());
            assert_eq!(result.column_types, ordinary.column_types);
            for (index, row) in result.rows().iter().enumerate() {
                assert_eq!(row, &[Value::Int64(4095 - i64::try_from(index).unwrap())]);
            }
            if let Some(expected) = &column_types {
                assert_eq!(&result.column_types, expected);
            } else {
                column_types = Some(result.column_types.clone());
            }
        }
        assert_eq!(
            database.execute(query).unwrap().column_types,
            column_types.unwrap()
        );
        let empty = "UNWIND [] AS i RETURN i AS value ORDER BY i";
        let AsyncSortDispatch::Completed(result) = database
            .execute_or_prepare_async_sort(empty, HashMap::new(), ExecutionOptions::default())
            .unwrap()
        else {
            panic!("empty literal source fits the resident work bound");
        };
        let expected = database.execute(empty).unwrap();
        assert_eq!(result.rows(), expected.rows());
        assert_eq!(result.column_types, expected.column_types);

        // Explicit output aliases retain precedence: ORDER BY i here names
        // the projected negative value, not the original positive input.
        let shadowed = "UNWIND [0, 1, 2] AS i RETURN i AS value, -i AS i ORDER BY i";
        let expected = vec![
            vec![Value::Int64(2), Value::Int64(-2)],
            vec![Value::Int64(1), Value::Int64(-1)],
            vec![Value::Int64(0), Value::Int64(0)],
        ];
        for _ in 0..2 {
            let AsyncSortDispatch::Completed(result) = database
                .execute_or_prepare_async_sort(
                    shadowed,
                    HashMap::new(),
                    ExecutionOptions::default(),
                )
                .unwrap()
            else {
                panic!("the three-row literal source fits the resident work bound");
            };
            assert_eq!(result.rows(), expected.as_slice());
            assert_eq!(
                ordinary_database.execute(shadowed).unwrap().rows(),
                expected.as_slice()
            );
        }
    }

    #[test]
    fn literal_work_bound_preserves_scheduling_above_threshold_and_with_spill() {
        runtime().block_on(async {
            let database = GrafeoDB::new_in_memory();
            // A second UNWIND multiplies source work. The bound must account for
            // both sources even though each range independently fits.
            for (query, rows, scheduled) in [
                ("UNWIND range(0, 4096) AS i RETURN i AS value ORDER BY i", 4097, true),
                ("UNWIND range(0, 63) AS i UNWIND range(0, 63) AS j RETURN i * 64 + j AS value ORDER BY value", 4096, false),
                ("UNWIND range(0, 63) AS i UNWIND range(0, 64) AS j RETURN i * 65 + j AS value ORDER BY value", 4160, true),
            ] {
                let dispatch = database.execute_or_prepare_async_sort(query, HashMap::new(), ExecutionOptions::default()).unwrap();
                assert_eq!(matches!(&dispatch, AsyncSortDispatch::Prepared(_)), scheduled);
                let result = match dispatch {
                    AsyncSortDispatch::Completed(result) => result,
                    AsyncSortDispatch::Prepared(prepared) => tokio::time::timeout(Duration::from_secs(10), prepared.execute()).await.unwrap().unwrap(),
                };
                assert_eq!(result.row_count(), rows);
                let ordinary = database.execute(query).unwrap();
                assert_eq!(result.rows(), ordinary.rows());
                assert_eq!(result.column_types, ordinary.column_types);
                for (index, row) in result.rows().iter().enumerate() {
                    assert_eq!(row, &[Value::Int64(i64::try_from(index).unwrap())]);
                }
            }
            // The same parameterized text must be reproved after substitution;
            // a small first invocation cannot authorize a larger later input.
            let query = "UNWIND range(0, $end) AS i RETURN i AS value ORDER BY i";
            for (end, scheduled) in [(63_i64, false), (4096_i64, true)] {
                let dispatch = database.execute_or_prepare_async_sort(query, HashMap::from([("end".into(), Value::Int64(end))]), ExecutionOptions::default()).unwrap();
                assert_eq!(matches!(&dispatch, AsyncSortDispatch::Prepared(_)), scheduled);
                let result = match dispatch {
                    AsyncSortDispatch::Completed(result) => result,
                    AsyncSortDispatch::Prepared(prepared) => prepared.execute().await.unwrap(),
                };
                assert_eq!(result.row_count(), usize::try_from(end + 1).unwrap());
            }
            let root = tempfile::tempdir().unwrap();
            let configured = GrafeoDB::with_config(Config::in_memory().with_spill_path(root.path())).unwrap();
            let result = prepare(&configured, "UNWIND range(0, 63) AS i RETURN i ORDER BY i", ExecutionOptions::default()).execute().await.unwrap();
            assert_eq!(result.row_count(), 64);
            assert_no_leaves(root.path(), &configured);
        });
    }

    #[test]
    fn bounded_literal_sort_honors_cold_and_cached_result_admission() {
        fn refuse(_: &QueryResult, _: ResultLimits) -> Result<()> {
            Err(Error::Internal("binding rejected copied output".into()))
        }
        let query = "UNWIND [3, 1, 2] AS i RETURN i ORDER BY i";
        for warm in [false, true] {
            let database = GrafeoDB::new_in_memory();
            if warm {
                assert!(matches!(
                    database
                        .execute_or_prepare_async_sort(
                            query,
                            HashMap::new(),
                            ExecutionOptions::default()
                        )
                        .unwrap(),
                    AsyncSortDispatch::Completed(_)
                ));
            }
            let error = database.execute_or_prepare_async_sort(
                query,
                HashMap::new(),
                ExecutionOptions {
                    result_limits: Some(ResultLimits {
                        max_rows: 2,
                        ..ResultLimits::default()
                    }),
                    ..ExecutionOptions::default()
                },
            );
            assert!(matches!(error, Err(error) if error.error_code() == ErrorCode::StorageFull));
            let error = database.execute_or_prepare_async_sort(
                query,
                HashMap::new(),
                ExecutionOptions {
                    result_admission: Some(refuse),
                    ..ExecutionOptions::default()
                },
            );
            assert!(
                matches!(error, Err(error) if error.to_string().contains("binding rejected copied output"))
            );
            let control = QueryExecutionControl::new();
            control.cancellation_handle().cancel();
            let error = database.execute_or_prepare_async_sort(
                query,
                HashMap::new(),
                ExecutionOptions {
                    control,
                    ..ExecutionOptions::default()
                },
            );
            assert!(matches!(error, Err(error) if error.error_code() == ErrorCode::QueryCancelled));
        }
    }

    #[test]
    fn hidden_wide_key_forces_spill_and_zero_quota_refuses_same_query() {
        runtime().block_on(async {
            let root = tempfile::tempdir().unwrap();
            let database = forced_database(root.path(), None);
            let result = prepare(&database, FORCED_QUERY, ExecutionOptions::default())
                .execute()
                .await
                .unwrap();
            assert_eq!(result.row_count(), 4096);
            for (index, row) in result.rows().iter().enumerate() {
                assert_eq!(row, &[Value::Int64(4095 - i64::try_from(index).unwrap())]);
            }
            assert_no_leaves(root.path(), &database);
            drop(result);
            let denied_root = tempfile::tempdir().unwrap();
            let denied = forced_database(denied_root.path(), Some(0));
            let error = prepare(&denied, FORCED_QUERY, ExecutionOptions::default())
                .execute()
                .await
                .unwrap_err();
            fn retains_operator(error: &Error) -> bool {
                match error {
                    Error::RetainedContext { source, .. } => {
                        source.inspect::<OperatorError, _>(|_| ()).is_some()
                            || source
                                .inspect::<RetainedErrorContext, _>(|context| {
                                    context.primary().is_some_and(retains_operator)
                                })
                                .unwrap_or(false)
                    }
                    _ => false,
                }
            }
            assert!(
                retains_operator(&error),
                "owned error lost its typed operator authority"
            );
            assert_eq!(error.error_code(), ErrorCode::StorageFull);
            assert!(error.to_string().contains("spill disk quota exceeded"));
            assert_no_leaves(denied_root.path(), &denied);
        });
    }

    #[test]
    fn row_limit_and_cancelled_preparation_preserve_primary_classification() {
        runtime().block_on(async {
            let database = GrafeoDB::new_in_memory();
            database.execute("INSERT (:AsyncSortCap {value: 3}), (:AsyncSortCap {value: 1}), (:AsyncSortCap {value: 2})").unwrap();
            let query = "MATCH (n:AsyncSortCap) RETURN n.value ORDER BY n.value";
            let options = ExecutionOptions {
                result_limits: Some(ResultLimits {
                    max_rows: 2,
                    ..ResultLimits::default()
                }),
                ..ExecutionOptions::default()
            };
            let error = prepare(&database, query, options)
                .execute()
                .await
                .unwrap_err();
            assert_eq!(error.error_code(), ErrorCode::StorageFull);
            let control = QueryExecutionControl::new();
            control.cancellation_handle().cancel();
            let result = database.execute_or_prepare_async_sort(
                query,
                HashMap::new(),
                ExecutionOptions {
                    control,
                    ..ExecutionOptions::default()
                },
            );
            assert!(
                matches!(result, Err(error) if error.error_code() == ErrorCode::QueryCancelled)
            );
        });
    }

    #[test]
    fn reserved_cleanup_survives_exhausted_query_memory() {
        runtime().block_on(async {
            let database = GrafeoDB::new_in_memory();
            let prepared = prepare(
                &database,
                "MATCH (n:AsyncSortFixture) RETURN n.value AS value ORDER BY n.value",
                ExecutionOptions::default(),
            );
            let shared = Arc::clone(&prepared.owner.as_ref().unwrap().shared);
            let stats = shared.resources.query_stats();
            let available = stats.growth_limit_bytes - stats.allocated_bytes;
            let exhaustion = shared.resources.try_allocate(available).unwrap();
            let error = prepared.execute().await.unwrap_err();
            assert_eq!(error.error_code(), ErrorCode::StorageFull);
            assert!(shared.leaf_finish_attempted.load(Ordering::Acquire));
            assert!(
                shared.state.lock().is_none(),
                "terminal state extracted only after explicit cleanup"
            );
            drop(exhaustion);
        });
    }

    #[test]
    fn abandoned_preparation_and_dropped_waiter_retain_cleanup_owner() {
        runtime().block_on(async {
            let database = GrafeoDB::new_in_memory();
            let query = "UNWIND range(0, 8191) AS i RETURN i ORDER BY i";
            let prepared = prepare(&database, query, ExecutionOptions::default());
            let shared = Arc::clone(&prepared.owner.as_ref().unwrap().shared);
            let token = shared.resources.cancellation_token().clone();
            drop(prepared);
            tokio::time::timeout(Duration::from_secs(10), async {
                while !shared.leaf_finish_attempted.load(Ordering::Acquire) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert!(token.check().is_err());
            assert!(shared.state.lock().as_ref().unwrap().cleaned);
            drop(shared);

            let prepared = prepare(&database, query, ExecutionOptions::default());
            let shared = Arc::clone(&prepared.owner.as_ref().unwrap().shared);
            let (entered_rx, release_tx) = block_next_input(&prepared);
            let mut waiter = Box::pin(prepared.execute());
            std::future::poll_fn(|context| {
                assert!(waiter.as_mut().poll(context).is_pending());
                std::task::Poll::Ready(())
            })
            .await;
            entered_rx.await.unwrap();
            drop(waiter);
            release_tx.send(()).unwrap();
            tokio::time::timeout(Duration::from_secs(10), async {
                while shared.state.lock().is_some() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert!(shared.resources.check_cancelled().is_err());
            assert!(shared.state.lock().is_none());
        });
    }

    #[test]
    fn close_after_abandon_does_not_need_another_runtime_poll() {
        for started in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let database =
                GrafeoDB::with_config(Config::in_memory().with_spill_path(root.path())).unwrap();
            runtime().block_on(async {
                let prepared = prepare(
                    &database,
                    "UNWIND [3, 1, 2] AS i RETURN i ORDER BY i",
                    ExecutionOptions::default(),
                );
                let shared = Arc::clone(&prepared.owner.as_ref().unwrap().shared);
                shared.resources.ensure_spill_manager().unwrap();
                if started {
                    let (entered, release) = block_next_input(&prepared);
                    let mut waiter = Box::pin(prepared.execute());
                    std::future::poll_fn(|context| {
                        assert!(waiter.as_mut().poll(context).is_pending());
                        std::task::Poll::Ready(())
                    })
                    .await;
                    entered.await.unwrap();
                    drop(waiter);
                    release.send(()).unwrap();
                } else {
                    drop(prepared);
                }
                // Deliberately synchronous: cleanup must run on the single
                // physical worker while this runtime thread waits for close.
                database.close().unwrap();
                assert!(shared.publication.lock().is_none());
                assert!(shared.leaf_finish_attempted.load(Ordering::Acquire));
                assert_no_leaves(root.path(), &database);
            });
        }
    }

    #[test]
    fn dropping_started_waiter_outside_entered_runtime_still_closes() {
        let database = GrafeoDB::new_in_memory();
        let runtime = runtime();
        let (waiter, release) = runtime.block_on(async {
            let prepared = prepare(
                &database,
                "MATCH (n:AsyncSortFixture) RETURN n.value AS value ORDER BY n.value",
                ExecutionOptions::default(),
            );
            let (entered, release) = block_next_input(&prepared);
            let mut waiter = Box::pin(prepared.execute());
            std::future::poll_fn(|context| {
                assert!(waiter.as_mut().poll(context).is_pending());
                std::task::Poll::Ready(())
            })
            .await;
            entered.await.unwrap();
            (waiter, release)
        });
        assert!(tokio::runtime::Handle::try_current().is_err());
        drop(waiter);
        release.send(()).unwrap();
        // The retained async owner cannot poll on this current-thread runtime.
        // Its saved handle must still queue the physical terminal ticket.
        database.close().unwrap();
    }

    #[test]
    fn undelivered_preparation_retains_last_database_owner_until_cleanup() {
        let database = Arc::new(GrafeoDB::new_in_memory());
        let weak = Arc::downgrade(&database);
        runtime().block_on(async {
            let preparation = tokio::task::spawn_blocking(move || {
                prepare(
                    &database,
                    "MATCH (n:AsyncSortFixture) RETURN n.value AS value ORDER BY n.value",
                    ExecutionOptions::default(),
                )
                .retain_database_owner(database.clone())
            });
            // The initial blocking closure and its result now own every strong
            // database reference. Its undelivered result must queue cleanup
            // behind this worker, without closing under the publication guard.
            drop(preparation);
            tokio::time::timeout(Duration::from_secs(10), async {
                while weak.strong_count() != 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
        });
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn cleanup_unwind_still_finishes_leaf_on_physical_worker() {
        use grafeo_core::execution::operators::{Operator, OperatorResult};
        struct PanicOnDrop;
        impl Operator for PanicOnDrop {
            fn next(&mut self) -> OperatorResult {
                Ok(None)
            }
            fn reset(&mut self) {}
            fn name(&self) -> &'static str {
                "panic-drop-fixture"
            }
            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }
        impl Drop for PanicOnDrop {
            fn drop(&mut self) {
                std::panic::panic_any("injected child destruction");
            }
        }
        runtime().block_on(async {
            let root = tempfile::tempdir().unwrap();
            let database =
                GrafeoDB::with_config(Config::in_memory().with_spill_path(root.path())).unwrap();
            let prepared = prepare(
                &database,
                "RETURN 1 AS value ORDER BY value",
                ExecutionOptions::default(),
            );
            let shared = Arc::clone(&prepared.owner.as_ref().unwrap().shared);
            shared.resources.ensure_spill_manager().unwrap();
            let mut sort = SortOperator::new(Box::new(PanicOnDrop), vec![], vec![]);
            sort.install_resource_context(&shared.resources).unwrap();
            shared.state.lock().as_mut().unwrap().sort = Some(sort);
            let error = prepared.execute().await.unwrap_err();
            assert_eq!(error.error_code(), ErrorCode::Internal);
            assert!(error.to_string().contains("physical operation panicked"));
            assert!(shared.leaf_finish_attempted.load(Ordering::Acquire));
            assert_no_leaves(root.path(), &database);
        });
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    mod failure_parity {
        use super::*;
        use grafeo_core::execution::spill::{SpillIo, SpillIoOperation};
        use std::sync::atomic::AtomicUsize;

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Primary {
            Success,
            RowLimit,
            Cancel,
            MergeSuccess,
            MergeCancel,
            MergeIo,
            MergeIoAndCancel,
            Io {
                operation: SpillIoOperation,
                occurrence: usize,
            },
        }

        const PRIMARIES: [Primary; 13] = [
            Primary::Success,
            Primary::RowLimit,
            Primary::Cancel,
            Primary::Io {
                operation: SpillIoOperation::Create,
                occurrence: 1,
            },
            Primary::Io {
                operation: SpillIoOperation::WriteHeader,
                occurrence: 1,
            },
            Primary::Io {
                operation: SpillIoOperation::WritePayload,
                occurrence: 1,
            },
            Primary::Io {
                operation: SpillIoOperation::Flush,
                occurrence: 1,
            },
            Primary::Io {
                operation: SpillIoOperation::Sync,
                occurrence: 1,
            },
            Primary::Io {
                operation: SpillIoOperation::ReadOpen,
                occurrence: 1,
            },
            Primary::Io {
                operation: SpillIoOperation::ReadHeader,
                occurrence: 1,
            },
            Primary::Io {
                operation: SpillIoOperation::ReadPayload,
                occurrence: 1,
            },
            Primary::Io {
                operation: SpillIoOperation::Delete,
                occurrence: 1,
            },
            Primary::Io {
                operation: SpillIoOperation::ReadPayload,
                occurrence: 1024,
            },
        ];

        const MERGE_PRIMARIES: [Primary; 4] = [
            Primary::MergeSuccess,
            Primary::MergeCancel,
            Primary::MergeIo,
            Primary::MergeIoAndCancel,
        ];
        // The qualified 32768-row fixture produced sixteen 2048-row runs.
        // Eighteen runs reach the next carry: after the first sixteen-way
        // merge, batch eighteen has one trailing zero and merges batches
        // seventeen/eighteen. Keep the existing budget, keys and deadline.
        const MERGE_ROWS: usize = 36864;
        const MERGE_BUDGET: usize = 8 << 20;
        const MERGE_KEY_BYTES: usize = 2048;

        impl Primary {
            fn succeeds(self) -> bool {
                matches!(self, Self::Success | Self::MergeSuccess)
            }

            fn merge(self) -> bool {
                matches!(
                    self,
                    Self::MergeSuccess | Self::MergeCancel | Self::MergeIo | Self::MergeIoAndCancel
                )
            }
        }

        struct Faults {
            primary: Primary,
            cancel: QueryCancellationHandle,
            data_file_started: AtomicBool,
            creates: AtomicUsize,
            writes: AtomicUsize,
            read_opens: AtomicUsize,
            read_payloads: AtomicUsize,
            matching_operations: AtomicUsize,
            injected: AtomicUsize,
            work_after_failure: AtomicUsize,
            writer_active: AtomicBool,
            writer_has_reads: AtomicBool,
            initial_publications: AtomicUsize,
            intermediate_read_opens: AtomicUsize,
            intermediate_payloads: AtomicUsize,
            merge_triggered: AtomicBool,
            initial_at_trigger: AtomicUsize,
            cleanup_attempts: AtomicUsize,
            cleanup_entered: AtomicBool,
            fail_cleanup: AtomicBool,
            released: Mutex<bool>,
            release: parking_lot::Condvar,
        }

        impl Faults {
            fn release_cleanup(&self) {
                *self.released.lock() = true;
                self.release.notify_all();
            }
        }

        // Also releases a blocked worker if an assertion or timeout unwinds.
        struct ReleaseCleanup(Arc<Faults>);
        impl Drop for ReleaseCleanup {
            fn drop(&mut self) {
                self.0.release_cleanup();
            }
        }

        impl SpillIo for Faults {
            fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
                if operation == SpillIoOperation::Create {
                    // Owner-marker writes also use WritePayload. The first
                    // data-file Create occurs only after lazy leaf admission.
                    self.data_file_started.store(true, Ordering::Release);
                    self.creates.fetch_add(1, Ordering::AcqRel);
                    self.writer_active.store(true, Ordering::Release);
                    self.writer_has_reads.store(false, Ordering::Release);
                }
                if self.data_file_started.load(Ordering::Acquire) {
                    let query_work = matches!(
                        operation,
                        SpillIoOperation::Create
                            | SpillIoOperation::WriteHeader
                            | SpillIoOperation::WritePayload
                            | SpillIoOperation::Flush
                            | SpillIoOperation::Sync
                            | SpillIoOperation::ReadOpen
                            | SpillIoOperation::ReadHeader
                            | SpillIoOperation::ReadPayload
                    );
                    if query_work && self.injected.load(Ordering::Acquire) != 0 {
                        self.work_after_failure.fetch_add(1, Ordering::AcqRel);
                    }
                    match operation {
                        SpillIoOperation::WritePayload => {
                            let prior = self.writes.fetch_add(1, Ordering::AcqRel);
                            if self.primary == Primary::Cancel && prior == 0 {
                                self.cancel.cancel();
                            }
                        }
                        SpillIoOperation::ReadOpen => {
                            self.read_opens.fetch_add(1, Ordering::AcqRel);
                            if self.writer_active.load(Ordering::Acquire) {
                                self.writer_has_reads.store(true, Ordering::Release);
                                self.intermediate_read_opens.fetch_add(1, Ordering::AcqRel);
                            }
                        }
                        SpillIoOperation::ReadPayload => {
                            self.read_payloads.fetch_add(1, Ordering::AcqRel);
                        }
                        _ => {}
                    }
                    if operation == SpillIoOperation::Sync
                        && self.writer_active.swap(false, Ordering::AcqRel)
                        && !self.writer_has_reads.load(Ordering::Acquire)
                    {
                        self.initial_publications.fetch_add(1, Ordering::AcqRel);
                    }
                    if self.primary.merge()
                        && operation == SpillIoOperation::WritePayload
                        && self.writer_active.load(Ordering::Acquire)
                        && self.writer_has_reads.load(Ordering::Acquire)
                        && self.initial_publications.load(Ordering::Acquire) > 16
                    {
                        let payload = self.intermediate_payloads.fetch_add(1, Ordering::AcqRel);
                        // One intermediate output payload has already completed
                        // after its readers opened. Fault the following payload.
                        if payload == 1 {
                            self.merge_triggered.store(true, Ordering::Release);
                            self.initial_at_trigger.store(
                                self.initial_publications.load(Ordering::Acquire),
                                Ordering::Release,
                            );
                            if matches!(
                                self.primary,
                                Primary::MergeCancel | Primary::MergeIoAndCancel
                            ) {
                                self.cancel.cancel();
                            }
                            if self.primary != Primary::MergeSuccess {
                                self.injected.fetch_add(1, Ordering::AcqRel);
                            }
                            if matches!(self.primary, Primary::MergeIo | Primary::MergeIoAndCancel)
                            {
                                return Err(std::io::ErrorKind::BrokenPipe.into());
                            }
                        }
                    }
                    if let Primary::Io {
                        operation: target,
                        occurrence,
                    } = self.primary
                        && operation == target
                        && self.matching_operations.fetch_add(1, Ordering::AcqRel) + 1 == occurrence
                    {
                        self.injected.fetch_add(1, Ordering::AcqRel);
                        return Err(std::io::ErrorKind::BrokenPipe.into());
                    }
                }
                if operation == SpillIoOperation::RemoveQueryDirectory {
                    self.cleanup_attempts.fetch_add(1, Ordering::AcqRel);
                    self.cleanup_entered.store(true, Ordering::Release);
                    let mut released = self.released.lock();
                    while !*released {
                        self.release.wait(&mut released);
                    }
                    if self.fail_cleanup.load(Ordering::Acquire) {
                        return Err(std::io::ErrorKind::PermissionDenied.into());
                    }
                }
                Ok(())
            }

            // Synchronization state is constructed before admission. No hook
            // grows shared state; only allocation-free ErrorKind values escape.
            fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
                Some(0)
            }
            fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
                Some(0)
            }
        }

        fn database(path: &std::path::Path, encrypted: bool, multipass: bool) -> GrafeoDB {
            let config = Config::in_memory()
                .with_memory_limit(if multipass { MERGE_BUDGET } else { 3 << 20 })
                .with_spill_path(path);
            #[cfg(feature = "encryption")]
            let config = {
                let mut config = config;
                if encrypted {
                    config.encryption = Some(crate::config::EncryptionConfig {
                        key_chain: Arc::new(grafeo_common::encryption::KeyChain::new([71; 32])),
                    });
                }
                config
            };
            #[cfg(not(feature = "encryption"))]
            assert!(!encrypted);
            let database = GrafeoDB::with_config(config).unwrap();
            let rows = if multipass { MERGE_ROWS } else { 4096 };
            let key_bytes = if multipass { MERGE_KEY_BYTES } else { 128 };
            for i in 0..i64::try_from(rows).unwrap() {
                let descending = i64::try_from(rows).unwrap() - 1 - i;
                database.create_node_with_props(
                    &["AsyncSort"],
                    [
                        ("value", Value::Int64(if multipass { i / 2 } else { i })),
                        (
                            "key",
                            Value::from(format!(
                                "{:0width$}-{}",
                                if multipass {
                                    descending / 2
                                } else {
                                    descending
                                },
                                "x".repeat(key_bytes),
                                width = if multipass { 5 } else { 4 }
                            )),
                        ),
                    ],
                );
            }
            database
        }

        fn typed_primary(error: &Error, expected: Primary) -> bool {
            match error {
                Error::Context { source, .. } => typed_primary(source, expected),
                Error::Storage(grafeo_common::utils::error::StorageError::Full) => {
                    expected == Primary::RowLimit
                }
                Error::RetainedContext { source, .. } => {
                    source
                        .inspect::<OperatorError, _>(|operator| match expected {
                            Primary::Io { .. } | Primary::MergeIo | Primary::MergeIoAndCancel => {
                                matches!(
                                    operator,
                                    OperatorError::ClassifiedAccountedFailure {
                                        classification: AccountedFailureClassification::Execution,
                                        authority,
                                    } if authority.granted_bytes() > 0
                                )
                            }
                            Primary::Cancel | Primary::MergeCancel => {
                                operator_code(operator) == ErrorCode::QueryCancelled
                            }
                            _ => false,
                        })
                        .unwrap_or(false)
                        || source
                            .inspect::<RetainedErrorContext, _>(|context| {
                                context
                                    .primary()
                                    .is_some_and(|primary| typed_primary(primary, expected))
                            })
                            .unwrap_or(false)
                }
                _ => false,
            }
        }

        fn cleanup_secondary(error: &Error) -> bool {
            match error {
                Error::RetainedContext { source, .. } => source
                    .inspect::<RetainedErrorContext, _>(|context| {
                        matches!(context.secondary(), Some(Error::Io(error)) if error.kind() == std::io::ErrorKind::PermissionDenied)
                            || context.primary().is_some_and(cleanup_secondary)
                    })
                    .unwrap_or(false),
                _ => false,
            }
        }

        fn primary_io_kind(error: &Error) -> Option<std::io::ErrorKind> {
            match error {
                Error::Io(error) => Some(error.kind()),
                Error::Context { source, .. } => primary_io_kind(source),
                Error::RetainedContext { source, .. } => source
                    .inspect::<RetainedErrorContext, _>(|context| {
                        context.primary().and_then(primary_io_kind)
                    })
                    .flatten(),
                _ => None,
            }
        }

        fn primary_operator_diagnostic(error: &Error) -> Option<String> {
            match error {
                Error::Context { source, .. } => primary_operator_diagnostic(source),
                Error::RetainedContext { source, .. } => source
                    .inspect::<OperatorError, _>(ToString::to_string)
                    .or_else(|| {
                        source
                            .inspect::<RetainedErrorContext, _>(|context| {
                                context.primary().and_then(primary_operator_diagnostic)
                            })
                            .flatten()
                    }),
                _ => None,
            }
        }

        #[test]
        fn real_sort_primary_and_leaf_cleanup_failure_matrix() {
            run_matrix(false);
        }

        #[test]
        fn real_sort_intermediate_merge_failure_matrix() {
            run_matrix(true);
        }

        fn run_matrix(multipass: bool) {
            runtime().block_on(async {
                for encrypted in [false, true] {
                    #[cfg(not(feature = "encryption"))]
                    if encrypted {
                        continue;
                    }
                    for &primary in if multipass { MERGE_PRIMARIES.as_slice() } else { PRIMARIES.as_slice() } {
                        for fail_cleanup in [false, true] {
                            let root = tempfile::tempdir().unwrap();
                            let database = database(root.path(), encrypted, multipass);
                            let control = QueryExecutionControl::new();
                            let faults = Arc::new(Faults {
                                primary,
                                cancel: control.cancellation_handle(),
                                data_file_started: AtomicBool::new(false),
                                creates: AtomicUsize::new(0),
                                writes: AtomicUsize::new(0),
                                read_opens: AtomicUsize::new(0),
                                read_payloads: AtomicUsize::new(0),
                                matching_operations: AtomicUsize::new(0),
                                injected: AtomicUsize::new(0),
                                work_after_failure: AtomicUsize::new(0),
                                writer_active: AtomicBool::new(false),
                                writer_has_reads: AtomicBool::new(false),
                                initial_publications: AtomicUsize::new(0),
                                intermediate_read_opens: AtomicUsize::new(0),
                                intermediate_payloads: AtomicUsize::new(0),
                                merge_triggered: AtomicBool::new(false),
                                initial_at_trigger: AtomicUsize::new(0),
                                cleanup_attempts: AtomicUsize::new(0),
                                cleanup_entered: AtomicBool::new(false),
                                fail_cleanup: AtomicBool::new(fail_cleanup),
                                released: Mutex::new(false),
                                release: parking_lot::Condvar::new(),
                            });
                            crate::database::testing::install_spill_io(&database, faults.clone());
                            let release = ReleaseCleanup(Arc::clone(&faults));
                            let prepared = prepare(&database, FORCED_QUERY, ExecutionOptions {
                                control,
                                result_limits: (primary == Primary::RowLimit).then_some(ResultLimits {
                                    max_rows: 1,
                                    ..ResultLimits::default()
                                }),
                                ..ExecutionOptions::default()
                            });
                            let shared = Arc::clone(&prepared.owner.as_ref().unwrap().shared);
                            let mut waiter = tokio::spawn(prepared.execute());
                            tokio::time::timeout(Duration::from_secs(10), async {
                                while !faults.cleanup_entered.load(Ordering::Acquire) {
                                    if waiter.is_finished() {
                                        let terminal = (&mut waiter).await;
                                        panic!("query missed physical cleanup: {primary:?}, encrypted={encrypted}, cleanup_failure={fail_cleanup}, terminal={terminal:?}");
                                    }
                                    tokio::task::yield_now().await;
                                }
                            }).await.unwrap();
                            assert!(faults.data_file_started.load(Ordering::Acquire), "fixture must create a data file after admitted query metadata");
                            assert!(faults.creates.load(Ordering::Acquire) > 0);
                            if !matches!(primary, Primary::Io { operation: SpillIoOperation::Create | SpillIoOperation::WriteHeader, .. }) {
                                assert!(faults.writes.load(Ordering::Acquire) > 0, "fixture must spill through its real operator");
                            } else {
                                assert_eq!(faults.writes.load(Ordering::Acquire), 0, "early writer fault precedes payloads");
                            }
                            assert!(!waiter.is_finished());
                            assert!(shared.publication.lock().is_some(), "publication must outlive physical cleanup");
                            assert!(shared.state.lock().as_ref().unwrap().cleaned);
                            let manager = Arc::clone(shared.resources.spill_manager().unwrap());
                            assert_eq!(manager.active_file_count(), 0);
                            drop(release);
                            let result = waiter.await.unwrap();
                            assert!(shared.publication.lock().is_none());
                            assert!(shared.state.lock().is_none());
                            assert_eq!(faults.cleanup_attempts.load(Ordering::Acquire), 1);
                            if multipass {
                                assert!(faults.merge_triggered.load(Ordering::Acquire), "fixture never reached the declared intermediate output phase: primary={primary:?}, encrypted={encrypted}, cleanup_failure={fail_cleanup}, initial_publications={}, creates={}, read_opens={}, intermediate_read_opens={}, writes={}, read_payloads={}, intermediate_payloads={}, writer_active={}, writer_has_reads={}, terminal={:?}",
                                    faults.initial_publications.load(Ordering::Acquire),
                                    faults.creates.load(Ordering::Acquire),
                                    faults.read_opens.load(Ordering::Acquire),
                                    faults.intermediate_read_opens.load(Ordering::Acquire),
                                    faults.writes.load(Ordering::Acquire),
                                    faults.read_payloads.load(Ordering::Acquire),
                                    faults.intermediate_payloads.load(Ordering::Acquire),
                                    faults.writer_active.load(Ordering::Acquire),
                                    faults.writer_has_reads.load(Ordering::Acquire),
                                    result.as_ref().map(|value| value.row_count()).map_err(|error| (error.error_code(), error.to_string())));
                                assert!(faults.initial_at_trigger.load(Ordering::Acquire) > 16);
                                assert!(faults.intermediate_payloads.load(Ordering::Acquire) >= 2);
                                assert_eq!(faults.injected.load(Ordering::Acquire), usize::from(!primary.succeeds()));
                                assert_eq!(faults.work_after_failure.load(Ordering::Acquire), 0, "intermediate failure resumed query I/O: {primary:?}");
                            }
                            if let Primary::Io { operation, occurrence } = primary {
                                assert_eq!(faults.injected.load(Ordering::Acquire), 1, "exactly one physical fault must execute: {primary:?}");
                                assert_eq!(faults.work_after_failure.load(Ordering::Acquire), 0, "physical failure must not resume query work: {primary:?}");
                                if operation != SpillIoOperation::Delete {
                                    assert_eq!(faults.matching_operations.load(Ordering::Acquire), occurrence, "physical failure must not retry query work: {primary:?}");
                                }
                                if matches!(operation, SpillIoOperation::ReadOpen | SpillIoOperation::ReadHeader | SpillIoOperation::ReadPayload) {
                                    assert!(faults.read_opens.load(Ordering::Acquire) > 0, "read fault must reach published-run merge readers");
                                }
                                if operation == SpillIoOperation::ReadPayload && occurrence > 1 {
                                    // Each opened sort run has only FileStart,
                                    // SortRunStart and FileEnd control payloads.
                                    // This proves sustained data consumption,
                                    // not just the first reader's control open.
                                    assert!(faults.read_payloads.load(Ordering::Acquire) > 3 * faults.read_opens.load(Ordering::Acquire), "late read must advance beyond control payloads");
                                }
                            }
                            let retained_error = if primary.succeeds() && !fail_cleanup {
                                let result = result.unwrap();
                                let rows = if multipass { MERGE_ROWS } else { 4096 };
                                assert_eq!(result.row_count(), rows);
                                for (index, row) in result.rows().iter().enumerate() {
                                    let value = i64::try_from(rows - 1 - index).unwrap();
                                    assert_eq!(row, &[Value::Int64(if multipass { value / 2 } else { value })]);
                                }
                                None
                            } else {
                                let error = result.unwrap_err();
                                let expected = match primary {
                                    Primary::Success | Primary::MergeSuccess => ErrorCode::IoError,
                                    Primary::RowLimit => ErrorCode::StorageFull,
                                    Primary::Cancel | Primary::MergeCancel => ErrorCode::QueryCancelled,
                                    Primary::Io { .. } | Primary::MergeIo | Primary::MergeIoAndCancel => ErrorCode::Internal,
                                };
                                assert_eq!(error.error_code(), expected, "{primary:?}, encrypted={encrypted}: {error}");
                                if !primary.succeeds() {
                                    assert!(typed_primary(&error, primary), "typed primary was replaced: {error}");
                                    assert_eq!(cleanup_secondary(&error), fail_cleanup, "secondary leaf error: {error}");
                                } else {
                                    assert_eq!(primary_io_kind(&error), Some(std::io::ErrorKind::PermissionDenied));
                                }
                                let io_operation = match primary {
                                    Primary::Io { operation, .. } => Some(operation),
                                    Primary::MergeIo | Primary::MergeIoAndCancel => Some(SpillIoOperation::WritePayload),
                                    _ => None,
                                };
                                if let Some(operation) = io_operation {
                                    let diagnostic = primary_operator_diagnostic(&error)
                                        .unwrap_or_else(|| panic!("I/O primary owner unavailable: {primary:?}: {error}"));
                                    // Qualified readers expose the copy-only kind,
                                    // keeping opaque provider payloads behind their
                                    // accounted owner rather than formatting them.
                                    let expected = if matches!(operation, SpillIoOperation::ReadOpen | SpillIoOperation::ReadHeader | SpillIoOperation::ReadPayload) {
                                        "qualified reader I/O failure (BrokenPipe)".to_owned()
                                    } else {
                                        std::io::Error::from(std::io::ErrorKind::BrokenPipe).to_string()
                                    };
                                    assert!(diagnostic.contains(&expected), "wrong retained I/O primary: {primary:?}, expected={expected:?}, primary={diagnostic:?}, combined={error}");
                                }
                                Some(error)
                            };
                            let stats = shared.resources.profile_stats().spill_physical.unwrap();
                            assert_eq!(stats.cleanup_failed, fail_cleanup);
                            assert_eq!(stats.observed_file_bytes, 0);
                            if fail_cleanup {
                                assert!(manager.spill_dir().is_dir());
                                assert!(stats.cleanup_debt_bytes > 0);
                                assert_eq!(stats.cleanup_debt_bytes, stats.reserved_bytes);
                                faults.fail_cleanup.store(false, Ordering::Release);
                                manager.finish_query().unwrap();
                            } else {
                                assert_eq!(stats.cleanup_debt_bytes, 0);
                                assert_eq!(stats.reserved_bytes, 0);
                            }
                            let recovered = shared.resources.profile_stats().spill_physical.unwrap();
                            assert_eq!(recovered.reserved_bytes, 0);
                            assert_eq!(recovered.cleanup_debt_bytes, 0);
                            assert!(!recovered.cleanup_failed);
                            assert_no_leaves(root.path(), &database);
                            if let Some(error) = retained_error {
                                let retained_bytes = shared.resources.query_stats().allocated_bytes;
                                drop(error);
                                assert!(shared.resources.query_stats().allocated_bytes < retained_bytes, "escaped diagnostics must retain grants until destruction");
                            }
                            drop(manager);
                            database.close().unwrap();
                            eprintln!(
                                "accepted async sort primary={primary:?} encrypted={encrypted} cleanup_failure={fail_cleanup} creates={} payload_writes={} read_opens={} read_payloads={} injected={}",
                                faults.creates.load(Ordering::Acquire),
                                faults.writes.load(Ordering::Acquire),
                                faults.read_opens.load(Ordering::Acquire),
                                faults.read_payloads.load(Ordering::Acquire),
                                faults.injected.load(Ordering::Acquire),
                            );
                            if multipass {
                                eprintln!("accepted intermediate merge initial_publications_at_trigger={} intermediate_payloads={} rows={MERGE_ROWS} budget={MERGE_BUDGET} key_bytes={MERGE_KEY_BYTES}", faults.initial_at_trigger.load(Ordering::Acquire), faults.intermediate_payloads.load(Ordering::Acquire));
                            }
                        }
                    }
                }
            });
        }
    }

    #[test]
    fn unsupported_shapes_use_existing_complete_path() {
        let database = GrafeoDB::new_in_memory();
        for query in [
            "INSERT (:Fallback {value: 1})",
            "MATCH (n:Fallback) RETURN n.value ORDER BY n.value LIMIT 1",
            "RETURN 7",
        ] {
            assert!(matches!(
                database
                    .execute_or_prepare_async_sort(
                        query,
                        HashMap::new(),
                        ExecutionOptions::default()
                    )
                    .unwrap(),
                AsyncSortDispatch::Completed(_)
            ));
        }
        assert_eq!(
            database
                .execute("MATCH (n:Fallback) RETURN n.value")
                .unwrap()
                .rows(),
            &[vec![Value::Int64(1)]]
        );
    }
}
