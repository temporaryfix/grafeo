//! Lazy, cursor-based query result streams.
//!
//! Today `Session::execute` drains the entire operator pipeline into a
//! `QueryResult { rows: Vec<Vec<Value>> }` before returning. For large result
//! sets this either exhausts memory or forces the caller to wait until the
//! final row has been produced before they can see the first one.
//!
//! `ResultStream` exposes the pipeline lazily: the consumer pulls one
//! admitted `StreamChunk` at a time from the root operator. Each output retains
//! its own query-memory reservation until the caller drops it. Dropping
//! the stream releases the operator tree and decrements the owning session's
//! `active_streams` counter.
//!
//! # Stability: Experimental
//!
//! This module is new in 0.5.40. Signatures may change before being promoted
//! to Beta. Use from embedded callers that want first-row latency or bounded
//! memory; use `Session::execute` when you want a fully materialized result.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use grafeo_common::memory::buffer::MemoryGrant;

use grafeo_common::types::{LogicalType, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::execution::operators::Operator;
use grafeo_core::execution::{
    DataChunk, QueryExecutionControl, QueryExecutionId, QueryLifecycleError, QueryResourceContext,
};

use super::{ResultAccumulator, ResultLimits};
use crate::database::QueryResult;
use crate::query::profile::ProfileNode;

/// RAII guard that increments/decrements a session's active-stream counter.
///
/// The counter prevents `commit()` / `rollback()` from racing with in-flight
/// streams that still hold references to the session's read snapshot.
pub(crate) struct StreamGuard<'s> {
    counter: &'s AtomicUsize,
}

impl<'s> StreamGuard<'s> {
    pub(crate) fn new(counter: &'s AtomicUsize) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self { counter }
    }
}

impl Drop for StreamGuard<'_> {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Observable lifecycle of a lazy query, including partial PROFILE results.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamStatus {
    /// More chunks may be pulled.
    Running,
    /// The operator reached EOF and mandatory cleanup succeeded.
    Completed,
    /// The caller closed the query before EOF.
    Closed,
    /// Execution, cancellation, or cleanup failed.
    Failed,
}

/// Admitted output from one stream pull.
///
/// This move-only owner retains its query-memory reservation independently of
/// the stream, including after EOF, cancellation, or explicit close. Access is
/// borrowed: independently cloning rows or values creates copies outside this
/// chunk's reservation. Upstream operator storage is accounted separately.
#[derive(Debug)]
#[must_use = "retain the chunk while using its admitted output"]
pub struct StreamChunk {
    // Declaration order keeps both physical and cached storage ahead of grants.
    chunk: DataChunk,
    rows: OnceLock<std::result::Result<Vec<Vec<Value>>, ()>>,
    schema: Arc<QueryResult>,
    grant: MemoryGrant,
}

impl StreamChunk {
    /// Borrowed logical rows, initialized once within the admitted capacity.
    ///
    /// # Errors
    /// Returns a resource error if the allocator cannot supply the reserved row
    /// view, or reports a capacity larger than the admitted maximum.
    pub fn rows(&self) -> Result<&[Vec<Value>]> {
        self.rows
            .get_or_init(|| {
                let mut rows = Vec::new();
                rows.try_reserve_exact(self.row_count()).map_err(|_| ())?;
                if rows.capacity() > self.row_count() {
                    return Err(());
                }
                let width = self.chunk.column_count();
                for index in 0..self.row_count() {
                    let mut row = Vec::new();
                    row.try_reserve_exact(width).map_err(|_| ())?;
                    if row.capacity() > width {
                        return Err(());
                    }
                    let physical = self.physical_row(index);
                    for column in self.chunk.columns() {
                        row.push(column.get_value(physical).unwrap_or(Value::Null));
                    }
                    rows.push(row);
                }
                Ok(rows)
            })
            .as_ref()
            .map(Vec::as_slice)
            .map_err(|()| {
                super::result_full("stream row-view allocation exceeded admitted capacity")
            })
    }

    /// Number of logical rows in this output.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.chunk.row_count()
    }

    /// Column names in row order.
    #[must_use]
    pub fn columns(&self) -> &[String] {
        &self.schema.columns
    }

    /// Logical column types for this output.
    #[must_use]
    pub fn column_types(&self) -> &[LogicalType] {
        &self.schema.column_types
    }

    fn physical_row(&self, index: usize) -> usize {
        self.chunk
            .selection()
            .map_or(index, |selection| usize::from(selection.as_slice()[index]))
    }

    // Returned rows belong to the caller; the immutable cache is never built
    // for this path, and no selection-index vector is retained.
    fn take_adapter_row(&self, index: usize) -> Vec<Value> {
        let physical = self.physical_row(index);
        self.chunk
            .columns()
            .iter()
            .map(|column| column.get_value(physical).unwrap_or(Value::Null))
            .collect()
    }

    /// Query-memory bytes retained by this output owner.
    #[must_use]
    pub fn granted_bytes(&self) -> usize {
        self.schema
            .result_reservation
            .as_ref()
            .map_or(0, |reservation| reservation.size())
            + self.grant.size()
    }
}

/// The unique owner shared by borrowed and database-owned stream wrappers.
pub(crate) struct StreamExecution {
    operator: Option<Box<dyn Operator>>,
    columns: Vec<String>,
    column_types: Vec<LogicalType>,
    control: QueryExecutionControl,
    resources: Option<QueryResourceContext>,
    output_schema: Option<Arc<QueryResult>>,
    query_id: QueryExecutionId,
    status: StreamStatus,
    cleanup_failure: Option<(std::io::ErrorKind, String)>,
    profile: Option<ProfileNode>,
    result_limits: Option<ResultLimits>,
}

impl StreamExecution {
    pub(crate) fn new(
        operator: Box<dyn Operator>,
        columns: Vec<String>,
        control: QueryExecutionControl,
        resources: QueryResourceContext,
    ) -> Self {
        let column_types = vec![LogicalType::Any; columns.len()];
        let query_id = resources.query_id();
        Self {
            operator: Some(operator),
            columns,
            column_types,
            control,
            resources: Some(resources),
            output_schema: None,
            query_id,
            status: StreamStatus::Running,
            cleanup_failure: None,
            profile: None,
            result_limits: None,
        }
    }

    pub(crate) fn with_profile(mut self, profile: ProfileNode) -> Self {
        self.profile = Some(profile);
        self
    }

    pub(crate) fn with_result_limits(mut self, limits: Option<ResultLimits>) -> Self {
        self.result_limits = limits;
        self
    }

    fn accumulator(&self, mut limits: ResultLimits) -> Result<ResultAccumulator> {
        if let Some(cap) = self.result_limits {
            limits.max_rows = limits.max_rows.min(cap.max_rows);
            limits.max_bytes = limits.max_bytes.min(cap.max_bytes);
        }
        let resources = self.resources.as_ref().ok_or_else(|| {
            Error::Query(grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Semantic,
                "cannot collect an already finalized stream",
            ))
        })?;
        ResultAccumulator::new(&self.columns, &self.column_types, resources.clone(), limits)
    }

    fn cleanup_result(&self) -> Result<()> {
        match &self.cleanup_failure {
            Some((kind, message)) => Err(Error::Io(std::io::Error::new(*kind, message.clone()))),
            None => Ok(()),
        }
    }

    fn release_resources(&mut self) {
        #[cfg(feature = "spill")]
        let manager = self
            .resources
            .as_ref()
            .and_then(|resources| resources.spill_manager().cloned());
        drop(self.operator.take());
        drop(self.output_schema.take());
        drop(self.resources.take());
        #[cfg(feature = "spill")]
        if let Some(manager) = manager
            && let Err(error) = manager.finish_query()
        {
            self.cleanup_failure = Some((error.kind(), error.to_string()));
        }
    }

    fn cleanup(&mut self) {
        // Retain only the profiled account for a final observation, including
        // foreign cleanup unwinds. This does not retry any operation.
        let profile_resources = self
            .resources
            .as_ref()
            .filter(|_| self.profile.is_some())
            .cloned();
        if let Err(panic) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.release_resources()))
        {
            self.cleanup_failure = Some((
                std::io::ErrorKind::Other,
                "foreign stream cleanup panicked".into(),
            ));
            std::mem::forget(panic);
            // A panicking operator destructor was already consumed. Release
            // the remaining context; never retry a manager's failed finish.
            if self.resources.is_some()
                && let Err(panic) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.release_resources();
                }))
            {
                std::mem::forget(panic);
            }
        }
        if let (Some(profile), Some(resources)) = (&self.profile, &profile_resources) {
            profile.record_query_resources(resources.profile_stats());
        }
    }

    fn protect<T>(&mut self, action: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| action(self))) {
            Ok(outcome) => outcome,
            Err(primary) => {
                self.status = StreamStatus::Failed;
                self.cleanup();
                std::panic::resume_unwind(primary);
            }
        }
    }

    fn finish(&mut self, status: StreamStatus, primary: Option<Error>) -> Result<()> {
        // Mark terminal before destruction; neither an error nor a foreign Drop
        // panic may make the operator eligible for another pull.
        self.status = status;
        self.cleanup();
        if let Some(primary) = primary {
            self.status = StreamStatus::Failed;
            return Err(match &self.cleanup_failure {
                Some((_, message)) => {
                    primary.with_context(format!("stream cleanup also failed: {message}"))
                }
                None => primary,
            });
        }
        if let Err(error) = self.cleanup_result() {
            self.status = StreamStatus::Failed;
            return Err(error);
        }
        self.control.complete().map_err(|error| {
            self.status = StreamStatus::Failed;
            match error {
                QueryLifecycleError::Cancelled(reason) => super::convert_cancellation_error(reason),
                other => Error::Internal(format!("stream completion failed: {other}")),
            }
        })
    }

    fn next_chunk(&mut self) -> Result<Option<DataChunk>> {
        self.protect(Self::next_chunk_inner)
    }

    fn next_output_chunk(&mut self) -> Result<Option<StreamChunk>> {
        self.protect(|execution| {
            let Some(mut chunk) = execution.next_chunk_inner()? else {
                return Ok(None);
            };
            // Zone hints are upstream execution metadata, not public output.
            chunk.clear_zone_hints();
            // On failure owned output drops before terminal cleanup.
            let output = (|| {
                let resources = execution.resources.as_ref().ok_or_else(|| {
                    Error::Internal("running stream lost its query resources".into())
                })?;
                // Stream result caps belong to collect and binding copies. The
                // internal output buffer is bounded by this query's live grants.
                // next_chunk_inner has already refined the advertised types.
                // Preserve that signature: later physical types do not replace
                // a concrete stream type in ResultAccumulator either.
                let schema_matches = execution.output_schema.as_ref().is_some_and(|schema| {
                    schema.columns == execution.columns
                        && schema.column_types == execution.column_types
                });
                if !schema_matches {
                    let mut accumulator = ResultAccumulator::new(
                        &execution.columns,
                        &execution.column_types,
                        resources.clone(),
                        ResultLimits {
                            max_rows: usize::MAX,
                            max_bytes: usize::MAX,
                        },
                    )?;
                    accumulator.capture_types(chunk.column_count(), |index| {
                        chunk.column(index).map(|column| column.data_type())
                    })?;
                    let schema = accumulator.finish();
                    let reservation = schema.result_reservation.as_ref().ok_or_else(|| {
                        Error::Internal("stream output schema lost its reservation".into())
                    })?;
                    let bytes = super::checked_result_bytes(
                        reservation
                            .size()
                            .checked_add(size_of::<QueryResult>())
                            .and_then(|bytes| bytes.checked_add(2 * size_of::<usize>())),
                    )?;
                    reservation.try_resize(bytes)?;
                    execution.output_schema = Some(Arc::new(schema));
                }
                let schema = execution
                    .output_schema
                    .as_ref()
                    .ok_or_else(|| {
                        Error::Internal("stream output schema was not initialized".into())
                    })?
                    .clone();
                let raw_bytes = chunk
                    .output_retained_bytes()
                    .map_err(|error| super::result_full(error.to_string()))?;
                let cache_bytes = super::checked_result_bytes(
                    chunk
                        .column_count()
                        .checked_mul(size_of::<Value>())
                        .and_then(|width| width.checked_add(size_of::<Vec<Value>>()))
                        .and_then(|row| row.checked_mul(chunk.row_count())),
                )?;
                let bytes = super::checked_result_bytes(raw_bytes.checked_add(cache_bytes))?;
                let grant = resources
                    .try_allocate(bytes)
                    .map_err(|error| super::result_full(error.to_string()))?;
                let output = StreamChunk {
                    chunk,
                    rows: OnceLock::new(),
                    schema,
                    grant,
                };
                execution
                    .control
                    .check()
                    .map_err(super::convert_cancellation_error)?;
                Ok(output)
            })();
            match output {
                Ok(output) => Ok(Some(output)),
                Err(error) => execution
                    .finish(StreamStatus::Failed, Some(error))
                    .map(|()| None),
            }
        })
    }

    fn next_chunk_inner(&mut self) -> Result<Option<DataChunk>> {
        if self.status != StreamStatus::Running {
            return Ok(None);
        }
        if let Err(error) = self.control.check() {
            self.finish(
                StreamStatus::Failed,
                Some(super::convert_cancellation_error(error)),
            )?;
            return Ok(None);
        }
        let outcome = match self.operator.as_mut() {
            Some(operator) => operator.next(),
            None => {
                return self
                    .finish(
                        StreamStatus::Failed,
                        Some(Error::Internal("running stream has no operator".into())),
                    )
                    .map(|()| None);
            }
        };
        // The operator's error wins over cancellation observed after its call.
        match outcome {
            Err(error) => self
                .finish(
                    StreamStatus::Failed,
                    Some(super::convert_operator_error(error)),
                )
                .map(|()| None),
            Ok(chunk) => {
                if let Err(error) = self.control.check() {
                    drop(chunk);
                    self.finish(
                        StreamStatus::Failed,
                        Some(super::convert_cancellation_error(error)),
                    )?;
                    return Ok(None);
                }
                match chunk {
                    Some(chunk) => {
                        refine_column_types(&chunk, &mut self.column_types);
                        Ok(Some(chunk))
                    }
                    None => self.finish(StreamStatus::Completed, None).map(|()| None),
                }
            }
        }
    }

    fn close(&mut self) -> Result<()> {
        self.protect(Self::close_inner)
    }

    fn close_inner(&mut self) -> Result<()> {
        if self.status != StreamStatus::Running {
            return self.cleanup_result();
        }
        let primary = self
            .control
            .check()
            .err()
            .map(super::convert_cancellation_error);
        self.finish(StreamStatus::Closed, primary)
    }
}

impl Drop for StreamExecution {
    fn drop(&mut self) {
        // Foreign operator destructors can panic. Drop is best effort; explicit
        // close remains fallible and spill managers retain their cleanup debt.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.close())) {
            Ok(Ok(())) => {}
            Ok(Err(_)) => {}
            Err(panic) => std::mem::forget(panic),
        }
    }
}

/// Lazy, chunk-based result stream bound to a session's lifetime.
///
/// Created by [`Session::execute_streaming`](crate::Session::execute_streaming).
/// Iterate via [`next_chunk`](Self::next_chunk) for chunk granularity or
/// [`into_row_iter`](Self::into_row_iter) for a row iterator.
///
/// # Stability: Experimental
pub struct ResultStream<'session> {
    execution: StreamExecution,
    guard: Option<StreamGuard<'session>>,
    /// Pins one committed publication cut until terminalization.
    publication: Option<parking_lot::RwLockReadGuard<'session, ()>>,
}

impl<'s> ResultStream<'s> {
    pub(crate) fn new(
        execution: StreamExecution,
        guard: StreamGuard<'s>,
        publication: parking_lot::RwLockReadGuard<'s, ()>,
    ) -> Self {
        Self {
            execution,
            guard: Some(guard),
            publication: Some(publication),
        }
    }

    /// Column names in the order they appear in each row.
    #[must_use]
    pub fn columns(&self) -> &[String] {
        &self.execution.columns
    }

    /// Column types. Initially `Any`; refined after the first non-empty chunk.
    #[must_use]
    pub fn column_types(&self) -> &[LogicalType] {
        &self.execution.column_types
    }

    /// Pulls the next chunk from the pipeline.
    ///
    /// Returns `Ok(None)` when the stream is exhausted. Every returned chunk
    /// retains admitted output storage until dropped, even after the stream closes.
    ///
    /// # Errors
    ///
    /// Propagates execution, cancellation, deadline, and output admission errors.
    pub fn next_chunk(&mut self) -> Result<Option<StreamChunk>> {
        self.pull(StreamExecution::next_output_chunk)
    }

    fn next_physical_chunk(&mut self) -> Result<Option<DataChunk>> {
        self.pull(StreamExecution::next_chunk)
    }

    fn pull<T>(
        &mut self,
        next: impl FnOnce(&mut StreamExecution) -> Result<Option<T>>,
    ) -> Result<Option<T>> {
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| next(&mut self.execution)));
        if self.execution.status != StreamStatus::Running {
            self.guard.take();
            self.publication.take();
        }
        match outcome {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }

    /// Converts to a row-level iterator that buffers one chunk internally.
    #[must_use]
    pub fn into_row_iter(self) -> RowIterator<'s> {
        RowIterator {
            stream: self,
            current: None,
            cursor: 0,
        }
    }

    /// Drains the stream into a fully materialized [`QueryResult`].
    ///
    /// Useful as an escape hatch when a caller requested streaming but then
    /// decides to collect everything (e.g., `stream.collect(Default::default())?` in tests).
    ///
    /// # Errors
    ///
    /// Propagates execution, cancellation, and result admission errors.
    /// Returns a semantic error if the cursor was already finalized.
    pub fn collect(mut self, limits: ResultLimits) -> Result<QueryResult> {
        let result = (|| {
            let mut accumulator = self.execution.accumulator(limits)?;
            while let Some(chunk) = self.next_physical_chunk()? {
                accumulator.consume(&chunk, usize::MAX)?;
            }
            Ok(accumulator.finish())
        })();
        match result {
            Err(error) if self.execution.status == StreamStatus::Running => {
                self.execution
                    .protect(|execution| execution.finish(StreamStatus::Failed, Some(error)))?;
                Err(Error::Internal(
                    "failed stream unexpectedly finalized successfully".into(),
                ))
            }
            result => result,
        }
    }

    /// Closes execution and releases its query resources and publication guard.
    /// Repeated calls return the stored cleanup resolution without another pull.
    ///
    /// # Errors
    /// Returns cancellation or mandatory cleanup failure.
    pub fn close(&mut self) -> Result<()> {
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.execution.close()));
        self.guard.take();
        self.publication.take();
        match outcome {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }

    /// Returns the lifecycle state, including whether PROFILE data is partial.
    #[must_use]
    pub fn status(&self) -> StreamStatus {
        self.execution.status
    }

    /// Returns the actual execution profile when PROFILE was requested.
    #[must_use]
    pub fn profile(&self) -> Option<&ProfileNode> {
        self.execution.profile.as_ref()
    }

    /// Returns the unique resource identity of this execution.
    #[must_use]
    pub fn query_id(&self) -> QueryExecutionId {
        self.execution.query_id
    }
}

/// Row-level iterator adapter over a [`ResultStream`].
///
/// The buffered chunk retains its reservation. Each returned `Vec<Value>` is
/// independently owned outside that reservation; retaining rows does not retain
/// or extend the chunk's accounting authority.
///
/// # Stability: Experimental
pub struct RowIterator<'s> {
    stream: ResultStream<'s>,
    current: Option<StreamChunk>,
    cursor: usize,
}

impl RowIterator<'_> {
    /// Column names from the source stream.
    #[must_use]
    pub fn columns(&self) -> &[String] {
        self.stream.columns()
    }
}

impl Iterator for RowIterator<'_> {
    type Item = Result<Vec<Value>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_some() && self.stream.execution.control.check().is_err() {
            self.current = None;
            self.cursor = 0;
            return match self.stream.next_chunk() {
                Err(error) => Some(Err(error)),
                Ok(_) => None,
            };
        }
        loop {
            if let Some(chunk) = &mut self.current {
                if self.cursor < chunk.row_count() {
                    let row = chunk.take_adapter_row(self.cursor);
                    self.cursor += 1;
                    return Some(Ok(row));
                }
                self.current = None;
                self.cursor = 0;
            }
            match self.stream.next_chunk() {
                Ok(Some(chunk)) => {
                    if chunk.row_count() == 0 {
                        continue;
                    }
                    self.current = Some(chunk);
                    self.cursor = 0;
                }
                Ok(None) => return None,
                Err(err) => return Some(Err(err)),
            }
        }
    }
}

/// Binding-friendly result stream with no lifetime parameter.
///
/// Used by language bindings (Python, Node.js, WASM) where Rust lifetimes
/// cannot be expressed at the FFI boundary. The operator tree is `'static`
/// because operators hold `Arc<dyn GraphStoreSearch>` rather than borrows; the
/// stores remain alive as long as the stream does.
///
/// Callers that need to tie the stream's lifetime to something else (e.g.
/// a wrapping `Arc<RwLock<GrafeoDB>>` in a binding) should carry that
/// keepalive in their own wrapper alongside the stream.
///
/// # Stability: Experimental
pub struct OwnedResultStream {
    execution: StreamExecution,
    publication: Option<parking_lot::ArcRwLockReadGuard<parking_lot::RawRwLock, ()>>,
}

impl std::fmt::Debug for OwnedResultStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedResultStream")
            .field("columns", &self.execution.columns)
            .field("column_types", &self.execution.column_types)
            .field("status", &self.execution.status)
            .finish_non_exhaustive()
    }
}

impl OwnedResultStream {
    pub(crate) fn new(
        execution: StreamExecution,
        publication: parking_lot::ArcRwLockReadGuard<parking_lot::RawRwLock, ()>,
    ) -> Self {
        Self {
            execution,
            publication: Some(publication),
        }
    }

    /// Column names in the order they appear in each row.
    #[must_use]
    pub fn columns(&self) -> &[String] {
        &self.execution.columns
    }

    /// Column types. Initially `Any`; refined after the first non-empty chunk.
    #[must_use]
    pub fn column_types(&self) -> &[LogicalType] {
        &self.execution.column_types
    }

    /// Pulls the next chunk. See [`ResultStream::next_chunk`].
    ///
    /// # Errors
    ///
    /// Propagates operator errors and deadline timeouts.
    pub fn next_chunk(&mut self) -> Result<Option<StreamChunk>> {
        self.pull(StreamExecution::next_output_chunk)
    }

    fn next_physical_chunk(&mut self) -> Result<Option<DataChunk>> {
        self.pull(StreamExecution::next_chunk)
    }

    fn pull<T>(
        &mut self,
        next: impl FnOnce(&mut StreamExecution) -> Result<Option<T>>,
    ) -> Result<Option<T>> {
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| next(&mut self.execution)));
        if self.execution.status != StreamStatus::Running {
            self.publication.take();
        }
        match outcome {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }

    /// Converts to a row iterator that buffers one chunk internally.
    #[must_use]
    pub fn into_row_iter(self) -> OwnedRowIterator {
        OwnedRowIterator {
            stream: self,
            current: None,
            cursor: 0,
        }
    }

    /// Drains into a [`QueryResult`].
    ///
    /// # Errors
    ///
    /// Propagates execution, cancellation, and result admission errors.
    /// Returns a semantic error if the cursor was already finalized.
    pub fn collect(mut self, limits: ResultLimits) -> Result<QueryResult> {
        let result = (|| {
            let mut accumulator = self.execution.accumulator(limits)?;
            while let Some(chunk) = self.next_physical_chunk()? {
                accumulator.consume(&chunk, usize::MAX)?;
            }
            Ok(accumulator.finish())
        })();
        match result {
            Err(error) if self.execution.status == StreamStatus::Running => {
                self.execution
                    .protect(|execution| execution.finish(StreamStatus::Failed, Some(error)))?;
                Err(Error::Internal(
                    "failed stream unexpectedly finalized successfully".into(),
                ))
            }
            result => result,
        }
    }

    /// Closes execution and releases its query resources and publication guard.
    /// Repeated calls return the stored cleanup resolution without another pull.
    ///
    /// # Errors
    /// Returns cancellation or mandatory cleanup failure.
    pub fn close(&mut self) -> Result<()> {
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.execution.close()));

        self.publication.take();
        match outcome {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }

    /// Returns the lifecycle state, including whether PROFILE data is partial.
    #[must_use]
    pub fn status(&self) -> StreamStatus {
        self.execution.status
    }

    /// Returns the actual execution profile when PROFILE was requested.
    #[must_use]
    pub fn profile(&self) -> Option<&ProfileNode> {
        self.execution.profile.as_ref()
    }

    /// Returns the unique resource identity of this execution.
    #[must_use]
    pub fn query_id(&self) -> QueryExecutionId {
        self.execution.query_id
    }
}

/// Row-level iterator over an [`OwnedResultStream`].
///
/// The buffered chunk retains its reservation. Returned `Vec<Value>` rows are
/// independently owned outside that reservation. Binding callers
/// remain responsible for their own copied-output limits.
///
/// # Stability: Experimental
pub struct OwnedRowIterator {
    stream: OwnedResultStream,
    current: Option<StreamChunk>,
    cursor: usize,
}

impl OwnedRowIterator {
    /// Column names from the source stream.
    #[must_use]
    pub fn columns(&self) -> &[String] {
        self.stream.columns()
    }

    /// Checks the cursor's composed cancellation/deadline before a binding
    /// delivers a row it has already pulled into a bounded output buffer.
    ///
    /// # Errors
    /// Returns cancellation/deadline and performs the normal terminal cleanup.
    pub fn check_execution(&mut self) -> Result<()> {
        if self.stream.execution.control.check().is_err() {
            self.current = None;
            self.cursor = 0;
            self.stream.next_chunk()?;
        }
        Ok(())
    }

    /// Discards buffered rows and closes the underlying execution.
    ///
    /// # Errors
    /// Returns the stream's retained cancellation or cleanup resolution.
    /// Repeated calls never pull rows or retry failed cleanup.
    pub fn close(&mut self) -> Result<()> {
        self.current = None;
        self.cursor = 0;
        self.stream.close()
    }
}

impl Iterator for OwnedRowIterator {
    type Item = Result<Vec<Value>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_some() && self.stream.execution.control.check().is_err() {
            self.current = None;
            self.cursor = 0;
            return match self.stream.next_chunk() {
                Err(error) => Some(Err(error)),
                Ok(_) => None,
            };
        }
        loop {
            if let Some(chunk) = &mut self.current {
                if self.cursor < chunk.row_count() {
                    let row = chunk.take_adapter_row(self.cursor);
                    self.cursor += 1;
                    return Some(Ok(row));
                }
                self.current = None;
                self.cursor = 0;
            }
            match self.stream.next_chunk() {
                Ok(Some(chunk)) => {
                    if chunk.row_count() == 0 {
                        continue;
                    }
                    self.current = Some(chunk);
                    self.cursor = 0;
                }
                Ok(None) => return None,
                Err(err) => return Some(Err(err)),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn refine_column_types(chunk: &DataChunk, types: &mut Vec<LogicalType>) {
    let col_count = chunk.column_count();
    if col_count == 0 {
        return;
    }
    if types.len() != col_count {
        types.resize(col_count, LogicalType::Any);
    }
    for (col_idx, slot) in types.iter_mut().enumerate().take(col_count) {
        if matches!(slot, LogicalType::Any)
            && let Some(col) = chunk.column(col_idx)
        {
            *slot = col.data_type().clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::memory::buffer::BufferManager;
    use grafeo_core::execution::QueryCancellationHandle;
    use grafeo_core::execution::operators::{OperatorError, OperatorResult};
    use std::sync::Arc;

    enum ProbeOutput {
        Rows,
        Failure,
        Eof,
    }

    struct Probe {
        pulls: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
        cancel: Option<QueryCancellationHandle>,
        output: ProbeOutput,
        panic_pull: bool,
        panic_drop: bool,
        float_rows: bool,
    }

    impl Operator for Probe {
        fn next(&mut self) -> OperatorResult {
            self.pulls.fetch_add(1, Ordering::SeqCst);
            assert!(!self.panic_pull, "foreign pull panic");
            if let Some(cancel) = &self.cancel {
                cancel.cancel();
            }
            if matches!(self.output, ProbeOutput::Failure) {
                Err(OperatorError::Execution("primary operator failure".into()))
            } else if matches!(self.output, ProbeOutput::Eof) {
                Ok(None)
            } else {
                let mut chunk = if self.float_rows {
                    let mut chunk = DataChunk::with_capacity(&[LogicalType::Float64], 2);
                    chunk.column_mut(0).unwrap().push_float64(7.5);
                    chunk.column_mut(0).unwrap().push_float64(8.5);
                    chunk
                } else {
                    let mut chunk = DataChunk::with_capacity(&[LogicalType::Int64], 2);
                    chunk.column_mut(0).unwrap().push_int64(7);
                    chunk.column_mut(0).unwrap().push_int64(8);
                    chunk
                };
                chunk.set_count(2);
                Ok(Some(chunk))
            }
        }
        fn reset(&mut self) {
            panic!("a terminal stream must never reset");
        }
        fn name(&self) -> &'static str {
            "StreamProbe"
        }
        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    impl Drop for Probe {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
            assert!(!self.panic_drop, "foreign cleanup panic");
        }
    }

    fn execution(
        fails: bool,
        eof: bool,
        cancel_in_pull: bool,
    ) -> (
        StreamExecution,
        QueryCancellationHandle,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        let control = QueryExecutionControl::new();
        let handle = control.cancellation_handle();
        let resources = QueryResourceContext::new_with_cancellation(
            BufferManager::with_budget(1_000_000),
            control.token(),
        )
        .unwrap();
        let pulls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let operator = Box::new(Probe {
            pulls: Arc::clone(&pulls),
            drops: Arc::clone(&drops),
            cancel: cancel_in_pull.then(|| handle.clone()),
            output: if fails {
                ProbeOutput::Failure
            } else if eof {
                ProbeOutput::Eof
            } else {
                ProbeOutput::Rows
            },
            panic_pull: false,
            panic_drop: false,
            float_rows: false,
        });
        (
            StreamExecution::new(operator, vec!["value".into()], control, resources),
            handle,
            pulls,
            drops,
        )
    }

    #[test]
    fn stream_output_schema_reuses_admission_and_outlives_cleanup() {
        let (mut execution, _, _, _) = execution(false, false, false);
        let resources = execution.resources.as_ref().unwrap().clone();
        let first = execution.next_output_chunk().unwrap().unwrap();
        let second = execution.next_output_chunk().unwrap().unwrap();
        assert!(Arc::ptr_eq(&first.schema, &second.schema));
        assert_eq!(first.column_types(), &[LogicalType::Int64]);
        execution.operator = Some(Box::new(Probe {
            pulls: Arc::new(AtomicUsize::new(0)),
            drops: Arc::new(AtomicUsize::new(0)),
            cancel: None,
            output: ProbeOutput::Rows,
            panic_pull: false,
            panic_drop: false,
            float_rows: true,
        }));
        let changed_physical = execution.next_output_chunk().unwrap().unwrap();
        assert!(Arc::ptr_eq(&first.schema, &changed_physical.schema));
        assert_eq!(changed_physical.column_types(), &[LogicalType::Int64]);
        assert_eq!(
            changed_physical.rows().unwrap()[0],
            vec![Value::Float64(7.5)]
        );
        execution.columns[0] = "renamed".into();
        let renamed = execution.next_output_chunk().unwrap().unwrap();
        assert!(!Arc::ptr_eq(&first.schema, &renamed.schema));
        assert_eq!(first.columns(), &["value"]);
        assert_eq!(renamed.columns(), &["renamed"]);
        execution.finish(StreamStatus::Closed, None).unwrap();
        assert!(execution.output_schema.is_none());
        assert!(resources.query_stats().allocated_bytes > 0);
        assert_eq!(first.rows().unwrap()[0], vec![Value::Int64(7)]);
        drop((first, second, changed_physical, renamed));
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[test]
    fn stream_operator_error_is_delivered_once_and_releases_borrowed_guards() {
        let (execution, _, pulls, drops) = execution(true, false, false);
        let active = AtomicUsize::new(0);
        let publication = parking_lot::RwLock::new(());
        let mut stream =
            ResultStream::new(execution, StreamGuard::new(&active), publication.read());
        assert_eq!(active.load(Ordering::SeqCst), 1);
        assert!(
            stream
                .next_chunk()
                .unwrap_err()
                .to_string()
                .contains("primary operator failure")
        );
        assert_eq!(stream.status(), StreamStatus::Failed);
        assert_eq!(active.load(Ordering::SeqCst), 0);
        assert!(publication.try_write().is_some());
        assert!(stream.next_chunk().unwrap().is_none());
        stream.close().unwrap();
        stream.close().unwrap();
        drop(stream);
        assert_eq!(pulls.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stream_pre_pull_cancel_never_calls_operator() {
        let (mut execution, handle, pulls, drops) = execution(true, false, false);
        handle.cancel();
        let error = execution.next_chunk().unwrap_err();
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::QueryCancelled
        );
        assert!(execution.next_chunk().unwrap().is_none());
        assert_eq!(pulls.load(Ordering::SeqCst), 0);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stream_operator_error_precedes_cancel_during_pull() {
        let (mut execution, _, pulls, drops) = execution(true, false, true);
        assert!(
            matches!(execution.next_chunk(), Err(Error::Internal(message)) if message == "primary operator failure")
        );
        assert!(execution.next_chunk().unwrap().is_none());
        assert_eq!(pulls.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stream_post_pull_cancel_does_not_publish_chunk() {
        let (mut execution, _, pulls, drops) = execution(false, false, true);
        assert_eq!(
            execution.next_chunk().unwrap_err().error_code(),
            grafeo_common::utils::error::ErrorCode::QueryCancelled
        );
        assert!(execution.next_chunk().unwrap().is_none());
        execution.close().unwrap();
        assert_eq!(pulls.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stream_eof_completes_once_and_owned_close_releases_publication() {
        for eof in [false, true] {
            let (execution, handle, pulls, drops) = execution(false, eof, false);
            let publication = Arc::new(parking_lot::RwLock::new(()));
            let mut stream = OwnedResultStream::new(execution, publication.read_arc());
            assert_eq!(stream.next_chunk().unwrap().is_none(), eof);
            stream.close().unwrap();
            stream.close().unwrap();
            assert_eq!(
                stream.status(),
                if eof {
                    StreamStatus::Completed
                } else {
                    StreamStatus::Closed
                }
            );
            assert!(!handle.try_cancel());
            assert!(publication.try_write().is_some());
            assert!(stream.next_chunk().unwrap().is_none());
            assert_eq!(pulls.load(Ordering::SeqCst), 1);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn stream_row_adapter_cancellation_discards_buffered_rows() {
        let (execution, handle, pulls, drops) = execution(false, false, false);
        let publication = Arc::new(parking_lot::RwLock::new(()));
        let mut rows = OwnedResultStream::new(execution, publication.read_arc()).into_row_iter();
        assert_eq!(rows.next().unwrap().unwrap(), vec![Value::Int64(7)]);
        handle.cancel();
        assert_eq!(
            rows.next().unwrap().unwrap_err().error_code(),
            grafeo_common::utils::error::ErrorCode::QueryCancelled
        );
        assert!(rows.next().is_none());
        assert_eq!(pulls.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(publication.try_write().is_some());
    }

    #[test]
    fn stream_unwind_terminalizes_retained_cursor_and_releases_guards() {
        for panic_pull in [false, true] {
            let (mut execution, _, pulls, drops) = execution(false, false, false);
            // Replace the ordinary probe without counting its construction cleanup.
            drop(execution.operator.take());
            drops.store(0, Ordering::SeqCst);
            execution.operator = Some(Box::new(Probe {
                pulls: Arc::clone(&pulls),
                drops: Arc::clone(&drops),
                cancel: None,
                output: ProbeOutput::Rows,
                panic_pull,
                panic_drop: !panic_pull,
                float_rows: false,
            }));
            let grant = execution
                .resources
                .as_ref()
                .unwrap()
                .try_allocate(128)
                .unwrap();
            drop(grant);
            execution.profile = Some(ProfileNode {
                name: "Probe".into(),
                label: String::new(),
                stats: Arc::new(parking_lot::Mutex::new(Default::default())),
                children: vec![],
            });
            let active = AtomicUsize::new(0);
            let publication = parking_lot::RwLock::new(());
            let mut stream =
                ResultStream::new(execution, StreamGuard::new(&active), publication.read());
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if panic_pull {
                    let _ = stream.next_chunk();
                } else {
                    assert!(stream.close().is_err());
                }
            }));
            assert_eq!(panic.is_err(), panic_pull);
            assert_eq!(stream.status(), StreamStatus::Failed);
            assert!(stream.next_chunk().unwrap().is_none());
            assert_eq!(active.load(Ordering::SeqCst), 0);
            assert!(publication.try_write().is_some());
            assert_eq!(pulls.load(Ordering::SeqCst), usize::from(panic_pull));
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            assert!(stream.execution.resources.is_none());
            let stats = stream
                .profile()
                .unwrap()
                .stats
                .lock()
                .query_resources
                .unwrap();
            assert_eq!(stats.resident_granted_bytes, 0);
            assert!(stats.resident_peak_bytes >= 128);
        }
    }

    #[test]
    fn stream_drop_releases_without_pulling() {
        let (execution, _, pulls, drops) = execution(false, false, false);
        let active = AtomicUsize::new(0);
        let publication = parking_lot::RwLock::new(());
        drop(ResultStream::new(
            execution,
            StreamGuard::new(&active),
            publication.read(),
        ));
        assert_eq!(active.load(Ordering::SeqCst), 0);
        assert_eq!(pulls.load(Ordering::SeqCst), 0);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(publication.try_write().is_some());
    }

    #[cfg(all(feature = "spill", any(target_os = "linux", target_os = "macos")))]
    #[test]
    fn stream_physical_profile_retains_cleanup_failure_without_retry() {
        let directory = tempfile::tempdir().unwrap();
        let (mut execution, _, pulls, drops) = execution(true, false, false);
        let root_owner = crate::spill_crypto::DatabaseSpillRoot::new(&crate::Config::in_memory());
        let root = root_owner
            .open(
                directory.path(),
                grafeo_common::types::StoreId::from_bytes([3; 32]).unwrap(),
                &execution.control.token(),
            )
            .unwrap();
        let resources = QueryResourceContext::with_spill_root(
            BufferManager::with_budget(1_000_000),
            &root,
            execution.control.token(),
        )
        .unwrap();
        let manager = Arc::clone(resources.ensure_spill_manager().unwrap().unwrap());
        let unknown = manager.spill_dir().join("unknown");
        std::fs::write(&unknown, b"keep").unwrap();
        execution.resources = Some(resources);
        execution.profile = Some(ProfileNode {
            name: "Probe".into(),
            label: String::new(),
            stats: Arc::new(parking_lot::Mutex::new(Default::default())),
            children: vec![],
        });
        let error = execution.next_chunk().unwrap_err();
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::Internal
        );
        assert!(error.to_string().contains("primary operator failure"));
        assert!(error.to_string().contains("stream cleanup also failed"));
        let stats = execution
            .profile
            .as_ref()
            .unwrap()
            .stats
            .lock()
            .query_resources
            .unwrap();
        let physical = stats.spill_physical.unwrap();
        assert!(physical.cleanup_failed);
        assert!(physical.cleanup_debt_bytes > 0);
        assert_eq!(physical.cleanup_debt_bytes, physical.reserved_bytes);
        assert_eq!(stats.resident_granted_bytes, 0);
        let cleanup = execution.close().unwrap_err().to_string();
        std::fs::remove_file(unknown).unwrap();
        assert_eq!(execution.close().unwrap_err().to_string(), cleanup);
        assert_eq!(
            execution
                .profile
                .as_ref()
                .unwrap()
                .stats
                .lock()
                .query_resources,
            Some(stats)
        );
        assert!(
            manager.spill_dir().exists(),
            "terminal close must not retry cleanup"
        );
        assert!(execution.next_chunk().unwrap().is_none());
        assert_eq!(pulls.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        manager.finish_query().unwrap();
    }

    #[cfg(feature = "spill")]
    #[test]
    fn stream_primary_retains_cleanup_failure_and_close_does_not_retry() {
        use grafeo_core::execution::spill::{
            CleartextSpillRecordProvider, NoopSpillIo, SpillDiskQuota, SpillFrameLimits,
        };
        let directory = tempfile::tempdir().unwrap();
        let (mut execution, _, pulls, drops) = execution(true, false, false);
        let (resources, manager) = crate::spill_crypto::admitted_spill_test_resources(
            directory.path(),
            BufferManager::with_budget(1_000_000),
            execution.control.token(),
            Arc::new(CleartextSpillRecordProvider),
            SpillFrameLimits::format_max(),
            Arc::new(NoopSpillIo),
            SpillDiskQuota::new(u64::MAX),
        );
        let unknown = manager.spill_dir().join("unknown");
        std::fs::write(&unknown, b"keep").unwrap();
        execution.resources = Some(resources);
        let error = execution.next_chunk().unwrap_err();
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::Internal
        );
        assert!(error.to_string().contains("primary operator failure"));
        assert!(error.to_string().contains("stream cleanup also failed"));
        let cleanup = execution.close().unwrap_err().to_string();
        std::fs::remove_file(unknown).unwrap();
        assert_eq!(execution.close().unwrap_err().to_string(), cleanup);
        assert!(
            manager.spill_dir().exists(),
            "terminal close must not retry cleanup"
        );
        assert!(execution.next_chunk().unwrap().is_none());
        assert_eq!(pulls.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        manager.finish_query().unwrap();
    }

    #[test]
    fn stream_collection_limits_intersect_options_and_release_all_ownership() {
        let (execution, _, pulls, drops) = execution(false, false, false);
        let resources = execution.resources.as_ref().unwrap().clone();
        let execution = execution.with_result_limits(Some(ResultLimits {
            max_rows: 1,
            max_bytes: 100_000,
        }));
        let active = AtomicUsize::new(0);
        let publication = parking_lot::RwLock::new(());
        let stream = ResultStream::new(execution, StreamGuard::new(&active), publication.read());
        let error = stream
            .collect(ResultLimits {
                max_rows: 10,
                max_bytes: 100_000,
            })
            .unwrap_err();
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert!(error.to_string().contains("row limit"));
        assert_eq!(pulls.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(active.load(Ordering::SeqCst), 0);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert!(publication.try_write().is_some());
    }

    #[test]
    fn stream_empty_collection_with_zero_limits_is_accounted_but_closed_collection_rejects() {
        let (first_execution, _, _, _) = execution(false, true, false);
        let resources = first_execution.resources.as_ref().unwrap().clone();
        let publication = Arc::new(parking_lot::RwLock::new(()));
        let stream = OwnedResultStream::new(first_execution, publication.read_arc());
        let result = stream
            .collect(ResultLimits {
                max_rows: 0,
                max_bytes: 0,
            })
            .unwrap();
        assert_eq!(result.row_count(), 0);
        assert!(resources.query_stats().allocated_bytes > 0);
        drop(result);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        let (execution, _, pulls, _) = execution(false, false, false);
        let mut stream = OwnedResultStream::new(execution, publication.read_arc());
        stream.close().unwrap();
        let error = stream.collect(ResultLimits::default()).unwrap_err();
        assert!(error.to_string().contains("already finalized"));
        assert_eq!(pulls.load(Ordering::SeqCst), 0);
    }
}
