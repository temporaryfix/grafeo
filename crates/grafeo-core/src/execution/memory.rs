//! Execution memory context for memory-aware query execution.

use grafeo_common::memory::buffer::{
    BufferManager, BufferStats, ConsumerRegistration, ConsumerRegistrationError, MemoryConsumer,
    MemoryGrant, MemoryGrantError, MemoryRegion, PressureLevel, QueryMemoryPool,
};
use parking_lot::Mutex;
use std::num::NonZeroU64;
use std::sync::Arc;
use thiserror::Error;

use super::cancellation::{QueryCancellationError, QueryCancellationToken, QueryExecutionControl};

#[cfg(test)]
std::thread_local! {
    // Actual context admission attempts, including refusals. Thread isolation
    // keeps independently running resource tests from contaminating the count.
    static QUERY_ADMISSION_ATTEMPTS: std::cell::Cell<(usize, usize)> = const {
        std::cell::Cell::new((0, 0))
    };
}

/// Takes this test thread's zero-byte and positive-byte admission attempts.
#[cfg(test)]
pub(crate) fn take_query_admission_attempts() -> (usize, usize) {
    QUERY_ADMISSION_ATTEMPTS.with(|attempts| attempts.replace((0, 0)))
}

/// Default chunk size for execution buffers.
pub const DEFAULT_CHUNK_SIZE: usize = 2048;

/// Chunk size under moderate memory pressure.
pub const MODERATE_PRESSURE_CHUNK_SIZE: usize = 1024;

/// Chunk size under high memory pressure.
pub const HIGH_PRESSURE_CHUNK_SIZE: usize = 512;

/// Chunk size under critical memory pressure.
pub const CRITICAL_PRESSURE_CHUNK_SIZE: usize = 256;

/// Execution context with memory awareness.
///
/// This context provides memory allocation for query execution operators
/// and adjusts chunk sizes based on memory pressure.
pub struct ExecutionMemoryContext {
    /// Reference to the buffer manager.
    manager: Arc<BufferManager>,
    /// Total bytes allocated for this execution context.
    allocated: usize,
    /// Grants held by this context.
    grants: Vec<MemoryGrant>,
}

impl ExecutionMemoryContext {
    /// Creates a new execution memory context.
    #[must_use]
    pub fn new(manager: Arc<BufferManager>) -> Self {
        Self {
            manager,
            allocated: 0,
            grants: Vec::new(),
        }
    }

    /// Requests memory for execution buffers.
    ///
    /// Returns `None` if the allocation cannot be satisfied.
    pub fn allocate(&mut self, size: usize) -> Option<MemoryGrant> {
        let grant = self
            .manager
            .try_allocate(size, MemoryRegion::ExecutionBuffers)?;
        self.allocated += size;
        Some(grant)
    }

    /// Allocates and stores a grant internally.
    ///
    /// The grant will be released when this context is dropped.
    pub fn allocate_tracked(&mut self, size: usize) -> bool {
        if let Some(grant) = self
            .manager
            .try_allocate(size, MemoryRegion::ExecutionBuffers)
        {
            self.allocated += size;
            self.grants.push(grant);
            true
        } else {
            false
        }
    }

    /// Returns the current pressure level.
    #[must_use]
    pub fn pressure_level(&self) -> PressureLevel {
        self.manager.pressure_level()
    }

    /// Returns whether chunk size should be reduced due to memory pressure.
    #[must_use]
    pub fn should_reduce_chunk_size(&self) -> bool {
        matches!(
            self.pressure_level(),
            PressureLevel::High | PressureLevel::Critical
        )
    }

    /// Computes adjusted chunk size based on memory pressure.
    #[must_use]
    pub fn adjusted_chunk_size(&self, requested: usize) -> usize {
        match self.pressure_level() {
            PressureLevel::Normal => requested,
            PressureLevel::Moderate => requested.min(MODERATE_PRESSURE_CHUNK_SIZE),
            PressureLevel::High => requested.min(HIGH_PRESSURE_CHUNK_SIZE),
            PressureLevel::Critical => requested.min(CRITICAL_PRESSURE_CHUNK_SIZE),
            _ => requested.min(CRITICAL_PRESSURE_CHUNK_SIZE),
        }
    }

    /// Returns the optimal chunk size for the current memory state.
    #[must_use]
    pub fn optimal_chunk_size(&self) -> usize {
        self.adjusted_chunk_size(DEFAULT_CHUNK_SIZE)
    }

    /// Returns total bytes allocated through this context.
    #[must_use]
    pub fn total_allocated(&self) -> usize {
        self.allocated
    }

    /// Returns the buffer manager.
    #[must_use]
    pub fn manager(&self) -> &Arc<BufferManager> {
        &self.manager
    }

    /// Releases all tracked grants.
    pub fn release_all(&mut self) {
        self.grants.clear();
        self.allocated = 0;
    }
}

impl Drop for ExecutionMemoryContext {
    fn drop(&mut self) {
        // Grants are automatically released when dropped
        self.grants.clear();
    }
}

/// Builder for execution memory contexts with pre-allocation.
pub struct ExecutionMemoryContextBuilder {
    manager: Arc<BufferManager>,
    initial_allocation: usize,
}

impl ExecutionMemoryContextBuilder {
    /// Creates a new builder with the given buffer manager.
    #[must_use]
    pub fn new(manager: Arc<BufferManager>) -> Self {
        Self {
            manager,
            initial_allocation: 0,
        }
    }

    /// Sets the initial allocation size.
    #[must_use]
    pub fn with_initial_allocation(mut self, size: usize) -> Self {
        self.initial_allocation = size;
        self
    }

    /// Builds the execution memory context.
    ///
    /// Returns `None` if the initial allocation cannot be satisfied.
    pub fn build(self) -> Option<ExecutionMemoryContext> {
        let mut ctx = ExecutionMemoryContext::new(self.manager);

        if self.initial_allocation > 0 && !ctx.allocate_tracked(self.initial_allocation) {
            return None;
        }

        Some(ctx)
    }
}

/// Process-wide source for execution identities.
///
/// A portable mutex is deliberate: the checked `u64` protocol must behave the
/// same on 32-bit and WebAssembly targets, and exhaustion must never wrap.
static NEXT_QUERY_EXECUTION_ID: Mutex<u64> = Mutex::new(0);

/// Process-local identity for one query execution.
///
/// The value is monotonic and nonzero within this process. It is diagnostic
/// execution identity, not a durable identifier, transaction identifier, or
/// cryptographic nonce. Values may have gaps when later context setup fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QueryExecutionId(NonZeroU64);

impl QueryExecutionId {
    /// Returns the nonzero numeric identity.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

impl std::fmt::Display for QueryExecutionId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

fn next_query_execution_id(
    source: &Mutex<u64>,
) -> Result<QueryExecutionId, QueryResourceContextError> {
    let mut current = source.lock();
    let next = current
        .checked_add(1)
        .ok_or(QueryResourceContextError::QueryExecutionIdExhausted)?;
    let id = NonZeroU64::new(next).ok_or(QueryResourceContextError::QueryExecutionIdExhausted)?;
    *current = next;
    Ok(QueryExecutionId(id))
}

/// Failure to create or use a query resource context.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueryResourceContextError {
    /// The process-local checked query identity space is exhausted.
    #[error("process-local query execution identity space exhausted")]
    QueryExecutionIdExhausted,
    /// Resident-memory admission or accounting failed.
    #[error(transparent)]
    Memory(#[from] MemoryGrantError),
    /// A unique scoped consumer identity could not be created.
    #[error(transparent)]
    ConsumerRegistration(#[from] ConsumerRegistrationError),
    /// An accounted partition could not admit its fixed constructor storage.
    #[cfg(feature = "spill")]
    #[error(transparent)]
    PartitionAdmission(#[from] crate::execution::spill::PartitionAdmissionError),
    /// A spill-capable operator was requested without a configured manager.
    #[cfg(feature = "spill")]
    #[error("query execution has no configured spill manager")]
    SpillManagerUnavailable,
    /// Authenticated root or query-leaf admission failed before publication.
    #[cfg(feature = "spill")]
    #[error("spill query admission failed: {message}")]
    SpillAdmission {
        /// Original I/O error classification.
        kind: std::io::ErrorKind,
        /// Diagnostic context retained from the root/lease owner.
        message: String,
        /// Typed cancellation retained through primary/cleanup I/O wrappers.
        cancellation: Option<QueryCancellationError>,
    },
}

#[cfg(feature = "spill")]
fn spill_admission_error(error: std::io::Error) -> QueryResourceContextError {
    let mut cancellation = None;
    let mut source: Option<&(dyn std::error::Error + 'static)> = error.get_ref().map(|e| e as _);
    // Bounded traversal also contains a hostile provider's cyclic source chain.
    for _ in 0..8 {
        let Some(current) = source else { break };
        if let Some(reason) = current.downcast_ref::<QueryCancellationError>() {
            cancellation = Some(*reason);
            break;
        }
        source = if let Some(io) = current.downcast_ref::<std::io::Error>() {
            io.get_ref().map(|e| e as _)
        } else {
            current.source()
        };
    }
    QueryResourceContextError::SpillAdmission {
        kind: error.kind(),
        message: error.to_string(),
        cancellation,
    }
}

/// Point-in-time statistics for one query's resident-memory account.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryResourceStats {
    /// Identity of the query whose account was sampled.
    pub query_id: QueryExecutionId,
    /// Resident bytes currently granted to this query.
    pub allocated_bytes: usize,
    /// Current dynamic fair-share growth limit.
    ///
    /// This is not reserved capacity; global pressure can deny smaller growth.
    pub growth_limit_bytes: usize,
}

// Clones share one admission owner. A configured resident query needs no
// filesystem leaf; the first spill-capable operator acquires it fallibly.
#[cfg(feature = "spill")]
struct QuerySpillState {
    root: Option<Arc<super::spill::SpillRoot>>,
    manager: std::sync::OnceLock<Arc<super::spill::SpillManager>>,
    admission: Mutex<()>,
    profile_merge: std::sync::atomic::AtomicBool,
}

/// Always-available resources shared by one query execution.
///
/// Clones retain the same query identity and [`QueryMemoryPool`]. The optional
/// spill manager exists only when the `spill` feature and database
/// configuration provide one. Merely carrying this context does not account
/// an operator's containers: operators must retain real grants from
/// [`Self::try_allocate`] as they grow.
///
/// Arbitrary manager injection and the former context alias are unavailable:
///
/// ```compile_fail,E0599
/// use grafeo_core::execution::QueryResourceContext;
/// let _ = QueryResourceContext::with_spill_manager;
/// ```
/// ```compile_fail,E0599
/// use grafeo_core::execution::QueryResourceContext;
/// let _ = QueryResourceContext::with_spill_manager_and_cancellation;
/// ```
/// ```compile_fail,E0432
/// use grafeo_core::execution::OperatorMemoryContext;
/// ```
#[derive(Clone)]
pub struct QueryResourceContext {
    query_id: QueryExecutionId,
    buffer_manager: Arc<BufferManager>,
    query_pool: Arc<QueryMemoryPool>,
    cancellation: QueryCancellationToken,
    #[cfg(feature = "spill")]
    spill_manager: Option<Arc<QuerySpillState>>,
}

impl QueryResourceContext {
    /// Creates a resident-only context with a fresh identity and query pool.
    ///
    /// # Errors
    ///
    /// Returns a structured error if the checked identity source or live-pool
    /// count is exhausted.
    pub fn new(buffer_manager: Arc<BufferManager>) -> Result<Self, QueryResourceContextError> {
        Self::new_with_cancellation(buffer_manager, QueryExecutionControl::new().token())
    }

    /// Creates a resident-only context with immutable cancellation control.
    ///
    /// Supplying the token during construction prevents clones of one query
    /// identity and memory pool from diverging onto unrelated cancellation
    /// states.
    ///
    /// # Errors
    ///
    /// Returns a structured error if the checked identity source or live-pool
    /// count is exhausted.
    pub fn new_with_cancellation(
        buffer_manager: Arc<BufferManager>,
        cancellation: QueryCancellationToken,
    ) -> Result<Self, QueryResourceContextError> {
        let query_id = next_query_execution_id(&NEXT_QUERY_EXECUTION_ID)?;
        let query_pool = buffer_manager.new_query_pool()?;
        Ok(Self {
            query_id,
            buffer_manager,
            query_pool,
            cancellation,
            #[cfg(feature = "spill")]
            spill_manager: None,
        })
    }

    /// Creates one query account retaining its configured spill authority.
    ///
    /// The first spill-capable operator admits a leaf with this account's exact
    /// execution identity and cancellation token. Clones share that admission;
    /// resident queries do not create or reserve filesystem objects.
    ///
    /// # Errors
    /// Returns an error if root validation, cancellation, or account creation fails.
    #[cfg(feature = "spill")]
    pub fn with_spill_root(
        buffer_manager: Arc<BufferManager>,
        root: &Arc<super::spill::SpillRoot>,
        cancellation: QueryCancellationToken,
    ) -> Result<Self, QueryResourceContextError> {
        root.validate_for_query(&cancellation)
            .map_err(spill_admission_error)?;
        let mut context = Self::new_with_cancellation(buffer_manager, cancellation)?;
        context.spill_manager = Some(Arc::new(QuerySpillState {
            root: Some(Arc::clone(root)),
            manager: std::sync::OnceLock::new(),
            admission: Mutex::new(()),
            profile_merge: std::sync::atomic::AtomicBool::new(false),
        }));
        Ok(context)
    }

    /// Whether this account is configured to admit spill storage.
    #[cfg(feature = "spill")]
    #[must_use]
    pub fn has_spill_manager(&self) -> bool {
        self.spill_manager.is_some()
    }

    /// Admits and shares this query's manager when configured.
    ///
    /// # Errors
    /// Returns cancellation, bounded admission contention, quota, or filesystem
    /// authority failure. Failure never downgrades the query to resident mode.
    #[cfg(feature = "spill")]
    pub fn ensure_spill_manager(
        &self,
    ) -> Result<Option<&Arc<super::spill::SpillManager>>, QueryResourceContextError> {
        let Some(state) = &self.spill_manager else {
            return Ok(None);
        };
        if let Some(manager) = state.manager.get() {
            return Ok(Some(manager));
        }
        let _guard = state
            .admission
            .try_lock_for(std::time::Duration::from_millis(512))
            .ok_or_else(|| {
                spill_admission_error(std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    "query spill admission is busy",
                ))
            })?;
        if let Some(manager) = state.manager.get() {
            return Ok(Some(manager));
        }
        let root = state
            .root
            .as_ref()
            .ok_or(QueryResourceContextError::SpillManagerUnavailable)?;
        let lease = root
            .begin_query(self.query_id, self.cancellation.clone())
            .map_err(spill_admission_error)?;
        let manager = Arc::new(super::spill::SpillManager::from_query_lease(lease));
        // Admission is serialized, so exactly one successful manager is installed.
        let _ = state.manager.set(manager);
        if state
            .profile_merge
            .load(std::sync::atomic::Ordering::SeqCst)
            && let Some(manager) = state.manager.get()
        {
            manager.enable_profile_merge();
        }
        Ok(state.manager.get())
    }

    #[cfg(feature = "spill")]
    pub(crate) fn enable_spill_profile_merge(&self) {
        if let Some(state) = &self.spill_manager {
            state
                .profile_merge
                .store(true, std::sync::atomic::Ordering::SeqCst);
            if let Some(manager) = state.manager.get() {
                manager.enable_profile_merge();
            }
        }
    }

    /// Returns this execution's process-local identity.
    #[must_use]
    pub const fn query_id(&self) -> QueryExecutionId {
        self.query_id
    }

    /// Returns this query's shared cooperative cancellation control.
    #[must_use]
    pub const fn cancellation_token(&self) -> &QueryCancellationToken {
        &self.cancellation
    }

    /// Checks whether this query was cancelled or exceeded its deadline.
    ///
    /// # Errors
    ///
    /// Returns the typed cancellation reason without mutating resource state.
    pub fn check_cancelled(&self) -> Result<(), QueryCancellationError> {
        self.cancellation.check()
    }

    /// Attempts to reserve resident execution bytes in both the query-local
    /// and database-wide accounts.
    ///
    /// The returned RAII grant must remain alive for as long as the bytes are
    /// resident. Denial leaves both accounts unchanged.
    ///
    /// # Errors
    ///
    /// Returns a structured query/global limit, arithmetic, or accounting
    /// failure without granting unaccounted bytes.
    pub fn try_allocate(&self, bytes: usize) -> Result<MemoryGrant, QueryResourceContextError> {
        #[cfg(test)]
        QUERY_ADMISSION_ATTEMPTS.with(|attempts| {
            let (zero, positive) = attempts.get();
            attempts.set((
                zero + usize::from(bytes == 0),
                positive + usize::from(bytes != 0),
            ));
        });
        self.query_pool
            .try_allocate(bytes, MemoryRegion::ExecutionBuffers)
            .map_err(Into::into)
    }

    /// Registers one exact consumer identity and returns its RAII guard.
    ///
    /// # Errors
    ///
    /// Returns a structured error if the manager's checked registration
    /// identity source is exhausted.
    pub fn register_consumer_scoped(
        &self,
        consumer: Arc<dyn MemoryConsumer>,
    ) -> Result<ConsumerRegistration, QueryResourceContextError> {
        self.buffer_manager
            .register_consumer_scoped(consumer)
            .map_err(Into::into)
    }

    /// Returns point-in-time query account statistics.
    #[must_use]
    pub fn query_stats(&self) -> QueryResourceStats {
        QueryResourceStats {
            query_id: self.query_id,
            allocated_bytes: self.query_pool.allocated(),
            growth_limit_bytes: self.query_pool.growth_limit(),
        }
    }

    /// Returns query-wide resource history, retained independently of cleanup.
    #[must_use]
    pub fn profile_stats(&self) -> super::profile::QueryProfileStats {
        let stats = super::profile::QueryProfileStats {
            resident_granted_bytes: self.query_pool.allocated(),
            resident_peak_bytes: self.query_pool.peak_allocated(),
            #[cfg(feature = "spill")]
            spill_recovery: self.spill_manager.as_ref().and_then(|state| {
                state
                    .root
                    .as_ref()
                    .and_then(|root| root.last_scavenge_report())
            }),
            merge_time_ns: if cfg!(target_arch = "wasm32") {
                None
            } else {
                Some(0)
            },
            ..super::profile::QueryProfileStats::default()
        };
        #[cfg(feature = "spill")]
        if let Some(manager) = self.spill_manager() {
            let spill = manager.profile_totals();
            return super::profile::QueryProfileStats {
                spill_physical: manager.physical_stats(),
                spilled_bytes: spill.0,
                spill_runs: spill.1,
                spill_partitions: spill.2,
                merge_time_ns: spill.3,
                ..stats
            };
        }
        stats
    }

    /// Returns point-in-time database-wide buffer statistics.
    #[must_use]
    pub fn buffer_stats(&self) -> BufferStats {
        self.buffer_manager.stats()
    }

    /// Returns the current database-wide pressure level.
    #[must_use]
    pub fn pressure_level(&self) -> PressureLevel {
        self.buffer_manager.pressure_level()
    }

    /// Returns whether system pressure warrants spilling (High or Critical).
    #[must_use]
    pub fn should_spill(&self) -> bool {
        self.pressure_level().should_spill()
    }

    /// Returns a reference to the buffer manager.
    #[must_use]
    pub fn buffer_manager(&self) -> &Arc<BufferManager> {
        &self.buffer_manager
    }

    /// Returns the admitted per-query manager without creating a leaf.
    #[cfg(feature = "spill")]
    #[must_use]
    pub fn spill_manager(&self) -> Option<&Arc<super::spill::SpillManager>> {
        self.spill_manager
            .as_ref()
            .and_then(|state| state.manager.get())
    }

    /// Returns a coherent logical spill-disk snapshot when this query has a
    /// configured manager.
    #[cfg(feature = "spill")]
    #[must_use]
    pub fn spill_disk_stats(&self) -> Option<super::spill::SpillDiskStats> {
        self.spill_manager().map(|manager| manager.disk_stats())
    }
}

// Unit operator fixtures consume a concrete codec-directory builder. They cannot
// attach an independently admitted manager to another query's resource account.
#[cfg(all(test, feature = "spill"))]
impl super::spill::BorrowedSpillFixture {
    pub(crate) fn build_operator_resources(
        self,
        buffer_manager: Arc<BufferManager>,
        cancellation: QueryCancellationToken,
    ) -> Result<(QueryResourceContext, Arc<super::spill::SpillManager>), QueryResourceContextError>
    {
        let mut context =
            QueryResourceContext::new_with_cancellation(buffer_manager, cancellation)?;
        let manager = Arc::new(self.build().map_err(spill_admission_error)?);
        context.spill_manager = Some(Arc::new(QuerySpillState {
            root: None,
            manager: std::sync::OnceLock::from(Arc::clone(&manager)),
            admission: Mutex::new(()),
            profile_merge: std::sync::atomic::AtomicBool::new(false),
        }));
        Ok((context, manager))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::memory::buffer::{
        BufferManagerConfig, MemoryConsumer, MemoryLimitScope, SpillError,
    };
    use std::collections::HashSet;
    use std::sync::Barrier;

    #[cfg(feature = "spill")]
    #[test]
    fn spill_admission_retains_typed_cancellation_through_io_context() {
        let typed = std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            QueryCancellationError::DeadlineExceeded {
                timeout: Some(std::time::Duration::from_secs(2)),
            },
        );
        let wrapped = std::io::Error::new(std::io::ErrorKind::Interrupted, typed);
        assert!(matches!(
            spill_admission_error(wrapped),
            QueryResourceContextError::SpillAdmission {
                cancellation: Some(QueryCancellationError::DeadlineExceeded { timeout: Some(value) }),
                ..
            } if value == std::time::Duration::from_secs(2)
        ));
        assert!(matches!(
            spill_admission_error(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "ordinary interrupted I/O"
            )),
            QueryResourceContextError::SpillAdmission {
                cancellation: None,
                ..
            }
        ));
    }

    #[test]
    fn test_execution_context_creation() {
        let manager = BufferManager::with_budget(1024 * 1024);
        let ctx = ExecutionMemoryContext::new(manager);

        assert_eq!(ctx.total_allocated(), 0);
        assert_eq!(ctx.pressure_level(), PressureLevel::Normal);
    }

    #[test]
    fn test_execution_context_allocation() {
        let manager = BufferManager::with_budget(1024 * 1024);
        let mut ctx = ExecutionMemoryContext::new(manager);

        let grant = ctx.allocate(1024);
        assert!(grant.is_some());
        assert_eq!(ctx.total_allocated(), 1024);
    }

    #[test]
    fn test_execution_context_tracked_allocation() {
        let manager = BufferManager::with_budget(1024 * 1024);
        let mut ctx = ExecutionMemoryContext::new(manager);

        assert!(ctx.allocate_tracked(1024));
        assert_eq!(ctx.total_allocated(), 1024);

        ctx.release_all();
        assert_eq!(ctx.total_allocated(), 0);
    }

    #[test]
    fn test_adjusted_chunk_size_normal() {
        let manager = BufferManager::with_budget(1024 * 1024);
        let ctx = ExecutionMemoryContext::new(manager);

        assert_eq!(ctx.adjusted_chunk_size(2048), 2048);
        assert_eq!(ctx.optimal_chunk_size(), 2048);
    }

    #[test]
    fn test_adjusted_chunk_size_under_pressure() {
        let config = BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 0.70,
            evict_limit_fraction: 0.85,
            hard_limit_fraction: 0.95,
            background_eviction: false,
            spill_path: None,
        };
        let manager = BufferManager::new(config);

        // Allocate to reach high pressure (>85%)
        let _g = manager.try_allocate(860, MemoryRegion::ExecutionBuffers);

        let ctx = ExecutionMemoryContext::new(manager);
        assert_eq!(ctx.pressure_level(), PressureLevel::High);
        assert_eq!(ctx.adjusted_chunk_size(2048), HIGH_PRESSURE_CHUNK_SIZE);
        assert!(ctx.should_reduce_chunk_size());
    }

    #[test]
    fn test_builder() {
        let manager = BufferManager::with_budget(1024 * 1024);

        let ctx = ExecutionMemoryContextBuilder::new(manager)
            .with_initial_allocation(4096)
            .build();

        assert!(ctx.is_some());
        let ctx = ctx.unwrap();
        assert_eq!(ctx.total_allocated(), 4096);
    }

    #[test]
    fn test_builder_insufficient_memory() {
        let manager = BufferManager::with_budget(1000);

        // Try to allocate more than available
        let ctx = ExecutionMemoryContextBuilder::new(manager)
            .with_initial_allocation(10000)
            .build();

        assert!(ctx.is_none());
    }

    struct EqualLabelConsumer;

    impl MemoryConsumer for EqualLabelConsumer {
        fn name(&self) -> &str {
            "same-diagnostic-label"
        }

        fn memory_usage(&self) -> usize {
            0
        }

        fn eviction_priority(&self) -> u8 {
            0
        }

        fn region(&self) -> MemoryRegion {
            MemoryRegion::ExecutionBuffers
        }

        fn evict(&self, _target_bytes: usize) -> usize {
            0
        }

        fn can_spill(&self) -> bool {
            false
        }

        fn spill(&self, _target_bytes: usize) -> Result<usize, SpillError> {
            Ok(0)
        }

        fn current_tier(&self) -> grafeo_common::memory::buffer::StorageTier {
            grafeo_common::memory::buffer::StorageTier::Uninitialized
        }
    }

    #[test]
    fn query_context_grant_accounts_locally_and_globally_until_drop() {
        let manager = BufferManager::with_budget(1_000);
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();

        let grant = context.try_allocate(128).unwrap();
        assert_eq!(grant.size(), 128);
        assert_eq!(context.query_stats().allocated_bytes, 128);
        assert_eq!(context.buffer_stats().total_allocated, 128);
        assert_eq!(
            context
                .buffer_stats()
                .region_usage(MemoryRegion::ExecutionBuffers),
            128
        );

        drop(grant);
        assert_eq!(context.query_stats().allocated_bytes, 0);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn query_fair_share_denial_does_not_change_global_accounting() {
        let manager = BufferManager::with_budget(100);
        let first = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let _second = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let fair_limit = first.query_stats().growth_limit_bytes;
        let grant = first.try_allocate(fair_limit).unwrap();
        let before = manager.allocated();

        let error = first.try_allocate(1).unwrap_err();
        assert!(matches!(
            error,
            QueryResourceContextError::Memory(
                grafeo_common::memory::buffer::MemoryGrantError::LimitExceeded {
                    scope: MemoryLimitScope::Query,
                    ..
                }
            )
        ));
        assert_eq!(manager.allocated(), before);

        drop(grant);
    }

    #[test]
    fn cloned_context_shares_id_pool_and_accounting_but_new_context_does_not() {
        let manager = BufferManager::with_budget(10_000);
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let clone = context.clone();
        let independent = QueryResourceContext::new(manager).unwrap();

        assert_eq!(clone.query_id(), context.query_id());
        assert_ne!(independent.query_id(), context.query_id());

        let grant = clone.try_allocate(64).unwrap();
        assert_eq!(context.query_stats().allocated_bytes, 64);
        assert_eq!(independent.query_stats().allocated_bytes, 0);
        drop(grant);
    }

    #[test]
    fn cloned_context_shares_cancellation_but_independent_context_does_not() {
        let manager = BufferManager::with_budget(10_000);
        let control = super::super::cancellation::QueryExecutionControl::new();
        let token = control.token();
        let cancellation = control.cancellation_handle();
        let context =
            QueryResourceContext::new_with_cancellation(Arc::clone(&manager), token.clone())
                .unwrap();
        let clone = context.clone();
        let independent = QueryResourceContext::new(manager).unwrap();

        assert!(cancellation.try_cancel());

        assert_eq!(
            context.check_cancelled(),
            Err(QueryCancellationError::Cancelled)
        );
        assert_eq!(
            clone.check_cancelled(),
            Err(QueryCancellationError::Cancelled)
        );
        assert_eq!(independent.check_cancelled(), Ok(()));
    }

    #[test]
    fn query_ids_increase_across_distinct_buffer_managers() {
        let first = QueryResourceContext::new(BufferManager::with_budget(1_000)).unwrap();
        let second = QueryResourceContext::new(BufferManager::with_budget(1_000)).unwrap();

        assert!(second.query_id() > first.query_id());
    }

    #[test]
    fn outstanding_grant_keeps_query_pool_live_after_context_drop() {
        let manager = BufferManager::with_budget(100);
        let first = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let second = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let grant = first.try_allocate(1).unwrap();
        let shared_limit = second.query_stats().growth_limit_bytes;

        drop(first);
        assert_eq!(second.query_stats().growth_limit_bytes, shared_limit);

        drop(grant);
        assert!(second.query_stats().growth_limit_bytes > shared_limit);
    }

    #[test]
    fn concurrent_context_creation_never_reuses_a_query_id() {
        const THREADS: usize = 64;
        let manager = BufferManager::with_budget(1_000_000);
        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let manager = Arc::clone(&manager);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    QueryResourceContext::new(manager).unwrap().query_id()
                })
            })
            .collect();
        let ids: HashSet<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        assert_eq!(ids.len(), THREADS);
        assert!(ids.iter().all(|id| id.get() != 0));
    }

    #[test]
    fn local_query_id_source_issues_max_once_then_fails_closed() {
        let source = parking_lot::Mutex::new(u64::MAX - 1);

        assert_eq!(next_query_execution_id(&source).unwrap().get(), u64::MAX);
        assert_eq!(
            next_query_execution_id(&source).unwrap_err(),
            QueryResourceContextError::QueryExecutionIdExhausted
        );
        assert_eq!(*source.lock(), u64::MAX);
    }

    #[test]
    fn scoped_registrations_with_equal_labels_are_independent() {
        let manager = BufferManager::with_budget(1_000);
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let first = context
            .register_consumer_scoped(Arc::new(EqualLabelConsumer))
            .unwrap();
        let second = context
            .register_consumer_scoped(Arc::new(EqualLabelConsumer))
            .unwrap();
        assert_ne!(first.id(), second.id());
        assert_eq!(manager.stats().consumer_count, 2);

        drop(first);
        assert_eq!(manager.stats().consumer_count, 1);
        drop(second);
        assert_eq!(manager.stats().consumer_count, 0);
    }
}
