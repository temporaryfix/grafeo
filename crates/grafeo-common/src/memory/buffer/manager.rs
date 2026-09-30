//! Unified buffer manager implementation.

use super::consumer::{MemoryConsumer, MemoryConsumerCallbackContext, in_memory_consumer_callback};
use super::grant::{GrantAccount, MemoryGrant, MemoryGrantError, MemoryLimitScope};
use super::region::MemoryRegion;
use super::stats::{BufferStats, PressureLevel};
use parking_lot::{Condvar, Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use thiserror::Error;

/// Default memory budget as a fraction of system memory.
const DEFAULT_MEMORY_FRACTION: f64 = 0.75;

/// Opaque identity of one exact consumer registration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConsumerRegistrationId(usize);

/// Failure to create a unique consumer registration.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConsumerRegistrationError {
    /// This manager's monotonic identity space is exhausted.
    #[error("buffer-manager consumer-registration identity space exhausted")]
    IdentityExhausted,
}

/// Failure to synchronously prove a deactivated registration is quiescent.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConsumerRegistrationCloseError {
    /// Deactivation succeeded without waiting. Callbacks with older leases may
    /// still finish because waiting from another callback could form a cycle.
    #[error(
        "consumer registration is deactivated without a quiescence wait; callback context cannot wait for in-flight callbacks"
    )]
    NonQuiescentCallbackContext,
}

struct RegisteredConsumer {
    id: Option<ConsumerRegistrationId>,
    name: Arc<str>,
    consumer: Arc<dyn MemoryConsumer>,
    lifecycle: Arc<RegistrationLifecycle>,
}

impl fmt::Debug for RegisteredConsumer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RegisteredConsumer")
            .field("id", &self.id)
            .field("name", &self.name)
            .finish()
    }
}

struct ConsumerSnapshot {
    name: Arc<str>,
    consumer: Arc<dyn MemoryConsumer>,
    lifecycle: Arc<RegistrationLifecycle>,
}

struct RegistrationCallbackState {
    active: bool,
    in_flight: usize,
}

struct RegistrationLifecycle {
    state: Mutex<RegistrationCallbackState>,
    quiescent: Condvar,
}

impl RegistrationLifecycle {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(RegistrationCallbackState {
                active: true,
                in_flight: 0,
            }),
            quiescent: Condvar::new(),
        })
    }

    fn try_acquire(self: &Arc<Self>) -> Option<ConsumerCallbackLease> {
        let mut state = self.state.lock();
        if !state.active {
            return None;
        }
        state.in_flight = state.in_flight.checked_add(1)?;
        Some(ConsumerCallbackLease {
            lifecycle: Arc::clone(self),
        })
    }

    fn deactivate(&self) {
        self.state.lock().active = false;
    }

    fn wait_until_quiescent(&self) {
        let mut state = self.state.lock();
        while state.in_flight != 0 {
            self.quiescent.wait(&mut state);
        }
    }
}

struct ConsumerCallbackLease {
    lifecycle: Arc<RegistrationLifecycle>,
}

#[derive(Default)]
struct ForceRamState {
    pinned: HashSet<String>,
    destructive_in_flight: HashMap<Arc<str>, usize>,
}

struct DestructiveCallbackLease<'a> {
    manager: &'a BufferManager,
    name: Arc<str>,
    _registration: ConsumerCallbackLease,
}

impl Drop for DestructiveCallbackLease<'_> {
    fn drop(&mut self) {
        let mut force_ram = self.manager.force_ram_consumers.lock();
        if let Some(in_flight) = force_ram.destructive_in_flight.get_mut(&self.name) {
            *in_flight = in_flight.saturating_sub(1);
            if *in_flight == 0 {
                force_ram.destructive_in_flight.remove(&self.name);
            }
        }
        self.manager.force_ram_quiescent.notify_all();
    }
}

impl Drop for ConsumerCallbackLease {
    fn drop(&mut self) {
        let mut state = self.lifecycle.state.lock();
        state.in_flight = state.in_flight.saturating_sub(1);
        if state.in_flight == 0 {
            self.lifecycle.quiescent.notify_all();
        }
    }
}

/// RAII ownership of one exact consumer registration.
///
/// Dropping this guard unregisters only its identity, even when concurrent
/// operators use the same diagnostic [`MemoryConsumer::name`].
#[must_use = "dropping the registration immediately unregisters the consumer"]
pub struct ConsumerRegistration {
    manager: Weak<BufferManager>,
    id: ConsumerRegistrationId,
    active: bool,
    lifecycle: Arc<RegistrationLifecycle>,
}

impl ConsumerRegistration {
    /// Returns the opaque identity owned by this guard.
    #[must_use]
    pub fn id(&self) -> ConsumerRegistrationId {
        self.id
    }

    /// Deactivates this registration and, when cycle-safe, waits for callbacks
    /// that already acquired a lease to finish.
    ///
    /// A registration removed from the manager cannot start new callbacks,
    /// including through an older consumer snapshot. Plain [`Drop`] performs
    /// the same deactivation without waiting, so it is safe from inside a
    /// consumer callback.
    ///
    /// Outside all memory-consumer callbacks, `Ok(())` is a quiescence
    /// guarantee. Inside any consumer callback (regardless of manager, name,
    /// or callback kind), this method never waits: it returns
    /// [`ConsumerRegistrationCloseError::NonQuiescentCallbackContext`]
    /// after deactivation rather than making a synchronous quiescence claim.
    /// No new callback can start, but older leases may still finish.
    ///
    /// # Errors
    ///
    /// Returns a structured nonquiescent result when waiting from the current
    /// callback context could form a cross-registration cycle.
    pub fn close(mut self) -> Result<(), ConsumerRegistrationCloseError> {
        if self.active {
            self.active = false;
            if let Some(manager) = self.manager.upgrade() {
                manager.remove_registration(self.id);
            }
        }
        self.lifecycle.deactivate();
        if in_memory_consumer_callback() {
            Err(ConsumerRegistrationCloseError::NonQuiescentCallbackContext)
        } else {
            self.lifecycle.wait_until_quiescent();
            Ok(())
        }
    }
}

impl fmt::Debug for ConsumerRegistration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConsumerRegistration")
            .field("id", &self.id)
            .field("active", &self.active)
            .finish_non_exhaustive()
    }
}

impl Drop for ConsumerRegistration {
    fn drop(&mut self) {
        if self.active {
            self.active = false;
            if let Some(manager) = self.manager.upgrade() {
                manager.remove_registration(self.id);
            }
        }
        self.lifecycle.deactivate();
    }
}

/// Configuration for the buffer manager.
#[derive(Debug, Clone)]
pub struct BufferManagerConfig {
    /// Total memory budget in bytes.
    pub budget: usize,
    /// Soft limit threshold (default: 70%).
    pub soft_limit_fraction: f64,
    /// Eviction threshold (default: 85%).
    pub evict_limit_fraction: f64,
    /// Hard limit threshold (default: 95%).
    pub hard_limit_fraction: f64,
    /// Enable background eviction thread.
    pub background_eviction: bool,
    /// Directory for spilling data to disk.
    pub spill_path: Option<PathBuf>,
}

impl BufferManagerConfig {
    /// Detects system memory size.
    ///
    /// Returns a conservative estimate if detection fails.
    #[must_use]
    pub fn detect_system_memory() -> usize {
        // Under Miri, file I/O is blocked by isolation: use fallback directly
        #[cfg(miri)]
        {
            return Self::fallback_system_memory();
        }

        // Try to detect system memory
        // On failure, return a conservative 1GB default
        #[cfg(not(miri))]
        {
            #[cfg(target_os = "windows")]
            {
                // Windows: Use GetPhysicallyInstalledSystemMemory or GlobalMemoryStatusEx
                // For now, use a fallback
                Self::fallback_system_memory()
            }

            #[cfg(target_os = "linux")]
            {
                // Linux: Read from /proc/meminfo
                if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
                    for line in contents.lines() {
                        if line.starts_with("MemTotal:")
                            && let Some(kb_str) = line.split_whitespace().nth(1)
                            && let Ok(kb) = kb_str.parse::<usize>()
                        {
                            return kb * 1024;
                        }
                    }
                }
                Self::fallback_system_memory()
            }

            #[cfg(target_os = "macos")]
            {
                // macOS: Use sysctl
                Self::fallback_system_memory()
            }

            #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
            {
                Self::fallback_system_memory()
            }
        }
    }

    fn fallback_system_memory() -> usize {
        // Default to 1GB if detection fails
        1024 * 1024 * 1024
    }

    /// Creates a config with the given budget.
    #[must_use]
    pub fn with_budget(budget: usize) -> Self {
        Self {
            budget,
            ..Default::default()
        }
    }
}

impl Default for BufferManagerConfig {
    fn default() -> Self {
        let system_memory = Self::detect_system_memory();
        Self {
            // reason: memory fraction (0.0..1.0) of a positive usize is always a valid positive usize
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            budget: (system_memory as f64 * DEFAULT_MEMORY_FRACTION) as usize,
            soft_limit_fraction: 0.70,
            evict_limit_fraction: 0.85,
            hard_limit_fraction: 0.95,
            background_eviction: false, // Disabled by default for simplicity
            spill_path: None,
        }
    }
}

/// The central unified buffer manager.
///
/// Manages memory allocation across all subsystems with pressure-aware
/// eviction and optional spilling support.
pub struct BufferManager {
    /// Configuration.
    config: BufferManagerConfig,
    /// Total allocated bytes.
    allocated: AtomicUsize,
    /// Fail-closed fence for an unrecoverable multi-counter transition.
    accounting_poisoned: AtomicBool,
    /// Number of live query memory pools used to derive a dynamic fair share.
    active_query_pools: AtomicUsize,
    /// Per-region allocated bytes.
    region_allocated: [AtomicUsize; 4],
    /// Registered memory consumers.
    consumers: RwLock<Vec<RegisteredConsumer>>,
    /// Monotonic identity source for scoped consumer registrations.
    next_consumer_registration: AtomicUsize,
    /// Set of consumer names pinned to RAM via `TierOverride::ForceRam`.
    ///
    /// Phase 8g enforcement: any spill loop ([`Self::run_eviction_internal`],
    /// [`Self::spill_all`], [`Self::spill_consumer_by_name`]) skips consumers
    /// whose name is in this set. Populated by the engine at startup based on
    /// `Config.section_configs`.
    force_ram_consumers: Mutex<ForceRamState>,
    /// Wakes pin callers when pre-existing destructive callback leases finish.
    force_ram_quiescent: Condvar,
    /// Computed soft limit in bytes.
    soft_limit: usize,
    /// Computed eviction limit in bytes.
    evict_limit: usize,
    /// Computed hard limit in bytes.
    hard_limit: usize,
    /// Shutdown flag.
    shutdown: AtomicBool,
}

struct PendingGlobalReservation<'a> {
    manager: &'a BufferManager,
    size: usize,
    region: MemoryRegion,
    armed: bool,
}

struct PendingQueryReservation<'a> {
    pool: &'a QueryMemoryPool,
    size: usize,
    armed: bool,
}

impl PendingQueryReservation<'_> {
    fn commit(mut self) -> Result<(), MemoryGrantError> {
        if self.size == 0 {
            self.armed = false;
            return Ok(());
        }

        let _transition = self.pool.local_transition.lock();
        if self.pool.accounting_poisoned.load(Ordering::Acquire) {
            return Err(MemoryGrantError::AccountingPoisoned {
                account: "query memory pool",
            });
        }
        let allocated = self.pool.allocated.load(Ordering::Acquire);
        let pending = self.pool.pending.load(Ordering::Acquire);
        let (Some(next_allocated), Some(next_pending)) = (
            allocated.checked_add(self.size),
            pending.checked_sub(self.size),
        ) else {
            self.pool.accounting_poisoned.store(true, Ordering::Release);
            return Err(MemoryGrantError::AccountingPoisoned {
                account: "query memory pool",
            });
        };

        // The global reservation is already live. Publish committed local
        // ownership before removing the pending admission so another request
        // can never reuse this capacity between the two stages.
        self.pool.allocated.store(next_allocated, Ordering::Release);
        self.pool
            .peak_allocated
            .fetch_max(next_allocated, Ordering::Relaxed);
        self.pool.pending.store(next_pending, Ordering::Release);
        self.armed = false;
        Ok(())
    }
}

impl Drop for PendingQueryReservation<'_> {
    fn drop(&mut self) {
        if !self.armed || self.size == 0 {
            return;
        }
        let _transition = self.pool.local_transition.lock();
        let pending = self.pool.pending.load(Ordering::Acquire);
        if let Some(next) = pending.checked_sub(self.size) {
            self.pool.pending.store(next, Ordering::Release);
        } else {
            self.pool.accounting_poisoned.store(true, Ordering::Release);
        }
    }
}

impl PendingGlobalReservation<'_> {
    fn commit(mut self) {
        self.armed = false;
    }
}

impl Drop for PendingGlobalReservation<'_> {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.manager.release_accounted(self.size, self.region);
        }
    }
}

fn checked_add_counter(counter: &AtomicUsize, amount: usize) -> Result<(), MemoryGrantError> {
    let mut current = counter.load(Ordering::Acquire);
    loop {
        let Some(next) = current.checked_add(amount) else {
            return Err(MemoryGrantError::ArithmeticOverflow {
                current_bytes: current,
                additional_bytes: amount,
            });
        };
        match counter.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Ok(()),
            Err(observed) => current = observed,
        }
    }
}

fn checked_sub_counter(
    counter: &AtomicUsize,
    amount: usize,
    account: &'static str,
) -> Result<(), MemoryGrantError> {
    let mut current = counter.load(Ordering::Acquire);
    loop {
        let Some(next) = current.checked_sub(amount) else {
            return Err(MemoryGrantError::AccountingUnderflow {
                account,
                accounted_bytes: current,
                release_bytes: amount,
            });
        };
        match counter.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Ok(()),
            Err(observed) => current = observed,
        }
    }
}

impl BufferManager {
    /// Creates a new buffer manager with the given configuration.
    #[must_use]
    pub fn new(config: BufferManagerConfig) -> Arc<Self> {
        // reason: limit fractions (0.0..1.0) of a positive usize are always valid positive usizes
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let soft_limit = (config.budget as f64 * config.soft_limit_fraction) as usize;
        // reason: limit fractions (0.0..1.0) of a positive usize are always valid positive usizes
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let evict_limit = (config.budget as f64 * config.evict_limit_fraction) as usize;
        // reason: limit fractions (0.0..1.0) of a positive usize are always valid positive usizes
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let hard_limit = (config.budget as f64 * config.hard_limit_fraction) as usize;

        Arc::new(Self {
            config,
            allocated: AtomicUsize::new(0),
            accounting_poisoned: AtomicBool::new(false),
            active_query_pools: AtomicUsize::new(0),
            region_allocated: [
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
            ],
            consumers: RwLock::new(Vec::new()),
            next_consumer_registration: AtomicUsize::new(1),
            force_ram_consumers: Mutex::new(ForceRamState::default()),
            force_ram_quiescent: Condvar::new(),
            soft_limit,
            evict_limit,
            hard_limit,
            shutdown: AtomicBool::new(false),
        })
    }

    /// Creates a buffer manager with default configuration.
    #[must_use]
    pub fn with_defaults() -> Arc<Self> {
        Self::new(BufferManagerConfig::default())
    }

    /// Creates a buffer manager with a specific budget.
    #[must_use]
    pub fn with_budget(budget: usize) -> Arc<Self> {
        Self::new(BufferManagerConfig::with_budget(budget))
    }

    /// Creates one RAII query memory pool backed by this manager.
    ///
    /// Every live pool receives a dynamic equal growth cap under the global
    /// hard limit. This is not a reservation or availability guarantee:
    /// existing grants are not revoked when another query starts, and global
    /// pressure may still deny growth below the query-local cap.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryGrantError::QueryPoolCountExhausted`] if the live-query
    /// counter cannot be incremented.
    pub fn new_query_pool(self: &Arc<Self>) -> Result<Arc<QueryMemoryPool>, MemoryGrantError> {
        let previous = self
            .active_query_pools
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .map_err(|_| MemoryGrantError::QueryPoolCountExhausted)?;
        debug_assert!(previous < usize::MAX);
        Ok(Arc::new(QueryMemoryPool {
            manager: Arc::clone(self),
            peak_allocated: AtomicUsize::new(0),
            allocated: AtomicUsize::new(0),
            pending: AtomicUsize::new(0),
            local_transition: Mutex::new(()),
            accounting_poisoned: AtomicBool::new(false),
        }))
    }

    /// Attempts to allocate memory for the given region.
    ///
    /// Returns `None` if allocation would exceed the hard limit after
    /// eviction attempts.
    pub fn try_allocate(
        self: &Arc<Self>,
        size: usize,
        region: MemoryRegion,
    ) -> Option<MemoryGrant> {
        let reservation = self.reserve_with_eviction(size, region).ok()?;

        // Check pressure and potentially trigger background eviction
        self.check_pressure();

        let grant = MemoryGrant::new(Arc::clone(self) as Arc<dyn GrantAccount>, size, region);
        reservation.commit();
        Some(grant)
    }

    /// Returns the current pressure level.
    #[must_use]
    pub fn pressure_level(&self) -> PressureLevel {
        let current = self.allocated.load(Ordering::Relaxed);
        self.compute_pressure_level(current)
    }

    /// Returns current buffer statistics.
    ///
    /// This is an observational, non-atomic snapshot: concurrent two-stage
    /// account transitions can become visible between individual field loads.
    /// Every completed transition preserves the aggregate invariants.
    #[must_use]
    pub fn stats(&self) -> BufferStats {
        let total_allocated = self.allocated.load(Ordering::Relaxed);
        BufferStats {
            budget: self.config.budget,
            total_allocated,
            region_allocated: [
                self.region_allocated[0].load(Ordering::Relaxed),
                self.region_allocated[1].load(Ordering::Relaxed),
                self.region_allocated[2].load(Ordering::Relaxed),
                self.region_allocated[3].load(Ordering::Relaxed),
            ],
            pressure_level: self.compute_pressure_level(total_allocated),
            consumer_count: self.consumers.read().len(),
        }
    }

    /// Registers a memory consumer for eviction callbacks.
    pub fn register_consumer(&self, consumer: Arc<dyn MemoryConsumer>) {
        let name: Arc<str> = {
            let _callback_context = MemoryConsumerCallbackContext::enter();
            Arc::from(consumer.name())
        };
        let registration = RegisteredConsumer {
            id: None,
            name,
            consumer,
            lifecycle: RegistrationLifecycle::new(),
        };
        let mut consumers = self.consumers.write();
        consumers.reserve(1);
        consumers.push(registration);
    }

    /// Registers a consumer under one unique RAII identity.
    ///
    /// # Errors
    ///
    /// Returns [`ConsumerRegistrationError::IdentityExhausted`] without
    /// registering the consumer if the monotonic identity space is exhausted.
    pub fn register_consumer_scoped(
        self: &Arc<Self>,
        consumer: Arc<dyn MemoryConsumer>,
    ) -> Result<ConsumerRegistration, ConsumerRegistrationError> {
        let raw_id = self
            .next_consumer_registration
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .map_err(|_| ConsumerRegistrationError::IdentityExhausted)?;
        let id = ConsumerRegistrationId(raw_id);
        let name: Arc<str> = {
            let _callback_context = MemoryConsumerCallbackContext::enter();
            Arc::from(consumer.name())
        };
        let lifecycle = RegistrationLifecycle::new();
        let registration = RegisteredConsumer {
            id: Some(id),
            name,
            consumer,
            lifecycle: Arc::clone(&lifecycle),
        };
        let mut consumers = self.consumers.write();
        consumers.reserve(1);
        consumers.push(registration);
        drop(consumers);
        Ok(ConsumerRegistration {
            manager: Arc::downgrade(self),
            id,
            active: true,
            lifecycle,
        })
    }

    /// Unregisters legacy unscoped consumers by name.
    ///
    /// Scoped operator registrations are deliberately preserved; they can
    /// only be removed by their unique RAII guard. This name-based seam remains
    /// for section-level ForceRam lifecycle compatibility. Removal deactivates
    /// old snapshots before returning, but callbacks that already hold a lease
    /// may finish. Removed consumer destructors run after the registry unlocks.
    pub fn unregister_consumer(&self, name: &str) {
        let mut removed = Vec::new();
        {
            let mut consumers = self.consumers.write();
            removed.reserve(consumers.len());
            let mut index = 0;
            while index < consumers.len() {
                let remove =
                    consumers[index].id.is_none() && consumers[index].name.as_ref() == name;
                if remove {
                    consumers[index].lifecycle.deactivate();
                    removed.push(consumers.swap_remove(index));
                } else {
                    index += 1;
                }
            }
            let same_name_survives = consumers
                .iter()
                .any(|registration| registration.name.as_ref() == name);
            if !same_name_survives {
                self.force_ram_consumers.lock().pinned.remove(name);
            }
        }
        drop(removed);
    }

    fn remove_registration(&self, id: ConsumerRegistrationId) -> bool {
        let removed = {
            let mut consumers = self.consumers.write();
            let removed = consumers
                .iter()
                .position(|registration| registration.id == Some(id))
                .map(|index| {
                    consumers[index].lifecycle.deactivate();
                    consumers.swap_remove(index)
                });
            if let Some(registration) = &removed {
                let same_name_survives = consumers
                    .iter()
                    .any(|candidate| candidate.name == registration.name);
                if !same_name_survives {
                    self.force_ram_consumers
                        .lock()
                        .pinned
                        .remove(registration.name.as_ref());
                }
            }
            removed
        };
        if let Some(registration) = removed {
            drop(registration);
            true
        } else {
            false
        }
    }

    /// Pins a consumer to RAM via [`crate::storage::TierOverride::ForceRam`].
    ///
    /// Insertion is the linearization point: it blocks new destructive leases,
    /// then this call waits for leases that started earlier to finish. Thus an
    /// external caller returns quiescent. Any memory-consumer callback takes
    /// the cycle-free nonblocking path, across every manager/name: it blocks
    /// every new lease, but pre-existing callbacks may finish afterward.
    /// Consumer code never runs under the pin lock.
    ///
    /// State-idempotent: repeated calls retain one pin. An external repeated
    /// call still acts as a quiescence barrier for destructive leases that
    /// started before the pin became visible.
    pub fn mark_force_ram(&self, name: &str) {
        let callback_context = in_memory_consumer_callback();
        let mut force_ram = self.force_ram_consumers.lock();
        force_ram.pinned.insert(name.to_string());
        if !callback_context {
            while force_ram
                .destructive_in_flight
                .get(name)
                .copied()
                .unwrap_or(0)
                != 0
            {
                self.force_ram_quiescent.wait(&mut force_ram);
            }
        }
    }

    /// Removes the [`crate::storage::TierOverride::ForceRam`] pin from a
    /// consumer (Phase 8g). After this call, the consumer participates in
    /// spill again like any other.
    pub fn clear_force_ram(&self, name: &str) {
        self.force_ram_consumers.lock().pinned.remove(name);
    }

    /// Returns `true` if a consumer is currently pinned via ForceRam.
    #[must_use]
    pub fn is_force_ram(&self, name: &str) -> bool {
        self.force_ram_consumers.lock().pinned.contains(name)
    }

    /// Forces eviction to reach the target usage.
    ///
    /// Returns the number of bytes actually freed.
    pub fn evict_to_target(&self, target_bytes: usize) -> usize {
        let current = self.allocated.load(Ordering::Relaxed);
        if current <= target_bytes {
            return 0;
        }

        let to_free = current - target_bytes;
        self.run_eviction_internal(to_free)
    }

    /// Spills all consumers that support it, regardless of memory pressure.
    ///
    /// Used when `TierOverride::ForceDisk` is configured. Returns total bytes freed.
    /// Consumers pinned via [`Self::mark_force_ram`] are skipped.
    pub fn spill_all(&self) -> usize {
        let consumers = self.consumer_snapshot();
        let mut total_freed: usize = 0;
        for consumer in consumers {
            let can_spill = self
                .with_consumer_callback(&consumer, true, |callback| callback.can_spill())
                .unwrap_or(false);
            if can_spill
                && let Some(Ok(freed)) = self
                    .with_consumer_callback(&consumer, true, |callback| callback.spill(usize::MAX))
            {
                total_freed = total_freed.saturating_add(freed);
            }
        }
        total_freed
    }

    /// Spills all consumers whose [`MemoryConsumer::name`] equals `name`.
    ///
    /// Used for targeted [`crate::storage::TierOverride::ForceDisk`] enforcement
    /// at database open: each section type configured as `ForceDisk` triggers a
    /// spill on its matching consumer only, leaving other consumers untouched.
    ///
    /// Best-effort: a failure on one consumer does not stop the others. Returns
    /// total bytes freed across all matching consumers.
    ///
    /// If `name` is pinned via [`Self::mark_force_ram`], this call is a no-op
    /// and returns `0`. The pin is honored even on explicit-by-name spill
    /// requests: `ForceRam` is a hard contract.
    pub fn spill_consumer_by_name(&self, name: &str) -> usize {
        if self.is_force_ram(name) {
            #[cfg(feature = "tracing")]
            tracing::debug!(
                target: "grafeo::buffer",
                consumer = name,
                "spill skipped: consumer pinned ForceRam"
            );
            return 0;
        }
        let consumers = self.consumer_snapshot();
        let mut total_freed: usize = 0;
        for consumer in consumers {
            if consumer.name.as_ref() != name {
                continue;
            }
            let can_spill = self
                .with_consumer_callback(&consumer, true, |callback| callback.can_spill())
                .unwrap_or(false);
            if can_spill
                && let Some(Ok(freed)) = self
                    .with_consumer_callback(&consumer, true, |callback| callback.spill(usize::MAX))
            {
                total_freed = total_freed.saturating_add(freed);
            }
        }
        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "grafeo::buffer",
            consumer = name,
            freed_bytes = total_freed,
            "tier transition: spill"
        );
        total_freed
    }

    /// Reloads OnDisk consumers back into RAM, in priority order (highest
    /// priority first), as long as projected usage stays below
    /// `target_fraction` of the budget.
    ///
    /// Phase 9a: closes the loop on the spill / reload lifecycle. Today
    /// consumers spill on memory pressure and stay OnDisk forever; this
    /// method gives users (or a future background thread) an explicit
    /// trigger to bring spilled state back into RAM after pressure drops.
    ///
    /// The walk visits consumers whose
    /// [`MemoryConsumer::current_tier`] is
    /// [`super::tiered::StorageTier::OnDisk`] and calls
    /// [`MemoryConsumer::reload`] on each. After each reload, if current
    /// allocation exceeds `target_fraction * budget`, the loop stops and
    /// leaves remaining consumers on disk. `reload()` errors are
    /// logged-and-skipped: the operation is best-effort.
    ///
    /// Returns the number of consumers successfully reloaded.
    ///
    /// `target_fraction` is clamped to `[0.0, 1.0]`. A value of 0.7 means
    /// "stop bringing things back when we'd hit 70% of the budget" —
    /// matching the soft-limit threshold default.
    pub fn reload_eligible(&self, target_fraction: f64) -> usize {
        let target_fraction = target_fraction.clamp(0.0, 1.0);
        // reason: target byte count from a bounded fraction is non-negative and bounded by budget
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let target_bytes = (self.config.budget as f64 * target_fraction) as usize;

        let candidates: Vec<ConsumerSnapshot> = {
            let mut out: Vec<_> = self
                .consumer_snapshot()
                .into_iter()
                .filter_map(|consumer| {
                    let tier = self.with_consumer_callback(&consumer, false, |callback| {
                        callback.current_tier()
                    })?;
                    if tier != super::tiered::StorageTier::OnDisk {
                        return None;
                    }
                    let priority = self.with_consumer_callback(&consumer, false, |callback| {
                        callback.eviction_priority()
                    })?;
                    Some((priority, consumer))
                })
                .collect();
            // Highest priority first: graph storage > active txn > index > query cache.
            out.sort_by_key(|(priority, _)| std::cmp::Reverse(*priority));
            out.into_iter().map(|(_, consumer)| consumer).collect()
        };

        let mut reloaded = 0;
        for consumer in candidates {
            let current = self.allocated.load(Ordering::Relaxed);
            if current >= target_bytes {
                break;
            }
            match self.with_consumer_callback(&consumer, false, |callback| callback.reload()) {
                Some(Ok(())) => {
                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        target: "grafeo::buffer",
                        consumer = consumer.name.as_ref(),
                        "tier transition: reload"
                    );
                    reloaded += 1;
                }
                Some(Err(_e)) => {
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        target: "grafeo::buffer",
                        consumer = consumer.name.as_ref(),
                        error = %_e,
                        "tier reload failed"
                    );
                    continue;
                }
                None => continue,
            }
        }
        #[cfg(feature = "tracing")]
        tracing::debug!(
            target: "grafeo::buffer",
            reloaded_count = reloaded,
            target_fraction = target_fraction,
            "reload_eligible cycle complete"
        );
        reloaded
    }

    /// Returns the current tier reported by each registered consumer that
    /// wraps a section.
    ///
    /// Tier is sourced from [`MemoryConsumer::current_tier`]. Consumers whose
    /// names don't follow the `"section:<TypeName>"` convention are skipped
    /// (e.g. CDC, overlay).
    #[must_use]
    pub fn snapshot_consumer_tiers(&self) -> Vec<(String, super::tiered::StorageTier)> {
        self.consumer_snapshot()
            .into_iter()
            .filter_map(|consumer| {
                let name = Arc::clone(&consumer.name);
                if !name.starts_with("section:") {
                    return None;
                }
                let tier = self
                    .with_consumer_callback(&consumer, false, |callback| callback.current_tier())?;
                Some((name.to_string(), tier))
            })
            .collect()
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &BufferManagerConfig {
        &self.config
    }

    /// Returns the memory budget.
    #[must_use]
    pub fn budget(&self) -> usize {
        self.config.budget
    }

    /// Returns currently allocated bytes.
    ///
    /// This is a point-in-time atomic read, not a reservation or a consistent
    /// snapshot with [`Self::stats`].
    #[must_use]
    pub fn allocated(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    /// Returns available bytes.
    #[must_use]
    pub fn available(&self) -> usize {
        self.config
            .budget
            .saturating_sub(self.allocated.load(Ordering::Relaxed))
    }

    /// Shuts down the buffer manager.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    // === Internal methods ===

    fn compute_pressure_level(&self, current: usize) -> PressureLevel {
        if current >= self.hard_limit {
            PressureLevel::Critical
        } else if current >= self.evict_limit {
            PressureLevel::High
        } else if current >= self.soft_limit {
            PressureLevel::Moderate
        } else {
            PressureLevel::Normal
        }
    }

    fn check_pressure(&self) {
        let level = self.pressure_level();
        if level.requires_eviction() {
            // In a more complete implementation, this would signal
            // a background thread. For now, do synchronous eviction.
            let aggressive = level >= PressureLevel::High;
            self.run_eviction_cycle(aggressive);
        }
    }

    fn run_eviction_cycle(&self, aggressive: bool) -> usize {
        let target = if aggressive {
            self.soft_limit
        } else {
            self.evict_limit
        };

        let current = self.allocated.load(Ordering::Relaxed);
        if current <= target {
            return 0;
        }

        let to_free = current - target;
        self.run_eviction_internal(to_free)
    }

    fn run_eviction_internal(&self, to_free: usize) -> usize {
        // Sort consumers by priority (lowest first = evict first)
        let mut sorted: Vec<_> = self
            .consumer_snapshot()
            .into_iter()
            .filter_map(|consumer| {
                let priority = self.with_consumer_callback(&consumer, false, |callback| {
                    callback.eviction_priority()
                })?;
                Some((priority, consumer))
            })
            .collect();
        sorted.sort_by_key(|(priority, _)| *priority);

        let mut total_freed = 0;
        for (_, consumer) in &sorted {
            if total_freed >= to_free {
                break;
            }

            let remaining = to_free - total_freed;
            let Some(consumer_usage) =
                self.with_consumer_callback(consumer, true, |callback| callback.memory_usage())
            else {
                continue;
            };

            // Ask consumer to evict up to half its usage or remaining needed
            let target_evict = remaining.min(consumer_usage / 2);
            if target_evict > 0
                && let Some(freed) = self
                    .with_consumer_callback(consumer, true, |callback| callback.evict(target_evict))
            {
                total_freed += freed.min(target_evict);
            }
        }

        // If eviction was not enough, try spilling to disk for consumers
        // that support it (e.g., vector indexes with mmap storage).
        // Phase 8g: ForceRam consumers are skipped here too — that's the
        // hard contract ("never move me to disk").
        if total_freed < to_free {
            for (_, consumer) in &sorted {
                if total_freed >= to_free {
                    break;
                }
                let can_spill = self
                    .with_consumer_callback(consumer, true, |callback| callback.can_spill())
                    .unwrap_or(false);
                if !can_spill {
                    continue;
                }
                let remaining = to_free - total_freed;
                match self
                    .with_consumer_callback(consumer, true, |callback| callback.spill(remaining))
                {
                    Some(Ok(freed)) => total_freed += freed.min(remaining),
                    Some(Err(_)) | None => continue,
                }
            }
        }

        total_freed
    }

    fn reserve_global(
        &self,
        size: usize,
        region: MemoryRegion,
    ) -> Result<PendingGlobalReservation<'_>, MemoryGrantError> {
        if self.accounting_poisoned.load(Ordering::Acquire) {
            return Err(MemoryGrantError::AccountingPoisoned {
                account: "buffer manager",
            });
        }
        if size == 0 {
            return Ok(PendingGlobalReservation {
                manager: self,
                size,
                region,
                armed: true,
            });
        }

        let mut current = self.allocated.load(Ordering::Acquire);
        loop {
            let Some(next) = current.checked_add(size) else {
                return Err(MemoryGrantError::ArithmeticOverflow {
                    current_bytes: current,
                    additional_bytes: size,
                });
            };
            if next > self.hard_limit {
                return Err(MemoryGrantError::LimitExceeded {
                    scope: MemoryLimitScope::Global,
                    requested_bytes: next,
                    limit_bytes: self.hard_limit,
                });
            }
            match self.allocated.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if let Err(error) =
                        checked_add_counter(&self.region_allocated[region.index()], size)
                    {
                        if checked_sub_counter(&self.allocated, size, "buffer manager total")
                            .is_err()
                        {
                            self.accounting_poisoned.store(true, Ordering::Release);
                            return Err(MemoryGrantError::AccountingPoisoned {
                                account: "buffer manager",
                            });
                        }
                        return Err(error);
                    }
                    return Ok(PendingGlobalReservation {
                        manager: self,
                        size,
                        region,
                        armed: true,
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }

    fn reserve_with_eviction(
        &self,
        size: usize,
        region: MemoryRegion,
    ) -> Result<PendingGlobalReservation<'_>, MemoryGrantError> {
        match self.reserve_global(size, region) {
            Ok(reservation) => Ok(reservation),
            Err(MemoryGrantError::LimitExceeded { .. }) => {
                self.run_eviction_cycle(true);
                self.reserve_global(size, region)
            }
            Err(error) => Err(error),
        }
    }

    /// Releases region accounting before the global total. A failed second
    /// stage restores the region or poisons the manager; it never undercounts.
    fn release_accounted(&self, size: usize, region: MemoryRegion) -> Result<(), MemoryGrantError> {
        if size == 0 {
            return Ok(());
        }
        if self.accounting_poisoned.load(Ordering::Acquire) {
            return Err(MemoryGrantError::AccountingPoisoned {
                account: "buffer manager",
            });
        }

        checked_sub_counter(
            &self.region_allocated[region.index()],
            size,
            "buffer manager region",
        )?;
        if let Err(error) = checked_sub_counter(&self.allocated, size, "buffer manager total") {
            if checked_add_counter(&self.region_allocated[region.index()], size).is_err() {
                self.accounting_poisoned.store(true, Ordering::Release);
                return Err(MemoryGrantError::AccountingPoisoned {
                    account: "buffer manager",
                });
            }
            return Err(error);
        }
        Ok(())
    }

    /// Clones strong consumer handles while holding the registry lock, then
    /// releases it before any user-supplied consumer method is invoked.
    fn consumer_snapshot(&self) -> Vec<ConsumerSnapshot> {
        self.consumers
            .read()
            .iter()
            .map(|registration| ConsumerSnapshot {
                name: Arc::clone(&registration.name),
                consumer: Arc::clone(&registration.consumer),
                lifecycle: Arc::clone(&registration.lifecycle),
            })
            .collect()
    }

    /// Acquires an active-registration lease, and for destructive callbacks a
    /// ForceRam permit, before invoking consumer code without manager locks.
    fn with_consumer_callback<T>(
        &self,
        consumer: &ConsumerSnapshot,
        respect_force_ram: bool,
        callback: impl FnOnce(&dyn MemoryConsumer) -> T,
    ) -> Option<T> {
        if respect_force_ram {
            let _lease = self.acquire_destructive_callback(consumer)?;
            let _callback_context = MemoryConsumerCallbackContext::enter();
            return Some(callback(consumer.consumer.as_ref()));
        }
        let _lease = consumer.lifecycle.try_acquire()?;
        let _callback_context = MemoryConsumerCallbackContext::enter();
        Some(callback(consumer.consumer.as_ref()))
    }

    fn acquire_destructive_callback<'a>(
        &'a self,
        consumer: &ConsumerSnapshot,
    ) -> Option<DestructiveCallbackLease<'a>> {
        let mut force_ram = self.force_ram_consumers.lock();
        if force_ram.pinned.contains(consumer.name.as_ref()) {
            return None;
        }
        let registration = consumer.lifecycle.try_acquire()?;
        let in_flight = force_ram
            .destructive_in_flight
            .entry(Arc::clone(&consumer.name))
            .or_default();
        *in_flight = in_flight.checked_add(1)?;
        drop(force_ram);
        Some(DestructiveCallbackLease {
            manager: self,
            name: Arc::clone(&consumer.name),
            _registration: registration,
        })
    }
}

impl GrantAccount for BufferManager {
    fn release_accounted(&self, size: usize, region: MemoryRegion) -> Result<(), MemoryGrantError> {
        self.release_accounted(size, region)
    }

    fn try_reserve_growth(
        &self,
        size: usize,
        region: MemoryRegion,
    ) -> Result<(), MemoryGrantError> {
        self.reserve_with_eviction(size, region).map(|reservation| {
            reservation.commit();
        })
    }

    fn allows_untracked_consumption(&self) -> bool {
        false
    }
}

/// Per-query resident-memory account with a dynamic fair share.
///
/// Grants created by this pool reserve both the query-local account and the
/// global [`BufferManager`]. Dropping the last pool/grant handle releases one
/// active-query slot; dropping each grant releases both byte counters.
pub struct QueryMemoryPool {
    manager: Arc<BufferManager>,
    peak_allocated: AtomicUsize,
    /// Committed bytes backed by a completed global reservation.
    allocated: AtomicUsize,
    /// In-flight local admissions not yet committed globally.
    pending: AtomicUsize,
    /// Serializes local pending/committed state transitions.
    local_transition: Mutex<()>,
    accounting_poisoned: AtomicBool,
}

impl QueryMemoryPool {
    /// Highest committed resident reservation observed during this query.
    #[must_use]
    pub fn peak_allocated(&self) -> usize {
        self.peak_allocated.load(Ordering::Relaxed)
    }

    /// Returns this query's currently accounted resident bytes.
    ///
    /// This is a point-in-time atomic read; global and regional counters may
    /// be observed at a different stage of a concurrent transition.
    #[must_use]
    pub fn allocated(&self) -> usize {
        self.allocated.load(Ordering::Acquire)
    }

    /// Returns this query's dynamic growth cap.
    ///
    /// The cap is an equal division among live pools, not reserved capacity or
    /// an availability guarantee. Global pressure can deny a smaller request.
    #[must_use]
    pub fn growth_limit(&self) -> usize {
        let active = self
            .manager
            .active_query_pools
            .load(Ordering::Acquire)
            .max(1);
        self.manager.hard_limit / active
    }

    /// Allocates a resizable RAII grant within the query and global limits.
    ///
    /// # Errors
    ///
    /// Returns a structured arithmetic or limit failure without changing
    /// either account.
    pub fn try_allocate(
        self: &Arc<Self>,
        size: usize,
        region: MemoryRegion,
    ) -> Result<MemoryGrant, MemoryGrantError> {
        self.reserve_growth(size, region)?;
        Ok(MemoryGrant::new(
            Arc::clone(self) as Arc<dyn GrantAccount>,
            size,
            region,
        ))
    }

    fn reserve_local_pending(
        &self,
        size: usize,
    ) -> Result<PendingQueryReservation<'_>, MemoryGrantError> {
        let _transition = self.local_transition.lock();
        if self.accounting_poisoned.load(Ordering::Acquire) {
            return Err(MemoryGrantError::AccountingPoisoned {
                account: "query memory pool",
            });
        }
        if size == 0 {
            return Ok(PendingQueryReservation {
                pool: self,
                size,
                armed: true,
            });
        }

        let allocated = self.allocated.load(Ordering::Acquire);
        let pending = self.pending.load(Ordering::Acquire);
        let Some(current) = allocated.checked_add(pending) else {
            self.accounting_poisoned.store(true, Ordering::Release);
            return Err(MemoryGrantError::AccountingPoisoned {
                account: "query memory pool",
            });
        };
        let Some(next) = current.checked_add(size) else {
            return Err(MemoryGrantError::ArithmeticOverflow {
                current_bytes: current,
                additional_bytes: size,
            });
        };
        let limit = self.growth_limit();
        if next > limit {
            return Err(MemoryGrantError::LimitExceeded {
                scope: MemoryLimitScope::Query,
                requested_bytes: next,
                limit_bytes: limit,
            });
        }
        let Some(next_pending) = pending.checked_add(size) else {
            self.accounting_poisoned.store(true, Ordering::Release);
            return Err(MemoryGrantError::AccountingPoisoned {
                account: "query memory pool",
            });
        };
        self.pending.store(next_pending, Ordering::Release);
        Ok(PendingQueryReservation {
            pool: self,
            size,
            armed: true,
        })
    }

    /// Admits query-local pending capacity before any global work. The local
    /// permit rolls back on denial or panic; committed local bytes publish only
    /// after the global reservation exists.
    fn reserve_growth(&self, size: usize, region: MemoryRegion) -> Result<(), MemoryGrantError> {
        let local = self.reserve_local_pending(size)?;
        let global = self.manager.reserve_with_eviction(size, region)?;
        local.commit()?;
        global.commit();
        Ok(())
    }

    /// Releases query-local accounting first, then the global/region account.
    /// If the global transition fails, local accounting is restored.
    fn release_accounted(&self, size: usize, region: MemoryRegion) -> Result<(), MemoryGrantError> {
        if size == 0 {
            return Ok(());
        }
        {
            let _transition = self.local_transition.lock();
            if self.accounting_poisoned.load(Ordering::Acquire) {
                return Err(MemoryGrantError::AccountingPoisoned {
                    account: "query memory pool",
                });
            }
            checked_sub_counter(&self.allocated, size, "query memory pool")?;
        }
        if let Err(error) = self.manager.release_accounted(size, region) {
            let _transition = self.local_transition.lock();
            if checked_add_counter(&self.allocated, size).is_err() {
                self.accounting_poisoned.store(true, Ordering::Release);
                return Err(MemoryGrantError::AccountingPoisoned {
                    account: "query memory pool",
                });
            }
            return Err(error);
        }
        Ok(())
    }
}

impl GrantAccount for QueryMemoryPool {
    fn release_accounted(&self, size: usize, region: MemoryRegion) -> Result<(), MemoryGrantError> {
        self.release_accounted(size, region)
    }

    fn try_reserve_growth(
        &self,
        size: usize,
        region: MemoryRegion,
    ) -> Result<(), MemoryGrantError> {
        self.reserve_growth(size, region)
    }

    fn allows_untracked_consumption(&self) -> bool {
        false
    }
}

impl Drop for QueryMemoryPool {
    fn drop(&mut self) {
        if self
            .manager
            .active_query_pools
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_sub(1)
            })
            .is_err()
        {
            self.manager
                .accounting_poisoned
                .store(true, Ordering::Release);
        }
    }
}

impl Drop for BufferManager {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::buffer::consumer::priorities;
    use parking_lot::{Condvar, Mutex};
    use std::sync::Barrier;
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    struct TestConsumer {
        name: String,
        usage: AtomicUsize,
        priority: u8,
        region: MemoryRegion,
        evicted: AtomicUsize,
    }

    impl TestConsumer {
        fn new(name: &str, usage: usize, priority: u8, region: MemoryRegion) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                usage: AtomicUsize::new(usage),
                priority,
                region,
                evicted: AtomicUsize::new(0),
            })
        }
    }

    impl MemoryConsumer for TestConsumer {
        fn name(&self) -> &str {
            &self.name
        }

        fn memory_usage(&self) -> usize {
            self.usage.load(Ordering::Relaxed)
        }

        fn eviction_priority(&self) -> u8 {
            self.priority
        }

        fn region(&self) -> MemoryRegion {
            self.region
        }

        fn evict(&self, target_bytes: usize) -> usize {
            let current = self.usage.load(Ordering::Relaxed);
            let to_evict = target_bytes.min(current);
            self.usage.fetch_sub(to_evict, Ordering::Relaxed);
            self.evicted.fetch_add(to_evict, Ordering::Relaxed);
            to_evict
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            if self.memory_usage() == 0 {
                super::super::tiered::StorageTier::Uninitialized
            } else {
                super::super::tiered::StorageTier::InMemory
            }
        }
    }

    #[test]
    fn test_basic_allocation() {
        let config = BufferManagerConfig {
            budget: 1024 * 1024, // 1MB
            ..Default::default()
        };
        let manager = BufferManager::new(config);

        let grant = manager.try_allocate(1024, MemoryRegion::ExecutionBuffers);
        assert!(grant.is_some());
        assert_eq!(manager.stats().total_allocated, 1024);
    }

    #[test]
    fn test_grant_raii_release() {
        let config = BufferManagerConfig {
            budget: 1024,
            ..Default::default()
        };
        let manager = BufferManager::new(config);

        {
            let _grant = manager.try_allocate(512, MemoryRegion::ExecutionBuffers);
            assert_eq!(manager.stats().total_allocated, 512);
        }

        // Grant dropped, memory should be released
        assert_eq!(manager.stats().total_allocated, 0);
    }

    #[test]
    fn test_pressure_levels() {
        let config = BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 0.70,
            evict_limit_fraction: 0.85,
            hard_limit_fraction: 0.95,
            background_eviction: false,
            spill_path: None,
        };
        let manager = BufferManager::new(config);

        assert_eq!(manager.pressure_level(), PressureLevel::Normal);

        // Allocate to 70% (soft limit)
        let _g1 = manager.try_allocate(700, MemoryRegion::ExecutionBuffers);
        assert_eq!(manager.pressure_level(), PressureLevel::Moderate);

        // Allocate to 85% (evict limit)
        let _g2 = manager.try_allocate(150, MemoryRegion::ExecutionBuffers);
        assert_eq!(manager.pressure_level(), PressureLevel::High);

        // Note: Can't easily test Critical without blocking
    }

    #[test]
    fn test_region_tracking() {
        let config = BufferManagerConfig {
            budget: 10000,
            ..Default::default()
        };
        let manager = BufferManager::new(config);

        let _g1 = manager.try_allocate(100, MemoryRegion::GraphStorage);
        let _g2 = manager.try_allocate(200, MemoryRegion::IndexBuffers);
        let _g3 = manager.try_allocate(300, MemoryRegion::ExecutionBuffers);

        let stats = manager.stats();
        assert_eq!(stats.region_usage(MemoryRegion::GraphStorage), 100);
        assert_eq!(stats.region_usage(MemoryRegion::IndexBuffers), 200);
        assert_eq!(stats.region_usage(MemoryRegion::ExecutionBuffers), 300);
        assert_eq!(stats.total_allocated, 600);
    }

    #[test]
    fn test_consumer_registration() {
        let manager = BufferManager::with_budget(10000);

        let consumer = TestConsumer::new(
            "test",
            1000,
            priorities::INDEX_BUFFERS,
            MemoryRegion::IndexBuffers,
        );

        manager.register_consumer(consumer);
        assert_eq!(manager.stats().consumer_count, 1);

        manager.unregister_consumer("test");
        assert_eq!(manager.stats().consumer_count, 0);
    }

    #[test]
    fn scoped_same_name_registration_drops_only_its_own_consumer() {
        let manager = BufferManager::with_budget(10_000);
        let first = TestConsumer::new(
            "same-kind-operator",
            100,
            priorities::EXECUTION_BUFFERS,
            MemoryRegion::ExecutionBuffers,
        );
        let second = TestConsumer::new(
            "same-kind-operator",
            200,
            priorities::EXECUTION_BUFFERS,
            MemoryRegion::ExecutionBuffers,
        );

        let first_registration = manager
            .register_consumer_scoped(first)
            .expect("first registration");
        let second_registration = manager
            .register_consumer_scoped(second)
            .expect("second registration");
        assert_ne!(first_registration.id(), second_registration.id());
        assert_eq!(manager.stats().consumer_count, 2);

        drop(first_registration);
        assert_eq!(manager.stats().consumer_count, 1);

        drop(second_registration);
        assert_eq!(manager.stats().consumer_count, 0);
    }

    #[test]
    fn dropping_last_scoped_registration_clears_its_force_ram_pin() {
        let manager = BufferManager::with_budget(1000);
        let consumer = TestConsumer::new(
            "scoped-pin",
            10,
            priorities::EXECUTION_BUFFERS,
            MemoryRegion::ExecutionBuffers,
        );
        let registration = manager
            .register_consumer_scoped(consumer)
            .expect("scoped registration");
        manager.mark_force_ram("scoped-pin");

        drop(registration);
        assert!(!manager.is_force_ram("scoped-pin"));

        let replacement = TestConsumer::new(
            "scoped-pin",
            10,
            priorities::EXECUTION_BUFFERS,
            MemoryRegion::ExecutionBuffers,
        );
        let _replacement_registration = manager
            .register_consumer_scoped(replacement)
            .expect("replacement registration");
        assert!(!manager.is_force_ram("scoped-pin"));
    }

    #[test]
    fn dropping_one_scoped_registration_preserves_pin_for_same_name_survivor() {
        let manager = BufferManager::with_budget(1000);
        let first = TestConsumer::new(
            "scoped-pin-survivor",
            10,
            priorities::EXECUTION_BUFFERS,
            MemoryRegion::ExecutionBuffers,
        );
        let second = TestConsumer::new(
            "scoped-pin-survivor",
            10,
            priorities::EXECUTION_BUFFERS,
            MemoryRegion::ExecutionBuffers,
        );
        let first_registration = manager
            .register_consumer_scoped(first)
            .expect("first registration");
        let second_registration = manager
            .register_consumer_scoped(second)
            .expect("second registration");
        manager.mark_force_ram("scoped-pin-survivor");

        drop(first_registration);
        assert!(manager.is_force_ram("scoped-pin-survivor"));
        drop(second_registration);
        assert!(!manager.is_force_ram("scoped-pin-survivor"));
    }

    fn wait_for_registration_deactivation(lifecycle: &RegistrationLifecycle) -> bool {
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if !lifecycle.state.lock().active {
                return true;
            }
            std::thread::yield_now();
        }
        false
    }

    #[test]
    fn scoped_removal_deactivates_old_snapshots_before_pin_cleanup() {
        let manager = BufferManager::with_budget(1000);
        let registration = manager
            .register_consumer_scoped(TestConsumer::new(
                "scoped-removal-order",
                10,
                priorities::EXECUTION_BUFFERS,
                MemoryRegion::ExecutionBuffers,
            ))
            .expect("scoped registration");
        manager.mark_force_ram("scoped-removal-order");
        let snapshot = manager
            .consumer_snapshot()
            .pop()
            .expect("registered consumer snapshot");

        let pin_state = manager.force_ram_consumers.lock();
        let removal = {
            let registration = registration;
            std::thread::spawn(move || drop(registration))
        };
        let deactivated_before_pin_cleanup =
            wait_for_registration_deactivation(&snapshot.lifecycle);
        drop(pin_state);
        removal.join().expect("scoped removal thread");

        assert!(
            deactivated_before_pin_cleanup,
            "old snapshots must be deactivated before removal can clear the ForceRam pin"
        );
    }

    #[test]
    fn legacy_removal_deactivates_old_snapshots_before_pin_cleanup() {
        let manager = BufferManager::with_budget(1000);
        manager.register_consumer(TestConsumer::new(
            "legacy-removal-order",
            10,
            priorities::EXECUTION_BUFFERS,
            MemoryRegion::ExecutionBuffers,
        ));
        manager.mark_force_ram("legacy-removal-order");
        let snapshot = manager
            .consumer_snapshot()
            .pop()
            .expect("registered consumer snapshot");

        let pin_state = manager.force_ram_consumers.lock();
        let removal = {
            let manager = Arc::clone(&manager);
            std::thread::spawn(move || manager.unregister_consumer("legacy-removal-order"))
        };
        let deactivated_before_pin_cleanup =
            wait_for_registration_deactivation(&snapshot.lifecycle);
        drop(pin_state);
        removal.join().expect("legacy removal thread");

        assert!(
            deactivated_before_pin_cleanup,
            "old snapshots must be deactivated before removal can clear the ForceRam pin"
        );
    }

    struct SelfUnregisteringConsumer {
        manager: Weak<BufferManager>,
        registration: Mutex<Option<ConsumerRegistration>>,
        usage: AtomicUsize,
    }

    impl MemoryConsumer for SelfUnregisteringConsumer {
        fn name(&self) -> &str {
            "self-unregistering"
        }

        fn memory_usage(&self) -> usize {
            self.usage.load(Ordering::Acquire)
        }

        fn eviction_priority(&self) -> u8 {
            priorities::EXECUTION_BUFFERS
        }

        fn region(&self) -> MemoryRegion {
            MemoryRegion::ExecutionBuffers
        }

        fn evict(&self, target_bytes: usize) -> usize {
            let manager = self.manager.upgrade().expect("manager is live");
            let callbacks_are_unlocked = manager.consumers.try_write().is_some();
            assert!(
                callbacks_are_unlocked,
                "consumer callback ran while the registration lock was held"
            );
            drop(self.registration.lock().take());

            let current = self.usage.load(Ordering::Acquire);
            let released = current.min(target_bytes);
            self.usage.fetch_sub(released, Ordering::AcqRel);
            released
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            super::super::tiered::StorageTier::InMemory
        }
    }

    #[test]
    fn eviction_callback_can_drop_its_own_scoped_registration() {
        let manager = BufferManager::with_budget(10_000);
        let consumer = Arc::new(SelfUnregisteringConsumer {
            manager: Arc::downgrade(&manager),
            registration: Mutex::new(None),
            usage: AtomicUsize::new(100),
        });
        let registration = manager
            .register_consumer_scoped(Arc::clone(&consumer) as Arc<dyn MemoryConsumer>)
            .expect("scoped registration");
        *consumer.registration.lock() = Some(registration);
        manager.allocated.store(100, Ordering::Release);

        assert_eq!(manager.evict_to_target(0), 50);
        assert_eq!(manager.stats().consumer_count, 0);
    }

    struct PanickingEvictionConsumer;

    impl MemoryConsumer for PanickingEvictionConsumer {
        fn name(&self) -> &str {
            "panicking-eviction"
        }

        fn memory_usage(&self) -> usize {
            100
        }

        fn eviction_priority(&self) -> u8 {
            priorities::EXECUTION_BUFFERS
        }

        fn region(&self) -> MemoryRegion {
            MemoryRegion::ExecutionBuffers
        }

        fn evict(&self, _target_bytes: usize) -> usize {
            panic!("deterministic eviction callback panic")
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            super::super::tiered::StorageTier::InMemory
        }
    }

    fn panic_rollback_manager() -> Arc<BufferManager> {
        BufferManager::new(BufferManagerConfig {
            budget: 100,
            soft_limit_fraction: 0.5,
            evict_limit_fraction: 0.5,
            hard_limit_fraction: 1.0,
            background_eviction: false,
            spill_path: None,
        })
    }

    struct CountingEvictionConsumer {
        callbacks: AtomicUsize,
    }

    impl MemoryConsumer for CountingEvictionConsumer {
        fn name(&self) -> &str {
            "query-admission-observer"
        }

        fn memory_usage(&self) -> usize {
            self.callbacks.fetch_add(1, Ordering::AcqRel);
            100
        }

        fn eviction_priority(&self) -> u8 {
            self.callbacks.fetch_add(1, Ordering::AcqRel);
            priorities::EXECUTION_BUFFERS
        }

        fn region(&self) -> MemoryRegion {
            MemoryRegion::ExecutionBuffers
        }

        fn evict(&self, _target_bytes: usize) -> usize {
            self.callbacks.fetch_add(1, Ordering::AcqRel);
            0
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            super::super::tiered::StorageTier::InMemory
        }
    }

    struct BlockingPriorityConsumer {
        entered: mpsc::Sender<()>,
        blocked: Mutex<bool>,
        release: Condvar,
        callbacks: AtomicUsize,
    }

    impl BlockingPriorityConsumer {
        fn unblock(&self) {
            *self.blocked.lock() = false;
            self.release.notify_all();
        }
    }

    impl MemoryConsumer for BlockingPriorityConsumer {
        fn name(&self) -> &str {
            "pending-query-admission"
        }

        fn memory_usage(&self) -> usize {
            0
        }

        fn eviction_priority(&self) -> u8 {
            self.callbacks.fetch_add(1, Ordering::AcqRel);
            self.entered.send(()).expect("observer remains live");
            let mut blocked = self.blocked.lock();
            while *blocked {
                self.release.wait(&mut blocked);
            }
            priorities::EXECUTION_BUFFERS
        }

        fn region(&self) -> MemoryRegion {
            MemoryRegion::ExecutionBuffers
        }

        fn evict(&self, _target_bytes: usize) -> usize {
            0
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            super::super::tiered::StorageTier::InMemory
        }
    }

    fn constrained_query_manager() -> Arc<BufferManager> {
        BufferManager::new(BufferManagerConfig {
            budget: 100,
            soft_limit_fraction: 0.5,
            evict_limit_fraction: 0.5,
            hard_limit_fraction: 1.0,
            background_eviction: false,
            spill_path: None,
        })
    }

    #[test]
    fn query_limit_denial_precedes_global_reservation_and_eviction() {
        let manager = constrained_query_manager();
        let existing = manager
            .try_allocate(100, MemoryRegion::ExecutionBuffers)
            .expect("existing global allocation");
        let query = manager.new_query_pool().expect("query pool");
        let _peer = manager.new_query_pool().expect("peer query pool");
        let observer = Arc::new(CountingEvictionConsumer {
            callbacks: AtomicUsize::new(0),
        });
        manager.register_consumer(Arc::clone(&observer) as Arc<dyn MemoryConsumer>);

        assert!(matches!(
            query.try_allocate(51, MemoryRegion::ExecutionBuffers),
            Err(MemoryGrantError::LimitExceeded {
                scope: MemoryLimitScope::Query,
                requested_bytes: 51,
                limit_bytes: 50,
            })
        ));
        assert_eq!(observer.callbacks.load(Ordering::Acquire), 0);
        assert_eq!(query.allocated(), 0);
        assert_eq!(manager.allocated(), 100);
        assert_eq!(existing.size(), 100);
    }

    #[test]
    fn concurrent_pending_query_admission_counts_toward_growth_cap() {
        let manager = constrained_query_manager();
        let existing = manager
            .try_allocate(100, MemoryRegion::ExecutionBuffers)
            .expect("existing global allocation");
        let query = manager.new_query_pool().expect("query pool");
        let _peer = manager.new_query_pool().expect("peer query pool");
        let (entered_sender, entered_receiver) = mpsc::channel();
        let blocker = Arc::new(BlockingPriorityConsumer {
            entered: entered_sender,
            blocked: Mutex::new(true),
            release: Condvar::new(),
            callbacks: AtomicUsize::new(0),
        });
        manager.register_consumer(Arc::clone(&blocker) as Arc<dyn MemoryConsumer>);

        let first_query = Arc::clone(&query);
        let first = std::thread::spawn(move || {
            first_query.try_allocate(30, MemoryRegion::ExecutionBuffers)
        });
        entered_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("first request reaches global eviction");

        let second_query = Arc::clone(&query);
        let (second_sender, second_receiver) = mpsc::channel();
        let second = std::thread::spawn(move || {
            second_sender
                .send(second_query.try_allocate(30, MemoryRegion::ExecutionBuffers))
                .expect("result observer remains live");
        });
        let prompt_second = second_receiver.recv_timeout(Duration::from_secs(2));

        blocker.unblock();
        let first_result = first.join().expect("first allocation thread");
        second.join().expect("second allocation thread");
        let second_result = prompt_second.expect("pending-cap denial must not enter eviction");

        assert!(matches!(
            second_result,
            Err(MemoryGrantError::LimitExceeded {
                scope: MemoryLimitScope::Query,
                requested_bytes: 60,
                limit_bytes: 50,
            })
        ));
        assert!(matches!(
            first_result,
            Err(MemoryGrantError::LimitExceeded {
                scope: MemoryLimitScope::Global,
                ..
            })
        ));
        assert_eq!(blocker.callbacks.load(Ordering::Acquire), 1);
        assert_eq!(query.allocated(), 0);
        drop(existing);
        let retry = query
            .try_allocate(50, MemoryRegion::ExecutionBuffers)
            .expect("failed global request rolls back pending query admission");
        assert_eq!(retry.size(), 50);
    }

    #[test]
    fn callback_panic_rolls_back_a_provisional_manager_allocation() {
        let manager = panic_rollback_manager();
        manager.register_consumer(Arc::new(PanickingEvictionConsumer));

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let manager = Arc::clone(&manager);
            move || {
                let _ = manager.try_allocate(90, MemoryRegion::ExecutionBuffers);
            }
        }));
        assert!(result.is_err());
        assert_eq!(manager.allocated(), 0);
        assert_eq!(
            manager.stats().region_usage(MemoryRegion::ExecutionBuffers),
            0
        );
    }

    #[test]
    fn callback_panic_does_not_leak_a_query_provisional_reservation() {
        let manager = panic_rollback_manager();
        let existing = manager
            .try_allocate(90, MemoryRegion::ExecutionBuffers)
            .expect("initial allocation without consumers");
        let query = manager.new_query_pool().expect("query pool");
        manager.register_consumer(Arc::new(PanickingEvictionConsumer));

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let query = Arc::clone(&query);
            move || {
                let _ = query.try_allocate(20, MemoryRegion::ExecutionBuffers);
            }
        }));
        assert!(result.is_err());
        assert_eq!(query.allocated(), 0);
        assert_eq!(query.pending.load(Ordering::Acquire), 0);
        assert_eq!(manager.allocated(), 90);
        assert_eq!(existing.size(), 90);
        drop(existing);
        let retry = query
            .try_allocate(100, MemoryRegion::ExecutionBuffers)
            .expect("panic rolls back pending local admission");
        assert_eq!(retry.size(), 100);
    }

    struct DropLockCheckingConsumer {
        manager: Weak<BufferManager>,
        dropped_without_registry_lock: Arc<AtomicBool>,
    }

    impl Drop for DropLockCheckingConsumer {
        fn drop(&mut self) {
            let unlocked = self
                .manager
                .upgrade()
                .is_none_or(|manager| manager.consumers.try_write().is_some());
            self.dropped_without_registry_lock
                .store(unlocked, Ordering::Release);
        }
    }

    impl MemoryConsumer for DropLockCheckingConsumer {
        fn name(&self) -> &str {
            "drop-lock-check"
        }

        fn memory_usage(&self) -> usize {
            0
        }

        fn eviction_priority(&self) -> u8 {
            priorities::EXECUTION_BUFFERS
        }

        fn region(&self) -> MemoryRegion {
            MemoryRegion::ExecutionBuffers
        }

        fn evict(&self, _target_bytes: usize) -> usize {
            0
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            super::super::tiered::StorageTier::Uninitialized
        }
    }

    #[test]
    fn unregister_drops_removed_consumer_after_releasing_registry_lock() {
        let manager = BufferManager::with_budget(1000);
        let dropped_without_registry_lock = Arc::new(AtomicBool::new(false));
        manager.register_consumer(Arc::new(DropLockCheckingConsumer {
            manager: Arc::downgrade(&manager),
            dropped_without_registry_lock: Arc::clone(&dropped_without_registry_lock),
        }));

        manager.unregister_consumer("drop-lock-check");
        assert!(dropped_without_registry_lock.load(Ordering::Acquire));
    }

    #[test]
    fn legacy_unregister_preserves_force_ram_for_surviving_scoped_registration() {
        let manager = BufferManager::with_budget(1000);
        let legacy = TestConsumer::new(
            "shared-name",
            10,
            priorities::EXECUTION_BUFFERS,
            MemoryRegion::ExecutionBuffers,
        );
        let scoped = TestConsumer::new(
            "shared-name",
            10,
            priorities::EXECUTION_BUFFERS,
            MemoryRegion::ExecutionBuffers,
        );
        manager.register_consumer(legacy);
        let _registration = manager
            .register_consumer_scoped(scoped)
            .expect("scoped registration");
        manager.mark_force_ram("shared-name");

        manager.unregister_consumer("shared-name");
        assert_eq!(manager.stats().consumer_count, 1);
        assert!(manager.is_force_ram("shared-name"));
    }

    #[test]
    fn test_eviction_ordering() {
        let manager = BufferManager::with_budget(10000);

        // Low priority consumer (evict first)
        let low_priority = TestConsumer::new(
            "low",
            500,
            priorities::SPILL_STAGING,
            MemoryRegion::SpillStaging,
        );

        // High priority consumer (evict last)
        let high_priority = TestConsumer::new(
            "high",
            500,
            priorities::ACTIVE_TRANSACTION,
            MemoryRegion::ExecutionBuffers,
        );

        manager.register_consumer(Arc::clone(&low_priority) as Arc<dyn MemoryConsumer>);
        manager.register_consumer(Arc::clone(&high_priority) as Arc<dyn MemoryConsumer>);

        // Manually set allocated to simulate memory usage
        // (consumers track their own usage separately from manager's allocation tracking)
        manager.allocated.store(1000, Ordering::Relaxed);

        // Request eviction to target 700 (need to free 300 bytes)
        let freed = manager.evict_to_target(700);

        // Low priority should be evicted first (up to half = 250)
        assert!(low_priority.evicted.load(Ordering::Relaxed) > 0);
        assert!(freed > 0);
    }

    #[test]
    fn test_hard_limit_blocking() {
        let config = BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 0.70,
            evict_limit_fraction: 0.85,
            hard_limit_fraction: 0.95,
            background_eviction: false,
            spill_path: None,
        };
        let manager = BufferManager::new(config);

        // Allocate up to hard limit (950 bytes)
        let _g1 = manager.try_allocate(950, MemoryRegion::ExecutionBuffers);

        // This should fail (would exceed hard limit)
        let g2 = manager.try_allocate(100, MemoryRegion::ExecutionBuffers);
        assert!(g2.is_none());
    }

    #[test]
    fn allocation_overflow_fails_without_changing_existing_accounting() {
        let manager = BufferManager::new(BufferManagerConfig {
            budget: usize::MAX,
            soft_limit_fraction: 1.0,
            evict_limit_fraction: 1.0,
            hard_limit_fraction: 1.0,
            background_eviction: false,
            spill_path: None,
        });
        let existing = manager
            .try_allocate(1, MemoryRegion::ExecutionBuffers)
            .expect("one byte fits");

        assert!(
            manager
                .try_allocate(usize::MAX, MemoryRegion::ExecutionBuffers)
                .is_none()
        );
        assert_eq!(manager.allocated(), 1);
        assert_eq!(existing.size(), 1);
    }

    #[test]
    fn query_pool_count_exhaustion_is_specific_and_non_mutating() {
        let manager = BufferManager::with_budget(1000);
        manager
            .active_query_pools
            .store(usize::MAX, Ordering::Release);

        assert!(matches!(
            manager.new_query_pool(),
            Err(MemoryGrantError::QueryPoolCountExhausted)
        ));
        assert_eq!(
            manager.active_query_pools.load(Ordering::Acquire),
            usize::MAX
        );
    }

    #[test]
    fn region_overflow_rolls_back_the_first_stage_global_reservation() {
        let manager = BufferManager::new(BufferManagerConfig {
            budget: usize::MAX,
            soft_limit_fraction: 1.0,
            evict_limit_fraction: 1.0,
            hard_limit_fraction: 1.0,
            background_eviction: false,
            spill_path: None,
        });
        manager.allocated.store(5, Ordering::Release);
        manager.region_allocated[MemoryRegion::ExecutionBuffers.index()]
            .store(usize::MAX, Ordering::Release);

        assert!(matches!(
            manager.reserve_global(1, MemoryRegion::ExecutionBuffers),
            Err(MemoryGrantError::ArithmeticOverflow { .. })
        ));
        assert_eq!(manager.allocated(), 5);
        assert_eq!(
            manager.stats().region_usage(MemoryRegion::ExecutionBuffers),
            usize::MAX
        );
        assert!(!manager.accounting_poisoned.load(Ordering::Acquire));
    }

    #[test]
    fn failed_global_release_restores_the_region_without_undercounting() {
        let manager = BufferManager::with_budget(1000);
        manager.allocated.store(50, Ordering::Release);
        manager.region_allocated[MemoryRegion::ExecutionBuffers.index()]
            .store(100, Ordering::Release);

        assert!(matches!(
            manager.release_accounted(60, MemoryRegion::ExecutionBuffers),
            Err(MemoryGrantError::AccountingUnderflow {
                account: "buffer manager total",
                accounted_bytes: 50,
                release_bytes: 60,
            })
        ));
        assert_eq!(manager.allocated(), 50);
        assert_eq!(
            manager.stats().region_usage(MemoryRegion::ExecutionBuffers),
            100
        );
        assert!(!manager.accounting_poisoned.load(Ordering::Acquire));
    }

    #[test]
    fn query_local_denial_rolls_back_the_first_stage_global_reservation() {
        let manager = BufferManager::new(BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 1.0,
            evict_limit_fraction: 1.0,
            hard_limit_fraction: 1.0,
            background_eviction: false,
            spill_path: None,
        });
        let query = manager.new_query_pool().expect("query pool");
        let _peer = manager.new_query_pool().expect("peer query pool");
        query.allocated.store(500, Ordering::Release);

        assert!(matches!(
            query.reserve_growth(1, MemoryRegion::ExecutionBuffers),
            Err(MemoryGrantError::LimitExceeded {
                scope: MemoryLimitScope::Query,
                requested_bytes: 501,
                limit_bytes: 500,
            })
        ));
        assert_eq!(query.allocated(), 500);
        assert_eq!(manager.allocated(), 0);
        assert_eq!(
            manager.stats().region_usage(MemoryRegion::ExecutionBuffers),
            0
        );
    }

    #[test]
    fn invalid_query_release_does_not_touch_the_global_account() {
        let manager = BufferManager::with_budget(1000);
        let query = manager.new_query_pool().expect("query pool");
        query.allocated.store(10, Ordering::Release);
        manager.allocated.store(50, Ordering::Release);
        manager.region_allocated[MemoryRegion::ExecutionBuffers.index()]
            .store(50, Ordering::Release);

        assert!(matches!(
            query.release_accounted(20, MemoryRegion::ExecutionBuffers),
            Err(MemoryGrantError::AccountingUnderflow {
                account: "query memory pool",
                accounted_bytes: 10,
                release_bytes: 20,
            })
        ));
        assert_eq!(query.allocated(), 10);
        assert_eq!(manager.allocated(), 50);
        assert_eq!(
            manager.stats().region_usage(MemoryRegion::ExecutionBuffers),
            50
        );
    }

    #[test]
    fn failed_global_stage_restores_query_local_accounting() {
        let manager = BufferManager::with_budget(1000);
        let query = manager.new_query_pool().expect("query pool");
        query.allocated.store(100, Ordering::Release);
        manager.allocated.store(50, Ordering::Release);
        manager.region_allocated[MemoryRegion::ExecutionBuffers.index()]
            .store(100, Ordering::Release);

        assert!(matches!(
            query.release_accounted(60, MemoryRegion::ExecutionBuffers),
            Err(MemoryGrantError::AccountingUnderflow {
                account: "buffer manager total",
                accounted_bytes: 50,
                release_bytes: 60,
            })
        ));
        assert_eq!(query.allocated(), 100);
        assert_eq!(manager.allocated(), 50);
        assert_eq!(
            manager.stats().region_usage(MemoryRegion::ExecutionBuffers),
            100
        );
        assert!(!query.accounting_poisoned.load(Ordering::Acquire));
    }

    #[test]
    fn zero_sized_query_grants_do_not_change_either_account() {
        let manager = BufferManager::with_budget(1000);
        let query = manager.new_query_pool().expect("query pool");

        let mut grant = query
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .expect("zero-sized grant");
        assert!(grant.is_empty());
        assert_eq!(query.allocated(), 0);
        assert_eq!(manager.allocated(), 0);
        grant.try_resize(0).expect("zero-sized no-op resize");
        drop(grant);
        assert_eq!(query.allocated(), 0);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn concurrent_global_grants_never_overcommit_the_hard_limit() {
        const WORKERS: usize = 32;
        const GRANT_BYTES: usize = 100;
        let manager = BufferManager::new(BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 1.0,
            evict_limit_fraction: 1.0,
            hard_limit_fraction: 1.0,
            background_eviction: false,
            spill_path: None,
        });
        let start = Arc::new(Barrier::new(WORKERS + 1));
        let release = Arc::new(Barrier::new(WORKERS + 1));
        let (result_sender, result_receiver) = mpsc::channel();
        let mut workers = Vec::with_capacity(WORKERS);

        for _ in 0..WORKERS {
            let manager = Arc::clone(&manager);
            let start = Arc::clone(&start);
            let release = Arc::clone(&release);
            let result_sender = result_sender.clone();
            workers.push(std::thread::spawn(move || {
                start.wait();
                let grant = manager.try_allocate(GRANT_BYTES, MemoryRegion::ExecutionBuffers);
                result_sender
                    .send(grant.is_some())
                    .expect("result receiver remains live");
                release.wait();
                drop(grant);
            }));
        }
        drop(result_sender);

        start.wait();
        let successful = (0..WORKERS)
            .filter(|_| result_receiver.recv().expect("allocation result"))
            .count();
        assert_eq!(successful, 10);
        assert_eq!(manager.allocated(), successful * GRANT_BYTES);
        assert!(manager.allocated() <= 1000);

        release.wait();
        for worker in workers {
            worker.join().expect("allocation worker");
        }
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn committed_query_bytes_never_publish_before_their_global_reservation() {
        let manager = BufferManager::new(BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 1.0,
            evict_limit_fraction: 1.0,
            hard_limit_fraction: 1.0,
            background_eviction: false,
            spill_path: None,
        });
        let query = manager.new_query_pool().expect("query pool");
        let running = Arc::new(AtomicBool::new(true));
        let observer = {
            let manager = Arc::clone(&manager);
            let query = Arc::clone(&query);
            let running = Arc::clone(&running);
            std::thread::spawn(move || {
                while running.load(Ordering::Acquire) {
                    let _transition = query.local_transition.lock();
                    assert!(query.allocated() <= manager.allocated());
                    std::hint::spin_loop();
                }
            })
        };

        for _ in 0..10_000 {
            let grant = query
                .try_allocate(1, MemoryRegion::ExecutionBuffers)
                .expect("single-byte query grant");
            assert_eq!(grant.size(), 1);
            drop(grant);
        }
        running.store(false, Ordering::Release);
        observer.join().expect("accounting observer");
        assert_eq!(query.allocated(), 0);
        assert_eq!(query.pending.load(Ordering::Acquire), 0);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn concurrent_query_pools_receive_a_dynamic_growth_cap() {
        let manager = BufferManager::new(BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 1.0,
            evict_limit_fraction: 1.0,
            hard_limit_fraction: 1.0,
            background_eviction: false,
            spill_path: None,
        });
        let first = manager.new_query_pool().expect("first query pool");
        assert_eq!(first.growth_limit(), 1000);

        let second = manager.new_query_pool().expect("second query pool");
        assert_eq!(first.growth_limit(), 500);
        assert_eq!(second.growth_limit(), 500);

        let first_grant = first
            .try_allocate(500, MemoryRegion::ExecutionBuffers)
            .expect("first query growth cap");
        let second_grant = second
            .try_allocate(500, MemoryRegion::ExecutionBuffers)
            .expect("second query growth cap");
        assert_eq!(manager.allocated(), 1000);
        drop(second_grant);
        assert_eq!(manager.allocated(), 500);
        assert!(matches!(
            first.try_allocate(1, MemoryRegion::ExecutionBuffers),
            Err(MemoryGrantError::LimitExceeded {
                scope: MemoryLimitScope::Query,
                requested_bytes: 501,
                limit_bytes: 500,
            })
        ));

        drop(first_grant);
        drop(second);
        assert_eq!(first.growth_limit(), 1000);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn denied_query_grant_resize_preserves_both_accounts_and_prior_size() {
        let manager = BufferManager::new(BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 1.0,
            evict_limit_fraction: 1.0,
            hard_limit_fraction: 1.0,
            background_eviction: false,
            spill_path: None,
        });
        let query = manager.new_query_pool().expect("query pool");
        let _peer = manager.new_query_pool().expect("peer query pool");
        let mut grant = query
            .try_allocate(400, MemoryRegion::ExecutionBuffers)
            .expect("initial grant");

        assert!(matches!(
            grant.try_resize(600),
            Err(MemoryGrantError::LimitExceeded {
                scope: MemoryLimitScope::Query,
                requested_bytes: 600,
                limit_bytes: 500,
            })
        ));
        assert_eq!(grant.size(), 400);
        assert_eq!(query.allocated(), 400);
        assert_eq!(manager.allocated(), 400);
    }

    #[test]
    fn cross_query_grant_merge_is_rejected_without_account_corruption() {
        let manager = BufferManager::new(BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 1.0,
            evict_limit_fraction: 1.0,
            hard_limit_fraction: 1.0,
            background_eviction: false,
            spill_path: None,
        });
        let first_query = manager.new_query_pool().expect("first query");
        let second_query = manager.new_query_pool().expect("second query");
        let mut first_grant = first_query
            .try_allocate(100, MemoryRegion::ExecutionBuffers)
            .expect("first grant");
        let second_grant = second_query
            .try_allocate(200, MemoryRegion::ExecutionBuffers)
            .expect("second grant");

        let second_grant = first_grant
            .try_merge(second_grant)
            .expect_err("different query accounts cannot merge");
        assert_eq!(first_grant.size(), 100);
        assert_eq!(second_grant.size(), 200);
        assert_eq!(first_query.allocated(), 100);
        assert_eq!(second_query.allocated(), 200);
        assert_eq!(manager.allocated(), 300);

        drop(first_grant);
        drop(second_grant);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn query_grant_cannot_discard_its_releaser_token() {
        let manager = BufferManager::new(BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 1.0,
            evict_limit_fraction: 1.0,
            hard_limit_fraction: 1.0,
            background_eviction: false,
            spill_path: None,
        });
        let query = manager.new_query_pool().expect("query pool");
        let grant = query
            .try_allocate(100, MemoryRegion::ExecutionBuffers)
            .expect("query grant");

        let grant = grant
            .try_consume()
            .expect_err("a byte count alone cannot release a query account");
        assert_eq!(grant.size(), 100);
        assert_eq!(query.allocated(), 100);
        assert_eq!(manager.allocated(), 100);

        drop(grant);
        assert_eq!(query.allocated(), 0);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn detached_grant_retains_accounting_until_release_or_reattach() {
        let manager = BufferManager::with_budget(1000);
        let grant = manager
            .try_allocate(100, MemoryRegion::ExecutionBuffers)
            .expect("managed grant");

        let detached = grant.detach();
        assert_eq!(detached.size(), 100);
        assert_eq!(manager.allocated(), 100);
        let grant = detached.reattach();
        assert_eq!(grant.size(), 100);
        assert_eq!(manager.allocated(), 100);

        let detached = grant.detach();
        detached.release();
        assert_eq!(manager.allocated(), 0);
        assert_eq!(
            manager.stats().region_usage(MemoryRegion::ExecutionBuffers),
            0
        );
    }

    #[test]
    fn test_available_memory() {
        let manager = BufferManager::with_budget(1000);

        assert_eq!(manager.available(), 1000);

        let _g = manager.try_allocate(300, MemoryRegion::ExecutionBuffers);
        assert_eq!(manager.available(), 700);
    }

    // --- Spill-aware test consumer ---

    struct SpillableConsumer {
        name: String,
        usage: AtomicUsize,
        priority: u8,
        region: MemoryRegion,
        evicted: AtomicUsize,
        spilled: AtomicUsize,
        spillable: bool,
        evict_returns_zero: bool,
    }

    impl SpillableConsumer {
        fn new(
            name: &str,
            usage: usize,
            priority: u8,
            region: MemoryRegion,
            spillable: bool,
        ) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                usage: AtomicUsize::new(usage),
                priority,
                region,
                evicted: AtomicUsize::new(0),
                spilled: AtomicUsize::new(0),
                spillable,
                evict_returns_zero: false,
            })
        }

        fn new_evict_fails(
            name: &str,
            usage: usize,
            priority: u8,
            region: MemoryRegion,
            spillable: bool,
        ) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                usage: AtomicUsize::new(usage),
                priority,
                region,
                evicted: AtomicUsize::new(0),
                spilled: AtomicUsize::new(0),
                spillable,
                evict_returns_zero: true,
            })
        }
    }

    impl MemoryConsumer for SpillableConsumer {
        fn name(&self) -> &str {
            &self.name
        }

        fn memory_usage(&self) -> usize {
            self.usage.load(Ordering::Relaxed)
        }

        fn eviction_priority(&self) -> u8 {
            self.priority
        }

        fn region(&self) -> MemoryRegion {
            self.region
        }

        fn evict(&self, target_bytes: usize) -> usize {
            if self.evict_returns_zero {
                return 0;
            }
            let current = self.usage.load(Ordering::Relaxed);
            let to_evict = target_bytes.min(current);
            self.usage.fetch_sub(to_evict, Ordering::Relaxed);
            self.evicted.fetch_add(to_evict, Ordering::Relaxed);
            to_evict
        }

        fn can_spill(&self) -> bool {
            self.spillable
        }

        fn spill(
            &self,
            target_bytes: usize,
        ) -> Result<usize, crate::memory::buffer::consumer::SpillError> {
            if !self.spillable {
                return Err(crate::memory::buffer::consumer::SpillError::NotSupported);
            }
            let current = self.usage.load(Ordering::Relaxed);
            let to_spill = target_bytes.min(current);
            self.usage.fetch_sub(to_spill, Ordering::Relaxed);
            self.spilled.fetch_add(to_spill, Ordering::Relaxed);
            Ok(to_spill)
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            if self.spilled.load(Ordering::Relaxed) > 0 {
                super::super::tiered::StorageTier::OnDisk
            } else if self.memory_usage() == 0 {
                super::super::tiered::StorageTier::Uninitialized
            } else {
                super::super::tiered::StorageTier::InMemory
            }
        }
    }

    struct BlockingSpillConsumer {
        name: &'static str,
        spill_started: mpsc::SyncSender<()>,
        blocked: Mutex<bool>,
        release: Condvar,
        spill_count: AtomicUsize,
    }

    impl BlockingSpillConsumer {
        fn unblock(&self) {
            *self.blocked.lock() = false;
            self.release.notify_all();
        }
    }

    struct ReentrantForceRamConsumer {
        manager: Weak<BufferManager>,
    }

    struct CrossManagerPinConsumer {
        name: &'static str,
        other_name: &'static str,
        other_manager: Weak<BufferManager>,
        callbacks_ready: Arc<Barrier>,
    }

    struct CrossRegistrationCloseConsumer {
        name: &'static str,
        other_registration: Mutex<Option<ConsumerRegistration>>,
        callbacks_ready: Arc<Barrier>,
        observed_nonquiescent: AtomicBool,
    }

    impl MemoryConsumer for CrossRegistrationCloseConsumer {
        fn name(&self) -> &str {
            self.name
        }

        fn memory_usage(&self) -> usize {
            100
        }

        fn eviction_priority(&self) -> u8 {
            priorities::EXECUTION_BUFFERS
        }

        fn region(&self) -> MemoryRegion {
            MemoryRegion::ExecutionBuffers
        }

        fn evict(&self, _target_bytes: usize) -> usize {
            0
        }

        fn can_spill(&self) -> bool {
            true
        }

        fn spill(
            &self,
            _target_bytes: usize,
        ) -> Result<usize, crate::memory::buffer::consumer::SpillError> {
            self.callbacks_ready.wait();
            let registration = self
                .other_registration
                .lock()
                .take()
                .expect("peer registration is present");
            self.observed_nonquiescent
                .store(registration.close().is_err(), Ordering::Release);
            Ok(0)
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            super::super::tiered::StorageTier::InMemory
        }
    }

    impl MemoryConsumer for CrossManagerPinConsumer {
        fn name(&self) -> &str {
            self.name
        }

        fn memory_usage(&self) -> usize {
            100
        }

        fn eviction_priority(&self) -> u8 {
            priorities::EXECUTION_BUFFERS
        }

        fn region(&self) -> MemoryRegion {
            MemoryRegion::ExecutionBuffers
        }

        fn evict(&self, _target_bytes: usize) -> usize {
            0
        }

        fn can_spill(&self) -> bool {
            true
        }

        fn spill(
            &self,
            _target_bytes: usize,
        ) -> Result<usize, crate::memory::buffer::consumer::SpillError> {
            self.callbacks_ready.wait();
            self.other_manager
                .upgrade()
                .expect("other manager remains live")
                .mark_force_ram(self.other_name);
            Ok(0)
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            super::super::tiered::StorageTier::InMemory
        }
    }

    #[test]
    fn cross_manager_callbacks_can_cross_pin_without_wait_cycle() {
        let first_manager = BufferManager::with_budget(1000);
        let second_manager = BufferManager::with_budget(1000);
        let callbacks_ready = Arc::new(Barrier::new(2));
        first_manager.register_consumer(Arc::new(CrossManagerPinConsumer {
            name: "cross-pin-a",
            other_name: "cross-pin-b",
            other_manager: Arc::downgrade(&second_manager),
            callbacks_ready: Arc::clone(&callbacks_ready),
        }));
        second_manager.register_consumer(Arc::new(CrossManagerPinConsumer {
            name: "cross-pin-b",
            other_name: "cross-pin-a",
            other_manager: Arc::downgrade(&first_manager),
            callbacks_ready,
        }));

        let first = {
            let manager = Arc::clone(&first_manager);
            std::thread::spawn(move || manager.spill_all())
        };
        let second = {
            let manager = Arc::clone(&second_manager);
            std::thread::spawn(move || manager.spill_all())
        };

        assert_eq!(first.join().expect("first cross-pin callback"), 0);
        assert_eq!(second.join().expect("second cross-pin callback"), 0);
        assert!(first_manager.is_force_ram("cross-pin-a"));
        assert!(second_manager.is_force_ram("cross-pin-b"));
    }

    #[test]
    fn cross_callback_registration_close_is_nonblocking_and_explicitly_nonquiescent() {
        let first_manager = BufferManager::with_budget(1000);
        let second_manager = BufferManager::with_budget(1000);
        let callbacks_ready = Arc::new(Barrier::new(2));
        let first_consumer = Arc::new(CrossRegistrationCloseConsumer {
            name: "cross-close-a",
            other_registration: Mutex::new(None),
            callbacks_ready: Arc::clone(&callbacks_ready),
            observed_nonquiescent: AtomicBool::new(false),
        });
        let second_consumer = Arc::new(CrossRegistrationCloseConsumer {
            name: "cross-close-b",
            other_registration: Mutex::new(None),
            callbacks_ready,
            observed_nonquiescent: AtomicBool::new(false),
        });
        let first_registration = first_manager
            .register_consumer_scoped(Arc::clone(&first_consumer) as Arc<dyn MemoryConsumer>)
            .expect("first registration");
        let second_registration = second_manager
            .register_consumer_scoped(Arc::clone(&second_consumer) as Arc<dyn MemoryConsumer>)
            .expect("second registration");
        *first_consumer.other_registration.lock() = Some(second_registration);
        *second_consumer.other_registration.lock() = Some(first_registration);

        let first = {
            let manager = Arc::clone(&first_manager);
            std::thread::spawn(move || manager.spill_all())
        };
        let second = {
            let manager = Arc::clone(&second_manager);
            std::thread::spawn(move || manager.spill_all())
        };

        assert_eq!(first.join().expect("first cross-close callback"), 0);
        assert_eq!(second.join().expect("second cross-close callback"), 0);
        assert!(first_consumer.observed_nonquiescent.load(Ordering::Acquire));
        assert!(
            second_consumer
                .observed_nonquiescent
                .load(Ordering::Acquire)
        );
        assert_eq!(first_manager.stats().consumer_count, 0);
        assert_eq!(second_manager.stats().consumer_count, 0);
    }

    impl MemoryConsumer for ReentrantForceRamConsumer {
        fn name(&self) -> &str {
            "reentrant-force-ram"
        }

        fn memory_usage(&self) -> usize {
            100
        }

        fn eviction_priority(&self) -> u8 {
            priorities::EXECUTION_BUFFERS
        }

        fn region(&self) -> MemoryRegion {
            MemoryRegion::ExecutionBuffers
        }

        fn evict(&self, _target_bytes: usize) -> usize {
            0
        }

        fn can_spill(&self) -> bool {
            true
        }

        fn spill(
            &self,
            _target_bytes: usize,
        ) -> Result<usize, crate::memory::buffer::consumer::SpillError> {
            let manager = self.manager.upgrade().expect("manager remains live");
            let callback_is_unlocked = manager.force_ram_consumers.try_lock().is_some();
            assert!(
                callback_is_unlocked,
                "consumer callback ran while the ForceRam lock was held"
            );
            manager.mark_force_ram(self.name());
            Ok(0)
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            super::super::tiered::StorageTier::InMemory
        }
    }

    #[test]
    fn spill_callback_can_reentrantly_pin_itself_force_ram() {
        let manager = BufferManager::with_budget(1000);
        manager.register_consumer(Arc::new(ReentrantForceRamConsumer {
            manager: Arc::downgrade(&manager),
        }));

        assert_eq!(manager.spill_all(), 0);
        assert!(manager.is_force_ram("reentrant-force-ram"));
    }

    struct ConcurrentSelfPinConsumer {
        manager: Weak<BufferManager>,
        both_callbacks_started: Arc<Barrier>,
    }

    impl MemoryConsumer for ConcurrentSelfPinConsumer {
        fn name(&self) -> &str {
            "concurrent-self-pin"
        }

        fn memory_usage(&self) -> usize {
            100
        }

        fn eviction_priority(&self) -> u8 {
            priorities::EXECUTION_BUFFERS
        }

        fn region(&self) -> MemoryRegion {
            MemoryRegion::ExecutionBuffers
        }

        fn evict(&self, _target_bytes: usize) -> usize {
            0
        }

        fn can_spill(&self) -> bool {
            true
        }

        fn spill(
            &self,
            _target_bytes: usize,
        ) -> Result<usize, crate::memory::buffer::consumer::SpillError> {
            self.both_callbacks_started.wait();
            self.manager
                .upgrade()
                .expect("manager remains live")
                .mark_force_ram(self.name());
            Ok(0)
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            super::super::tiered::StorageTier::InMemory
        }
    }

    #[test]
    fn concurrent_same_name_callbacks_can_both_self_pin_without_deadlock() {
        let manager = BufferManager::with_budget(1000);
        manager.register_consumer(Arc::new(ConcurrentSelfPinConsumer {
            manager: Arc::downgrade(&manager),
            both_callbacks_started: Arc::new(Barrier::new(2)),
        }));

        let first_manager = Arc::clone(&manager);
        let first =
            std::thread::spawn(move || first_manager.spill_consumer_by_name("concurrent-self-pin"));
        let second_manager = Arc::clone(&manager);
        let second = std::thread::spawn(move || {
            second_manager.spill_consumer_by_name("concurrent-self-pin")
        });

        assert_eq!(first.join().expect("first self-pin callback"), 0);
        assert_eq!(second.join().expect("second self-pin callback"), 0);
        assert!(manager.is_force_ram("concurrent-self-pin"));
    }

    impl MemoryConsumer for BlockingSpillConsumer {
        fn name(&self) -> &str {
            self.name
        }

        fn memory_usage(&self) -> usize {
            100
        }

        fn eviction_priority(&self) -> u8 {
            priorities::EXECUTION_BUFFERS
        }

        fn region(&self) -> MemoryRegion {
            MemoryRegion::ExecutionBuffers
        }

        fn evict(&self, _target_bytes: usize) -> usize {
            0
        }

        fn can_spill(&self) -> bool {
            true
        }

        fn spill(
            &self,
            _target_bytes: usize,
        ) -> Result<usize, crate::memory::buffer::consumer::SpillError> {
            self.spill_count.fetch_add(1, Ordering::AcqRel);
            self.spill_started
                .send(())
                .expect("spill-start observer remains live");
            let mut blocked = self.blocked.lock();
            while *blocked {
                self.release.wait(&mut blocked);
            }
            Ok(100)
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            super::super::tiered::StorageTier::InMemory
        }
    }

    struct ObservationalPinConsumer {
        manager: Weak<BufferManager>,
        name: &'static str,
    }

    impl MemoryConsumer for ObservationalPinConsumer {
        fn name(&self) -> &str {
            self.name
        }

        fn memory_usage(&self) -> usize {
            0
        }

        fn eviction_priority(&self) -> u8 {
            priorities::EXECUTION_BUFFERS
        }

        fn region(&self) -> MemoryRegion {
            MemoryRegion::ExecutionBuffers
        }

        fn evict(&self, _target_bytes: usize) -> usize {
            0
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            self.manager
                .upgrade()
                .expect("manager remains live")
                .mark_force_ram(self.name);
            super::super::tiered::StorageTier::InMemory
        }
    }

    struct RegistrationNamePinConsumer {
        target_manager: Weak<BufferManager>,
        target_name: &'static str,
    }

    impl MemoryConsumer for RegistrationNamePinConsumer {
        fn name(&self) -> &str {
            self.target_manager
                .upgrade()
                .expect("target manager remains live")
                .mark_force_ram(self.target_name);
            "registration-name-pin"
        }

        fn memory_usage(&self) -> usize {
            0
        }

        fn eviction_priority(&self) -> u8 {
            priorities::EXECUTION_BUFFERS
        }

        fn region(&self) -> MemoryRegion {
            MemoryRegion::ExecutionBuffers
        }

        fn evict(&self, _target_bytes: usize) -> usize {
            0
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            super::super::tiered::StorageTier::Uninitialized
        }
    }

    #[test]
    fn observational_callback_cross_pin_never_waits_for_destructive_callback() {
        let manager = BufferManager::with_budget(10_000);
        let (spill_started_sender, spill_started_receiver) = mpsc::sync_channel(1);
        let blocker = Arc::new(BlockingSpillConsumer {
            name: "section:observational-pin",
            spill_started: spill_started_sender,
            blocked: Mutex::new(true),
            release: Condvar::new(),
            spill_count: AtomicUsize::new(0),
        });
        manager.register_consumer(Arc::clone(&blocker) as Arc<dyn MemoryConsumer>);
        manager.register_consumer(Arc::new(ObservationalPinConsumer {
            manager: Arc::downgrade(&manager),
            name: "section:observational-pin",
        }));

        let spill_manager = Arc::clone(&manager);
        let spill_thread = std::thread::spawn(move || {
            spill_manager.spill_consumer_by_name("section:observational-pin")
        });
        spill_started_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("destructive callback starts");

        let (snapshot_sender, snapshot_receiver) = mpsc::channel();
        let snapshot_manager = Arc::clone(&manager);
        let snapshot_thread = std::thread::spawn(move || {
            snapshot_sender
                .send(snapshot_manager.snapshot_consumer_tiers())
                .expect("snapshot observer remains live");
        });
        let tiers = snapshot_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("observational callback must take nonblocking pin path");
        assert_eq!(tiers.len(), 2);
        assert!(manager.is_force_ram("section:observational-pin"));

        blocker.unblock();
        snapshot_thread.join().expect("snapshot thread");
        assert_eq!(spill_thread.join().expect("spill thread"), 100);
    }

    #[test]
    fn registration_name_callback_cross_pin_is_nonblocking() {
        let target_manager = BufferManager::with_budget(10_000);
        let (spill_started_sender, spill_started_receiver) = mpsc::sync_channel(1);
        let blocker = Arc::new(BlockingSpillConsumer {
            name: "registration-pin-target",
            spill_started: spill_started_sender,
            blocked: Mutex::new(true),
            release: Condvar::new(),
            spill_count: AtomicUsize::new(0),
        });
        target_manager.register_consumer(Arc::clone(&blocker) as Arc<dyn MemoryConsumer>);
        let spill_manager = Arc::clone(&target_manager);
        let spill_thread = std::thread::spawn(move || {
            spill_manager.spill_consumer_by_name("registration-pin-target")
        });
        spill_started_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("destructive callback starts");

        let registration_manager = BufferManager::with_budget(1000);
        let (registered_sender, registered_receiver) = mpsc::channel();
        let registration_thread = {
            let registration_manager = Arc::clone(&registration_manager);
            let target_manager = Arc::downgrade(&target_manager);
            std::thread::spawn(move || {
                registration_manager.register_consumer(Arc::new(RegistrationNamePinConsumer {
                    target_manager,
                    target_name: "registration-pin-target",
                }));
                registered_sender
                    .send(())
                    .expect("registration observer remains live");
            })
        };
        registered_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("registration-time name callback must not wait");
        assert!(target_manager.is_force_ram("registration-pin-target"));

        blocker.unblock();
        registration_thread.join().expect("registration thread");
        assert_eq!(spill_thread.join().expect("spill thread"), 100);
        assert_eq!(registration_manager.stats().consumer_count, 1);
    }

    #[test]
    fn force_ram_pin_return_linearizes_after_an_in_flight_spill_cycle() {
        let manager = BufferManager::with_budget(10_000);
        let (spill_started_sender, spill_started_receiver) = mpsc::sync_channel(1);
        let blocker = Arc::new(BlockingSpillConsumer {
            name: "pin-target",
            spill_started: spill_started_sender,
            blocked: Mutex::new(true),
            release: Condvar::new(),
            spill_count: AtomicUsize::new(0),
        });
        let target = SpillableConsumer::new_evict_fails(
            "pin-target",
            100,
            priorities::EXECUTION_BUFFERS,
            MemoryRegion::ExecutionBuffers,
            true,
        );
        manager.register_consumer(Arc::clone(&blocker) as Arc<dyn MemoryConsumer>);
        manager.register_consumer(Arc::clone(&target) as Arc<dyn MemoryConsumer>);

        let spill_manager = Arc::clone(&manager);
        let spill_thread = std::thread::spawn(move || spill_manager.spill_all());
        spill_started_receiver
            .recv()
            .expect("first spill callback starts");

        let pin_manager = Arc::clone(&manager);
        let (pin_calling_sender, pin_calling_receiver) = mpsc::sync_channel(1);
        let (pin_returned_sender, pin_returned_receiver) = mpsc::sync_channel(1);
        let pin_thread = std::thread::spawn(move || {
            pin_calling_sender.send(()).expect("pin caller observed");
            pin_manager.mark_force_ram("pin-target");
            pin_returned_sender.send(()).expect("pin return observed");
        });
        pin_calling_receiver.recv().expect("pin call begins");
        while !manager.is_force_ram("pin-target") {
            std::thread::yield_now();
        }
        assert!(pin_returned_receiver.try_recv().is_err());

        blocker.unblock();
        assert_eq!(spill_thread.join().expect("spill cycle"), 100);
        pin_returned_receiver.recv().expect("pin call returns");
        pin_thread.join().expect("pin thread");
        assert_eq!(target.spilled.load(Ordering::Acquire), 0);

        assert_eq!(manager.spill_all(), 0);
        assert_eq!(target.spilled.load(Ordering::Acquire), 0);
    }

    #[test]
    fn removed_snapshot_consumer_cannot_start_a_later_callback() {
        let manager = BufferManager::with_budget(10_000);
        let (spill_started_sender, spill_started_receiver) = mpsc::sync_channel(1);
        let blocker = Arc::new(BlockingSpillConsumer {
            name: "snapshot-blocker",
            spill_started: spill_started_sender,
            blocked: Mutex::new(true),
            release: Condvar::new(),
            spill_count: AtomicUsize::new(0),
        });
        let target = SpillableConsumer::new_evict_fails(
            "snapshot-target",
            100,
            priorities::EXECUTION_BUFFERS,
            MemoryRegion::ExecutionBuffers,
            true,
        );
        manager.register_consumer(Arc::clone(&blocker) as Arc<dyn MemoryConsumer>);
        let target_registration = manager
            .register_consumer_scoped(Arc::clone(&target) as Arc<dyn MemoryConsumer>)
            .expect("target registration");

        let spill_manager = Arc::clone(&manager);
        let spill_thread = std::thread::spawn(move || spill_manager.spill_all());
        spill_started_receiver
            .recv()
            .expect("first spill callback starts");
        drop(target_registration);
        blocker.unblock();

        assert_eq!(spill_thread.join().expect("spill cycle"), 100);
        assert_eq!(target.spilled.load(Ordering::Acquire), 0);
    }

    #[test]
    fn explicit_registration_close_waits_for_in_flight_callback() {
        let manager = BufferManager::with_budget(10_000);
        let (spill_started_sender, spill_started_receiver) = mpsc::sync_channel(1);
        let consumer = Arc::new(BlockingSpillConsumer {
            name: "quiescent-close",
            spill_started: spill_started_sender,
            blocked: Mutex::new(true),
            release: Condvar::new(),
            spill_count: AtomicUsize::new(0),
        });
        let registration = manager
            .register_consumer_scoped(Arc::clone(&consumer) as Arc<dyn MemoryConsumer>)
            .expect("scoped registration");
        let spill_manager = Arc::clone(&manager);
        let spill_thread = std::thread::spawn(move || spill_manager.spill_all());
        spill_started_receiver
            .recv()
            .expect("spill callback starts");

        let (close_calling_sender, close_calling_receiver) = mpsc::sync_channel(1);
        let (close_returned_sender, close_returned_receiver) = mpsc::sync_channel(1);
        let close_thread = std::thread::spawn(move || {
            close_calling_sender
                .send(())
                .expect("close caller observed");
            registration
                .close()
                .expect("external close waits to quiescence");
            close_returned_sender
                .send(())
                .expect("close observer remains live");
        });
        close_calling_receiver.recv().expect("close call begins");
        while manager.stats().consumer_count != 0 {
            std::thread::yield_now();
        }
        assert!(close_returned_receiver.try_recv().is_err());

        consumer.unblock();
        assert_eq!(spill_thread.join().expect("spill callback"), 100);
        close_returned_receiver.recv().expect("close returns");
        close_thread.join().expect("close thread");
        assert_eq!(manager.stats().consumer_count, 0);
    }

    #[test]
    fn test_spill_all_calls_spillable_consumers() {
        let manager = BufferManager::with_budget(10000);
        let spillable = SpillableConsumer::new(
            "spillable",
            500,
            priorities::QUERY_CACHE,
            MemoryRegion::ExecutionBuffers,
            true,
        );
        let non_spillable = SpillableConsumer::new(
            "non_spillable",
            500,
            priorities::QUERY_CACHE,
            MemoryRegion::ExecutionBuffers,
            false,
        );
        manager.register_consumer(Arc::clone(&spillable) as Arc<dyn MemoryConsumer>);
        manager.register_consumer(Arc::clone(&non_spillable) as Arc<dyn MemoryConsumer>);

        let freed = manager.spill_all();
        assert_eq!(freed, 500);
        assert_eq!(spillable.spilled.load(Ordering::Relaxed), 500);
        assert_eq!(non_spillable.spilled.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_spill_all_skips_non_spillable() {
        let manager = BufferManager::with_budget(10000);
        let consumer = SpillableConsumer::new(
            "no_spill",
            1000,
            priorities::INDEX_BUFFERS,
            MemoryRegion::IndexBuffers,
            false,
        );
        manager.register_consumer(Arc::clone(&consumer) as Arc<dyn MemoryConsumer>);

        assert_eq!(manager.spill_all(), 0);
        assert_eq!(consumer.memory_usage(), 1000);
    }

    #[test]
    fn test_eviction_falls_back_to_spill() {
        let manager = BufferManager::with_budget(10000);
        let consumer = SpillableConsumer::new_evict_fails(
            "spill_fallback",
            1000,
            priorities::QUERY_CACHE,
            MemoryRegion::ExecutionBuffers,
            true,
        );
        manager.register_consumer(Arc::clone(&consumer) as Arc<dyn MemoryConsumer>);
        manager.allocated.store(2000, Ordering::Relaxed);

        let freed = manager.evict_to_target(1500);
        assert_eq!(consumer.evicted.load(Ordering::Relaxed), 0);
        assert!(consumer.spilled.load(Ordering::Relaxed) > 0);
        assert!(freed > 0);
    }

    struct UntrustedFreedByteConsumer;

    impl MemoryConsumer for UntrustedFreedByteConsumer {
        fn name(&self) -> &str {
            "untrusted-freed-byte-report"
        }

        fn memory_usage(&self) -> usize {
            100
        }

        fn eviction_priority(&self) -> u8 {
            priorities::EXECUTION_BUFFERS
        }

        fn region(&self) -> MemoryRegion {
            MemoryRegion::ExecutionBuffers
        }

        fn evict(&self, _target_bytes: usize) -> usize {
            usize::MAX
        }

        fn can_spill(&self) -> bool {
            true
        }

        fn spill(
            &self,
            _target_bytes: usize,
        ) -> Result<usize, crate::memory::buffer::consumer::SpillError> {
            Ok(usize::MAX)
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            super::super::tiered::StorageTier::InMemory
        }
    }

    #[test]
    fn eviction_clamps_untrusted_consumer_freed_byte_reports() {
        let manager = BufferManager::with_budget(10_000);
        manager.register_consumer(Arc::new(UntrustedFreedByteConsumer));
        manager.register_consumer(Arc::new(UntrustedFreedByteConsumer));
        manager.allocated.store(100, Ordering::Release);

        assert_eq!(manager.evict_to_target(50), 50);
        assert_eq!(
            manager.spill_consumer_by_name("untrusted-freed-byte-report"),
            usize::MAX
        );
    }

    #[test]
    fn test_eviction_no_spill_when_sufficient() {
        let manager = BufferManager::with_budget(10000);
        let consumer = SpillableConsumer::new(
            "eviction_enough",
            1000,
            priorities::QUERY_CACHE,
            MemoryRegion::ExecutionBuffers,
            true,
        );
        manager.register_consumer(Arc::clone(&consumer) as Arc<dyn MemoryConsumer>);
        manager.allocated.store(1200, Ordering::Relaxed);

        let freed = manager.evict_to_target(1000);
        assert_eq!(freed, 200);
        assert_eq!(consumer.spilled.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_eviction_spill_skips_non_spillable() {
        let manager = BufferManager::with_budget(10000);
        let consumer = SpillableConsumer::new_evict_fails(
            "no_spill",
            1000,
            priorities::QUERY_CACHE,
            MemoryRegion::ExecutionBuffers,
            false,
        );
        manager.register_consumer(Arc::clone(&consumer) as Arc<dyn MemoryConsumer>);
        manager.allocated.store(2000, Ordering::Relaxed);

        let freed = manager.evict_to_target(1500);
        assert_eq!(freed, 0);
        assert_eq!(consumer.memory_usage(), 1000);
    }

    #[test]
    fn alix_with_defaults_creates_manager() {
        let manager = BufferManager::with_defaults();
        // with_defaults uses system memory detection, budget should be > 0
        assert!(manager.budget() > 0);
        assert_eq!(manager.allocated(), 0);
        assert_eq!(manager.available(), manager.budget());
    }

    #[test]
    fn gus_config_accessor_returns_budget() {
        let manager = BufferManager::with_budget(4096);
        let config = manager.config();
        assert_eq!(config.budget, 4096);
        assert!(!config.background_eviction);
        assert!(config.spill_path.is_none());
    }

    #[test]
    fn vincent_shutdown_sets_flag() {
        let manager = BufferManager::with_budget(1000);
        manager.shutdown();
        // shutdown stores true; drop also stores true, so this just verifies
        // the method runs without error and the manager remains usable
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn jules_critical_pressure_level() {
        let config = BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 0.70,
            evict_limit_fraction: 0.85,
            hard_limit_fraction: 0.95,
            background_eviction: false,
            spill_path: None,
        };
        let manager = BufferManager::new(config);

        // Manually set allocated above hard limit to test Critical level
        manager.allocated.store(960, Ordering::Relaxed);
        assert_eq!(manager.pressure_level(), PressureLevel::Critical);
    }

    #[test]
    fn mia_evict_to_target_already_below() {
        let manager = BufferManager::with_budget(10000);
        // allocated is 0, target is 5000: already below target
        let freed = manager.evict_to_target(5000);
        assert_eq!(freed, 0);
    }

    #[test]
    fn butch_safe_grant_allocation_succeeds_under_hard_limit() {
        let config = BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 0.70,
            evict_limit_fraction: 0.85,
            hard_limit_fraction: 0.95,
            background_eviction: false,
            spill_path: None,
        };
        let manager = BufferManager::new(config);

        let grant = manager
            .try_allocate(100, MemoryRegion::GraphStorage)
            .expect("safe grant allocation");
        assert_eq!(manager.allocated(), 100);
        assert_eq!(
            manager.stats().region_usage(MemoryRegion::GraphStorage),
            100
        );
        drop(grant);
    }

    #[test]
    fn django_safe_grant_allocation_fails_at_hard_limit() {
        let config = BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 0.70,
            evict_limit_fraction: 0.85,
            hard_limit_fraction: 0.95,
            background_eviction: false,
            spill_path: None,
        };
        let manager = BufferManager::new(config);

        // Fill up to hard limit
        manager.allocated.store(940, Ordering::Relaxed);

        // This exceeds hard limit (940 + 100 = 1040 > 950), no consumers to evict
        let grant = manager.try_allocate(100, MemoryRegion::ExecutionBuffers);
        assert!(grant.is_none());
    }

    #[test]
    fn shosanna_drop_sets_shutdown() {
        // Create and immediately drop to exercise the Drop impl
        let manager = BufferManager::with_budget(512);
        drop(manager);
        // If we get here without panic, the Drop impl ran successfully.
    }

    #[test]
    fn hans_eviction_with_zero_usage_consumer() {
        let manager = BufferManager::with_budget(10000);
        // Consumer with zero usage: target_evict will be 0, so evict is skipped
        let consumer = TestConsumer::new(
            "empty",
            0,
            priorities::SPILL_STAGING,
            MemoryRegion::SpillStaging,
        );
        manager.register_consumer(Arc::clone(&consumer) as Arc<dyn MemoryConsumer>);
        manager.allocated.store(500, Ordering::Relaxed);

        let freed = manager.evict_to_target(200);
        // Consumer has 0 usage, so target_evict = min(300, 0/2) = 0, evict skipped
        assert_eq!(consumer.evicted.load(Ordering::Relaxed), 0);
        assert_eq!(freed, 0);
    }

    #[test]
    fn beatrix_grant_drop_decrements_accounting() {
        let config = BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 0.70,
            evict_limit_fraction: 0.85,
            hard_limit_fraction: 0.95,
            background_eviction: false,
            spill_path: None,
        };
        let manager = BufferManager::new(config);

        let grant = manager
            .try_allocate(200, MemoryRegion::IndexBuffers)
            .expect("safe grant allocation");
        assert_eq!(manager.allocated(), 200);

        drop(grant);
        assert_eq!(manager.allocated(), 0);
        assert_eq!(manager.stats().region_usage(MemoryRegion::IndexBuffers), 0);
    }

    /// Consumer whose spill() returns an error to exercise the Err(_) => continue path.
    struct FailingSpillConsumer {
        name: String,
        usage: AtomicUsize,
        priority: u8,
        region: MemoryRegion,
    }

    impl FailingSpillConsumer {
        fn new(name: &str, usage: usize, priority: u8, region: MemoryRegion) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                usage: AtomicUsize::new(usage),
                priority,
                region,
            })
        }
    }

    impl MemoryConsumer for FailingSpillConsumer {
        fn name(&self) -> &str {
            &self.name
        }

        fn memory_usage(&self) -> usize {
            self.usage.load(Ordering::Relaxed)
        }

        fn eviction_priority(&self) -> u8 {
            self.priority
        }

        fn region(&self) -> MemoryRegion {
            self.region
        }

        fn evict(&self, _target_bytes: usize) -> usize {
            0 // eviction always fails
        }

        fn can_spill(&self) -> bool {
            true
        }

        fn spill(
            &self,
            _target_bytes: usize,
        ) -> Result<usize, crate::memory::buffer::consumer::SpillError> {
            Err(crate::memory::buffer::consumer::SpillError::IoError(
                "disk full".to_string(),
            ))
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            if self.memory_usage() == 0 {
                super::super::tiered::StorageTier::Uninitialized
            } else {
                super::super::tiered::StorageTier::InMemory
            }
        }
    }

    #[test]
    fn vincent_spill_error_continues_to_next_consumer() {
        let manager = BufferManager::with_budget(10000);

        // First consumer: spill fails
        let failing = FailingSpillConsumer::new(
            "failing_spill",
            500,
            priorities::SPILL_STAGING,
            MemoryRegion::SpillStaging,
        );

        // Second consumer: spill succeeds
        let working = SpillableConsumer::new_evict_fails(
            "working_spill",
            500,
            priorities::QUERY_CACHE,
            MemoryRegion::ExecutionBuffers,
            true,
        );

        manager.register_consumer(Arc::clone(&failing) as Arc<dyn MemoryConsumer>);
        manager.register_consumer(Arc::clone(&working) as Arc<dyn MemoryConsumer>);
        manager.allocated.store(2000, Ordering::Relaxed);

        let freed = manager.evict_to_target(1500);
        // failing consumer's spill errors out, working consumer's spill succeeds
        assert!(working.spilled.load(Ordering::Relaxed) > 0);
        assert!(freed > 0);
    }

    #[test]
    fn django_detect_system_memory_returns_positive() {
        let mem = BufferManagerConfig::detect_system_memory();
        assert!(mem > 0);
    }

    #[test]
    fn shosanna_spill_path_config() {
        let config = BufferManagerConfig {
            budget: 1024,
            spill_path: Some(PathBuf::from("/tmp/grafeo-spill")),
            ..Default::default()
        };
        assert_eq!(
            config.spill_path.as_ref().unwrap().to_str().unwrap(),
            "/tmp/grafeo-spill"
        );
        let manager = BufferManager::new(config);
        assert!(manager.config().spill_path.is_some());
    }
}
