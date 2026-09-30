//! Cooperative query cancellation and deadline control.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use thiserror::Error;

/// A typed reason why cooperative query execution stopped.
///
/// Manual cancellation and deadline expiry remain distinct so engine and
/// binding boundaries never need to classify strings. The optional timeout is
/// diagnostic configuration carried with the deadline, not a second clock.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueryCancellationError {
    /// A caller explicitly requested cancellation.
    #[error("query was cancelled")]
    Cancelled,
    /// The execution deadline was observed after it elapsed.
    #[error("query deadline exceeded")]
    DeadlineExceeded {
        /// Configured duration, when the execution owner supplied one.
        timeout: Option<Duration>,
    },
    /// The execution owner observed a phase outside the private state machine.
    #[error("query execution state is invalid (phase {phase})")]
    InvalidState {
        /// Raw phase value observed by the checker.
        phase: u8,
    },
}

/// Failure to construct a finite monotonic query deadline.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueryDeadlineError {
    /// Adding the requested duration exceeded the platform's `Instant` range.
    #[error("query timeout {timeout:?} exceeds the monotonic clock range")]
    DeadlineOverflow {
        /// Requested timeout duration.
        timeout: Duration,
    },
}

/// Invalid or cancelled execution-owner lifecycle transition.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueryLifecycleError {
    /// Cancellation or deadline expiry won before this transition.
    #[error(transparent)]
    Cancelled(#[from] QueryCancellationError),
    /// A commit fence has already been installed.
    #[error("query commit has already begun")]
    CommitAlreadyStarted,
    /// Successful execution has already been closed.
    #[error("query execution has already finished")]
    ExecutionAlreadyFinished,
    /// An internal lifecycle phase was not one of the validated states.
    #[error("query execution state is invalid")]
    InvalidState {
        /// Raw phase value observed by the lifecycle transition.
        phase: u8,
    },
}

#[derive(Debug)]
struct QueryExecutionState {
    phase: AtomicU8,
    deadline_has_timeout: AtomicBool,
    deadline_timeout_secs: AtomicU64,
    deadline_timeout_nanos: AtomicU32,
}

impl QueryExecutionState {
    const RUNNING: u8 = 0;
    const CANCELLED: u8 = 1;
    const DEADLINE_EXCEEDED: u8 = 2;
    const COMMITTING: u8 = 3;
    const FINISHED: u8 = 4;
    const RECORDING_DEADLINE: u8 = 5;

    fn new() -> Self {
        Self {
            phase: AtomicU8::new(Self::RUNNING),
            deadline_has_timeout: AtomicBool::new(false),
            deadline_timeout_secs: AtomicU64::new(0),
            deadline_timeout_nanos: AtomicU32::new(0),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn try_record_deadline(&self, timeout: Option<Duration>) -> bool {
        if self
            .phase
            .compare_exchange(
                Self::RUNNING,
                Self::RECORDING_DEADLINE,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return false;
        }

        if let Some(timeout) = timeout {
            self.deadline_timeout_secs
                .store(timeout.as_secs(), Ordering::Relaxed);
            self.deadline_timeout_nanos
                .store(timeout.subsec_nanos(), Ordering::Relaxed);
            self.deadline_has_timeout.store(true, Ordering::Relaxed);
        }
        self.phase.store(Self::DEADLINE_EXCEEDED, Ordering::Release);
        true
    }

    fn deadline_error(&self) -> QueryCancellationError {
        let timeout = self.deadline_has_timeout.load(Ordering::Relaxed).then(|| {
            Duration::new(
                self.deadline_timeout_secs.load(Ordering::Relaxed),
                self.deadline_timeout_nanos.load(Ordering::Relaxed),
            )
        });
        QueryCancellationError::DeadlineExceeded { timeout }
    }
}

fn check_state(
    state: &QueryExecutionState,
    #[cfg(not(target_arch = "wasm32"))] deadline: Option<Instant>,
    #[cfg(not(target_arch = "wasm32"))] timeout: Option<Duration>,
) -> Result<(), QueryCancellationError> {
    loop {
        match state.phase.load(Ordering::Acquire) {
            QueryExecutionState::CANCELLED => return Err(QueryCancellationError::Cancelled),
            QueryExecutionState::DEADLINE_EXCEEDED => {
                return Err(state.deadline_error());
            }
            QueryExecutionState::COMMITTING | QueryExecutionState::FINISHED => return Ok(()),
            QueryExecutionState::RECORDING_DEADLINE => {
                std::hint::spin_loop();
                continue;
            }
            QueryExecutionState::RUNNING => {}
            phase => return Err(QueryCancellationError::InvalidState { phase }),
        }

        #[cfg(not(target_arch = "wasm32"))]
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            if state.try_record_deadline(timeout) {
                return Err(state.deadline_error());
            }
            continue;
        }

        return Ok(());
    }
}

/// Check-only cooperative cancellation capability supplied to query workers.
///
/// Clones share immutable deadline metadata and one terminal atomic state.
/// Checks and clones do not allocate. This capability cannot request
/// cancellation or close the commit/success window, so downstream operators
/// cannot disable an engine-owned deadline.
#[derive(Clone, Debug)]
pub struct QueryCancellationToken {
    state: Arc<QueryExecutionState>,
    #[cfg(not(target_arch = "wasm32"))]
    deadline: Option<Instant>,
    #[cfg(not(target_arch = "wasm32"))]
    timeout: Option<Duration>,
}

impl QueryCancellationToken {
    /// Checks the first terminal cancellation reason.
    ///
    /// # Errors
    ///
    /// Returns the typed, allocation-free winning cancellation reason.
    pub fn check(&self) -> Result<(), QueryCancellationError> {
        check_state(
            &self.state,
            #[cfg(not(target_arch = "wasm32"))]
            self.deadline,
            #[cfg(not(target_arch = "wasm32"))]
            self.timeout,
        )
    }

    /// Returns whether cancellation or deadline expiry won for this query.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.check().is_err()
    }

    /// Returns the configured absolute deadline, if any.
    #[cfg(not(target_arch = "wasm32"))]
    #[must_use]
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Returns the configured timeout duration used for diagnostics, if any.
    #[cfg(not(target_arch = "wasm32"))]
    #[must_use]
    pub fn timeout_duration(&self) -> Option<Duration> {
        self.timeout
    }
}

/// Orchestration capability for execution checks and deadline composition.
///
/// Unlike [`QueryCancellationToken`], this capability may add a compatibility
/// deadline before deriving worker tokens. Keep it at the executor/pipeline
/// boundary; distribute only [`Self::token`] to operators and resource
/// contexts.
#[derive(Clone, Debug)]
pub struct QueryExecutionCheckpoint {
    state: Arc<QueryExecutionState>,
    #[cfg(not(target_arch = "wasm32"))]
    deadline: Option<Instant>,
    #[cfg(not(target_arch = "wasm32"))]
    timeout: Option<Duration>,
}

impl QueryExecutionCheckpoint {
    /// Checks the first terminal cancellation reason.
    ///
    /// # Errors
    ///
    /// Returns the shared winning cancellation reason.
    pub fn check(&self) -> Result<(), QueryCancellationError> {
        check_state(
            &self.state,
            #[cfg(not(target_arch = "wasm32"))]
            self.deadline,
            #[cfg(not(target_arch = "wasm32"))]
            self.timeout,
        )
    }

    /// Adds a deadline snapshot while preserving shared cancellation state.
    ///
    /// The earlier deadline wins. Call this only at an orchestration boundary,
    /// before deriving tokens for workers.
    #[cfg(not(target_arch = "wasm32"))]
    #[must_use]
    pub fn with_additional_deadline(
        mut self,
        deadline: Instant,
        timeout: Option<Duration>,
    ) -> Self {
        if self.deadline.is_none_or(|current| deadline < current) {
            self.deadline = Some(deadline);
            self.timeout = timeout;
        }
        self
    }

    /// Derives a check-only worker token with this exact deadline snapshot.
    #[must_use]
    pub fn token(&self) -> QueryCancellationToken {
        QueryCancellationToken {
            state: Arc::clone(&self.state),
            #[cfg(not(target_arch = "wasm32"))]
            deadline: self.deadline,
            #[cfg(not(target_arch = "wasm32"))]
            timeout: self.timeout,
        }
    }
}

/// Cancel-only capability safe to share with callers or supervising threads.
///
/// Cancellation is one atomic compare-and-exchange. It never locks, performs
/// I/O, closes resources, runs cleanup, or waits for acknowledgement. The
/// query thread observes the request at a cooperative checkpoint.
#[derive(Clone, Debug)]
pub struct QueryCancellationHandle {
    state: Arc<QueryExecutionState>,
}

impl QueryCancellationHandle {
    /// Requests cancellation for the associated execution.
    pub fn cancel(&self) {
        let _ = self.try_cancel();
    }

    /// Tries to win the transition from running to cancelled.
    ///
    /// Returns `true` only when this call won the transition out of cancellable
    /// execution. It returns `false` after an earlier cancellation, deadline,
    /// commit fence, or successful completion. Commit/result outcomes are
    /// authoritative once their fence wins.
    #[must_use]
    pub fn try_cancel(&self) -> bool {
        self.state
            .phase
            .compare_exchange(
                QueryExecutionState::RUNNING,
                QueryExecutionState::CANCELLED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }
}

/// Execution-owner capability for cancellation, commit, and success fencing.
///
/// Create one control per execution, retain it at the orchestration boundary,
/// and distribute only [`Self::token`] to operators and
/// [`Self::cancellation_handle`] to cancellers. Give [`Self::checkpoint`] only
/// to executor/pipeline orchestration. The control is deliberately neither
/// `Clone` nor `Sync`: lifecycle authority must have one mutable owner.
#[derive(Debug)]
pub struct QueryExecutionControl {
    state: Arc<QueryExecutionState>,
    #[cfg(not(target_arch = "wasm32"))]
    deadline: Option<Instant>,
    #[cfg(not(target_arch = "wasm32"))]
    timeout: Option<Duration>,
    _not_sync: PhantomData<Cell<()>>,
}

impl QueryExecutionControl {
    /// Creates execution control with no deadline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(QueryExecutionState::new()),
            #[cfg(not(target_arch = "wasm32"))]
            deadline: None,
            #[cfg(not(target_arch = "wasm32"))]
            timeout: None,
            _not_sync: PhantomData,
        }
    }

    /// Creates execution control with one absolute monotonic deadline.
    ///
    /// Monotonic deadline enforcement is unavailable on WebAssembly targets;
    /// explicit cancellation remains portable there.
    #[cfg(not(target_arch = "wasm32"))]
    #[must_use]
    pub fn with_deadline(deadline: Instant) -> Self {
        Self {
            state: Arc::new(QueryExecutionState::new()),
            deadline: Some(deadline),
            timeout: None,
            _not_sync: PhantomData,
        }
    }

    /// Creates execution control whose deadline is `timeout` from now.
    ///
    /// # Errors
    ///
    /// Returns [`QueryDeadlineError::DeadlineOverflow`] instead of panicking
    /// when the public duration cannot be represented by the platform clock.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_timeout(timeout: Duration) -> Result<Self, QueryDeadlineError> {
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or(QueryDeadlineError::DeadlineOverflow { timeout })?;
        Ok(Self {
            state: Arc::new(QueryExecutionState::new()),
            deadline: Some(deadline),
            timeout: Some(timeout),
            _not_sync: PhantomData,
        })
    }

    /// Adds an optional deadline snapshot while preserving the earliest bound.
    ///
    /// This consumes the owner so an orchestration boundary can compose its
    /// timeout once before distributing checkpoints and worker tokens. A
    /// later or absent deadline never restarts or replaces the existing one.
    #[cfg(not(target_arch = "wasm32"))]
    #[must_use]
    pub fn with_additional_deadline(
        mut self,
        deadline: Option<Instant>,
        timeout: Option<Duration>,
    ) -> Self {
        if let Some(deadline) = deadline
            && self.deadline.is_none_or(|current| deadline < current)
        {
            self.deadline = Some(deadline);
            self.timeout = timeout;
        }
        self
    }

    /// Returns an orchestration checkpoint capable of composing deadlines.
    #[must_use]
    pub fn checkpoint(&self) -> QueryExecutionCheckpoint {
        QueryExecutionCheckpoint {
            state: Arc::clone(&self.state),
            #[cfg(not(target_arch = "wasm32"))]
            deadline: self.deadline,
            #[cfg(not(target_arch = "wasm32"))]
            timeout: self.timeout,
        }
    }

    /// Checks cancellation and the owner deadline without deriving a
    /// checkpoint or cloning the shared state.
    ///
    /// # Errors
    ///
    /// Returns the recorded cancellation, deadline or invalid-state error.
    pub fn check(&self) -> Result<(), QueryCancellationError> {
        check_state(
            &self.state,
            #[cfg(not(target_arch = "wasm32"))]
            self.deadline,
            #[cfg(not(target_arch = "wasm32"))]
            self.timeout,
        )
    }

    /// Returns a check-only capability for query workers.
    #[must_use]
    pub fn token(&self) -> QueryCancellationToken {
        self.checkpoint().token()
    }

    /// Returns a cancel-only capability for a supervising caller or thread.
    #[must_use]
    pub fn cancellation_handle(&self) -> QueryCancellationHandle {
        QueryCancellationHandle {
            state: Arc::clone(&self.state),
        }
    }

    /// Atomically closes the cancellation window before durable commit.
    ///
    /// Call this only after all operator/worker activity and required
    /// pre-commit execution cleanup have quiesced. Once the fence wins, worker
    /// tokens intentionally stop reporting later cancellation or deadline
    /// expiry so the commit outcome remains authoritative.
    ///
    /// A cancellation request or elapsed deadline observed first causes an
    /// error and the caller must roll back. If this transition wins, later
    /// cancellation returns `false`; commit success or failure is authoritative
    /// and must never be reclassified as cancellation.
    ///
    /// # Errors
    ///
    /// Returns the first cancellation reason observed before the commit fence,
    /// or an invalid-transition error if commit/completion was already fenced.
    pub fn try_begin_commit(&mut self) -> Result<(), QueryLifecycleError> {
        loop {
            check_state(
                &self.state,
                #[cfg(not(target_arch = "wasm32"))]
                self.deadline,
                #[cfg(not(target_arch = "wasm32"))]
                self.timeout,
            )
            .map_err(|error| match error {
                QueryCancellationError::InvalidState { phase } => {
                    QueryLifecycleError::InvalidState { phase }
                }
                cancellation => QueryLifecycleError::Cancelled(cancellation),
            })?;
            match self.state.phase.compare_exchange(
                QueryExecutionState::RUNNING,
                QueryExecutionState::COMMITTING,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(QueryExecutionState::CANCELLED) => {
                    return Err(QueryCancellationError::Cancelled.into());
                }
                Err(QueryExecutionState::DEADLINE_EXCEEDED) => {
                    return Err(self.state.deadline_error().into());
                }
                Err(QueryExecutionState::COMMITTING) => {
                    return Err(QueryLifecycleError::CommitAlreadyStarted);
                }
                Err(QueryExecutionState::FINISHED) => {
                    return Err(QueryLifecycleError::ExecutionAlreadyFinished);
                }
                Err(QueryExecutionState::RECORDING_DEADLINE) => {
                    std::hint::spin_loop();
                    continue;
                }
                Err(QueryExecutionState::RUNNING) => continue,
                Err(phase) => return Err(QueryLifecycleError::InvalidState { phase }),
            }
        }
    }

    /// Marks non-committing execution complete or closes a successful commit.
    ///
    /// Call this only after result materialization and mandatory cleanup have
    /// succeeded. When completion wins, later cancellation returns `false` and
    /// successful output remains authoritative.
    ///
    /// # Errors
    ///
    /// Returns the first cancellation reason if it won while execution was
    /// still cancellable, or an invalid-transition error after completion.
    pub fn complete(&mut self) -> Result<(), QueryLifecycleError> {
        loop {
            check_state(
                &self.state,
                #[cfg(not(target_arch = "wasm32"))]
                self.deadline,
                #[cfg(not(target_arch = "wasm32"))]
                self.timeout,
            )
            .map_err(|error| match error {
                QueryCancellationError::InvalidState { phase } => {
                    QueryLifecycleError::InvalidState { phase }
                }
                cancellation => QueryLifecycleError::Cancelled(cancellation),
            })?;
            let current = self.state.phase.load(Ordering::Acquire);
            match current {
                QueryExecutionState::RUNNING | QueryExecutionState::COMMITTING => {
                    match self.state.phase.compare_exchange(
                        current,
                        QueryExecutionState::FINISHED,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => return Ok(()),
                        Err(_) => continue,
                    }
                }
                QueryExecutionState::FINISHED => {
                    return Err(QueryLifecycleError::ExecutionAlreadyFinished);
                }
                QueryExecutionState::CANCELLED => {
                    return Err(QueryCancellationError::Cancelled.into());
                }
                QueryExecutionState::DEADLINE_EXCEEDED => {
                    return Err(self.state.deadline_error().into());
                }
                QueryExecutionState::RECORDING_DEADLINE => {
                    std::hint::spin_loop();
                    continue;
                }
                phase => return Err(QueryLifecycleError::InvalidState { phase }),
            }
        }
    }
}

impl Default for QueryExecutionControl {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_is_one_way_shared_and_first_reason_wins() {
        let control = QueryExecutionControl::new();
        let token = control.token();
        let observer = token.clone();
        let handle = control.cancellation_handle();

        assert_eq!(observer.check(), Ok(()));
        assert!(handle.try_cancel());
        assert!(!handle.try_cancel());

        assert_eq!(observer.check(), Err(QueryCancellationError::Cancelled));
        assert_eq!(control.check(), Err(QueryCancellationError::Cancelled));
        assert!(token.is_cancelled());
    }

    #[test]
    fn fire_and_forget_cancellation_is_idempotent() {
        let control = QueryExecutionControl::new();
        let token = control.token();
        let handle = control.cancellation_handle();

        handle.cancel();
        handle.cancel();

        assert_eq!(token.check(), Err(QueryCancellationError::Cancelled));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn expired_deadline_retains_configured_duration() {
        let timeout = Duration::ZERO;
        let control = QueryExecutionControl::with_timeout(timeout).unwrap();
        let token = control.token();

        assert_eq!(
            token.check(),
            Err(QueryCancellationError::DeadlineExceeded {
                timeout: Some(timeout),
            })
        );
        assert!(
            !control.cancellation_handle().try_cancel(),
            "the first terminal reason remains authoritative"
        );
        assert_eq!(
            token.check(),
            Err(QueryCancellationError::DeadlineExceeded {
                timeout: Some(timeout),
            })
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn timeout_construction_fails_closed_on_clock_overflow() {
        assert!(matches!(
            QueryExecutionControl::with_timeout(Duration::MAX),
            Err(QueryDeadlineError::DeadlineOverflow {
                timeout: Duration::MAX,
            })
        ));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn additional_deadline_uses_earliest_clock_and_its_diagnostic() {
        let control = QueryExecutionControl::with_deadline(
            Instant::now().checked_add(Duration::from_mins(1)).unwrap(),
        );
        let timeout = Duration::from_millis(7);
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();

        assert_eq!(
            control
                .checkpoint()
                .with_additional_deadline(expired, Some(timeout))
                .check(),
            Err(QueryCancellationError::DeadlineExceeded {
                timeout: Some(timeout),
            })
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn control_deadline_composition_keeps_earliest_and_ignores_later_bounds() {
        let original_deadline = Instant::now().checked_add(Duration::from_mins(1)).unwrap();
        let later_deadline = original_deadline
            .checked_add(Duration::from_mins(1))
            .unwrap();
        let control = QueryExecutionControl::with_deadline(original_deadline)
            .with_additional_deadline(Some(later_deadline), Some(Duration::from_mins(2)))
            .with_additional_deadline(None, Some(Duration::ZERO));

        assert_eq!(control.deadline, Some(original_deadline));
        assert_eq!(control.timeout, None);
        assert_eq!(control.checkpoint().check(), Ok(()));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn control_deadline_composition_replaces_with_earlier_bound_once() {
        let earlier = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        let timeout = Duration::from_millis(7);
        let control =
            QueryExecutionControl::new().with_additional_deadline(Some(earlier), Some(timeout));

        assert_eq!(control.deadline, Some(earlier));
        assert_eq!(control.timeout, Some(timeout));
        assert_eq!(
            control.checkpoint().check(),
            Err(QueryCancellationError::DeadlineExceeded {
                timeout: Some(timeout)
            })
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn composed_control_deadline_reaches_commit_fence() {
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        let timeout = Duration::from_millis(11);
        let mut control =
            QueryExecutionControl::new().with_additional_deadline(Some(expired), Some(timeout));

        assert_eq!(
            control.try_begin_commit(),
            Err(QueryLifecycleError::Cancelled(
                QueryCancellationError::DeadlineExceeded {
                    timeout: Some(timeout)
                }
            ))
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn composing_a_deadline_preserves_an_existing_cancel_handle() {
        let control = QueryExecutionControl::new();
        let handle = control.cancellation_handle();
        let token = control
            .checkpoint()
            .with_additional_deadline(
                Instant::now().checked_add(Duration::from_mins(1)).unwrap(),
                None,
            )
            .token();

        assert!(handle.try_cancel());
        assert_eq!(token.check(), Err(QueryCancellationError::Cancelled));
    }

    #[test]
    fn commit_and_cancellation_have_one_unambiguous_winner() {
        use std::sync::Barrier;

        let mut control = QueryExecutionControl::new();
        let token = control.token();
        let handle = control.cancellation_handle();
        let barrier = Arc::new(Barrier::new(2));
        let cancel_barrier = Arc::clone(&barrier);
        let thread = std::thread::spawn(move || {
            cancel_barrier.wait();
            handle.try_cancel()
        });

        barrier.wait();
        let commit = control.try_begin_commit();
        let cancelled = thread.join().unwrap();

        match (commit, cancelled) {
            (Ok(()), false) => {
                assert_eq!(token.check(), Ok(()));
                control.complete().unwrap();
            }
            (Err(QueryLifecycleError::Cancelled(QueryCancellationError::Cancelled)), true) => {
                assert_eq!(token.check(), Err(QueryCancellationError::Cancelled));
            }
            outcome => panic!("commit/cancel race had no unique winner: {outcome:?}"),
        }
    }

    #[test]
    fn completed_result_cannot_be_retroactively_cancelled() {
        let mut control = QueryExecutionControl::new();
        let token = control.token();
        let handle = control.cancellation_handle();
        control.complete().unwrap();

        assert!(!handle.try_cancel());
        assert_eq!(token.check(), Ok(()));
    }

    #[test]
    fn lifecycle_fences_are_exclusive_and_fail_closed() {
        let mut committed = QueryExecutionControl::new();
        committed.try_begin_commit().unwrap();
        assert_eq!(
            committed.try_begin_commit(),
            Err(QueryLifecycleError::CommitAlreadyStarted)
        );
        committed.complete().unwrap();
        assert_eq!(
            committed.complete(),
            Err(QueryLifecycleError::ExecutionAlreadyFinished)
        );

        let mut completed = QueryExecutionControl::new();
        completed.complete().unwrap();
        assert_eq!(
            completed.try_begin_commit(),
            Err(QueryLifecycleError::ExecutionAlreadyFinished)
        );
    }

    #[test]
    fn malformed_private_phase_is_typed_for_token_commit_and_complete() {
        let invalid_phase = 0xff;
        let mut control = QueryExecutionControl::new();
        control.state.phase.store(invalid_phase, Ordering::Release);

        assert_eq!(
            control.token().check(),
            Err(QueryCancellationError::InvalidState {
                phase: invalid_phase
            })
        );
        assert_eq!(
            control.check(),
            Err(QueryCancellationError::InvalidState {
                phase: invalid_phase
            })
        );
        assert_eq!(
            control.try_begin_commit(),
            Err(QueryLifecycleError::InvalidState {
                phase: invalid_phase
            })
        );
        assert_eq!(
            control.complete(),
            Err(QueryLifecycleError::InvalidState {
                phase: invalid_phase
            })
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn owner_check_matches_checkpoint_before_and_after_commit_fence() {
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        let timeout = Duration::from_millis(13);
        let control =
            QueryExecutionControl::new().with_additional_deadline(Some(expired), Some(timeout));
        let expected = Err(QueryCancellationError::DeadlineExceeded {
            timeout: Some(timeout),
        });
        assert_eq!(control.check(), expected);
        assert_eq!(control.checkpoint().check(), expected);

        let mut committed = QueryExecutionControl::new();
        committed.try_begin_commit().unwrap();
        let handle = committed.cancellation_handle();
        assert_eq!(committed.check(), Ok(()));
        assert!(!handle.try_cancel(), "commit fence remains authoritative");
        committed.complete().unwrap();
        assert_eq!(committed.check(), Ok(()));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn winning_deadline_diagnostic_is_shared_across_all_observers() {
        let mut control = QueryExecutionControl::with_deadline(
            Instant::now().checked_add(Duration::from_mins(1)).unwrap(),
        );
        let base = control.token();
        let timeout = Duration::from_millis(11);
        let expired = control
            .checkpoint()
            .with_additional_deadline(
                Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
                Some(timeout),
            )
            .token();
        let expected = QueryCancellationError::DeadlineExceeded {
            timeout: Some(timeout),
        };

        assert_eq!(expired.check(), Err(expected));
        assert_eq!(base.check(), Err(expected));
        assert_eq!(
            control.try_begin_commit(),
            Err(QueryLifecycleError::Cancelled(expected))
        );
    }
}
