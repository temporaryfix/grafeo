//! External merge sort for out-of-core sorting.
//!
//! This module implements external sorting using sorted runs on disk.
//! When memory is exhausted, sorted buffers are written as runs to disk.
//! On finalization, all runs are merged using k-way merge.

use super::file::{
    ExactOwnedReaderQualification, ProviderAccountedReaderError,
    ProviderAccountedReaderFailureClassification, ProviderAccountedReaderOperationError,
    ProviderAccountedReaderReceipt, ProviderAccountedReaderResolutionError,
    ProviderAccountedSortRow, ProviderAccountedSpillFileReader, ScalarReaderCleanup, SpillFile,
    SpillFileReader, SpillFileRole, SpillFinishError, SpillRecordBuffer, SpillWriterBuffer,
    qualified_writer_buffer_requested_bytes,
};
use super::manager::SpillManager;
use crate::execution::accounted_chunk::{AccountedOrdinalRow, AccountedSortRowError, SortRowShape};
use crate::execution::operators::push::spill_state::OperatorSpillState;
use crate::execution::operators::value_utils::compare_values_total;
use crate::execution::operators::{AccountedFailureClassification, OperatorError};
use crate::execution::operators::{AccountedValueComparator, SemanticComparisonError};
#[cfg(test)]
use crate::execution::value_codec::serialize_row_with_limits;
use crate::execution::value_codec::{
    CodecLimits, CounterSortScratch, DecodedFramedRow, QualifiedCodecError,
    conservative_decoded_row_retained_bytes, deserialize_framed_row_exact,
    deserialize_framed_row_exact_with_receipt, deserialize_framed_row_exact_with_receipt_qualified,
    measure_serialized_row_with_limits, serialize_row_with_prepared_scratch,
};
use crate::execution::{QueryCancellationError, QueryCancellationToken};
use allocator_api2::alloc::Global;
use allocator_api2::vec::Vec as ExactVec;
use grafeo_common::memory::buffer::{
    AccountedError, AccountedErrorPublisher, AccountedErrorPublisherBuildError,
    AccountedErrorPublisherBuildFailure, MemoryGrant, MemoryGrantError,
};
use grafeo_common::types::Value;
use std::cell::Cell;
use std::cmp::Ordering;
#[cfg(test)]
use std::collections::BinaryHeap;
use std::io::Write as _;
use std::sync::Arc;

#[cfg(test)]
#[path = "external_sort/scalar_failure_tests.rs"]
mod scalar_failure_tests;

const INITIAL_RUN_CATALOG_CAPACITY: usize = 4;
/// Conservative descriptor and heap-head ceiling for one merge step.
const DEFAULT_MERGE_FAN_IN: usize = 16;

/// Sealed scalar publication for resource-qualified sorter grant changes.
///
/// Unlike an arbitrary callback, this owner cannot
/// execute caller code. Publishing uses only pre-existing `Cell` and operator
/// telemetry storage, performs no allocation, and cannot unwind. The retained
/// component is read at publication time so a moved-row release is reflected
/// without rebuilding the observer.
pub(crate) struct ExternalSortGrantObserver<'a> {
    target: ExternalSortGrantObserverTarget<'a>,
    /// Sticky evidence that a transferred owner lost its only release token
    /// during `Drop`. Ordinary exact publications must never clear this latch.
    unacknowledged_retained_poisoned: Cell<bool>,
    #[cfg(test)]
    peak_external_bytes: Option<&'a Cell<usize>>,
}

enum ExternalSortGrantObserverTarget<'a> {
    Inert,
    Accounted {
        external_bytes: ObserverCounter<'a>,
        retained_bytes: ObserverCounter<'a>,
        spill_state: Option<ObserverSpillState<'a>>,
    },
}

enum ObserverCounter<'a> {
    Borrowed(&'a Cell<usize>),
    Owned(Cell<usize>),
}
impl std::ops::Deref for ObserverCounter<'_> {
    type Target = Cell<usize>;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(counter) => counter,
            Self::Owned(counter) => counter,
        }
    }
}
enum ObserverSpillState<'a> {
    Borrowed(&'a OperatorSpillState),
    Owned(Arc<OperatorSpillState>),
}
impl std::ops::Deref for ObserverSpillState<'_> {
    type Target = OperatorSpillState;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(state) => state,
            Self::Owned(state) => state,
        }
    }
}

impl<'a> ExternalSortGrantObserver<'a> {
    pub(crate) fn new(
        external_bytes: &'a Cell<usize>,
        retained_bytes: &'a Cell<usize>,
        spill_state: Option<&'a OperatorSpillState>,
    ) -> Self {
        Self {
            target: ExternalSortGrantObserverTarget::Accounted {
                external_bytes: ObserverCounter::Borrowed(external_bytes),
                retained_bytes: ObserverCounter::Borrowed(retained_bytes),
                spill_state: spill_state.map(ObserverSpillState::Borrowed),
            },
            unacknowledged_retained_poisoned: Cell::new(false),
            #[cfg(test)]
            peak_external_bytes: None,
        }
    }

    /// Owns telemetry for a resumable consumer without borrowing its state.
    pub(crate) fn owned(state: Arc<OperatorSpillState>) -> ExternalSortGrantObserver<'static> {
        ExternalSortGrantObserver {
            target: ExternalSortGrantObserverTarget::Accounted {
                external_bytes: ObserverCounter::Owned(Cell::new(0)),
                retained_bytes: ObserverCounter::Owned(Cell::new(0)),
                spill_state: Some(ObserverSpillState::Owned(state)),
            },
            unacknowledged_retained_poisoned: Cell::new(false),
            #[cfg(test)]
            peak_external_bytes: None,
        }
    }

    fn inert() -> Self {
        Self {
            target: ExternalSortGrantObserverTarget::Inert,
            unacknowledged_retained_poisoned: Cell::new(false),
            #[cfg(test)]
            peak_external_bytes: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn tracking_peak(mut self, peak_external_bytes: &'a Cell<usize>) -> Self {
        self.peak_external_bytes = Some(peak_external_bytes);
        self
    }

    pub(crate) fn publish(&self, bytes: usize) -> Result<(), MemoryGrantError> {
        self.reject_if_unacknowledged_retained_poisoned()?;
        let ExternalSortGrantObserverTarget::Accounted {
            external_bytes,
            retained_bytes,
            spill_state,
        } = &self.target
        else {
            return Ok(());
        };
        external_bytes.set(bytes);
        #[cfg(test)]
        if let Some(peak) = self.peak_external_bytes {
            peak.set(peak.get().max(bytes));
        }
        let retained = retained_bytes.get();
        let Some(total) = retained.checked_add(bytes) else {
            // Keep any available eviction telemetry conservative while
            // returning the allocation-free typed invariant failure. The sum
            // is checked even when this internal observer has no telemetry
            // sink, so `None` cannot weaken qualified overflow detection.
            self.publish_unrepresentable();
            return Err(MemoryGrantError::ArithmeticOverflow {
                current_bytes: retained,
                additional_bytes: bytes,
            });
        };
        if let Some(state) = spill_state {
            state.set_usage(total);
        }
        Ok(())
    }

    fn publish_sum(
        &self,
        current_bytes: usize,
        additional_bytes: usize,
    ) -> Result<(), MemoryGrantError> {
        let Some(total) = current_bytes.checked_add(additional_bytes) else {
            self.publish_unrepresentable();
            return Err(MemoryGrantError::ArithmeticOverflow {
                current_bytes,
                additional_bytes,
            });
        };
        self.publish(total)
    }

    pub(crate) fn publish_unrepresentable(&self) {
        let ExternalSortGrantObserverTarget::Accounted {
            external_bytes,
            spill_state,
            ..
        } = &self.target
        else {
            return;
        };
        external_bytes.set(usize::MAX);
        #[cfg(test)]
        if let Some(peak) = self.peak_external_bytes {
            peak.set(usize::MAX);
        }
        if let Some(state) = spill_state {
            state.set_usage(usize::MAX);
        }
    }

    /// Publishes a transferred row/chunk charge beside the current exact-sort
    /// frontier without exposing either scalar sink to callers.
    pub(crate) fn publish_retained(&self, bytes: usize) -> Result<(), MemoryGrantError> {
        self.reject_if_unacknowledged_retained_poisoned()?;
        let ExternalSortGrantObserverTarget::Accounted {
            external_bytes,
            retained_bytes,
            spill_state,
        } = &self.target
        else {
            return Ok(());
        };
        retained_bytes.set(bytes);
        let external = external_bytes.get();
        let Some(total) = external.checked_add(bytes) else {
            self.publish_unrepresentable();
            return Err(MemoryGrantError::ArithmeticOverflow {
                current_bytes: external,
                additional_bytes: bytes,
            });
        };
        if let Some(state) = spill_state {
            state.set_usage(total);
        }
        Ok(())
    }

    pub(crate) fn publish_retained_preserving_primary(&self, bytes: usize) {
        let _ = self.publish_retained(bytes);
    }

    fn poison_unacknowledged_retained(&self) {
        self.unacknowledged_retained_poisoned.set(true);
        self.publish_unrepresentable();
    }

    fn reject_if_unacknowledged_retained_poisoned(&self) -> Result<(), MemoryGrantError> {
        if !self.unacknowledged_retained_poisoned.get() {
            return Ok(());
        }
        self.publish_unrepresentable();
        Err(MemoryGrantError::AccountingPoisoned {
            account: "exact owned sort unacknowledged retained handoff",
        })
    }

    fn has_unacknowledged_retained_poison(&self) -> bool {
        self.unacknowledged_retained_poisoned.get()
    }

    /// Atomically transfers one live row charge out of the cursor frontier
    /// and into the downstream-retained slot. All arithmetic is validated
    /// before either scalar changes, so observers never see a transient
    /// undercount between the two owners.
    fn transfer_row_to_retained(
        &self,
        stable_sorter_bytes: usize,
        reader_bytes: usize,
        row_bytes: &Cell<usize>,
        transferred_bytes: usize,
    ) -> Result<(), MemoryGrantError> {
        self.reject_if_unacknowledged_retained_poisoned()?;
        let current_rows = row_bytes.get();
        let Some(remaining_rows) = current_rows.checked_sub(transferred_bytes) else {
            self.publish_unrepresentable();
            return Err(MemoryGrantError::AccountingUnderflow {
                account: "exact owned sort row transfer",
                accounted_bytes: current_rows,
                release_bytes: transferred_bytes,
            });
        };
        let Some(children) = reader_bytes.checked_add(remaining_rows) else {
            self.publish_unrepresentable();
            return Err(MemoryGrantError::ArithmeticOverflow {
                current_bytes: reader_bytes,
                additional_bytes: remaining_rows,
            });
        };
        let Some(external) = stable_sorter_bytes.checked_add(children) else {
            self.publish_unrepresentable();
            return Err(MemoryGrantError::ArithmeticOverflow {
                current_bytes: stable_sorter_bytes,
                additional_bytes: children,
            });
        };
        let Some(total) = external.checked_add(transferred_bytes) else {
            self.publish_unrepresentable();
            return Err(MemoryGrantError::ArithmeticOverflow {
                current_bytes: external,
                additional_bytes: transferred_bytes,
            });
        };
        let Some(previous_children) = reader_bytes.checked_add(current_rows) else {
            self.publish_unrepresentable();
            return Err(MemoryGrantError::ArithmeticOverflow {
                current_bytes: reader_bytes,
                additional_bytes: current_rows,
            });
        };
        let Some(previous_external) = stable_sorter_bytes.checked_add(previous_children) else {
            self.publish_unrepresentable();
            return Err(MemoryGrantError::ArithmeticOverflow {
                current_bytes: stable_sorter_bytes,
                additional_bytes: previous_children,
            });
        };

        match &self.target {
            ExternalSortGrantObserverTarget::Inert => row_bytes.set(remaining_rows),
            ExternalSortGrantObserverTarget::Accounted {
                external_bytes,
                retained_bytes,
                spill_state,
            } => {
                if external_bytes.get() != previous_external {
                    self.publish_unrepresentable();
                    return Err(MemoryGrantError::AccountingPoisoned {
                        account: "exact owned sort atomic row transfer",
                    });
                }
                if retained_bytes.get() != 0 {
                    self.publish_unrepresentable();
                    return Err(MemoryGrantError::AccountingUnderflow {
                        account: "exact owned sort retained transfer slot",
                        accounted_bytes: 0,
                        release_bytes: retained_bytes.get(),
                    });
                }
                row_bytes.set(remaining_rows);
                external_bytes.set(external);
                retained_bytes.set(transferred_bytes);
                #[cfg(test)]
                if let Some(peak) = self.peak_external_bytes {
                    peak.set(peak.get().max(external));
                }
                if let Some(state) = spill_state {
                    state.set_usage(total);
                }
            }
        }
        Ok(())
    }

    /// Atomically returns a failed downstream-construction grant from the
    /// retained slot to the cursor's sorter frontier. The physical output has
    /// already been destroyed; this changes only which sealed owner holds the
    /// sole retry capability and leaves total observed usage unchanged.
    fn transfer_retained_to_external(
        &self,
        previous_external: usize,
        current_external: usize,
        transferred_bytes: usize,
    ) -> Result<(), MemoryGrantError> {
        self.reject_if_unacknowledged_retained_poisoned()?;
        let Some(expected_external) = previous_external.checked_add(transferred_bytes) else {
            self.publish_unrepresentable();
            return Err(MemoryGrantError::ArithmeticOverflow {
                current_bytes: previous_external,
                additional_bytes: transferred_bytes,
            });
        };
        if current_external != expected_external {
            self.publish_unrepresentable();
            return Err(MemoryGrantError::AccountingPoisoned {
                account: "exact failed output grant transfer",
            });
        }
        match &self.target {
            ExternalSortGrantObserverTarget::Inert => Ok(()),
            ExternalSortGrantObserverTarget::Accounted {
                external_bytes,
                retained_bytes,
                spill_state,
            } => {
                if external_bytes.get() != previous_external
                    || retained_bytes.get() != transferred_bytes
                {
                    self.publish_unrepresentable();
                    return Err(MemoryGrantError::AccountingPoisoned {
                        account: "exact failed output retained attribution",
                    });
                }
                external_bytes.set(current_external);
                retained_bytes.set(0);
                #[cfg(test)]
                if let Some(peak) = self.peak_external_bytes {
                    peak.set(peak.get().max(current_external));
                }
                if let Some(state) = spill_state {
                    state.set_usage(current_external);
                }
                Ok(())
            }
        }
    }

    /// Moves the pre-admitted final-error block from cursor-local attribution
    /// into the sealed retained slot without changing total observed usage.
    /// The block remains globally/query-accounted while cleanup runs and until
    /// it is either published or physically destroyed.
    fn transfer_control_to_retained(
        &self,
        external_without_control: usize,
        control_bytes: usize,
    ) -> Result<(), MemoryGrantError> {
        self.reject_if_unacknowledged_retained_poisoned()?;
        let Some(previous_external) = external_without_control.checked_add(control_bytes) else {
            self.publish_unrepresentable();
            return Err(MemoryGrantError::ArithmeticOverflow {
                current_bytes: external_without_control,
                additional_bytes: control_bytes,
            });
        };
        let Some(total) = external_without_control.checked_add(control_bytes) else {
            unreachable!("the identical checked sum succeeded above")
        };
        match &self.target {
            ExternalSortGrantObserverTarget::Inert => Ok(()),
            ExternalSortGrantObserverTarget::Accounted {
                external_bytes,
                retained_bytes,
                spill_state,
            } => {
                if external_bytes.get() != previous_external || retained_bytes.get() != 0 {
                    self.publish_unrepresentable();
                    return Err(MemoryGrantError::AccountingPoisoned {
                        account: "exact final publisher attribution transfer",
                    });
                }
                external_bytes.set(external_without_control);
                retained_bytes.set(control_bytes);
                if let Some(state) = spill_state {
                    state.set_usage(total);
                }
                Ok(())
            }
        }
    }

    /// Completes the sealed publication hand-off. No caller code can run
    /// between the successful transfer and this step: `publisher.publish` is
    /// allocation-free and infallible, so clearing operator attribution is an
    /// infallible assignment rather than a second fallible protocol edge.
    fn finish_control_transfer(&self, control_bytes: usize) {
        if self.reject_if_unacknowledged_retained_poisoned().is_err() {
            return;
        }
        if let ExternalSortGrantObserverTarget::Accounted {
            external_bytes,
            retained_bytes,
            spill_state,
        } = &self.target
        {
            debug_assert_eq!(retained_bytes.get(), control_bytes);
            retained_bytes.set(0);
            if let Some(state) = spill_state {
                state.set_usage(external_bytes.get());
            }
        }
    }
}

/// Checked view over exact-cursor child grants while one nested operation is
/// allowed to resize a single reader or row child. The stable sorter component
/// is captured only for that operation; all mutable aggregate components live
/// in `Cell`s owned by the cursor and are updated before publication.
pub(crate) struct ExactOwnedGrantTransitionObserver<'a, 'observer> {
    observer: &'a ExternalSortGrantObserver<'observer>,
    stable_sorter_bytes: usize,
    reader_bytes: &'a Cell<usize>,
    row_bytes: &'a Cell<usize>,
}

impl ExactOwnedGrantTransitionObserver<'_, '_> {
    fn replace_component(
        &self,
        component: &Cell<usize>,
        previous: usize,
        current: usize,
        account: &'static str,
    ) -> Result<(), MemoryGrantError> {
        let aggregate = component.get();
        let Some(without_previous) = aggregate.checked_sub(previous) else {
            self.publish_unrepresentable();
            return Err(MemoryGrantError::AccountingUnderflow {
                account,
                accounted_bytes: aggregate,
                release_bytes: previous,
            });
        };
        let Some(replacement) = without_previous.checked_add(current) else {
            self.publish_unrepresentable();
            return Err(MemoryGrantError::ArithmeticOverflow {
                current_bytes: without_previous,
                additional_bytes: current,
            });
        };
        component.set(replacement);
        self.publish()
    }

    pub(crate) fn replace_reader(
        &self,
        previous: usize,
        current: usize,
    ) -> Result<(), MemoryGrantError> {
        self.replace_component(
            self.reader_bytes,
            previous,
            current,
            "exact owned sort reader frontier",
        )
    }

    pub(crate) fn replace_row(
        &self,
        previous: usize,
        current: usize,
    ) -> Result<(), MemoryGrantError> {
        self.replace_component(
            self.row_bytes,
            previous,
            current,
            "exact owned sort row frontier",
        )
    }

    pub(crate) fn publish(&self) -> Result<(), MemoryGrantError> {
        let Some(children) = self.reader_bytes.get().checked_add(self.row_bytes.get()) else {
            self.publish_unrepresentable();
            return Err(MemoryGrantError::ArithmeticOverflow {
                current_bytes: self.reader_bytes.get(),
                additional_bytes: self.row_bytes.get(),
            });
        };
        self.observer
            .publish_sum(self.stable_sorter_bytes, children)
    }

    pub(crate) fn publish_unrepresentable(&self) {
        self.observer.publish_unrepresentable();
    }

    fn transfer_row_to_retained(&self, bytes: usize) -> Result<(), MemoryGrantError> {
        self.observer.transfer_row_to_retained(
            self.stable_sorter_bytes,
            self.reader_bytes.get(),
            self.row_bytes,
            bytes,
        )
    }
}

impl crate::execution::accounted_chunk::AccountedRowGrantObserver
    for ExactOwnedGrantTransitionObserver<'_, '_>
{
    fn replace_row(&self, previous: usize, current: usize) -> Result<(), MemoryGrantError> {
        Self::replace_row(self, previous, current)
    }
}

impl crate::execution::accounted_chunk::AccountedChunkGrantObserver
    for ExternalSortGrantObserver<'_>
{
    fn publish_retained(&self, bytes: usize) -> Result<(), MemoryGrantError> {
        Self::publish_retained(self, bytes)
    }

    fn publish_retained_preserving_primary(&self, bytes: usize) {
        Self::publish_retained_preserving_primary(self, bytes);
    }

    fn poison_unacknowledged_retained(&self) {
        Self::poison_unacknowledged_retained(self);
    }
}

impl super::file::ProviderGrantTransitionObserver for ExactOwnedGrantTransitionObserver<'_, '_> {
    fn replace_reader(&self, previous: usize, current: usize) -> Result<(), MemoryGrantError> {
        Self::replace_reader(self, previous, current)
    }

    fn replace_row(&self, previous: usize, current: usize) -> Result<(), MemoryGrantError> {
        Self::replace_row(self, previous, current)
    }

    fn publish_unrepresentable(&self) {
        Self::publish_unrepresentable(self);
    }
}

/// Structured internal failure for grant-qualified external-sort operations.
#[derive(Debug)]
pub(crate) enum ExternalSortOperationError {
    Cancelled(QueryCancellationError),
    CancelledWithCleanup {
        error: QueryCancellationError,
        cleanup: std::io::Error,
        phase: &'static str,
    },
    Memory(MemoryGrantError),
    MemoryWithCleanup {
        error: MemoryGrantError,
        cleanup: std::io::Error,
        phase: &'static str,
    },
    Allocation(std::io::Error),
    Io(std::io::Error),
    WithGrantRelease {
        primary: ExternalSortPrimary,
        release: MemoryGrantError,
        cleanup: Option<std::io::Error>,
        phase: &'static str,
    },
}

/// Allocation-free primary classification for compound spill failures.
#[derive(Debug)]
pub(crate) enum ExternalSortPrimary {
    Cancelled(QueryCancellationError),
    Memory(MemoryGrantError),
    Allocation(std::io::Error),
    Io(std::io::Error),
}

impl ExternalSortPrimary {
    fn into_io(self) -> std::io::Error {
        match self {
            Self::Cancelled(error) => cancellation_io_error(error),
            Self::Memory(error) => std::io::Error::new(std::io::ErrorKind::OutOfMemory, error),
            Self::Allocation(error) | Self::Io(error) => error,
        }
    }

    fn into_operation(self) -> ExternalSortOperationError {
        match self {
            Self::Cancelled(error) => ExternalSortOperationError::Cancelled(error),
            Self::Memory(error) => ExternalSortOperationError::Memory(error),
            Self::Allocation(error) => ExternalSortOperationError::Allocation(error),
            Self::Io(error) => ExternalSortOperationError::Io(error),
        }
    }
}

impl std::fmt::Display for ExternalSortPrimary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled(error) => std::fmt::Display::fmt(error, formatter),
            Self::Memory(error) => std::fmt::Display::fmt(error, formatter),
            Self::Allocation(error) | Self::Io(error) => std::fmt::Display::fmt(error, formatter),
        }
    }
}

impl ExternalSortOperationError {
    fn into_io(self) -> std::io::Error {
        match self {
            Self::Cancelled(error) => cancellation_io_error(error),
            Self::CancelledWithCleanup {
                error,
                cleanup,
                phase,
            } => super::combine_primary_and_cleanup(cancellation_io_error(error), cleanup, phase),
            Self::Memory(error) => std::io::Error::new(std::io::ErrorKind::OutOfMemory, error),
            Self::MemoryWithCleanup {
                error,
                cleanup,
                phase,
            } => super::combine_primary_and_cleanup(
                std::io::Error::new(std::io::ErrorKind::OutOfMemory, error),
                cleanup,
                phase,
            ),
            Self::Allocation(error) | Self::Io(error) => error,
            Self::WithGrantRelease {
                primary,
                release,
                cleanup,
                phase,
            } => {
                let error = super::combine_primary_and_cleanup(
                    primary.into_io(),
                    std::io::Error::other(release),
                    phase,
                );
                match cleanup {
                    Some(cleanup) => {
                        super::combine_primary_and_cleanup(error, cleanup, "sort spill cleanup")
                    }
                    None => error,
                }
            }
        }
    }

    fn workspace_preparation(error: std::io::Error) -> Self {
        if error.kind() == std::io::ErrorKind::OutOfMemory {
            Self::Allocation(error)
        } else {
            Self::Io(error)
        }
    }

    fn into_side_effect_free_primary(self) -> ExternalSortPrimary {
        match self {
            Self::Cancelled(error) => ExternalSortPrimary::Cancelled(error),
            Self::Memory(error) => ExternalSortPrimary::Memory(error),
            Self::Allocation(error) => ExternalSortPrimary::Allocation(error),
            Self::Io(error) => ExternalSortPrimary::Io(error),
            Self::CancelledWithCleanup { .. }
            | Self::MemoryWithCleanup { .. }
            | Self::WithGrantRelease { .. } => {
                unreachable!("side-effect-free sizing cannot carry cleanup failure")
            }
        }
    }
}

impl std::fmt::Display for ExternalSortOperationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled(error) => std::fmt::Display::fmt(error, formatter),
            Self::CancelledWithCleanup {
                error,
                cleanup,
                phase,
            } => write!(formatter, "{error}; {phase} also failed: {cleanup}"),
            Self::Memory(error) => std::fmt::Display::fmt(error, formatter),
            Self::MemoryWithCleanup {
                error,
                cleanup,
                phase,
            } => write!(formatter, "{error}; {phase} also failed: {cleanup}"),
            Self::Allocation(error) | Self::Io(error) => std::fmt::Display::fmt(error, formatter),
            Self::WithGrantRelease {
                primary,
                release,
                cleanup,
                phase,
            } => {
                write!(formatter, "{primary}; {phase} also failed: {release}")?;
                if let Some(cleanup) = cleanup {
                    write!(formatter, "; sort spill cleanup also failed: {cleanup}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ExternalSortOperationError {}

fn cancellation_io_error(error: QueryCancellationError) -> std::io::Error {
    let kind = match error {
        QueryCancellationError::Cancelled => std::io::ErrorKind::Interrupted,
        QueryCancellationError::DeadlineExceeded { .. } => std::io::ErrorKind::TimedOut,
        QueryCancellationError::InvalidState { .. } => std::io::ErrorKind::Other,
    };
    std::io::Error::new(kind, error)
}

fn with_io_cleanup(
    primary: ExternalSortOperationError,
    cleanup: std::io::Error,
    context: &'static str,
) -> ExternalSortOperationError {
    match primary {
        ExternalSortOperationError::Cancelled(error) => {
            ExternalSortOperationError::CancelledWithCleanup {
                error,
                cleanup,
                phase: context,
            }
        }
        ExternalSortOperationError::CancelledWithCleanup {
            error,
            cleanup: existing,
            phase,
        } => ExternalSortOperationError::CancelledWithCleanup {
            error,
            cleanup: super::combine_primary_and_cleanup(existing, cleanup, context),
            phase,
        },
        ExternalSortOperationError::Memory(error) => {
            ExternalSortOperationError::MemoryWithCleanup {
                error,
                cleanup,
                phase: context,
            }
        }
        ExternalSortOperationError::MemoryWithCleanup {
            error,
            cleanup: existing,
            phase,
        } => ExternalSortOperationError::MemoryWithCleanup {
            error,
            cleanup: super::combine_primary_and_cleanup(existing, cleanup, context),
            phase,
        },
        ExternalSortOperationError::Allocation(error) => ExternalSortOperationError::Allocation(
            super::combine_primary_and_cleanup(error, cleanup, context),
        ),
        ExternalSortOperationError::Io(error) => ExternalSortOperationError::Io(
            super::combine_primary_and_cleanup(error, cleanup, context),
        ),
        ExternalSortOperationError::WithGrantRelease {
            primary,
            release,
            cleanup: existing,
            phase,
        } => ExternalSortOperationError::WithGrantRelease {
            primary,
            release,
            cleanup: Some(match existing {
                Some(existing) => super::combine_primary_and_cleanup(existing, cleanup, context),
                None => cleanup,
            }),
            phase,
        },
    }
}

fn with_writer_release_failure(
    primary: ExternalSortPrimary,
    release: Option<MemoryGrantError>,
    cleanup: Option<std::io::Error>,
    phase: &'static str,
) -> ExternalSortOperationError {
    match release {
        Some(release) => ExternalSortOperationError::WithGrantRelease {
            primary,
            release,
            cleanup,
            phase,
        },
        None => {
            let primary = primary.into_operation();
            match cleanup {
                Some(cleanup) => with_io_cleanup(primary, cleanup, "sort spill cleanup"),
                None => primary,
            }
        }
    }
}

impl From<MemoryGrantError> for ExternalSortOperationError {
    fn from(error: MemoryGrantError) -> Self {
        Self::Memory(error)
    }
}

impl From<QueryCancellationError> for ExternalSortOperationError {
    fn from(error: QueryCancellationError) -> Self {
        Self::Cancelled(error)
    }
}

impl From<std::io::Error> for ExternalSortOperationError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

fn check_cancellation(
    cancellation: Option<&QueryCancellationToken>,
) -> Result<(), QueryCancellationError> {
    match cancellation {
        Some(cancellation) => cancellation.check(),
        None => Ok(()),
    }
}

/// Reusable codec allocations plus their optional query-memory capability.
///
/// Physical allocations are declared before the grant so ordinary field drop
/// releases their backing storage before the accounting token is released.
struct ExternalSortWorkspace {
    row_staging: SpillRecordBuffer,
    counter_scratch: CounterSortScratch,
    #[cfg(test)]
    release_error: Option<MemoryGrantError>,
    grant: Option<MemoryGrant>,
}

impl ExternalSortWorkspace {
    fn new(maximum_record_bytes: usize, grant: Option<MemoryGrant>) -> Self {
        Self {
            row_staging: SpillRecordBuffer::new(maximum_record_bytes),
            counter_scratch: CounterSortScratch::new(),
            #[cfg(test)]
            release_error: None,
            grant,
        }
    }

    fn granted_bytes(&self) -> usize {
        self.grant.as_ref().map_or(0, MemoryGrant::size)
    }

    fn observed_bytes(&self) -> Result<usize, ExternalSortOperationError> {
        let row_bytes = self.row_staging.capacity();
        let counter_bytes = self
            .counter_scratch
            .observed_capacity_bytes()
            .map_err(ExternalSortOperationError::workspace_preparation)?;
        checked_workspace_sum(row_bytes, counter_bytes).map_err(Into::into)
    }

    fn resize_grant(&mut self, bytes: usize) -> Result<(), ExternalSortOperationError> {
        if let Some(grant) = self.grant.as_mut() {
            grant.try_resize(bytes)?;
        }
        Ok(())
    }

    fn rollback_grant(
        &mut self,
        previous_bytes: usize,
        primary: ExternalSortOperationError,
    ) -> ExternalSortOperationError {
        match self.resize_grant(previous_bytes) {
            Ok(()) => primary,
            Err(accounting) => accounting,
        }
    }

    fn prepare_row_staging(
        &mut self,
        required_bytes: usize,
    ) -> Result<(), ExternalSortOperationError> {
        if self.grant.is_none() || required_bytes <= self.row_staging.capacity() {
            self.row_staging.prepare_record(required_bytes)?;
            return Ok(());
        }

        let previous_bytes = self.observed_bytes()?;
        let provisional = checked_workspace_sum(previous_bytes, required_bytes)?;
        self.resize_grant(provisional)?;

        let mut replacement = SpillRecordBuffer::new(self.row_staging.maximum());
        if let Err(error) = replacement.prepare_record(required_bytes) {
            drop(replacement);
            return Err(self.rollback_grant(
                previous_bytes,
                ExternalSortOperationError::workspace_preparation(error),
            ));
        }
        let replacement_bytes = replacement.capacity();
        let observed_peak = match checked_workspace_sum(previous_bytes, replacement_bytes) {
            Ok(bytes) => bytes,
            Err(error) => {
                drop(replacement);
                return Err(self.rollback_grant(previous_bytes, error.into()));
            }
        };
        if observed_peak > provisional
            && let Err(error) = self.resize_grant(observed_peak)
        {
            drop(replacement);
            return Err(self.rollback_grant(previous_bytes, error));
        }

        let old = std::mem::replace(&mut self.row_staging, replacement);
        drop(old);
        let observed = self.observed_bytes()?;
        self.resize_grant(observed)
    }

    fn prepare_counter_scratch(
        &mut self,
        required_entries: usize,
        required_key_bytes: usize,
    ) -> Result<(), ExternalSortOperationError> {
        let current = self.counter_scratch.observed_capacity();
        if self.grant.is_none()
            || (required_entries <= current.entry_capacity
                && required_key_bytes <= current.key_capacity)
        {
            self.counter_scratch
                .prepare(required_entries, required_key_bytes)?;
            return Ok(());
        }

        let previous_bytes = self.observed_bytes()?;
        let requested =
            CounterSortScratch::requested_capacity_bytes(required_entries, required_key_bytes)
                .map_err(ExternalSortOperationError::workspace_preparation)?;
        let provisional = checked_workspace_sum(previous_bytes, requested)?;
        self.resize_grant(provisional)?;

        let mut replacement = CounterSortScratch::new();
        if let Err(error) = replacement.prepare(required_entries, required_key_bytes) {
            drop(replacement);
            return Err(self.rollback_grant(
                previous_bytes,
                ExternalSortOperationError::workspace_preparation(error),
            ));
        }
        let replacement_bytes = match replacement.observed_capacity_bytes() {
            Ok(bytes) => bytes,
            Err(error) => {
                drop(replacement);
                return Err(self.rollback_grant(
                    previous_bytes,
                    ExternalSortOperationError::workspace_preparation(error),
                ));
            }
        };
        let observed_peak = match checked_workspace_sum(previous_bytes, replacement_bytes) {
            Ok(bytes) => bytes,
            Err(error) => {
                drop(replacement);
                return Err(self.rollback_grant(previous_bytes, error.into()));
            }
        };
        if observed_peak > provisional
            && let Err(error) = self.resize_grant(observed_peak)
        {
            drop(replacement);
            return Err(self.rollback_grant(previous_bytes, error));
        }

        let old = std::mem::replace(&mut self.counter_scratch, replacement);
        drop(old);
        let observed = self.observed_bytes()?;
        self.resize_grant(observed)
    }

    fn prepare(
        &mut self,
        required_row_bytes: usize,
        required_counter_entries: usize,
        required_counter_key_bytes: usize,
    ) -> Result<(), ExternalSortOperationError> {
        self.prepare_row_staging(required_row_bytes)?;
        self.prepare_counter_scratch(required_counter_entries, required_counter_key_bytes)
    }

    fn release_capacity(&mut self) -> Result<(), MemoryGrantError> {
        self.row_staging.discard_capacity();
        self.counter_scratch.discard_capacity();
        #[cfg(test)]
        if self.granted_bytes() != 0
            && let Some(error) = &self.release_error
        {
            return Err(error.clone());
        }
        if let Some(grant) = self.grant.as_mut() {
            grant.try_resize(0)?;
        }
        Ok(())
    }
}

/// Query-memory ownership for the fixed synchronous spill-writer backing.
///
/// No physical buffer survives an operation. A failed release deliberately
/// leaves the exact non-zero token here so telemetry and later cleanup remain
/// conservative and retryable.
struct ExternalSortWriterWorkspace {
    grant: Option<MemoryGrant>,
    retain_failure: bool,
    failure_retained: bool,
    hook_authority: Option<AccountedError>,
    scalar_cleanup: Option<AccountedError>,
    scalar_retry_file: Option<SpillFile>,
    #[cfg(test)]
    release_error: Option<MemoryGrantError>,
    #[cfg(test)]
    prepared_capacity: Option<usize>,
}

impl ExternalSortWriterWorkspace {
    fn new(grant: Option<MemoryGrant>) -> Self {
        Self {
            grant,
            retain_failure: false,
            failure_retained: false,
            hook_authority: None,
            scalar_cleanup: None,
            scalar_retry_file: None,
            #[cfg(test)]
            release_error: None,
            #[cfg(test)]
            prepared_capacity: None,
        }
    }

    fn granted_bytes(&self) -> usize {
        self.grant.as_ref().map_or(0, MemoryGrant::size)
    }

    fn resize_grant(&mut self, bytes: usize) -> Result<(), MemoryGrantError> {
        if let Some(grant) = self.grant.as_mut() {
            grant.try_resize(bytes)?;
        }
        Ok(())
    }

    fn release_capacity(&mut self) -> Result<(), MemoryGrantError> {
        if self.granted_bytes() == 0 {
            return Ok(());
        }
        #[cfg(test)]
        if let Some(error) = &self.release_error {
            return Err(error.clone());
        }
        self.resize_grant(0)
    }
}

/// Physical ownership state for one writer publication attempt.
enum WriterPublicationState {
    Empty,
    Prepared(SpillWriterBuffer),
    File(SpillFile),
}

/// Couples one transient writer backing to its retained query-memory token.
///
/// `state` is deliberately declared first. The explicit `Drop` backstop also
/// removes a buffer from a live `SpillFile` before attempting grant release,
/// including while unwinding through hostile provider or I/O callbacks.
struct WriterPublication<'a> {
    state: WriterPublicationState,
    workspace: &'a mut ExternalSortWriterWorkspace,
    retained_base_bytes: usize,
    armed: bool,
}

impl<'a> WriterPublication<'a> {
    fn new(workspace: &'a mut ExternalSortWriterWorkspace) -> Self {
        Self {
            state: WriterPublicationState::Empty,
            workspace,
            retained_base_bytes: 0,
            armed: true,
        }
    }

    fn granted_bytes(&self) -> usize {
        self.workspace.granted_bytes()
    }

    fn prepare(&mut self, manager: &SpillManager) -> Result<(), ExternalSortOperationError> {
        debug_assert!(matches!(self.state, WriterPublicationState::Empty));
        debug_assert!(self.workspace.grant.is_some());

        let declared_workspace = if self.workspace.retain_failure {
            // Even a provider whose declaration changes between preflight and
            // this operation must fail without constructing an unadmitted box.
            manager
                .qualified_sort_provider_workspace_bound()
                .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::Unsupported))
        } else {
            manager.qualified_file_workspace_bound()
        };
        let file_workspace = match declared_workspace {
            Ok(bytes) => bytes,
            Err(error) => {
                return Err(self.fail_preparation(ExternalSortPrimary::Io(error)));
            }
        };
        let buffer_requested = qualified_writer_buffer_requested_bytes();
        let requested_capacity = {
            #[cfg(test)]
            {
                self.workspace.prepared_capacity.unwrap_or(buffer_requested)
            }
            #[cfg(not(test))]
            {
                buffer_requested
            }
        };
        let requested = match checked_workspace_sum(requested_capacity, file_workspace) {
            Ok(bytes) => bytes,
            Err(error) => {
                return Err(self.fail_preparation(ExternalSortPrimary::Memory(error)));
            }
        };
        let provisional = self.workspace.granted_bytes().max(requested);
        if let Err(error) = self.workspace.resize_grant(provisional) {
            return Err(self.fail_preparation(ExternalSortPrimary::Memory(error)));
        }

        let preparation = SpillWriterBuffer::prepare_with_capacity(requested_capacity);
        let buffer = match preparation {
            Ok(buffer) => buffer,
            Err(error) => {
                return Err(self.fail_preparation(ExternalSortPrimary::Allocation(error)));
            }
        };
        let observed = match checked_workspace_sum(buffer.capacity(), file_workspace) {
            Ok(observed) => observed,
            Err(error) => {
                drop(buffer);
                return Err(self.fail_preparation(ExternalSortPrimary::Memory(error)));
            }
        };
        self.state = WriterPublicationState::Prepared(buffer);
        if observed > self.workspace.granted_bytes()
            && let Err(error) = self.workspace.resize_grant(observed)
        {
            return Err(self.fail_preparation(ExternalSortPrimary::Memory(error)));
        }
        self.retained_base_bytes = observed;
        Ok(())
    }

    fn write_sort_row_accounted(
        &mut self,
        payload: &[u8],
        other_granted_bytes: usize,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortPrimary> {
        let retained_base_bytes = self.retained_base_bytes;
        let workspace = &mut *self.workspace;
        let file = match &mut self.state {
            WriterPublicationState::File(file) => file,
            WriterPublicationState::Empty | WriterPublicationState::Prepared(_) => {
                return Err(ExternalSortPrimary::Io(std::io::Error::other(
                    "spill writer publication has no file",
                )));
            }
        };
        let mut admission_failure = None;
        let write = file.write_sort_row_with_admission(payload, |required| {
            let requested = match checked_workspace_sum(retained_base_bytes, required) {
                Ok(requested) => requested.max(workspace.granted_bytes()),
                Err(error) => {
                    admission_failure = Some(error);
                    return Err(std::io::Error::from(std::io::ErrorKind::OutOfMemory));
                }
            };
            match workspace.resize_grant(requested) {
                Ok(()) => {
                    match observer.publish_sum(other_granted_bytes, workspace.granted_bytes()) {
                        Ok(()) => Ok(()),
                        Err(error) => {
                            admission_failure = Some(error);
                            Err(std::io::Error::from(std::io::ErrorKind::OutOfMemory))
                        }
                    }
                }
                Err(error) => {
                    admission_failure = Some(error);
                    let _ = observer.publish_sum(other_granted_bytes, workspace.granted_bytes());
                    Err(std::io::Error::from(std::io::ErrorKind::OutOfMemory))
                }
            }
        });
        match (write, admission_failure) {
            (_, Some(error)) => Err(ExternalSortPrimary::Memory(error)),
            (Ok(()), None) => Ok(()),
            (Err(error), None) => Err(ExternalSortPrimary::Io(error)),
        }
    }

    fn fail_preparation(&mut self, primary: ExternalSortPrimary) -> ExternalSortOperationError {
        self.workspace.failure_retained = self.workspace.retain_failure;
        self.discard_writer_backing();
        let release = self.release_charge().err();
        with_writer_release_failure(primary, release, None, "writer preparation rollback")
    }

    fn create(
        &mut self,
        manager: &SpillManager,
        role: SpillFileRole,
        other_granted_bytes: usize,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortPrimary> {
        let buffer = match std::mem::replace(&mut self.state, WriterPublicationState::Empty) {
            WriterPublicationState::Prepared(buffer) => buffer,
            WriterPublicationState::Empty | WriterPublicationState::File(_) => {
                return Err(ExternalSortPrimary::Io(std::io::Error::other(
                    "spill writer publication is not prepared",
                )));
            }
        };
        let buffer_bytes = buffer.capacity();
        let workspace = &mut *self.workspace;
        let mut admission_failure = None;
        let file = manager.create_qualified_file_with_writer_buffer(role, buffer, |required| {
            let requested = match checked_workspace_sum(buffer_bytes, required) {
                Ok(requested) => requested.max(workspace.granted_bytes()),
                Err(error) => {
                    admission_failure = Some(error);
                    return Err(std::io::Error::from(std::io::ErrorKind::OutOfMemory));
                }
            };
            match workspace.resize_grant(requested) {
                Ok(()) => {
                    match observer.publish_sum(other_granted_bytes, workspace.granted_bytes()) {
                        Ok(()) => Ok(()),
                        Err(error) => {
                            admission_failure = Some(error);
                            Err(std::io::Error::from(std::io::ErrorKind::OutOfMemory))
                        }
                    }
                }
                Err(error) => {
                    admission_failure = Some(error);
                    let _ = observer.publish_sum(other_granted_bytes, workspace.granted_bytes());
                    Err(std::io::Error::from(std::io::ErrorKind::OutOfMemory))
                }
            }
        });
        match (file, admission_failure) {
            (_, Some(error)) => Err(ExternalSortPrimary::Memory(error)),
            (Ok(file), None) => {
                self.state = WriterPublicationState::File(file);
                Ok(())
            }
            (Err(error), None) => Err(ExternalSortPrimary::Io(error)),
        }
    }

    fn file_mut(&mut self) -> &mut SpillFile {
        match &mut self.state {
            WriterPublicationState::File(file) => file,
            WriterPublicationState::Empty | WriterPublicationState::Prepared(_) => {
                panic!("spill writer publication has no file")
            }
        }
    }

    fn finish_and_release(
        &mut self,
        cancellation: Option<&QueryCancellationToken>,
    ) -> Result<(), ExternalSortOperationError> {
        let mut retry_file = None;
        let result = self.finish_and_release_retaining_file(cancellation, &mut retry_file);
        if retry_file.is_some() {
            quarantine_pull_sort_hook(self.workspace.hook_authority.as_ref());
        }
        drop(retry_file);
        result
    }

    fn finish_and_release_retaining_file(
        &mut self,
        cancellation: Option<&QueryCancellationToken>,
        retry_file: &mut Option<SpillFile>,
    ) -> Result<(), ExternalSortOperationError> {
        debug_assert!(retry_file.is_none());
        let mut release_succeeded = false;
        let outcome = match &mut self.state {
            WriterPublicationState::File(file) => file.finish_write_before_publish(|| {
                self.workspace
                    .release_capacity()
                    .map_err(ExternalSortPrimary::Memory)?;
                release_succeeded = true;
                check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)
            }),
            WriterPublicationState::Empty | WriterPublicationState::Prepared(_) => {
                return Err(std::io::Error::other(
                    "spill writer publication has no file to finish",
                )
                .into());
            }
        };
        if release_succeeded {
            self.armed = false;
        }
        match outcome {
            Ok(()) => Ok(()),
            Err(SpillFinishError::Io(error)) => {
                Err(self.abort_io_retaining_file(error, "writer finish release", retry_file))
            }
            Err(SpillFinishError::BeforePublish(ExternalSortPrimary::Memory(error))) => {
                if self.workspace.scalar_cleanup.is_some() {
                    return Err(self.abort_primary_retaining_file(
                        ExternalSortPrimary::Memory(error),
                        "writer finish release",
                        retry_file,
                    ));
                }
                let (cleanup, retained_file) = self.cleanup_file_retaining_failure();
                *retry_file = retained_file;
                let primary = ExternalSortOperationError::Memory(error);
                let error = match cleanup {
                    Some(cleanup) => with_io_cleanup(primary, cleanup, "sort spill cleanup"),
                    None => primary,
                };
                Err(error)
            }
            Err(SpillFinishError::BeforePublish(primary)) => Err(self
                .abort_primary_retaining_file(primary, "writer cancellation release", retry_file)),
        }
    }

    fn abort_io_retaining_file(
        &mut self,
        primary: std::io::Error,
        release_phase: &'static str,
        retry_file: &mut Option<SpillFile>,
    ) -> ExternalSortOperationError {
        self.abort_primary_retaining_file(
            ExternalSortPrimary::Io(primary),
            release_phase,
            retry_file,
        )
    }

    fn abort_primary(
        &mut self,
        primary: ExternalSortPrimary,
        release_phase: &'static str,
    ) -> ExternalSortOperationError {
        let mut retry_file = None;
        let error = self.abort_primary_retaining_file(primary, release_phase, &mut retry_file);
        if retry_file.is_some() {
            quarantine_pull_sort_hook(self.workspace.hook_authority.as_ref());
        }
        drop(retry_file);
        error
    }

    fn abort_primary_retaining_file(
        &mut self,
        primary: ExternalSortPrimary,
        release_phase: &'static str,
        retry_file: &mut Option<SpillFile>,
    ) -> ExternalSortOperationError {
        debug_assert!(retry_file.is_none());
        if self.workspace.scalar_cleanup.is_some() {
            self.workspace.failure_retained = true;
            *retry_file = self.take_scalar_failed_file();
            self.armed = false;
            return primary.into_operation();
        }
        self.workspace.failure_retained = self.workspace.retain_failure;
        let (cleanup, retained_file) = self.cleanup_file_retaining_failure();
        *retry_file = retained_file;
        let release = self.release_charge().err();
        with_writer_release_failure(primary, release, cleanup, release_phase)
    }

    fn abort_operation_retaining_file(
        &mut self,
        mut primary: ExternalSortOperationError,
        release_phase: &'static str,
        retry_file: &mut Option<SpillFile>,
    ) -> ExternalSortOperationError {
        debug_assert!(retry_file.is_none());
        if self.workspace.scalar_cleanup.is_some() {
            self.workspace.failure_retained = true;
            *retry_file = self.take_scalar_failed_file();
            self.armed = false;
            return primary;
        }
        self.workspace.failure_retained = self.workspace.retain_failure;
        let (cleanup, retained_file) = self.cleanup_file_retaining_failure();
        *retry_file = retained_file;
        if let Err(release) = self.release_charge() {
            primary = with_io_cleanup(primary, std::io::Error::other(release), release_phase);
        }
        match cleanup {
            Some(cleanup) => with_io_cleanup(primary, cleanup, "sort spill cleanup"),
            None => primary,
        }
    }

    fn take_scalar_failed_file(&mut self) -> Option<SpillFile> {
        match std::mem::replace(&mut self.state, WriterPublicationState::Empty) {
            WriterPublicationState::File(mut file) => {
                file.discard_writer_backing();
                Some(file)
            }
            WriterPublicationState::Prepared(buffer) => {
                drop(buffer);
                None
            }
            WriterPublicationState::Empty => None,
        }
    }

    fn cleanup_file_retaining_failure(&mut self) -> (Option<std::io::Error>, Option<SpillFile>) {
        let cleanup = match &mut self.state {
            WriterPublicationState::Empty => None,
            WriterPublicationState::Prepared(_) => {
                self.discard_writer_backing();
                None
            }
            WriterPublicationState::File(file) => {
                file.discard_writer_backing();
                file.close_and_delete().err()
            }
        };
        let retry_file = if cleanup.is_some() {
            match std::mem::replace(&mut self.state, WriterPublicationState::Empty) {
                WriterPublicationState::File(file) => Some(file),
                WriterPublicationState::Empty | WriterPublicationState::Prepared(_) => {
                    unreachable!("only a spill file can report a deletion failure")
                }
            }
        } else {
            // Explicit deletion proved this owner retired. Remove it now so
            // the Drop backstop can distinguish success from an unproven file.
            let retired = std::mem::replace(&mut self.state, WriterPublicationState::Empty);
            drop(retired);
            None
        };
        (cleanup, retry_file)
    }

    fn discard_writer_backing(&mut self) {
        match &mut self.state {
            WriterPublicationState::Empty => {}
            WriterPublicationState::Prepared(_) => {
                let prepared = std::mem::replace(&mut self.state, WriterPublicationState::Empty);
                drop(prepared);
            }
            WriterPublicationState::File(file) => file.discard_writer_backing(),
        }
    }

    fn release_charge(&mut self) -> Result<(), MemoryGrantError> {
        self.discard_writer_backing();
        if self.workspace.failure_retained {
            self.armed = false;
            return Ok(());
        }
        if !self.armed {
            return Ok(());
        }
        match self.workspace.release_capacity() {
            Ok(()) => {
                self.armed = false;
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    fn take_file_after_release(mut self) -> SpillFile {
        assert!(
            !self.armed,
            "writer charge must be released before publication"
        );
        match std::mem::replace(&mut self.state, WriterPublicationState::Empty) {
            WriterPublicationState::File(file) => file,
            WriterPublicationState::Empty | WriterPublicationState::Prepared(_) => {
                panic!("finished spill writer publication has no file")
            }
        }
    }
}

impl Drop for WriterPublication<'_> {
    fn drop(&mut self) {
        if self.workspace.scalar_cleanup.is_some() {
            // During unwind, the outer scalar owner receives this exact staging
            // handle and its grant before any cleanup callback can run.
            if self.armed || !matches!(self.state, WriterPublicationState::Empty) {
                self.workspace.failure_retained = true;
                self.workspace.scalar_retry_file = self.take_scalar_failed_file();
            }
            return;
        }
        let physical = std::mem::replace(&mut self.state, WriterPublicationState::Empty);
        let retained_file = match physical {
            WriterPublicationState::Empty => None,
            WriterPublicationState::Prepared(buffer) => {
                drop(buffer);
                None
            }
            WriterPublicationState::File(mut file) => {
                file.discard_writer_backing();
                Some(file)
            }
        };
        if retained_file.is_some() {
            // SpillFile's Drop can forget a fresh cleanup error. The physical
            // owner was not proved deleted before this implicit retirement.
            quarantine_pull_sort_hook(self.workspace.hook_authority.as_ref());
        }
        let _ = super::run_cleanup_backstop(|| {
            drop(retained_file);
            Ok::<(), std::convert::Infallible>(())
        });
        if self.armed {
            if self.workspace.retain_failure && std::thread::panicking() {
                // The opaque unwind payload can outlive this sorter. Its
                // complete admitted workspace remains charged fail-closed.
                if let Some(grant) = self.workspace.grant.take() {
                    std::mem::forget(grant);
                }
            } else {
                let _ = super::run_cleanup_backstop(|| self.workspace.release_capacity());
            }
        }
    }
}

/// One atomically published run and its declared row count.
struct ExternalSortRunEntry {
    file: SpillFile,
    rows: usize,
}

/// Pinned allocator-backed catalog storage whose requested layout is its
/// reported capacity. Qualified growth checks this contract before moving any
/// published run into a replacement allocation.
type RunCatalogEntries = ExactVec<ExternalSortRunEntry, Global>;

/// Paired run metadata plus its optional resident-memory capability.
///
/// Entries are declared before the grant so file handles and the catalog
/// allocation are physically released before their accounting token.
struct ExternalSortRunCatalog {
    entries: RunCatalogEntries,
    #[cfg(test)]
    forced_next_observed_capacity: Option<usize>,
    #[cfg(test)]
    forced_next_shrink_failure: Option<MemoryGrantError>,
    grant: Option<MemoryGrant>,
}

impl ExternalSortRunCatalog {
    fn new(grant: Option<MemoryGrant>) -> Self {
        Self {
            entries: RunCatalogEntries::new_in(Global),
            #[cfg(test)]
            forced_next_observed_capacity: None,
            #[cfg(test)]
            forced_next_shrink_failure: None,
            grant,
        }
    }

    fn allocate_exact_entries(
        target_capacity: usize,
    ) -> Result<RunCatalogEntries, ExternalSortOperationError> {
        let mut entries = RunCatalogEntries::new_in(Global);
        if entries.try_reserve_exact(target_capacity).is_err() {
            // The unretained allocator error expires with the condition. Drop
            // the empty physical allocation before the caller rolls back the
            // catalog's provisional authority.
            drop(entries);
            return Err(ExternalSortOperationError::workspace_preparation(
                std::io::Error::from(std::io::ErrorKind::OutOfMemory),
            ));
        }
        Ok(entries)
    }

    fn observed_replacement_capacity(&mut self, actual_capacity: usize) -> usize {
        #[cfg(test)]
        if let Some(forced) = self.forced_next_observed_capacity.take() {
            return forced;
        }
        actual_capacity
    }

    #[cfg(test)]
    fn force_next_observed_capacity_for_test(&mut self, capacity: usize) {
        self.forced_next_observed_capacity = Some(capacity);
    }

    #[cfg(test)]
    fn force_next_shrink_failure_for_test(&mut self, error: MemoryGrantError) {
        self.forced_next_shrink_failure = Some(error);
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn get(&self, index: usize) -> &ExternalSortRunEntry {
        &self.entries[index]
    }

    fn iter(&self) -> impl Iterator<Item = &ExternalSortRunEntry> {
        self.entries.iter()
    }

    fn granted_bytes(&self) -> usize {
        self.grant.as_ref().map_or(0, MemoryGrant::size)
    }

    fn observed_bytes(&self) -> Result<usize, ExternalSortOperationError> {
        run_catalog_capacity_bytes(self.entries.capacity())
    }

    fn resize_grant(&mut self, bytes: usize) -> Result<(), ExternalSortOperationError> {
        #[cfg(test)]
        if self
            .grant
            .as_ref()
            .is_some_and(|grant| bytes < grant.size())
            && let Some(error) = self.forced_next_shrink_failure.take()
        {
            return Err(error.into());
        }
        if let Some(grant) = self.grant.as_mut() {
            grant.try_resize(bytes)?;
        }
        Ok(())
    }

    fn rollback_grant(
        &mut self,
        previous_bytes: usize,
        primary: ExternalSortOperationError,
    ) -> ExternalSortOperationError {
        match self.resize_grant(previous_bytes) {
            Ok(()) => primary,
            Err(accounting) => accounting,
        }
    }

    fn reconcile_grant_with_observed_capacity(&mut self) -> Result<(), ExternalSortOperationError> {
        let Some(granted_bytes) = self.grant.as_ref().map(MemoryGrant::size) else {
            return Ok(());
        };
        let observed_bytes = self.observed_bytes()?;
        match granted_bytes.cmp(&observed_bytes) {
            Ordering::Greater => self.resize_grant(observed_bytes),
            Ordering::Equal => Ok(()),
            // A physical catalog may never gain authority after observation.
            // Fence an impossible under-covered state without allocating an
            // error payload or changing either the grant or the catalog.
            Ordering::Less => Err(ExternalSortOperationError::Io(std::io::Error::from(
                std::io::ErrorKind::InvalidData,
            ))),
        }
    }

    fn next_capacity(&self, required: usize) -> Result<usize, ExternalSortOperationError> {
        if required <= self.entries.capacity() {
            return Ok(self.entries.capacity());
        }
        if self.entries.capacity() == 0 {
            return Ok(required.max(INITIAL_RUN_CATALOG_CAPACITY));
        }
        let doubled = self.entries.capacity().checked_mul(2).ok_or_else(|| {
            ExternalSortOperationError::workspace_preparation(std::io::Error::from(
                std::io::ErrorKind::OutOfMemory,
            ))
        })?;
        Ok(doubled.max(required))
    }

    fn prepare_push(&mut self) -> Result<(), ExternalSortOperationError> {
        self.reconcile_grant_with_observed_capacity()?;
        let required = self.entries.len().checked_add(1).ok_or_else(|| {
            ExternalSortOperationError::workspace_preparation(std::io::Error::from(
                std::io::ErrorKind::OutOfMemory,
            ))
        })?;
        if required <= self.entries.capacity() {
            return Ok(());
        }
        if self.grant.is_none() {
            self.entries.try_reserve(1).map_err(|error| {
                ExternalSortOperationError::workspace_preparation(std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    format!("reserve sort run catalog: {error}"),
                ))
            })?;
            return Ok(());
        }

        let previous_bytes = self.observed_bytes()?;
        let target_capacity = self.next_capacity(required)?;
        let requested_bytes = run_catalog_capacity_bytes(target_capacity)?;
        let provisional = checked_workspace_sum(previous_bytes, requested_bytes)?;
        self.resize_grant(provisional)?;

        let replacement = match Self::allocate_exact_entries(target_capacity) {
            Ok(replacement) => replacement,
            Err(primary) => return Err(self.rollback_grant(previous_bytes, primary)),
        };
        let actual_capacity = replacement.capacity();
        let observed_capacity = self.observed_replacement_capacity(actual_capacity);
        if observed_capacity != target_capacity {
            // A violated pinned-allocator contract is fenced before published
            // entries move. Tear down the replacement before rolling back the
            // provisional peak grant, and keep the failure allocation-free.
            drop(replacement);
            let primary = ExternalSortOperationError::Io(std::io::Error::from(
                std::io::ErrorKind::InvalidData,
            ));
            return Err(self.rollback_grant(previous_bytes, primary));
        }

        debug_assert_eq!(actual_capacity, target_capacity);
        debug_assert!(target_capacity >= required);
        let mut old = std::mem::replace(&mut self.entries, replacement);
        self.entries.append(&mut old);
        drop(old);
        // Only a shrink remains after the old physical allocation is gone. A
        // failed shrink leaves the unchanged logical catalog fully covered by
        // the conservative old-plus-new grant; it never triggers another grow.
        self.resize_grant(requested_bytes)
    }

    fn push(&mut self, file: SpillFile, rows: usize) {
        assert!(
            self.entries.len() < self.entries.capacity(),
            "sort run publication requires pre-admitted catalog capacity"
        );
        self.entries.push(ExternalSortRunEntry { file, rows });
    }

    fn remove(&mut self, index: usize) {
        self.entries.remove(index);
    }

    /// Releases an already-empty catalog's physical backing before shrinking
    /// its authority. A failed shrink leaves the conservative grant in place
    /// and is retryable because the empty catalog remains a valid owner.
    fn release_empty_capacity(&mut self) -> Result<(), ExternalSortOperationError> {
        if !self.entries.is_empty() {
            return Err(ExternalSortOperationError::Io(std::io::Error::from(
                std::io::ErrorKind::InvalidInput,
            )));
        }
        let physical = std::mem::replace(&mut self.entries, RunCatalogEntries::new_in(Global));
        drop(physical);
        self.resize_grant(0)
    }

    #[cfg(test)]
    fn pointer(&self) -> *const ExternalSortRunEntry {
        self.entries.as_ptr()
    }

    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.entries.capacity()
    }
}

fn run_catalog_capacity_bytes(capacity: usize) -> Result<usize, ExternalSortOperationError> {
    std::alloc::Layout::array::<ExternalSortRunEntry>(capacity)
        .map(|layout| layout.size())
        .map_err(|_| {
            ExternalSortOperationError::workspace_preparation(std::io::Error::from(
                std::io::ErrorKind::OutOfMemory,
            ))
        })
}

fn checked_workspace_sum(
    current_bytes: usize,
    additional_bytes: usize,
) -> Result<usize, MemoryGrantError> {
    current_bytes
        .checked_add(additional_bytes)
        .ok_or(MemoryGrantError::ArithmeticOverflow {
            current_bytes,
            additional_bytes,
        })
}

fn cursor_frontier_bytes(
    base_bytes: usize,
    reader_bytes: usize,
    row_bytes: usize,
) -> Result<usize, MemoryGrantError> {
    checked_workspace_sum(base_bytes, checked_workspace_sum(reader_bytes, row_bytes)?)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the existing scalar admission boundary also carries the pull failure-lifetime policy"
)]
fn open_cursor_reader(
    file: &SpillFile,
    frontier_grant: &mut Option<MemoryGrant>,
    total_without_frontier: usize,
    frontier_base_bytes: usize,
    frontier_reader_bytes: usize,
    frontier_row_bytes: usize,
    observer: &ExternalSortGrantObserver<'_>,
    retain_failure: bool,
    scalar_cleanup: Option<&AccountedError>,
) -> Result<(SpillFileReader, usize), ExternalSortOperationError> {
    let Some(grant) = frontier_grant.as_mut() else {
        return file.reader().map(|reader| (reader, 0)).map_err(Into::into);
    };

    let previous = cursor_frontier_bytes(
        frontier_base_bytes,
        frontier_reader_bytes,
        frontier_row_bytes,
    )?;
    let mut admitted_bytes = 0usize;
    let mut admission_failure = None;
    let mut admit = |required| {
        let retained = admitted_bytes.max(required);
        let requested =
            checked_workspace_sum(frontier_reader_bytes, retained).and_then(|readers| {
                cursor_frontier_bytes(frontier_base_bytes, readers, frontier_row_bytes)
            });
        let requested = match requested {
            Ok(requested) => requested,
            Err(error) => {
                admission_failure = Some(ExternalSortOperationError::Memory(error));
                return Err(std::io::Error::from(std::io::ErrorKind::OutOfMemory));
            }
        };
        match grant.try_resize(requested) {
            Ok(()) => match observer.publish_sum(total_without_frontier, grant.size()) {
                Ok(()) => {
                    admitted_bytes = retained;
                    Ok(())
                }
                Err(error) => {
                    admission_failure = Some(ExternalSortOperationError::Memory(error));
                    Err(std::io::Error::from(std::io::ErrorKind::OutOfMemory))
                }
            },
            Err(error) => {
                admission_failure = Some(ExternalSortOperationError::Memory(error));
                let _ = observer.publish_sum(total_without_frontier, grant.size());
                Err(std::io::Error::from(std::io::ErrorKind::OutOfMemory))
            }
        }
    };
    let reader = match scalar_cleanup {
        Some(cleanup) => file.reader_with_admission_and_cleanup(&mut admit, cleanup.clone()),
        None => file.reader_with_admission(&mut admit),
    };
    match reader {
        Ok(reader) if admission_failure.is_none() => Ok((reader, admitted_bytes)),
        result => {
            let primary = match (admission_failure, result) {
                (Some(primary), Ok(reader)) => {
                    drop(reader);
                    primary
                }
                (Some(primary), Err(_sentinel)) => primary,
                (None, Err(error)) => ExternalSortOperationError::Io(error),
                (None, Ok(_)) => unreachable!("successful qualified reader has no failure"),
            };
            let release = if retain_failure {
                Ok(())
            } else {
                grant.try_resize(previous)
            };
            // Preserve the reader/admission primary even if an impossible
            // in-crate observer mismatch is discovered during rollback. The
            // observer has already made both scalar sinks fail closed.
            let _ = observer.publish_sum(total_without_frontier, grant.size());
            match release {
                Ok(()) => Err(primary),
                Err(release) => Err(with_io_cleanup(
                    primary,
                    std::io::Error::other(release),
                    "sort cursor reader-workspace release",
                )),
            }
        }
    }
}

fn cursor_capacity_bytes<T>(capacity: usize) -> Result<usize, ExternalSortOperationError> {
    std::alloc::Layout::array::<T>(capacity)
        .map(|layout| layout.size())
        .map_err(|_| {
            ExternalSortOperationError::Allocation(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "external sort cursor capacity exceeds the platform allocation maximum",
            ))
        })
}

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SortDirection {
    /// Ascending order.
    Ascending,
    /// Descending order.
    Descending,
}

/// Null handling in sort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NullOrder {
    /// NULLs come first.
    First,
    /// NULLs come last.
    Last,
}

/// Sort key specification.
#[derive(Debug, Clone)]
pub struct SortKey {
    /// Column index to sort by.
    pub column: usize,
    /// Sort direction.
    pub direction: SortDirection,
    /// Null handling.
    pub null_order: NullOrder,
}

impl SortKey {
    /// Creates a new ascending sort key.
    #[must_use]
    pub fn ascending(column: usize) -> Self {
        Self {
            column,
            direction: SortDirection::Ascending,
            null_order: NullOrder::Last,
        }
    }

    /// Creates a new descending sort key.
    #[must_use]
    pub fn descending(column: usize) -> Self {
        Self {
            column,
            direction: SortDirection::Descending,
            null_order: NullOrder::First,
        }
    }
}

/// Immutable semantic row comparison supplied to the external merge engine.
///
/// The callback compares only query-visible sort semantics. [`ExternalSort`]
/// applies its durable input ordinal as an ascending final tie-break, so a
/// descending user key never reverses equal-key encounter order. Callbacks
/// must be deterministic and define a total ordering for every row accepted
/// by the sorter.
#[derive(Clone)]
pub struct SemanticRowComparator {
    compare: Comparison,
}

#[derive(Clone)]
enum Comparison {
    Released,
    Infallible(Arc<RowCompareFn>),
    Accounted(Arc<AccountedComparison>),
}

struct AccountedComparison {
    keys: Vec<SortKey>,
    provider: Arc<dyn AccountedValueComparator>,
    fixed_bytes: usize,
    grant: parking_lot::Mutex<MemoryGrant>,
    resources: crate::execution::QueryResourceContext,
}

type RowCompareFn = dyn Fn(&[Value], &[Value]) -> Ordering + Send + Sync;

impl SemanticRowComparator {
    /// Wraps a deterministic, thread-safe semantic row comparator.
    #[must_use]
    pub fn new(compare: impl Fn(&[Value], &[Value]) -> Ordering + Send + Sync + 'static) -> Self {
        Self {
            compare: Comparison::Infallible(Arc::new(compare)),
        }
    }

    /// Installs an explicit resource-contract provider. This is not proof of
    /// arbitrary callback behavior: the implementor must honor its declared
    /// allocation bound and retain no comparison scratch or allocated errors.
    ///
    /// # Errors
    /// Returns failure to create the comparator's query-owned scratch grant.
    pub fn new_accounted(
        keys: Vec<SortKey>,
        provider: Arc<dyn AccountedValueComparator>,
        resources: crate::execution::QueryResourceContext,
    ) -> Result<Self, crate::execution::QueryResourceContextError> {
        let fixed_bytes = keys
            .capacity()
            .checked_mul(std::mem::size_of::<SortKey>())
            .and_then(|bytes| {
                bytes.checked_add(
                    std::mem::size_of::<AccountedComparison>() + 2 * std::mem::size_of::<usize>(),
                )
            })
            .ok_or(MemoryGrantError::ArithmeticOverflow {
                current_bytes: usize::MAX,
                additional_bytes: 1,
            })?;
        let grant = resources.try_allocate(fixed_bytes)?;
        Ok(Self {
            compare: Comparison::Accounted(Arc::new(AccountedComparison {
                keys,
                provider,
                fixed_bytes,
                grant: parking_lot::Mutex::new(grant),
                resources,
            })),
        })
    }

    pub(crate) fn checked_granted_bytes(&self) -> Result<usize, MemoryGrantError> {
        Ok(match &self.compare {
            Comparison::Infallible(_) | Comparison::Released => 0,
            Comparison::Accounted(accounted) => accounted.grant.lock().size(),
        })
    }

    fn is_accounted(&self) -> bool {
        matches!(self.compare, Comparison::Accounted(_))
    }

    pub(crate) fn try_compare(
        &self,
        left: &[Value],
        right: &[Value],
    ) -> Result<Ordering, ExternalSortOperationError> {
        let accounted = match &self.compare {
            Comparison::Released => {
                return Err(ExternalSortOperationError::Io(std::io::Error::from(
                    std::io::ErrorKind::InvalidData,
                )));
            }
            Comparison::Infallible(compare) => return Ok(compare(left, right)),
            Comparison::Accounted(accounted) => accounted,
        };
        check_cancellation(Some(accounted.resources.cancellation_token()))?;
        let mut grant = accounted.grant.try_lock().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "reentrant accounted comparison",
            )
        })?;
        for key in &accounted.keys {
            let left = left.get(key.column);
            let right = right.get(key.column);
            let bound = accounted
                .provider
                .scratch_bytes(left, right)
                .map_err(semantic_comparison_error)?;
            let bound = accounted.fixed_bytes.checked_add(bound).ok_or(
                MemoryGrantError::ArithmeticOverflow {
                    current_bytes: accounted.fixed_bytes,
                    additional_bytes: bound,
                },
            )?;
            if bound > grant.size() {
                grant.try_resize(bound)?;
            }
            let order = accounted
                .provider
                .compare(left, right)
                .map_err(semantic_comparison_error)?;
            let order = if key.direction == SortDirection::Descending {
                order.reverse()
            } else {
                order
            };
            if order != Ordering::Equal {
                return Ok(order);
            }
        }
        Ok(Ordering::Equal)
    }
}

fn semantic_comparison_error(error: SemanticComparisonError) -> ExternalSortOperationError {
    match error {
        SemanticComparisonError::Resource(error) => ExternalSortOperationError::Memory(error),
        SemanticComparisonError::Cancelled(error) => ExternalSortOperationError::Cancelled(error),
        SemanticComparisonError::Invalid(message) => ExternalSortOperationError::Io(
            std::io::Error::new(std::io::ErrorKind::InvalidData, message),
        ),
    }
}

impl std::fmt::Debug for SemanticRowComparator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SemanticRowComparator")
            .finish_non_exhaustive()
    }
}

/// Admission classification for exact comparison.
///
/// Built-in keys use allocation-free comparison. Explicit accounted providers
/// supply checked scratch bounds and must share the sorter's query account.
/// Plain custom callbacks retain compatibility merge support and remain
/// unqualified for exact owned-row output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ComparatorQualification {
    BuiltInSortKeys,
    AccountedProvider,
    UnqualifiedCustom,
}

#[derive(Debug)]
struct OrdinalRow {
    values: Vec<Value>,
    ordinal: u64,
}

/// Construction-time ownership for an encoded row and its dedicated child.
///
/// The child already accounts for the observed encoded `Vec` capacity on
/// entry. It expands to cover the simultaneous encoded-plus-decoded peak and
/// cannot be released before the encoded allocation on any exit path.
struct AccountedEncodedSortRow {
    payload: Option<Vec<u8>>,
    grant: Option<MemoryGrant>,
}

impl AccountedEncodedSortRow {
    fn try_new(payload: Vec<u8>, grant: MemoryGrant) -> Result<Self, ExternalSortOperationError> {
        let owner = Self {
            payload: Some(payload),
            grant: Some(grant),
        };
        if owner.granted_bytes() < owner.payload_capacity() {
            return Err(ExternalSortOperationError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "encoded sort-row capacity exceeds its incoming child grant",
            )));
        }
        Ok(owner)
    }

    fn payload(&self) -> &[u8] {
        self.payload
            .as_deref()
            .expect("live encoded row owner retains its payload")
    }

    fn payload_capacity(&self) -> usize {
        self.payload
            .as_ref()
            .expect("live encoded row owner retains its payload")
            .capacity()
    }

    fn granted_bytes(&self) -> usize {
        self.grant
            .as_ref()
            .expect("live encoded row owner retains its child grant")
            .size()
    }

    fn resize_grant(&mut self, bytes: usize) -> Result<(), MemoryGrantError> {
        self.grant
            .as_mut()
            .expect("live encoded row owner retains its child grant")
            .try_resize(bytes)
    }

    fn into_released_grant(mut self) -> MemoryGrant {
        drop(self.payload.take());
        self.grant
            .take()
            .expect("encoded row authority transfers exactly once")
    }
}

impl Drop for AccountedEncodedSortRow {
    fn drop(&mut self) {
        // This explicit order is load-bearing for every error and unwind.
        drop(self.payload.take());
        drop(self.grant.take());
    }
}

/// One bounded batch borrowed from an [`ExternalSortCursor`].
///
/// The borrow prevents advancing the cursor while a batch is live, allowing
/// the merge engine to reuse one accounted output buffer rather than letting
/// callers accumulate an unbounded number of sorter-owned batches.
#[derive(Debug)]
pub struct ExternalSortChunk<'cursor> {
    rows: &'cursor [Vec<Value>],
}

impl ExternalSortChunk<'_> {
    /// Returns the rows in this batch in semantic stable-sort order.
    #[must_use]
    pub fn rows(&self) -> &[Vec<Value>] {
        self.rows
    }

    /// Returns the number of rows in this batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Returns whether this batch contains no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// Fallible, bounded-output cursor over a consuming external merge.
///
/// Dropping the cursor closes every live reader before attempting best-effort
/// spill cleanup. A disk-backed cursor is consuming: after construction, the
/// originating sorter cannot be merged again even when iteration stops early.
pub struct ExternalSortCursor<'sort> {
    output_rows: Vec<Vec<Value>>,
    heap: ComparatorMinHeap<HeapEntry>,
    run_readers: Vec<Option<CursorRunReader>>,
    memory_iter: std::vec::IntoIter<OrdinalRow>,
    sorter: &'sort mut ExternalSort,
    memory_run_index: usize,
    max_chunk_rows: usize,
    frontier_base_bytes: usize,
    frontier_reader_bytes: usize,
    frontier_row_bytes: usize,
    output_base_bytes: usize,
    output_row_bytes: usize,
    terminal: bool,
    drop_cleaned: bool,
}

/// Resource-qualified cursor that keeps its sealed scalar observer attached
/// through downstream unwinding and final physical cleanup.
#[cfg(test)]
pub(crate) struct AccountedExternalSortCursor<'sort> {
    cursor: ExternalSortCursor<'sort>,
    observer: ExternalSortGrantObserver<'sort>,
}

/// Failure from the exact, move-consuming disk-only merge lane.
///
/// Reader-operation failures retain the original pre-admitted shared owner;
/// no stringification or source erasure can detach its diagnostic from the
/// reader/payload authority it carries.
enum ExactOwnedSortPrimary {
    Sort(ExternalSortOperationError),
    PublisherBuild(ExactOwnedPublisherBuildFailure),
    Accounted {
        classification: AccountedFailureClassification,
        authority: AccountedError,
    },
    Decoded(ExactOwnedDecodedError),
    DecodedPanic(ExactOwnedDecodedPanic),
    Terminal(ExactOwnedTerminalCleanup),
}

impl std::fmt::Debug for ExactOwnedSortPrimary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sort(error) => formatter.debug_tuple("Sort").field(error).finish(),
            Self::PublisherBuild(error) => formatter
                .debug_tuple("PublisherBuild")
                .field(error)
                .finish(),
            Self::Accounted {
                classification,
                authority,
            } => formatter
                .debug_struct("Accounted")
                .field("classification", classification)
                .field("authority", authority)
                .finish(),
            Self::Decoded(error) => formatter.debug_tuple("Decoded").field(error).finish(),
            Self::DecodedPanic(error) => {
                formatter.debug_tuple("DecodedPanic").field(error).finish()
            }
            Self::Terminal(error) => formatter.debug_tuple("Terminal").field(error).finish(),
        }
    }
}

/// Original exact-decoder failure retained beside the grant that covered its
/// complete encoded-plus-decoded construction peak.
enum ExactOwnedDecodedPrimary {
    Codec(QualifiedCodecError),
    Memory(MemoryGrantError),
    Allocation(&'static str),
    InvalidInput(&'static str),
    InvalidData(&'static str),
    #[cfg(test)]
    Hostile(Box<dyn std::error::Error + Send>),
}

impl ExactOwnedDecodedPrimary {
    #[cfg(test)]
    fn kind(&self) -> std::io::ErrorKind {
        match self {
            Self::Codec(error) => error.kind(),
            Self::Memory(_) => std::io::ErrorKind::OutOfMemory,
            Self::Allocation(_) => std::io::ErrorKind::OutOfMemory,
            Self::InvalidInput(_) => std::io::ErrorKind::InvalidInput,
            Self::InvalidData(_) => std::io::ErrorKind::InvalidData,
            Self::Hostile(_) => std::io::ErrorKind::Other,
        }
    }
}

impl std::fmt::Debug for ExactOwnedDecodedPrimary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Codec(error) => formatter.debug_tuple("Codec").field(error).finish(),
            Self::Memory(error) => formatter.debug_tuple("Memory").field(error).finish(),
            Self::Allocation(message) => {
                formatter.debug_tuple("Allocation").field(message).finish()
            }
            Self::InvalidInput(message) => formatter
                .debug_tuple("InvalidInput")
                .field(message)
                .finish(),
            Self::InvalidData(message) => {
                formatter.debug_tuple("InvalidData").field(message).finish()
            }
            #[cfg(test)]
            Self::Hostile(error) => formatter.debug_tuple("Hostile").field(error).finish(),
        }
    }
}

impl std::fmt::Display for ExactOwnedDecodedPrimary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Codec(error) => std::fmt::Display::fmt(error, formatter),
            Self::Memory(error) => std::fmt::Display::fmt(error, formatter),
            Self::Allocation(message)
            | Self::InvalidInput(message)
            | Self::InvalidData(message) => formatter.write_str(message),
            #[cfg(test)]
            Self::Hostile(error) => std::fmt::Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for ExactOwnedDecodedPrimary {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Codec(error) => Some(error),
            Self::Memory(error) => Some(error),
            Self::Allocation(_) | Self::InvalidInput(_) | Self::InvalidData(_) => None,
            #[cfg(test)]
            Self::Hostile(error) => Some(error.as_ref()),
        }
    }
}

/// Move-only decoded-row diagnostic whose construction authority cannot be
/// detached through `source`, conversion, or formatting.
#[must_use = "dropping the decoded-row failure releases its construction authority"]
struct ExactOwnedDecodedError {
    primary: Option<ExactOwnedDecodedPrimary>,
    grant: Option<MemoryGrant>,
}

impl ExactOwnedDecodedError {
    fn new(primary: ExactOwnedDecodedPrimary, grant: MemoryGrant) -> Self {
        Self {
            primary: Some(primary),
            grant: Some(grant),
        }
    }

    fn primary(&self) -> &ExactOwnedDecodedPrimary {
        self.primary
            .as_ref()
            .expect("live exact decoded-row failure retains its primary")
    }

    #[cfg(test)]
    fn kind(&self) -> std::io::ErrorKind {
        self.primary().kind()
    }

    fn granted_bytes(&self) -> usize {
        self.grant.as_ref().map_or(0, MemoryGrant::size)
    }
}

impl std::fmt::Debug for ExactOwnedDecodedError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExactOwnedDecodedError")
            .field("primary", &self.primary())
            .field("granted_bytes", &self.granted_bytes())
            .finish()
    }
}

impl std::fmt::Display for ExactOwnedDecodedError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.primary(), formatter)
    }
}

impl std::error::Error for ExactOwnedDecodedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.primary())
    }
}

impl Drop for ExactOwnedDecodedError {
    fn drop(&mut self) {
        let primary_dropped = self.primary.take().is_none_or(|primary| {
            super::run_cleanup_backstop(|| {
                drop(primary);
                Ok::<(), std::convert::Infallible>(())
            })
        });
        if primary_dropped {
            if let Some(mut grant) = self.grant.take()
                && !super::run_cleanup_backstop(|| grant.try_resize(0))
            {
                // `Drop` cannot return the failed reconciliation. Keep the
                // account conservatively charged instead of raw-dropping the
                // last token and pretending its release completed.
                std::mem::forget(grant);
            }
        } else if let Some(grant) = self.grant.take() {
            // A hostile diagnostic destructor can strand physical state. Its
            // construction authority must then remain charged permanently.
            std::mem::forget(grant);
        }
    }
}

/// Decoder panic payload paired with the complete encoded-row owner.
pub(crate) struct ExactOwnedDecodedPanic {
    payload: Option<Box<dyn std::any::Any + Send>>,
    encoded: Option<ProviderAccountedSortRow>,
}

impl ExactOwnedDecodedPanic {
    fn new(payload: Box<dyn std::any::Any + Send>, encoded: ProviderAccountedSortRow) -> Self {
        Self {
            payload: Some(payload),
            encoded: Some(encoded),
        }
    }

    /// Borrows the original panic without separating it from its authority.
    #[cfg(test)]
    pub(crate) fn payload(&self) -> &(dyn std::any::Any + Send) {
        self.payload
            .as_deref()
            .expect("live exact decoder panic retains its original payload")
    }

    #[cfg(test)]
    fn granted_bytes(&self) -> usize {
        self.encoded
            .as_ref()
            .map_or(0, ProviderAccountedSortRow::granted_bytes_for_test)
    }
}

impl std::fmt::Debug for ExactOwnedDecodedPanic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExactOwnedDecodedPanic")
            .field(
                "granted_bytes",
                &self
                    .encoded
                    .as_ref()
                    .map_or(0, ProviderAccountedSortRow::granted_bytes),
            )
            .finish_non_exhaustive()
    }
}

impl std::fmt::Display for ExactOwnedDecodedPanic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("captured fatal panic in qualified sort-row decoding")
    }
}

impl std::error::Error for ExactOwnedDecodedPanic {}

impl Drop for ExactOwnedDecodedPanic {
    fn drop(&mut self) {
        let payload_dropped = self.payload.take().is_none_or(|payload| {
            super::run_cleanup_backstop(|| {
                drop(payload);
                Ok::<(), std::convert::Infallible>(())
            })
        });
        if payload_dropped {
            if let Some(encoded) = self.encoded.take() {
                let mut grant = encoded.into_released_grant();
                if !super::run_cleanup_backstop(|| grant.try_resize(0)) {
                    // The encoded allocation is already gone. A refused
                    // accounting release has no caller in Drop, so retain its
                    // sole token fail-closed rather than silently discarding
                    // retry authority through the row owner's raw destructor.
                    std::mem::forget(grant);
                }
            }
        } else if let Some(encoded) = self.encoded.take() {
            // If the panic payload's destructor strands state, leaking the
            // complete encoded owner preserves the matching authority.
            std::mem::forget(encoded);
        }
    }
}

/// Allocation-free classification of a failed pre-admitted publisher build.
///
/// The build error's grant is not attached to any payload. Classifying it and
/// then dropping it releases that authority before this compact value enters
/// the cursor error state, avoiding both a large `Result` and a fresh boxing
/// allocation on an allocator-failure path.
#[derive(Debug)]
enum ExactOwnedPublisherBuildFailure {
    Memory(MemoryGrantError),
    NonZeroGrant,
    Allocation,
    Other,
}

impl ExactOwnedPublisherBuildFailure {
    fn classify(error: &AccountedErrorPublisherBuildError) -> Self {
        match error.failure() {
            AccountedErrorPublisherBuildFailure::Admission(error)
            | AccountedErrorPublisherBuildFailure::AllocationWithRollback(error) => {
                Self::Memory(error.clone())
            }
            AccountedErrorPublisherBuildFailure::NonZeroGrant { .. } => Self::NonZeroGrant,
            AccountedErrorPublisherBuildFailure::Allocation => Self::Allocation,
            _ => Self::Other,
        }
    }
}

impl std::fmt::Display for ExactOwnedPublisherBuildFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Memory(error) => std::fmt::Display::fmt(error, formatter),
            Self::NonZeroGrant => {
                formatter.write_str("exact failure publisher received a nonzero child grant")
            }
            Self::Allocation => formatter
                .write_str("allocator refused the pre-admitted exact failure control block"),
            Self::Other => formatter.write_str("exact failure publisher construction failed"),
        }
    }
}

/// One typed cause in the exact lane's fixed-capacity terminal envelope.
///
/// The three variants cover failures produced by the cursor itself, failures
/// returned by its downstream operator/sink, and a cleanup failure produced
/// while terminalizing the cursor. Keeping them as owned values avoids
/// formatting, boxing, or overwriting an earlier authority-bearing cause.
enum ExactOwnedFinalCause {
    Stream(ExactOwnedSortStreamError),
    Operator(OperatorError),
    Terminal(ExactOwnedTerminalCleanup),
}

impl ExactOwnedFinalCause {
    fn classification(&self) -> AccountedFailureClassification {
        match self {
            Self::Stream(error) => error.classification(),
            Self::Operator(error) => classify_operator_error(error),
            Self::Terminal(error) => error.classification(),
        }
    }
}

impl std::fmt::Debug for ExactOwnedFinalCause {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stream(error) => formatter.debug_tuple("Stream").field(error).finish(),
            Self::Operator(error) => formatter.debug_tuple("Operator").field(error).finish(),
            Self::Terminal(error) => formatter.debug_tuple("Terminal").field(error).finish(),
        }
    }
}

impl std::fmt::Display for ExactOwnedFinalCause {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stream(error) => std::fmt::Display::fmt(error, formatter),
            Self::Operator(error) => std::fmt::Display::fmt(error, formatter),
            Self::Terminal(error) => std::fmt::Display::fmt(error, formatter),
        }
    }
}

/// Allocation-free result of exact terminal cleanup. Run deletion and grant
/// release remain distinct, so neither can flatten or overwrite the other.
#[derive(Debug, Default)]
struct ExactOwnedPendingReaderCleanup {
    accounted: Option<AccountedError>,
    publication_release: Option<MemoryGrantError>,
    panic: Option<ExactOwnedCleanupPanic>,
}

impl ExactOwnedPendingReaderCleanup {
    fn is_empty(&self) -> bool {
        self.accounted.is_none() && self.publication_release.is_none() && self.panic.is_none()
    }

    fn defers_frontier_release(&self) -> bool {
        self.publication_release.is_some()
    }

    fn write_failures(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut wrote = false;
        if let Some(error) = &self.accounted {
            write!(formatter, "exact local reader cleanup failed: {error}")?;
            wrote = true;
        }
        if let Some(error) = &self.publication_release {
            if wrote {
                formatter.write_str("; ")?;
            }
            write!(
                formatter,
                "exact local reader publication release failed: {error}"
            )?;
            wrote = true;
        }
        if let Some(error) = &self.panic {
            if wrote {
                formatter.write_str("; ")?;
            }
            write!(
                formatter,
                "exact local reader cleanup captured a fatal panic: {error}"
            )?;
            wrote = true;
        }
        if !wrote {
            formatter.write_str("exact local reader cleanup reported no failure")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for ExactOwnedPendingReaderCleanup {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_failures(formatter)
    }
}

impl Drop for ExactOwnedPendingReaderCleanup {
    fn drop(&mut self) {
        if let Some(error) = self.accounted.take() {
            let _ = super::run_cleanup_backstop(|| {
                drop(error);
                Ok::<(), std::convert::Infallible>(())
            });
        }
        let _ = self.publication_release.take();
        if let Some(error) = self.panic.take() {
            let _ = super::run_cleanup_backstop(|| {
                drop(error);
                Ok::<(), std::convert::Infallible>(())
            });
        }
    }
}

/// Allocation-free result of exact terminal cleanup. Run deletion and grant
/// release remain distinct, so neither can flatten or overwrite the other.
#[derive(Debug)]
pub(crate) struct ExactOwnedTerminalCleanup {
    pending_reader_cleanup: ExactOwnedPendingReaderCleanup,
    reader_cleanup: Option<AccountedError>,
    reader_publication_release: Option<MemoryGrantError>,
    reader_cleanup_panic: Option<ExactOwnedCleanupPanic>,
    run_cleanup: Option<ExternalSortOperationError>,
    run_cleanup_panic: Option<ExactOwnedCleanupPanic>,
    resource_release: Option<ExactOwnedResourceRelease>,
    resource_release_panic: Option<ExactOwnedCleanupPanic>,
}

impl ExactOwnedTerminalCleanup {
    #[allow(
        clippy::too_many_arguments,
        reason = "the fixed-capacity terminal record keeps eight independently typed failure slots without allocation"
    )]
    fn from_parts(
        pending_reader_cleanup: ExactOwnedPendingReaderCleanup,
        reader_cleanup: Option<AccountedError>,
        reader_publication_release: Option<MemoryGrantError>,
        reader_cleanup_panic: Option<ExactOwnedCleanupPanic>,
        run_cleanup: Option<ExternalSortOperationError>,
        run_cleanup_panic: Option<ExactOwnedCleanupPanic>,
        resource_release: Option<ExactOwnedResourceRelease>,
        resource_release_panic: Option<ExactOwnedCleanupPanic>,
    ) -> Option<Self> {
        if pending_reader_cleanup.is_empty()
            && reader_cleanup.is_none()
            && reader_publication_release.is_none()
            && reader_cleanup_panic.is_none()
            && run_cleanup.is_none()
            && run_cleanup_panic.is_none()
            && resource_release.is_none()
            && resource_release_panic.is_none()
        {
            None
        } else {
            Some(Self {
                pending_reader_cleanup,
                reader_cleanup,
                reader_publication_release,
                reader_cleanup_panic,
                run_cleanup,
                run_cleanup_panic,
                resource_release,
                resource_release_panic,
            })
        }
    }

    fn into_operator_error_without_publisher(mut self) -> OperatorError {
        if let Some(error) = self.pending_reader_cleanup.accounted.take() {
            let classification = classify_accounted_authority(&error);
            return OperatorError::ClassifiedAccountedFailure {
                classification,
                authority: error,
            };
        }
        if let Some(error) = self.pending_reader_cleanup.publication_release.take() {
            return OperatorError::ResidentMemory(error);
        }
        if let Some(error) = self.reader_cleanup.take() {
            let classification = classify_accounted_authority(&error);
            return OperatorError::ClassifiedAccountedFailure {
                classification,
                authority: error,
            };
        }
        if let Some(error) = self.reader_publication_release.take() {
            return OperatorError::ResidentMemory(error);
        }
        if let Some(error) = self.run_cleanup.take() {
            return ExactOwnedSortStreamError::map_sort_error(error);
        }
        if let Some(release) = self.resource_release.as_mut()
            && let Some(error) = release.take_operator_error()
        {
            return error;
        }
        OperatorError::ResidentContainerInvariant {
            container: "exact owned sort terminal cleanup",
            message: "terminal cleanup captured a fatal panic without its final publisher",
        }
    }

    fn classification(&self) -> AccountedFailureClassification {
        if let Some(error) = self.pending_reader_cleanup.accounted.as_ref() {
            return classify_accounted_authority(error);
        }
        if let Some(error) = self.pending_reader_cleanup.publication_release.as_ref() {
            return AccountedFailureClassification::ResidentMemory(error.clone());
        }
        if let Some(error) = self.reader_cleanup.as_ref() {
            return classify_accounted_authority(error);
        }
        if let Some(error) = self.reader_publication_release.as_ref() {
            return AccountedFailureClassification::ResidentMemory(error.clone());
        }
        if let Some(error) = self.run_cleanup.as_ref() {
            return classify_external_sort_error(error);
        }
        if let Some(error) = self.resource_release.as_ref() {
            return error.classification();
        }
        if self.reader_cleanup_panic.is_some()
            || self.run_cleanup_panic.is_some()
            || self.resource_release_panic.is_some()
            || self.pending_reader_cleanup.panic.is_some()
        {
            AccountedFailureClassification::ResidentInvariant
        } else {
            AccountedFailureClassification::Execution
        }
    }
}

/// Original payload from a caught terminal-cleanup unwind. It is retained as
/// a fixed cause instead of resuming and replacing the operation primary.
struct ExactOwnedCleanupPanic {
    payload: Option<Box<dyn std::any::Any + Send>>,
}

impl ExactOwnedCleanupPanic {
    fn new(payload: Box<dyn std::any::Any + Send>) -> Self {
        Self {
            payload: Some(payload),
        }
    }

    #[cfg(test)]
    fn payload(&self) -> &(dyn std::any::Any + Send) {
        self.payload
            .as_deref()
            .expect("live cleanup panic retains its original payload")
    }
}

fn retain_first_cleanup_panic(
    first: &mut Option<ExactOwnedCleanupPanic>,
    payload: Box<dyn std::any::Any + Send>,
) {
    let panic = ExactOwnedCleanupPanic::new(payload);
    if first.is_none() {
        *first = Some(panic);
    } else {
        drop(panic);
    }
}

impl std::fmt::Debug for ExactOwnedCleanupPanic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExactOwnedCleanupPanic")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Display for ExactOwnedCleanupPanic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("captured fatal panic during exact terminal cleanup")
    }
}

impl std::error::Error for ExactOwnedCleanupPanic {}

impl Drop for ExactOwnedCleanupPanic {
    fn drop(&mut self) {
        if let Some(payload) = self.payload.take() {
            let _ = super::run_cleanup_backstop(|| {
                drop(payload);
                Ok::<(), std::convert::Infallible>(())
            });
        }
    }
}

/// Every independently retryable release/publication result from exact
/// terminal cleanup. Fixed slots avoid allocating a collection while retaining
/// each structured failure.
#[derive(Debug)]
struct ExactOwnedResourceRelease {
    workspace: Option<MemoryGrantError>,
    writer: Option<MemoryGrantError>,
    ordinal: Option<MemoryGrantError>,
    frontier: Option<MemoryGrantError>,
    payload: Option<MemoryGrantError>,
    output: Option<MemoryGrantError>,
    catalog: Option<ExternalSortOperationError>,
    publication: Option<ExternalSortOperationError>,
}

impl ExactOwnedResourceRelease {
    fn is_empty(&self) -> bool {
        self.workspace.is_none()
            && self.writer.is_none()
            && self.ordinal.is_none()
            && self.frontier.is_none()
            && self.payload.is_none()
            && self.output.is_none()
            && self.catalog.is_none()
            && self.publication.is_none()
    }

    fn take_operator_error(&mut self) -> Option<OperatorError> {
        for slot in [
            &mut self.workspace,
            &mut self.writer,
            &mut self.ordinal,
            &mut self.frontier,
            &mut self.payload,
            &mut self.output,
        ] {
            if let Some(error) = slot.take() {
                return Some(OperatorError::ResidentMemory(error));
            }
        }
        self.catalog
            .take()
            .or_else(|| self.publication.take())
            .map(ExactOwnedSortStreamError::map_sort_error)
    }

    fn classification(&self) -> AccountedFailureClassification {
        if let Some(error) = [
            &self.workspace,
            &self.writer,
            &self.ordinal,
            &self.frontier,
            &self.payload,
            &self.output,
        ]
        .into_iter()
        .find_map(|slot| slot.as_ref())
        {
            return AccountedFailureClassification::ResidentMemory(error.clone());
        }
        self.catalog
            .as_ref()
            .or(self.publication.as_ref())
            .map_or(AccountedFailureClassification::Execution, |error| {
                classify_external_sort_error(error)
            })
    }

    fn write_failures(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut wrote = false;
        macro_rules! write_slot {
            ($slot:expr, $phase:literal) => {
                if let Some(error) = $slot {
                    if wrote {
                        formatter.write_str("; ")?;
                    }
                    write!(formatter, concat!($phase, " failed: {}"), error)?;
                    wrote = true;
                }
            };
        }
        write_slot!(&self.workspace, "sort workspace release");
        write_slot!(&self.writer, "sort writer release");
        write_slot!(&self.ordinal, "sort ordinal release");
        write_slot!(&self.frontier, "sort frontier release");
        write_slot!(&self.payload, "sort payload release");
        write_slot!(&self.output, "sort output release");
        write_slot!(&self.catalog, "sort catalog release");
        write_slot!(&self.publication, "sort observer publication");
        if !wrote {
            formatter.write_str("exact resource release reported no failure")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for ExactOwnedResourceRelease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_failures(formatter)
    }
}

impl std::error::Error for ExactOwnedResourceRelease {}

impl Drop for ExactOwnedResourceRelease {
    fn drop(&mut self) {
        let _ = self.workspace.take();
        let _ = self.writer.take();
        let _ = self.ordinal.take();
        let _ = self.frontier.take();
        let _ = self.payload.take();
        let _ = self.output.take();
        if let Some(catalog) = self.catalog.take() {
            let _ = super::run_cleanup_backstop(|| {
                drop(catalog);
                Ok::<(), std::convert::Infallible>(())
            });
        }
        if let Some(publication) = self.publication.take() {
            let _ = super::run_cleanup_backstop(|| {
                drop(publication);
                Ok::<(), std::convert::Infallible>(())
            });
        }
    }
}

impl std::fmt::Display for ExactOwnedTerminalCleanup {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut wrote = false;
        macro_rules! write_cleanup {
            ($slot:expr, $label:literal) => {
                if let Some(error) = $slot {
                    if wrote {
                        formatter.write_str("; ")?;
                    }
                    write!(formatter, concat!($label, ": {}"), error)?;
                    wrote = true;
                }
            };
        }
        if !self.pending_reader_cleanup.is_empty() {
            write_cleanup!(
                Some(&self.pending_reader_cleanup),
                "exact local reader cleanup failed"
            );
        }
        write_cleanup!(&self.run_cleanup, "exact run cleanup failed");
        write_cleanup!(&self.reader_cleanup, "exact reader cleanup failed");
        write_cleanup!(
            &self.reader_publication_release,
            "exact reader publication release failed"
        );
        write_cleanup!(
            &self.reader_cleanup_panic,
            "exact reader cleanup captured a fatal panic"
        );
        write_cleanup!(
            &self.run_cleanup_panic,
            "exact run cleanup captured a fatal panic"
        );
        write_cleanup!(&self.resource_release, "exact resource release failed");
        write_cleanup!(
            &self.resource_release_panic,
            "exact resource release captured a fatal panic"
        );
        if !wrote {
            formatter.write_str("exact terminal cleanup reported no failure")?;
        }
        Ok(())
    }
}

impl std::error::Error for ExactOwnedTerminalCleanup {}

impl Drop for ExactOwnedTerminalCleanup {
    fn drop(&mut self) {
        let mut resolved = true;
        let pending = std::mem::take(&mut self.pending_reader_cleanup);
        resolved &= super::run_cleanup_backstop(|| {
            drop(pending);
            Ok::<(), std::convert::Infallible>(())
        });
        if let Some(reader_cleanup) = self.reader_cleanup.take() {
            resolved &= super::run_cleanup_backstop(|| {
                drop(reader_cleanup);
                Ok::<(), std::convert::Infallible>(())
            });
        }
        let _ = self.reader_publication_release.take();
        if let Some(reader_cleanup_panic) = self.reader_cleanup_panic.take() {
            resolved &= super::run_cleanup_backstop(|| {
                drop(reader_cleanup_panic);
                Ok::<(), std::convert::Infallible>(())
            });
        }
        if let Some(run_cleanup) = self.run_cleanup.take() {
            resolved &= retire_pull_operation_error(run_cleanup);
        }
        if let Some(run_cleanup_panic) = self.run_cleanup_panic.take() {
            resolved &= super::run_cleanup_backstop(|| {
                drop(run_cleanup_panic);
                Ok::<(), std::convert::Infallible>(())
            });
        }
        if let Some(resource_release) = self.resource_release.take() {
            resolved &= super::run_cleanup_backstop(|| {
                drop(resource_release);
                Ok::<(), std::convert::Infallible>(())
            });
        }
        if let Some(resource_release_panic) = self.resource_release_panic.take() {
            resolved &= super::run_cleanup_backstop(|| {
                drop(resource_release_panic);
                Ok::<(), std::convert::Infallible>(())
            });
        }
        assert!(resolved, "exact terminal diagnostic destructor panicked");
    }
}

/// Pre-admitted, cloneable-at-the-handle final failure payload.
///
/// Slots are installed exactly once. There is deliberately no incremental
/// overwrite API: a later transfer or cleanup failure can never destroy the
/// original diagnostic or its memory authority before publication.
#[must_use = "the exact final failure must be published or dropped"]
struct ExactOwnedFinalFailure {
    hook_workspace: Option<AccountedError>,
    primary: Option<ExactOwnedFinalCause>,
    secondary: Option<(ExactOwnedFinalCause, &'static str)>,
    cleanup: Option<(ExactOwnedFinalCause, &'static str)>,
    reconciliation: Option<(ExactOwnedFinalCause, &'static str)>,
}

fn map_provider_failure_classification(
    classification: ProviderAccountedReaderFailureClassification,
) -> AccountedFailureClassification {
    match classification {
        ProviderAccountedReaderFailureClassification::Memory(error) => {
            AccountedFailureClassification::ResidentMemory(error)
        }
        ProviderAccountedReaderFailureClassification::Allocation => {
            AccountedFailureClassification::ResidentAllocation
        }
        ProviderAccountedReaderFailureClassification::ExactAllocation(error) => {
            AccountedFailureClassification::ResidentExactVectorAllocation(error)
        }
        ProviderAccountedReaderFailureClassification::StorageFull => {
            AccountedFailureClassification::StorageFull
        }
        ProviderAccountedReaderFailureClassification::Execution => {
            AccountedFailureClassification::Execution
        }
    }
}

fn classify_accounted_authority(error: &AccountedError) -> AccountedFailureClassification {
    error
        .inspect::<ProviderAccountedReaderOperationError, _>(|error| {
            map_provider_failure_classification(error.operator_classification())
        })
        .or_else(|| {
            error.inspect::<ProviderAccountedReaderError, _>(|error| {
                map_provider_failure_classification(error.operator_classification())
            })
        })
        .unwrap_or(AccountedFailureClassification::Execution)
}

fn classify_io_kind(kind: std::io::ErrorKind) -> AccountedFailureClassification {
    if matches!(
        kind,
        std::io::ErrorKind::QuotaExceeded | std::io::ErrorKind::StorageFull
    ) {
        AccountedFailureClassification::StorageFull
    } else {
        AccountedFailureClassification::Execution
    }
}

pub(crate) fn classify_external_sort_error(
    error: &ExternalSortOperationError,
) -> AccountedFailureClassification {
    match error {
        ExternalSortOperationError::Cancelled(error)
        | ExternalSortOperationError::CancelledWithCleanup { error, .. } => {
            AccountedFailureClassification::QueryCancelled(*error)
        }
        ExternalSortOperationError::Memory(error)
        | ExternalSortOperationError::MemoryWithCleanup { error, .. } => {
            AccountedFailureClassification::ResidentMemory(error.clone())
        }
        ExternalSortOperationError::Allocation(_) => {
            AccountedFailureClassification::ResidentAllocation
        }
        ExternalSortOperationError::Io(error) => classify_io_kind(error.kind()),
        ExternalSortOperationError::WithGrantRelease { primary, .. } => match primary {
            ExternalSortPrimary::Cancelled(error) => {
                AccountedFailureClassification::QueryCancelled(*error)
            }
            ExternalSortPrimary::Memory(error) => {
                AccountedFailureClassification::ResidentMemory(error.clone())
            }
            ExternalSortPrimary::Allocation(_) => {
                AccountedFailureClassification::ResidentAllocation
            }
            ExternalSortPrimary::Io(error) => classify_io_kind(error.kind()),
        },
    }
}

pub(crate) fn classify_operator_error(error: &OperatorError) -> AccountedFailureClassification {
    match error {
        OperatorError::UnsupportedAccountedTransport { .. } => {
            AccountedFailureClassification::UnsupportedAccountedTransport
        }
        OperatorError::AccountedFailure(authority) => classify_accounted_authority(authority),
        OperatorError::ClassifiedAccountedFailure { classification, .. } => classification.clone(),
        OperatorError::TypeMismatch { .. } => AccountedFailureClassification::TypeMismatch,
        OperatorError::ColumnNotFound(_) => AccountedFailureClassification::ColumnNotFound,
        OperatorError::Execution(_) => AccountedFailureClassification::Execution,
        OperatorError::ConstraintViolation(_) => {
            AccountedFailureClassification::ConstraintViolation
        }
        OperatorError::WriteConflict(_) => AccountedFailureClassification::WriteConflict,
        OperatorError::ResidentMemory(error) => {
            AccountedFailureClassification::ResidentMemory(error.clone())
        }
        OperatorError::Context { source, .. } => classify_operator_error(source),
        OperatorError::ResidentAllocation(_)
        | OperatorError::ResidentContainerAllocation { .. }
        | OperatorError::ResidentNativeMapAllocation { .. }
        | OperatorError::ResidentNativeMapAllocationWithRollback { .. } => {
            AccountedFailureClassification::ResidentAllocation
        }
        OperatorError::ResidentExactVectorAllocation(error) => {
            AccountedFailureClassification::ResidentExactVectorAllocation(error.clone())
        }
        OperatorError::ResidentContainerInvariant { .. }
        | OperatorError::ResidentContainerInvariantWithRollback { .. } => {
            AccountedFailureClassification::ResidentInvariant
        }
        OperatorError::StorageFull(_) => AccountedFailureClassification::StorageFull,
        OperatorError::QueryCancelled(error) => {
            AccountedFailureClassification::QueryCancelled(*error)
        }
    }
}

impl ExactOwnedFinalFailure {
    fn from_parts(
        primary: ExactOwnedFinalCause,
        secondary: Option<(ExactOwnedFinalCause, &'static str)>,
        cleanup: Option<(ExactOwnedFinalCause, &'static str)>,
        reconciliation: Option<(ExactOwnedFinalCause, &'static str)>,
    ) -> Self {
        Self {
            primary: Some(primary),
            secondary,
            cleanup,
            reconciliation,
            hook_workspace: None,
        }
    }

    fn primary(&self) -> &ExactOwnedFinalCause {
        self.primary
            .as_ref()
            .expect("live exact final failure retains its primary")
    }

    fn cloned_operator_primary(&self) -> Option<OperatorError> {
        match self.primary() {
            ExactOwnedFinalCause::Operator(error) => Some(error.clone()),
            ExactOwnedFinalCause::Stream(_) | ExactOwnedFinalCause::Terminal(_) => None,
        }
    }

    #[cfg(test)]
    fn decoded_primary(&self) -> Option<&ExactOwnedDecodedError> {
        match self.primary() {
            ExactOwnedFinalCause::Stream(ExactOwnedSortStreamError {
                primary: ExactOwnedSortPrimary::Decoded(error),
                ..
            }) => Some(error),
            _ => None,
        }
    }

    #[cfg(test)]
    fn decoded_panic_primary(&self) -> Option<&ExactOwnedDecodedPanic> {
        match self.primary() {
            ExactOwnedFinalCause::Stream(ExactOwnedSortStreamError {
                primary: ExactOwnedSortPrimary::DecodedPanic(panic),
            }) => Some(panic),
            _ => None,
        }
    }

    #[cfg(test)]
    fn pending_reader_publication_release(&self) -> Option<&MemoryGrantError> {
        let (ExactOwnedFinalCause::Terminal(cleanup), _) = self.cleanup.as_ref()? else {
            return None;
        };
        cleanup.pending_reader_cleanup.publication_release.as_ref()
    }
}

/// One pre-admitted allowance shared by every non-reader hook diagnostic from
/// a pull sort. Forgotten opaque owners poison this authority permanently.
#[derive(Debug)]
pub(crate) struct PullSortHookAuthority {
    grant: Option<MemoryGrant>,
    unresolved: std::sync::atomic::AtomicBool,
}
impl PullSortHookAuthority {
    pub(crate) fn new(grant: MemoryGrant) -> Self {
        Self {
            grant: Some(grant),
            unresolved: std::sync::atomic::AtomicBool::new(false),
        }
    }
    pub(crate) fn granted_bytes(&self) -> usize {
        self.grant.as_ref().map_or(0, MemoryGrant::size)
    }
}
impl std::fmt::Display for PullSortHookAuthority {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.write_str("retained pull-sort hook workspace")
    }
}
impl std::error::Error for PullSortHookAuthority {}
impl Drop for PullSortHookAuthority {
    fn drop(&mut self) {
        if self.unresolved.load(std::sync::atomic::Ordering::Acquire)
            && let Some(grant) = self.grant.take()
        {
            std::mem::forget(grant);
        }
    }
}
pub(crate) fn quarantine_pull_sort_hook(authority: Option<&AccountedError>) {
    if let Some(authority) = authority {
        authority.inspect::<PullSortHookAuthority, _>(|owner| {
            owner
                .unresolved
                .store(true, std::sync::atomic::Ordering::Release);
        });
    }
}
/// Keeps authority live and marks opaque unwinds even when callers catch the
/// unwind while retaining the sorter itself.
pub(crate) struct PullSortHookGuard(pub(crate) Option<AccountedError>);
impl Drop for PullSortHookGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            quarantine_pull_sort_hook(self.0.as_ref());
        }
    }
}

/// One terminal scalar merge diagnostic and the exact grants that admitted it.
/// Its publication block is allocated before the merge consumes any run.
struct ScalarSortFailure {
    operation: Option<ExternalSortOperationError>,
    operator: Option<OperatorError>,
    panic: Option<Box<dyn std::any::Any + Send>>,
    cleanup: Option<ExternalSortOperationError>,
    cleanup_panic: Option<Box<dyn std::any::Any + Send>>,
    cleanup_authority: AccountedError,
}

impl std::fmt::Debug for ScalarSortFailure {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("ScalarSortFailure")
            .field(
                "reader_panicked",
                &self
                    .cleanup_authority
                    .inspect::<ScalarReaderCleanup, _>(ScalarReaderCleanup::operation_panicked)
                    .unwrap_or(false),
            )
            .finish_non_exhaustive()
    }
}

impl std::fmt::Display for ScalarSortFailure {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Opaque extension Display implementations may allocate, panic, or
        // expose cloneable payloads. Classification carries the public reason.
        out.write_str("scalar spill sort failed")
    }
}

impl std::error::Error for ScalarSortFailure {}

impl Drop for ScalarSortFailure {
    fn drop(&mut self) {
        let mut resolved = true;
        for error in [self.operation.take(), self.cleanup.take()]
            .into_iter()
            .flatten()
        {
            resolved &= retire_pull_operation_error(error);
        }
        if let Some(error) = self.operator.take() {
            resolved &= super::run_cleanup_backstop(|| {
                drop(error);
                Ok::<(), std::convert::Infallible>(())
            });
        }
        for payload in [self.panic.take(), self.cleanup_panic.take()]
            .into_iter()
            .flatten()
        {
            resolved &= super::run_cleanup_backstop(|| {
                drop(payload);
                Ok::<(), std::convert::Infallible>(())
            });
        }
        resolved &= !self
            .cleanup_authority
            .inspect::<ScalarReaderCleanup, _>(ScalarReaderCleanup::failed)
            .unwrap_or(true);
        if !resolved {
            self.cleanup_authority
                .inspect::<ScalarReaderCleanup, _>(ScalarReaderCleanup::mark_failed);
            // Opaque cleanup may have retained a physical payload. Its exact
            // reader/frame authority is quarantined; this emptied carrier can
            // still be physically deallocated before its own grant retires.
        }
    }
}

/// Scalar output keeps the established borrowed-row algorithm and provider
/// contract, while its terminal diagnostic owns every admitted callback span.
pub(crate) struct ScalarExternalSortCursor<'sort> {
    cursor: ExternalSortCursor<'sort>,
    observer: ExternalSortGrantObserver<'sort>,
    transport: Option<ScalarFailureTransport>,
}

impl std::fmt::Debug for ScalarExternalSortCursor<'_> {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        output
            .debug_struct("ScalarExternalSortCursor")
            .field("cursor", &self.cursor)
            .finish_non_exhaustive()
    }
}

struct ScalarFailureTransport {
    publisher: AccountedErrorPublisher<ScalarSortFailure>,
    cleanup: AccountedError,
}

fn scalar_reader_cleanup_check(
    cleanup: Option<&AccountedError>,
) -> Result<(), ExternalSortOperationError> {
    if cleanup.is_some_and(|authority| {
        authority
            .inspect::<ScalarReaderCleanup, _>(ScalarReaderCleanup::failed)
            .unwrap_or(true)
    }) {
        Err(ExternalSortOperationError::Io(
            std::io::ErrorKind::Other.into(),
        ))
    } else {
        Ok(())
    }
}

fn mark_scalar_cleanup_failed(cleanup: Option<&AccountedError>) {
    if let Some(authority) = cleanup {
        authority.inspect::<ScalarReaderCleanup, _>(ScalarReaderCleanup::mark_failed);
    }
}

/// Bounded typed transport for the existing pull-sort spill transitions.
/// Publication is admitted before any destructive spill operation.
#[derive(Debug)]
pub(crate) struct PullSortFailure {
    pub(crate) primary: Option<OperatorError>,
    pub(crate) operation: Option<ExternalSortOperationError>,
    pub(crate) release: Option<MemoryGrantError>,
    pub(crate) workspaces: [Option<MemoryGrant>; 3],
    pub(crate) hook_workspace: Option<AccountedError>,
}

#[cfg(test)]
impl PullSortFailure {
    pub(crate) fn payload_granted_bytes(&self) -> usize {
        let workspace = self
            .workspaces
            .iter()
            .flatten()
            .map(MemoryGrant::size)
            .sum::<usize>();
        let hook = self.hook_workspace.as_ref().map_or(0, |authority| {
            authority.granted_bytes()
                + authority
                    .inspect::<PullSortHookAuthority, _>(PullSortHookAuthority::granted_bytes)
                    .unwrap()
        });
        workspace + hook
    }
}

impl std::fmt::Display for PullSortFailure {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(primary) = &self.primary {
            write!(out, "{primary}")?;
        }
        if let Some(operation) = &self.operation {
            if self.primary.is_some() {
                out.write_str("; sort spill cleanup also failed: ")?;
            }
            write!(out, "{operation}")?;
        }
        if let Some(release) = &self.release {
            write!(out, "; sort row release also failed: {release}")?;
        }
        Ok(())
    }
}
impl std::error::Error for PullSortFailure {}

fn retire_pull_io_error(error: std::io::Error) -> bool {
    // Decompose our own compound I/O carrier before invoking either provider's
    // destructor; one unwind must never drop the other opaque error implicitly.
    if error
        .get_ref()
        .is_some_and(|inner| inner.is::<super::SpillCleanupContext>())
    {
        let inner = error.into_inner().expect("checked compound error");
        let context = inner
            .downcast::<super::SpillCleanupContext>()
            .expect("checked compound error type");
        let super::SpillCleanupContext {
            primary, cleanup, ..
        } = *context;
        let primary = retire_pull_io_error(primary);
        let cleanup = retire_pull_io_error(cleanup);
        return primary && cleanup;
    }
    super::run_cleanup_backstop(|| {
        drop(error);
        Ok::<(), std::convert::Infallible>(())
    })
}

fn retire_pull_operation_error(operation: ExternalSortOperationError) -> bool {
    match operation {
        ExternalSortOperationError::Cancelled(_) | ExternalSortOperationError::Memory(_) => true,
        ExternalSortOperationError::CancelledWithCleanup { cleanup, .. }
        | ExternalSortOperationError::MemoryWithCleanup { cleanup, .. }
        | ExternalSortOperationError::Allocation(cleanup)
        | ExternalSortOperationError::Io(cleanup) => retire_pull_io_error(cleanup),
        ExternalSortOperationError::WithGrantRelease {
            primary, cleanup, ..
        } => {
            let primary = match primary {
                ExternalSortPrimary::Io(error) | ExternalSortPrimary::Allocation(error) => {
                    retire_pull_io_error(error)
                }
                _ => true,
            };
            let cleanup = cleanup.is_none_or(retire_pull_io_error);
            primary && cleanup
        }
    }
}

impl Drop for PullSortFailure {
    fn drop(&mut self) {
        // Each known opaque owner is resolved independently. AccountedError
        // retains this carrier's grant if any destructor fails.
        let mut resolved = true;
        if let Some(primary) = self.primary.take() {
            resolved &= super::run_cleanup_backstop(|| {
                drop(primary);
                Ok::<(), std::convert::Infallible>(())
            });
        }
        if let Some(operation) = self.operation.take() {
            resolved &= retire_pull_operation_error(operation);
        }
        if !resolved {
            quarantine_pull_sort_hook(self.hook_workspace.as_ref());
            for grant in &mut self.workspaces {
                if let Some(grant) = grant.take() {
                    std::mem::forget(grant);
                }
            }
        }
        assert!(resolved, "pull sort diagnostic destructor panicked");
    }
}

pub(crate) fn recover_exact_final_operator_primary(
    authority: &AccountedError,
) -> Option<OperatorError> {
    authority
        .inspect::<ExactOwnedFinalFailure, _>(ExactOwnedFinalFailure::cloned_operator_primary)
        .flatten()
        .or_else(|| {
            authority
                .inspect::<PullSortFailure, _>(|failure| failure.primary.clone())
                .flatten()
        })
}

impl std::fmt::Debug for ExactOwnedFinalFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExactOwnedFinalFailure")
            .field("primary", &self.primary())
            .field("secondary", &self.secondary)
            .field("cleanup", &self.cleanup)
            .field("reconciliation", &self.reconciliation)
            .finish()
    }
}

impl std::fmt::Display for ExactOwnedFinalFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.primary(), formatter)?;
        if let Some((secondary, phase)) = &self.secondary {
            write!(formatter, "; {phase} also failed: {secondary}")?;
        }
        if let Some((cleanup, phase)) = &self.cleanup {
            write!(formatter, "; {phase} also failed: {cleanup}")?;
        }
        if let Some((reconciliation, phase)) = &self.reconciliation {
            write!(formatter, "; {phase} also failed: {reconciliation}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ExactOwnedFinalFailure {}

fn retire_exact_final_cause(cause: ExactOwnedFinalCause) -> bool {
    match cause {
        ExactOwnedFinalCause::Stream(ExactOwnedSortStreamError {
            primary: ExactOwnedSortPrimary::Sort(error),
        }) => retire_pull_operation_error(error),
        other => super::run_cleanup_backstop(|| {
            drop(other);
            Ok::<(), std::convert::Infallible>(())
        }),
    }
}

impl Drop for ExactOwnedFinalFailure {
    fn drop(&mut self) {
        let mut resolved = true;
        if let Some(primary) = self.primary.take() {
            resolved &= retire_exact_final_cause(primary);
        }
        if let Some((secondary, _)) = self.secondary.take() {
            resolved &= retire_exact_final_cause(secondary);
        }
        if let Some((cleanup, _)) = self.cleanup.take() {
            resolved &= retire_exact_final_cause(cleanup);
        }
        if let Some((reconciliation, _)) = self.reconciliation.take() {
            resolved &= retire_exact_final_cause(reconciliation);
        }
        if !resolved {
            quarantine_pull_sort_hook(self.hook_workspace.as_ref());
        }
    }
}

#[derive(Debug)]
pub(crate) struct ExactOwnedSortStreamError {
    primary: ExactOwnedSortPrimary,
}

impl ExactOwnedSortStreamError {
    fn primary(primary: ExactOwnedSortPrimary) -> Self {
        Self { primary }
    }

    fn accounted(error: AccountedError) -> Self {
        let classification = classify_accounted_authority(&error);
        Self::classified_accounted(classification, error)
    }

    fn classified_accounted(
        classification: AccountedFailureClassification,
        authority: AccountedError,
    ) -> Self {
        Self::primary(ExactOwnedSortPrimary::Accounted {
            classification,
            authority,
        })
    }

    fn decoded(error: ExactOwnedDecodedError) -> Self {
        Self::primary(ExactOwnedSortPrimary::Decoded(error))
    }

    fn decoded_panic(error: ExactOwnedDecodedPanic) -> Self {
        Self::primary(ExactOwnedSortPrimary::DecodedPanic(error))
    }

    fn terminal(error: ExactOwnedTerminalCleanup) -> Self {
        Self::primary(ExactOwnedSortPrimary::Terminal(error))
    }

    fn publisher_build(failure: ExactOwnedPublisherBuildFailure) -> Self {
        Self::primary(ExactOwnedSortPrimary::PublisherBuild(failure))
    }

    fn can_return_without_final_publisher(&self) -> bool {
        matches!(
            &self.primary,
            ExactOwnedSortPrimary::Sort(
                ExternalSortOperationError::Cancelled(_) | ExternalSortOperationError::Memory(_)
            ) | ExactOwnedSortPrimary::PublisherBuild(_)
                | ExactOwnedSortPrimary::Accounted { .. }
        )
    }

    fn classification(&self) -> AccountedFailureClassification {
        match &self.primary {
            ExactOwnedSortPrimary::Sort(error) => classify_external_sort_error(error),
            ExactOwnedSortPrimary::PublisherBuild(ExactOwnedPublisherBuildFailure::Memory(
                error,
            )) => AccountedFailureClassification::ResidentMemory(error.clone()),
            ExactOwnedSortPrimary::PublisherBuild(ExactOwnedPublisherBuildFailure::Allocation) => {
                AccountedFailureClassification::ResidentAllocation
            }
            ExactOwnedSortPrimary::PublisherBuild(_) => {
                AccountedFailureClassification::ResidentInvariant
            }
            ExactOwnedSortPrimary::Accounted { classification, .. } => classification.clone(),
            ExactOwnedSortPrimary::Decoded(error) => match error.primary() {
                ExactOwnedDecodedPrimary::Memory(error) => {
                    AccountedFailureClassification::ResidentMemory(error.clone())
                }
                ExactOwnedDecodedPrimary::Allocation(_) => {
                    AccountedFailureClassification::ResidentAllocation
                }
                ExactOwnedDecodedPrimary::Codec(_)
                | ExactOwnedDecodedPrimary::InvalidInput(_)
                | ExactOwnedDecodedPrimary::InvalidData(_) => {
                    AccountedFailureClassification::Execution
                }
                #[cfg(test)]
                ExactOwnedDecodedPrimary::Hostile(_) => AccountedFailureClassification::Execution,
            },
            ExactOwnedSortPrimary::DecodedPanic(_) => AccountedFailureClassification::Execution,
            ExactOwnedSortPrimary::Terminal(error) => error.classification(),
        }
    }

    fn map_sort_error(error: ExternalSortOperationError) -> OperatorError {
        match error {
            ExternalSortOperationError::Cancelled(error) => OperatorError::QueryCancelled(error),
            ExternalSortOperationError::CancelledWithCleanup {
                error,
                cleanup,
                phase,
            } => OperatorError::QueryCancelled(error)
                .with_context(format!("{phase} also failed: {cleanup}")),
            ExternalSortOperationError::Memory(error) => OperatorError::ResidentMemory(error),
            ExternalSortOperationError::MemoryWithCleanup {
                error,
                cleanup,
                phase,
            } => OperatorError::ResidentMemory(error)
                .with_context(format!("{phase} also failed: {cleanup}")),
            ExternalSortOperationError::Allocation(error) => {
                OperatorError::ResidentAllocation(error.to_string())
            }
            ExternalSortOperationError::Io(error) => OperatorError::from_spill_io_error(error),
            ExternalSortOperationError::WithGrantRelease {
                primary,
                release,
                cleanup,
                phase,
            } => {
                let primary = match primary {
                    ExternalSortPrimary::Cancelled(error) => OperatorError::QueryCancelled(error),
                    ExternalSortPrimary::Memory(error) => OperatorError::ResidentMemory(error),
                    ExternalSortPrimary::Allocation(error) => {
                        OperatorError::ResidentAllocation(error.to_string())
                    }
                    ExternalSortPrimary::Io(error) => OperatorError::from_spill_io_error(error),
                };
                let context = match cleanup {
                    Some(cleanup) => format!(
                        "{phase} grant release also failed: {release}; sort spill cleanup also failed: {cleanup}"
                    ),
                    None => format!("{phase} grant release also failed: {release}"),
                };
                primary.with_context(context)
            }
        }
    }

    /// Preserves the existing operator classifications while keeping owned
    /// reader-operation failures in their pre-admitted cloneable envelope.
    pub(crate) fn into_operator_error(self) -> OperatorError {
        match self.primary {
            ExactOwnedSortPrimary::Sort(error) => Self::map_sort_error(error),
            ExactOwnedSortPrimary::PublisherBuild(error) => match error {
                ExactOwnedPublisherBuildFailure::Memory(error) => {
                    OperatorError::ResidentMemory(error)
                }
                ExactOwnedPublisherBuildFailure::NonZeroGrant => {
                    OperatorError::ResidentContainerInvariant {
                        container: "exact reader-open error publisher",
                        message: "publisher received a nonzero child grant",
                    }
                }
                ExactOwnedPublisherBuildFailure::Allocation => {
                    OperatorError::ResidentContainerInvariant {
                        container: "exact reader-open error publisher",
                        message: "allocator refused the pre-admitted control block",
                    }
                }
                ExactOwnedPublisherBuildFailure::Other => {
                    OperatorError::ResidentContainerInvariant {
                        container: "exact reader-open error publisher",
                        message: "publisher construction failed",
                    }
                }
            },
            ExactOwnedSortPrimary::Accounted {
                classification,
                authority,
            } => OperatorError::ClassifiedAccountedFailure {
                classification,
                authority,
            },
            ExactOwnedSortPrimary::Decoded(_) => {
                panic!("exact decoded failures must pass through the pre-admitted final publisher")
            }
            ExactOwnedSortPrimary::DecodedPanic(_) => {
                panic!("exact decoder panics must pass through the pre-admitted final publisher")
            }
            ExactOwnedSortPrimary::Terminal(error) => error.into_operator_error_without_publisher(),
        }
    }
}

impl std::fmt::Display for ExactOwnedSortStreamError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.primary {
            ExactOwnedSortPrimary::Sort(error) => std::fmt::Display::fmt(error, formatter)?,
            ExactOwnedSortPrimary::PublisherBuild(error) => {
                std::fmt::Display::fmt(error, formatter)?;
            }
            ExactOwnedSortPrimary::Accounted { authority, .. } => {
                std::fmt::Display::fmt(authority, formatter)?;
            }
            ExactOwnedSortPrimary::Decoded(error) => {
                std::fmt::Display::fmt(error, formatter)?;
            }
            ExactOwnedSortPrimary::DecodedPanic(error) => {
                std::fmt::Display::fmt(error, formatter)?;
            }
            ExactOwnedSortPrimary::Terminal(error) => {
                std::fmt::Display::fmt(error, formatter)?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for ExactOwnedSortStreamError {}

impl From<ExternalSortOperationError> for ExactOwnedSortStreamError {
    fn from(error: ExternalSortOperationError) -> Self {
        Self::primary(ExactOwnedSortPrimary::Sort(error))
    }
}

impl From<MemoryGrantError> for ExactOwnedSortStreamError {
    fn from(error: MemoryGrantError) -> Self {
        ExternalSortOperationError::Memory(error).into()
    }
}

impl From<QueryCancellationError> for ExactOwnedSortStreamError {
    fn from(error: QueryCancellationError) -> Self {
        ExternalSortOperationError::Cancelled(error).into()
    }
}

impl From<std::io::Error> for ExactOwnedSortStreamError {
    fn from(error: std::io::Error) -> Self {
        ExternalSortOperationError::Io(error).into()
    }
}

/// One provider-accounted final-run reader.
struct ExactOwnedRunReader {
    reader: Option<ProviderAccountedSpillFileReader>,
    receipt: ProviderAccountedReaderReceipt,
    remaining: u64,
    limits: super::file::SpillFrameLimits,
}

impl ExactOwnedRunReader {
    fn take_reader(&mut self) -> ProviderAccountedSpillFileReader {
        self.reader
            .take()
            .expect("live exact sort run retains its reader")
    }

    fn restore_reader(&mut self, reader: ProviderAccountedSpillFileReader) {
        assert!(
            self.reader.replace(reader).is_none(),
            "exact sort reader restores exactly once"
        );
    }
}

/// One heap head whose decoded allocation and grant move together.
struct ExactOwnedHeapEntry {
    row: AccountedOrdinalRow,
    run_index: usize,
}

/// Exact-layout reader slots paired with the exact-layout merge heap.
type ExactOwnedReaderSlots = ExactVec<Option<ExactOwnedRunReader>, Global>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExactOwnedCursorState {
    Building,
    Active,
    Failed,
    Terminal,
}

/// Safe, explicitly consumable owner used so cursor Drop can publish only
/// after the sorter's own final retrying destructor has completed.
struct ExactOwnedSorterOwner(Option<ExternalSort>);

impl ExactOwnedSorterOwner {
    fn new(sorter: ExternalSort) -> Self {
        Self(Some(sorter))
    }

    fn drop_now(&mut self) -> bool {
        let Some(mut sorter) = self.0.take() else {
            return true;
        };
        let cleanup_succeeded = super::run_cleanup_backstop(|| sorter.cleanup_for_drop());
        if cleanup_succeeded && sorter.runs.is_empty() {
            // No more hook can run from this sorter. Escaped diagnostics own
            // their own shared authority; release only these local clones
            // before testing physical quiescence.
            sorter.writer_workspace.hook_authority = None;
            sorter.pull_hook_workspace = None;
        }
        let quiescent = cleanup_succeeded
            && sorter.exact_failure_publisher.is_none()
            && sorter.runs.is_empty()
            && sorter.runs.entries.capacity() == 0
            && matches!(
                (sorter.checked_total_granted_bytes(), sorter.comparator.checked_granted_bytes()),
                (Ok(total), Ok(comparator)) if total == comparator
            );
        if quiescent {
            drop(sorter);
            true
        } else {
            if !sorter.runs.is_empty() {
                let stranded = u64::try_from(sorter.runs.len()).unwrap_or(u64::MAX);
                super::manager::record_orphan_cleanup_failures(stranded);
            }
            // The final retry did not prove that every fallible release
            // completed. There is no caller to receive the error from Drop,
            // so retain the complete owner graph and its accounting authority
            // fail-closed instead of letting field destructors discard tokens.
            std::mem::forget(sorter);
            false
        }
    }
}

impl std::ops::Deref for ExactOwnedSorterOwner {
    type Target = ExternalSort;

    fn deref(&self) -> &Self::Target {
        self.0
            .as_ref()
            .expect("live exact cursor retains its sorter owner")
    }
}

impl std::ops::DerefMut for ExactOwnedSorterOwner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
            .as_mut()
            .expect("live exact cursor retains its mutable sorter owner")
    }
}

/// Consuming, disk-only final merge that yields one uniquely accounted row.
///
/// Fields are declaration-ordered: decoded heads and readers are destroyed
/// before the sorter can delete runs or release their container authority.
#[must_use = "the exact owned sort cursor must be drained or dropped"]
pub(crate) struct ExactOwnedSortCursor<'observer> {
    heap: ComparatorMinHeap<ExactOwnedHeapEntry>,
    readers: ExactOwnedReaderSlots,
    /// A local reader can fail while being explicitly aborted after another
    /// operation has already established the primary. These fixed slots keep
    /// that independent cleanup result until terminal publication; in
    /// particular, a failed unused-publisher release also keeps the merged
    /// frontier authority retryable.
    pending_reader_cleanup: ExactOwnedPendingReaderCleanup,
    sorter: ExactOwnedSorterOwner,
    reader_bytes: Cell<usize>,
    row_bytes: Cell<usize>,
    observer: ExternalSortGrantObserver<'observer>,
    state: ExactOwnedCursorState,
}

// Counts active merge work, excluding time a consumer holds a yielded row.
struct ProfileMergeTimer {
    manager: Arc<SpillManager>,
    #[cfg(not(target_arch = "wasm32"))]
    started: std::time::Instant,
}
impl ProfileMergeTimer {
    fn new(manager: &Arc<SpillManager>) -> Option<Self> {
        if !manager.profile_merge_enabled() {
            return None;
        }
        Some(Self {
            manager: Arc::clone(manager),
            #[cfg(not(target_arch = "wasm32"))]
            started: std::time::Instant::now(),
        })
    }
}
impl Drop for ProfileMergeTimer {
    fn drop(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        self.manager.record_merge_time(
            u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        );
        #[cfg(target_arch = "wasm32")]
        let _ = &self.manager;
    }
}

/// Send-capable owner for the exact cursor used by resumable consumers.
///
/// Only owned counters and telemetry cross calls. The existing cursor is
/// reconstructed temporarily, so its borrowed-observer compatibility variants
/// never become part of this owner's type or require a self-reference.
pub(crate) struct OwnedExactSortCursor {
    storage: Option<OwnedExactSortCursorStorage>,
    telemetry: OwnedExactSortTelemetry,
}

struct OwnedExactSortTelemetry {
    external_bytes: Cell<usize>,
    retained_bytes: Cell<usize>,
    spill_state: Arc<OperatorSpillState>,
    unacknowledged_retained_poisoned: Cell<bool>,
}

impl OwnedExactSortTelemetry {
    fn observer(&self) -> ExternalSortGrantObserver<'_> {
        let mut observer = ExternalSortGrantObserver::new(
            &self.external_bytes,
            &self.retained_bytes,
            Some(&self.spill_state),
        );
        observer.unacknowledged_retained_poisoned =
            Cell::new(self.unacknowledged_retained_poisoned.get());
        observer
    }
}

struct OwnedExactSortCursorStorage {
    heap: ComparatorMinHeap<ExactOwnedHeapEntry>,
    readers: ExactOwnedReaderSlots,
    pending_reader_cleanup: ExactOwnedPendingReaderCleanup,
    sorter: ExactOwnedSorterOwner,
    reader_bytes: Cell<usize>,
    row_bytes: Cell<usize>,
    state: ExactOwnedCursorState,
}

impl OwnedExactSortCursorStorage {
    fn from_cursor(mut cursor: ExactOwnedSortCursor<'_>) -> Self {
        Self {
            heap: std::mem::replace(&mut cursor.heap, ComparatorMinHeap::new()),
            readers: std::mem::replace(&mut cursor.readers, ExactOwnedReaderSlots::new_in(Global)),
            pending_reader_cleanup: std::mem::take(&mut cursor.pending_reader_cleanup),
            // The empty shell owns no grants and must not clean up transferred state.
            sorter: ExactOwnedSorterOwner(cursor.sorter.0.take()),
            reader_bytes: Cell::new(cursor.reader_bytes.get()),
            row_bytes: Cell::new(cursor.row_bytes.get()),
            state: cursor.state,
        }
    }

    fn into_cursor(self, observer: ExternalSortGrantObserver<'_>) -> ExactOwnedSortCursor<'_> {
        ExactOwnedSortCursor {
            heap: self.heap,
            readers: self.readers,
            pending_reader_cleanup: self.pending_reader_cleanup,
            sorter: self.sorter,
            reader_bytes: self.reader_bytes,
            row_bytes: self.row_bytes,
            observer,
            state: self.state,
        }
    }
}

#[expect(
    clippy::result_large_err,
    reason = "the exact lane keeps rich cleanup failures inline because boxing could allocate on an already-failing, resource-accounted path"
)]
impl OwnedExactSortCursor {
    fn missing_storage() -> ExactOwnedSortStreamError {
        ExternalSortOperationError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "owned exact cursor has no resumable storage",
        ))
        .into()
    }

    fn with_cursor<T>(
        &mut self,
        action: impl FnOnce(&mut ExactOwnedSortCursor<'_>) -> T,
    ) -> Result<T, ExactOwnedSortStreamError> {
        let storage = self.storage.take().ok_or_else(Self::missing_storage)?;
        let mut cursor = storage.into_cursor(self.telemetry.observer());
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| action(&mut cursor)));
        // Restoration cannot fail: telemetry is owned independently and the
        // transient cursor contains only the resource fields moved out above.
        self.telemetry
            .unacknowledged_retained_poisoned
            .set(cursor.observer.unacknowledged_retained_poisoned.get());
        self.storage = Some(OwnedExactSortCursorStorage::from_cursor(cursor));
        match outcome {
            Ok(value) => Ok(value),
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    pub(crate) fn next_owned_row(
        &mut self,
    ) -> Result<Option<AccountedOrdinalRow>, ExactOwnedSortStreamError> {
        self.with_cursor(|cursor| cursor.next_owned_row())?
    }

    #[cfg(test)]
    fn num_columns(&self) -> Result<usize, ExactOwnedSortStreamError> {
        let storage = self.storage.as_ref().ok_or_else(Self::missing_storage)?;
        Ok(storage.sorter.num_columns)
    }

    #[cfg(test)]
    fn checked_granted_bytes(&self) -> Result<usize, ExactOwnedSortStreamError> {
        let storage = self.storage.as_ref().ok_or_else(Self::missing_storage)?;
        Ok(checked_workspace_sum(
            storage.sorter.checked_total_granted_bytes()?,
            checked_workspace_sum(storage.reader_bytes.get(), storage.row_bytes.get())?,
        )?)
    }

    pub(crate) fn release_transferred_retained(&mut self) -> Result<(), ExactOwnedSortStreamError> {
        self.with_cursor(|cursor| cursor.release_transferred_retained())?
    }

    pub(crate) fn finish_stream_failure(
        &mut self,
        primary: ExactOwnedSortStreamError,
        cleanup_phase: &'static str,
    ) -> OperatorError {
        if self.storage.is_none() {
            // Missing storage cannot replace an already established primary.
            return primary.into_operator_error();
        }
        self.with_cursor(|cursor| cursor.finish_stream_failure(primary, cleanup_phase))
            .unwrap_or_else(ExactOwnedSortStreamError::into_operator_error)
    }

    pub(crate) fn finish_operator_failure(
        &mut self,
        primary: OperatorError,
        secondary: Option<(ExactOwnedSortStreamError, &'static str)>,
        cleanup_phase: &'static str,
    ) -> OperatorError {
        if self.storage.is_none() {
            return primary;
        }
        self.with_cursor(|cursor| cursor.finish_operator_failure(primary, secondary, cleanup_phase))
            .unwrap_or_else(ExactOwnedSortStreamError::into_operator_error)
    }

    pub(crate) fn finish_early_stop(&mut self) -> Result<(), OperatorError> {
        self.with_cursor(|cursor| cursor.finish_early_stop())
            .map_err(ExactOwnedSortStreamError::into_operator_error)?
    }
}

impl Drop for OwnedExactSortCursor {
    fn drop(&mut self) {
        if let Some(storage) = self.storage.take() {
            drop(storage.into_cursor(self.telemetry.observer()));
        }
    }
}

impl std::fmt::Debug for ExternalSortCursor<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExternalSortCursor")
            .field("buffered_output_rows", &self.output_rows.len())
            .field("frontier_heads", &self.heap.len())
            .field("live_readers", &self.run_readers.iter().flatten().count())
            .field("max_chunk_rows", &self.max_chunk_rows)
            .field("terminal", &self.terminal)
            .finish_non_exhaustive()
    }
}

struct ExternalSortCursorState {
    output_rows: Vec<Vec<Value>>,
    heap: ComparatorMinHeap<HeapEntry>,
    run_readers: Vec<Option<CursorRunReader>>,
    memory_iter: std::vec::IntoIter<OrdinalRow>,
    memory_run_index: usize,
    max_chunk_rows: usize,
    frontier_base_bytes: usize,
    frontier_reader_bytes: usize,
    frontier_row_bytes: usize,
    output_base_bytes: usize,
    output_row_bytes: usize,
}

/// External merge sort for out-of-core sorting.
///
/// Manages sorted runs on disk and provides stable, capped-fan-in merge.
/// `merge_all` still collects the complete output; the streaming cursor stage
/// will replace that remaining compatibility limitation.
pub struct ExternalSort {
    /// Spill manager for file creation.
    manager: Arc<SpillManager>,
    /// Reusable, optionally grant-owned row-codec workspace.
    workspace: ExternalSortWorkspace,
    /// Atomically paired, optionally grant-owned run catalog.
    runs: ExternalSortRunCatalog,
    /// Transient fixed writer backing and its optional query-memory token.
    writer_workspace: ExternalSortWriterWorkspace,
    pull_hook_workspace: Option<AccountedError>,
    pull_control_bytes: usize,
    /// Outer slots used while assigning durable ordinals to the resident tail.
    ordinal_grant: Option<MemoryGrant>,
    /// Reader/head/heap storage retained by an active merge cursor.
    frontier_grant: Option<MemoryGrant>,
    /// Reusable stored/plaintext payload peak retained through head decoding.
    payload_grant: Option<MemoryGrant>,
    /// The single reusable output batch retained by an active merge cursor.
    output_grant: Option<MemoryGrant>,
    /// One exact final-failure carrier, present only while the move-consuming
    /// cursor exclusively owns this sorter.
    exact_failure_publisher: Option<AccountedErrorPublisher<ExactOwnedFinalFailure>>,
    scalar_failure_publication_bytes: usize,
    scalar_cleanup: Option<AccountedError>,
    /// One-shot publisher-release failure injection for terminal retry tests.
    #[cfg(test)]
    exact_final_publisher_release_error: Option<MemoryGrantError>,
    /// Deterministic, owner-local decoder unwind injection for hostile tests.
    #[cfg(test)]
    exact_decoder_panic: std::sync::Mutex<Option<Box<dyn std::any::Any + Send>>>,
    /// Immutable check-only capability for resource-qualified operations.
    cancellation: Option<QueryCancellationToken>,
    /// Number of public columns per row.
    num_columns: usize,
    row_shape: SortRowShape,
    /// Caller-selected semantic ordering; ordinal stability is added here.
    comparator: SemanticRowComparator,
    /// Unforgeable admission for the exact allocation-bounded comparator lane.
    comparator_qualification: ComparatorQualification,
    /// First ordinal not yet assigned to an accepted input row.
    next_input_ordinal: u64,
    /// Maximum readers and heap heads participating in one merge step.
    merge_fan_in: usize,
    /// Binary carries over DISTINCT input batches; levels are implicit in the
    /// batch-count bits, so no independently allocated level catalog is needed.
    distinct_run_schedule: DistinctRunSchedule,
    // One means DISTINCT; pull sort freezes a power-of-two initial fan-in.
    initial_run_group: Option<usize>,
    /// Actual row visits by successful intermediate merges, used to verify
    /// the production schedule's work bound independently of its counters.
    #[cfg(test)]
    merged_row_visits: u128,
    /// Disk merge is consuming and cannot be retried after partial cleanup.
    disk_merge_started: bool,
}

#[derive(Clone, Copy)]
enum DistinctRunSchedule {
    Unused,
    Appending(u64),
    Sealed,
    Failed,
}

#[expect(
    clippy::result_large_err,
    reason = "the exact lane keeps rich cleanup failures inline because boxing could allocate on an already-failing, resource-accounted path"
)]
impl ExternalSort {
    /// Enables the graph sort owner's private, optional per-row provenance.
    /// Generic callers retain strict fixed-width rows.
    pub(crate) fn enable_edge_provenance(&mut self) {
        assert_eq!(
            self.next_input_ordinal, 0,
            "row shape is fixed before accepting runs"
        );
        self.row_shape = SortRowShape::with_edge_trailer(self.num_columns);
    }

    /// Creates a new external sort.
    #[must_use]
    pub fn new(manager: Arc<SpillManager>, num_columns: usize, sort_keys: Vec<SortKey>) -> Self {
        let comparator = SemanticRowComparator::new(move |left, right| {
            compare_rows(
                &left[..left.len().min(num_columns)],
                &right[..right.len().min(num_columns)],
                &sort_keys,
            )
        });
        Self::new_inner(
            manager,
            num_columns,
            comparator,
            ComparatorQualification::BuiltInSortKeys,
            None,
            None,
        )
    }

    /// Creates an external sort using caller-defined semantic row ordering.
    ///
    /// Stable encounter order is owned by the sorter and must not be included
    /// in `comparator`; an ascending durable ordinal is applied after it.
    #[must_use]
    pub fn new_with_comparator(
        manager: Arc<SpillManager>,
        num_columns: usize,
        compare: impl Fn(&[Value], &[Value]) -> Ordering + Send + Sync + 'static,
    ) -> Self {
        Self::new_inner(
            manager,
            num_columns,
            SemanticRowComparator::new(compare),
            ComparatorQualification::UnqualifiedCustom,
            None,
            None,
        )
    }

    /// Test-only accounted constructor with an inert execution control.
    ///
    /// Release builds expose only the resource-qualified accounted constructor,
    /// so production code cannot accidentally omit cancellation authority.
    #[cfg(test)]
    pub(crate) fn new_accounted(
        manager: Arc<SpillManager>,
        num_columns: usize,
        sort_keys: Vec<SortKey>,
        grant: MemoryGrant,
    ) -> Self {
        assert_eq!(
            grant.size(),
            0,
            "accounted external sort requires a dedicated zero-sized root grant"
        );
        let comparator = SemanticRowComparator::new(move |left, right| {
            compare_rows(
                &left[..left.len().min(num_columns)],
                &right[..right.len().min(num_columns)],
                &sort_keys,
            )
        });
        Self::new_inner(
            manager,
            num_columns,
            comparator,
            ComparatorQualification::BuiltInSortKeys,
            Some(grant),
            Some(crate::execution::QueryExecutionControl::new().token()),
        )
    }

    pub(crate) fn new_accounted_with_cancellation(
        manager: Arc<SpillManager>,
        num_columns: usize,
        sort_keys: Vec<SortKey>,
        grant: MemoryGrant,
        cancellation: QueryCancellationToken,
    ) -> Self {
        assert_eq!(
            grant.size(),
            0,
            "accounted external sort requires a dedicated zero-sized root grant"
        );
        let comparator = SemanticRowComparator::new(move |left, right| {
            compare_rows(
                &left[..left.len().min(num_columns)],
                &right[..right.len().min(num_columns)],
                &sort_keys,
            )
        });
        Self::new_inner(
            manager,
            num_columns,
            comparator,
            ComparatorQualification::BuiltInSortKeys,
            Some(grant),
            Some(cancellation),
        )
    }

    /// Creates a resource-accounted external sort with caller-defined ordering.
    ///
    /// The zero-sized root grant grows only through fallible qualified paths;
    /// `cancellation` is checked throughout spill and merge work. An accounted
    /// comparator must use the same query resource account as `grant`; terminal
    /// cleanup transfers its uniquely owned scratch grant into that account's
    /// existing release owner. Unrelated accounts retain compatibility ordering
    /// but are rejected by exact owned-output qualification.
    ///
    /// # Panics
    ///
    /// Panics when `grant` is not the dedicated zero-sized root grant required
    /// to keep every later allocation transition fallible and attributable.
    pub fn new_accounted_with_comparator_and_cancellation(
        manager: Arc<SpillManager>,
        num_columns: usize,
        comparator: SemanticRowComparator,
        mut grant: MemoryGrant,
        cancellation: QueryCancellationToken,
    ) -> Self {
        assert_eq!(
            grant.size(),
            0,
            "accounted external sort requires a dedicated zero-sized root grant"
        );
        // Prove shared account identity without allocating or moving charged
        // bytes. A foreign comparator remains usable by compatibility paths,
        // but cannot enter the exact lane's terminal grant-transfer contract.
        let same_account = match &comparator.compare {
            Comparison::Accounted(accounted) => accounted
                .grant
                .lock()
                .split(0)
                .is_some_and(|child| grant.try_merge(child).is_ok()),
            Comparison::Infallible(_) | Comparison::Released => false,
        };
        let qualification = if same_account {
            ComparatorQualification::AccountedProvider
        } else {
            ComparatorQualification::UnqualifiedCustom
        };
        Self::new_inner(
            manager,
            num_columns,
            comparator,
            qualification,
            Some(grant),
            Some(cancellation),
        )
    }

    fn new_inner(
        manager: Arc<SpillManager>,
        num_columns: usize,
        comparator: SemanticRowComparator,
        comparator_qualification: ComparatorQualification,
        grant: Option<MemoryGrant>,
        cancellation: Option<QueryCancellationToken>,
    ) -> Self {
        debug_assert_eq!(grant.is_some(), cancellation.is_some());
        let (
            workspace_grant,
            run_catalog_grant,
            writer_grant,
            ordinal_grant,
            frontier_grant,
            payload_grant,
            output_grant,
        ) = match grant {
            Some(mut grant) => {
                let mut catalog = grant
                    .split(0)
                    .expect("zero-sized accounted sorter grant must split at zero");
                let mut writer = catalog
                    .split(0)
                    .expect("zero-sized accounted sorter grant must split twice at zero");
                let mut ordinal = writer
                    .split(0)
                    .expect("zero-sized accounted sorter grant must split three times at zero");
                let mut frontier = ordinal
                    .split(0)
                    .expect("zero-sized accounted sorter grant must split four times at zero");
                let mut payload = frontier
                    .split(0)
                    .expect("zero-sized accounted sorter grant must split five times at zero");
                let output = payload
                    .split(0)
                    .expect("zero-sized accounted sorter grant must split six times at zero");
                (
                    Some(grant),
                    Some(catalog),
                    Some(writer),
                    Some(ordinal),
                    Some(frontier),
                    Some(payload),
                    Some(output),
                )
            }
            None => (None, None, None, None, None, None, None),
        };
        let maximum =
            usize::try_from(manager.frame_limits().max_plaintext_bytes()).unwrap_or(usize::MAX);
        Self {
            manager,
            workspace: ExternalSortWorkspace::new(maximum, workspace_grant),
            runs: ExternalSortRunCatalog::new(run_catalog_grant),
            writer_workspace: ExternalSortWriterWorkspace::new(writer_grant),
            pull_hook_workspace: None,
            pull_control_bytes: 0,
            ordinal_grant,
            frontier_grant,
            payload_grant,
            output_grant,
            exact_failure_publisher: None,
            scalar_failure_publication_bytes: 0,
            scalar_cleanup: None,
            #[cfg(test)]
            exact_final_publisher_release_error: None,
            #[cfg(test)]
            exact_decoder_panic: std::sync::Mutex::new(None),
            cancellation,
            num_columns,
            row_shape: SortRowShape::strict(num_columns),
            comparator,
            comparator_qualification,
            next_input_ordinal: 0,
            merge_fan_in: DEFAULT_MERGE_FAN_IN,
            distinct_run_schedule: DistinctRunSchedule::Unused,
            initial_run_group: None,
            #[cfg(test)]
            merged_row_visits: 0,
            disk_merge_started: false,
        }
    }

    fn recover_exact_publisher_build_error(
        &mut self,
        error: AccountedErrorPublisherBuildError,
    ) -> ExactOwnedPublisherBuildFailure {
        let failure = ExactOwnedPublisherBuildFailure::classify(&error);
        self.merge_exact_publisher_grant(error.into_grant());
        failure
    }

    fn merge_exact_publisher_grant(&mut self, grant: MemoryGrant) {
        self.merge_exact_frontier_grant(grant, "publisher build authority");
    }

    fn merge_exact_frontier_grant(&mut self, grant: MemoryGrant, owner: &'static str) {
        let frontier = self
            .frontier_grant
            .as_mut()
            .expect("accounted exact sorter retains its frontier grant");
        if let Err(grant) = frontier.try_merge(grant) {
            let stranded_bytes = grant.size();
            // Every caller transfers a child of this sorter's query account,
            // so identity, region, and checked sum make rejection unreachable.
            // If that invariant is ever broken, fail closed instead of
            // dropping the sole retry authority.
            std::mem::forget(grant);
            panic!(
                "exact {owner} could not rejoin its frontier ({stranded_bytes} bytes retained fail-closed)"
            );
        }
    }

    fn try_admit_exact_final_publisher(&mut self) -> Result<(), ExactOwnedPublisherBuildFailure> {
        assert!(
            self.exact_failure_publisher.is_none(),
            "exact final failure publisher is admitted exactly once"
        );
        let grant = self
            .frontier_grant
            .as_mut()
            .and_then(|grant| grant.split(0))
            .expect("accounted exact sorter retains a splittable frontier grant");
        match AccountedErrorPublisher::try_new(grant) {
            Ok(publisher) => {
                self.exact_failure_publisher = Some(publisher);
                Ok(())
            }
            Err(error) => Err(self.recover_exact_publisher_build_error(error)),
        }
    }

    #[cfg(test)]
    fn inject_exact_decoder_panic(&mut self, payload: Box<dyn std::any::Any + Send>) {
        assert!(
            self.exact_decoder_panic
                .lock()
                .expect("exact decoder panic injection mutex is not poisoned")
                .replace(payload)
                .is_none(),
            "exact decoder panic injection is single-shot"
        );
    }

    #[cfg(test)]
    pub(crate) fn inject_persistent_exact_workspace_release_failure(
        &mut self,
        error: MemoryGrantError,
    ) {
        assert!(
            self.workspace.granted_bytes() > 0,
            "persistent exact release failure requires live workspace authority"
        );
        self.workspace.release_error = Some(error);
    }

    /// Attempts the optional exact-output admission without consuming the
    /// sorter. Denial leaves every compatibility merge path available.
    pub(crate) fn try_enable_exact_owned_output(
        &mut self,
        spill_state: Option<&OperatorSpillState>,
    ) -> bool {
        if self.exact_failure_publisher.is_some() {
            return true;
        }
        let Ok(current_bytes) = self.checked_total_granted_bytes() else {
            if let Some(state) = spill_state {
                state.set_usage(usize::MAX);
            }
            return false;
        };
        let Some(admission_bytes) = current_bytes
            .checked_add(AccountedErrorPublisher::<ExactOwnedFinalFailure>::required_bytes())
        else {
            if let Some(state) = spill_state {
                state.set_usage(current_bytes);
            }
            return false;
        };
        let external_bytes = Cell::new(current_bytes);
        let retained_bytes = Cell::new(0);
        let observer =
            ExternalSortGrantObserver::new(&external_bytes, &retained_bytes, spill_state);
        // The control-block layout is exact. Publish it conservatively before
        // grant admission can run eviction machinery, then reconcile to the
        // actual sorter total on both success and optional-lane denial.
        if observer.publish(admission_bytes).is_err() {
            let _ = observer.publish(current_bytes);
            return false;
        }
        match self.try_admit_exact_final_publisher() {
            Ok(()) => {
                let _ = observer.publish(admission_bytes);
                true
            }
            Err(_) => {
                let actual = self.checked_total_granted_bytes().unwrap_or(usize::MAX);
                let _ = observer.publish(actual);
                false
            }
        }
    }

    pub(crate) fn workspace_granted_bytes(&self) -> usize {
        self.workspace.granted_bytes()
    }

    pub(crate) fn run_catalog_granted_bytes(&self) -> usize {
        self.runs.granted_bytes()
    }

    pub(crate) fn writer_workspace_granted_bytes(&self) -> usize {
        self.writer_workspace.granted_bytes()
    }

    pub(crate) fn checked_total_granted_bytes(&self) -> Result<usize, MemoryGrantError> {
        [
            self.comparator.checked_granted_bytes()?,
            self.pull_retained_bytes()?,
            self.workspace_granted_bytes(),
            self.run_catalog_granted_bytes(),
            self.writer_workspace_granted_bytes(),
            self.ordinal_granted_bytes(),
            self.frontier_granted_bytes(),
            self.payload_granted_bytes(),
            self.output_granted_bytes(),
            self.exact_failure_publisher
                .as_ref()
                .map_or(0, AccountedErrorPublisher::granted_bytes),
            self.scalar_failure_publication_bytes,
            self.scalar_cleanup
                .as_ref()
                .map_or(0, AccountedError::granted_bytes),
            self.scalar_cleanup.as_ref().map_or(0, |authority| {
                authority
                    .inspect::<ScalarReaderCleanup, _>(ScalarReaderCleanup::granted_bytes)
                    .unwrap_or(usize::MAX)
            }),
        ]
        .into_iter()
        .try_fold(0usize, |current_bytes, additional_bytes| {
            current_bytes.checked_add(additional_bytes).ok_or(
                MemoryGrantError::ArithmeticOverflow {
                    current_bytes,
                    additional_bytes,
                },
            )
        })
    }

    #[cfg(test)]
    pub(crate) fn total_granted_bytes(&self) -> usize {
        self.checked_total_granted_bytes()
            .expect("split sorter grants remain one representable query allocation")
    }

    fn publish_granted_bytes(
        &self,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        let total = match self.checked_total_granted_bytes() {
            Ok(total) => total,
            Err(error) => {
                observer.publish_unrepresentable();
                return Err(error.into());
            }
        };
        observer.publish(total).map_err(Into::into)
    }

    fn publish_granted_bytes_preserving_primary(&self, observer: &ExternalSortGrantObserver<'_>) {
        match self.checked_total_granted_bytes() {
            Ok(total) => {
                let _ = observer.publish(total);
            }
            Err(_) => observer.publish_unrepresentable(),
        }
    }

    fn ordinal_granted_bytes(&self) -> usize {
        self.ordinal_grant.as_ref().map_or(0, MemoryGrant::size)
    }

    fn frontier_granted_bytes(&self) -> usize {
        self.frontier_grant.as_ref().map_or(0, MemoryGrant::size)
    }

    fn output_granted_bytes(&self) -> usize {
        self.output_grant.as_ref().map_or(0, MemoryGrant::size)
    }

    fn payload_granted_bytes(&self) -> usize {
        self.payload_grant.as_ref().map_or(0, MemoryGrant::size)
    }

    fn resize_ordinal_grant(
        &mut self,
        bytes: usize,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        let result = self
            .ordinal_grant
            .as_mut()
            .map_or(Ok(()), |grant| grant.try_resize(bytes));
        let observation = self.publish_granted_bytes(observer);
        match result {
            Ok(()) => observation,
            Err(error) => Err(error.into()),
        }
    }

    fn resize_frontier_grant(
        &mut self,
        bytes: usize,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        let result = self
            .frontier_grant
            .as_mut()
            .map_or(Ok(()), |grant| grant.try_resize(bytes));
        let observation = self.publish_granted_bytes(observer);
        match result {
            Ok(()) => observation,
            Err(error) => Err(error.into()),
        }
    }

    fn resize_output_grant(
        &mut self,
        bytes: usize,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        let result = self
            .output_grant
            .as_mut()
            .map_or(Ok(()), |grant| grant.try_resize(bytes));
        let observation = self.publish_granted_bytes(observer);
        match result {
            Ok(()) => observation,
            Err(error) => Err(error.into()),
        }
    }

    fn resize_payload_grant(
        &mut self,
        bytes: usize,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        let bytes = bytes.max(self.payload_granted_bytes());
        let result = self
            .payload_grant
            .as_mut()
            .map_or(Ok(()), |grant| grant.try_resize(bytes));
        let observation = self.publish_granted_bytes(observer);
        match result {
            Ok(()) => observation,
            Err(error) => Err(error.into()),
        }
    }

    fn release_payload_grant(
        &mut self,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        let result = self
            .payload_grant
            .as_mut()
            .map_or(Ok(()), |grant| grant.try_resize(0));
        let observation = self.publish_granted_bytes(observer);
        match result {
            Ok(()) => observation,
            Err(error) => Err(error.into()),
        }
    }

    fn release_cursor_grants(
        &mut self,
        release_output: bool,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        let mut first_failure = None;
        for grant in [
            &mut self.ordinal_grant,
            &mut self.frontier_grant,
            &mut self.payload_grant,
        ] {
            if let Some(grant) = grant
                && let Err(error) = grant.try_resize(0)
                && first_failure.is_none()
            {
                first_failure = Some(error);
            }
        }
        if release_output
            && let Some(grant) = self.output_grant.as_mut()
            && let Err(error) = grant.try_resize(0)
            && first_failure.is_none()
        {
            first_failure = Some(error);
        }
        let observation = self.publish_granted_bytes(observer);
        match first_failure {
            Some(error) => Err(ExternalSortOperationError::Memory(error)),
            None => observation,
        }
    }

    /// Publishes the live post-unwind total before resuming the primary panic.
    ///
    /// The sealed observer cannot allocate, unwind, or invoke caller code, so
    /// reconciliation needs no secondary catch/leak path.
    fn resume_after_accounted_unwind(
        &self,
        observer: &ExternalSortGrantObserver<'_>,
        panic: Box<dyn std::any::Any + Send>,
    ) -> ! {
        // A forged/mispaired in-crate observer could report an impossible
        // cross-grant overflow. The external scalar is already reconciled and
        // telemetry is fail-closed at `usize::MAX`; never replace the original
        // operation panic with that secondary invariant failure.
        self.publish_granted_bytes_preserving_primary(observer);
        std::panic::resume_unwind(panic)
    }

    #[cfg(test)]
    fn run_catalog_observed_bytes(&self) -> Result<usize, ExternalSortOperationError> {
        self.runs.observed_bytes()
    }

    #[cfg(test)]
    fn run_catalog_pointer(&self) -> *const ExternalSortRunEntry {
        self.runs.pointer()
    }

    #[cfg(test)]
    fn run_catalog_capacity(&self) -> usize {
        self.runs.capacity()
    }

    #[cfg(test)]
    fn set_merge_fan_in(&mut self, merge_fan_in: usize) {
        assert!(merge_fan_in >= 2, "merge fan-in must permit progress");
        assert!(
            !self.disk_merge_started,
            "merge fan-in is immutable after merge starts"
        );
        self.merge_fan_in = merge_fan_in;
    }

    /// Returns the number of runs on disk.
    #[must_use]
    pub fn num_runs(&self) -> usize {
        self.runs.len()
    }

    /// Returns whether this sorter has the structural shape admitted by the
    /// first production owned-row lane, without touching the optional exact
    /// reader capability seam. Callers must establish a qualified sink before
    /// invoking [`Self::exact_owned_disk_shape_eligible`].
    pub(crate) fn exact_owned_disk_base_shape_eligible(&self) -> bool {
        matches!(
            self.comparator_qualification,
            ComparatorQualification::BuiltInSortKeys | ComparatorQualification::AccountedProvider
        ) && self.cancellation.is_some()
            && self.frontier_grant.is_some()
            && self.payload_grant.is_some()
            && self.num_columns > 0
            && !self.runs.is_empty()
            && self.runs.len() <= self.merge_fan_in
            && !self.disk_merge_started
    }

    /// Returns whether the structurally eligible sorter also has a cached,
    /// hard-qualified reader contract for every run. This check mints and then
    /// discards only identity receipts; provider and hook callbacks are cached
    /// once per file, and no reader/open authority is consumed.
    pub(crate) fn exact_owned_disk_shape_eligible(&self) -> bool {
        self.exact_owned_disk_base_shape_eligible()
            && self
                .runs
                .iter()
                .all(|run| run.file.exact_owned_reader_qualification().is_some())
    }

    fn split_exact_owned_child(&mut self) -> Result<MemoryGrant, ExactOwnedSortStreamError> {
        self.frontier_grant
            .as_mut()
            .and_then(|grant| grant.split(0))
            .ok_or_else(|| {
                ExternalSortOperationError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "exact owned sort lost its frontier grant",
                ))
                .into()
            })
    }

    fn take_exact_owned_payload_grant(&mut self) -> Result<MemoryGrant, ExactOwnedSortStreamError> {
        let replacement = self.split_exact_owned_child()?;
        let payload = self.payload_grant.replace(replacement).ok_or_else(|| {
            ExactOwnedSortStreamError::from(ExternalSortOperationError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "exact owned sort transferred its payload grant more than once",
            )))
        })?;
        if payload.size() != 0 {
            let replacement = self
                .payload_grant
                .replace(payload)
                .expect("payload replacement was installed before validation");
            drop(replacement);
            return Err(ExternalSortOperationError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "exact owned sort payload grant was not quiescent before transfer",
            ))
            .into());
        }
        Ok(payload)
    }

    /// Consumes an eligible, disk-only final frontier into exact decoded row
    /// owners. No compatibility state is modified until every shape check and
    /// both exact container admissions have succeeded.
    pub(crate) fn into_exact_owned_disk_cursor(
        self,
        observer: ExternalSortGrantObserver<'_>,
    ) -> Result<ExactOwnedSortCursor<'_>, ExactOwnedSortStreamError> {
        if self.exact_failure_publisher.is_none() {
            return Err(ExternalSortOperationError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "exact owned output requires successful pre-admission before sorter consumption",
            ))
            .into());
        }
        self.publish_granted_bytes(&observer)?;
        let mut cursor = ExactOwnedSortCursor {
            heap: ComparatorMinHeap::new(),
            readers: ExactOwnedReaderSlots::new_in(Global),
            pending_reader_cleanup: ExactOwnedPendingReaderCleanup::default(),
            sorter: ExactOwnedSorterOwner::new(self),
            reader_bytes: Cell::new(0),
            row_bytes: Cell::new(0),
            observer,
            state: ExactOwnedCursorState::Building,
        };
        let initialization = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cursor.initialize()?;
            cursor.state = ExactOwnedCursorState::Active;
            cursor.publish_live()
        }));
        match initialization {
            Ok(Ok(())) => Ok(cursor),
            Ok(Err(primary)) => {
                cursor.state = ExactOwnedCursorState::Failed;
                Err(cursor.publish_initialization_failure(primary))
            }
            Err(payload) => {
                cursor.state = ExactOwnedCursorState::Failed;
                let _ = super::run_cleanup_backstop(|| {
                    cursor
                        .finish_resources_preserving_final_publisher()
                        .map_or(Ok(()), Err)
                });
                std::panic::resume_unwind(payload);
            }
        }
    }

    /// Consumes the exact frontier with exclusively owned observation, making
    /// the resumable owner Send without changing the borrowed sort interface.
    pub(crate) fn into_send_owned_disk_cursor(
        self,
        observer: ExternalSortGrantObserver<'static>,
    ) -> Result<OwnedExactSortCursor, ExactOwnedSortStreamError> {
        #[cfg(test)]
        if observer.peak_external_bytes.is_some() {
            return Err(ExternalSortOperationError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Send exact cursor requires exclusively owned observation",
            ))
            .into());
        }
        let ExternalSortGrantObserverTarget::Accounted {
            external_bytes: ObserverCounter::Owned(external_bytes),
            retained_bytes: ObserverCounter::Owned(retained_bytes),
            spill_state: Some(ObserverSpillState::Owned(spill_state)),
        } = observer.target
        else {
            return Err(ExternalSortOperationError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Send exact cursor requires exclusively owned observation",
            ))
            .into());
        };
        let telemetry = OwnedExactSortTelemetry {
            external_bytes,
            retained_bytes,
            spill_state,
            unacknowledged_retained_poisoned: observer.unacknowledged_retained_poisoned,
        };
        let cursor = self.into_exact_owned_disk_cursor(telemetry.observer())?;
        telemetry
            .unacknowledged_retained_poisoned
            .set(cursor.observer.unacknowledged_retained_poisoned.get());
        let storage = OwnedExactSortCursorStorage::from_cursor(cursor);
        Ok(OwnedExactSortCursor {
            storage: Some(storage),
            telemetry,
        })
    }

    /// Returns the total number of rows across all runs, saturating at
    /// `usize::MAX` for compatibility with this infallible observer.
    #[must_use]
    pub fn total_rows(&self) -> usize {
        self.runs
            .iter()
            .map(|entry| entry.rows)
            .fold(0, usize::saturating_add)
    }

    #[cfg(test)]
    fn checked_total_rows(&self) -> std::io::Result<usize> {
        self.runs.iter().try_fold(0usize, |total, entry| {
            total.checked_add(entry.rows).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "tracked sort row count exceeds the platform address space",
                )
            })
        })
    }

    fn checked_total_rows_with_cancellation(
        &self,
        cancellation: Option<&QueryCancellationToken>,
    ) -> Result<usize, ExternalSortOperationError> {
        let mut total = 0usize;
        for entry in self.runs.iter() {
            check_cancellation(cancellation)?;
            total = total.checked_add(entry.rows).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "tracked sort row count exceeds the platform address space",
                )
            })?;
            check_cancellation(cancellation)?;
        }
        Ok(total)
    }

    fn attach_input_ordinals(
        &mut self,
        rows: Vec<Vec<Value>>,
    ) -> Result<Vec<OrdinalRow>, ExternalSortOperationError> {
        for row in &rows {
            self.row_shape.edge_mask(row).map_err(|message| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
            })?;
        }
        let row_count = u64::try_from(rows.len()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "sort input row count exceeds u64",
            )
        })?;
        let first_ordinal = self.next_input_ordinal;
        let next_input_ordinal = first_ordinal.checked_add(row_count).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "sort input ordinal exceeds u64",
            )
        })?;
        let mut stable_rows = Vec::new();
        stable_rows.try_reserve_exact(rows.len()).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                format!("reserve ordinal sort rows: {error}"),
            )
        })?;
        for (offset, values) in rows.into_iter().enumerate() {
            let offset = u64::try_from(offset).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "sort input ordinal exceeds u64",
                )
            })?;
            stable_rows.push(OrdinalRow {
                values,
                ordinal: first_ordinal + offset,
            });
        }
        self.next_input_ordinal = next_input_ordinal;
        Ok(stable_rows)
    }

    fn attach_input_ordinals_for_cursor(
        &mut self,
        rows: Vec<Vec<Value>>,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<Vec<OrdinalRow>, ExternalSortOperationError> {
        for row in &rows {
            self.row_shape.edge_mask(row).map_err(|message| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
            })?;
        }
        let row_count = u64::try_from(rows.len()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "sort input row count exceeds u64",
            )
        })?;
        let first_ordinal = self.next_input_ordinal;
        let next_input_ordinal = first_ordinal.checked_add(row_count).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "sort input ordinal exceeds u64",
            )
        })?;
        let previous = self.ordinal_granted_bytes();
        let requested = cursor_capacity_bytes::<OrdinalRow>(rows.len())?;
        self.resize_ordinal_grant(requested, observer)?;

        let mut stable_rows = Vec::new();
        if let Err(error) = stable_rows.try_reserve_exact(rows.len()) {
            let primary = ExternalSortOperationError::Allocation(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                format!("reserve ordinal cursor rows: {error}"),
            ));
            return match self.resize_ordinal_grant(previous, observer) {
                Ok(()) => Err(primary),
                Err(accounting) => Err(accounting),
            };
        }
        for (offset, values) in rows.into_iter().enumerate() {
            let offset = u64::try_from(offset).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "sort input ordinal exceeds u64",
                )
            })?;
            stable_rows.push(OrdinalRow {
                values,
                ordinal: first_ordinal + offset,
            });
        }
        self.next_input_ordinal = next_input_ordinal;
        Ok(stable_rows)
    }

    fn read_cursor_payload(
        &mut self,
        reader: &mut SpillFileReader,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<Vec<u8>, ExternalSortOperationError> {
        // Compatibility cursors deliberately retain the legacy, unqualified
        // reader contract. Only a caller that supplied resource grants opts
        // into providers' trusted allocation-bound requirements.
        if self.payload_grant.is_none() {
            return reader
                .read_sort_row()
                .map_err(ExternalSortOperationError::Io);
        }

        let mut admission_failure = None;
        let read = reader.read_sort_row_with_admission(|bytes| {
            match self.resize_payload_grant(bytes, observer) {
                Ok(()) => Ok(()),
                Err(error) => {
                    admission_failure = Some(error);
                    Err(std::io::Error::from(std::io::ErrorKind::OutOfMemory))
                }
            }
        });
        if self.scalar_cleanup.is_some() {
            match (read, admission_failure) {
                (_, Some(primary)) => return Err(primary),
                (Err(error), None) => return Err(ExternalSortOperationError::Io(error)),
                (Ok(payload), None) => return Ok(payload),
            }
        }
        match (read, admission_failure) {
            (_, Some(primary)) => {
                let release = self.release_payload_grant(observer);
                match release {
                    Ok(()) => Err(primary),
                    Err(release) => Err(with_io_cleanup(
                        primary,
                        release.into_io(),
                        "sort cursor payload grant release",
                    )),
                }
            }
            (Ok(payload), None) => Ok(payload),
            (Err(error), None) => {
                let primary = ExternalSortOperationError::Io(error);
                match self.release_payload_grant(observer) {
                    Ok(()) => Err(primary),
                    Err(release) => Err(with_io_cleanup(
                        primary,
                        release.into_io(),
                        "sort cursor payload grant release",
                    )),
                }
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "cursor head decode transfers reader, grant, cancellation, and frontier state atomically"
    )]
    fn read_decode_cursor_head(
        &mut self,
        reader: &mut SpillFileReader,
        row_shape: SortRowShape,
        limits: super::file::SpillFrameLimits,
        frontier_base_bytes: usize,
        frontier_reader_bytes: usize,
        frontier_row_bytes: usize,
        cancellation: Option<&QueryCancellationToken>,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(OrdinalRow, usize), ExternalSortOperationError> {
        let payload = self.read_cursor_payload(reader, observer)?;
        let result = (|| {
            check_cancellation(cancellation)?;
            let retained_bytes = conservative_decoded_row_retained_bytes(payload.len())?;
            let admitted = cursor_frontier_bytes(
                frontier_base_bytes,
                frontier_reader_bytes,
                checked_workspace_sum(frontier_row_bytes, retained_bytes)?,
            )?;
            self.resize_frontier_grant(admitted, observer)?;
            let row = decode_row_payload_with_shape(&payload, row_shape, limits)?;
            check_cancellation(cancellation)?;
            Ok((row, retained_bytes))
        })();
        drop(payload);
        if self.scalar_cleanup.is_some() && result.is_err() {
            return result;
        }
        let release = self.release_payload_grant(observer);
        match (result, release) {
            (Ok(row), Ok(())) => Ok(row),
            (Err(primary), Ok(())) => Err(primary),
            (Ok(_), Err(release)) => Err(release),
            (Err(primary), Err(release)) => Err(with_io_cleanup(
                primary,
                release.into_io(),
                "sort cursor payload grant release",
            )),
        }
    }

    /// Spills an already-sorted buffer as a run to disk.
    ///
    /// The buffer must already be stably sorted according to this sorter's
    /// semantic comparator. The sorter durably assigns encounter ordinals.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to disk fails.
    pub fn spill_sorted_run(&mut self, rows: Vec<Vec<Value>>) -> std::io::Result<()> {
        let observer = ExternalSortGrantObserver::inert();
        self.spill_sorted_run_inner(&rows, None, &observer)
            .map_err(ExternalSortOperationError::into_io)
    }

    #[cfg(test)]
    pub(crate) fn spill_sorted_run_accounted(
        &mut self,
        rows: &[Vec<Value>],
    ) -> Result<(), ExternalSortOperationError> {
        debug_assert!(self.workspace.grant.is_some());
        let cancellation = self.cancellation.clone();
        let observer = ExternalSortGrantObserver::inert();
        self.spill_sorted_run_inner(rows, cancellation.as_ref(), &observer)
    }

    /// Appends one admitted, already-sorted batch using binary carries.
    ///
    /// DISTINCT's ordered passes retain the immediate binary schedule. This
    /// operation preserves every row and stable ordinal; it does not deduplicate.
    ///
    /// Each live run represents a power-of-two number of batches. Equal levels
    /// are adjacent at the tail and merge exactly once into the next level.
    /// Every row therefore participates in at most floor(log2(batch count))
    /// ingestion merges, including for unequal batch sizes. The checked u64
    /// batch count permits at most 64 live runs, plus the existing admitted
    /// retry slot while a replacement is being published. Each carry opens
    /// only two readers, within the configured fan-in.
    pub(crate) fn spill_distinct_run_accounted(
        &mut self,
        rows: &[Vec<Value>],
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        self.spill_scheduled_run_accounted(rows, observer, 1)
    }

    pub(crate) fn set_pull_control_bytes(&mut self, bytes: usize) {
        self.pull_control_bytes = bytes;
    }

    fn pull_retained_bytes(&self) -> Result<usize, MemoryGrantError> {
        let hook = self
            .pull_hook_workspace
            .as_ref()
            .map_or(Ok(0), |authority| {
                checked_workspace_sum(
                    authority.granted_bytes(),
                    authority
                        .inspect::<PullSortHookAuthority, _>(PullSortHookAuthority::granted_bytes)
                        .expect("sort owns its hook authority"),
                )
            })?;
        checked_workspace_sum(hook, self.pull_control_bytes)
    }

    pub(crate) fn retain_pull_hook_workspace(&mut self, grant: AccountedError) {
        debug_assert!(self.pull_hook_workspace.is_none());
        self.writer_workspace.hook_authority = Some(grant.clone());
        self.pull_hook_workspace = Some(grant);
    }

    pub(crate) fn retain_pull_failure_workspaces(&mut self) {
        self.writer_workspace.retain_failure = true;
    }

    pub(crate) fn take_pull_failure_workspaces(&mut self) -> [Option<MemoryGrant>; 3] {
        [
            self.writer_workspace.grant.take(),
            self.frontier_grant.take(),
            self.payload_grant.take(),
        ]
    }

    /// Defers pull SORT's initial carries until one bounded fan-in is full.
    /// Later carries retain their original batch ordinals and binary levels.
    pub(crate) fn spill_pull_run_accounted(
        &mut self,
        rows: &[Vec<Value>],
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        let maximum = self.merge_fan_in.min(64);
        let initial = 1usize << (usize::BITS - 1 - maximum.leading_zeros());
        self.spill_scheduled_run_accounted(rows, observer, initial)
    }

    fn spill_scheduled_run_accounted(
        &mut self,
        rows: &[Vec<Value>],
        observer: &ExternalSortGrantObserver<'_>,
        initial: usize,
    ) -> Result<(), ExternalSortOperationError> {
        check_cancellation(self.cancellation.as_ref())?;
        if rows.is_empty() {
            return Ok(());
        }
        if self
            .initial_run_group
            .is_some_and(|frozen| frozen != initial)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "sort run schedule cannot change mode or initial fan-in",
            )
            .into());
        }
        let batches = match self.distinct_run_schedule {
            DistinctRunSchedule::Unused if self.runs.is_empty() => 0,
            DistinctRunSchedule::Appending(batches) => batches,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "DISTINCT run schedule cannot append after foreign input, sealing or failure",
                )
                .into());
            }
        };
        let next = batches.checked_add(1).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "DISTINCT input batch ordinal overflow",
            )
        })?;
        let cancellation = self.cancellation.clone();
        // A partially published append/merge cannot be replayed. Cleanup still
        // owns every run and retry slot through the existing sorter machinery.
        self.initial_run_group = Some(initial);
        self.distinct_run_schedule = DistinctRunSchedule::Failed;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.spill_sorted_run_accounted_observing(rows, observer)?;
            if initial > 1 && next <= initial as u64 {
                if next == initial as u64 {
                    check_cancellation(cancellation.as_ref())?;
                    self.merge_run_group(0, initial, cancellation.as_ref(), observer)?;
                }
                return Ok(());
            }
            for _ in 0..next.trailing_zeros() {
                check_cancellation(cancellation.as_ref())?;
                let start = self.runs.len().checked_sub(2).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "DISTINCT binary carry lost an input run",
                    )
                })?;
                self.merge_run_group(start, 2, cancellation.as_ref(), observer)?;
            }
            Ok(())
        }));
        match outcome {
            Ok(Ok(())) => {
                self.distinct_run_schedule = DistinctRunSchedule::Appending(next);
                Ok(())
            }
            Ok(Err(error)) => Err(error),
            Err(panic) => self.resume_after_accounted_unwind(observer, panic),
        }
    }

    /// Seals binary-carry input and reduces the bounded level catalog once at EOF.
    /// This must precede exact owned-output qualification. Unlike reduction
    /// after each append, this final pass cannot repeatedly rewrite a growing
    /// prefix: its input catalog has at most 64 runs and never receives input.
    pub(crate) fn finish_distinct_runs_accounted(
        &mut self,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        check_cancellation(self.cancellation.as_ref())?;
        match self.distinct_run_schedule {
            DistinctRunSchedule::Sealed => return Ok(()),
            DistinctRunSchedule::Appending(_) => {}
            DistinctRunSchedule::Unused if self.runs.is_empty() => {}
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "DISTINCT run schedule cannot seal foreign input or failure",
                )
                .into());
            }
        }
        self.distinct_run_schedule = DistinctRunSchedule::Failed;
        let cancellation = self.cancellation.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.reduce_runs_for_final_merge(false, cancellation.as_ref(), observer)
        }));
        match outcome {
            Ok(Ok(())) => {
                self.distinct_run_schedule = DistinctRunSchedule::Sealed;
                Ok(())
            }
            Ok(Err(error)) => Err(error),
            Err(panic) => self.resume_after_accounted_unwind(observer, panic),
        }
    }

    pub(crate) fn spill_sorted_run_accounted_observing(
        &mut self,
        rows: &[Vec<Value>],
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        debug_assert!(self.workspace.grant.is_some());
        let cancellation = self.cancellation.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.spill_sorted_run_inner(rows, cancellation.as_ref(), observer)
        }));
        match outcome {
            Ok(result) => result,
            Err(panic) => self.resume_after_accounted_unwind(observer, panic),
        }
    }

    fn spill_sorted_run_inner(
        &mut self,
        rows: &[Vec<Value>],
        cancellation: Option<&QueryCancellationToken>,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        check_cancellation(cancellation)?;
        if self.disk_merge_started {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "external sort cannot spill after a disk merge has started",
            )
            .into());
        }
        if rows.is_empty() {
            return Ok(());
        }

        let row_count = rows.len();
        let row_count_u64 = u64::try_from(row_count)
            .map_err(|_| std::io::Error::other("sort run row count exceeds u64"))?;
        let first_ordinal = self.next_input_ordinal;
        let next_input_ordinal = first_ordinal.checked_add(row_count_u64).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "sort input ordinal exceeds u64",
            )
        })?;
        let columns = u32::try_from(self.num_columns).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "sort run column count exceeds u32",
            )
        })?;
        for row in rows {
            check_cancellation(cancellation)?;
            self.row_shape.edge_mask(row).map_err(|message| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
            })?;
        }
        let limits = self.manager.frame_limits();
        let mut required_staging = 0usize;
        let mut required_counter_entries = 0usize;
        let mut required_counter_key_bytes = 0usize;
        for row in rows {
            check_cancellation(cancellation)?;
            let measurement = measure_serialized_row_with_limits(row, limits.codec_limits())?;
            let record_bytes = measurement
                .encoded_bytes
                .checked_add(std::mem::size_of::<u64>())
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "sort row plus ordinal length exceeds the platform address space",
                    )
                })?;
            required_staging = required_staging.max(record_bytes);
            required_counter_entries =
                required_counter_entries.max(measurement.counter_sort_entries);
            required_counter_key_bytes =
                required_counter_key_bytes.max(measurement.counter_sort_key_bytes);
            check_cancellation(cancellation)?;
        }
        let preparation = self.workspace.prepare(
            required_staging,
            required_counter_entries,
            required_counter_key_bytes,
        );
        let observation = self.publish_granted_bytes(observer);
        preparation?;
        observation?;
        check_cancellation(cancellation)?;
        debug_assert!(self.workspace.row_staging.capacity() >= required_staging);
        let observed_counter = self.workspace.counter_scratch.observed_capacity();
        debug_assert!(observed_counter.entry_capacity >= required_counter_entries);
        debug_assert!(observed_counter.key_capacity >= required_counter_key_bytes);

        let catalog_preparation = self.runs.prepare_push();
        let observation = self.publish_granted_bytes(observer);
        catalog_preparation?;
        observation?;
        check_cancellation(cancellation)?;

        let write_rows =
            |workspace: &mut ExternalSortWorkspace,
             write_payload: &mut dyn FnMut(&[u8]) -> Result<(), ExternalSortPrimary>|
             -> Result<(), ExternalSortPrimary> {
                for (row_index, row) in rows.iter().enumerate() {
                    check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;
                    let row_offset = u64::try_from(row_index).map_err(|_| {
                        ExternalSortPrimary::Io(std::io::Error::new(
                            std::io::ErrorKind::OutOfMemory,
                            "sort input ordinal exceeds u64",
                        ))
                    })?;
                    let ordinal = first_ordinal.checked_add(row_offset).ok_or_else(|| {
                        ExternalSortPrimary::Io(std::io::Error::new(
                            std::io::ErrorKind::OutOfMemory,
                            "sort input ordinal exceeds u64",
                        ))
                    })?;
                    let mut payload = workspace
                        .row_staging
                        .record_scope(required_staging)
                        .map_err(ExternalSortPrimary::Io)?;
                    serialize_row_with_prepared_scratch(
                        row,
                        &mut payload,
                        limits.codec_limits(),
                        &mut workspace.counter_scratch,
                    )
                    .map_err(ExternalSortPrimary::Io)?;
                    payload
                        .write_all(&ordinal.to_le_bytes())
                        .map_err(ExternalSortPrimary::Io)?;
                    write_payload(payload.as_slice())?;
                    check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;
                }
                Ok(())
            };

        let spill_file = if self.writer_workspace.grant.is_some() {
            let stable_bytes = checked_workspace_sum(
                checked_workspace_sum(
                    self.workspace_granted_bytes(),
                    self.run_catalog_granted_bytes(),
                )?,
                self.pull_retained_bytes()?,
            )?;
            let manager = &self.manager;
            let workspace = &mut self.workspace;
            let writer_workspace = &mut self.writer_workspace;
            let result = {
                let mut publication = WriterPublication::new(writer_workspace);
                let attempt = (|| {
                    publication.prepare(manager)?;
                    observer.publish_sum(stable_bytes, publication.granted_bytes())?;
                    if let Err(error) = check_cancellation(cancellation) {
                        return Err(publication.abort_primary(
                            ExternalSortPrimary::Cancelled(error),
                            "writer cancellation release",
                        ));
                    }
                    if let Err(error) =
                        publication.create(manager, SpillFileRole::SortRun, stable_bytes, observer)
                    {
                        return Err(publication.abort_primary(error, "writer creation release"));
                    }
                    if let Err(error) = check_cancellation(cancellation) {
                        return Err(publication.abort_primary(
                            ExternalSortPrimary::Cancelled(error),
                            "writer cancellation release",
                        ));
                    }
                    let write_result = (|| -> Result<(), ExternalSortPrimary> {
                        publication
                            .file_mut()
                            .write_sort_run_start(columns, row_count_u64)
                            .map_err(ExternalSortPrimary::Io)?;
                        check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;
                        debug_assert_eq!(publication.file_mut().limits(), limits);
                        write_rows(workspace, &mut |payload| {
                            publication.write_sort_row_accounted(payload, stable_bytes, observer)
                        })
                    })();
                    if let Err(error) = write_result {
                        let phase = match error {
                            ExternalSortPrimary::Cancelled(_) => "writer cancellation release",
                            ExternalSortPrimary::Memory(_)
                            | ExternalSortPrimary::Allocation(_)
                            | ExternalSortPrimary::Io(_) => "writer failure release",
                        };
                        return Err(publication.abort_primary(error, phase));
                    }
                    publication.finish_and_release(cancellation)
                })();
                match attempt {
                    Ok(()) => Ok(publication.take_file_after_release()),
                    Err(error) => Err(error),
                }
            };
            match result {
                Ok(file) => {
                    self.publish_granted_bytes(observer)?;
                    file
                }
                Err(error) => {
                    // The physical/provider primary retains precedence over
                    // an impossible in-crate observer mismatch. Publication
                    // has already made both scalar sinks fail closed.
                    self.publish_granted_bytes_preserving_primary(observer);
                    return Err(error);
                }
            }
        } else {
            let mut spill_file = self.manager.create_file(SpillFileRole::SortRun)?;
            let spill_result = (|| -> Result<(), ExternalSortPrimary> {
                spill_file
                    .write_sort_run_start(columns, row_count_u64)
                    .map_err(ExternalSortPrimary::Io)?;
                check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;
                debug_assert_eq!(spill_file.limits(), limits);
                write_rows(&mut self.workspace, &mut |payload| {
                    spill_file
                        .write_sort_row(payload)
                        .map_err(ExternalSortPrimary::Io)
                })
            })()
            .map_err(ExternalSortPrimary::into_io)
            .and_then(|()| spill_file.finish_write());
            if let Err(spill_error) = spill_result {
                return match spill_file.close_and_delete() {
                    Ok(()) => Err(spill_error.into()),
                    Err(cleanup_error) => Err(super::combine_primary_and_cleanup(
                        spill_error,
                        cleanup_error,
                        "sort spill cleanup",
                    )
                    .into()),
                };
            }
            spill_file
        };

        self.runs.push(spill_file, row_count);
        self.next_input_ordinal = next_input_ordinal;

        Ok(())
    }

    /// Starts a consuming merge that yields at most `max_chunk_rows` at once.
    ///
    /// The returned chunks borrow the cursor, so only one sorter-owned output
    /// batch can be live at a time. Compatibility callers that need one
    /// complete `Vec` can continue to use [`Self::merge_all`].
    ///
    /// # Errors
    ///
    /// Returns an error for a zero batch bound, resource exhaustion, malformed
    /// spill data, cancellation on a resource-qualified sorter, or cleanup
    /// failure. Cancellation is preserved as `Interrupted` and deadline expiry
    /// as `TimedOut`, with the typed reason available as the error source.
    pub fn merge_cursor(
        &mut self,
        in_memory_buffer: Vec<Vec<Value>>,
        max_chunk_rows: usize,
    ) -> std::io::Result<ExternalSortCursor<'_>> {
        let cancellation = self.cancellation.clone();
        let observer = ExternalSortGrantObserver::inert();
        let state = self
            .merge_cursor_inner(
                in_memory_buffer,
                max_chunk_rows,
                cancellation.as_ref(),
                &observer,
            )
            .map_err(ExternalSortOperationError::into_io)?;
        Ok(ExternalSortCursor::from_state(self, state))
    }

    #[cfg(test)]
    pub(crate) fn inspect_scalar_failure<R>(
        error: &OperatorError,
        inspect: impl FnOnce(Option<&OperatorError>, Option<&ExternalSortOperationError>) -> R,
    ) -> Option<R> {
        match error {
            OperatorError::ClassifiedAccountedFailure { authority, .. } => authority
                .inspect::<ScalarSortFailure, _>(
                |failure| inspect(failure.operator.as_ref(), failure.cleanup.as_ref()),
            ),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn inspect_scalar_panic<R>(
        error: &OperatorError,
        inspect: impl FnOnce(Option<&(dyn std::any::Any + Send)>, usize) -> R,
    ) -> Option<R> {
        match error {
            OperatorError::ClassifiedAccountedFailure { authority, .. } => {
                authority.inspect::<ScalarSortFailure, _>(|failure| {
                    let retained = failure.cleanup_authority.granted_bytes().saturating_add(
                        failure
                            .cleanup_authority
                            .inspect::<ScalarReaderCleanup, _>(ScalarReaderCleanup::granted_bytes)
                            .unwrap_or(usize::MAX),
                    );
                    inspect(failure.panic.as_deref(), retained)
                })
            }
            _ => None,
        }
    }

    fn prepare_scalar_failure_transport(
        &mut self,
    ) -> Result<ScalarFailureTransport, ExternalSortOperationError> {
        let child = self
            .frontier_grant
            .as_mut()
            .ok_or_else(|| ExternalSortOperationError::Io(std::io::ErrorKind::InvalidInput.into()))?
            .split(0)
            .ok_or_else(|| {
                ExternalSortOperationError::Io(std::io::ErrorKind::InvalidInput.into())
            })?;
        let cleanup = AccountedErrorPublisher::<ScalarReaderCleanup>::try_new(child)
            .map_err(|error| self.recover_scalar_publisher_failure(error))?;
        let child = self
            .frontier_grant
            .as_mut()
            .ok_or_else(|| ExternalSortOperationError::Io(std::io::ErrorKind::InvalidInput.into()))?
            .split(0)
            .ok_or_else(|| {
                ExternalSortOperationError::Io(std::io::ErrorKind::InvalidInput.into())
            })?;
        let publisher = match AccountedErrorPublisher::<ScalarSortFailure>::try_new(child) {
            Ok(publisher) => publisher,
            Err(error) => {
                self.merge_exact_frontier_grant(
                    cleanup.into_unpublished_grant(),
                    "unused scalar cleanup publisher",
                );
                return Err(self.recover_scalar_publisher_failure(error));
            }
        };
        let cleanup = cleanup.publish(ScalarReaderCleanup::new());
        self.writer_workspace.retain_failure = true;
        self.writer_workspace.scalar_cleanup = Some(cleanup.clone());
        self.scalar_cleanup = Some(cleanup.clone());
        self.scalar_failure_publication_bytes = publisher.granted_bytes();
        Ok(ScalarFailureTransport { publisher, cleanup })
    }

    fn recover_scalar_publisher_failure(
        &mut self,
        error: AccountedErrorPublisherBuildError,
    ) -> ExternalSortOperationError {
        let primary = match error.failure() {
            AccountedErrorPublisherBuildFailure::Admission(error)
            | AccountedErrorPublisherBuildFailure::AllocationWithRollback(error) => {
                ExternalSortOperationError::Memory(error.clone())
            }
            AccountedErrorPublisherBuildFailure::Allocation => {
                ExternalSortOperationError::Allocation(std::io::ErrorKind::OutOfMemory.into())
            }
            AccountedErrorPublisherBuildFailure::NonZeroGrant { .. } => {
                ExternalSortOperationError::Io(std::io::ErrorKind::InvalidInput.into())
            }
            _ => ExternalSortOperationError::Io(std::io::ErrorKind::Other.into()),
        };
        self.merge_exact_frontier_grant(error.into_grant(), "scalar publisher refusal");
        primary
    }

    fn scalar_failure(
        &mut self,
        transport: ScalarFailureTransport,
        operation: Option<ExternalSortOperationError>,
        operator: Option<OperatorError>,
        panic: Option<Box<dyn std::any::Any + Send>>,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> OperatorError {
        let classification = operation
            .as_ref()
            .map(classify_external_sort_error)
            .or_else(|| operator.as_ref().map(classify_operator_error))
            .unwrap_or(AccountedFailureClassification::Execution);
        let mut cleanup = None;
        let mut cleanup_panic = None;
        if let Some(file) = self.writer_workspace.scalar_retry_file.take() {
            // The catalog slot was admitted before intermediate publication.
            self.runs.push(file, 0);
        }
        let mut index = 0;
        while index < self.runs.len() {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.runs.entries[index].file.close_and_delete()
            }));
            match outcome {
                Ok(Ok(())) => {
                    self.runs.remove(index);
                }
                Ok(Err(error)) => {
                    if cleanup.is_none() {
                        cleanup = Some(ExternalSortOperationError::Io(error));
                    } else if !super::run_cleanup_backstop(|| {
                        drop(error);
                        Ok::<(), std::convert::Infallible>(())
                    }) {
                        mark_scalar_cleanup_failed(self.scalar_cleanup.as_ref());
                    }
                    index += 1;
                }
                Err(payload) => {
                    if cleanup_panic.is_none() {
                        cleanup_panic = Some(payload);
                    } else if !super::run_cleanup_backstop(|| {
                        drop(payload);
                        Ok::<(), std::convert::Infallible>(())
                    }) {
                        mark_scalar_cleanup_failed(self.scalar_cleanup.as_ref());
                    }
                    index += 1;
                }
            }
        }
        let mut workspaces = Some(self.take_pull_failure_workspaces());
        transport
            .cleanup
            .inspect::<ScalarReaderCleanup, _>(|cleanup| {
                if let Some(grants) = workspaces.take() {
                    cleanup.retain_workspaces(grants);
                }
            });
        if let Some(grants) = workspaces {
            for grant in grants.into_iter().flatten() {
                std::mem::forget(grant);
            }
        }
        self.scalar_failure_publication_bytes = 0;
        let ScalarFailureTransport {
            publisher,
            cleanup: cleanup_authority,
        } = transport;
        let authority = publisher.publish(ScalarSortFailure {
            operation,
            operator,
            panic,
            cleanup,
            cleanup_panic,
            cleanup_authority,
        });
        self.publish_granted_bytes_preserving_primary(observer);
        OperatorError::ClassifiedAccountedFailure {
            classification,
            authority,
        }
    }

    /// Production scalar fallback: the shared merge algorithm retains its
    /// existing Vec provider/platform contract and gains terminal ownership.
    pub(crate) fn merge_scalar_cursor_accounted_observing<'sort>(
        &'sort mut self,
        in_memory_buffer: Vec<Vec<Value>>,
        max_chunk_rows: usize,
        observer: ExternalSortGrantObserver<'sort>,
        map_error: fn(ExternalSortOperationError) -> OperatorError,
    ) -> Result<ScalarExternalSortCursor<'sort>, OperatorError> {
        let transport = self.prepare_scalar_failure_transport().map_err(map_error)?;
        let cancellation = self.cancellation.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.merge_cursor_inner(
                in_memory_buffer,
                max_chunk_rows,
                cancellation.as_ref(),
                &observer,
            )
        }));
        let state = match outcome {
            Ok(Ok(state)) => state,
            Ok(Err(error)) => {
                return Err(self.scalar_failure(transport, Some(error), None, None, &observer));
            }
            Err(payload) => {
                return Err(self.scalar_failure(transport, None, None, Some(payload), &observer));
            }
        };
        Ok(ScalarExternalSortCursor {
            cursor: ExternalSortCursor::from_state(self, state),
            observer,
            transport: Some(transport),
        })
    }

    #[cfg(test)]
    pub(crate) fn merge_cursor_accounted_observing<'sort>(
        &'sort mut self,
        in_memory_buffer: Vec<Vec<Value>>,
        max_chunk_rows: usize,
        observer: ExternalSortGrantObserver<'sort>,
    ) -> Result<AccountedExternalSortCursor<'sort>, ExternalSortOperationError> {
        debug_assert!(self.workspace.grant.is_some());
        let cancellation = self.cancellation.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.merge_cursor_inner(
                in_memory_buffer,
                max_chunk_rows,
                cancellation.as_ref(),
                &observer,
            )
        }));
        let state = match outcome {
            Ok(result) => result?,
            Err(panic) => self.resume_after_accounted_unwind(&observer, panic),
        };
        Ok(AccountedExternalSortCursor {
            cursor: ExternalSortCursor::from_state(self, state),
            observer,
        })
    }

    fn merge_cursor_inner(
        &mut self,
        in_memory_buffer: Vec<Vec<Value>>,
        max_chunk_rows: usize,
        cancellation: Option<&QueryCancellationToken>,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<ExternalSortCursorState, ExternalSortOperationError> {
        if max_chunk_rows == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "external sort cursor chunk size must be greater than zero",
            )
            .into());
        }
        if self.disk_merge_started {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "external sort disk merge is consuming and cannot be retried",
            )
            .into());
        }

        let num_runs = self.runs.len();
        let has_memory = !in_memory_buffer.is_empty();
        if num_runs > 0 {
            // Install the consuming fence before the first cancellation poll:
            // a pre-cancelled qualified attempt still owns cleanup of all runs.
            self.disk_merge_started = true;
        }

        let workspace_release = self.release_memory_workspaces();
        let observation = self.publish_granted_bytes(observer);
        if let Err(primary) = workspace_release {
            if self.scalar_cleanup.is_some() {
                return Err(primary);
            }
            let cleanup = self.cleanup_runs();
            self.publish_granted_bytes_preserving_primary(observer);
            return match cleanup {
                Ok(()) => Err(primary),
                Err(cleanup) => Err(with_io_cleanup(
                    primary,
                    cleanup.into_io(),
                    "sort spill cleanup",
                )),
            };
        }
        observation?;
        if let Err(error) = check_cancellation(cancellation) {
            let primary = ExternalSortOperationError::Cancelled(error);
            if self.scalar_cleanup.is_some() {
                return Err(primary);
            }
            let cleanup = self.cleanup_runs();
            self.publish_granted_bytes_preserving_primary(observer);
            return match cleanup {
                Ok(()) => Err(primary),
                Err(cleanup) => Err(with_io_cleanup(
                    primary,
                    cleanup.into_io(),
                    "sort spill cleanup",
                )),
            };
        }

        let initialization = (|| {
            let requested_output_bytes = cursor_capacity_bytes::<Vec<Value>>(max_chunk_rows)?;
            self.resize_output_grant(requested_output_bytes, observer)?;
            let mut output_rows = Vec::new();
            output_rows
                .try_reserve_exact(max_chunk_rows)
                .map_err(|error| {
                    ExternalSortOperationError::Allocation(std::io::Error::new(
                        std::io::ErrorKind::OutOfMemory,
                        format!("reserve external sort cursor output: {error}"),
                    ))
                })?;
            let output_base_bytes = cursor_capacity_bytes::<Vec<Value>>(output_rows.capacity())?;
            self.resize_output_grant(output_base_bytes, observer)?;

            self.checked_total_rows_with_cancellation(cancellation)?
                .checked_add(in_memory_buffer.len())
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::OutOfMemory,
                        "merged sort row count exceeds the platform address space",
                    )
                })?;
            self.reduce_runs_for_final_merge(has_memory, cancellation, observer)?;

            let mut memory_rows =
                self.attach_input_ordinals_for_cursor(in_memory_buffer, observer)?;
            if !memory_rows.is_empty() {
                check_cancellation(cancellation)?;
                if let Comparison::Infallible(compare) = &self.comparator.compare {
                    memory_rows.sort_unstable_by(|left, right| {
                        compare_ordinal_rows(left, right, compare.as_ref())
                    });
                } else {
                    sort_ordinal_rows(&mut memory_rows, &self.comparator)?;
                }
                check_cancellation(cancellation)?;
            }

            let memory_slot = usize::from(!memory_rows.is_empty());
            let frontier_capacity = self.runs.len().checked_add(memory_slot).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "sort cursor frontier exceeds the platform address space",
                )
            })?;
            let requested_frontier_bytes = checked_workspace_sum(
                cursor_capacity_bytes::<HeapEntry>(frontier_capacity)?,
                cursor_capacity_bytes::<Option<CursorRunReader>>(self.runs.len())?,
            )?;
            self.resize_frontier_grant(requested_frontier_bytes, observer)?;

            let mut heap = ComparatorMinHeap::try_with_exact_capacity(frontier_capacity)
                .map_err(ComparatorMinHeapAllocationError::into_operation)?;
            let mut run_readers = Vec::new();
            run_readers
                .try_reserve_exact(self.runs.len())
                .map_err(|error| {
                    std::io::Error::new(
                        std::io::ErrorKind::OutOfMemory,
                        format!("reserve sort cursor readers: {error}"),
                    )
                })?;
            let frontier_base_bytes = checked_workspace_sum(
                cursor_capacity_bytes::<HeapEntry>(heap.capacity())?,
                cursor_capacity_bytes::<Option<CursorRunReader>>(run_readers.capacity())?,
            )?;
            self.resize_frontier_grant(frontier_base_bytes, observer)?;

            let mut frontier_reader_bytes = 0usize;
            let mut frontier_row_bytes = 0usize;
            for run_index in 0..self.runs.len() {
                check_cancellation(cancellation)?;
                let limits = self.runs.get(run_index).file.limits();
                let total_without_frontier = [
                    self.scalar_failure_publication_bytes,
                    self.scalar_cleanup
                        .as_ref()
                        .map_or(0, AccountedError::granted_bytes),
                    self.workspace_granted_bytes(),
                    self.run_catalog_granted_bytes(),
                    self.writer_workspace_granted_bytes(),
                    self.ordinal_granted_bytes(),
                    self.payload_granted_bytes(),
                    self.output_granted_bytes(),
                ]
                .into_iter()
                .try_fold(0usize, checked_workspace_sum)?;
                let (mut reader, reader_workspace_bytes) = open_cursor_reader(
                    &self.runs.entries[run_index].file,
                    &mut self.frontier_grant,
                    total_without_frontier,
                    frontier_base_bytes,
                    frontier_reader_bytes,
                    frontier_row_bytes,
                    observer,
                    self.scalar_cleanup.is_some(),
                    self.scalar_cleanup.as_ref(),
                )?;
                frontier_reader_bytes = frontier_reader_bytes
                    .checked_add(reader_workspace_bytes)
                    .ok_or(MemoryGrantError::ArithmeticOverflow {
                    current_bytes: frontier_reader_bytes,
                    additional_bytes: reader_workspace_bytes,
                })?;
                let (columns, row_count) = reader.read_sort_run_start()?;
                self.validate_columns(columns)?;
                self.validate_row_count(run_index, row_count)?;
                check_cancellation(cancellation)?;
                if row_count == 0 {
                    reader.finish()?;
                    drop(reader);
                    scalar_reader_cleanup_check(self.scalar_cleanup.as_ref())?;
                    frontier_reader_bytes = frontier_reader_bytes
                        .checked_sub(reader_workspace_bytes)
                        .expect("newly opened reader owns its workspace charge");
                    let frontier_bytes = cursor_frontier_bytes(
                        frontier_base_bytes,
                        frontier_reader_bytes,
                        frontier_row_bytes,
                    )?;
                    self.resize_frontier_grant(frontier_bytes, observer)?;
                    run_readers.push(None);
                } else {
                    let (row, retained_bytes) = self.read_decode_cursor_head(
                        &mut reader,
                        self.row_shape,
                        limits,
                        frontier_base_bytes,
                        frontier_reader_bytes,
                        frontier_row_bytes,
                        cancellation,
                        observer,
                    )?;
                    check_cancellation(cancellation)?;
                    run_readers.push(Some(CursorRunReader {
                        reader,
                        remaining: row_count - 1,
                        row_shape: self.row_shape,
                        limits,
                        reader_workspace_bytes,
                    }));
                    heap.push_entry(
                        HeapEntry {
                            row,
                            run_index,
                            retained_bytes,
                        },
                        &self.comparator,
                    )?;
                    frontier_row_bytes = frontier_row_bytes.checked_add(retained_bytes).ok_or(
                        MemoryGrantError::ArithmeticOverflow {
                            current_bytes: frontier_row_bytes,
                            additional_bytes: retained_bytes,
                        },
                    )?;
                }
                check_cancellation(cancellation)?;
            }

            let memory_run_index = self.runs.len();
            let mut memory_iter = memory_rows.into_iter();
            if let Some(row) = memory_iter.next() {
                heap.push_entry(
                    HeapEntry {
                        row,
                        run_index: memory_run_index,
                        retained_bytes: 0,
                    },
                    &self.comparator,
                )?;
            }
            check_cancellation(cancellation)?;
            Ok(ExternalSortCursorState {
                output_rows,
                heap,
                run_readers,
                memory_iter,
                memory_run_index,
                max_chunk_rows,
                frontier_base_bytes,
                frontier_reader_bytes,
                frontier_row_bytes,
                output_base_bytes,
                output_row_bytes: 0,
            })
        })();

        match initialization {
            Ok(state) => Ok(state),
            Err(mut primary) => {
                if self.scalar_cleanup.is_some() {
                    return Err(primary);
                }
                if let Err(release) = self.release_cursor_grants(true, observer) {
                    primary =
                        with_io_cleanup(primary, release.into_io(), "sort cursor grant release");
                }
                let cleanup = self.cleanup_runs();
                self.publish_granted_bytes_preserving_primary(observer);
                match cleanup {
                    Ok(()) => Err(primary),
                    Err(cleanup) => Err(with_io_cleanup(
                        primary,
                        cleanup.into_io(),
                        "spill-read cleanup",
                    )),
                }
            }
        }
    }

    /// Merges all runs and an optional in-memory buffer into sorted output.
    ///
    /// Uses k-way merge with a min-heap.
    ///
    /// # Errors
    ///
    /// Returns an error if reading from disk fails.
    pub fn merge_all(
        &mut self,
        in_memory_buffer: Vec<Vec<Value>>,
    ) -> std::io::Result<Vec<Vec<Value>>> {
        let observer = ExternalSortGrantObserver::inert();
        self.merge_all_inner(in_memory_buffer, None, &observer)
            .map_err(ExternalSortOperationError::into_io)
    }

    #[cfg(test)]
    pub(crate) fn merge_all_accounted(
        &mut self,
        in_memory_buffer: Vec<Vec<Value>>,
    ) -> Result<Vec<Vec<Value>>, ExternalSortOperationError> {
        debug_assert!(self.workspace.grant.is_some());
        let cancellation = self.cancellation.clone();
        let observer = ExternalSortGrantObserver::inert();
        self.merge_all_inner(in_memory_buffer, cancellation.as_ref(), &observer)
    }

    #[cfg(test)]
    pub(crate) fn merge_all_accounted_observing(
        &mut self,
        in_memory_buffer: Vec<Vec<Value>>,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<Vec<Vec<Value>>, ExternalSortOperationError> {
        debug_assert!(self.workspace.grant.is_some());
        let cancellation = self.cancellation.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.merge_all_inner(in_memory_buffer, cancellation.as_ref(), observer)
        }));
        match outcome {
            Ok(result) => result,
            Err(panic) => self.resume_after_accounted_unwind(observer, panic),
        }
    }

    fn merge_all_inner(
        &mut self,
        in_memory_buffer: Vec<Vec<Value>>,
        cancellation: Option<&QueryCancellationToken>,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<Vec<Vec<Value>>, ExternalSortOperationError> {
        if self.disk_merge_started {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "external sort disk merge is consuming and cannot be retried",
            )
            .into());
        }
        let num_runs = self.runs.len();
        let has_memory = !in_memory_buffer.is_empty();
        if num_runs > 0 {
            // A merge attempt that must destroy runs after an accounting
            // failure or cancellation is consuming just like a partially
            // completed merge. Install this fence before the first poll so a
            // pre-cancelled qualified merge still owns and cleans every run.
            self.disk_merge_started = true;
        }
        let workspace_release = self.release_memory_workspaces();
        let observation = self.publish_granted_bytes(observer);
        if let Err(error) = workspace_release {
            let cleanup_result = self.cleanup_runs();
            self.publish_granted_bytes_preserving_primary(observer);
            return match cleanup_result {
                Ok(()) => Err(error),
                Err(cleanup) => Err(with_io_cleanup(
                    error,
                    cleanup.into_io(),
                    "sort spill cleanup",
                )),
            };
        }
        observation?;
        if let Err(error) = check_cancellation(cancellation) {
            let primary = ExternalSortOperationError::Cancelled(error);
            let cleanup_result = self.cleanup_runs();
            self.publish_granted_bytes_preserving_primary(observer);
            return match cleanup_result {
                Ok(()) => Err(primary),
                Err(cleanup) => Err(with_io_cleanup(
                    primary,
                    cleanup.into_io(),
                    "sort spill cleanup",
                )),
            };
        }

        // Special case: no runs and no memory buffer
        if num_runs == 0 && !has_memory {
            self.cleanup_runs()?;
            self.publish_granted_bytes(observer)?;
            check_cancellation(cancellation)?;
            return Ok(Vec::new());
        }

        // Special case: only in-memory buffer, no disk runs
        if num_runs == 0 {
            self.cleanup_runs()?;
            self.publish_granted_bytes(observer)?;
            check_cancellation(cancellation)?;
            let mut sorted_buffer = self.attach_input_ordinals(in_memory_buffer)?;
            sort_ordinal_rows(&mut sorted_buffer, &self.comparator)?;
            check_cancellation(cancellation)?;
            return Ok(sorted_buffer.into_iter().map(|row| row.values).collect());
        }

        let result = (|| {
            self.checked_total_rows_with_cancellation(cancellation)?
                .checked_add(in_memory_buffer.len())
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::OutOfMemory,
                        "merged sort row count exceeds the platform address space",
                    )
                })?;
            self.reduce_runs_for_final_merge(has_memory, cancellation, observer)?;
            if self.runs.len() == 1 && !has_memory {
                self.read_single_run(0, cancellation)
            } else {
                self.k_way_merge(in_memory_buffer, cancellation)
            }
        })();

        match result {
            Ok(rows) => {
                self.cleanup_runs()?;
                self.publish_granted_bytes(observer)?;
                check_cancellation(cancellation)?;
                Ok(rows)
            }
            Err(read_error) => {
                let cleanup_result = self.cleanup_runs();
                self.publish_granted_bytes_preserving_primary(observer);
                match cleanup_result {
                    Ok(()) => Err(read_error),
                    Err(cleanup_error) => Err(with_io_cleanup(
                        read_error,
                        cleanup_error.into_io(),
                        "spill-read cleanup",
                    )),
                }
            }
        }
    }

    /// Reduces consecutive input segments until the final merge fits in one
    /// bounded reader/head set. Consecutive grouping is load-bearing: a merged
    /// run occupies the same ordinal interval as its inputs, so equal-key rows
    /// remain stable across every pass without serializing a new tie-breaker.
    fn reduce_runs_for_final_merge(
        &mut self,
        has_memory: bool,
        cancellation: Option<&QueryCancellationToken>,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        debug_assert!(self.merge_fan_in >= 2);
        let final_disk_limit = self
            .merge_fan_in
            .checked_sub(usize::from(has_memory))
            .expect("merge fan-in is at least two");

        while self.runs.len() > final_disk_limit {
            check_cancellation(cancellation)?;
            let pass_inputs = self.runs.len();
            let minimum_pass_outputs = pass_inputs.div_ceil(self.merge_fan_in);
            let pass_outputs = minimum_pass_outputs.max(final_disk_limit);
            let mut remaining_reduction = pass_inputs - pass_outputs;
            let merged_groups = remaining_reduction.div_ceil(self.merge_fan_in - 1);
            let mut input_cursor = 0usize;
            for _ in 0..merged_groups {
                check_cancellation(cancellation)?;
                let group_reduction = remaining_reduction.min(self.merge_fan_in - 1);
                let group_len = group_reduction + 1;
                self.merge_run_group(input_cursor, group_len, cancellation, observer)?;
                input_cursor = input_cursor.checked_add(1).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::OutOfMemory,
                        "sort merge pass cursor exceeds the platform address space",
                    )
                })?;
                remaining_reduction -= group_reduction;
            }
            debug_assert_eq!(remaining_reduction, 0);
            debug_assert_eq!(self.runs.len(), pass_outputs);
            check_cancellation(cancellation)?;
        }
        Ok(())
    }

    /// Streams one consecutive group into a replacement run, then atomically
    /// transfers catalog ownership after every input reader has closed and
    /// every replaced file has been explicitly deleted.
    fn merge_run_group(
        &mut self,
        start: usize,
        run_count: usize,
        cancellation: Option<&QueryCancellationToken>,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        let _profile_merge = ProfileMergeTimer::new(&self.manager);
        debug_assert!(run_count >= 2);
        debug_assert!(run_count <= self.merge_fan_in);
        let end = start.checked_add(run_count).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "sort merge group exceeds the platform address space",
            )
        })?;
        if end > self.runs.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "sort merge group exceeds the run catalog",
            )
            .into());
        }

        let mut row_count = 0usize;
        for entry in &self.runs.entries[start..end] {
            check_cancellation(cancellation)?;
            row_count = row_count.checked_add(entry.rows).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "intermediate sort row count exceeds the platform address space",
                )
            })?;
            check_cancellation(cancellation)?;
        }

        // Reserve one catalog slot before publishing a replacement. The slot
        // is normally unused because a successful group replaces inputs in
        // place. It is the retry capability for the exceptional case where a
        // published intermediate output cannot be deleted after a later
        // cancellation or input-cleanup failure.
        let catalog_preparation = self.runs.prepare_push();
        let observation = self.publish_granted_bytes(observer);
        catalog_preparation?;
        observation?;
        check_cancellation(cancellation)?;

        // This out-slot exists before any publication callback. A deletion
        // failure can therefore transfer the only retry handle without a
        // post-failure allocation or an oversized error value.
        let mut retry_file = None;
        let output_result: Result<SpillFile, ExternalSortOperationError> = {
            let entries = &self.runs.entries[start..end];
            let manager = &self.manager;
            let comparator = &self.comparator;
            let row_shape = self.row_shape;
            if self.writer_workspace.grant.is_some() {
                let stable_bytes = [
                    self.scalar_failure_publication_bytes,
                    self.scalar_cleanup
                        .as_ref()
                        .map_or(0, AccountedError::granted_bytes),
                    self.pull_retained_bytes()?,
                    self.workspace_granted_bytes(),
                    self.run_catalog_granted_bytes(),
                    self.ordinal_granted_bytes(),
                    self.output_granted_bytes(),
                ]
                .into_iter()
                .try_fold(0usize, checked_workspace_sum)?;
                let writer_workspace = &mut self.writer_workspace;
                let frontier_grant = &mut self.frontier_grant;
                let payload_grant = &mut self.payload_grant;
                {
                    let mut publication = WriterPublication::new(writer_workspace);
                    let mut merge_workspace = IntermediateMergeWorkspace {
                        retain_failure: publication.workspace.retain_failure,
                        scalar_cleanup: publication.workspace.scalar_cleanup.clone(),
                        failed: false,
                        frontier_grant,
                        payload_grant,
                        stable_granted_bytes: stable_bytes,
                        comparator,
                    };
                    let attempt = (|| -> Result<(), ExternalSortOperationError> {
                        publication.prepare(manager)?;
                        observer.publish(
                            merge_workspace.observed_total(publication.granted_bytes())?,
                        )?;
                        if let Err(error) = check_cancellation(cancellation) {
                            return Err(publication.abort_primary_retaining_file(
                                ExternalSortPrimary::Cancelled(error),
                                "intermediate writer cancellation release",
                                &mut retry_file,
                            ));
                        }
                        if let Err(error) = publication.create(
                            manager,
                            SpillFileRole::SortRun,
                            checked_workspace_sum(stable_bytes, merge_workspace.granted_bytes()?)?,
                            observer,
                        ) {
                            return Err(publication.abort_primary_retaining_file(
                                error,
                                "intermediate writer creation release",
                                &mut retry_file,
                            ));
                        }
                        if let Err(error) = check_cancellation(cancellation) {
                            return Err(publication.abort_primary_retaining_file(
                                ExternalSortPrimary::Cancelled(error),
                                "intermediate writer cancellation release",
                                &mut retry_file,
                            ));
                        }
                        if let Err(error) = merge_run_slice_to_file_accounted(
                            entries,
                            start,
                            row_count,
                            row_shape,
                            comparator,
                            &mut publication,
                            &mut merge_workspace,
                            cancellation,
                            observer,
                        ) {
                            return Err(publication.abort_operation_retaining_file(
                                error,
                                "intermediate writer failure release",
                                &mut retry_file,
                            ));
                        }
                        publication.finish_and_release_retaining_file(cancellation, &mut retry_file)
                    })();
                    match attempt {
                        Ok(()) => Ok(publication.take_file_after_release()),
                        Err(error) => Err(error),
                    }
                }
            } else {
                debug_assert!(cancellation.is_none());
                (|| -> Result<SpillFile, ExternalSortOperationError> {
                    let mut file = manager.create_file(SpillFileRole::SortRun)?;
                    let merge_result = merge_run_slice_to_file(
                        entries, start, row_count, row_shape, comparator, &mut file, None,
                    )
                    .map_err(ExternalSortPrimary::into_io)
                    .and_then(|()| file.finish_write());
                    if let Err(primary) = merge_result {
                        return match file.close_and_delete() {
                            Ok(()) => Err(primary.into()),
                            Err(cleanup) => {
                                retry_file = Some(file);
                                Err(super::combine_primary_and_cleanup(
                                    primary,
                                    cleanup,
                                    "intermediate sort spill cleanup",
                                )
                                .into())
                            }
                        };
                    }
                    Ok(file)
                })()
            }
        };

        let mut output = match output_result {
            Ok(file) => {
                self.publish_granted_bytes(observer)?;
                file
            }
            Err(error) => {
                if let Some(file) = retry_file.take() {
                    // `prepare_push` admitted this exact exceptional owner
                    // before publication began, so failure recovery cannot
                    // allocate or lose the only retry handle.
                    self.runs.push(file, row_count);
                }
                self.publish_granted_bytes_preserving_primary(observer);
                return Err(error);
            }
        };

        if self.scalar_cleanup.is_some() {
            self.runs.push(output, row_count);
            check_cancellation(cancellation)?;
            for index in start..end {
                self.runs.entries[index].file.close_and_delete()?;
            }
            let output_index = self.runs.len() - 1;
            let output = self.runs.entries.remove(output_index);
            self.runs.entries[start] = output;
            for _ in 1..run_count {
                self.runs.remove(start + 1);
            }
            #[cfg(test)]
            {
                self.merged_row_visits += row_count as u128;
            }
            check_cancellation(cancellation)?;
            return Ok(());
        }

        if let Err(error) = check_cancellation(cancellation) {
            let primary = ExternalSortOperationError::Cancelled(error);
            return match output.close_and_delete() {
                Ok(()) => Err(primary),
                Err(cleanup) => {
                    self.runs.push(output, row_count);
                    Err(with_io_cleanup(
                        primary,
                        cleanup,
                        "intermediate sort spill cleanup",
                    ))
                }
            };
        }

        // Readers owned by `merge_run_slice_to_file` are gone before deletion.
        // Do not poll cancellation through cleanup: every owned input must get
        // its explicit deletion attempt once replacement has been published.
        for index in start..end {
            if let Err(input_cleanup) = self.runs.entries[index].file.close_and_delete() {
                let primary = ExternalSortOperationError::Io(input_cleanup);
                return match output.close_and_delete() {
                    Ok(()) => Err(primary),
                    Err(output_cleanup) => {
                        self.runs.push(output, row_count);
                        Err(with_io_cleanup(
                            primary,
                            output_cleanup,
                            "intermediate sort output cleanup",
                        ))
                    }
                };
            }
        }

        self.runs.entries[start] = ExternalSortRunEntry {
            file: output,
            rows: row_count,
        };
        for _ in 1..run_count {
            self.runs.remove(start + 1);
        }
        #[cfg(test)]
        {
            self.merged_row_visits += row_count as u128;
        }
        check_cancellation(cancellation)?;
        Ok(())
    }

    /// Reads a single run from disk.
    fn read_single_run(
        &mut self,
        run_index: usize,
        cancellation: Option<&QueryCancellationToken>,
    ) -> Result<Vec<Vec<Value>>, ExternalSortOperationError> {
        check_cancellation(cancellation)?;
        let spill_file = &self.runs.get(run_index).file;
        let mut reader = spill_file.reader()?;
        let (columns, row_count) = reader.read_sort_run_start()?;
        self.validate_columns(columns)?;
        self.validate_row_count(run_index, row_count)?;
        check_cancellation(cancellation)?;
        let mut rows = Vec::new();
        for _ in 0..row_count {
            check_cancellation(cancellation)?;
            let payload = reader.read_sort_row()?;
            let row = decode_row_payload_with_shape(&payload, self.row_shape, spill_file.limits())?;
            rows.try_reserve(1).map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    format!("reserve sort output: {error}"),
                )
            })?;
            rows.push(row.values);
            check_cancellation(cancellation)?;
        }
        reader.finish()?;
        check_cancellation(cancellation)?;
        Ok(rows)
    }

    /// Performs k-way merge of all runs and an optional in-memory buffer.
    fn k_way_merge(
        &mut self,
        in_memory_buffer: Vec<Vec<Value>>,
        cancellation: Option<&QueryCancellationToken>,
    ) -> Result<Vec<Vec<Value>>, ExternalSortOperationError> {
        let _profile_merge = ProfileMergeTimer::new(&self.manager);
        debug_assert!(
            self.runs.len() + usize::from(!in_memory_buffer.is_empty()) <= self.merge_fan_in,
            "final external merge must respect the configured fan-in"
        );
        check_cancellation(cancellation)?;
        let mut in_memory_buffer = self.attach_input_ordinals(in_memory_buffer)?;
        let total_rows = self
            .checked_total_rows_with_cancellation(cancellation)?
            .checked_add(in_memory_buffer.len())
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "merged sort row count exceeds the platform address space",
                )
            })?;
        let mut result = Vec::new();
        result.try_reserve_exact(total_rows).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                format!("reserve merged sort output: {error}"),
            )
        })?;
        check_cancellation(cancellation)?;

        let comparator = &self.comparator;

        // Sort the in-memory buffer first.
        if !in_memory_buffer.is_empty() {
            check_cancellation(cancellation)?;
            sort_ordinal_rows(&mut in_memory_buffer, comparator)?;
            check_cancellation(cancellation)?;
        }

        let memory_heap_slot = usize::from(!in_memory_buffer.is_empty());
        let initial_heap_capacity =
            self.runs
                .len()
                .checked_add(memory_heap_slot)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::OutOfMemory,
                        "sort merge heap count exceeds the platform address space",
                    )
                })?;
        let mut heap = ComparatorMinHeap::try_with_exact_capacity(initial_heap_capacity)
            .map_err(ComparatorMinHeapAllocationError::into_operation)?;
        check_cancellation(cancellation)?;

        // Create readers for all runs. The reader vector and heap have already
        // reserved their maximum entry counts, so publishing a decoded first
        // row into the merge frontier cannot allocate.
        let mut run_readers: Vec<Option<RunReader>> = Vec::new();
        run_readers
            .try_reserve_exact(self.runs.len())
            .map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    format!("reserve sort run readers: {error}"),
                )
            })?;
        for (idx, entry) in self.runs.iter().enumerate() {
            check_cancellation(cancellation)?;
            let spill_file = &entry.file;
            let mut reader = spill_file.reader()?;
            let (columns, row_count) = reader.read_sort_run_start()?;
            self.validate_columns(columns)?;
            self.validate_row_count(idx, row_count)?;
            check_cancellation(cancellation)?;

            if row_count > 0 {
                let payload = reader.read_sort_row()?;
                let first_row =
                    decode_row_payload_with_shape(&payload, self.row_shape, spill_file.limits())?;
                check_cancellation(cancellation)?;
                run_readers.push(Some(RunReader {
                    reader,
                    remaining: row_count - 1,
                    row_shape: self.row_shape,
                    limits: spill_file.limits(),
                    finished: false,
                }));
                heap.push_entry(
                    HeapEntry {
                        row: first_row,
                        run_index: idx,
                        retained_bytes: 0,
                    },
                    comparator,
                )?;
            } else {
                reader.finish()?;
                run_readers.push(None);
            }
            check_cancellation(cancellation)?;
        }

        // Move the first in-memory row into the heap; no row or key clone is
        // required anywhere in the merge frontier.
        let mut memory_iter = in_memory_buffer.into_iter();
        let memory_run_index = self.runs.len();
        if let Some(row) = memory_iter.next() {
            heap.push_entry(
                HeapEntry {
                    row,
                    run_index: memory_run_index,
                    retained_bytes: 0,
                },
                comparator,
            )?;
        }
        check_cancellation(cancellation)?;

        // Merge loop
        loop {
            check_cancellation(cancellation)?;
            let Some(entry) = heap.pop_entry(comparator)? else {
                break;
            };
            result.push(entry.row.values);

            if entry.run_index == memory_run_index {
                // Advance memory iterator
                if let Some(row) = memory_iter.next() {
                    heap.push_entry(
                        HeapEntry {
                            row,
                            run_index: memory_run_index,
                            retained_bytes: 0,
                        },
                        comparator,
                    )?;
                }
            } else {
                // Advance file run
                let run_reader = run_readers[entry.run_index].as_mut().ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "heap referenced an empty sort run",
                    )
                })?;
                let next_row = run_reader.next_row()?;
                check_cancellation(cancellation)?;
                if let Some(next_row) = next_row {
                    heap.push_entry(
                        HeapEntry {
                            row: next_row,
                            run_index: entry.run_index,
                            retained_bytes: 0,
                        },
                        comparator,
                    )?;
                }
            }
            check_cancellation(cancellation)?;
        }

        check_cancellation(cancellation)?;
        Ok(result)
    }

    fn validate_columns(&self, columns: u32) -> std::io::Result<()> {
        let expected = u32::try_from(self.num_columns).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "sort column count exceeds u32",
            )
        })?;
        if columns != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("sort run declares {columns} columns, expected {expected}"),
            ));
        }
        Ok(())
    }

    fn validate_row_count(&self, run_index: usize, declared: u64) -> std::io::Result<()> {
        let tracked = u64::try_from(self.runs.get(run_index).rows).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "tracked sort row count exceeds u64",
            )
        })?;
        if declared != tracked {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("sort run declares {declared} rows, but manager tracked {tracked}"),
            ));
        }
        Ok(())
    }

    /// Explicitly deletes every run, retaining failed handles for retry.
    ///
    /// # Errors
    ///
    /// Returns the first owned deletion error without formatting it.
    pub fn cleanup(&mut self) -> std::io::Result<()> {
        self.cleanup_inner()
            .map_err(ExternalSortOperationError::into_io)
    }

    /// Cleans up the DISTINCT sort owner without boxing or erasing its typed
    /// primary and cleanup failures through the compatibility I/O interface.
    /// The caller retains its observer until every resource owner is released
    /// and publishes failures through its pre-admitted terminal carrier.
    pub(crate) fn cleanup_distinct_accounted(&mut self) -> Result<(), ExternalSortOperationError> {
        self.cleanup_inner()
    }

    fn cleanup_inner(&mut self) -> Result<(), ExternalSortOperationError> {
        let workspace_result = self.release_memory_workspaces();
        let cleanup_result = self.cleanup_runs();
        match (workspace_result, cleanup_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) => Err(error),
            (Ok(()), Err(error)) => Err(error),
            (Err(error), Err(cleanup)) => Err(with_io_cleanup(
                error,
                cleanup.into_io(),
                "sort spill cleanup",
            )),
        }
    }

    fn release_memory_workspaces(&mut self) -> Result<(), ExternalSortOperationError> {
        let observer = ExternalSortGrantObserver::inert();
        self.release_memory_workspaces_observing(&observer)
    }

    fn release_memory_workspaces_observing(
        &mut self,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        let workspace = self.workspace.release_capacity();
        let writer = self.writer_workspace.release_capacity();
        let primary = match (workspace, writer) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => {
                Err(ExternalSortOperationError::Memory(error))
            }
            (Err(primary), Err(release)) => Err(ExternalSortOperationError::WithGrantRelease {
                primary: ExternalSortPrimary::Memory(primary),
                release,
                cleanup: None,
                phase: "writer workspace cleanup",
            }),
        };
        let cursor = self.release_cursor_grants(true, observer);
        match (primary, cursor) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(primary), Ok(())) => Err(primary),
            (Ok(()), Err(cursor)) => Err(cursor),
            (Err(primary), Err(cursor)) => Err(with_io_cleanup(
                primary,
                cursor.into_io(),
                "sort cursor grant release",
            )),
        }
    }

    /// Releases every allocation retained by a terminal exact cursor. Run
    /// handles must have been deleted first; an incomplete cleanup deliberately
    /// retains the catalog backing and its retry authority.
    fn release_exact_terminal_resources(
        &mut self,
        observer: &ExternalSortGrantObserver<'_>,
        defer_frontier_release: bool,
    ) -> (
        Option<ExactOwnedResourceRelease>,
        Option<ExactOwnedCleanupPanic>,
    ) {
        let mut first_panic = None;
        // Exact cursors have already reclaimed every head and reader. Relinquish
        // this comparator owner before final publication, without shrinking a
        // grant that another comparator clone still owns.
        if self.comparator.is_accounted() {
            let previous = std::mem::replace(&mut self.comparator.compare, Comparison::Released);
            if let Comparison::Accounted(accounted) = previous
                && let Some(accounted) = Arc::into_inner(accounted)
            {
                let AccountedComparison {
                    keys,
                    provider,
                    fixed_bytes: _,
                    grant,
                    resources,
                } = accounted;
                let grant = grant.into_inner();
                // The retained grant outlives all physical metadata, including
                // a provider destructor panic. Existing frontier cleanup then
                // owns explicit retry if its release fails.
                let destruction = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    drop((keys, provider, resources));
                }));
                self.merge_exact_frontier_grant(grant, "terminal comparator authority");
                if let Err(payload) = destruction {
                    retain_first_cleanup_panic(&mut first_panic, payload);
                }
            }
        }
        macro_rules! memory_release {
            ($operation:expr) => {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $operation)) {
                    Ok(result) => result.err(),
                    Err(payload) => {
                        retain_first_cleanup_panic(&mut first_panic, payload);
                        None
                    }
                }
            };
        }
        let workspace = memory_release!(self.workspace.release_capacity());
        let writer = memory_release!(self.writer_workspace.release_capacity());
        let ordinal = memory_release!(match self.ordinal_grant.as_mut() {
            Some(grant) => grant.try_resize(0),
            None => Ok(()),
        });
        let frontier = if defer_frontier_release {
            None
        } else {
            memory_release!(match self.frontier_grant.as_mut() {
                Some(grant) => grant.try_resize(0),
                None => Ok(()),
            })
        };
        let payload = memory_release!(match self.payload_grant.as_mut() {
            Some(grant) => grant.try_resize(0),
            None => Ok(()),
        });
        let output = memory_release!(match self.output_grant.as_mut() {
            Some(grant) => grant.try_resize(0),
            None => Ok(()),
        });
        let catalog = if self.runs.is_empty() {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.runs.release_empty_capacity()
            })) {
                Ok(result) => result.err(),
                Err(payload) => {
                    retain_first_cleanup_panic(&mut first_panic, payload);
                    None
                }
            }
        } else {
            None
        };
        let publication = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.publish_granted_bytes(observer)
        })) {
            Ok(result) => result.err(),
            Err(payload) => {
                retain_first_cleanup_panic(&mut first_panic, payload);
                None
            }
        };
        let release = ExactOwnedResourceRelease {
            workspace,
            writer,
            ordinal,
            frontier,
            payload,
            output,
            catalog,
            publication,
        };
        ((!release.is_empty()).then_some(release), first_panic)
    }

    /// Best-effort cleanup used only from `Drop`.
    ///
    /// Unlike the explicit cleanup path, this never formats or destroys an
    /// opaque downstream failure. Each file is isolated so one hostile hook
    /// cannot prevent later runs from receiving their own cleanup attempt.
    fn cleanup_for_drop(&mut self) -> Result<(), ()> {
        // The unpublished block's physical allocation must disappear before
        // its child grant and before the remaining sorter grants are released.
        if let Some(publisher) = self.exact_failure_publisher.take() {
            self.merge_exact_publisher_grant(publisher.into_unpublished_grant());
        }
        let workspace_released = super::run_cleanup_backstop(|| self.workspace.release_capacity());
        let writer_released =
            super::run_cleanup_backstop(|| self.writer_workspace.release_capacity());
        let observer = ExternalSortGrantObserver::inert();
        let cursor_released =
            super::run_cleanup_backstop(|| self.release_cursor_grants(true, &observer));
        let runs_cleaned = self.cleanup_runs_for_drop();
        let catalog_released = runs_cleaned
            && self.runs.is_empty()
            && super::run_cleanup_backstop(|| self.runs.release_empty_capacity());
        if workspace_released
            && writer_released
            && cursor_released
            && runs_cleaned
            && catalog_released
        {
            Ok(())
        } else {
            Err(())
        }
    }

    fn cleanup_runs_for_drop(&mut self) -> bool {
        let mut index = 0;
        let mut removed_any = false;
        let mut failed = false;
        while index < self.runs.len() {
            if super::run_cleanup_backstop(|| self.runs.entries[index].file.close_and_delete()) {
                self.runs.remove(index);
                removed_any = true;
            } else {
                quarantine_pull_sort_hook(self.pull_hook_workspace.as_ref());
                mark_scalar_cleanup_failed(self.scalar_cleanup.as_ref());
                failed = true;
                index += 1;
            }
        }
        if failed && removed_any {
            self.disk_merge_started = true;
        }
        !failed
    }

    fn cleanup_runs_scalar(&mut self) -> Result<(), ExternalSortOperationError> {
        while !self.runs.is_empty() {
            self.runs.entries[0].file.close_and_delete()?;
            self.runs.remove(0);
        }
        Ok(())
    }

    fn cleanup_runs(&mut self) -> Result<(), ExternalSortOperationError> {
        let mut index = 0;
        let mut removed_any = false;
        let mut first_error = None;
        while index < self.runs.len() {
            match self.runs.entries[index].file.close_and_delete() {
                Ok(()) => {
                    self.runs.remove(index);
                    removed_any = true;
                }
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    } else {
                        // A later opaque failure must not replace the first by
                        // formatting, dropping, or panicking during disposal.
                        quarantine_pull_sort_hook(self.pull_hook_workspace.as_ref());
                        let _ = super::run_cleanup_backstop(|| Err::<(), _>(error));
                    }
                    index += 1;
                }
            }
        }
        if let Some(error) = first_error {
            if removed_any {
                // Some runs have already been destroyed. Refuse subsequent
                // publication/merge rather than treating the remainder as a
                // complete sorter; a later cleanup call can still retry deletion.
                self.disk_merge_started = true;
            }
            Err(error.into())
        } else {
            Ok(())
        }
    }

    /// Exact-lane deletion keeps the first typed failure and the first unwind
    /// in independent fixed slots while still attempting every run. A panic
    /// from one extension hook therefore cannot erase an earlier error or
    /// prevent later handles from receiving cleanup.
    fn cleanup_runs_exact(
        &mut self,
    ) -> (
        Option<ExternalSortOperationError>,
        Option<ExactOwnedCleanupPanic>,
    ) {
        let mut index = 0;
        let mut removed_any = false;
        let mut first_error = None;
        let mut first_panic = None;
        while index < self.runs.len() {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.runs.entries[index].file.close_and_delete()
            }));
            match outcome {
                Ok(Ok(())) => {
                    self.runs.remove(index);
                    removed_any = true;
                }
                Ok(Err(error)) => {
                    let error = ExternalSortOperationError::Io(error);
                    if first_error.is_none() {
                        first_error = Some(error);
                    } else {
                        quarantine_pull_sort_hook(self.pull_hook_workspace.as_ref());
                        let _ = super::run_cleanup_backstop(|| Err::<(), _>(error));
                    }
                    index += 1;
                }
                Err(payload) => {
                    // Secondary panic owners may be forgotten by the bounded
                    // first-panic carrier. Retain the full hook allowance.
                    quarantine_pull_sort_hook(self.pull_hook_workspace.as_ref());
                    retain_first_cleanup_panic(&mut first_panic, payload);
                    index += 1;
                }
            }
        }
        if (first_error.is_some() || first_panic.is_some()) && removed_any {
            self.disk_merge_started = true;
        }
        (first_error, first_panic)
    }
}

impl Drop for ExternalSort {
    fn drop(&mut self) {
        let unwinding = std::thread::panicking();
        let clean = super::run_cleanup_backstop(|| self.cleanup_for_drop());
        if unwinding || !clean {
            quarantine_pull_sort_hook(self.pull_hook_workspace.as_ref());
        }
        if !clean && !self.runs.is_empty() {
            let stranded = u64::try_from(self.runs.len()).unwrap_or(u64::MAX);
            super::manager::record_orphan_cleanup_failures(stranded);
        }
    }
}

#[expect(
    clippy::result_large_err,
    reason = "the exact lane keeps rich cleanup failures inline because boxing could allocate on an already-failing, resource-accounted path"
)]
impl ExactOwnedSortCursor<'_> {
    fn initialize(&mut self) -> Result<(), ExactOwnedSortStreamError> {
        let _profile_merge = ProfileMergeTimer::new(&self.sorter.manager);
        if !self.sorter.exact_owned_disk_shape_eligible() {
            return Err(ExternalSortOperationError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "external sort shape is not eligible for exact owned output",
            ))
            .into());
        }
        self.check_cancellation_raw()?;

        let run_count = self.sorter.runs.len();
        let heap_bytes = cursor_capacity_bytes::<ExactOwnedHeapEntry>(run_count)?;
        let reader_slot_bytes = cursor_capacity_bytes::<Option<ExactOwnedRunReader>>(run_count)?;
        let frontier_bytes = checked_workspace_sum(heap_bytes, reader_slot_bytes)?;
        self.sorter
            .resize_frontier_grant(frontier_bytes, &self.observer)?;

        let heap = ComparatorMinHeap::try_with_exact_capacity(run_count)
            .map_err(ComparatorMinHeapAllocationError::into_operation)?;
        let mut readers = ExactOwnedReaderSlots::new_in(Global);
        readers.try_reserve_exact(run_count).map_err(|_| {
            ExternalSortOperationError::Allocation(std::io::Error::from(
                std::io::ErrorKind::OutOfMemory,
            ))
        })?;
        if readers.capacity() != run_count || heap.capacity() != run_count {
            return Err(ExternalSortOperationError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "exact owned sort container capacity diverged from its admitted layout",
            ))
            .into());
        }
        self.heap = heap;
        self.readers = readers;
        self.sorter.disk_merge_started = true;
        let expected_columns = u32::try_from(self.sorter.num_columns).map_err(|_| {
            ExternalSortOperationError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "sort column count exceeds u32",
            ))
        })?;
        for run_index in 0..run_count {
            self.check_cancellation_raw()?;
            self.initialize_run(run_index, expected_columns)?;
            if self.sorter.comparator.is_accounted() {
                self.publish_live()?;
            }
        }
        self.check_cancellation_raw()?;
        Ok(())
    }

    fn initialize_run(
        &mut self,
        run_index: usize,
        expected_columns: u32,
    ) -> Result<(), ExactOwnedSortStreamError> {
        let (qualification, limits, expected_rows) = {
            let run = self.sorter.runs.get(run_index);
            let qualification = run.file.exact_owned_reader_qualification().ok_or_else(|| {
                ExternalSortOperationError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "sort run lost its cached exact-reader qualification",
                ))
            })?;
            let rows = u64::try_from(run.rows).map_err(|_| {
                ExternalSortOperationError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "tracked sort row count exceeds u64",
                ))
            })?;
            (qualification, run.file.limits(), rows)
        };
        let reader = self.open_reader(run_index, qualification)?;
        let receipt = reader.receipt();
        let current_reader_bytes = self.reader_bytes.get();
        let Some(base_reader_bytes) = current_reader_bytes.checked_sub(receipt.bytes()) else {
            let primary = MemoryGrantError::AccountingUnderflow {
                account: "exact owned sort reader frontier",
                accounted_bytes: current_reader_bytes,
                release_bytes: receipt.bytes(),
            };
            self.abort_local_reader_preserving_primary(reader);
            self.observer.publish_unrepresentable();
            return Err(primary.into());
        };
        let reader = match reader.read_sort_run_start_owned(expected_columns, expected_rows) {
            Ok(reader) => reader,
            Err(error) => {
                self.reset_reader_preserving_primary(base_reader_bytes);
                return Err(ExactOwnedSortStreamError::accounted(error));
            }
        };
        if expected_rows == 0 {
            let finish = reader.finish_sort_run_owned();
            match finish {
                Ok(()) => self.set_reader_bytes(base_reader_bytes)?,
                Err(error) => {
                    let error = self.adopt_reader_resolution_error(error);
                    self.reset_reader_preserving_primary(base_reader_bytes);
                    return Err(error);
                }
            }
            self.readers.push(None);
            return Ok(());
        }

        let (reader, row) = self.read_decode_head(reader, receipt, limits)?;
        debug_assert_eq!(
            self.readers.len(),
            run_index,
            "exact reader slots remain index-aligned with the run catalog"
        );
        self.readers.push(Some(ExactOwnedRunReader {
            reader: Some(reader),
            receipt,
            remaining: expected_rows - 1,
            limits,
        }));
        // Install the reader owner before comparator code can unwind. The heap
        // operation either retains or physically drops `row`; in both cases the
        // outer initialization fence can now find and explicitly abort the
        // paired reader instead of relying on its lossy Drop backstop.
        let push = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.heap.push_exact_owned(
                ExactOwnedHeapEntry { row, run_index },
                &self.sorter.comparator,
            )
        }));
        if let Ok(Err(error)) = push {
            self.state = ExactOwnedCursorState::Failed;
            let _ = self.publish_live();
            return Err(error.into());
        }
        if let Err(payload) = push {
            self.state = ExactOwnedCursorState::Failed;
            let _ = self.publish_live();
            std::panic::resume_unwind(payload);
        }
        Ok(())
    }

    /// Number of output columns fixed by the decoded run headers.
    pub(crate) fn num_columns(&self) -> usize {
        self.sorter.num_columns
    }

    #[cfg(test)]
    fn inject_next_decoder_panic(&mut self, payload: Box<dyn std::any::Any + Send>) {
        self.sorter.inject_exact_decoder_panic(payload);
    }

    pub(crate) fn output_observer(&self) -> &ExternalSortGrantObserver<'_> {
        &self.observer
    }

    pub(crate) fn release_transferred_retained(&self) -> Result<(), ExactOwnedSortStreamError> {
        self.observer.publish_retained(0)?;
        self.publish_live()
    }

    /// Checked live cursor authority, excluding any row already transferred
    /// to its downstream accounted chunk.
    pub(crate) fn checked_granted_bytes(&self) -> Result<usize, MemoryGrantError> {
        checked_workspace_sum(
            self.sorter.checked_total_granted_bytes()?,
            checked_workspace_sum(self.reader_bytes.get(), self.row_bytes.get())?,
        )
    }

    fn check_cancellation_raw(&self) -> Result<(), ExactOwnedSortStreamError> {
        check_cancellation(self.sorter.cancellation.as_ref()).map_err(Into::into)
    }

    fn transition_observer(
        &self,
    ) -> Result<ExactOwnedGrantTransitionObserver<'_, '_>, ExactOwnedSortStreamError> {
        let stable_sorter_bytes = match self.sorter.checked_total_granted_bytes() {
            Ok(bytes) => bytes,
            Err(error) => {
                self.observer.publish_unrepresentable();
                return Err(error.into());
            }
        };
        Ok(ExactOwnedGrantTransitionObserver {
            observer: &self.observer,
            stable_sorter_bytes,
            reader_bytes: &self.reader_bytes,
            row_bytes: &self.row_bytes,
        })
    }

    fn publish_live(&self) -> Result<(), ExactOwnedSortStreamError> {
        let bytes = match self.checked_granted_bytes() {
            Ok(bytes) => bytes,
            Err(error) => {
                self.observer.publish_unrepresentable();
                return Err(error.into());
            }
        };
        self.observer.publish(bytes)?;
        Ok(())
    }

    fn publish_final_failure(
        &mut self,
        primary: ExactOwnedFinalCause,
        secondary: Option<(ExactOwnedFinalCause, &'static str)>,
        cleanup: Option<(ExactOwnedFinalCause, &'static str)>,
    ) -> Option<OperatorError> {
        let publisher_bytes = self
            .sorter
            .exact_failure_publisher
            .as_ref()
            .map(AccountedErrorPublisher::granted_bytes)?;
        let classification = primary.classification();
        let mut final_failure =
            ExactOwnedFinalFailure::from_parts(primary, secondary, cleanup, None);
        final_failure
            .hook_workspace
            .clone_from(&self.sorter.pull_hook_workspace);
        let lower_total = self.checked_granted_bytes().and_then(|bytes| {
            bytes
                .checked_sub(publisher_bytes)
                .ok_or(MemoryGrantError::AccountingUnderflow {
                    account: "exact final publisher attribution",
                    accounted_bytes: bytes,
                    release_bytes: publisher_bytes,
                })
        });
        let publisher = self
            .sorter
            .exact_failure_publisher
            .take()
            .expect("the final publisher was borrowed above");
        let mut transferred = false;
        match lower_total {
            Ok(bytes) => {
                if let Err(error) = self
                    .observer
                    .transfer_control_to_retained(bytes, publisher_bytes)
                {
                    final_failure.reconciliation = Some((
                        ExactOwnedFinalCause::Stream(error.into()),
                        "exact final publisher attribution transfer",
                    ));
                } else {
                    transferred = true;
                }
            }
            Err(error) => {
                self.observer.publish_unrepresentable();
                final_failure.reconciliation = Some((
                    ExactOwnedFinalCause::Stream(error.into()),
                    "exact final publisher attribution calculation",
                ));
            }
        }
        let accounted = publisher.publish(final_failure);
        if transferred {
            self.observer.finish_control_transfer(publisher_bytes);
        }
        Some(OperatorError::ClassifiedAccountedFailure {
            classification,
            authority: accounted,
        })
    }

    fn publish_initialization_failure(
        &mut self,
        primary: ExactOwnedSortStreamError,
    ) -> ExactOwnedSortStreamError {
        let cleanup = self.finish_resources_preserving_final_publisher();
        assert!(
            self.sorter.exact_failure_publisher.is_some(),
            "qualified exact initialization failure lost its pre-admitted final publisher"
        );
        if cleanup.is_none() && primary.can_return_without_final_publisher() {
            match self.release_unused_final_publisher() {
                Ok(()) => {
                    self.state = ExactOwnedCursorState::Terminal;
                    return primary;
                }
                Err(error) => {
                    self.retry_terminal_after_publisher_resolution_failure();
                    drop(error);
                    return primary;
                }
            }
        }
        let accounted = self
            .publish_final_failure(
                ExactOwnedFinalCause::Stream(primary),
                None,
                cleanup.map(|error| {
                    (
                        ExactOwnedFinalCause::Terminal(error),
                        "exact owned sort initialization cleanup",
                    )
                }),
            )
            .expect("initialization retained its pre-admitted final publisher");
        match accounted {
            OperatorError::ClassifiedAccountedFailure {
                classification,
                authority,
            } => ExactOwnedSortStreamError::classified_accounted(classification, authority),
            _ => unreachable!("exact final publication always creates an accounted failure"),
        }
    }

    /// Terminalizes a failed exact stream and publishes its primary plus any
    /// cleanup/reconciliation secondaries in one pre-admitted owner.
    pub(crate) fn finish_stream_failure(
        &mut self,
        primary: ExactOwnedSortStreamError,
        cleanup_phase: &'static str,
    ) -> OperatorError {
        if self.sorter.exact_failure_publisher.is_none() {
            self.retry_terminal_after_publisher_resolution_failure();
            return primary.into_operator_error();
        }
        let cleanup = self.finish_resources_preserving_final_publisher();
        if cleanup.is_none() && primary.can_return_without_final_publisher() {
            match self.release_unused_final_publisher() {
                Ok(()) => {
                    self.state = ExactOwnedCursorState::Terminal;
                    return primary.into_operator_error();
                }
                Err(error) => {
                    self.retry_terminal_after_publisher_resolution_failure();
                    drop(error);
                    return primary.into_operator_error();
                }
            }
        }
        let terminalized_cleanly = cleanup.is_none();
        let result = self
            .publish_final_failure(
                ExactOwnedFinalCause::Stream(primary),
                None,
                cleanup.map(|error| (ExactOwnedFinalCause::Terminal(error), cleanup_phase)),
            )
            .expect("active exact cursor failures retain their pre-admitted final publisher");
        if terminalized_cleanly {
            self.state = ExactOwnedCursorState::Terminal;
        }
        result
    }

    /// Publishes a downstream primary, an optional exact transfer failure, and
    /// terminal cleanup without allocating or invoking either diagnostic's
    /// formatter on the failure path.
    pub(crate) fn finish_operator_failure(
        &mut self,
        primary: OperatorError,
        secondary: Option<(ExactOwnedSortStreamError, &'static str)>,
        cleanup_phase: &'static str,
    ) -> OperatorError {
        let cleanup = self.finish_resources_preserving_final_publisher();
        if secondary.is_none() && cleanup.is_none() {
            match self.release_unused_final_publisher() {
                Ok(()) => {
                    self.state = ExactOwnedCursorState::Terminal;
                    return primary;
                }
                Err(error) => {
                    self.retry_terminal_after_publisher_resolution_failure();
                    drop(error);
                    return primary;
                }
            }
        }
        let terminalized_cleanly = cleanup.is_none();
        let result = self
            .publish_final_failure(
                ExactOwnedFinalCause::Operator(primary),
                secondary.map(|(error, phase)| (ExactOwnedFinalCause::Stream(error), phase)),
                cleanup.map(|error| (ExactOwnedFinalCause::Terminal(error), cleanup_phase)),
            )
            .expect("active exact cursor failures retain their pre-admitted final publisher");
        if terminalized_cleanly {
            self.state = ExactOwnedCursorState::Terminal;
        }
        result
    }

    /// Handles a downstream early-stop request. A successful cleanup drops the
    /// unused publisher; a failure is transported through that same slot.
    pub(crate) fn finish_early_stop(&mut self) -> Result<(), OperatorError> {
        match self.abort() {
            Ok(()) => Ok(()),
            Err(error) if self.sorter.exact_failure_publisher.is_none() => {
                self.retry_terminal_after_publisher_resolution_failure();
                Err(error.into_operator_error_without_publisher())
            }
            Err(error) => Err(self
                .publish_final_failure(ExactOwnedFinalCause::Terminal(error), None, None)
                .expect("failed exact cleanup retains its pre-admitted final publisher")),
        }
    }

    fn retry_terminal_after_publisher_resolution_failure(&mut self) {
        if let Some(error) = self.finish_resources_preserving_final_publisher() {
            let _ = super::run_cleanup_backstop(|| {
                drop(error);
                Ok::<(), std::convert::Infallible>(())
            });
            return;
        }
        match self.release_unused_final_publisher() {
            Ok(()) => self.state = ExactOwnedCursorState::Terminal,
            Err(error) => {
                let _ = super::run_cleanup_backstop(|| {
                    drop(error);
                    Ok::<(), std::convert::Infallible>(())
                });
            }
        }
    }

    fn set_reader_bytes(&self, bytes: usize) -> Result<(), ExactOwnedSortStreamError> {
        self.reader_bytes.set(bytes);
        self.publish_live()
    }

    fn reclaim_row_to_frontier_preserving_primary(&mut self, row: AccountedOrdinalRow) {
        let bytes = row.granted_bytes();
        let grant = row.into_released_grant();
        self.sorter
            .merge_exact_frontier_grant(grant, "discarded row authority");
        match self.row_bytes.get().checked_sub(bytes) {
            Some(remaining) => {
                self.row_bytes.set(remaining);
                let _ = self.publish_live();
            }
            None => self.observer.publish_unrepresentable(),
        }
    }

    fn reclaim_heap_entry_preserving_primary(&mut self, entry: ExactOwnedHeapEntry) {
        self.reclaim_row_to_frontier_preserving_primary(entry.row);
    }

    /// Reclaims every internally owned heap row before terminal grant release.
    /// Row storage and recursive receipts are destroyed first; their grants
    /// then move into the sorter frontier while no observer callback can see
    /// the transient internal ownership change.
    fn reclaim_terminal_heap(&mut self) {
        let heap = std::mem::replace(&mut self.heap, ComparatorMinHeap::new());
        let mut reclaimed_bytes = Some(0usize);
        for entry in heap.entries {
            let bytes = entry.row.granted_bytes();
            let grant = entry.row.into_released_grant();
            self.sorter
                .merge_exact_frontier_grant(grant, "terminal row authority");
            reclaimed_bytes = reclaimed_bytes.and_then(|total| total.checked_add(bytes));
        }
        if reclaimed_bytes != Some(self.row_bytes.get()) {
            self.observer.publish_unrepresentable();
        }
        self.row_bytes.set(0);
    }

    /// Reclaims a failed row-to-chunk construction grant from downstream
    /// retained attribution into the cursor's retryable frontier.
    pub(crate) fn reclaim_failed_output_grant(
        &mut self,
        grant: MemoryGrant,
    ) -> Result<(), ExactOwnedSortStreamError> {
        let transferred_bytes = grant.size();
        let previous_external = self.checked_granted_bytes();
        self.sorter
            .merge_exact_frontier_grant(grant, "failed output construction authority");
        let current_external = self.checked_granted_bytes();
        let previous_external = match previous_external {
            Ok(bytes) => bytes,
            Err(error) => {
                self.observer.publish_unrepresentable();
                return Err(error.into());
            }
        };
        let current_external = match current_external {
            Ok(bytes) => bytes,
            Err(error) => {
                self.observer.publish_unrepresentable();
                return Err(error.into());
            }
        };
        self.observer
            .transfer_retained_to_external(previous_external, current_external, transferred_bytes)
            .map_err(Into::into)
    }

    fn reset_reader_preserving_primary(&self, bytes: usize) {
        self.reader_bytes.set(bytes);
        let _ = self.publish_live();
    }

    fn reset_frontier_preserving_primary(&self, reader_bytes: usize, row_bytes: usize) {
        // Both sealed scalar cells are committed before the one externally
        // visible publication, so a reader-to-diagnostic hand-off cannot
        // expose a half-reset frontier.
        self.reader_bytes.set(reader_bytes);
        self.row_bytes.set(row_bytes);
        let _ = self.publish_live();
    }

    fn adopt_reader_resolution_error(
        &mut self,
        error: ProviderAccountedReaderResolutionError,
    ) -> ExactOwnedSortStreamError {
        match error {
            ProviderAccountedReaderResolutionError::Accounted(error) => {
                ExactOwnedSortStreamError::accounted(error)
            }
            ProviderAccountedReaderResolutionError::UnusedPublication(error) => {
                let (primary, grant) = error.into_parts();
                self.sorter.merge_exact_publisher_grant(grant);
                ExternalSortOperationError::Memory(primary).into()
            }
        }
    }

    /// Explicitly resolves a reader that is still locally owned after another
    /// operation has already established the primary failure.
    ///
    /// The result is retained in fixed cursor slots rather than returned as a
    /// replacement primary. A failed unused-publisher release first transfers
    /// its still-live grant into the sorter frontier; callers then reduce the
    /// reader scalar and publish the combined post-transfer total exactly once.
    fn abort_local_reader_preserving_primary(&mut self, reader: ProviderAccountedSpillFileReader) {
        debug_assert!(
            self.pending_reader_cleanup.is_empty(),
            "one primary failure cannot strand more than one local exact reader"
        );
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| reader.abort_owned()));
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(ProviderAccountedReaderResolutionError::Accounted(error))) => {
                if self.pending_reader_cleanup.accounted.is_none() {
                    self.pending_reader_cleanup.accounted = Some(error);
                } else {
                    // This is unreachable under the cursor's terminal-on-first-
                    // primary protocol. Fail closed if that protocol is ever
                    // violated: never overwrite or release the earlier owner.
                    self.observer.publish_unrepresentable();
                    std::mem::forget(error);
                }
            }
            Ok(Err(ProviderAccountedReaderResolutionError::UnusedPublication(error))) => {
                let (primary, grant) = error.into_parts();
                self.sorter.merge_exact_publisher_grant(grant);
                if self.pending_reader_cleanup.publication_release.is_none() {
                    self.pending_reader_cleanup.publication_release = Some(primary);
                } else {
                    self.observer.publish_unrepresentable();
                }
            }
            Err(payload) => {
                let panic = ExactOwnedCleanupPanic::new(payload);
                if self.pending_reader_cleanup.panic.is_none() {
                    self.pending_reader_cleanup.panic = Some(panic);
                } else {
                    self.observer.publish_unrepresentable();
                    std::mem::forget(panic);
                }
            }
        }
    }

    fn open_reader(
        &mut self,
        run_index: usize,
        qualification: ExactOwnedReaderQualification,
    ) -> Result<ProviderAccountedSpillFileReader, ExactOwnedSortStreamError> {
        let base_reader_bytes = self.reader_bytes.get();
        let open_publication_bytes =
            AccountedErrorPublisher::<ProviderAccountedReaderError>::required_bytes();
        let with_open_publisher = base_reader_bytes
            .checked_add(open_publication_bytes)
            .ok_or(MemoryGrantError::ArithmeticOverflow {
                current_bytes: base_reader_bytes,
                additional_bytes: open_publication_bytes,
            })?;
        // Publish the exact known control-block envelope before its grant can
        // invoke eviction or allocation machinery. This is conservative until
        // construction succeeds and prevents a transient undercount across
        // every fallible grant transition.
        if let Err(error) = self.set_reader_bytes(with_open_publisher) {
            self.reset_reader_preserving_primary(base_reader_bytes);
            return Err(error);
        }
        let publisher_grant = match self.sorter.split_exact_owned_child() {
            Ok(grant) => grant,
            Err(error) => {
                self.reset_reader_preserving_primary(base_reader_bytes);
                return Err(error);
            }
        };
        let failure_publisher =
            match AccountedErrorPublisher::<ProviderAccountedReaderError>::try_new(publisher_grant)
            {
                Ok(publisher) => publisher,
                Err(error) => {
                    let failure = self.sorter.recover_exact_publisher_build_error(error);
                    let primary = ExactOwnedSortStreamError::publisher_build(failure);
                    self.reset_reader_preserving_primary(base_reader_bytes);
                    return Err(primary);
                }
            };
        debug_assert_eq!(failure_publisher.granted_bytes(), open_publication_bytes);

        let reader_grant = match self.sorter.split_exact_owned_child() {
            Ok(grant) => grant,
            Err(error) => {
                self.sorter
                    .merge_exact_publisher_grant(failure_publisher.into_unpublished_grant());
                self.reset_reader_preserving_primary(base_reader_bytes);
                return Err(error);
            }
        };
        let transition = match self.transition_observer() {
            Ok(transition) => transition,
            Err(error) => {
                drop(reader_grant);
                self.sorter
                    .merge_exact_publisher_grant(failure_publisher.into_unpublished_grant());
                self.reset_reader_preserving_primary(base_reader_bytes);
                return Err(error);
            }
        };
        let result = self
            .sorter
            .runs
            .get(run_index)
            .file
            .reader_with_exact_owned_provider_admission(
                qualification,
                reader_grant,
                failure_publisher,
                &transition,
            );
        match result {
            Ok(reader) => {
                let receipt = reader.receipt();
                let Some(target) = base_reader_bytes.checked_add(receipt.bytes()) else {
                    let primary = MemoryGrantError::ArithmeticOverflow {
                        current_bytes: base_reader_bytes,
                        additional_bytes: receipt.bytes(),
                    };
                    self.abort_local_reader_preserving_primary(reader);
                    self.reset_reader_preserving_primary(base_reader_bytes);
                    return Err(primary.into());
                };
                if let Err(error) = self.set_reader_bytes(target) {
                    self.abort_local_reader_preserving_primary(reader);
                    self.reset_reader_preserving_primary(base_reader_bytes);
                    return Err(error);
                }
                Ok(reader)
            }
            Err(error) => {
                self.reset_reader_preserving_primary(base_reader_bytes);
                Err(ExactOwnedSortStreamError::accounted(error))
            }
        }
    }

    fn read_decode_head(
        &mut self,
        reader: ProviderAccountedSpillFileReader,
        receipt: ProviderAccountedReaderReceipt,
        limits: super::file::SpillFrameLimits,
    ) -> Result<(ProviderAccountedSpillFileReader, AccountedOrdinalRow), ExactOwnedSortStreamError>
    {
        let current_reader_bytes = self.reader_bytes.get();
        let Some(base_reader_bytes) = current_reader_bytes.checked_sub(receipt.bytes()) else {
            self.abort_local_reader_preserving_primary(reader);
            self.observer.publish_unrepresentable();
            return Err(MemoryGrantError::AccountingUnderflow {
                account: "exact owned sort reader frontier",
                accounted_bytes: current_reader_bytes,
                release_bytes: receipt.bytes(),
            }
            .into());
        };
        let base_row_bytes = self.row_bytes.get();
        let payload_grant = match self.sorter.take_exact_owned_payload_grant() {
            Ok(grant) => grant,
            Err(error) => {
                self.abort_local_reader_preserving_primary(reader);
                self.reset_frontier_preserving_primary(base_reader_bytes, base_row_bytes);
                return Err(error);
            }
        };
        let transition = match self.transition_observer() {
            Ok(transition) => transition,
            Err(error) => {
                drop(payload_grant);
                self.abort_local_reader_preserving_primary(reader);
                self.reset_frontier_preserving_primary(base_reader_bytes, base_row_bytes);
                return Err(error);
            }
        };
        let (reader, encoded) =
            match reader.read_sort_row_owned_observing(payload_grant, &transition) {
                Ok(result) => result,
                Err(error) => {
                    self.reset_frontier_preserving_primary(base_reader_bytes, base_row_bytes);
                    return Err(ExactOwnedSortStreamError::accounted(error));
                }
            };
        #[cfg(test)]
        let decoder_panic = self
            .sorter
            .exact_decoder_panic
            .lock()
            .expect("exact decoder panic injection mutex is not poisoned")
            .take();
        #[cfg(test)]
        let decoded = match decoder_panic {
            Some(payload) => decode_provider_accounted_sort_row_with(
                encoded,
                self.sorter.row_shape,
                limits,
                Some(&transition),
                move |_, _, _| std::panic::resume_unwind(payload),
            ),
            None => decode_provider_accounted_sort_row(
                encoded,
                self.sorter.row_shape,
                limits,
                Some(&transition),
            ),
        };
        #[cfg(not(test))]
        let decoded = decode_provider_accounted_sort_row(
            encoded,
            self.sorter.row_shape,
            limits,
            Some(&transition),
        );
        let row = match decoded {
            Ok(row) => row,
            Err(error) => {
                self.abort_local_reader_preserving_primary(reader);
                self.reset_frontier_preserving_primary(base_reader_bytes, base_row_bytes);
                return Err(error);
            }
        };
        let Some(expected) = base_row_bytes.checked_add(row.granted_bytes()) else {
            let error = MemoryGrantError::ArithmeticOverflow {
                current_bytes: base_row_bytes,
                additional_bytes: row.granted_bytes(),
            };
            self.reclaim_row_to_frontier_preserving_primary(row);
            self.abort_local_reader_preserving_primary(reader);
            self.reset_frontier_preserving_primary(base_reader_bytes, base_row_bytes);
            return Err(error.into());
        };
        if self.row_bytes.get() != expected {
            self.observer.publish_unrepresentable();
            self.reclaim_row_to_frontier_preserving_primary(row);
            self.abort_local_reader_preserving_primary(reader);
            self.reset_frontier_preserving_primary(base_reader_bytes, base_row_bytes);
            return Err(ExternalSortOperationError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "exact decoded row authority diverged from transition telemetry",
            ))
            .into());
        }
        Ok((reader, row))
    }

    /// Advances before publishing the popped row. A failure reading the next
    /// head therefore cannot expose a prefix while hiding an earlier failure
    /// from the same merge step.
    pub(crate) fn next_owned_row(
        &mut self,
    ) -> Result<Option<AccountedOrdinalRow>, ExactOwnedSortStreamError> {
        let _profile_merge = self
            .sorter
            .0
            .as_ref()
            .map(|sorter| ProfileMergeTimer::new(&sorter.manager));
        match self.state {
            ExactOwnedCursorState::Terminal => return Ok(None),
            ExactOwnedCursorState::Failed | ExactOwnedCursorState::Building => {
                return Err(ExternalSortOperationError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "failed exact owned sort cursor cannot resume",
                ))
                .into());
            }
            ExactOwnedCursorState::Active => {}
        }
        if let Err(error) = self.check_cancellation_raw() {
            self.state = ExactOwnedCursorState::Failed;
            let _ = self.publish_live();
            return Err(error);
        }
        let popped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.heap.pop_exact_owned(&self.sorter.comparator)
        }));
        let Some(entry) = (match popped {
            Ok(Ok(entry)) => entry,
            Ok(Err(error)) => {
                self.state = ExactOwnedCursorState::Failed;
                let _ = self.publish_live();
                return Err(error.into());
            }
            Err(payload) => {
                // The pop guard restored the removed minimum without
                // allocating. The poisoned heap therefore still owns every
                // row and terminal cleanup can reclaim every grant explicitly.
                self.state = ExactOwnedCursorState::Failed;
                let _ = self.publish_live();
                std::panic::resume_unwind(payload);
            }
        }) else {
            self.finish_terminal()
                .map_err(ExactOwnedSortStreamError::terminal)?;
            return Ok(None);
        };
        let returned_bytes = entry.row.granted_bytes();
        // Pop comparisons can grow scratch before the replacement reader's
        // atomic transitions. Publish that owner while the popped row remains
        // in row_bytes, and reclaim the row if reconciliation fails.
        if self.sorter.comparator.is_accounted()
            && let Err(error) = self.publish_live()
        {
            self.reclaim_heap_entry_preserving_primary(entry);
            self.state = ExactOwnedCursorState::Failed;
            return Err(error);
        }
        let advance = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.advance_run(entry.run_index)
        }));
        match advance {
            Ok(Ok(())) => {}
            Ok(Err(primary)) => {
                self.reclaim_heap_entry_preserving_primary(entry);
                self.state = ExactOwnedCursorState::Failed;
                return Err(primary);
            }
            Err(payload) => {
                // `advance_run` may already have installed the replacement
                // head before its comparator unwinds. That head remains in
                // `row_bytes`; only the popped owner is physically destroyed
                // and moved into the retry frontier before the panic resumes.
                self.reclaim_heap_entry_preserving_primary(entry);
                self.state = ExactOwnedCursorState::Failed;
                let _ = self.publish_live();
                std::panic::resume_unwind(payload);
            }
        }
        if let Err(primary) = self.check_cancellation_raw() {
            self.reclaim_heap_entry_preserving_primary(entry);
            self.state = ExactOwnedCursorState::Failed;
            return Err(primary);
        }
        // Advancing may compare a replacement head and grow scratch again.
        // Atomic row transfer must start from the freshly published total.
        if self.sorter.comparator.is_accounted()
            && let Err(error) = self.publish_live()
        {
            self.reclaim_heap_entry_preserving_primary(entry);
            self.state = ExactOwnedCursorState::Failed;
            return Err(error);
        }
        let transition = match self.transition_observer() {
            Ok(transition) => transition,
            Err(error) => {
                self.reclaim_heap_entry_preserving_primary(entry);
                self.state = ExactOwnedCursorState::Failed;
                return Err(error);
            }
        };
        if let Err(error) = transition.transfer_row_to_retained(returned_bytes) {
            self.reclaim_heap_entry_preserving_primary(entry);
            self.state = ExactOwnedCursorState::Failed;
            self.observer.publish_retained_preserving_primary(0);
            return Err(error.into());
        }
        Ok(Some(entry.row))
    }

    fn advance_run(&mut self, run_index: usize) -> Result<(), ExactOwnedSortStreamError> {
        let mut slot = self
            .readers
            .get_mut(run_index)
            .and_then(Option::take)
            .ok_or_else(|| {
                ExternalSortOperationError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "exact owned sort heap referenced an empty reader slot",
                ))
            })?;
        let receipt = slot.receipt;
        if slot.remaining == 0 {
            let current_reader_bytes = self.reader_bytes.get();
            let Some(base_reader_bytes) = current_reader_bytes.checked_sub(receipt.bytes()) else {
                let reader = slot.take_reader();
                self.abort_local_reader_preserving_primary(reader);
                self.observer.publish_unrepresentable();
                return Err(MemoryGrantError::AccountingUnderflow {
                    account: "exact owned sort reader frontier",
                    accounted_bytes: current_reader_bytes,
                    release_bytes: receipt.bytes(),
                }
                .into());
            };
            let reader = slot.take_reader();
            let result = reader.finish_sort_run_owned();
            match result {
                Ok(()) => self.set_reader_bytes(base_reader_bytes)?,
                Err(error) => {
                    let error = self.adopt_reader_resolution_error(error);
                    self.reset_reader_preserving_primary(base_reader_bytes);
                    return Err(error);
                }
            }
            return Ok(());
        }

        let Some(remaining) = slot.remaining.checked_sub(1) else {
            let primary = ExternalSortOperationError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "exact owned sort remaining-row count underflow",
            ));
            let reader = slot.take_reader();
            self.abort_local_reader_preserving_primary(reader);
            match self.reader_bytes.get().checked_sub(receipt.bytes()) {
                Some(base_reader_bytes) => {
                    self.reset_reader_preserving_primary(base_reader_bytes);
                }
                None => self.observer.publish_unrepresentable(),
            }
            return Err(primary.into());
        };
        let reader = slot.take_reader();
        let (reader, row) = self.read_decode_head(reader, receipt, slot.limits)?;
        slot.remaining = remaining;
        slot.restore_reader(reader);
        self.readers[run_index] = Some(slot);
        let push = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.heap.push_exact_owned(
                ExactOwnedHeapEntry { row, run_index },
                &self.sorter.comparator,
            )
        }));
        if let Ok(Err(error)) = push {
            self.state = ExactOwnedCursorState::Failed;
            let _ = self.publish_live();
            return Err(error.into());
        }
        if let Err(payload) = push {
            self.state = ExactOwnedCursorState::Failed;
            let _ = self.publish_live();
            std::panic::resume_unwind(payload);
        }
        Ok(())
    }

    /// Explicit terminal cleanup used for both complete drain and downstream
    /// early stop. Readers/heads and exact containers disappear before file
    /// deletion and before the frontier grant is shrunk.
    pub(crate) fn abort(&mut self) -> Result<(), ExactOwnedTerminalCleanup> {
        if self.state == ExactOwnedCursorState::Terminal {
            return Ok(());
        }
        if let Some(cleanup) = self.finish_resources_preserving_final_publisher() {
            return Err(cleanup);
        }
        self.release_unused_final_publisher()?;
        self.state = ExactOwnedCursorState::Terminal;
        Ok(())
    }

    fn cleanup_readers_exact(
        &mut self,
    ) -> (
        Option<AccountedError>,
        Option<MemoryGrantError>,
        Option<ExactOwnedCleanupPanic>,
    ) {
        let readers = std::mem::replace(&mut self.readers, ExactOwnedReaderSlots::new_in(Global));
        let mut first_error = None;
        let mut first_publication_release = None;
        let mut first_panic = None;
        for mut run_reader in readers.into_iter().flatten() {
            let reader = run_reader.take_reader();
            let outcome =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| reader.abort_owned()));
            match outcome {
                Ok(Ok(())) => {}
                Ok(Err(ProviderAccountedReaderResolutionError::Accounted(error)))
                    if first_error.is_none() =>
                {
                    first_error = Some(error);
                }
                Ok(Err(ProviderAccountedReaderResolutionError::Accounted(error))) => {
                    let _ = super::run_cleanup_backstop(|| {
                        drop(error);
                        Ok::<(), std::convert::Infallible>(())
                    });
                }
                Ok(Err(ProviderAccountedReaderResolutionError::UnusedPublication(error))) => {
                    let (primary, grant) = error.into_parts();
                    self.sorter.merge_exact_publisher_grant(grant);
                    if first_publication_release.is_none() {
                        first_publication_release = Some(primary);
                    }
                }
                Err(payload) => {
                    retain_first_cleanup_panic(&mut first_panic, payload);
                }
            }
        }
        (first_error, first_publication_release, first_panic)
    }

    fn finish_resources_preserving_final_publisher(&mut self) -> Option<ExactOwnedTerminalCleanup> {
        self.reclaim_terminal_heap();
        let pending_reader_cleanup = std::mem::take(&mut self.pending_reader_cleanup);
        let defer_pending_publication = pending_reader_cleanup.defers_frontier_release();
        let (reader_cleanup, reader_publication_release, reader_cleanup_panic) =
            self.cleanup_readers_exact();
        self.reader_bytes.set(0);
        self.row_bytes.set(0);
        self.observer.publish_retained_preserving_primary(0);
        // A physical or grant cleanup failure remains explicitly retryable.
        // Failed cursors cannot emit again, but `abort` and `Drop` continue to
        // drive the retained handles until all cleanup succeeds.
        self.state = ExactOwnedCursorState::Failed;

        let (cleanup, cleanup_panic) = self.sorter.cleanup_runs_exact();
        let (release, release_panic) = self.sorter.release_exact_terminal_resources(
            &self.observer,
            defer_pending_publication || reader_publication_release.is_some(),
        );
        ExactOwnedTerminalCleanup::from_parts(
            pending_reader_cleanup,
            reader_cleanup,
            reader_publication_release,
            reader_cleanup_panic,
            cleanup,
            cleanup_panic,
            release,
            release_panic,
        )
    }

    fn release_unused_final_publisher(&mut self) -> Result<(), ExactOwnedTerminalCleanup> {
        let Some(publisher) = self.sorter.exact_failure_publisher.take() else {
            self.publish_live()
                .map_err(|error| ExactOwnedTerminalCleanup {
                    pending_reader_cleanup: ExactOwnedPendingReaderCleanup::default(),
                    reader_cleanup: None,
                    reader_publication_release: None,
                    reader_cleanup_panic: None,
                    run_cleanup: Some(match error.primary {
                        ExactOwnedSortPrimary::Sort(error) => error,
                        _ => ExternalSortOperationError::Io(std::io::Error::other(
                            "exact terminal observer reconciliation failed",
                        )),
                    }),
                    run_cleanup_panic: None,
                    resource_release: None,
                    resource_release_panic: None,
                })?;
            return Ok(());
        };
        // Destroy the empty physical block first, then move its still-accounted
        // grant back under the sorter's explicit retry authority. The checked
        // sorter total is unchanged across this non-callback handoff.
        self.sorter
            .merge_exact_publisher_grant(publisher.into_unpublished_grant());
        #[cfg(test)]
        let release = match self.sorter.exact_final_publisher_release_error.take() {
            Some(error) => Some(error),
            None => self
                .sorter
                .frontier_grant
                .as_mut()
                .and_then(|grant| grant.try_resize(0).err()),
        };
        #[cfg(not(test))]
        let release = self
            .sorter
            .frontier_grant
            .as_mut()
            .and_then(|grant| grant.try_resize(0).err());
        let publication = self.publish_live().err().map(|error| match error.primary {
            ExactOwnedSortPrimary::Sort(error) => error,
            _ => ExternalSortOperationError::Io(std::io::Error::other(
                "exact terminal observer reconciliation failed",
            )),
        });
        let resource_release = ExactOwnedResourceRelease {
            workspace: None,
            writer: None,
            ordinal: None,
            frontier: release,
            payload: None,
            output: None,
            catalog: None,
            publication,
        };
        if resource_release.is_empty() {
            Ok(())
        } else {
            Err(ExactOwnedTerminalCleanup {
                pending_reader_cleanup: ExactOwnedPendingReaderCleanup::default(),
                reader_cleanup: None,
                reader_publication_release: None,
                reader_cleanup_panic: None,
                run_cleanup: None,
                run_cleanup_panic: None,
                resource_release: Some(resource_release),
                resource_release_panic: None,
            })
        }
    }

    fn finish_terminal(&mut self) -> Result<(), ExactOwnedTerminalCleanup> {
        self.abort()
    }
}

impl Drop for ExactOwnedSortCursor<'_> {
    #[allow(
        clippy::result_large_err,
        reason = "terminal cleanup is a fixed-capacity, allocation-free failure record by design"
    )]
    fn drop(&mut self) {
        if self.sorter.0.is_none() {
            // All fields with resource authority were moved into the owned
            // resumable wrapper; this temporary shell has nothing to release.
            return;
        }
        self.observer.publish_retained_preserving_primary(0);
        let terminal_cleanup_succeeded = if self.state == ExactOwnedCursorState::Terminal {
            true
        } else {
            super::run_cleanup_backstop(|| {
                if let Some(cleanup) = self.finish_resources_preserving_final_publisher() {
                    return Err(cleanup);
                }
                Ok(())
            })
        };
        if let Some(publisher) = self.sorter.exact_failure_publisher.take() {
            self.sorter
                .merge_exact_publisher_grant(publisher.into_unpublished_grant());
        }
        // Run one bounded sorter retry while the prior conservative scalar
        // remains visible. Zero is honest only when both the cursor cleanup
        // and the final owner teardown proved complete; an unreportable Drop
        // failure is published as unrepresentable and its owner is retained
        // fail-closed by `drop_now`.
        if !terminal_cleanup_succeeded {
            quarantine_pull_sort_hook(self.sorter.pull_hook_workspace.as_ref());
        }
        let sorter_cleanup_succeeded = self.sorter.drop_now();
        if terminal_cleanup_succeeded
            && sorter_cleanup_succeeded
            && !self.observer.has_unacknowledged_retained_poison()
        {
            if self.observer.publish(0).is_err() {
                self.observer.publish_unrepresentable();
            }
        } else {
            self.observer.publish_unrepresentable();
        }
    }
}

impl ExternalSortCursor<'_> {
    fn from_state(
        sorter: &mut ExternalSort,
        state: ExternalSortCursorState,
    ) -> ExternalSortCursor<'_> {
        ExternalSortCursor {
            output_rows: state.output_rows,
            heap: state.heap,
            run_readers: state.run_readers,
            memory_iter: state.memory_iter,
            sorter,
            memory_run_index: state.memory_run_index,
            max_chunk_rows: state.max_chunk_rows,
            frontier_base_bytes: state.frontier_base_bytes,
            frontier_reader_bytes: state.frontier_reader_bytes,
            frontier_row_bytes: state.frontier_row_bytes,
            output_base_bytes: state.output_base_bytes,
            output_row_bytes: state.output_row_bytes,
            terminal: false,
            drop_cleaned: false,
        }
    }

    /// Advances the merge and returns one bounded borrowed batch.
    ///
    /// # Errors
    ///
    /// Returns an error for cancellation, deadline expiry, malformed or
    /// unreadable spill data, allocation failure, or explicit spill cleanup
    /// failure. No rows from the failing call are returned.
    pub fn next_chunk(&mut self) -> std::io::Result<Option<ExternalSortChunk<'_>>> {
        let observer = ExternalSortGrantObserver::inert();
        let result = self.next_chunk_inner(&observer);
        let has_rows = result.map_err(ExternalSortOperationError::into_io)?;
        Ok(has_rows.then(|| ExternalSortChunk {
            rows: &self.output_rows,
        }))
    }

    fn abort_inner(
        &mut self,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        self.discard_frontier();
        if self.sorter.scalar_cleanup.is_some() {
            scalar_reader_cleanup_check(self.sorter.scalar_cleanup.as_ref())?;
            self.sorter.cleanup_runs_scalar()?;
            self.sorter.release_cursor_grants(true, observer)?;
            self.terminal = true;
            return Ok(());
        }
        let grant_release = self.sorter.release_cursor_grants(true, observer);
        let cleanup = self.sorter.cleanup_runs();
        let observation = self.sorter.publish_granted_bytes(observer);
        self.terminal = true;
        match (grant_release, cleanup) {
            (Ok(()), Ok(())) => observation,
            (Err(primary), Ok(())) => Err(primary),
            (Ok(()), Err(cleanup)) => Err(cleanup),
            (Err(primary), Err(cleanup)) => Err(with_io_cleanup(
                primary,
                cleanup.into_io(),
                "abandoned sort cursor cleanup",
            )),
        }
    }

    fn next_chunk_inner(
        &mut self,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<bool, ExternalSortOperationError> {
        let _profile_merge = ProfileMergeTimer::new(&self.sorter.manager);
        if self.terminal {
            self.output_rows = Vec::new();
            self.output_row_bytes = 0;
            self.sorter.resize_output_grant(0, observer)?;
            return Ok(false);
        }
        self.output_rows.clear();
        self.output_row_bytes = 0;
        if let Err(primary) = self
            .sorter
            .resize_output_grant(self.output_base_bytes, observer)
        {
            return Err(self.fail(primary, observer));
        }
        let cancellation = self.sorter.cancellation.clone();
        if let Err(error) = check_cancellation(cancellation.as_ref()) {
            return Err(self.fail(ExternalSortOperationError::Cancelled(error), observer));
        }

        while self.output_rows.len() < self.max_chunk_rows {
            if let Err(error) = check_cancellation(cancellation.as_ref()) {
                return Err(self.fail(ExternalSortOperationError::Cancelled(error), observer));
            }
            let entry = match self.heap.pop_entry(&self.sorter.comparator) {
                Ok(Some(entry)) => entry,
                Ok(None) => break,
                Err(error) => return Err(self.fail(error, observer)),
            };
            let run_index = entry.run_index;
            let Some(output_row_bytes) = self.output_row_bytes.checked_add(entry.retained_bytes)
            else {
                return Err(self.fail(
                    ExternalSortOperationError::Memory(MemoryGrantError::ArithmeticOverflow {
                        current_bytes: self.output_row_bytes,
                        additional_bytes: entry.retained_bytes,
                    }),
                    observer,
                ));
            };
            let output_bytes = match checked_workspace_sum(self.output_base_bytes, output_row_bytes)
            {
                Ok(bytes) => bytes,
                Err(error) => {
                    return Err(self.fail(ExternalSortOperationError::Memory(error), observer));
                }
            };
            if let Err(primary) = self.sorter.resize_output_grant(output_bytes, observer) {
                return Err(self.fail(primary, observer));
            }
            self.output_rows.push(entry.row.values);
            self.output_row_bytes = output_row_bytes;
            self.frontier_row_bytes = self
                .frontier_row_bytes
                .checked_sub(entry.retained_bytes)
                .expect("cursor frontier owns every heap-head charge");
            let frontier_bytes = match cursor_frontier_bytes(
                self.frontier_base_bytes,
                self.frontier_reader_bytes,
                self.frontier_row_bytes,
            ) {
                Ok(bytes) => bytes,
                Err(error) => {
                    return Err(self.fail(ExternalSortOperationError::Memory(error), observer));
                }
            };
            if let Err(primary) = self.sorter.resize_frontier_grant(frontier_bytes, observer) {
                return Err(self.fail(primary, observer));
            }

            let next = (|| -> Result<Option<(OrdinalRow, usize)>, ExternalSortOperationError> {
                if run_index == self.memory_run_index {
                    return Ok(self.memory_iter.next().map(|row| (row, 0)));
                }
                let remaining = self
                    .run_readers
                    .get(run_index)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "sort cursor heap referenced an empty run",
                        )
                    })?
                    .remaining;
                if remaining == 0 {
                    let mut reader = self.run_readers[run_index]
                        .take()
                        .expect("validated live cursor reader disappeared");
                    reader.reader.finish()?;
                    let reader_workspace_bytes = reader.reader_workspace_bytes;
                    drop(reader);
                    scalar_reader_cleanup_check(self.sorter.scalar_cleanup.as_ref())?;
                    self.frontier_reader_bytes = self
                        .frontier_reader_bytes
                        .checked_sub(reader_workspace_bytes)
                        .expect("cursor reader owns its retained workspace charge");
                    let frontier_bytes = cursor_frontier_bytes(
                        self.frontier_base_bytes,
                        self.frontier_reader_bytes,
                        self.frontier_row_bytes,
                    )?;
                    self.sorter
                        .resize_frontier_grant(frontier_bytes, observer)?;
                    Ok(None)
                } else {
                    let reader = self.run_readers[run_index]
                        .as_mut()
                        .expect("validated live cursor reader disappeared");
                    let (row, retained_bytes) = self.sorter.read_decode_cursor_head(
                        &mut reader.reader,
                        reader.row_shape,
                        reader.limits,
                        self.frontier_base_bytes,
                        self.frontier_reader_bytes,
                        self.frontier_row_bytes,
                        cancellation.as_ref(),
                        observer,
                    )?;
                    reader.remaining -= 1;
                    self.frontier_row_bytes = self
                        .frontier_row_bytes
                        .checked_add(retained_bytes)
                        .ok_or(MemoryGrantError::ArithmeticOverflow {
                        current_bytes: self.frontier_row_bytes,
                        additional_bytes: retained_bytes,
                    })?;
                    Ok(Some((row, retained_bytes)))
                }
            })();
            let next = match next {
                Ok(next) => next,
                Err(primary) => return Err(self.fail(primary, observer)),
            };
            if let Err(error) = check_cancellation(cancellation.as_ref()) {
                return Err(self.fail(ExternalSortOperationError::Cancelled(error), observer));
            }
            match next {
                Some((row, retained_bytes)) => {
                    if let Err(error) = self.heap.push_entry(
                        HeapEntry {
                            row,
                            run_index,
                            retained_bytes,
                        },
                        &self.sorter.comparator,
                    ) {
                        return Err(self.fail(error, observer));
                    }
                }
                None if run_index != self.memory_run_index => {}
                None => {}
            }
        }

        if self.heap.is_empty()
            && let Err(primary) = self.complete(observer)
        {
            self.output_rows.clear();
            return Err(primary);
        }
        if let Err(error) = check_cancellation(cancellation.as_ref()) {
            return Err(self.fail(ExternalSortOperationError::Cancelled(error), observer));
        }
        if self.output_rows.is_empty() {
            Ok(false)
        } else {
            Ok(true)
        }
    }

    fn discard_frontier(&mut self) {
        self.output_rows = Vec::new();
        self.heap = ComparatorMinHeap::new();
        self.run_readers = Vec::new();
        self.memory_iter = Vec::new().into_iter();
        self.frontier_reader_bytes = 0;
        self.frontier_row_bytes = 0;
        self.output_row_bytes = 0;
    }

    fn complete(
        &mut self,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortOperationError> {
        self.heap = ComparatorMinHeap::new();
        self.run_readers = Vec::new();
        self.memory_iter = Vec::new().into_iter();
        self.frontier_reader_bytes = 0;
        self.frontier_row_bytes = 0;
        if self.sorter.scalar_cleanup.is_some() {
            scalar_reader_cleanup_check(self.sorter.scalar_cleanup.as_ref())?;
            self.sorter.cleanup_runs_scalar()?;
            self.sorter.release_cursor_grants(false, observer)?;
            self.terminal = true;
            return Ok(());
        }
        let grant_release = self.sorter.release_cursor_grants(false, observer);
        let cleanup = self.sorter.cleanup_runs();
        let observation = self.sorter.publish_granted_bytes(observer);
        self.terminal = true;
        match (grant_release, cleanup) {
            (Ok(()), Ok(())) => observation,
            (Err(primary), Ok(())) => Err(primary),
            (Ok(()), Err(cleanup)) => Err(cleanup),
            (Err(primary), Err(cleanup)) => Err(with_io_cleanup(
                primary,
                cleanup.into_io(),
                "spill-read cleanup",
            )),
        }
    }

    fn fail(
        &mut self,
        primary: ExternalSortOperationError,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> ExternalSortOperationError {
        if self.sorter.scalar_cleanup.is_some() {
            return primary;
        }
        self.discard_frontier();
        let mut primary = primary;
        if let Err(release) = self.sorter.release_cursor_grants(true, observer) {
            primary = with_io_cleanup(primary, release.into_io(), "sort cursor grant release");
        }
        let cleanup = self.sorter.cleanup_runs();
        self.sorter
            .publish_granted_bytes_preserving_primary(observer);
        self.terminal = true;
        match cleanup {
            Ok(()) => primary,
            Err(cleanup) => with_io_cleanup(primary, cleanup.into_io(), "spill-read cleanup"),
        }
    }

    fn cleanup_for_drop_with_observer(&mut self, observer: &ExternalSortGrantObserver<'_>) {
        let needs_run_cleanup = !self.terminal;
        self.discard_frontier();
        let _ = super::run_cleanup_backstop(|| self.sorter.release_cursor_grants(true, observer));
        if needs_run_cleanup {
            let _ = self.sorter.cleanup_runs_for_drop();
        }
        self.sorter
            .publish_granted_bytes_preserving_primary(observer);
        self.terminal = true;
        self.drop_cleaned = true;
    }
}

#[cfg(test)]
impl std::fmt::Debug for AccountedExternalSortCursor<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("AccountedExternalSortCursor")
            .field(&self.cursor)
            .finish()
    }
}

#[cfg(test)]
impl AccountedExternalSortCursor<'_> {
    pub(crate) fn next_chunk_accounted_observing(
        &mut self,
    ) -> Result<Option<ExternalSortChunk<'_>>, ExternalSortOperationError> {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.cursor.next_chunk_inner(&self.observer)
        }));
        let has_rows = match outcome {
            Ok(result) => result?,
            Err(panic) => self
                .cursor
                .sorter
                .resume_after_accounted_unwind(&self.observer, panic),
        };
        Ok(has_rows.then(|| ExternalSortChunk {
            rows: &self.cursor.output_rows,
        }))
    }
}

impl ScalarExternalSortCursor<'_> {
    fn fail(
        &mut self,
        transport: ScalarFailureTransport,
        operation: Option<ExternalSortOperationError>,
        operator: Option<OperatorError>,
        panic: Option<Box<dyn std::any::Any + Send>>,
    ) -> OperatorError {
        self.cursor.discard_frontier();
        self.cursor.terminal = true;
        self.cursor
            .sorter
            .scalar_failure(transport, operation, operator, panic, &self.observer)
    }

    pub(crate) fn next_chunk_accounted_observing(
        &mut self,
    ) -> Result<Option<ExternalSortChunk<'_>>, OperatorError> {
        let Some(transport) = self.transport.take() else {
            return Ok(None);
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.cursor.next_chunk_inner(&self.observer)
        }));
        let has_rows = match outcome {
            Ok(Ok(rows)) => rows,
            Ok(Err(error)) => return Err(self.fail(transport, Some(error), None, None)),
            Err(payload) => return Err(self.fail(transport, None, None, Some(payload))),
        };
        self.transport = Some(transport);
        Ok(has_rows.then(|| ExternalSortChunk {
            rows: &self.cursor.output_rows,
        }))
    }

    pub(crate) fn abort_accounted_observing(&mut self) -> Result<(), OperatorError> {
        let Some(transport) = self.transport.take() else {
            return Ok(());
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.cursor.abort_inner(&self.observer)
        }));
        match outcome {
            Ok(Ok(())) => {
                self.transport = Some(transport);
                Ok(())
            }
            Ok(Err(error)) => Err(self.fail(transport, Some(error), None, None)),
            Err(payload) => Err(self.fail(transport, None, None, Some(payload))),
        }
    }

    pub(crate) fn abort_with_primary(&mut self, primary: OperatorError) -> OperatorError {
        match self.transport.take() {
            Some(transport) => self.fail(transport, None, Some(primary), None),
            None => primary,
        }
    }
}

impl Drop for ScalarExternalSortCursor<'_> {
    fn drop(&mut self) {
        let mut clean = false;
        if let Some(transport) = self.transport.take() {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.cursor.abort_inner(&self.observer)
            }));
            match outcome {
                Ok(Ok(())) => {
                    drop(transport);
                    clean = true;
                }
                Ok(Err(error)) => {
                    let diagnostic = self.fail(transport, Some(error), None, None);
                    drop(diagnostic);
                }
                Err(payload) => {
                    let diagnostic = self.fail(transport, None, None, Some(payload));
                    drop(diagnostic);
                }
            }
        }
        self.cursor.drop_cleaned = true;
        // AccountedError deliberately quarantines last-drop authority during
        // unwinding. Keep the shared cleanup block on the surviving sorter so
        // post-catch telemetry includes it and ordinary teardown can retire it.
        if clean && !std::thread::panicking() {
            self.cursor.sorter.writer_workspace.scalar_cleanup = None;
            self.cursor.sorter.scalar_cleanup = None;
        }
        self.cursor.sorter.scalar_failure_publication_bytes = 0;
        self.cursor
            .sorter
            .publish_granted_bytes_preserving_primary(&self.observer);
    }
}

#[cfg(test)]
impl Drop for AccountedExternalSortCursor<'_> {
    fn drop(&mut self) {
        self.cursor.cleanup_for_drop_with_observer(&self.observer);
    }
}

impl Drop for ExternalSortCursor<'_> {
    fn drop(&mut self) {
        if !self.drop_cleaned {
            let observer = ExternalSortGrantObserver::inert();
            self.cleanup_for_drop_with_observer(&observer);
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "accounted multipass merge carries semantic, resource, and publication context"
)]
fn merge_run_slice_to_file_accounted(
    entries: &[ExternalSortRunEntry],
    first_run_index: usize,
    row_count: usize,
    row_shape: SortRowShape,
    comparator: &SemanticRowComparator,
    output: &mut WriterPublication<'_>,
    workspace: &mut IntermediateMergeWorkspace<'_>,
    cancellation: Option<&QueryCancellationToken>,
    observer: &ExternalSortGrantObserver<'_>,
) -> Result<(), ExternalSortOperationError> {
    let mut heap = IntermediateHeap::new();
    let mut readers: Vec<Option<AccountedIntermediateRunReader>> = Vec::new();
    let mut frontier_reader_bytes = 0usize;
    let mut frontier_head_bytes = 0usize;
    let merge_result = (|| -> Result<(), ExternalSortOperationError> {
        let columns = u32::try_from(row_shape.logical_columns).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "sort run column count exceeds u32",
            )
        })?;
        let row_count_u64 = u64::try_from(row_count).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "intermediate sort row count exceeds u64",
            )
        })?;
        check_cancellation(cancellation)?;
        output
            .file_mut()
            .write_sort_run_start(columns, row_count_u64)?;
        check_cancellation(cancellation)?;

        let requested_frontier_base = checked_workspace_sum(
            cursor_capacity_bytes::<AccountedIntermediateHeapEntry<'_>>(entries.len())
                .map_err(ExternalSortOperationError::into_side_effect_free_primary)
                .map_err(ExternalSortPrimary::into_operation)?,
            cursor_capacity_bytes::<Option<AccountedIntermediateRunReader>>(entries.len())
                .map_err(ExternalSortOperationError::into_side_effect_free_primary)
                .map_err(ExternalSortPrimary::into_operation)?,
        )?;
        workspace
            .resize_frontier(requested_frontier_base, output.granted_bytes(), observer)
            .map_err(ExternalSortPrimary::into_operation)?;

        heap.try_reserve_exact(entries.len()).map_err(|error| {
            ExternalSortOperationError::Allocation(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                format!("reserve accounted intermediate sort merge heap: {error}"),
            ))
        })?;
        readers.try_reserve_exact(entries.len()).map_err(|error| {
            ExternalSortOperationError::Allocation(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                format!("reserve accounted intermediate sort readers: {error}"),
            ))
        })?;
        let frontier_base_bytes = checked_workspace_sum(
            cursor_capacity_bytes::<AccountedIntermediateHeapEntry<'_>>(heap.capacity())?,
            cursor_capacity_bytes::<Option<AccountedIntermediateRunReader>>(readers.capacity())?,
        )?;
        workspace
            .resize_frontier(frontier_base_bytes, output.granted_bytes(), observer)
            .map_err(ExternalSortPrimary::into_operation)?;

        for (local_index, entry) in entries.iter().enumerate() {
            check_cancellation(cancellation)?;
            let total_without_frontier = checked_workspace_sum(
                checked_workspace_sum(
                    workspace.stable_with_comparator_bytes()?,
                    output.granted_bytes(),
                )?,
                workspace.payload_granted_bytes(),
            )?;
            let (mut reader, reader_workspace_bytes) = open_cursor_reader(
                &entry.file,
                workspace.frontier_grant,
                total_without_frontier,
                frontier_base_bytes,
                frontier_reader_bytes,
                frontier_head_bytes,
                observer,
                workspace.retain_failure,
                workspace.scalar_cleanup.as_ref(),
            )?;
            frontier_reader_bytes = frontier_reader_bytes
                .checked_add(reader_workspace_bytes)
                .ok_or(MemoryGrantError::ArithmeticOverflow {
                    current_bytes: frontier_reader_bytes,
                    additional_bytes: reader_workspace_bytes,
                })?;
            let (declared_columns, declared_rows) = reader.read_sort_run_start()?;
            if declared_columns != columns {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("sort run declares {declared_columns} columns, expected {columns}"),
                )
                .into());
            }
            let tracked_rows = u64::try_from(entry.rows).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "tracked sort row count exceeds u64",
                )
            })?;
            if declared_rows != tracked_rows {
                let run_index = first_run_index.saturating_add(local_index);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "sort run {run_index} declares {declared_rows} rows, but manager tracked {tracked_rows}"
                    ),
                )
                .into());
            }
            check_cancellation(cancellation)?;

            if declared_rows == 0 {
                reader.finish()?;
                drop(reader);
                scalar_reader_cleanup_check(workspace.scalar_cleanup.as_ref())?;
                frontier_reader_bytes = frontier_reader_bytes
                    .checked_sub(reader_workspace_bytes)
                    .expect("new intermediate reader owns its workspace charge");
                let frontier_bytes = cursor_frontier_bytes(
                    frontier_base_bytes,
                    frontier_reader_bytes,
                    frontier_head_bytes,
                )?;
                workspace
                    .resize_frontier(frontier_bytes, output.granted_bytes(), observer)
                    .map_err(ExternalSortPrimary::into_operation)?;
                readers.push(None);
            } else {
                let (row, payload, retained_bytes) = read_accounted_intermediate_head(
                    &mut reader,
                    row_shape,
                    entry.file.limits(),
                    frontier_base_bytes,
                    frontier_reader_bytes,
                    frontier_head_bytes,
                    workspace,
                    output.granted_bytes(),
                    cancellation,
                    observer,
                )
                .map_err(ExternalSortPrimary::into_operation)?;
                frontier_head_bytes = frontier_head_bytes.checked_add(retained_bytes).ok_or(
                    MemoryGrantError::ArithmeticOverflow {
                        current_bytes: frontier_head_bytes,
                        additional_bytes: retained_bytes,
                    },
                )?;
                readers.push(Some(AccountedIntermediateRunReader {
                    reader,
                    remaining: declared_rows - 1,
                    row_shape,
                    limits: entry.file.limits(),
                    reader_workspace_bytes,
                }));
                heap.push(AccountedIntermediateHeapEntry {
                    row,
                    payload,
                    run_index: local_index,
                    retained_bytes,
                    comparator,
                })?;
            }
            check_cancellation(cancellation)?;
        }

        let mut emitted = 0usize;
        loop {
            check_cancellation(cancellation)?;
            let Some(entry) = heap.pop()? else {
                break;
            };
            let AccountedIntermediateHeapEntry {
                row,
                payload,
                run_index,
                retained_bytes,
                comparator: _,
            } = entry;
            let other_granted_bytes = checked_workspace_sum(
                workspace.stable_with_comparator_bytes()?,
                workspace.granted_bytes()?,
            )?;
            output
                .write_sort_row_accounted(&payload, other_granted_bytes, observer)
                .map_err(ExternalSortPrimary::into_operation)?;
            emitted = emitted.checked_add(1).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "intermediate sort output count exceeds the platform address space",
                )
            })?;
            check_cancellation(cancellation)?;
            drop(row);
            drop(payload);
            frontier_head_bytes = frontier_head_bytes
                .checked_sub(retained_bytes)
                .expect("intermediate heap head owns its retained charge");
            let frontier_bytes = cursor_frontier_bytes(
                frontier_base_bytes,
                frontier_reader_bytes,
                frontier_head_bytes,
            )?;
            workspace
                .resize_frontier(frontier_bytes, output.granted_bytes(), observer)
                .map_err(ExternalSortPrimary::into_operation)?;

            let remaining = readers[run_index]
                .as_ref()
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "intermediate heap referenced an empty sort run",
                    )
                })?
                .remaining;
            if remaining == 0 {
                let mut reader = readers[run_index]
                    .take()
                    .expect("validated intermediate reader disappeared");
                reader.reader.finish()?;
                let reader_workspace_bytes = reader.reader_workspace_bytes;
                drop(reader);
                scalar_reader_cleanup_check(workspace.scalar_cleanup.as_ref())?;
                frontier_reader_bytes = frontier_reader_bytes
                    .checked_sub(reader_workspace_bytes)
                    .expect("intermediate reader owns its retained workspace charge");
                let frontier_bytes = cursor_frontier_bytes(
                    frontier_base_bytes,
                    frontier_reader_bytes,
                    frontier_head_bytes,
                )?;
                workspace
                    .resize_frontier(frontier_bytes, output.granted_bytes(), observer)
                    .map_err(ExternalSortPrimary::into_operation)?;
            } else {
                let reader = readers[run_index]
                    .as_mut()
                    .expect("validated intermediate reader disappeared");
                let (row, payload, retained_bytes) = read_accounted_intermediate_head(
                    &mut reader.reader,
                    reader.row_shape,
                    reader.limits,
                    frontier_base_bytes,
                    frontier_reader_bytes,
                    frontier_head_bytes,
                    workspace,
                    output.granted_bytes(),
                    cancellation,
                    observer,
                )
                .map_err(ExternalSortPrimary::into_operation)?;
                reader.remaining -= 1;
                frontier_head_bytes = frontier_head_bytes.checked_add(retained_bytes).ok_or(
                    MemoryGrantError::ArithmeticOverflow {
                        current_bytes: frontier_head_bytes,
                        additional_bytes: retained_bytes,
                    },
                )?;
                heap.push(AccountedIntermediateHeapEntry {
                    row,
                    payload,
                    run_index,
                    retained_bytes,
                    comparator,
                })?;
            }
            check_cancellation(cancellation)?;
        }

        if emitted != row_count {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("intermediate sort emitted {emitted} rows, expected {row_count}"),
            )
            .into());
        }
        check_cancellation(cancellation)?;
        Ok(())
    })();

    drop(heap);
    drop(readers);
    let merge_result =
        merge_result.and_then(|()| scalar_reader_cleanup_check(workspace.scalar_cleanup.as_ref()));
    workspace.failed = merge_result.is_err() && workspace.retain_failure;
    let release = if workspace.failed {
        Ok(())
    } else {
        workspace.release_all(output.granted_bytes(), observer)
    };
    match (merge_result, release) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(primary), Ok(())) => Err(primary),
        (Ok(()), Err(release)) => Err(release.into_operation()),
        (Err(primary), Err(release)) => Err(with_io_cleanup(
            primary,
            release.into_io(),
            "intermediate merge workspace release",
        )),
    }
}

/// Merges one bounded, consecutive run slice directly into a new framed run.
///
/// The original plaintext row payload is forwarded after its decoded value is
/// used for comparison. This keeps codec work bounded by one payload and one
/// decoded head per reader and avoids an intermediate `Vec` or lossy re-encode.
fn merge_run_slice_to_file(
    entries: &[ExternalSortRunEntry],
    first_run_index: usize,
    row_count: usize,
    row_shape: SortRowShape,
    comparator: &SemanticRowComparator,
    output: &mut SpillFile,
    cancellation: Option<&QueryCancellationToken>,
) -> Result<(), ExternalSortPrimary> {
    check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;
    let columns = u32::try_from(row_shape.logical_columns).map_err(|_| {
        ExternalSortPrimary::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "sort run column count exceeds u32",
        ))
    })?;
    let row_count_u64 = u64::try_from(row_count).map_err(|_| {
        ExternalSortPrimary::Io(std::io::Error::new(
            std::io::ErrorKind::OutOfMemory,
            "intermediate sort row count exceeds u64",
        ))
    })?;
    output
        .write_sort_run_start(columns, row_count_u64)
        .map_err(ExternalSortPrimary::Io)?;
    check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;

    let mut heap = IntermediateHeap::new();
    heap.try_reserve(entries.len()).map_err(|error| {
        ExternalSortPrimary::Io(std::io::Error::new(
            std::io::ErrorKind::OutOfMemory,
            format!("reserve intermediate sort merge heap: {error}"),
        ))
    })?;
    let mut readers: Vec<Option<RunReader>> = Vec::new();
    readers.try_reserve_exact(entries.len()).map_err(|error| {
        ExternalSortPrimary::Io(std::io::Error::new(
            std::io::ErrorKind::OutOfMemory,
            format!("reserve intermediate sort run readers: {error}"),
        ))
    })?;

    for (local_index, entry) in entries.iter().enumerate() {
        check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;
        let mut reader = entry.file.reader().map_err(ExternalSortPrimary::Io)?;
        let (declared_columns, declared_rows) = reader
            .read_sort_run_start()
            .map_err(ExternalSortPrimary::Io)?;
        if declared_columns != columns {
            return Err(ExternalSortPrimary::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("sort run declares {declared_columns} columns, expected {columns}"),
            )));
        }
        let tracked_rows = u64::try_from(entry.rows).map_err(|_| {
            ExternalSortPrimary::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "tracked sort row count exceeds u64",
            ))
        })?;
        if declared_rows != tracked_rows {
            let run_index = first_run_index.saturating_add(local_index);
            return Err(ExternalSortPrimary::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "sort run {run_index} declares {declared_rows} rows, but manager tracked {tracked_rows}"
                ),
            )));
        }
        check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;

        if declared_rows == 0 {
            reader.finish().map_err(ExternalSortPrimary::Io)?;
            readers.push(None);
        } else {
            let payload = reader.read_sort_row().map_err(ExternalSortPrimary::Io)?;
            let row = decode_row_payload_with_shape(&payload, row_shape, entry.file.limits())
                .map_err(ExternalSortPrimary::Io)?;
            check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;
            readers.push(Some(RunReader {
                reader,
                remaining: declared_rows - 1,
                row_shape,
                limits: entry.file.limits(),
                finished: false,
            }));
            heap.push(IntermediateHeapEntry {
                row,
                payload,
                run_index: local_index,
                comparator,
            })
            .map_err(ExternalSortOperationError::into_side_effect_free_primary)?;
        }
        check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;
    }

    let mut emitted = 0usize;
    loop {
        check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;
        let Some(entry) = heap
            .pop()
            .map_err(ExternalSortOperationError::into_side_effect_free_primary)?
        else {
            break;
        };
        output
            .write_sort_row(&entry.payload)
            .map_err(ExternalSortPrimary::Io)?;
        emitted = emitted.checked_add(1).ok_or_else(|| {
            ExternalSortPrimary::Io(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "intermediate sort output count exceeds the platform address space",
            ))
        })?;
        check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;

        let run_index = entry.run_index;
        let next = readers[run_index]
            .as_mut()
            .ok_or_else(|| {
                ExternalSortPrimary::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "intermediate heap referenced an empty sort run",
                ))
            })?
            .next_record()
            .map_err(ExternalSortPrimary::Io)?;
        check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;
        match next {
            Some(next) => heap
                .push(IntermediateHeapEntry {
                    row: next.row,
                    payload: next.payload,
                    run_index,
                    comparator,
                })
                .map_err(ExternalSortOperationError::into_side_effect_free_primary)?,
            None => readers[run_index] = None,
        }
        check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;
    }

    if emitted != row_count {
        return Err(ExternalSortPrimary::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("intermediate sort emitted {emitted} rows, expected {row_count}"),
        )));
    }
    check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;
    Ok(())
}

/// Helper struct for reading from a run.
struct RunReader {
    reader: SpillFileReader,
    remaining: u64,
    row_shape: SortRowShape,
    limits: super::file::SpillFrameLimits,
    finished: bool,
}

/// One final-cursor reader and the query-memory workspace retained for its
/// entire provider/file lifetime.
struct CursorRunReader {
    reader: SpillFileReader,
    remaining: u64,
    row_shape: SortRowShape,
    limits: super::file::SpillFrameLimits,
    reader_workspace_bytes: usize,
}

struct IntermediateMergeWorkspace<'a> {
    retain_failure: bool,
    scalar_cleanup: Option<AccountedError>,
    failed: bool,
    comparator: &'a SemanticRowComparator,
    frontier_grant: &'a mut Option<MemoryGrant>,
    payload_grant: &'a mut Option<MemoryGrant>,
    stable_granted_bytes: usize,
}

impl IntermediateMergeWorkspace<'_> {
    fn stable_with_comparator_bytes(&self) -> Result<usize, MemoryGrantError> {
        checked_workspace_sum(
            self.stable_granted_bytes,
            self.comparator.checked_granted_bytes()?,
        )
    }
    fn frontier_granted_bytes(&self) -> usize {
        self.frontier_grant.as_ref().map_or(0, MemoryGrant::size)
    }

    fn payload_granted_bytes(&self) -> usize {
        self.payload_grant.as_ref().map_or(0, MemoryGrant::size)
    }

    fn granted_bytes(&self) -> Result<usize, MemoryGrantError> {
        checked_workspace_sum(self.frontier_granted_bytes(), self.payload_granted_bytes())
    }

    fn observed_total(&self, writer_granted_bytes: usize) -> Result<usize, MemoryGrantError> {
        checked_workspace_sum(
            checked_workspace_sum(
                checked_workspace_sum(
                    self.stable_granted_bytes,
                    self.comparator.checked_granted_bytes()?,
                )?,
                writer_granted_bytes,
            )?,
            self.granted_bytes()?,
        )
    }

    fn publish_total(
        &self,
        writer_granted_bytes: usize,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortPrimary> {
        let total = match self.observed_total(writer_granted_bytes) {
            Ok(total) => total,
            Err(error) => {
                observer.publish_unrepresentable();
                return Err(ExternalSortPrimary::Memory(error));
            }
        };
        observer.publish(total).map_err(ExternalSortPrimary::Memory)
    }

    fn resize_frontier(
        &mut self,
        bytes: usize,
        writer_granted_bytes: usize,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortPrimary> {
        let result = self
            .frontier_grant
            .as_mut()
            .map_or(Ok(()), |grant| grant.try_resize(bytes));
        let observation = self.publish_total(writer_granted_bytes, observer);
        match result {
            Ok(()) => observation,
            Err(error) => Err(ExternalSortPrimary::Memory(error)),
        }
    }

    fn resize_payload(
        &mut self,
        bytes: usize,
        writer_granted_bytes: usize,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortPrimary> {
        let bytes = bytes.max(self.payload_granted_bytes());
        let result = self
            .payload_grant
            .as_mut()
            .map_or(Ok(()), |grant| grant.try_resize(bytes));
        let observation = self.publish_total(writer_granted_bytes, observer);
        match result {
            Ok(()) => observation,
            Err(error) => Err(ExternalSortPrimary::Memory(error)),
        }
    }

    fn release_payload(
        &mut self,
        writer_granted_bytes: usize,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortPrimary> {
        let result = self
            .payload_grant
            .as_mut()
            .map_or(Ok(()), |grant| grant.try_resize(0));
        let observation = self.publish_total(writer_granted_bytes, observer);
        match result {
            Ok(()) => observation,
            Err(error) => Err(ExternalSortPrimary::Memory(error)),
        }
    }

    fn release_all(
        &mut self,
        writer_granted_bytes: usize,
        observer: &ExternalSortGrantObserver<'_>,
    ) -> Result<(), ExternalSortPrimary> {
        let mut first = None;
        for grant in [&mut *self.frontier_grant, &mut *self.payload_grant] {
            if let Some(grant) = grant.as_mut()
                && let Err(error) = grant.try_resize(0)
                && first.is_none()
            {
                first = Some(error);
            }
        }
        let observation = self.publish_total(writer_granted_bytes, observer);
        match first {
            Some(error) => Err(ExternalSortPrimary::Memory(error)),
            None => observation,
        }
    }
}

impl Drop for IntermediateMergeWorkspace<'_> {
    fn drop(&mut self) {
        if self.failed {
            return;
        }
        if self.retain_failure && std::thread::panicking() {
            if self.scalar_cleanup.is_some() {
                return;
            }
            for grant in [&mut *self.frontier_grant, &mut *self.payload_grant] {
                if let Some(grant) = grant.take() {
                    std::mem::forget(grant);
                }
            }
            return;
        }
        if let Some(grant) = self.frontier_grant.as_mut() {
            let _ = super::run_cleanup_backstop(|| grant.try_resize(0));
        }
        if let Some(grant) = self.payload_grant.as_mut() {
            let _ = super::run_cleanup_backstop(|| grant.try_resize(0));
        }
    }
}

struct AccountedIntermediateRunReader {
    reader: SpillFileReader,
    remaining: u64,
    row_shape: SortRowShape,
    limits: super::file::SpillFrameLimits,
    reader_workspace_bytes: usize,
}

trait FallibleIntermediateEntry {
    fn try_compare(&self, other: &Self) -> Result<Ordering, ExternalSortOperationError>;
}
struct IntermediateHeap<T>(ComparatorMinHeap<T>);
impl<T: FallibleIntermediateEntry> IntermediateHeap<T> {
    fn new() -> Self {
        Self(ComparatorMinHeap::new())
    }
    fn try_reserve_exact(&mut self, count: usize) -> Result<(), ExternalSortOperationError> {
        self.0 = ComparatorMinHeap::try_with_exact_capacity(count)
            .map_err(ComparatorMinHeapAllocationError::into_operation)?;
        Ok(())
    }
    fn try_reserve(&mut self, count: usize) -> Result<(), ExternalSortOperationError> {
        self.try_reserve_exact(count)
    }
    fn capacity(&self) -> usize {
        self.0.capacity()
    }
    fn push(&mut self, entry: T) -> Result<(), ExternalSortOperationError> {
        self.0
            .try_push_by(entry, &|left, right| left.try_compare(right))
    }
    fn pop(&mut self) -> Result<Option<T>, ExternalSortOperationError> {
        self.0.try_pop_by(&|left, right| left.try_compare(right))
    }
}
impl FallibleIntermediateEntry for AccountedIntermediateHeapEntry<'_> {
    fn try_compare(&self, other: &Self) -> Result<Ordering, ExternalSortOperationError> {
        try_compare_ordinal_rows(&self.row, &other.row, self.comparator)
    }
}
impl FallibleIntermediateEntry for IntermediateHeapEntry<'_> {
    fn try_compare(&self, other: &Self) -> Result<Ordering, ExternalSortOperationError> {
        try_compare_ordinal_rows(&self.row, &other.row, self.comparator)
    }
}

struct AccountedIntermediateHeapEntry<'a> {
    row: OrdinalRow,
    payload: Vec<u8>,
    run_index: usize,
    retained_bytes: usize,
    comparator: &'a SemanticRowComparator,
}

#[cfg(test)]
impl Eq for AccountedIntermediateHeapEntry<'_> {}

#[cfg(test)]
impl PartialEq for AccountedIntermediateHeapEntry<'_> {
    fn eq(&self, other: &Self) -> bool {
        try_compare_ordinal_rows(&self.row, &other.row, self.comparator).expect("test comparator")
            == Ordering::Equal
    }
}

#[cfg(test)]
impl Ord for AccountedIntermediateHeapEntry<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        try_compare_ordinal_rows(&other.row, &self.row, self.comparator).expect("test comparator")
    }
}

#[cfg(test)]
impl PartialOrd for AccountedIntermediateHeapEntry<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "an intermediate head atomically transfers payload, decode, reader, and grant state"
)]
fn read_accounted_intermediate_head(
    reader: &mut SpillFileReader,
    row_shape: SortRowShape,
    limits: super::file::SpillFrameLimits,
    frontier_base_bytes: usize,
    frontier_reader_bytes: usize,
    frontier_head_bytes: usize,
    workspace: &mut IntermediateMergeWorkspace<'_>,
    writer_granted_bytes: usize,
    cancellation: Option<&QueryCancellationToken>,
    observer: &ExternalSortGrantObserver<'_>,
) -> Result<(OrdinalRow, Vec<u8>, usize), ExternalSortPrimary> {
    let mut admission_failure = None;
    let payload = reader.read_sort_row_with_admission(|required| {
        match workspace.resize_payload(required, writer_granted_bytes, observer) {
            Ok(()) => Ok(()),
            Err(ExternalSortPrimary::Memory(error)) => {
                admission_failure = Some(error);
                Err(std::io::Error::from(std::io::ErrorKind::OutOfMemory))
            }
            Err(_) => unreachable!("payload grant resize only returns memory failures"),
        }
    });
    let payload = match (payload, admission_failure) {
        (_, Some(error)) => return Err(ExternalSortPrimary::Memory(error)),
        (Ok(payload), None) => payload,
        (Err(error), None) => return Err(ExternalSortPrimary::Io(error)),
    };
    check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;
    let decoded_bytes = conservative_decoded_row_retained_bytes(payload.len())
        .map_err(ExternalSortPrimary::Memory)?;
    let retained_bytes = payload
        .capacity()
        .checked_add(decoded_bytes)
        .ok_or_else(|| {
            ExternalSortPrimary::Memory(MemoryGrantError::ArithmeticOverflow {
                current_bytes: payload.capacity(),
                additional_bytes: decoded_bytes,
            })
        })?;
    let frontier_bytes = cursor_frontier_bytes(
        frontier_base_bytes,
        frontier_reader_bytes,
        checked_workspace_sum(frontier_head_bytes, retained_bytes)
            .map_err(ExternalSortPrimary::Memory)?,
    )
    .map_err(ExternalSortPrimary::Memory)?;
    workspace.resize_frontier(frontier_bytes, writer_granted_bytes, observer)?;
    let row = decode_row_payload_with_shape(&payload, row_shape, limits)
        .map_err(ExternalSortPrimary::Io)?;
    check_cancellation(cancellation).map_err(ExternalSortPrimary::Cancelled)?;
    workspace.release_payload(writer_granted_bytes, observer)?;
    Ok((row, payload, retained_bytes))
}

impl RunReader {
    fn next_record(&mut self) -> std::io::Result<Option<IntermediateRunRecord>> {
        if self.remaining == 0 {
            if !self.finished {
                self.reader.finish()?;
                self.finished = true;
            }
            return Ok(None);
        }

        let payload = self.reader.read_sort_row()?;
        let row = decode_row_payload_with_shape(&payload, self.row_shape, self.limits)?;
        self.remaining -= 1;
        Ok(Some(IntermediateRunRecord { row, payload }))
    }

    fn next_row(&mut self) -> std::io::Result<Option<OrdinalRow>> {
        self.next_record()
            .map(|record| record.map(|record| record.row))
    }
}

struct IntermediateRunRecord {
    row: OrdinalRow,
    payload: Vec<u8>,
}

struct IntermediateHeapEntry<'a> {
    row: OrdinalRow,
    payload: Vec<u8>,
    run_index: usize,
    comparator: &'a SemanticRowComparator,
}

#[cfg(test)]
impl Eq for IntermediateHeapEntry<'_> {}

#[cfg(test)]
impl PartialEq for IntermediateHeapEntry<'_> {
    fn eq(&self, other: &Self) -> bool {
        try_compare_ordinal_rows(&self.row, &other.row, self.comparator).expect("test comparator")
            == Ordering::Equal
    }
}

#[cfg(test)]
impl Ord for IntermediateHeapEntry<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        try_compare_ordinal_rows(&other.row, &self.row, self.comparator).expect("test comparator")
    }
}

#[cfg(test)]
impl PartialOrd for IntermediateHeapEntry<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Pinned allocator-backed heap storage whose requested layout is its reported
/// capacity. Qualified callers admit that exact layout before construction.
type ComparatorHeapEntries<T> = ExactVec<T, Global>;

#[derive(Debug, Clone, Copy)]
enum ComparatorMinHeapAllocationError {
    Allocation,
    CapacityContract,
}

impl ComparatorMinHeapAllocationError {
    fn into_operation(self) -> ExternalSortOperationError {
        match self {
            Self::Allocation => ExternalSortOperationError::Allocation(std::io::Error::from(
                std::io::ErrorKind::OutOfMemory,
            )),
            Self::CapacityContract => ExternalSortOperationError::Io(std::io::Error::from(
                std::io::ErrorKind::InvalidData,
            )),
        }
    }
}

/// Fixed-capacity min-heap whose ordering is borrowed for each operation.
///
/// Sifting swaps fully initialized entries rather than leaving a temporary
/// hole, so an unwinding comparator cannot make an entry unreachable or
/// suppress its destructor. Pushes never reserve: callers must construct the
/// exact maximum capacity before publishing the heap.
struct ComparatorMinHeap<T> {
    entries: ComparatorHeapEntries<T>,
    poisoned: bool,
}

/// Leaves its heap poisoned during unwinding and clears the poison only after
/// a complete sift has restored the ordering invariant.
#[must_use]
struct ComparatorMinHeapOperation<'a> {
    poisoned: &'a mut bool,
}

impl<'a> ComparatorMinHeapOperation<'a> {
    fn begin(poisoned: &'a mut bool) -> Self {
        *poisoned = true;
        Self { poisoned }
    }

    fn finish(self) {
        *self.poisoned = false;
    }
}

/// Restores a removed minimum if comparator code unwinds during sift-down.
///
/// The heap is already one entry below its admitted capacity, so the guard's
/// `push` cannot allocate. Ordering may remain poisoned, but every move-only
/// row owner stays reachable for explicit terminal reclamation.
struct ComparatorMinHeapPoppedEntry<'a, T> {
    entries: &'a mut ComparatorHeapEntries<T>,
    minimum: Option<T>,
}

impl<'a, T> ComparatorMinHeapPoppedEntry<'a, T> {
    fn new(entries: &'a mut ComparatorHeapEntries<T>, minimum: T) -> Self {
        Self {
            entries,
            minimum: Some(minimum),
        }
    }

    fn entries(&mut self) -> &mut ComparatorHeapEntries<T> {
        self.entries
    }

    fn finish(mut self) -> T {
        self.minimum
            .take()
            .expect("live heap pop guard retains its removed minimum")
    }
}

impl<T> Drop for ComparatorMinHeapPoppedEntry<'_, T> {
    fn drop(&mut self) {
        if let Some(minimum) = self.minimum.take() {
            self.entries.push(minimum);
        }
    }
}

impl<T> ComparatorMinHeap<T> {
    fn new() -> Self {
        Self {
            entries: ComparatorHeapEntries::new_in(Global),
            poisoned: false,
        }
    }

    fn try_with_exact_capacity(capacity: usize) -> Result<Self, ComparatorMinHeapAllocationError> {
        let mut entries = ComparatorHeapEntries::new_in(Global);
        if entries.try_reserve_exact(capacity).is_err() {
            drop(entries);
            return Err(ComparatorMinHeapAllocationError::Allocation);
        }
        if entries.capacity() != capacity {
            // Fence a violated pinned-allocator contract before the heap can
            // publish an entry or report an under-accounted layout.
            drop(entries);
            return Err(ComparatorMinHeapAllocationError::CapacityContract);
        }
        Ok(Self {
            entries,
            poisoned: false,
        })
    }

    fn push_by(&mut self, entry: T, compare: &impl Fn(&T, &T) -> Ordering) {
        assert!(
            !self.poisoned,
            "comparator heap is poisoned by an earlier comparator panic"
        );
        assert!(
            self.entries.len() < self.entries.capacity(),
            "comparator heap push requires pre-admitted capacity"
        );
        let Self { entries, poisoned } = self;
        let operation = ComparatorMinHeapOperation::begin(poisoned);
        entries.push(entry);
        let mut index = entries.len() - 1;
        while index > 0 {
            let parent = (index - 1) / 2;
            if compare(&entries[index], &entries[parent]) != Ordering::Less {
                break;
            }
            entries.swap(index, parent);
            index = parent;
        }
        operation.finish();
    }

    fn pop_by(&mut self, compare: &impl Fn(&T, &T) -> Ordering) -> Option<T> {
        assert!(
            !self.poisoned,
            "comparator heap is poisoned by an earlier comparator panic"
        );
        if self.entries.is_empty() {
            return None;
        }
        let Self { entries, poisoned } = self;
        let operation = ComparatorMinHeapOperation::begin(poisoned);
        let minimum = entries.swap_remove(0);
        let mut popped = ComparatorMinHeapPoppedEntry::new(entries, minimum);
        let entries = popped.entries();
        let mut index = 0usize;
        loop {
            let Some(left) = index.checked_mul(2).and_then(|index| index.checked_add(1)) else {
                break;
            };
            if left >= entries.len() {
                break;
            }
            let right = left + 1;
            let child = if right < entries.len()
                && compare(&entries[right], &entries[left]) == Ordering::Less
            {
                right
            } else {
                left
            };
            if compare(&entries[child], &entries[index]) != Ordering::Less {
                break;
            }
            entries.swap(index, child);
            index = child;
        }
        let minimum = popped.finish();
        operation.finish();
        Some(minimum)
    }
    fn try_push_by(
        &mut self,
        entry: T,
        compare: &impl Fn(&T, &T) -> Result<Ordering, ExternalSortOperationError>,
    ) -> Result<(), ExternalSortOperationError> {
        if self.poisoned {
            return Err(ExternalSortOperationError::Io(std::io::Error::from(
                std::io::ErrorKind::InvalidData,
            )));
        }
        if self.entries.len() >= self.entries.capacity() {
            return Err(ExternalSortOperationError::Io(std::io::Error::from(
                std::io::ErrorKind::InvalidData,
            )));
        }
        let Self { entries, poisoned } = self;
        let operation = ComparatorMinHeapOperation::begin(poisoned);
        entries.push(entry);
        let mut index = entries.len() - 1;
        while index > 0 {
            let parent = (index - 1) / 2;
            if compare(&entries[index], &entries[parent])? != Ordering::Less {
                break;
            }
            entries.swap(index, parent);
            index = parent;
        }
        operation.finish();
        Ok(())
    }

    fn try_pop_by(
        &mut self,
        compare: &impl Fn(&T, &T) -> Result<Ordering, ExternalSortOperationError>,
    ) -> Result<Option<T>, ExternalSortOperationError> {
        if self.poisoned {
            return Err(ExternalSortOperationError::Io(std::io::Error::from(
                std::io::ErrorKind::InvalidData,
            )));
        }
        if self.entries.is_empty() {
            return Ok(None);
        }
        let Self { entries, poisoned } = self;
        let operation = ComparatorMinHeapOperation::begin(poisoned);
        let minimum = entries.swap_remove(0);
        let mut popped = ComparatorMinHeapPoppedEntry::new(entries, minimum);
        let entries = popped.entries();
        let mut index = 0usize;
        loop {
            let Some(left) = index.checked_mul(2).and_then(|index| index.checked_add(1)) else {
                break;
            };
            if left >= entries.len() {
                break;
            }
            let right = left + 1;
            let child = if right < entries.len()
                && compare(&entries[right], &entries[left])? == Ordering::Less
            {
                right
            } else {
                left
            };
            if compare(&entries[child], &entries[index])? != Ordering::Less {
                break;
            }
            entries.swap(index, child);
            index = child;
        }
        let minimum = popped.finish();
        operation.finish();
        Ok(Some(minimum))
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn capacity(&self) -> usize {
        self.entries.capacity()
    }

    #[cfg(test)]
    fn pointer(&self) -> *const T {
        self.entries.as_ptr()
    }
}

/// Comparator-free entry in a final merge heap.
struct HeapEntry {
    row: OrdinalRow,
    run_index: usize,
    retained_bytes: usize,
}

impl ComparatorMinHeap<HeapEntry> {
    fn push_entry(
        &mut self,
        entry: HeapEntry,
        comparator: &SemanticRowComparator,
    ) -> Result<(), ExternalSortOperationError> {
        if let Comparison::Infallible(compare) = &comparator.compare {
            self.push_by(entry, &|left, right| {
                compare_ordinal_rows(&left.row, &right.row, compare.as_ref())
            });
            Ok(())
        } else {
            self.try_push_by(entry, &|left, right| {
                try_compare_ordinal_rows(&left.row, &right.row, comparator)
            })
        }
    }
    fn pop_entry(
        &mut self,
        comparator: &SemanticRowComparator,
    ) -> Result<Option<HeapEntry>, ExternalSortOperationError> {
        if let Comparison::Infallible(compare) = &comparator.compare {
            Ok(self.pop_by(&|left, right| {
                compare_ordinal_rows(&left.row, &right.row, compare.as_ref())
            }))
        } else {
            self.try_pop_by(&|left, right| {
                try_compare_ordinal_rows(&left.row, &right.row, comparator)
            })
        }
    }
}
impl ComparatorMinHeap<ExactOwnedHeapEntry> {
    fn push_exact_owned(
        &mut self,
        entry: ExactOwnedHeapEntry,
        comparator: &SemanticRowComparator,
    ) -> Result<(), ExternalSortOperationError> {
        if let Comparison::Infallible(compare) = &comparator.compare {
            self.push_by(entry, &|left, right| {
                compare(left.row.values(), right.row.values())
                    .then_with(|| left.row.ordinal().cmp(&right.row.ordinal()))
            });
            Ok(())
        } else {
            self.try_push_by(entry, &|left, right| {
                Ok(comparator
                    .try_compare(left.row.values(), right.row.values())?
                    .then_with(|| left.row.ordinal().cmp(&right.row.ordinal())))
            })
        }
    }
    fn pop_exact_owned(
        &mut self,
        comparator: &SemanticRowComparator,
    ) -> Result<Option<ExactOwnedHeapEntry>, ExternalSortOperationError> {
        if let Comparison::Infallible(compare) = &comparator.compare {
            Ok(self.pop_by(&|left, right| {
                compare(left.row.values(), right.row.values())
                    .then_with(|| left.row.ordinal().cmp(&right.row.ordinal()))
            }))
        } else {
            self.try_pop_by(&|left, right| {
                Ok(comparator
                    .try_compare(left.row.values(), right.row.values())?
                    .then_with(|| left.row.ordinal().cmp(&right.row.ordinal())))
            })
        }
    }
}

// Stable encounter ordinals make an in-place heapsort semantically stable,
// without allocating scratch or converting a comparison failure into equality.
fn sort_ordinal_rows(
    rows: &mut [OrdinalRow],
    comparator: &SemanticRowComparator,
) -> Result<(), ExternalSortOperationError> {
    if let Comparison::Infallible(compare) = &comparator.compare {
        rows.sort_by(|left, right| compare_ordinal_rows(left, right, compare.as_ref()));
        return Ok(());
    }
    fn sift(
        rows: &mut [OrdinalRow],
        mut index: usize,
        end: usize,
        comparator: &SemanticRowComparator,
    ) -> Result<(), ExternalSortOperationError> {
        while index < end / 2 {
            let left = index * 2 + 1;
            let right = left + 1;
            let child = if right < end
                && try_compare_ordinal_rows(&rows[left], &rows[right], comparator)?
                    == Ordering::Less
            {
                right
            } else {
                left
            };
            if try_compare_ordinal_rows(&rows[index], &rows[child], comparator)? != Ordering::Less {
                break;
            }
            rows.swap(index, child);
            index = child;
        }
        Ok(())
    }
    for index in (0..rows.len() / 2).rev() {
        sift(rows, index, rows.len(), comparator)?;
    }
    for end in (1..rows.len()).rev() {
        rows.swap(0, end);
        sift(rows, 0, end, comparator)?;
    }
    Ok(())
}

fn try_compare_ordinal_rows(
    left: &OrdinalRow,
    right: &OrdinalRow,
    comparator: &SemanticRowComparator,
) -> Result<Ordering, ExternalSortOperationError> {
    Ok(comparator
        .try_compare(&left.values, &right.values)?
        .then_with(|| left.ordinal.cmp(&right.ordinal)))
}

/// Compares two rows by sort keys.
fn compare_rows(a: &[Value], b: &[Value], keys: &[SortKey]) -> Ordering {
    for key in keys {
        let a_val = a.get(key.column);
        let b_val = b.get(key.column);

        let ordering = match (a_val, b_val) {
            (Some(Value::Null), Some(Value::Null)) => Ordering::Equal,
            (Some(Value::Null), _) => match key.null_order {
                NullOrder::First => Ordering::Less,
                NullOrder::Last => Ordering::Greater,
            },
            (_, Some(Value::Null)) => match key.null_order {
                NullOrder::First => Ordering::Greater,
                NullOrder::Last => Ordering::Less,
            },
            (Some(a), Some(b)) => {
                let ordering = compare_values_total(a, b);
                match key.direction {
                    SortDirection::Ascending => ordering,
                    SortDirection::Descending => ordering.reverse(),
                }
            }
            _ => Ordering::Equal,
        };

        if ordering != Ordering::Equal {
            return ordering;
        }
    }

    Ordering::Equal
}

fn compare_ordinal_rows(
    left: &OrdinalRow,
    right: &OrdinalRow,
    comparator: &RowCompareFn,
) -> Ordering {
    comparator(&left.values, &right.values).then_with(|| left.ordinal.cmp(&right.ordinal))
}

#[cfg(test)]
fn checked_accounted_decoded_retained_bytes(
    direct_bytes: usize,
    payload_bytes: usize,
    broad_decoded_bytes: usize,
) -> Result<usize, ExternalSortOperationError> {
    let retained_bytes = checked_workspace_sum(direct_bytes, payload_bytes)?;
    if retained_bytes > broad_decoded_bytes {
        // Keep the invariant failure allocation-free: the construction grant
        // still owns all live decoded and encoded allocations on this path.
        return Err(ExternalSortOperationError::Io(std::io::Error::from(
            std::io::ErrorKind::InvalidData,
        )));
    }
    Ok(retained_bytes)
}

/// Decodes one owned spill frame under a conservative construction grant and
/// returns exact move-only retained ownership.
///
/// `construction_grant` must be a dedicated child which already covers the
/// encoded payload's observed capacity. Before decoding, it is expanded to
/// the checked sum of that capacity and the broad decoded-row envelope. Once
/// decoding has produced its unforgeable receipt, the encoded allocation is
/// destroyed and the child shrinks to top-level `Vec<Value>` capacity plus
/// decoder-witnessed recursive payload bytes.
#[allow(
    dead_code,
    reason = "the next move-consuming row-to-column builder will call this sealed mint"
)]
fn decode_accounted_sort_row(
    payload: Vec<u8>,
    num_columns: usize,
    limits: super::file::SpillFrameLimits,
    construction_grant: MemoryGrant,
) -> Result<AccountedOrdinalRow, ExternalSortOperationError> {
    let mut encoded = AccountedEncodedSortRow::try_new(payload, construction_grant)?;
    let values_end = encoded
        .payload()
        .len()
        .checked_sub(std::mem::size_of::<u64>())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "sort row record is missing its input ordinal",
            )
        })?;
    let ordinal_bytes: [u8; 8] = encoded.payload()[values_end..].try_into().map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "sort row record has an invalid input ordinal",
        )
    })?;
    let broad_decoded_bytes = conservative_decoded_row_retained_bytes(values_end)?;
    let construction_bytes =
        checked_workspace_sum(encoded.payload_capacity(), broad_decoded_bytes)?;
    encoded.resize_grant(construction_bytes)?;

    let decoded = deserialize_framed_row_exact_with_receipt(
        &encoded.payload()[..values_end],
        num_columns,
        limits.codec_limits(),
    )?;

    // The physical encoded allocation must cease before its construction
    // authority is reduced to the decoded row's exact retained charge.
    let grant = encoded.into_released_grant();
    AccountedOrdinalRow::try_from_decoded(decoded, u64::from_le_bytes(ordinal_bytes), grant, None)
        .map_err(map_accounted_sort_row_build_error)
}

/// Decodes the exact provider-owned payload without converting or copying its
/// `allocator_api2` backing. The same child grant expands around the
/// encoded-plus-decoded peak, then transfers to the decoded row only after the
/// encoded allocation has been destroyed.
#[expect(
    clippy::result_large_err,
    reason = "the exact lane keeps the non-detachable accounted failure inline instead of allocating a box after decode failure"
)]
fn decode_provider_accounted_sort_row(
    encoded: ProviderAccountedSortRow,
    row_shape: SortRowShape,
    limits: super::file::SpillFrameLimits,
    observer: Option<&ExactOwnedGrantTransitionObserver<'_, '_>>,
) -> Result<AccountedOrdinalRow, ExactOwnedSortStreamError> {
    decode_provider_accounted_sort_row_with(
        encoded,
        row_shape,
        limits,
        observer,
        deserialize_framed_row_exact_with_receipt_qualified,
    )
}

#[expect(
    clippy::result_large_err,
    reason = "the exact lane keeps the non-detachable accounted failure inline instead of allocating a box after decode failure"
)]
fn decode_provider_accounted_sort_row_with(
    mut encoded: ProviderAccountedSortRow,
    row_shape: SortRowShape,
    limits: super::file::SpillFrameLimits,
    observer: Option<&ExactOwnedGrantTransitionObserver<'_, '_>>,
    decode: impl FnOnce(&[u8], usize, CodecLimits) -> Result<DecodedFramedRow, QualifiedCodecError>,
) -> Result<AccountedOrdinalRow, ExactOwnedSortStreamError> {
    if encoded.granted_bytes() < encoded.payload_capacity() {
        return Err(exact_decoded_failure(
            encoded,
            ExactOwnedDecodedPrimary::InvalidInput(
                "encoded sort-row capacity exceeds its provider-owned child grant",
            ),
        ));
    }
    let Some(values_end) = encoded
        .payload()
        .len()
        .checked_sub(std::mem::size_of::<u64>())
    else {
        return Err(exact_decoded_failure(
            encoded,
            ExactOwnedDecodedPrimary::InvalidData("sort row record is missing its input ordinal"),
        ));
    };
    let ordinal_bytes: [u8; 8] = match encoded.payload()[values_end..].try_into() {
        Ok(ordinal) => ordinal,
        Err(_) => {
            return Err(exact_decoded_failure(
                encoded,
                ExactOwnedDecodedPrimary::InvalidData(
                    "sort row record has an invalid input ordinal",
                ),
            ));
        }
    };
    let physical_columns = match row_shape.physical_columns(&encoded.payload()[..values_end]) {
        Ok(columns) => columns,
        Err(message) => {
            return Err(exact_decoded_failure(
                encoded,
                ExactOwnedDecodedPrimary::InvalidData(message),
            ));
        }
    };
    let broad_decoded_bytes = match conservative_decoded_row_retained_bytes(values_end) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(exact_decoded_failure(
                encoded,
                ExactOwnedDecodedPrimary::Memory(error),
            ));
        }
    };
    let construction_bytes =
        match checked_workspace_sum(encoded.payload_capacity(), broad_decoded_bytes) {
            Ok(bytes) => bytes,
            Err(error) => {
                return Err(exact_decoded_failure(
                    encoded,
                    ExactOwnedDecodedPrimary::Memory(error),
                ));
            }
        };
    let previous = encoded.granted_bytes();
    if let Err(error) = encoded.grow_grant_to(construction_bytes) {
        return Err(exact_decoded_failure(
            encoded,
            ExactOwnedDecodedPrimary::Memory(error),
        ));
    }
    if let Some(Err(error)) =
        observer.map(|observer| observer.replace_row(previous, encoded.granted_bytes()))
    {
        return Err(exact_decoded_failure(
            encoded,
            ExactOwnedDecodedPrimary::Memory(error),
        ));
    }

    let decoded = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        decode(
            &encoded.payload()[..values_end],
            physical_columns,
            limits.codec_limits(),
        )
    })) {
        Ok(Ok(decoded)) => decoded,
        Ok(Err(error)) => {
            return Err(exact_decoded_failure(
                encoded,
                ExactOwnedDecodedPrimary::Codec(error),
            ));
        }
        Err(payload) => {
            return Err(ExactOwnedSortStreamError::decoded_panic(
                ExactOwnedDecodedPanic::new(payload, encoded),
            ));
        }
    };
    if let Err(message) = row_shape.edge_mask(decoded.values()) {
        drop(decoded);
        return Err(exact_decoded_failure(
            encoded,
            ExactOwnedDecodedPrimary::InvalidData(message),
        ));
    }
    let grant = encoded.into_released_grant();
    match AccountedOrdinalRow::try_from_decoded(
        decoded,
        u64::from_le_bytes(ordinal_bytes),
        grant,
        observer.map(|observer| {
            observer as &dyn crate::execution::accounted_chunk::AccountedRowGrantObserver
        }),
    ) {
        Ok(row) => Ok(row),
        Err(error) => {
            let (primary, grant) = error.into_parts();
            let primary = match primary {
                AccountedSortRowError::Memory(error) => ExactOwnedDecodedPrimary::Memory(error),
                AccountedSortRowError::Arithmetic => ExactOwnedDecodedPrimary::Allocation(
                    "decoded sort-row retained layout is not representable",
                ),
                AccountedSortRowError::InvalidAuthority => ExactOwnedDecodedPrimary::InvalidData(
                    "decoded sort row authority does not match its retained shape",
                ),
            };
            Err(ExactOwnedSortStreamError::decoded(
                ExactOwnedDecodedError::new(primary, grant),
            ))
        }
    }
}

fn exact_decoded_failure(
    encoded: ProviderAccountedSortRow,
    primary: ExactOwnedDecodedPrimary,
) -> ExactOwnedSortStreamError {
    let grant = encoded.into_released_grant();
    ExactOwnedSortStreamError::decoded(ExactOwnedDecodedError::new(primary, grant))
}

fn map_accounted_sort_row_build_error(
    error: crate::execution::accounted_chunk::AccountedSortRowBuildError,
) -> ExternalSortOperationError {
    let (primary, grant) = error.into_parts();
    // `try_from_decoded` destroys the physical values and receipt before this
    // owner can escape. Only then may the compatibility lane release the
    // construction authority it does not transport in its legacy error.
    drop(grant);
    map_accounted_sort_row_error(primary)
}

fn map_accounted_sort_row_error(error: AccountedSortRowError) -> ExternalSortOperationError {
    match error {
        AccountedSortRowError::Memory(error) => ExternalSortOperationError::Memory(error),
        AccountedSortRowError::Arithmetic => ExternalSortOperationError::Allocation(
            std::io::Error::from(std::io::ErrorKind::OutOfMemory),
        ),
        AccountedSortRowError::InvalidAuthority => {
            ExternalSortOperationError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "decoded sort row authority does not match its retained shape",
            ))
        }
    }
}

fn decode_row_payload_with_shape(
    payload: &[u8],
    row_shape: SortRowShape,
    limits: super::file::SpillFrameLimits,
) -> std::io::Result<OrdinalRow> {
    let values_end = payload
        .len()
        .checked_sub(std::mem::size_of::<u64>())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "sort row record is missing its input ordinal",
            )
        })?;
    let (value_payload, ordinal_payload) = payload.split_at(values_end);
    let ordinal_bytes: [u8; 8] = ordinal_payload.try_into().map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "sort row record has an invalid input ordinal",
        )
    })?;
    let physical_columns = row_shape
        .physical_columns(value_payload)
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidData, message))?;
    let values =
        deserialize_framed_row_exact(value_payload, physical_columns, limits.codec_limits())?;
    row_shape
        .edge_mask(&values)
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidData, message))?;
    Ok(OrdinalRow {
        values,
        ordinal: u64::from_le_bytes(ordinal_bytes),
    })
}

#[cfg(test)]
fn decode_row_payload(
    payload: &[u8],
    num_columns: usize,
    limits: super::file::SpillFrameLimits,
) -> std::io::Result<OrdinalRow> {
    decode_row_payload_with_shape(payload, SortRowShape::strict(num_columns), limits)
}

#[cfg(test)]
// reason: test indices are small known values
#[allow(clippy::cast_possible_wrap)]
mod tests {
    use super::*;
    use arcstr::ArcStr;
    use grafeo_common::memory::buffer::{
        BufferManager, BufferManagerConfig, MemoryConsumer, MemoryGrantError, MemoryRegion,
        StorageTier, priorities,
    };
    use std::cell::Cell;
    use std::collections::{BTreeMap, HashMap};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::{Arc, Mutex};
    use tempfile::TempDir;

    struct ContractComparator {
        bound: usize,
        fail: bool,
        calls: Arc<AtomicUsize>,
    }

    impl AccountedValueComparator for ContractComparator {
        fn scratch_bytes(
            &self,
            _: Option<&Value>,
            _: Option<&Value>,
        ) -> Result<usize, SemanticComparisonError> {
            Ok(self.bound)
        }
        fn compare(
            &self,
            left: Option<&Value>,
            right: Option<&Value>,
        ) -> Result<Ordering, SemanticComparisonError> {
            self.calls.fetch_add(1, AtomicOrdering::Relaxed);
            if self.fail {
                return Err(SemanticComparisonError::Invalid("semantic witness"));
            }
            Ok(match (left, right) {
                (Some(left), Some(right)) => compare_values_total(left, right),
                (None, None) => Ordering::Equal,
                (None, _) => Ordering::Less,
                (_, None) => Ordering::Greater,
            })
        }
    }

    fn semantic_fault_sort(
        manager: Arc<SpillManager>,
        columns: usize,
        keys: Vec<SortKey>,
        grant: MemoryGrant,
        resources: &crate::execution::QueryResourceContext,
    ) -> ExternalSort {
        let comparator = SemanticRowComparator::new_accounted(
            keys,
            Arc::new(ContractComparator {
                bound: 4096,
                fail: false,
                calls: Arc::new(AtomicUsize::new(0)),
            }),
            resources.clone(),
        )
        .unwrap();
        ExternalSort::new_accounted_with_comparator_and_cancellation(
            manager,
            columns,
            comparator,
            grant,
            resources.cancellation_token().clone(),
        )
    }

    #[test]
    fn accounted_comparator_denies_scratch_before_callback_and_retains_shared_metadata() {
        let buffers = buffer_manager_with_exact_budget(1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new(buffers.clone()).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let comparator = SemanticRowComparator::new_accounted(
            vec![SortKey::ascending(0)],
            Arc::new(ContractComparator {
                bound: 2 * 1024 * 1024,
                fail: false,
                calls: calls.clone(),
            }),
            resources,
        )
        .unwrap();
        let fixed = comparator.checked_granted_bytes().unwrap();
        assert!(fixed > 0);
        assert!(matches!(
            comparator.try_compare(&[Value::Int64(1)], &[Value::Int64(2)]),
            Err(ExternalSortOperationError::Memory(_))
        ));
        assert_eq!(calls.load(AtomicOrdering::Relaxed), 0);
        let retained = comparator.clone();
        drop(comparator);
        assert_eq!(buffers.allocated(), fixed);
        drop(retained);
        assert_eq!(buffers.allocated(), 0);
    }

    #[test]
    fn accounted_comparator_error_poisoned_heap_retains_every_entry() {
        let buffers = buffer_manager_with_exact_budget(1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new(buffers.clone()).unwrap();
        let comparator = SemanticRowComparator::new_accounted(
            vec![SortKey::ascending(0)],
            Arc::new(ContractComparator {
                bound: 4096,
                fail: true,
                calls: Arc::new(AtomicUsize::new(0)),
            }),
            resources,
        )
        .unwrap();
        let mut heap = ComparatorMinHeap::try_with_exact_capacity(2).unwrap();
        let compare = |left: &Vec<Value>, right: &Vec<Value>| comparator.try_compare(left, right);
        heap.try_push_by(vec![Value::Int64(2)], &compare).unwrap();
        let error = heap
            .try_push_by(vec![Value::Int64(1)], &compare)
            .unwrap_err();
        assert!(
            matches!(error, ExternalSortOperationError::Io(ref error) if error.to_string() == "semantic witness")
        );
        assert!(heap.poisoned);
        assert_eq!(heap.entries.len(), 2);
        drop(heap);
        drop(comparator);
        assert_eq!(buffers.allocated(), 0);
    }

    #[test]
    fn accounted_comparator_exact_qualification_requires_same_query_account() {
        for same_account in [false, true] {
            let (_directory, manager) = create_manager();
            let buffers = buffer_manager_with_exact_budget(4 * 1024 * 1024);
            let resources = crate::execution::QueryResourceContext::new(buffers.clone()).unwrap();
            let comparator_resources = if same_account {
                resources.clone()
            } else {
                crate::execution::QueryResourceContext::new(buffers.clone()).unwrap()
            };
            let comparator = SemanticRowComparator::new_accounted(
                vec![SortKey::ascending(0)],
                Arc::new(ContractComparator {
                    bound: 4096,
                    fail: false,
                    calls: Arc::new(AtomicUsize::new(0)),
                }),
                comparator_resources,
            )
            .unwrap();
            let mut sort = ExternalSort::new_accounted_with_comparator_and_cancellation(
                manager.clone(),
                1,
                comparator,
                resources.try_allocate(0).unwrap(),
                resources.cancellation_token().clone(),
            );
            sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
            assert_eq!(sort.exact_owned_disk_base_shape_eligible(), same_account);
            assert_eq!(sort.exact_owned_disk_shape_eligible(), same_account);
            assert_eq!(
                sort.comparator.try_compare(&row(&[1]), &row(&[2])).unwrap(),
                Ordering::Less
            );
            drop(sort);
            assert_eq!(manager.active_file_count(), 0);
            assert_eq!(buffers.allocated(), 0);
        }
    }

    #[test]
    fn accounted_comparator_exact_cursor_merges_stable_duplicates() {
        let (_directory, manager) = create_manager();
        let buffers = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new(buffers.clone()).unwrap();
        let comparator = SemanticRowComparator::new_accounted(
            vec![SortKey::ascending(0)],
            Arc::new(ContractComparator {
                bound: 4096,
                fail: false,
                calls: Arc::new(AtomicUsize::new(0)),
            }),
            resources.clone(),
        )
        .unwrap();
        let state = Arc::new(OperatorSpillState::new("accounted semantic merge".into()));
        let observer = ExternalSortGrantObserver::owned(state.clone());
        let mut sort = ExternalSort::new_accounted_with_comparator_and_cancellation(
            manager.clone(),
            2,
            comparator,
            resources.try_allocate(0).unwrap(),
            resources.cancellation_token().clone(),
        );
        sort.set_merge_fan_in(2);
        for ordinal in 0..7_i64 {
            sort.spill_distinct_run_accounted(
                &[vec![Value::Int64(1), Value::Int64(ordinal)]],
                &observer,
            )
            .unwrap();
        }
        sort.finish_distinct_runs_accounted(&observer).unwrap();
        assert!(sort.try_enable_exact_owned_output(Some(&state)));
        let mut cursor = sort.into_send_owned_disk_cursor(observer).unwrap();
        for ordinal in 0..7_i64 {
            let row = cursor.next_owned_row().unwrap().unwrap();
            assert_eq!(row.values(), &[Value::Int64(1), Value::Int64(ordinal)]);
            drop(row.into_released_grant());
            cursor.release_transferred_retained().unwrap();
        }
        assert!(cursor.next_owned_row().unwrap().is_none());
        drop(cursor);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(buffers.allocated(), 0);
        assert_eq!(state.usage(), 0);
    }

    #[test]
    fn invalid_cancellation_state_is_internal_io() {
        let error = cancellation_io_error(QueryCancellationError::InvalidState { phase: 255 });
        assert_eq!(error.kind(), std::io::ErrorKind::Other);
    }

    struct TestGrantObservation {
        external_bytes: Cell<usize>,
        retained_bytes: Cell<usize>,
        peak_external_bytes: Cell<usize>,
    }

    impl TestGrantObservation {
        fn new(initial_external_bytes: usize) -> Self {
            Self {
                external_bytes: Cell::new(initial_external_bytes),
                retained_bytes: Cell::new(0),
                peak_external_bytes: Cell::new(initial_external_bytes),
            }
        }

        fn observer(&self) -> ExternalSortGrantObserver<'_> {
            ExternalSortGrantObserver::new(&self.external_bytes, &self.retained_bytes, None)
                .tracking_peak(&self.peak_external_bytes)
        }

        fn current(&self) -> usize {
            self.external_bytes.get()
        }

        fn peak(&self) -> usize {
            self.peak_external_bytes.get()
        }
    }

    fn unwrap_exact_final_failure(error: ExactOwnedSortStreamError) -> AccountedError {
        let ExactOwnedSortStreamError {
            primary:
                ExactOwnedSortPrimary::Accounted {
                    authority: accounted,
                    ..
                },
        } = error
        else {
            panic!("exact failure did not pass through its final accounted envelope")
        };
        assert!(accounted.is::<ExactOwnedFinalFailure>());
        accounted
    }

    struct HeapDropProbe {
        key: i64,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for HeapDropProbe {
        fn drop(&mut self) {
            self.drops.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    #[derive(Default)]
    struct CountingCreateIo {
        creates: AtomicUsize,
    }

    impl super::super::SpillIo for CountingCreateIo {
        fn check(&self, operation: super::super::SpillIoOperation) -> std::io::Result<()> {
            if operation == super::super::SpillIoOperation::Create {
                self.creates.fetch_add(1, AtomicOrdering::Relaxed);
            }
            Ok(())
        }
    }

    #[derive(Default)]
    struct CountingReadPayloadIo {
        opens: AtomicUsize,
        reads: AtomicUsize,
        writes: AtomicUsize,
    }

    impl super::super::SpillIo for CountingReadPayloadIo {
        fn check(&self, operation: super::super::SpillIoOperation) -> std::io::Result<()> {
            match operation {
                super::super::SpillIoOperation::ReadOpen => {
                    self.opens.fetch_add(1, AtomicOrdering::Relaxed);
                }
                super::super::SpillIoOperation::ReadPayload => {
                    self.reads.fetch_add(1, AtomicOrdering::Relaxed);
                }
                super::super::SpillIoOperation::WritePayload => {
                    self.writes.fetch_add(1, AtomicOrdering::Relaxed);
                }
                _ => {}
            }
            Ok(())
        }
    }

    struct QualificationTrapIo {
        queries: Arc<AtomicUsize>,
        panic_on_query: bool,
    }

    impl super::super::SpillIo for QualificationTrapIo {
        fn check(&self, _operation: super::super::SpillIoOperation) -> std::io::Result<()> {
            Ok(())
        }

        fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
            self.queries.fetch_add(1, AtomicOrdering::Relaxed);
            assert!(
                !self.panic_on_query,
                "ineligible compatibility shape queried exact reader capability"
            );
            Some(0)
        }
    }

    struct QualifiedOpenFailureIo;

    impl super::super::SpillIo for QualifiedOpenFailureIo {
        fn check(&self, operation: super::super::SpillIoOperation) -> std::io::Result<()> {
            if operation == super::super::SpillIoOperation::ReadOpen {
                return Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied));
            }
            Ok(())
        }

        fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
            Some(0)
        }
    }

    #[derive(Clone, Copy)]
    enum ArmedQualifiedReadFailure {
        Error,
        Panic,
    }

    struct ArmedQualifiedReadIo {
        armed: AtomicBool,
        failure: ArmedQualifiedReadFailure,
    }

    impl ArmedQualifiedReadIo {
        fn new(failure: ArmedQualifiedReadFailure) -> Self {
            Self {
                armed: AtomicBool::new(false),
                failure,
            }
        }

        fn arm(&self) {
            self.armed.store(true, AtomicOrdering::Release);
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct ExactReaderPanicSentinel;

    impl super::super::SpillIo for ArmedQualifiedReadIo {
        fn check(&self, operation: super::super::SpillIoOperation) -> std::io::Result<()> {
            if operation != super::super::SpillIoOperation::ReadPayload
                || !self.armed.swap(false, AtomicOrdering::AcqRel)
            {
                return Ok(());
            }
            match self.failure {
                ArmedQualifiedReadFailure::Error => Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "deterministic midstream exact payload failure",
                )),
                ArmedQualifiedReadFailure::Panic => std::panic::panic_any(ExactReaderPanicSentinel),
            }
        }

        fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
            Some(4096)
        }
    }

    #[derive(Default)]
    struct ReaderPeakProvider {
        begin_counts: Mutex<HashMap<super::super::SpillFileIdentity, usize>>,
        active_readers: Arc<AtomicUsize>,
        peak_readers: Arc<AtomicUsize>,
    }

    impl ReaderPeakProvider {
        fn active_readers(&self) -> usize {
            self.active_readers.load(AtomicOrdering::Acquire)
        }

        fn peak_readers(&self) -> usize {
            self.peak_readers.load(AtomicOrdering::Acquire)
        }
    }

    impl super::super::SpillRecordProvider for ReaderPeakProvider {
        fn seals(&self) -> bool {
            false
        }

        fn begin_file(
            &self,
            identity: super::super::SpillFileIdentity,
        ) -> std::io::Result<Box<dyn super::super::OpenSpillRecord>> {
            let track_reader = {
                let mut begin_counts = self.begin_counts.lock().unwrap();
                let count = begin_counts.entry(identity).or_default();
                let track_reader = *count != 0;
                *count += 1;
                track_reader
            };
            let active_readers = track_reader.then(|| Arc::clone(&self.active_readers));
            if active_readers.is_some() {
                let active = self.active_readers.fetch_add(1, AtomicOrdering::AcqRel) + 1;
                self.peak_readers.fetch_max(active, AtomicOrdering::AcqRel);
            }
            Ok(Box::new(ReaderPeakOpenRecord { active_readers }))
        }

        fn file_workspace_allocation_bound(&self) -> Option<usize> {
            Some(4096)
        }
    }

    struct ReaderPeakOpenRecord {
        active_readers: Option<Arc<AtomicUsize>>,
    }

    impl super::super::OpenSpillRecord for ReaderPeakOpenRecord {
        fn stored_len(&self, plaintext_len: usize) -> std::io::Result<usize> {
            Ok(plaintext_len)
        }

        fn seal_allocation_bound(&self, plaintext_len: usize) -> Option<usize> {
            plaintext_len.checked_mul(2)
        }

        fn open_allocation_bound(&self, stored_len: usize) -> Option<usize> {
            stored_len.checked_mul(2)
        }

        fn seal(
            &mut self,
            _meta: &super::super::SpillRecordMeta,
            _aad: &[u8; 32],
            plaintext: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            Ok(plaintext.to_vec())
        }

        fn open(
            &mut self,
            _meta: &super::super::SpillRecordMeta,
            _aad: &[u8; 32],
            stored: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            Ok(stored.to_vec())
        }
    }

    impl Drop for ReaderPeakOpenRecord {
        fn drop(&mut self) {
            if let Some(active_readers) = &self.active_readers {
                active_readers.fetch_sub(1, AtomicOrdering::AcqRel);
            }
        }
    }

    /// A deliberately legacy-only provider: it supports the original framed
    /// I/O contract but declares none of the allocation bounds required by a
    /// qualified query. Public compatibility cursors must remain usable with
    /// providers that predate the opt-in resource contract.
    struct LegacyOnlyProvider;

    impl super::super::SpillRecordProvider for LegacyOnlyProvider {
        fn seals(&self) -> bool {
            false
        }

        fn begin_file(
            &self,
            _identity: super::super::SpillFileIdentity,
        ) -> std::io::Result<Box<dyn super::super::OpenSpillRecord>> {
            Ok(Box::new(LegacyOnlyOpenRecord))
        }
    }

    struct LegacyOnlyOpenRecord;

    impl super::super::OpenSpillRecord for LegacyOnlyOpenRecord {
        fn stored_len(&self, plaintext_len: usize) -> std::io::Result<usize> {
            Ok(plaintext_len)
        }

        fn seal(
            &mut self,
            _meta: &super::super::SpillRecordMeta,
            _aad: &[u8; 32],
            plaintext: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            Ok(plaintext.to_vec())
        }

        fn open(
            &mut self,
            _meta: &super::super::SpillRecordMeta,
            _aad: &[u8; 32],
            stored: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            Ok(stored.to_vec())
        }
    }

    /// Adds trusted test-only allocation bounds around the pre-existing
    /// framing failpoint providers without changing their callback behavior.
    struct QualifiedTestProvider {
        inner: Arc<dyn super::super::SpillRecordProvider>,
    }

    impl super::super::SpillRecordProvider for QualifiedTestProvider {
        fn seals(&self) -> bool {
            self.inner.seals()
        }

        fn begin_file(
            &self,
            identity: super::super::SpillFileIdentity,
        ) -> std::io::Result<Box<dyn super::super::OpenSpillRecord>> {
            self.inner.begin_file(identity).map(|inner| {
                Box::new(QualifiedTestOpenRecord { inner })
                    as Box<dyn super::super::OpenSpillRecord>
            })
        }

        fn file_workspace_allocation_bound(&self) -> Option<usize> {
            // Two boxed record adapters plus the largest fixed-frame copy.
            Some(64)
        }
    }

    struct QualifiedTestOpenRecord {
        inner: Box<dyn super::super::OpenSpillRecord>,
    }

    impl super::super::OpenSpillRecord for QualifiedTestOpenRecord {
        fn stored_len(&self, plaintext_len: usize) -> std::io::Result<usize> {
            self.inner.stored_len(plaintext_len)
        }

        fn seal_allocation_bound(&self, plaintext_len: usize) -> Option<usize> {
            plaintext_len.checked_mul(2)
        }

        fn open_allocation_bound(&self, stored_len: usize) -> Option<usize> {
            stored_len.checked_mul(2)
        }

        fn seal(
            &mut self,
            meta: &super::super::SpillRecordMeta,
            aad: &[u8; 32],
            plaintext: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            self.inner.seal(meta, aad, plaintext)
        }

        fn open(
            &mut self,
            meta: &super::super::SpillRecordMeta,
            aad: &[u8; 32],
            stored: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            self.inner.open(meta, aad, stored)
        }
    }

    fn qualified_test_provider(
        inner: Arc<dyn super::super::SpillRecordProvider>,
    ) -> Arc<dyn super::super::SpillRecordProvider> {
        Arc::new(QualifiedTestProvider { inner })
    }

    struct CorruptOpenedSortRowProvider;

    impl super::super::SpillRecordProvider for CorruptOpenedSortRowProvider {
        fn seals(&self) -> bool {
            false
        }

        fn begin_file(
            &self,
            _identity: super::super::SpillFileIdentity,
        ) -> std::io::Result<Box<dyn super::super::OpenSpillRecord>> {
            Ok(Box::new(CorruptOpenedSortRow))
        }

        fn file_workspace_allocation_bound(&self) -> Option<usize> {
            Some(24)
        }

        fn supports_qualified_exact_open(&self) -> bool {
            true
        }
    }

    struct CorruptOpenedSortRow;

    impl super::super::OpenSpillRecord for CorruptOpenedSortRow {
        fn stored_len(&self, plaintext_len: usize) -> std::io::Result<usize> {
            Ok(plaintext_len)
        }

        fn seal_allocation_bound(&self, plaintext_len: usize) -> Option<usize> {
            plaintext_len.checked_mul(2)
        }

        fn open_allocation_bound(&self, stored_len: usize) -> Option<usize> {
            stored_len.checked_mul(2)
        }

        fn seal(
            &mut self,
            _meta: &super::super::SpillRecordMeta,
            _aad: &[u8; 32],
            plaintext: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            Ok(plaintext.to_vec())
        }

        fn open(
            &mut self,
            meta: &super::super::SpillRecordMeta,
            _aad: &[u8; 32],
            stored: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            let mut plaintext = stored.to_vec();
            if meta.kind() == super::super::SpillRecordKind::SortRow
                && let Some(tag) = plaintext.get_mut(8)
            {
                *tag = u8::MAX;
            }
            Ok(plaintext)
        }

        fn open_qualified_into(
            &mut self,
            meta: &super::super::SpillRecordMeta,
            _aad: &[u8; 32],
            stored: &[u8],
            plaintext: &mut [u8],
        ) -> Option<std::io::Result<()>> {
            if stored.len() != plaintext.len() {
                return Some(Err(std::io::Error::from(std::io::ErrorKind::InvalidData)));
            }
            plaintext.copy_from_slice(stored);
            if meta.kind() == super::super::SpillRecordKind::SortRow
                && let Some(tag) = plaintext.get_mut(8)
            {
                *tag = u8::MAX;
            }
            Some(Ok(()))
        }
    }

    struct CancelNthIo {
        target: super::super::SpillIoOperation,
        trigger: usize,
        armed: AtomicBool,
        matching: AtomicUsize,
        flushes: AtomicUsize,
        syncs: AtomicUsize,
        read_opens: AtomicUsize,
        cancellation: crate::execution::QueryCancellationHandle,
        failure: Option<(std::io::ErrorKind, &'static str)>,
        fail_delete: AtomicBool,
        delete_failure: Option<(std::io::ErrorKind, &'static str)>,
    }

    impl CancelNthIo {
        fn new(
            target: super::super::SpillIoOperation,
            trigger: usize,
            cancellation: crate::execution::QueryCancellationHandle,
        ) -> Self {
            Self {
                target,
                trigger,
                armed: AtomicBool::new(true),
                matching: AtomicUsize::new(0),
                flushes: AtomicUsize::new(0),
                syncs: AtomicUsize::new(0),
                read_opens: AtomicUsize::new(0),
                cancellation,
                failure: None,
                fail_delete: AtomicBool::new(false),
                delete_failure: None,
            }
        }

        fn with_failure(
            target: super::super::SpillIoOperation,
            trigger: usize,
            cancellation: crate::execution::QueryCancellationHandle,
            kind: std::io::ErrorKind,
            message: &'static str,
        ) -> Self {
            Self {
                failure: Some((kind, message)),
                ..Self::new(target, trigger, cancellation)
            }
        }

        fn disarm(&self) {
            self.armed.store(false, AtomicOrdering::Release);
        }

        fn with_delete_failure(mut self, kind: std::io::ErrorKind, message: &'static str) -> Self {
            self.delete_failure = Some((kind, message));
            self.fail_delete.store(true, AtomicOrdering::Relaxed);
            self
        }

        fn permit_delete(&self) {
            self.fail_delete.store(false, AtomicOrdering::Release);
        }

        fn arm(&self) {
            self.matching.store(0, AtomicOrdering::Relaxed);
            self.armed.store(true, AtomicOrdering::Release);
        }
    }

    impl super::super::SpillIo for CancelNthIo {
        fn check(&self, operation: super::super::SpillIoOperation) -> std::io::Result<()> {
            match operation {
                super::super::SpillIoOperation::Flush => {
                    self.flushes.fetch_add(1, AtomicOrdering::Relaxed);
                }
                super::super::SpillIoOperation::Sync => {
                    self.syncs.fetch_add(1, AtomicOrdering::Relaxed);
                }
                super::super::SpillIoOperation::ReadOpen => {
                    self.read_opens.fetch_add(1, AtomicOrdering::Relaxed);
                }
                _ => {}
            }
            if self.armed.load(AtomicOrdering::Acquire) && operation == self.target {
                let matching = self.matching.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                if matching == self.trigger {
                    self.cancellation.cancel();
                    if let Some((kind, message)) = self.failure {
                        return Err(std::io::Error::new(kind, message));
                    }
                }
            }
            if operation == super::super::SpillIoOperation::Delete
                && self.fail_delete.load(AtomicOrdering::Acquire)
                && let Some((kind, message)) = self.delete_failure
            {
                return Err(std::io::Error::new(kind, message));
            }
            Ok(())
        }
    }

    struct AccountingCreateIo {
        buffer_manager: Arc<BufferManager>,
        allocations: Mutex<Vec<usize>>,
    }

    impl AccountingCreateIo {
        fn new(buffer_manager: Arc<BufferManager>) -> Self {
            Self {
                buffer_manager,
                allocations: Mutex::new(Vec::new()),
            }
        }

        fn allocations(&self) -> Vec<usize> {
            self.allocations.lock().unwrap().clone()
        }
    }

    impl super::super::SpillIo for AccountingCreateIo {
        fn check(&self, operation: super::super::SpillIoOperation) -> std::io::Result<()> {
            if operation == super::super::SpillIoOperation::Create {
                self.allocations
                    .lock()
                    .unwrap()
                    .push(self.buffer_manager.allocated());
            }
            Ok(())
        }
    }

    struct ToggleDeleteIo {
        fail_delete: std::sync::atomic::AtomicBool,
    }

    impl ToggleDeleteIo {
        fn failing() -> Self {
            Self {
                fail_delete: std::sync::atomic::AtomicBool::new(true),
            }
        }

        fn permit_delete(&self) {
            self.fail_delete
                .store(false, std::sync::atomic::Ordering::Release);
        }
    }

    impl super::super::SpillIo for ToggleDeleteIo {
        fn check(&self, operation: super::super::SpillIoOperation) -> std::io::Result<()> {
            if operation == super::super::SpillIoOperation::Delete
                && self.fail_delete.load(std::sync::atomic::Ordering::Acquire)
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "deterministic writer cleanup denial",
                ));
            }
            Ok(())
        }
    }

    struct PanicOnceIo {
        operation: super::super::SpillIoOperation,
        fired: std::sync::atomic::AtomicBool,
    }

    impl PanicOnceIo {
        fn new(operation: super::super::SpillIoOperation) -> Self {
            Self {
                operation,
                fired: std::sync::atomic::AtomicBool::new(false),
            }
        }
    }

    impl super::super::SpillIo for PanicOnceIo {
        fn check(&self, operation: super::super::SpillIoOperation) -> std::io::Result<()> {
            if operation == self.operation
                && !self.fired.swap(true, std::sync::atomic::Ordering::AcqRel)
            {
                panic!("deterministic {operation:?} I/O callback panic")
            }
            Ok(())
        }
    }

    #[derive(Clone, Copy)]
    enum CompoundDeleteOrder {
        ErrorThenPanic,
        PanicThenError,
    }

    struct CompoundDeleteIo {
        order: CompoundDeleteOrder,
        deletes: AtomicUsize,
    }

    impl CompoundDeleteIo {
        fn new(order: CompoundDeleteOrder) -> Self {
            Self {
                order,
                deletes: AtomicUsize::new(0),
            }
        }
    }

    impl super::super::SpillIo for CompoundDeleteIo {
        fn check(&self, operation: super::super::SpillIoOperation) -> std::io::Result<()> {
            if operation != super::super::SpillIoOperation::Delete {
                return Ok(());
            }
            match (
                self.order,
                self.deletes.fetch_add(1, AtomicOrdering::AcqRel),
            ) {
                (CompoundDeleteOrder::ErrorThenPanic, 0)
                | (CompoundDeleteOrder::PanicThenError, 1) => Err(std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    "deterministic exact cleanup deletion error",
                )),
                (CompoundDeleteOrder::ErrorThenPanic, 1)
                | (CompoundDeleteOrder::PanicThenError, 0) => {
                    std::panic::panic_any("deterministic exact cleanup deletion panic")
                }
                _ => Ok(()),
            }
        }

        fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
            Some(0)
        }
    }

    struct FailWriteAndToggleDeleteIo {
        write_payloads: AtomicUsize,
        fail_delete: std::sync::atomic::AtomicBool,
    }

    impl FailWriteAndToggleDeleteIo {
        fn failing() -> Self {
            Self {
                write_payloads: AtomicUsize::new(0),
                fail_delete: std::sync::atomic::AtomicBool::new(true),
            }
        }

        fn permit_delete(&self) {
            self.fail_delete
                .store(false, std::sync::atomic::Ordering::Release);
        }
    }

    impl super::super::SpillIo for FailWriteAndToggleDeleteIo {
        fn check(&self, operation: super::super::SpillIoOperation) -> std::io::Result<()> {
            if operation == super::super::SpillIoOperation::WritePayload
                && self.write_payloads.fetch_add(1, AtomicOrdering::Relaxed) == 2
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "deterministic writer primary failure",
                ));
            }
            if operation == super::super::SpillIoOperation::Delete
                && self.fail_delete.load(std::sync::atomic::Ordering::Acquire)
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    "deterministic writer deletion failure",
                ));
            }
            Ok(())
        }
    }

    struct PanickingEvictionConsumer;

    #[derive(Debug)]
    struct PanicOnDrop;

    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("secondary panic payload dropped");
        }
    }

    #[derive(Debug)]
    struct DoublePanicError {
        _nested: PanicOnDrop,
    }

    impl std::fmt::Display for DoublePanicError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("hostile external-sort cleanup error")
        }
    }

    impl std::error::Error for DoublePanicError {}

    impl Drop for DoublePanicError {
        fn drop(&mut self) {
            panic!("outer external-sort cleanup error dropped");
        }
    }

    struct HostileBeginErrorProvider;

    struct PanicOnProviderDrop;

    impl super::super::SpillRecordProvider for PanicOnProviderDrop {
        fn seals(&self) -> bool {
            false
        }

        fn begin_file(
            &self,
            identity: super::super::SpillFileIdentity,
        ) -> std::io::Result<Box<dyn super::super::OpenSpillRecord>> {
            super::super::SpillRecordProvider::begin_file(
                &super::super::CleartextSpillRecordProvider,
                identity,
            )
        }

        fn file_workspace_allocation_bound(&self) -> Option<usize> {
            super::super::SpillRecordProvider::file_workspace_allocation_bound(
                &super::super::CleartextSpillRecordProvider,
            )
        }
    }

    impl Drop for PanicOnProviderDrop {
        fn drop(&mut self) {
            panic!("deterministic spill provider destructor panic");
        }
    }

    impl super::super::SpillRecordProvider for HostileBeginErrorProvider {
        fn seals(&self) -> bool {
            false
        }

        fn begin_file(
            &self,
            _identity: super::super::SpillFileIdentity,
        ) -> std::io::Result<Box<dyn super::super::OpenSpillRecord>> {
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                DoublePanicError {
                    _nested: PanicOnDrop,
                },
            ))
        }

        fn file_workspace_allocation_bound(&self) -> Option<usize> {
            Some(0)
        }
    }

    struct HostileDoubleErrorDeleteIo {
        fired: std::sync::atomic::AtomicBool,
    }

    impl HostileDoubleErrorDeleteIo {
        fn new() -> Self {
            Self {
                fired: std::sync::atomic::AtomicBool::new(false),
            }
        }
    }

    impl super::super::SpillIo for HostileDoubleErrorDeleteIo {
        fn check(&self, operation: super::super::SpillIoOperation) -> std::io::Result<()> {
            if operation == super::super::SpillIoOperation::Delete
                && !self.fired.swap(true, std::sync::atomic::Ordering::AcqRel)
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    DoublePanicError {
                        _nested: PanicOnDrop,
                    },
                ));
            }
            Ok(())
        }
    }

    #[derive(Debug)]
    struct PrimaryDropPanic;

    impl MemoryConsumer for PanickingEvictionConsumer {
        fn name(&self) -> &str {
            "external-sort-panicking-eviction"
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
            panic!("deterministic external-sort eviction panic");
        }

        fn current_tier(&self) -> StorageTier {
            StorageTier::InMemory
        }
    }

    /// Returns (TempDir, SpillManager). TempDir must be kept alive as long as manager is used.
    fn create_manager() -> (TempDir, Arc<SpillManager>) {
        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build()
                .unwrap(),
        );
        (temp_dir, manager)
    }

    fn qualification_trap_manager(
        directory: &std::path::Path,
        queries: Arc<AtomicUsize>,
        panic_on_query: bool,
    ) -> Arc<SpillManager> {
        Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory)
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(QualificationTrapIo {
                    queries,
                    panic_on_query,
                }))
                .build()
                .unwrap(),
        )
    }

    fn buffer_manager_with_exact_budget(budget: usize) -> Arc<BufferManager> {
        let mut config = BufferManagerConfig::with_budget(budget);
        config.soft_limit_fraction = 1.0;
        config.evict_limit_fraction = 1.0;
        config.hard_limit_fraction = 1.0;
        BufferManager::new(config)
    }

    fn qualified_writer_base_bytes(manager: &SpillManager) -> usize {
        checked_workspace_sum(
            qualified_writer_buffer_requested_bytes(),
            manager.qualified_file_workspace_bound().unwrap(),
        )
        .unwrap()
    }

    fn row(values: &[i64]) -> Vec<Value> {
        values.iter().map(|&v| Value::Int64(v)).collect()
    }

    fn framed_sort_payload(values: &[Value], ordinal: u64) -> Vec<u8> {
        let mut payload = Vec::new();
        serialize_row_with_limits(
            values,
            &mut payload,
            super::super::SpillFrameLimits::format_max().codec_limits(),
        )
        .unwrap();
        payload.extend_from_slice(&ordinal.to_le_bytes());
        payload.shrink_to_fit();
        payload
    }

    fn string_and_blob(values: &[Value]) -> (&ArcStr, &Arc<[u8]>) {
        let Value::String(text) = &values[0] else {
            panic!("row lost its string shape")
        };
        let Value::Bytes(blob) = &values[1] else {
            panic!("row lost its blob shape")
        };
        (text, blob)
    }

    fn provenance_row(key: i64, typed: bool) -> Vec<Value> {
        let mut values = vec![Value::Int64(key), Value::List(vec![Value::Int64(7)].into())];
        if typed {
            values.push(Value::Bytes(vec![4u8].into()));
        }
        values
    }

    fn scalar_provenance_row(key: i64) -> Vec<Value> {
        let kind = key % 4;
        let value = if kind == 3 {
            Value::List(vec![Value::Int64(7)].into())
        } else {
            Value::Int64(7)
        };
        let mut row = vec![
            Value::Int64(key),
            value.clone(),
            Value::Bytes(vec![0xff].into()),
            Value::Null,
            value,
        ];
        let mask = match kind {
            1 => Some([0b0000_1000, 0b0000_0011]), // Edge at1, Node at4
            2 => Some([0b0000_1100, 0b0000_0010]), // Node at1, Edge at4
            3 => Some([0b0000_0100, 0b0000_0001]), // List(Edge) at1 and4
            _ => None,
        };
        if let Some(mask) = mask {
            row.push(Value::Bytes(Arc::from(mask)));
        }
        row
    }

    #[test]
    fn scalar_provenance_decoder_validates_two_bit_width_padding_and_strict_default() {
        let shape = SortRowShape::with_edge_trailer(5);
        let limits = super::super::SpillFrameLimits::format_max();
        for key in 0..4 {
            let row = scalar_provenance_row(key);
            let decoded =
                decode_row_payload_with_shape(&framed_sort_payload(&row, 29), shape, limits)
                    .unwrap();
            assert_eq!(decoded.values, row);
            assert_eq!(decoded.ordinal, 29);
            if key != 0 {
                assert!(decode_row_payload(&framed_sort_payload(&row, 29), 5, limits).is_err());
            }
        }
        for tail in [
            Value::Null,
            Value::Bytes(vec![0, 0].into()),
            Value::Bytes(vec![8].into()),
            Value::Bytes(vec![8, 4].into()),
            Value::Bytes(vec![8, 3, 0].into()),
        ] {
            let mut row = scalar_provenance_row(0);
            row.push(tail);
            assert!(
                decode_row_payload_with_shape(&framed_sort_payload(&row, 0), shape, limits)
                    .is_err()
            );
        }
    }

    #[test]
    fn distinct_binary_runs_bound_actual_merge_work_as_cardinality_grows() {
        // These counts exercise the real framed merge implementation. The old
        // 17-to-16 prefix reducer exceeds the N*log2(N) bound already at N=128.
        for count in [32_u64, 128, 512] {
            let (_directory, manager) = create_manager();
            let buffers = buffer_manager_with_exact_budget(4 * 1024 * 1024);
            let resources = crate::execution::QueryResourceContext::new(buffers.clone()).unwrap();
            let state = Arc::new(OperatorSpillState::new("balanced DISTINCT runs".into()));
            let observer = ExternalSortGrantObserver::owned(state.clone());
            let mut sort = ExternalSort::new_accounted(
                manager.clone(),
                2,
                vec![SortKey::ascending(0)],
                resources.try_allocate(0).unwrap(),
            );
            for ordinal in 0..count {
                let values = vec![
                    Value::Int64(i64::try_from(ordinal % 7).unwrap()),
                    Value::Int64(i64::try_from(ordinal).unwrap()),
                ];
                sort.spill_distinct_run_accounted(&[values], &observer)
                    .unwrap();
                assert_eq!(sort.num_runs(), (ordinal + 1).count_ones() as usize);
                assert!(sort.num_runs() <= 64);
                assert_eq!(state.usage(), sort.total_granted_bytes());
            }
            let expected_visits = u128::from(count) * u128::from(count.ilog2());
            assert_eq!(sort.merged_row_visits, expected_visits, "N={count}");
            sort.finish_distinct_runs_accounted(&observer).unwrap();
            assert_eq!(sort.merged_row_visits, expected_visits);
            assert!(sort.try_enable_exact_owned_output(Some(&state)));
            let mut cursor = sort.into_send_owned_disk_cursor(observer).unwrap();
            for key in 0..7_u64 {
                for ordinal in (key..count).step_by(7) {
                    let row = cursor.next_owned_row().unwrap().unwrap();
                    assert_eq!(
                        row.values(),
                        &[
                            Value::Int64(i64::try_from(key).unwrap()),
                            Value::Int64(i64::try_from(ordinal).unwrap()),
                        ]
                    );
                    drop(row.into_released_grant());
                    cursor.release_transferred_retained().unwrap();
                }
            }
            assert!(cursor.next_owned_row().unwrap().is_none());
            drop(cursor);
            assert_eq!(manager.active_file_count(), 0);
            assert_eq!(state.usage(), 0);
            assert_eq!(buffers.allocated(), 0);
        }
    }

    #[test]
    fn distinct_binary_runs_reduce_only_at_eof_and_preserve_unequal_batch_order() {
        let (_directory, manager) = create_manager();
        let buffers = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new(buffers.clone()).unwrap();
        let state = Arc::new(OperatorSpillState::new("unequal DISTINCT batches".into()));
        let observer = ExternalSortGrantObserver::owned(state.clone());
        let mut sort = ExternalSort::new_accounted(
            manager.clone(),
            1,
            vec![SortKey::ascending(0)],
            resources.try_allocate(0).unwrap(),
        );
        sort.set_merge_fan_in(2);
        let mut total_rows = 0_u128;
        for batch in 0..63_usize {
            let rows: Vec<_> = (0..=(batch % 5)).map(|_| vec![Value::Int64(1)]).collect();
            total_rows += rows.len() as u128;
            sort.spill_distinct_run_accounted(&rows, &observer).unwrap();
        }
        assert_eq!(sort.num_runs(), 6);
        assert!(sort.merged_row_visits <= total_rows * 5);
        sort.finish_distinct_runs_accounted(&observer).unwrap();
        assert_eq!(sort.num_runs(), 2);
        assert!(sort.merged_row_visits <= total_rows * 8);
        let visits = sort.merged_row_visits;
        sort.finish_distinct_runs_accounted(&observer).unwrap();
        assert_eq!(sort.merged_row_visits, visits, "EOF sealing is idempotent");
        assert!(
            sort.spill_distinct_run_accounted(&[vec![Value::Int64(2)]], &observer)
                .is_err()
        );
        assert_eq!(sort.merged_row_visits, visits);
        assert!(sort.try_enable_exact_owned_output(Some(&state)));
        let mut cursor = sort.into_send_owned_disk_cursor(observer).unwrap();
        for ordinal in 0..total_rows {
            let row = cursor.next_owned_row().unwrap().unwrap();
            assert_eq!(u128::from(row.ordinal()), ordinal);
            drop(row.into_released_grant());
            cursor.release_transferred_retained().unwrap();
        }
        assert!(cursor.next_owned_row().unwrap().is_none());
        drop(cursor);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(state.usage(), 0);
        assert_eq!(buffers.allocated(), 0);
    }

    #[test]
    fn pull_initial_fan_in_defers_carries_and_preserves_stable_unequal_batches() {
        for batches in [3usize, 4, 7, 8] {
            let (_directory, manager) = create_manager();
            let buffers = buffer_manager_with_exact_budget(4 << 20);
            let resources = crate::execution::QueryResourceContext::new(buffers.clone()).unwrap();
            let state = Arc::new(OperatorSpillState::new("deferred pull batches".into()));
            let observer = ExternalSortGrantObserver::owned(state.clone());
            let mut sort = ExternalSort::new_accounted(
                manager.clone(),
                1,
                vec![SortKey::ascending(0)],
                resources.try_allocate(0).unwrap(),
            );
            sort.set_merge_fan_in(4);
            let mut total = 0u64;
            for batch in 0..batches {
                let rows: Vec<_> = (0..=(batch % 3)).map(|_| vec![Value::Int64(1)]).collect();
                total += rows.len() as u64;
                sort.spill_pull_run_accounted(&rows, &observer).unwrap();
                match batch.cmp(&3) {
                    std::cmp::Ordering::Less => {
                        assert_eq!(sort.num_runs(), batch + 1);
                        assert_eq!(sort.merged_row_visits, 0);
                    }
                    std::cmp::Ordering::Equal => {
                        assert_eq!(sort.num_runs(), 1);
                        assert_eq!(sort.merged_row_visits, u128::from(total));
                    }
                    std::cmp::Ordering::Greater => {
                        assert_eq!(sort.num_runs(), (batch + 1).count_ones() as usize);
                    }
                }
                if batch == 0 {
                    assert!(
                        sort.spill_distinct_run_accounted(&[row(&[1])], &observer)
                            .is_err()
                    );
                    sort.set_merge_fan_in(8);
                    assert!(
                        sort.spill_pull_run_accounted(&[row(&[1])], &observer)
                            .is_err()
                    );
                    sort.set_merge_fan_in(4);
                    assert_eq!(sort.num_runs(), 1);
                }
            }
            sort.finish_distinct_runs_accounted(&observer).unwrap();
            assert!(sort.try_enable_exact_owned_output(Some(&state)));
            let mut cursor = sort.into_send_owned_disk_cursor(observer).unwrap();
            for ordinal in 0..total {
                let row = cursor.next_owned_row().unwrap().unwrap();
                assert_eq!(row.ordinal(), ordinal);
                assert_eq!(row.values(), &[Value::Int64(1)]);
                drop(row.into_released_grant());
                cursor.release_transferred_retained().unwrap();
            }
            assert!(cursor.next_owned_row().unwrap().is_none());
            drop(cursor);
            assert_eq!(manager.active_file_count(), 0);
            assert_eq!(state.usage(), 0);
            assert_eq!(buffers.allocated(), 0);
        }
    }

    #[test]
    fn pull_initial_fan_in_write_failure_is_not_replayable() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(FailWriteAndToggleDeleteIo::failing());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(io.clone() as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffers = buffer_manager_with_exact_budget(4 << 20);
        let resources = crate::execution::QueryResourceContext::new(buffers.clone()).unwrap();
        let state = Arc::new(OperatorSpillState::new("failed deferred pull".into()));
        let observer = ExternalSortGrantObserver::owned(state);
        let mut sort = ExternalSort::new_accounted(
            manager.clone(),
            1,
            vec![SortKey::ascending(0)],
            resources.try_allocate(0).unwrap(),
        );
        sort.set_merge_fan_in(4);
        let error = sort
            .spill_pull_run_accounted(&[row(&[1])], &observer)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("deterministic writer primary failure")
        );
        assert!(matches!(
            sort.distinct_run_schedule,
            DistinctRunSchedule::Failed
        ));
        assert!(
            sort.spill_pull_run_accounted(&[row(&[1])], &observer)
                .is_err()
        );
        io.permit_delete();
        sort.cleanup_distinct_accounted().unwrap();
        drop(error);
        drop(sort);
        // Failed unpublished writer deletion belongs to the manager ledger,
        // not the sort run catalog; retry its retained cleanup authority.
        assert_eq!(manager.active_file_count(), 1);
        manager.cleanup().unwrap();
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        assert_eq!(buffers.allocated(), 0);
    }

    #[test]
    fn pull_initial_fan_in_cancellation_preserves_owned_cleanup() {
        let (_directory, manager) = create_manager();
        let control = crate::execution::QueryExecutionControl::new();
        let buffers = buffer_manager_with_exact_budget(4 << 20);
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            buffers.clone(),
            control.token(),
        )
        .unwrap();
        let state = Arc::new(OperatorSpillState::new("cancel deferred pull".into()));
        let observer = ExternalSortGrantObserver::owned(state.clone());
        let mut sort = ExternalSort::new_accounted_with_cancellation(
            manager.clone(),
            1,
            vec![SortKey::ascending(0)],
            resources.try_allocate(0).unwrap(),
            control.token(),
        );
        sort.set_merge_fan_in(4);
        for _ in 0..3 {
            sort.spill_pull_run_accounted(&[row(&[1])], &observer)
                .unwrap();
        }
        control.cancellation_handle().cancel();
        assert!(matches!(
            sort.spill_pull_run_accounted(&[row(&[1])], &observer),
            Err(ExternalSortOperationError::Cancelled(_))
        ));
        assert_eq!(sort.num_runs(), 3);
        assert_eq!(sort.merged_row_visits, 0);
        sort.cleanup_distinct_accounted().unwrap();
        drop(sort);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(buffers.allocated(), 0);
    }

    #[test]
    fn send_owned_exact_cursor_restores_authority_across_drain_stop_error_and_unwind() {
        fn assert_send<T: Send>() {}
        assert_send::<OwnedExactSortCursor>();
        for mode in 0..4 {
            let (_directory, manager) = create_manager();
            let buffers = buffer_manager_with_exact_budget(4 * 1024 * 1024);
            let resources = crate::execution::QueryResourceContext::new(buffers.clone()).unwrap();
            let state = Arc::new(OperatorSpillState::new("owned exact cursor".into()));
            let mut sort = ExternalSort::new_accounted(
                manager.clone(),
                1,
                vec![SortKey::ascending(0)],
                resources.try_allocate(0).unwrap(),
            );
            sort.spill_sorted_run_accounted(&[row(&[1]), row(&[3])])
                .unwrap();
            sort.spill_sorted_run_accounted(&[row(&[2])]).unwrap();
            assert!(sort.try_enable_exact_owned_output(Some(&state)));
            let mut cursor = sort
                .into_send_owned_disk_cursor(ExternalSortGrantObserver::owned(state.clone()))
                .unwrap();
            assert_eq!(cursor.num_columns().unwrap(), 1);
            assert_eq!(state.usage(), cursor.checked_granted_bytes().unwrap());
            let first = cursor.next_owned_row().unwrap().unwrap();
            assert_eq!(first.values(), &[Value::Int64(1)]);
            assert_eq!(
                state.usage(),
                cursor.checked_granted_bytes().unwrap() + first.granted_bytes()
            );
            drop(first.into_released_grant());
            cursor.release_transferred_retained().unwrap();
            match mode {
                0 => {
                    for expected in [2, 3] {
                        let next = cursor.next_owned_row().unwrap().unwrap();
                        assert_eq!(next.values(), &[Value::Int64(expected)]);
                        drop(next.into_released_grant());
                        cursor.release_transferred_retained().unwrap();
                    }
                    assert!(cursor.next_owned_row().unwrap().is_none());
                }
                1 => cursor.finish_early_stop().unwrap(),
                2 => {
                    let primary = OperatorError::UnsupportedAccountedTransport { consumer: "test" };
                    assert!(matches!(
                        cursor.finish_operator_failure(primary, None, "owned cursor test cleanup"),
                        OperatorError::UnsupportedAccountedTransport { consumer: "test" }
                    ));
                }
                _ => {
                    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        cursor
                            .with_cursor(|inner| {
                                inner.state = ExactOwnedCursorState::Failed;
                                std::panic::panic_any(17u32);
                            })
                            .unwrap();
                    }))
                    .unwrap_err();
                    assert_eq!(*panic.downcast::<u32>().unwrap(), 17);
                    assert_eq!(
                        cursor.storage.as_ref().unwrap().state,
                        ExactOwnedCursorState::Failed
                    );
                    assert_eq!(state.usage(), cursor.checked_granted_bytes().unwrap());
                    cursor.finish_early_stop().unwrap();
                }
            }
            drop(cursor);
            assert_eq!(state.usage(), 0);
            assert_eq!(buffers.allocated(), 0);
            assert_eq!(manager.active_file_count(), 0);
        }
    }

    #[test]
    fn scalar_provenance_two_bit_trailer_survives_multipass_and_exact_cursor() {
        for mode in 0..3 {
            let (_directory, manager) = create_manager();
            let buffers = buffer_manager_with_exact_budget(64 * 1024 * 1024);
            let resources = crate::execution::QueryResourceContext::new(buffers.clone()).unwrap();
            let mut sort = if mode == 0 {
                ExternalSort::new(manager.clone(), 5, vec![SortKey::ascending(0)])
            } else {
                ExternalSort::new_accounted(
                    manager.clone(),
                    5,
                    vec![SortKey::ascending(0)],
                    resources.try_allocate(0).unwrap(),
                )
            };
            sort.enable_edge_provenance();
            sort.set_merge_fan_in(2);
            if mode == 2 {
                // Exact owned output admits only an already bounded final
                // frontier. The production caller uses the multipass fallback
                // for larger frontiers; exercise that separately below.
                // Interleaved runs still force every cursor head to refill
                // across ordinary, Edge, Node and List(Edge) row shapes.
                for parity in 0..2 {
                    sort.spill_sorted_run(
                        (parity..9).step_by(2).map(scalar_provenance_row).collect(),
                    )
                    .unwrap();
                }
                assert_eq!(sort.num_runs(), 2);
                assert!(sort.exact_owned_disk_shape_eligible());
            } else {
                for key in (0..9).rev() {
                    sort.spill_sorted_run(vec![scalar_provenance_row(key)])
                        .unwrap();
                }
                assert_eq!(sort.num_runs(), 9, "must require intermediate merge passes");
            }
            let expected: Vec<_> = (0..9).map(scalar_provenance_row).collect();
            let actual = if mode == 2 {
                assert!(sort.try_enable_exact_owned_output(None));
                let external = Cell::new(sort.total_granted_bytes());
                let retained = Cell::new(0);
                let observer = ExternalSortGrantObserver::new(&external, &retained, None);
                let mut cursor = sort.into_exact_owned_disk_cursor(observer).unwrap();
                let mut actual = Vec::new();
                while let Some(row) = cursor.next_owned_row().unwrap() {
                    actual.push(row.values().to_vec());
                    assert_eq!(retained.get(), row.granted_bytes());
                    drop(row);
                    cursor.release_transferred_retained().unwrap();
                }
                drop(cursor);
                actual
            } else {
                let rows = if mode == 0 {
                    sort.merge_all(Vec::new()).unwrap()
                } else {
                    sort.merge_all_accounted(Vec::new()).unwrap()
                };
                drop(sort);
                rows
            };
            assert_eq!(actual, expected);
            assert_eq!(manager.active_file_count(), 0);
            assert_eq!(buffers.allocated(), 0);
        }
    }

    #[test]
    fn sort_provenance_decoder_rejects_malformed_trailers_and_strict_width() {
        let shape = SortRowShape::with_edge_trailer(2);
        let limits = super::super::SpillFrameLimits::format_max();
        let ordinary = provenance_row(1, false);
        let typed = provenance_row(1, true);
        for values in [&ordinary, &typed] {
            let payload = framed_sort_payload(values, 19);
            let decoded = decode_row_payload_with_shape(&payload, shape, limits).unwrap();
            assert_eq!(&decoded.values, values);
            assert_eq!(decoded.ordinal, 19);
        }
        assert!(decode_row_payload(&framed_sort_payload(&typed, 19), 2, limits).is_err());
        for tail in [
            Value::Null,
            Value::Bytes(Vec::new().into()),
            Value::Bytes(vec![0].into()),
            Value::Bytes(vec![16].into()),
            Value::Bytes(vec![4, 0].into()),
        ] {
            let mut invalid = ordinary.clone();
            invalid.push(tail);
            assert!(
                decode_row_payload_with_shape(&framed_sort_payload(&invalid, 0), shape, limits)
                    .is_err()
            );
        }
        let mut invalid_width = typed.clone();
        invalid_width.push(Value::Null);
        assert!(
            decode_row_payload_with_shape(&framed_sort_payload(&invalid_width, 0), shape, limits)
                .is_err()
        );
        let (_directory, manager) = create_manager();
        let mut strict = ExternalSort::new(manager, 2, vec![SortKey::ascending(0)]);
        assert!(strict.spill_sorted_run(vec![typed.clone()]).is_err());
        assert_eq!(strict.num_runs(), 0);
        assert!(strict.merge_all(vec![typed]).is_err());
    }

    #[test]
    fn sort_provenance_survives_ordinary_and_accounted_multipass_merge() {
        for accounted in [false, true] {
            let (_directory, manager) = create_manager();
            let buffers = buffer_manager_with_exact_budget(64 * 1024 * 1024);
            let resources = crate::execution::QueryResourceContext::new(buffers.clone()).unwrap();
            let mut sort = if accounted {
                ExternalSort::new_accounted(
                    manager.clone(),
                    2,
                    vec![SortKey::ascending(0)],
                    resources.try_allocate(0).unwrap(),
                )
            } else {
                ExternalSort::new(manager.clone(), 2, vec![SortKey::ascending(0)])
            };
            sort.enable_edge_provenance();
            sort.set_merge_fan_in(2);
            for key in (0..9).rev() {
                sort.spill_sorted_run(vec![provenance_row(key, key % 2 == 0)])
                    .unwrap();
            }
            assert_eq!(sort.num_runs(), 9);
            let expected: Vec<_> = (0..10)
                .map(|key| provenance_row(key, key % 2 == 0))
                .collect();
            let tail = vec![provenance_row(9, false)];
            let actual = if accounted {
                sort.merge_all_accounted(tail).unwrap()
            } else {
                sort.merge_all(tail).unwrap()
            };
            assert_eq!(actual, expected);
            assert_eq!(manager.active_file_count(), 0);
            drop(sort);
            assert_eq!(buffers.allocated(), 0);
        }
    }

    #[test]
    fn sort_provenance_exact_cursor_resumes_with_mixed_physical_widths() {
        let (_directory, manager) = create_manager();
        let buffers = buffer_manager_with_exact_budget(64 * 1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new(buffers.clone()).unwrap();
        let mut sort = ExternalSort::new_accounted(
            manager.clone(),
            2,
            vec![SortKey::ascending(0)],
            resources.try_allocate(0).unwrap(),
        );
        sort.enable_edge_provenance();
        // Every resumed reader alternates logical and physical width.
        sort.spill_sorted_run_accounted(&[
            provenance_row(0, true),
            provenance_row(2, false),
            provenance_row(4, true),
        ])
        .unwrap();
        sort.spill_sorted_run_accounted(&[
            provenance_row(1, false),
            provenance_row(3, true),
            provenance_row(5, false),
        ])
        .unwrap();
        assert!(sort.try_enable_exact_owned_output(None));
        let external = Cell::new(sort.total_granted_bytes());
        let retained = Cell::new(0);
        let observer = ExternalSortGrantObserver::new(&external, &retained, None);
        let mut cursor = sort.into_exact_owned_disk_cursor(observer).unwrap();
        for (key, typed) in [
            (0, true),
            (1, false),
            (2, false),
            (3, true),
            (4, true),
            (5, false),
        ] {
            let row = cursor.next_owned_row().unwrap().unwrap();
            assert_eq!(row.values(), provenance_row(key, typed));
            assert_eq!(retained.get(), row.granted_bytes());
            // Suspending here must preserve row shape on the next head refill.
            assert!(buffers.allocated() >= row.granted_bytes());
            drop(row);
            cursor.release_transferred_retained().unwrap();
        }
        assert!(cursor.next_owned_row().unwrap().is_none());
        drop(cursor);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(buffers.allocated(), 0);
    }

    #[test]
    fn test_external_sort_empty() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);

        let result = sort.merge_all(Vec::new()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_external_sort_memory_only() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);

        let buffer = vec![row(&[3]), row(&[1]), row(&[2])];
        let result = sort.merge_all(buffer).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], row(&[1]));
        assert_eq!(result[1], row(&[2]));
        assert_eq!(result[2], row(&[3]));
    }

    #[test]
    fn semantic_comparator_and_input_ordinal_order_three_runs() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new_with_comparator(manager, 2, |left, right| {
            let magnitude = |value: &Value| match value {
                Value::Int64(value) => value.unsigned_abs(),
                other => panic!("expected integer sort key, found {other:?}"),
            };
            magnitude(&left[0]).cmp(&magnitude(&right[0]))
        });

        sort.spill_sorted_run(vec![row(&[-1, 0]), row(&[2, 1])])
            .unwrap();
        sort.spill_sorted_run(vec![row(&[1, 2]), row(&[3, 3])])
            .unwrap();
        sort.spill_sorted_run(vec![row(&[-1, 4]), row(&[2, 5])])
            .unwrap();

        assert_eq!(
            sort.merge_all(Vec::new()).unwrap(),
            vec![
                row(&[-1, 0]),
                row(&[1, 2]),
                row(&[-1, 4]),
                row(&[2, 1]),
                row(&[2, 5]),
                row(&[3, 3]),
            ]
        );
    }

    #[test]
    fn descending_equal_keys_keep_input_order_through_multipass_merge() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 2, vec![SortKey::descending(0)]);
        sort.set_merge_fan_in(3);
        sort.spill_sorted_run(vec![row(&[9, 0]), row(&[8, 0])])
            .unwrap();
        sort.spill_sorted_run(vec![row(&[9, 1]), row(&[8, 1])])
            .unwrap();
        sort.spill_sorted_run(vec![row(&[9, 2]), row(&[8, 2])])
            .unwrap();
        sort.spill_sorted_run(vec![row(&[9, 3]), row(&[8, 3])])
            .unwrap();

        assert_eq!(
            sort.merge_all(Vec::new()).unwrap(),
            vec![
                row(&[9, 0]),
                row(&[9, 1]),
                row(&[9, 2]),
                row(&[9, 3]),
                row(&[8, 0]),
                row(&[8, 1]),
                row(&[8, 2]),
                row(&[8, 3]),
            ]
        );
    }

    #[test]
    fn input_ordinal_exhaustion_precedes_file_or_catalog_publication() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(Arc::clone(&manager), 1, vec![SortKey::ascending(0)]);
        sort.next_input_ordinal = u64::MAX;

        let error = sort.spill_sorted_run(vec![row(&[1])]).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::OutOfMemory);
        assert!(error.to_string().contains("input ordinal"));
        assert_eq!(sort.next_input_ordinal, u64::MAX);
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
    }

    #[test]
    fn missing_durable_input_ordinal_fails_closed() {
        let error =
            decode_row_payload(&[], 1, super::super::SpillFrameLimits::format_max()).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("missing its input ordinal"));
    }

    #[test]
    fn accounted_decoded_row_shrinks_large_payloads_below_construction_envelope() {
        for value in [
            Value::String(ArcStr::from("s".repeat(128 * 1024))),
            Value::Bytes(Arc::from(vec![0xabu8; 128 * 1024])),
        ] {
            let payload = framed_sort_payload(&[value], 41);
            let value_frame_len = payload.len() - std::mem::size_of::<u64>();
            let broad = conservative_decoded_row_retained_bytes(value_frame_len).unwrap();
            let peak = payload.capacity().checked_add(broad).unwrap();
            let buffer_manager = buffer_manager_with_exact_budget(peak);
            let resources =
                crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
            let payload_capacity = payload.capacity();
            let grant = resources.try_allocate(payload_capacity).unwrap();

            let decoded = decode_accounted_sort_row(
                payload,
                1,
                super::super::SpillFrameLimits::format_max(),
                grant,
            )
            .unwrap();

            assert_eq!(decoded.ordinal(), 41);
            assert_eq!(decoded.values().len(), 1);
            assert!(decoded.granted_bytes() < broad / 16);
            assert_eq!(buffer_manager.allocated(), decoded.granted_bytes());
            drop(decoded);
            assert_eq!(buffer_manager.allocated(), 0);
        }
    }

    #[test]
    fn accounted_decoded_row_scalar_retains_observed_top_level_vec_capacity() {
        let payload = framed_sort_payload(&[Value::Int64(17)], 12);
        let value_frame_len = payload.len() - std::mem::size_of::<u64>();
        let broad = conservative_decoded_row_retained_bytes(value_frame_len).unwrap();
        let peak = payload.capacity().checked_add(broad).unwrap();
        let buffer_manager = buffer_manager_with_exact_budget(peak);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(payload.capacity()).unwrap();

        let decoded = decode_accounted_sort_row(
            payload,
            1,
            super::super::SpillFrameLimits::format_max(),
            grant,
        )
        .unwrap();
        let top_level_capacity = decoded.top_level_capacity_bytes();

        assert!(top_level_capacity > 0);
        assert_eq!(decoded.granted_bytes(), top_level_capacity);
        assert_eq!(buffer_manager.allocated(), top_level_capacity);
    }

    #[test]
    fn accounted_decoded_row_equal_frames_own_independent_arc_payloads() {
        let source_text = ArcStr::from("independent decoder text".repeat(257));
        let source_blob: Arc<[u8]> = Arc::from(vec![0x5au8; 8 * 1024]);
        let values = [
            Value::String(source_text.clone()),
            Value::Bytes(Arc::clone(&source_blob)),
        ];
        let first_payload = framed_sort_payload(&values, 7);
        let second_payload = framed_sort_payload(&values, 7);
        let first_capacity = first_payload.capacity();
        let second_capacity = second_payload.capacity();
        let value_frame_len = first_payload.len() - std::mem::size_of::<u64>();
        let broad = conservative_decoded_row_retained_bytes(value_frame_len).unwrap();
        let budget = first_capacity
            .checked_add(second_capacity)
            .and_then(|bytes| bytes.checked_add(broad.checked_mul(2)?))
            .unwrap();
        let buffer_manager = buffer_manager_with_exact_budget(budget);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let first_grant = resources.try_allocate(first_capacity).unwrap();
        let second_grant = resources.try_allocate(second_capacity).unwrap();

        let first = decode_accounted_sort_row(
            first_payload,
            values.len(),
            super::super::SpillFrameLimits::format_max(),
            first_grant,
        )
        .unwrap();
        let second = decode_accounted_sort_row(
            second_payload,
            values.len(),
            super::super::SpillFrameLimits::format_max(),
            second_grant,
        )
        .unwrap();
        let (first_text, first_blob) = string_and_blob(first.values());
        let (second_text, second_blob) = string_and_blob(second.values());

        assert!(!ArcStr::ptr_eq(&source_text, first_text));
        assert!(!ArcStr::ptr_eq(&source_text, second_text));
        assert!(!ArcStr::ptr_eq(first_text, second_text));
        assert!(!Arc::ptr_eq(&source_blob, first_blob));
        assert!(!Arc::ptr_eq(&source_blob, second_blob));
        assert!(!Arc::ptr_eq(first_blob, second_blob));
        assert_eq!(first.granted_bytes(), second.granted_bytes());
    }

    #[test]
    fn accounted_decoded_row_retains_authority_after_decode_scope_and_context_drop() {
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let retained = {
            let resources =
                crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
            let payload = framed_sort_payload(
                &[Value::List(Arc::from([
                    Value::String(ArcStr::from("survives its decode cursor")),
                    Value::Bytes(Arc::from([9_u8; 257])),
                ]))],
                99,
            );
            let grant = resources.try_allocate(payload.capacity()).unwrap();
            let decoded = decode_accounted_sort_row(
                payload,
                1,
                super::super::SpillFrameLimits::format_max(),
                grant,
            )
            .unwrap();
            drop(resources);
            decoded
        };
        let retained_bytes = retained.granted_bytes();

        assert!(retained_bytes > 257);
        assert_eq!(retained.ordinal(), 99);
        assert_eq!(buffer_manager.allocated(), retained_bytes);
        assert_eq!(retained.values().len(), 1);
        drop(retained);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_decoded_row_rejects_retained_bytes_beyond_broad_envelope() {
        let error = checked_accounted_decoded_retained_bytes(65, 0, 64).unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::InvalidData
        ));
    }

    #[test]
    fn accounted_decoded_row_rejects_uncovered_payload_and_rolls_back_malformed_frame() {
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let payload = framed_sort_payload(&[Value::String(ArcStr::from("covered"))], 3);
        let payload_capacity = payload.capacity();
        let short_grant = resources.try_allocate(payload_capacity - 1).unwrap();

        let uncovered = decode_accounted_sort_row(
            payload,
            1,
            super::super::SpillFrameLimits::format_max(),
            short_grant,
        )
        .unwrap_err();

        assert!(matches!(
            uncovered,
            ExternalSortOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::InvalidInput
        ));
        assert_eq!(buffer_manager.allocated(), 0);

        let mut malformed = framed_sort_payload(&[Value::String(ArcStr::from("malformed"))], 5);
        malformed[std::mem::size_of::<u64>()] = 0xff;
        let malformed_grant = resources.try_allocate(malformed.capacity()).unwrap();
        let error = decode_accounted_sort_row(
            malformed,
            1,
            super::super::SpillFrameLimits::format_max(),
            malformed_grant,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::InvalidData
        ));
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_decoded_row_is_not_clone() {
        trait AmbiguousIfClone<Marker> {
            fn marker() {}
        }
        impl<T: ?Sized> AmbiguousIfClone<()> for T {}
        impl<T: Clone> AmbiguousIfClone<u8> for T {}

        let _ = <AccountedOrdinalRow as AmbiguousIfClone<_>>::marker;
    }

    #[test]
    fn compatibility_cursor_remains_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ExternalSortCursor<'static>>();
    }

    #[test]
    fn merge_cursor_yields_only_the_requested_rows_per_chunk() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);
        sort.spill_sorted_run(vec![row(&[1]), row(&[4])]).unwrap();
        sort.spill_sorted_run(vec![row(&[2]), row(&[5])]).unwrap();
        sort.spill_sorted_run(vec![row(&[3]), row(&[6])]).unwrap();

        let mut cursor = sort.merge_cursor(Vec::new(), 2).unwrap();
        assert_eq!(
            cursor.next_chunk().unwrap().unwrap().rows(),
            &[row(&[1]), row(&[2])]
        );
        assert_eq!(
            cursor.next_chunk().unwrap().unwrap().rows(),
            &[row(&[3]), row(&[4])]
        );
        assert_eq!(
            cursor.next_chunk().unwrap().unwrap().rows(),
            &[row(&[5]), row(&[6])]
        );
        assert!(cursor.next_chunk().unwrap().is_none());
    }

    #[test]
    fn compatibility_cursor_does_not_require_qualified_provider_bounds() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(LegacyOnlyProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .build()
                .unwrap(),
        );
        let mut sort = ExternalSort::new(Arc::clone(&manager), 1, vec![SortKey::ascending(0)]);
        sort.spill_sorted_run(vec![row(&[1]), row(&[3])]).unwrap();
        sort.spill_sorted_run(vec![row(&[2]), row(&[4])]).unwrap();

        let mut cursor = sort.merge_cursor(Vec::new(), 2).unwrap();
        assert_eq!(
            cursor.next_chunk().unwrap().unwrap().rows(),
            &[row(&[1]), row(&[2])]
        );
        assert_eq!(
            cursor.next_chunk().unwrap().unwrap().rows(),
            &[row(&[3]), row(&[4])]
        );
        assert!(cursor.next_chunk().unwrap().is_none());
        drop(cursor);
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    fn accounted_cursor_bounds_and_reuses_one_output_grant() {
        let (_temp_dir, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        for values in [[1, 4], [2, 5], [3, 6]] {
            sort.spill_sorted_run_accounted(&[row(&[values[0]]), row(&[values[1]])])
                .unwrap();
        }

        let observer = ExternalSortGrantObserver::inert();
        let mut cursor = sort
            .merge_cursor_accounted_observing(Vec::new(), 2, observer)
            .unwrap();
        assert_eq!(cursor.cursor.output_rows.capacity(), 2);
        assert!(cursor.cursor.sorter.frontier_granted_bytes() > 0);
        assert_eq!(cursor.cursor.sorter.payload_granted_bytes(), 0);
        for expected in [[1, 2], [3, 4], [5, 6]] {
            let chunk = cursor.next_chunk_accounted_observing().unwrap().unwrap();
            assert_eq!(chunk.rows(), &[row(&[expected[0]]), row(&[expected[1]])]);
            assert_eq!(cursor.cursor.output_rows.len(), 2);
            assert_eq!(
                cursor.cursor.sorter.output_granted_bytes(),
                cursor.cursor.output_base_bytes + cursor.cursor.output_row_bytes
            );
            assert_eq!(cursor.cursor.sorter.payload_granted_bytes(), 0);
            assert_eq!(
                buffer_manager.allocated(),
                cursor.cursor.sorter.total_granted_bytes()
            );
        }
        assert!(cursor.next_chunk_accounted_observing().unwrap().is_none());
        assert_eq!(cursor.cursor.output_rows.capacity(), 0);
        assert_eq!(cursor.cursor.heap.capacity(), 0);
        assert_eq!(cursor.cursor.run_readers.capacity(), 0);
        assert_eq!(cursor.cursor.sorter.output_granted_bytes(), 0);
        assert_eq!(cursor.cursor.sorter.frontier_granted_bytes(), 0);
        drop(cursor);
        assert_eq!(sort.total_granted_bytes(), sort.run_catalog_granted_bytes());
        assert_eq!(manager.active_file_count(), 0);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn cursor_tail_ordinal_denial_precedes_allocation_and_preserves_ordinal() {
        let (_temp_dir, manager) = create_manager();
        let output_bytes = cursor_capacity_bytes::<Vec<Value>>(1).unwrap();
        let buffer_manager = buffer_manager_with_exact_budget(output_bytes);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(manager, 1, vec![SortKey::ascending(0)], grant);

        let observer = ExternalSortGrantObserver::inert();
        let error = sort
            .merge_cursor_accounted_observing(vec![row(&[2]), row(&[1])], 1, observer)
            .unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Memory(MemoryGrantError::LimitExceeded { .. })
        ));
        assert_eq!(sort.next_input_ordinal, 0);
        assert_eq!(sort.total_granted_bytes(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn cursor_payload_denial_precedes_record_payload_allocation() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(CountingReadPayloadIo::default());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let budget = 4 * 1024 * 1024;
        let buffer_manager = buffer_manager_with_exact_budget(budget);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
        let stable = sort.total_granted_bytes();
        let output = cursor_capacity_bytes::<Vec<Value>>(1).unwrap();
        let heap_calibration = ComparatorMinHeap::<HeapEntry>::try_with_exact_capacity(1).unwrap();
        let mut reader_calibration: Vec<Option<CursorRunReader>> = Vec::new();
        reader_calibration.try_reserve_exact(1).unwrap();
        let frontier = checked_workspace_sum(
            cursor_capacity_bytes::<HeapEntry>(heap_calibration.capacity()).unwrap(),
            cursor_capacity_bytes::<Option<CursorRunReader>>(reader_calibration.capacity())
                .unwrap(),
        )
        .unwrap();
        let reader_workspace = manager.qualified_file_workspace_bound().unwrap();
        let blocker_bytes = budget
            .checked_sub(stable)
            .and_then(|bytes| bytes.checked_sub(output))
            .and_then(|bytes| bytes.checked_sub(frontier))
            .and_then(|bytes| bytes.checked_sub(reader_workspace))
            .unwrap();
        let blocker = resources.try_allocate(blocker_bytes).unwrap();

        let observer = ExternalSortGrantObserver::inert();
        let error = sort
            .merge_cursor_accounted_observing(Vec::new(), 1, observer)
            .unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Memory(MemoryGrantError::LimitExceeded { .. })
        ));
        assert_eq!(
            io.reads.load(AtomicOrdering::Relaxed),
            2,
            "FileStart and SortRunStart may read; denied SortRow must not"
        );
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(sort.payload_granted_bytes(), 0);
        assert_eq!(manager.active_file_count(), 0);
        drop(blocker);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn cursor_reader_control_denial_precedes_control_payload() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(CountingReadPayloadIo::default());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let budget = 4 * 1024 * 1024;
        let buffer_manager = buffer_manager_with_exact_budget(budget);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
        let stable = sort.total_granted_bytes();
        let output = cursor_capacity_bytes::<Vec<Value>>(1).unwrap();
        let heap_calibration = ComparatorMinHeap::<HeapEntry>::try_with_exact_capacity(1).unwrap();
        let mut reader_calibration: Vec<Option<CursorRunReader>> = Vec::new();
        reader_calibration.try_reserve_exact(1).unwrap();
        let frontier = checked_workspace_sum(
            cursor_capacity_bytes::<HeapEntry>(heap_calibration.capacity()).unwrap(),
            cursor_capacity_bytes::<Option<CursorRunReader>>(reader_calibration.capacity())
                .unwrap(),
        )
        .unwrap();
        let blocker_bytes = budget
            .checked_sub(stable)
            .and_then(|bytes| bytes.checked_sub(output))
            .and_then(|bytes| bytes.checked_sub(frontier))
            .unwrap();
        let blocker = resources.try_allocate(blocker_bytes).unwrap();

        let observer = ExternalSortGrantObserver::inert();
        let error = sort
            .merge_cursor_accounted_observing(Vec::new(), 1, observer)
            .unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Memory(MemoryGrantError::LimitExceeded { .. })
        ));
        assert_eq!(io.opens.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(
            io.reads.load(AtomicOrdering::Relaxed),
            0,
            "fixed-control workspace must be admitted before reading FileStart"
        );
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(sort.payload_granted_bytes(), 0);
        assert_eq!(manager.active_file_count(), 0);
        drop(blocker);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn cursor_multipass_reader_denial_precedes_intermediate_reader_open() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(CountingReadPayloadIo::default());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let budget = 4 * 1024 * 1024;
        let buffer_manager = buffer_manager_with_exact_budget(budget);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.set_merge_fan_in(3);
        for value in 0..5 {
            sort.spill_sorted_run_accounted(&[row(&[value])]).unwrap();
        }
        sort.release_memory_workspaces().unwrap();

        let stable = sort.total_granted_bytes();
        let output = cursor_capacity_bytes::<Vec<Value>>(1).unwrap();
        let prepared =
            SpillWriterBuffer::prepare_with_capacity(qualified_writer_buffer_requested_bytes())
                .unwrap();
        let writer = checked_workspace_sum(
            prepared.capacity(),
            manager.qualified_file_workspace_bound().unwrap(),
        )
        .unwrap();
        drop(prepared);
        let mut heap_calibration: BinaryHeap<AccountedIntermediateHeapEntry<'_>> =
            BinaryHeap::new();
        heap_calibration.try_reserve_exact(3).unwrap();
        let mut reader_calibration: Vec<Option<AccountedIntermediateRunReader>> = Vec::new();
        reader_calibration.try_reserve_exact(3).unwrap();
        let frontier = checked_workspace_sum(
            cursor_capacity_bytes::<AccountedIntermediateHeapEntry<'_>>(
                heap_calibration.capacity(),
            )
            .unwrap(),
            cursor_capacity_bytes::<Option<AccountedIntermediateRunReader>>(
                reader_calibration.capacity(),
            )
            .unwrap(),
        )
        .unwrap();
        let blocker_bytes = budget
            .checked_sub(stable)
            .and_then(|bytes| bytes.checked_sub(output))
            .and_then(|bytes| bytes.checked_sub(writer))
            .and_then(|bytes| bytes.checked_sub(frontier))
            .unwrap();
        let blocker = resources.try_allocate(blocker_bytes).unwrap();

        let observer = ExternalSortGrantObserver::inert();
        let error = sort
            .merge_cursor_accounted_observing(Vec::new(), 1, observer)
            .unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Memory(MemoryGrantError::LimitExceeded { .. })
        ));
        assert_eq!(
            io.opens.load(AtomicOrdering::Relaxed),
            0,
            "intermediate reader workspace must be admitted before reader open"
        );
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(sort.frontier_granted_bytes(), 0);
        assert_eq!(sort.payload_granted_bytes(), 0);
        assert_eq!(manager.active_file_count(), 0);
        drop(blocker);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn intermediate_payload_denial_precedes_sort_row_allocation() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(CountingReadPayloadIo::default());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let budget = 4 * 1024 * 1024;
        let buffer_manager = buffer_manager_with_exact_budget(budget);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.set_merge_fan_in(3);
        sort.spill_sorted_run_accounted(&[vec![Value::String("x".repeat(4_096).into())]])
            .unwrap();
        for value in 1..5 {
            sort.spill_sorted_run_accounted(&[row(&[value])]).unwrap();
        }
        sort.release_memory_workspaces().unwrap();
        io.opens.store(0, AtomicOrdering::Relaxed);
        io.reads.store(0, AtomicOrdering::Relaxed);

        let stable = sort.total_granted_bytes();
        let output = cursor_capacity_bytes::<Vec<Value>>(1).unwrap();
        let writer = qualified_writer_base_bytes(&manager);
        let mut heap_calibration: BinaryHeap<AccountedIntermediateHeapEntry<'_>> =
            BinaryHeap::new();
        heap_calibration.try_reserve_exact(3).unwrap();
        let mut reader_calibration: Vec<Option<AccountedIntermediateRunReader>> = Vec::new();
        reader_calibration.try_reserve_exact(3).unwrap();
        let frontier = checked_workspace_sum(
            cursor_capacity_bytes::<AccountedIntermediateHeapEntry<'_>>(
                heap_calibration.capacity(),
            )
            .unwrap(),
            cursor_capacity_bytes::<Option<AccountedIntermediateRunReader>>(
                reader_calibration.capacity(),
            )
            .unwrap(),
        )
        .unwrap();
        // Enough for the first reader's fixed control workspace, but far less
        // than the 4 KiB SortRow's declared stored/plaintext allocation peak.
        let reader_slack = 1_024;
        let blocker_bytes = budget
            .checked_sub(stable)
            .and_then(|bytes| bytes.checked_sub(output))
            .and_then(|bytes| bytes.checked_sub(writer))
            .and_then(|bytes| bytes.checked_sub(frontier))
            .and_then(|bytes| bytes.checked_sub(reader_slack))
            .unwrap();
        let blocker = resources.try_allocate(blocker_bytes).unwrap();

        let observer = ExternalSortGrantObserver::inert();
        let error = sort
            .merge_cursor_accounted_observing(Vec::new(), 1, observer)
            .unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Memory(MemoryGrantError::LimitExceeded { .. })
        ));
        assert_eq!(io.opens.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(
            io.reads.load(AtomicOrdering::Relaxed),
            2,
            "FileStart and SortRunStart may read; denied SortRow must not"
        );
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(sort.frontier_granted_bytes(), 0);
        assert_eq!(sort.payload_granted_bytes(), 0);
        assert_eq!(manager.active_file_count(), 0);
        drop(blocker);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_cursor_streams_stable_order_through_multipass_merge() {
        let directory = TempDir::new().unwrap();
        let provider = Arc::new(ReaderPeakProvider::default());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::clone(&provider) as Arc<dyn super::super::SpillRecordProvider>,
                    super::super::SpillFrameLimits::format_max(),
                )
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            2,
            vec![SortKey::descending(0)],
            grant,
        );
        sort.set_merge_fan_in(3);
        for ordinal in 0..10 {
            sort.spill_sorted_run_accounted(&[row(&[1, ordinal])])
                .unwrap();
        }

        let observer = ExternalSortGrantObserver::inert();
        let mut cursor = sort
            .merge_cursor_accounted_observing(Vec::new(), 2, observer)
            .unwrap();
        let mut observed = Vec::new();
        while let Some(chunk) = cursor.next_chunk_accounted_observing().unwrap() {
            observed.extend_from_slice(chunk.rows());
        }
        drop(cursor);

        assert_eq!(
            observed,
            (0..10)
                .map(|ordinal| row(&[1, ordinal]))
                .collect::<Vec<_>>()
        );
        assert!(provider.peak_readers() <= 3);
        assert_eq!(provider.active_readers(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(sort.total_granted_bytes(), sort.run_catalog_granted_bytes());
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn cursor_cancellation_during_next_read_discards_batch_and_cleans_everything() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(CancelNthIo::new(
            super::super::SpillIoOperation::ReadPayload,
            1,
            control.cancellation_handle(),
        ));
        io.disarm();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            Arc::clone(&buffer_manager),
            control.token(),
        )
        .unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted_with_cancellation(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
            control.token(),
        );
        sort.spill_sorted_run_accounted(&[row(&[1]), row(&[3])])
            .unwrap();
        sort.spill_sorted_run_accounted(&[row(&[2]), row(&[4])])
            .unwrap();
        let observer = ExternalSortGrantObserver::inert();
        let mut cursor = sort
            .merge_cursor_accounted_observing(Vec::new(), 2, observer)
            .unwrap();
        io.arm();

        let error = cursor.next_chunk_accounted_observing().unwrap_err();
        assert!(matches!(
            error,
            ExternalSortOperationError::Cancelled(
                crate::execution::QueryCancellationError::Cancelled
            )
        ));
        assert!(cursor.cursor.output_rows.is_empty());
        assert_eq!(cursor.cursor.output_rows.capacity(), 0);
        assert_eq!(cursor.cursor.heap.capacity(), 0);
        assert_eq!(cursor.cursor.run_readers.capacity(), 0);
        assert_eq!(cursor.cursor.sorter.output_granted_bytes(), 0);
        assert_eq!(cursor.cursor.sorter.frontier_granted_bytes(), 0);
        drop(cursor);
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(sort.total_granted_bytes(), sort.run_catalog_granted_bytes());
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn cursor_decode_failure_closes_readers_releases_grants_and_deletes_runs() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(CorruptOpenedSortRowProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run_accounted(&[row(&[1]), row(&[2])])
            .unwrap();

        let observer = ExternalSortGrantObserver::inert();
        let error = sort
            .merge_cursor_accounted_observing(Vec::new(), 1, observer)
            .unwrap_err();
        assert!(matches!(
            error,
            ExternalSortOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::InvalidData
                    && error.to_string().contains("Unknown value tag")
        ));
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(sort.total_granted_bytes(), sort.run_catalog_granted_bytes());
        assert_eq!(manager.active_file_count(), 0);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn dropping_partially_consumed_cursor_closes_readers_and_cleans_runs() {
        let directory = TempDir::new().unwrap();
        let provider = Arc::new(ReaderPeakProvider::default());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::clone(&provider) as Arc<dyn super::super::SpillRecordProvider>,
                    super::super::SpillFrameLimits::format_max(),
                )
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run_accounted(&[row(&[1]), row(&[3])])
            .unwrap();
        sort.spill_sorted_run_accounted(&[row(&[2]), row(&[4])])
            .unwrap();
        {
            let observer = ExternalSortGrantObserver::inert();
            let mut cursor = sort
                .merge_cursor_accounted_observing(Vec::new(), 1, observer)
                .unwrap();
            assert_eq!(
                cursor
                    .next_chunk_accounted_observing()
                    .unwrap()
                    .unwrap()
                    .rows(),
                &[row(&[1])]
            );
            assert_eq!(provider.active_readers(), 2);
        }
        assert_eq!(provider.active_readers(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(sort.total_granted_bytes(), sort.run_catalog_granted_bytes());
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_multipass_merge_caps_peak_open_readers() {
        let directory = TempDir::new().unwrap();
        let provider = Arc::new(ReaderPeakProvider::default());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::clone(&provider) as Arc<dyn super::super::SpillRecordProvider>,
                    super::super::SpillFrameLimits::format_max(),
                )
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(manager, 2, vec![SortKey::ascending(0)], grant);
        sort.set_merge_fan_in(3);
        for ordinal in 0..10 {
            sort.spill_sorted_run_accounted(&[row(&[1, ordinal])])
                .unwrap();
        }

        let result = sort.merge_all_accounted(Vec::new()).unwrap();

        assert_eq!(result.len(), 10);
        assert_eq!(result.first(), Some(&row(&[1, 0])));
        assert_eq!(result.last(), Some(&row(&[1, 9])));
        for (ordinal, result_row) in result.iter().enumerate() {
            assert_eq!(result_row, &row(&[1, ordinal as i64]));
        }
        assert!(
            provider.peak_readers() <= 3,
            "merge opened {} readers simultaneously",
            provider.peak_readers()
        );
        assert_eq!(provider.active_readers(), 0);
    }

    #[test]
    fn accounted_intermediate_merge_cancels_after_complete_output_record() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(CancelNthIo::new(
            super::super::SpillIoOperation::WritePayload,
            3,
            control.cancellation_handle(),
        ));
        io.disarm();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            Arc::clone(&buffer_manager),
            control.token(),
        )
        .unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted_with_cancellation(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
            control.token(),
        );
        sort.set_merge_fan_in(3);
        for value in 0..4 {
            sort.spill_sorted_run_accounted(&[row(&[value])]).unwrap();
        }
        io.arm();

        let error = sort.merge_all_accounted(Vec::new()).unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Cancelled(
                crate::execution::QueryCancellationError::Cancelled
            )
        ));
        assert_eq!(io.matching.load(AtomicOrdering::Relaxed), 3);
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
        assert_eq!(buffer_manager.allocated(), sort.run_catalog_granted_bytes());
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn intermediate_write_error_wins_when_same_callback_cancels() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(CancelNthIo::with_failure(
            super::super::SpillIoOperation::WritePayload,
            3,
            control.cancellation_handle(),
            std::io::ErrorKind::PermissionDenied,
            "deterministic intermediate merge write failure",
        ));
        io.disarm();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            Arc::clone(&buffer_manager),
            control.token(),
        )
        .unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted_with_cancellation(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
            control.token(),
        );
        sort.set_merge_fan_in(3);
        for value in 0..4 {
            sort.spill_sorted_run_accounted(&[row(&[value])]).unwrap();
        }
        io.arm();

        let error = sort.merge_all_accounted(Vec::new()).unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::PermissionDenied
                    && error.to_string().contains("intermediate merge write failure")
        ));
        assert!(control.token().is_cancelled());
        assert_eq!(io.matching.load(AtomicOrdering::Relaxed), 3);
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn intermediate_output_cleanup_failure_remains_sorter_owned_for_retry() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(ToggleDeleteIo::failing());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.set_merge_fan_in(3);
        for value in 0..4 {
            sort.spill_sorted_run_accounted(&[row(&[value])]).unwrap();
        }

        let error = sort.merge_all_accounted(Vec::new()).unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::PermissionDenied
                    && error.to_string().contains("writer cleanup denial")
        ));
        assert_eq!(sort.num_runs(), 5);
        assert_eq!(manager.active_file_count(), 5);
        assert!(manager.disk_stats().reserved_live_bytes > 0);

        io.permit_delete();
        sort.cleanup().unwrap();
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn cancelled_partial_intermediate_cleanup_failure_remains_sorter_owned_for_retry() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(
            CancelNthIo::new(
                super::super::SpillIoOperation::WritePayload,
                3,
                control.cancellation_handle(),
            )
            .with_delete_failure(
                std::io::ErrorKind::WouldBlock,
                "deterministic partial intermediate deletion failure",
            ),
        );
        io.disarm();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            Arc::clone(&buffer_manager),
            control.token(),
        )
        .unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted_with_cancellation(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
            control.token(),
        );
        sort.set_merge_fan_in(3);
        for value in 0..4 {
            sort.spill_sorted_run_accounted(&[row(&[value])]).unwrap();
        }
        io.arm();

        let error = sort.merge_all_accounted(Vec::new()).unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::CancelledWithCleanup {
                error: crate::execution::QueryCancellationError::Cancelled,
                ref cleanup,
                ..
            } if cleanup.to_string().contains("partial intermediate deletion failure")
        ));
        assert_eq!(sort.num_runs(), 5);
        assert_eq!(manager.active_file_count(), 5);
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);

        io.permit_delete();
        sort.cleanup().unwrap();
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn partial_intermediate_write_and_delete_failures_retain_primary_and_output() {
        for semantic in [false, true] {
            let directory = TempDir::new().unwrap();
            let control = crate::execution::QueryExecutionControl::new();
            let io = Arc::new(
                CancelNthIo::with_failure(
                    super::super::SpillIoOperation::WritePayload,
                    3,
                    control.cancellation_handle(),
                    std::io::ErrorKind::PermissionDenied,
                    "deterministic partial intermediate write failure",
                )
                .with_delete_failure(
                    std::io::ErrorKind::WouldBlock,
                    "deterministic partial intermediate deletion failure",
                ),
            );
            io.disarm();
            let manager = Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .provider(
                        Arc::new(super::super::CleartextSpillRecordProvider),
                        super::super::SpillFrameLimits::format_max(),
                    )
                    .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                    .build()
                    .unwrap(),
            );
            let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
            let resources = crate::execution::QueryResourceContext::new_with_cancellation(
                Arc::clone(&buffer_manager),
                control.token(),
            )
            .unwrap();
            let grant = resources.try_allocate(0).unwrap();
            let mut sort = if semantic {
                semantic_fault_sort(
                    Arc::clone(&manager),
                    1,
                    vec![SortKey::ascending(0)],
                    grant,
                    &resources,
                )
            } else {
                ExternalSort::new_accounted_with_cancellation(
                    Arc::clone(&manager),
                    1,
                    vec![SortKey::ascending(0)],
                    grant,
                    control.token(),
                )
            };
            sort.set_merge_fan_in(3);
            for value in 0..4 {
                sort.spill_sorted_run_accounted(&[row(&[value])]).unwrap();
            }
            io.arm();

            let error = sort.merge_all_accounted(Vec::new()).unwrap_err();

            assert!(matches!(
                error,
                ExternalSortOperationError::Io(ref error)
                    if error.kind() == std::io::ErrorKind::PermissionDenied
                        && error.to_string().contains("partial intermediate write failure")
                        && error.to_string().contains("partial intermediate deletion failure")
            ));
            assert!(control.token().is_cancelled());
            assert_eq!(sort.num_runs(), 5);
            assert_eq!(manager.active_file_count(), 5);

            io.permit_delete();
            sort.cleanup().unwrap();
            assert_eq!(sort.num_runs(), 0);
            assert_eq!(manager.active_file_count(), 0);
            drop(sort);
            assert_eq!(buffer_manager.allocated(), 0);
        }
    }

    #[test]
    fn partial_intermediate_read_and_delete_failures_retain_primary_and_output() {
        for semantic in [false, true] {
            let directory = TempDir::new().unwrap();
            let control = crate::execution::QueryExecutionControl::new();
            let io = Arc::new(
                CancelNthIo::with_failure(
                    super::super::SpillIoOperation::ReadPayload,
                    1,
                    control.cancellation_handle(),
                    std::io::ErrorKind::PermissionDenied,
                    "deterministic partial intermediate read failure",
                )
                .with_delete_failure(
                    std::io::ErrorKind::WouldBlock,
                    "deterministic partial intermediate deletion failure",
                ),
            );
            io.disarm();
            let manager = Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .provider(
                        Arc::new(super::super::CleartextSpillRecordProvider),
                        super::super::SpillFrameLimits::format_max(),
                    )
                    .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                    .build()
                    .unwrap(),
            );
            let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
            let resources = crate::execution::QueryResourceContext::new_with_cancellation(
                Arc::clone(&buffer_manager),
                control.token(),
            )
            .unwrap();
            let grant = resources.try_allocate(0).unwrap();
            let mut sort = if semantic {
                semantic_fault_sort(
                    Arc::clone(&manager),
                    1,
                    vec![SortKey::ascending(0)],
                    grant,
                    &resources,
                )
            } else {
                ExternalSort::new_accounted_with_cancellation(
                    Arc::clone(&manager),
                    1,
                    vec![SortKey::ascending(0)],
                    grant,
                    control.token(),
                )
            };
            sort.set_merge_fan_in(3);
            for value in 0..4 {
                sort.spill_sorted_run_accounted(&[row(&[value])]).unwrap();
            }
            io.arm();

            let error = sort.merge_all_accounted(Vec::new()).unwrap_err();

            assert!(matches!(
                error,
                ExternalSortOperationError::Io(ref error)
                    if error.kind() == std::io::ErrorKind::PermissionDenied
                        && error.to_string().contains("partial intermediate read failure")
                        && error.to_string().contains("partial intermediate deletion failure")
            ));
            assert!(control.token().is_cancelled());
            assert_eq!(sort.num_runs(), 5);
            assert_eq!(manager.active_file_count(), 5);

            io.permit_delete();
            sort.cleanup().unwrap();
            assert_eq!(sort.num_runs(), 0);
            assert_eq!(manager.active_file_count(), 0);
            drop(sort);
            assert_eq!(buffer_manager.allocated(), 0);
        }
    }

    #[test]
    fn partial_intermediate_decode_and_delete_failures_retain_primary_and_output() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(ToggleDeleteIo::failing());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(CorruptOpenedSortRowProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.set_merge_fan_in(3);
        for value in 0..4 {
            sort.spill_sorted_run_accounted(&[row(&[value])]).unwrap();
        }

        let error = sort.merge_all_accounted(Vec::new()).unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::InvalidData
                    && error.to_string().contains("Unknown value tag")
                    && error.to_string().contains("writer cleanup denial")
        ));
        assert_eq!(sort.num_runs(), 5);
        assert_eq!(manager.active_file_count(), 5);

        io.permit_delete();
        sort.cleanup().unwrap();
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn tokenless_partial_intermediate_delete_failure_remains_sorter_owned_for_retry() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(
            CancelNthIo::with_failure(
                super::super::SpillIoOperation::WritePayload,
                3,
                control.cancellation_handle(),
                std::io::ErrorKind::PermissionDenied,
                "deterministic tokenless intermediate write failure",
            )
            .with_delete_failure(
                std::io::ErrorKind::WouldBlock,
                "deterministic tokenless intermediate deletion failure",
            ),
        );
        io.disarm();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let mut sort = ExternalSort::new(Arc::clone(&manager), 1, vec![SortKey::ascending(0)]);
        sort.set_merge_fan_in(3);
        for value in 0..4 {
            sort.spill_sorted_run(vec![row(&[value])]).unwrap();
        }
        io.arm();

        let error = sort.merge_all(Vec::new()).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(
            error
                .to_string()
                .contains("tokenless intermediate write failure")
        );
        assert!(
            error
                .to_string()
                .contains("tokenless intermediate deletion failure")
        );
        assert!(control.token().is_cancelled());
        assert_eq!(sort.num_runs(), 5);
        assert_eq!(manager.active_file_count(), 5);

        io.permit_delete();
        sort.cleanup().unwrap();
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    fn accounted_spill_stops_between_successful_row_records() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(CancelNthIo::new(
            super::super::SpillIoOperation::WritePayload,
            4,
            control.cancellation_handle(),
        ));
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            Arc::clone(&buffer_manager),
            control.token(),
        )
        .unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted_with_cancellation(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
            control.token(),
        );

        let error = sort
            .spill_sorted_run_accounted(&[row(&[1]), row(&[2]), row(&[3])])
            .unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Cancelled(
                crate::execution::QueryCancellationError::Cancelled
            )
        ));
        assert_eq!(io.matching.load(AtomicOrdering::Relaxed), 4);
        assert_eq!(io.flushes.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(io.syncs.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
        assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_spill_checks_after_final_row_before_finish() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(CancelNthIo::new(
            super::super::SpillIoOperation::WritePayload,
            3,
            control.cancellation_handle(),
        ));
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            Arc::clone(&buffer_manager),
            control.token(),
        )
        .unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted_with_cancellation(
            Arc::clone(&manager),
            1,
            vec![],
            grant,
            control.token(),
        );

        let error = sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Cancelled(
                crate::execution::QueryCancellationError::Cancelled
            )
        ));
        assert_eq!(io.matching.load(AtomicOrdering::Relaxed), 3);
        assert_eq!(io.flushes.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(io.syncs.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_pre_cancelled_merge_still_cleans_owned_runs() {
        let (_directory, manager) = create_manager();
        let control = crate::execution::QueryExecutionControl::new();
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            Arc::clone(&buffer_manager),
            control.token(),
        )
        .unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted_with_cancellation(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
            control.token(),
        );
        sort.spill_sorted_run_accounted(&[row(&[1]), row(&[2])])
            .unwrap();
        control.cancellation_handle().cancel();

        let error = sort.merge_all_accounted(Vec::new()).unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Cancelled(
                crate::execution::QueryCancellationError::Cancelled
            )
        ));
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(sort.workspace_granted_bytes(), 0);
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
        assert_eq!(buffer_manager.allocated(), sort.run_catalog_granted_bytes());
        drop(sort);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_record_io_error_wins_when_same_callback_cancels() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(CancelNthIo::with_failure(
            super::super::SpillIoOperation::WritePayload,
            3,
            control.cancellation_handle(),
            std::io::ErrorKind::PermissionDenied,
            "deterministic cancelled record write failure",
        ));
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            Arc::clone(&buffer_manager),
            control.token(),
        )
        .unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted_with_cancellation(
            Arc::clone(&manager),
            1,
            vec![],
            grant,
            control.token(),
        );

        let error = sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::PermissionDenied
                    && error.to_string().contains("cancelled record write failure")
        ));
        assert!(control.token().is_cancelled());
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn cancelled_writer_preserves_release_and_delete_failures() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(
            CancelNthIo::new(
                super::super::SpillIoOperation::WritePayload,
                3,
                control.cancellation_handle(),
            )
            .with_delete_failure(
                std::io::ErrorKind::WouldBlock,
                "deterministic cancellation deletion failure",
            ),
        );
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            Arc::clone(&buffer_manager),
            control.token(),
        )
        .unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted_with_cancellation(
            Arc::clone(&manager),
            1,
            vec![],
            grant,
            control.token(),
        );
        sort.writer_workspace.release_error = Some(MemoryGrantError::AccountingPoisoned {
            account: "deterministic cancellation writer release",
        });

        let error = sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap_err();

        let ExternalSortOperationError::WithGrantRelease {
            primary:
                ExternalSortPrimary::Cancelled(crate::execution::QueryCancellationError::Cancelled),
            release,
            cleanup: Some(cleanup),
            phase,
        } = error
        else {
            panic!("cancellation, release, and deletion failures were flattened")
        };
        assert!(matches!(
            release,
            MemoryGrantError::AccountingPoisoned {
                account: "deterministic cancellation writer release"
            }
        ));
        assert_eq!(cleanup.kind(), std::io::ErrorKind::WouldBlock);
        assert!(
            cleanup
                .to_string()
                .contains("cancellation deletion failure")
        );
        assert_eq!(phase, "writer cancellation release");
        assert!(sort.writer_workspace_granted_bytes() > 0);
        assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());
        assert_eq!(
            resources.query_stats().allocated_bytes,
            sort.total_granted_bytes()
        );
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 1);
        assert_eq!(manager.spilled_bytes(), 0);
        assert!(manager.disk_stats().reserved_live_bytes > 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);

        sort.writer_workspace.release_error = None;
        sort.cleanup().unwrap();
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(manager.active_file_count(), 1);

        io.permit_delete();
        manager.cleanup().unwrap();
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        drop(sort);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_k_way_merge_stops_after_cancelled_record_read() {
        for semantic in [false, true] {
            let directory = TempDir::new().unwrap();
            let control = crate::execution::QueryExecutionControl::new();
            let io = Arc::new(CancelNthIo::new(
                super::super::SpillIoOperation::ReadPayload,
                7,
                control.cancellation_handle(),
            ));
            io.disarm();
            let manager = Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .provider(
                        Arc::new(super::super::CleartextSpillRecordProvider),
                        super::super::SpillFrameLimits::format_max(),
                    )
                    .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                    .build()
                    .unwrap(),
            );
            let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
            let resources = crate::execution::QueryResourceContext::new_with_cancellation(
                Arc::clone(&buffer_manager),
                control.token(),
            )
            .unwrap();
            let grant = resources.try_allocate(0).unwrap();
            let mut sort = if semantic {
                semantic_fault_sort(
                    Arc::clone(&manager),
                    1,
                    vec![SortKey::ascending(0)],
                    grant,
                    &resources,
                )
            } else {
                ExternalSort::new_accounted_with_cancellation(
                    Arc::clone(&manager),
                    1,
                    vec![SortKey::ascending(0)],
                    grant,
                    control.token(),
                )
            };
            sort.spill_sorted_run_accounted(&[row(&[1]), row(&[3])])
                .unwrap();
            sort.spill_sorted_run_accounted(&[row(&[2]), row(&[4])])
                .unwrap();
            io.arm();

            let error = sort.merge_all_accounted(Vec::new()).unwrap_err();

            assert!(matches!(
                error,
                ExternalSortOperationError::Cancelled(
                    crate::execution::QueryCancellationError::Cancelled
                )
            ));
            assert_eq!(io.matching.load(AtomicOrdering::Relaxed), 7);
            assert_eq!(io.read_opens.load(AtomicOrdering::Relaxed), 2);
            assert_eq!(sort.num_runs(), 0);
            assert_eq!(sort.workspace_granted_bytes(), 0);
            assert_eq!(sort.writer_workspace_granted_bytes(), 0);
            assert_eq!(manager.active_file_count(), 0);
            assert_eq!(manager.spilled_bytes(), 0);
            assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
            assert_eq!(
                buffer_manager.allocated(),
                sort.run_catalog_granted_bytes() + sort.comparator.checked_granted_bytes().unwrap()
            );
            drop(sort);
            assert_eq!(buffer_manager.allocated(), 0);
        }
    }

    #[test]
    fn accounted_read_io_error_wins_when_same_callback_cancels() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(CancelNthIo::with_failure(
            super::super::SpillIoOperation::ReadPayload,
            7,
            control.cancellation_handle(),
            std::io::ErrorKind::PermissionDenied,
            "deterministic cancelled record read failure",
        ));
        io.disarm();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            Arc::clone(&buffer_manager),
            control.token(),
        )
        .unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted_with_cancellation(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
            control.token(),
        );
        sort.spill_sorted_run_accounted(&[row(&[1]), row(&[3])])
            .unwrap();
        sort.spill_sorted_run_accounted(&[row(&[2]), row(&[4])])
            .unwrap();
        io.arm();

        let error = sort.merge_all_accounted(Vec::new()).unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::PermissionDenied
                    && error.to_string().contains("cancelled record read failure")
        ));
        assert!(control.token().is_cancelled());
        assert_eq!(io.matching.load(AtomicOrdering::Relaxed), 7);
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn cancelled_merge_preserves_delete_failure_and_cleanup_ignores_token() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(
            CancelNthIo::new(
                super::super::SpillIoOperation::ReadPayload,
                7,
                control.cancellation_handle(),
            )
            .with_delete_failure(
                std::io::ErrorKind::WouldBlock,
                "deterministic cancelled merge deletion failure",
            ),
        );
        io.disarm();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            Arc::clone(&buffer_manager),
            control.token(),
        )
        .unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted_with_cancellation(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
            control.token(),
        );
        sort.spill_sorted_run_accounted(&[row(&[1]), row(&[3])])
            .unwrap();
        sort.spill_sorted_run_accounted(&[row(&[2]), row(&[4])])
            .unwrap();
        io.arm();

        let error = sort.merge_all_accounted(Vec::new()).unwrap_err();

        let ExternalSortOperationError::CancelledWithCleanup {
            error: crate::execution::QueryCancellationError::Cancelled,
            cleanup,
            phase,
        } = error
        else {
            panic!("cancelled merge cleanup failure lost typed cancellation")
        };
        assert_eq!(cleanup.kind(), std::io::ErrorKind::WouldBlock);
        assert!(
            cleanup
                .to_string()
                .contains("cancelled merge deletion failure")
        );
        assert_eq!(phase, "spill-read cleanup");
        assert_eq!(sort.num_runs(), 2);
        assert_eq!(manager.active_file_count(), 2);
        assert!(manager.disk_stats().reserved_live_bytes > 0);

        io.permit_delete();
        sort.cleanup().unwrap();
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
        drop(sort);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn test_external_sort_single_run() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);

        // Spill a sorted run
        let sorted_run = vec![row(&[1]), row(&[2]), row(&[3])];
        sort.spill_sorted_run(sorted_run).unwrap();

        assert_eq!(sort.num_runs(), 1);
        assert_eq!(sort.total_rows(), 3);

        let result = sort.merge_all(Vec::new()).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], row(&[1]));
        assert_eq!(result[1], row(&[2]));
        assert_eq!(result[2], row(&[3]));
    }

    #[test]
    fn accounted_codec_workspace_is_charged_and_released_before_merge() {
        let (_temp_dir, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(manager, 1, vec![], grant);
        let expected = vec![Value::GCounter(Arc::new(HashMap::from([
            ("replica-a".to_string(), 1),
            ("replica-b".to_string(), 2),
        ])))];
        let smaller = vec![Value::GCounter(Arc::new(HashMap::from([(
            "a".to_string(),
            3,
        )])))];
        let rows = vec![expected.clone()];

        let spill_observation = TestGrantObservation::new(sort.total_granted_bytes());
        let observer = spill_observation.observer();
        sort.spill_sorted_run_accounted_observing(&rows, &observer)
            .unwrap();

        let granted = sort.workspace_granted_bytes();
        let catalog_granted = sort.run_catalog_granted_bytes();
        let row_pointer = sort.workspace.row_staging.as_slice().as_ptr();
        let counter_entry_pointer = sort.workspace.counter_scratch.entry_pointer();
        let counter_key_pointer = sort.workspace.counter_scratch.key_pointer();
        assert!(granted > 0);
        assert!(catalog_granted > 0);
        assert_eq!(spill_observation.current(), sort.total_granted_bytes());
        assert_eq!(sort.workspace.observed_bytes().unwrap(), granted);
        assert_eq!(
            buffer_manager.allocated(),
            sort.total_granted_bytes(),
            "every retained codec and catalog capacity must remain covered by grants"
        );
        assert_eq!(
            resources.query_stats().allocated_bytes,
            sort.total_granted_bytes()
        );

        sort.spill_sorted_run_accounted(std::slice::from_ref(&smaller))
            .unwrap();

        assert_eq!(sort.workspace_granted_bytes(), granted);
        assert_eq!(sort.workspace.observed_bytes().unwrap(), granted);
        assert_eq!(sort.workspace.row_staging.as_slice().as_ptr(), row_pointer);
        assert_eq!(
            sort.workspace.counter_scratch.entry_pointer(),
            counter_entry_pointer
        );
        assert_eq!(
            sort.workspace.counter_scratch.key_pointer(),
            counter_key_pointer
        );
        assert_eq!(sort.run_catalog_granted_bytes(), catalog_granted);
        assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());
        let merge_observation = TestGrantObservation::new(sort.total_granted_bytes());
        let observer = merge_observation.observer();
        assert_eq!(
            sort.merge_all_accounted_observing(Vec::new(), &observer)
                .unwrap(),
            vec![expected, smaller]
        );
        assert_eq!(merge_observation.current(), catalog_granted);
        assert_eq!(sort.workspace_granted_bytes(), 0);
        assert_eq!(sort.run_catalog_granted_bytes(), catalog_granted);
        assert_eq!(buffer_manager.allocated(), catalog_granted);
        assert_eq!(resources.query_stats().allocated_bytes, catalog_granted);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[test]
    fn run_catalog_pinned_allocator_reports_exact_capacity_across_dense_targets() {
        fn require_exact_allocator(
            _entries: &allocator_api2::vec::Vec<
                ExternalSortRunEntry,
                allocator_api2::alloc::Global,
            >,
        ) {
        }

        for target_capacity in 1..=4096 {
            let entries = ExternalSortRunCatalog::allocate_exact_entries(target_capacity).unwrap();
            require_exact_allocator(&entries);
            assert!(entries.is_empty());
            assert_eq!(entries.capacity(), target_capacity);
            assert_eq!(
                run_catalog_capacity_bytes(entries.capacity()).unwrap(),
                std::alloc::Layout::array::<ExternalSortRunEntry>(target_capacity)
                    .unwrap()
                    .size()
            );
        }
    }

    #[test]
    fn run_catalog_capacity_contract_failure_rolls_back_exact_peak_without_publication() {
        let old_capacity = INITIAL_RUN_CATALOG_CAPACITY;
        let target_capacity = old_capacity.checked_mul(2).unwrap();
        let old_bytes = run_catalog_capacity_bytes(old_capacity).unwrap();
        let target_bytes = run_catalog_capacity_bytes(target_capacity).unwrap();
        let exact_peak = old_bytes.checked_add(target_bytes).unwrap();
        let buffer_manager = buffer_manager_with_exact_budget(exact_peak);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let (directory, manager) = create_manager();
        let mut catalog = ExternalSortRunCatalog::new(Some(grant));

        for _ in 0..old_capacity {
            catalog.prepare_push().unwrap();
            let file = manager.create_file(SpillFileRole::SortRun).unwrap();
            catalog.push(file, 1);
        }
        let old_pointer = catalog.pointer();
        assert_eq!(catalog.len(), old_capacity);
        assert_eq!(catalog.capacity(), old_capacity);
        assert_eq!(catalog.granted_bytes(), old_bytes);
        assert_eq!(buffer_manager.allocated(), old_bytes);

        catalog.force_next_observed_capacity_for_test(target_capacity + 1);
        let error = catalog.prepare_push().unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Io(error)
                if error.kind() == std::io::ErrorKind::InvalidData
        ));
        assert_eq!(catalog.len(), old_capacity);
        assert_eq!(catalog.capacity(), old_capacity);
        assert_eq!(catalog.pointer(), old_pointer);
        assert_eq!(catalog.granted_bytes(), old_bytes);
        assert_eq!(catalog.observed_bytes().unwrap(), old_bytes);
        assert_eq!(buffer_manager.allocated(), old_bytes);
        assert_eq!(resources.query_stats().allocated_bytes, old_bytes);
        assert_eq!(manager.active_file_count(), old_capacity);

        drop(catalog);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[test]
    fn run_catalog_shrink_failure_leaves_replacement_overcovered_and_retryable() {
        let old_capacity = INITIAL_RUN_CATALOG_CAPACITY;
        let target_capacity = old_capacity.checked_mul(2).unwrap();
        let old_bytes = run_catalog_capacity_bytes(old_capacity).unwrap();
        let target_bytes = run_catalog_capacity_bytes(target_capacity).unwrap();
        let exact_peak = old_bytes.checked_add(target_bytes).unwrap();
        let buffer_manager = buffer_manager_with_exact_budget(exact_peak);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let (directory, manager) = create_manager();
        let mut catalog = ExternalSortRunCatalog::new(Some(grant));

        for _ in 0..old_capacity {
            catalog.prepare_push().unwrap();
            let file = manager.create_file(SpillFileRole::SortRun).unwrap();
            catalog.push(file, 1);
        }
        let old_pointer = catalog.pointer();
        catalog.force_next_shrink_failure_for_test(MemoryGrantError::AccountingPoisoned {
            account: "deterministic run-catalog shrink",
        });

        let error = catalog.prepare_push().unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Memory(MemoryGrantError::AccountingPoisoned {
                account: "deterministic run-catalog shrink"
            })
        ));
        assert_eq!(catalog.len(), old_capacity);
        assert_eq!(catalog.capacity(), target_capacity);
        assert_ne!(catalog.pointer(), old_pointer);
        assert_eq!(catalog.observed_bytes().unwrap(), target_bytes);
        assert_eq!(catalog.granted_bytes(), exact_peak);
        assert_eq!(buffer_manager.allocated(), exact_peak);

        // Retrying the same publication performs only the deferred shrink;
        // it neither reallocates nor moves the already-published entries.
        let replacement_pointer = catalog.pointer();
        catalog.prepare_push().unwrap();
        assert_eq!(catalog.pointer(), replacement_pointer);
        assert_eq!(catalog.capacity(), target_capacity);
        assert_eq!(catalog.granted_bytes(), target_bytes);
        assert_eq!(buffer_manager.allocated(), target_bytes);
        let file = manager.create_file(SpillFileRole::SortRun).unwrap();
        catalog.push(file, 1);
        assert_eq!(catalog.len(), old_capacity + 1);

        drop(catalog);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[test]
    fn run_catalog_fast_path_fences_an_undercovered_physical_capacity() {
        let old_capacity = INITIAL_RUN_CATALOG_CAPACITY;
        let old_bytes = run_catalog_capacity_bytes(old_capacity).unwrap();
        let buffer_manager = buffer_manager_with_exact_budget(old_bytes);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let (_directory, manager) = create_manager();
        let mut catalog = ExternalSortRunCatalog::new(Some(grant));

        catalog.prepare_push().unwrap();
        let file = manager.create_file(SpillFileRole::SortRun).unwrap();
        catalog.push(file, 1);
        let pointer = catalog.pointer();
        catalog
            .grant
            .as_mut()
            .unwrap()
            .try_resize(old_bytes - 1)
            .unwrap();

        let error = catalog.prepare_push().unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Io(error)
                if error.kind() == std::io::ErrorKind::InvalidData
        ));
        assert_eq!(catalog.len(), 1);
        assert_eq!(catalog.capacity(), old_capacity);
        assert_eq!(catalog.pointer(), pointer);
        assert_eq!(catalog.observed_bytes().unwrap(), old_bytes);
        assert_eq!(catalog.granted_bytes(), old_bytes - 1);
        assert_eq!(buffer_manager.allocated(), old_bytes - 1);

        drop(catalog);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[test]
    fn accounted_run_catalog_is_exact_reused_and_retained_until_drop() {
        let (_temp_dir, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(manager, 1, vec![SortKey::ascending(0)], grant);
        let observation = TestGrantObservation::new(sort.total_granted_bytes());
        let observer = observation.observer();
        sort.spill_sorted_run_accounted_observing(&[row(&[1])], &observer)
            .unwrap();

        let codec_bytes = sort.workspace_granted_bytes();
        let catalog_bytes = sort.run_catalog_granted_bytes();
        let catalog_pointer = sort.run_catalog_pointer();
        let catalog_capacity = sort.run_catalog_capacity();
        assert!(codec_bytes > 0);
        assert!(catalog_bytes > 0);
        assert_eq!(
            catalog_bytes,
            sort.run_catalog_observed_bytes().unwrap(),
            "the retained run catalog must be charged by observed capacity"
        );
        let independent_catalog_bytes = catalog_capacity
            .checked_mul(std::mem::size_of::<ExternalSortRunEntry>())
            .unwrap();
        assert_eq!(catalog_bytes, independent_catalog_bytes);
        assert_eq!(sort.total_granted_bytes(), codec_bytes + catalog_bytes);
        assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());
        assert_eq!(observation.current(), sort.total_granted_bytes());

        for value in 2..=catalog_capacity {
            let value = i64::try_from(value).unwrap();
            sort.spill_sorted_run_accounted(&[row(&[value])]).unwrap();
        }

        assert_eq!(sort.run_catalog_granted_bytes(), catalog_bytes);
        assert_eq!(sort.run_catalog_pointer(), catalog_pointer);
        assert_eq!(sort.num_runs(), catalog_capacity);
        assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());

        let expected = (1..=catalog_capacity)
            .map(|value| row(&[i64::try_from(value).unwrap()]))
            .collect::<Vec<_>>();

        let merge_observation = TestGrantObservation::new(sort.total_granted_bytes());
        let observer = merge_observation.observer();
        assert_eq!(
            sort.merge_all_accounted_observing(Vec::new(), &observer)
                .unwrap(),
            expected
        );
        assert_eq!(sort.workspace_granted_bytes(), 0);
        assert_eq!(sort.run_catalog_granted_bytes(), catalog_bytes);
        assert_eq!(sort.total_granted_bytes(), catalog_bytes);
        assert_eq!(merge_observation.current(), catalog_bytes);
        assert_eq!(buffer_manager.allocated(), catalog_bytes);
        assert_eq!(resources.query_stats().allocated_bytes, catalog_bytes);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[test]
    fn accounted_run_catalog_growth_replaces_storage_and_preserves_entries() {
        let (_temp_dir, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(manager, 1, vec![SortKey::ascending(0)], grant);

        sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
        let initial_capacity = sort.run_catalog_capacity();
        for value in 2..=initial_capacity {
            sort.spill_sorted_run_accounted(&[row(&[i64::try_from(value).unwrap()])])
                .unwrap();
        }
        let initial_pointer = sort.run_catalog_pointer();
        let initial_catalog_bytes = sort.run_catalog_granted_bytes();
        assert_eq!(sort.num_runs(), initial_capacity);

        let growth_value = initial_capacity.checked_add(1).unwrap();
        sort.spill_sorted_run_accounted(&[row(&[i64::try_from(growth_value).unwrap()])])
            .unwrap();

        let grown_capacity = sort.run_catalog_capacity();
        let grown_catalog_bytes = sort.run_catalog_granted_bytes();
        assert!(grown_capacity > initial_capacity);
        assert_ne!(sort.run_catalog_pointer(), initial_pointer);
        assert!(grown_catalog_bytes > initial_catalog_bytes);
        assert_eq!(
            grown_catalog_bytes,
            grown_capacity
                .checked_mul(std::mem::size_of::<ExternalSortRunEntry>())
                .unwrap()
        );
        assert_eq!(
            grown_catalog_bytes,
            sort.run_catalog_observed_bytes().unwrap()
        );
        assert_eq!(
            sort.total_granted_bytes(),
            sort.workspace_granted_bytes()
                .checked_add(grown_catalog_bytes)
                .unwrap()
        );
        assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());
        assert_eq!(
            resources.query_stats().allocated_bytes,
            sort.total_granted_bytes()
        );

        let expected = (1..=growth_value)
            .map(|value| row(&[i64::try_from(value).unwrap()]))
            .collect::<Vec<_>>();
        assert_eq!(sort.merge_all_accounted(Vec::new()).unwrap(), expected);
        assert_eq!(buffer_manager.allocated(), grown_catalog_bytes);
        assert_eq!(resources.query_stats().allocated_bytes, grown_catalog_bytes);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[test]
    fn accounted_run_catalog_denial_precedes_file_creation() {
        let row = row(&[1]);
        let codec_bytes = {
            let (_directory, manager) = create_manager();
            let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
            let resources = crate::execution::QueryResourceContext::new(buffer_manager).unwrap();
            let grant = resources.try_allocate(0).unwrap();
            let mut calibration = ExternalSort::new_accounted(manager, 1, vec![], grant);
            calibration
                .spill_sorted_run_accounted(std::slice::from_ref(&row))
                .unwrap();
            calibration.workspace_granted_bytes()
        };
        let directory = TempDir::new().unwrap();
        let io = Arc::new(CountingCreateIo::default());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(codec_bytes);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(manager, 1, vec![], grant);
        let observation = TestGrantObservation::new(sort.total_granted_bytes());
        let observer = observation.observer();
        let error = sort
            .spill_sorted_run_accounted_observing(std::slice::from_ref(&row), &observer)
            .unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Memory(MemoryGrantError::LimitExceeded { .. })
        ));
        assert_eq!(sort.workspace_granted_bytes(), codec_bytes);
        assert_eq!(sort.run_catalog_granted_bytes(), 0);
        assert_eq!(sort.total_granted_bytes(), codec_bytes);
        assert_eq!(observation.current(), codec_bytes);
        assert_eq!(buffer_manager.allocated(), codec_bytes);
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(io.creates.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);

        let public_error = sort.spill_sorted_run(vec![row]).unwrap_err();
        assert_eq!(public_error.kind(), std::io::ErrorKind::OutOfMemory);
        assert!(
            public_error
                .get_ref()
                .and_then(|source| source.downcast_ref::<MemoryGrantError>())
                .is_some(),
            "the public compatibility error must retain the catalog grant denial"
        );
        assert_eq!(io.creates.load(AtomicOrdering::Relaxed), 0);

        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_writer_denial_precedes_file_and_provider_creation() {
        let staged = row(&[1]);
        let stable_bytes = {
            let (_directory, manager) = create_manager();
            let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
            let resources = crate::execution::QueryResourceContext::new(buffer_manager).unwrap();
            let grant = resources.try_allocate(0).unwrap();
            let mut calibration = ExternalSort::new_accounted(manager, 1, vec![], grant);
            calibration
                .spill_sorted_run_accounted(std::slice::from_ref(&staged))
                .unwrap();
            assert_eq!(calibration.writer_workspace_granted_bytes(), 0);
            calibration.total_granted_bytes()
        };
        let directory = TempDir::new().unwrap();
        let io = Arc::new(CountingCreateIo::default());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let writer_bytes = qualified_writer_base_bytes(&manager);
        assert!(writer_bytes > 0);
        let budget = stable_bytes.checked_add(writer_bytes - 1).unwrap();
        let buffer_manager = buffer_manager_with_exact_budget(budget);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);

        let error = sort
            .spill_sorted_run_accounted(std::slice::from_ref(&staged))
            .unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Memory(MemoryGrantError::LimitExceeded { .. })
        ));
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(sort.total_granted_bytes(), stable_bytes);
        assert_eq!(resources.query_stats().allocated_bytes, stable_bytes);
        assert_eq!(buffer_manager.allocated(), stable_bytes);
        assert_eq!(io.creates.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);

        drop(sort);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_writer_row_denial_precedes_payload_write() {
        let staged = vec![Value::String("x".repeat(4_096).into())];
        let stable_bytes = {
            let (_directory, manager) = create_manager();
            let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
            let resources = crate::execution::QueryResourceContext::new(buffer_manager).unwrap();
            let grant = resources.try_allocate(0).unwrap();
            let mut calibration = ExternalSort::new_accounted(manager, 1, vec![], grant);
            calibration
                .spill_sorted_run_accounted(std::slice::from_ref(&staged))
                .unwrap();
            calibration.total_granted_bytes()
        };
        let directory = TempDir::new().unwrap();
        let io = Arc::new(CountingReadPayloadIo::default());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let writer_bytes = qualified_writer_base_bytes(&manager);
        let seal_slack = 1_024;
        let budget = stable_bytes
            .checked_add(writer_bytes)
            .and_then(|bytes| bytes.checked_add(seal_slack))
            .unwrap();
        let buffer_manager = buffer_manager_with_exact_budget(budget);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );

        let error = sort
            .spill_sorted_run_accounted(std::slice::from_ref(&staged))
            .unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Memory(MemoryGrantError::LimitExceeded { .. })
        ));
        assert_eq!(
            io.writes.load(AtomicOrdering::Relaxed),
            2,
            "FileStart and SortRunStart may publish; denied SortRow payload must not"
        );
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(sort.total_granted_bytes(), stable_bytes);
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn writer_capacity_override_denial_precedes_allocation_and_create() {
        let staged = row(&[1]);
        let base = {
            let (_directory, manager) = create_manager();
            let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
            let resources = crate::execution::QueryResourceContext::new(buffer_manager).unwrap();
            let grant = resources.try_allocate(0).unwrap();
            let mut calibration = ExternalSort::new_accounted(manager, 1, vec![], grant);
            calibration
                .spill_sorted_run_accounted(std::slice::from_ref(&staged))
                .unwrap();
            calibration.total_granted_bytes()
        };
        let directory = TempDir::new().unwrap();
        let io = Arc::new(CountingCreateIo::default());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let requested = qualified_writer_buffer_requested_bytes();
        let writer_base = qualified_writer_base_bytes(&manager);
        let buffer_manager =
            buffer_manager_with_exact_budget(base.checked_add(writer_base).unwrap());
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);
        sort.writer_workspace.prepared_capacity = Some(requested.checked_add(1).unwrap());

        let error = sort
            .spill_sorted_run_accounted(std::slice::from_ref(&staged))
            .unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Memory(MemoryGrantError::LimitExceeded { .. })
        ));
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(sort.total_granted_bytes(), base);
        assert_eq!(buffer_manager.allocated(), base);
        assert_eq!(resources.query_stats().allocated_bytes, base);
        assert_eq!(io.creates.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);

        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn writer_capacity_reconciliation_charges_the_forced_actual_peak() {
        let requested = qualified_writer_buffer_requested_bytes();
        let forced_request = requested.checked_add(1).unwrap();
        let forced_actual = SpillWriterBuffer::prepare_with_capacity(forced_request)
            .unwrap()
            .capacity();
        assert!(forced_actual > requested);

        let directory = TempDir::new().unwrap();
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let io = Arc::new(AccountingCreateIo::new(Arc::clone(&buffer_manager)));
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);
        sort.writer_workspace.prepared_capacity = Some(forced_request);
        let observation = TestGrantObservation::new(sort.total_granted_bytes());
        let observer = observation.observer();
        sort.spill_sorted_run_accounted_observing(&[row(&[1])], &observer)
            .unwrap();

        let stable = sort.total_granted_bytes();
        let create_allocations = io.allocations();
        assert_eq!(create_allocations.len(), 1);
        let peak = create_allocations[0];
        assert_eq!(
            peak.checked_sub(stable),
            Some(
                forced_actual
                    .checked_add(manager.qualified_file_workspace_bound().unwrap())
                    .unwrap()
            )
        );
        assert!(observation.peak() >= peak);
        assert_eq!(observation.current(), stable);
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(buffer_manager.allocated(), stable);
        assert_eq!(resources.query_stats().allocated_bytes, stable);
        assert_eq!(sort.num_runs(), 1);

        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn impossible_writer_capacity_fails_before_allocation_or_create() {
        let staged = row(&[1]);
        let base = {
            let (_directory, manager) = create_manager();
            let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
            let resources = crate::execution::QueryResourceContext::new(buffer_manager).unwrap();
            let grant = resources.try_allocate(0).unwrap();
            let mut calibration = ExternalSort::new_accounted(manager, 1, vec![], grant);
            calibration
                .spill_sorted_run_accounted(std::slice::from_ref(&staged))
                .unwrap();
            calibration.total_granted_bytes()
        };
        let directory = TempDir::new().unwrap();
        let io = Arc::new(CountingCreateIo::default());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let writer_base = qualified_writer_base_bytes(&manager);
        let buffer_manager =
            buffer_manager_with_exact_budget(base.checked_add(writer_base).unwrap());
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(manager, 1, vec![], grant);
        sort.writer_workspace.prepared_capacity = Some(usize::MAX);

        let error = sort
            .spill_sorted_run_accounted(std::slice::from_ref(&staged))
            .unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Memory(MemoryGrantError::ArithmeticOverflow { .. })
        ));
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(sort.total_granted_bytes(), base);
        assert_eq!(buffer_manager.allocated(), base);
        assert_eq!(resources.query_stats().allocated_bytes, base);
        assert_eq!(io.creates.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);

        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_writer_charge_is_visible_at_create_and_released_before_catalog_push() {
        let directory = TempDir::new().unwrap();
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let io = Arc::new(AccountingCreateIo::new(Arc::clone(&buffer_manager)));
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);
        let staged = row(&[1]);
        let observation = TestGrantObservation::new(sort.total_granted_bytes());
        let observer = observation.observer();
        sort.spill_sorted_run_accounted_observing(std::slice::from_ref(&staged), &observer)
            .unwrap();

        let base = sort.total_granted_bytes();
        let create_allocations = io.allocations();
        assert_eq!(create_allocations.len(), 1);
        let peak = create_allocations[0];
        let writer_bytes = peak.checked_sub(base).unwrap();
        assert!(writer_bytes >= qualified_writer_base_bytes(&manager));
        let operation_peak = observation.peak();
        assert!(operation_peak >= peak);
        assert_eq!(observation.current(), base);
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(resources.query_stats().allocated_bytes, base);
        assert_eq!(buffer_manager.allocated(), base);
        assert_eq!(sort.num_runs(), 1);

        let second_observation = TestGrantObservation::new(base);
        let observer = second_observation.observer();
        sort.spill_sorted_run_accounted_observing(std::slice::from_ref(&staged), &observer)
            .unwrap();

        assert_eq!(io.allocations(), vec![peak, peak]);
        assert_eq!(second_observation.peak(), operation_peak);
        assert_eq!(second_observation.current(), base);
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(sort.total_granted_bytes(), base);
        assert_eq!(buffer_manager.allocated(), base);
        assert_eq!(sort.num_runs(), 2);

        drop(sort);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_scalar_observer_publishes_exact_live_usage() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let retained_grant = resources.try_allocate(37).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(manager, 1, vec![], grant);
        let retained_bytes = Cell::new(retained_grant.size());
        let external_bytes = Cell::new(sort.total_granted_bytes());
        let state = crate::execution::operators::push::spill_state::OperatorSpillState::new(
            "sealed sort observer".to_string(),
        );
        let observer =
            ExternalSortGrantObserver::new(&external_bytes, &retained_bytes, Some(&state));

        sort.spill_sorted_run_accounted_observing(&[row(&[1])], &observer)
            .unwrap();

        let expected = retained_grant
            .size()
            .checked_add(sort.total_granted_bytes())
            .unwrap();
        assert_eq!(external_bytes.get(), sort.total_granted_bytes());
        assert_eq!(state.usage(), expected);
        assert_eq!(buffer_manager.allocated(), expected);
    }

    #[test]
    fn poisoned_retained_handoff_rejects_every_direct_transition_without_mutation() {
        let external_bytes = Cell::new(41usize);
        let retained_bytes = Cell::new(0usize);
        let row_bytes = Cell::new(9usize);
        let state = crate::execution::operators::push::spill_state::OperatorSpillState::new(
            "poisoned retained handoff".to_string(),
        );
        state.set_usage(external_bytes.get());
        let observer =
            ExternalSortGrantObserver::new(&external_bytes, &retained_bytes, Some(&state));
        observer.poison_unacknowledged_retained();

        let assert_poisoned = |error: MemoryGrantError| {
            assert!(matches!(
                error,
                MemoryGrantError::AccountingPoisoned {
                    account: "exact owned sort unacknowledged retained handoff"
                }
            ));
            assert_eq!(external_bytes.get(), usize::MAX);
            assert_eq!(retained_bytes.get(), 0);
            assert_eq!(row_bytes.get(), 9);
            assert_eq!(state.usage(), usize::MAX);
        };

        assert_poisoned(observer.publish(0).unwrap_err());
        assert_poisoned(observer.publish_retained(0).unwrap_err());
        assert_poisoned(
            observer
                .transfer_row_to_retained(32, 0, &row_bytes, 9)
                .unwrap_err(),
        );
        assert_poisoned(
            observer
                .transfer_retained_to_external(41, 50, 9)
                .unwrap_err(),
        );
        assert_poisoned(observer.transfer_control_to_retained(41, 9).unwrap_err());

        observer.finish_control_transfer(9);
        assert_eq!(external_bytes.get(), usize::MAX);
        assert_eq!(retained_bytes.get(), 0);
        assert_eq!(row_bytes.get(), 9);
        assert_eq!(state.usage(), usize::MAX);
    }

    #[test]
    fn accounted_final_heap_keeps_single_comparator_owner_through_replacement() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let comparator =
            SemanticRowComparator::new(|left, right| compare_values_total(&left[0], &right[0]));
        let weak_comparator = Arc::downgrade(match &comparator.compare {
            Comparison::Infallible(compare) => compare,
            _ => panic!("test native comparator"),
        });
        let cancellation = crate::execution::QueryExecutionControl::new().token();
        let mut sort = ExternalSort::new_accounted_with_comparator_and_cancellation(
            Arc::clone(&manager),
            1,
            comparator,
            grant,
            cancellation,
        );
        for values in [[1, 5], [2, 6], [3, 7], [4, 8]] {
            sort.spill_sorted_run_accounted(&[row(&[values[0]]), row(&[values[1]])])
                .unwrap();
        }
        assert_eq!(weak_comparator.strong_count(), 1);

        let mut cursor = sort
            .merge_cursor_accounted_observing(Vec::new(), 1, ExternalSortGrantObserver::inert())
            .unwrap();
        assert_eq!(weak_comparator.strong_count(), 1);
        let heap_pointer = cursor.cursor.heap.pointer();
        let heap_capacity = cursor.cursor.heap.capacity();
        let mut output = Vec::new();
        for _ in 0..7 {
            let chunk = cursor.next_chunk_accounted_observing().unwrap().unwrap();
            output.extend_from_slice(chunk.rows());
            assert_eq!(weak_comparator.strong_count(), 1);
            assert_eq!(cursor.cursor.heap.pointer(), heap_pointer);
            assert_eq!(cursor.cursor.heap.capacity(), heap_capacity);
        }
        let chunk = cursor.next_chunk_accounted_observing().unwrap().unwrap();
        output.extend_from_slice(chunk.rows());
        assert!(cursor.next_chunk_accounted_observing().unwrap().is_none());

        assert_eq!(
            output,
            [
                row(&[1]),
                row(&[2]),
                row(&[3]),
                row(&[4]),
                row(&[5]),
                row(&[6]),
                row(&[7]),
                row(&[8]),
            ]
        );
        drop(cursor);
        assert_eq!(weak_comparator.strong_count(), 1);
        drop(sort);
        assert_eq!(weak_comparator.strong_count(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn exact_owned_container_capacities_and_layout_math_are_exact() {
        for capacity in 0..=1024usize {
            let heap = ComparatorMinHeap::<ExactOwnedHeapEntry>::try_with_exact_capacity(capacity)
                .unwrap();
            assert_eq!(heap.capacity(), capacity);
            assert_eq!(
                cursor_capacity_bytes::<ExactOwnedHeapEntry>(capacity).unwrap(),
                std::alloc::Layout::array::<ExactOwnedHeapEntry>(capacity)
                    .unwrap()
                    .size()
            );

            let mut readers = ExactOwnedReaderSlots::new_in(Global);
            readers.try_reserve_exact(capacity).unwrap();
            assert_eq!(readers.capacity(), capacity);
            assert_eq!(
                cursor_capacity_bytes::<Option<ExactOwnedRunReader>>(capacity).unwrap(),
                std::alloc::Layout::array::<Option<ExactOwnedRunReader>>(capacity)
                    .unwrap()
                    .size()
            );
        }
        assert!(cursor_capacity_bytes::<ExactOwnedHeapEntry>(usize::MAX).is_err());
        assert!(cursor_capacity_bytes::<Option<ExactOwnedRunReader>>(usize::MAX).is_err());
    }

    #[test]
    fn custom_comparator_fallback_does_not_probe_exact_reader_capability() {
        let directory = TempDir::new().unwrap();
        let queries = Arc::new(AtomicUsize::new(0));
        let manager = qualification_trap_manager(directory.path(), Arc::clone(&queries), true);
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let cancellation = crate::execution::QueryExecutionControl::new().token();
        let mut sort = ExternalSort::new_accounted_with_comparator_and_cancellation(
            Arc::clone(&manager),
            1,
            SemanticRowComparator::new(|left, right| compare_values_total(&left[0], &right[0])),
            grant,
            cancellation,
        );
        sort.spill_sorted_run_accounted(&[row(&[2])]).unwrap();
        sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();

        assert!(!sort.exact_owned_disk_base_shape_eligible());
        assert!(!sort.exact_owned_disk_shape_eligible());
        assert_eq!(queries.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(
            sort.merge_all_accounted(Vec::new()).unwrap(),
            [row(&[1]), row(&[2])]
        );
        assert_eq!(queries.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(manager.active_file_count(), 0);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn discarded_exact_probe_is_cached_and_consumes_no_reader_authority() {
        let directory = TempDir::new().unwrap();
        let queries = Arc::new(AtomicUsize::new(0));
        let manager = qualification_trap_manager(directory.path(), Arc::clone(&queries), false);
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run_accounted(&[row(&[2])]).unwrap();
        sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();

        assert!(sort.exact_owned_disk_shape_eligible());
        assert!(sort.exact_owned_disk_shape_eligible());
        assert_eq!(queries.load(AtomicOrdering::Relaxed), 2);
        assert!(sort.try_enable_exact_owned_output(None));
        let external = Cell::new(sort.total_granted_bytes());
        let retained = Cell::new(0);
        let observer = ExternalSortGrantObserver::new(&external, &retained, None);
        let mut cursor = sort.into_exact_owned_disk_cursor(observer).unwrap();
        assert_eq!(queries.load(AtomicOrdering::Relaxed), 2);
        cursor.abort().unwrap();
        assert_eq!(cursor.checked_granted_bytes().unwrap(), 0);
        assert_eq!(external.get(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn exact_final_publisher_denial_falls_back_with_sorter_and_charge_intact() {
        let budget = 4 * 1024 * 1024;
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(budget);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run_accounted(&[row(&[2])]).unwrap();
        sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
        assert!(sort.exact_owned_disk_shape_eligible());
        let before = sort.total_granted_bytes();
        let required = AccountedErrorPublisher::<ExactOwnedFinalFailure>::required_bytes();
        assert!(required > 0);
        let available = budget.checked_sub(buffer_manager.allocated()).unwrap();
        assert!(available >= required);
        let blocker = buffer_manager
            .try_allocate(available - (required - 1), MemoryRegion::ExecutionBuffers)
            .unwrap();
        let spill_state = crate::execution::operators::push::spill_state::OperatorSpillState::new(
            "optional exact publisher denial".to_string(),
        );
        spill_state.set_usage(before);

        assert!(!sort.try_enable_exact_owned_output(Some(&spill_state)));
        assert!(sort.exact_failure_publisher.is_none());
        assert_eq!(sort.total_granted_bytes(), before);
        assert_eq!(spill_state.usage(), before);
        assert_eq!(sort.num_runs(), 2);
        assert!(!sort.disk_merge_started);

        drop(blocker);
        assert_eq!(
            sort.merge_all_accounted(Vec::new()).unwrap(),
            [row(&[1]), row(&[2])]
        );
        assert_eq!(manager.active_file_count(), 0);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn exact_terminal_release_failures_are_retryable_and_cannot_resume() {
        for semantic in [false, true] {
            enum FailureLane {
                Workspace,
                Writer,
                Catalog,
            }

            for lane in [
                FailureLane::Workspace,
                FailureLane::Writer,
                FailureLane::Catalog,
            ] {
                let (_directory, manager) = create_manager();
                let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
                let resources =
                    crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager))
                        .unwrap();
                let grant = resources.try_allocate(0).unwrap();
                let mut sort = if semantic {
                    semantic_fault_sort(
                        Arc::clone(&manager),
                        1,
                        vec![SortKey::ascending(0)],
                        grant,
                        &resources,
                    )
                } else {
                    ExternalSort::new_accounted(
                        Arc::clone(&manager),
                        1,
                        vec![SortKey::ascending(0)],
                        grant,
                    )
                };
                sort.spill_sorted_run_accounted(&[row(&[2])]).unwrap();
                sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
                let failure = MemoryGrantError::AccountingPoisoned {
                    account: "deterministic exact terminal release",
                };
                match lane {
                    FailureLane::Workspace => {
                        assert!(sort.workspace_granted_bytes() > 0);
                        sort.workspace.release_error = Some(failure);
                    }
                    FailureLane::Writer => {
                        sort.writer_workspace.resize_grant(17).unwrap();
                        sort.writer_workspace.release_error = Some(failure);
                    }
                    FailureLane::Catalog => {
                        sort.runs.force_next_shrink_failure_for_test(failure);
                    }
                }
                assert!(sort.try_enable_exact_owned_output(None));
                let external = Cell::new(sort.total_granted_bytes());
                let retained = Cell::new(0);
                let observer = ExternalSortGrantObserver::new(&external, &retained, None);
                let mut cursor = sort.into_exact_owned_disk_cursor(observer).unwrap();

                assert!(cursor.abort().is_err());
                assert_eq!(cursor.state, ExactOwnedCursorState::Failed);
                assert!(cursor.checked_granted_bytes().unwrap() > 0);
                assert_eq!(external.get(), cursor.checked_granted_bytes().unwrap());
                assert!(cursor.next_owned_row().is_err());
                assert_eq!(manager.active_file_count(), 0);

                cursor.sorter.workspace.release_error = None;
                cursor.sorter.writer_workspace.release_error = None;
                cursor.abort().unwrap();
                assert_eq!(cursor.state, ExactOwnedCursorState::Terminal);
                assert_eq!(cursor.checked_granted_bytes().unwrap(), 0);
                assert_eq!(external.get(), 0);
                assert_eq!(buffer_manager.allocated(), 0);
            }
        }
    }

    #[test]
    fn exact_cursor_drop_keeps_failed_catalog_cleanup_fail_closed() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run_accounted(&[row(&[2]), row(&[1])])
            .unwrap();
        sort.runs
            .force_next_shrink_failure_for_test(MemoryGrantError::AccountingPoisoned {
                account: "one-shot exact catalog drop reconciliation",
            });
        assert!(sort.try_enable_exact_owned_output(None));
        let external = Cell::new(sort.total_granted_bytes());
        let retained = Cell::new(0);
        let observer = ExternalSortGrantObserver::new(&external, &retained, None);
        let cursor = sort.into_exact_owned_disk_cursor(observer).unwrap();

        drop(cursor);

        assert_eq!(external.get(), usize::MAX);
        assert_eq!(retained.get(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn exact_final_operator_primary_recovers_without_failure_path_cloning() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
        assert!(sort.try_enable_exact_owned_output(None));
        let external = Cell::new(sort.total_granted_bytes());
        let retained = Cell::new(0);
        let observer = ExternalSortGrantObserver::new(&external, &retained, None);
        let mut cursor = sort.into_exact_owned_disk_cursor(observer).unwrap();
        let final_error = cursor.finish_operator_failure(
            OperatorError::TypeMismatch {
                expected: "exact expected type".to_string(),
                found: "exact found type".to_string(),
            },
            Some((
                ExactOwnedSortStreamError::from(ExternalSortOperationError::Memory(
                    MemoryGrantError::AccountingPoisoned {
                        account: "exact secondary reconciliation",
                    },
                )),
                "synthetic exact secondary",
            )),
            "synthetic exact cleanup",
        );
        assert!(matches!(
            &final_error,
            OperatorError::ClassifiedAccountedFailure {
                classification: AccountedFailureClassification::TypeMismatch,
                ..
            }
        ));

        let recovered = final_error.recover_accounted_primary();
        let OperatorError::Context { source, context } = recovered else {
            panic!("exact final operator primary was not recovered")
        };
        assert!(matches!(
            *source,
            OperatorError::TypeMismatch { ref expected, ref found }
                if expected == "exact expected type" && found == "exact found type"
        ));
        assert!(context.contains("synthetic exact secondary"));
        assert!(context.contains("exact secondary reconciliation"));
        drop(cursor);
        assert_eq!(external.get(), 0);
        assert_eq!(retained.get(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn unused_final_publisher_release_failure_is_retryable_not_false_success() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
        assert!(sort.try_enable_exact_owned_output(None));
        let publisher_bytes = sort
            .exact_failure_publisher
            .as_ref()
            .map(AccountedErrorPublisher::granted_bytes)
            .unwrap();
        let external = Cell::new(sort.total_granted_bytes());
        let retained = Cell::new(0);
        let observer = ExternalSortGrantObserver::new(&external, &retained, None);
        let mut cursor = sort.into_exact_owned_disk_cursor(observer).unwrap();
        cursor.sorter.exact_final_publisher_release_error =
            Some(MemoryGrantError::AccountingPoisoned {
                account: "deterministic unused exact final publisher release",
            });

        assert!(cursor.abort().is_err());
        assert_eq!(cursor.state, ExactOwnedCursorState::Failed);
        assert!(cursor.sorter.exact_failure_publisher.is_none());
        assert_eq!(cursor.checked_granted_bytes().unwrap(), publisher_bytes);
        assert_eq!(external.get(), publisher_bytes);
        assert_eq!(retained.get(), 0);
        assert_eq!(buffer_manager.allocated(), publisher_bytes);
        assert!(cursor.next_owned_row().is_err());

        cursor.abort().unwrap();
        assert_eq!(cursor.state, ExactOwnedCursorState::Terminal);
        assert_eq!(cursor.checked_granted_bytes().unwrap(), 0);
        assert_eq!(external.get(), 0);
        assert_eq!(retained.get(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn live_reader_unused_publisher_release_failure_defers_terminal_success() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run_accounted(&[row(&[1]), row(&[2])])
            .unwrap();
        assert!(sort.try_enable_exact_owned_output(None));
        let external = Cell::new(sort.total_granted_bytes());
        let retained = Cell::new(0);
        let observer = ExternalSortGrantObserver::new(&external, &retained, None);
        let mut cursor = sort.into_exact_owned_disk_cursor(observer).unwrap();
        let injected = MemoryGrantError::AccountingPoisoned {
            account: "deterministic unused exact reader publisher release",
        };
        let _failure_guard =
            super::super::file::fail_next_unused_publisher_release_for_test(injected.clone());

        let cleanup = cursor.abort().unwrap_err();
        assert_eq!(cleanup.reader_publication_release.as_ref(), Some(&injected));
        assert_eq!(cursor.state, ExactOwnedCursorState::Failed);
        let charged = cursor.checked_granted_bytes().unwrap();
        assert!(charged > 0);
        assert_eq!(external.get(), charged);
        assert_eq!(retained.get(), 0);
        assert_eq!(buffer_manager.allocated(), charged);
        assert!(cursor.next_owned_row().is_err());
        drop(cleanup);

        cursor.abort().unwrap();
        assert_eq!(cursor.state, ExactOwnedCursorState::Terminal);
        assert_eq!(cursor.checked_granted_bytes().unwrap(), 0);
        assert_eq!(external.get(), 0);
        assert_eq!(retained.get(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn exact_open_failure_escapes_typed_with_global_charge_but_not_sort_attribution() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(QualifiedOpenFailureIo))
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
        let spill_state = crate::execution::operators::push::spill_state::OperatorSpillState::new(
            "typed exact reader error attribution".to_string(),
        );
        assert!(sort.try_enable_exact_owned_output(Some(&spill_state)));
        let external = Cell::new(sort.total_granted_bytes());
        let retained = Cell::new(0);
        let observer = ExternalSortGrantObserver::new(&external, &retained, Some(&spill_state));

        let error = match sort.into_exact_owned_disk_cursor(observer) {
            Err(error) => error,
            Ok(mut cursor) => {
                cursor.abort().unwrap();
                panic!("injected exact reader-open failure did not fire")
            }
        };

        assert_eq!(external.get(), 0);
        assert_eq!(retained.get(), 0);
        assert_eq!(spill_state.usage(), 0);
        assert_eq!(manager.active_file_count(), 0);
        let ExactOwnedSortStreamError {
            primary:
                ExactOwnedSortPrimary::Accounted {
                    authority: accounted,
                    ..
                },
        } = error
        else {
            panic!("typed reader-open failure was flattened or cleanup failed")
        };
        assert!(accounted.is::<ProviderAccountedReaderError>());
        assert_eq!(
            accounted
                .inspect::<ProviderAccountedReaderError, _>(ProviderAccountedReaderError::kind),
            Some(std::io::ErrorKind::PermissionDenied)
        );
        let charged = buffer_manager.allocated();
        assert!(charged >= accounted.granted_bytes());
        assert!(charged > 0);
        assert_eq!(resources.query_stats().allocated_bytes, charged);
        let clone = accounted.clone();
        assert!(accounted.ptr_eq(&clone));
        drop(accounted);
        assert_eq!(buffer_manager.allocated(), charged);
        drop(clone);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn exact_corrupt_decode_retains_primary_and_reader_release_secondary_through_retry() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(CorruptOpenedSortRowProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run_accounted(&[row(&[1]), row(&[2])])
            .unwrap();
        assert!(sort.try_enable_exact_owned_output(None));
        let injected = MemoryGrantError::AccountingPoisoned {
            account: "deterministic corrupt-row reader publisher release",
        };
        let _failure_guard =
            super::super::file::fail_next_unused_publisher_release_for_test(injected.clone());
        let external = Cell::new(sort.total_granted_bytes());
        let retained = Cell::new(0);
        let observer = ExternalSortGrantObserver::new(&external, &retained, None);

        let error = match sort.into_exact_owned_disk_cursor(observer) {
            Err(error) => error,
            Ok(mut cursor) => {
                cursor.abort().unwrap();
                panic!("corrupt exact row unexpectedly decoded")
            }
        };
        let accounted = unwrap_exact_final_failure(error);

        assert_eq!(external.get(), 0);
        assert_eq!(retained.get(), 0);
        assert_eq!(manager.active_file_count(), 0);
        let decoded_bytes = accounted
            .inspect::<ExactOwnedFinalFailure, _>(|failure| {
                let decoded = failure
                    .decoded_primary()
                    .unwrap_or_else(|| panic!("final failure lost decoder primary: {failure:?}"));
                assert_eq!(decoded.kind(), std::io::ErrorKind::InvalidData);
                assert!(decoded.to_string().contains("Unknown value tag"));
                assert_eq!(
                    failure.pending_reader_publication_release(),
                    Some(&injected),
                    "reader publication failure remains independent of decoder primary"
                );
                decoded.granted_bytes()
            })
            .expect("accounted handle retains exact final failure");
        assert!(decoded_bytes > 0);
        let charged = buffer_manager.allocated();
        assert!(charged >= accounted.granted_bytes() + decoded_bytes);
        let clone = accounted.clone();
        drop(resources);
        drop(accounted);
        assert_eq!(buffer_manager.allocated(), charged);
        drop(clone);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn exact_midstream_reader_error_and_panic_retain_typed_authority_after_cursor_cleanup() {
        for finalization in [false, true] {
            for semantic in [false, true] {
                for failure in [
                    ArmedQualifiedReadFailure::Error,
                    ArmedQualifiedReadFailure::Panic,
                ] {
                    let directory = TempDir::new().unwrap();
                    let io = Arc::new(ArmedQualifiedReadIo::new(failure));
                    let manager = Arc::new(
                        crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                            .provider(
                                Arc::new(super::super::CleartextSpillRecordProvider),
                                super::super::SpillFrameLimits::format_max(),
                            )
                            .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                            .build()
                            .unwrap(),
                    );
                    let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
                    let resources =
                        crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager))
                            .unwrap();
                    let grant = resources.try_allocate(0).unwrap();
                    let mut sort = if semantic {
                        semantic_fault_sort(
                            Arc::clone(&manager),
                            1,
                            vec![SortKey::ascending(0)],
                            grant,
                            &resources,
                        )
                    } else {
                        ExternalSort::new_accounted(
                            Arc::clone(&manager),
                            1,
                            vec![SortKey::ascending(0)],
                            grant,
                        )
                    };
                    sort.spill_sorted_run_accounted(&[row(&[1]), row(&[2]), row(&[3])])
                        .unwrap();
                    assert!(sort.try_enable_exact_owned_output(None));
                    let external = Cell::new(sort.total_granted_bytes());
                    let retained = Cell::new(0);
                    let observer = ExternalSortGrantObserver::new(&external, &retained, None);
                    let mut cursor = sort.into_exact_owned_disk_cursor(observer).unwrap();

                    let first = cursor.next_owned_row().unwrap().unwrap();
                    let first_bytes = first.granted_bytes();
                    cursor.release_transferred_retained().unwrap();
                    if finalization {
                        let second = cursor.next_owned_row().unwrap().unwrap();
                        assert_eq!(second.values(), &[Value::Int64(2)]);
                        cursor.release_transferred_retained().unwrap();
                        drop(second);
                        // The third row is already the head. Its advance reads
                        // FileEnd through the qualified provider finalization seam.
                    }
                    io.arm();
                    let primary = cursor.next_owned_row().unwrap_err();
                    assert!(cursor.next_owned_row().is_err());
                    let ExactOwnedSortStreamError {
                        primary:
                            ExactOwnedSortPrimary::Accounted {
                                classification,
                                authority: operation,
                            },
                    } = &primary
                    else {
                        panic!("midstream qualified reader failure lost its accounted owner")
                    };
                    assert!(matches!(
                        classification,
                        AccountedFailureClassification::Execution
                    ));
                    let authority = operation
                        .inspect::<super::super::file::ProviderAccountedReaderOperationError, _>(
                            |error| {
                                match failure {
                                    ArmedQualifiedReadFailure::Error => {
                                        assert_eq!(
                                            error.kind(),
                                            std::io::ErrorKind::PermissionDenied
                                        );
                                        assert!(!error.is_fatal_captured_panic());
                                    }
                                    ArmedQualifiedReadFailure::Panic => {
                                        assert!(error.is_fatal_captured_panic());
                                        assert!(
                                            error.panic_payload_is::<ExactReaderPanicSentinel>()
                                        );
                                    }
                                }
                                error.retained_authority_bytes()
                            },
                        )
                        .expect("typed reader-operation payload remains inspectable");
                    assert!(authority > 0);

                    let final_error =
                        cursor.finish_stream_failure(primary, "midstream reader terminal cleanup");
                    let OperatorError::ClassifiedAccountedFailure {
                        classification,
                        authority: accounted,
                    } = final_error
                    else {
                        panic!("midstream reader owner was flattened during terminal cleanup")
                    };
                    assert!(matches!(
                        classification,
                        AccountedFailureClassification::Execution
                    ));
                    assert!(
                        accounted.is::<super::super::file::ProviderAccountedReaderOperationError>()
                    );
                    assert_eq!(cursor.state, ExactOwnedCursorState::Terminal);
                    assert!(cursor.next_owned_row().unwrap().is_none());
                    assert_eq!(cursor.checked_granted_bytes().unwrap(), 0);
                    assert_eq!(external.get(), 0);
                    assert_eq!(retained.get(), 0);
                    assert_eq!(manager.active_file_count(), 0);
                    let charged = buffer_manager.allocated();
                    assert!(charged >= first_bytes + accounted.granted_bytes() + authority);
                    assert_eq!(resources.query_stats().allocated_bytes, charged);

                    let clone = accounted.clone();
                    drop(cursor);
                    drop(resources);
                    drop(accounted);
                    assert_eq!(buffer_manager.allocated(), charged);
                    drop(clone);
                    assert_eq!(buffer_manager.allocated(), first_bytes);
                    drop(first);
                    assert_eq!(buffer_manager.allocated(), 0);
                }
            }
        }
    }

    #[test]
    fn exact_partial_cursor_drop_cleans_runs_but_leaves_returned_row_accounted() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run_accounted(&[row(&[1]), row(&[2]), row(&[3])])
            .unwrap();
        assert!(sort.try_enable_exact_owned_output(None));
        let external = Cell::new(sort.total_granted_bytes());
        let retained = Cell::new(0);
        let observer = ExternalSortGrantObserver::new(&external, &retained, None);
        let mut cursor = sort.into_exact_owned_disk_cursor(observer).unwrap();

        let first = cursor.next_owned_row().unwrap().unwrap();
        let first_bytes = first.granted_bytes();
        assert!(first_bytes > 0);
        assert_eq!(retained.get(), first_bytes);
        assert_eq!(first.values(), &[Value::Int64(1)]);
        drop(cursor);

        assert_eq!(external.get(), 0);
        assert_eq!(retained.get(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(resources.query_stats().allocated_bytes, first_bytes);
        assert_eq!(buffer_manager.allocated(), first_bytes);
        drop(resources);
        assert_eq!(buffer_manager.allocated(), first_bytes);
        drop(first);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn failed_output_grant_returns_atomically_from_retained_slot_to_retry_frontier() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
        assert!(sort.try_enable_exact_owned_output(None));
        let external = Cell::new(sort.total_granted_bytes());
        let retained = Cell::new(0);
        let observer = ExternalSortGrantObserver::new(&external, &retained, None);
        let mut cursor = sort.into_exact_owned_disk_cursor(observer).unwrap();
        let row = cursor.next_owned_row().unwrap().unwrap();

        let error = crate::execution::accounted_chunk::try_accounted_chunk_from_sort_row(
            row,
            2,
            cursor.output_observer(),
        )
        .unwrap_err();
        let (primary, grant) = error.into_parts();
        assert!(matches!(
            primary,
            crate::execution::accounted_chunk::AccountedSortChunkPrimary::InvalidShape(_)
        ));
        let transferred_bytes = grant.size();
        assert!(transferred_bytes > 0);
        assert_eq!(retained.get(), transferred_bytes);
        let total_before = buffer_manager.allocated();

        cursor.reclaim_failed_output_grant(grant).unwrap();

        assert_eq!(retained.get(), 0);
        assert_eq!(external.get(), cursor.checked_granted_bytes().unwrap());
        assert_eq!(buffer_manager.allocated(), total_before);
        cursor.abort().unwrap();
        assert_eq!(cursor.checked_granted_bytes().unwrap(), 0);
        assert_eq!(external.get(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn exact_cleanup_keeps_typed_delete_error_and_panic_in_distinct_slots() {
        for order in [
            CompoundDeleteOrder::ErrorThenPanic,
            CompoundDeleteOrder::PanicThenError,
        ] {
            let directory = TempDir::new().unwrap();
            let io = Arc::new(CompoundDeleteIo::new(order));
            let manager = Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .provider(
                        Arc::new(CorruptOpenedSortRowProvider),
                        super::super::SpillFrameLimits::format_max(),
                    )
                    .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                    .build()
                    .unwrap(),
            );
            let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
            let resources =
                crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
            let grant = resources.try_allocate(0).unwrap();
            let mut sort = ExternalSort::new_accounted(
                Arc::clone(&manager),
                1,
                vec![SortKey::ascending(0)],
                grant,
            );
            sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
            sort.spill_sorted_run_accounted(&[row(&[2])]).unwrap();
            assert!(sort.try_enable_exact_owned_output(None));
            let external = Cell::new(sort.total_granted_bytes());
            let retained = Cell::new(0);
            let observer = ExternalSortGrantObserver::new(&external, &retained, None);

            let error = match sort.into_exact_owned_disk_cursor(observer) {
                Err(error) => error,
                Ok(mut cursor) => {
                    cursor.abort().unwrap();
                    panic!("compound exact cleanup fixture did not fail decoding")
                }
            };
            let accounted = unwrap_exact_final_failure(error);

            accounted
                .inspect::<ExactOwnedFinalFailure, _>(|failure| {
                    assert!(
                        failure.decoded_primary().is_some(),
                        "compound fixture lost decoder primary: {failure:?}"
                    );
                    let (ExactOwnedFinalCause::Terminal(terminal), _) = failure
                        .cleanup
                        .as_ref()
                        .expect("terminal cleanup remains a distinct cause")
                    else {
                        panic!("cleanup cause lost its terminal structure")
                    };
                    let deletion = terminal
                        .run_cleanup
                        .as_ref()
                        .expect("typed deletion error survives cleanup panic");
                    assert!(
                        deletion
                            .to_string()
                            .contains("deterministic exact cleanup deletion error")
                    );
                    let panic = terminal
                        .run_cleanup_panic
                        .as_ref()
                        .expect("cleanup panic survives typed deletion error");
                    assert_eq!(
                        panic.payload().downcast_ref::<&str>(),
                        Some(&"deterministic exact cleanup deletion panic")
                    );
                })
                .unwrap();
            assert_eq!(external.get(), 0);
            assert_eq!(retained.get(), 0);
            assert_eq!(manager.active_file_count(), 0);
            drop(resources);
            drop(accounted);
            assert_eq!(buffer_manager.allocated(), 0);
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct ExactDecoderPanicSentinel(&'static str);

    #[derive(Debug)]
    struct CountedDecoderPanicOnDrop {
        drops: Arc<AtomicUsize>,
    }

    impl Drop for CountedDecoderPanicOnDrop {
        fn drop(&mut self) {
            self.drops.fetch_add(1, AtomicOrdering::Relaxed);
            panic!("hostile exact decoder panic-payload destructor")
        }
    }

    #[derive(Debug)]
    struct CountedDecodedErrorOnDrop {
        message: &'static str,
        drops: Arc<AtomicUsize>,
    }

    impl std::fmt::Display for CountedDecodedErrorOnDrop {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.message)
        }
    }

    impl std::error::Error for CountedDecodedErrorOnDrop {}

    impl Drop for CountedDecodedErrorOnDrop {
        fn drop(&mut self) {
            self.drops.fetch_add(1, AtomicOrdering::Relaxed);
            panic!("hostile exact decoded-error destructor")
        }
    }

    fn hostile_decoded_error(
        resources: &crate::execution::QueryResourceContext,
        bytes: usize,
        message: &'static str,
        drops: Arc<AtomicUsize>,
    ) -> ExactOwnedDecodedError {
        ExactOwnedDecodedError::new(
            ExactOwnedDecodedPrimary::Hostile(Box::new(CountedDecodedErrorOnDrop {
                message,
                drops,
            })),
            resources.try_allocate(bytes).unwrap(),
        )
    }

    #[test]
    fn exact_decoded_error_destructor_strands_only_its_matching_authority() {
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let error = hostile_decoded_error(
            &resources,
            8192,
            "hostile decoded error",
            Arc::clone(&drops),
        );
        assert_eq!(error.granted_bytes(), 8192);
        drop(resources);

        drop(error);
        assert_eq!(drops.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(buffer_manager.allocated(), 8192);
    }

    #[test]
    fn exact_final_envelope_cleans_every_hostile_slot_and_strands_exact_authority() {
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let primary_drops = Arc::new(AtomicUsize::new(0));
        let cleanup_drops = Arc::new(AtomicUsize::new(0));
        let failure = ExactOwnedFinalFailure::from_parts(
            ExactOwnedFinalCause::Stream(ExactOwnedSortStreamError::decoded(
                hostile_decoded_error(
                    &resources,
                    4096,
                    "hostile final primary",
                    Arc::clone(&primary_drops),
                ),
            )),
            None,
            Some((
                ExactOwnedFinalCause::Stream(ExactOwnedSortStreamError::decoded(
                    hostile_decoded_error(
                        &resources,
                        16384,
                        "hostile final cleanup",
                        Arc::clone(&cleanup_drops),
                    ),
                )),
                "hostile exact final cleanup slot",
            )),
            None,
        );
        drop(resources);

        drop(failure);
        assert_eq!(primary_drops.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(cleanup_drops.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(buffer_manager.allocated(), 4096 + 16384);
    }

    #[test]
    fn exact_decoder_panic_is_owned_on_first_and_middle_rows() {
        for panic_on_middle_row in [false, true] {
            let (_directory, manager) = create_manager();
            let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
            let resources =
                crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
            let grant = resources.try_allocate(0).unwrap();
            let mut sort = ExternalSort::new_accounted(
                Arc::clone(&manager),
                1,
                vec![SortKey::ascending(0)],
                grant,
            );
            sort.spill_sorted_run_accounted(&[row(&[1]), row(&[2]), row(&[3])])
                .unwrap();
            if !panic_on_middle_row {
                sort.inject_exact_decoder_panic(Box::new(ExactDecoderPanicSentinel("first")));
            }
            assert!(sort.try_enable_exact_owned_output(None));
            let external = Cell::new(sort.total_granted_bytes());
            let retained = Cell::new(0);
            let observer = ExternalSortGrantObserver::new(&external, &retained, None);
            let mut cursor = match sort.into_exact_owned_disk_cursor(observer) {
                Ok(cursor) => cursor,
                Err(error) if !panic_on_middle_row => {
                    let accounted = unwrap_exact_final_failure(error);
                    assert_eq!(external.get(), 0);
                    assert_eq!(retained.get(), 0);
                    assert_eq!(manager.active_file_count(), 0);
                    let panic_bytes = accounted
                        .inspect::<ExactOwnedFinalFailure, _>(|failure| {
                            let panic = failure
                                .decoded_panic_primary()
                                .expect("first-row decoder panic retains its typed owner");
                            assert_eq!(
                                panic.payload().downcast_ref::<ExactDecoderPanicSentinel>(),
                                Some(&ExactDecoderPanicSentinel("first"))
                            );
                            panic.granted_bytes()
                        })
                        .unwrap();
                    assert!(panic_bytes > 0);
                    let charged = buffer_manager.allocated();
                    assert!(charged >= accounted.granted_bytes() + panic_bytes);
                    assert_eq!(resources.query_stats().allocated_bytes, charged);
                    let clone = accounted.clone();
                    drop(resources);
                    drop(accounted);
                    assert_eq!(buffer_manager.allocated(), charged);
                    drop(clone);
                    assert_eq!(buffer_manager.allocated(), 0);
                    continue;
                }
                Err(error) => panic!("middle-row fixture failed during initialization: {error}"),
            };

            let first = cursor.next_owned_row().unwrap().unwrap();
            assert_eq!(first.values(), &[Value::Int64(1)]);
            let first_bytes = first.granted_bytes();
            cursor.release_transferred_retained().unwrap();
            cursor.inject_next_decoder_panic(Box::new(ExactDecoderPanicSentinel("middle")));
            let injected = MemoryGrantError::AccountingPoisoned {
                account: "deterministic middle-row reader publisher release",
            };
            let _failure_guard =
                super::super::file::fail_next_unused_publisher_release_for_test(injected.clone());
            let error = cursor.next_owned_row().unwrap_err();
            assert!(matches!(
                &error.primary,
                ExactOwnedSortPrimary::DecodedPanic(_)
            ));
            let final_error =
                cursor.finish_stream_failure(error, "exact decoder-panic test terminal cleanup");
            let OperatorError::ClassifiedAccountedFailure {
                authority: accounted,
                ..
            } = final_error
            else {
                panic!("middle-row decoder panic lost its accounted final owner")
            };
            let panic_bytes = accounted
                .inspect::<ExactOwnedFinalFailure, _>(|failure| {
                    let panic = failure
                        .decoded_panic_primary()
                        .expect("middle-row decoder panic remains typed");
                    assert_eq!(
                        panic.payload().downcast_ref::<ExactDecoderPanicSentinel>(),
                        Some(&ExactDecoderPanicSentinel("middle"))
                    );
                    assert_eq!(
                        failure.pending_reader_publication_release(),
                        Some(&injected),
                        "reader release diagnostic must not replace decoder panic"
                    );
                    panic.granted_bytes()
                })
                .unwrap();
            assert_eq!(cursor.state, ExactOwnedCursorState::Failed);
            let retryable_bytes = cursor.checked_granted_bytes().unwrap();
            assert!(retryable_bytes > 0);
            assert_eq!(external.get(), retryable_bytes);
            assert_eq!(retained.get(), 0);
            assert_eq!(manager.active_file_count(), 0);
            assert!(cursor.next_owned_row().is_err());

            cursor.abort().unwrap();
            assert_eq!(cursor.state, ExactOwnedCursorState::Terminal);
            assert_eq!(cursor.checked_granted_bytes().unwrap(), 0);
            assert_eq!(external.get(), 0);
            let charged = buffer_manager.allocated();
            assert!(charged >= first_bytes + accounted.granted_bytes() + panic_bytes);
            assert_eq!(resources.query_stats().allocated_bytes, charged);
            let clone = accounted.clone();
            drop(cursor);
            drop(resources);
            drop(accounted);
            assert_eq!(buffer_manager.allocated(), charged);
            drop(clone);
            assert_eq!(buffer_manager.allocated(), first_bytes);
            drop(first);
            assert_eq!(buffer_manager.allocated(), 0);
        }
    }

    #[test]
    fn exact_decoder_panic_payload_destructor_strands_its_matching_authority() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        sort.inject_exact_decoder_panic(Box::new(CountedDecoderPanicOnDrop {
            drops: Arc::clone(&drops),
        }));
        assert!(sort.try_enable_exact_owned_output(None));
        let external = Cell::new(sort.total_granted_bytes());
        let retained = Cell::new(0);
        let observer = ExternalSortGrantObserver::new(&external, &retained, None);

        let error = match sort.into_exact_owned_disk_cursor(observer) {
            Err(error) => error,
            Ok(mut cursor) => {
                cursor.abort().unwrap();
                panic!("hostile exact decoder panic fixture did not fire")
            }
        };
        let accounted = unwrap_exact_final_failure(error);
        let encoded_bytes = accounted
            .inspect::<ExactOwnedFinalFailure, _>(|failure| {
                let panic = failure
                    .decoded_panic_primary()
                    .expect("hostile decoder panic retains its encoded-row owner");
                assert!(panic.payload().is::<CountedDecoderPanicOnDrop>());
                panic.granted_bytes()
            })
            .unwrap();
        assert!(encoded_bytes > 0);
        assert_eq!(external.get(), 0);
        assert_eq!(retained.get(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert!(buffer_manager.allocated() >= accounted.granted_bytes() + encoded_bytes);

        drop(resources);
        drop(accounted);
        assert_eq!(drops.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(buffer_manager.allocated(), encoded_bytes);
    }

    #[test]
    fn exact_cursor_comparator_panic_is_terminal_reconciled_and_cleanup_retryable() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        for value in [4, 1, 3, 2] {
            sort.spill_sorted_run_accounted(&[row(&[value])]).unwrap();
        }
        let armed = Arc::new(AtomicBool::new(false));
        let comparisons = Arc::new(AtomicUsize::new(0));
        let armed_for_compare = Arc::clone(&armed);
        let comparisons_for_compare = Arc::clone(&comparisons);
        sort.comparator = SemanticRowComparator::new(move |left, right| {
            comparisons_for_compare.fetch_add(1, AtomicOrdering::Relaxed);
            if armed_for_compare.load(AtomicOrdering::Relaxed) {
                std::panic::panic_any(String::from("exact cursor comparator panic"));
            }
            compare_values_total(&left[0], &right[0])
        });
        assert!(sort.try_enable_exact_owned_output(None));
        let external = Cell::new(sort.total_granted_bytes());
        let retained = Cell::new(0);
        let observer = ExternalSortGrantObserver::new(&external, &retained, None);
        let mut cursor = sort.into_exact_owned_disk_cursor(observer).unwrap();
        armed.store(true, AtomicOrdering::Relaxed);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = cursor.next_owned_row();
        }))
        .unwrap_err();

        assert_eq!(
            panic.downcast_ref::<String>().map(String::as_str),
            Some("exact cursor comparator panic")
        );
        assert_eq!(cursor.state, ExactOwnedCursorState::Failed);
        assert_eq!(external.get(), cursor.checked_granted_bytes().unwrap());
        let comparison_count = comparisons.load(AtomicOrdering::Relaxed);
        assert!(cursor.next_owned_row().is_err());
        assert_eq!(comparisons.load(AtomicOrdering::Relaxed), comparison_count);
        armed.store(false, AtomicOrdering::Relaxed);
        cursor.abort().unwrap();
        assert_eq!(cursor.checked_granted_bytes().unwrap(), 0);
        assert_eq!(external.get(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn exact_initial_heap_comparator_panic_aborts_installed_readers_and_runs() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run_accounted(&[row(&[2])]).unwrap();
        sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
        sort.comparator = SemanticRowComparator::new(|_, _| {
            std::panic::panic_any(String::from("initial exact heap comparator panic"))
        });
        assert!(sort.try_enable_exact_owned_output(None));
        let external = Cell::new(sort.total_granted_bytes());
        let retained = Cell::new(0);
        let observer = ExternalSortGrantObserver::new(&external, &retained, None);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = sort.into_exact_owned_disk_cursor(observer);
        }))
        .unwrap_err();

        assert_eq!(
            panic.downcast_ref::<String>().map(String::as_str),
            Some("initial exact heap comparator panic")
        );
        assert_eq!(external.get(), 0);
        assert_eq!(retained.get(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn comparator_min_heap_panic_keeps_every_entry_destructible() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut heap = ComparatorMinHeap::try_with_exact_capacity(2).unwrap();
        heap.push_by(
            HeapDropProbe {
                key: 2,
                drops: Arc::clone(&drops),
            },
            &|left, right| left.key.cmp(&right.key),
        );

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            heap.push_by(
                HeapDropProbe {
                    key: 1,
                    drops: Arc::clone(&drops),
                },
                &|_, _| std::panic::panic_any(String::from("heap comparator panic")),
            );
        }))
        .unwrap_err();

        assert_eq!(
            panic.downcast_ref::<String>().map(String::as_str),
            Some("heap comparator panic")
        );
        assert_eq!(heap.len(), 2);
        let retry_comparisons = AtomicUsize::new(0);
        let retry = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = heap.pop_by(&|left, right| {
                retry_comparisons.fetch_add(1, AtomicOrdering::Relaxed);
                left.key.cmp(&right.key)
            });
        }))
        .unwrap_err();
        assert_eq!(
            retry.downcast_ref::<&str>(),
            Some(&"comparator heap is poisoned by an earlier comparator panic")
        );
        assert_eq!(retry_comparisons.load(AtomicOrdering::Relaxed), 0);
        drop(heap);
        assert_eq!(drops.load(AtomicOrdering::Relaxed), 2);
    }

    #[test]
    fn comparator_min_heap_pop_panic_poison_rejects_retry_and_drops_every_entry() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut heap = ComparatorMinHeap::try_with_exact_capacity(4).unwrap();
        for key in [1, 2, 3, 4] {
            heap.push_by(
                HeapDropProbe {
                    key,
                    drops: Arc::clone(&drops),
                },
                &|left, right| left.key.cmp(&right.key),
            );
        }

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = heap
                .pop_by(&|_, _| std::panic::panic_any(String::from("heap pop comparator panic")));
        }))
        .unwrap_err();

        assert_eq!(
            panic.downcast_ref::<String>().map(String::as_str),
            Some("heap pop comparator panic")
        );
        assert_eq!(heap.len(), 4);
        assert_eq!(drops.load(AtomicOrdering::Relaxed), 0);
        let retry_comparisons = AtomicUsize::new(0);
        let retry = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            heap.push_by(
                HeapDropProbe {
                    key: 5,
                    drops: Arc::clone(&drops),
                },
                &|left, right| {
                    retry_comparisons.fetch_add(1, AtomicOrdering::Relaxed);
                    left.key.cmp(&right.key)
                },
            );
        }))
        .unwrap_err();
        assert_eq!(
            retry.downcast_ref::<&str>(),
            Some(&"comparator heap is poisoned by an earlier comparator panic")
        );
        assert_eq!(retry_comparisons.load(AtomicOrdering::Relaxed), 0);
        drop(heap);
        assert_eq!(drops.load(AtomicOrdering::Relaxed), 5);
    }

    #[test]
    fn compatibility_final_heap_keeps_single_comparator_owner() {
        let (_directory, manager) = create_manager();
        let observed_max = Arc::new(AtomicUsize::new(0));
        let comparator_weak = Arc::new(Mutex::new(None::<std::sync::Weak<RowCompareFn>>));
        let observed_max_for_compare = Arc::clone(&observed_max);
        let weak_for_compare = Arc::clone(&comparator_weak);
        let mut sort =
            ExternalSort::new_with_comparator(Arc::clone(&manager), 1, move |left, right| {
                let strong_count = weak_for_compare
                    .lock()
                    .unwrap()
                    .as_ref()
                    .expect("test installs the comparator weak reference before merging")
                    .strong_count();
                observed_max_for_compare.fetch_max(strong_count, AtomicOrdering::Relaxed);
                compare_values_total(&left[0], &right[0])
            });
        *comparator_weak.lock().unwrap() = Some(Arc::downgrade(match &sort.comparator.compare {
            Comparison::Infallible(compare) => compare,
            _ => panic!("test native comparator"),
        }));
        for value in [4, 1, 3, 2] {
            sort.spill_sorted_run(vec![row(&[value])]).unwrap();
        }

        let output = sort.merge_all(Vec::new()).unwrap();

        assert_eq!(output, [row(&[1]), row(&[2]), row(&[3]), row(&[4])]);
        assert_eq!(observed_max.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    fn final_heaps_respect_final_null_placement_direction_and_equal_ordinal_order() {
        let inputs = [
            vec![Value::Int64(1), Value::Int64(10)],
            vec![Value::Null, Value::Int64(20)],
            vec![Value::Int64(2), Value::Int64(30)],
            vec![Value::Int64(1), Value::Int64(40)],
        ];
        let null_cases = [
            (
                SortKey {
                    column: 0,
                    direction: SortDirection::Ascending,
                    null_order: NullOrder::First,
                },
                vec![
                    inputs[1].clone(),
                    inputs[0].clone(),
                    inputs[3].clone(),
                    inputs[2].clone(),
                ],
            ),
            (
                SortKey {
                    column: 0,
                    direction: SortDirection::Ascending,
                    null_order: NullOrder::Last,
                },
                vec![
                    inputs[0].clone(),
                    inputs[3].clone(),
                    inputs[2].clone(),
                    inputs[1].clone(),
                ],
            ),
            (
                SortKey {
                    column: 0,
                    direction: SortDirection::Descending,
                    null_order: NullOrder::First,
                },
                vec![
                    inputs[1].clone(),
                    inputs[2].clone(),
                    inputs[0].clone(),
                    inputs[3].clone(),
                ],
            ),
            (
                SortKey {
                    column: 0,
                    direction: SortDirection::Descending,
                    null_order: NullOrder::Last,
                },
                vec![
                    inputs[2].clone(),
                    inputs[0].clone(),
                    inputs[3].clone(),
                    inputs[1].clone(),
                ],
            ),
        ];

        for (key, expected) in null_cases {
            let (_legacy_directory, legacy_manager) = create_manager();
            let mut legacy = ExternalSort::new(Arc::clone(&legacy_manager), 2, vec![key.clone()]);
            for input in &inputs {
                legacy.spill_sorted_run(vec![input.clone()]).unwrap();
            }
            let legacy_output = legacy.merge_all(Vec::new()).unwrap();
            assert_eq!(legacy_output, expected);
            assert_eq!(legacy_manager.active_file_count(), 0);

            let (_accounted_directory, accounted_manager) = create_manager();
            let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
            let resources =
                crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
            let grant = resources.try_allocate(0).unwrap();
            let mut accounted = ExternalSort::new_accounted(accounted_manager, 2, vec![key], grant);
            for input in &inputs {
                accounted
                    .spill_sorted_run_accounted(std::slice::from_ref(input))
                    .unwrap();
            }
            let mut cursor = accounted
                .merge_cursor_accounted_observing(Vec::new(), 1, ExternalSortGrantObserver::inert())
                .unwrap();
            let mut actual = Vec::new();
            while let Some(chunk) = cursor.next_chunk_accounted_observing().unwrap() {
                actual.extend_from_slice(chunk.rows());
            }
            assert_eq!(actual, expected);
            drop(cursor);
            drop(accounted);
            assert_eq!(buffer_manager.allocated(), 0);
        }

        let direction_cases = [
            (
                SortKey::ascending(0),
                vec![inputs[0].clone(), inputs[3].clone(), inputs[2].clone()],
            ),
            (
                SortKey::descending(0),
                vec![inputs[2].clone(), inputs[0].clone(), inputs[3].clone()],
            ),
        ];
        for (key, expected) in direction_cases {
            let (_directory, manager) = create_manager();
            let mut sort = ExternalSort::new(manager, 2, vec![key]);
            for input in [inputs[0].clone(), inputs[2].clone(), inputs[3].clone()] {
                sort.spill_sorted_run(vec![input]).unwrap();
            }
            assert_eq!(sort.merge_all(Vec::new()).unwrap(), expected);
        }
    }

    #[test]
    fn accounted_comparator_panic_keeps_grants_owned_until_sorter_drop() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let comparator = SemanticRowComparator::new(|_, _| {
            std::panic::panic_any(String::from("accounted heap comparator panic"))
        });
        let cancellation = crate::execution::QueryExecutionControl::new().token();
        let mut sort = ExternalSort::new_accounted_with_comparator_and_cancellation(
            Arc::clone(&manager),
            1,
            comparator,
            grant,
            cancellation,
        );
        sort.spill_sorted_run_accounted(&[row(&[2])]).unwrap();
        sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = sort.merge_cursor_accounted_observing(
                Vec::new(),
                1,
                ExternalSortGrantObserver::inert(),
            );
        }))
        .unwrap_err();

        assert_eq!(
            panic.downcast_ref::<String>().map(String::as_str),
            Some("accounted heap comparator panic")
        );
        assert_eq!(
            buffer_manager.allocated(),
            sort.checked_total_granted_bytes().unwrap()
        );
        assert_eq!(
            resources.query_stats().allocated_bytes,
            buffer_manager.allocated()
        );
        drop(sort);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_scalar_observer_without_telemetry_rejects_combined_overflow() {
        let retained_bytes = Cell::new(usize::MAX);
        let external_bytes = Cell::new(0);
        let observer = ExternalSortGrantObserver::new(&external_bytes, &retained_bytes, None);

        let error = observer.publish(1).unwrap_err();

        assert!(matches!(
            error,
            MemoryGrantError::ArithmeticOverflow {
                current_bytes: usize::MAX,
                additional_bytes: 1
            }
        ));
        assert_eq!(external_bytes.get(), usize::MAX);
    }

    #[test]
    fn accounted_scalar_observer_overflow_preserves_publication_and_retry_invariants() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);
        let external_bytes = Cell::new(sort.total_granted_bytes());
        let mispaired_retained_bytes = Cell::new(usize::MAX);
        let state = crate::execution::operators::push::spill_state::OperatorSpillState::new(
            "sealed sort observer overflow".to_string(),
        );
        let invalid_observer = ExternalSortGrantObserver::new(
            &external_bytes,
            &mispaired_retained_bytes,
            Some(&state),
        );

        let error = sort
            .spill_sorted_run_accounted_observing(&[row(&[1])], &invalid_observer)
            .unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Memory(MemoryGrantError::ArithmeticOverflow {
                current_bytes: usize::MAX,
                additional_bytes,
            }) if additional_bytes > 0
        ));
        assert_eq!(external_bytes.get(), usize::MAX);
        assert_eq!(state.usage(), usize::MAX);
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(
            sort.workspace_granted_bytes(),
            sort.workspace.observed_bytes().unwrap()
        );
        assert_eq!(
            sort.run_catalog_granted_bytes(),
            sort.run_catalog_observed_bytes().unwrap()
        );
        assert_eq!(
            sort.checked_total_granted_bytes().unwrap(),
            buffer_manager.allocated()
        );
        assert_eq!(
            resources.query_stats().allocated_bytes,
            buffer_manager.allocated()
        );

        let retained_bytes = Cell::new(0);
        let retry_external_bytes = Cell::new(sort.total_granted_bytes());
        let valid_observer =
            ExternalSortGrantObserver::new(&retry_external_bytes, &retained_bytes, Some(&state));
        sort.spill_sorted_run_accounted_observing(&[row(&[1])], &valid_observer)
            .unwrap();

        assert_eq!(sort.num_runs(), 1);
        assert_eq!(manager.active_file_count(), 1);
        assert_eq!(retry_external_bytes.get(), sort.total_granted_bytes());
        assert_eq!(state.usage(), sort.total_granted_bytes());
        assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());

        drop(sort);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_scalar_observer_reconciles_before_resuming_operation_panic() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(PanicOnceIo::new(
                    super::super::SpillIoOperation::Create,
                )))
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let retained_grant = resources.try_allocate(41).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(manager, 1, vec![], grant);
        let retained_bytes = Cell::new(retained_grant.size());
        let external_bytes = Cell::new(sort.total_granted_bytes());
        let state = crate::execution::operators::push::spill_state::OperatorSpillState::new(
            "sealed sort observer unwind".to_string(),
        );
        let observer =
            ExternalSortGrantObserver::new(&external_bytes, &retained_bytes, Some(&state));

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sort.spill_sorted_run_accounted_observing(&[row(&[1])], &observer)
        }))
        .unwrap_err();

        assert_eq!(
            panic.downcast_ref::<String>().map(String::as_str),
            Some("deterministic Create I/O callback panic")
        );
        let expected = retained_grant
            .size()
            .checked_add(sort.total_granted_bytes())
            .unwrap();
        assert_eq!(external_bytes.get(), sort.total_granted_bytes());
        assert_eq!(state.usage(), expected);
        assert_eq!(buffer_manager.allocated(), expected);
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(sort.num_runs(), 0);
    }

    #[test]
    fn accounted_run_catalog_growth_denial_preserves_published_entries() {
        let (calibrated_budget, catalog_capacity) = {
            let (_directory, manager) = create_manager();
            let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
            let resources = crate::execution::QueryResourceContext::new(buffer_manager).unwrap();
            let grant = resources.try_allocate(0).unwrap();
            let mut calibration = ExternalSort::new_accounted(manager, 1, vec![], grant);
            calibration
                .spill_sorted_run_accounted(&[row(&[1])])
                .unwrap();
            let capacity = calibration.run_catalog_capacity();
            for value in 2..=capacity {
                let value = i64::try_from(value).unwrap();
                calibration
                    .spill_sorted_run_accounted(&[row(&[value])])
                    .unwrap();
            }
            (calibration.total_granted_bytes(), capacity)
        };
        let writer_bytes = SpillWriterBuffer::prepare().unwrap().capacity();
        let directory = TempDir::new().unwrap();
        let io = Arc::new(CountingCreateIo::default());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager =
            buffer_manager_with_exact_budget(calibrated_budget.checked_add(writer_bytes).unwrap());
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(manager, 1, vec![SortKey::ascending(0)], grant);
        for value in 1..=catalog_capacity {
            let value = i64::try_from(value).unwrap();
            sort.spill_sorted_run_accounted(&[row(&[value])]).unwrap();
        }
        let pointer = sort.run_catalog_pointer();
        // Successful runs need one transient writer allowance. Occupy that
        // allowance only after filling the existing catalog so the next
        // persistent catalog growth, which precedes writer preparation, is
        // denied at its own boundary.
        let blocker = resources.try_allocate(writer_bytes).unwrap();
        let denied_value = i64::try_from(catalog_capacity.checked_add(1).unwrap()).unwrap();

        let error = sort
            .spill_sorted_run_accounted(&[row(&[denied_value])])
            .unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Memory(MemoryGrantError::LimitExceeded { .. })
        ));
        assert_eq!(sort.num_runs(), catalog_capacity);
        assert_eq!(sort.total_rows(), catalog_capacity);
        assert_eq!(sort.run_catalog_pointer(), pointer);
        assert_eq!(sort.total_granted_bytes(), calibrated_budget);
        assert_eq!(
            buffer_manager.allocated(),
            calibrated_budget.checked_add(writer_bytes).unwrap()
        );
        assert_eq!(io.creates.load(AtomicOrdering::Relaxed), catalog_capacity);
        let expected = (1..=catalog_capacity)
            .map(|value| row(&[i64::try_from(value).unwrap()]))
            .collect::<Vec<_>>();
        drop(blocker);
        assert_eq!(sort.merge_all_accounted(Vec::new()).unwrap(), expected);
        assert_eq!(buffer_manager.allocated(), sort.run_catalog_granted_bytes());
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_constructor_rejects_a_nonzero_workspace_grant() {
        let (_temp_dir, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(1).unwrap();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _sort = ExternalSort::new_accounted(manager, 1, vec![], grant);
        }));

        assert!(panic.is_err());
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn partial_accounted_workspace_preparation_remains_exactly_charged() {
        let (_temp_dir, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(manager, 1, vec![], grant);

        let error = sort.workspace.prepare(32, usize::MAX, 0).unwrap_err();

        assert!(matches!(error, ExternalSortOperationError::Allocation(_)));
        assert!(sort.workspace.row_staging.capacity() >= 32);
        assert_eq!(
            sort.workspace_granted_bytes(),
            sort.workspace.observed_bytes().unwrap()
        );
        assert_eq!(buffer_manager.allocated(), sort.workspace_granted_bytes());
        assert_eq!(sort.num_runs(), 0);

        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_workspace_drop_during_unwind_releases_query_and_global_bytes() {
        let (_temp_dir, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();

        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);
            sort.spill_sorted_run_accounted(&[vec![Value::String("retained".into())]])
                .unwrap();
            assert!(buffer_manager.allocated() > 0);
            panic!("drop accounted external sort during unwind");
        }));

        assert!(unwind.is_err());
        assert_eq!(buffer_manager.allocated(), 0);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    fn accounted_codec_workspace_denial_precedes_file_creation() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(CountingCreateIo::default());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(manager, 1, vec![], grant);
        let rows = vec![vec![Value::String("requires workspace".into())]];

        let error = sort.spill_sorted_run_accounted(&rows).unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Memory(MemoryGrantError::LimitExceeded { .. })
        ));
        assert_eq!(rows.len(), 1, "the internal path must borrow retry state");
        assert_eq!(sort.workspace_granted_bytes(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(sort.manager.active_file_count(), 0);
        assert_eq!(io.creates.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);

        let public_error = sort.spill_sorted_run(rows).unwrap_err();
        assert_eq!(public_error.kind(), std::io::ErrorKind::OutOfMemory);
        assert!(
            public_error
                .get_ref()
                .and_then(|source| source.downcast_ref::<MemoryGrantError>())
                .is_some(),
            "the compatibility io::Error must retain the structured grant denial"
        );
        assert_eq!(io.creates.load(AtomicOrdering::Relaxed), 0);
    }

    #[test]
    fn accounted_growth_denial_preserves_the_previous_retry_workspace() {
        let (_temp_dir, manager) = create_manager();
        let catalog_budget = INITIAL_RUN_CATALOG_CAPACITY
            .checked_mul(std::mem::size_of::<ExternalSortRunEntry>())
            .unwrap();
        let writer_bytes = SpillWriterBuffer::prepare().unwrap().capacity();
        let buffer_manager = buffer_manager_with_exact_budget(catalog_budget + 128 + writer_bytes);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);
        let small = vec![Value::Null];

        sort.spill_sorted_run_accounted(std::slice::from_ref(&small))
            .unwrap();
        let granted = sort.workspace_granted_bytes();
        let capacity = sort.workspace.row_staging.capacity();
        let pointer = sort.workspace.row_staging.as_slice().as_ptr();
        // Preserve the transient writer allowance for publication, then occupy
        // it while exercising a later persistent codec-workspace denial.
        let blocker = resources.try_allocate(writer_bytes).unwrap();

        let error = sort
            .spill_sorted_run_accounted(&[vec![Value::String("x".repeat(4_096).into())]])
            .unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Memory(MemoryGrantError::LimitExceeded { .. })
        ));
        assert_eq!(sort.workspace_granted_bytes(), granted);
        assert_eq!(sort.workspace.observed_bytes().unwrap(), granted);
        assert_eq!(sort.workspace.row_staging.capacity(), capacity);
        assert_eq!(sort.workspace.row_staging.as_slice().as_ptr(), pointer);
        assert_eq!(
            buffer_manager.allocated(),
            sort.total_granted_bytes()
                .checked_add(writer_bytes)
                .unwrap()
        );
        assert_eq!(sort.num_runs(), 1);
        assert_eq!(manager.active_file_count(), 1);
        drop(blocker);
        assert_eq!(sort.merge_all_accounted(Vec::new()).unwrap(), vec![small]);
        assert_eq!(buffer_manager.allocated(), sort.run_catalog_granted_bytes());
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn external_sort_reuses_plaintext_staging_across_runs() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![]);
        let tiny = vec![Value::String("tiny".into())];
        let large = vec![Value::String("x".repeat(4_096).into())];
        let medium = vec![Value::String("medium row".into())];
        let small = vec![Value::String("small".into())];

        sort.spill_sorted_run(vec![tiny.clone(), large.clone(), medium.clone()])
            .unwrap();
        let first_capacity = sort.workspace.row_staging.capacity();
        let first_pointer = sort.workspace.row_staging.as_slice().as_ptr();
        assert_eq!(sort.workspace.row_staging.write_growths(), 0);

        sort.spill_sorted_run(vec![small.clone()]).unwrap();

        assert!(first_capacity >= 4_096);
        assert_eq!(sort.workspace.row_staging.capacity(), first_capacity);
        assert_eq!(
            sort.workspace.row_staging.as_slice().as_ptr(),
            first_pointer
        );
        assert_eq!(sort.workspace.row_staging.write_growths(), 0);
        assert_eq!(
            sort.merge_all(Vec::new()).unwrap(),
            vec![tiny, large, medium, small]
        );
    }

    #[test]
    fn external_sort_prepares_and_reuses_largest_counter_workspace() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![]);
        let small = vec![Value::GCounter(Arc::new(HashMap::from([(
            "a".to_string(),
            1,
        )])))];
        let largest = vec![Value::GCounter(Arc::new(HashMap::from([
            ("long-replica-a".to_string(), 1),
            ("long-replica-b".to_string(), 2),
            ("long-replica-c".to_string(), 3),
            ("long-replica-d".to_string(), 4),
        ])))];
        let medium = vec![Value::GCounter(Arc::new(HashMap::from([
            ("b".to_string(), 2),
            ("c".to_string(), 3),
        ])))];

        sort.spill_sorted_run(vec![small.clone(), largest.clone(), medium.clone()])
            .unwrap();
        let entry_capacity = sort.workspace.counter_scratch.entry_capacity();
        let key_capacity = sort.workspace.counter_scratch.key_capacity();
        let entry_pointer = sort.workspace.counter_scratch.entry_pointer();
        let key_pointer = sort.workspace.counter_scratch.key_pointer();
        assert!(entry_capacity >= 4);
        assert!(key_capacity >= 56);
        assert_eq!(sort.workspace.counter_scratch.entry_len(), 0);
        assert_eq!(sort.workspace.counter_scratch.key_len(), 0);
        assert_eq!(sort.workspace.counter_scratch.write_growths(), 0);

        sort.spill_sorted_run(vec![small.clone()]).unwrap();

        assert_eq!(
            sort.workspace.counter_scratch.entry_capacity(),
            entry_capacity
        );
        assert_eq!(sort.workspace.counter_scratch.key_capacity(), key_capacity);
        assert_eq!(
            sort.workspace.counter_scratch.entry_pointer(),
            entry_pointer
        );
        assert_eq!(sort.workspace.counter_scratch.key_pointer(), key_pointer);
        assert_eq!(sort.workspace.counter_scratch.entry_len(), 0);
        assert_eq!(sort.workspace.counter_scratch.key_len(), 0);
        assert_eq!(sort.workspace.counter_scratch.write_growths(), 0);
        assert_eq!(
            sort.merge_all(Vec::new()).unwrap(),
            vec![small.clone(), largest, medium, small]
        );
    }

    #[test]
    fn oversized_row_is_rejected_before_staging_growth_and_retry() {
        let directory = TempDir::new().unwrap();
        // A one-NULL row is nine codec bytes plus the eight-byte durable
        // ordinal; the longer string remains above this record ceiling.
        let limits = super::super::SpillFrameLimits::new(24, 24).unwrap();
        let io = Arc::new(CountingCreateIo::default());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(Arc::new(super::super::CleartextSpillRecordProvider), limits)
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let mut sort = ExternalSort::new(Arc::clone(&manager), 1, vec![]);

        let error = sort
            .spill_sorted_run(vec![
                vec![Value::Null],
                vec![Value::String("too large".into())],
            ])
            .unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(sort.workspace.row_staging.len(), 0);
        assert_eq!(sort.workspace.row_staging.capacity(), 0);
        assert_eq!(sort.workspace.row_staging.write_growths(), 0);
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(sort.runs.entries.capacity(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        assert_eq!(io.creates.load(AtomicOrdering::Relaxed), 0);

        sort.spill_sorted_run(vec![vec![Value::Null]]).unwrap();

        assert_eq!(sort.num_runs(), 1);
        assert_eq!(io.creates.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(sort.merge_all(Vec::new()).unwrap(), vec![vec![Value::Null]]);
    }

    #[test]
    fn prepared_record_scope_refuses_growth_past_its_declared_length() {
        let mut staging = SpillRecordBuffer::new(64);
        let observed = staging.prepare_record(16).unwrap();
        let pointer = staging.as_slice().as_ptr();

        {
            let mut record = staging.record_scope(16).unwrap();
            let error = record.write_all(&[0_u8; 17]).unwrap_err();
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        }

        assert!(observed >= 16);
        assert_eq!(staging.capacity(), observed);
        assert_eq!(staging.as_slice().as_ptr(), pointer);
        assert_eq!(staging.len(), 0);
        assert_eq!(staging.write_growths(), 0);
    }

    #[test]
    fn private_finish_hook_is_one_shot_while_public_finish_remains_idempotent() {
        let (_directory, manager) = create_manager();
        let mut file = manager
            .create_file(SpillFileRole::RdfAggregateState)
            .unwrap();
        file.finish_write().unwrap();
        let mut callback_ran = false;

        let result = file.finish_write_before_publish(|| {
            callback_ran = true;
            Ok::<_, std::convert::Infallible>(())
        });

        assert!(matches!(
            result,
            Err(SpillFinishError::Io(ref error))
                if error.kind() == std::io::ErrorKind::InvalidInput
        ));
        assert!(!callback_ran);
        file.finish_write().unwrap();
        file.close_and_delete().unwrap();
    }

    #[test]
    fn impossible_preparation_is_out_of_memory_and_preserves_capacity() {
        let mut staging = SpillRecordBuffer::new(usize::MAX);
        let observed = staging.prepare_record(32).unwrap();
        let pointer = staging.as_slice().as_ptr();

        let error = staging.prepare_record(usize::MAX).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::OutOfMemory);
        assert!(observed >= 32);
        assert_eq!(staging.capacity(), observed);
        assert_eq!(staging.as_slice().as_ptr(), pointer);
        assert_eq!(staging.len(), 0);
    }

    #[test]
    fn accounted_writer_failure_retains_exactly_charged_retry_workspace() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(super::super::framing_tests::FailNthIo::new(
                    super::super::SpillIoOperation::WritePayload,
                    3,
                    std::io::ErrorKind::PermissionDenied,
                )))
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);
        let staged = vec![Value::GCounter(Arc::new(HashMap::from([
            ("replica-b".to_string(), 2),
            ("replica-a".to_string(), 1),
        ])))];

        let error = sort
            .spill_sorted_run_accounted(std::slice::from_ref(&staged))
            .unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::PermissionDenied
        ));
        assert_eq!(sort.workspace.row_staging.len(), 0);
        assert_eq!(sort.workspace.counter_scratch.entry_len(), 0);
        assert_eq!(sort.workspace.counter_scratch.key_len(), 0);
        assert_eq!(
            sort.workspace_granted_bytes(),
            sort.workspace.observed_bytes().unwrap()
        );
        let retained_catalog_bytes = sort.run_catalog_granted_bytes();
        let retained_catalog_pointer = sort.run_catalog_pointer();
        assert!(retained_catalog_bytes > 0);
        assert_eq!(
            retained_catalog_bytes,
            sort.run_catalog_observed_bytes().unwrap()
        );
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);

        sort.spill_sorted_run_accounted(std::slice::from_ref(&staged))
            .unwrap();
        assert_eq!(sort.run_catalog_pointer(), retained_catalog_pointer);
        assert_eq!(sort.run_catalog_granted_bytes(), retained_catalog_bytes);
        assert_eq!(sort.merge_all_accounted(Vec::new()).unwrap(), vec![staged]);
        assert_eq!(buffer_manager.allocated(), sort.run_catalog_granted_bytes());
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_create_flush_and_sync_failures_release_writer_for_retry() {
        for semantic in [false, true] {
            for operation in [
                super::super::SpillIoOperation::Create,
                super::super::SpillIoOperation::Flush,
                super::super::SpillIoOperation::Sync,
            ] {
                let directory = TempDir::new().unwrap();
                let manager = Arc::new(
                    crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                        .provider(
                            Arc::new(super::super::CleartextSpillRecordProvider),
                            super::super::SpillFrameLimits::format_max(),
                        )
                        .io(Arc::new(super::super::framing_tests::FailNthIo::new(
                            operation,
                            1,
                            std::io::ErrorKind::PermissionDenied,
                        )))
                        .build()
                        .unwrap(),
                );
                let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
                let resources =
                    crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager))
                        .unwrap();
                let grant = resources.try_allocate(0).unwrap();
                let mut sort = if semantic {
                    semantic_fault_sort(Arc::clone(&manager), 1, vec![], grant, &resources)
                } else {
                    ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant)
                };

                let error = sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap_err();

                assert!(matches!(
                    error,
                    ExternalSortOperationError::Io(ref error)
                        if error.kind() == std::io::ErrorKind::PermissionDenied
                ));
                assert_eq!(sort.writer_workspace_granted_bytes(), 0);
                assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());
                assert_eq!(sort.num_runs(), 0);
                assert_eq!(manager.active_file_count(), 0);
                assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);

                sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
                assert_eq!(sort.writer_workspace_granted_bytes(), 0);
                assert_eq!(sort.num_runs(), 1);
                drop(sort);
                assert_eq!(resources.query_stats().allocated_bytes, 0);
                assert_eq!(buffer_manager.allocated(), 0);
            }
        }
    }

    #[test]
    fn accounted_create_flush_and_sync_panics_release_writer_for_retry() {
        for operation in [
            super::super::SpillIoOperation::Create,
            super::super::SpillIoOperation::Flush,
            super::super::SpillIoOperation::Sync,
        ] {
            let directory = TempDir::new().unwrap();
            let manager = Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .provider(
                        Arc::new(super::super::CleartextSpillRecordProvider),
                        super::super::SpillFrameLimits::format_max(),
                    )
                    .io(Arc::new(PanicOnceIo::new(operation)))
                    .build()
                    .unwrap(),
            );
            let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
            let resources =
                crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
            let grant = resources.try_allocate(0).unwrap();
            let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);

            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                sort.spill_sorted_run_accounted(&[row(&[1])])
            }));

            assert!(panic.is_err(), "{operation:?} failpoint did not panic");
            assert_eq!(sort.writer_workspace_granted_bytes(), 0);
            assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());
            assert_eq!(
                resources.query_stats().allocated_bytes,
                sort.total_granted_bytes()
            );
            assert_eq!(sort.num_runs(), 0);
            assert_eq!(manager.active_file_count(), 0);
            assert_eq!(manager.spilled_bytes(), 0);
            assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);

            sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
            assert_eq!(sort.writer_workspace_granted_bytes(), 0);
            assert_eq!(sort.num_runs(), 1);
            drop(sort);
            assert_eq!(resources.query_stats().allocated_bytes, 0);
            assert_eq!(buffer_manager.allocated(), 0);
        }
    }

    #[test]
    fn accounted_provider_begin_panic_releases_writer_for_retry() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    qualified_test_provider(
                        super::super::framing_tests::panic_once_begin_file_provider(),
                    ),
                    super::super::SpillFrameLimits::format_max(),
                )
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sort.spill_sorted_run_accounted(&[row(&[1])])
        }));

        assert!(panic.is_err());
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());
        assert_eq!(
            resources.query_stats().allocated_bytes,
            sort.total_granted_bytes()
        );
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);

        sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(sort.num_runs(), 1);
        drop(sort);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_writer_unwind_drops_provider_state_before_releasing_workspace() {
        let directory = TempDir::new().unwrap();
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let (provider, witness) =
            super::super::framing_tests::grant_lifetime_provider(resources.clone(), 4096);
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(provider, super::super::SpillFrameLimits::format_max())
                .build()
                .unwrap(),
        );
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);
        let stable_bytes = sort.total_granted_bytes();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut publication = WriterPublication::new(&mut sort.writer_workspace);
            publication.prepare(&manager).unwrap();
            let observer = ExternalSortGrantObserver::inert();
            publication
                .create(&manager, SpillFileRole::SortRun, stable_bytes, &observer)
                .unwrap();
            panic!("deterministic writer publication unwind");
        }))
        .unwrap_err();

        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"deterministic writer publication unwind")
        );
        assert_eq!(witness.drops(), 1);
        assert!(
            !witness.released_before_drop(),
            "provider state outlived its admitted file-workspace grant"
        );
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(manager.active_file_count(), 0);
        drop(sort);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn hostile_provider_destructor_cannot_replace_writer_unwind_or_strand_grant() {
        const CHILD_ENV: &str = "GRAFEO_WRITER_PROVIDER_DROP_CHILD";
        const HANDSHAKE: &str = "GRAFEO_WRITER_PROVIDER_DROP_OK";
        if std::env::var_os(CHILD_ENV).is_some() {
            let directory = TempDir::new().unwrap();
            let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
            let resources =
                crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
            let manager = Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .provider(
                        Arc::new(PanicOnProviderDrop),
                        super::super::SpillFrameLimits::format_max(),
                    )
                    .build()
                    .unwrap(),
            );
            let grant = resources.try_allocate(0).unwrap();
            let mut writer_workspace = ExternalSortWriterWorkspace::new(Some(grant));

            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut publication = WriterPublication::new(&mut writer_workspace);
                publication.prepare(&manager).unwrap();
                let observer = ExternalSortGrantObserver::inert();
                publication
                    .create(&manager, SpillFileRole::SortRun, 0, &observer)
                    .unwrap();
                drop(manager);
                panic!("deterministic writer primary panic");
            }))
            .unwrap_err();

            assert_eq!(
                panic.downcast_ref::<&str>(),
                Some(&"deterministic writer primary panic")
            );
            assert_eq!(writer_workspace.granted_bytes(), 0);
            assert_eq!(resources.query_stats().allocated_bytes, 0);
            println!("{HANDSHAKE}");
            return;
        }

        let test_name = "execution::spill::external_sort::tests::hostile_provider_destructor_cannot_replace_writer_unwind_or_strand_grant";
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg(test_name)
            .arg("--exact")
            .arg("--nocapture")
            .env(CHILD_ENV, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains(HANDSHAKE),
            "provider-drop child did not preserve the writer primary and release its grant\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }

    #[test]
    fn accounted_provider_begin_error_releases_writer_and_removes_staging() {
        let directory = TempDir::new().unwrap();
        let manager =
            Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .provider(
                        qualified_test_provider(
                            super::super::framing_tests::fail_begin_file_provider(),
                        ),
                        super::super::SpillFrameLimits::format_max(),
                    )
                    .build()
                    .unwrap(),
            );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);

        let error = sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::PermissionDenied
        ));
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());
        assert_eq!(
            resources.query_stats().allocated_bytes,
            sort.total_granted_bytes()
        );
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);

        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn provider_primary_survives_panicking_construction_cleanup_and_drop_retry() {
        let directory = TempDir::new().unwrap();
        let before = SpillManager::orphan_cleanup_failures();
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .provider(
                super::super::framing_tests::fail_begin_file_provider(),
                super::super::SpillFrameLimits::format_max(),
            )
            .io(Arc::new(PanicOnceIo::new(
                super::super::SpillIoOperation::Delete,
            )))
            .build()
            .unwrap();

        let error = manager.create_file(SpillFileRole::SortRun).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(error.to_string().contains("begin-file failpoint"));
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        assert!(SpillManager::orphan_cleanup_failures() > before);
    }

    #[test]
    fn provider_begin_and_delete_failures_preserve_primary_and_retry_state() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(ToggleDeleteIo::failing());
        let manager =
            Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .provider(
                        qualified_test_provider(
                            super::super::framing_tests::fail_begin_file_provider(),
                        ),
                        super::super::SpillFrameLimits::format_max(),
                    )
                    .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                    .build()
                    .unwrap(),
            );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);

        let error = sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap_err();

        let ExternalSortOperationError::Io(error) = error else {
            panic!("provider initialization failure lost its I/O classification");
        };
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        let rendered = error.to_string();
        assert!(rendered.contains("begin-file failpoint"));
        assert!(rendered.contains("spill construction cleanup"));
        assert!(rendered.contains("deterministic writer cleanup denial"));
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 1);
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);

        io.permit_delete();
        manager.cleanup().unwrap();
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        drop(sort);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn writer_release_failure_retains_token_and_refuses_false_success() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);
        sort.writer_workspace.release_error = Some(MemoryGrantError::AccountingPoisoned {
            account: "deterministic writer release",
        });

        let error = sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::Memory(MemoryGrantError::AccountingPoisoned {
                account: "deterministic writer release"
            })
        ));
        let retained_writer = sort.writer_workspace_granted_bytes();
        assert!(retained_writer >= qualified_writer_base_bytes(&manager));
        assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());
        assert_eq!(
            resources.query_stats().allocated_bytes,
            sort.total_granted_bytes()
        );
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);

        sort.writer_workspace.release_error = None;
        sort.cleanup().unwrap();
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(buffer_manager.allocated(), sort.run_catalog_granted_bytes());

        sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap();
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(sort.num_runs(), 1);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn writer_release_and_delete_failures_remain_independently_retryable() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(ToggleDeleteIo::failing());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);
        sort.writer_workspace.release_error = Some(MemoryGrantError::AccountingPoisoned {
            account: "deterministic writer release",
        });

        let error = sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::MemoryWithCleanup {
                error: MemoryGrantError::AccountingPoisoned {
                    account: "deterministic writer release"
                },
                ref cleanup,
                phase: "sort spill cleanup",
            } if cleanup.kind() == std::io::ErrorKind::PermissionDenied
        ));
        assert!(sort.writer_workspace_granted_bytes() > 0);
        assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 1);
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);

        sort.writer_workspace.release_error = None;
        sort.cleanup().unwrap();
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(manager.active_file_count(), 1);

        io.permit_delete();
        manager.cleanup().unwrap();
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        drop(sort);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn io_primary_survives_writer_release_failure() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(super::super::framing_tests::FailNthIo::new(
                    super::super::SpillIoOperation::WritePayload,
                    3,
                    std::io::ErrorKind::PermissionDenied,
                )))
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);
        sort.writer_workspace.release_error = Some(MemoryGrantError::AccountingPoisoned {
            account: "deterministic writer release",
        });

        let error = sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap_err();

        assert!(matches!(
            error,
            ExternalSortOperationError::WithGrantRelease {
                primary: ExternalSortPrimary::Io(ref primary),
                release: MemoryGrantError::AccountingPoisoned {
                    account: "deterministic writer release"
                },
                cleanup: None,
                ..
            } if primary.kind() == std::io::ErrorKind::PermissionDenied
        ));
        assert!(sort.writer_workspace_granted_bytes() > 0);
        assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);

        sort.writer_workspace.release_error = None;
        sort.cleanup().unwrap();
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn io_release_and_delete_failures_remain_typed_and_independently_retryable() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(FailWriteAndToggleDeleteIo::failing());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);
        sort.writer_workspace.release_error = Some(MemoryGrantError::AccountingPoisoned {
            account: "deterministic writer release",
        });

        let error = sort.spill_sorted_run_accounted(&[row(&[1])]).unwrap_err();

        let ExternalSortOperationError::WithGrantRelease {
            primary: ExternalSortPrimary::Io(primary),
            release,
            cleanup: Some(cleanup),
            phase,
        } = error
        else {
            panic!("three independent failures were flattened")
        };
        assert_eq!(primary.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(primary.to_string().contains("writer primary failure"));
        assert!(matches!(
            release,
            MemoryGrantError::AccountingPoisoned {
                account: "deterministic writer release"
            }
        ));
        assert_eq!(cleanup.kind(), std::io::ErrorKind::WouldBlock);
        assert!(cleanup.to_string().contains("writer deletion failure"));
        assert_eq!(phase, "writer failure release");
        assert!(sort.writer_workspace_granted_bytes() > 0);
        assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());
        assert_eq!(
            resources.query_stats().allocated_bytes,
            sort.total_granted_bytes()
        );
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 1);
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);

        sort.writer_workspace.release_error = None;
        sort.cleanup().unwrap();
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(manager.active_file_count(), 1);

        io.permit_delete();
        manager.cleanup().unwrap();
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        drop(sort);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_provider_panic_retains_exactly_charged_retry_workspace() {
        let directory = TempDir::new().unwrap();
        let manager =
            Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .provider(
                        qualified_test_provider(
                            super::super::framing_tests::panic_once_seal_provider(),
                        ),
                        super::super::SpillFrameLimits::format_max(),
                    )
                    .build()
                    .unwrap(),
            );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(Arc::clone(&manager), 1, vec![], grant);
        let staged = vec![Value::GCounter(Arc::new(HashMap::from([
            ("replica-b".to_string(), 2),
            ("replica-a".to_string(), 1),
        ])))];

        let observation = TestGrantObservation::new(sort.total_granted_bytes());
        let observer = observation.observer();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sort.spill_sorted_run_accounted_observing(std::slice::from_ref(&staged), &observer)
        }))
        .unwrap_err();

        assert_eq!(panic.downcast_ref::<&str>(), Some(&"seal callback panic"));
        assert_eq!(sort.workspace.row_staging.len(), 0);
        assert_eq!(sort.workspace.counter_scratch.entry_len(), 0);
        assert_eq!(sort.workspace.counter_scratch.key_len(), 0);
        assert_eq!(
            sort.workspace_granted_bytes(),
            sort.workspace.observed_bytes().unwrap()
        );
        assert_eq!(sort.writer_workspace_granted_bytes(), 0);
        assert_eq!(buffer_manager.allocated(), sort.total_granted_bytes());
        assert_eq!(
            observation.current(),
            sort.total_granted_bytes(),
            "telemetry must be current before provider code can unwind"
        );
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);

        sort.spill_sorted_run_accounted(std::slice::from_ref(&staged))
            .unwrap();
        assert_eq!(sort.merge_all_accounted(Vec::new()).unwrap(), vec![staged]);
        assert_eq!(buffer_manager.allocated(), sort.run_catalog_granted_bytes());
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_observer_reconciles_a_panic_between_workspace_subgrants() {
        let staged = vec![Value::GCounter(Arc::new(HashMap::from([
            ("replica-b".to_string(), 2),
            ("replica-a".to_string(), 1),
        ])))];
        let measurement = measure_serialized_row_with_limits(
            &staged,
            super::super::SpillFrameLimits::format_max().codec_limits(),
        )
        .unwrap();
        let (row_workspace_bytes, full_workspace_bytes) = {
            let calibration_manager = buffer_manager_with_exact_budget(1024 * 1024);
            let calibration_resources =
                crate::execution::QueryResourceContext::new(Arc::clone(&calibration_manager))
                    .unwrap();
            let grant = calibration_resources.try_allocate(0).unwrap();
            let mut workspace = ExternalSortWorkspace::new(usize::MAX, Some(grant));
            workspace
                .prepare_row_staging(
                    measurement
                        .encoded_bytes
                        .checked_add(std::mem::size_of::<u64>())
                        .unwrap(),
                )
                .unwrap();
            let row_bytes = workspace.granted_bytes();
            workspace
                .prepare_counter_scratch(
                    measurement.counter_sort_entries,
                    measurement.counter_sort_key_bytes,
                )
                .unwrap();
            (row_bytes, workspace.granted_bytes())
        };
        assert!(row_workspace_bytes > 0);
        assert!(full_workspace_bytes > row_workspace_bytes);

        let budget = full_workspace_bytes.checked_add(1).unwrap();
        let blocker_bytes = budget.checked_sub(row_workspace_bytes).unwrap();
        let mut config = BufferManagerConfig::with_budget(budget);
        config.soft_limit_fraction = 0.5;
        config.evict_limit_fraction = 0.5;
        config.hard_limit_fraction = 1.0;
        let buffer_manager = BufferManager::new(config);
        let blocker = buffer_manager
            .try_allocate(blocker_bytes, MemoryRegion::ExecutionBuffers)
            .unwrap();
        buffer_manager.register_consumer(Arc::new(PanickingEvictionConsumer));
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let (_directory, spill_manager) = create_manager();
        let mut sort = ExternalSort::new_accounted(spill_manager, 1, vec![], grant);
        let observation = TestGrantObservation::new(sort.total_granted_bytes());
        let observer = observation.observer();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sort.spill_sorted_run_accounted_observing(std::slice::from_ref(&staged), &observer)
        }));

        assert!(panic.is_err());
        assert_eq!(sort.workspace_granted_bytes(), row_workspace_bytes);
        assert_eq!(sort.run_catalog_granted_bytes(), 0);
        assert_eq!(sort.total_granted_bytes(), row_workspace_bytes);
        assert_eq!(
            observation.current(),
            row_workspace_bytes,
            "caught unwind must publish the live successfully-grown subgrant"
        );
        assert_eq!(resources.query_stats().allocated_bytes, row_workspace_bytes);
        assert_eq!(buffer_manager.allocated(), budget);
        assert_eq!(sort.num_runs(), 0);

        drop(sort);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(buffer_manager.allocated(), blocker_bytes);
        drop(blocker);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn hostile_initialization_primary_survives_panicking_cleanup() {
        const CHILD_ENV: &str = "GRAFEO_SPILL_HOSTILE_CONSTRUCTION_CHILD";
        const HANDSHAKE: &str = "GRAFEO_HOSTILE_CONSTRUCTION_PRIMARY_OK";
        if std::env::var_os(CHILD_ENV).is_some() {
            let directory = TempDir::new().unwrap();
            let before = SpillManager::orphan_cleanup_failures();
            let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(HostileBeginErrorProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(PanicOnceIo::new(
                    super::super::SpillIoOperation::Delete,
                )))
                .build()
                .unwrap();

            let error = manager.create_file(SpillFileRole::SortRun).unwrap_err();

            assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
            std::mem::forget(error);
            assert_eq!(manager.active_file_count(), 0);
            assert_eq!(manager.spilled_bytes(), 0);
            assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
            assert!(SpillManager::orphan_cleanup_failures() > before);
            println!("{HANDSHAKE}");
            return;
        }

        let test_name = "execution::spill::external_sort::tests::hostile_initialization_primary_survives_panicking_cleanup";
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg(test_name)
            .arg("--exact")
            .arg("--nocapture")
            .env(CHILD_ENV, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains(HANDSHAKE),
            "construction-cleanup child did not preserve the primary\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn hostile_nested_error_destructors_cannot_abort_external_sort_drop() {
        const CHILD_ENV: &str = "GRAFEO_EXTERNAL_SORT_HOSTILE_DROP_CHILD";
        const HANDSHAKE: &str = "GRAFEO_HOSTILE_EXTERNAL_SORT_DROP_OK";
        if std::env::var_os(CHILD_ENV).is_some() {
            let directory = TempDir::new().unwrap();
            let manager = Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .provider(
                        Arc::new(super::super::CleartextSpillRecordProvider),
                        super::super::SpillFrameLimits::format_max(),
                    )
                    .io(Arc::new(HostileDoubleErrorDeleteIo::new()))
                    .build()
                    .unwrap(),
            );
            let mut sort = ExternalSort::new(Arc::clone(&manager), 1, vec![]);
            sort.spill_sorted_run(vec![row(&[1])]).unwrap();
            let path = sort.runs.get(0).file.path().to_path_buf();
            let before = SpillManager::orphan_cleanup_failures();
            let primary = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _sort = sort;
                std::panic::panic_any(PrimaryDropPanic);
            }))
            .unwrap_err();
            assert!(primary.is::<PrimaryDropPanic>());
            assert!(!path.exists());
            assert_eq!(manager.active_file_count(), 0);
            assert!(SpillManager::orphan_cleanup_failures() > before);
            println!("{HANDSHAKE}");
            return;
        }

        let test_name = "execution::spill::external_sort::tests::hostile_nested_error_destructors_cannot_abort_external_sort_drop";
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg(test_name)
            .arg("--exact")
            .arg("--nocapture")
            .env(CHILD_ENV, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains(HANDSHAKE),
            "external-sort Drop child did not complete its exact scenario\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }

    #[test]
    fn failed_row_write_resets_staging_and_publishes_no_run() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(super::super::framing_tests::FailNthIo::new(
                    super::super::SpillIoOperation::WritePayload,
                    3,
                    std::io::ErrorKind::PermissionDenied,
                )))
                .build()
                .unwrap(),
        );
        let mut sort = ExternalSort::new(Arc::clone(&manager), 1, vec![]);
        let staged = vec![Value::GCounter(Arc::new(HashMap::from([
            ("replica-b".to_string(), 2),
            ("replica-a".to_string(), 1),
        ])))];

        let error = sort.spill_sorted_run(vec![staged.clone()]).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(sort.workspace.row_staging.len(), 0);
        assert!(sort.workspace.row_staging.capacity() > 0);
        assert_eq!(sort.workspace.counter_scratch.entry_len(), 0);
        assert_eq!(sort.workspace.counter_scratch.key_len(), 0);
        assert!(sort.workspace.counter_scratch.entry_capacity() >= 2);
        assert!(sort.workspace.counter_scratch.key_capacity() >= 18);
        assert_eq!(sort.workspace.counter_scratch.write_growths(), 0);
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);

        sort.spill_sorted_run(vec![staged.clone()]).unwrap();
        assert_eq!(sort.merge_all(Vec::new()).unwrap(), vec![staged]);
    }

    #[test]
    fn caught_provider_panic_resets_staging_and_preserves_retry() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    super::super::framing_tests::panic_once_seal_provider(),
                    super::super::SpillFrameLimits::format_max(),
                )
                .build()
                .unwrap(),
        );
        let mut sort = ExternalSort::new(Arc::clone(&manager), 1, vec![]);
        let staged = vec![Value::GCounter(Arc::new(HashMap::from([
            ("replica-b".to_string(), 2),
            ("replica-a".to_string(), 1),
        ])))];

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sort.spill_sorted_run(vec![staged.clone()])
        }));

        assert!(panic.is_err());
        assert_eq!(sort.workspace.row_staging.len(), 0);
        assert!(sort.workspace.row_staging.capacity() > 0);
        assert_eq!(sort.workspace.counter_scratch.entry_len(), 0);
        assert_eq!(sort.workspace.counter_scratch.key_len(), 0);
        assert!(sort.workspace.counter_scratch.entry_capacity() >= 2);
        assert!(sort.workspace.counter_scratch.key_capacity() >= 18);
        assert_eq!(sort.workspace.counter_scratch.write_growths(), 0);
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);

        sort.spill_sorted_run(vec![vec![Value::Null]]).unwrap();
        assert_eq!(sort.merge_all(Vec::new()).unwrap(), vec![vec![Value::Null]]);
    }

    #[test]
    fn merge_transition_discards_plaintext_staging_capacity() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![]);
        let expected = vec![Value::List(Arc::from([
            Value::String("x".repeat(4_096).into()),
            Value::GCounter(Arc::new(HashMap::from([("replica-a".to_string(), 1)]))),
        ]))];
        sort.spill_sorted_run(vec![expected.clone()]).unwrap();
        assert!(sort.workspace.row_staging.capacity() >= 4_096);
        assert!(sort.workspace.counter_scratch.entry_capacity() >= 1);
        assert!(sort.workspace.counter_scratch.key_capacity() >= 9);

        let rows = sort.merge_all(Vec::new()).unwrap();

        assert_eq!(rows, vec![expected]);
        assert_eq!(sort.workspace.row_staging.capacity(), 0);
        assert_eq!(sort.workspace.counter_scratch.entry_capacity(), 0);
        assert_eq!(sort.workspace.counter_scratch.key_capacity(), 0);
    }

    #[test]
    fn memory_only_merge_discards_prepared_staging_after_create_failure() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(super::super::framing_tests::FailNthIo::new(
                    super::super::SpillIoOperation::Create,
                    1,
                    std::io::ErrorKind::PermissionDenied,
                )))
                .build()
                .unwrap(),
        );
        let mut sort = ExternalSort::new(manager, 1, vec![]);
        let staged = vec![Value::List(Arc::from([
            Value::String("x".repeat(4_096).into()),
            Value::GCounter(Arc::new(HashMap::from([("replica-a".to_string(), 1)]))),
        ]))];
        let error = sort.spill_sorted_run(vec![staged]).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(sort.workspace.row_staging.capacity() >= 4_096);
        assert_eq!(sort.workspace.row_staging.write_growths(), 0);
        assert!(sort.workspace.counter_scratch.entry_capacity() >= 1);
        assert!(sort.workspace.counter_scratch.key_capacity() >= 9);
        assert_eq!(sort.workspace.counter_scratch.write_growths(), 0);

        let rows = sort.merge_all(vec![vec![Value::Null]]).unwrap();

        assert_eq!(rows, vec![vec![Value::Null]]);
        assert_eq!(sort.workspace.row_staging.capacity(), 0);
        assert_eq!(sort.workspace.counter_scratch.entry_capacity(), 0);
        assert_eq!(sort.workspace.counter_scratch.key_capacity(), 0);
    }

    #[test]
    fn framed_zero_column_sort_row_round_trips_exactly() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 0, vec![]);

        sort.spill_sorted_run(vec![vec![]]).unwrap();

        assert_eq!(
            sort.merge_all(Vec::new()).unwrap(),
            vec![Vec::<Value>::new()]
        );
    }

    #[test]
    fn fallible_merge_size_rejects_overflow_before_allocation_or_reader_open() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 0, vec![]);
        sort.spill_sorted_run(vec![vec![]]).unwrap();
        sort.spill_sorted_run(vec![vec![]]).unwrap();
        sort.runs.entries[0].rows = usize::MAX;
        sort.runs.entries[1].rows = 1;

        assert_eq!(sort.total_rows(), usize::MAX);
        assert_eq!(
            sort.checked_total_rows().unwrap_err().kind(),
            std::io::ErrorKind::OutOfMemory
        );

        sort.runs.entries[1].file.close_and_delete().unwrap();
        sort.runs.remove(1);
        for buffer in [Vec::new(), vec![Vec::new()]] {
            let ExternalSortOperationError::Io(error) = sort.k_way_merge(buffer, None).unwrap_err()
            else {
                panic!("merge overflow lost its I/O classification")
            };
            assert_eq!(error.kind(), std::io::ErrorKind::OutOfMemory);
        }
    }

    #[test]
    fn failed_run_publication_keeps_handles_and_counts_paired() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(super::super::framing_tests::FailNthIo::new(
                    super::super::SpillIoOperation::Sync,
                    2,
                    std::io::ErrorKind::PermissionDenied,
                )))
                .build()
                .unwrap(),
        );
        let mut sort = ExternalSort::new(Arc::clone(&manager), 1, vec![SortKey::ascending(0)]);
        sort.spill_sorted_run(vec![row(&[1])]).unwrap();
        let first_bytes = manager.spilled_bytes();

        let error = sort.spill_sorted_run(vec![row(&[2])]).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(sort.runs.len(), 1);
        assert_eq!(sort.runs.get(0).rows, 1);
        assert_eq!(sort.num_runs(), 1);
        assert_eq!(sort.total_rows(), 1);
        assert_eq!(manager.active_file_count(), 1);
        assert_eq!(manager.spilled_bytes(), first_bytes);
        assert_eq!(sort.merge_all(Vec::new()).unwrap(), vec![row(&[1])]);
    }

    #[test]
    fn spilled_sort_run_starts_with_the_v1_frame_magic() {
        let (temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);
        sort.spill_sorted_run(vec![row(&[1])]).unwrap();

        let path = std::fs::read_dir(temp_dir.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let bytes = std::fs::read(path).unwrap();

        assert_eq!(&bytes[..4], b"GRSP");
    }

    #[test]
    fn framed_sort_row_round_trips_exact_and_nested_task_one_values() {
        let (_directory, manager) = create_manager();
        let mut counter = HashMap::new();
        counter.insert("replica-b".to_owned(), 2);
        counter.insert("replica-a".to_owned(), 1);
        let mut positive = HashMap::new();
        positive.insert("replica-a".to_owned(), 7);
        let mut negative = HashMap::new();
        negative.insert("replica-b".to_owned(), 3);
        let mut map = BTreeMap::new();
        map.insert(
            grafeo_common::types::PropertyKey::new("nested"),
            Value::List(Arc::from([Value::Null, Value::Bool(true)])),
        );
        let expected = vec![
            Value::RdfLiteral {
                lexical: "plain".into(),
                language: None,
                datatype: None,
            },
            Value::RdfLiteral {
                lexical: "colour".into(),
                language: Some("EN".into()),
                datatype: None,
            },
            Value::RdfLiteral {
                lexical: "18446744073709551616".into(),
                language: None,
                datatype: Some("http://www.w3.org/2001/XMLSchema#integer".into()),
            },
            Value::Time(
                grafeo_common::types::Time::from_nanos(12_345)
                    .unwrap()
                    .with_offset(-3_600),
            ),
            Value::List(Arc::from([
                Value::Vector(Arc::from([1.0_f32, -2.5])),
                Value::String("nested".into()),
            ])),
            Value::Path {
                nodes: Arc::from([Value::Int64(1), Value::Int64(2)]),
                edges: Arc::from([Value::String("edge".into())]),
            },
            Value::Map(Arc::new(map)),
            Value::GCounter(Arc::new(counter)),
            Value::OnCounter {
                pos: Arc::new(positive),
                neg: Arc::new(negative),
            },
        ];
        let mut sort = ExternalSort::new(manager, expected.len(), vec![]);
        sort.spill_sorted_run(vec![expected.clone()]).unwrap();

        let rows = sort.merge_all(Vec::new()).unwrap();

        assert_eq!(rows, vec![expected]);
    }

    #[test]
    fn framed_sort_row_requires_exact_codec_payload_consumption() {
        let mut payload = Vec::new();
        serialize_row_with_limits(
            &[Value::Int64(1)],
            &mut payload,
            super::super::SpillFrameLimits::format_max().codec_limits(),
        )
        .unwrap();
        payload.push(0xff);

        let error = decode_row_payload(&payload, 1, super::super::SpillFrameLimits::format_max())
            .unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn read_and_cleanup_failure_cannot_return_a_successful_merge() {
        let (_directory, manager) = create_manager();
        let mut sort = ExternalSort::new(Arc::clone(&manager), 1, vec![SortKey::ascending(0)]);
        sort.spill_sorted_run(vec![row(&[1])]).unwrap();
        let path = sort.runs.get(0).file.path().to_path_buf();
        let published = manager.spilled_bytes();
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();

        let error = sort.merge_all(Vec::new()).unwrap_err();

        assert!(error.to_string().contains("cleanup also failed"));
        let primary = error
            .get_ref()
            .and_then(std::error::Error::source)
            .and_then(|source| source.downcast_ref::<std::io::Error>())
            .expect("combined spill error must retain the primary I/O error as source");
        assert_eq!(primary.kind(), error.kind());
        assert_eq!(sort.num_runs(), 1);
        assert_eq!(manager.spilled_bytes(), published);

        std::fs::remove_dir(path).unwrap();
        sort.cleanup().unwrap();
    }

    #[test]
    fn partial_cleanup_after_merge_cannot_be_retried_as_partial_output() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(super::super::framing_tests::FailNthIo::new(
                    super::super::SpillIoOperation::Delete,
                    2,
                    std::io::ErrorKind::PermissionDenied,
                )))
                .build()
                .unwrap(),
        );
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);
        sort.spill_sorted_run(vec![row(&[1])]).unwrap();
        sort.spill_sorted_run(vec![row(&[2])]).unwrap();

        let first_error = sort.merge_all(Vec::new()).unwrap_err();
        assert_eq!(first_error.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(sort.num_runs(), 1, "the first run was already deleted");
        assert_eq!(
            sort.merge_all(Vec::new()).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );

        sort.cleanup().unwrap();
    }

    #[test]
    fn partial_explicit_cleanup_makes_sort_terminal_but_remains_retryable() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(super::super::framing_tests::FailNthIo::new(
                    super::super::SpillIoOperation::Delete,
                    2,
                    std::io::ErrorKind::PermissionDenied,
                )))
                .build()
                .unwrap(),
        );
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);
        sort.spill_sorted_run(vec![row(&[1])]).unwrap();
        sort.spill_sorted_run(vec![row(&[2])]).unwrap();

        assert_eq!(
            sort.cleanup().unwrap_err().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        assert_eq!(sort.num_runs(), 1);
        assert_eq!(
            sort.merge_all(Vec::new()).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert_eq!(
            sort.spill_sorted_run(vec![row(&[3])]).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        sort.cleanup().unwrap();
    }

    #[test]
    fn cleanup_runs_retains_first_owned_error_without_formatting_or_dropping_later_errors() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(super::super::framing_tests::HostileCleanupIo::new());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let mut sort = ExternalSort::new(Arc::clone(&manager), 1, vec![SortKey::ascending(0)]);
        for value in 0..3 {
            sort.spill_sorted_run(vec![row(&[value])]).unwrap();
        }

        let cleanup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sort.cleanup()));

        assert!(
            cleanup.is_ok(),
            "explicit run cleanup must not invoke hostile error Display or Drop"
        );
        let error = cleanup.unwrap().unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(
            error
                .get_ref()
                .and_then(|source| {
                    source.downcast_ref::<super::super::framing_tests::HostileCleanupError>()
                })
                .map(super::super::framing_tests::HostileCleanupError::attempt),
            Some(0),
            "cleanup must return the first owned error, not a reconstruction"
        );
        assert_eq!(io.attempts(), 3, "every run receives one cleanup attempt");
        assert_eq!(io.display_calls(), 0);
        assert_eq!(io.secondary_drops(), 0);
        assert_eq!(sort.num_runs(), 2, "only failed run handles remain");
        assert_eq!(manager.active_file_count(), 2);
        assert!(sort.disk_merge_started);

        drop(error);
        assert_eq!(io.primary_drops(), 1);
        io.permit_delete();
        sort.cleanup().unwrap();
        assert_eq!(io.attempts(), 5, "retry visits only retained failures");
        assert_eq!(sort.num_runs(), 0);
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    fn successful_explicit_cleanup_preserves_sorter_reuse() {
        let (_directory, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);
        sort.spill_sorted_run(vec![vec![Value::List(Arc::from([
            Value::String("x".repeat(4_096).into()),
            Value::GCounter(Arc::new(HashMap::from([("replica-a".to_string(), 1)]))),
        ]))]])
        .unwrap();
        assert!(sort.workspace.row_staging.capacity() >= 4_096);
        assert!(sort.workspace.counter_scratch.entry_capacity() >= 1);
        assert!(sort.workspace.counter_scratch.key_capacity() >= 9);

        sort.cleanup().unwrap();

        assert_eq!(sort.num_runs(), 0);
        assert_eq!(sort.total_rows(), 0);
        assert_eq!(sort.workspace.row_staging.capacity(), 0);
        assert_eq!(sort.workspace.counter_scratch.entry_capacity(), 0);
        assert_eq!(sort.workspace.counter_scratch.key_capacity(), 0);
        sort.spill_sorted_run(vec![row(&[2])]).unwrap();
        assert_eq!(sort.merge_all(Vec::new()).unwrap(), vec![row(&[2])]);
    }

    #[test]
    fn failed_first_explicit_cleanup_preserves_complete_sorter_for_retry() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(super::super::CleartextSpillRecordProvider),
                    super::super::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(super::super::framing_tests::FailNthIo::new(
                    super::super::SpillIoOperation::Delete,
                    1,
                    std::io::ErrorKind::PermissionDenied,
                )))
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let grant = resources.try_allocate(0).unwrap();
        let mut sort = ExternalSort::new_accounted(
            Arc::clone(&manager),
            1,
            vec![SortKey::ascending(0)],
            grant,
        );
        sort.spill_sorted_run(vec![row(&[1])]).unwrap();
        let published = manager.spilled_bytes();
        assert!(buffer_manager.allocated() > 0);

        assert_eq!(
            sort.cleanup().unwrap_err().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        assert_eq!(sort.num_runs(), 1);
        assert_eq!(sort.total_rows(), 1);
        assert_eq!(manager.active_file_count(), 1);
        assert_eq!(manager.spilled_bytes(), published);
        assert_eq!(sort.workspace_granted_bytes(), 0);
        assert_eq!(buffer_manager.allocated(), sort.run_catalog_granted_bytes());
        assert_eq!(sort.merge_all(Vec::new()).unwrap(), vec![row(&[1])]);
        assert_eq!(buffer_manager.allocated(), sort.run_catalog_granted_bytes());
        drop(sort);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn successful_disk_merge_is_one_shot_even_with_new_memory_rows() {
        let (_directory, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);
        sort.spill_sorted_run(vec![row(&[1])]).unwrap();
        assert_eq!(sort.merge_all(Vec::new()).unwrap(), vec![row(&[1])]);

        assert_eq!(
            sort.merge_all(vec![row(&[2])]).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
    }

    #[test]
    // reason: test values 1..=6 fit i64
    #[allow(clippy::cast_possible_wrap)]
    fn test_external_sort_two_runs() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);

        // Spill two sorted runs
        sort.spill_sorted_run(vec![row(&[1]), row(&[3]), row(&[5])])
            .unwrap();
        sort.spill_sorted_run(vec![row(&[2]), row(&[4]), row(&[6])])
            .unwrap();

        assert_eq!(sort.num_runs(), 2);

        let result = sort.merge_all(Vec::new()).unwrap();
        assert_eq!(result.len(), 6);
        for (i, r) in result.iter().enumerate() {
            assert_eq!(r, &row(&[(i + 1) as i64]));
        }
    }

    #[test]
    fn equal_keys_preserve_input_segment_order_across_runs_and_memory() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 2, vec![SortKey::ascending(0)]);
        sort.set_merge_fan_in(2);

        sort.spill_sorted_run(vec![row(&[1, 10]), row(&[1, 11])])
            .unwrap();
        sort.spill_sorted_run(vec![row(&[1, 20]), row(&[1, 21])])
            .unwrap();

        assert_eq!(
            sort.merge_all(vec![row(&[1, 30]), row(&[1, 31])]).unwrap(),
            vec![
                row(&[1, 10]),
                row(&[1, 11]),
                row(&[1, 20]),
                row(&[1, 21]),
                row(&[1, 30]),
                row(&[1, 31]),
            ]
        );
    }

    #[test]
    fn spilled_merge_uses_the_resident_cross_numeric_comparator() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);

        sort.spill_sorted_run(vec![vec![Value::Int64(2)]]).unwrap();
        sort.spill_sorted_run(vec![vec![Value::Float64(1.5)]])
            .unwrap();

        assert_eq!(
            sort.merge_all(Vec::new()).unwrap(),
            vec![vec![Value::Float64(1.5)], vec![Value::Int64(2)]]
        );
    }

    #[test]
    // reason: test values 1..=7 fit i64
    #[allow(clippy::cast_possible_wrap)]
    fn test_external_sort_runs_with_memory() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);

        // Spill a run
        sort.spill_sorted_run(vec![row(&[1]), row(&[4]), row(&[7])])
            .unwrap();

        // Merge with in-memory buffer
        let buffer = vec![row(&[6]), row(&[3]), row(&[5]), row(&[2])];
        let result = sort.merge_all(buffer).unwrap();

        assert_eq!(result.len(), 7);
        for (i, r) in result.iter().enumerate() {
            assert_eq!(r, &row(&[(i + 1) as i64]));
        }
    }

    #[test]
    // reason: test values 1..=6 fit i64
    #[allow(clippy::cast_possible_wrap)]
    fn test_external_sort_descending() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::descending(0)]);

        sort.spill_sorted_run(vec![row(&[5]), row(&[3]), row(&[1])])
            .unwrap();
        sort.spill_sorted_run(vec![row(&[6]), row(&[4]), row(&[2])])
            .unwrap();

        let result = sort.merge_all(Vec::new()).unwrap();
        assert_eq!(result.len(), 6);
        for (i, r) in result.iter().enumerate() {
            assert_eq!(r, &row(&[(6 - i) as i64]));
        }
    }

    #[test]
    fn test_external_sort_multi_column() {
        let (_temp_dir, manager) = create_manager();
        let sort_keys = vec![SortKey::ascending(0), SortKey::descending(1)];
        let mut sort = ExternalSort::new(manager, 2, sort_keys);

        // Rows: (group, value)
        sort.spill_sorted_run(vec![
            vec![Value::Int64(1), Value::Int64(30)],
            vec![Value::Int64(1), Value::Int64(10)],
            vec![Value::Int64(2), Value::Int64(20)],
        ])
        .unwrap();

        sort.spill_sorted_run(vec![
            vec![Value::Int64(1), Value::Int64(20)],
            vec![Value::Int64(2), Value::Int64(30)],
            vec![Value::Int64(2), Value::Int64(10)],
        ])
        .unwrap();

        let result = sort.merge_all(Vec::new()).unwrap();

        // Expected: sorted by col0 asc, then col1 desc
        // (1,30), (1,20), (1,10), (2,30), (2,20), (2,10)
        assert_eq!(result.len(), 6);
        assert_eq!(result[0], vec![Value::Int64(1), Value::Int64(30)]);
        assert_eq!(result[1], vec![Value::Int64(1), Value::Int64(20)]);
        assert_eq!(result[2], vec![Value::Int64(1), Value::Int64(10)]);
        assert_eq!(result[3], vec![Value::Int64(2), Value::Int64(30)]);
        assert_eq!(result[4], vec![Value::Int64(2), Value::Int64(20)]);
        assert_eq!(result[5], vec![Value::Int64(2), Value::Int64(10)]);
    }

    #[test]
    fn test_external_sort_with_nulls() {
        let (_temp_dir, manager) = create_manager();
        let sort_keys = vec![SortKey {
            column: 0,
            direction: SortDirection::Ascending,
            null_order: NullOrder::Last,
        }];
        let mut sort = ExternalSort::new(manager, 1, sort_keys);

        sort.spill_sorted_run(vec![
            vec![Value::Int64(1)],
            vec![Value::Int64(3)],
            vec![Value::Null],
        ])
        .unwrap();

        sort.spill_sorted_run(vec![vec![Value::Int64(2)], vec![Value::Null]])
            .unwrap();

        let result = sort.merge_all(Vec::new()).unwrap();

        // Nulls should be last
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], vec![Value::Int64(1)]);
        assert_eq!(result[1], vec![Value::Int64(2)]);
        assert_eq!(result[2], vec![Value::Int64(3)]);
        assert_eq!(result[3], vec![Value::Null]);
        assert_eq!(result[4], vec![Value::Null]);
    }

    #[test]
    // reason: test values 0..100 fit i64
    #[allow(clippy::cast_possible_wrap)]
    fn test_external_sort_many_runs() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);

        // Create 10 runs with interleaved values
        for i in 0..10 {
            let run: Vec<Vec<Value>> = (0..10).map(|j| row(&[i + j * 10])).collect();
            sort.spill_sorted_run(run).unwrap();
        }

        assert_eq!(sort.num_runs(), 10);
        assert_eq!(sort.total_rows(), 100);

        let result = sort.merge_all(Vec::new()).unwrap();
        assert_eq!(result.len(), 100);

        // Verify sorted order
        for (i, r) in result.iter().enumerate() {
            assert_eq!(r, &row(&[i as i64]));
        }
    }

    #[test]
    fn test_external_sort_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build()
                .unwrap(),
        );

        {
            let mut sort = ExternalSort::new(Arc::clone(&manager), 1, vec![SortKey::ascending(0)]);
            sort.spill_sorted_run(vec![row(&[1]), row(&[2])]).unwrap();
            sort.spill_sorted_run(vec![row(&[3]), row(&[4])]).unwrap();

            assert!(manager.spilled_bytes() > 0);
            // sort dropped here
        }

        // After drop, spilled bytes should be cleaned up
        // (The manager still exists, but files are deleted)
    }
    #[test]
    fn pull_failure_two_hostile_destructors_keep_the_carrier_charged() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
        #[derive(Debug)]
        struct PanicDrop(Arc<AtomicUsize>);
        impl std::fmt::Display for PanicDrop {
            fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                panic!("opaque error must not be formatted")
            }
        }
        impl std::error::Error for PanicDrop {}
        impl Drop for PanicDrop {
            fn drop(&mut self) {
                self.0.fetch_add(1, AtomicOrdering::Relaxed);
                panic!("opaque error destructor");
            }
        }
        for compound_io in [false, true] {
            let memory = grafeo_common::memory::buffer::BufferManager::with_budget(1 << 20);
            let resources = crate::execution::QueryResourceContext::new(memory.clone()).unwrap();
            let drops = Arc::new(AtomicUsize::new(0));
            let primary = std::io::Error::other(PanicDrop(drops.clone()));
            let cleanup = std::io::Error::other(PanicDrop(drops.clone()));
            let operation = if compound_io {
                ExternalSortOperationError::Io(super::super::combine_primary_and_cleanup(
                    primary,
                    cleanup,
                    "hostile pair",
                ))
            } else {
                ExternalSortOperationError::WithGrantRelease {
                    primary: ExternalSortPrimary::Io(primary),
                    release: MemoryGrantError::Denied {
                        additional_bytes: 1,
                    },
                    cleanup: Some(cleanup),
                    phase: "hostile pair",
                }
            };
            let publisher = grafeo_common::memory::buffer::AccountedErrorPublisher::try_new(
                resources.try_allocate(0).unwrap(),
            )
            .unwrap();
            let charged = publisher.granted_bytes();
            let authority = publisher.publish(PullSortFailure {
                primary: None,
                operation: Some(operation),
                release: None,
                workspaces: [None, None, None],
                hook_workspace: None,
            });
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(authority)));
            assert!(panic.is_err());
            assert_eq!(
                drops.load(AtomicOrdering::Relaxed),
                2,
                "each known opaque owner needs its own unwind boundary"
            );
            assert_eq!(
                memory.allocated(),
                charged,
                "failed opaque destruction must not release its carrier grant"
            );
        }
    }
    #[test]
    fn pull_failure_retains_provider_workspace_until_escaped_payload_drops() {
        const PAYLOAD_BYTES: usize = 4096;
        struct Payload {
            bytes: Box<[u8; PAYLOAD_BYTES]>,
            memory: Arc<grafeo_common::memory::buffer::BufferManager>,
            drops: Arc<std::sync::atomic::AtomicUsize>,
        }
        impl std::fmt::Debug for Payload {
            fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                out.write_str("ProviderPayload")
            }
        }
        impl std::fmt::Display for Payload {
            fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                panic!("provider payload must remain opaque")
            }
        }
        impl std::error::Error for Payload {}
        impl Drop for Payload {
            fn drop(&mut self) {
                assert_eq!(self.bytes[0], 3);
                assert!(
                    self.memory.allocated() >= PAYLOAD_BYTES,
                    "provider diagnostic lost its admitted workspace before destruction"
                );
                self.drops
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        struct Provider {
            memory: Arc<grafeo_common::memory::buffer::BufferManager>,
            drops: Arc<std::sync::atomic::AtomicUsize>,
        }
        impl super::super::SpillRecordProvider for Provider {
            fn seals(&self) -> bool {
                false
            }
            fn file_workspace_allocation_bound(&self) -> Option<usize> {
                Some(
                    PAYLOAD_BYTES
                        + std::mem::size_of::<Payload>()
                        + std::mem::size_of::<(
                            std::io::ErrorKind,
                            Box<dyn std::error::Error + Send + Sync>,
                        )>(),
                )
            }
            fn begin_file(
                &self,
                _: super::super::SpillFileIdentity,
            ) -> std::io::Result<Box<dyn super::super::OpenSpillRecord>> {
                Err(std::io::Error::other(Payload {
                    bytes: Box::new([3; PAYLOAD_BYTES]),
                    memory: self.memory.clone(),
                    drops: self.drops.clone(),
                }))
            }
        }
        let memory = grafeo_common::memory::buffer::BufferManager::with_budget(1 << 20);
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let directory = tempfile::tempdir().unwrap();
        let (resources, manager) = super::super::BorrowedSpillFixture::new(directory.path())
            .provider(
                Arc::new(Provider {
                    memory: memory.clone(),
                    drops: drops.clone(),
                }),
                super::super::SpillFrameLimits::format_max(),
            )
            .build_operator_resources(
                memory.clone(),
                crate::execution::QueryExecutionControl::new().token(),
            )
            .unwrap();
        let mut sort = ExternalSort::new_accounted(
            manager.clone(),
            1,
            vec![SortKey::ascending(0)],
            resources.try_allocate(0).unwrap(),
        );
        sort.retain_pull_failure_workspaces();
        let publisher =
            AccountedErrorPublisher::try_new(resources.try_allocate(0).unwrap()).unwrap();
        let operation = sort
            .spill_sorted_run_accounted_observing(&[row(&[1])], &ExternalSortGrantObserver::inert())
            .unwrap_err();
        let workspaces = sort.take_pull_failure_workspaces();
        let retained = workspaces
            .iter()
            .flatten()
            .map(MemoryGrant::size)
            .sum::<usize>();
        assert!(retained >= PAYLOAD_BYTES);
        let authority = publisher.publish(PullSortFailure {
            primary: None,
            operation: Some(operation),
            release: None,
            workspaces,
            hook_workspace: None,
        });
        drop(sort);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(drops.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(memory.allocated(), retained + authority.granted_bytes());
        drop(authority);
        assert_eq!(drops.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(memory.allocated(), 0);
    }
}
