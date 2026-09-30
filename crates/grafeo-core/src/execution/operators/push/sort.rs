//! Push-based sort operator (pipeline breaker).

use crate::execution::QueryCancellationToken;
#[cfg(feature = "spill")]
use crate::execution::accounted_chunk::{
    AccountedSortChunkPrimary, try_accounted_chunk_from_sort_row,
};
use crate::execution::accounted_chunk::{SortRowShape, SortValueType};
use crate::execution::chunk::DataChunk;
#[cfg(all(test, feature = "spill"))]
use crate::execution::operators::AccountedFailureClassification;
use crate::execution::operators::OperatorError;
use crate::execution::operators::value_utils::compare_values_total;
use crate::execution::pipeline::{ChunkSizeHint, PushOperator, Sink};
#[cfg(feature = "spill")]
use crate::execution::spill::{
    ExternalSort, ExternalSortGrantObserver, ExternalSortOperationError, ExternalSortPrimary,
    PullSortFailure, SpillManager, classify_external_sort_error, classify_operator_error,
};
use crate::execution::value_codec::decoded_bytes_payload_retained_bytes;
#[cfg(feature = "spill")]
use crate::execution::vector::ResidentCapacityError;
use crate::execution::vector::ValueVector;
#[cfg(feature = "spill")]
use grafeo_common::memory::buffer::{AccountedErrorPublisher, AccountedErrorPublisherBuildFailure};
use grafeo_common::memory::buffer::{MemoryGrant, MemoryGrantError};
use grafeo_common::types::Value;
#[cfg(feature = "spill")]
use std::cell::Cell;
use std::cmp::Ordering;
use std::sync::Arc;

/// Initial outer-row capacity used by grant-aware sort ingestion.
const INITIAL_SORT_ROW_CAPACITY: usize = 4;

/// Conservative qualified emission batch. Decoded spill heads carry a
/// deliberately pessimistic nested-value envelope, so a small fixed batch
/// preserves progress under tight query budgets while remaining amortized.
const QUALIFIED_SORT_OUTPUT_CHUNK_ROWS: usize = 32;

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
    /// Create a new ascending sort key.
    pub fn ascending(column: usize) -> Self {
        Self {
            column,
            direction: SortDirection::Ascending,
            null_order: NullOrder::Last,
        }
    }

    /// Create a new descending sort key.
    pub fn descending(column: usize) -> Self {
        Self {
            column,
            direction: SortDirection::Descending,
            null_order: NullOrder::First,
        }
    }
}

/// Failure while growing the sort-owned row containers.
#[derive(Debug)]
enum SortBufferGrowthError {
    Memory(MemoryGrantError),
    Allocation {
        container: &'static str,
        detail: String,
    },
    CapacityOverflow {
        container: &'static str,
    },
}

impl SortBufferGrowthError {
    fn allocation(container: &'static str, error: std::collections::TryReserveError) -> Self {
        Self::Allocation {
            container,
            detail: error.to_string(),
        }
    }

    fn into_operator_error(self) -> OperatorError {
        match self {
            Self::Memory(error) => OperatorError::ResidentMemory(error),
            Self::Allocation { container, detail } => OperatorError::ResidentAllocation(format!(
                "allocator refused {container} capacity: {detail}"
            )),
            Self::CapacityOverflow { container } => OperatorError::ResidentAllocation(format!(
                "{container} capacity exceeds the platform address space"
            )),
        }
    }

    #[cfg(feature = "spill")]
    fn is_resident_grant_denial(&self) -> bool {
        matches!(
            self,
            Self::Memory(MemoryGrantError::LimitExceeded { .. } | MemoryGrantError::Denied { .. })
        )
    }
}

impl From<MemoryGrantError> for SortBufferGrowthError {
    fn from(error: MemoryGrantError) -> Self {
        Self::Memory(error)
    }
}

fn capacity_bytes<T>(
    capacity: usize,
    container: &'static str,
) -> Result<usize, SortBufferGrowthError> {
    capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(SortBufferGrowthError::CapacityOverflow { container })
}

fn checked_capacity_sum(
    current: usize,
    additional: usize,
    container: &'static str,
) -> Result<usize, SortBufferGrowthError> {
    current
        .checked_add(additional)
        .ok_or(SortBufferGrowthError::CapacityOverflow { container })
}

/// Sort-owned row containers plus an optional real resident-memory grant.
///
/// `rows` is deliberately declared before `grant`: Rust drops fields in
/// declaration order, so every allocation is freed before its accounting
/// capability releases the corresponding bytes.
struct SortRowBuffer {
    rows: Vec<Vec<Value>>,
    /// Checked row Vec backing plus newly allocated private mask payloads.
    row_capacity_bytes: usize,
    grant: Option<MemoryGrant>,
    #[cfg(all(test, feature = "spill"))]
    release_error: Option<MemoryGrantError>,
}

/// Releases the source buffer's grant after moved rows are physically gone.
///
/// The grant stays attached to the reusable buffer throughout the operation.
/// A failed explicit or unwind release therefore retains the exact accounting
/// capability instead of dropping the only token for conservatively accounted
/// bytes.
#[cfg(feature = "spill")]
struct SortRowGrantRelease<'a> {
    buffer: &'a mut SortRowBuffer,
    remaining_bytes: &'a Cell<usize>,
    armed: bool,
}

#[cfg(feature = "spill")]
impl SortRowGrantRelease<'_> {
    fn release(&mut self) -> Result<(), MemoryGrantError> {
        match self.buffer.release_empty_grant() {
            Ok(()) => {
                self.remaining_bytes.set(0);
                self.armed = false;
                Ok(())
            }
            Err(error) => {
                self.remaining_bytes.set(self.buffer.granted_bytes());
                Err(error)
            }
        }
    }
}

#[cfg(feature = "spill")]
impl Drop for SortRowGrantRelease<'_> {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.release();
        }
    }
}

/// Rows temporarily moved out of the reusable buffer.
///
/// `rows` is declared before the release guard so normal return and unwinding
/// both free every physical row allocation before attempting to shrink the
/// source buffer's retained accounting capability.
#[cfg(feature = "spill")]
struct TakenSortRows<'a> {
    rows: Vec<Vec<Value>>,
    grant_release: SortRowGrantRelease<'a>,
}

#[cfg(feature = "spill")]
impl TakenSortRows<'_> {
    fn rows(&self) -> &[Vec<Value>] {
        &self.rows
    }

    fn take_rows(&mut self) -> Vec<Vec<Value>> {
        std::mem::take(&mut self.rows)
    }

    fn release(mut self) -> Result<(), MemoryGrantError> {
        // Release physical storage before shrinking the corresponding grant.
        self.rows = Vec::new();
        self.grant_release.release()
    }
}

/// Restores consumer telemetry after rows have moved out of the reusable buffer.
///
/// The sealed external-sort observer reports the live row grant plus every external
/// grant while work is in flight. This guard is declared before
/// [`TakenSortRows`], so normal return and unwinding both drop the physical
/// rows and attempt their grant release first, then publish any conservatively
/// retained row grant plus the latest external total. `Cell` is sufficient
/// because external-sort publications are synchronous on the operator thread and
/// avoids introducing a separate heap allocation.
#[cfg(feature = "spill")]
struct SpillUsageAfterMovedRows<'a> {
    state: Option<&'a super::spill_state::OperatorSpillState>,
    row_bytes: &'a Cell<usize>,
    external_bytes: &'a Cell<usize>,
}

#[cfg(feature = "spill")]
impl Drop for SpillUsageAfterMovedRows<'_> {
    fn drop(&mut self) {
        if let Some(state) = self.state {
            // Drop must not unwind. A same-query sum overflow is an internal
            // protocol defect, so keep eviction telemetry fail closed instead
            // of advertising a smaller live allocation.
            state.set_usage(
                self.row_bytes
                    .get()
                    .saturating_add(self.external_bytes.get()),
            );
        }
    }
}

/// Reconciles exact-finalization telemetry from the cursor's final sealed
/// publication, including while a hostile comparator or sink unwinds.
///
/// Clean teardown publishes zero. An unreportable Drop cleanup failure leaves
/// `external_bytes` at `usize::MAX`; this guard must preserve that fail-closed
/// signal rather than inferring success merely because the sorter was moved.
#[cfg(feature = "spill")]
struct ExactFinalizationUsageReconcile<'a> {
    state: Option<&'a super::spill_state::OperatorSpillState>,
    external_bytes: &'a Cell<usize>,
}

#[cfg(feature = "spill")]
impl Drop for ExactFinalizationUsageReconcile<'_> {
    fn drop(&mut self) {
        if let Some(state) = self.state {
            state.set_usage(self.external_bytes.get());
        }
    }
}

#[cfg(feature = "spill")]
fn fail_closed_external_observer_initialization(
    result: Result<usize, MemoryGrantError>,
    spill_state: Option<&super::spill_state::OperatorSpillState>,
) -> Result<usize, OperatorError> {
    match result {
        Ok(bytes) => Ok(bytes),
        Err(error) => {
            if let Some(state) = spill_state {
                state.set_usage(usize::MAX);
            }
            Err(OperatorError::ResidentMemory(error))
        }
    }
}

impl SortRowBuffer {
    #[cfg(feature = "spill")]
    fn new() -> Self {
        Self {
            rows: Vec::new(),
            row_capacity_bytes: 0,
            grant: None,
            #[cfg(all(test, feature = "spill"))]
            release_error: None,
        }
    }

    fn with_resource_context(
        context: &crate::execution::memory::QueryResourceContext,
    ) -> Result<Self, crate::execution::memory::QueryResourceContextError> {
        Ok(Self {
            rows: Vec::new(),
            row_capacity_bytes: 0,
            grant: Some(context.try_allocate(0)?),
            #[cfg(all(test, feature = "spill"))]
            release_error: None,
        })
    }

    fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    #[cfg(feature = "spill")]
    fn len(&self) -> usize {
        self.rows.len()
    }

    fn rows(&self) -> &[Vec<Value>] {
        &self.rows
    }

    fn granted_bytes(&self) -> usize {
        self.grant.as_ref().map_or(0, MemoryGrant::size)
    }

    /// Returns the resident envelope charged for the current input buffer.
    ///
    /// This stage charges private provenance payloads, observed row/outer `Vec` capacities, and the
    /// standard library stable-sort scratch upper bound of one outer element
    /// per buffered row. Codec, merge-frontier, and output allocations remain
    /// separate stages and are not included here.
    fn observed_charge(&self) -> Result<usize, SortBufferGrowthError> {
        let bytes = checked_capacity_sum(
            capacity_bytes::<Vec<Value>>(self.rows.capacity(), "sort outer rows")?,
            self.row_capacity_bytes,
            "sort row capacities",
        )?;
        checked_capacity_sum(
            bytes,
            capacity_bytes::<Vec<Value>>(self.rows.len(), "stable-sort scratch")?,
            "sort resident envelope",
        )
    }

    fn next_outer_capacity(&self, required: usize) -> Result<usize, SortBufferGrowthError> {
        if required <= self.rows.capacity() {
            return Ok(self.rows.capacity());
        }
        if self.rows.capacity() == 0 {
            return Ok(required.max(INITIAL_SORT_ROW_CAPACITY));
        }
        let doubled =
            self.rows
                .capacity()
                .checked_mul(2)
                .ok_or(SortBufferGrowthError::CapacityOverflow {
                    container: "sort outer rows",
                })?;
        Ok(doubled.max(required))
    }

    fn rollback_grant(
        &mut self,
        previous: usize,
        primary: SortBufferGrowthError,
    ) -> SortBufferGrowthError {
        let Some(grant) = self.grant.as_mut() else {
            return primary;
        };
        match grant.try_resize(previous) {
            Ok(()) => primary,
            Err(accounting_error) => SortBufferGrowthError::Memory(accounting_error),
        }
    }

    fn try_push_chunk_row(
        &mut self,
        chunk: &DataChunk,
        row_index: usize,
        num_columns: usize,
    ) -> Result<(), SortBufferGrowthError> {
        let value_type = |column| {
            chunk
                .column(column)
                .map_or(SortValueType::Ordinary, |values| {
                    SortValueType::from_logical(values.data_type())
                })
        };
        let typed = (0..num_columns).any(|column| value_type(column) != SortValueType::Ordinary);
        let mask_len = if typed { num_columns.div_ceil(4) } else { 0 };
        let retained = if typed {
            decoded_bytes_payload_retained_bytes(mask_len)?
        } else {
            0
        };
        let physical_columns =
            checked_capacity_sum(num_columns, usize::from(typed), "sort physical width")?;
        // Reserve both the temporary mask Vec and the retained Arc payload
        // before either is allocated. The mask remains part of the row owner.
        self.try_push_row_with_payload(physical_columns, retained, mask_len, |column| {
            if column < num_columns {
                return Ok(chunk
                    .column(column)
                    .and_then(|values| values.get_value(row_index))
                    .unwrap_or(Value::Null));
            }
            // The pinned Global allocator gives this temporary an exact
            // capacity contract, matching the already admitted transient peak.
            let mut mask = allocator_api2::vec::Vec::new_in(allocator_api2::alloc::Global);
            mask.try_reserve_exact(mask_len).map_err(|error| {
                SortBufferGrowthError::Allocation {
                    container: "sort edge provenance mask",
                    detail: error.to_string(),
                }
            })?;
            if mask.capacity() != mask_len {
                return Err(SortBufferGrowthError::Allocation {
                    container: "sort edge provenance mask",
                    detail: "exact allocator violated its requested capacity".to_string(),
                });
            }
            mask.resize(mask_len, 0u8);
            for column in 0..num_columns {
                mask[column / 4] |= (value_type(column) as u8) << (2 * (column % 4));
            }
            Ok(Value::Bytes(Arc::from(mask.as_slice())))
        })
    }

    fn try_push_row_with_payload(
        &mut self,
        num_columns: usize,
        retained_payload: usize,
        transient_payload: usize,
        mut value_at: impl FnMut(usize) -> Result<Value, SortBufferGrowthError>,
    ) -> Result<(), SortBufferGrowthError> {
        if self.grant.is_none() {
            let mut row = Vec::new();
            row.try_reserve_exact(num_columns)
                .map_err(|error| SortBufferGrowthError::allocation("sort row values", error))?;
            let row_bytes = capacity_bytes::<Value>(row.capacity(), "sort row values")?;
            let row_capacity_bytes = checked_capacity_sum(
                checked_capacity_sum(self.row_capacity_bytes, row_bytes, "sort row capacities")?,
                retained_payload,
                "sort row payload",
            )?;
            for column in 0..num_columns {
                row.push(value_at(column)?);
            }
            self.rows
                .try_reserve(1)
                .map_err(|error| SortBufferGrowthError::allocation("sort outer rows", error))?;
            self.rows.push(row);
            self.row_capacity_bytes = row_capacity_bytes;
            return Ok(());
        }

        let previous_charge = self.granted_bytes();
        let required_rows =
            self.rows
                .len()
                .checked_add(1)
                .ok_or(SortBufferGrowthError::CapacityOverflow {
                    container: "sort row count",
                })?;
        let previous_outer_bytes =
            capacity_bytes::<Vec<Value>>(self.rows.capacity(), "sort outer rows")?;
        let target_outer_capacity = self.next_outer_capacity(required_rows)?;
        let target_outer_bytes =
            capacity_bytes::<Vec<Value>>(target_outer_capacity, "sort outer rows")?;
        let target_row_bytes = capacity_bytes::<Value>(num_columns, "sort row values")?;
        let scratch_growth = capacity_bytes::<Vec<Value>>(1, "stable-sort scratch")?;

        let without_old_outer = previous_charge.checked_sub(previous_outer_bytes).ok_or(
            SortBufferGrowthError::CapacityOverflow {
                container: "sort grant reconciliation",
            },
        )?;
        let mut provisional_charge = checked_capacity_sum(
            checked_capacity_sum(
                checked_capacity_sum(without_old_outer, target_outer_bytes, "sort outer rows")?,
                target_row_bytes,
                "sort row values",
            )?,
            scratch_growth,
            "stable-sort scratch",
        )?;
        provisional_charge = checked_capacity_sum(
            provisional_charge,
            checked_capacity_sum(retained_payload, transient_payload, "sort payload peak")?,
            "sort row payload",
        )?;
        let outer_growth = target_outer_capacity != self.rows.capacity();
        if outer_growth {
            // During reallocation both the old and replacement outer vectors
            // coexist. Charge that transient peak before allocating either the
            // row or the replacement container.
            provisional_charge = checked_capacity_sum(
                provisional_charge,
                previous_outer_bytes,
                "sort outer reallocation peak",
            )?;
        }
        self.grant
            .as_mut()
            .expect("checked grant presence")
            .try_resize(provisional_charge)?;

        let mut row = Vec::new();
        if let Err(error) = row.try_reserve_exact(num_columns) {
            let primary = SortBufferGrowthError::allocation("sort row values", error);
            drop(row);
            return Err(self.rollback_grant(previous_charge, primary));
        }
        let observed_row_bytes = match capacity_bytes::<Value>(row.capacity(), "sort row values") {
            Ok(bytes) => bytes,
            Err(primary) => {
                drop(row);
                return Err(self.rollback_grant(previous_charge, primary));
            }
        };
        if observed_row_bytes > target_row_bytes {
            let additional = observed_row_bytes - target_row_bytes;
            let Some(reconciled) = provisional_charge.checked_add(additional) else {
                drop(row);
                return Err(self.rollback_grant(
                    previous_charge,
                    SortBufferGrowthError::CapacityOverflow {
                        container: "sort row reconciliation",
                    },
                ));
            };
            if let Err(error) = self
                .grant
                .as_mut()
                .expect("checked grant presence")
                .try_resize(reconciled)
            {
                drop(row);
                return Err(
                    self.rollback_grant(previous_charge, SortBufferGrowthError::Memory(error))
                );
            }
            provisional_charge = reconciled;
        }
        let row_capacity_bytes =
            match checked_capacity_sum(observed_row_bytes, retained_payload, "sort row payload")
                .and_then(|row_bytes| {
                    checked_capacity_sum(self.row_capacity_bytes, row_bytes, "sort row capacities")
                }) {
                Ok(bytes) => bytes,
                Err(primary) => {
                    drop(row);
                    return Err(self.rollback_grant(previous_charge, primary));
                }
            };
        for column in 0..num_columns {
            // Existing values only clone Arc owners. The optional final mask
            // creates a payload covered by the admitted transient/retained peak.
            match value_at(column) {
                Ok(value) => row.push(value),
                Err(primary) => {
                    drop(row);
                    return Err(self.rollback_grant(previous_charge, primary));
                }
            }
        }

        let mut replacement = if outer_growth {
            let mut candidate = Vec::new();
            if let Err(error) = candidate.try_reserve_exact(target_outer_capacity) {
                let primary = SortBufferGrowthError::allocation("sort outer rows", error);
                drop(candidate);
                drop(row);
                return Err(self.rollback_grant(previous_charge, primary));
            }
            let observed_outer_bytes =
                match capacity_bytes::<Vec<Value>>(candidate.capacity(), "sort outer rows") {
                    Ok(bytes) => bytes,
                    Err(primary) => {
                        drop(candidate);
                        drop(row);
                        return Err(self.rollback_grant(previous_charge, primary));
                    }
                };
            if observed_outer_bytes > target_outer_bytes {
                let additional = observed_outer_bytes - target_outer_bytes;
                let Some(reconciled) = provisional_charge.checked_add(additional) else {
                    drop(candidate);
                    drop(row);
                    return Err(self.rollback_grant(
                        previous_charge,
                        SortBufferGrowthError::CapacityOverflow {
                            container: "sort outer reconciliation",
                        },
                    ));
                };
                if let Err(error) = self
                    .grant
                    .as_mut()
                    .expect("checked grant presence")
                    .try_resize(reconciled)
                {
                    drop(candidate);
                    drop(row);
                    return Err(
                        self.rollback_grant(previous_charge, SortBufferGrowthError::Memory(error))
                    );
                }
                provisional_charge = reconciled;
            }
            Some(candidate)
        } else {
            None
        };

        if let Some(mut replacement_rows) = replacement.take() {
            let old_rows = std::mem::take(&mut self.rows);
            for existing in old_rows {
                replacement_rows.push(existing);
            }
            self.rows = replacement_rows;
        }
        self.rows.push(row);
        self.row_capacity_bytes = row_capacity_bytes;

        let observed = self.observed_charge()?;
        debug_assert!(
            observed <= provisional_charge,
            "pre-growth sort charge must cover observed capacities"
        );
        self.grant
            .as_mut()
            .expect("checked grant presence")
            .try_resize(observed)?;
        Ok(())
    }

    fn sort_by(&mut self, mut compare: impl FnMut(&[Value], &[Value]) -> Ordering) {
        self.rows.sort_by(|left, right| compare(left, right));
    }

    #[cfg(feature = "spill")]
    fn release_empty_grant(&mut self) -> Result<(), MemoryGrantError> {
        debug_assert!(self.rows.is_empty());
        debug_assert_eq!(self.row_capacity_bytes, 0);
        #[cfg(all(test, feature = "spill"))]
        if let Some(error) = &self.release_error {
            return Err(error.clone());
        }
        if let Some(grant) = self.grant.as_mut() {
            grant.try_resize(0)?;
        }
        Ok(())
    }

    #[cfg(feature = "spill")]
    fn take_rows_with_grant<'a>(
        &'a mut self,
        remaining_bytes: &'a Cell<usize>,
    ) -> TakenSortRows<'a> {
        self.row_capacity_bytes = 0;
        remaining_bytes.set(self.granted_bytes());
        TakenSortRows {
            rows: std::mem::take(&mut self.rows),
            grant_release: SortRowGrantRelease {
                buffer: self,
                remaining_bytes,
                armed: true,
            },
        }
    }
}

/// Push-based sort operator.
///
/// This is a pipeline breaker that must buffer all input before producing
/// sorted output in the finalize phase. Construction requires the query's
/// resource context through [`Self::with_resource_context`].
///
/// The context-free constructors have been removed:
///
/// ```compile_fail,E0599
/// use grafeo_core::execution::operators::push::SortPushOperator;
/// let _ = SortPushOperator::new;
/// ```
///
/// ```compile_fail,E0599
/// use grafeo_core::execution::operators::push::SortPushOperator;
/// let _ = SortPushOperator::ascending;
/// ```
///
/// ```compile_fail,E0599
/// use grafeo_core::execution::operators::push::SortPushOperator;
/// let _ = SortPushOperator::descending;
/// ```
pub struct SortPushOperator {
    /// Sort keys.
    keys: Vec<SortKey>,
    /// Buffered rows as (row_values...).
    buffer: SortRowBuffer,
    /// Number of columns per row.
    num_columns: Option<usize>,
    /// Check-only capability present only for resource-qualified execution.
    cancellation: Option<QueryCancellationToken>,
}

impl SortPushOperator {
    /// Creates a resident sort whose owned row capacities and stable-sort
    /// scratch envelope consume the query's real memory grant.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_common::memory::buffer::BufferManager;
    /// use grafeo_core::execution::QueryResourceContext;
    /// use grafeo_core::execution::operators::push::{SortKey, SortPushOperator};
    ///
    /// let resources = QueryResourceContext::new(BufferManager::with_budget(1024 * 1024)).unwrap();
    /// let _sort = SortPushOperator::with_resource_context(
    ///     vec![SortKey::ascending(0)],
    ///     resources,
    /// ).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a structured accounting failure if the initial zero-byte grant
    /// cannot be created. Later growth denial is returned by [`PushOperator::push`].
    pub fn with_resource_context(
        keys: Vec<SortKey>,
        context: crate::execution::memory::QueryResourceContext,
    ) -> Result<Self, crate::execution::memory::QueryResourceContextError> {
        let cancellation = context.cancellation_token().clone();
        Ok(Self {
            keys,
            buffer: SortRowBuffer::with_resource_context(&context)?,
            num_columns: None,
            cancellation: Some(cancellation),
        })
    }
}

/// Compare two rows by sort keys.
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

fn emit_sorted_rows(
    rows: &[Vec<Value>],
    num_columns: usize,
    cancellation: Option<&QueryCancellationToken>,
    sink: &mut dyn Sink,
) -> Result<(), OperatorError> {
    emit_sorted_rows_with_continuation(rows, num_columns, cancellation, sink).map(|_| ())
}

fn emit_sorted_rows_with_continuation(
    rows: &[Vec<Value>],
    num_columns: usize,
    cancellation: Option<&QueryCancellationToken>,
    sink: &mut dyn Sink,
) -> Result<bool, OperatorError> {
    poll_cancellation(cancellation)?;
    if rows.is_empty() || num_columns == 0 {
        return Ok(true);
    }

    let shape = SortRowShape::with_edge_trailer(num_columns);
    let mut start = 0;
    while start < rows.len() {
        let mask = shape
            .edge_mask(&rows[start])
            .map_err(|message| OperatorError::Execution(message.to_string()))?;
        let mut end = start + 1;
        while end < rows.len() {
            let next = shape
                .edge_mask(&rows[end])
                .map_err(|message| OperatorError::Execution(message.to_string()))?;
            if next != mask {
                break;
            }
            end += 1;
        }
        let mut columns = Vec::new();
        columns.try_reserve_exact(num_columns).map_err(|error| {
            OperatorError::ResidentAllocation(format!(
                "allocator refused bounded sort output columns: {error}"
            ))
        })?;
        for column in 0..num_columns {
            let data_type = SortRowShape::column_type(mask, column).logical_type();
            columns.push(ValueVector::with_capacity(data_type, end - start));
        }
        for row in &rows[start..end] {
            poll_cancellation(cancellation)?;
            for (column_index, column) in columns.iter_mut().enumerate() {
                column.push(row[column_index].clone());
            }
        }
        poll_cancellation(cancellation)?;
        let continuation = sink.consume(DataChunk::new(columns))?;
        poll_cancellation(cancellation)?;
        if !continuation {
            return Ok(false);
        }
        start = end;
    }
    Ok(true)
}

fn emit_sorted_rows_bounded(
    rows: &[Vec<Value>],
    num_columns: usize,
    cancellation: Option<&QueryCancellationToken>,
    sink: &mut dyn Sink,
) -> Result<(), OperatorError> {
    for rows in rows.chunks(QUALIFIED_SORT_OUTPUT_CHUNK_ROWS) {
        let continuation =
            emit_sorted_rows_with_continuation(rows, num_columns, cancellation, sink)?;
        poll_cancellation(cancellation)?;
        if !continuation {
            break;
        }
    }
    Ok(())
}

fn poll_cancellation(cancellation: Option<&QueryCancellationToken>) -> Result<(), OperatorError> {
    if let Some(cancellation) = cancellation {
        cancellation.check()?;
    }
    Ok(())
}

impl PushOperator for SortPushOperator {
    fn push(&mut self, chunk: DataChunk, _sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        poll_cancellation(self.cancellation.as_ref())?;
        if chunk.is_empty() {
            return Ok(true);
        }

        let num_cols = chunk.column_count();
        match self.num_columns {
            None => self.num_columns = Some(num_cols),
            Some(expected) if expected != num_cols => {
                return Err(OperatorError::TypeMismatch {
                    expected: format!("sort row with {expected} columns"),
                    found: format!("sort row with {num_cols} columns"),
                });
            }
            Some(_) => {}
        }

        // Buffer all rows
        for i in chunk.selected_indices() {
            poll_cancellation(self.cancellation.as_ref())?;
            self.buffer
                .try_push_chunk_row(&chunk, i, num_cols)
                .map_err(SortBufferGrowthError::into_operator_error)?;
            poll_cancellation(self.cancellation.as_ref())?;
        }

        Ok(true)
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        poll_cancellation(self.cancellation.as_ref())?;
        if self.buffer.is_empty() {
            return Ok(());
        }

        // Sort the buffer - borrow keys separately to avoid borrow conflict
        let keys = &self.keys;
        self.buffer.sort_by(|a, b| {
            compare_rows(
                &a[..self.num_columns.unwrap_or(0)],
                &b[..self.num_columns.unwrap_or(0)],
                keys,
            )
        });
        poll_cancellation(self.cancellation.as_ref())?;

        // Emit sorted rows in chunks
        let num_cols = self.num_columns.unwrap_or(0);
        if num_cols == 0 {
            return Ok(());
        }

        if self.cancellation.is_some() {
            emit_sorted_rows_bounded(
                self.buffer.rows(),
                num_cols,
                self.cancellation.as_ref(),
                sink,
            )
        } else {
            emit_sorted_rows(
                self.buffer.rows(),
                num_cols,
                self.cancellation.as_ref(),
                sink,
            )
        }
    }

    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        // Sort is a breaker, chunk size doesn't matter much
        ChunkSizeHint::Default
    }

    fn name(&self) -> &'static str {
        "SortPush"
    }
}

/// Default spill threshold (number of rows before spilling).
#[cfg(feature = "spill")]
pub const DEFAULT_SPILL_THRESHOLD: usize = 100_000;

/// Minimum buffer size (rows) before memory-pressure spilling can trigger.
///
/// Prevents "noisy neighbor" scenarios where a tiny sort buffer gets spilled
/// because unrelated subsystems consumed memory.
#[cfg(feature = "spill")]
const SORT_MIN_BUFFER_ROWS: usize = 1000;

#[cfg(feature = "spill")]
struct PullSortFailureTransport {
    publisher: AccountedErrorPublisher<PullSortFailure>,
    hook_workspace: grafeo_common::memory::buffer::AccountedError,
}

/// Push-based sort operator with spilling support.
///
/// This is a pipeline breaker that buffers input and spills to disk
/// when memory pressure is high. It uses external merge sort for
/// out-of-core sorting.
///
/// Three construction modes are supported:
///
/// 1. **Resource-context mode** (when constructed with
///    `with_resource_context`): registers as a scoped `MemoryConsumer` and
///    charges observed input-row capacities plus the stable-sort scratch
///    envelope, reusable row-codec workspace, exact run-catalog slot backing,
///    and transient fixed synchronous writer backing to separate real grants.
///    Denied catalog or writer growth returns typed resource exhaustion before
///    file creation. A denied input-row grant spills a non-empty resident run
///    once and retries that exact row once; every continued denial or concrete
///    spill failure remains structured. Existing pressure-triggered spilling
///    is unchanged.
///
/// 2. **Row-count spill mode** (when constructed with `with_spilling`): spills
///    when `buffer.len() >= spill_threshold`.
///
/// 3. **Resident compatibility mode** (when constructed with `new`): no spill
///    manager is attached, so the row threshold cannot spill or bound the
///    buffer. This mode may grow without bound and is not bounded execution.
///
/// The former memory-context constructor is unavailable:
///
/// ```compile_fail,E0599
/// use grafeo_core::execution::operators::push::SpillableSortPushOperator;
/// let _ = SpillableSortPushOperator::with_memory_context;
/// ```
#[cfg(feature = "spill")]
pub struct SpillableSortPushOperator {
    /// Exact scoped registration. Declared first so it deactivates before any
    /// state a future callback adapter could reference is destroyed.
    _consumer_registration: Option<grafeo_common::memory::buffer::ConsumerRegistration>,
    /// Sort keys.
    keys: Vec<SortKey>,
    semantic_comparator: Option<crate::execution::spill::SemanticRowComparator>,
    /// Buffered rows as (row_values...).
    buffer: SortRowBuffer,
    /// Number of columns per row.
    num_columns: Option<usize>,
    /// Spill manager for explicit row-count mode.
    ///
    /// Resource-context mode obtains its manager through `memory_ctx`; resident
    /// compatibility mode has neither manager.
    spill_manager: Option<Arc<SpillManager>>,
    /// Zero-sized grant transferred into the lazy qualified external sorter.
    pending_workspace_grant: Option<MemoryGrant>,
    /// External sort state (created when first spill occurs).
    external_sort: Option<ExternalSort>,
    /// Row threshold used only when an explicit spill manager is attached.
    spill_threshold: usize,
    /// Memory context for pressure-aware spilling.
    memory_ctx: Option<crate::execution::memory::QueryResourceContext>,
    /// Shared state with the registered MemoryConsumer adapter.
    spill_state: Option<std::sync::Arc<super::spill_state::OperatorSpillState>>,
    /// Once exact dispatch consumes the sorter, no later finalize call may
    /// reinterpret its empty fields as a successful no-spill result.
    exact_finalize_consumed: bool,
    /// Pull boundaries can flush one physical row at a time. Binary carries
    /// bound live run handles without repeatedly rewriting a growing prefix.
    bounded_pull_runs: bool,
    pull_failure_publisher: Option<PullSortFailureTransport>,
    pull_cleanup_publisher: Option<PullSortFailureTransport>,
    pull_terminal_error: Option<OperatorError>,
}

#[cfg(feature = "spill")]
impl SpillableSortPushOperator {
    /// Creates a resident compatibility sort operator with the given keys.
    ///
    /// No spill manager is attached, so this constructor does not spill and
    /// does not bound resident memory. Use [`Self::with_spilling`] for explicit
    /// row-threshold spill mode.
    pub fn new(keys: Vec<SortKey>) -> Self {
        Self {
            keys,
            semantic_comparator: None,
            buffer: SortRowBuffer::new(),
            num_columns: None,
            spill_manager: None,
            pending_workspace_grant: None,
            external_sort: None,
            spill_threshold: DEFAULT_SPILL_THRESHOLD,
            memory_ctx: None,
            spill_state: None,
            exact_finalize_consumed: false,
            bounded_pull_runs: false,
            pull_failure_publisher: None,
            pull_cleanup_publisher: None,
            pull_terminal_error: None,
            _consumer_registration: None,
        }
    }

    /// Create a new spillable sort operator with spilling enabled (row-count mode).
    pub fn with_spilling(keys: Vec<SortKey>, manager: Arc<SpillManager>, threshold: usize) -> Self {
        Self {
            keys,
            semantic_comparator: None,
            buffer: SortRowBuffer::new(),
            num_columns: None,
            spill_manager: Some(manager),
            pending_workspace_grant: None,
            external_sort: None,
            spill_threshold: threshold,
            memory_ctx: None,
            spill_state: None,
            exact_finalize_consumed: false,
            bounded_pull_runs: false,
            pull_failure_publisher: None,
            pull_cleanup_publisher: None,
            pull_terminal_error: None,
            _consumer_registration: None,
        }
    }

    /// Create a spillable sort operator with memory-aware spilling.
    ///
    /// Registers as a `MemoryConsumer` with the `BufferManager` and spills
    /// based on system memory pressure rather than row count thresholds.
    ///
    /// # Errors
    ///
    /// Returns a structured error if no spill manager is attached or the
    /// scoped consumer registration identity cannot be created.
    pub fn with_resource_context(
        keys: Vec<SortKey>,
        ctx: crate::execution::memory::QueryResourceContext,
    ) -> Result<Self, crate::execution::memory::QueryResourceContextError> {
        use super::spill_state::{OperatorConsumerAdapter, OperatorSpillState};

        if ctx.ensure_spill_manager()?.is_none() {
            return Err(
                crate::execution::memory::QueryResourceContextError::SpillManagerUnavailable,
            );
        }
        let buffer = SortRowBuffer::with_resource_context(&ctx)?;
        let workspace_grant = ctx.try_allocate(0)?;
        let state = std::sync::Arc::new(OperatorSpillState::new("SpillableSortPush".to_string()));
        let adapter =
            std::sync::Arc::new(OperatorConsumerAdapter::new(std::sync::Arc::clone(&state)));
        let consumer_registration = ctx.register_consumer_scoped(adapter)?;

        Ok(Self {
            keys,
            semantic_comparator: None,
            buffer,
            num_columns: None,
            spill_manager: None,
            pending_workspace_grant: Some(workspace_grant),
            external_sort: None,
            spill_threshold: DEFAULT_SPILL_THRESHOLD,
            memory_ctx: Some(ctx),
            spill_state: Some(state),
            exact_finalize_consumed: false,
            bounded_pull_runs: false,
            pull_failure_publisher: None,
            pull_cleanup_publisher: None,
            pull_terminal_error: None,
            _consumer_registration: Some(consumer_registration),
        })
    }

    pub(crate) fn with_semantic_comparator(
        mut self,
        provider: Arc<dyn crate::execution::operators::AccountedValueComparator>,
        resources: &crate::execution::QueryResourceContext,
    ) -> Result<Self, OperatorError> {
        let key_bytes = self
            .keys
            .len()
            .checked_mul(std::mem::size_of::<crate::execution::spill::SortKey>())
            .ok_or(OperatorError::ResidentMemory(
                MemoryGrantError::ArithmeticOverflow {
                    current_bytes: self.keys.len(),
                    additional_bytes: usize::MAX,
                },
            ))?;
        let temporary_key_grant =
            resources
                .try_allocate(key_bytes)
                .map_err(|error| match error {
                    crate::execution::QueryResourceContextError::Memory(error) => {
                        OperatorError::ResidentMemory(error)
                    }
                    other => OperatorError::Execution(other.to_string()),
                })?;
        let mut keys = Vec::new();
        keys.try_reserve_exact(self.keys.len()).map_err(|source| {
            OperatorError::ResidentContainerAllocation {
                container: "semantic sort keys",
                source,
            }
        })?;
        if keys.capacity() != self.keys.len() {
            return Err(OperatorError::ResidentContainerInvariant {
                container: "semantic sort keys",
                message: "exact capacity contract violated",
            });
        }
        keys.extend(
            self.keys
                .iter()
                .map(|key| crate::execution::spill::SortKey {
                    column: key.column,
                    direction: match key.direction {
                        SortDirection::Ascending => {
                            crate::execution::spill::SortDirection::Ascending
                        }
                        SortDirection::Descending => {
                            crate::execution::spill::SortDirection::Descending
                        }
                    },
                    null_order: match key.null_order {
                        NullOrder::First => crate::execution::spill::NullOrder::First,
                        NullOrder::Last => crate::execution::spill::NullOrder::Last,
                    },
                }),
        );
        self.semantic_comparator = Some(
            crate::execution::spill::SemanticRowComparator::new_accounted(
                keys,
                provider,
                resources.clone(),
            )
            .map_err(|error| match error {
                crate::execution::QueryResourceContextError::Memory(error) => {
                    OperatorError::ResidentMemory(error)
                }
                other => OperatorError::Execution(other.to_string()),
            })?,
        );
        drop(temporary_key_grant);
        Ok(self)
    }

    /// Create a sort operator with a single ascending key and spilling.
    pub fn ascending_with_spilling(
        column: usize,
        manager: Arc<SpillManager>,
        threshold: usize,
    ) -> Self {
        Self::with_spilling(vec![SortKey::ascending(column)], manager, threshold)
    }

    /// Create a sort operator with a single descending key and spilling.
    pub fn descending_with_spilling(
        column: usize,
        manager: Arc<SpillManager>,
        threshold: usize,
    ) -> Self {
        Self::with_spilling(vec![SortKey::descending(column)], manager, threshold)
    }

    /// Sets the threshold for explicit row-count spill mode.
    ///
    /// This does not attach a spill manager, so it does not make a value from
    /// [`Self::new`] spill-capable.
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.spill_threshold = threshold;
        self
    }

    fn configured_spill_manager(&self) -> Option<Arc<SpillManager>> {
        self.memory_ctx
            .as_ref()
            .and_then(|context| context.spill_manager().cloned())
            .or_else(|| self.spill_manager.clone())
    }

    fn ensure_external_sort(&mut self, manager: Arc<SpillManager>) {
        if self.external_sort.is_some() {
            return;
        }
        let num_cols = self.num_columns.unwrap_or(0);
        let spill_keys = if self.semantic_comparator.is_some() {
            Vec::new()
        } else {
            self.keys
                .iter()
                .map(|key| crate::execution::spill::SortKey {
                    column: key.column,
                    direction: match key.direction {
                        SortDirection::Ascending => {
                            crate::execution::spill::SortDirection::Ascending
                        }
                        SortDirection::Descending => {
                            crate::execution::spill::SortDirection::Descending
                        }
                    },
                    null_order: match key.null_order {
                        NullOrder::First => crate::execution::spill::NullOrder::First,
                        NullOrder::Last => crate::execution::spill::NullOrder::Last,
                    },
                })
                .collect()
        };
        let mut external_sort = if let Some(grant) = self.pending_workspace_grant.take() {
            let cancellation = self
                .memory_ctx
                .as_ref()
                .expect("accounted external sort requires its resource context")
                .cancellation_token()
                .clone();
            if let Some(comparator) = &self.semantic_comparator {
                ExternalSort::new_accounted_with_comparator_and_cancellation(
                    manager,
                    num_cols,
                    comparator.clone(),
                    grant,
                    cancellation,
                )
            } else {
                ExternalSort::new_accounted_with_cancellation(
                    manager,
                    num_cols,
                    spill_keys,
                    grant,
                    cancellation,
                )
            }
        } else {
            ExternalSort::new(manager, num_cols, spill_keys)
        };
        external_sort.enable_edge_provenance();
        if self.bounded_pull_runs {
            external_sort.retain_pull_failure_workspaces();
            external_sort.set_pull_control_bytes(self.pull_control_bytes());
            if let Some(transport) = &self.pull_cleanup_publisher {
                external_sort.retain_pull_hook_workspace(transport.hook_workspace.clone());
            }
        }
        self.external_sort = Some(external_sort);
    }

    fn pull_control_bytes(&self) -> usize {
        self.pull_failure_publisher
            .as_ref()
            .map_or(0, |slot| slot.publisher.granted_bytes())
            .saturating_add(
                self.pull_cleanup_publisher
                    .as_ref()
                    .map_or(0, |slot| slot.publisher.granted_bytes()),
            )
    }

    fn checked_total_granted_bytes(&self) -> Result<usize, MemoryGrantError> {
        let external_sort_bytes = match self.external_sort.as_ref() {
            Some(sort) => sort.checked_total_granted_bytes()?,
            None => {
                let comparator = self
                    .semantic_comparator
                    .as_ref()
                    .map_or(Ok(0), |comparator| comparator.checked_granted_bytes())?;
                let hook = self
                    .pull_cleanup_publisher
                    .as_ref()
                    .or(self.pull_failure_publisher.as_ref())
                    .map_or(0, |slot| {
                        slot.hook_workspace.granted_bytes()
                            + slot
                                .hook_workspace
                                .inspect::<crate::execution::spill::PullSortHookAuthority, _>(
                                    crate::execution::spill::PullSortHookAuthority::granted_bytes,
                                )
                                .expect("sort owns its hook authority")
                    });
                comparator
                    .checked_add(hook)
                    .and_then(|bytes| bytes.checked_add(self.pull_control_bytes()))
                    .ok_or(MemoryGrantError::ArithmeticOverflow {
                        current_bytes: comparator,
                        additional_bytes: hook.saturating_add(self.pull_control_bytes()),
                    })?
            }
        };
        self.buffer
            .granted_bytes()
            .checked_add(external_sort_bytes)
            .ok_or(MemoryGrantError::ArithmeticOverflow {
                current_bytes: self.buffer.granted_bytes(),
                additional_bytes: external_sort_bytes,
            })
    }

    #[cfg(test)]
    fn total_granted_bytes(&self) -> usize {
        self.checked_total_granted_bytes()
            .expect("sort buffer and external grants share one representable query account")
    }

    fn checked_external_granted_bytes_for_observer(&self) -> Result<usize, OperatorError> {
        let result = self
            .external_sort
            .as_ref()
            .expect("qualified observer requires an initialized external sort")
            .checked_total_granted_bytes();
        fail_closed_external_observer_initialization(result, self.spill_state.as_deref())
    }

    fn refresh_spill_usage(&self) {
        if let Some(state) = &self.spill_state {
            state.set_usage(self.checked_total_granted_bytes().unwrap_or(usize::MAX));
        }
    }

    fn prepare_pull_failure_transport(&mut self) -> Result<(), OperatorError> {
        if !self.bounded_pull_runs || self.pull_failure_publisher.is_some() {
            return Ok(());
        }
        let manager =
            self.configured_spill_manager()
                .ok_or(OperatorError::ResidentContainerInvariant {
                    container: "pull sort failure transport",
                    message: "missing spill manager",
                })?;
        if manager.qualified_sort_provider_workspace_bound().is_none() {
            return Err(OperatorError::ResidentContainerInvariant {
                container: "pull sort failure transport",
                message: "spill provider does not declare a qualified file-workspace bound",
            });
        }
        let hook_bytes = manager.qualified_sort_hook_workspace_bound().ok_or(
            OperatorError::ResidentContainerInvariant {
                container: "pull sort failure transport",
                message: "spill hook does not declare a complete sort diagnostic bound",
            },
        )?;
        let resources =
            self.memory_ctx
                .as_ref()
                .ok_or(OperatorError::ResidentContainerInvariant {
                    container: "pull sort failure transport",
                    message: "missing resource account",
                })?;
        let allocate = |bytes| {
            resources.try_allocate(bytes).map_err(|error| match error {
                crate::execution::QueryResourceContextError::Memory(error) => {
                    OperatorError::ResidentMemory(error)
                }
                other => OperatorError::Execution(other.to_string()),
            })
        };
        let map_build =
            |error: grafeo_common::memory::buffer::AccountedErrorPublisherBuildError| match error
                .failure()
            {
                AccountedErrorPublisherBuildFailure::Admission(error) => {
                    OperatorError::ResidentMemory(error.clone())
                }
                _ => OperatorError::ResidentContainerInvariant {
                    container: "pull sort failure transport",
                    message: "publisher allocation failed",
                },
            };
        let hook_grant = allocate(hook_bytes)?;
        let hook_publisher = AccountedErrorPublisher::try_new(allocate(0)?).map_err(map_build)?;
        let hook_authority = hook_publisher.publish(
            crate::execution::spill::PullSortHookAuthority::new(hook_grant),
        );
        let create = || {
            let publisher = AccountedErrorPublisher::try_new(allocate(0)?).map_err(map_build)?;
            Ok::<_, OperatorError>(PullSortFailureTransport {
                publisher,
                hook_workspace: hook_authority.clone(),
            })
        };
        self.pull_cleanup_publisher = Some(create()?);
        self.pull_failure_publisher = Some(create()?);
        Ok(())
    }

    fn map_pull_operation_failure(
        publisher: &mut Option<PullSortFailureTransport>,
        operation: ExternalSortOperationError,
        release: Option<MemoryGrantError>,
        workspaces: [Option<MemoryGrant>; 3],
    ) -> OperatorError {
        if let Some(transport) = publisher.take() {
            return OperatorError::ClassifiedAccountedFailure {
                classification: classify_external_sort_error(&operation),
                authority: transport.publisher.publish(PullSortFailure {
                    primary: None,
                    operation: Some(operation),
                    release,
                    workspaces,
                    hook_workspace: Some(transport.hook_workspace),
                }),
            };
        }
        let primary = Self::map_external_sort_error(operation);
        match release {
            Some(release) => Self::with_row_release_context(primary, &release, "spill"),
            None => primary,
        }
    }

    fn map_external_sort_error(error: ExternalSortOperationError) -> OperatorError {
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
                let primary = primary.with_context(format!("{phase} also failed: {release}"));
                match cleanup {
                    Some(cleanup) => {
                        primary.with_context(format!("sort spill cleanup also failed: {cleanup}"))
                    }
                    None => primary,
                }
            }
        }
    }

    fn map_accounted_chunk_error(error: AccountedSortChunkPrimary) -> OperatorError {
        match error {
            AccountedSortChunkPrimary::Memory(error) => OperatorError::ResidentMemory(error),
            AccountedSortChunkPrimary::Capacity(ResidentCapacityError::Allocation {
                container,
                source,
            }) => OperatorError::ResidentContainerAllocation { container, source },
            AccountedSortChunkPrimary::Capacity(
                error @ ResidentCapacityError::ExactAllocation { .. },
            ) => OperatorError::ResidentExactVectorAllocation(error),
            AccountedSortChunkPrimary::Capacity(ResidentCapacityError::ExactCapacityContract {
                container,
                ..
            }) => OperatorError::ResidentContainerInvariant {
                container,
                message: "pinned exact allocator violated its capacity contract",
            },
            AccountedSortChunkPrimary::Capacity(ResidentCapacityError::ArithmeticOverflow {
                container,
            }) => OperatorError::ResidentContainerInvariant {
                container,
                message: "capacity exceeds the platform address space",
            },
            AccountedSortChunkPrimary::Arithmetic(context) => {
                OperatorError::ResidentContainerInvariant {
                    container: context,
                    message: "capacity exceeds the platform address space",
                }
            }
            AccountedSortChunkPrimary::InvalidShape(message) => {
                OperatorError::ResidentContainerInvariant {
                    container: "accounted sort output",
                    message,
                }
            }
            AccountedSortChunkPrimary::CapacityContract => {
                OperatorError::ResidentContainerInvariant {
                    container: "accounted sort output",
                    message: "observed capacity exceeded its pre-admitted envelope",
                }
            }
        }
    }

    fn with_row_release_context(
        primary: OperatorError,
        accounting: &MemoryGrantError,
        phase: &str,
    ) -> OperatorError {
        primary.with_context(format!(
            "resident-memory release after {phase} also failed: {accounting}"
        ))
    }

    fn with_cursor_cleanup_context(
        primary: OperatorError,
        cleanup: ExternalSortOperationError,
        phase: &str,
    ) -> OperatorError {
        primary.with_context(format!(
            "sort cursor cleanup after {phase} also failed: {}",
            Self::map_external_sort_error(cleanup)
        ))
    }

    /// Sorts and publishes the current run, then releases its resident grant.
    ///
    /// The grant already includes the stable-sort scratch upper envelope. It
    /// remains live while `spill_sorted_run` owns the rows and shrinks only
    /// after that call returns, on both success and failure.
    fn spill_current_buffer(&mut self, manager: Arc<SpillManager>) -> Result<(), OperatorError> {
        let cancellation = self
            .memory_ctx
            .as_ref()
            .map(|context| context.cancellation_token().clone());
        poll_cancellation(cancellation.as_ref())?;
        if self.buffer.is_empty() {
            return Ok(());
        }

        self.prepare_pull_failure_transport()?;
        let _hook_guard = crate::execution::spill::PullSortHookGuard(
            self.pull_cleanup_publisher
                .as_ref()
                .map(|transport| transport.hook_workspace.clone()),
        );
        let keys = &self.keys;
        let width = self.num_columns.unwrap_or(0);
        let failure_publisher = &mut self.pull_failure_publisher;
        if let Some(comparator) = &self.semantic_comparator {
            let sorted = crate::execution::operators::sort::try_stable_sort_by(
                &mut self.buffer.rows,
                |left, right| {
                    comparator
                        .try_compare(&left[..width], &right[..width])
                        .map_err(|error| {
                            Self::map_pull_operation_failure(
                                failure_publisher,
                                error,
                                None,
                                [None, None, None],
                            )
                        })
                },
            );
            if let Err(error) = sorted {
                let bytes = self.pull_control_bytes();
                if let Some(sort) = &mut self.external_sort {
                    sort.set_pull_control_bytes(bytes);
                }
                self.refresh_spill_usage();
                return Err(error);
            }
        } else {
            self.buffer
                .sort_by(|left, right| compare_rows(&left[..width], &right[..width], keys));
        }
        poll_cancellation(cancellation.as_ref())?;
        self.ensure_external_sort(manager);

        let spill_state = self.spill_state.clone();
        let row_bytes = Cell::new(self.buffer.granted_bytes());
        let external_bytes = Cell::new(self.checked_external_granted_bytes_for_observer()?);
        let _usage_after_rows = SpillUsageAfterMovedRows {
            state: spill_state.as_deref(),
            row_bytes: &row_bytes,
            external_bytes: &external_bytes,
        };
        poll_cancellation(cancellation.as_ref())?;
        let mut rows = self.buffer.take_rows_with_grant(&row_bytes);
        let spill_result = if self.memory_ctx.is_some() {
            let observer =
                ExternalSortGrantObserver::new(&external_bytes, &row_bytes, spill_state.as_deref());
            let sort = self.external_sort.as_mut().ok_or_else(|| {
                OperatorError::Execution("external sort missing before spilling".into())
            })?;
            if self.bounded_pull_runs {
                // Pull SORT defers its first bounded fan-in of runs, then
                // uses binary carries; stable duplicates remain intact.
                sort.spill_pull_run_accounted(rows.rows(), &observer)
            } else {
                sort.spill_sorted_run_accounted_observing(rows.rows(), &observer)
            }
        } else {
            let rows = rows.take_rows();
            self.external_sort
                .as_mut()
                .expect("external sort initialized before spilling")
                .spill_sorted_run(rows)
                .map_err(ExternalSortOperationError::Io)
        };
        let failure_workspaces = if spill_result.is_err() && self.bounded_pull_runs {
            self.external_sort.as_mut().map_or(
                [None, None, None],
                ExternalSort::take_pull_failure_workspaces,
            )
        } else {
            [None, None, None]
        };
        let release_result = rows.release();
        self.refresh_spill_usage();

        let result = match (spill_result, release_result) {
            (Ok(()), Ok(())) => poll_cancellation(cancellation.as_ref()),
            (Err(primary), release) => Err(Self::map_pull_operation_failure(
                &mut self.pull_failure_publisher,
                primary,
                release.err(),
                failure_workspaces,
            )),
            (Ok(()), Err(accounting)) => Err(OperatorError::ResidentMemory(accounting)),
        };
        let control_bytes = self.pull_control_bytes();
        if let Some(sort) = &mut self.external_sort {
            sort.set_pull_control_bytes(control_bytes);
        }
        // The moved-row observer owns its own scalar snapshot. Retire it
        // before the final recomputation following grant/control transfer.
        drop(_usage_after_rows);
        self.refresh_spill_usage();
        result
    }

    /// Feeds a borrowed physical chunk through the same admitted row buffer
    /// used by push execution. The pull owner retains its columnar grant until
    /// this call returns; no input ownership is inferred from the borrow.
    pub(crate) fn ingest_pull_chunk(&mut self, chunk: &DataChunk) -> Result<(), OperatorError> {
        self.bounded_pull_runs = true;
        let cancellation = self
            .memory_ctx
            .as_ref()
            .map(|context| context.cancellation_token().clone());
        poll_cancellation(cancellation.as_ref())?;
        let width = chunk.column_count();
        if let Some(expected) = self.num_columns {
            if expected != width {
                return Err(OperatorError::TypeMismatch {
                    expected: format!("sort row with {expected} columns"),
                    found: format!("sort row with {width} columns"),
                });
            }
        } else {
            self.num_columns = Some(width);
        }
        for row in chunk.selected_indices() {
            poll_cancellation(cancellation.as_ref())?;
            self.push_row_with_spill_retry(chunk, row, width, cancellation.as_ref())?;
        }
        self.maybe_spill()
    }

    pub(crate) fn pull_buffered_rows(&self) -> usize {
        self.buffer.len()
    }

    /// Drains a transition batch before the pull owner releases its raw input.
    pub(crate) fn flush_pull_batch(&mut self) -> Result<(), OperatorError> {
        let manager = self
            .configured_spill_manager()
            .ok_or_else(|| OperatorError::Execution("pull sort has no spill manager".into()))?;
        self.spill_current_buffer(manager)
    }

    pub(crate) fn finish_pull_input(
        &mut self,
    ) -> Result<crate::execution::spill::OwnedExactSortCursor, OperatorError> {
        self.flush_pull_batch()?;
        let _hook_guard = crate::execution::spill::PullSortHookGuard(
            self.pull_cleanup_publisher
                .as_ref()
                .map(|transport| transport.hook_workspace.clone()),
        );
        let state = self.spill_state.clone().ok_or_else(|| {
            OperatorError::Execution("pull sort has no accounted spill observer".into())
        })?;
        let observer = ExternalSortGrantObserver::owned(state.clone());
        let sort = self
            .external_sort
            .as_mut()
            .ok_or_else(|| OperatorError::Execution("pull sort has no input runs".into()))?;
        sort.finish_distinct_runs_accounted(&observer)
            .map_err(|error| {
                let workspaces = sort.take_pull_failure_workspaces();
                sort.set_pull_control_bytes(
                    self.pull_cleanup_publisher
                        .as_ref()
                        .map_or(0, |slot| slot.publisher.granted_bytes()),
                );
                Self::map_pull_operation_failure(
                    &mut self.pull_failure_publisher,
                    error,
                    None,
                    workspaces,
                )
            })?;
        if !sort.try_enable_exact_owned_output(Some(&state)) {
            return Err(OperatorError::ResidentMemory(MemoryGrantError::Denied {
                additional_bytes: 1,
            }));
        }
        self.exact_finalize_consumed = true;
        // Exact output has its own pre-admitted terminal publisher from here.
        self.pull_failure_publisher = None;
        self.pull_cleanup_publisher = None;
        sort.set_pull_control_bytes(0);
        // The final cursor now owns the only admitted comparator reference.
        self.semantic_comparator = None;
        self.external_sort
            .take()
            .ok_or_else(|| OperatorError::Execution("pull sort already consumed its runs".into()))?
            .into_send_owned_disk_cursor(observer)
            .map_err(|error| error.into_operator_error())
    }

    pub(crate) fn finish_pull_failure(&mut self, primary: OperatorError) -> OperatorError {
        self.finish_pull_cleanup(Some(primary))
            .expect_err("a primary always survives cleanup")
    }

    pub(crate) fn finish_pull_cleanup(
        &mut self,
        primary: Option<OperatorError>,
    ) -> Result<(), OperatorError> {
        if let Some(error) = &self.pull_terminal_error {
            return Err(error.clone());
        }
        let _hook_guard = crate::execution::spill::PullSortHookGuard(
            self.pull_cleanup_publisher
                .as_ref()
                .map(|transport| transport.hook_workspace.clone()),
        );
        let cleanup = self
            .external_sort
            .as_mut()
            .and_then(|sort| sort.cleanup_distinct_accounted().err());
        if let Some(cleanup) = cleanup {
            let error = if let Some(transport) = self.pull_cleanup_publisher.take() {
                let classification = primary.as_ref().map_or_else(
                    || classify_external_sort_error(&cleanup),
                    classify_operator_error,
                );
                OperatorError::ClassifiedAccountedFailure {
                    classification,
                    authority: transport.publisher.publish(PullSortFailure {
                        primary,
                        operation: Some(cleanup),
                        release: None,
                        workspaces: [None, None, None],
                        hook_workspace: Some(transport.hook_workspace),
                    }),
                }
            } else {
                match primary {
                    Some(primary) => {
                        Self::with_cursor_cleanup_context(primary, cleanup, "pull sort cleanup")
                    }
                    None => Self::map_external_sort_error(cleanup),
                }
            };
            self.pull_terminal_error = Some(error.clone());
            let control_bytes = self.pull_control_bytes();
            if let Some(sort) = &mut self.external_sort {
                sort.set_pull_control_bytes(control_bytes);
            }
            self.refresh_spill_usage();
            return Err(error);
        }
        self.pull_failure_publisher = None;
        self.pull_cleanup_publisher = None;
        if let Some(sort) = &mut self.external_sort {
            sort.set_pull_control_bytes(0);
        }
        self.refresh_spill_usage();
        primary.map_or(Ok(()), Err)
    }

    /// Checks whether spilling should occur and performs it if needed.
    fn maybe_spill(&mut self) -> Result<(), OperatorError> {
        let cancellation = self
            .memory_ctx
            .as_ref()
            .map(crate::execution::memory::QueryResourceContext::cancellation_token);
        poll_cancellation(cancellation)?;
        let should_spill = if let Some(ref state) = self.spill_state {
            // Memory-aware: eviction requested OR system pressure is High/Critical
            let eviction = state.take_eviction_request().is_some();
            let pressure = self.memory_ctx.as_ref().map_or(false, |c| c.should_spill());
            // Minimum buffer guard: don't spill tiny buffers from noisy neighbors
            let above_minimum = self.buffer.len() >= SORT_MIN_BUFFER_ROWS;
            (eviction || pressure) && above_minimum
        } else {
            // Non-context path. This can spill only when an explicit manager
            // was attached by `with_spilling`.
            self.buffer.len() >= self.spill_threshold
        };

        if !should_spill {
            return Ok(());
        }

        // Get SpillManager: prefer memory_ctx, fall back to self.spill_manager
        let manager = self.configured_spill_manager();

        let Some(manager) = manager else {
            return Ok(()); // No spilling configured
        };

        self.spill_current_buffer(manager)
    }

    fn push_row_with_spill_retry(
        &mut self,
        chunk: &DataChunk,
        row_index: usize,
        num_columns: usize,
        cancellation: Option<&QueryCancellationToken>,
    ) -> Result<(), OperatorError> {
        let push_row =
            |buffer: &mut SortRowBuffer| buffer.try_push_chunk_row(chunk, row_index, num_columns);

        let first_error = match push_row(&mut self.buffer) {
            Ok(()) => {
                self.refresh_spill_usage();
                return Ok(());
            }
            Err(error) => error,
        };
        self.refresh_spill_usage();

        if !first_error.is_resident_grant_denial()
            || self.memory_ctx.is_none()
            || self.buffer.is_empty()
        {
            return Err(first_error.into_operator_error());
        }
        let Some(manager) = self.configured_spill_manager() else {
            return Err(first_error.into_operator_error());
        };

        self.spill_current_buffer(manager)?;
        poll_cancellation(cancellation)?;

        let retry = push_row(&mut self.buffer);
        self.refresh_spill_usage();
        retry.map_err(SortBufferGrowthError::into_operator_error)
    }
}

#[cfg(feature = "spill")]
impl PushOperator for SpillableSortPushOperator {
    fn push(&mut self, chunk: DataChunk, _sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        if self.exact_finalize_consumed {
            return Err(OperatorError::ResidentContainerInvariant {
                container: "exact owned sort finalization",
                message: "the consuming finalization fence has already fired",
            });
        }
        let cancellation = self
            .memory_ctx
            .as_ref()
            .map(|context| context.cancellation_token().clone());
        poll_cancellation(cancellation.as_ref())?;
        if chunk.is_empty() {
            return Ok(true);
        }

        let num_cols = chunk.column_count();
        match self.num_columns {
            None => self.num_columns = Some(num_cols),
            Some(expected) if expected != num_cols => {
                return Err(OperatorError::TypeMismatch {
                    expected: format!("sort row with {expected} columns"),
                    found: format!("sort row with {num_cols} columns"),
                });
            }
            Some(_) => {}
        }

        for i in chunk.selected_indices() {
            poll_cancellation(cancellation.as_ref())?;
            self.push_row_with_spill_retry(&chunk, i, num_cols, cancellation.as_ref())?;
            poll_cancellation(cancellation.as_ref())?;
        }

        // Check if we should spill
        self.maybe_spill()?;

        Ok(true)
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        if self.exact_finalize_consumed {
            return Err(OperatorError::ResidentContainerInvariant {
                container: "exact owned sort finalization",
                message: "the consuming finalization fence has already fired",
            });
        }
        let cancellation = self
            .memory_ctx
            .as_ref()
            .map(|context| context.cancellation_token().clone());
        poll_cancellation(cancellation.as_ref())?;
        let num_cols = self.num_columns.unwrap_or(0);
        if num_cols == 0 && self.buffer.is_empty() {
            return Ok(());
        }

        if self.external_sort.is_none() {
            // No spilling occurred: retain the charged rows through output
            // construction and until operator drop.
            let keys = &self.keys;
            self.buffer.sort_by(|a, b| {
                compare_rows(
                    &a[..self.num_columns.unwrap_or(0)],
                    &b[..self.num_columns.unwrap_or(0)],
                    keys,
                )
            });
            poll_cancellation(cancellation.as_ref())?;
            return if self.memory_ctx.is_some() {
                emit_sorted_rows_bounded(self.buffer.rows(), num_cols, cancellation.as_ref(), sink)
            } else {
                emit_sorted_rows(self.buffer.rows(), num_cols, cancellation.as_ref(), sink)
            };
        }

        // The first exact owned-output lane is intentionally narrow. Probe
        // the unforgeable sink permit before consuming the sorter; every
        // unqualified sink or sorter shape continues through the established
        // borrowed-chunk compatibility path below without state change.
        let exact_sort_base_eligible = self.buffer.is_empty()
            && self.memory_ctx.is_some()
            && self
                .external_sort
                .as_ref()
                .is_some_and(ExternalSort::exact_owned_disk_base_shape_eligible);
        let exact_admission_spill_state = self.spill_state.clone();
        if exact_sort_base_eligible
            && let Some(mut sink_permit) = sink.__accounted_sink_permit()
            && self
                .external_sort
                .as_ref()
                .is_some_and(ExternalSort::exact_owned_disk_shape_eligible)
            && self.external_sort.as_mut().is_some_and(|sorter| {
                sorter.try_enable_exact_owned_output(exact_admission_spill_state.as_deref())
            })
        {
            let spill_state = self.spill_state.clone();
            let retained_bytes = Cell::new(0usize);
            let external_bytes = Cell::new(self.checked_external_granted_bytes_for_observer()?);
            let _usage_reconcile = ExactFinalizationUsageReconcile {
                state: spill_state.as_deref(),
                external_bytes: &external_bytes,
            };
            let observer = ExternalSortGrantObserver::new(
                &external_bytes,
                &retained_bytes,
                spill_state.as_deref(),
            );
            self.exact_finalize_consumed = true;
            let sorter = self
                .external_sort
                .take()
                .expect("the exact sorter shape was just qualified");
            let mut cursor = match sorter.into_exact_owned_disk_cursor(observer) {
                Ok(cursor) => cursor,
                Err(error) => {
                    return Err(error.into_operator_error());
                }
            };
            let emission_result = loop {
                let row = match cursor.next_owned_row() {
                    Ok(row) => row,
                    Err(error) => {
                        break Err(cursor.finish_stream_failure(
                            error,
                            "exact owned sort cleanup after cursor failure",
                        ));
                    }
                };
                let Some(row) = row else {
                    break Ok(());
                };
                let chunk = match try_accounted_chunk_from_sort_row(
                    row,
                    cursor.num_columns(),
                    cursor.output_observer(),
                ) {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        let (primary, grant) = error.into_parts();
                        let primary = Self::map_accounted_chunk_error(primary);
                        let recovery =
                            cursor
                                .reclaim_failed_output_grant(grant)
                                .err()
                                .map(|error| {
                                    (error, "exact owned sort failed-output authority recovery")
                                });
                        break Err(cursor.finish_operator_failure(
                            primary,
                            recovery,
                            "exact owned sort cleanup after output construction failure",
                        ));
                    }
                };
                let continuation = sink_permit.consume(chunk);
                let transfer_publication = cursor.release_transferred_retained();
                let continuation = match continuation {
                    Ok(continuation) => match transfer_publication {
                        Ok(()) => continuation,
                        Err(error) => {
                            break Err(cursor.finish_stream_failure(
                                error,
                                "exact owned sort cleanup after transfer publication failure",
                            ));
                        }
                    },
                    Err(primary) => {
                        break Err(cursor.finish_operator_failure(
                            primary,
                            transfer_publication.err().map(|error| {
                                (
                                    error,
                                    "exact owned sort transfer publication after sink failure",
                                )
                            }),
                            "exact owned sort cleanup after sink failure",
                        ));
                    }
                };
                if let Err(primary) = poll_cancellation(cancellation.as_ref()) {
                    break Err(cursor.finish_operator_failure(
                        primary,
                        None,
                        "exact owned sort cleanup after post-sink cancellation",
                    ));
                }
                if !continuation {
                    break cursor.finish_early_stop();
                }
            };
            drop(cursor);
            return emission_result;
        }

        // The resident-tail grant remains live until the cursor has emitted or
        // discarded every moved row. Qualified execution streams bounded
        // cursor chunks; compatibility spilling retains its collecting API.
        let spill_state = self.spill_state.clone();
        let row_bytes = Cell::new(self.buffer.granted_bytes());
        let external_bytes = Cell::new(self.checked_external_granted_bytes_for_observer()?);
        let _usage_after_rows = SpillUsageAfterMovedRows {
            state: spill_state.as_deref(),
            row_bytes: &row_bytes,
            external_bytes: &external_bytes,
        };
        poll_cancellation(cancellation.as_ref())?;
        let mut remaining = self.buffer.take_rows_with_grant(&row_bytes);
        let remaining_rows = remaining.take_rows();
        if self.memory_ctx.is_some() {
            let observer =
                ExternalSortGrantObserver::new(&external_bytes, &row_bytes, spill_state.as_deref());
            let mut cursor = match self
                .external_sort
                .as_mut()
                .expect("checked external sort presence")
                .merge_scalar_cursor_accounted_observing(
                    remaining_rows,
                    QUALIFIED_SORT_OUTPUT_CHUNK_ROWS,
                    observer,
                    Self::map_external_sort_error,
                ) {
                Ok(cursor) => cursor,
                Err(error) => {
                    let primary = error;
                    let release_result = remaining.release();
                    return match release_result {
                        Ok(()) => Err(primary),
                        Err(accounting) => Err(Self::with_row_release_context(
                            primary,
                            &accounting,
                            "cursor creation",
                        )),
                    };
                }
            };

            let emission_result = loop {
                let next = match cursor.next_chunk_accounted_observing() {
                    Ok(next) => next,
                    Err(primary) => break Err(primary),
                };
                let Some(chunk) = next else {
                    break Ok(());
                };
                let continuation = emit_sorted_rows_with_continuation(
                    chunk.rows(),
                    num_cols,
                    cancellation.as_ref(),
                    sink,
                );
                let continuation = match continuation {
                    Ok(continuation) => continuation,
                    Err(primary) => {
                        break Err(cursor.abort_with_primary(primary));
                    }
                };
                if let Err(primary) = poll_cancellation(cancellation.as_ref()) {
                    break Err(cursor.abort_with_primary(primary));
                }
                if !continuation {
                    break cursor.abort_accounted_observing();
                }
            };
            drop(cursor);
            let release_result = remaining.release();
            self.refresh_spill_usage();
            return match (emission_result, release_result) {
                (Ok(()), Ok(())) => Ok(()),
                (Err(primary), Ok(())) => Err(primary),
                (Ok(()), Err(accounting)) => Err(OperatorError::ResidentMemory(accounting)),
                (Err(primary), Err(accounting)) => Err(Self::with_row_release_context(
                    primary,
                    &accounting,
                    "streamed emission",
                )),
            };
        }

        let merge_result = self
            .external_sort
            .as_mut()
            .expect("checked external sort presence")
            .merge_all(remaining_rows)
            .map_err(OperatorError::from_spill_io_error);
        let sorted_rows = match merge_result {
            Ok(rows) => rows,
            Err(primary) => {
                let release_result = remaining.release();
                self.refresh_spill_usage();
                return match release_result {
                    Ok(()) => Err(primary),
                    Err(accounting) => Err(Self::with_row_release_context(
                        primary,
                        &accounting,
                        "merge",
                    )),
                };
            }
        };
        let emit_result = emit_sorted_rows(&sorted_rows, num_cols, cancellation.as_ref(), sink);
        drop(sorted_rows);
        let release_result = remaining.release();
        self.refresh_spill_usage();
        match (emit_result, release_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(primary), Ok(())) => Err(primary),
            (Ok(()), Err(accounting)) => Err(OperatorError::ResidentMemory(accounting)),
            (Err(primary), Err(accounting)) => Err(Self::with_row_release_context(
                primary,
                &accounting,
                "emission",
            )),
        }
    }

    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        // Sort is a breaker, chunk size doesn't matter much
        ChunkSizeHint::Default
    }

    fn name(&self) -> &'static str {
        "SpillableSortPush"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "spill")]
    use crate::execution::AccountedDataChunk;
    use crate::execution::QueryResourceContext;
    #[cfg(feature = "spill")]
    use crate::execution::pipeline::{
        AccountedSinkPermit, qualified_accounted_transport::QualifiedSink,
    };
    use crate::execution::sink::CollectorSink;
    use grafeo_common::types::LogicalType;
    use std::sync::Arc;
    #[cfg(feature = "spill")]
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};

    fn buffer_manager_with_exact_budget(
        budget: usize,
    ) -> std::sync::Arc<grafeo_common::memory::buffer::BufferManager> {
        let mut config = grafeo_common::memory::buffer::BufferManagerConfig::with_budget(budget);
        config.soft_limit_fraction = 1.0;
        config.evict_limit_fraction = 1.0;
        config.hard_limit_fraction = 1.0;
        grafeo_common::memory::buffer::BufferManager::new(config)
    }

    fn one_integer_row_charge() -> usize {
        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let mut sort =
            SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                .unwrap();
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[1]), &mut sink).unwrap();
        let charge = manager.allocated();
        assert!(
            charge > std::mem::size_of::<Value>() + std::mem::size_of::<Vec<Value>>(),
            "the grant must cover observed row/outer capacity and stable-sort scratch"
        );
        drop(sort);
        assert_eq!(manager.allocated(), 0);
        charge
    }

    #[test]
    #[cfg(feature = "spill")]
    fn observer_initialization_overflow_fails_telemetry_closed() {
        let state = super::super::spill_state::OperatorSpillState::new(
            "sort observer initialization overflow".to_string(),
        );
        state.set_usage(17);

        let error = fail_closed_external_observer_initialization(
            Err(MemoryGrantError::ArithmeticOverflow {
                current_bytes: usize::MAX,
                additional_bytes: 1,
            }),
            Some(&state),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            OperatorError::ResidentMemory(MemoryGrantError::ArithmeticOverflow { .. })
        ));
        assert_eq!(state.usage(), usize::MAX);
    }

    fn create_test_chunk(values: &[i64]) -> DataChunk {
        let v: Vec<Value> = values.iter().map(|&i| Value::Int64(i)).collect();
        let vector = ValueVector::from_values(&v);
        DataChunk::new(vec![vector])
    }

    #[cfg(feature = "spill")]
    fn create_value_chunk(values: &[Value]) -> DataChunk {
        DataChunk::new(vec![ValueVector::from_values(values)])
    }

    #[cfg(feature = "spill")]
    fn create_rows_chunk(rows: &[Vec<Value>]) -> DataChunk {
        let columns = rows.first().map_or(0, Vec::len);
        assert!(rows.iter().all(|row| row.len() == columns));
        DataChunk::new(
            (0..columns)
                .map(|column| {
                    ValueVector::from_values(
                        &rows
                            .iter()
                            .map(|row| row[column].clone())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect(),
        )
    }

    #[cfg(feature = "spill")]
    fn collected_first_column(sink: CollectorSink) -> Vec<Value> {
        let mut values = Vec::new();
        for chunk in sink.into_chunks() {
            let column = chunk.column(0).expect("test sort has one column");
            for row in 0..chunk.len() {
                values.push(column.get_value(row).expect("test row has one value"));
            }
        }
        values
    }

    #[cfg(feature = "spill")]
    #[derive(Default)]
    struct RetainingAccountedSortSink {
        accounted: Vec<AccountedDataChunk>,
        plain: Vec<DataChunk>,
        stop_after_first: bool,
        fail_after_first: bool,
        cancel_after_first: Option<crate::execution::QueryCancellationHandle>,
    }

    #[cfg(feature = "spill")]
    impl Sink for RetainingAccountedSortSink {
        fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
            self.plain.push(chunk);
            Ok(true)
        }

        fn consume_accounted(&mut self, chunk: AccountedDataChunk) -> Result<bool, OperatorError> {
            <Self as QualifiedSink>::consume_accounted_qualified(self, chunk)
        }

        fn __accounted_sink_permit(&mut self) -> Option<AccountedSinkPermit<'_>> {
            Some(AccountedSinkPermit::new(self))
        }

        fn finalize(&mut self) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "RetainingAccountedSortSink"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
            self
        }
    }

    #[cfg(feature = "spill")]
    impl QualifiedSink for RetainingAccountedSortSink {
        fn consume_accounted_qualified(
            &mut self,
            chunk: AccountedDataChunk,
        ) -> Result<bool, OperatorError> {
            self.accounted.push(chunk);
            if let Some(cancellation) = &self.cancel_after_first {
                cancellation.cancel();
            }
            if self.fail_after_first {
                return Err(OperatorError::ResidentContainerInvariant {
                    container: "test exact accounted sink",
                    message: "injected downstream failure",
                });
            }
            Ok(!self.stop_after_first)
        }
    }

    #[cfg(feature = "spill")]
    fn accounted_rows(sink: &RetainingAccountedSortSink) -> Vec<Vec<Value>> {
        sink.accounted
            .iter()
            .map(|accounted| {
                let chunk = accounted.chunk();
                (0..chunk.column_count())
                    .map(|column| {
                        chunk
                            .column(column)
                            .expect("accounted test column exists")
                            .get_value(0)
                            .expect("accounted test row exists")
                    })
                    .collect()
            })
            .collect()
    }

    #[cfg(feature = "spill")]
    struct PanickingQualificationIo {
        qualification_queries: Arc<AtomicUsize>,
        panic_enabled: Arc<AtomicBool>,
    }

    #[cfg(feature = "spill")]
    impl crate::execution::spill::SpillIo for PanickingQualificationIo {
        fn check(
            &self,
            _operation: crate::execution::spill::SpillIoOperation,
        ) -> std::io::Result<()> {
            Ok(())
        }

        fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
            self.qualification_queries
                .fetch_add(1, AtomicOrdering::Relaxed);
            assert!(
                !self.panic_enabled.load(AtomicOrdering::Relaxed),
                "exact reader qualification trap"
            );
            Some(0)
        }
    }

    #[cfg(feature = "spill")]
    fn fixture_with_panicking_qualification(
        directory: &std::path::Path,
        qualification_queries: Arc<AtomicUsize>,
        panic_enabled: Arc<AtomicBool>,
    ) -> crate::execution::spill::BorrowedSpillFixture {
        crate::execution::spill::BorrowedSpillFixture::new(directory)
            .provider(
                Arc::new(crate::execution::spill::CleartextSpillRecordProvider),
                crate::execution::spill::SpillFrameLimits::format_max(),
            )
            .io(Arc::new(PanickingQualificationIo {
                qualification_queries,
                panic_enabled,
            }))
    }

    #[cfg(feature = "spill")]
    fn create_stable_sort_chunk(start: usize, len: usize) -> DataChunk {
        let keys = (start..start + len)
            .map(|ordinal| Value::Int64(i64::try_from(ordinal % 4).unwrap()))
            .collect::<Vec<_>>();
        let ordinals = (start..start + len)
            .map(|ordinal| Value::Int64(i64::try_from(ordinal).unwrap()))
            .collect::<Vec<_>>();
        DataChunk::new(vec![
            ValueVector::from_values(&keys),
            ValueVector::from_values(&ordinals),
        ])
    }

    #[cfg(feature = "spill")]
    fn stable_sort_prefix_charge(rows: usize) -> usize {
        let manager = buffer_manager_with_exact_budget(64 * 1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let mut sort =
            SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                .unwrap();
        let mut discard = CollectorSink::new();
        sort.push(create_stable_sort_chunk(0, rows), &mut discard)
            .unwrap();
        let charge = manager.allocated();
        drop(sort);
        assert_eq!(manager.allocated(), 0);
        charge
    }

    #[cfg(feature = "spill")]
    fn hard_denial_sort_budget(prefix_rows: usize) -> (usize, usize) {
        let prefix_charge = stable_sort_prefix_charge(prefix_rows);
        // The 25% headroom admits the fixed 64 KiB writer plus codec/catalog
        // workspaces, but not the next outer-row replacement peak.
        let budget = prefix_charge.checked_add(prefix_charge / 4).unwrap();
        (prefix_charge, budget)
    }

    #[test]
    fn test_sort_ascending() {
        let resources =
            QueryResourceContext::new(buffer_manager_with_exact_budget(1024 * 1024)).unwrap();
        let mut sort =
            SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                .unwrap();
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[3, 1, 4, 1, 5, 9, 2, 6]), &mut sink)
            .unwrap();
        sort.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);

        let col = chunks[0].column(0).unwrap();
        assert_eq!(col.get_value(0), Some(Value::Int64(1)));
        assert_eq!(col.get_value(1), Some(Value::Int64(1)));
        assert_eq!(col.get_value(2), Some(Value::Int64(2)));
        assert_eq!(col.get_value(3), Some(Value::Int64(3)));
    }

    #[test]
    fn test_sort_descending() {
        let resources =
            QueryResourceContext::new(buffer_manager_with_exact_budget(1024 * 1024)).unwrap();
        let mut sort =
            SortPushOperator::with_resource_context(vec![SortKey::descending(0)], resources)
                .unwrap();
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[3, 1, 4, 1, 5]), &mut sink)
            .unwrap();
        sort.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        let col = chunks[0].column(0).unwrap();
        assert_eq!(col.get_value(0), Some(Value::Int64(5)));
        assert_eq!(col.get_value(1), Some(Value::Int64(4)));
        assert_eq!(col.get_value(2), Some(Value::Int64(3)));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn final_null_placement_is_direction_independent_and_spilling_matches_memory() {
        use tempfile::TempDir;

        let input = [
            Value::Int64(2),
            Value::Null,
            Value::Int64(1),
            Value::Null,
            Value::Int64(3),
        ];
        let cases = [
            (
                SortKey {
                    column: 0,
                    direction: SortDirection::Ascending,
                    null_order: NullOrder::First,
                },
                vec![
                    Value::Null,
                    Value::Null,
                    Value::Int64(1),
                    Value::Int64(2),
                    Value::Int64(3),
                ],
            ),
            (
                SortKey {
                    column: 0,
                    direction: SortDirection::Ascending,
                    null_order: NullOrder::Last,
                },
                vec![
                    Value::Int64(1),
                    Value::Int64(2),
                    Value::Int64(3),
                    Value::Null,
                    Value::Null,
                ],
            ),
            (
                SortKey {
                    column: 0,
                    direction: SortDirection::Descending,
                    null_order: NullOrder::First,
                },
                vec![
                    Value::Null,
                    Value::Null,
                    Value::Int64(3),
                    Value::Int64(2),
                    Value::Int64(1),
                ],
            ),
            (
                SortKey {
                    column: 0,
                    direction: SortDirection::Descending,
                    null_order: NullOrder::Last,
                },
                vec![
                    Value::Int64(3),
                    Value::Int64(2),
                    Value::Int64(1),
                    Value::Null,
                    Value::Null,
                ],
            ),
        ];

        for (key, expected) in cases {
            let resources =
                QueryResourceContext::new(buffer_manager_with_exact_budget(1024 * 1024)).unwrap();
            let mut memory_sort =
                SortPushOperator::with_resource_context(vec![key.clone()], resources).unwrap();
            let mut memory_sink = CollectorSink::new();
            memory_sort
                .push(create_value_chunk(&input), &mut memory_sink)
                .unwrap();
            memory_sort.finalize(&mut memory_sink).unwrap();
            assert_eq!(collected_first_column(memory_sink), expected);

            let directory = TempDir::new().unwrap();
            let manager = Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .build()
                    .unwrap(),
            );
            let mut spilled_sort = SpillableSortPushOperator::with_spilling(vec![key], manager, 2);
            let mut spilled_sink = CollectorSink::new();
            spilled_sort
                .push(create_value_chunk(&input), &mut spilled_sink)
                .unwrap();
            spilled_sort.finalize(&mut spilled_sink).unwrap();
            assert_eq!(collected_first_column(spilled_sink), expected);
        }
    }

    fn scalar_provenance_chunk(kind: u8, key: i64, ordinal: i64) -> DataChunk {
        let entity_type = match kind {
            1 => LogicalType::List(Box::new(LogicalType::Edge)),
            2 => LogicalType::Edge,
            3 => LogicalType::Node,
            _ => LogicalType::Any,
        };
        let types = [
            LogicalType::Int64,
            entity_type.clone(),
            LogicalType::Any,
            LogicalType::Int64,
            entity_type,
            LogicalType::Any,
        ];
        let mut columns: Vec<_> = types
            .into_iter()
            .map(|ty| ValueVector::with_capacity(ty, 1))
            .collect();
        columns[0].push_value(Value::Int64(key));
        columns[3].push_value(Value::Int64(ordinal));
        for index in [1, 4] {
            match kind {
                1 => columns[index].push_value(Value::List(vec![Value::Int64(7)].into())),
                2 => columns[index].push_edge_id(grafeo_common::types::EdgeId(7)),
                3 => columns[index].push_node_id(grafeo_common::types::NodeId(7)),
                _ => columns[index].push_value(Value::Int64(7)),
            }
        }
        for index in [2, 5] {
            columns[index].push_value(Value::Bytes(vec![0xff, 0x03].into()));
        }
        DataChunk::new(columns)
    }

    fn scalar_provenance_inputs() -> impl Iterator<Item = DataChunk> {
        [(0, 2), (3, 0), (2, 1), (1, 0), (0, 0), (3, 2)]
            .into_iter()
            .enumerate()
            .map(|(ordinal, (kind, key))| {
                scalar_provenance_chunk(kind, key, i64::try_from(ordinal).unwrap())
            })
    }

    fn assert_scalar_provenance_rows<'a>(chunks: impl IntoIterator<Item = &'a DataChunk>) {
        let mut actual = Vec::new();
        for chunk in chunks {
            assert_eq!(chunk.column_count(), 6, "private trailer must not escape");
            for row in chunk.selected_indices() {
                let ordinal = chunk.column(3).unwrap().get_value(row).unwrap();
                for index in [1, 4] {
                    let column = chunk.column(index).unwrap();
                    assert_eq!(column.data_type(), chunk.column(1).unwrap().data_type());
                    match column.data_type() {
                        LogicalType::Edge => {
                            assert_eq!(
                                column.get_edge_id(row),
                                Some(grafeo_common::types::EdgeId(7))
                            );
                            assert_eq!(
                                column.get_node_id(row),
                                None,
                                "typed edge must not resolve colliding node"
                            );
                        }
                        LogicalType::Node => {
                            assert_eq!(
                                column.get_node_id(row),
                                Some(grafeo_common::types::NodeId(7))
                            );
                            assert_eq!(
                                column.get_edge_id(row),
                                None,
                                "typed node must not resolve colliding edge"
                            );
                        }
                        LogicalType::List(_) => assert_eq!(
                            column.get_value(row),
                            Some(Value::List(vec![Value::Int64(7)].into()))
                        ),
                        _ => assert_eq!(column.get_value(row), Some(Value::Int64(7))),
                    }
                }
                assert_eq!(
                    chunk.column(5).unwrap().get_value(row),
                    Some(Value::Bytes(vec![0xff, 0x03].into()))
                );
                actual.push((ordinal, chunk.column(1).unwrap().data_type().clone()));
            }
        }
        assert_eq!(
            actual,
            vec![
                (Value::Int64(1), LogicalType::Node),
                (
                    Value::Int64(3),
                    LogicalType::List(Box::new(LogicalType::Edge))
                ),
                (Value::Int64(4), LogicalType::Any),
                (Value::Int64(2), LogicalType::Edge),
                (Value::Int64(0), LogicalType::Any),
                (Value::Int64(5), LogicalType::Node),
            ]
        );
    }

    #[test]
    fn scalar_provenance_resident_sort_preserves_mixed_rows_and_second_tag_byte() {
        let resources =
            QueryResourceContext::new(buffer_manager_with_exact_budget(1024 * 1024)).unwrap();
        let mut sort =
            SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                .unwrap();
        let mut sink = CollectorSink::new();
        for chunk in scalar_provenance_inputs() {
            sort.push(chunk, &mut sink).unwrap();
        }
        sort.finalize(&mut sink).unwrap();
        let chunks = sink.into_chunks();
        assert_scalar_provenance_rows(chunks.iter());
    }

    #[test]
    #[cfg(feature = "spill")]
    fn scalar_provenance_forced_spill_preserves_mixed_rows() {
        let directory = tempfile::TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .build()
                .unwrap(),
        );
        let mut sort = SpillableSortPushOperator::with_spilling(
            vec![SortKey::ascending(0)],
            manager.clone(),
            usize::MAX,
        );
        let mut sink = CollectorSink::new();
        for chunk in scalar_provenance_inputs() {
            sort.push(chunk, &mut sink).unwrap();
            sort.spill_current_buffer(manager.clone()).unwrap();
        }
        assert!(manager.active_file_count() > 0);
        sort.finalize(&mut sink).unwrap();
        let chunks = sink.into_chunks();
        assert_scalar_provenance_rows(chunks.iter());
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn scalar_provenance_exact_spill_preserves_types_accessors_and_grants() {
        let manager = buffer_manager_with_exact_budget(64 * 1024 * 1024);
        let directory = tempfile::TempDir::new().unwrap();
        let (resources, spills) =
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .build_operator_resources(
                    manager.clone(),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut ignored = CollectorSink::new();
        for chunk in scalar_provenance_inputs() {
            sort.push(chunk, &mut ignored).unwrap();
            sort.spill_current_buffer(spills.clone()).unwrap();
        }
        let mut sink = RetainingAccountedSortSink::default();
        sort.finalize(&mut sink).unwrap();
        assert!(sink.plain.is_empty());
        assert_scalar_provenance_rows(sink.accounted.iter().map(|chunk| chunk.chunk()));
        assert_eq!(spills.active_file_count(), 0);
        drop(sort);
        assert!(manager.allocated() > 0);
        drop(sink);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn resident_sort_preserves_mixed_row_edge_list_provenance() {
        let resources =
            QueryResourceContext::new(buffer_manager_with_exact_budget(1024 * 1024)).unwrap();
        let mut sort =
            SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                .unwrap();
        let mut sink = CollectorSink::new();
        for (typed, rows) in [
            (true, [(2, 0), (0, 1)]),
            (false, [(1, 2), (0, 3)]),
            (true, [(0, 4), (3, 5)]),
        ] {
            sort.push(edge_provenance_chunk(typed, &rows), &mut sink)
                .unwrap();
        }
        sort.finalize(&mut sink).unwrap();
        let chunks = sink.into_chunks();
        assert_edge_provenance_rows(chunks.iter());
    }

    fn edge_provenance_chunk(typed: bool, rows: &[(i64, i64)]) -> DataChunk {
        let list_type = if typed {
            LogicalType::List(Box::new(LogicalType::Edge))
        } else {
            LogicalType::Any
        };
        let mut columns = vec![
            ValueVector::with_capacity(LogicalType::Int64, rows.len()),
            ValueVector::with_capacity(list_type, rows.len()),
            ValueVector::with_capacity(LogicalType::Int64, rows.len()),
            ValueVector::with_capacity(LogicalType::Any, rows.len()),
        ];
        for &(key, ordinal) in rows {
            columns[0].push_value(Value::Int64(key));
            // Identical scalar values deliberately carry different provenance.
            columns[1].push_value(Value::List(vec![Value::Int64(7), Value::Int64(9)].into()));
            columns[2].push_value(Value::Int64(ordinal));
            // A real trailing Bytes value must not be interpreted as metadata.
            columns[3].push_value(Value::Bytes(vec![0b0000_0010].into()));
        }
        DataChunk::new(columns)
    }

    fn assert_edge_provenance_rows<'a>(chunks: impl IntoIterator<Item = &'a DataChunk>) {
        let mut actual = Vec::new();
        for chunk in chunks {
            assert_eq!(chunk.column_count(), 4, "private metadata must not escape");
            let typed = matches!(chunk.column(1).unwrap().data_type(), LogicalType::List(element) if element.as_ref() == &LogicalType::Edge);
            for row in chunk.selected_indices() {
                assert_eq!(
                    chunk.column(1).unwrap().get_value(row),
                    Some(Value::List(vec![Value::Int64(7), Value::Int64(9)].into()))
                );
                assert_eq!(
                    chunk.column(3).unwrap().get_value(row),
                    Some(Value::Bytes(vec![0b0000_0010].into()))
                );
                actual.push((chunk.column(2).unwrap().get_value(row).unwrap(), typed));
            }
        }
        let expected = [
            (1, true),
            (3, false),
            (4, true),
            (2, false),
            (0, true),
            (5, true),
        ]
        .map(|(ordinal, typed)| (Value::Int64(ordinal), typed));
        assert_eq!(
            actual, expected,
            "stable row ordering and each row's provenance must survive sorting"
        );
    }

    #[test]
    fn sort_provenance_resident_mask_peak_is_admitted_before_row_publication() {
        let mut charges = Vec::new();
        for typed in [false, true] {
            let manager = buffer_manager_with_exact_budget(1024 * 1024);
            let resources = crate::execution::QueryResourceContext::new(manager.clone()).unwrap();
            let mut sort =
                SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                    .unwrap();
            let mut sink = CollectorSink::new();
            sort.push(edge_provenance_chunk(typed, &[(0, 0)]), &mut sink)
                .unwrap();
            charges.push(sort.buffer.granted_bytes());
            assert_eq!(
                sort.buffer.granted_bytes(),
                sort.buffer.observed_charge().unwrap()
            );
            assert_eq!(manager.allocated(), sort.buffer.granted_bytes());
            drop(sort);
            assert_eq!(manager.allocated(), 0);
        }
        assert_eq!(
            charges[1] - charges[0],
            std::mem::size_of::<Value>() + decoded_bytes_payload_retained_bytes(1).unwrap()
        );
        // The final retained row would fit, but creating its Arc while the
        // temporary mask Vec exists needs one additional byte of authority.
        let manager = buffer_manager_with_exact_budget(charges[1]);
        let resources = crate::execution::QueryResourceContext::new(manager.clone()).unwrap();
        let mut sort =
            SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                .unwrap();
        let mut sink = CollectorSink::new();
        assert!(matches!(
            sort.push(edge_provenance_chunk(true, &[(0, 0)]), &mut sink),
            Err(OperatorError::ResidentMemory(_))
        ));
        assert!(sort.buffer.is_empty());
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn forced_spill_sort_preserves_mixed_row_edge_list_provenance() {
        let directory = tempfile::TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .build()
                .unwrap(),
        );
        let mut sort = SpillableSortPushOperator::with_spilling(
            vec![SortKey::ascending(0)],
            manager.clone(),
            usize::MAX,
        );
        let mut sink = CollectorSink::new();
        for (typed, rows) in [
            (true, [(2, 0), (0, 1)]),
            (false, [(1, 2), (0, 3)]),
            (true, [(0, 4), (3, 5)]),
        ] {
            sort.push(edge_provenance_chunk(typed, &rows), &mut sink)
                .unwrap();
            sort.spill_current_buffer(manager.clone()).unwrap();
        }
        assert!(manager.active_file_count() > 0);
        sort.finalize(&mut sink).unwrap();
        let chunks = sink.into_chunks();
        assert_edge_provenance_rows(chunks.iter());
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn exact_spill_sort_preserves_mixed_row_edge_list_provenance_and_grants() {
        let manager = buffer_manager_with_exact_budget(64 * 1024 * 1024);
        let directory = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .build_operator_resources(
                    manager.clone(),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut ignored = CollectorSink::new();
        for (typed, rows) in [
            (true, [(2, 0), (0, 1)]),
            (false, [(1, 2), (0, 3)]),
            (true, [(0, 4), (3, 5)]),
        ] {
            sort.push(edge_provenance_chunk(typed, &rows), &mut ignored)
                .unwrap();
            sort.spill_current_buffer(spill_manager.clone()).unwrap();
        }
        assert!(spill_manager.active_file_count() > 0);
        let mut sink = RetainingAccountedSortSink::default();
        sort.finalize(&mut sink).unwrap();
        assert!(
            sink.plain.is_empty(),
            "must exercise exact accounted output"
        );
        assert!(!sink.accounted.is_empty());
        assert_edge_provenance_rows(sink.accounted.iter().map(|chunk| chunk.chunk()));
        assert_eq!(spill_manager.active_file_count(), 0);
        drop(sort);
        assert!(
            manager.allocated() > 0,
            "retained sink chunks must retain their grant"
        );
        drop(sink);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn test_sort_multiple_chunks() {
        let resources =
            QueryResourceContext::new(buffer_manager_with_exact_budget(1024 * 1024)).unwrap();
        let mut sort =
            SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                .unwrap();
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[5, 3, 1]), &mut sink).unwrap();
        sort.push(create_test_chunk(&[4, 2, 6]), &mut sink).unwrap();
        sort.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks[0].len(), 6);

        let col = chunks[0].column(0).unwrap();
        assert_eq!(col.get_value(0), Some(Value::Int64(1)));
        assert_eq!(col.get_value(5), Some(Value::Int64(6)));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn exact_owned_sort_streams_stably_retains_sink_charge_and_deletes_files() {
        let manager = buffer_manager_with_exact_budget(64 * 1024 * 1024);
        let directory = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut ignored = CollectorSink::new();

        sort.push(create_stable_sort_chunk(0, 4), &mut ignored)
            .unwrap();
        sort.spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap();
        sort.push(create_stable_sort_chunk(4, 4), &mut ignored)
            .unwrap();
        sort.spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap();
        assert!(sort.buffer.is_empty());
        assert_eq!(spill_manager.active_file_count(), 2);

        let mut sink = RetainingAccountedSortSink::default();
        sort.finalize(&mut sink).unwrap();

        assert!(
            sort.external_sort.is_none(),
            "exact dispatch consumes the sorter"
        );
        assert!(sink.plain.is_empty());
        assert_eq!(
            accounted_rows(&sink),
            vec![
                vec![Value::Int64(0), Value::Int64(0)],
                vec![Value::Int64(0), Value::Int64(4)],
                vec![Value::Int64(1), Value::Int64(1)],
                vec![Value::Int64(1), Value::Int64(5)],
                vec![Value::Int64(2), Value::Int64(2)],
                vec![Value::Int64(2), Value::Int64(6)],
                vec![Value::Int64(3), Value::Int64(3)],
                vec![Value::Int64(3), Value::Int64(7)],
            ]
        );
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        assert!(matches!(
            sort.finalize(&mut sink),
            Err(OperatorError::ResidentContainerInvariant {
                container: "exact owned sort finalization",
                ..
            })
        ));

        let retained = sink
            .accounted
            .iter()
            .map(AccountedDataChunk::granted_bytes)
            .sum::<usize>();
        assert!(retained > 0);
        assert_eq!(manager.allocated(), retained);
        drop(sort);
        assert_eq!(manager.allocated(), retained);
        drop(sink);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn exact_finalize_drop_failure_keeps_usage_fail_closed_and_fences_reuse() {
        let manager = buffer_manager_with_exact_budget(64 * 1024 * 1024);
        let directory = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut ignored = CollectorSink::new();
        sort.push(create_test_chunk(&[2, 1]), &mut ignored).unwrap();
        sort.spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap();
        sort.external_sort
            .as_mut()
            .unwrap()
            .inject_persistent_exact_workspace_release_failure(
                MemoryGrantError::AccountingPoisoned {
                    account: "persistent exact finalization release",
                },
            );
        let spill_state = Arc::clone(sort.spill_state.as_ref().unwrap());
        let mut sink = RetainingAccountedSortSink::default();

        let error = sort.finalize(&mut sink).unwrap_err();

        assert!(matches!(
            &error,
            OperatorError::ClassifiedAccountedFailure {
                classification: AccountedFailureClassification::ResidentMemory(
                    MemoryGrantError::AccountingPoisoned {
                        account: "persistent exact finalization release"
                    }
                ),
                ..
            }
        ));
        assert!(sort.exact_finalize_consumed);
        assert!(sort.external_sort.is_none());
        assert_eq!(spill_state.usage(), usize::MAX);
        assert_eq!(spill_manager.active_file_count(), 0);
        drop(error);
        drop(sink);
        assert!(manager.allocated() > 0);

        let reuse = sort
            .push(create_test_chunk(&[3]), &mut ignored)
            .unwrap_err();
        assert!(matches!(
            reuse,
            OperatorError::ResidentContainerInvariant {
                container: "exact owned sort finalization",
                ..
            }
        ));
        assert_eq!(spill_state.usage(), usize::MAX);
        drop(sort);
        assert!(manager.allocated() > 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn exact_chunk_construction_unwind_keeps_usage_fail_closed_and_fences_reuse() {
        let manager = buffer_manager_with_exact_budget(64 * 1024 * 1024);
        let directory = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut ignored = CollectorSink::new();
        sort.push(create_test_chunk(&[2, 1]), &mut ignored).unwrap();
        sort.spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap();
        let spill_state = Arc::clone(sort.spill_state.as_ref().unwrap());
        let mut sink = RetainingAccountedSortSink::default();
        let _panic_guard =
            crate::execution::accounted_chunk::inject_sort_chunk_construction_panic_once();

        let panic =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sort.finalize(&mut sink)))
                .unwrap_err();

        let message = panic
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str));
        assert_eq!(
            message,
            Some("deterministic accounted sort-chunk construction panic")
        );
        assert!(sort.exact_finalize_consumed);
        assert!(sort.external_sort.is_none());
        assert!(sink.accounted.is_empty());
        assert_eq!(spill_state.usage(), usize::MAX);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert!(manager.allocated() > 0);

        let reuse = sort
            .push(create_test_chunk(&[3]), &mut ignored)
            .unwrap_err();
        assert!(matches!(
            reuse,
            OperatorError::ResidentContainerInvariant {
                container: "exact owned sort finalization",
                ..
            }
        ));
        assert_eq!(spill_state.usage(), usize::MAX);
        let finalize_reuse = sort.finalize(&mut sink).unwrap_err();
        assert!(matches!(
            finalize_reuse,
            OperatorError::ResidentContainerInvariant {
                container: "exact owned sort finalization",
                ..
            }
        ));
        assert_eq!(spill_state.usage(), usize::MAX);
        drop(sort);
        assert!(manager.allocated() > 0);
        assert_eq!(spill_state.usage(), usize::MAX);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn exact_owned_sort_preserves_descending_and_explicit_null_last() {
        let manager = buffer_manager_with_exact_budget(64 * 1024 * 1024);
        let directory = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .build_operator_resources(
                    manager,
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey {
                column: 0,
                direction: SortDirection::Descending,
                null_order: NullOrder::Last,
            }],
            resources,
        )
        .unwrap();
        let mut ignored = CollectorSink::new();

        sort.push(
            create_value_chunk(&[Value::Null, Value::Int64(1), Value::Int64(3)]),
            &mut ignored,
        )
        .unwrap();
        sort.spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap();
        sort.push(
            create_value_chunk(&[Value::Int64(2), Value::Null]),
            &mut ignored,
        )
        .unwrap();
        sort.spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap();

        let mut sink = RetainingAccountedSortSink::default();
        sort.finalize(&mut sink).unwrap();
        assert_eq!(
            accounted_rows(&sink),
            vec![
                vec![Value::Int64(3)],
                vec![Value::Int64(2)],
                vec![Value::Int64(1)],
                vec![Value::Null],
                vec![Value::Null],
            ]
        );
        assert!(sink.plain.is_empty());
        assert_eq!(spill_manager.active_file_count(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn exact_owned_sort_matches_memory_for_multikey_nan_null_string_and_nested_payloads() {
        let rows = vec![
            vec![
                Value::from("a"),
                Value::Float64(1.0),
                Value::List(Arc::from([Value::from("nested-0"), Value::Int64(0)])),
                Value::Int64(0),
            ],
            vec![
                Value::Null,
                Value::Float64(2.0),
                Value::List(Arc::from([Value::from("nested-1")])),
                Value::Int64(1),
            ],
            vec![
                Value::from("b"),
                Value::Float64(f64::NAN),
                Value::List(Arc::from([Value::from("nested-2")])),
                Value::Int64(2),
            ],
            vec![
                Value::from("a"),
                Value::Float64(3.0),
                Value::List(Arc::from([Value::from("nested-3")])),
                Value::Int64(3),
            ],
            vec![
                Value::from("b"),
                Value::Null,
                Value::List(Arc::from([Value::from("nested-4")])),
                Value::Int64(4),
            ],
            vec![
                Value::from("a"),
                Value::Float64(1.0),
                Value::List(Arc::from([Value::from("nested-5"), Value::Int64(5)])),
                Value::Int64(5),
            ],
        ];
        let keys = vec![
            SortKey {
                column: 0,
                direction: SortDirection::Ascending,
                null_order: NullOrder::Last,
            },
            SortKey {
                column: 1,
                direction: SortDirection::Descending,
                null_order: NullOrder::Last,
            },
        ];

        let resources =
            QueryResourceContext::new(buffer_manager_with_exact_budget(1024 * 1024)).unwrap();
        let mut memory_sort =
            SortPushOperator::with_resource_context(keys.clone(), resources).unwrap();
        let mut memory_sink = CollectorSink::new();
        memory_sort
            .push(create_rows_chunk(&rows), &mut memory_sink)
            .unwrap();
        memory_sort.finalize(&mut memory_sink).unwrap();
        let expected_ids = memory_sink
            .into_chunks()
            .iter()
            .flat_map(|chunk| {
                (0..chunk.len()).map(|row| chunk.column(3).unwrap().get_value(row).unwrap())
            })
            .collect::<Vec<_>>();

        let manager = buffer_manager_with_exact_budget(64 * 1024 * 1024);
        let directory = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut exact_sort =
            SpillableSortPushOperator::with_resource_context(keys, resources).unwrap();
        let mut ignored = CollectorSink::new();
        for pair in rows.chunks(2) {
            exact_sort
                .push(create_rows_chunk(pair), &mut ignored)
                .unwrap();
            exact_sort
                .spill_current_buffer(Arc::clone(&spill_manager))
                .unwrap();
        }
        let mut exact_sink = RetainingAccountedSortSink::default();
        exact_sort.finalize(&mut exact_sink).unwrap();
        let exact_rows = accounted_rows(&exact_sink);
        assert_eq!(
            exact_rows
                .iter()
                .map(|row| row[3].clone())
                .collect::<Vec<_>>(),
            expected_ids
        );
        assert!(
            exact_rows
                .iter()
                .all(|row| matches!(row[2], Value::List(_)))
        );
        assert!(exact_sink.plain.is_empty());
        assert_eq!(spill_manager.active_file_count(), 0);
        drop(exact_sort);
        drop(exact_sink);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn exact_owned_sort_falls_back_without_permit_or_with_resident_tail() {
        let manager = buffer_manager_with_exact_budget(64 * 1024 * 1024);
        let directory = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut no_permit = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut ignored = CollectorSink::new();
        no_permit
            .push(create_test_chunk(&[2, 1]), &mut ignored)
            .unwrap();
        no_permit
            .spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap();
        let mut plain_sink = CollectorSink::new();
        no_permit.finalize(&mut plain_sink).unwrap();
        assert!(no_permit.external_sort.is_some());
        assert_eq!(
            collected_first_column(plain_sink),
            vec![Value::Int64(1), Value::Int64(2)]
        );
        drop(no_permit);

        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .build_operator_resources(
                    manager,
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut resident_tail = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        resident_tail
            .push(create_test_chunk(&[3, 1]), &mut ignored)
            .unwrap();
        resident_tail
            .spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap();
        resident_tail
            .push(create_test_chunk(&[2]), &mut ignored)
            .unwrap();
        let mut qualified_sink = RetainingAccountedSortSink::default();
        resident_tail.finalize(&mut qualified_sink).unwrap();
        assert!(qualified_sink.accounted.is_empty());
        assert_eq!(
            qualified_sink
                .plain
                .iter()
                .flat_map(|chunk| (0..chunk.len()).map(|row| chunk
                    .column(0)
                    .unwrap()
                    .get_value(row)
                    .unwrap()))
                .collect::<Vec<_>>(),
            vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)]
        );
        assert!(resident_tail.external_sort.is_some());
    }

    #[test]
    #[cfg(feature = "spill")]
    fn compatibility_fallback_never_probes_optional_exact_reader_capability() {
        for (run_count, qualified_sink) in [(1usize, false), (17, true)] {
            let memory = buffer_manager_with_exact_budget(64 * 1024 * 1024);
            let directory = tempfile::TempDir::new().unwrap();
            let queries = Arc::new(AtomicUsize::new(0));
            let panic_enabled = Arc::new(AtomicBool::new(true));
            let (resources, spill_manager) = fixture_with_panicking_qualification(
                directory.path(),
                Arc::clone(&queries),
                panic_enabled,
            )
            .build_operator_resources(
                Arc::clone(&memory),
                crate::execution::QueryExecutionControl::new().token(),
            )
            .unwrap();
            let mut sort = SpillableSortPushOperator::with_resource_context(
                vec![SortKey::ascending(0)],
                resources,
            )
            .unwrap();
            let mut ignored = CollectorSink::new();
            for value in (0..run_count).rev() {
                sort.push(
                    create_test_chunk(&[i64::try_from(value).unwrap()]),
                    &mut ignored,
                )
                .unwrap();
                sort.spill_current_buffer(Arc::clone(&spill_manager))
                    .unwrap();
            }

            if qualified_sink {
                let mut sink = RetainingAccountedSortSink::default();
                sort.finalize(&mut sink).unwrap();
                assert!(sink.accounted.is_empty());
                assert_eq!(
                    sink.plain.iter().map(DataChunk::len).sum::<usize>(),
                    run_count
                );
            } else {
                let mut sink = CollectorSink::new();
                sort.finalize(&mut sink).unwrap();
                assert_eq!(
                    sink.into_chunks().iter().map(DataChunk::len).sum::<usize>(),
                    run_count
                );
            }
            assert_eq!(queries.load(AtomicOrdering::Relaxed), 0);
            assert!(sort.external_sort.is_some());
            assert_eq!(spill_manager.active_file_count(), 0);
            drop(sort);
            assert_eq!(memory.allocated(), 0);
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    fn qualification_callback_panic_leaves_sorter_unconsumed_for_compatibility_retry() {
        let memory = buffer_manager_with_exact_budget(64 * 1024 * 1024);
        let directory = tempfile::TempDir::new().unwrap();
        let queries = Arc::new(AtomicUsize::new(0));
        let panic_enabled = Arc::new(AtomicBool::new(true));
        let (resources, spill_manager) = fixture_with_panicking_qualification(
            directory.path(),
            Arc::clone(&queries),
            Arc::clone(&panic_enabled),
        )
        .build_operator_resources(
            Arc::clone(&memory),
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut ignored = CollectorSink::new();
        sort.push(create_test_chunk(&[2, 1]), &mut ignored).unwrap();
        sort.spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap();
        let mut qualified = RetainingAccountedSortSink::default();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = sort.finalize(&mut qualified);
        }))
        .unwrap_err();

        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"exact reader qualification trap")
        );
        assert_eq!(queries.load(AtomicOrdering::Relaxed), 1);
        assert!(!sort.exact_finalize_consumed);
        assert!(sort.external_sort.is_some());
        assert_eq!(spill_manager.active_file_count(), 1);
        assert!(qualified.accounted.is_empty());
        panic_enabled.store(false, AtomicOrdering::Relaxed);
        let mut plain = CollectorSink::new();
        sort.finalize(&mut plain).unwrap();
        assert_eq!(
            collected_first_column(plain),
            [Value::Int64(1), Value::Int64(2)]
        );
        assert_eq!(queries.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(spill_manager.active_file_count(), 0);
        drop(sort);
        assert_eq!(memory.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn exact_owned_sort_dispatches_at_fan_in_boundaries_one_sixteen_seventeen() {
        for (run_count, expect_exact) in [(1usize, true), (16, true), (17, false)] {
            let manager = buffer_manager_with_exact_budget(64 * 1024 * 1024);
            let directory = tempfile::TempDir::new().unwrap();
            let (resources, spill_manager) =
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .build_operator_resources(
                        Arc::clone(&manager),
                        crate::execution::QueryExecutionControl::new().token(),
                    )
                    .unwrap();
            let mut sort = SpillableSortPushOperator::with_resource_context(
                vec![SortKey::ascending(0)],
                resources,
            )
            .unwrap();
            let mut ignored = CollectorSink::new();
            for value in (0..run_count).rev() {
                sort.push(
                    create_test_chunk(&[i64::try_from(value).unwrap()]),
                    &mut ignored,
                )
                .unwrap();
                sort.spill_current_buffer(Arc::clone(&spill_manager))
                    .unwrap();
            }

            let mut sink = RetainingAccountedSortSink::default();
            sort.finalize(&mut sink).unwrap();
            assert_eq!(!sink.accounted.is_empty(), expect_exact, "{run_count} runs");
            assert_eq!(!sink.plain.is_empty(), !expect_exact, "{run_count} runs");
            assert_eq!(
                sort.external_sort.is_none(),
                expect_exact,
                "{run_count} runs"
            );
            let output = if expect_exact {
                accounted_rows(&sink)
                    .into_iter()
                    .map(|row| row[0].clone())
                    .collect::<Vec<_>>()
            } else {
                sink.plain
                    .iter()
                    .flat_map(|chunk| {
                        (0..chunk.len()).map(|row| chunk.column(0).unwrap().get_value(row).unwrap())
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                output,
                (0..run_count)
                    .map(|value| Value::Int64(i64::try_from(value).unwrap()))
                    .collect::<Vec<_>>()
            );
            assert_eq!(spill_manager.active_file_count(), 0);
            drop(sink);
            drop(sort);
            assert_eq!(manager.allocated(), 0);
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    fn exact_owned_sort_stop_error_and_post_sink_cancel_cleanup_without_losing_retained_charge() {
        enum Outcome {
            Stop,
            Error,
            Cancel,
        }

        for outcome in [Outcome::Stop, Outcome::Error, Outcome::Cancel] {
            let manager = buffer_manager_with_exact_budget(64 * 1024 * 1024);
            let directory = tempfile::TempDir::new().unwrap();
            let control = crate::execution::QueryExecutionControl::new();
            let (resources, spill_manager) =
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .build_operator_resources(Arc::clone(&manager), control.token())
                    .unwrap();
            let mut sort = SpillableSortPushOperator::with_resource_context(
                vec![SortKey::ascending(0)],
                resources,
            )
            .unwrap();
            let mut ignored = CollectorSink::new();
            for values in [&[3, 1][..], &[4, 2][..]] {
                sort.push(create_test_chunk(values), &mut ignored).unwrap();
                sort.spill_current_buffer(Arc::clone(&spill_manager))
                    .unwrap();
            }
            let mut sink = RetainingAccountedSortSink {
                stop_after_first: matches!(outcome, Outcome::Stop),
                fail_after_first: matches!(outcome, Outcome::Error),
                cancel_after_first: matches!(outcome, Outcome::Cancel)
                    .then(|| control.cancellation_handle()),
                ..RetainingAccountedSortSink::default()
            };

            let result = sort.finalize(&mut sink);
            match outcome {
                Outcome::Stop => assert!(result.is_ok()),
                Outcome::Error => assert!(matches!(
                    result,
                    Err(OperatorError::ResidentContainerInvariant {
                        container: "test exact accounted sink",
                        ..
                    })
                )),
                Outcome::Cancel => {
                    assert!(matches!(result, Err(OperatorError::QueryCancelled(_))));
                }
            }
            assert_eq!(sink.accounted.len(), 1);
            assert!(sink.plain.is_empty());
            assert!(sort.external_sort.is_none());
            assert_eq!(spill_manager.active_file_count(), 0);
            assert_eq!(spill_manager.spilled_bytes(), 0);
            assert_eq!(sort.spill_state.as_ref().unwrap().usage(), 0);
            let retained = sink.accounted[0].granted_bytes();
            assert_eq!(manager.allocated(), retained);
            drop(sort);
            assert_eq!(manager.allocated(), retained);
            drop(sink);
            assert_eq!(manager.allocated(), 0);
        }
    }

    #[test]
    fn resource_sort_charges_capacity_and_releases_it_on_drop() {
        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let mut sort =
            SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                .unwrap();
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[3, 1, 2]), &mut sink).unwrap();
        let charged = manager.allocated();
        assert!(
            charged > 0,
            "operator-owned capacity must consume a real grant"
        );
        assert_eq!(charged, sort.buffer.observed_charge().unwrap());

        sort.finalize(&mut sink).unwrap();
        assert_eq!(
            manager.allocated(),
            charged,
            "resident rows and their reserved stable-sort scratch live through finalize"
        );

        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn resource_sort_emits_bounded_output_chunks_without_spill_support() {
        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let mut sort =
            SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                .unwrap();
        let row_count = QUALIFIED_SORT_OUTPUT_CHUNK_ROWS + 3;
        let descending = (0..row_count)
            .rev()
            .map(|value| i64::try_from(value).unwrap())
            .collect::<Vec<_>>();
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&descending), &mut sink)
            .unwrap();
        sort.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), QUALIFIED_SORT_OUTPUT_CHUNK_ROWS);
        assert_eq!(chunks[1].len(), 3);
        let values = chunks
            .iter()
            .flat_map(|chunk| {
                let column = chunk.column(0).expect("test sort has one column");
                (0..chunk.len()).map(move |index| column.get_value(index))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            values,
            (0..row_count)
                .map(|value| Some(Value::Int64(i64::try_from(value).unwrap())))
                .collect::<Vec<_>>()
        );
        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn resource_sort_cancelled_push_precedes_shape_and_buffer_growth() {
        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let control = crate::execution::QueryExecutionControl::new();
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            Arc::clone(&manager),
            control.token(),
        )
        .unwrap();
        let mut sort =
            SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                .unwrap();
        let mut sink = CollectorSink::new();

        control.cancellation_handle().cancel();
        let error = sort.push(create_test_chunk(&[1]), &mut sink).unwrap_err();

        assert!(matches!(
            error,
            OperatorError::QueryCancelled(crate::execution::QueryCancellationError::Cancelled)
        ));
        assert_eq!(sort.num_columns, None);
        assert!(sort.buffer.rows().is_empty());
        assert_eq!(sort.buffer.rows.capacity(), 0);
        assert_eq!(sort.buffer.row_capacity_bytes, 0);
        assert_eq!(manager.allocated(), 0);
        assert!(sink.into_chunks().is_empty());
    }

    #[test]
    fn resource_sort_cancelled_finalize_publishes_no_output() {
        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let control = crate::execution::QueryExecutionControl::new();
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            Arc::clone(&manager),
            control.token(),
        )
        .unwrap();
        let mut sort =
            SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                .unwrap();
        let mut sink = CollectorSink::new();
        sort.push(create_test_chunk(&[3, 1, 2]), &mut sink).unwrap();
        let charged = manager.allocated();
        let original_rows = sort.buffer.rows().to_vec();

        control.cancellation_handle().cancel();
        let error = sort.finalize(&mut sink).unwrap_err();

        assert!(matches!(
            error,
            OperatorError::QueryCancelled(crate::execution::QueryCancellationError::Cancelled)
        ));
        assert!(sink.into_chunks().is_empty());
        assert_eq!(sort.buffer.rows(), original_rows);
        assert_eq!(manager.allocated(), charged);
        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn sort_sink_error_beats_cancellation_requested_by_sink() {
        struct CancellingFailingSink {
            cancellation: crate::execution::QueryCancellationHandle,
        }

        impl Sink for CancellingFailingSink {
            fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
                self.cancellation.cancel();
                Err(OperatorError::Execution(
                    "deterministic sort sink failure".to_string(),
                ))
            }

            fn finalize(&mut self) -> Result<(), OperatorError> {
                Ok(())
            }

            fn name(&self) -> &'static str {
                "CancellingFailingSink"
            }

            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
                self
            }
        }

        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let control = crate::execution::QueryExecutionControl::new();
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            Arc::clone(&manager),
            control.token(),
        )
        .unwrap();
        let mut sort =
            SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                .unwrap();
        let mut discard = CollectorSink::new();
        sort.push(create_test_chunk(&[3, 1, 2]), &mut discard)
            .unwrap();
        let charged = manager.allocated();
        let mut sink = CancellingFailingSink {
            cancellation: control.cancellation_handle(),
        };

        let error = sort.finalize(&mut sink).unwrap_err();

        assert!(matches!(
            error,
            OperatorError::Execution(ref message)
                if message == "deterministic sort sink failure"
        ));
        assert!(control.token().is_cancelled());
        assert_eq!(manager.allocated(), charged);
        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn resource_sort_tracks_row_capacity_without_rescanning_history() {
        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let mut sort =
            SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                .unwrap();
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[4, 3, 2, 1]), &mut sink)
            .unwrap();
        let scanned = sort
            .buffer
            .rows()
            .iter()
            .map(|row| row.capacity() * std::mem::size_of::<Value>())
            .sum::<usize>();
        assert_eq!(sort.buffer.row_capacity_bytes, scanned);
    }

    #[test]
    fn resource_sort_denial_is_typed_and_does_not_grow_unaccounted() {
        let required = one_integer_row_charge();
        let manager = buffer_manager_with_exact_budget(required - 1);
        let resources = crate::execution::QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let mut sort =
            SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                .unwrap();
        let mut sink = CollectorSink::new();

        let error = sort.push(create_test_chunk(&[1]), &mut sink).unwrap_err();
        assert!(matches!(
            error,
            OperatorError::ResidentMemory(
                grafeo_common::memory::buffer::MemoryGrantError::LimitExceeded { .. }
            )
        ));
        assert_eq!(
            manager.allocated(),
            0,
            "denial must roll back every provisional grant"
        );
        assert!(sort.buffer.rows().is_empty());
        assert_eq!(sort.buffer.rows.capacity(), 0);
        assert_eq!(sort.buffer.row_capacity_bytes, 0);

        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn resource_sort_drop_during_unwind_releases_every_granted_byte() {
        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new(Arc::clone(&manager)).unwrap();

        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut sort =
                SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                    .unwrap();
            let mut sink = CollectorSink::new();
            sort.push(create_test_chunk(&[3, 2, 1]), &mut sink).unwrap();
            assert!(manager.allocated() > 0);
            panic!("exercise grant cleanup");
        }));

        assert!(unwind.is_err());
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn sort_rejects_chunk_arity_drift_before_buffering_it() {
        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let resources = crate::execution::QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let mut sort =
            SortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
                .unwrap();
        let mut sink = CollectorSink::new();
        sort.push(create_test_chunk(&[1]), &mut sink).unwrap();
        let before = manager.allocated();
        let two_columns = DataChunk::new(vec![
            ValueVector::from_values(&[Value::Int64(2)]),
            ValueVector::from_values(&[Value::Int64(3)]),
        ]);

        let error = sort.push(two_columns, &mut sink).unwrap_err();
        assert!(matches!(error, OperatorError::TypeMismatch { .. }));
        assert_eq!(manager.allocated(), before);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_sort_no_spill() {
        // When threshold is not reached, should work like normal sort
        let mut sort =
            SpillableSortPushOperator::new(vec![SortKey::ascending(0)]).with_threshold(100);
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[3, 1, 4, 1, 5, 9, 2, 6]), &mut sink)
            .unwrap();
        sort.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);

        let col = chunks[0].column(0).unwrap();
        assert_eq!(col.get_value(0), Some(Value::Int64(1)));
        assert_eq!(col.get_value(1), Some(Value::Int64(1)));
        assert_eq!(col.get_value(2), Some(Value::Int64(2)));
        assert_eq!(col.get_value(3), Some(Value::Int64(3)));
    }

    #[test]
    #[cfg(feature = "spill")]
    // reason: test values 1..=10 fit i64
    #[allow(clippy::cast_possible_wrap)]
    fn test_spillable_sort_with_spilling() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build()
                .unwrap(),
        );

        // Set very low threshold to force spilling
        let mut sort = SpillableSortPushOperator::ascending_with_spilling(0, manager, 5);
        let mut sink = CollectorSink::new();

        // Push more than threshold
        sort.push(create_test_chunk(&[10, 8, 6, 4, 2]), &mut sink)
            .unwrap();
        sort.push(create_test_chunk(&[9, 7, 5, 3, 1]), &mut sink)
            .unwrap();
        sort.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 10);

        // Verify sorted order
        let col = chunks[0].column(0).unwrap();
        for i in 0..10 {
            assert_eq!(col.get_value(i), Some(Value::Int64((i + 1) as i64)));
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    fn row_count_sort_quota_failure_remains_structured_storage_full() {
        use crate::execution::spill::SpillDiskQuota;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .quota(SpillDiskQuota::new(0))
                .build()
                .unwrap(),
        );
        let mut sort = SpillableSortPushOperator::ascending_with_spilling(0, manager, 1);
        let mut sink = CollectorSink::new();

        let error = sort.push(create_test_chunk(&[1]), &mut sink).unwrap_err();

        assert!(matches!(error, OperatorError::StorageFull(_)));
    }

    #[test]
    #[cfg(feature = "spill")]
    // reason: test values 1..=15 fit i64
    #[allow(clippy::cast_possible_wrap)]
    fn test_spillable_sort_many_runs() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build()
                .unwrap(),
        );

        // Set very low threshold to force multiple spills
        let mut sort = SpillableSortPushOperator::ascending_with_spilling(0, manager, 3);
        let mut sink = CollectorSink::new();

        // Push data in multiple chunks
        for i in 0..5 {
            sort.push(
                create_test_chunk(&[i * 3 + 3, i * 3 + 2, i * 3 + 1]),
                &mut sink,
            )
            .unwrap();
        }
        sort.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 15);

        // Verify sorted order
        let col = chunks[0].column(0).unwrap();
        for i in 0..15 {
            assert_eq!(col.get_value(i), Some(Value::Int64((i + 1) as i64)));
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    // reason: test values 1..=6 fit i64
    #[allow(clippy::cast_possible_wrap)]
    fn test_spillable_sort_descending_with_spilling() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build()
                .unwrap(),
        );

        let mut sort = SpillableSortPushOperator::descending_with_spilling(0, manager, 3);
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[1, 3, 5]), &mut sink).unwrap();
        sort.push(create_test_chunk(&[2, 4, 6]), &mut sink).unwrap();
        sort.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        let col = chunks[0].column(0).unwrap();

        // Should be descending: 6, 5, 4, 3, 2, 1
        for i in 0..6 {
            assert_eq!(col.get_value(i), Some(Value::Int64((6 - i) as i64)));
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    fn resource_sort_hard_denial_spills_once_and_retries_the_exact_row() {
        const PREFIX_ROWS: usize = 4096;
        let (prefix_charge, budget) = hard_denial_sort_budget(PREFIX_ROWS);
        let manager = buffer_manager_with_exact_budget(budget);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();

        sort.push(create_stable_sort_chunk(0, PREFIX_ROWS), &mut sink)
            .unwrap();
        assert_eq!(manager.allocated(), prefix_charge);
        sort.push(create_stable_sort_chunk(PREFIX_ROWS, 1), &mut sink)
            .expect("the denied row must be retried after one successful spill");

        assert_eq!(spill_manager.active_file_count(), 1);
        assert_eq!(sort.buffer.rows().len(), 1);
        assert_eq!(
            sort.buffer.rows()[0],
            vec![
                Value::Int64(0),
                Value::Int64(i64::try_from(PREFIX_ROWS).unwrap())
            ]
        );
        assert_eq!(
            sort.spill_state.as_ref().unwrap().usage(),
            sort.total_granted_bytes()
        );

        sort.finalize(&mut sink).unwrap();
        let chunks = sink.into_chunks();
        let mut previous_key = i64::MIN;
        let mut previous_ordinal = 0;
        let mut observed = 0;
        for chunk in chunks {
            let keys = chunk.column(0).unwrap();
            let ordinals = chunk.column(1).unwrap();
            for row in 0..chunk.len() {
                let Value::Int64(key) = keys.get_value(row).unwrap() else {
                    panic!("sort key lost its type");
                };
                let Value::Int64(ordinal) = ordinals.get_value(row).unwrap() else {
                    panic!("sort ordinal lost its type");
                };
                assert!(key >= previous_key);
                if key == previous_key {
                    assert!(ordinal > previous_ordinal, "equal-key rows lost stability");
                }
                previous_key = key;
                previous_ordinal = ordinal;
                observed += 1;
            }
        }
        assert_eq!(observed, PREFIX_ROWS + 1);

        drop(sort);
        assert_eq!(manager.allocated(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn resource_sort_hard_denial_preserves_spill_quota_failure_and_cleanup() {
        use crate::execution::spill::SpillDiskQuota;

        const PREFIX_ROWS: usize = 4096;
        let (_, budget) = hard_denial_sort_budget(PREFIX_ROWS);
        let manager = buffer_manager_with_exact_budget(budget);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .quota(SpillDiskQuota::new(0))
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();

        sort.push(create_stable_sort_chunk(0, PREFIX_ROWS), &mut sink)
            .unwrap();
        let error = sort
            .push(create_stable_sort_chunk(PREFIX_ROWS, 1), &mut sink)
            .unwrap_err();

        assert!(matches!(error, OperatorError::StorageFull(_)));
        assert!(sort.buffer.is_empty());
        assert_eq!(sort.buffer.granted_bytes(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(spill_manager.disk_stats().reserved_live_bytes, 0);
        let external_bytes = sort.external_sort.as_ref().unwrap().total_granted_bytes();
        assert_eq!(manager.allocated(), external_bytes);
        assert_eq!(sort.spill_state.as_ref().unwrap().usage(), external_bytes);

        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn resource_sort_hard_denial_cancellation_during_spill_wins_and_cleans() {
        struct CancelOnSync {
            cancellation: crate::execution::QueryCancellationHandle,
        }

        impl crate::execution::spill::SpillIo for CancelOnSync {
            fn check(
                &self,
                operation: crate::execution::spill::SpillIoOperation,
            ) -> std::io::Result<()> {
                if operation == crate::execution::spill::SpillIoOperation::Sync {
                    self.cancellation.cancel();
                }
                Ok(())
            }
        }

        const PREFIX_ROWS: usize = 4096;
        let (_, budget) = hard_denial_sort_budget(PREFIX_ROWS);
        let manager = buffer_manager_with_exact_budget(budget);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .provider(
                    Arc::new(crate::execution::spill::CleartextSpillRecordProvider),
                    crate::execution::spill::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(CancelOnSync {
                    cancellation: control.cancellation_handle(),
                }))
                .build_operator_resources(Arc::clone(&manager), control.token())
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();

        sort.push(create_stable_sort_chunk(0, PREFIX_ROWS), &mut sink)
            .unwrap();
        let error = sort
            .push(create_stable_sort_chunk(PREFIX_ROWS, 1), &mut sink)
            .unwrap_err();

        assert!(matches!(
            error,
            OperatorError::QueryCancelled(crate::execution::QueryCancellationError::Cancelled)
        ));
        assert!(control.token().is_cancelled());
        assert!(sort.buffer.is_empty());
        assert_eq!(sort.buffer.granted_bytes(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(spill_manager.disk_stats().reserved_live_bytes, 0);
        let external_bytes = sort.external_sort.as_ref().unwrap().total_granted_bytes();
        assert_eq!(manager.allocated(), external_bytes);
        assert_eq!(sort.spill_state.as_ref().unwrap().usage(), external_bytes);

        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn resource_sort_hard_denial_retries_only_once_when_the_retry_is_denied() {
        struct SplitQueryLimitOnSync {
            manager: Arc<grafeo_common::memory::buffer::BufferManager>,
            competitors: std::sync::Mutex<Vec<crate::execution::memory::QueryResourceContext>>,
        }

        impl crate::execution::spill::SpillIo for SplitQueryLimitOnSync {
            fn check(
                &self,
                operation: crate::execution::spill::SpillIoOperation,
            ) -> std::io::Result<()> {
                if operation != crate::execution::spill::SpillIoOperation::Sync {
                    return Ok(());
                }
                let mut competitors = self.competitors.lock().unwrap();
                if competitors.is_empty() {
                    for _ in 0..1024 {
                        competitors.push(
                            crate::execution::QueryResourceContext::new(Arc::clone(&self.manager))
                                .unwrap(),
                        );
                    }
                }
                Ok(())
            }
        }

        const PREFIX_ROWS: usize = 4096;
        let (_, budget) = hard_denial_sort_budget(PREFIX_ROWS);
        let manager = buffer_manager_with_exact_budget(budget);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .provider(
                    Arc::new(crate::execution::spill::CleartextSpillRecordProvider),
                    crate::execution::spill::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(SplitQueryLimitOnSync {
                    manager: Arc::clone(&manager),
                    competitors: std::sync::Mutex::new(Vec::new()),
                }))
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();

        sort.push(create_stable_sort_chunk(0, PREFIX_ROWS), &mut sink)
            .unwrap();
        let error = sort
            .push(create_stable_sort_chunk(PREFIX_ROWS, 1), &mut sink)
            .unwrap_err();

        assert!(matches!(
            error,
            OperatorError::ResidentMemory(MemoryGrantError::LimitExceeded {
                scope: grafeo_common::memory::buffer::MemoryLimitScope::Query,
                ..
            })
        ));
        assert!(sort.buffer.is_empty());
        assert_eq!(sort.buffer.granted_bytes(), 0);
        let external = sort.external_sort.as_ref().unwrap();
        assert_eq!(external.total_rows(), PREFIX_ROWS);
        assert_eq!(spill_manager.active_file_count(), 1);
        assert!(spill_manager.spilled_bytes() > 0);
        let external_bytes = external.total_granted_bytes();
        assert_eq!(manager.allocated(), external_bytes);
        assert_eq!(sort.spill_state.as_ref().unwrap().usage(), external_bytes);

        drop(sort);
        assert_eq!(manager.allocated(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn resource_sort_hard_denial_spill_grant_failure_precedes_original_and_cleans() {
        let per_run_budget = one_integer_row_charge();

        let original_requested_bytes = {
            let original_manager = buffer_manager_with_exact_budget(per_run_budget);
            let original_resources =
                crate::execution::QueryResourceContext::new(Arc::clone(&original_manager)).unwrap();
            let mut original = SortPushOperator::with_resource_context(
                vec![SortKey::ascending(0)],
                original_resources,
            )
            .unwrap();
            let mut discard = CollectorSink::new();
            let error = original
                .push(create_test_chunk(&[3, 1]), &mut discard)
                .unwrap_err();
            let OperatorError::ResidentMemory(MemoryGrantError::LimitExceeded {
                requested_bytes,
                ..
            }) = error
            else {
                panic!("the calibration must reach the original row grant denial");
            };
            drop(original);
            assert_eq!(original_manager.allocated(), 0);
            requested_bytes
        };

        let manager = buffer_manager_with_exact_budget(per_run_budget);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();

        let error = sort
            .push(create_test_chunk(&[3, 1, 2]), &mut sink)
            .unwrap_err();
        let OperatorError::ResidentMemory(MemoryGrantError::LimitExceeded {
            requested_bytes: spill_requested_bytes,
            ..
        }) = error
        else {
            panic!("the concrete spill-workspace grant failure was reclassified");
        };
        assert!(
            spill_requested_bytes < original_requested_bytes,
            "the returned grant denial must be the spill failure, not the triggering row denial"
        );
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(spill_manager.disk_stats().reserved_live_bytes, 0);
        assert!(sort.buffer.is_empty());
        assert_eq!(sort.buffer.granted_bytes(), 0);
        assert_eq!(manager.allocated(), 0);
        assert_eq!(sort.spill_state.as_ref().unwrap().usage(), 0);

        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn resource_spill_sort_cancelled_push_creates_no_run() {
        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build_operator_resources(Arc::clone(&manager), control.token())
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();
        let values = vec![1; SORT_MIN_BUFFER_ROWS];

        control.cancellation_handle().cancel();
        let error = sort
            .push(create_test_chunk(&values), &mut sink)
            .unwrap_err();

        assert!(matches!(
            error,
            OperatorError::QueryCancelled(crate::execution::QueryCancellationError::Cancelled)
        ));
        assert_eq!(sort.num_columns, None);
        assert!(sort.buffer.rows().is_empty());
        assert_eq!(sort.buffer.rows.capacity(), 0);
        assert_eq!(sort.buffer.row_capacity_bytes, 0);
        assert!(sort.external_sort.is_none());
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(spill_manager.disk_stats().reserved_live_bytes, 0);
        assert_eq!(manager.allocated(), 0);
        assert!(sink.into_chunks().is_empty());
    }

    #[test]
    fn sort_capacity_arithmetic_fails_closed_on_overflow() {
        assert!(matches!(
            capacity_bytes::<Value>(usize::MAX, "test values"),
            Err(SortBufferGrowthError::CapacityOverflow {
                container: "test values"
            })
        ));
        assert!(matches!(
            checked_capacity_sum(usize::MAX, 1, "test sum"),
            Err(SortBufferGrowthError::CapacityOverflow {
                container: "test sum"
            })
        ));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn external_sort_allocator_failure_remains_structured() {
        let error = SpillableSortPushOperator::map_external_sort_error(
            ExternalSortOperationError::Allocation(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "deterministic allocator refusal",
            )),
        );

        assert!(matches!(error, OperatorError::ResidentAllocation(_)));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn external_sort_quota_failure_remains_structured_storage_full() {
        let error = SpillableSortPushOperator::map_external_sort_error(
            ExternalSortOperationError::Io(std::io::Error::new(
                std::io::ErrorKind::QuotaExceeded,
                "deterministic per-query spill quota",
            )),
        );

        assert!(matches!(error, OperatorError::StorageFull(_)));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn writer_release_context_does_not_reclassify_io_primary() {
        let error = SpillableSortPushOperator::map_external_sort_error(
            ExternalSortOperationError::WithGrantRelease {
                primary: ExternalSortPrimary::Io(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "deterministic writer failure",
                )),
                release: MemoryGrantError::AccountingPoisoned {
                    account: "deterministic writer release",
                },
                cleanup: Some(std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    "deterministic delete failure",
                )),
                phase: "writer failure release",
            },
        );

        let OperatorError::Context {
            source: cleanup_source,
            context: cleanup_context,
        } = error
        else {
            panic!("writer cleanup context was flattened");
        };
        assert!(cleanup_context.contains("delete failure"));
        let OperatorError::Context {
            source: release_source,
            context: release_context,
        } = *cleanup_source
        else {
            panic!("writer release context was flattened");
        };
        assert!(release_context.starts_with("writer failure release also failed:"));
        assert!(!release_context.contains("writer failure release release"));
        assert!(release_context.contains("deterministic writer release"));
        assert!(matches!(*release_source, OperatorError::Execution(_)));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn writer_release_and_cleanup_context_preserve_typed_cancellation() {
        let error = SpillableSortPushOperator::map_external_sort_error(
            ExternalSortOperationError::WithGrantRelease {
                primary: ExternalSortPrimary::Cancelled(
                    crate::execution::QueryCancellationError::Cancelled,
                ),
                release: MemoryGrantError::AccountingPoisoned {
                    account: "deterministic cancellation writer release",
                },
                cleanup: Some(std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    "deterministic cancellation delete failure",
                )),
                phase: "writer cancellation release",
            },
        );

        let OperatorError::Context {
            source: cleanup_source,
            context: cleanup_context,
        } = error
        else {
            panic!("cancellation cleanup context was flattened");
        };
        assert!(cleanup_context.contains("cancellation delete failure"));
        let OperatorError::Context {
            source: release_source,
            context: release_context,
        } = *cleanup_source
        else {
            panic!("cancellation release context was flattened");
        };
        assert!(release_context.contains("cancellation writer release"));
        assert!(matches!(
            *release_source,
            OperatorError::QueryCancelled(crate::execution::QueryCancellationError::Cancelled)
        ));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn row_grant_release_failure_preserves_every_primary_classification() {
        let secondary = MemoryGrantError::AccountingPoisoned {
            account: "test row grant",
        };
        let cases = [
            OperatorError::ResidentMemory(MemoryGrantError::LimitExceeded {
                scope: grafeo_common::memory::buffer::MemoryLimitScope::Query,
                requested_bytes: 65,
                limit_bytes: 64,
            }),
            OperatorError::ResidentAllocation("allocator refused workspace".to_string()),
            OperatorError::WriteConflict("concurrent writer".to_string()),
        ];

        for primary in cases {
            let expected = primary.to_string();
            let combined =
                SpillableSortPushOperator::with_row_release_context(primary, &secondary, "spill");
            match combined {
                OperatorError::Context { source, context } => {
                    assert_eq!(source.to_string(), expected);
                    assert!(context.contains("release after spill"));
                    assert!(context.contains("test row grant"));
                }
                other => panic!("primary was flattened instead of retained: {other}"),
            }
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spilled_sort_retains_tail_grant_until_emission_finishes() {
        struct GrantObservingSink {
            manager: Arc<grafeo_common::memory::buffer::BufferManager>,
            observed: bool,
        }

        impl Sink for GrantObservingSink {
            fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
                if self.manager.allocated() == 0 {
                    return Err(OperatorError::Execution(
                        "tail grant was released before sorted emission".to_string(),
                    ));
                }
                self.observed = true;
                Ok(true)
            }

            fn finalize(&mut self) -> Result<(), OperatorError> {
                Ok(())
            }

            fn name(&self) -> &'static str {
                "GrantObservingSink"
            }

            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
                self
            }
        }

        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut discard = CollectorSink::new();

        sort.push(create_test_chunk(&[3, 1]), &mut discard).unwrap();
        sort.spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap();
        let external_sort_bytes = sort.external_sort.as_ref().unwrap().total_granted_bytes();
        assert!(external_sort_bytes > 0);
        assert_eq!(manager.allocated(), external_sort_bytes);
        assert_eq!(
            sort.spill_state.as_ref().unwrap().usage(),
            external_sort_bytes
        );
        sort.push(create_test_chunk(&[2]), &mut discard).unwrap();
        assert!(manager.allocated() > external_sort_bytes);

        let mut sink = GrantObservingSink {
            manager: Arc::clone(&manager),
            observed: false,
        };
        sort.finalize(&mut sink).unwrap();
        assert!(sink.observed);
        let catalog_bytes = sort
            .external_sort
            .as_ref()
            .unwrap()
            .run_catalog_granted_bytes();
        assert!(catalog_bytes > 0);
        assert_eq!(manager.allocated(), catalog_bytes);
        assert_eq!(sort.spill_state.as_ref().unwrap().usage(), catalog_bytes);
        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn resource_sort_emits_bounded_chunks_before_and_after_spill() {
        struct BoundedChunkSink {
            chunk_lengths: Vec<usize>,
            values: Vec<i64>,
        }

        impl Sink for BoundedChunkSink {
            fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
                if chunk.len() > QUALIFIED_SORT_OUTPUT_CHUNK_ROWS {
                    return Err(OperatorError::Execution(format!(
                        "sort emitted an unbounded {}-row chunk",
                        chunk.len()
                    )));
                }
                self.chunk_lengths.push(chunk.len());
                let column = chunk.column(0).expect("test sort has one column");
                for index in 0..chunk.len() {
                    let Some(Value::Int64(value)) = column.get_value(index) else {
                        panic!("test sort emitted a non-integer value");
                    };
                    self.values.push(value);
                }
                Ok(true)
            }

            fn finalize(&mut self) -> Result<(), OperatorError> {
                Ok(())
            }

            fn name(&self) -> &'static str {
                "BoundedChunkSink"
            }

            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
                self
            }
        }

        let manager = buffer_manager_with_exact_budget(16 * 1024 * 1024);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut resident_sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources.clone(),
        )
        .unwrap();
        let mut discard = CollectorSink::new();
        let row_count = QUALIFIED_SORT_OUTPUT_CHUNK_ROWS + 3;
        let descending = (0..row_count)
            .rev()
            .map(|value| i64::try_from(value).unwrap())
            .collect::<Vec<_>>();
        resident_sort
            .push(create_test_chunk(&descending), &mut discard)
            .unwrap();
        let mut resident_sink = BoundedChunkSink {
            chunk_lengths: Vec::new(),
            values: Vec::new(),
        };

        resident_sort.finalize(&mut resident_sink).unwrap();

        assert_eq!(
            resident_sink.chunk_lengths,
            vec![QUALIFIED_SORT_OUTPUT_CHUNK_ROWS, 3]
        );
        assert_eq!(
            resident_sink.values,
            (0..row_count)
                .map(|value| i64::try_from(value).unwrap())
                .collect::<Vec<_>>()
        );
        drop(resident_sort);
        assert_eq!(manager.allocated(), 0);

        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        sort.push(create_test_chunk(&descending), &mut discard)
            .unwrap();
        sort.spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap();
        let mut sink = BoundedChunkSink {
            chunk_lengths: Vec::new(),
            values: Vec::new(),
        };

        sort.finalize(&mut sink).unwrap();

        assert_eq!(
            sink.chunk_lengths,
            vec![QUALIFIED_SORT_OUTPUT_CHUNK_ROWS, 3]
        );
        assert_eq!(
            sink.values,
            (0..row_count)
                .map(|value| i64::try_from(value).unwrap())
                .collect::<Vec<_>>()
        );
        assert_eq!(spill_manager.active_file_count(), 0);
        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn resource_spilled_sort_early_stop_aborts_cursor_and_cleans_runs() {
        struct StopAfterFirstChunk {
            chunks: usize,
        }

        impl Sink for StopAfterFirstChunk {
            fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
                self.chunks += 1;
                Ok(false)
            }

            fn finalize(&mut self) -> Result<(), OperatorError> {
                Ok(())
            }

            fn name(&self) -> &'static str {
                "StopAfterFirstChunk"
            }

            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
                self
            }
        }

        let manager = buffer_manager_with_exact_budget(16 * 1024 * 1024);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut discard = CollectorSink::new();
        let row_count = QUALIFIED_SORT_OUTPUT_CHUNK_ROWS + 1;
        let values = (0..row_count)
            .map(|value| i64::try_from(value).unwrap())
            .collect::<Vec<_>>();
        sort.push(create_test_chunk(&values), &mut discard).unwrap();
        sort.spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap();
        let mut sink = StopAfterFirstChunk { chunks: 0 };

        sort.finalize(&mut sink).unwrap();

        assert_eq!(sink.chunks, 1);
        assert_eq!(spill_manager.active_file_count(), 0);
        let external = sort.external_sort.as_ref().unwrap();
        assert_eq!(external.num_runs(), 0);
        assert_eq!(
            external.total_granted_bytes(),
            external.run_catalog_granted_bytes()
        );
        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn resource_spilled_sort_sink_error_preserves_cleanup_and_cancellation_precedence() {
        struct ToggleDeleteIo {
            deny: AtomicBool,
        }

        impl crate::execution::spill::SpillIo for ToggleDeleteIo {
            fn check(
                &self,
                operation: crate::execution::spill::SpillIoOperation,
            ) -> std::io::Result<()> {
                if operation == crate::execution::spill::SpillIoOperation::Delete
                    && self.deny.load(AtomicOrdering::Acquire)
                {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::WouldBlock,
                        "deterministic streamed-sort delete denial",
                    ));
                }
                Ok(())
            }
        }

        struct CancellingFailingSink {
            cancellation: crate::execution::QueryCancellationHandle,
        }

        impl Sink for CancellingFailingSink {
            fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
                self.cancellation.cancel();
                Err(OperatorError::Execution(
                    "deterministic streamed-sort sink failure".to_string(),
                ))
            }

            fn finalize(&mut self) -> Result<(), OperatorError> {
                Ok(())
            }

            fn name(&self) -> &'static str {
                "CancellingFailingSink"
            }

            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
                self
            }
        }

        let manager = buffer_manager_with_exact_budget(16 * 1024 * 1024);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let io = Arc::new(ToggleDeleteIo {
            deny: AtomicBool::new(true),
        });
        let control = crate::execution::QueryExecutionControl::new();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .provider(
                    Arc::new(crate::execution::spill::CleartextSpillRecordProvider),
                    crate::execution::spill::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn crate::execution::spill::SpillIo>)
                .build_operator_resources(Arc::clone(&manager), control.token())
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut discard = CollectorSink::new();
        let row_count = QUALIFIED_SORT_OUTPUT_CHUNK_ROWS + 1;
        let values = (0..row_count)
            .map(|value| i64::try_from(value).unwrap())
            .collect::<Vec<_>>();
        sort.push(create_test_chunk(&values), &mut discard).unwrap();
        sort.spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap();
        let mut sink = CancellingFailingSink {
            cancellation: control.cancellation_handle(),
        };

        let error = sort.finalize(&mut sink).unwrap_err();

        assert!(matches!(
            &error,
            OperatorError::ClassifiedAccountedFailure {
                classification: AccountedFailureClassification::Execution,
                ..
            }
        ));
        ExternalSort::inspect_scalar_failure(&error, |primary, cleanup| {
            assert!(matches!(
                primary,
                Some(OperatorError::Execution(message))
                    if message == "deterministic streamed-sort sink failure"
            ));
            let Some(ExternalSortOperationError::Io(cleanup)) = cleanup else {
                panic!("typed cursor cleanup failure was lost");
            };
            assert_eq!(cleanup.kind(), std::io::ErrorKind::WouldBlock);
            assert_eq!(
                cleanup.to_string(),
                "deterministic streamed-sort delete denial"
            );
        })
        .expect("accounted scalar diagnostic retains primary and cleanup");
        assert!(control.token().is_cancelled());
        assert_eq!(spill_manager.active_file_count(), 1);

        // The sorter still owns retryable files after the escaped diagnostic
        // is destroyed; its shared failure authority must remain live.
        drop(error);
        assert!(manager.allocated() > 0);
        io.deny.store(false, AtomicOrdering::Release);
        sort.external_sort.as_mut().unwrap().cleanup().unwrap();
        assert_eq!(spill_manager.active_file_count(), 0);
        assert!(manager.allocated() > 0);
        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn failed_proactive_spill_releases_the_moved_row_grant() {
        struct CancelThenFailSync {
            cancellation: crate::execution::QueryCancellationHandle,
        }

        impl crate::execution::spill::SpillIo for CancelThenFailSync {
            fn check(
                &self,
                operation: crate::execution::spill::SpillIoOperation,
            ) -> std::io::Result<()> {
                if operation == crate::execution::spill::SpillIoOperation::Sync {
                    self.cancellation.cancel();
                    Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "deterministic sort spill failure",
                    ))
                } else {
                    Ok(())
                }
            }
        }

        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .provider(
                    Arc::new(crate::execution::spill::CleartextSpillRecordProvider),
                    crate::execution::spill::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(CancelThenFailSync {
                    cancellation: control.cancellation_handle(),
                }))
                .build_operator_resources(Arc::clone(&manager), control.token())
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[2, 1]), &mut sink).unwrap();
        assert!(manager.allocated() > 0);
        let error = sort
            .spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap_err();

        assert!(matches!(
            error,
            OperatorError::Execution(ref message)
                if message.contains("deterministic sort spill failure")
        ));
        assert!(control.token().is_cancelled());
        let external_sort_bytes = sort.external_sort.as_ref().unwrap().total_granted_bytes();
        assert!(external_sort_bytes > 0);
        assert_eq!(manager.allocated(), external_sort_bytes);
        assert_eq!(
            sort.spill_state.as_ref().unwrap().usage(),
            external_sort_bytes
        );
        assert_eq!(spill_manager.active_file_count(), 0);

        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn cancellation_before_spill_publication_cleans_staged_file_and_row_grant() {
        struct CancelOnSync {
            cancellation: crate::execution::QueryCancellationHandle,
        }

        impl crate::execution::spill::SpillIo for CancelOnSync {
            fn check(
                &self,
                operation: crate::execution::spill::SpillIoOperation,
            ) -> std::io::Result<()> {
                if operation == crate::execution::spill::SpillIoOperation::Sync {
                    self.cancellation.cancel();
                }
                Ok(())
            }
        }

        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .provider(
                    Arc::new(crate::execution::spill::CleartextSpillRecordProvider),
                    crate::execution::spill::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(CancelOnSync {
                    cancellation: control.cancellation_handle(),
                }))
                .build_operator_resources(Arc::clone(&manager), control.token())
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();
        sort.push(create_test_chunk(&[2, 1]), &mut sink).unwrap();

        let error = sort
            .spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap_err();

        assert!(matches!(
            error,
            OperatorError::QueryCancelled(crate::execution::QueryCancellationError::Cancelled)
        ));
        assert!(sort.buffer.rows().is_empty());
        assert_eq!(sort.buffer.granted_bytes(), 0);
        let external_sort_bytes = sort.external_sort.as_ref().unwrap().total_granted_bytes();
        assert!(external_sort_bytes > 0);
        assert_eq!(manager.allocated(), external_sort_bytes);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(spill_manager.disk_stats().reserved_live_bytes, 0);

        drop(sort);
        assert_eq!(manager.allocated(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn failed_row_grant_release_retains_its_token_and_exact_telemetry() {
        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[2, 1]), &mut sink).unwrap();
        let row_bytes = sort.buffer.granted_bytes();
        assert!(row_bytes > 0);
        sort.buffer.release_error = Some(MemoryGrantError::AccountingPoisoned {
            account: "deterministic sort row release",
        });

        let error = sort
            .spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap_err();

        assert!(matches!(
            error,
            OperatorError::ResidentMemory(MemoryGrantError::AccountingPoisoned {
                account: "deterministic sort row release"
            })
        ));
        assert!(sort.buffer.rows().is_empty());
        assert_eq!(sort.buffer.granted_bytes(), row_bytes);
        let external_bytes = sort.external_sort.as_ref().unwrap().total_granted_bytes();
        let expected = row_bytes.checked_add(external_bytes).unwrap();
        assert_eq!(manager.allocated(), expected);
        assert_eq!(sort.spill_state.as_ref().unwrap().usage(), expected);
        assert_eq!(spill_manager.active_file_count(), 1);

        sort.buffer.release_error = None;
        sort.buffer.release_empty_grant().unwrap();
        sort.refresh_spill_usage();
        assert_eq!(manager.allocated(), external_bytes);
        assert_eq!(sort.spill_state.as_ref().unwrap().usage(), external_bytes);
        drop(sort);
        assert_eq!(manager.allocated(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn caught_spill_callback_panic_releases_the_moved_row_grant() {
        struct PanicOnceWriteIo {
            panicked: AtomicBool,
        }

        impl crate::execution::spill::SpillIo for PanicOnceWriteIo {
            fn check(
                &self,
                operation: crate::execution::spill::SpillIoOperation,
            ) -> std::io::Result<()> {
                if operation == crate::execution::spill::SpillIoOperation::WritePayload
                    && !self.panicked.swap(true, AtomicOrdering::Relaxed)
                {
                    panic!("deterministic spill callback panic");
                }
                Ok(())
            }
        }

        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .provider(
                    Arc::new(crate::execution::spill::CleartextSpillRecordProvider),
                    crate::execution::spill::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(PanicOnceWriteIo {
                    panicked: AtomicBool::new(false),
                }))
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[2, 1]), &mut sink).unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sort.spill_current_buffer(Arc::clone(&spill_manager))
        }));

        assert!(panic.is_err());
        assert!(sort.buffer.rows().is_empty());
        assert_eq!(sort.buffer.granted_bytes(), 0);
        let external_bytes = sort.external_sort.as_ref().unwrap().total_granted_bytes();
        assert!(external_bytes > 0);
        assert_eq!(manager.allocated(), external_bytes);
        assert_eq!(sort.spill_state.as_ref().unwrap().usage(), external_bytes);
        assert_eq!(spill_manager.active_file_count(), 0);

        sort.push(create_test_chunk(&[3]), &mut sink).unwrap();
        sort.spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap();
        sort.finalize(&mut sink).unwrap();
        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn caught_merge_callback_panic_releases_the_moved_tail_grant() {
        struct PanicOnceReadIo {
            panicked: AtomicBool,
        }

        impl crate::execution::spill::SpillIo for PanicOnceReadIo {
            fn check(
                &self,
                operation: crate::execution::spill::SpillIoOperation,
            ) -> std::io::Result<()> {
                if operation == crate::execution::spill::SpillIoOperation::ReadPayload
                    && !self.panicked.swap(true, AtomicOrdering::Relaxed)
                {
                    panic!("deterministic merge callback panic");
                }
                Ok(())
            }
        }

        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .provider(
                    Arc::new(crate::execution::spill::CleartextSpillRecordProvider),
                    crate::execution::spill::SpillFrameLimits::format_max(),
                )
                .io(Arc::new(PanicOnceReadIo {
                    panicked: AtomicBool::new(false),
                }))
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[2]), &mut sink).unwrap();
        sort.spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap();
        sort.push(create_test_chunk(&[1]), &mut sink).unwrap();
        let error =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sort.finalize(&mut sink)))
                .expect("reader callback panic is retained in the accounted diagnostic")
                .unwrap_err();
        let OperatorError::ClassifiedAccountedFailure {
            classification: AccountedFailureClassification::Execution,
            authority,
        } = &error
        else {
            panic!("reader callback panic lost its classified owner");
        };
        let diagnostic_bytes = authority.granted_bytes();
        let retained_bytes = ExternalSort::inspect_scalar_panic(&error, |panic, bytes| {
            assert_eq!(
                panic.and_then(|payload| payload.downcast_ref::<&str>()),
                Some(&"deterministic merge callback panic")
            );
            bytes
        })
        .expect("the original scalar panic remains inspectable");

        assert!(sort.buffer.rows().is_empty());
        assert_eq!(sort.buffer.granted_bytes(), 0);
        let external_bytes = sort.external_sort.as_ref().unwrap().total_granted_bytes();
        assert!(external_bytes > 0);
        assert_eq!(manager.allocated(), external_bytes + diagnostic_bytes);
        assert_eq!(sort.spill_state.as_ref().unwrap().usage(), external_bytes);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(std::fs::read_dir(temp_dir.path()).unwrap().count(), 0);

        drop(sort);
        assert_eq!(manager.allocated(), diagnostic_bytes + retained_bytes);
        assert!(retained_bytes > 0);
        drop(error);
        assert_eq!(manager.allocated(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn caught_sink_panic_reconciles_accounted_cursor_drop_usage() {
        struct PanickingSink;

        impl Sink for PanickingSink {
            fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
                panic!("deterministic downstream sort sink panic")
            }

            fn finalize(&mut self) -> Result<(), OperatorError> {
                Ok(())
            }

            fn name(&self) -> &'static str {
                "PanickingSortSink"
            }

            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
                self
            }
        }

        let manager = buffer_manager_with_exact_budget(1024 * 1024);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut discard = CollectorSink::new();

        sort.push(create_test_chunk(&[2]), &mut discard).unwrap();
        sort.spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap();
        sort.push(create_test_chunk(&[1]), &mut discard).unwrap();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sort.finalize(&mut PanickingSink)
        }))
        .unwrap_err();

        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"deterministic downstream sort sink panic")
        );
        assert!(sort.buffer.rows().is_empty());
        assert_eq!(sort.buffer.granted_bytes(), 0);
        let external_bytes = sort
            .external_sort
            .as_ref()
            .unwrap()
            .checked_total_granted_bytes()
            .unwrap();
        assert_eq!(manager.allocated(), external_bytes);
        assert_eq!(sort.spill_state.as_ref().unwrap().usage(), external_bytes);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(spill_manager.spilled_bytes(), 0);

        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn codec_workspace_denial_is_typed_and_creates_no_spill_file() {
        let required = one_integer_row_charge();
        let manager = buffer_manager_with_exact_budget(required);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[1]), &mut sink).unwrap();
        assert_eq!(manager.allocated(), required);

        let error = sort
            .spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap_err();

        assert!(matches!(
            error,
            OperatorError::ResidentMemory(MemoryGrantError::LimitExceeded { .. })
        ));
        assert_eq!(manager.allocated(), 0);
        assert_eq!(sort.spill_state.as_ref().unwrap().usage(), 0);
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(std::fs::read_dir(temp_dir.path()).unwrap().count(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn run_catalog_denial_is_typed_and_precedes_spill_file_creation() {
        let row_bytes = one_integer_row_charge();
        let codec_bytes = {
            let calibration_manager = buffer_manager_with_exact_budget(1024 * 1024);
            let resources =
                crate::execution::QueryResourceContext::new(calibration_manager).unwrap();
            let grant = resources.try_allocate(0).unwrap();
            let directory = tempfile::TempDir::new().unwrap();
            let spill_manager = Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .build()
                    .unwrap(),
            );
            let mut external = ExternalSort::new_accounted(spill_manager, 1, vec![], grant);
            external
                .spill_sorted_run_accounted(&[vec![Value::Int64(1)]])
                .unwrap();
            external.workspace_granted_bytes()
        };
        let manager = buffer_manager_with_exact_budget(row_bytes + codec_bytes);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();
        sort.push(create_test_chunk(&[1]), &mut sink).unwrap();

        let error = sort
            .spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap_err();

        assert!(matches!(
            error,
            OperatorError::ResidentMemory(MemoryGrantError::LimitExceeded { .. })
        ));
        assert!(sort.buffer.rows().is_empty());
        assert_eq!(sort.buffer.granted_bytes(), 0);
        assert_eq!(manager.allocated(), codec_bytes);
        assert_eq!(sort.spill_state.as_ref().unwrap().usage(), codec_bytes);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(std::fs::read_dir(temp_dir.path()).unwrap().count(), 0);

        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn writer_denial_is_resident_memory_and_reconciles_external_only_usage() {
        let row_bytes = one_integer_row_charge();
        let (external_base, writer_bytes) = {
            let calibration_manager = buffer_manager_with_exact_budget(1024 * 1024);
            let resources =
                crate::execution::QueryResourceContext::new(calibration_manager).unwrap();
            let grant = resources.try_allocate(0).unwrap();
            let directory = tempfile::TempDir::new().unwrap();
            let spill_manager = Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .build()
                    .unwrap(),
            );
            let mut external = ExternalSort::new_accounted(spill_manager, 1, vec![], grant);
            let external_bytes = Cell::new(external.total_granted_bytes());
            let retained_bytes = Cell::new(0);
            let peak = Cell::new(external_bytes.get());
            let observer = ExternalSortGrantObserver::new(&external_bytes, &retained_bytes, None)
                .tracking_peak(&peak);
            external
                .spill_sorted_run_accounted_observing(&[vec![Value::Int64(1)]], &observer)
                .unwrap();
            let base = external.total_granted_bytes();
            (base, peak.get().checked_sub(base).unwrap())
        };
        assert!(writer_bytes > 0);
        let budget = row_bytes
            .checked_add(external_base)
            .and_then(|bytes| bytes.checked_add(writer_bytes - 1))
            .unwrap();
        let manager = buffer_manager_with_exact_budget(budget);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();
        sort.push(create_test_chunk(&[1]), &mut sink).unwrap();
        assert_eq!(manager.allocated(), row_bytes);

        let error = sort
            .spill_current_buffer(Arc::clone(&spill_manager))
            .unwrap_err();

        assert!(matches!(
            error,
            OperatorError::ResidentMemory(MemoryGrantError::LimitExceeded { .. })
        ));
        assert!(sort.buffer.rows().is_empty());
        assert_eq!(sort.buffer.granted_bytes(), 0);
        let external = sort.external_sort.as_ref().unwrap();
        assert_eq!(external.writer_workspace_granted_bytes(), 0);
        assert_eq!(external.total_granted_bytes(), external_base);
        assert_eq!(manager.allocated(), external_base);
        assert_eq!(sort.spill_state.as_ref().unwrap().usage(), external_base);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(std::fs::read_dir(temp_dir.path()).unwrap().count(), 0);

        drop(sort);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn first_row_capacity_denial_does_not_create_an_empty_spill_run() {
        let required = one_integer_row_charge();
        let manager = buffer_manager_with_exact_budget(required - 1);
        let temp_dir = tempfile::TempDir::new().unwrap();
        let (resources, spill_manager) =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build_operator_resources(
                    Arc::clone(&manager),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();

        let error = sort.push(create_test_chunk(&[1]), &mut sink).unwrap_err();
        assert!(matches!(
            error,
            OperatorError::ResidentMemory(MemoryGrantError::LimitExceeded { .. })
        ));
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(manager.allocated(), 0);
        assert!(sort.buffer.is_empty());
        assert_eq!(sort.buffer.granted_bytes(), 0);
        assert!(sort.external_sort.is_none());
        assert_eq!(sort.spill_state.as_ref().unwrap().usage(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn pull_run_catalog_is_bounded_and_preserves_duplicate_rows() {
        let directory = tempfile::tempdir().unwrap();
        let memory = grafeo_common::memory::buffer::BufferManager::with_budget(2 << 20);
        let (resources, manager) =
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .build_operator_resources(
                    memory.clone(),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        for count in 1usize..=129 {
            sort.ingest_pull_chunk(&create_test_chunk(&[7])).unwrap();
            sort.flush_pull_batch().unwrap();
            assert_eq!(
                sort.external_sort.as_ref().unwrap().num_runs(),
                if count < 16 {
                    count
                } else {
                    count.count_ones() as usize
                }
            );
            assert!(manager.active_file_count() <= 16);
        }
        let mut cursor = sort.finish_pull_input().unwrap();
        let mut count = 0;
        while let Some(row) = cursor.next_owned_row().unwrap() {
            assert_eq!(row.values()[0], Value::Int64(7));
            count += 1;
            drop(row.into_released_grant());
            cursor.release_transferred_retained().unwrap();
        }
        assert_eq!(count, 129);
        drop(cursor);
        drop(sort);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(memory.allocated(), 0);
    }

    #[test]
    #[cfg(all(feature = "spill", any(target_os = "linux", target_os = "macos")))]
    fn scoped_registration_lives_through_empty_sort_finalize() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let buffer_manager = grafeo_common::memory::buffer::BufferManager::with_budget(1024 * 1024);
        let root = crate::execution::spill::RootedSpillFixture::new(temp_dir.path())
            .root()
            .unwrap();
        let resources = crate::execution::QueryResourceContext::with_spill_root(
            Arc::clone(&buffer_manager),
            &root,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        resources.ensure_spill_manager().unwrap().unwrap();
        let baseline = buffer_manager.stats().consumer_count;
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();
        assert_eq!(buffer_manager.stats().consumer_count, baseline + 1);

        sort.finalize(&mut sink).unwrap();
        assert_eq!(
            buffer_manager.stats().consumer_count,
            baseline + 1,
            "finalize must not end the registration while the operator remains alive"
        );

        drop(sort);
        assert_eq!(buffer_manager.stats().consumer_count, baseline);
    }
    #[cfg(feature = "spill")]
    #[test]
    fn pull_sort_terminal_cleanup_preserves_typed_primary_and_hostile_payload() {
        struct Payload {
            formats: Arc<AtomicUsize>,
            drops: Arc<AtomicUsize>,
        }
        impl std::fmt::Debug for Payload {
            fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                out.write_str("Payload")
            }
        }
        impl std::fmt::Display for Payload {
            fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.formats.fetch_add(1, AtomicOrdering::Relaxed);
                panic!("cleanup payload must not be formatted during transport");
            }
        }
        impl std::error::Error for Payload {}
        impl Drop for Payload {
            fn drop(&mut self) {
                self.drops.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
        struct Fault {
            armed: AtomicBool,
            formats: Arc<AtomicUsize>,
            drops: Arc<AtomicUsize>,
        }
        impl crate::execution::spill::SpillIo for Fault {
            fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
                // One armed delete yields one error; all owned payload fields
                // are fixed-size references to pre-existing shared counters.
                Some(
                    std::mem::size_of::<Payload>()
                        + std::mem::size_of::<(
                            std::io::ErrorKind,
                            Box<dyn std::error::Error + Send + Sync>,
                        )>(),
                )
            }
            fn check(
                &self,
                operation: crate::execution::spill::SpillIoOperation,
            ) -> std::io::Result<()> {
                if operation == crate::execution::spill::SpillIoOperation::Delete
                    && self.armed.swap(false, AtomicOrdering::Relaxed)
                {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        Payload {
                            formats: self.formats.clone(),
                            drops: self.drops.clone(),
                        },
                    ));
                }
                Ok(())
            }
        }
        let formats = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let fault = Arc::new(Fault {
            armed: AtomicBool::new(false),
            formats: formats.clone(),
            drops: drops.clone(),
        });
        let directory = tempfile::tempdir().unwrap();
        let memory = buffer_manager_with_exact_budget(64 << 20);
        let (resources, manager) =
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .io(fault.clone())
                .build_operator_resources(
                    memory.clone(),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        assert!(
            sort.pull_failure_publisher.is_none(),
            "resident input must not allocate failure transport"
        );
        sort.ingest_pull_chunk(&create_test_chunk(&[2, 1])).unwrap();
        assert!(sort.pull_failure_publisher.is_none());
        sort.flush_pull_batch().unwrap();
        assert_eq!(manager.active_file_count(), 1);
        fault.armed.store(true, AtomicOrdering::Relaxed);
        let error = sort.finish_pull_failure(OperatorError::ColumnNotFound("original".into()));
        let OperatorError::ClassifiedAccountedFailure {
            classification,
            authority,
        } = &error
        else {
            panic!("typed primary/cleanup carrier missing")
        };
        assert!(matches!(
            classification,
            AccountedFailureClassification::ColumnNotFound
        ));
        assert!(authority.inspect::<PullSortFailure, _>(|failure| {
            matches!(&failure.primary, Some(OperatorError::ColumnNotFound(name)) if name == "original") && failure.operation.is_some()
        }).unwrap());
        assert_eq!(formats.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(drops.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(manager.active_file_count(), 1);
        fault.armed.store(false, AtomicOrdering::Relaxed);
        drop(sort);
        assert_eq!(manager.active_file_count(), 0);
        let payload_grants = authority
            .inspect::<PullSortFailure, _>(|failure| failure.payload_granted_bytes())
            .unwrap();
        assert_eq!(
            memory.allocated(),
            authority.granted_bytes() + payload_grants
        );
        drop(error);
        assert_eq!(drops.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(memory.allocated(), 0);
    }
    #[cfg(feature = "spill")]
    #[test]
    fn pull_sort_rejects_unbounded_hook_before_destructive_spill() {
        struct Unbounded(AtomicUsize);
        impl crate::execution::spill::SpillIo for Unbounded {
            fn check(&self, _: crate::execution::spill::SpillIoOperation) -> std::io::Result<()> {
                self.0.fetch_add(1, AtomicOrdering::Relaxed);
                Ok(())
            }
        }
        let io = Arc::new(Unbounded(AtomicUsize::new(0)));
        let directory = tempfile::tempdir().unwrap();
        let memory = buffer_manager_with_exact_budget(1 << 20);
        let (resources, manager) =
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .io(io.clone())
                .build_operator_resources(
                    memory.clone(),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        sort.ingest_pull_chunk(&create_test_chunk(&[2, 1])).unwrap();
        assert!(matches!(
            sort.flush_pull_batch(),
            Err(OperatorError::ResidentContainerInvariant {
                container: "pull sort failure transport",
                ..
            })
        ));
        assert_eq!(io.0.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(sort.pull_buffered_rows(), 2);
        drop(sort);
        assert_eq!(memory.allocated(), 0);
    }

    #[cfg(feature = "spill")]
    #[test]
    fn pull_sort_rejects_unbounded_provider_without_allocating_failure_transport() {
        struct Unbounded;
        impl crate::execution::spill::SpillRecordProvider for Unbounded {
            fn seals(&self) -> bool {
                false
            }
            fn begin_file(
                &self,
                _: crate::execution::spill::SpillFileIdentity,
            ) -> std::io::Result<Box<dyn crate::execution::spill::OpenSpillRecord>> {
                panic!("provider work must follow allocation-bound admission")
            }
        }
        let directory = tempfile::tempdir().unwrap();
        let memory = buffer_manager_with_exact_budget(1 << 20);
        let (resources, manager) =
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(Unbounded),
                    crate::execution::spill::SpillFrameLimits::format_max(),
                )
                .build_operator_resources(
                    memory.clone(),
                    crate::execution::QueryExecutionControl::new().token(),
                )
                .unwrap();
        let mut sort = SpillableSortPushOperator::with_resource_context(
            vec![SortKey::ascending(0)],
            resources,
        )
        .unwrap();
        sort.ingest_pull_chunk(&create_test_chunk(&[2, 1])).unwrap();
        let before = memory.allocated();
        assert!(matches!(
            sort.flush_pull_batch(),
            Err(OperatorError::ResidentContainerInvariant {
                container: "pull sort failure transport",
                ..
            })
        ));
        assert_eq!(memory.allocated(), before);
        assert!(sort.pull_failure_publisher.is_none());
        assert_eq!(manager.active_file_count(), 0);
        drop(sort);
        assert_eq!(memory.allocated(), 0);
    }

    #[cfg(feature = "spill")]
    #[test]
    fn pull_sort_hook_unwind_and_secondary_cleanup_keep_declared_authority() {
        const PAYLOAD_BYTES: usize = 1024;
        #[derive(Debug)]
        struct Payload(Box<[u8; PAYLOAD_BYTES]>);
        impl std::fmt::Display for Payload {
            fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                panic!("opaque payload formatted")
            }
        }
        impl std::error::Error for Payload {}
        impl Drop for Payload {
            fn drop(&mut self) {
                assert_eq!(self.0[0], 7);
            }
        }
        struct Fault {
            panic_create: bool,
            remaining_deletes: AtomicUsize,
        }
        impl crate::execution::spill::SpillIo for Fault {
            fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
                // At most two payloads escape, and each owns one exact array,
                // its fixed Box field, and the erased I/O owner.
                Some(
                    2 * (PAYLOAD_BYTES
                        + std::mem::size_of::<Payload>()
                        + std::mem::size_of::<(
                            std::io::ErrorKind,
                            Box<dyn std::error::Error + Send + Sync>,
                        )>()),
                )
            }
            fn check(
                &self,
                operation: crate::execution::spill::SpillIoOperation,
            ) -> std::io::Result<()> {
                if self.panic_create
                    && operation == crate::execution::spill::SpillIoOperation::Create
                {
                    std::panic::panic_any(Payload(Box::new([7; PAYLOAD_BYTES])));
                }
                if operation == crate::execution::spill::SpillIoOperation::Delete
                    && self
                        .remaining_deletes
                        .fetch_update(AtomicOrdering::Relaxed, AtomicOrdering::Relaxed, |left| {
                            left.checked_sub(1)
                        })
                        .is_ok()
                {
                    return Err(std::io::Error::other(Payload(Box::new([7; PAYLOAD_BYTES]))));
                }
                Ok(())
            }
        }
        for panic_create in [false, true] {
            let fault = Arc::new(Fault {
                panic_create,
                remaining_deletes: AtomicUsize::new(0),
            });
            let directory = tempfile::tempdir().unwrap();
            let memory = buffer_manager_with_exact_budget(4 << 20);
            let (resources, manager) =
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .io(fault.clone())
                    .build_operator_resources(
                        memory.clone(),
                        crate::execution::QueryExecutionControl::new().token(),
                    )
                    .unwrap();
            let mut sort = SpillableSortPushOperator::with_resource_context(
                vec![SortKey::ascending(0)],
                resources,
            )
            .unwrap();
            sort.ingest_pull_chunk(&create_test_chunk(&[2, 1])).unwrap();
            let bound =
                crate::execution::spill::SpillIo::qualified_sort_hook_workspace_bound(&*fault)
                    .unwrap();
            if panic_create {
                let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    sort.flush_pull_batch()
                }))
                .unwrap_err();
                drop(sort);
                // The parent may retain this opaque panic indefinitely. Its
                // declared hook authority must survive even a later drop.
                assert!(memory.allocated() >= bound);
                drop(payload);
                assert!(memory.allocated() >= bound);
            } else {
                sort.flush_pull_batch().unwrap();
                sort.ingest_pull_chunk(&create_test_chunk(&[4, 3])).unwrap();
                sort.flush_pull_batch().unwrap();
                assert_eq!(manager.active_file_count(), 2);
                fault.remaining_deletes.store(2, AtomicOrdering::Relaxed);
                let failure = sort.finish_pull_cleanup(None).unwrap_err();
                drop(sort);
                drop(failure);
                assert_eq!(manager.active_file_count(), 0);
                assert_eq!(
                    memory.allocated(),
                    bound,
                    "the forgotten secondary diagnostic retains the shared whole-sort allowance"
                );
            }
        }
    }
}
