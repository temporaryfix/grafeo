//! Sort operator for ordering results.
//!
//! This module provides:
//! - `SortOperator`: Orders results by one or more columns

use crate::execution::vector::ValueVector;
use crate::execution::{QueryResourceContext, QueryResourceContextError};
use grafeo_common::memory::buffer::{MemoryGrant, MemoryGrantError};
use std::cmp::Ordering;
use std::mem::size_of;

use grafeo_common::types::{LogicalType, Value};

use super::value_utils::{compare_values_total, compare_values_with_nulls};
use super::{Operator, OperatorError, OperatorPipelineDecomposition, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::chunk::DataChunkBuilder;

mod fallible;
#[cfg(test)]
mod semantic_tests;
pub(crate) use fallible::try_stable_sort_by;

/// Allocation-free failure from an admitted semantic comparison.
#[derive(Debug, Clone)]
pub enum SemanticComparisonError {
    /// The comparison scratch envelope could not be admitted.
    Resource(MemoryGrantError),
    /// The provider rejected an invalid comparison input.
    Invalid(&'static str),
    /// The query was cancelled before comparison completed.
    Cancelled(crate::execution::QueryCancellationError),
}

impl From<SemanticComparisonError> for OperatorError {
    fn from(error: SemanticComparisonError) -> Self {
        match error {
            SemanticComparisonError::Resource(error) => Self::ResidentMemory(error),
            SemanticComparisonError::Invalid(message) => Self::ResidentContainerInvariant {
                container: "semantic comparison",
                message,
            },
            SemanticComparisonError::Cancelled(error) => Self::QueryCancelled(error),
        }
    }
}

/// Exact ascending value ordering with an explicit temporary allocation contract.
///
/// Implementations must compute `scratch_bytes` without allocating and include
/// every simultaneous temporary allocation used by `compare`. Comparison must
/// retain no scratch or allocated error payload after returning. This is an
/// implementor contract, not a sandbox for arbitrary callbacks. Callers reserve
/// the checked bound before invoking comparison and keep it alive through return.
pub trait AccountedValueComparator: Send + Sync {
    /// Checked upper bound for temporary bytes used by this comparison.
    ///
    /// # Errors
    /// Returns a resource error for unrepresentable bounds or a concrete
    /// semantic error for inputs the provider cannot compare.
    fn scratch_bytes(
        &self,
        left: Option<&Value>,
        right: Option<&Value>,
    ) -> Result<usize, SemanticComparisonError>;

    /// Exact ascending order, including the provider's unbound/null semantics.
    ///
    /// # Errors
    /// Returns the first semantic, resource, or cancellation failure without
    /// retaining temporary allocations or an allocated error payload.
    fn compare(
        &self,
        left: Option<&Value>,
        right: Option<&Value>,
    ) -> Result<Ordering, SemanticComparisonError>;
}

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SortDirection {
    /// Ascending order (smallest first).
    Ascending,
    /// Descending order (largest first).
    Descending,
}

/// Null ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NullOrder {
    /// Nulls come first.
    NullsFirst,
    /// Nulls come last.
    NullsLast,
}

/// A sort key specification.
#[derive(Debug, Clone)]
pub struct SortKey {
    /// Column index to sort by.
    pub column: usize,
    /// Sort direction.
    pub direction: SortDirection,
    /// Null ordering.
    pub null_order: NullOrder,
}

impl SortKey {
    /// Creates a new sort key with ascending order.
    pub fn ascending(column: usize) -> Self {
        Self {
            column,
            direction: SortDirection::Ascending,
            null_order: NullOrder::NullsLast,
        }
    }

    /// Creates a new sort key with descending order.
    pub fn descending(column: usize) -> Self {
        Self {
            column,
            direction: SortDirection::Descending,
            null_order: NullOrder::NullsLast,
        }
    }

    /// Sets the null ordering.
    pub fn with_null_order(mut self, null_order: NullOrder) -> Self {
        self.null_order = null_order;
        self
    }
}

/// Compares two sort-key values in their final requested order.
///
/// `NULLS FIRST` / `NULLS LAST` describes final placement independently of
/// direction. Descending order therefore reverses only a comparison between
/// two non-null values, never the already-resolved null placement.
pub(super) fn compare_sort_values(
    a: &Option<Value>,
    b: &Option<Value>,
    direction: SortDirection,
    null_order: NullOrder,
) -> Ordering {
    match (a, b) {
        (Some(a), Some(b)) if !matches!(a, Value::Null) && !matches!(b, Value::Null) => {
            let ordering = compare_values_total(a, b);
            match direction {
                SortDirection::Ascending => ordering,
                SortDirection::Descending => ordering.reverse(),
            }
        }
        _ => compare_values_with_nulls(a, b, null_order),
    }
}

fn into_push_sort_key(key: SortKey) -> super::push::SortKey {
    super::push::SortKey {
        column: key.column,
        direction: match key.direction {
            SortDirection::Ascending => super::push::SortDirection::Ascending,
            SortDirection::Descending => super::push::SortDirection::Descending,
        },
        null_order: match key.null_order {
            NullOrder::NullsFirst => super::push::NullOrder::First,
            NullOrder::NullsLast => super::push::NullOrder::Last,
        },
    }
}

/// A row reference for sorting.
#[derive(Debug, Clone)]
struct SortRow {
    /// Index of the chunk this row belongs to.
    chunk_index: usize,
    /// Row index within the chunk.
    row_index: usize,
}

#[cfg(feature = "spill")]
struct PendingSortInput {
    // Raw storage drops before its grant; the containing operator drops the
    // spill state before this pair, including during unwinding.
    chunk: DataChunk,
    _grant: MemoryGrant,
}

/// Sort operator.
///
/// Materializes all input and sorts by the specified keys.
pub struct SortOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Sort keys.
    sort_keys: Vec<SortKey>,
    semantic_comparator: Option<std::sync::Arc<dyn AccountedValueComparator>>,
    comparison_grant: Option<MemoryGrant>,
    /// Physical input width, including hidden expression keys.
    input_width: Option<usize>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Materialized chunks.
    chunks: Vec<DataChunk>,
    /// Sorted row references.
    sorted_rows: Vec<SortRow>,
    /// The child has reported EOF; finishing cannot truncate an active child.
    input_complete: bool,
    /// Explicit cleanup fences every subsequent input/output operation.
    cleaned: bool,
    /// Whether sorting is complete.
    sort_complete: bool,
    /// Current position in output.
    output_position: usize,
    // Physical inputs and private sorting capacities drop before their grant.
    resident_bytes: usize,
    #[cfg(feature = "spill")]
    cursor: Option<parking_lot::Mutex<crate::execution::spill::OwnedExactSortCursor>>,
    #[cfg(feature = "spill")]
    spill: Option<super::push::SpillableSortPushOperator>,
    #[cfg(feature = "spill")]
    pending_input: Option<PendingSortInput>,
    // Inherited shared payloads in the spill buffer drop before this grant.
    #[cfg(feature = "spill")]
    buffered_input_grant: Option<MemoryGrant>,
    resident_grant: Option<MemoryGrant>,
    output_grant: Option<MemoryGrant>,
    resources: Option<QueryResourceContext>,
    failure: Option<OperatorError>,
}

fn sort_phase_error() -> OperatorError {
    OperatorError::ResidentContainerInvariant {
        container: "owned sort phase",
        message: "input, preparation and cleanup steps are out of order",
    }
}

impl SortOperator {
    /// Creates a new sort operator.
    pub fn new(
        child: Box<dyn Operator>,
        sort_keys: Vec<SortKey>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            child,
            sort_keys,
            output_schema,
            semantic_comparator: None,
            comparison_grant: None,
            input_width: None,
            chunks: Vec::new(),
            sorted_rows: Vec::new(),
            input_complete: false,
            cleaned: false,
            sort_complete: false,
            output_position: 0,
            resident_bytes: 0,
            resident_grant: None,
            output_grant: None,
            resources: None,
            #[cfg(feature = "spill")]
            cursor: None,
            #[cfg(feature = "spill")]
            spill: None,
            #[cfg(feature = "spill")]
            pending_input: None,
            #[cfg(feature = "spill")]
            buffered_input_grant: None,
            failure: None,
        }
    }

    /// Installs exact semantic ordering for the pull operator and its spill runs.
    /// The provider orders ascending, including unbound values; DESC reverses
    /// the complete result. Hidden input columns remain owned until emission.
    pub fn with_semantic_comparator(
        mut self,
        comparator: std::sync::Arc<dyn AccountedValueComparator>,
    ) -> Self {
        self.semantic_comparator = Some(comparator);
        self
    }

    /// Decomposes this operator into its child and sort keys for push-based conversion.
    pub fn into_parts(self) -> (Box<dyn Operator>, Vec<SortKey>) {
        (self.child, self.sort_keys)
    }

    /// Resolves the schema carried by rows from one source chunk.
    ///
    /// `Any` is the planner's open type for projected values. Preserve the
    /// source's typed edge-list marker there so a modifier cannot erase path
    /// provenance. Other values remain `Any`; in particular, an ordinary
    /// integer list is never promoted to `List(Edge)` from its contents.
    fn effective_schema(&self, source: &DataChunk) -> Vec<LogicalType> {
        self.output_schema
            .iter()
            .enumerate()
            .map(|(index, configured)| {
                if !matches!(configured, LogicalType::Any) {
                    return configured.clone();
                }
                let Some(source_type) = source.column(index).map(|column| column.data_type())
                else {
                    return configured.clone();
                };
                if matches!(source_type, LogicalType::Node | LogicalType::Edge)
                    || matches!(
                        source_type,
                        LogicalType::List(item) if item.as_ref() == &LogicalType::Edge
                    )
                {
                    source_type.clone()
                } else {
                    configured.clone()
                }
            })
            .collect()
    }

    fn schema_matches(&self, source: &DataChunk, expected: &[LogicalType]) -> bool {
        self.output_schema
            .iter()
            .enumerate()
            .all(|(index, configured)| {
                let Some(expected_type) = expected.get(index) else {
                    return false;
                };
                if !matches!(configured, LogicalType::Any) {
                    return configured == expected_type;
                }
                match source.column(index).map(|column| column.data_type()) {
                    Some(source_type)
                        if matches!(source_type, LogicalType::Node | LogicalType::Edge)
                            || matches!(
                                source_type,
                                LogicalType::List(item) if item.as_ref() == &LogicalType::Edge
                            ) =>
                    {
                        expected_type == source_type
                    }
                    _ => expected_type == configured,
                }
            })
    }

    /// Materializes and sorts the input through the same bounded steps used by
    /// an owned scheduling job.
    fn sort(&mut self) -> Result<(), OperatorError> {
        self.check_cancelled()?;
        while !self.input_complete && self.ingest_next_input_chunk_inner()? {}
        self.finish_input_inner()
    }

    /// Pulls and admits at most one child chunk, retaining all inherited payload
    /// grants in this owner before it can be enqueued for another job.
    /// Returns `false` only after the child reports EOF; empty chunks count as a step.
    ///
    /// # Errors
    /// Returns input, cancellation, admission or spill failure. Finished or
    /// explicitly cleaned owners cannot ingest more input.
    pub fn ingest_next_input_chunk(&mut self) -> Result<bool, OperatorError> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        if self.cleaned || self.sort_complete {
            return Err(sort_phase_error());
        }
        if self.input_complete {
            return Ok(false);
        }
        if let Err(error) = self.check_cancelled() {
            return Err(self.fail(error));
        }
        match self.ingest_next_input_chunk_inner() {
            Ok(more) => Ok(more),
            Err(error) => Err(self.fail(error)),
        }
    }

    fn ingest_next_input_chunk_inner(&mut self) -> Result<bool, OperatorError> {
        // Check each delivered chunk before admission. Rechecking the
        // deadline at both sides of every singleton pull duplicates clock
        // reads; EOF and all spill/publication boundaries check separately.
        let Some(mut chunk) = self.child.next()? else {
            self.input_complete = true;
            return Ok(false);
        };
        self.check_cancelled()?;
        if chunk.is_empty() {
            return Ok(true);
        }
        chunk.clear_zone_hints();
        self.input_width = Some(chunk.column_count());
        let bytes = if self.resources.is_some() {
            chunk
                .output_retained_bytes()
                .map_err(|error| OperatorError::ResidentAllocation(error.to_string()))?
        } else {
            0
        };
        #[cfg(feature = "spill")]
        if self.spill.is_some() || self.should_transition(bytes)? {
            if self.spill.is_none() {
                self.trim_resident_grant()?;
            }
            let resources = self.resources.as_ref().ok_or_else(sort_invariant)?;
            let batch_limit = resources.buffer_manager().budget() / 4;
            if self.buffered_input_grant.is_none() {
                self.buffered_input_grant =
                    Some(resources.try_allocate(0).map_err(sort_context_error)?);
            }
            let grant = self
                .buffered_input_grant
                .as_mut()
                .ok_or_else(sort_invariant)?;
            let held = grant.size();
            if let Err(error) = grant.try_resize(sort_add(held, bytes)?) {
                if held == 0 {
                    return Err(error.into());
                }
                // Flush with the old grant still field-owned; retry once
                // after the inherited buffer pointees have been released.
                self.spill
                    .as_mut()
                    .ok_or_else(sort_invariant)?
                    .flush_pull_batch()?;
                self.buffered_input_grant
                    .as_mut()
                    .ok_or_else(sort_invariant)?
                    .try_resize(bytes)?;
            }
            self.pending_input = Some(PendingSortInput {
                chunk,
                _grant: self
                    .buffered_input_grant
                    .take()
                    .ok_or_else(sort_invariant)?,
            });
            if self.spill.is_none() {
                self.transition_to_spill()?;
            }
            let pending = self.pending_input.as_ref().ok_or_else(sort_invariant)?;
            let spill = self.spill.as_mut().ok_or_else(sort_invariant)?;
            spill.ingest_pull_chunk(&pending.chunk)?;
            if pending._grant.size() >= batch_limit || spill.pull_buffered_rows() == 0 {
                spill.flush_pull_batch()?;
                self.pending_input = None;
            } else {
                // The row buffer can share payloads with the raw chunk.
                // Release the raw vectors, but retain their conservative
                // physical grant until the entire buffered batch is gone.
                let PendingSortInput { chunk, _grant } =
                    self.pending_input.take().ok_or_else(sort_invariant)?;
                drop(chunk);
                self.buffered_input_grant = Some(_grant);
            }
            return Ok(true);
        }
        self.admit_chunk(&chunk, bytes)?;
        let chunk_idx = self.chunks.len();
        for row_idx in chunk.selected_indices() {
            self.sorted_rows.push(SortRow {
                chunk_index: chunk_idx,
                row_index: row_idx,
            });
        }
        self.chunks.push(chunk);
        Ok(true)
    }

    /// Prepares output after observed input EOF, using the existing resident
    /// stable sort or bounded spill reductions. Repeated successful calls are inert.
    ///
    /// # Errors
    /// Rejects unfinished input and returns cancellation, sorting or cleanup failure.
    pub fn finish_input(&mut self) -> Result<(), OperatorError> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        if self.cleaned || !self.input_complete {
            return Err(sort_phase_error());
        }
        if self.sort_complete {
            return Ok(());
        }
        match self.finish_input_inner() {
            Ok(()) => Ok(()),
            Err(error) => Err(self.fail(error)),
        }
    }

    fn finish_input_inner(&mut self) -> Result<(), OperatorError> {
        self.check_cancelled()?;
        #[cfg(feature = "spill")]
        if let Some(spill) = &mut self.spill {
            self.cursor = Some(parking_lot::Mutex::new(spill.finish_pull_input()?));
            self.buffered_input_grant = None;
            self.sort_complete = true;
            return Ok(());
        }

        // Input growth is complete: release optional reservation headroom
        // before stable sorting and subsequent output admission.
        self.trim_resident_grant()?;
        // Sort the row references
        let chunks = &self.chunks;
        let sort_keys = &self.sort_keys;

        if let Some(comparator) = &self.semantic_comparator {
            let grant = &mut self.comparison_grant;
            let resources = &self.resources;
            try_stable_sort_by(&mut self.sorted_rows, |a, b| {
                if let Some(resources) = resources {
                    resources.check_cancelled()?;
                }
                for key in sort_keys {
                    let left = chunks[a.chunk_index]
                        .column(key.column)
                        .and_then(|column| column.get_value(a.row_index));
                    let right = chunks[b.chunk_index]
                        .column(key.column)
                        .and_then(|column| column.get_value(b.row_index));
                    let bytes = comparator.scratch_bytes(left.as_ref(), right.as_ref())?;
                    if let Some(grant) = grant.as_mut()
                        && bytes > grant.size()
                    {
                        grant.try_resize(bytes)?;
                    }
                    let ordering = comparator.compare(left.as_ref(), right.as_ref())?;
                    let ordering = match key.direction {
                        SortDirection::Ascending => ordering,
                        SortDirection::Descending => ordering.reverse(),
                    };
                    if ordering != Ordering::Equal {
                        return Ok(ordering);
                    }
                }
                Ok(Ordering::Equal)
            })?;
            self.comparison_grant = None;
        } else {
            self.sorted_rows.sort_by(|a, b| {
                for key in sort_keys {
                    let chunk_a = &chunks[a.chunk_index];
                    let chunk_b = &chunks[b.chunk_index];

                    let val_a = chunk_a
                        .column(key.column)
                        .and_then(|c| c.get_value(a.row_index));
                    let val_b = chunk_b
                        .column(key.column)
                        .and_then(|c| c.get_value(b.row_index));

                    let cmp = compare_sort_values(&val_a, &val_b, key.direction, key.null_order);

                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                }
                Ordering::Equal
            });
        }

        self.check_cancelled()?;
        self.sort_complete = true;
        Ok(())
    }

    /// Reads one already-prepared bounded output chunk. The caller must consume
    /// and drop it before the next output call changes this owner's output grant.
    /// This method never materializes an unfinished child.
    ///
    /// # Errors
    /// Returns a phase error before preparation, or cancellation/output failure.
    pub fn next_prepared_output(&mut self) -> OperatorResult {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        if self.cleaned || !self.sort_complete {
            return Err(sort_phase_error());
        }
        self.next()
    }

    /// Explicitly releases the retained sort owners without resetting the child.
    /// A primary failure keeps precedence over any cleanup failures.
    ///
    /// # Errors
    /// Returns the supplied primary, retained failure, or concrete cleanup failure.
    pub fn finish_owned_cleanup(
        &mut self,
        primary: Option<OperatorError>,
    ) -> Result<(), OperatorError> {
        let primary = self.failure.clone().or(primary);
        if self.cleaned {
            return primary.map_or(Ok(()), Err);
        }
        self.cleaned = true;
        if let Some(primary) = primary {
            return Err(self.fail(primary));
        }
        #[cfg(feature = "spill")]
        {
            let mut error = self
                .cursor
                .as_mut()
                .and_then(|cursor| cursor.get_mut().finish_early_stop().err());
            if let Some(spill) = &mut self.spill {
                error = spill.finish_pull_cleanup(error).err();
            }
            if let Some(error) = error {
                // The exact cursor already consumed its terminal publisher.
                // Retire it without a second explicit cleanup/publication.
                self.retire_owned_storage();
                self.failure = Some(error.clone());
                return Err(error);
            }
        }
        self.retire_owned_storage();
        Ok(())
    }

    fn retire_owned_storage(&mut self) {
        #[cfg(feature = "spill")]
        {
            self.cursor = None;
            self.spill = None;
            self.pending_input = None;
            self.buffered_input_grant = None;
        }
        self.chunks = Vec::new();
        self.sorted_rows = Vec::new();
        self.resident_bytes = 0;
        self.resident_grant = None;
        self.output_grant = None;
        self.comparison_grant = None;
    }

    fn check_cancelled(&self) -> Result<(), OperatorError> {
        if let Some(resources) = &self.resources {
            resources.check_cancelled()?;
        }
        Ok(())
    }

    fn admit_chunk(&mut self, chunk: &DataChunk, payload: usize) -> Result<(), OperatorError> {
        let chunks = sort_capacity(self.chunks.capacity(), sort_add(self.chunks.len(), 1)?)?;
        let rows = sort_capacity(
            self.sorted_rows.capacity(),
            sort_add(self.sorted_rows.len(), chunk.row_count())?,
        )?;
        let resident = sort_add(self.resident_bytes, payload)?;
        // Full old/new vector capacities overlap during growth. A full row-ref
        // capacity additionally covers the stable sort's temporary scratch.
        let final_bytes = sort_add(
            resident,
            sort_add(
                sort_mul(chunks, size_of::<DataChunk>())?,
                sort_mul(rows, 2 * size_of::<SortRow>())?,
            )?,
        )?;
        let peak = sort_add(
            final_bytes,
            sort_add(
                if chunks > self.chunks.capacity() {
                    sort_mul(self.chunks.capacity(), size_of::<DataChunk>())?
                } else {
                    0
                },
                if rows > self.sorted_rows.capacity() {
                    sort_mul(self.sorted_rows.capacity(), size_of::<SortRow>())?
                } else {
                    0
                },
            )?,
        )?;
        const RESERVATION_HEADROOM: usize = 64 << 10;
        if let Some(grant) = &mut self.resident_grant
            && peak > grant.size()
        {
            let target = peak.checked_add(RESERVATION_HEADROOM).unwrap_or(peak);
            // Headroom is optional. A query that fits the exact allocation
            // peak must still succeed when the larger reservation is denied.
            if grant.try_resize(target).is_err() {
                grant.try_resize(peak)?;
            }
        }
        if chunks > self.chunks.capacity() {
            self.chunks
                .try_reserve_exact(chunks - self.chunks.len())
                .map_err(sort_allocation)?;
        }
        if rows > self.sorted_rows.capacity() {
            self.sorted_rows
                .try_reserve_exact(rows - self.sorted_rows.len())
                .map_err(sort_allocation)?;
        }
        if self.chunks.capacity() > chunks || self.sorted_rows.capacity() > rows {
            return Err(sort_invariant());
        }
        if let Some(grant) = &mut self.resident_grant {
            let retained = final_bytes
                .checked_add(RESERVATION_HEADROOM)
                .unwrap_or(final_bytes);
            if grant.size() > retained {
                grant.try_resize(retained)?;
            }
        }
        self.resident_bytes = resident;
        Ok(())
    }

    fn resident_requirement(&self) -> Result<usize, OperatorError> {
        sort_add(
            self.resident_bytes,
            sort_add(
                sort_mul(self.chunks.capacity(), size_of::<DataChunk>())?,
                sort_mul(self.sorted_rows.capacity(), 2 * size_of::<SortRow>())?,
            )?,
        )
    }

    fn trim_resident_grant(&mut self) -> Result<(), OperatorError> {
        let required = self.resident_requirement()?;
        if let Some(grant) = &mut self.resident_grant {
            grant.try_resize(required)?;
        }
        Ok(())
    }

    #[cfg(feature = "spill")]
    fn should_transition(&self, bytes: usize) -> Result<bool, OperatorError> {
        let Some(resources) = &self.resources else {
            return Ok(false);
        };
        if !resources.has_spill_manager() {
            return Ok(false);
        }
        // Reserve headroom for the existing row buffer, framing workspace and
        // the raw input during the admitted columnar-to-row transition.
        let held = self.resident_requirement()?;
        Ok(
            sort_add(held, bytes)? > resources.buffer_manager().budget() / 4
                || resources.should_spill(),
        )
    }

    #[cfg(feature = "spill")]
    fn transition_to_spill(&mut self) -> Result<(), OperatorError> {
        let resources = self.resources.as_ref().ok_or_else(sort_invariant)?.clone();
        let key_bytes = sort_mul(self.sort_keys.len(), size_of::<super::push::SortKey>())?;
        let grant = self.resident_grant.as_mut().ok_or_else(sort_invariant)?;
        grant.try_resize(sort_add(grant.size(), key_bytes)?)?;
        let mut keys = Vec::new();
        keys.try_reserve_exact(self.sort_keys.len())
            .map_err(sort_allocation)?;
        if keys.capacity() > self.sort_keys.len() {
            return Err(sort_invariant());
        }
        keys.extend(self.sort_keys.iter().cloned().map(into_push_sort_key));
        let mut spill =
            super::push::SpillableSortPushOperator::with_resource_context(keys, resources.clone())
                .map_err(sort_context_error)?;
        if let Some(comparator) = &self.semantic_comparator {
            spill = spill.with_semantic_comparator(comparator.clone(), &resources)?;
        }
        self.spill = Some(spill);
        self.comparison_grant = None;
        // References have not been sorted yet. Preserve input order for stable
        // ties, and retain the full columnar grant until every source is gone.
        self.sorted_rows = Vec::new();
        let chunks = std::mem::take(&mut self.chunks);
        for chunk in chunks {
            self.check_cancelled()?;
            let spill = self.spill.as_mut().ok_or_else(sort_invariant)?;
            spill.ingest_pull_chunk(&chunk)?;
        }
        self.spill
            .as_mut()
            .ok_or_else(sort_invariant)?
            .flush_pull_batch()?;
        self.resident_bytes = 0;
        if let Some(grant) = &mut self.resident_grant {
            grant.try_resize(key_bytes)?;
        }
        Ok(())
    }

    fn fail(&mut self, primary: OperatorError) -> OperatorError {
        if let Some(error) = &self.failure {
            return error.clone();
        }
        #[cfg(feature = "spill")]
        let primary = {
            let mut primary = primary;
            if let Some(cursor) = &mut self.cursor {
                let cursor = cursor.get_mut();
                primary =
                    cursor.finish_operator_failure(primary, None, "pull sort terminal cleanup");
            }
            if let Some(spill) = &mut self.spill {
                primary = spill.finish_pull_failure(primary);
            }
            primary
        };
        self.retire_owned_storage();
        self.failure = Some(primary.clone());
        primary
    }

    #[cfg(feature = "spill")]
    fn next_spilled(&mut self) -> OperatorResult {
        let cursor = self.cursor.as_mut().ok_or_else(sort_invariant)?.get_mut();
        let row = match cursor.next_owned_row() {
            Ok(row) => row,
            Err(error) => {
                let error = cursor.finish_stream_failure(error, "pull sort merge cleanup");
                self.cursor = None;
                return Err(error);
            }
        };
        let Some(row) = row else {
            return Ok(None);
        };
        let output_result = (|| {
            let shape = crate::execution::accounted_chunk::SortRowShape::with_edge_trailer(
                self.input_width.ok_or_else(sort_invariant)?,
            );
            let values = row.values();
            let mask = shape
                .edge_mask(values)
                .map_err(|message| OperatorError::Execution(message.into()))?;
            let mut bytes = sort_mul(
                self.output_schema.len(),
                size_of::<ValueVector>() + size_of::<LogicalType>() + size_of::<Value>() + 16,
            )?;
            for value in values {
                bytes = sort_add(
                    bytes,
                    value.retained_size_bytes().ok_or_else(sort_overflow)?,
                )?;
            }
            for ty in &self.output_schema {
                bytes = sort_add(bytes, sort_schema_bytes(ty, 0)?)?;
            }
            let resources = self.resources.as_ref().ok_or_else(sort_invariant)?;
            let grant = resources.try_allocate(bytes).map_err(sort_context_error)?;
            let mut columns = Vec::new();
            columns
                .try_reserve_exact(self.output_schema.len())
                .map_err(sort_allocation)?;
            for (index, configured) in self.output_schema.iter().enumerate() {
                let ty = if *configured == LogicalType::Any {
                    crate::execution::accounted_chunk::SortRowShape::column_type(mask, index)
                        .logical_type()
                } else {
                    configured.clone()
                };
                let mut column = ValueVector::try_with_capacity(ty, 1)
                    .map_err(|error| OperatorError::ResidentAllocation(error.to_string()))?;
                column
                    .try_reserve_validity_capacity(1)
                    .map_err(|error| OperatorError::ResidentAllocation(error.to_string()))?;
                column.push_value(values[index].clone());
                columns.push(column);
            }
            let output = DataChunk::new(columns);
            if output
                .output_retained_bytes()
                .map_err(|error| OperatorError::ResidentAllocation(error.to_string()))?
                > grant.size()
            {
                return Err(sort_invariant());
            }
            Ok::<_, OperatorError>((output, grant))
        })();
        drop(row.into_released_grant());
        let cursor = self.cursor.as_mut().ok_or_else(sort_invariant)?.get_mut();
        let release = cursor.release_transferred_retained();
        let (output, grant) = match output_result {
            Ok(output) => {
                if let Err(error) = release {
                    let error = cursor.finish_stream_failure(error, "pull sort output release");
                    self.cursor = None;
                    return Err(error);
                }
                output
            }
            Err(primary) => {
                let error = cursor.finish_operator_failure(
                    primary,
                    release
                        .err()
                        .map(|error| (error, "pull sort output release")),
                    "pull sort output construction cleanup",
                );
                self.cursor = None;
                return Err(error);
            }
        };
        self.output_grant = Some(grant);
        self.check_cancelled()?;
        Ok(Some(output))
    }
}

fn sort_overflow() -> OperatorError {
    OperatorError::ResidentMemory(MemoryGrantError::ArithmeticOverflow {
        current_bytes: usize::MAX,
        additional_bytes: 1,
    })
}
fn sort_invariant() -> OperatorError {
    OperatorError::ResidentContainerInvariant {
        container: "pull sort",
        message: "retained capacity or lifecycle invariant",
    }
}
fn sort_add(a: usize, b: usize) -> Result<usize, OperatorError> {
    a.checked_add(b).ok_or_else(sort_overflow)
}
fn sort_mul(a: usize, b: usize) -> Result<usize, OperatorError> {
    a.checked_mul(b).ok_or_else(sort_overflow)
}
fn sort_capacity(capacity: usize, needed: usize) -> Result<usize, OperatorError> {
    if needed <= capacity {
        Ok(capacity)
    } else {
        needed
            .max(4)
            .checked_next_power_of_two()
            .ok_or_else(sort_overflow)
    }
}
fn sort_allocation(source: std::collections::TryReserveError) -> OperatorError {
    OperatorError::ResidentContainerAllocation {
        container: "pull sort",
        source,
    }
}
fn sort_context_error(error: QueryResourceContextError) -> OperatorError {
    match error {
        QueryResourceContextError::Memory(error) => error.into(),
        other => OperatorError::Execution(other.to_string()),
    }
}
fn sort_schema_bytes(ty: &LogicalType, depth: usize) -> Result<usize, OperatorError> {
    if depth > 256 {
        return Err(sort_overflow());
    }
    let heap = match ty {
        LogicalType::List(item) => sort_schema_bytes(item, depth + 1)?,
        LogicalType::Map { key, value } => sort_add(
            sort_schema_bytes(key, depth + 1)?,
            sort_schema_bytes(value, depth + 1)?,
        )?,
        LogicalType::Struct(fields) => {
            let mut bytes = sort_mul(fields.capacity(), size_of::<(String, LogicalType)>())?;
            for (name, ty) in fields {
                bytes = sort_add(
                    bytes,
                    sort_add(name.capacity(), sort_schema_bytes(ty, depth + 1)?)?,
                )?;
            }
            bytes
        }
        _ => 0,
    };
    sort_add(size_of::<LogicalType>(), heap)
}

impl SortOperator {
    fn next_inner(&mut self) -> OperatorResult {
        self.check_cancelled()?;
        if !self.sort_complete {
            self.sort()?;
        }

        #[cfg(feature = "spill")]
        if self.cursor.is_some() {
            return self.next_spilled();
        }
        if self.output_position >= self.sorted_rows.len() {
            return Ok(None);
        }

        let first_row = &self.sorted_rows[self.output_position];
        let first_source = &self.chunks[first_row.chunk_index];
        let output_grant = if let Some(resources) = &self.resources {
            let mut bytes = sort_mul(
                self.output_schema.len(),
                size_of::<ValueVector>() + 2048 * (size_of::<Value>() + 1),
            )?;
            for (index, configured) in self.output_schema.iter().enumerate() {
                bytes = sort_add(bytes, sort_mul(sort_schema_bytes(configured, 0)?, 2)?)?;
                if let Some(column) = first_source.column(index) {
                    bytes = sort_add(
                        bytes,
                        sort_mul(sort_schema_bytes(column.data_type(), 0)?, 2)?,
                    )?;
                }
            }
            // Resident inputs retain every shared Value pointee through this
            // output handoff. Only the new vectors and cloned schemas need a
            // second grant; scanning/recharging each shared value is redundant.
            Some(resources.try_allocate(bytes).map_err(sort_context_error)?)
        } else {
            None
        };
        let effective_schema = self.effective_schema(first_source);
        let mut builder = DataChunkBuilder::with_capacity(&effective_schema, 2048);

        while self.output_position < self.sorted_rows.len() && !builder.is_full() {
            let row_ref = &self.sorted_rows[self.output_position];
            let source_chunk = &self.chunks[row_ref.chunk_index];
            if !self.schema_matches(source_chunk, &effective_schema) {
                break;
            }

            // Copy all columns
            for col_idx in 0..self.output_schema.len() {
                if let (Some(src_col), Some(dst_col)) =
                    (source_chunk.column(col_idx), builder.column_mut(col_idx))
                {
                    if let Some(value) = src_col.get_value(row_ref.row_index) {
                        dst_col.push_value(value);
                    } else {
                        dst_col.push_value(Value::Null);
                    }
                }
            }

            builder.advance_row();
            self.output_position += 1;
        }

        if builder.row_count() > 0 {
            let output = builder.finish();
            if let Some(grant) = &output_grant {
                let mut private_bytes = output
                    .observed_column_capacity_bytes()
                    .map_err(|error| OperatorError::ResidentAllocation(error.to_string()))?;
                for column in 0..output.column_count() {
                    let column = output.column(column).ok_or_else(sort_invariant)?;
                    private_bytes =
                        sort_add(private_bytes, sort_schema_bytes(column.data_type(), 0)?)?;
                }
                if private_bytes > grant.size() {
                    return Err(sort_invariant());
                }
            }
            self.output_grant = output_grant;
            self.check_cancelled()?;
            Ok(Some(output))
        } else {
            Ok(None)
        }
    }
}

impl Operator for SortOperator {
    fn next(&mut self) -> OperatorResult {
        if self.cleaned {
            return self
                .failure
                .clone()
                .map_or_else(|| Err(sort_phase_error()), Err);
        }
        self.output_grant = None;
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        match self.next_inner() {
            Ok(result) => Ok(result),
            Err(error) => Err(self.fail(error)),
        }
    }

    fn reset(&mut self) {
        self.child.reset();
        #[cfg(feature = "spill")]
        let cleanup_error = {
            let error = self
                .cursor
                .as_mut()
                .and_then(|cursor| cursor.get_mut().finish_early_stop().err());
            self.cursor = None;
            self.spill = None;
            self.pending_input = None;
            self.buffered_input_grant = None;
            error
        };
        self.chunks = Vec::new();
        self.sorted_rows = Vec::new();
        self.resident_bytes = 0;
        self.resident_grant = None;
        self.output_grant = None;
        self.resources = None;
        self.comparison_grant = None;
        self.input_width = None;
        self.failure = None;
        #[cfg(feature = "spill")]
        {
            self.failure = cleanup_error;
        }
        self.input_complete = false;
        self.cleaned = false;
        self.sort_complete = false;
        self.output_position = 0;
    }

    fn name(&self) -> &'static str {
        "Sort"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn install_resource_context(
        &mut self,
        resources: &crate::execution::QueryResourceContext,
    ) -> Result<(), crate::execution::QueryResourceContextError> {
        if self.resources.is_some() {
            self.reset();
        }
        self.child.install_resource_context(resources)?;
        self.resident_grant = Some(resources.try_allocate(0)?);
        if self.semantic_comparator.is_some() {
            self.comparison_grant = Some(resources.try_allocate(0)?);
        }
        self.resources = Some(resources.clone());
        Ok(())
    }

    fn decompose_pipeline_with_resources(
        self: Box<Self>,
        resources: &crate::execution::QueryResourceContext,
    ) -> Result<OperatorPipelineDecomposition, crate::execution::QueryResourceContextError> {
        if self.semantic_comparator.is_some() {
            return Ok(OperatorPipelineDecomposition::Boundary(self));
        }
        let (child, sort_keys) = (*self).into_parts();
        let push_keys = sort_keys.into_iter().map(into_push_sort_key).collect();
        #[cfg(feature = "spill")]
        if resources.has_spill_manager() {
            return Ok(OperatorPipelineDecomposition::unary(
                child,
                Box::new(
                    super::push::SpillableSortPushOperator::with_resource_context(
                        push_keys,
                        resources.clone(),
                    )?,
                ),
            ));
        }
        Ok(OperatorPipelineDecomposition::unary(
            child,
            Box::new(super::push::SortPushOperator::with_resource_context(
                push_keys,
                resources.clone(),
            )?),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use crate::execution::selection::SelectionVector;

    struct MockOperator {
        chunks: Vec<DataChunk>,
        original_chunks: Vec<DataChunk>,
        position: usize,
    }

    impl MockOperator {
        fn new(chunks: Vec<DataChunk>) -> Self {
            Self {
                original_chunks: chunks.clone(),
                chunks,
                position: 0,
            }
        }
    }

    impl Operator for MockOperator {
        fn next(&mut self) -> OperatorResult {
            if self.position < self.chunks.len() {
                let chunk = std::mem::replace(&mut self.chunks[self.position], DataChunk::empty());
                self.position += 1;
                Ok(Some(chunk))
            } else {
                Ok(None)
            }
        }

        fn reset(&mut self) {
            self.chunks.clone_from(&self.original_chunks);
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "Mock"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    fn create_unsorted_chunk() -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String]);

        let data = [(3i64, "cherry"), (1, "apple"), (4, "date"), (2, "banana")];

        for (num, text) in data {
            builder.column_mut(0).unwrap().push_int64(num);
            builder.column_mut(1).unwrap().push_string(text);
            builder.advance_row();
        }

        builder.finish()
    }

    fn create_value_chunk(values: &[Value]) -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for value in values {
            builder.column_mut(0).unwrap().push_value(value.clone());
            builder.advance_row();
        }
        builder.finish()
    }

    fn collect_first_column(operator: &mut dyn Operator) -> Vec<Value> {
        let mut values = Vec::new();
        while let Some(chunk) = operator.next().unwrap() {
            for row in chunk.selected_indices() {
                values.push(
                    chunk
                        .column(0)
                        .unwrap()
                        .get_value(row)
                        .expect("test row has one value"),
                );
            }
        }
        values
    }

    #[test]
    fn test_sort_ascending() {
        let mock = MockOperator::new(vec![create_unsorted_chunk()]);

        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64, LogicalType::String],
        );

        let mut results = Vec::new();
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                let num = chunk.column(0).unwrap().get_int64(row).unwrap();
                let text = chunk
                    .column(1)
                    .unwrap()
                    .get_string(row)
                    .unwrap()
                    .to_string();
                results.push((num, text));
            }
        }

        assert_eq!(
            results,
            vec![
                (1, "apple".to_string()),
                (2, "banana".to_string()),
                (3, "cherry".to_string()),
                (4, "date".to_string()),
            ]
        );
    }

    #[test]
    fn test_sort_descending() {
        let mock = MockOperator::new(vec![create_unsorted_chunk()]);

        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            vec![LogicalType::Int64, LogicalType::String],
        );

        let mut results = Vec::new();
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                let num = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(num);
            }
        }

        assert_eq!(results, vec![4, 3, 2, 1]);
    }

    #[test]
    fn final_null_placement_is_direction_independent() {
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
                    null_order: NullOrder::NullsFirst,
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
                    null_order: NullOrder::NullsLast,
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
                    null_order: NullOrder::NullsFirst,
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
                    null_order: NullOrder::NullsLast,
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
            let mock = MockOperator::new(vec![create_value_chunk(&input)]);
            let mut sort = SortOperator::new(Box::new(mock), vec![key], vec![LogicalType::Int64]);
            assert_eq!(collect_first_column(&mut sort), expected);
        }
    }

    #[test]
    fn test_sort_by_string() {
        let mock = MockOperator::new(vec![create_unsorted_chunk()]);

        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(1)], // Sort by string column
            vec![LogicalType::Int64, LogicalType::String],
        );

        let mut results = Vec::new();
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                let text = chunk
                    .column(1)
                    .unwrap()
                    .get_string(row)
                    .unwrap()
                    .to_string();
                results.push(text);
            }
        }

        assert_eq!(
            results,
            vec![
                "apple".to_string(),
                "banana".to_string(),
                "cherry".to_string(),
                "date".to_string(),
            ]
        );
    }

    #[test]
    fn test_sort_empty_input() {
        let mock = MockOperator::new(vec![]);

        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );

        assert!(sort.next().unwrap().is_none());
    }

    #[test]
    fn test_sort_already_sorted() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for v in [1i64, 2, 3, 4] {
            builder.column_mut(0).unwrap().push_int64(v);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );

        let mut results = Vec::new();
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                results.push(chunk.column(0).unwrap().get_int64(row).unwrap());
            }
        }

        assert_eq!(results, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_sort_duplicate_values() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for v in [3i64, 1, 3, 2, 1] {
            builder.column_mut(0).unwrap().push_int64(v);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );

        let mut results = Vec::new();
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                results.push(chunk.column(0).unwrap().get_int64(row).unwrap());
            }
        }

        assert_eq!(results, vec![1, 1, 2, 3, 3]);
    }

    #[test]
    fn test_sort_multi_key() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::String, LogicalType::Int64]);
        let data = [("b", 2i64), ("a", 3), ("b", 1), ("a", 1)];
        for (s, n) in data {
            builder.column_mut(0).unwrap().push_string(s);
            builder.column_mut(1).unwrap().push_int64(n);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0), SortKey::ascending(1)],
            vec![LogicalType::String, LogicalType::Int64],
        );

        let mut results = Vec::new();
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                let s = chunk
                    .column(0)
                    .unwrap()
                    .get_string(row)
                    .unwrap()
                    .to_string();
                let n = chunk.column(1).unwrap().get_int64(row).unwrap();
                results.push((s, n));
            }
        }

        assert_eq!(
            results,
            vec![
                ("a".to_string(), 1),
                ("a".to_string(), 3),
                ("b".to_string(), 1),
                ("b".to_string(), 2),
            ]
        );
    }

    #[test]
    fn test_sort_multiple_chunks() {
        // Two separate chunks that get merged during sort
        let mut b1 = DataChunkBuilder::new(&[LogicalType::Int64]);
        b1.column_mut(0).unwrap().push_int64(5);
        b1.advance_row();
        b1.column_mut(0).unwrap().push_int64(1);
        b1.advance_row();
        let chunk1 = b1.finish();

        let mut b2 = DataChunkBuilder::new(&[LogicalType::Int64]);
        b2.column_mut(0).unwrap().push_int64(3);
        b2.advance_row();
        b2.column_mut(0).unwrap().push_int64(2);
        b2.advance_row();
        let chunk2 = b2.finish();

        let mock = MockOperator::new(vec![chunk1, chunk2]);
        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );

        let mut results = Vec::new();
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                results.push(chunk.column(0).unwrap().get_int64(row).unwrap());
            }
        }

        assert_eq!(results, vec![1, 2, 3, 5]);
    }

    #[test]
    fn test_sort_reverse_sorted() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for v in [4i64, 3, 2, 1] {
            builder.column_mut(0).unwrap().push_int64(v);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );

        let mut results = Vec::new();
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                results.push(chunk.column(0).unwrap().get_int64(row).unwrap());
            }
        }

        assert_eq!(results, vec![1, 2, 3, 4]);
    }

    #[test]
    fn qualified_columnar_sort_retains_grants_and_resets_for_fresh_query() {
        use grafeo_common::memory::buffer::BufferManager;
        let manager = BufferManager::with_budget(2 << 20);
        let resources = QueryResourceContext::new(manager.clone()).unwrap();
        let mut sort = SortOperator::new(
            Box::new(MockOperator::new(vec![create_unsorted_chunk()])),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64, LogicalType::String],
        );
        sort.install_resource_context(&resources).unwrap();
        let chunk = sort.next().unwrap().unwrap();
        assert_eq!(chunk.column(0).unwrap().get_int64(0), Some(1));
        assert!(manager.allocated() > 0);
        drop(chunk);
        sort.reset();
        assert_eq!(manager.allocated(), 0);
        let fresh = QueryResourceContext::new(manager.clone()).unwrap();
        sort.install_resource_context(&fresh).unwrap();
        assert_eq!(sort.next().unwrap().unwrap().row_count(), 4);
        sort.reset();
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn resident_sort_duplicate_selection_shares_large_payload_under_budget() {
        use crate::execution::selection::SelectionVector;
        use grafeo_common::memory::buffer::BufferManager;
        let manager = BufferManager::with_budget(1 << 20);
        let resources = QueryResourceContext::new(manager.clone()).unwrap();
        let mut column = ValueVector::with_capacity(LogicalType::String, 1);
        column.push_string("x".repeat(512 << 10));
        let mut input = DataChunk::new(vec![column]);
        let mut selected = SelectionVector::new_empty();
        for _ in 0..4 {
            selected.push(0);
        }
        input.set_selection(selected);
        let mut sort = SortOperator::new(
            Box::new(MockOperator::new(vec![input])),
            vec![SortKey::ascending(0)],
            vec![LogicalType::String],
        );
        sort.install_resource_context(&resources).unwrap();
        let output = sort.next().unwrap().unwrap();
        assert_eq!(output.row_count(), 4);
        for row in 0..4 {
            let Value::String(value) = output.column(0).unwrap().get_value(row).unwrap() else {
                panic!("selected string must retain its type");
            };
            assert_eq!(value.len(), 512 << 10);
            assert!(value.bytes().all(|byte| byte == b'x'));
        }
        assert!(manager.allocated() >= 512 << 10);
        assert!(manager.allocated() <= 1 << 20);
        drop(output);
        sort.reset();
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn resident_reservation_headroom_falls_back_and_is_released_before_output() {
        use grafeo_common::memory::buffer::BufferManager;
        // Normalize physical capacities to the cloned inputs used below.
        let input = create_unsorted_chunk().clone();
        let required = input.output_retained_bytes().unwrap()
            + 4 * size_of::<DataChunk>()
            + 8 * size_of::<SortRow>();
        for budget in [required - 1, required, required + (128 << 10)] {
            // Exercise the exact admission boundary, rather than the
            // default hard limit of 95% of the configured pool budget.
            let mut config =
                grafeo_common::memory::buffer::BufferManagerConfig::with_budget(budget);
            config.soft_limit_fraction = 1.0;
            config.evict_limit_fraction = 1.0;
            config.hard_limit_fraction = 1.0;
            let manager = BufferManager::new(config);
            let resources = QueryResourceContext::new(manager.clone()).unwrap();
            let mut sort = SortOperator::new(
                Box::new(MockOperator::new(vec![input.clone()])),
                vec![SortKey::ascending(0)],
                vec![LogicalType::Int64, LogicalType::String],
            );
            sort.install_resource_context(&resources).unwrap();
            let result = sort.sort();
            if budget < required {
                assert!(matches!(result, Err(OperatorError::ResidentMemory(_))));
            } else {
                result.unwrap();
                assert_eq!(sort.resident_grant.as_ref().unwrap().size(), required);
                assert_eq!(manager.allocated(), required);
                assert_eq!(sort.sorted_rows.len(), 4);
            }
            sort.reset();
            assert_eq!(manager.allocated(), 0);
        }
    }

    #[test]
    fn qualified_columnar_sort_denies_resident_growth_before_publication() {
        use grafeo_common::memory::buffer::BufferManager;
        let manager = BufferManager::with_budget(64);
        let resources = QueryResourceContext::new(manager.clone()).unwrap();
        let mut sort = SortOperator::new(
            Box::new(MockOperator::new(vec![create_unsorted_chunk()])),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64, LogicalType::String],
        );
        sort.install_resource_context(&resources).unwrap();
        assert!(matches!(sort.next(), Err(OperatorError::ResidentMemory(_))));
        assert!(sort.next().is_err(), "terminal denial must stay sticky");
        sort.reset();
        assert_eq!(manager.allocated(), 0);
    }

    #[cfg(feature = "spill")]
    fn pressure_sort_chunks() -> Vec<DataChunk> {
        (0..64)
            .rev()
            .map(|batch| {
                let mut chunk = DataChunk::with_capacity(&[LogicalType::Int64], 1024);
                for row in (0..1024).rev() {
                    chunk
                        .column_mut(0)
                        .unwrap()
                        .push_int64((batch * 1024 + row) / 4);
                }
                chunk.set_count(1024);
                chunk
            })
            .collect()
    }

    #[cfg(feature = "spill")]
    #[test]
    fn qualified_pull_sort_spills_stably_then_cancels_and_releases() {
        use crate::execution::QueryExecutionControl;
        use crate::execution::spill::SpillDiskQuota;
        use grafeo_common::memory::buffer::BufferManager;
        let directory = tempfile::tempdir().unwrap();
        let spill_fixture = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .quota(SpillDiskQuota::new(64 << 20));
        let manager = BufferManager::with_budget(2 << 20);
        let control = QueryExecutionControl::new();
        let cancel = control.cancellation_handle();
        let (resources, spill) = spill_fixture
            .build_operator_resources(manager.clone(), control.token())
            .unwrap();
        let mut sort = SortOperator::new(
            Box::new(MockOperator::new(pressure_sort_chunks())),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );
        sort.install_resource_context(&resources).unwrap();
        for expected in 0..16 {
            let chunk = sort.next().unwrap().unwrap();
            assert_eq!(chunk.row_count(), 1);
            assert_eq!(chunk.column(0).unwrap().get_int64(0), Some(expected / 4));
        }
        assert!(spill.active_file_count() > 0);
        cancel.cancel();
        assert!(sort.next().is_err());
        sort.reset();
        assert_eq!(spill.active_file_count(), 0);
        assert_eq!(manager.allocated(), 0);
    }

    #[cfg(feature = "spill")]
    #[test]
    fn pending_input_grant_survives_spill_failure_cleanup() {
        use crate::execution::spill::{
            CleartextSpillRecordProvider, SpillDiskQuota, SpillFrameLimits, SpillIo,
            SpillIoOperation,
        };
        use grafeo_common::memory::buffer::BufferManager;
        use std::sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        };
        struct FailWriteObserveCleanup {
            manager: Arc<BufferManager>,
            creates: AtomicUsize,
            minimum: AtomicUsize,
            observed: AtomicBool,
        }
        impl SpillIo for FailWriteObserveCleanup {
            fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
                // Only one write failure escapes this terminal test. Its
                // static string plus the I/O error's erased owner are fixed.
                Some(
                    std::mem::size_of::<&'static str>()
                        + std::mem::size_of::<(
                            std::io::ErrorKind,
                            Box<dyn std::error::Error + Send + Sync>,
                        )>(),
                )
            }
            fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
                if operation == SpillIoOperation::Create {
                    self.creates.fetch_add(1, Ordering::Relaxed);
                }
                // The second run cannot begin until the first has been
                // published. Fail its payload independently of rows per run,
                // leaving the first run for outer terminal cleanup to visit.
                if operation == SpillIoOperation::WritePayload
                    && self.creates.load(Ordering::Relaxed) >= 2
                {
                    return Err(std::io::Error::other("injected later-run write failure"));
                }
                let minimum = self.minimum.load(Ordering::Relaxed);
                if operation == SpillIoOperation::Delete && minimum != 0 {
                    assert!(
                        self.manager.allocated() >= minimum,
                        "input payload grant disappeared before spill cleanup callback"
                    );
                    self.observed.store(true, Ordering::Relaxed);
                }
                Ok(())
            }
        }
        let directory = tempfile::tempdir().unwrap();
        let manager = BufferManager::with_budget(2 << 20);
        let io = Arc::new(FailWriteObserveCleanup {
            manager: manager.clone(),
            creates: AtomicUsize::new(0),
            minimum: AtomicUsize::new(0),
            observed: AtomicBool::new(false),
        });
        let spill_fixture = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .provider(
                Arc::new(CleartextSpillRecordProvider),
                SpillFrameLimits::format_max(),
            )
            .io(io.clone())
            .quota(SpillDiskQuota::new(64 << 20));
        let (resources, spill) = spill_fixture
            .build_operator_resources(
                manager.clone(),
                crate::execution::QueryExecutionControl::new().token(),
            )
            .unwrap();
        let chunks = pressure_sort_chunks()
            .into_iter()
            .map(|chunk| {
                let mut strings = ValueVector::with_capacity(LogicalType::String, 1024);
                for _ in 0..1024 {
                    strings.push_string("shared-sort-input-payload");
                }
                DataChunk::new(vec![chunk.column(0).unwrap().clone(), strings])
            })
            .collect();
        let mut sort = SortOperator::new(
            Box::new(MockOperator::new(chunks)),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64, LogicalType::String],
        );
        sort.install_resource_context(&resources).unwrap();
        // Stop between propagation and the outer terminal cleanup, exactly
        // where the former local grant had already left scope.
        let primary = sort.sort().unwrap_err();
        assert!(
            primary
                .to_string()
                .contains("injected later-run write failure")
        );
        assert!(io.creates.load(Ordering::Relaxed) >= 2);
        assert!(
            spill.active_file_count() > 0,
            "a published run awaits cleanup"
        );
        let pending = sort
            .pending_input
            .as_ref()
            .expect("failure retains physical input owner");
        assert!(pending._grant.size() >= pending.chunk.output_retained_bytes().unwrap());
        io.minimum.store(pending._grant.size(), Ordering::Relaxed);
        let retained_primary = sort.fail(primary);
        assert!(
            io.observed.load(Ordering::Relaxed),
            "terminal cleanup must cross the observer"
        );
        assert!(sort.pending_input.is_none());
        assert_eq!(spill.active_file_count(), 0);
        sort.reset();
        let OperatorError::ClassifiedAccountedFailure { authority, .. } = &retained_primary else {
            panic!("qualified spill failure lost its admitted carrier");
        };
        let payload_grants = authority
            .inspect::<crate::execution::spill::PullSortFailure, _>(|failure| {
                failure.payload_granted_bytes()
            })
            .unwrap();
        assert_eq!(
            manager.allocated(),
            authority.granted_bytes() + payload_grants
        );
        drop(retained_primary);
        assert_eq!(manager.allocated(), 0);
    }

    #[cfg(feature = "spill")]
    #[test]
    fn qualified_pull_sort_zero_disk_quota_proves_transition() {
        use crate::execution::spill::SpillDiskQuota;
        use grafeo_common::memory::buffer::BufferManager;
        let directory = tempfile::tempdir().unwrap();
        let spill_fixture = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .quota(SpillDiskQuota::new(0));
        let manager = BufferManager::with_budget(2 << 20);
        let (resources, spill) = spill_fixture
            .build_operator_resources(
                manager.clone(),
                crate::execution::QueryExecutionControl::new().token(),
            )
            .unwrap();
        let mut sort = SortOperator::new(
            Box::new(MockOperator::new(pressure_sort_chunks())),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );
        sort.install_resource_context(&resources).unwrap();
        let error = sort.next().unwrap_err();
        assert!(
            error.to_string().contains("spill disk quota exceeded"),
            "{error}"
        );
        sort.reset();
        assert_eq!(spill.active_file_count(), 0);
        assert!(matches!(
            &error,
            OperatorError::ClassifiedAccountedFailure {
                classification: super::super::AccountedFailureClassification::StorageFull,
                ..
            }
        ));
        drop(error);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn test_sort_single_row() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(42);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );

        let mut count = 0;
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                assert_eq!(chunk.column(0).unwrap().get_int64(row), Some(42));
                count += 1;
            }
        }
        assert_eq!(count, 1);
    }

    #[test]
    fn test_sort_name() {
        let mock = MockOperator::new(vec![]);
        let sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );
        assert_eq!(sort.name(), "Sort");
    }

    #[test]
    fn test_sort_into_any() {
        let mock = MockOperator::new(vec![]);
        let op = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<SortOperator>().is_ok());
    }

    #[test]
    fn test_sort_into_parts() {
        let mock = MockOperator::new(vec![]);
        let op = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0), SortKey::descending(1)],
            vec![LogicalType::Int64, LogicalType::String],
        );
        let (mut child, sort_keys) = op.into_parts();
        assert_eq!(sort_keys.len(), 2);
        assert_eq!(sort_keys[0].column, 0);
        assert_eq!(sort_keys[1].column, 1);
        assert!(child.next().unwrap().is_none());
    }

    fn typed_edge_list_chunk(rows: &[(i64, i64)]) -> DataChunk {
        let edge_list = LogicalType::List(Box::new(LogicalType::Edge));
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, edge_list]);
        for &(key, edge_id) in rows {
            builder.column_mut(0).unwrap().push_int64(key);
            builder
                .column_mut(1)
                .unwrap()
                .push_value(Value::List(vec![Value::Int64(edge_id)].into()));
            builder.advance_row();
        }
        builder.finish()
    }

    fn ordinary_list_chunk(rows: &[(i64, i64)]) -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[
            LogicalType::Int64,
            LogicalType::List(Box::new(LogicalType::Int64)),
        ]);
        for &(key, value) in rows {
            builder.column_mut(0).unwrap().push_int64(key);
            builder
                .column_mut(1)
                .unwrap()
                .push_value(Value::List(vec![Value::Int64(value)].into()));
            builder.advance_row();
        }
        builder.finish()
    }

    fn collect_mixed_list_rows(operator: &mut dyn Operator) -> Vec<(LogicalType, i64, Value)> {
        let mut rows = Vec::new();
        while let Some(chunk) = operator.next().unwrap() {
            let schema = chunk.column(1).unwrap().data_type().clone();
            for row in chunk.selected_indices() {
                rows.push((
                    schema.clone(),
                    chunk.column(0).unwrap().get_int64(row).unwrap(),
                    chunk.column(1).unwrap().get_value(row).unwrap(),
                ));
            }
        }
        rows
    }

    fn any_list_chunk(rows: &[(i64, i64)]) -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::Any]);
        for &(key, value) in rows {
            builder.column_mut(0).unwrap().push_int64(key);
            builder
                .column_mut(1)
                .unwrap()
                .push_value(Value::List(vec![Value::Int64(value)].into()));
            builder.advance_row();
        }
        builder.finish()
    }

    fn typed_entity_chunk(entity_type: LogicalType, rows: &[(i64, i64)]) -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, entity_type]);
        for &(key, entity_id) in rows {
            builder.column_mut(0).unwrap().push_int64(key);
            builder
                .column_mut(1)
                .unwrap()
                .push_value(Value::Int64(entity_id));
            builder.advance_row();
        }
        builder.finish()
    }

    fn collect_typed_entity_rows(operator: &mut dyn Operator) -> Vec<(LogicalType, i64, i64)> {
        let mut rows = Vec::new();
        while let Some(chunk) = operator.next().unwrap() {
            let schema = chunk.column(1).unwrap().data_type().clone();
            for row in chunk.selected_indices() {
                rows.push((
                    schema.clone(),
                    chunk.column(0).unwrap().get_int64(row).unwrap(),
                    chunk
                        .column(1)
                        .unwrap()
                        .get_value(row)
                        .and_then(|value| value.as_int64())
                        .unwrap(),
                ));
            }
        }
        rows
    }

    #[test]
    fn sort_preserves_mixed_list_provenance_for_selected_rows_and_reset() {
        let mut edge = typed_edge_list_chunk(&[(2, 200), (1, 100), (99, 9900)]);
        edge.set_selection(SelectionVector::from_predicate(3, |row| row != 2));
        let ordinary = ordinary_list_chunk(&[(3, 300), (0, 0)]);
        let expected = vec![
            (
                LogicalType::Any,
                0,
                Value::List(vec![Value::Int64(0)].into()),
            ),
            (
                LogicalType::List(Box::new(LogicalType::Edge)),
                1,
                Value::List(vec![Value::Int64(100)].into()),
            ),
            (
                LogicalType::List(Box::new(LogicalType::Edge)),
                2,
                Value::List(vec![Value::Int64(200)].into()),
            ),
            (
                LogicalType::Any,
                3,
                Value::List(vec![Value::Int64(300)].into()),
            ),
        ];
        let mut sort = SortOperator::new(
            Box::new(MockOperator::new(vec![edge, ordinary])),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64, LogicalType::Any],
        );

        assert_eq!(collect_mixed_list_rows(&mut sort), expected);
        sort.reset();
        assert_eq!(collect_mixed_list_rows(&mut sort), expected);
    }

    #[test]
    fn sort_keeps_tied_mixed_rows_in_input_order_without_inference() {
        let mut edge = typed_edge_list_chunk(&[(7, 700)]);
        edge.set_selection(SelectionVector::new_all(1));
        let ordinary = any_list_chunk(&[(7, 7)]);
        let mut sort = SortOperator::new(
            Box::new(MockOperator::new(vec![edge, ordinary])),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64, LogicalType::Any],
        );

        let rows = collect_mixed_list_rows(&mut sort);
        assert_eq!(rows[0].0, LogicalType::List(Box::new(LogicalType::Edge)));
        assert_eq!(rows[0].2, Value::List(vec![Value::Int64(700)].into()));
        assert_eq!(rows[1].0, LogicalType::Any);
        assert_eq!(rows[1].2, Value::List(vec![Value::Int64(7)].into()));
    }

    #[test]
    fn sort_preserves_node_and_edge_provenance_for_selected_same_ids() {
        let mut node = typed_entity_chunk(LogicalType::Node, &[(2, 42), (9, 42)]);
        node.set_selection(SelectionVector::from_predicate(2, |row| row == 0));
        let edge = typed_entity_chunk(LogicalType::Edge, &[(1, 42)]);
        let mut sort = SortOperator::new(
            Box::new(MockOperator::new(vec![node, edge])),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64, LogicalType::Any],
        );

        assert_eq!(
            collect_typed_entity_rows(&mut sort),
            vec![(LogicalType::Edge, 1, 42), (LogicalType::Node, 2, 42)]
        );
    }
    #[test]
    fn owned_sort_steps_preserve_resident_rows_and_require_observed_eof() {
        fn assert_send<T: Send>() {}
        assert_send::<SortOperator>();
        struct EofOnce {
            chunks: std::collections::VecDeque<DataChunk>,
            eof: bool,
        }
        impl Operator for EofOnce {
            fn next(&mut self) -> OperatorResult {
                assert!(!self.eof, "the child must not be polled after observed EOF");
                let chunk = self.chunks.pop_front();
                self.eof = chunk.is_none();
                Ok(chunk)
            }
            fn reset(&mut self) {
                self.eof = false;
            }
            fn name(&self) -> &'static str {
                "EofOnce"
            }
            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }
        let manager = grafeo_common::memory::buffer::BufferManager::with_budget(1 << 20);
        let resources = QueryResourceContext::new(manager.clone()).unwrap();
        let mut sort = SortOperator::new(
            Box::new(EofOnce {
                chunks: [
                    create_unsorted_chunk(),
                    DataChunk::empty(),
                    create_unsorted_chunk(),
                ]
                .into(),
                eof: false,
            }),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64, LogicalType::String],
        );
        sort.install_resource_context(&resources).unwrap();
        assert!(sort.finish_input().is_err());
        assert!(sort.next_prepared_output().is_err());
        assert!(sort.chunks.is_empty());
        assert!(sort.ingest_next_input_chunk().unwrap());
        assert_eq!(sort.chunks.len(), 1);
        assert!(
            sort.ingest_next_input_chunk().unwrap(),
            "an empty delivered chunk is one step, not EOF"
        );
        assert_eq!(sort.chunks.len(), 1);
        assert!(sort.finish_input().is_err());
        assert!(sort.ingest_next_input_chunk().unwrap());
        assert!(!sort.ingest_next_input_chunk().unwrap());
        assert!(!sort.ingest_next_input_chunk().unwrap());
        // Ordinary next may finish already-exhausted stepped input, but must
        // never poll its child again or start a second sorter.
        let chunk = sort.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 8);
        let keys: Vec<_> = (0..8)
            .map(|row| chunk.column(0).unwrap().get_int64(row).unwrap())
            .collect();
        assert_eq!(keys, [1, 1, 2, 2, 3, 3, 4, 4]);
        drop(chunk);
        sort.finish_input().unwrap();
        assert!(sort.ingest_next_input_chunk().is_err());
        assert!(sort.next_prepared_output().unwrap().is_none());
        #[cfg(feature = "spill")]
        assert!(sort.spill.is_none());
        sort.finish_owned_cleanup(None).unwrap();
        sort.finish_owned_cleanup(None).unwrap();
        assert!(sort.next_prepared_output().is_err());
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn owned_sort_empty_input_and_between_step_cancellation_are_terminal() {
        let mut empty = SortOperator::new(Box::new(MockOperator::new(vec![])), vec![], vec![]);
        assert!(empty.finish_input().is_err());
        assert!(!empty.ingest_next_input_chunk().unwrap());
        empty.finish_input().unwrap();
        assert!(empty.next_prepared_output().unwrap().is_none());
        empty.finish_owned_cleanup(None).unwrap();

        let manager = grafeo_common::memory::buffer::BufferManager::with_budget(1 << 20);
        let control = crate::execution::QueryExecutionControl::new();
        let resources =
            QueryResourceContext::new_with_cancellation(manager.clone(), control.token()).unwrap();
        let mut sort = SortOperator::new(
            Box::new(MockOperator::new(vec![
                create_unsorted_chunk(),
                create_unsorted_chunk(),
            ])),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64, LogicalType::String],
        );
        sort.install_resource_context(&resources).unwrap();
        assert!(sort.ingest_next_input_chunk().unwrap());
        assert!(manager.allocated() > 0);
        control.cancellation_handle().cancel();
        assert!(matches!(
            sort.ingest_next_input_chunk(),
            Err(OperatorError::QueryCancelled(_))
        ));
        assert!(matches!(
            sort.finish_owned_cleanup(Some(OperatorError::ColumnNotFound("later".into()))),
            Err(OperatorError::QueryCancelled(_))
        ));
        assert!(matches!(
            sort.finish_owned_cleanup(None),
            Err(OperatorError::QueryCancelled(_))
        ));
        assert_eq!(manager.allocated(), 0);
    }
    #[cfg(feature = "spill")]
    #[test]
    fn owned_sort_failed_early_stop_is_published_once_and_remains_sticky() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        };
        struct DeleteFault {
            persistent: bool,
            armed: AtomicBool,
            attempts: AtomicUsize,
        }
        impl crate::execution::spill::SpillIo for DeleteFault {
            fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
                Some(0)
            }
            fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
                Some(0)
            }
            fn check(
                &self,
                operation: crate::execution::spill::SpillIoOperation,
            ) -> std::io::Result<()> {
                if operation == crate::execution::spill::SpillIoOperation::Delete
                    && self.armed.load(Ordering::Relaxed)
                {
                    let attempt = self.attempts.fetch_add(1, Ordering::Relaxed);
                    if self.persistent || attempt == 0 {
                        return Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied));
                    }
                }
                Ok(())
            }
        }
        for persistent in [false, true] {
            let fault = Arc::new(DeleteFault {
                persistent,
                armed: AtomicBool::new(false),
                attempts: AtomicUsize::new(0),
            });
            let directory = tempfile::tempdir().unwrap();
            let memory = grafeo_common::memory::buffer::BufferManager::with_budget(2 << 20);
            let (resources, manager) =
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .io(fault.clone())
                    .build_operator_resources(
                        memory.clone(),
                        crate::execution::QueryExecutionControl::new().token(),
                    )
                    .unwrap();
            let mut sort = SortOperator::new(
                Box::new(MockOperator::new(pressure_sort_chunks())),
                vec![SortKey::ascending(0)],
                vec![LogicalType::Int64],
            );
            sort.install_resource_context(&resources).unwrap();
            while sort.ingest_next_input_chunk().unwrap() {}
            sort.finish_input().unwrap();
            drop(sort.next_prepared_output().unwrap().unwrap());
            fault.armed.store(true, Ordering::Relaxed);
            let first = sort.finish_owned_cleanup(None).unwrap_err();
            let attempts = fault.attempts.load(Ordering::Relaxed);
            assert!(attempts > 0);
            if persistent {
                assert!(manager.active_file_count() > 0);
            } else {
                assert_eq!(
                    manager.active_file_count(),
                    0,
                    "a transient explicit cleanup failure must recover in the physical Drop backstop"
                );
                let OperatorError::ClassifiedAccountedFailure { authority, .. } = &first else {
                    panic!("escaped cleanup failure lost its accounted owner");
                };
                assert!(memory.allocated() >= authority.granted_bytes());
            }
            let second = sort
                .finish_owned_cleanup(Some(OperatorError::ColumnNotFound(
                    "replacement must not win".into(),
                )))
                .unwrap_err();
            assert_eq!(
                fault.attempts.load(Ordering::Relaxed),
                attempts,
                "a consumed terminal publisher cannot be reused"
            );
            assert!(matches!(
                &second,
                OperatorError::ClassifiedAccountedFailure {
                    classification: super::super::AccountedFailureClassification::Execution,
                    ..
                }
            ));
            fault.armed.store(false, Ordering::Relaxed);
            if persistent {
                // A failed final physical retirement intentionally quarantines
                // the complete owner graph. Its handles and real grants must stay
                // live; manager cleanup cannot bypass that retained authority.
                assert_eq!(
                    manager.cleanup().unwrap_err().kind(),
                    std::io::ErrorKind::WouldBlock
                );
                drop(first);
                drop(second);
                drop(sort);
                assert!(manager.active_file_count() > 0);
                assert!(
                    memory.allocated()
                        >= manager.active_file_count()
                            * std::mem::size_of::<crate::execution::spill::SpillFile>(),
                    "quarantined file owners must retain their admitted catalog backing"
                );
            } else {
                manager.cleanup().unwrap();
                assert_eq!(manager.active_file_count(), 0);
                drop(first);
                drop(second);
                drop(sort);
                assert_eq!(memory.allocated(), 0);
            }
        }
    }
}
