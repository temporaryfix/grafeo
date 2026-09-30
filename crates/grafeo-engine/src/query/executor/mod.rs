//! Query executor.
//!
//! Executes physical plans and produces results.
//!
//! Catalog procedures execute through `Session`, not a detached body adapter.
//! The removed direct procedure surface is not part of the 0.0.1 API:
//!
//! ```compile_fail,E0432
//! use grafeo_engine::query::executor::user_procedure::{ProcedureContext, UserProcedureOperator};
//! ```

#[cfg(all(
    feature = "gql",
    feature = "lpg",
    feature = "spill",
    feature = "async-storage"
))]
mod async_sort;
#[cfg(any(feature = "lpg", feature = "algos"))]
pub mod procedure_call;
#[cfg(all(any(feature = "lpg", feature = "algos"), feature = "gql"))]
pub(crate) mod procedure_output;
#[cfg(all(feature = "gql", feature = "lpg"))]
pub mod stream;
#[cfg(all(
    feature = "gql",
    feature = "lpg",
    feature = "spill",
    feature = "async-storage"
))]
pub use async_sort::{AsyncSortDispatch, PreparedAsyncSort};
use std::borrow::Cow;

use crate::config::AdaptiveConfig;
use crate::database::QueryResult;
use grafeo_common::grafeo_debug_span;
use grafeo_common::types::{LogicalType, Value};
use grafeo_common::utils::error::{Error, QueryError, Result};
use grafeo_core::execution::operators::{FactorizedOperator, Operator, OperatorError};
use grafeo_core::execution::{
    AdaptiveContext, AdaptiveSummary, CardinalityTrackingWrapper, DataChunk, Pipeline,
    QueryCancellationError, QueryCancellationToken, QueryExecutionCheckpoint,
    QueryExecutionControl, QueryResourceContext, SharedAdaptiveContext,
};

/// Limits on the rows and retained bytes returned by one eager query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResultLimits {
    /// Maximum number of returned rows; zero permits only an empty result.
    pub max_rows: usize,
    /// Maximum retained row storage and backing bytes, constrained by query grants.
    /// Column metadata is separately charged to the query grant.
    pub max_bytes: usize,
}

impl Default for ResultLimits {
    fn default() -> Self {
        Self {
            max_rows: 1_000_000,
            max_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Pure admission of a binding's copied output before statement publication.
///
/// The function must use only Rust data, must not reenter the database or a
/// foreign runtime, and may be called more than once for the same result.
pub type ResultAdmission = fn(&QueryResult, ResultLimits) -> Result<()>;

/// Caller-owned execution options consumed by one query.
pub struct ExecutionOptions {
    /// Lifecycle owner for cancellation, deadlines, and the commit fence.
    pub control: QueryExecutionControl,
    /// Named query language. `None` uses the session default.
    pub language: Option<String>,
    /// Explicit result limits. `None` uses the database configuration.
    pub result_limits: Option<ResultLimits>,
    /// Optional copied-output admission, run inside the statement rollback fence.
    /// Eager execution only: stream callers enforce their chunk conversion cap
    /// and must leave this unset.
    pub result_admission: Option<ResultAdmission>,
}

impl Default for ExecutionOptions {
    fn default() -> Self {
        Self {
            control: QueryExecutionControl::new(),
            language: None,
            result_limits: None,
            result_admission: None,
        }
    }
}

/// A formatting target that admits capacity before growing its String.
pub(crate) struct BoundedResultWriter {
    output: String,
    grant: grafeo_common::memory::buffer::MemoryGrant,
    resources: QueryResourceContext,
    max_bytes: usize,
    base_bytes: usize,
    error: Option<Error>,
}

impl BoundedResultWriter {
    pub(crate) fn new(
        resources: QueryResourceContext,
        max_bytes: usize,
        base_bytes: usize,
    ) -> Result<Self> {
        let grant = resources
            .try_allocate(base_bytes)
            .map_err(|error| result_full(error.to_string()))?;
        Ok(Self {
            output: String::new(),
            grant,
            resources,
            max_bytes,
            base_bytes,
            error: None,
        })
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    pub(crate) fn output(&self) -> &str {
        &self.output
    }

    fn append(&mut self, text: &str) -> Result<()> {
        self.resources
            .check_cancelled()
            .map_err(convert_cancellation_error)?;
        let needed = checked_result_bytes(self.output.len().checked_add(text.len()))?;
        if needed > self.max_bytes {
            return Err(result_full("formatted result exceeds byte limit"));
        }
        if needed > self.output.capacity() {
            let target = needed
                .max(self.output.capacity().saturating_mul(2))
                .min(self.max_bytes);
            self.grant
                .try_resize(checked_result_bytes(self.base_bytes.checked_add(target))?)
                .map_err(|error| result_full(error.to_string()))?;
            self.output
                .try_reserve_exact(target - self.output.len())
                .map_err(|error| result_full(error.to_string()))?;
            if self.output.capacity() > target {
                drop(std::mem::take(&mut self.output));
                return Err(result_full(
                    "formatted result allocator exceeded admitted capacity",
                ));
            }
        }
        self.output.push_str(text);
        Ok(())
    }

    pub(crate) fn resolve(&mut self, outcome: std::fmt::Result) -> Result<()> {
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        outcome.map_err(|_| Error::Internal("result formatter failed".into()))
    }
}

impl std::fmt::Write for BoundedResultWriter {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        if self.error.is_some() {
            return Err(std::fmt::Error);
        }
        self.append(text).map_err(|error| {
            self.error = Some(error);
            std::fmt::Error
        })
    }
}

/// Formats one generated report row with temporary and retained grants.
pub(crate) fn bounded_text_result(
    column: &str,
    resources: QueryResourceContext,
    limits: ResultLimits,
    format: impl FnOnce(&mut dyn std::fmt::Write) -> std::fmt::Result,
) -> Result<QueryResult> {
    if limits.max_rows == 0 {
        return Err(result_full("result row limit excludes report row"));
    }
    let metadata = checked_result_bytes(column.len().checked_add(size_of::<String>()))?;
    let _metadata = resources
        .try_allocate(metadata)
        .map_err(|error| result_full(error.to_string()))?;
    let columns = [column.to_owned()];
    let mut accumulator =
        ResultAccumulator::new(&columns, &[LogicalType::String], resources.clone(), limits)?;
    let mut writer = BoundedResultWriter::new(resources.clone(), limits.max_bytes, 0)?;
    let outcome = format(&mut writer);
    writer.resolve(outcome)?;
    // ArcStr's pinned header/alignment is bounded by this conservative 128-byte
    // allowance. Keep both the formatted String and its shared copy charged.
    let temporary =
        checked_result_bytes(writer.output.len().checked_add(128 + size_of::<Value>()))?;
    let _temporary = resources
        .try_allocate(temporary)
        .map_err(|error| result_full(error.to_string()))?;
    let value = Value::from(writer.output.as_str());
    accumulator.append_values(&[value])?;
    Ok(accumulator.finish())
}

/// Formats metadata-only command status under the query's resident grant.
#[cfg(any(test, feature = "gql"))]
pub(crate) fn bounded_status_result(
    resources: QueryResourceContext,
    format: std::fmt::Arguments<'_>,
) -> Result<QueryResult> {
    use std::fmt::Write;
    let base = size_of::<crate::database::ResultReservation>() + 2 * size_of::<usize>();
    let mut writer = BoundedResultWriter::new(resources, usize::MAX, base)?;
    let outcome = writer.write_fmt(format);
    writer.resolve(outcome)?;
    Ok(QueryResult::status(writer.output).with_result_reservation(writer.grant))
}

/// One output accumulator for pull, push, factorized, and public stream collection.
pub(crate) struct ResultAccumulator {
    result: QueryResult,
    resources: QueryResourceContext,
    limits: ResultLimits,
    row_bytes: usize,
    metadata_bytes: usize,
    rows: usize,
    types_captured: bool,
    dense_companion_bytes: usize,
}

fn result_full(message: impl Into<String>) -> Error {
    Error::Storage(grafeo_common::utils::error::StorageError::Full).with_context(message)
}

fn checked_result_bytes(bytes: Option<usize>) -> Result<usize> {
    bytes.ok_or_else(|| result_full("result retained-size arithmetic overflow"))
}

fn type_bytes(ty: &LogicalType) -> Result<usize> {
    let nested = match ty {
        LogicalType::List(item) => type_bytes(item)?,
        LogicalType::Map { key, value } => {
            checked_result_bytes(type_bytes(key)?.checked_add(type_bytes(value)?))?
        }
        LogicalType::Struct(fields) => {
            let mut bytes =
                checked_result_bytes(fields.len().checked_mul(size_of::<(String, LogicalType)>()))?;
            for (name, ty) in fields {
                bytes = checked_result_bytes(bytes.checked_add(name.len()))?;
                bytes = checked_result_bytes(bytes.checked_add(checked_result_bytes(
                    type_bytes(ty)?.checked_sub(size_of::<LogicalType>()),
                )?))?;
            }
            bytes
        }
        LogicalType::Any
        | LogicalType::Null
        | LogicalType::Bool
        | LogicalType::Int8
        | LogicalType::Int16
        | LogicalType::Int32
        | LogicalType::Int64
        | LogicalType::Float32
        | LogicalType::Float64
        | LogicalType::String
        | LogicalType::Bytes
        | LogicalType::Date
        | LogicalType::Time
        | LogicalType::Timestamp
        | LogicalType::Duration
        | LogicalType::ZonedTime
        | LogicalType::ZonedDatetime
        | LogicalType::Node
        | LogicalType::Edge
        | LogicalType::Path
        | LogicalType::Vector(_) => 0,
        _ => {
            return Err(result_full(
                "unknown result column type cannot be accounted",
            ));
        }
    };
    checked_result_bytes(size_of::<LogicalType>().checked_add(nested))
}

impl ResultAccumulator {
    pub(crate) fn new(
        columns: &[String],
        types: &[LogicalType],
        resources: QueryResourceContext,
        limits: ResultLimits,
    ) -> Result<Self> {
        Self::new_inner(columns, types, resources, limits, false)
    }

    fn new_inner(
        columns: &[String],
        types: &[LogicalType],
        resources: QueryResourceContext,
        limits: ResultLimits,
        default_unknown_types: bool,
    ) -> Result<Self> {
        resources
            .check_cancelled()
            .map_err(convert_cancellation_error)?;
        let mut metadata = checked_result_bytes(columns.len().checked_mul(size_of::<String>()))?;
        for column in columns {
            metadata = checked_result_bytes(metadata.checked_add(column.len()))?;
        }
        let type_count = if default_unknown_types && types.is_empty() {
            columns.len()
        } else {
            types.len()
        };
        if types.is_empty() {
            metadata = checked_result_bytes(metadata.checked_add(checked_result_bytes(
                type_count.checked_mul(size_of::<LogicalType>()),
            )?))?;
        } else {
            for ty in types {
                metadata = checked_result_bytes(metadata.checked_add(type_bytes(ty)?))?;
            }
        }
        metadata = checked_result_bytes(metadata.checked_add(
            size_of::<crate::database::ResultReservation>() + 2 * size_of::<usize>(),
        ))?;
        let grant = resources
            .try_allocate(metadata)
            .map_err(|error| result_full(error.to_string()))?;
        let mut names = Vec::new();
        names
            .try_reserve_exact(columns.len())
            .map_err(|error| result_full(error.to_string()))?;
        if names.capacity() > columns.len() {
            return Err(result_full(
                "result metadata allocator exceeded admitted capacity",
            ));
        }
        for name in columns {
            let mut copy = String::new();
            copy.try_reserve_exact(name.len())
                .map_err(|error| result_full(error.to_string()))?;
            if copy.capacity() > name.len() {
                return Err(result_full(
                    "result name allocator exceeded admitted capacity",
                ));
            }
            copy.push_str(name);
            names.push(copy);
        }
        let mut copied_types = Vec::new();
        copied_types
            .try_reserve_exact(type_count)
            .map_err(|error| result_full(error.to_string()))?;
        if copied_types.capacity() > type_count {
            return Err(result_full(
                "result metadata allocator exceeded admitted capacity",
            ));
        }
        if types.is_empty() {
            copied_types.resize(type_count, LogicalType::Any);
        } else {
            copied_types.extend_from_slice(types);
        }
        Ok(Self {
            result: QueryResult::with_types(names, copied_types).with_result_reservation(grant),
            resources,
            limits,
            row_bytes: 0,
            metadata_bytes: metadata,
            rows: 0,
            types_captured: !types.iter().all(|ty| *ty == LogicalType::Any),
            dense_companion_bytes: 0,
        })
    }

    fn resize_reservation(&self, row_bytes: usize, metadata_bytes: usize) -> Result<()> {
        if row_bytes > self.limits.max_bytes {
            return Err(result_full(format!(
                "result byte limit exceeded: requested {row_bytes}, limit {}",
                self.limits.max_bytes
            )));
        }
        let total = checked_result_bytes(row_bytes.checked_add(metadata_bytes))?;
        self.result
            .result_reservation
            .as_ref()
            .ok_or_else(|| Error::Internal("bounded result lost its reservation".into()))?
            .try_resize(total)
    }

    fn add_backing(&mut self, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let target = checked_result_bytes(self.row_bytes.checked_add(bytes))?;
        self.resize_reservation(target, self.metadata_bytes)?;
        self.row_bytes = target;
        Ok(())
    }

    fn check_rows(&self, count: usize) -> Result<()> {
        let target = self
            .rows
            .checked_add(count)
            .ok_or_else(|| result_full("result row count overflow"))?;
        if target > self.limits.max_rows {
            Err(result_full(format!(
                "result row limit exceeded: requested {target}, limit {}",
                self.limits.max_rows
            )))
        } else {
            Ok(())
        }
    }

    fn reserve_vec<T>(
        vector: &mut Vec<T>,
        needed: usize,
        row_bytes: &mut usize,
        metadata_bytes: usize,
        limits: ResultLimits,
        reservation: &crate::database::ResultReservation,
        grow: bool,
    ) -> Result<()> {
        if needed <= vector.capacity() {
            return Ok(());
        }
        let unit = size_of::<T>();
        let old_capacity = vector.capacity();
        let remaining = limits.max_bytes.saturating_sub(*row_bytes);
        let available = old_capacity.saturating_add(remaining / unit.max(1));
        let target = if grow {
            needed.max(old_capacity.saturating_mul(2)).max(4)
        } else {
            needed
        }
        .min(available);
        if target < needed {
            return Err(result_full(
                "result byte limit cannot admit container capacity",
            ));
        }
        let growth = checked_result_bytes(
            target
                .checked_sub(old_capacity)
                .and_then(|n| n.checked_mul(unit)),
        )?;
        let next_bytes = checked_result_bytes(row_bytes.checked_add(growth))?;
        reservation.try_resize(checked_result_bytes(
            metadata_bytes.checked_add(next_bytes),
        )?)?;
        if let Err(error) = vector.try_reserve_exact(target - vector.len()) {
            reservation.try_resize(checked_result_bytes(
                metadata_bytes.checked_add(*row_bytes),
            )?)?;
            return Err(result_full(format!(
                "result container allocation failed: {error}"
            )));
        }
        if vector.capacity() > target {
            // Do not retain an allocator-expanded container under a smaller grant.
            drop(std::mem::take(vector));
            return Err(result_full("result allocator exceeded admitted capacity"));
        }
        *row_bytes = next_bytes;
        Ok(())
    }

    fn capture_types<'a>(
        &mut self,
        count: usize,
        mut ty: impl FnMut(usize) -> Option<&'a LogicalType>,
    ) -> Result<()> {
        if self.types_captured || count == 0 {
            return Ok(());
        }
        let mut new_bytes = 0usize;
        for index in 0..count {
            new_bytes = checked_result_bytes(
                new_bytes.checked_add(type_bytes(ty(index).unwrap_or(&LogicalType::Any))?),
            )?;
        }
        // Admit replacement while the original type vector is still alive.
        let metadata = checked_result_bytes(self.metadata_bytes.checked_add(new_bytes))?;
        self.resize_reservation(self.row_bytes, metadata)?;
        let mut types = Vec::new();
        types
            .try_reserve_exact(count)
            .map_err(|error| result_full(error.to_string()))?;
        if types.capacity() > count {
            return Err(result_full(
                "result type allocator exceeded admitted capacity",
            ));
        }
        for index in 0..count {
            types.push(ty(index).unwrap_or(&LogicalType::Any).clone());
        }
        self.result.column_types = types;
        self.metadata_bytes = metadata;
        self.types_captured = true;
        Ok(())
    }

    pub(crate) fn append_values(&mut self, values: &[Value]) -> Result<()> {
        self.append_cells(
            values.len(),
            |index| {
                values[index]
                    .retained_size_bytes()
                    .ok_or_else(|| result_full("result value retained-size overflow"))
            },
            |index| values[index].clone(),
        )
    }

    fn append_row<'a>(
        &mut self,
        width: usize,
        cell: impl Fn(usize) -> Option<(&'a grafeo_core::execution::vector::ValueVector, usize)>,
    ) -> Result<()> {
        self.append_cells(
            width,
            |index| match cell(index) {
                Some((column, index)) => column
                    .retained_value_bytes(index)
                    .ok_or_else(|| result_full("result value retained-size overflow")),
                None => Ok(size_of::<Value>()),
            },
            |index| {
                cell(index)
                    .and_then(|(column, index)| column.get_value(index))
                    .unwrap_or(Value::Null)
            },
        )
    }

    fn append_cells(
        &mut self,
        width: usize,
        mut cell_bytes: impl FnMut(usize) -> Result<usize>,
        mut cell_value: impl FnMut(usize) -> Value,
    ) -> Result<()> {
        self.check_rows(1)?;
        self.flush_dense()?;
        let reservation = self
            .result
            .result_reservation
            .as_ref()
            .ok_or_else(|| Error::Internal("bounded result lost reservation".into()))?;
        Self::reserve_vec(
            &mut self.result.rows,
            self.rows + 1,
            &mut self.row_bytes,
            self.metadata_bytes,
            self.limits,
            reservation,
            true,
        )?;
        let mut row = Vec::new();
        Self::reserve_vec(
            &mut row,
            width,
            &mut self.row_bytes,
            self.metadata_bytes,
            self.limits,
            reservation,
            false,
        )?;
        for index in 0..width {
            let bytes = cell_bytes(index)?;
            let backing = bytes
                .checked_sub(size_of::<Value>())
                .ok_or_else(|| result_full("result value bound omitted its inline slot"))?;
            self.add_backing(backing)?;
            row.push(cell_value(index));
        }
        self.result.rows.push(row);
        self.rows += 1;
        Ok(())
    }

    fn flush_dense(&mut self) -> Result<()> {
        let Some(columns) = self.result.int64_cols.take() else {
            return Ok(());
        };
        // The dense path already reserved the exact lazy row companion so
        // infallible borrowed rows() cannot allocate an unaccounted copy.
        self.result
            .rows
            .try_reserve_exact(self.rows)
            .map_err(|error| result_full(error.to_string()))?;
        if self.result.rows.capacity() > self.rows {
            return Err(result_full(
                "result row allocator exceeded admitted capacity",
            ));
        }
        for index in 0..self.rows {
            let mut row = Vec::new();
            row.try_reserve_exact(columns.len())
                .map_err(|error| result_full(error.to_string()))?;
            if row.capacity() > columns.len() {
                return Err(result_full(
                    "result row allocator exceeded admitted capacity",
                ));
            }
            row.extend(columns.iter().map(|column| Value::Int64(column[index])));
            self.result.rows.push(row);
        }
        self.dense_companion_bytes = 0;
        // Retain the admitted high-water reservation: both layouts coexisted.
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn append_chunk(&mut self, chunk: &DataChunk) -> Result<usize> {
        self.consume(chunk, usize::MAX)
    }

    pub(crate) fn consume(&mut self, chunk: &DataChunk, limit: usize) -> Result<usize> {
        let count = chunk.row_count().min(limit);
        if self
            .result
            .int64_cols
            .as_ref()
            .is_some_and(|columns| columns.len() != chunk.column_count())
        {
            return Err(Error::Internal("result column count changed".into()));
        }
        self.check_rows(count)?;
        self.capture_types(chunk.column_count(), |index| {
            chunk.column(index).map(|column| column.data_type())
        })?;
        if count == 0 {
            return Ok(0);
        }
        let dense = chunk.column_count() > 0
            && chunk.selection().is_none()
            && self.result.rows.is_empty()
            && (0..chunk.column_count()).all(|index| {
                chunk.column(index).is_some_and(|column| {
                    (column.as_int64_slice().is_some() || column.as_node_id_slice().is_some())
                        && (column.validity_capacity() == 0
                            || (0..count).all(|row| !column.is_null(row)))
                })
            });
        if dense {
            let per_row = checked_result_bytes(
                chunk
                    .column_count()
                    .checked_mul(size_of::<Value>())
                    .and_then(|bytes| bytes.checked_add(size_of::<Vec<Value>>())),
            )?;
            let companion = checked_result_bytes((self.rows + count).checked_mul(per_row))?;
            self.add_backing(companion.saturating_sub(self.dense_companion_bytes))?;
            self.dense_companion_bytes = companion;
            let reservation = self
                .result
                .result_reservation
                .as_ref()
                .ok_or_else(|| Error::Internal("bounded result lost reservation".into()))?;
            let columns = self.result.int64_cols.get_or_insert_with(Vec::new);
            Self::reserve_vec(
                columns,
                chunk.column_count(),
                &mut self.row_bytes,
                self.metadata_bytes,
                self.limits,
                reservation,
                false,
            )?;
            while columns.len() < chunk.column_count() {
                columns.push(Vec::new());
            }
            if columns.len() != chunk.column_count() {
                return Err(Error::Internal("result column count changed".into()));
            }
            for (index, output) in columns.iter_mut().enumerate() {
                Self::reserve_vec(
                    output,
                    self.rows + count,
                    &mut self.row_bytes,
                    self.metadata_bytes,
                    self.limits,
                    reservation,
                    true,
                )?;
                let column = chunk
                    .column(index)
                    .ok_or_else(|| Error::Internal("missing dense result column".into()))?;
                if let Some(values) = column.as_int64_slice() {
                    output.extend_from_slice(&values[..count]);
                } else if let Some(values) = column.as_node_id_slice() {
                    output.extend(values[..count].iter().map(|id| {
                        #[allow(clippy::cast_possible_wrap)]
                        {
                            id.as_u64() as i64
                        }
                    }));
                }
            }
            self.rows += count;
        } else {
            for row in chunk.selected_indices().take(count) {
                self.resources
                    .check_cancelled()
                    .map_err(convert_cancellation_error)?;
                self.append_row(chunk.column_count(), |index| {
                    chunk.column(index).map(|column| (column, row))
                })?;
            }
        }
        Ok(count)
    }

    #[cfg(all(
        feature = "gql",
        feature = "lpg",
        feature = "spill",
        feature = "async-storage"
    ))]
    fn consume_visible_sort_columns(&mut self, chunk: &DataChunk, width: usize) -> Result<()> {
        if width == chunk.column_count() {
            self.consume(chunk, usize::MAX)?;
            return Ok(());
        }
        if width > chunk.column_count() {
            return Err(Error::Internal("sort output lost a visible column".into()));
        }
        self.check_rows(chunk.row_count())?;
        self.capture_types(width, |index| {
            chunk.column(index).map(|column| column.data_type())
        })?;
        for row in chunk.selected_indices() {
            self.resources
                .check_cancelled()
                .map_err(convert_cancellation_error)?;
            self.append_row(width, |index| {
                chunk.column(index).map(|column| (column, row))
            })?;
        }
        Ok(())
    }

    fn consume_factorized(
        &mut self,
        chunk: &grafeo_core::execution::factorized_chunk::FactorizedChunk,
        limit: usize,
    ) -> Result<usize> {
        let levels = chunk.level_count();
        let scratch = checked_result_bytes(levels.checked_mul(2 * size_of::<usize>()))?;
        let _scratch = self
            .resources
            .try_allocate(scratch)
            .map_err(|error| result_full(error.to_string()))?;
        let width = chunk.total_column_count();
        self.capture_types(width, |index| {
            let mut offset = index;
            for level in 0..levels {
                let level = chunk.level(level)?;
                if offset < level.column_count() {
                    return level.column(offset).map(|column| column.data().data_type());
                }
                offset -= level.column_count();
            }
            None
        })?;
        let mut count = 0;
        for indices in chunk.logical_row_iter() {
            if count == limit {
                break;
            }
            self.resources
                .check_cancelled()
                .map_err(convert_cancellation_error)?;
            if let Some(selection) = chunk.chunk_state().selection()
                && let Some(deepest) = levels.checked_sub(1)
                && !selection.is_selected(deepest, indices[deepest])
            {
                continue;
            }
            self.append_row(width, |index| {
                let mut offset = index;
                for (level_index, physical) in indices.iter().copied().enumerate() {
                    let level = chunk.level(level_index)?;
                    if offset < level.column_count() {
                        return level.column(offset).map(|column| (column.data(), physical));
                    }
                    offset -= level.column_count();
                }
                None
            })?;
            count += 1;
        }
        Ok(count)
    }

    pub(crate) fn finish(self) -> QueryResult {
        self.result
    }
}

struct ResultAccumulatorSink {
    accumulator: ResultAccumulator,
    error: Option<Error>,
}

impl grafeo_core::execution::pipeline::ResultConsumer for ResultAccumulatorSink {
    fn consume(&mut self, chunk: &DataChunk) -> std::result::Result<bool, OperatorError> {
        match self.accumulator.consume(chunk, usize::MAX) {
            Ok(_) => Ok(true),
            Err(error) => {
                self.error = Some(error);
                Err(OperatorError::Execution(
                    "bounded result admission failed".into(),
                ))
            }
        }
    }
    fn consume_variant(
        &mut self,
        chunk: &grafeo_core::execution::factorized_chunk::ChunkVariant,
    ) -> std::result::Result<bool, OperatorError> {
        use grafeo_core::execution::factorized_chunk::ChunkVariant;
        let result = match chunk {
            ChunkVariant::Flat(chunk) => self.accumulator.consume(chunk, usize::MAX),
            ChunkVariant::Factorized(chunk) => {
                self.accumulator.consume_factorized(chunk, usize::MAX)
            }
            _ => Err(Error::Internal("unknown result chunk variant".into())),
        };
        match result {
            Ok(_) => Ok(true),
            Err(error) => {
                self.error = Some(error);
                Err(OperatorError::Execution(
                    "bounded result admission failed".into(),
                ))
            }
        }
    }
}

/// Executes a physical operator tree and collects results.
///
/// Raw deadline and timeout builders have been removed. Configure execution
/// through [`Self::with_execution_checkpoint`].
///
/// ```compile_fail,E0599
/// use grafeo_engine::query::executor::Executor;
/// let _ = Executor::with_deadline;
/// ```
///
/// ```compile_fail,E0599
/// use grafeo_engine::query::executor::Executor;
/// let _ = Executor::with_timeout_duration;
/// ```
pub struct Executor<'a> {
    /// Column names for the result.
    columns: Cow<'a, [String]>,
    /// Column types for the result.
    column_types: Cow<'a, [LogicalType]>,
    /// Explicit orchestration capability for this execution.
    ///
    /// Standalone execution creates one checkpoint when none is supplied.
    checkpoint: Option<QueryExecutionCheckpoint>,
    result_resources: Option<QueryResourceContext>,
    result_limits: ResultLimits,
}

impl<'a> Executor<'a> {
    /// Creates a new executor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            columns: Cow::Borrowed(&[]),
            column_types: Cow::Borrowed(&[]),
            checkpoint: None,
            result_resources: None,
            result_limits: ResultLimits::default(),
        }
    }

    /// Creates an executor with specified column names.
    #[must_use]
    pub fn with_columns(columns: Vec<String>) -> Self {
        let len = columns.len();
        Self {
            columns: Cow::Owned(columns),
            column_types: Cow::Owned(vec![LogicalType::Any; len]),
            checkpoint: None,
            result_resources: None,
            result_limits: ResultLimits::default(),
        }
    }

    /// Creates an executor with specified column names and types.
    #[must_use]
    pub fn with_columns_and_types(columns: Vec<String>, column_types: Vec<LogicalType>) -> Self {
        Self {
            columns: Cow::Owned(columns),
            column_types: Cow::Owned(column_types),
            checkpoint: None,
            result_resources: None,
            result_limits: ResultLimits::default(),
        }
    }

    /// Borrows plan metadata; collection admits capacity before copying it.
    ///
    /// # Errors
    /// Returns cancellation if the supplied execution has been cancelled.
    pub fn with_bounded_columns(
        columns: &'a [String],
        resources: QueryResourceContext,
        limits: ResultLimits,
    ) -> Result<Self> {
        resources
            .check_cancelled()
            .map_err(convert_cancellation_error)?;
        Ok(Self {
            columns: Cow::Borrowed(columns),
            column_types: Cow::Borrowed(&[]),
            checkpoint: None,
            result_resources: Some(resources),
            result_limits: limits,
        })
    }

    /// Installs this execution's shared resources and eager output limits.
    #[must_use]
    pub fn with_result_resources(
        mut self,
        resources: QueryResourceContext,
        limits: ResultLimits,
    ) -> Self {
        self.result_resources = Some(resources);
        self.result_limits = limits;
        self
    }

    /// Installs an orchestration checkpoint for cooperative cancellation.
    ///
    /// Compose deadlines on the checkpoint before deriving resource tokens.
    /// A terminal cancellation remains authoritative; reuse after cancellation
    /// requires a fresh checkpoint. Supplied result resources must belong to
    /// the same execution.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_core::execution::QueryExecutionControl;
    /// use grafeo_engine::query::executor::Executor;
    ///
    /// let control = QueryExecutionControl::new();
    /// let _executor = Executor::with_columns(vec!["value".to_owned()])
    ///     .with_execution_checkpoint(control.checkpoint());
    /// ```
    #[must_use]
    pub fn with_execution_checkpoint(mut self, checkpoint: QueryExecutionCheckpoint) -> Self {
        self.checkpoint = Some(checkpoint);
        self
    }

    /// Resolves the checkpoint once at the start of an execution.
    fn execution_checkpoint(&self) -> QueryExecutionCheckpoint {
        self.checkpoint
            .clone()
            .unwrap_or_else(|| QueryExecutionControl::new().checkpoint())
    }

    /// Checks whether explicit cancellation or the effective deadline has won.
    fn check_cancellation(token: &QueryCancellationToken) -> Result<()> {
        token.check().map_err(convert_cancellation_error)
    }

    fn accumulator(&self, cancellation: &QueryCancellationToken) -> Result<ResultAccumulator> {
        let resources = match &self.result_resources {
            Some(resources) => resources.clone(),
            None => {
                let buffer =
                    grafeo_common::memory::buffer::BufferManager::with_budget(64 * 1024 * 1024);
                QueryResourceContext::new_with_cancellation(buffer, cancellation.clone())
                    .map_err(|error| result_full(error.to_string()))?
            }
        };
        ResultAccumulator::new_inner(
            &self.columns,
            &self.column_types,
            resources,
            self.result_limits,
            true,
        )
    }

    /// Executes a physical operator and collects results under the output limits.
    ///
    /// # Errors
    /// Returns execution, cancellation, or result admission failures.
    pub fn execute(&self, operator: &mut dyn Operator) -> Result<QueryResult> {
        self.execute_collect(operator, None)
    }

    fn execute_collect(
        &self,
        operator: &mut dyn Operator,
        limit: Option<usize>,
    ) -> Result<QueryResult> {
        let _span = grafeo_debug_span!("grafeo::query::execute");
        let checkpoint = self.execution_checkpoint();
        let cancellation = checkpoint.token();
        Self::check_cancellation(&cancellation)?;
        let mut accumulator = self.accumulator(&cancellation)?;
        let mut collected = 0usize;
        loop {
            if limit.is_some_and(|limit| collected >= limit) {
                break;
            }
            Self::check_cancellation(&cancellation)?;
            let remaining = limit.map_or(usize::MAX, |limit| limit - collected);
            if let Some(factorized) = operator.as_factorized_mut() {
                let next = FactorizedOperator::next_factorized(factorized)
                    .map_err(convert_operator_error)?;
                Self::check_cancellation(&cancellation)?;
                match next {
                    Some(chunk) => {
                        collected += accumulator.consume_factorized(&chunk, remaining)?;
                    }
                    None => break,
                }
            } else {
                let next = operator.next().map_err(convert_operator_error)?;
                Self::check_cancellation(&cancellation)?;
                match next {
                    Some(chunk) => collected += accumulator.consume(&chunk, remaining)?,
                    None => break,
                }
            }
        }
        Self::check_cancellation(&cancellation)?;
        Ok(accumulator.finish())
    }

    /// Executes a push pipeline directly into the bounded result accumulator.
    ///
    /// # Errors
    /// Returns execution, cancellation, or result admission failures.
    pub fn execute_pipeline(
        &self,
        source: Box<dyn Operator>,
        push_ops: Vec<Box<dyn grafeo_core::execution::pipeline::PushOperator>>,
    ) -> Result<QueryResult> {
        use grafeo_core::execution::OperatorSource;
        use grafeo_core::execution::pipeline::ResultConsumerSink;
        let checkpoint = self.execution_checkpoint();
        let cancellation = checkpoint.token();
        Self::check_cancellation(&cancellation)?;
        let consumer = ResultAccumulatorSink {
            accumulator: self.accumulator(&cancellation)?,
            error: None,
        };
        let sink = ResultConsumerSink::new(consumer);
        let mut pipeline = Pipeline::new(
            Box::new(OperatorSource::new(source)),
            push_ops,
            Box::new(sink),
        );
        pipeline.set_execution_checkpoint(checkpoint);
        let execution = pipeline.execute();
        let sink = pipeline
            .into_sink()
            .into_any()
            .downcast::<ResultConsumerSink<ResultAccumulatorSink>>()
            .map_err(|_| Error::Internal("bounded result sink identity changed".into()))?;
        let consumer = sink.into_inner();
        if let Some(error) = consumer.error {
            return Err(error);
        }
        execution.map_err(convert_operator_error)?;
        Self::check_cancellation(&cancellation)?;
        Ok(consumer.accumulator.finish())
    }

    /// Executes and returns at most `limit` rows, still enforcing output limits.
    ///
    /// # Errors
    /// Returns execution, cancellation, or result admission failures.
    pub fn execute_with_limit(
        &self,
        operator: &mut dyn Operator,
        limit: usize,
    ) -> Result<QueryResult> {
        self.execute_collect(operator, Some(limit))
    }

    /// Executes a physical operator with adaptive cardinality tracking.
    ///
    /// This wraps the operator in a cardinality tracking layer and monitors
    /// deviation from estimates during execution. The adaptive summary is
    /// returned alongside the query result.
    ///
    /// # Arguments
    ///
    /// * `operator` - The root physical operator to execute
    /// * `adaptive_context` - Context with cardinality estimates from planning
    /// * `config` - Adaptive execution configuration
    ///
    /// # Errors
    ///
    /// Returns an error if operator execution fails.
    pub fn execute_adaptive(
        &self,
        operator: Box<dyn Operator>,
        adaptive_context: Option<AdaptiveContext>,
        config: &AdaptiveConfig,
    ) -> Result<(QueryResult, Option<AdaptiveSummary>)> {
        // If adaptive is disabled or no context, fall back to normal execution
        if !config.enabled {
            let mut op = operator;
            let result = self.execute(op.as_mut())?;
            return Ok((result, None));
        }

        let Some(ctx) = adaptive_context else {
            let mut op = operator;
            let result = self.execute(op.as_mut())?;
            return Ok((result, None));
        };

        // Create shared context for tracking
        let shared_ctx = SharedAdaptiveContext::from_context(AdaptiveContext::with_thresholds(
            config.threshold,
            config.min_rows,
        ));

        // Copy estimates from the planning context to the shared tracking context
        for (op_id, checkpoint) in ctx.all_checkpoints() {
            if let Some(mut inner) = shared_ctx.snapshot() {
                inner.set_estimate(op_id, checkpoint.estimated);
            }
        }

        // Wrap operator with tracking
        let mut wrapped = CardinalityTrackingWrapper::new(operator, "root", shared_ctx.clone());

        let result = self.execute(&mut wrapped)?;

        // Get final summary
        let summary = shared_ctx.snapshot().map(|ctx| ctx.summary());

        Ok((result, summary))
    }
}

impl Default for Executor<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts an operator error to a common error.
pub(crate) fn convert_operator_error(err: OperatorError) -> Error {
    match err {
        OperatorError::TypeMismatch { expected, found } => Error::TypeMismatch { expected, found },
        OperatorError::ColumnNotFound(name) => {
            Error::InvalidValue(format!("Column not found: {name}"))
        }
        OperatorError::Execution(msg) => Error::Internal(msg),
        OperatorError::ConstraintViolation(msg) => {
            Error::InvalidValue(format!("Constraint violation: {msg}"))
        }
        OperatorError::WriteConflict(msg) => {
            Error::Transaction(grafeo_common::utils::error::TransactionError::WriteConflict(msg))
        }
        OperatorError::ResidentMemory(error) => convert_resident_memory_error(error),
        OperatorError::ResidentAllocation(message) => {
            Error::Storage(grafeo_common::utils::error::StorageError::Full).with_context(message)
        }
        OperatorError::ResidentContainerAllocation { container, source } => {
            Error::Storage(grafeo_common::utils::error::StorageError::Full)
                .with_context(format!("allocator refused {container} capacity: {source}"))
        }
        #[cfg(feature = "spill")]
        OperatorError::ResidentExactVectorAllocation(error) => {
            Error::Storage(grafeo_common::utils::error::StorageError::Full)
                .with_context(error.to_string())
        }
        OperatorError::ResidentNativeMapAllocation { source } => {
            Error::Storage(grafeo_common::utils::error::StorageError::Full)
                .with_context(format!("native partition-map allocation failed: {source}"))
        }
        OperatorError::ResidentNativeMapAllocationWithRollback { source, rollback } => {
            Error::Storage(grafeo_common::utils::error::StorageError::Full)
                .with_context(format!("native partition-map allocation failed: {source}"))
                .with_context(format!("partition grant rollback also failed: {rollback}"))
        }
        OperatorError::ResidentContainerInvariant { container, message } => Error::Internal(
            format!("resident-memory invariant failed for {container}: {message}"),
        ),
        OperatorError::ResidentContainerInvariantWithRollback {
            container,
            message,
            rollback,
        } => Error::Internal(format!(
            "resident-memory invariant failed for {container}: {message}; grant rollback also failed: {rollback}"
        )),
        OperatorError::StorageFull(message) => {
            Error::Storage(grafeo_common::utils::error::StorageError::Full).with_context(message)
        }
        OperatorError::QueryCancelled(error) => convert_cancellation_error(error),
        OperatorError::Context { source, context } => {
            convert_operator_error(*source).with_context(context)
        }
        #[cfg(feature = "spill")]
        OperatorError::ClassifiedAccountedFailure {
            classification,
            authority,
        } => convert_classified_accounted_failure(classification, authority),
        #[cfg(feature = "spill")]
        OperatorError::AccountedFailure(authority) => Error::Internal(authority.to_string()),
        _ => Error::Internal(format!("{err}")),
    }
}

fn convert_resident_memory_error(error: grafeo_common::memory::buffer::MemoryGrantError) -> Error {
    match error {
        error @ (grafeo_common::memory::buffer::MemoryGrantError::LimitExceeded { .. }
        | grafeo_common::memory::buffer::MemoryGrantError::ArithmeticOverflow { .. }
        | grafeo_common::memory::buffer::MemoryGrantError::Denied { .. }) => {
            Error::Storage(grafeo_common::utils::error::StorageError::Full)
                .with_context(error.to_string())
        }
        error => Error::Internal(error.to_string()),
    }
}

#[cfg(feature = "spill")]
fn convert_classified_accounted_failure(
    classification: grafeo_core::execution::operators::AccountedFailureClassification,
    authority: grafeo_common::memory::buffer::AccountedError,
) -> Error {
    use grafeo_common::memory::buffer::MemoryGrantError;
    use grafeo_common::utils::error::ErrorCode;
    use grafeo_core::execution::operators::AccountedFailureClassification;

    // Classification is fixed-size metadata. Preserve the original opaque
    // primary and its authority without formatting, cloning or source walking;
    // trusted concrete owners provide their own bounded diagnostic text.
    let code = match classification {
        AccountedFailureClassification::ResidentMemory(
            MemoryGrantError::LimitExceeded { .. }
            | MemoryGrantError::ArithmeticOverflow { .. }
            | MemoryGrantError::Denied { .. },
        )
        | AccountedFailureClassification::ResidentAllocation
        | AccountedFailureClassification::ResidentExactVectorAllocation(_)
        | AccountedFailureClassification::StorageFull => ErrorCode::StorageFull,
        AccountedFailureClassification::QueryCancelled(QueryCancellationError::Cancelled) => {
            ErrorCode::QueryCancelled
        }
        AccountedFailureClassification::QueryCancelled(
            QueryCancellationError::DeadlineExceeded { .. },
        ) => ErrorCode::QueryTimeout,
        AccountedFailureClassification::TypeMismatch => ErrorCode::TypeMismatch,
        AccountedFailureClassification::ColumnNotFound
        | AccountedFailureClassification::ConstraintViolation => ErrorCode::InvalidInput,
        AccountedFailureClassification::WriteConflict => ErrorCode::TransactionConflict,
        _ => ErrorCode::Internal,
    };
    Error::RetainedContext {
        code,
        source: authority.into(),
    }
}

pub(crate) fn convert_cancellation_error(error: QueryCancellationError) -> Error {
    match error {
        QueryCancellationError::Cancelled => Error::Query(QueryError::cancelled()),
        QueryCancellationError::DeadlineExceeded {
            timeout: Some(timeout),
        } => Error::Query(QueryError::timeout_with_limit(timeout)),
        QueryCancellationError::DeadlineExceeded { timeout: None } => {
            Error::Query(QueryError::timeout())
        }
        QueryCancellationError::InvalidState { phase } => {
            Error::Internal(format!("query execution state is invalid: {phase}"))
        }
        _ => Error::Internal(format!("unknown query cancellation reason: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::LogicalType;
    use grafeo_core::execution::DataChunk;
    use std::time::Duration;
    #[cfg(not(target_arch = "wasm32"))]
    use std::time::Instant;

    /// A mock operator that generates chunks with integer data on demand.
    struct MockIntOperator {
        values: Vec<i64>,
        position: usize,
        chunk_size: usize,
    }

    impl MockIntOperator {
        fn new(values: Vec<i64>, chunk_size: usize) -> Self {
            Self {
                values,
                position: 0,
                chunk_size,
            }
        }
    }

    impl Operator for MockIntOperator {
        fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
            if self.position >= self.values.len() {
                return Ok(None);
            }

            let end = (self.position + self.chunk_size).min(self.values.len());
            let mut chunk = DataChunk::with_capacity(&[LogicalType::Int64], self.chunk_size);

            {
                let col = chunk.column_mut(0).unwrap();
                for i in self.position..end {
                    col.push_int64(self.values[i]);
                }
            }
            chunk.set_count(end - self.position);
            self.position = end;

            Ok(Some(chunk))
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "MockInt"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// Empty mock operator for testing empty results.
    struct EmptyOperator;

    impl Operator for EmptyOperator {
        fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
            Ok(None)
        }

        fn reset(&mut self) {}

        fn name(&self) -> &'static str {
            "Empty"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    struct CancellingOperator {
        cancellation: grafeo_core::execution::QueryCancellationHandle,
        error_after_cancel: bool,
        factorized: bool,
    }

    impl CancellingOperator {
        fn cancel(&self) -> std::result::Result<(), OperatorError> {
            self.cancellation.cancel();
            if self.error_after_cancel {
                Err(OperatorError::Execution("source failed".to_string()))
            } else {
                Ok(())
            }
        }
    }

    #[cfg(feature = "spill")]
    #[derive(Debug)]
    struct AccountedConversionFailure {
        context: &'static str,
        primary: Option<grafeo_common::memory::buffer::MemoryGrantError>,
    }

    #[cfg(feature = "spill")]
    impl std::fmt::Display for AccountedConversionFailure {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            if let Some(primary) = &self.primary {
                write!(formatter, "{primary}; ")?;
            }
            formatter.write_str(self.context)
        }
    }

    #[cfg(feature = "spill")]
    impl std::error::Error for AccountedConversionFailure {}

    #[cfg(feature = "spill")]
    fn accounted_conversion_authority(
        message: &'static str,
    ) -> grafeo_common::memory::buffer::AccountedError {
        accounted_conversion_authority_with_primary(message, None)
    }

    #[cfg(feature = "spill")]
    fn accounted_conversion_authority_with_primary(
        message: &'static str,
        primary: Option<grafeo_common::memory::buffer::MemoryGrantError>,
    ) -> grafeo_common::memory::buffer::AccountedError {
        use grafeo_common::memory::buffer::{AccountedErrorPublisher, BufferManager, MemoryRegion};

        let required = AccountedErrorPublisher::<AccountedConversionFailure>::required_bytes();
        let manager = BufferManager::with_budget(required * 2);
        let grant = manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .expect("zero-byte publication grant");
        AccountedErrorPublisher::try_new(grant)
            .expect("accounted error publisher")
            .publish(AccountedConversionFailure {
                context: message,
                primary,
            })
    }

    impl Operator for CancellingOperator {
        fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
            self.cancel()?;
            Ok(None)
        }

        fn reset(&mut self) {}

        fn name(&self) -> &'static str {
            "Cancelling"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }

        fn as_factorized_mut(&mut self) -> Option<&mut dyn FactorizedOperator> {
            if self.factorized { Some(self) } else { None }
        }
    }

    impl FactorizedOperator for CancellingOperator {
        fn next_factorized(&mut self) -> grafeo_core::execution::operators::FactorizedResult {
            self.cancel()?;
            Ok(None)
        }
    }

    #[test]
    fn test_executor_empty() {
        let executor = Executor::with_columns(vec!["a".to_string()]);
        let mut op = EmptyOperator;

        let result = executor.execute(&mut op).unwrap();
        assert!(result.is_empty());
        assert_eq!(result.column_count(), 1);
    }

    #[test]
    fn test_executor_single_chunk() {
        let executor = Executor::with_columns(vec!["value".to_string()]);
        let mut op = MockIntOperator::new(vec![1, 2, 3], 10);

        let result = executor.execute(&mut op).unwrap();
        assert_eq!(result.row_count(), 3);
        assert_eq!(result.rows()[0][0], Value::Int64(1));
        assert_eq!(result.rows()[1][0], Value::Int64(2));
        assert_eq!(result.rows()[2][0], Value::Int64(3));
    }

    #[test]
    fn test_executor_with_limit() {
        let executor = Executor::with_columns(vec!["value".to_string()]);
        let mut op = MockIntOperator::new((0..10).collect(), 100);

        let result = executor.execute_with_limit(&mut op, 5).unwrap();
        assert_eq!(result.row_count(), 5);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_executor_timeout_expired() {
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        let control = QueryExecutionControl::with_deadline(expired);
        let mut executor = Executor::with_columns(vec!["value".to_string()])
            .with_execution_checkpoint(control.checkpoint());
        let mut op = MockIntOperator::new(vec![1, 2, 3], 10);

        let result = executor.execute(&mut op);
        assert!(result.is_err());
        assert_eq!(op.position, 0, "expired checkpoints stop before a pull");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Query exceeded timeout"),
            "Expected timeout error, got: {err}"
        );

        let retry = executor.execute(&mut op).unwrap_err();
        assert_eq!(retry.error_code(), err.error_code());
        assert_eq!(op.position, 0);

        executor = executor.with_execution_checkpoint(QueryExecutionControl::new().checkpoint());
        let result = executor.execute(&mut op).unwrap();
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn executor_deadline_is_observed_even_when_limit_is_zero() {
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        let checkpoint = QueryExecutionControl::new()
            .checkpoint()
            .with_additional_deadline(expired, Some(Duration::from_secs(3)));
        let executor =
            Executor::with_columns(vec!["value".to_string()]).with_execution_checkpoint(checkpoint);
        let mut op = MockIntOperator::new(vec![1, 2, 3], 10);

        let error = executor.execute_with_limit(&mut op, 0).unwrap_err();
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::QueryTimeout
        );
        assert!(error.to_string().contains("3s timeout"));
        assert_eq!(op.position, 0);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn standalone_accumulator_shares_the_execution_checkpoint_state() {
        let executor = Executor::with_columns(vec!["value".to_string()]);
        let checkpoint = executor.execution_checkpoint();
        let cancellation = checkpoint.token();
        let accumulator = executor.accumulator(&cancellation).unwrap();
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        let error = checkpoint
            .with_additional_deadline(expired, Some(Duration::from_secs(3)))
            .check()
            .unwrap_err();

        assert_eq!(cancellation.check().unwrap_err(), error);
        assert_eq!(accumulator.resources.check_cancelled().unwrap_err(), error);
    }

    #[test]
    fn explicit_checkpoint_is_observed_even_when_limit_is_zero() {
        let control = QueryExecutionControl::new();
        let cancellation = control.cancellation_handle();
        let executor = Executor::with_columns(vec!["value".to_string()])
            .with_execution_checkpoint(control.checkpoint());
        let mut op = MockIntOperator::new(vec![1, 2, 3], 10);
        cancellation.cancel();

        let error = executor.execute_with_limit(&mut op, 0).unwrap_err();
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::QueryCancelled
        );
        assert_eq!(op.position, 0);
    }

    #[test]
    fn test_executor_no_timeout() {
        let executor = Executor::with_columns(vec!["value".to_string()]);
        let mut op = MockIntOperator::new(vec![1, 2, 3], 10);

        let result = executor.execute(&mut op).unwrap();
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn executor_checks_cancellation_after_the_final_pull() {
        for (pipeline, factorized) in [(false, false), (false, true), (true, false)] {
            let control = QueryExecutionControl::new();
            let mut operator = CancellingOperator {
                cancellation: control.cancellation_handle(),
                error_after_cancel: false,
                factorized,
            };
            let executor = Executor::with_columns(vec!["value".to_string()])
                .with_execution_checkpoint(control.checkpoint());
            let error = if pipeline {
                executor
                    .execute_pipeline(Box::new(operator), vec![])
                    .unwrap_err()
            } else {
                executor.execute(&mut operator).unwrap_err()
            };

            assert_eq!(
                error.error_code(),
                grafeo_common::utils::error::ErrorCode::QueryCancelled
            );
        }
    }

    #[test]
    fn concrete_operator_error_beats_racing_cancellation() {
        for (pipeline, factorized) in [(false, false), (false, true), (true, false)] {
            let control = QueryExecutionControl::new();
            let mut operator = CancellingOperator {
                cancellation: control.cancellation_handle(),
                error_after_cancel: true,
                factorized,
            };
            let executor = Executor::with_columns(vec!["value".to_string()])
                .with_execution_checkpoint(control.checkpoint());
            let error = if pipeline {
                executor
                    .execute_pipeline(Box::new(operator), vec![])
                    .unwrap_err()
            } else {
                executor.execute(&mut operator).unwrap_err()
            };

            assert!(matches!(error, Error::Internal(ref message) if message == "source failed"));
        }
    }

    #[test]
    fn test_executor_type_capture_from_first_chunk() {
        // When column_types are all Any, types should be captured from the first
        // non-empty chunk.
        let executor = Executor::with_columns(vec!["value".to_string()]);
        // column_types starts as [Any] from with_columns
        let mut op = MockIntOperator::new(vec![42, 99], 10);

        let result = executor.execute(&mut op).unwrap();
        assert_eq!(result.row_count(), 2);
        // After execution, column types should be captured as Int64
        assert_eq!(result.column_types, vec![LogicalType::Int64]);
    }

    #[test]
    fn test_executor_type_capture_with_explicit_types() {
        // When column_types are explicitly set (not all Any), types should NOT be
        // overwritten from chunks.
        let executor =
            Executor::with_columns_and_types(vec!["value".to_string()], vec![LogicalType::String]);
        let mut op = MockIntOperator::new(vec![1], 10);

        let result = executor.execute(&mut op).unwrap();
        assert_eq!(result.row_count(), 1);
        // Types should remain as explicitly set (String), not changed to Int64
        assert_eq!(result.column_types, vec![LogicalType::String]);
    }

    #[test]
    fn test_execute_pipeline_basic() {
        let source = Box::new(MockIntOperator::new(vec![10, 20, 30], 10));
        let executor = Executor::with_columns(vec!["value".to_string()]);

        let result = executor.execute_pipeline(source, vec![]).unwrap();
        assert_eq!(result.row_count(), 3);
        assert_eq!(result.rows()[0][0], Value::Int64(10));
        assert_eq!(result.rows()[1][0], Value::Int64(20));
        assert_eq!(result.rows()[2][0], Value::Int64(30));
    }

    #[test]
    fn test_execute_pipeline_empty_source() {
        let source = Box::new(EmptyOperator);
        let executor = Executor::with_columns(vec!["value".to_string()]);

        let result = executor.execute_pipeline(source, vec![]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_execute_pipeline_type_capture() {
        // Pipeline should capture column types from first non-empty chunk when
        // column_types are all Any.
        let source = Box::new(MockIntOperator::new(vec![1, 2], 10));
        let executor = Executor::with_columns(vec!["value".to_string()]);

        let result = executor.execute_pipeline(source, vec![]).unwrap();
        assert_eq!(result.column_types, vec![LogicalType::Int64]);
    }

    #[test]
    fn test_execute_pipeline_explicit_types_preserved() {
        // Pipeline should preserve explicitly set column types.
        let source = Box::new(MockIntOperator::new(vec![1], 10));
        let executor =
            Executor::with_columns_and_types(vec!["value".to_string()], vec![LogicalType::String]);

        let result = executor.execute_pipeline(source, vec![]).unwrap();
        // Explicit types should not be overwritten
        assert_eq!(result.column_types, vec![LogicalType::String]);
    }

    #[test]
    fn test_execute_with_limit_type_capture() {
        // execute_with_limit should also capture types from first chunk
        let executor = Executor::with_columns(vec!["value".to_string()]);
        let mut op = MockIntOperator::new(vec![1, 2, 3, 4, 5], 2);

        let result = executor.execute_with_limit(&mut op, 3).unwrap();
        assert_eq!(result.row_count(), 3);
        assert_eq!(result.column_types, vec![LogicalType::Int64]);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_execute_with_limit_timeout_expired() {
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        let executor = Executor::with_columns(vec!["value".to_string()])
            .with_execution_checkpoint(QueryExecutionControl::with_deadline(expired).checkpoint());
        let mut op = MockIntOperator::new(vec![1, 2, 3], 10);

        let result = executor.execute_with_limit(&mut op, 10);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Query exceeded timeout")
        );
    }

    #[test]
    fn test_convert_operator_error_variants() {
        // Test all OperatorError conversion branches
        let err = convert_operator_error(OperatorError::TypeMismatch {
            expected: "Int64".to_string(),
            found: "String".to_string(),
        });
        assert!(matches!(err, Error::TypeMismatch { .. }));

        let err = convert_operator_error(OperatorError::ColumnNotFound("col_x".to_string()));
        assert!(matches!(err, Error::InvalidValue(_)));
        assert!(err.to_string().contains("col_x"));

        let err = convert_operator_error(OperatorError::Execution("internal issue".to_string()));
        assert!(matches!(err, Error::Internal(_)));

        let err = convert_operator_error(OperatorError::QueryCancelled(
            grafeo_core::execution::QueryCancellationError::DeadlineExceeded {
                timeout: Some(Duration::from_secs(3)),
            },
        ));
        assert_eq!(
            err.error_code(),
            grafeo_common::utils::error::ErrorCode::QueryTimeout
        );
        assert!(err.to_string().contains("3s"));

        let err = convert_operator_error(OperatorError::QueryCancelled(
            grafeo_core::execution::QueryCancellationError::Cancelled,
        ));
        assert_eq!(
            err.error_code(),
            grafeo_common::utils::error::ErrorCode::QueryCancelled
        );

        let err = convert_operator_error(
            OperatorError::QueryCancelled(QueryCancellationError::InvalidState { phase: 255 })
                .with_context("cleanup also failed"),
        );
        assert!(
            matches!(&err, Error::Context { source, .. } if matches!(source.as_ref(), Error::Internal(_)))
        );
        assert!(err.to_string().contains("255"));
        assert!(err.to_string().contains("cleanup also failed"));

        for (reason, expected) in [
            (
                grafeo_core::execution::QueryCancellationError::DeadlineExceeded { timeout: None },
                grafeo_common::utils::error::ErrorCode::QueryTimeout,
            ),
            (
                grafeo_core::execution::QueryCancellationError::Cancelled,
                grafeo_common::utils::error::ErrorCode::QueryCancelled,
            ),
        ] {
            let contextual = convert_operator_error(
                OperatorError::QueryCancelled(reason).with_context("cleanup also failed"),
            );
            assert_eq!(contextual.error_code(), expected);
            assert!(contextual.to_string().contains("cleanup also failed"));
        }

        let err = convert_operator_error(OperatorError::ConstraintViolation("unique".to_string()));
        assert!(matches!(err, Error::InvalidValue(_)));
        assert!(err.to_string().contains("unique"));

        let err =
            convert_operator_error(OperatorError::WriteConflict("concurrent write".to_string()));
        assert!(matches!(err, Error::Transaction(_)));

        let err = convert_operator_error(OperatorError::ResidentMemory(
            grafeo_common::memory::buffer::MemoryGrantError::LimitExceeded {
                scope: grafeo_common::memory::buffer::MemoryLimitScope::Query,
                requested_bytes: 65,
                limit_bytes: 64,
            },
        ));
        assert_eq!(
            err.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );

        let err = convert_operator_error(OperatorError::ResidentMemory(
            grafeo_common::memory::buffer::MemoryGrantError::AccountingPoisoned { account: "test" },
        ));
        assert!(matches!(err, Error::Internal(_)));

        let err = convert_operator_error(
            OperatorError::ResidentMemory(
                grafeo_common::memory::buffer::MemoryGrantError::LimitExceeded {
                    scope: grafeo_common::memory::buffer::MemoryLimitScope::Query,
                    requested_bytes: 65,
                    limit_bytes: 64,
                },
            )
            .with_context("spill cleanup also failed"),
        );
        assert_eq!(
            err.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert!(err.to_string().contains("spill cleanup also failed"));

        let err = convert_operator_error(
            OperatorError::WriteConflict("concurrent writer".to_string())
                .with_context("row grant release also failed"),
        );
        assert_eq!(
            err.error_code(),
            grafeo_common::utils::error::ErrorCode::TransactionConflict
        );
        assert!(err.to_string().contains("row grant release also failed"));

        let err = convert_operator_error(OperatorError::ResidentAllocation(
            "allocator refused sort row".to_string(),
        ));
        assert_eq!(
            err.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );

        let mut impossible = Vec::<u8>::new();
        let catalog_error = impossible.try_reserve(usize::MAX).unwrap_err();
        let err = convert_operator_error(OperatorError::ResidentContainerAllocation {
            container: "partition catalog",
            source: catalog_error,
        });
        assert_eq!(
            err.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert!(err.to_string().contains("partition catalog"));

        let err = convert_operator_error(OperatorError::ResidentNativeMapAllocation {
            source: grafeo_core::execution::NativeMapAllocationError::capacity_overflow(),
        });
        assert_eq!(
            err.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert!(err.to_string().contains("native partition-map"));

        let err = convert_operator_error(OperatorError::ResidentNativeMapAllocationWithRollback {
            source: grafeo_core::execution::NativeMapAllocationError::capacity_overflow(),
            rollback: grafeo_common::memory::buffer::MemoryGrantError::AccountingPoisoned {
                account: "partition test",
            },
        });
        assert_eq!(
            err.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert!(
            err.to_string()
                .contains("partition grant rollback also failed")
        );

        let err = convert_operator_error(OperatorError::ResidentContainerInvariant {
            container: "native partition map",
            message: "allocation exceeded receipt",
        });
        assert!(matches!(err, Error::Internal(_)));
        assert!(err.to_string().contains("allocation exceeded receipt"));

        let err = convert_operator_error(OperatorError::ResidentContainerInvariantWithRollback {
            container: "native partition map",
            message: "allocation exceeded receipt",
            rollback: grafeo_common::memory::buffer::MemoryGrantError::AccountingPoisoned {
                account: "partition test",
            },
        });
        assert!(matches!(err, Error::Internal(_)));
        assert!(err.to_string().contains("grant rollback also failed"));

        let err = convert_operator_error(OperatorError::StorageFull(
            "per-query spill quota exceeded".to_string(),
        ));
        assert_eq!(
            err.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert!(err.to_string().contains("per-query spill quota exceeded"));
    }

    #[cfg(feature = "spill")]
    #[test]
    fn direct_exact_vector_allocation_reaches_engine_as_storage_full() {
        let error = convert_operator_error(OperatorError::ResidentExactVectorAllocation(
            grafeo_core::execution::ResidentCapacityError::ArithmeticOverflow {
                container: "exact sort output",
            },
        ));

        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert!(error.to_string().contains("exact sort output"));
    }

    #[cfg(feature = "spill")]
    #[test]
    fn classified_resident_limit_reaches_storage_full_with_authority_context() {
        use grafeo_common::memory::buffer::{MemoryGrantError, MemoryLimitScope};
        use grafeo_core::execution::operators::AccountedFailureClassification;

        let primary = MemoryGrantError::LimitExceeded {
            scope: MemoryLimitScope::Query,
            requested_bytes: 65,
            limit_bytes: 64,
        };
        let error = convert_operator_error(OperatorError::ClassifiedAccountedFailure {
            classification: AccountedFailureClassification::ResidentMemory(primary.clone()),
            authority: accounted_conversion_authority_with_primary(
                "accounted sort cleanup context",
                Some(primary),
            ),
        });

        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        let diagnostic = error.to_string();
        assert!(diagnostic.contains("requested 65 bytes, limit 64 bytes"));
        assert!(diagnostic.contains("accounted sort cleanup context"));
    }

    #[cfg(feature = "spill")]
    #[test]
    fn classified_cancellation_preserves_query_cancelled_code() {
        use grafeo_core::execution::operators::AccountedFailureClassification;

        let error = convert_operator_error(OperatorError::ClassifiedAccountedFailure {
            classification: AccountedFailureClassification::QueryCancelled(
                QueryCancellationError::Cancelled,
            ),
            authority: accounted_conversion_authority("accounted cancellation cleanup"),
        });

        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::QueryCancelled
        );
        assert!(error.to_string().contains("accounted cancellation cleanup"));
    }

    #[cfg(feature = "spill")]
    #[test]
    fn classified_public_error_retains_original_payload_and_grant_without_formatting() {
        use grafeo_common::memory::buffer::{
            AccountedErrorPublisher, BufferManager, MemoryGrant, MemoryRegion,
        };
        use grafeo_common::utils::error::{ErrorCode, RetainedErrorOwner};
        use grafeo_core::execution::operators::AccountedFailureClassification;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Observation {
            manager: Arc<BufferManager>,
            drops: AtomicUsize,
            drop_charge: AtomicUsize,
            formats: AtomicUsize,
            source_calls: AtomicUsize,
        }
        struct OriginalFailure {
            bytes: Box<[u8]>,
            observation: Arc<Observation>,
            // The exact original payload owns its admitted heap throughout Drop.
            grant: MemoryGrant,
        }
        impl std::fmt::Debug for OriginalFailure {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter
                    .debug_struct("OriginalFailure")
                    .finish_non_exhaustive()
            }
        }
        impl std::fmt::Display for OriginalFailure {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.observation.formats.fetch_add(1, Ordering::Relaxed);
                formatter.write_str("trusted retained conversion diagnostic")
            }
        }
        impl std::error::Error for OriginalFailure {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                self.observation
                    .source_calls
                    .fetch_add(1, Ordering::Relaxed);
                None
            }
        }
        impl Drop for OriginalFailure {
            fn drop(&mut self) {
                self.observation.drops.fetch_add(1, Ordering::Relaxed);
                self.observation
                    .drop_charge
                    .store(self.observation.manager.allocated(), Ordering::Relaxed);
            }
        }

        const PAYLOAD_BYTES: usize = 8192;
        for public_drops_first in [false, true] {
            let manager = BufferManager::with_budget(1 << 20);
            let observation = Arc::new(Observation {
                manager: Arc::clone(&manager),
                drops: AtomicUsize::new(0),
                drop_charge: AtomicUsize::new(0),
                formats: AtomicUsize::new(0),
                source_calls: AtomicUsize::new(0),
            });
            let publisher = AccountedErrorPublisher::try_new(
                manager
                    .try_allocate(0, MemoryRegion::ExecutionBuffers)
                    .unwrap(),
            )
            .unwrap();
            let grant = manager
                .try_allocate(PAYLOAD_BYTES, MemoryRegion::ExecutionBuffers)
                .unwrap();
            let bytes = vec![0x5a; PAYLOAD_BYTES].into_boxed_slice();
            let address = bytes.as_ptr() as usize;
            let authority = publisher.publish(OriginalFailure {
                bytes,
                observation: Arc::clone(&observation),
                grant,
            });
            let state_owner = authority.clone();
            let charged = manager.allocated();
            let error = convert_operator_error(OperatorError::ClassifiedAccountedFailure {
                classification: AccountedFailureClassification::StorageFull,
                authority,
            });
            assert_eq!(error.error_code(), ErrorCode::StorageFull);
            assert_eq!(observation.formats.load(Ordering::Relaxed), 0);
            assert_eq!(observation.source_calls.load(Ordering::Relaxed), 0);
            assert_eq!(manager.allocated(), charged);
            let source = std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<RetainedErrorOwner>()
                .unwrap();
            source
                .inspect::<OriginalFailure, _>(|original| {
                    assert_eq!(original.bytes.as_ptr() as usize, address);
                    assert_eq!(original.bytes.len(), PAYLOAD_BYTES);
                    assert_eq!(original.grant.size(), PAYLOAD_BYTES);
                })
                .unwrap();
            if public_drops_first {
                drop(error);
                assert_eq!(observation.drops.load(Ordering::Relaxed), 0);
                assert_eq!(manager.allocated(), charged);
                drop(state_owner);
            } else {
                drop(state_owner);
                assert_eq!(observation.drops.load(Ordering::Relaxed), 0);
                assert_eq!(manager.allocated(), charged);
                drop(error);
            }
            assert_eq!(observation.drops.load(Ordering::Relaxed), 1);
            assert_eq!(observation.drop_charge.load(Ordering::Relaxed), charged);
            assert_eq!(observation.formats.load(Ordering::Relaxed), 0);
            assert_eq!(observation.source_calls.load(Ordering::Relaxed), 0);
            assert_eq!(manager.allocated(), 0);
        }
    }

    #[cfg(feature = "spill")]
    #[test]
    fn classified_public_error_preserves_canonical_primary_codes() {
        use grafeo_common::memory::buffer::MemoryGrantError;
        use grafeo_common::utils::error::ErrorCode;
        use grafeo_core::execution::operators::AccountedFailureClassification as Classification;

        for (classification, expected) in [
            (Classification::TypeMismatch, ErrorCode::TypeMismatch),
            (Classification::ColumnNotFound, ErrorCode::InvalidInput),
            (Classification::ConstraintViolation, ErrorCode::InvalidInput),
            (
                Classification::WriteConflict,
                ErrorCode::TransactionConflict,
            ),
            (Classification::StorageFull, ErrorCode::StorageFull),
            (Classification::ResidentAllocation, ErrorCode::StorageFull),
            (
                Classification::ResidentExactVectorAllocation(
                    grafeo_core::execution::ResidentCapacityError::ArithmeticOverflow {
                        container: "conversion fixture",
                    },
                ),
                ErrorCode::StorageFull,
            ),
            (
                Classification::ResidentMemory(MemoryGrantError::Denied {
                    additional_bytes: 1,
                }),
                ErrorCode::StorageFull,
            ),
            (
                Classification::ResidentMemory(MemoryGrantError::AccountingPoisoned {
                    account: "conversion fixture",
                }),
                ErrorCode::Internal,
            ),
            (
                Classification::QueryCancelled(QueryCancellationError::Cancelled),
                ErrorCode::QueryCancelled,
            ),
            (
                Classification::QueryCancelled(QueryCancellationError::DeadlineExceeded {
                    timeout: Some(Duration::from_millis(5)),
                }),
                ErrorCode::QueryTimeout,
            ),
            (
                Classification::QueryCancelled(QueryCancellationError::DeadlineExceeded {
                    timeout: None,
                }),
                ErrorCode::QueryTimeout,
            ),
            (
                Classification::QueryCancelled(QueryCancellationError::InvalidState { phase: 255 }),
                ErrorCode::Internal,
            ),
            (
                Classification::UnsupportedAccountedTransport,
                ErrorCode::Internal,
            ),
            (Classification::ResidentInvariant, ErrorCode::Internal),
            (Classification::Execution, ErrorCode::Internal),
        ] {
            let error = convert_operator_error(OperatorError::ClassifiedAccountedFailure {
                classification,
                authority: accounted_conversion_authority("retained primary"),
            });
            assert_eq!(error.error_code(), expected);
            assert!(matches!(error, Error::RetainedContext { .. }));
        }
    }

    #[cfg(feature = "spill")]
    #[test]
    fn legacy_accounted_failure_tuple_remains_an_internal_source() {
        let legacy = OperatorError::AccountedFailure(accounted_conversion_authority(
            "legacy accounted operator source",
        ));
        let error = convert_operator_error(legacy);

        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::Internal
        );
        assert!(
            matches!(error, Error::Internal(ref message) if message.contains(
                "legacy accounted operator source"
            ))
        );
    }

    #[cfg(feature = "spill")]
    #[test]
    fn forced_sort_spill_quota_reaches_engine_boundary_as_storage_full() {
        use grafeo_core::execution::operators::SpillableSortPushOperator;
        use grafeo_core::execution::pipeline::PushOperator;
        use grafeo_core::execution::spill::{
            CleartextSpillRecordProvider, NoopSpillIo, SpillFrameLimits,
        };
        use grafeo_core::execution::{CollectorSink, SpillDiskQuota};
        use std::sync::Arc;

        let directory = tempfile::tempdir().unwrap();
        let (resources, manager) = crate::spill_crypto::admitted_spill_test_resources(
            directory.path(),
            grafeo_common::memory::buffer::BufferManager::with_budget(1 << 20),
            grafeo_core::execution::QueryExecutionControl::new().token(),
            Arc::new(CleartextSpillRecordProvider),
            SpillFrameLimits::format_max(),
            Arc::new(NoopSpillIo),
            SpillDiskQuota::new(0),
        );
        let mut sort = SpillableSortPushOperator::ascending_with_spilling(0, manager, 1);
        let mut chunk = DataChunk::with_capacity(&[LogicalType::Int64], 1);
        chunk.column_mut(0).unwrap().push_int64(1);
        chunk.set_count(1);

        let operator_error = sort.push(chunk, &mut CollectorSink::new()).unwrap_err();
        let error = convert_operator_error(operator_error);

        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert!(error.to_string().contains("spill disk quota exceeded"));
        drop(sort);
        resources.spill_manager().unwrap().finish_query().unwrap();
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_execute_pipeline_timeout_expired() {
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        for empty in [false, true] {
            let source = || -> Box<dyn Operator> {
                if empty {
                    Box::new(EmptyOperator)
                } else {
                    Box::new(MockIntOperator::new(vec![1, 2, 3], 10))
                }
            };
            let mut executor = Executor::with_columns(vec!["value".to_string()])
                .with_execution_checkpoint(
                    QueryExecutionControl::with_deadline(expired).checkpoint(),
                );

            let error = executor.execute_pipeline(source(), vec![]).unwrap_err();
            assert_eq!(
                error.error_code(),
                grafeo_common::utils::error::ErrorCode::QueryTimeout
            );
            assert!(error.to_string().contains("Query exceeded timeout"));
            let retry = executor.execute_pipeline(source(), vec![]).unwrap_err();
            assert_eq!(retry.error_code(), error.error_code());

            executor =
                executor.with_execution_checkpoint(QueryExecutionControl::new().checkpoint());
            let result = executor.execute_pipeline(source(), vec![]).unwrap();
            assert_eq!(result.row_count(), if empty { 0 } else { 3 });
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn executor_checkpoint_composition_keeps_earliest_deadline_and_diagnostic() {
        let now = Instant::now();
        let earlier = now.checked_sub(Duration::from_secs(2)).unwrap();
        let later = now.checked_sub(Duration::from_secs(1)).unwrap();
        for pipeline in [false, true] {
            for earlier_first in [false, true] {
                let checkpoint = QueryExecutionControl::new().checkpoint();
                let checkpoint = if earlier_first {
                    checkpoint
                        .with_additional_deadline(earlier, Some(Duration::from_secs(3)))
                        .with_additional_deadline(later, Some(Duration::from_secs(5)))
                } else {
                    checkpoint
                        .with_additional_deadline(later, Some(Duration::from_secs(5)))
                        .with_additional_deadline(earlier, Some(Duration::from_secs(3)))
                };
                let executor = Executor::with_columns(vec!["value".to_string()])
                    .with_execution_checkpoint(checkpoint);
                let error = if pipeline {
                    executor
                        .execute_pipeline(Box::new(EmptyOperator), vec![])
                        .unwrap_err()
                } else {
                    executor.execute(&mut EmptyOperator).unwrap_err()
                };
                assert_eq!(
                    error.error_code(),
                    grafeo_common::utils::error::ErrorCode::QueryTimeout
                );
                assert!(error.to_string().contains("3s timeout"));
            }
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn executor_checkpoint_preserves_first_terminal_reason() {
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        for pipeline in [false, true] {
            for cancellation_first in [false, true] {
                let control = QueryExecutionControl::new();
                let cancellation = control.cancellation_handle();
                let checkpoint = control
                    .checkpoint()
                    .with_additional_deadline(expired, Some(Duration::from_secs(3)));
                if cancellation_first {
                    cancellation.cancel();
                } else {
                    assert_eq!(
                        checkpoint.check().unwrap_err(),
                        QueryCancellationError::DeadlineExceeded {
                            timeout: Some(Duration::from_secs(3)),
                        }
                    );
                    cancellation.cancel();
                }
                let executor = Executor::with_columns(vec!["value".to_string()])
                    .with_execution_checkpoint(checkpoint);
                let error = if pipeline {
                    executor
                        .execute_pipeline(Box::new(EmptyOperator), vec![])
                        .unwrap_err()
                } else {
                    executor.execute(&mut EmptyOperator).unwrap_err()
                };
                if cancellation_first {
                    assert_eq!(
                        error.error_code(),
                        grafeo_common::utils::error::ErrorCode::QueryCancelled
                    );
                } else {
                    assert_eq!(
                        error.error_code(),
                        grafeo_common::utils::error::ErrorCode::QueryTimeout
                    );
                    assert!(error.to_string().contains("3s timeout"));
                }
            }
        }
    }

    fn bounded_executor(
        max_rows: usize,
        max_bytes: usize,
    ) -> (Executor<'static>, QueryResourceContext) {
        let resources = QueryResourceContext::new(
            grafeo_common::memory::buffer::BufferManager::with_budget(1_000_000),
        )
        .unwrap();
        (
            Executor::with_columns(vec!["value".into()]).with_result_resources(
                resources.clone(),
                ResultLimits {
                    max_rows,
                    max_bytes,
                },
            ),
            resources,
        )
    }

    #[test]
    fn bounded_pull_and_push_reject_overlimit_without_retaining_partial_output() {
        for pipeline in [false, true] {
            let (executor, resources) = bounded_executor(4, 100_000);
            let mut source = MockIntOperator::new(vec![1, 2, 3, 4, 5, 6], 3);
            let error = if pipeline {
                executor
                    .execute_pipeline(Box::new(source), vec![])
                    .unwrap_err()
            } else {
                executor.execute(&mut source).unwrap_err()
            };
            assert_eq!(
                error.error_code(),
                grafeo_common::utils::error::ErrorCode::StorageFull
            );
            assert!(error.to_string().contains("row limit"));
            assert_eq!(resources.query_stats().allocated_bytes, 0);
        }
    }

    #[test]
    fn bounded_dense_result_reserves_lazy_rows_and_retains_grant_through_extraction() {
        let (executor, resources) = bounded_executor(8, 100_000);
        let mut source = MockIntOperator::new(vec![1, 2, 3], 3);
        let result = executor.execute(&mut source).unwrap();
        assert!(result.is_int64_columnar());
        let before = resources.query_stats().allocated_bytes;
        assert!(before >= 3 * (size_of::<Vec<Value>>() + size_of::<Value>() + size_of::<i64>()));
        assert_eq!(
            result.rows(),
            &[
                vec![Value::Int64(1)],
                vec![Value::Int64(2)],
                vec![Value::Int64(3)]
            ]
        );
        assert_eq!(resources.query_stats().allocated_bytes, before);
        let rows = result.into_rows().unwrap();
        assert_eq!(resources.query_stats().allocated_bytes, before);
        drop(rows);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[test]
    fn bounded_dense_rejects_changed_width_before_retaining_short_columns() {
        let (executor, _) = bounded_executor(8, 100_000);
        let mut accumulator = executor
            .accumulator(&executor.execution_checkpoint().token())
            .unwrap();
        use grafeo_core::execution::vector::ValueVector;
        let column = |value| {
            let mut column = ValueVector::with_type(LogicalType::Int64);
            column.push_value(Value::Int64(value));
            column
        };
        let first = DataChunk::new(vec![column(1)]);
        accumulator.consume(&first, usize::MAX).unwrap();
        assert!(accumulator.result.is_int64_columnar());
        let wider = DataChunk::new(vec![column(2), column(3)]);
        assert!(
            accumulator
                .consume(&wider, usize::MAX)
                .unwrap_err()
                .to_string()
                .contains("column count changed")
        );
        assert_eq!(accumulator.finish().rows(), &[vec![Value::Int64(1)]]);
    }

    #[test]
    fn bounded_empty_result_accepts_zero_limits_with_metadata_granted() {
        let (executor, resources) = bounded_executor(0, 0);
        let mut source = EmptyOperator;
        let result = executor.execute(&mut source).unwrap();
        assert_eq!(result.row_count(), 0);
        assert_eq!(result.columns, ["value"]);
        assert!(resources.query_stats().allocated_bytes > 0);
        drop(result);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[test]
    fn bounded_generic_backing_is_admitted_before_clone() {
        let resources = QueryResourceContext::new(
            grafeo_common::memory::buffer::BufferManager::with_budget(1_000_000),
        )
        .unwrap();
        let mut accumulator = ResultAccumulator::new(
            &["value".into()],
            &[LogicalType::Any],
            resources.clone(),
            ResultLimits {
                max_rows: 10,
                max_bytes: 1024,
            },
        )
        .unwrap();
        let value = Value::from("x".repeat(4096));
        let column = grafeo_core::execution::vector::ValueVector::from_values(&[value]);
        let chunk = DataChunk::new(vec![column]);
        let error = accumulator.append_chunk(&chunk).unwrap_err();
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert_eq!(accumulator.result.row_count(), 0);
        drop(accumulator);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[test]
    fn bounded_selected_null_and_zero_column_rows_are_exact() {
        let resources = QueryResourceContext::new(
            grafeo_common::memory::buffer::BufferManager::with_budget(1_000_000),
        )
        .unwrap();
        let limits = ResultLimits {
            max_rows: 10,
            max_bytes: 100_000,
        };
        let mut accumulator = ResultAccumulator::new(
            &["value".into()],
            &[LogicalType::Any],
            resources.clone(),
            limits,
        )
        .unwrap();
        let column = grafeo_core::execution::vector::ValueVector::from_values(&[
            Value::Int64(7),
            Value::Null,
            Value::Int64(9),
        ]);
        let mut chunk = DataChunk::new(vec![column]);
        chunk.set_selection(
            grafeo_core::execution::selection::SelectionVector::from_predicate(3, |index| {
                index > 0
            }),
        );
        accumulator.append_chunk(&chunk).unwrap();
        assert_eq!(
            accumulator.finish().rows(),
            &[vec![Value::Null], vec![Value::Int64(9)]]
        );
        let mut accumulator = ResultAccumulator::new(&[], &[], resources, limits).unwrap();
        let mut chunk = DataChunk::with_capacity(&[], 3);
        chunk.set_count(3);
        accumulator.append_chunk(&chunk).unwrap();
        assert_eq!(accumulator.finish().rows(), &[vec![], vec![], vec![]]);
    }

    #[test]
    fn bounded_factorized_rows_preserve_multiplicity_without_flattening() {
        use grafeo_core::execution::{factorized_chunk::FactorizedChunk, vector::ValueVector};
        let mut chunk = FactorizedChunk::with_flat_level(
            vec![ValueVector::from_values(&[
                Value::Int64(10),
                Value::Int64(20),
            ])],
            vec!["parent".into()],
        );
        chunk.add_level(
            vec![ValueVector::from_values(&[
                Value::Int64(1),
                Value::Int64(2),
                Value::Int64(3),
            ])],
            vec!["child".into()],
            &[0, 2, 3],
        );
        for max_rows in [2, 3] {
            let resources = QueryResourceContext::new(
                grafeo_common::memory::buffer::BufferManager::with_budget(1_000_000),
            )
            .unwrap();
            let mut accumulator = ResultAccumulator::new(
                &["parent".into(), "child".into()],
                &[LogicalType::Any, LogicalType::Any],
                resources.clone(),
                ResultLimits {
                    max_rows,
                    max_bytes: 100_000,
                },
            )
            .unwrap();
            let outcome = accumulator.consume_factorized(&chunk, usize::MAX);
            if max_rows == 2 {
                assert_eq!(
                    outcome.unwrap_err().error_code(),
                    grafeo_common::utils::error::ErrorCode::StorageFull
                );
                drop(accumulator);
            } else {
                assert_eq!(outcome.unwrap(), 3);
                assert_eq!(
                    accumulator.finish().rows(),
                    &[
                        vec![Value::Int64(10), Value::Int64(1)],
                        vec![Value::Int64(10), Value::Int64(2)],
                        vec![Value::Int64(20), Value::Int64(3)]
                    ]
                );
            }
            assert_eq!(resources.query_stats().allocated_bytes, 0);
        }
    }

    #[test]
    fn bounded_formatter_stops_before_oversized_text_or_zero_row_report() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        for limits in [
            ResultLimits {
                max_rows: 0,
                max_bytes: 100_000,
            },
            ResultLimits {
                max_rows: 1,
                max_bytes: 8,
            },
        ] {
            let resources = QueryResourceContext::new(
                grafeo_common::memory::buffer::BufferManager::with_budget(1_000_000),
            )
            .unwrap();
            let error = bounded_text_result("report", resources.clone(), limits, |output| {
                write!(output, "0123456789")?;
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
            .unwrap_err();
            assert_eq!(
                error.error_code(),
                grafeo_common::utils::error::ErrorCode::StorageFull
            );
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            assert_eq!(resources.query_stats().allocated_bytes, 0);
        }
    }

    #[test]
    fn bounded_status_adopts_its_string_and_retains_metadata_grant() {
        let resources = QueryResourceContext::new(
            grafeo_common::memory::buffer::BufferManager::with_budget(1_000_000),
        )
        .unwrap();
        let result =
            bounded_status_result(resources.clone(), format_args!("Created {}", "世界")).unwrap();
        assert_eq!(result.status_message.as_deref(), Some("Created 世界"));
        assert_eq!(result.row_count(), 0);
        assert!(resources.query_stats().allocated_bytes >= "Created 世界".len());
        drop(result);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[test]
    fn bounded_executor_metadata_is_admitted_before_copy_and_operator_pull() {
        let resources = QueryResourceContext::new(
            grafeo_common::memory::buffer::BufferManager::with_budget(1024),
        )
        .unwrap();
        let columns = ["x".repeat(4096)];
        let executor =
            Executor::with_bounded_columns(&columns, resources.clone(), ResultLimits::default())
                .unwrap();
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        let mut operator = MockIntOperator::new(vec![1], 1);
        let error = executor.execute(&mut operator).unwrap_err();
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert_eq!(
            operator.position, 0,
            "metadata denial must precede the first pull"
        );
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[test]
    fn borrowed_executor_reuses_schema_and_results_retain_their_own_grants() {
        let resources = QueryResourceContext::new(
            grafeo_common::memory::buffer::BufferManager::with_budget(100_000),
        )
        .unwrap();
        let columns = ["value".into()];
        let metadata_only =
            ResultAccumulator::new(&columns, &[], resources.clone(), ResultLimits::default())
                .unwrap()
                .finish();
        assert!(metadata_only.column_types.is_empty());
        drop(metadata_only);
        let executor =
            Executor::with_bounded_columns(&columns, resources.clone(), ResultLimits::default())
                .unwrap();
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        let empty = executor.execute(&mut EmptyOperator).unwrap();
        assert_eq!(empty.column_types, [LogicalType::Any]);
        let empty_bytes = resources.query_stats().allocated_bytes;
        assert!(empty_bytes > 0);
        let result = executor
            .execute(&mut MockIntOperator::new(vec![42], 1))
            .unwrap();
        assert_eq!(result.column_types, [LogicalType::Int64]);
        assert_eq!(result.rows()[0], [Value::Int64(42)]);
        assert!(resources.query_stats().allocated_bytes > empty_bytes);
        drop(executor);
        drop(columns);
        assert_eq!(result.columns, ["value"]);
        drop(result);
        assert_eq!(resources.query_stats().allocated_bytes, empty_bytes);
        drop(empty);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }
}
