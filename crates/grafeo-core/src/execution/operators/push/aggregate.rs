//! Push-based aggregate operator (pipeline breaker).

use crate::execution::QueryCancellationToken;
use crate::execution::chunk::DataChunk;
use crate::execution::operators::OperatorError;
use crate::execution::operators::accumulator::{AggregateExpr, AggregateFunction, AggregateState};
use crate::execution::pipeline::{ChunkSizeHint, PushOperator, Sink};
#[cfg(feature = "spill")]
use crate::execution::spill::{PartitionOperationError, PartitionedState, SpillManager};
use crate::execution::vector::ValueVector;
use grafeo_common::types::Value;
use std::collections::HashMap;
#[cfg(feature = "spill")]
use std::io::{Read, Write};
#[cfg(feature = "spill")]
use std::sync::Arc;

/// Creates a new [`AggregateState`] from an [`AggregateExpr`].
fn state_for_expr(expr: &AggregateExpr) -> AggregateState {
    AggregateState::new(
        expr.function,
        expr.distinct,
        expr.percentile,
        expr.separator.as_deref(),
    )
}

/// Updates a single accumulator from a data chunk row, handling bivariate
/// functions, `CountNonNull` null-skipping, and `COUNT(*)`.
fn update_accumulator(
    acc: &mut AggregateState,
    expr: &AggregateExpr,
    chunk: &DataChunk,
    row: usize,
) -> Result<(), OperatorError> {
    let distinct_key = if expr.distinct {
        expr.distinct_key_column
            .map(|column_index| {
                let column = chunk
                    .column(column_index)
                    .ok_or_else(|| OperatorError::ColumnNotFound(column_index.to_string()))?;
                column.get_value(row).ok_or_else(|| {
                    OperatorError::Execution(format!(
                        "DISTINCT key column {column_index} had no value for row {row}"
                    ))
                })
            })
            .transpose()?
    } else {
        None
    };

    // Bivariate set functions (COVAR, CORR, REGR_*) need two column values
    if expr.column2.is_some() {
        let y_val = expr
            .column
            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row)));
        let x_val = expr
            .column2
            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row)));
        acc.update_bivariate_with_distinct_key(y_val, x_val, distinct_key);
        return Ok(());
    }

    if let Some(col) = expr.column {
        let val = chunk.column(col).and_then(|c| c.get_value(row));
        // CountNonNull must skip null values
        if expr.function == AggregateFunction::CountNonNull
            && matches!(val, None | Some(Value::Null))
        {
            return Ok(());
        }
        acc.update_with_distinct_key(val, distinct_key);
    } else {
        // COUNT(*)
        acc.update(None);
    }
    Ok(())
}

/// Shared input inspection for nonallocating resource declarations and the
/// post-update capacity delta. `get_value` retains immutable Value ownership.
#[cfg(feature = "spill")]
fn aggregate_row_inputs(
    expression: &AggregateExpr,
    chunk: &DataChunk,
    row: usize,
) -> (Option<Value>, Option<Value>, Option<Value>) {
    let read = |column| {
        chunk
            .column(column)
            .and_then(|values| values.get_value(row))
    };
    (
        expression.column.and_then(read),
        expression.column2.and_then(read),
        if expression.distinct {
            expression.distinct_key_column.and_then(read)
        } else {
            None
        },
    )
}

/// Hash key for grouping.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GroupKey(Vec<u64>);

impl GroupKey {
    fn from_row(chunk: &DataChunk, row: usize, group_by: &[usize]) -> Self {
        let hashes: Vec<u64> = group_by
            .iter()
            .map(|&col| {
                chunk
                    .column(col)
                    .and_then(|c| c.get_value(row))
                    .map_or(0, |v| hash_value(&v))
            })
            .collect();
        Self(hashes)
    }
}

fn hash_value(value: &Value) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    // Discriminant tag prevents cross-type collisions (e.g. Null vs unknown)
    match value {
        Value::Null => 0u8.hash(&mut hasher),
        Value::Bool(b) => {
            1u8.hash(&mut hasher);
            b.hash(&mut hasher);
        }
        Value::Int64(i) => {
            2u8.hash(&mut hasher);
            i.hash(&mut hasher);
        }
        Value::Float64(f) => {
            3u8.hash(&mut hasher);
            // Canonicalize -0.0 to +0.0 so the two zeros form one group.
            grafeo_common::types::canonical_f64_bits(*f).hash(&mut hasher);
        }
        Value::String(s) => {
            4u8.hash(&mut hasher);
            s.hash(&mut hasher);
        }
        Value::Bytes(b) => {
            5u8.hash(&mut hasher);
            b.hash(&mut hasher);
        }
        Value::Timestamp(t) => {
            6u8.hash(&mut hasher);
            t.hash(&mut hasher);
        }
        Value::Date(d) => {
            7u8.hash(&mut hasher);
            d.hash(&mut hasher);
        }
        Value::Time(t) => {
            8u8.hash(&mut hasher);
            t.hash(&mut hasher);
        }
        Value::Duration(d) => {
            9u8.hash(&mut hasher);
            d.hash(&mut hasher);
        }
        Value::ZonedDatetime(zdt) => {
            10u8.hash(&mut hasher);
            zdt.hash(&mut hasher);
        }
        Value::List(list) => {
            11u8.hash(&mut hasher);
            list.len().hash(&mut hasher);
            for elem in list.iter() {
                hash_value(elem).hash(&mut hasher);
            }
        }
        Value::Map(map) => {
            12u8.hash(&mut hasher);
            map.len().hash(&mut hasher);
            // BTreeMap iterates in key order, so hashing is deterministic
            for (k, v) in map.as_ref() {
                k.as_str().hash(&mut hasher);
                hash_value(v).hash(&mut hasher);
            }
        }
        Value::Vector(vec) => {
            13u8.hash(&mut hasher);
            vec.len().hash(&mut hasher);
            for f in vec.iter() {
                f.to_bits().hash(&mut hasher);
            }
        }
        Value::Path { nodes, edges } => {
            14u8.hash(&mut hasher);
            nodes.len().hash(&mut hasher);
            for n in nodes.iter() {
                hash_value(n).hash(&mut hasher);
            }
            for e in edges.iter() {
                hash_value(e).hash(&mut hasher);
            }
        }
        Value::GCounter(map) => {
            15u8.hash(&mut hasher);
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by_key(|(k, _)| *k);
            for (k, v) in entries {
                k.hash(&mut hasher);
                v.hash(&mut hasher);
            }
        }
        Value::OnCounter { pos, neg } => {
            16u8.hash(&mut hasher);
            let mut pos_entries: Vec<_> = pos.iter().collect();
            pos_entries.sort_by_key(|(k, _)| *k);
            for (k, v) in pos_entries {
                k.hash(&mut hasher);
                v.hash(&mut hasher);
            }
            let mut neg_entries: Vec<_> = neg.iter().collect();
            neg_entries.sort_by_key(|(k, _)| *k);
            for (k, v) in neg_entries {
                k.hash(&mut hasher);
                v.hash(&mut hasher);
            }
        }
        other => {
            255u8.hash(&mut hasher);
            std::mem::discriminant(other).hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Group state with key values and accumulators.
#[derive(Clone)]
pub(crate) struct GroupState {
    pub(crate) key_values: Vec<Value>,
    pub(crate) accumulators: Vec<AggregateState>,
    #[cfg(feature = "spill")]
    retained_heap_bytes_cache: Option<usize>,
}

#[cfg(feature = "spill")]
fn aggregate_capacity_overflow() -> grafeo_common::memory::buffer::MemoryGrantError {
    grafeo_common::memory::buffer::MemoryGrantError::ArithmeticOverflow {
        current_bytes: usize::MAX,
        additional_bytes: 1,
    }
}

#[cfg(feature = "spill")]
impl GroupState {
    fn retained_heap_bytes(
        &self,
    ) -> Result<usize, grafeo_common::memory::buffer::MemoryGrantError> {
        if let Some(bytes) = self.retained_heap_bytes_cache {
            return Ok(bytes);
        }
        let measured = || {
            let mut bytes = self
                .key_values
                .capacity()
                .checked_mul(std::mem::size_of::<Value>())?
                .checked_add(
                    self.accumulators
                        .capacity()
                        .checked_mul(std::mem::size_of::<AggregateState>())?,
                )?;
            for value in &self.key_values {
                bytes = bytes.checked_add(
                    value
                        .retained_size_bytes()?
                        .checked_sub(std::mem::size_of::<Value>())?,
                )?;
            }
            for accumulator in &self.accumulators {
                bytes = bytes.checked_add(accumulator.retained_heap_bytes()?)?;
            }
            Some(bytes)
        };
        measured().ok_or_else(aggregate_capacity_overflow)
    }
}

/// Push-based aggregate operator.
///
/// This is a pipeline breaker that accumulates all input, groups by key,
/// and produces aggregated output in the finalize phase.
pub struct AggregatePushOperator {
    /// Columns to group by.
    group_by: Vec<usize>,
    /// Aggregate expressions.
    aggregates: Vec<AggregateExpr>,
    /// Group states by hash key.
    groups: HashMap<GroupKey, GroupState>,
    /// Global accumulator (for no GROUP BY).
    global_state: Option<Vec<AggregateState>>,
    /// Zero-byte foothold for future checked aggregate-container growth.
    /// Declared after physical state so those allocations drop first.
    _grant: Option<grafeo_common::memory::buffer::MemoryGrant>,
    /// Check-only capability present only for resource-qualified execution.
    cancellation: Option<QueryCancellationToken>,
}

impl AggregatePushOperator {
    /// Create a new aggregate operator.
    pub fn new(group_by: Vec<usize>, aggregates: Vec<AggregateExpr>) -> Self {
        let global_state = if group_by.is_empty() {
            Some(aggregates.iter().map(state_for_expr).collect())
        } else {
            None
        };

        Self {
            group_by,
            aggregates,
            groups: HashMap::new(),
            global_state,
            _grant: None,
            cancellation: None,
        }
    }

    /// Creates a resident aggregate carrying this execution's immutable
    /// cancellation capability.
    ///
    /// The retained zero-byte grant establishes the fallible API and ownership
    /// needed for later checked hash-table/output growth. This stage does not
    /// yet charge those aggregate containers.
    ///
    /// # Errors
    ///
    /// Returns a structured resource error if the query grant cannot be
    /// created.
    pub fn with_resource_context(
        group_by: Vec<usize>,
        aggregates: Vec<AggregateExpr>,
        context: crate::execution::memory::QueryResourceContext,
    ) -> Result<Self, crate::execution::memory::QueryResourceContextError> {
        let cancellation = context.cancellation_token().clone();
        let mut aggregate = Self::new(group_by, aggregates);
        aggregate._grant = Some(context.try_allocate(0)?);
        aggregate.cancellation = Some(cancellation);
        Ok(aggregate)
    }

    /// Create a simple global aggregate (no GROUP BY).
    pub fn global(aggregates: Vec<AggregateExpr>) -> Self {
        Self::new(Vec::new(), aggregates)
    }
}

impl PushOperator for AggregatePushOperator {
    fn push(&mut self, chunk: DataChunk, _sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        let cancellation = self.cancellation.clone();
        poll_cancellation(cancellation.as_ref())?;
        if chunk.is_empty() {
            return Ok(true);
        }

        for row in chunk.selected_indices() {
            poll_cancellation(cancellation.as_ref())?;
            if self.group_by.is_empty() {
                // Global aggregation
                if let Some(ref mut accumulators) = self.global_state {
                    for (acc, expr) in accumulators.iter_mut().zip(&self.aggregates) {
                        update_accumulator(acc, expr, &chunk, row)?;
                    }
                }
            } else {
                // Group by aggregation
                let key = GroupKey::from_row(&chunk, row, &self.group_by);

                let state = self.groups.entry(key).or_insert_with(|| {
                    let key_values: Vec<Value> = self
                        .group_by
                        .iter()
                        .map(|&col| {
                            chunk
                                .column(col)
                                .and_then(|c| c.get_value(row))
                                .unwrap_or(Value::Null)
                        })
                        .collect();

                    GroupState {
                        #[cfg(feature = "spill")]
                        retained_heap_bytes_cache: None,
                        key_values,
                        accumulators: self.aggregates.iter().map(state_for_expr).collect(),
                    }
                });

                for (acc, expr) in state.accumulators.iter_mut().zip(&self.aggregates) {
                    update_accumulator(acc, expr, &chunk, row)?;
                }
            }
            poll_cancellation(cancellation.as_ref())?;
        }

        Ok(true)
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        let cancellation = self.cancellation.clone();
        poll_cancellation(cancellation.as_ref())?;
        let num_output_cols = self.group_by.len() + self.aggregates.len();
        let mut columns: Vec<ValueVector> =
            (0..num_output_cols).map(|_| ValueVector::new()).collect();

        if self.group_by.is_empty() {
            // Global aggregation - single row output
            poll_cancellation(cancellation.as_ref())?;
            if let Some(ref accumulators) = self.global_state {
                for (i, acc) in accumulators.iter().enumerate() {
                    columns[i].push(acc.finalize());
                }
            }
            poll_cancellation(cancellation.as_ref())?;
        } else {
            // Group by - one row per group
            for state in self.groups.values() {
                poll_cancellation(cancellation.as_ref())?;
                // Output group key columns
                for (i, val) in state.key_values.iter().enumerate() {
                    columns[i].push(val.clone());
                }

                // Output aggregate results
                for (i, acc) in state.accumulators.iter().enumerate() {
                    columns[self.group_by.len() + i].push(acc.finalize());
                }
                poll_cancellation(cancellation.as_ref())?;
            }
        }

        if !columns.is_empty() && !columns[0].is_empty() {
            poll_cancellation(cancellation.as_ref())?;
            let chunk = DataChunk::new(columns);
            sink.consume(chunk)?;
        }

        Ok(())
    }

    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        ChunkSizeHint::Default
    }

    fn name(&self) -> &'static str {
        "AggregatePush"
    }
}

fn poll_cancellation(cancellation: Option<&QueryCancellationToken>) -> Result<(), OperatorError> {
    if let Some(cancellation) = cancellation {
        cancellation.check()?;
    }
    Ok(())
}

/// Default spill threshold for aggregates (number of groups).
#[cfg(feature = "spill")]
pub const DEFAULT_AGGREGATE_SPILL_THRESHOLD: usize = 50_000;

/// Minimum number of groups before memory-pressure spilling can trigger.
///
/// Prevents "noisy neighbor" scenarios where a tiny aggregate buffer gets
/// spilled because unrelated subsystems consumed memory.
#[cfg(feature = "spill")]
const AGGREGATE_MIN_BUFFER_GROUPS: usize = 500;

/// Each tag names a live accumulator representation. Spill files are scoped to
/// one query, so writer and reader evolve together beneath the existing frame.
#[cfg(feature = "spill")]
mod spill_tag {
    pub const COUNT: u8 = 0;
    pub const SUM_INT: u8 = 1;
    pub const SUM_FLOAT: u8 = 2;
    pub const AVG: u8 = 3;
    pub const MIN: u8 = 4;
    pub const MAX: u8 = 5;
    pub const FIRST: u8 = 6;
    pub const LAST: u8 = 7;
    pub const COLLECT: u8 = 8;
    pub const COUNT_DISTINCT: u8 = 9;
    pub const SUM_INT_DISTINCT: u8 = 10;
    pub const SUM_FLOAT_DISTINCT: u8 = 11;
    pub const AVG_DISTINCT: u8 = 12;
    pub const COLLECT_DISTINCT: u8 = 13;
    pub const GROUP_CONCAT: u8 = 14;
    pub const GROUP_CONCAT_DISTINCT: u8 = 15;
    pub const STDDEV: u8 = 16;
    pub const STDDEV_POP: u8 = 17;
    pub const VARIANCE: u8 = 18;
    pub const VARIANCE_POP: u8 = 19;
    pub const PERCENTILE_DISC: u8 = 20;
    pub const PERCENTILE_CONT: u8 = 21;
    pub const BIVARIATE: u8 = 22;
    pub const SAMPLE: u8 = 23;
    pub const MIN_DISTINCT: u8 = 24;
    pub const MAX_DISTINCT: u8 = 25;
    pub const LAST_DISTINCT: u8 = 26;
    /// Only an explicitly terminal Frozen state uses this tag.
    pub const FINALIZED: u8 = 255;
}

#[cfg(feature = "spill")]
fn spill_invalid(message: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

#[cfg(feature = "spill")]
fn write_count(writer: &mut dyn Write, count: usize) -> std::io::Result<()> {
    let count = u64::try_from(count).map_err(|_| spill_invalid("aggregate count exceeds u64"))?;
    writer.write_all(&count.to_le_bytes())
}

#[cfg(feature = "spill")]
fn read_u64(reader: &mut dyn Read) -> std::io::Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(feature = "spill")]
fn read_count(reader: &mut dyn Read) -> std::io::Result<usize> {
    usize::try_from(read_u64(reader)?)
        .map_err(|_| spill_invalid("aggregate count is not addressable"))
}

#[cfg(feature = "spill")]
fn read_i64(reader: &mut dyn Read) -> std::io::Result<i64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(i64::from_le_bytes(bytes))
}

#[cfg(feature = "spill")]
fn write_samples(writer: &mut dyn Write, count: i64) -> std::io::Result<()> {
    if count < 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "negative aggregate sample count",
        ));
    }
    writer.write_all(&count.to_le_bytes())
}

#[cfg(feature = "spill")]
fn read_samples(reader: &mut dyn Read) -> std::io::Result<i64> {
    let count = read_i64(reader)?;
    if count < 0 {
        return Err(spill_invalid("negative aggregate sample count"));
    }
    Ok(count)
}

#[cfg(feature = "spill")]
fn read_f64(reader: &mut dyn Read) -> std::io::Result<f64> {
    Ok(f64::from_bits(read_u64(reader)?))
}

#[cfg(feature = "spill")]
fn read_flag(reader: &mut dyn Read) -> std::io::Result<bool> {
    let mut byte = [0];
    reader.read_exact(&mut byte)?;
    match byte[0] {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(spill_invalid("invalid aggregate presence flag")),
    }
}

#[cfg(feature = "spill")]
fn spill_reserve<T>(values: &mut Vec<T>, count: usize) -> std::io::Result<()> {
    values.try_reserve_exact(count).map_err(|error| {
        std::io::Error::other(format!(
            "cannot reserve restored aggregate collection: {error}"
        ))
    })
}

/// Conservative hashbrown backing bytes beyond the live identity slots.
/// Empty buckets and control bytes consume the byte grant, not logical items;
/// tagged String/Other identity bytes are charged separately by the identity codec.
#[cfg(feature = "spill")]
fn seen_storage_bound(count: usize) -> std::io::Result<usize> {
    if count == 0 {
        return Ok(0);
    }
    let buckets = count
        .checked_mul(2)
        .and_then(usize::checked_next_power_of_two)
        .map(|count| count.max(4))
        .ok_or_else(|| spill_invalid("aggregate DISTINCT backing size overflow"))?;
    let controls = buckets
        .checked_add(64)
        .ok_or_else(|| spill_invalid("aggregate DISTINCT control size overflow"))?;
    (buckets - count)
        .checked_mul(std::mem::size_of::<
            crate::execution::operators::accumulator::HashableValue,
        >())
        .and_then(|bytes| bytes.checked_add(controls))
        .ok_or_else(|| spill_invalid("aggregate DISTINCT backing size overflow"))
}

/// Preserve the accumulator's identity enum, including its tagged Debug-text
/// fallback. Converting Other(text) through Value::String would change the key
/// variant and admit a duplicate after reload (notably for bivariate pairs).
#[cfg(feature = "spill")]
fn write_identity(
    identity: &crate::execution::operators::accumulator::HashableValue,
    writer: &mut dyn Write,
    budget: &mut crate::execution::spill::SpillCodecEncodeBudget,
) -> std::io::Result<()> {
    use crate::execution::operators::accumulator::HashableValue;
    match identity {
        HashableValue::Null => writer.write_all(&[0]),
        HashableValue::Bool(value) => writer.write_all(&[1, u8::from(*value)]),
        HashableValue::Int64(value) => {
            writer.write_all(&[2])?;
            writer.write_all(&value.to_le_bytes())
        }
        HashableValue::Float64Bits(value) => {
            writer.write_all(&[3])?;
            writer.write_all(&value.to_le_bytes())
        }
        HashableValue::String(value) => {
            writer.write_all(&[4])?;
            write_string(value, writer, budget)
        }
        HashableValue::Other(value) => {
            writer.write_all(&[5])?;
            write_string(value, writer, budget)
        }
    }
}

#[cfg(feature = "spill")]
fn read_identity(
    reader: &mut dyn Read,
    budget: &mut crate::execution::spill::SpillCodecDecodeBudget,
) -> std::io::Result<crate::execution::operators::accumulator::HashableValue> {
    use crate::execution::operators::accumulator::HashableValue;
    let mut tag = [0];
    reader.read_exact(&mut tag)?;
    match tag[0] {
        0 => Ok(HashableValue::Null),
        1 => Ok(HashableValue::Bool(read_flag(reader)?)),
        2 => Ok(HashableValue::Int64(read_i64(reader)?)),
        3 => Ok(HashableValue::Float64Bits(read_u64(reader)?)),
        4 => Ok(HashableValue::String(read_string(reader, budget)?)),
        5 => Ok(HashableValue::Other(read_string(reader, budget)?)),
        _ => Err(spill_invalid("unknown aggregate DISTINCT identity tag")),
    }
}

#[cfg(feature = "spill")]
fn write_seen(
    seen: &grafeo_common::utils::hash::FxHashSet<
        crate::execution::operators::accumulator::HashableValue,
    >,
    writer: &mut dyn Write,
    budget: &mut crate::execution::spill::SpillCodecEncodeBudget,
) -> std::io::Result<()> {
    let backing_bytes = seen_storage_bound(seen.len())?;
    budget.charge_items::<crate::execution::operators::accumulator::HashableValue>(
        seen.len(),
        "DISTINCT identities",
    )?;
    budget.charge_storage_bytes(backing_bytes, "DISTINCT hash backing")?;
    write_count(writer, seen.len())?;
    for key in seen {
        write_identity(key, writer, budget)?;
    }
    Ok(())
}

#[cfg(feature = "spill")]
fn read_seen(
    reader: &mut dyn Read,
    budget: &mut crate::execution::spill::SpillCodecDecodeBudget,
) -> std::io::Result<
    grafeo_common::utils::hash::FxHashSet<crate::execution::operators::accumulator::HashableValue>,
> {
    let count = read_count(reader)?;
    let backing_bytes = seen_storage_bound(count)?;
    budget.charge_items::<crate::execution::operators::accumulator::HashableValue>(
        count,
        "DISTINCT identities",
    )?;
    budget.charge_storage_bytes(backing_bytes, "DISTINCT hash backing")?;
    let mut seen = grafeo_common::utils::hash::FxHashSet::default();
    seen.try_reserve(count).map_err(|error| {
        std::io::Error::other(format!("cannot reserve restored DISTINCT keys: {error}"))
    })?;
    for _ in 0..count {
        let key = read_identity(reader, budget)?;
        if !seen.insert(key) {
            return Err(spill_invalid(
                "duplicate serialized aggregate DISTINCT identity",
            ));
        }
    }
    Ok(seen)
}

#[cfg(feature = "spill")]
fn write_optional_seen(
    seen: &Option<
        grafeo_common::utils::hash::FxHashSet<
            crate::execution::operators::accumulator::HashableValue,
        >,
    >,
    writer: &mut dyn Write,
    budget: &mut crate::execution::spill::SpillCodecEncodeBudget,
) -> std::io::Result<()> {
    writer.write_all(&[u8::from(seen.is_some())])?;
    if let Some(seen) = seen {
        write_seen(seen, writer, budget)?;
    }
    Ok(())
}

#[cfg(feature = "spill")]
fn read_optional_seen(
    reader: &mut dyn Read,
    budget: &mut crate::execution::spill::SpillCodecDecodeBudget,
) -> std::io::Result<
    Option<
        grafeo_common::utils::hash::FxHashSet<
            crate::execution::operators::accumulator::HashableValue,
        >,
    >,
> {
    if read_flag(reader)? {
        Ok(Some(read_seen(reader, budget)?))
    } else {
        Ok(None)
    }
}

#[cfg(feature = "spill")]
fn write_string(
    value: &str,
    writer: &mut dyn Write,
    budget: &mut crate::execution::spill::SpillCodecEncodeBudget,
) -> std::io::Result<()> {
    budget.charge_items::<u8>(value.len(), "aggregate string bytes")?;
    write_count(writer, value.len())?;
    writer.write_all(value.as_bytes())
}

#[cfg(feature = "spill")]
fn read_string(
    reader: &mut dyn Read,
    budget: &mut crate::execution::spill::SpillCodecDecodeBudget,
) -> std::io::Result<String> {
    let count = read_count(reader)?;
    budget.charge_items::<u8>(count, "aggregate string bytes")?;
    let mut bytes = Vec::new();
    spill_reserve(&mut bytes, count)?;
    bytes.resize(count, 0);
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|_| spill_invalid("aggregate string is not UTF-8"))
}

#[cfg(feature = "spill")]
fn bivariate_kind_tag(kind: AggregateFunction) -> std::io::Result<u8> {
    match kind {
        AggregateFunction::CovarSamp => Ok(0),
        AggregateFunction::CovarPop => Ok(1),
        AggregateFunction::Corr => Ok(2),
        AggregateFunction::RegrSlope => Ok(3),
        AggregateFunction::RegrIntercept => Ok(4),
        AggregateFunction::RegrR2 => Ok(5),
        AggregateFunction::RegrCount => Ok(6),
        AggregateFunction::RegrSxx => Ok(7),
        AggregateFunction::RegrSyy => Ok(8),
        AggregateFunction::RegrSxy => Ok(9),
        AggregateFunction::RegrAvgx => Ok(10),
        AggregateFunction::RegrAvgy => Ok(11),
        _ => Err(spill_invalid("non-bivariate function in bivariate state")),
    }
}

#[cfg(feature = "spill")]
fn read_bivariate_kind(reader: &mut dyn Read) -> std::io::Result<AggregateFunction> {
    let mut byte = [0];
    reader.read_exact(&mut byte)?;
    match byte[0] {
        0 => Ok(AggregateFunction::CovarSamp),
        1 => Ok(AggregateFunction::CovarPop),
        2 => Ok(AggregateFunction::Corr),
        3 => Ok(AggregateFunction::RegrSlope),
        4 => Ok(AggregateFunction::RegrIntercept),
        5 => Ok(AggregateFunction::RegrR2),
        6 => Ok(AggregateFunction::RegrCount),
        7 => Ok(AggregateFunction::RegrSxx),
        8 => Ok(AggregateFunction::RegrSyy),
        9 => Ok(AggregateFunction::RegrSxy),
        10 => Ok(AggregateFunction::RegrAvgx),
        11 => Ok(AggregateFunction::RegrAvgy),
        _ => Err(spill_invalid("unknown bivariate aggregate function")),
    }
}

/// Serializes live state, including compensation and DISTINCT identity. The
/// existing frame and Value codec bound every dynamic restored collection.
#[cfg(feature = "spill")]
fn serialize_group_state_bounded(
    state: &GroupState,
    writer: &mut dyn Write,
    codec_limits: crate::execution::spill::CodecLimits,
) -> std::io::Result<()> {
    use crate::execution::spill::SpillCodecEncodeBudget;
    let mut budget = SpillCodecEncodeBudget::new(codec_limits);
    budget.charge_items::<Value>(state.key_values.len(), "group key slot")?;
    write_count(writer, state.key_values.len())?;
    for value in &state.key_values {
        budget.encode_value(value, writer)?;
    }
    budget.charge_items::<AggregateState>(state.accumulators.len(), "group accumulator slot")?;
    write_count(writer, state.accumulators.len())?;
    for state in &state.accumulators {
        let tag = match state {
            AggregateState::Count(_) => spill_tag::COUNT,
            AggregateState::CountDistinct(..) => spill_tag::COUNT_DISTINCT,
            AggregateState::SumInt(..) => spill_tag::SUM_INT,
            AggregateState::SumIntDistinct(..) => spill_tag::SUM_INT_DISTINCT,
            AggregateState::SumFloat(..) => spill_tag::SUM_FLOAT,
            AggregateState::SumFloatDistinct(..) => spill_tag::SUM_FLOAT_DISTINCT,
            AggregateState::Avg(..) => spill_tag::AVG,
            AggregateState::AvgDistinct(..) => spill_tag::AVG_DISTINCT,
            AggregateState::Min(_) => spill_tag::MIN,
            AggregateState::MinDistinct(..) => spill_tag::MIN_DISTINCT,
            AggregateState::Max(_) => spill_tag::MAX,
            AggregateState::MaxDistinct(..) => spill_tag::MAX_DISTINCT,
            AggregateState::First(_) => spill_tag::FIRST,
            AggregateState::Last(_) => spill_tag::LAST,
            AggregateState::LastDistinct(..) => spill_tag::LAST_DISTINCT,
            AggregateState::Sample(_) => spill_tag::SAMPLE,
            AggregateState::Collect(_) => spill_tag::COLLECT,
            AggregateState::CollectDistinct(..) => spill_tag::COLLECT_DISTINCT,
            AggregateState::GroupConcat(..) => spill_tag::GROUP_CONCAT,
            AggregateState::GroupConcatDistinct(..) => spill_tag::GROUP_CONCAT_DISTINCT,
            AggregateState::StdDev { .. } => spill_tag::STDDEV,
            AggregateState::StdDevPop { .. } => spill_tag::STDDEV_POP,
            AggregateState::Variance { .. } => spill_tag::VARIANCE,
            AggregateState::VariancePop { .. } => spill_tag::VARIANCE_POP,
            AggregateState::PercentileDisc { .. } => spill_tag::PERCENTILE_DISC,
            AggregateState::PercentileCont { .. } => spill_tag::PERCENTILE_CONT,
            AggregateState::Bivariate { .. } => spill_tag::BIVARIATE,
            AggregateState::Frozen(_) => spill_tag::FINALIZED,
        };
        writer.write_all(&[tag])?;
        match state {
            AggregateState::Count(count) | AggregateState::CountDistinct(count, _) => {
                write_samples(writer, *count)?;
            }
            AggregateState::SumInt(sum, count) | AggregateState::SumIntDistinct(sum, count, _) => {
                writer.write_all(&sum.to_le_bytes())?;
                write_samples(writer, *count)?;
            }
            AggregateState::SumFloat(sum, comp, count)
            | AggregateState::SumFloatDistinct(sum, comp, count, _) => {
                writer.write_all(&sum.to_le_bytes())?;
                writer.write_all(&comp.to_le_bytes())?;
                write_samples(writer, *count)?;
            }
            AggregateState::Avg(sum, count) | AggregateState::AvgDistinct(sum, count, _) => {
                writer.write_all(&sum.to_le_bytes())?;
                write_samples(writer, *count)?;
            }
            AggregateState::Min(value)
            | AggregateState::MinDistinct(value, _)
            | AggregateState::Max(value)
            | AggregateState::MaxDistinct(value, _)
            | AggregateState::First(value)
            | AggregateState::Last(value)
            | AggregateState::LastDistinct(value, _)
            | AggregateState::Sample(value) => {
                writer.write_all(&[u8::from(value.is_some())])?;
                if let Some(value) = value {
                    budget.encode_value(value, writer)?;
                }
            }
            AggregateState::Collect(values) | AggregateState::CollectDistinct(values, _) => {
                budget.charge_items::<Value>(values.len(), "collected value slots")?;
                write_count(writer, values.len())?;
                for value in values {
                    budget.encode_value(value, writer)?;
                }
            }
            AggregateState::GroupConcat(values, separator)
            | AggregateState::GroupConcatDistinct(values, separator, _) => {
                budget.charge_items::<String>(values.len(), "collected string slots")?;
                write_count(writer, values.len())?;
                for value in values {
                    write_string(value, writer, &mut budget)?;
                }
                write_string(separator, writer, &mut budget)?;
            }
            AggregateState::StdDev {
                count,
                mean,
                m2,
                seen,
            }
            | AggregateState::StdDevPop {
                count,
                mean,
                m2,
                seen,
            }
            | AggregateState::Variance {
                count,
                mean,
                m2,
                seen,
            }
            | AggregateState::VariancePop {
                count,
                mean,
                m2,
                seen,
            } => {
                write_samples(writer, *count)?;
                writer.write_all(&mean.to_le_bytes())?;
                writer.write_all(&m2.to_le_bytes())?;
                write_optional_seen(seen, writer, &mut budget)?;
            }
            AggregateState::PercentileDisc {
                values,
                percentile,
                seen,
            }
            | AggregateState::PercentileCont {
                values,
                percentile,
                seen,
            } => {
                budget.charge_items::<f64>(values.len(), "percentile operand slots")?;
                write_count(writer, values.len())?;
                for value in values {
                    writer.write_all(&value.to_le_bytes())?;
                }
                writer.write_all(&percentile.to_le_bytes())?;
                write_optional_seen(seen, writer, &mut budget)?;
            }
            AggregateState::Bivariate {
                kind,
                count,
                mean_x,
                mean_y,
                m2_x,
                m2_y,
                c_xy,
                seen,
            } => {
                writer.write_all(&[bivariate_kind_tag(*kind)?])?;
                write_samples(writer, *count)?;
                for value in [mean_x, mean_y, m2_x, m2_y, c_xy] {
                    writer.write_all(&value.to_le_bytes())?;
                }
                write_optional_seen(seen, writer, &mut budget)?;
            }
            AggregateState::Frozen(value) => {
                budget.encode_value(value, writer)?;
            }
        }
        match state {
            AggregateState::CountDistinct(_, seen)
            | AggregateState::SumIntDistinct(_, _, seen)
            | AggregateState::SumFloatDistinct(_, _, _, seen)
            | AggregateState::AvgDistinct(_, _, seen)
            | AggregateState::CollectDistinct(_, seen)
            | AggregateState::MinDistinct(_, seen)
            | AggregateState::MaxDistinct(_, seen)
            | AggregateState::LastDistinct(_, seen)
            | AggregateState::GroupConcatDistinct(_, _, seen) => {
                write_seen(seen, writer, &mut budget)?;
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(all(feature = "spill", test))]
fn serialize_group_state(state: &GroupState, writer: &mut dyn Write) -> std::io::Result<()> {
    serialize_group_state_bounded(
        state,
        writer,
        crate::execution::spill::CodecLimits::format_max(),
    )
}

#[cfg(feature = "spill")]
fn deserialize_group_state_bounded(
    reader: &mut dyn Read,
    codec_limits: crate::execution::spill::CodecLimits,
) -> std::io::Result<GroupState> {
    use crate::execution::spill::SpillCodecDecodeBudget;
    let mut budget = SpillCodecDecodeBudget::new(codec_limits);
    let count = read_count(reader)?;
    budget.charge_items::<Value>(count, "group key slot")?;
    let mut key_values = Vec::new();
    spill_reserve(&mut key_values, count)?;
    for _ in 0..count {
        key_values.push(budget.decode_value(reader)?);
    }
    let count = read_count(reader)?;
    budget.charge_items::<AggregateState>(count, "group accumulator slot")?;
    let mut accumulators = Vec::new();
    spill_reserve(&mut accumulators, count)?;
    for _ in 0..count {
        let mut tag = [0];
        reader.read_exact(&mut tag)?;
        let state = match tag[0] {
            spill_tag::COUNT | spill_tag::COUNT_DISTINCT => {
                let count = read_samples(reader)?;
                if tag[0] == spill_tag::COUNT {
                    AggregateState::Count(count)
                } else {
                    AggregateState::CountDistinct(count, read_seen(reader, &mut budget)?)
                }
            }
            spill_tag::SUM_INT | spill_tag::SUM_INT_DISTINCT => {
                let sum = read_i64(reader)?;
                let count = read_samples(reader)?;
                if tag[0] == spill_tag::SUM_INT {
                    AggregateState::SumInt(sum, count)
                } else {
                    AggregateState::SumIntDistinct(sum, count, read_seen(reader, &mut budget)?)
                }
            }
            spill_tag::SUM_FLOAT | spill_tag::SUM_FLOAT_DISTINCT => {
                let sum = read_f64(reader)?;
                let comp = read_f64(reader)?;
                let count = read_samples(reader)?;
                if tag[0] == spill_tag::SUM_FLOAT {
                    AggregateState::SumFloat(sum, comp, count)
                } else {
                    AggregateState::SumFloatDistinct(
                        sum,
                        comp,
                        count,
                        read_seen(reader, &mut budget)?,
                    )
                }
            }
            spill_tag::AVG | spill_tag::AVG_DISTINCT => {
                let sum = read_f64(reader)?;
                let count = read_samples(reader)?;
                if tag[0] == spill_tag::AVG {
                    AggregateState::Avg(sum, count)
                } else {
                    AggregateState::AvgDistinct(sum, count, read_seen(reader, &mut budget)?)
                }
            }
            spill_tag::MIN
            | spill_tag::MIN_DISTINCT
            | spill_tag::MAX
            | spill_tag::MAX_DISTINCT
            | spill_tag::FIRST
            | spill_tag::LAST
            | spill_tag::LAST_DISTINCT
            | spill_tag::SAMPLE => {
                let value = if read_flag(reader)? {
                    Some(budget.decode_value(reader)?)
                } else {
                    None
                };
                match tag[0] {
                    spill_tag::MIN => AggregateState::Min(value),
                    spill_tag::MIN_DISTINCT => {
                        AggregateState::MinDistinct(value, read_seen(reader, &mut budget)?)
                    }
                    spill_tag::MAX => AggregateState::Max(value),
                    spill_tag::MAX_DISTINCT => {
                        AggregateState::MaxDistinct(value, read_seen(reader, &mut budget)?)
                    }
                    spill_tag::FIRST => AggregateState::First(value),
                    spill_tag::LAST => AggregateState::Last(value),
                    spill_tag::LAST_DISTINCT => {
                        AggregateState::LastDistinct(value, read_seen(reader, &mut budget)?)
                    }
                    _ => AggregateState::Sample(value),
                }
            }
            spill_tag::COLLECT | spill_tag::COLLECT_DISTINCT => {
                let count = read_count(reader)?;
                budget.charge_items::<Value>(count, "collected value slots")?;
                let mut values = Vec::new();
                spill_reserve(&mut values, count)?;
                for _ in 0..count {
                    values.push(budget.decode_value(reader)?);
                }
                if tag[0] == spill_tag::COLLECT {
                    AggregateState::Collect(values)
                } else {
                    AggregateState::CollectDistinct(values, read_seen(reader, &mut budget)?)
                }
            }
            spill_tag::GROUP_CONCAT | spill_tag::GROUP_CONCAT_DISTINCT => {
                let count = read_count(reader)?;
                budget.charge_items::<String>(count, "collected string slots")?;
                let mut values = Vec::new();
                spill_reserve(&mut values, count)?;
                for _ in 0..count {
                    values.push(read_string(reader, &mut budget)?);
                }
                let separator = read_string(reader, &mut budget)?;
                if tag[0] == spill_tag::GROUP_CONCAT {
                    AggregateState::GroupConcat(values, separator)
                } else {
                    AggregateState::GroupConcatDistinct(
                        values,
                        separator,
                        read_seen(reader, &mut budget)?,
                    )
                }
            }
            spill_tag::STDDEV
            | spill_tag::STDDEV_POP
            | spill_tag::VARIANCE
            | spill_tag::VARIANCE_POP => {
                let count = read_samples(reader)?;
                let mean = read_f64(reader)?;
                let m2 = read_f64(reader)?;
                let seen = read_optional_seen(reader, &mut budget)?;
                match tag[0] {
                    spill_tag::STDDEV => AggregateState::StdDev {
                        count,
                        mean,
                        m2,
                        seen,
                    },
                    spill_tag::STDDEV_POP => AggregateState::StdDevPop {
                        count,
                        mean,
                        m2,
                        seen,
                    },
                    spill_tag::VARIANCE => AggregateState::Variance {
                        count,
                        mean,
                        m2,
                        seen,
                    },
                    _ => AggregateState::VariancePop {
                        count,
                        mean,
                        m2,
                        seen,
                    },
                }
            }
            spill_tag::PERCENTILE_DISC | spill_tag::PERCENTILE_CONT => {
                let count = read_count(reader)?;
                budget.charge_items::<f64>(count, "percentile operand slots")?;
                let mut values = Vec::new();
                spill_reserve(&mut values, count)?;
                for _ in 0..count {
                    values.push(read_f64(reader)?);
                }
                let percentile = read_f64(reader)?;
                let seen = read_optional_seen(reader, &mut budget)?;
                if tag[0] == spill_tag::PERCENTILE_DISC {
                    AggregateState::PercentileDisc {
                        values,
                        percentile,
                        seen,
                    }
                } else {
                    AggregateState::PercentileCont {
                        values,
                        percentile,
                        seen,
                    }
                }
            }
            spill_tag::BIVARIATE => AggregateState::Bivariate {
                kind: read_bivariate_kind(reader)?,
                count: read_samples(reader)?,
                mean_x: read_f64(reader)?,
                mean_y: read_f64(reader)?,
                m2_x: read_f64(reader)?,
                m2_y: read_f64(reader)?,
                c_xy: read_f64(reader)?,
                seen: read_optional_seen(reader, &mut budget)?,
            },
            spill_tag::FINALIZED => AggregateState::Frozen(budget.decode_value(reader)?),
            _ => return Err(spill_invalid("unknown spilled aggregate-state tag")),
        };
        accumulators.push(state);
    }
    let mut state = GroupState {
        retained_heap_bytes_cache: None,
        key_values,
        accumulators,
    };
    state.retained_heap_bytes_cache = Some(
        state
            .retained_heap_bytes()
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::OutOfMemory, error))?,
    );
    Ok(state)
}

#[cfg(all(test, feature = "spill"))]
fn deserialize_group_state(reader: &mut dyn Read) -> std::io::Result<GroupState> {
    deserialize_group_state_bounded(reader, crate::execution::spill::CodecLimits::format_max())
}

/// Push-based aggregate operator with spilling support.
///
/// Uses partitioned hash table that can spill cold partitions to disk
/// when memory pressure is high.
///
/// Three construction modes are supported:
///
/// 1. **Resource-context mode** (when constructed with
///    `with_resource_context`): registers as a scoped `MemoryConsumer` and
///    spills when system pressure is High/Critical or eviction is requested.
///    Its current byte estimate is diagnostic pressure input, not a resident
///    grant; exact container charging is a later bounded-operator stage.
///
/// 2. **Row-count spill mode** (when constructed with `with_spilling`): spills
///    when the number of groups reaches its configured threshold.
///
/// 3. **Resident compatibility mode** (when constructed with `new`): no spill
///    manager is attached, so the row threshold cannot spill or bound the hash
///    table. This mode may grow without bound and is not bounded execution.
///
/// The former memory-context constructor is unavailable:
///
/// ```compile_fail,E0599
/// use grafeo_core::execution::operators::push::SpillableAggregatePushOperator;
/// let _ = SpillableAggregatePushOperator::with_memory_context;
/// ```
#[cfg(feature = "spill")]
pub struct SpillableAggregatePushOperator {
    /// Exact scoped registration. Declared first so it deactivates before any
    /// state a future callback adapter could reference is destroyed.
    _consumer_registration: Option<grafeo_common::memory::buffer::ConsumerRegistration>,
    /// Columns to group by.
    group_by: Vec<usize>,
    /// Aggregate expressions.
    aggregates: Vec<AggregateExpr>,
    /// Spill manager for explicit row-count mode.
    ///
    /// Resource-context mode owns its manager through partitioned state;
    /// resident compatibility mode has no manager.
    spill_manager: Option<Arc<SpillManager>>,
    /// Partitioned groups (used when spilling is enabled).
    partitioned_groups: Option<PartitionedState<GroupState>>,
    /// Non-partitioned groups (used when spilling is disabled).
    groups: HashMap<GroupKey, GroupState>,
    /// Global accumulator (for no GROUP BY).
    global_state: Option<Vec<AggregateState>>,
    /// Group threshold used only when an explicit spill manager is attached.
    spill_threshold: usize,
    /// Whether we've switched to partitioned mode.
    using_partitioned: bool,
    /// Memory context for pressure-aware spilling.
    memory_ctx: Option<crate::execution::memory::QueryResourceContext>,
    /// Shared state with the registered MemoryConsumer adapter.
    spill_state: Option<std::sync::Arc<super::spill_state::OperatorSpillState>>,
    /// Running total of estimated group memory in bytes (incremental tracking).
    estimated_bytes: usize,
    /// Fixed expression/catalog owner, after its physical allocations.
    metadata_grant: Option<grafeo_common::memory::buffer::MemoryGrant>,
}

#[cfg(feature = "spill")]
struct AggregateMetadataAdmission {
    group_by: Vec<usize>,
    aggregates: Vec<AggregateExpr>,
    // Constructor failures destroy the moved parameters before authority.
    grant: Option<grafeo_common::memory::buffer::MemoryGrant>,
}

#[cfg(feature = "spill")]
impl SpillableAggregatePushOperator {
    /// Creates a resident compatibility aggregate operator.
    ///
    /// No spill manager is attached, so this constructor does not spill and
    /// does not bound resident memory. Use [`Self::with_spilling`] for explicit
    /// row-threshold spill mode.
    pub fn new(group_by: Vec<usize>, aggregates: Vec<AggregateExpr>) -> Self {
        let global_state = if group_by.is_empty() {
            Some(aggregates.iter().map(state_for_expr).collect())
        } else {
            None
        };

        Self {
            group_by,
            aggregates,
            spill_manager: None,
            partitioned_groups: None,
            groups: HashMap::new(),
            global_state,
            spill_threshold: DEFAULT_AGGREGATE_SPILL_THRESHOLD,
            using_partitioned: false,
            memory_ctx: None,
            spill_state: None,
            _consumer_registration: None,
            estimated_bytes: 0,
            metadata_grant: None,
        }
    }

    /// Create a spillable aggregate operator with spilling enabled (row-count mode).
    pub fn with_spilling(
        group_by: Vec<usize>,
        aggregates: Vec<AggregateExpr>,
        manager: Arc<SpillManager>,
        threshold: usize,
    ) -> Self {
        let global_state = if group_by.is_empty() {
            Some(aggregates.iter().map(state_for_expr).collect())
        } else {
            None
        };

        let partitioned = PartitionedState::new_with_bounded_codec(
            Arc::clone(&manager),
            256, // Number of partitions
            serialize_group_state_bounded,
            deserialize_group_state_bounded,
        );

        Self {
            group_by,
            aggregates,
            spill_manager: Some(manager),
            partitioned_groups: Some(partitioned),
            groups: HashMap::new(),
            global_state,
            spill_threshold: threshold,
            using_partitioned: true,
            memory_ctx: None,
            spill_state: None,
            _consumer_registration: None,
            estimated_bytes: 0,
            metadata_grant: None,
        }
    }

    /// Create a spillable aggregate operator with memory-aware spilling.
    ///
    /// Registers as a `MemoryConsumer` with the `BufferManager` and spills
    /// based on system memory pressure rather than group count thresholds.
    /// This compatibility constructor estimates pressure; it does not impose
    /// the qualified query path's resident-memory admission contract.
    ///
    /// # Errors
    ///
    /// Returns a structured error if no spill manager is attached or the
    /// scoped consumer registration identity cannot be created.
    pub fn with_resource_context(
        group_by: Vec<usize>,
        aggregates: Vec<AggregateExpr>,
        ctx: crate::execution::memory::QueryResourceContext,
    ) -> Result<Self, crate::execution::memory::QueryResourceContextError> {
        use super::spill_state::{OperatorConsumerAdapter, OperatorSpillState};

        let global_state = if group_by.is_empty() {
            Some(aggregates.iter().map(state_for_expr).collect())
        } else {
            None
        };

        // Pre-create partitioned state using the spill manager from memory context
        let spill_manager = ctx
            .ensure_spill_manager()?
            .cloned()
            .ok_or(crate::execution::memory::QueryResourceContextError::SpillManagerUnavailable)?;
        let state = std::sync::Arc::new(OperatorSpillState::new(
            "SpillableAggregatePush".to_string(),
        ));
        let adapter =
            std::sync::Arc::new(OperatorConsumerAdapter::new(std::sync::Arc::clone(&state)));
        let consumer_registration = ctx.register_consumer_scoped(adapter)?;
        let cancellation = ctx.cancellation_token().clone();
        let partitioned = PartitionedState::new_with_bounded_codec_and_cancellation(
            spill_manager,
            256,
            serialize_group_state_bounded,
            deserialize_group_state_bounded,
            cancellation,
        );

        Ok(Self {
            group_by,
            aggregates,
            spill_manager: None,
            partitioned_groups: Some(partitioned),
            groups: HashMap::new(),
            global_state,
            spill_threshold: DEFAULT_AGGREGATE_SPILL_THRESHOLD,
            using_partitioned: true,
            memory_ctx: Some(ctx),
            spill_state: Some(state),
            _consumer_registration: Some(consumer_registration),
            estimated_bytes: 0,
            metadata_grant: None,
        })
    }

    /// Constructs the admitted grouped owner used by query pipeline conversion.
    /// The public resource constructor retains pressure-based compatibility.
    pub(crate) fn with_qualified_resource_context(
        group_by: Vec<usize>,
        aggregates: Vec<AggregateExpr>,
        ctx: crate::execution::memory::QueryResourceContext,
    ) -> Result<Self, crate::execution::memory::QueryResourceContextError> {
        use super::spill_state::{OperatorConsumerAdapter, OperatorSpillState};

        // The grouped route owns its expression storage and consumer controls
        // before constructing any group or registering an eviction callback.
        let metadata_grant = if group_by.is_empty() {
            None
        } else {
            let capacity = || {
                let mut bytes = group_by
                    .capacity()
                    .checked_mul(std::mem::size_of::<usize>())?
                    .checked_add(
                        aggregates
                            .capacity()
                            .checked_mul(std::mem::size_of::<AggregateExpr>())?,
                    )?;
                for expression in &aggregates {
                    for text in [&expression.alias, &expression.separator]
                        .into_iter()
                        .flatten()
                    {
                        bytes = bytes.checked_add(text.capacity())?;
                    }
                }
                bytes
                    .checked_add(std::mem::size_of::<OperatorSpillState>())?
                    .checked_add(std::mem::size_of::<OperatorConsumerAdapter>())?
                    .checked_add(4usize.checked_mul(std::mem::size_of::<usize>())?)?
                    .checked_add("SpillableAggregatePush".len())
            };
            Some(ctx.try_allocate(capacity().ok_or_else(aggregate_capacity_overflow)?)?)
        };
        let metadata = AggregateMetadataAdmission {
            group_by,
            aggregates,
            grant: metadata_grant,
        };
        let global_state = if metadata.group_by.is_empty() {
            Some(metadata.aggregates.iter().map(state_for_expr).collect())
        } else {
            None
        };
        let spill_manager = ctx
            .ensure_spill_manager()?
            .cloned()
            .ok_or(crate::execution::memory::QueryResourceContextError::SpillManagerUnavailable)?;
        let cancellation = ctx.cancellation_token().clone();
        let partitioned = if metadata.group_by.is_empty() {
            PartitionedState::new_with_bounded_codec_and_cancellation(
                spill_manager,
                256,
                serialize_group_state_bounded,
                deserialize_group_state_bounded,
                cancellation,
            )
        } else {
            PartitionedState::new_accounted_admitted_with_cancellation(
                spill_manager,
                256,
                serialize_group_state_bounded,
                deserialize_group_state_bounded,
                GroupState::retained_heap_bytes,
                ctx.try_allocate(0)?,
                cancellation,
            )?
        };
        let state = Arc::new(OperatorSpillState::new(
            "SpillableAggregatePush".to_string(),
        ));
        let adapter = Arc::new(OperatorConsumerAdapter::new(Arc::clone(&state)));
        let consumer_registration = ctx.register_consumer_scoped(adapter)?;
        let mut operator = Self {
            group_by: metadata.group_by,
            aggregates: metadata.aggregates,
            spill_manager: None,
            partitioned_groups: Some(partitioned),
            groups: HashMap::new(),
            global_state,
            spill_threshold: DEFAULT_AGGREGATE_SPILL_THRESHOLD,
            using_partitioned: true,
            memory_ctx: Some(ctx),
            spill_state: Some(state),
            _consumer_registration: Some(consumer_registration),
            estimated_bytes: 0,
            metadata_grant: metadata.grant,
        };
        operator.refresh_estimated_usage();
        Ok(operator)
    }

    /// Create a simple global aggregate (no GROUP BY).
    pub fn global(aggregates: Vec<AggregateExpr>) -> Self {
        Self::new(Vec::new(), aggregates)
    }

    /// Sets the threshold for explicit row-count spill mode.
    ///
    /// This does not attach a spill manager, so it does not make a value from
    /// [`Self::new`] spill-capable.
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.spill_threshold = threshold;
        self
    }

    fn ingest_chunk(&mut self, chunk: &DataChunk) -> Result<bool, OperatorError> {
        let cancellation = self
            .memory_ctx
            .as_ref()
            .map(|context| context.cancellation_token().clone());
        poll_cancellation(cancellation.as_ref())?;
        if chunk.is_empty() {
            return Ok(true);
        }

        for row in chunk.selected_indices() {
            if let Err(primary) = self.update_row(chunk, row) {
                self.refresh_estimated_usage();
                return Err(primary);
            }
            self.poll_after_group_mutation(cancellation.as_ref())?;
        }

        // Update memory consumer usage estimate
        self.refresh_estimated_usage();
        poll_cancellation(cancellation.as_ref())?;

        // Check if we need to spill
        self.maybe_spill()?;

        Ok(true)
    }
    fn resource_child_grant(
        &mut self,
    ) -> Result<grafeo_common::memory::buffer::MemoryGrant, OperatorError> {
        self.metadata_grant
            .as_mut()
            .and_then(|grant| grant.split(0))
            .ok_or(OperatorError::ResidentContainerInvariant {
                container: "aggregate resource owner",
                message: "grouped resource execution lost its metadata authority",
            })
    }

    fn update_accounted_row(&mut self, chunk: &DataChunk, row: usize) -> Result<(), OperatorError> {
        use crate::execution::spill::PartitionUpdateAdmission;

        // Declare the temporary key before allocation. The partition admits
        // its builder under the same one-spill budget as the group mutation.
        let key_bytes = self
            .group_by
            .iter()
            .try_fold(0usize, |bytes, &column| {
                let value = chunk
                    .column(column)
                    .and_then(|values| values.get_value(row))
                    .unwrap_or(Value::Null);
                bytes.checked_add(value.retained_size_bytes()?)
            })
            .ok_or_else(aggregate_capacity_overflow)?;
        let group_by = &self.group_by;
        let aggregates = &self.aggregates;
        let partitioned =
            self.partitioned_groups
                .as_mut()
                .ok_or(OperatorError::ResidentContainerInvariant {
                    container: "aggregate resource owner",
                    message: "grouped resource execution lost its partitions",
                })?;
        partitioned
            .try_update_accounted(
                key_bytes,
                || {
                    let mut key = Vec::new();
                    key.try_reserve_exact(group_by.len()).map_err(|source| {
                        OperatorError::ResidentContainerAllocation {
                            container: "aggregate row key",
                            source,
                        }
                    })?;
                    for &column in group_by {
                        key.push(
                            chunk
                                .column(column)
                                .and_then(|values| values.get_value(row))
                                .unwrap_or(Value::Null),
                        );
                    }
                    Ok(key)
                },
                |previous| {
                    // The longest diagnostic has fewer than 64 literal bytes
                    // and two at-most-20-digit usize values on supported targets.
                    // 256 bytes covers its final String plus a doubled backing
                    // while format! grows; column-only diagnostics are smaller.
                    let mut extra = 256usize;
                    if previous.is_none() {
                        extra = extra
                            .checked_add(key_bytes)
                            .and_then(|bytes| {
                                bytes.checked_add(
                                    aggregates
                                        .len()
                                        .checked_mul(std::mem::size_of::<AggregateState>())?,
                                )
                            })
                            .ok_or_else(aggregate_capacity_overflow)?;
                    }
                    for (index, expression) in aggregates.iter().enumerate() {
                        let (value, second, distinct_key) =
                            aggregate_row_inputs(expression, chunk, row);
                        let peak = match previous.and_then(|state| state.accumulators.get(index)) {
                            Some(accumulator) => accumulator.update_peak_bytes(
                                expression,
                                value.as_ref(),
                                second.as_ref(),
                                distinct_key.as_ref(),
                            ),
                            None => AggregateState::replacement_peak_bytes(
                                None,
                                expression,
                                value.as_ref(),
                                second.as_ref(),
                                distinct_key.as_ref(),
                            ),
                        }
                        .ok_or_else(aggregate_capacity_overflow)?;
                        extra = extra
                            .checked_add(peak)
                            .ok_or_else(aggregate_capacity_overflow)?;
                    }
                    let previous_bytes = previous.map_or(Ok(0), GroupState::retained_heap_bytes)?;
                    let retained_upper_bound = previous_bytes
                        .checked_add(extra)
                        .ok_or_else(aggregate_capacity_overflow)?;
                    // The custom group codec emits at most H + 16 bytes; the
                    // row key emits at most key_bytes + 8. Four u64 partition
                    // fields give P <= H + key_bytes + 56. Four P leaves room
                    // for two codec staging copies plus counter scratch, or a
                    // retained record plus the cleartext provider's two-copy
                    // seal/open allowance. This is scheduling policy only:
                    // actual provider bounds still admit independently and may
                    // refuse an operation. No old collection is rescanned.
                    let working_bytes = retained_upper_bound
                        .checked_add(key_bytes)
                        .and_then(|bytes| bytes.checked_add(7 * std::mem::size_of::<u64>()))
                        .and_then(|bytes| bytes.checked_mul(4))
                        .ok_or_else(aggregate_capacity_overflow)?;
                    Ok((
                        PartitionUpdateAdmission {
                            retained_upper_bound,
                            construction_peak: extra,
                        },
                        working_bytes,
                    ))
                },
                || {
                    let mut key_values = Vec::new();
                    key_values
                        .try_reserve_exact(group_by.len())
                        .map_err(|source| OperatorError::ResidentContainerAllocation {
                            container: "aggregate group key",
                            source,
                        })?;
                    for &column in group_by {
                        key_values.push(
                            chunk
                                .column(column)
                                .and_then(|values| values.get_value(row))
                                .unwrap_or(Value::Null),
                        );
                    }
                    let mut accumulators = Vec::new();
                    accumulators
                        .try_reserve_exact(aggregates.len())
                        .map_err(|source| OperatorError::ResidentContainerAllocation {
                            container: "aggregate accumulator slots",
                            source,
                        })?;
                    accumulators.extend(aggregates.iter().map(state_for_expr));
                    let mut state = GroupState {
                        key_values,
                        accumulators,
                        retained_heap_bytes_cache: None,
                    };
                    state.retained_heap_bytes_cache = Some(state.retained_heap_bytes()?);
                    Ok(state)
                },
                |state| {
                    let mut retained = state.retained_heap_bytes()?;
                    for (accumulator, expression) in state.accumulators.iter_mut().zip(aggregates) {
                        let before = accumulator
                            .retained_update_snapshot()
                            .ok_or_else(aggregate_capacity_overflow)?;
                        update_accumulator(accumulator, expression, chunk, row)?;
                        let (value, second, distinct_key) =
                            aggregate_row_inputs(expression, chunk, row);
                        let (removed, added) = accumulator
                            .retained_update_delta(
                                before,
                                expression,
                                value.as_ref(),
                                second.as_ref(),
                                distinct_key.as_ref(),
                            )
                            .ok_or_else(aggregate_capacity_overflow)?;
                        retained = retained
                            .checked_sub(removed)
                            .and_then(|bytes| bytes.checked_add(added))
                            .ok_or_else(aggregate_capacity_overflow)?;
                    }
                    state.retained_heap_bytes_cache = Some(retained);
                    Ok(())
                },
            )
            .map_err(Self::map_partition_error)
    }

    fn owns_grouped_resources(&self) -> bool {
        self.metadata_grant.is_some()
    }

    fn update_row(&mut self, chunk: &DataChunk, row: usize) -> Result<(), OperatorError> {
        if self.owns_grouped_resources() {
            return self.update_accounted_row(chunk, row);
        }
        let resource_qualified = self.memory_ctx.is_some();
        if self.group_by.is_empty() {
            if let Some(ref mut accumulators) = self.global_state {
                for (acc, expr) in accumulators.iter_mut().zip(&self.aggregates) {
                    update_accumulator(acc, expr, chunk, row)?;
                }
            }
        } else if self.using_partitioned {
            if let Some(ref mut partitioned) = self.partitioned_groups {
                let key_values: Vec<Value> = self
                    .group_by
                    .iter()
                    .map(|&column| {
                        chunk
                            .column(column)
                            .and_then(|values| values.get_value(row))
                            .unwrap_or(Value::Null)
                    })
                    .collect();
                let aggregates = &self.aggregates;
                let make_state = || GroupState {
                    #[cfg(feature = "spill")]
                    retained_heap_bytes_cache: None,
                    key_values: key_values.clone(),
                    accumulators: aggregates.iter().map(state_for_expr).collect(),
                };
                let state = if resource_qualified {
                    partitioned
                        .get_or_insert_with_controlled(key_values.clone(), make_state)
                        .map_err(Self::map_partition_error)?
                } else {
                    partitioned
                        .get_or_insert_with(key_values.clone(), make_state)
                        .map_err(OperatorError::from_spill_io_error)?
                };
                for (accumulator, expression) in state.accumulators.iter_mut().zip(&self.aggregates)
                {
                    update_accumulator(accumulator, expression, chunk, row)?;
                }
            }
        } else {
            let key = GroupKey::from_row(chunk, row, &self.group_by);
            let state = self.groups.entry(key).or_insert_with(|| {
                let key_values: Vec<Value> = self
                    .group_by
                    .iter()
                    .map(|&column| {
                        chunk
                            .column(column)
                            .and_then(|values| values.get_value(row))
                            .unwrap_or(Value::Null)
                    })
                    .collect();
                GroupState {
                    #[cfg(feature = "spill")]
                    retained_heap_bytes_cache: None,
                    key_values,
                    accumulators: self.aggregates.iter().map(state_for_expr).collect(),
                }
            });
            for (accumulator, expression) in state.accumulators.iter_mut().zip(&self.aggregates) {
                update_accumulator(accumulator, expression, chunk, row)?;
            }
        }
        Ok(())
    }

    fn refresh_estimated_usage(&mut self) {
        let Some(spill_state) = &self.spill_state else {
            return;
        };
        if self.owns_grouped_resources() {
            self.estimated_bytes = self
                .partitioned_groups
                .as_ref()
                .map_or(0, PartitionedState::granted_bytes)
                .saturating_add(
                    self.metadata_grant
                        .as_ref()
                        .map_or(0, grafeo_common::memory::buffer::MemoryGrant::size),
                );
            spill_state.set_usage(self.estimated_bytes);
            return;
        }
        let group_count = if self.using_partitioned {
            self.partitioned_groups
                .as_ref()
                .map_or(0, PartitionedState::total_size)
        } else {
            self.groups.len()
        };
        let key_size = self
            .group_by
            .len()
            .saturating_mul(std::mem::size_of::<Value>());
        let accumulator_size = self.aggregates.len().saturating_mul(64);
        let bytes_per_group = key_size.saturating_add(accumulator_size).saturating_add(48);
        self.estimated_bytes = group_count.saturating_mul(bytes_per_group);
        spill_state.set_usage(self.estimated_bytes);
    }

    fn map_partition_error(error: PartitionOperationError) -> OperatorError {
        // Grant denials and arithmetic overflow travel through the partition
        // I/O owner so any heap-bearing payload keeps its authority. Recover
        // the typed memory failure before consuming that owner; flattening it
        // to an execution string would misclassify ordinary resource
        // exhaustion as an internal engine defect.
        if matches!(&error, PartitionOperationError::Io(_))
            && let Some(memory_error) = error.resident_memory_error()
        {
            return OperatorError::ResidentMemory(memory_error.clone());
        }
        match error {
            PartitionOperationError::Accounted {
                classification,
                authority,
            } => OperatorError::ClassifiedAccountedFailure {
                classification,
                authority,
            },
            PartitionOperationError::Memory(error) => OperatorError::ResidentMemory(error),
            PartitionOperationError::Admission(error) => {
                use crate::execution::spill::PartitionAdmissionError;
                match error {
                    PartitionAdmissionError::Memory(error) => OperatorError::ResidentMemory(error),
                    PartitionAdmissionError::Allocation { container, error } => {
                        OperatorError::ResidentContainerAllocation {
                            container,
                            source: error,
                        }
                    }
                    PartitionAdmissionError::PublisherAllocation => {
                        OperatorError::ResidentContainerInvariant {
                            container: "partition failure publisher",
                            message: "allocator refused publisher storage",
                        }
                    }
                    PartitionAdmissionError::PublisherAllocationWithRollback(rollback) => {
                        OperatorError::ResidentContainerInvariantWithRollback {
                            container: "partition failure publisher",
                            message: "allocator refused publisher storage",
                            rollback,
                        }
                    }
                    PartitionAdmissionError::UnboundedHooks => {
                        OperatorError::ResidentContainerInvariant {
                            container: "partition hook admission",
                            message: "spill hooks do not declare a workspace bound",
                        }
                    }
                    PartitionAdmissionError::NonZeroPublisherGrant { .. }
                    | PartitionAdmissionError::PublisherInvariant => {
                        OperatorError::ResidentContainerInvariant {
                            container: "partition failure publisher",
                            message: "publisher admission invariant failed",
                        }
                    }
                }
            }
            PartitionOperationError::MemoryWithCleanup {
                error,
                cleanup,
                phase,
            } => OperatorError::ResidentMemory(error)
                .with_context(format!("{phase} also failed: {cleanup}")),
            PartitionOperationError::AdmissionWithCleanup {
                error,
                cleanup,
                phase,
            } => Self::map_partition_error(PartitionOperationError::Admission(error))
                .with_context(format!("{phase} also failed: {cleanup}")),
            PartitionOperationError::Cancelled(error) => OperatorError::QueryCancelled(error),
            PartitionOperationError::CancelledWithCleanup {
                error,
                cleanup,
                phase,
            } => OperatorError::QueryCancelled(error)
                .with_context(format!("{phase} also failed: {cleanup}")),
            PartitionOperationError::Allocation { container, error } => {
                OperatorError::ResidentContainerAllocation {
                    container,
                    source: error,
                }
            }
            PartitionOperationError::NativeMapAllocation { error } => {
                OperatorError::ResidentNativeMapAllocation { source: error }
            }
            PartitionOperationError::NativeMapAllocationWithRollback { error, rollback } => {
                OperatorError::ResidentNativeMapAllocationWithRollback {
                    source: error,
                    rollback,
                }
            }
            PartitionOperationError::NativeMapInvariant { message } => {
                OperatorError::ResidentContainerInvariant {
                    container: "native partition map",
                    message,
                }
            }
            PartitionOperationError::NativeMapInvariantWithRollback { message, rollback } => {
                OperatorError::ResidentContainerInvariantWithRollback {
                    container: "native partition map",
                    message,
                    rollback,
                }
            }
            PartitionOperationError::IoWithCleanup {
                error,
                cleanup,
                phase,
            } => OperatorError::from_spill_io_error(error)
                .with_context(format!("{phase} also failed: {cleanup}")),
            PartitionOperationError::Io(error) => OperatorError::from_spill_io_error(error),
        }
    }

    fn finalize_accounted(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        let consumer = sink.name();
        // Refuse unsupported transport before the consuming cursor is created.
        let mut permit = sink
            .__accounted_sink_permit()
            .ok_or(OperatorError::UnsupportedAccountedTransport { consumer })?;
        let mut output_root = self.resource_child_grant()?;
        let cancellation = self
            .memory_ctx
            .as_ref()
            .map(|context| context.cancellation_token().clone());
        let result = (|| {
            let partitioned = self.partitioned_groups.as_mut().ok_or(
                OperatorError::ResidentContainerInvariant {
                    container: "aggregate resource owner",
                    message: "grouped resource execution lost its partitions",
                },
            )?;
            let mut cursor = partitioned
                .drain_partitioned_accounted()
                .map_err(Self::map_partition_error)?;
            while let Some(entry) = cursor.next_entry().map_err(Self::map_partition_error)? {
                // The output grant stays outside the unwind boundary. In
                // addition to retained results, 256 bytes cover the standard
                // indexing panic's fixed text and two usize diagnostics.
                let admitted = (|| {
                    let mut grant =
                        output_root
                            .split(0)
                            .ok_or(OperatorError::ResidentContainerInvariant {
                                container: "aggregate output owner",
                                message: "zero-byte output grant could not be split",
                            })?;
                    let state = entry.value();
                    let columns = state
                        .key_values
                        .len()
                        .checked_add(state.accumulators.len())
                        .ok_or_else(aggregate_capacity_overflow)?;
                    let mut peak = columns
                        .checked_mul(
                            std::mem::size_of::<ValueVector>() + std::mem::size_of::<Value>(),
                        )
                        .and_then(|bytes| bytes.checked_add(256))
                        .ok_or_else(aggregate_capacity_overflow)?;
                    for value in &state.key_values {
                        let nested = value
                            .retained_size_bytes()
                            .and_then(|bytes| bytes.checked_sub(std::mem::size_of::<Value>()))
                            .ok_or_else(aggregate_capacity_overflow)?;
                        peak = peak
                            .checked_add(nested)
                            .ok_or_else(aggregate_capacity_overflow)?;
                    }
                    for accumulator in &state.accumulators {
                        peak = peak
                            .checked_add(
                                accumulator
                                    .finalize_peak_bytes()
                                    .ok_or_else(aggregate_capacity_overflow)?,
                            )
                            .ok_or_else(aggregate_capacity_overflow)?;
                    }
                    grant.try_resize(peak)?;
                    Ok::<_, OperatorError>((grant, columns))
                })();
                let (grant, columns) = match admitted {
                    Ok(admitted) => admitted,
                    Err(error) => {
                        return Err(Self::map_partition_error(cursor.fail_operator(error)));
                    }
                };
                let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let state = entry.value();
                    let mut values =
                        crate::execution::vector::try_exact_value_vector_catalog(columns)
                            .map_err(OperatorError::ResidentExactVectorAllocation)?;
                    for value in &state.key_values {
                        values.push(
                            ValueVector::try_exact_generic_one_with_sort_type(
                                value.clone(),
                                crate::execution::accounted_chunk::SortValueType::Ordinary,
                            )
                            .map_err(OperatorError::ResidentExactVectorAllocation)?,
                        );
                    }
                    for accumulator in &state.accumulators {
                        values.push(
                            ValueVector::try_exact_generic_one_with_sort_type(
                                accumulator.finalize(),
                                crate::execution::accounted_chunk::SortValueType::Ordinary,
                            )
                            .map_err(OperatorError::ResidentExactVectorAllocation)?,
                        );
                    }
                    Ok::<_, OperatorError>(DataChunk::new(values))
                }));
                let output = match built {
                    Ok(Ok(chunk)) => {
                        match crate::execution::accounted_chunk::try_accounted_distinct_output(
                            chunk, grant,
                        ) {
                            Ok(output) => output,
                            Err(error) => {
                                return Err(Self::map_partition_error(cursor.fail_operator(error)));
                            }
                        }
                    }
                    Ok(Err(error)) => {
                        return Err(Self::map_partition_error(cursor.fail_operator(error)));
                    }
                    Err(panic) => {
                        return Err(Self::map_partition_error(cursor.fail_panic(panic, grant)));
                    }
                };
                // The decoded entry and independent output grant overlap until
                // the receiver owns every shallow-cloned immutable payload.
                let keep_going = match permit.consume(output) {
                    Ok(keep_going) => keep_going,
                    Err(error) => {
                        return Err(Self::map_partition_error(cursor.fail_operator(error)));
                    }
                };
                if let Err(error) = poll_cancellation(cancellation.as_ref()) {
                    return Err(Self::map_partition_error(cursor.fail_operator(error)));
                }
                if !keep_going {
                    break;
                }
            }
            Ok(())
        })();
        self.refresh_estimated_usage();
        result
    }

    /// A row is the aggregate's smallest atomic update unit. If cancellation
    /// won during that unit, reconcile diagnostic usage before returning the
    /// terminal error rather than exposing stale consumer state.
    fn poll_after_group_mutation(
        &mut self,
        cancellation: Option<&QueryCancellationToken>,
    ) -> Result<(), OperatorError> {
        let result = poll_cancellation(cancellation);
        if result.is_err() {
            self.refresh_estimated_usage();
        }
        result
    }

    /// Checks whether spilling should occur and performs it if needed.
    fn maybe_spill(&mut self) -> Result<(), OperatorError> {
        let cancellation = self
            .memory_ctx
            .as_ref()
            .map(|context| context.cancellation_token().clone());
        poll_cancellation(cancellation.as_ref())?;
        if self.global_state.is_some() {
            // Global aggregation doesn't need spilling
            return Ok(());
        }

        if self.spill_state.is_some() {
            // Memory-aware mode
            self.maybe_spill_memory_aware()
        } else {
            // Non-context path. This can spill only when an explicit manager
            // was attached by `with_spilling`.
            self.maybe_spill_row_count()
        }
    }

    /// Memory-aware spill decision: check eviction flag and system pressure.
    fn maybe_spill_memory_aware(&mut self) -> Result<(), OperatorError> {
        let cancellation = self
            .memory_ctx
            .as_ref()
            .map(|context| context.cancellation_token().clone());
        let should_spill = if let Some(ref state) = self.spill_state {
            let eviction = state.take_eviction_request().is_some();
            let pressure = self.memory_ctx.as_ref().map_or(false, |c| c.should_spill());

            // Determine current group count for minimum buffer guard
            let group_count = if let Some(ref partitioned) = self.partitioned_groups {
                partitioned.total_size()
            } else {
                self.groups.len()
            };
            let above_minimum = group_count >= AGGREGATE_MIN_BUFFER_GROUPS;

            (eviction || pressure) && above_minimum
        } else {
            false
        };

        if should_spill {
            poll_cancellation(cancellation.as_ref())?;
            if let Some(ref mut partitioned) = self.partitioned_groups {
                partitioned
                    .spill_largest_controlled()
                    .map_err(Self::map_partition_error)?;
            }
            poll_cancellation(cancellation.as_ref())?;
        }

        Ok(())
    }

    /// Explicit row-count spill decision.
    fn maybe_spill_row_count(&mut self) -> Result<(), OperatorError> {
        // If using partitioned state, check if we need to spill
        if let Some(ref mut partitioned) = self.partitioned_groups {
            if partitioned.total_size() >= self.spill_threshold {
                partitioned
                    .spill_largest()
                    .map_err(OperatorError::from_spill_io_error)?;
            }
        } else if self.groups.len() >= self.spill_threshold {
            // Not using partitioned state yet, but reached threshold
            // If spilling is configured, switch to partitioned mode
            if let Some(ref manager) = self.spill_manager {
                let mut partitioned = PartitionedState::new_with_bounded_codec(
                    Arc::clone(manager),
                    256,
                    serialize_group_state_bounded,
                    deserialize_group_state_bounded,
                );

                // Move existing groups to partitioned state
                for (_key, state) in self.groups.drain() {
                    partitioned
                        .insert(state.key_values.clone(), state)
                        .map_err(OperatorError::from_spill_io_error)?;
                }

                self.partitioned_groups = Some(partitioned);
                self.using_partitioned = true;
            }
        }

        Ok(())
    }
}

#[cfg(feature = "spill")]
impl PushOperator for SpillableAggregatePushOperator {
    fn push(&mut self, chunk: DataChunk, _sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        self.ingest_chunk(&chunk)
    }

    fn push_accounted(
        &mut self,
        chunk: crate::execution::AccountedDataChunk,
        _sink: &mut dyn Sink,
    ) -> Result<bool, OperatorError> {
        if !self.owns_grouped_resources() {
            return Err(OperatorError::UnsupportedAccountedTransport {
                consumer: self.name(),
            });
        }
        self.ingest_chunk(chunk.chunk())
    }

    fn __accounted_push_permit(
        &mut self,
    ) -> Option<crate::execution::pipeline::AccountedPushPermit<'_>> {
        self.owns_grouped_resources()
            .then(|| crate::execution::pipeline::AccountedPushPermit::new(self))
    }

    fn admit_chunk_transport(
        &self,
        input: crate::execution::pipeline::ChunkTransport,
    ) -> Result<crate::execution::pipeline::ChunkTransport, OperatorError> {
        use crate::execution::pipeline::ChunkTransport;
        if self.owns_grouped_resources() {
            Ok(ChunkTransport::MayBeAccounted)
        } else if input == ChunkTransport::PlainOnly {
            Ok(input)
        } else {
            Err(OperatorError::UnsupportedAccountedTransport {
                consumer: self.name(),
            })
        }
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        let cancellation = self
            .memory_ctx
            .as_ref()
            .map(|context| context.cancellation_token().clone());
        poll_cancellation(cancellation.as_ref())?;
        if self.owns_grouped_resources() {
            return self.finalize_accounted(sink);
        }
        let num_output_cols = self.group_by.len() + self.aggregates.len();
        let mut columns: Vec<ValueVector> =
            (0..num_output_cols).map(|_| ValueVector::new()).collect();

        if self.group_by.is_empty() {
            // Global aggregation - single row output
            poll_cancellation(cancellation.as_ref())?;
            if let Some(ref accumulators) = self.global_state {
                for (i, acc) in accumulators.iter().enumerate() {
                    columns[i].push(acc.finalize());
                }
            }
            poll_cancellation(cancellation.as_ref())?;
        } else if self.using_partitioned {
            // This is a consuming terminal path: once drain_all starts, this
            // operator cannot retry finalization. A later cursor-based drain
            // must add rollback if reusable post-cancellation state is needed.
            if self.partitioned_groups.is_some() {
                poll_cancellation(cancellation.as_ref())?;
                let groups = {
                    let partitioned = self
                        .partitioned_groups
                        .as_mut()
                        .expect("partition presence checked above");
                    if cancellation.is_some() {
                        partitioned
                            .drain_all_controlled()
                            .map_err(Self::map_partition_error)
                    } else {
                        partitioned
                            .drain_all()
                            .map_err(OperatorError::from_spill_io_error)
                    }
                };
                // Draining is consuming. Reconcile telemetry whether it
                // completed or terminal cancellation explicitly cleaned the
                // owned state.
                self.refresh_estimated_usage();
                let groups = groups?;
                poll_cancellation(cancellation.as_ref())?;

                for (_key, state) in groups {
                    poll_cancellation(cancellation.as_ref())?;
                    // Output group key columns
                    for (i, val) in state.key_values.iter().enumerate() {
                        columns[i].push(val.clone());
                    }

                    // Output aggregate results
                    for (i, acc) in state.accumulators.iter().enumerate() {
                        columns[self.group_by.len() + i].push(acc.finalize());
                    }
                    poll_cancellation(cancellation.as_ref())?;
                }
            }
        } else {
            // Group by using regular hash map - one row per group
            for state in self.groups.values() {
                poll_cancellation(cancellation.as_ref())?;
                // Output group key columns
                for (i, val) in state.key_values.iter().enumerate() {
                    columns[i].push(val.clone());
                }

                // Output aggregate results
                for (i, acc) in state.accumulators.iter().enumerate() {
                    columns[self.group_by.len() + i].push(acc.finalize());
                }
                poll_cancellation(cancellation.as_ref())?;
            }
        }

        if !columns.is_empty() && !columns[0].is_empty() {
            poll_cancellation(cancellation.as_ref())?;
            let chunk = DataChunk::new(columns);
            sink.consume(chunk)?;
        }

        Ok(())
    }

    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        ChunkSizeHint::Default
    }

    fn name(&self) -> &'static str {
        "SpillableAggregatePush"
    }
}

#[cfg(feature = "spill")]
impl crate::execution::pipeline::qualified_accounted_transport::QualifiedPushOperator
    for SpillableAggregatePushOperator
{
    fn push_accounted_qualified(
        &mut self,
        chunk: crate::execution::AccountedDataChunk,
        _sink: &mut crate::execution::pipeline::AccountedSinkPermit<'_>,
    ) -> Result<bool, OperatorError> {
        self.ingest_chunk(chunk.chunk())
    }
}

#[cfg(all(test, feature = "spill"))]
#[path = "aggregate/owned_failure_tests.rs"]
mod owned_failure_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::operators::accumulator::AggregateFunction;
    use crate::execution::sink::CollectorSink;
    use std::sync::Arc;
    #[cfg(feature = "spill")]
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[cfg(feature = "spill")]
    struct CancelAtSync {
        cancellation: crate::execution::QueryCancellationHandle,
        failure: Option<&'static str>,
    }

    #[cfg(feature = "spill")]
    impl crate::execution::spill::SpillIo for CancelAtSync {
        fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
            // Cancellation is allocation-free; the optional fixed test error
            // and its std::io::Error wrapper fit within this declared bound.
            Some(256)
        }

        fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
            // Reader hooks only inspect atomics or request cancellation.
            Some(0)
        }

        fn check(
            &self,
            operation: crate::execution::spill::SpillIoOperation,
        ) -> std::io::Result<()> {
            if operation != crate::execution::spill::SpillIoOperation::Sync {
                return Ok(());
            }
            self.cancellation.cancel();
            match self.failure {
                Some(message) => Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    message,
                )),
                None => Ok(()),
            }
        }
    }

    #[cfg(feature = "spill")]
    struct ArmableCancelNthIo {
        target: crate::execution::spill::SpillIoOperation,
        trigger: usize,
        armed: AtomicBool,
        matching: AtomicUsize,
        cancellation: crate::execution::QueryCancellationHandle,
    }

    #[cfg(feature = "spill")]
    impl ArmableCancelNthIo {
        fn new(
            target: crate::execution::spill::SpillIoOperation,
            trigger: usize,
            cancellation: crate::execution::QueryCancellationHandle,
        ) -> Self {
            Self {
                target,
                trigger,
                armed: AtomicBool::new(false),
                matching: AtomicUsize::new(0),
                cancellation,
            }
        }

        fn arm(&self) {
            self.matching.store(0, Ordering::Relaxed);
            self.armed.store(true, Ordering::Release);
        }
    }

    #[cfg(feature = "spill")]
    impl crate::execution::spill::SpillIo for ArmableCancelNthIo {
        fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
            // Cancellation is allocation-free; the optional fixed test error
            // and its std::io::Error wrapper fit within this declared bound.
            Some(256)
        }

        fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
            // Reader hooks only inspect atomics or request cancellation.
            Some(0)
        }

        fn check(
            &self,
            operation: crate::execution::spill::SpillIoOperation,
        ) -> std::io::Result<()> {
            if self.armed.load(Ordering::Acquire) && operation == self.target {
                let matching = self.matching.fetch_add(1, Ordering::Relaxed) + 1;
                if matching == self.trigger {
                    self.cancellation.cancel();
                }
            }
            Ok(())
        }
    }

    #[cfg(feature = "spill")]
    fn cancellation_spill_fixture(
        root: &std::path::Path,
        cancellation: crate::execution::QueryCancellationHandle,
        failure: Option<&'static str>,
    ) -> crate::execution::spill::BorrowedSpillFixture {
        crate::execution::spill::BorrowedSpillFixture::new(root)
            .provider(
                Arc::new(crate::execution::spill::CleartextSpillRecordProvider),
                crate::execution::spill::SpillFrameLimits::format_max(),
            )
            .io(Arc::new(CancelAtSync {
                cancellation,
                failure,
            }))
    }

    #[cfg(feature = "spill")]
    fn minimum_spill_group_input() -> (Vec<i64>, Vec<i64>) {
        let upper = i64::try_from(AGGREGATE_MIN_BUFFER_GROUPS).unwrap();
        let keys: Vec<_> = (0..upper).collect();
        let values = vec![1; keys.len()];
        (keys, values)
    }

    fn create_test_chunk(values: &[i64]) -> DataChunk {
        let v: Vec<Value> = values.iter().map(|&i| Value::Int64(i)).collect();
        let vector = ValueVector::from_values(&v);
        DataChunk::new(vec![vector])
    }

    fn create_two_column_chunk(col1: &[i64], col2: &[i64]) -> DataChunk {
        let v1: Vec<Value> = col1.iter().map(|&i| Value::Int64(i)).collect();
        let v2: Vec<Value> = col2.iter().map(|&i| Value::Int64(i)).collect();
        DataChunk::new(vec![
            ValueVector::from_values(&v1),
            ValueVector::from_values(&v2),
        ])
    }

    fn create_distinct_value_and_key_chunk() -> DataChunk {
        DataChunk::new(vec![
            ValueVector::from_values(&[
                Value::Int64(1),
                Value::Int64(1),
                Value::Int64(3),
                Value::Int64(9),
            ]),
            ValueVector::from_values(&[
                Value::String("term-1".into()),
                Value::String("term-2".into()),
                Value::String("term-3".into()),
                Value::String("term-1".into()),
            ]),
        ])
    }

    #[test]
    fn resource_aggregate_cancelled_push_precedes_group_growth() {
        let buffer_manager = grafeo_common::memory::buffer::BufferManager::with_budget(1024 * 1024);
        let control = crate::execution::QueryExecutionControl::new();
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            Arc::clone(&buffer_manager),
            control.token(),
        )
        .unwrap();
        let baseline_bytes = buffer_manager.allocated();
        let mut aggregate = AggregatePushOperator::with_resource_context(
            vec![0],
            vec![AggregateExpr::sum(1)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();

        control.cancellation_handle().cancel();
        let error = aggregate
            .push(create_two_column_chunk(&[1], &[2]), &mut sink)
            .unwrap_err();

        assert!(matches!(
            error,
            OperatorError::QueryCancelled(crate::execution::QueryCancellationError::Cancelled)
        ));
        assert!(aggregate.groups.is_empty());
        assert_eq!(buffer_manager.allocated(), baseline_bytes);
        assert!(sink.into_chunks().is_empty());
    }

    #[test]
    fn resource_aggregate_cancelled_finalize_preserves_groups_and_publishes_no_output() {
        let buffer_manager = grafeo_common::memory::buffer::BufferManager::with_budget(1024 * 1024);
        let control = crate::execution::QueryExecutionControl::new();
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            buffer_manager,
            control.token(),
        )
        .unwrap();
        let mut aggregate = AggregatePushOperator::with_resource_context(
            vec![0],
            vec![AggregateExpr::sum(1)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();
        aggregate
            .push(create_two_column_chunk(&[1], &[2]), &mut sink)
            .unwrap();
        let original_keys: Vec<_> = aggregate.groups.keys().cloned().collect();

        control.cancellation_handle().cancel();
        let error = aggregate.finalize(&mut sink).unwrap_err();

        assert!(matches!(
            error,
            OperatorError::QueryCancelled(crate::execution::QueryCancellationError::Cancelled)
        ));
        assert_eq!(aggregate.groups.len(), 1);
        assert_eq!(
            aggregate.groups.keys().cloned().collect::<Vec<_>>(),
            original_keys
        );
        assert!(sink.into_chunks().is_empty());
    }

    #[test]
    #[cfg(feature = "spill")]
    fn public_resource_aggregate_preserves_undeclared_hook_compatibility() {
        #[derive(Default)]
        struct UndeclaredHooks {
            creates: AtomicUsize,
        }
        impl crate::execution::spill::SpillIo for UndeclaredHooks {
            fn check(
                &self,
                operation: crate::execution::spill::SpillIoOperation,
            ) -> std::io::Result<()> {
                if operation == crate::execution::spill::SpillIoOperation::Create {
                    self.creates.fetch_add(1, Ordering::Relaxed);
                }
                Ok(())
            }
        }

        let directory = tempfile::TempDir::new().unwrap();
        let buffer_manager = grafeo_common::memory::buffer::BufferManager::with_budget(1 << 20);
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(UndeclaredHooks::default());
        let (resources, manager) =
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .io(Arc::clone(&io) as Arc<dyn crate::execution::spill::SpillIo>)
                .build_operator_resources(Arc::clone(&buffer_manager), control.token())
                .unwrap();
        let baseline_bytes = resources.query_stats().allocated_bytes;
        let baseline_consumers = buffer_manager.stats().consumer_count;
        let mut aggregate = SpillableAggregatePushOperator::with_resource_context(
            vec![0],
            vec![AggregateExpr::sum(1)],
            resources.clone(),
        )
        .unwrap();
        assert!(aggregate.metadata_grant.is_none());
        assert!(!aggregate.owns_grouped_resources());
        assert!(aggregate.__accounted_push_permit().is_none());
        assert_eq!(resources.query_stats().allocated_bytes, baseline_bytes);
        assert_eq!(
            buffer_manager.stats().consumer_count,
            baseline_consumers + 1
        );

        let (keys, values) = minimum_spill_group_input();
        let mut sink = CollectorSink::new();
        aggregate.spill_state.as_ref().unwrap().request_eviction(1);
        aggregate
            .push(create_two_column_chunk(&keys, &values), &mut sink)
            .unwrap();
        assert_eq!(io.creates.load(Ordering::Relaxed), 1);
        assert_eq!(manager.active_file_count(), 1);
        aggregate.finalize(&mut sink).unwrap();
        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        let mut rows = Vec::new();
        for row in 0..chunks[0].len() {
            // Compatibility output has generic vector backing; the stored
            // values still retain their exact integer types.
            let Some(Value::Int64(key)) = chunks[0].column(0).unwrap().get_value(row) else {
                panic!("compatibility group key must remain an integer");
            };
            let Some(Value::Int64(sum)) = chunks[0].column(1).unwrap().get_value(row) else {
                panic!("compatibility SUM must remain an integer");
            };
            rows.push((key, sum));
        }
        rows.sort_unstable();
        assert_eq!(
            rows,
            keys.into_iter().map(|key| (key, 1)).collect::<Vec<_>>()
        );
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(resources.query_stats().allocated_bytes, baseline_bytes);
        drop(aggregate);
        assert_eq!(buffer_manager.stats().consumer_count, baseline_consumers);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn resource_spill_aggregate_cancelled_push_precedes_group_and_disk_growth() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let buffer_manager = grafeo_common::memory::buffer::BufferManager::with_budget(1024 * 1024);
        let spill_manager_fixture =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path());
        let control = crate::execution::QueryExecutionControl::new();
        let (resources, spill_manager) = spill_manager_fixture
            .build_operator_resources(Arc::clone(&buffer_manager), control.token())
            .unwrap();
        let baseline_consumers = buffer_manager.stats().consumer_count;
        let mut aggregate = SpillableAggregatePushOperator::with_resource_context(
            vec![0],
            vec![AggregateExpr::sum(1)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();
        let baseline_estimated_bytes = aggregate.estimated_bytes;
        let baseline_usage = aggregate.spill_state.as_ref().unwrap().usage();
        assert_eq!(
            buffer_manager.stats().consumer_count,
            baseline_consumers + 1
        );

        control.cancellation_handle().cancel();
        let error = aggregate
            .push(create_two_column_chunk(&[1], &[2]), &mut sink)
            .unwrap_err();

        assert!(matches!(
            error,
            OperatorError::QueryCancelled(crate::execution::QueryCancellationError::Cancelled)
        ));
        assert_eq!(
            aggregate.partitioned_groups.as_ref().unwrap().total_size(),
            0
        );
        assert_eq!(aggregate.estimated_bytes, baseline_estimated_bytes);
        assert_eq!(
            aggregate.spill_state.as_ref().unwrap().usage(),
            baseline_usage
        );
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(spill_manager.disk_stats().reserved_live_bytes, 0);
        assert!(sink.into_chunks().is_empty());
        assert_eq!(
            buffer_manager.stats().consumer_count,
            baseline_consumers + 1
        );

        drop(aggregate);
        assert_eq!(buffer_manager.stats().consumer_count, baseline_consumers);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn partition_cleanup_context_preserves_typed_aggregate_cancellation_source() {
        let error = SpillableAggregatePushOperator::map_partition_error(
            PartitionOperationError::CancelledWithCleanup {
                error: crate::execution::QueryCancellationError::Cancelled,
                cleanup: std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    "deterministic partition cleanup failure",
                ),
                phase: "cancelled partition drain cleanup",
            },
        );

        let OperatorError::Context { source, context } = error else {
            panic!("partition cleanup flattened aggregate cancellation")
        };
        assert!(matches!(
            *source,
            OperatorError::QueryCancelled(crate::execution::QueryCancellationError::Cancelled)
        ));
        assert!(context.contains("cancelled partition drain cleanup"));
        assert!(context.contains("deterministic partition cleanup failure"));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn partition_grant_denial_remains_typed_at_aggregate_boundary() {
        let memory_error = grafeo_common::memory::buffer::MemoryGrantError::LimitExceeded {
            scope: grafeo_common::memory::buffer::MemoryLimitScope::Query,
            requested_bytes: 65,
            limit_bytes: 64,
        };
        let error =
            SpillableAggregatePushOperator::map_partition_error(PartitionOperationError::Io(
                std::io::Error::new(std::io::ErrorKind::OutOfMemory, memory_error.clone()),
            ));

        assert!(matches!(
            error,
            OperatorError::ResidentMemory(ref retained) if retained == &memory_error
        ));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn resource_spill_aggregate_cancelled_finalize_precedes_destructive_drain() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let buffer_manager = grafeo_common::memory::buffer::BufferManager::with_budget(1024 * 1024);
        let spill_manager_fixture =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path());
        let control = crate::execution::QueryExecutionControl::new();
        let (resources, spill_manager) = spill_manager_fixture
            .build_operator_resources(buffer_manager, control.token())
            .unwrap();
        let mut aggregate = SpillableAggregatePushOperator::with_resource_context(
            vec![0],
            vec![AggregateExpr::sum(1)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();
        let (keys, values) = minimum_spill_group_input();
        aggregate.spill_state.as_ref().unwrap().request_eviction(1);
        aggregate
            .push(create_two_column_chunk(&keys, &values), &mut sink)
            .unwrap();
        assert_eq!(
            aggregate.partitioned_groups.as_ref().unwrap().total_size(),
            keys.len()
        );
        assert_eq!(
            aggregate
                .partitioned_groups
                .as_ref()
                .unwrap()
                .spilled_count(),
            1
        );
        assert_eq!(spill_manager.active_file_count(), 1);
        let before_spilled_bytes = spill_manager.spilled_bytes();
        let before_reserved_bytes = spill_manager.disk_stats().reserved_live_bytes;

        control.cancellation_handle().cancel();
        let error = aggregate.finalize(&mut sink).unwrap_err();

        assert!(matches!(
            error,
            OperatorError::QueryCancelled(crate::execution::QueryCancellationError::Cancelled)
        ));
        assert_eq!(
            aggregate.partitioned_groups.as_ref().unwrap().total_size(),
            keys.len()
        );
        assert_eq!(
            aggregate
                .partitioned_groups
                .as_ref()
                .unwrap()
                .spilled_count(),
            1
        );
        assert_eq!(spill_manager.active_file_count(), 1);
        assert_eq!(spill_manager.spilled_bytes(), before_spilled_bytes);
        assert_eq!(
            spill_manager.disk_stats().reserved_live_bytes,
            before_reserved_bytes
        );
        assert!(sink.into_chunks().is_empty());

        drop(aggregate);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(spill_manager.disk_stats().reserved_live_bytes, 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn aggregate_drain_cancellation_maps_typed_reason_and_reconciles_cleanup() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let buffer_manager = grafeo_common::memory::buffer::BufferManager::with_budget(1024 * 1024);
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(ArmableCancelNthIo::new(
            crate::execution::spill::SpillIoOperation::ReadPayload,
            2,
            control.cancellation_handle(),
        ));
        let spill_manager_fixture =
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .provider(
                    Arc::new(crate::execution::spill::CleartextSpillRecordProvider),
                    crate::execution::spill::SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn crate::execution::spill::SpillIo>);
        let (resources, spill_manager) = spill_manager_fixture
            .build_operator_resources(buffer_manager, control.token())
            .unwrap();
        let mut aggregate = SpillableAggregatePushOperator::with_qualified_resource_context(
            vec![0],
            vec![AggregateExpr::sum(1)],
            resources,
        )
        .unwrap();
        let mut sink = crate::execution::sink::CountingSink::new();
        let (keys, values) = minimum_spill_group_input();
        aggregate.spill_state.as_ref().unwrap().request_eviction(1);
        aggregate
            .push(create_two_column_chunk(&keys, &values), &mut sink)
            .unwrap();
        assert_eq!(
            aggregate
                .partitioned_groups
                .as_ref()
                .unwrap()
                .spilled_count(),
            1
        );
        // Make the first output attempt read a file, independent of which
        // hash partition won the eviction request. No decoded row may escape
        // the cancellation injected into that first record's read.
        for partition in 0..256 {
            aggregate
                .partitioned_groups
                .as_mut()
                .unwrap()
                .spill_partition_controlled(partition)
                .unwrap();
        }
        io.arm();

        let error = aggregate.finalize(&mut sink).unwrap_err();

        assert!(matches!(
            error,
            OperatorError::ClassifiedAccountedFailure {
                classification:
                    crate::execution::operators::AccountedFailureClassification::QueryCancelled(
                        crate::execution::QueryCancellationError::Cancelled
                    ),
                ..
            }
        ));
        assert_eq!(io.matching.load(Ordering::Relaxed), 2);
        assert_eq!(
            aggregate.partitioned_groups.as_ref().unwrap().total_size(),
            0
        );
        assert_eq!(
            aggregate
                .partitioned_groups
                .as_ref()
                .unwrap()
                .spilled_count(),
            0
        );
        let retained = aggregate
            .partitioned_groups
            .as_ref()
            .unwrap()
            .granted_bytes()
            + aggregate.metadata_grant.as_ref().unwrap().size();
        assert_eq!(aggregate.estimated_bytes, retained);
        assert_eq!(aggregate.spill_state.as_ref().unwrap().usage(), retained);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(spill_manager.disk_stats().reserved_live_bytes, 0);
        assert_eq!(sink.count(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn cancellation_during_successful_aggregate_spill_cleans_staging() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let buffer_manager = grafeo_common::memory::buffer::BufferManager::with_budget(1024 * 1024);
        let control = crate::execution::QueryExecutionControl::new();
        let spill_fixture =
            cancellation_spill_fixture(temp_dir.path(), control.cancellation_handle(), None);
        let (resources, spill_manager) = spill_fixture
            .build_operator_resources(Arc::clone(&buffer_manager), control.token())
            .unwrap();
        let baseline_consumers = buffer_manager.stats().consumer_count;
        let mut aggregate = SpillableAggregatePushOperator::with_qualified_resource_context(
            vec![0],
            vec![AggregateExpr::sum(1)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();
        let (keys, values) = minimum_spill_group_input();
        aggregate.spill_state.as_ref().unwrap().request_eviction(1);

        let error = aggregate
            .push(create_two_column_chunk(&keys, &values), &mut sink)
            .unwrap_err();

        assert!(matches!(
            error,
            OperatorError::ClassifiedAccountedFailure {
                classification:
                    crate::execution::operators::AccountedFailureClassification::QueryCancelled(
                        crate::execution::QueryCancellationError::Cancelled
                    ),
                ..
            }
        ));
        assert!(control.token().is_cancelled());
        let partitions = aggregate.partitioned_groups.as_ref().unwrap();
        assert_eq!(partitions.total_size(), keys.len());
        assert_eq!(partitions.spilled_count(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(spill_manager.disk_stats().reserved_live_bytes, 0);
        assert_eq!(
            buffer_manager.stats().consumer_count,
            baseline_consumers + 1
        );

        drop(aggregate);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(spill_manager.disk_stats().reserved_live_bytes, 0);
        assert_eq!(buffer_manager.stats().consumer_count, baseline_consumers);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn aggregate_spill_error_beats_racing_cancellation_and_cleans_staging() {
        const FAILURE: &str = "deterministic aggregate spill failure";

        let temp_dir = tempfile::TempDir::new().unwrap();
        let buffer_manager = grafeo_common::memory::buffer::BufferManager::with_budget(1024 * 1024);
        let control = crate::execution::QueryExecutionControl::new();
        let spill_fixture = cancellation_spill_fixture(
            temp_dir.path(),
            control.cancellation_handle(),
            Some(FAILURE),
        );
        let (resources, spill_manager) = spill_fixture
            .build_operator_resources(Arc::clone(&buffer_manager), control.token())
            .unwrap();
        let baseline_consumers = buffer_manager.stats().consumer_count;
        let mut aggregate = SpillableAggregatePushOperator::with_qualified_resource_context(
            vec![0],
            vec![AggregateExpr::sum(1)],
            resources,
        )
        .unwrap();
        let mut sink = CollectorSink::new();
        let (keys, values) = minimum_spill_group_input();
        aggregate.spill_state.as_ref().unwrap().request_eviction(1);

        let error = aggregate
            .push(create_two_column_chunk(&keys, &values), &mut sink)
            .unwrap_err();

        assert!(matches!(
            error,
            OperatorError::ClassifiedAccountedFailure {
                classification:
                    crate::execution::operators::AccountedFailureClassification::Execution,
                ..
            }
        ));
        assert_eq!(
            PartitionedState::<GroupState>::inspect_failure(&error, |primary, _, _, _, _| {
                matches!(primary, Some(PartitionOperationError::Io(error))
                if error.kind() == std::io::ErrorKind::PermissionDenied && error.to_string() == FAILURE)
            }),
            Some(true)
        );
        assert!(control.token().is_cancelled());
        let partitions = aggregate.partitioned_groups.as_ref().unwrap();
        assert_eq!(partitions.total_size(), keys.len());
        assert_eq!(partitions.spilled_count(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(spill_manager.spilled_bytes(), 0);
        assert_eq!(spill_manager.disk_stats().reserved_live_bytes, 0);
        assert_eq!(
            buffer_manager.stats().consumer_count,
            baseline_consumers + 1
        );

        drop(aggregate);
        assert_eq!(buffer_manager.stats().consumer_count, baseline_consumers);
    }

    #[test]
    fn push_distinct_uses_the_explicit_key_column() {
        let expression = AggregateExpr::sum(0)
            .with_distinct()
            .with_distinct_key_column(1);
        let mut aggregate = AggregatePushOperator::global(vec![expression]);
        let mut sink = CollectorSink::new();

        aggregate
            .push(create_distinct_value_and_key_chunk(), &mut sink)
            .unwrap();
        aggregate.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::Int64(5))
        );
    }

    #[test]
    fn push_distinct_identity_completion_keeps_first_key_operands() {
        let mut aggregate = AggregatePushOperator::global(identity_completion_expressions());
        let mut sink = CollectorSink::new();
        aggregate
            .push(identity_completion_chunk(0..3), &mut sink)
            .unwrap();
        aggregate.finalize(&mut sink).unwrap();
        let chunks = sink.into_chunks();
        let actual: Vec<_> = (0..5)
            .map(|column| chunks[0].column(column).unwrap().get_value(0).unwrap())
            .collect();
        assert_eq!(actual, identity_completion_expected());
    }

    #[test]
    fn push_identity_completion_preserves_non_distinct_results() {
        let expressions = identity_completion_expressions()
            .into_iter()
            .map(|mut expression| {
                expression.distinct = false;
                expression
            })
            .collect();
        let mut aggregate = AggregatePushOperator::global(expressions);
        let mut sink = CollectorSink::new();
        aggregate
            .push(identity_completion_chunk(0..3), &mut sink)
            .unwrap();
        aggregate.finalize(&mut sink).unwrap();
        let chunks = sink.into_chunks();
        let actual: Vec<_> = (0..5)
            .map(|column| chunks[0].column(column).unwrap().get_value(0).unwrap())
            .collect();
        assert_eq!(
            actual,
            vec![
                Value::Int64(1),
                Value::Int64(100),
                Value::Int64(2),
                Value::Int64(30),
                Value::Float64(15.0)
            ]
        );
    }

    fn identity_completion_expressions() -> Vec<AggregateExpr> {
        [
            AggregateFunction::Min,
            AggregateFunction::Max,
            AggregateFunction::Last,
            AggregateFunction::Sum,
            AggregateFunction::Avg,
        ]
        .into_iter()
        .enumerate()
        .map(|(column, function)| {
            let mut expression = AggregateExpr::min(column)
                .with_distinct()
                .with_distinct_key_column(5);
            expression.function = function;
            expression
        })
        .collect()
    }

    fn identity_completion_chunk(rows: std::ops::Range<usize>) -> DataChunk {
        let columns = [
            [Value::Int64(100), Value::Int64(50), Value::Int64(1)],
            [Value::Int64(1), Value::Int64(50), Value::Int64(100)],
            [Value::Int64(1), Value::Int64(3), Value::Int64(2)],
            [
                Value::String("bad".into()),
                Value::Int64(20),
                Value::Int64(10),
            ],
            [
                Value::String("bad".into()),
                Value::Int64(20),
                Value::Int64(10),
            ],
            [Value::Int64(1), Value::Int64(2), Value::Int64(1)],
            [Value::Int64(0), Value::Int64(0), Value::Int64(0)],
        ];
        DataChunk::new(
            columns
                .iter()
                .map(|column| ValueVector::from_values(&column[rows.clone()]))
                .collect(),
        )
    }

    fn identity_completion_expected() -> Vec<Value> {
        vec![
            Value::Int64(50),
            Value::Int64(50),
            Value::Int64(3),
            Value::Int64(30),
            Value::Float64(15.0),
        ]
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_distinct_identity_completion_codec_continues_after_reload() {
        let expressions = identity_completion_expressions();
        let mut state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![],
            accumulators: expressions
                .iter()
                .map(|expression| AggregateState::new(expression.function, true, None, None))
                .collect(),
        };
        let prefix = identity_completion_chunk(0..2);
        for row in 0..2 {
            for (accumulator, expression) in state.accumulators.iter_mut().zip(&expressions) {
                update_accumulator(accumulator, expression, &prefix, row).unwrap();
            }
        }
        let mut bytes = Vec::new();
        serialize_group_state(&state, &mut bytes).unwrap();
        let mut restored = deserialize_group_state(&mut bytes.as_slice()).unwrap();
        let suffix = identity_completion_chunk(2..3);
        for (accumulator, expression) in restored.accumulators.iter_mut().zip(&expressions) {
            update_accumulator(accumulator, expression, &suffix, 0).unwrap();
        }
        assert_eq!(
            restored
                .accumulators
                .iter()
                .map(AggregateState::finalize)
                .collect::<Vec<_>>(),
            identity_completion_expected()
        );
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_distinct_identity_last_null_key_survives_reload() {
        let mut state = AggregateState::new(AggregateFunction::Last, true, None, None);
        state.update_with_distinct_key(Some(Value::Int64(1)), Some(Value::Int64(1)));
        state.update_with_distinct_key(Some(Value::Null), Some(Value::Int64(2)));
        let group = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![],
            accumulators: vec![state],
        };
        let mut bytes = Vec::new();
        serialize_group_state(&group, &mut bytes).unwrap();
        let mut restored = deserialize_group_state(&mut bytes.as_slice()).unwrap();
        let state = &mut restored.accumulators[0];
        state.update_with_distinct_key(Some(Value::Int64(20)), Some(Value::Int64(2)));
        assert_eq!(state.finalize(), Value::Null);
        state.update_with_distinct_key(None, Some(Value::Int64(3)));
        state.update_with_distinct_key(Some(Value::Int64(30)), Some(Value::Int64(3)));
        state.update_with_distinct_key(Some(Value::Int64(99)), Some(Value::Int64(1)));
        assert_eq!(state.finalize(), Value::Int64(30));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_distinct_identity_completion_partition_continues_after_reload() {
        use crate::execution::spill::{
            CleartextSpillRecordProvider, CodecLimits, SpillFrameLimits,
        };
        use tempfile::TempDir;
        let directory = TempDir::new().unwrap();
        let limits = SpillFrameLimits::new(4096, 4096)
            .unwrap()
            .with_codec_limits(CodecLimits::new(64 * 1024, 1024, 32, 32));
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(Arc::new(CleartextSpillRecordProvider), limits)
                .build()
                .unwrap(),
        );
        let mut aggregate = SpillableAggregatePushOperator::with_spilling(
            vec![6],
            identity_completion_expressions(),
            Arc::clone(&manager),
            1,
        );
        let mut sink = CollectorSink::new();
        aggregate
            .push(identity_completion_chunk(0..2), &mut sink)
            .unwrap();
        let partitioned = aggregate.partitioned_groups.as_mut().unwrap();
        let partition = partitioned.partition_for(&[Value::Int64(0)]);
        partitioned.spill_partition(partition).unwrap();
        assert!(
            manager.active_file_count() > 0,
            "must evict a real partition"
        );
        aggregate
            .push(identity_completion_chunk(2..3), &mut sink)
            .unwrap();
        aggregate.finalize(&mut sink).unwrap();
        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        let actual: Vec<_> = (1..6)
            .map(|column| chunks[0].column(column).unwrap().get_value(0).unwrap())
            .collect();
        assert_eq!(actual, identity_completion_expected());
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    fn push_distinct_statistics_and_bivariate_use_explicit_identity() {
        let expressions = vec![
            AggregateExpr::stdev_pop(0)
                .with_distinct()
                .with_distinct_key_column(2),
            AggregateExpr::percentile_cont(0, 0.5)
                .with_distinct()
                .with_distinct_key_column(2),
            AggregateExpr {
                function: AggregateFunction::RegrCount,
                column: Some(0),
                column2: Some(1),
                distinct_key_column: Some(2),
                distinct: true,
                alias: None,
                percentile: None,
                separator: None,
            },
        ];
        // Identities 1,2,3 select operands 1,1,3; identity 1's later changed
        // operand 9 must not contribute. Value-only dedup would lose identity 2.
        let columns = [[1, 1, 3, 9], [10, 10, 30, 90], [1, 2, 3, 1]]
            .into_iter()
            .map(|values| ValueVector::from_values(&values.map(Value::Int64)))
            .collect();
        let mut aggregate = AggregatePushOperator::global(expressions);
        let mut sink = CollectorSink::new();
        aggregate.push(DataChunk::new(columns), &mut sink).unwrap();
        aggregate.finalize(&mut sink).unwrap();
        let chunks = sink.into_chunks();
        let actual: Vec<_> = (0..3)
            .map(|column| chunks[0].column(column).unwrap().get_value(0).unwrap())
            .collect();
        assert!(
            matches!(actual[0], Value::Float64(value) if (value - (8.0_f64 / 9.0).sqrt()).abs() < 1e-12)
                && actual[1] == Value::Float64(1.0)
                && actual[2] == Value::Int64(3),
            "push DISTINCT must retain identity-qualified statistics and pair count: {actual:?}"
        );
    }

    #[test]
    fn test_global_count() {
        let mut agg = AggregatePushOperator::global(vec![AggregateExpr::count_star()]);
        let mut sink = CollectorSink::new();

        agg.push(create_test_chunk(&[1, 2, 3, 4, 5]), &mut sink)
            .unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::Int64(5))
        );
    }

    #[test]
    fn test_global_sum() {
        let mut agg = AggregatePushOperator::global(vec![AggregateExpr::sum(0)]);
        let mut sink = CollectorSink::new();

        agg.push(create_test_chunk(&[1, 2, 3, 4, 5]), &mut sink)
            .unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        // AggregateState preserves integer type for SUM of integers
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::Int64(15))
        );
    }

    #[test]
    fn test_global_min_max() {
        let mut agg =
            AggregatePushOperator::global(vec![AggregateExpr::min(0), AggregateExpr::max(0)]);
        let mut sink = CollectorSink::new();

        agg.push(create_test_chunk(&[3, 1, 4, 1, 5, 9, 2, 6]), &mut sink)
            .unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::Int64(1))
        );
        assert_eq!(
            chunks[0].column(1).unwrap().get_value(0),
            Some(Value::Int64(9))
        );
    }

    #[test]
    fn test_group_by_sum() {
        // Group by column 0, sum column 1
        let mut agg = AggregatePushOperator::new(vec![0], vec![AggregateExpr::sum(1)]);
        let mut sink = CollectorSink::new();

        // Group 1: 10, 20 (sum=30), Group 2: 30, 40 (sum=70)
        agg.push(
            create_two_column_chunk(&[1, 1, 2, 2], &[10, 20, 30, 40]),
            &mut sink,
        )
        .unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks[0].len(), 2); // 2 groups
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_aggregate_no_spill() {
        // When threshold is not reached, should work like normal aggregate
        let mut agg = SpillableAggregatePushOperator::new(vec![0], vec![AggregateExpr::sum(1)])
            .with_threshold(100);
        let mut sink = CollectorSink::new();

        agg.push(
            create_two_column_chunk(&[1, 1, 2, 2], &[10, 20, 30, 40]),
            &mut sink,
        )
        .unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks[0].len(), 2); // 2 groups
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_aggregate_with_spilling() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build()
                .unwrap(),
        );

        // Set very low threshold to force spilling
        let mut agg = SpillableAggregatePushOperator::with_spilling(
            vec![0],
            vec![AggregateExpr::sum(1)],
            manager,
            3, // Spill after 3 groups
        );
        let mut sink = CollectorSink::new();

        // Create 10 different groups
        for i in 0..10 {
            let chunk = create_two_column_chunk(&[i], &[i * 10]);
            agg.push(chunk, &mut sink).unwrap();
        }
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 10); // 10 groups

        // Verify sums are correct (AggregateState preserves Int64 for integer sums)
        let mut sums: Vec<i64> = Vec::new();
        for i in 0..chunks[0].len() {
            if let Some(Value::Int64(sum)) = chunks[0].column(1).unwrap().get_value(i) {
                sums.push(sum);
            }
        }
        sums.sort_unstable();
        assert_eq!(sums, vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spillable_aggregate_quota_failure_remains_structured_storage_full() {
        use crate::execution::spill::SpillDiskQuota;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .quota(SpillDiskQuota::new(0))
                .build()
                .unwrap(),
        );
        let mut aggregate = SpillableAggregatePushOperator::with_spilling(
            vec![0],
            vec![AggregateExpr::sum(1)],
            manager,
            1,
        );
        let mut sink = CollectorSink::new();

        let error = aggregate
            .push(create_two_column_chunk(&[1], &[10]), &mut sink)
            .unwrap_err();

        assert!(matches!(error, OperatorError::StorageFull(_)));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_aggregate_global() {
        // Global aggregation shouldn't be affected by spilling
        let mut agg = SpillableAggregatePushOperator::global(vec![AggregateExpr::count_star()]);
        let mut sink = CollectorSink::new();

        agg.push(create_test_chunk(&[1, 2, 3, 4, 5]), &mut sink)
            .unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::Int64(5))
        );
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_aggregate_many_groups() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build()
                .unwrap(),
        );

        let mut agg = SpillableAggregatePushOperator::with_spilling(
            vec![0],
            vec![AggregateExpr::count_star()],
            manager,
            10, // Very low threshold
        );
        let mut sink = CollectorSink::new();

        // Create 100 different groups
        for i in 0..100 {
            let chunk = create_test_chunk(&[i]);
            agg.push(chunk, &mut sink).unwrap();
        }
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 100); // 100 groups

        // Each group should have count = 1
        for i in 0..100 {
            if let Some(Value::Int64(count)) = chunks[0].column(1).unwrap().get_value(i) {
                assert_eq!(count, 1);
            }
        }
    }

    // ---------------------------------------------------------------
    // hash_value coverage for all Value variants
    // ---------------------------------------------------------------

    #[test]
    fn hash_value_null() {
        let h = hash_value(&Value::Null);
        assert_ne!(h, 0); // hasher produces non-zero for Null discriminant
    }

    #[test]
    fn hash_value_bool() {
        let t = hash_value(&Value::Bool(true));
        let f = hash_value(&Value::Bool(false));
        assert_ne!(t, f);
    }

    #[test]
    fn hash_value_int64() {
        let a = hash_value(&Value::Int64(42));
        let b = hash_value(&Value::Int64(43));
        assert_ne!(a, b);
    }

    #[test]
    fn hash_value_float64() {
        let a = hash_value(&Value::Float64(19.88));
        let b = hash_value(&Value::Float64(3.19));
        assert_ne!(a, b);
    }

    #[test]
    fn hash_value_string() {
        let a = hash_value(&Value::String("hello".into()));
        let b = hash_value(&Value::String("world".into()));
        assert_ne!(a, b);
    }

    #[test]
    fn hash_value_bytes() {
        let a = hash_value(&Value::Bytes(vec![1, 2, 3].into()));
        let b = hash_value(&Value::Bytes(vec![4, 5, 6].into()));
        assert_ne!(a, b);
    }

    #[test]
    fn hash_value_list() {
        let a = hash_value(&Value::List(vec![Value::Int64(1), Value::Int64(2)].into()));
        let b = hash_value(&Value::List(vec![Value::Int64(3)].into()));
        assert_ne!(a, b);
    }

    #[test]
    fn hash_value_map() {
        use grafeo_common::types::PropertyKey;
        use std::collections::BTreeMap;
        use std::sync::Arc;
        let mut map = BTreeMap::new();
        map.insert(PropertyKey::new("key"), Value::Int64(42));
        let h = hash_value(&Value::Map(Arc::new(map)));
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_vector() {
        let h = hash_value(&Value::Vector(vec![1.0, 2.0, 3.0].into()));
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_path() {
        let h = hash_value(&Value::Path {
            nodes: vec![Value::Int64(1), Value::Int64(2)].into(),
            edges: vec![Value::Int64(10)].into(),
        });
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_gcounter() {
        use std::sync::Arc;
        let mut map = std::collections::HashMap::new();
        map.insert("replica1".to_string(), 10u64);
        let h = hash_value(&Value::GCounter(Arc::new(map)));
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_on_counter() {
        use std::sync::Arc;
        let mut pos = std::collections::HashMap::new();
        pos.insert("replica1".to_string(), 10u64);
        let neg = std::collections::HashMap::new();
        let h = hash_value(&Value::OnCounter {
            pos: Arc::new(pos),
            neg: Arc::new(neg),
        });
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_timestamp() {
        use grafeo_common::types::Timestamp;
        let h = hash_value(&Value::Timestamp(Timestamp::from_micros(1_700_000_000_000)));
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_date() {
        use grafeo_common::types::Date;
        let h = hash_value(&Value::Date(Date::from_days(19000)));
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_time() {
        use grafeo_common::types::Time;
        let h = hash_value(&Value::Time(Time::from_hms(12, 0, 0).unwrap()));
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_duration() {
        use grafeo_common::types::Duration;
        let h = hash_value(&Value::Duration(Duration::from_days(1)));
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_zoned_datetime() {
        use grafeo_common::types::{Timestamp, ZonedDatetime};
        let zdt =
            ZonedDatetime::from_timestamp_offset(Timestamp::from_micros(1_700_000_000_000), 3600);
        let h = hash_value(&Value::ZonedDatetime(zdt));
        assert_ne!(h, 0);
    }

    // ---------------------------------------------------------------
    // AggregateState in push context: advanced functions now work
    // ---------------------------------------------------------------

    #[test]
    fn aggregate_state_last_returns_last_value() {
        let mut state = AggregateState::new(AggregateFunction::Last, false, None, None);
        state.update(Some(Value::Int64(10)));
        state.update(Some(Value::Int64(20)));
        assert_eq!(state.finalize(), Value::Int64(20));
    }

    #[test]
    fn aggregate_state_collect_returns_list() {
        let mut state = AggregateState::new(AggregateFunction::Collect, false, None, None);
        state.update(Some(Value::Int64(1)));
        state.update(Some(Value::Int64(2)));
        assert_eq!(
            state.finalize(),
            Value::List(vec![Value::Int64(1), Value::Int64(2)].into())
        );
    }

    #[test]
    fn aggregate_state_stdev_returns_value() {
        let mut state = AggregateState::new(AggregateFunction::StdDev, false, None, None);
        state.update(Some(Value::Float64(2.0)));
        state.update(Some(Value::Float64(4.0)));
        state.update(Some(Value::Float64(6.0)));
        let result = state.finalize();
        assert!(matches!(result, Value::Float64(_)));
    }

    #[test]
    fn aggregate_state_first_returns_first_value() {
        let mut state = AggregateState::new(AggregateFunction::First, false, None, None);
        state.update(Some(Value::Int64(10)));
        state.update(Some(Value::Int64(20)));
        assert_eq!(state.finalize(), Value::Int64(10));
    }

    #[test]
    fn aggregate_state_avg_empty_returns_null() {
        let state = AggregateState::new(AggregateFunction::Avg, false, None, None);
        assert_eq!(state.finalize(), Value::Null);
    }

    #[test]
    fn aggregate_state_sum_empty_returns_null() {
        let state = AggregateState::new(AggregateFunction::Sum, false, None, None);
        assert_eq!(state.finalize(), Value::Null);
    }

    #[test]
    fn aggregate_state_min_max_empty_returns_null() {
        let min = AggregateState::new(AggregateFunction::Min, false, None, None);
        let max = AggregateState::new(AggregateFunction::Max, false, None, None);
        assert_eq!(min.finalize(), Value::Null);
        assert_eq!(max.finalize(), Value::Null);
    }

    #[test]
    fn aggregate_state_count_non_null_skips_nulls() {
        // CountNonNull maps to the Count(0) state variant, which increments
        // unconditionally. Callers (both push and pull operators) must filter
        // null values before calling update. This test verifies the expected
        // contract: only non-null values are fed to the accumulator.
        let mut state = AggregateState::new(AggregateFunction::CountNonNull, false, None, None);
        // Simulate what the operator should do: skip nulls, update only non-nulls
        // (Value::Null is skipped, Value::Int64(5) is the only non-null)
        state.update(Some(Value::Int64(5)));
        assert_eq!(state.finalize(), Value::Int64(1));
    }

    #[test]
    fn test_empty_chunk_returns_ok() {
        let mut agg = AggregatePushOperator::global(vec![AggregateExpr::count_star()]);
        let mut sink = CollectorSink::new();
        let empty = DataChunk::new(vec![ValueVector::new()]);
        let result = agg.push(empty, &mut sink).unwrap();
        assert!(result);
    }

    // ---------------------------------------------------------------
    // Spill serialization round-trip tests
    // ---------------------------------------------------------------

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_count() {
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::String("grp".into())],
            accumulators: vec![AggregateState::Count(42)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.key_values, vec![Value::String("grp".into())]);
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(42));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_sum_int() {
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::SumInt(100, 5)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(100));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_sum_float() {
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::SumFloat(3.125, 0.0, 2)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Float64(3.125));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_avg() {
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::Avg(30.0, 3)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Float64(10.0));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_min() {
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::Min(Some(Value::Int64(7)))],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(7));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_min_none() {
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::Min(None)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Null);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_max() {
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::Max(Some(Value::Int64(99)))],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(99));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_first() {
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::First(Some(Value::String("hello".into())))],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(
            restored.accumulators[0].finalize(),
            Value::String("hello".into())
        );
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_last() {
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::Last(Some(Value::Float64(2.75)))],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Float64(2.75));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_collect() {
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::Collect(vec![
                Value::Int64(10),
                Value::Int64(20),
                Value::Int64(30),
            ])],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(
            restored.accumulators[0].finalize(),
            Value::List(vec![Value::Int64(10), Value::Int64(20), Value::Int64(30)].into())
        );
    }

    #[test]
    #[cfg(feature = "spill")]
    fn finite_budget_partition_spill_reloads_nested_group_state_exactly() {
        use crate::execution::spill::{
            CleartextSpillRecordProvider, CodecLimits, SpillFrameLimits,
        };
        use grafeo_common::types::PropertyKey;
        use std::collections::BTreeMap;
        use tempfile::TempDir;

        let directory = TempDir::new().unwrap();
        let limits = SpillFrameLimits::new(4096, 4096)
            .unwrap()
            .with_codec_limits(CodecLimits::new(64 * 1024, 1024, 32, 32));
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(Arc::new(CleartextSpillRecordProvider), limits)
                .build()
                .unwrap(),
        );
        let mut partitioned = PartitionedState::new_with_bounded_codec(
            Arc::clone(&manager),
            4,
            serialize_group_state_bounded,
            deserialize_group_state_bounded,
        );
        let mut key_map = BTreeMap::new();
        key_map.insert(
            PropertyKey::new("key"),
            Value::List(Arc::from([Value::Null, Value::Bool(true)])),
        );
        let key = vec![
            Value::Map(Arc::new(key_map)),
            Value::Path {
                nodes: Arc::from([Value::Int64(1), Value::Int64(2)]),
                edges: Arc::from([Value::String("key-edge".into())]),
            },
        ];
        let mut collected_map = BTreeMap::new();
        collected_map.insert(PropertyKey::new("nested"), Value::String("value".into()));
        let collected = vec![
            Value::Map(Arc::new(collected_map)),
            Value::Path {
                nodes: Arc::from([Value::Int64(3), Value::Int64(4)]),
                edges: Arc::from([Value::String("collected-edge".into())]),
            },
            Value::List(Arc::from([Value::Null, Value::Int64(9)])),
        ];
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: key.clone(),
            accumulators: vec![AggregateState::Collect(collected.clone())],
        };
        partitioned.insert(key.clone(), state).unwrap();
        let partition = partitioned.partition_for(&key);
        partitioned.spill_partition(partition).unwrap();

        let restored = partitioned.get(&key).unwrap().unwrap();
        assert_eq!(restored.key_values, key);
        assert_eq!(
            restored.accumulators[0].finalize(),
            Value::List(Arc::from(collected.clone()))
        );
        let restored = partitioned
            .get_or_insert_with(key.clone(), || unreachable!())
            .unwrap();
        restored.accumulators[0].update(Some(Value::String("continued".into())));
        let mut expected = collected;
        expected.push(Value::String("continued".into()));
        assert_eq!(
            restored.accumulators[0].finalize(),
            Value::List(Arc::from(expected))
        );
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_distinct_codec_reload_accepts_new_and_rejects_duplicate() {
        let mut state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: [
                AggregateFunction::Count,
                AggregateFunction::StdDevPop,
                AggregateFunction::PercentileCont,
            ]
            .into_iter()
            .map(|function| AggregateState::new(function, true, Some(0.5), None))
            .collect(),
        };
        for value in [1, 3] {
            for accumulator in &mut state.accumulators {
                accumulator.update(Some(Value::Int64(value)));
            }
        }
        let mut bytes = Vec::new();
        serialize_group_state(&state, &mut bytes).unwrap();
        let mut restored = deserialize_group_state(&mut bytes.as_slice()).unwrap();
        // A finalized-value-only roundtrip passes before these updates. Both
        // retained DISTINCT identity and live numerical state must survive.
        for value in [3, 5] {
            for accumulator in &mut restored.accumulators {
                accumulator.update(Some(Value::Int64(value)));
            }
        }
        let actual: Vec<_> = restored
            .accumulators
            .iter()
            .map(AggregateState::finalize)
            .collect();
        assert!(
            actual[0] == Value::Int64(3)
                && matches!(actual[1], Value::Float64(value) if (value - (8.0_f64 / 3.0).sqrt()).abs() < 1e-12)
                && actual[2] == Value::Float64(3.0),
            "reloaded DISTINCT states must continue over unique operands [1,3,5]: {actual:?}"
        );
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_every_live_state_continues_like_uninterrupted_execution() {
        let functions = [
            AggregateFunction::Count,
            AggregateFunction::CountNonNull,
            AggregateFunction::Sum,
            AggregateFunction::Avg,
            AggregateFunction::Min,
            AggregateFunction::Max,
            AggregateFunction::First,
            AggregateFunction::Last,
            AggregateFunction::Collect,
            AggregateFunction::StdDev,
            AggregateFunction::StdDevPop,
            AggregateFunction::Variance,
            AggregateFunction::VariancePop,
            AggregateFunction::PercentileDisc,
            AggregateFunction::PercentileCont,
            AggregateFunction::GroupConcat,
            AggregateFunction::Sample,
            AggregateFunction::CovarSamp,
            AggregateFunction::CovarPop,
            AggregateFunction::Corr,
            AggregateFunction::RegrSlope,
            AggregateFunction::RegrIntercept,
            AggregateFunction::RegrR2,
            AggregateFunction::RegrCount,
            AggregateFunction::RegrSxx,
            AggregateFunction::RegrSyy,
            AggregateFunction::RegrSxy,
            AggregateFunction::RegrAvgx,
            AggregateFunction::RegrAvgy,
        ];
        let update = |state: &mut AggregateState, y, x, key| {
            if matches!(state, AggregateState::Bivariate { .. }) {
                state.update_bivariate_with_distinct_key(
                    Some(Value::Int64(y)),
                    Some(Value::Int64(x)),
                    Some(Value::Int64(key)),
                );
            } else {
                state.update_with_distinct_key(Some(Value::Int64(y)), Some(Value::Int64(key)));
            }
        };
        for function in functions {
            for distinct in [false, true] {
                let mut uninterrupted =
                    AggregateState::new(function, distinct, Some(0.5), Some("|"));
                update(&mut uninterrupted, 1, 2, 1);
                update(&mut uninterrupted, 3, 4, 2);
                let group = GroupState {
                    #[cfg(feature = "spill")]
                    retained_heap_bytes_cache: None,
                    key_values: vec![],
                    accumulators: vec![uninterrupted.clone()],
                };
                let mut bytes = Vec::new();
                serialize_group_state(&group, &mut bytes).unwrap();
                let mut restored = deserialize_group_state(&mut bytes.as_slice()).unwrap();
                assert!(!matches!(
                    restored.accumulators[0],
                    AggregateState::Frozen(_)
                ));
                // Existing key, new value/key, equal value with a new key,
                // then changed value with the oldest key.
                for (y, x, key) in [(3, 4, 2), (5, 8, 3), (5, 8, 4), (99, 88, 1)] {
                    update(&mut uninterrupted, y, x, key);
                    update(&mut restored.accumulators[0], y, x, key);
                    assert_eq!(
                        restored.accumulators[0].finalize(),
                        uninterrupted.finalize(),
                        "{function:?}, distinct={distinct}, continuation key={key}"
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_float_compensation_and_some_null_first_survive_reload() {
        for distinct in [false, true] {
            let mut uninterrupted =
                AggregateState::new(AggregateFunction::Sum, distinct, None, None);
            uninterrupted.update(Some(Value::Float64(1e16)));
            uninterrupted.update(Some(Value::Int64(1)));
            assert!(matches!(
                &uninterrupted,
                AggregateState::SumFloat(_, comp, _)
                    | AggregateState::SumFloatDistinct(_, comp, _, _) if *comp != 0.0
            ));
            let group = GroupState {
                #[cfg(feature = "spill")]
                retained_heap_bytes_cache: None,
                key_values: vec![],
                accumulators: vec![
                    uninterrupted.clone(),
                    AggregateState::First(Some(Value::Null)),
                ],
            };
            let mut bytes = Vec::new();
            serialize_group_state(&group, &mut bytes).unwrap();
            let mut restored = deserialize_group_state(&mut bytes.as_slice()).unwrap();
            uninterrupted.update(Some(Value::Int64(2)));
            restored.accumulators[0].update(Some(Value::Int64(2)));
            restored.accumulators[1].update(Some(Value::Int64(7)));
            assert_eq!(
                restored.accumulators[0].finalize(),
                uninterrupted.finalize()
            );
            assert_eq!(restored.accumulators[1].finalize(), Value::Null);
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_distinct_backing_bytes_do_not_consume_logical_item_budget() {
        use crate::execution::operators::accumulator::HashableValue;
        use crate::execution::spill::CodecLimits;

        let group = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![],
            accumulators: vec![AggregateState::CountDistinct(
                2,
                [HashableValue::Int64(1), HashableValue::Int64(3)]
                    .into_iter()
                    .collect(),
            )],
        };
        // One accumulator and two live identities are three logical items.
        // Two keys reserve four hash buckets and 4 + 64 control bytes.
        let backing_bytes =
            std::mem::size_of::<AggregateState>() + 4 * std::mem::size_of::<HashableValue>() + 68;
        let limits = CodecLimits::new(backing_bytes, 3, 1, 4);
        let mut bytes = Vec::new();
        serialize_group_state_bounded(&group, &mut bytes, limits).unwrap();
        let restored = deserialize_group_state_bounded(
            &mut bytes.as_slice(),
            limits.bounded_to_payload(bytes.len()),
        )
        .unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(2));

        for (limits, message) in [
            (
                CodecLimits::new(backing_bytes - 1, 3, 1, 4),
                "byte codec budget",
            ),
            (
                CodecLimits::new(backing_bytes, 2, 1, 4),
                "item codec budget",
            ),
        ] {
            let write_error =
                serialize_group_state_bounded(&group, &mut Vec::new(), limits).unwrap_err();
            assert_eq!(write_error.kind(), std::io::ErrorKind::InvalidInput);
            assert!(write_error.to_string().contains(message));
            let read_error = deserialize_group_state_bounded(&mut bytes.as_slice(), limits)
                .err()
                .unwrap();
            assert_eq!(read_error.kind(), std::io::ErrorKind::InvalidData);
            assert!(read_error.to_string().contains(message));
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_distinct_restore_rejects_excess_budget_and_duplicate_identities() {
        use crate::execution::spill::{CodecLimits, SpillCodecEncodeBudget};
        let mut bytes = Vec::new();
        write_count(&mut bytes, 0).unwrap(); // Group keys.
        write_count(&mut bytes, 1).unwrap(); // Accumulators.
        bytes.write_all(&[spill_tag::COUNT_DISTINCT]).unwrap();
        bytes.write_all(&2_i64.to_le_bytes()).unwrap();
        write_count(&mut bytes, 2).unwrap(); // Two serialized identities.
        let mut budget = SpillCodecEncodeBudget::new(CodecLimits::format_max());
        for _ in 0..2 {
            write_identity(
                &crate::execution::operators::accumulator::HashableValue::Int64(7),
                &mut bytes,
                &mut budget,
            )
            .unwrap();
        }
        let error = deserialize_group_state(&mut bytes.as_slice())
            .err()
            .unwrap();
        assert!(error.to_string().contains("duplicate serialized"));
        let error = deserialize_group_state_bounded(
            &mut bytes.as_slice(),
            CodecLimits::new(64 * 1024, 2, 4, 4),
        )
        .err()
        .unwrap();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("item codec budget"));
        // A declared huge collection is refused from its header, before any
        // key can be decoded or its backing set reserved.
        bytes.truncate(25);
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(deserialize_group_state(&mut bytes.as_slice()).is_err());
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_distinct_identity_variants_and_other_keys_survive_continuation() {
        use crate::execution::operators::accumulator::HashableValue;
        use grafeo_common::utils::hash::FxHashSet;
        let list = Value::List(Arc::from([Value::Int64(1)]));
        let spelling = format!("{list:?}");
        let identities: FxHashSet<_> = [
            HashableValue::Null,
            HashableValue::Bool(true),
            HashableValue::Int64(7),
            HashableValue::Float64Bits(7.0_f64.to_bits()),
            HashableValue::Float64Bits(f64::NAN.to_bits() | 0x11),
            HashableValue::String(spelling.clone()),
            HashableValue::Other(spelling.clone()),
        ]
        .into_iter()
        .collect();
        let mut pair_state = AggregateState::new(AggregateFunction::RegrCount, true, None, None);
        for x in [10, 20] {
            pair_state.update_bivariate(Some(Value::Int64(1)), Some(Value::Int64(x)));
        }
        let group = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![],
            accumulators: vec![
                AggregateState::CountDistinct(7, identities.clone()),
                pair_state,
            ],
        };
        let mut bytes = Vec::new();
        serialize_group_state(&group, &mut bytes).unwrap();
        let mut restored = deserialize_group_state(&mut bytes.as_slice()).unwrap();
        let AggregateState::CountDistinct(_, seen) = &restored.accumulators[0] else {
            panic!("DISTINCT must remain live");
        };
        assert_eq!(
            seen, &identities,
            "preserve every identity discriminant and float bit pattern"
        );
        restored.accumulators[0].update(Some(list));
        restored.accumulators[0].update(Some(Value::String(spelling.into())));
        restored.accumulators[0].update(Some(Value::List(Arc::from([Value::Int64(2)]))));
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(8));
        for x in [10, 30] {
            restored.accumulators[1].update_bivariate(Some(Value::Int64(1)), Some(Value::Int64(x)));
        }
        assert_eq!(restored.accumulators[1].finalize(), Value::Int64(3));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_live_state_rejects_negative_sample_counts() {
        // Offsets are inside the existing group payload: 8-byte key count,
        // 8-byte accumulator count, one state tag, then each state's fields.
        let cases = [
            (
                AggregateState::new(AggregateFunction::Count, true, None, None),
                17,
            ),
            (
                AggregateState::new(AggregateFunction::Sum, true, None, None),
                25,
            ),
            (AggregateState::SumFloat(1.0, 0.0, 1), 33),
            (
                AggregateState::new(AggregateFunction::Avg, true, None, None),
                25,
            ),
            (
                AggregateState::new(AggregateFunction::StdDevPop, true, None, None),
                17,
            ),
            (
                AggregateState::new(AggregateFunction::RegrCount, true, None, None),
                18,
            ),
        ];
        for (state, count_offset) in cases {
            let group = GroupState {
                #[cfg(feature = "spill")]
                retained_heap_bytes_cache: None,
                key_values: vec![],
                accumulators: vec![state],
            };
            let mut bytes = Vec::new();
            serialize_group_state(&group, &mut bytes).unwrap();
            bytes[count_offset..count_offset + 8].copy_from_slice(&(-1_i64).to_le_bytes());
            let error = deserialize_group_state(&mut bytes.as_slice())
                .err()
                .unwrap();
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
            assert!(
                error
                    .to_string()
                    .contains("negative aggregate sample count")
            );
        }
        // Signed sums and floating special values are legitimate state, unlike
        // negative numbers of observations; do not reject them wholesale.
        let group = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![],
            accumulators: vec![AggregateState::SumFloat(f64::INFINITY, f64::NAN, 1)],
        };
        let mut bytes = Vec::new();
        serialize_group_state(&group, &mut bytes).unwrap();
        let restored = deserialize_group_state(&mut bytes.as_slice()).unwrap();
        assert!(
            matches!(restored.accumulators[0], AggregateState::SumFloat(sum, comp, 1)
            if sum == f64::INFINITY && comp.is_nan())
        );
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_distinct_partition_reload_accepts_new_and_rejects_duplicate() {
        use crate::execution::spill::{
            CleartextSpillRecordProvider, CodecLimits, SpillFrameLimits,
        };
        use tempfile::TempDir;

        let directory = TempDir::new().unwrap();
        let limits = SpillFrameLimits::new(4096, 4096)
            .unwrap()
            .with_codec_limits(CodecLimits::new(64 * 1024, 1024, 32, 32));
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(Arc::new(CleartextSpillRecordProvider), limits)
                .build()
                .unwrap(),
        );
        let mut aggregate = SpillableAggregatePushOperator::with_spilling(
            vec![0],
            vec![
                AggregateExpr::count(1).with_distinct(),
                AggregateExpr::stdev_pop(1).with_distinct(),
                AggregateExpr::percentile_cont(1, 0.5).with_distinct(),
            ],
            Arc::clone(&manager),
            1,
        );
        let mut sink = CollectorSink::new();
        aggregate
            .push(create_two_column_chunk(&[1, 1], &[1, 3]), &mut sink)
            .unwrap();
        let partitioned = aggregate.partitioned_groups.as_mut().unwrap();
        let partition = partitioned.partition_for(&[Value::Int64(1)]);
        partitioned.spill_partition(partition).unwrap();
        assert!(
            manager.active_file_count() > 0,
            "must evict a real partition"
        );
        // Exercise the real push row updater, which reloads this exact group.
        aggregate
            .push(create_two_column_chunk(&[1, 1], &[3, 5]), &mut sink)
            .unwrap();
        aggregate.finalize(&mut sink).unwrap();
        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].row_count(), 1);
        let actual: Vec<_> = (1..4)
            .map(|column| chunks[0].column(column).unwrap().get_value(0).unwrap())
            .collect();
        assert_eq!(manager.active_file_count(), 0);
        assert!(
            actual[0] == Value::Int64(3)
                && matches!(actual[1], Value::Float64(value) if (value - (8.0_f64 / 3.0).sqrt()).abs() < 1e-12)
                && actual[2] == Value::Float64(3.0),
            "evicted DISTINCT group must resume over unique operands [1,3,5]: {actual:?}"
        );
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_all_variants_combined() {
        // A single GroupState with every common accumulator type
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::String("combined".into()), Value::Int64(42)],
            accumulators: vec![
                AggregateState::Count(10),
                AggregateState::SumInt(50, 5),
                AggregateState::SumFloat(7.5, 0.0, 3),
                AggregateState::Avg(20.0, 4),
                AggregateState::Min(Some(Value::Int64(1))),
                AggregateState::Max(Some(Value::Int64(99))),
                AggregateState::First(Some(Value::String("first".into()))),
                AggregateState::Last(Some(Value::String("last".into()))),
                AggregateState::Collect(vec![Value::Int64(1), Value::Int64(2)]),
            ],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();

        assert_eq!(restored.key_values.len(), 2);
        assert_eq!(restored.key_values[0], Value::String("combined".into()));
        assert_eq!(restored.key_values[1], Value::Int64(42));
        assert_eq!(restored.accumulators.len(), 9);

        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(10));
        assert_eq!(restored.accumulators[1].finalize(), Value::Int64(50));
        assert_eq!(restored.accumulators[2].finalize(), Value::Float64(7.5));
        assert_eq!(restored.accumulators[3].finalize(), Value::Float64(5.0));
        assert_eq!(restored.accumulators[4].finalize(), Value::Int64(1));
        assert_eq!(restored.accumulators[5].finalize(), Value::Int64(99));
        assert_eq!(
            restored.accumulators[6].finalize(),
            Value::String("first".into())
        );
        assert_eq!(
            restored.accumulators[7].finalize(),
            Value::String("last".into())
        );
        assert_eq!(
            restored.accumulators[8].finalize(),
            Value::List(vec![Value::Int64(1), Value::Int64(2)].into())
        );
    }

    // ---------------------------------------------------------------
    // DISTINCT variants retain their live state and seen identities
    // ---------------------------------------------------------------

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_count_distinct() {
        use crate::execution::operators::accumulator::HashableValue;
        use grafeo_common::utils::hash::FxHashSet;

        let mut seen = FxHashSet::default();
        seen.insert(HashableValue::from(Value::Int64(1)));
        seen.insert(HashableValue::from(Value::Int64(2)));
        seen.insert(HashableValue::from(Value::Int64(3)));
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::CountDistinct(3, seen)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        // The live DISTINCT accumulator retains its count and seen identities.
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(3));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_avg_distinct() {
        use crate::execution::operators::accumulator::HashableValue;
        use grafeo_common::utils::hash::FxHashSet;

        let mut seen = FxHashSet::default();
        seen.insert(HashableValue::from(Value::Float64(2.0)));
        seen.insert(HashableValue::from(Value::Float64(4.0)));
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::AvgDistinct(6.0, 2, seen)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Float64(3.0));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_collect_distinct() {
        use crate::execution::operators::accumulator::HashableValue;
        use grafeo_common::utils::hash::FxHashSet;

        let mut seen = FxHashSet::default();
        seen.insert(HashableValue::from(Value::Int64(10)));
        seen.insert(HashableValue::from(Value::Int64(20)));
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::CollectDistinct(
                vec![Value::Int64(10), Value::Int64(20)],
                seen,
            )],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        // Collection order and seen identities survive reload.
        let result = restored.accumulators[0].finalize();
        assert!(matches!(result, Value::List(_)));
    }

    // ---------------------------------------------------------------
    // Complex live-state roundtrips
    // ---------------------------------------------------------------

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_stddev() {
        // Build a StdDev state by feeding values
        let mut acc = AggregateState::new(AggregateFunction::StdDev, false, None, None);
        acc.update(Some(Value::Float64(2.0)));
        acc.update(Some(Value::Float64(4.0)));
        acc.update(Some(Value::Float64(6.0)));
        let expected = acc.finalize();

        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: vec![acc],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        // Sufficient statistics survive without premature finalization.
        assert_eq!(restored.accumulators[0].finalize(), expected);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_percentile_disc() {
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::PercentileDisc {
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                percentile: 0.5,
                seen: None,
            }],
        };
        let expected = state.accumulators[0].finalize();
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), expected);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_group_concat() {
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::GroupConcat(
                vec!["alix".to_string(), "gus".to_string(), "vincent".to_string()],
                ", ".to_string(),
            )],
        };
        let expected = state.accumulators[0].finalize();
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), expected);
    }

    // ---------------------------------------------------------------
    // SpillableAggregatePushOperator with Collect
    // ---------------------------------------------------------------

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_aggregate_collect() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build()
                .unwrap(),
        );

        let mut agg = SpillableAggregatePushOperator::with_spilling(
            vec![0],
            vec![AggregateExpr::collect(1)],
            manager,
            3, // Spill after 3 groups
        );
        let mut sink = CollectorSink::new();

        // Create groups: group 1 collects [10, 20], group 2 collects [30, 40]
        agg.push(
            create_two_column_chunk(&[1, 2, 1, 2], &[10, 30, 20, 40]),
            &mut sink,
        )
        .unwrap();
        // Add more groups to trigger spilling
        for i in 3..10 {
            agg.push(create_two_column_chunk(&[i], &[i * 10]), &mut sink)
                .unwrap();
        }
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 9); // 9 groups

        // Find group 1 and verify its collected list
        let mut found_group1 = false;
        for row in 0..chunks[0].len() {
            if let Some(Value::Int64(1)) = chunks[0].column(0).unwrap().get_value(row) {
                let collected = chunks[0].column(1).unwrap().get_value(row).unwrap();
                if let Value::List(list) = collected {
                    assert_eq!(list.len(), 2);
                    assert!(list.contains(&Value::Int64(10)));
                    assert!(list.contains(&Value::Int64(20)));
                    found_group1 = true;
                }
            }
        }
        assert!(found_group1, "Group 1 with collected values not found");
    }

    // ---------------------------------------------------------------
    // SpillableAggregatePushOperator with Min/Max
    // ---------------------------------------------------------------

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_aggregate_min_max() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build()
                .unwrap(),
        );

        let mut agg = SpillableAggregatePushOperator::with_spilling(
            vec![0],
            vec![AggregateExpr::min(1), AggregateExpr::max(1)],
            manager,
            3, // Spill after 3 groups
        );
        let mut sink = CollectorSink::new();

        // Group 1: values 50, 10, 30 => min=10, max=50
        // Group 2: values 20, 40 => min=20, max=40
        agg.push(
            create_two_column_chunk(&[1, 2, 1, 2, 1], &[50, 20, 10, 40, 30]),
            &mut sink,
        )
        .unwrap();

        // Add more groups to trigger spilling
        for i in 3..10 {
            agg.push(create_two_column_chunk(&[i], &[i * 10]), &mut sink)
                .unwrap();
        }
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 9); // 9 groups

        // Verify group 1: min=10, max=50
        let mut found_group1 = false;
        for row in 0..chunks[0].len() {
            if let Some(Value::Int64(1)) = chunks[0].column(0).unwrap().get_value(row) {
                assert_eq!(
                    chunks[0].column(1).unwrap().get_value(row),
                    Some(Value::Int64(10))
                );
                assert_eq!(
                    chunks[0].column(2).unwrap().get_value(row),
                    Some(Value::Int64(50))
                );
                found_group1 = true;
            }
        }
        assert!(found_group1, "Group 1 with min/max not found");

        // Verify group 2: min=20, max=40
        let mut found_group2 = false;
        for row in 0..chunks[0].len() {
            if let Some(Value::Int64(2)) = chunks[0].column(0).unwrap().get_value(row) {
                assert_eq!(
                    chunks[0].column(1).unwrap().get_value(row),
                    Some(Value::Int64(20))
                );
                assert_eq!(
                    chunks[0].column(2).unwrap().get_value(row),
                    Some(Value::Int64(40))
                );
                found_group2 = true;
            }
        }
        assert!(found_group2, "Group 2 with min/max not found");
    }

    // ---------------------------------------------------------------
    // Additional aggregate push operator coverage
    // ---------------------------------------------------------------

    #[test]
    fn test_aggregate_count_non_null() {
        // COUNT(column) with CountNonNull skips null values
        let expr = AggregateExpr::count(0);
        let mut agg = AggregatePushOperator::global(vec![expr]);
        let mut sink = CollectorSink::new();

        // Create a chunk with mixed values and nulls
        let mut col = ValueVector::new();
        col.push(Value::Int64(10)); // Alix's score
        col.push(Value::Null);
        col.push(Value::Int64(30)); // Gus's score
        col.push(Value::Null);
        col.push(Value::Int64(50)); // Vincent's score
        let chunk = DataChunk::new(vec![col]);

        agg.push(chunk, &mut sink).unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        // Only 3 non-null values should be counted
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::Int64(3))
        );
    }

    #[test]
    fn test_grouped_aggregate_empty_groups() {
        // Grouped aggregate with empty input produces no output
        let mut agg = AggregatePushOperator::new(vec![0], vec![AggregateExpr::sum(1)]);
        let mut sink = CollectorSink::new();

        // Push an empty chunk
        let empty = DataChunk::new(vec![ValueVector::new(), ValueVector::new()]);
        agg.push(empty, &mut sink).unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        // No groups produced, so no output chunk
        assert!(chunks.is_empty());
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_aggregate_threshold_transition() {
        // Test the transition from non-partitioned to partitioned mode
        // when the spill_manager is set but threshold is reached without
        // using with_spilling (tests the maybe_spill fallback path)
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build()
                .unwrap(),
        );

        // Use with_spilling to trigger the partitioned spill path
        let mut agg = SpillableAggregatePushOperator::with_spilling(
            vec![0],
            vec![AggregateExpr::count_star()],
            manager,
            2, // Very low threshold
        );
        let mut sink = CollectorSink::new();

        // Create 5 groups to force spilling
        for i in 0..5 {
            agg.push(create_test_chunk(&[i]), &mut sink).unwrap();
        }
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 5);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_explicit_frozen_state_remains_terminal() {
        let mut acc = AggregateState::new(AggregateFunction::StdDev, false, None, None);
        acc.update(Some(Value::Float64(2.0)));
        acc.update(Some(Value::Float64(4.0)));
        acc.update(Some(Value::Float64(6.0)));
        let expected = acc.finalize();

        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::Frozen(expected.clone())],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let mut restored = deserialize_group_state(&mut &buf[..]).unwrap();

        assert!(matches!(
            restored.accumulators[0],
            AggregateState::Frozen(_)
        ));

        restored.accumulators[0].update(Some(Value::Float64(100.0)));
        restored.accumulators[0].update(Some(Value::Float64(200.0)));

        assert_eq!(restored.accumulators[0].finalize(), expected);
    }

    // ---------------------------------------------------------------
    // Serialization roundtrip: end-to-end through push operator
    // ---------------------------------------------------------------

    #[test]
    #[cfg(feature = "spill")]
    fn test_serialize_deserialize_sum_state() {
        // Sum with float values to exercise SumFloat serialization path
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::String("Alix".into())],
            accumulators: vec![
                AggregateState::SumInt(42, 3),
                AggregateState::SumFloat(2.72, 0.001, 2),
            ],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();

        assert_eq!(restored.key_values, vec![Value::String("Alix".into())]);
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(42));
        assert_eq!(restored.accumulators[1].finalize(), Value::Float64(2.72));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_serialize_deserialize_avg_state() {
        // Avg with sum=30.0, count=6 => finalize should produce 5.0
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::String("Gus".into())],
            accumulators: vec![AggregateState::Avg(30.0, 6)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();

        assert_eq!(restored.key_values, vec![Value::String("Gus".into())]);
        assert_eq!(restored.accumulators[0].finalize(), Value::Float64(5.0));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_serialize_deserialize_count_state() {
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::String("Vincent".into())],
            accumulators: vec![AggregateState::Count(17)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();

        assert_eq!(restored.key_values, vec![Value::String("Vincent".into())]);
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(17));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_serialize_deserialize_min_max_state() {
        // Test with String values (not just Int64) to cover different value types
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::String("Jules".into())],
            accumulators: vec![
                AggregateState::Min(Some(Value::String("Amsterdam".into()))),
                AggregateState::Max(Some(Value::Float64(99.9))),
                AggregateState::Min(None),
                AggregateState::Max(None),
            ],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();

        assert_eq!(
            restored.accumulators[0].finalize(),
            Value::String("Amsterdam".into())
        );
        assert_eq!(restored.accumulators[1].finalize(), Value::Float64(99.9));
        // None values serialize as Null and deserialize as Min(None)
        assert_eq!(restored.accumulators[2].finalize(), Value::Null);
        assert_eq!(restored.accumulators[3].finalize(), Value::Null);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_serialize_deserialize_collect_state() {
        // Collect with mixed value types
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::String("Mia".into())],
            accumulators: vec![AggregateState::Collect(vec![
                Value::Int64(1),
                Value::String("Berlin".into()),
                Value::Float64(2.5),
                Value::Bool(true),
            ])],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();

        let result = restored.accumulators[0].finalize();
        if let Value::List(list) = result {
            assert_eq!(list.len(), 4);
            assert_eq!(list[0], Value::Int64(1));
            assert_eq!(list[1], Value::String("Berlin".into()));
            assert_eq!(list[2], Value::Float64(2.5));
            assert_eq!(list[3], Value::Bool(true));
        } else {
            panic!("expected List, got {result:?}");
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_serialize_deserialize_count_distinct() {
        use crate::execution::operators::accumulator::HashableValue;
        use grafeo_common::utils::hash::FxHashSet;

        let mut seen = FxHashSet::default();
        seen.insert(HashableValue::from(Value::String("Paris".into())));
        seen.insert(HashableValue::from(Value::String("Prague".into())));
        seen.insert(HashableValue::from(Value::String("Barcelona".into())));
        let state = GroupState {
            #[cfg(feature = "spill")]
            retained_heap_bytes_cache: None,
            key_values: vec![Value::String("Butch".into())],
            accumulators: vec![AggregateState::CountDistinct(3, seen)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();

        // CountDistinct remains live after decoding.
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(3));
        assert!(
            matches!(restored.accumulators[0], AggregateState::CountDistinct(..)),
            "DISTINCT must retain its live seen-key state"
        );
    }

    // ---------------------------------------------------------------
    // Push operator with empty chunks and empty input
    // ---------------------------------------------------------------

    #[test]
    fn test_global_aggregate_empty_input() {
        // Global aggregate with no input at all should produce correct defaults
        let mut agg = AggregatePushOperator::global(vec![
            AggregateExpr::count_star(),
            AggregateExpr::sum(0),
            AggregateExpr::min(0),
            AggregateExpr::max(0),
        ]);
        let mut sink = CollectorSink::new();

        // No push calls at all, directly finalize
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        // COUNT(*) with no input should be 0
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::Int64(0))
        );
        // SUM with no input should be Null
        assert_eq!(chunks[0].column(1).unwrap().get_value(0), Some(Value::Null));
        // MIN with no input should be Null
        assert_eq!(chunks[0].column(2).unwrap().get_value(0), Some(Value::Null));
        // MAX with no input should be Null
        assert_eq!(chunks[0].column(3).unwrap().get_value(0), Some(Value::Null));
    }

    // ---------------------------------------------------------------
    // Spillable aggregate: memory pressure triggers spilling
    // ---------------------------------------------------------------

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_aggregate_memory_pressure() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build()
                .unwrap(),
        );

        // Extremely low threshold of 2 to force spilling quickly
        let mut agg = SpillableAggregatePushOperator::with_spilling(
            vec![0],
            vec![AggregateExpr::sum(1)],
            Arc::clone(&manager),
            2,
        );
        let mut sink = CollectorSink::new();

        // Push many distinct groups to trigger memory pressure and spilling
        for i in 0..20 {
            let chunk = create_two_column_chunk(&[i], &[i * 5]);
            agg.push(chunk, &mut sink).unwrap();
        }

        // Verify spilling happened (manager should have active spill files)
        assert!(
            manager.active_file_count() > 0,
            "expected spill files to be created under memory pressure"
        );

        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 20);

        // Verify all sums are correct
        let mut sums: Vec<i64> = Vec::new();
        for i in 0..chunks[0].len() {
            if let Some(Value::Int64(sum)) = chunks[0].column(1).unwrap().get_value(i) {
                sums.push(sum);
            }
        }
        sums.sort_unstable();
        let expected: Vec<i64> = (0..20).map(|i| i * 5).collect();
        assert_eq!(sums, expected);
    }

    #[test]
    #[cfg(all(feature = "spill", any(target_os = "linux", target_os = "macos")))]
    fn scoped_registration_lives_through_aggregate_sink_error() {
        struct CancellingFailingSink {
            cancellation: crate::execution::QueryCancellationHandle,
        }

        impl Sink for CancellingFailingSink {
            fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
                self.cancellation.cancel();
                Err(OperatorError::Execution(
                    "injected sink failure".to_string(),
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

        let temp_dir = tempfile::TempDir::new().unwrap();
        let buffer_manager = grafeo_common::memory::buffer::BufferManager::with_budget(1024 * 1024);
        let spill_root = crate::execution::spill::RootedSpillFixture::new(temp_dir.path())
            .root()
            .unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let resources = crate::execution::QueryResourceContext::with_spill_root(
            Arc::clone(&buffer_manager),
            &spill_root,
            control.token(),
        )
        .unwrap();
        resources.ensure_spill_manager().unwrap().unwrap();
        let baseline = buffer_manager.stats().consumer_count;
        let mut aggregate = SpillableAggregatePushOperator::with_resource_context(
            Vec::new(),
            vec![AggregateExpr::count_star()],
            resources,
        )
        .unwrap();
        let mut sink = CancellingFailingSink {
            cancellation: control.cancellation_handle(),
        };
        assert_eq!(buffer_manager.stats().consumer_count, baseline + 1);

        let error = aggregate.finalize(&mut sink).unwrap_err();
        assert!(matches!(
            error,
            OperatorError::Execution(ref message) if message == "injected sink failure"
        ));
        assert!(control.token().is_cancelled());
        assert_eq!(
            buffer_manager.stats().consumer_count,
            baseline + 1,
            "a finalize error must leave cleanup to the operator's RAII scope"
        );

        drop(aggregate);
        assert_eq!(buffer_manager.stats().consumer_count, baseline);
    }
}
