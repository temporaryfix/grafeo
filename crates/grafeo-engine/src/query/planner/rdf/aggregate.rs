//! RDF-local aggregate execution.
//!
//! SPARQL set functions have materially different empty-group, error, numeric
//! promotion, and RDF-term DISTINCT semantics from SQL/GQL aggregates. Keeping
//! that state here prevents RDF correctness work from changing the generic LPG
//! pull, push, or spill contracts.

use grafeo_common::types::{HashableValue, LogicalType, Value};
use grafeo_core::execution::chunk::DataChunkBuilder;
use grafeo_core::execution::operators::accumulator::AggregateState;
use grafeo_core::execution::operators::value_utils::compare_values;
use grafeo_core::execution::operators::{
    AggregateExpr, AggregateFunction, Operator, OperatorError, OperatorResult,
};
use grafeo_core::execution::{QueryResourceContext, QueryResourceContextError};
use grafeo_core::graph::rdf::{Literal, Term};
use indexmap::IndexMap;
use std::cmp::Ordering;
#[cfg(feature = "spill")]
use std::io::{Cursor, Read, Write};

#[cfg(feature = "spill")]
use grafeo_core::execution::spill::{SpillCodecDecodeBudget, SpillCodecEncodeBudget};

use super::{decode_tagged_rdf_filter_term, numeric::RdfNumeric, sort::rdf_compare_terms};

mod bounded;
#[cfg(all(test, feature = "spill"))]
#[path = "aggregate/closure_tests.rs"]
mod closure_tests;
#[cfg(feature = "spill")]
mod spill;

const OUTPUT_CHUNK_SIZE: usize = 2048;
/// Pull-based SPARQL aggregation with exact RDF set-function semantics.
pub(super) struct RdfAggregateOperator {
    child: Box<dyn Operator>,
    group_columns: Vec<usize>,
    aggregates: Vec<AggregateExpr>,
    output_schema: Vec<LogicalType>,
    row_identity_columns: Option<Vec<RdfRowIdentityColumn>>,
    groups: IndexMap<Vec<HashableValue>, RdfGroupState>,
    results: Option<indexmap::map::IntoIter<Vec<HashableValue>, RdfGroupState>>,
    next_ordinal: u64,
    admission: Option<bounded::Admission>,
    resources: Option<QueryResourceContext>,
    failed: bool,
    #[cfg(feature = "spill")]
    spilled: Option<Box<dyn Operator>>,
    #[cfg(feature = "spill")]
    shared_input: Option<std::sync::Arc<parking_lot::Mutex<Box<dyn Operator>>>>,
}

#[derive(Clone, Copy)]
pub(super) struct RdfRowIdentityColumn {
    pub(super) visible: usize,
    pub(super) canonical_rdf: Option<usize>,
}

struct RdfGroupState {
    key_values: Vec<Value>,
    accumulators: Vec<RdfAccumulator>,
}

#[cfg(feature = "spill")]
const RDF_AGGREGATE_STATE_MAGIC: [u8; 4] = *b"RAGS";
#[cfg(feature = "spill")]
const RDF_AGGREGATE_STATE_VERSION: u8 = 1;
#[cfg(feature = "spill")]
const RDF_AGGREGATE_STATE_END: u8 = 0xa5;
#[cfg(feature = "spill")]
const RDF_AGGREGATE_STATE_HEADER_BYTES: usize = 4 + 1 + 8 + 4;

#[cfg(feature = "spill")]
struct RdfPayloadWriter {
    bytes: Vec<u8>,
    max_bytes: usize,
}

#[cfg(feature = "spill")]
impl RdfPayloadWriter {
    fn new(max_bytes: usize) -> std::io::Result<Self> {
        let mut bytes = Vec::new();
        bytes.try_reserve(max_bytes.min(64)).map_err(|error| {
            std::io::Error::other(format!(
                "cannot reserve RDF aggregate spill payload: {error}"
            ))
        })?;
        Ok(Self { bytes, max_bytes })
    }

    fn into_inner(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(feature = "spill")]
impl Write for RdfPayloadWriter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let encoded_length = self.bytes.len().checked_add(buffer.len()).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "RDF aggregate state payload length overflow",
            )
        })?;
        if encoded_length > self.max_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "RDF aggregate state payload exceeds codec byte limit {}",
                    self.max_bytes
                ),
            ));
        }
        self.bytes.try_reserve(buffer.len()).map_err(|error| {
            std::io::Error::other(format!("cannot grow RDF aggregate spill payload: {error}"))
        })?;
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(feature = "spill")]
impl RdfGroupState {
    fn encode_spill(
        &self,
        limits: grafeo_core::execution::spill::CodecLimits,
    ) -> std::io::Result<Vec<u8>> {
        let mut payload = RdfPayloadWriter::new(limits.max_bytes())?;
        let mut budget = SpillCodecEncodeBudget::new(limits);
        budget.charge_items::<Value>(self.key_values.len(), "RDF aggregate group key")?;
        write_count(
            &mut payload,
            self.key_values.len(),
            "RDF aggregate group keys",
        )?;
        for key in &self.key_values {
            budget.encode_value(key, &mut payload)?;
        }
        budget
            .charge_items::<RdfAccumulator>(self.accumulators.len(), "RDF aggregate accumulator")?;
        write_count(
            &mut payload,
            self.accumulators.len(),
            "RDF aggregate accumulators",
        )?;
        for accumulator in &self.accumulators {
            accumulator.write_spill(&mut payload, &mut budget)?;
        }
        payload.write_all(&[RDF_AGGREGATE_STATE_END])?;
        let payload = payload.into_inner();

        let payload_length = u64::try_from(payload.len()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "RDF aggregate state payload exceeds the spill format",
            )
        })?;
        let encoded_length = RDF_AGGREGATE_STATE_HEADER_BYTES
            .checked_add(payload.len())
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "RDF aggregate state encoded length overflow",
                )
            })?;
        let mut encoded = Vec::new();
        encoded.try_reserve_exact(encoded_length).map_err(|error| {
            std::io::Error::other(format!(
                "cannot reserve RDF aggregate spill record: {error}"
            ))
        })?;
        encoded.write_all(&RDF_AGGREGATE_STATE_MAGIC)?;
        encoded.write_all(&[RDF_AGGREGATE_STATE_VERSION])?;
        encoded.write_all(&payload_length.to_le_bytes())?;
        encoded.write_all(&rdf_aggregate_checksum(&payload).to_le_bytes())?;
        encoded.write_all(&payload)?;
        Ok(encoded)
    }

    fn decode_spill(
        encoded: &[u8],
        limits: grafeo_core::execution::spill::CodecLimits,
    ) -> std::io::Result<Self> {
        let mut header = Cursor::new(encoded);
        let mut magic = [0u8; 4];
        header.read_exact(&mut magic)?;
        if magic != RDF_AGGREGATE_STATE_MAGIC {
            return Err(invalid_spill("invalid RDF aggregate state magic"));
        }
        let version = read_byte(&mut header)?;
        if version != RDF_AGGREGATE_STATE_VERSION {
            return Err(invalid_spill(format!(
                "unsupported RDF aggregate state version {version}"
            )));
        }
        let mut payload_length = [0u8; 8];
        header.read_exact(&mut payload_length)?;
        let payload_length = usize::try_from(u64::from_le_bytes(payload_length)).map_err(|_| {
            invalid_spill("RDF aggregate state payload length does not fit this platform")
        })?;
        let mut checksum = [0u8; 4];
        header.read_exact(&mut checksum)?;
        let checksum = u32::from_le_bytes(checksum);
        let actual_payload_length = encoded
            .len()
            .checked_sub(RDF_AGGREGATE_STATE_HEADER_BYTES)
            .ok_or_else(|| invalid_spill("truncated RDF aggregate state header"))?;
        if payload_length != actual_payload_length {
            return Err(invalid_spill(format!(
                "RDF aggregate state payload length {payload_length} does not match {actual_payload_length} bytes"
            )));
        }
        if payload_length > limits.max_bytes() {
            return Err(invalid_spill(format!(
                "RDF aggregate state payload exceeds codec byte limit {}",
                limits.max_bytes()
            )));
        }
        let payload = &encoded[RDF_AGGREGATE_STATE_HEADER_BYTES..];
        if rdf_aggregate_checksum(payload) != checksum {
            return Err(invalid_spill("RDF aggregate state checksum mismatch"));
        }

        let mut reader = Cursor::new(payload);
        let mut budget = SpillCodecDecodeBudget::new(limits);
        let key_count = read_collection_count::<Value>(
            &mut reader,
            &mut budget,
            1,
            "RDF aggregate group keys",
        )?;
        let mut key_values = Vec::new();
        key_values.try_reserve_exact(key_count).map_err(|error| {
            std::io::Error::other(format!("cannot reserve RDF aggregate group keys: {error}"))
        })?;
        for _ in 0..key_count {
            key_values.push(budget.decode_value(&mut reader)?);
        }
        let accumulator_count = read_collection_count::<RdfAccumulator>(
            &mut reader,
            &mut budget,
            2,
            "RDF aggregate accumulators",
        )?;
        let mut accumulators = Vec::new();
        accumulators
            .try_reserve_exact(accumulator_count)
            .map_err(|error| {
                std::io::Error::other(format!(
                    "cannot reserve RDF aggregate accumulators: {error}"
                ))
            })?;
        for _ in 0..accumulator_count {
            accumulators.push(RdfAccumulator::read_spill(&mut reader, &mut budget)?);
        }
        if read_byte(&mut reader)? != RDF_AGGREGATE_STATE_END {
            return Err(invalid_spill("invalid RDF aggregate state terminator"));
        }
        if usize::try_from(reader.position()).ok() != Some(payload.len()) {
            return Err(invalid_spill("trailing RDF aggregate state bytes"));
        }
        Ok(Self {
            key_values,
            accumulators,
        })
    }

    #[cfg(test)]
    fn merge_from(&mut self, other: Self) -> std::io::Result<()> {
        if self.key_values.len() != other.key_values.len()
            || !self
                .key_values
                .iter()
                .zip(&other.key_values)
                .all(|(left, right)| {
                    HashableValue::from(left.clone()) == HashableValue::from(right.clone())
                })
        {
            return Err(invalid_spill(
                "cannot merge RDF aggregate states for different group keys",
            ));
        }
        if self.accumulators.len() != other.accumulators.len() {
            return Err(invalid_spill(
                "cannot merge RDF aggregate states with different accumulator counts",
            ));
        }
        for (left, right) in self.accumulators.iter_mut().zip(other.accumulators) {
            left.merge_from(right)?;
        }
        Ok(())
    }
}

#[cfg(feature = "spill")]
fn rdf_aggregate_checksum(bytes: &[u8]) -> u32 {
    let mut checksum = u32::MAX;
    for byte in bytes {
        checksum ^= u32::from(*byte);
        for _ in 0..8 {
            let low_bit = checksum & 1;
            checksum = (checksum >> 1) ^ (0xedb8_8320 & 0u32.wrapping_sub(low_bit));
        }
    }
    !checksum
}

#[cfg(feature = "spill")]
fn invalid_spill(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into())
}

#[cfg(feature = "spill")]
fn write_count(writer: &mut impl Write, count: usize, description: &str) -> std::io::Result<()> {
    let count = u32::try_from(count).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{description} exceed the spill format"),
        )
    })?;
    writer.write_all(&count.to_le_bytes())
}

#[cfg(feature = "spill")]
fn read_count(
    reader: &mut impl Read,
    max_count: usize,
    description: &str,
) -> std::io::Result<usize> {
    let mut count = [0u8; 4];
    reader.read_exact(&mut count)?;
    let count = usize::try_from(u32::from_le_bytes(count))
        .map_err(|_| invalid_spill(format!("{description} do not fit this platform")))?;
    if count > max_count {
        return Err(invalid_spill(format!(
            "{description} exceed their spill record"
        )));
    }
    Ok(count)
}

#[cfg(feature = "spill")]
fn remaining_spill_bytes(reader: &Cursor<&[u8]>) -> std::io::Result<usize> {
    let position = usize::try_from(reader.position())
        .map_err(|_| invalid_spill("RDF aggregate reader position does not fit this platform"))?;
    reader
        .get_ref()
        .len()
        .checked_sub(position)
        .ok_or_else(|| invalid_spill("RDF aggregate reader advanced beyond its payload"))
}

#[cfg(feature = "spill")]
fn read_collection_count<T>(
    reader: &mut Cursor<&[u8]>,
    budget: &mut SpillCodecDecodeBudget,
    minimum_item_bytes: usize,
    description: &str,
) -> std::io::Result<usize> {
    let count = read_count(reader, usize::MAX, description)?;
    let remaining = remaining_spill_bytes(reader)?;
    let wire_limit = remaining / minimum_item_bytes.max(1);
    if count > wire_limit {
        return Err(invalid_spill(format!(
            "{description} exceed the remaining spill record"
        )));
    }
    budget.charge_items::<T>(count, description)?;
    Ok(count)
}

#[cfg(feature = "spill")]
fn read_byte(reader: &mut impl Read) -> std::io::Result<u8> {
    let mut byte = [0u8; 1];
    reader.read_exact(&mut byte)?;
    Ok(byte[0])
}

#[cfg(feature = "spill")]
fn read_u64(reader: &mut impl Read) -> std::io::Result<u64> {
    let mut value = [0u8; 8];
    reader.read_exact(&mut value)?;
    Ok(u64::from_le_bytes(value))
}

#[cfg(feature = "spill")]
fn read_bool(reader: &mut impl Read, description: &str) -> std::io::Result<bool> {
    match read_byte(reader)? {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(invalid_spill(format!("invalid {description} flag {value}"))),
    }
}

#[cfg(feature = "spill")]
fn write_string(writer: &mut impl Write, value: &str) -> std::io::Result<()> {
    write_count(writer, value.len(), "RDF aggregate string bytes")?;
    writer.write_all(value.as_bytes())
}

#[cfg(feature = "spill")]
fn read_string(reader: &mut Cursor<&[u8]>, description: &str) -> std::io::Result<String> {
    let length = read_count(reader, usize::MAX, description)?;
    if length > remaining_spill_bytes(reader)? {
        return Err(invalid_spill(format!(
            "{description} exceeds the remaining spill record"
        )));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|error| std::io::Error::other(format!("cannot reserve {description}: {error}")))?;
    bytes.resize(length, 0);
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|error| invalid_spill(format!("{description} is not UTF-8: {error}")))
}

#[cfg(feature = "spill")]
fn write_optional_ordered_value(
    writer: &mut impl Write,
    value: Option<&RdfOrderedValue>,
    budget: &mut SpillCodecEncodeBudget,
) -> std::io::Result<()> {
    writer.write_all(&[u8::from(value.is_some())])?;
    if let Some(value) = value {
        writer.write_all(&value.ordinal.to_le_bytes())?;
        budget.encode_value(&value.value, writer)?;
    }
    Ok(())
}

#[cfg(feature = "spill")]
fn read_optional_ordered_value(
    reader: &mut Cursor<&[u8]>,
    budget: &mut SpillCodecDecodeBudget,
) -> std::io::Result<Option<RdfOrderedValue>> {
    match read_byte(reader)? {
        0 => Ok(None),
        1 => {
            let ordinal = read_u64(reader)?;
            Ok(Some(RdfOrderedValue {
                ordinal,
                value: budget.decode_value(reader)?,
            }))
        }
        value => Err(invalid_spill(format!(
            "invalid RDF aggregate ordered-value flag {value}"
        ))),
    }
}

#[cfg(feature = "spill")]
fn aggregate_function_tag(function: AggregateFunction) -> std::io::Result<u8> {
    match function {
        AggregateFunction::Count => Ok(0),
        AggregateFunction::CountNonNull => Ok(1),
        AggregateFunction::Sum => Ok(2),
        AggregateFunction::Avg => Ok(3),
        AggregateFunction::Min => Ok(4),
        AggregateFunction::Max => Ok(5),
        AggregateFunction::Sample => Ok(6),
        AggregateFunction::GroupConcat => Ok(7),
        _ => Err(invalid_spill(
            "unsupported non-SPARQL aggregate function cannot be spilled",
        )),
    }
}

#[cfg(feature = "spill")]
fn aggregate_function_from_tag(tag: u8) -> std::io::Result<AggregateFunction> {
    match tag {
        0 => Ok(AggregateFunction::Count),
        1 => Ok(AggregateFunction::CountNonNull),
        2 => Ok(AggregateFunction::Sum),
        3 => Ok(AggregateFunction::Avg),
        4 => Ok(AggregateFunction::Min),
        5 => Ok(AggregateFunction::Max),
        6 => Ok(AggregateFunction::Sample),
        7 => Ok(AggregateFunction::GroupConcat),
        _ => Err(invalid_spill(format!(
            "unknown RDF aggregate function tag {tag}"
        ))),
    }
}

impl RdfAggregateOperator {
    pub(super) fn new(
        child: Box<dyn Operator>,
        group_columns: Vec<usize>,
        aggregates: Vec<AggregateExpr>,
        output_schema: Vec<LogicalType>,
        row_identity_columns: Option<Vec<RdfRowIdentityColumn>>,
    ) -> Self {
        Self {
            child,
            group_columns,
            aggregates,
            output_schema,
            row_identity_columns,
            groups: IndexMap::new(),
            results: None,
            next_ordinal: 0,
            admission: None,
            resources: None,
            failed: false,
            #[cfg(feature = "spill")]
            spilled: None,
            #[cfg(feature = "spill")]
            shared_input: None,
        }
    }

    fn new_group(&self, key_values: Vec<Value>) -> RdfGroupState {
        RdfGroupState {
            key_values,
            accumulators: self.aggregates.iter().map(RdfAccumulator::new).collect(),
        }
    }

    fn aggregate(&mut self) -> Result<(), OperatorError> {
        while let Some(chunk) = self.child.next()? {
            for logical_row in 0..chunk.row_count() {
                let row = chunk.selection().map_or(logical_row, |selection| {
                    usize::from(selection.as_slice()[logical_row])
                });
                let input_bytes = match &mut self.admission {
                    Some(admission) => admission.prepare_row(&chunk, row)?,
                    None => 0,
                };
                let ordinal = self.next_ordinal;
                self.next_ordinal = self.next_ordinal.checked_add(1).ok_or_else(|| {
                    OperatorError::Execution("RDF aggregate input ordinal overflow".to_string())
                })?;
                let key_values = self
                    .group_columns
                    .iter()
                    .map(|&column| {
                        chunk
                            .column(column)
                            .and_then(|values| values.get_value(row))
                            .unwrap_or(Value::Null)
                    })
                    .collect::<Vec<_>>();
                let key = key_values
                    .iter()
                    .cloned()
                    .map(HashableValue::from)
                    .collect::<Vec<_>>();
                let new_group = !self.groups.contains_key(&key);
                if let Some(admission) = &mut self.admission {
                    let mut key_bytes = 0;
                    for value in &key_values {
                        key_bytes = bounded::add(key_bytes, bounded::value_bytes(value)?)?;
                    }
                    let growth =
                        bounded::group_growth(new_group, key_bytes, input_bytes, &self.aggregates)?;
                    #[cfg(feature = "spill")]
                    if admission.resources.has_spill_manager()
                        && bounded::add(admission.state.size(), growth)?
                            > admission.resources.query_stats().growth_limit_bytes / 8
                    {
                        admission.release_scratch()?;
                        self.spilled = Some(spill::start(self, chunk, logical_row, ordinal)?);
                        return Ok(());
                    }
                    #[cfg(not(feature = "spill"))]
                    let _ = logical_row;
                    admission.grow(growth)?;
                }
                if new_group {
                    let group = self.new_group(key_values);
                    self.groups.insert(key.clone(), group);
                } else {
                    drop(key_values);
                }
                let group = self
                    .groups
                    .get_mut(&key)
                    .expect("the RDF aggregate group was inserted above");

                if let Some(admission) = &mut self.admission {
                    admission.set_scratch(bounded::group_update_peak(group, input_bytes)?)?;
                }
                let snapshot = if self.admission.is_some() {
                    Some(bounded::snapshot_group(group)?)
                } else {
                    None
                };
                let row_identity = self.row_identity_columns.as_ref().map(|columns| {
                    Value::List(
                        columns
                            .iter()
                            .map(|column| {
                                let visible = chunk
                                    .column(column.visible)
                                    .and_then(|values| values.get_value(row))
                                    .unwrap_or(Value::Null);
                                let canonical = column
                                    .canonical_rdf
                                    .and_then(|index| chunk.column(index))
                                    .and_then(|values| values.get_value(row))
                                    .filter(|value| !value.is_null());
                                let (canonical_rdf, identity) =
                                    canonical.map_or((false, visible), |identity| (true, identity));
                                Value::List(vec![Value::Bool(canonical_rdf), identity].into())
                            })
                            .collect::<Vec<_>>()
                            .into(),
                    )
                });
                for (index, (accumulator, aggregate)) in group
                    .accumulators
                    .iter_mut()
                    .zip(&self.aggregates)
                    .enumerate()
                {
                    let value = aggregate.column.and_then(|column| {
                        chunk
                            .column(column)
                            .and_then(|values| values.get_value(row))
                    });
                    let distinct_key = aggregate.distinct_key_column.and_then(|column| {
                        chunk
                            .column(column)
                            .and_then(|values| values.get_value(row))
                    });
                    accumulator.update_at(
                        value,
                        distinct_key,
                        aggregate
                            .column
                            .is_none()
                            .then(|| row_identity.clone())
                            .flatten(),
                        ordinal,
                    )?;

                    debug_assert_eq!(
                        accumulator.function, aggregate.function,
                        "RDF accumulator {index} no longer matches its expression"
                    );
                }
                let delta = snapshot
                    .map(|snapshot| bounded::updated_group_delta(group, snapshot))
                    .transpose()?;
                drop(row_identity);
                drop(key);
                if let (Some(admission), Some((previous_bytes, current_bytes))) =
                    (&mut self.admission, delta)
                {
                    let base = admission.state.size().checked_sub(previous_bytes).ok_or(
                        OperatorError::ResidentContainerInvariant {
                            container: "RDF aggregate accounting",
                            message: "group bound exceeded retained reservation",
                        },
                    )?;
                    admission.adopt_updated_state(bounded::add(base, current_bytes)?)?;
                }
            }
        }

        // An implicit empty group produces one solution. Explicit grouping
        // has no keys and therefore no groups when its input is empty.
        if self.groups.is_empty() && self.group_columns.is_empty() {
            if let Some(admission) = &mut self.admission {
                admission.grow(bounded::group_growth(true, 0, 0, &self.aggregates)?)?;
            }
            let key_values = vec![Value::Null; self.group_columns.len()];
            let key = key_values
                .iter()
                .cloned()
                .map(HashableValue::from)
                .collect();
            self.groups.insert(key, self.new_group(key_values));
        }

        self.results = Some(std::mem::take(&mut self.groups).into_iter());
        if let Some(admission) = &mut self.admission {
            admission.release_scratch()?;
        }
        Ok(())
    }
}

impl RdfAggregateOperator {
    fn next_inner(&mut self) -> OperatorResult {
        #[cfg(feature = "spill")]
        if let Some(spilled) = &mut self.spilled {
            return spilled.next();
        }
        if self.results.is_none() {
            self.aggregate()?;
        }
        #[cfg(feature = "spill")]
        if let Some(spilled) = &mut self.spilled {
            return spilled.next();
        }
        let results = self
            .results
            .as_mut()
            .expect("RDF aggregate results are initialized above");
        let mut output_rows = results.len().min(OUTPUT_CHUNK_SIZE);
        if output_rows == 0 {
            self.results = Some(IndexMap::new().into_iter());
            self.admission = None;
            return Ok(None);
        }
        if let Some(admission) = &mut self.admission {
            loop {
                let mut bytes = bounded::mul(
                    output_rows.max(1),
                    bounded::mul(self.output_schema.len(), std::mem::size_of::<Value>() * 4)?,
                )?;
                bytes = bounded::add(bytes, bounded::mul(self.output_schema.len(), 512)?)?;
                for (_, group) in results.as_slice().iter().take(output_rows) {
                    bytes = bounded::add(bytes, bounded::group_finalize_peak(group)?)?;
                }
                match admission.output(bytes) {
                    Ok(()) => break,
                    Err(error) if output_rows <= 1 => return Err(error),
                    Err(_) => output_rows = (output_rows / 2).max(1),
                }
            }
        }
        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, output_rows.max(1));

        for (_, group) in results.take(output_rows) {
            for (column, value) in group.key_values.into_iter().enumerate() {
                builder
                    .column_mut(column)
                    .expect("group output schema matches the group columns")
                    .push_value(value);
            }
            let offset = self.group_columns.len();
            for (index, accumulator) in group.accumulators.iter().enumerate() {
                builder
                    .column_mut(offset + index)
                    .expect("aggregate output schema matches the accumulators")
                    .push_value(accumulator.finalize());
            }
            builder.advance_row();
        }

        if builder.row_count() == 0 {
            self.results = Some(IndexMap::new().into_iter());
            self.admission = None;
            Ok(None)
        } else {
            Ok(Some(builder.finish()))
        }
    }
}

impl Operator for RdfAggregateOperator {
    fn next(&mut self) -> OperatorResult {
        if self.failed {
            return Err(OperatorError::ResidentContainerInvariant {
                container: "RDF aggregate",
                message: "failed aggregate cannot resume",
            });
        }
        let result = self.next_inner();
        if result.is_err() {
            self.failed = true;
            self.groups = IndexMap::new();
            self.results = None;
            #[cfg(feature = "spill")]
            {
                self.spilled = None;
            }
            self.admission = None;
        }
        result
    }

    fn reset(&mut self) {
        self.groups = IndexMap::new();
        self.results = None;
        #[cfg(feature = "spill")]
        {
            self.spilled = None;
        }
        self.admission = None;
        self.child.reset();
        self.next_ordinal = 0;
        self.failed = false;
        if let Some(resources) = &self.resources {
            match bounded::Admission::new(resources) {
                Ok(admission) => self.admission = Some(admission),
                Err(_) => self.failed = true,
            }
        }
    }

    fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> Result<(), QueryResourceContextError> {
        self.child.install_resource_context(resources)?;
        self.admission = Some(bounded::Admission::new(resources)?);
        self.resources = Some(resources.clone());
        self.failed = false;
        Ok(())
    }

    fn name(&self) -> &'static str {
        // A distinct name prevents generic pipeline conversion from silently
        // routing SPARQL state through SQL/GQL push and spill accumulators.
        "RdfInMemoryAggregate"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

struct RdfAccumulator {
    function: AggregateFunction,
    distinct: Option<RdfDistinctState>,
    state: RdfSetFunctionState,
}

struct RdfDistinctState {
    entries: IndexMap<HashableValue, RdfDistinctEntry>,
    unkeyed_error: bool,
}

struct RdfDistinctEntry {
    #[cfg_attr(not(feature = "spill"), allow(dead_code))]
    key: Value,
    operand: Value,
    ordinal: u64,
}

enum RdfSetFunctionState {
    Count(i64),
    Generic(AggregateState),
    Sum(RdfNumericState),
    Average(RdfNumericState),
    Minimum {
        selected: Option<RdfOrderedValue>,
        saw_error: bool,
    },
    Maximum(Option<RdfOrderedValue>),
    Sample(Option<RdfOrderedValue>),
    GroupConcat {
        values: Vec<RdfOrderedString>,
        separator: String,
        error: bool,
    },
}

struct RdfOrderedValue {
    ordinal: u64,
    value: Value,
}

struct RdfOrderedString {
    ordinal: u64,
    value: String,
}

impl RdfAccumulator {
    fn new(expression: &AggregateExpr) -> Self {
        let state = match expression.function {
            AggregateFunction::Count | AggregateFunction::CountNonNull => {
                RdfSetFunctionState::Count(0)
            }
            AggregateFunction::Sum => RdfSetFunctionState::Sum(RdfNumericState::default()),
            AggregateFunction::Avg => RdfSetFunctionState::Average(RdfNumericState::default()),
            AggregateFunction::Min => RdfSetFunctionState::Minimum {
                selected: None,
                saw_error: false,
            },
            AggregateFunction::Max => RdfSetFunctionState::Maximum(None),
            AggregateFunction::Sample => RdfSetFunctionState::Sample(None),
            AggregateFunction::GroupConcat => RdfSetFunctionState::GroupConcat {
                values: Vec::new(),
                separator: expression.separator.as_deref().unwrap_or(" ").to_string(),
                error: false,
            },
            function => RdfSetFunctionState::Generic(AggregateState::new(
                function,
                false,
                expression.percentile,
                expression.separator.as_deref(),
            )),
        };
        Self {
            function: expression.function,
            distinct: expression.distinct.then(|| RdfDistinctState {
                entries: IndexMap::new(),
                unkeyed_error: false,
            }),
            state,
        }
    }

    fn update_at(
        &mut self,
        value: Option<Value>,
        distinct_key: Option<Value>,
        row_identity: Option<Value>,
        ordinal: u64,
    ) -> Result<(), OperatorError> {
        let value = value.unwrap_or(Value::Null);

        // Errors are retained by SPARQL ListEval. They are not members of an
        // expression DISTINCT set and must reach set functions that propagate
        // them. COUNT(DISTINCT *) is different: its null aggregate operand is
        // synthetic, while its logical solution-mapping identity is real.
        let mut rebuild = false;
        if let Some(distinct) = &mut self.distinct {
            let key = if let Some(row_identity) = row_identity {
                Some(row_identity)
            } else if value.is_null() {
                None
            } else {
                Some(distinct_key.unwrap_or_else(|| value.clone()))
            };
            if let Some(key) = key {
                let identity = HashableValue::from(key.clone());
                if let Some(existing) = distinct.entries.get(&identity) {
                    if existing.ordinal <= ordinal {
                        return Ok(());
                    }
                    rebuild = true;
                } else {
                    distinct.entries.try_reserve(1).map_err(|error| {
                        OperatorError::ResidentAllocation(format!(
                            "cannot reserve RDF DISTINCT entry: {error}"
                        ))
                    })?;
                }
                distinct.entries.insert(
                    identity,
                    RdfDistinctEntry {
                        key,
                        operand: value.clone(),
                        ordinal,
                    },
                );
            } else if matches!(
                self.function,
                AggregateFunction::Sum
                    | AggregateFunction::Avg
                    | AggregateFunction::Min
                    | AggregateFunction::GroupConcat
            ) {
                distinct.unkeyed_error = true;
            }
        }
        if rebuild {
            return self.rebuild_distinct_state();
        }

        self.apply_value_at(value, ordinal)
    }

    fn apply_value_at(&mut self, value: Value, ordinal: u64) -> Result<(), OperatorError> {
        match &mut self.state {
            RdfSetFunctionState::Count(count) => match self.function {
                AggregateFunction::Count => {
                    *count = count.checked_add(1).ok_or_else(|| {
                        OperatorError::Execution("RDF COUNT state overflow".to_string())
                    })?;
                }
                AggregateFunction::CountNonNull if !value.is_null() => {
                    *count = count.checked_add(1).ok_or_else(|| {
                        OperatorError::Execution("RDF COUNT state overflow".to_string())
                    })?;
                }
                AggregateFunction::CountNonNull => {}
                _ => debug_assert!(false, "RDF count state has a non-count function"),
            },
            RdfSetFunctionState::Sum(state) | RdfSetFunctionState::Average(state) => {
                state.update_at(&value, ordinal)?;
            }
            RdfSetFunctionState::Minimum {
                selected,
                saw_error,
            } => {
                if value.is_null() {
                    // ListEval represents both an unbound operand and an
                    // expression failure as an error. MIN is the first member
                    // of ascending SPARQL order, where errors are lowest.
                    *saw_error = true;
                } else if selected.as_ref().is_none_or(|current| {
                    match rdf_aggregate_compare(&value, &current.value) {
                        Some(Ordering::Less) => true,
                        Some(Ordering::Equal) | None => ordinal < current.ordinal,
                        Some(Ordering::Greater) => false,
                    }
                }) {
                    *selected = Some(RdfOrderedValue { ordinal, value });
                }
            }
            RdfSetFunctionState::Maximum(maximum) => {
                if !value.is_null()
                    && maximum.as_ref().is_none_or(|current| {
                        match rdf_aggregate_compare(&value, &current.value) {
                            Some(Ordering::Greater) => true,
                            Some(Ordering::Equal) | None => ordinal < current.ordinal,
                            Some(Ordering::Less) => false,
                        }
                    })
                {
                    *maximum = Some(RdfOrderedValue { ordinal, value });
                }
            }
            RdfSetFunctionState::Sample(sample) => {
                if !value.is_null()
                    && sample
                        .as_ref()
                        .is_none_or(|selected| ordinal < selected.ordinal)
                {
                    *sample = Some(RdfOrderedValue { ordinal, value });
                }
            }
            RdfSetFunctionState::GroupConcat { values, error, .. } => {
                match rdf_aggregate_string(&value) {
                    Some(value) => {
                        if values
                            .last()
                            .is_some_and(|previous| previous.ordinal >= ordinal)
                        {
                            return Err(OperatorError::Execution(
                                "RDF GROUP_CONCAT ordinals are not strictly increasing".to_string(),
                            ));
                        }
                        values.try_reserve(1).map_err(|error| {
                            OperatorError::ResidentAllocation(format!(
                                "cannot reserve RDF GROUP_CONCAT value: {error}"
                            ))
                        })?;
                        values.push(RdfOrderedString { ordinal, value });
                    }
                    None => *error = true,
                }
            }
            RdfSetFunctionState::Generic(state) => match self.function {
                AggregateFunction::Count => state.update(None),
                AggregateFunction::CountNonNull if !value.is_null() => {
                    state.update(Some(value));
                }
                AggregateFunction::CountNonNull => {}
                _ if !value.is_null() => state.update(Some(value)),
                _ => {}
            },
        }
        Ok(())
    }

    fn rebuild_distinct_state(&mut self) -> Result<(), OperatorError> {
        let separator = match &mut self.state {
            RdfSetFunctionState::GroupConcat { separator, .. } => Some(std::mem::take(separator)),
            _ => None,
        };
        let distinct = self.distinct.as_mut().ok_or_else(|| {
            OperatorError::Execution(
                "cannot rebuild a non-DISTINCT RDF aggregate state".to_string(),
            )
        })?;
        distinct
            .entries
            .sort_unstable_by_key(|_, entry| entry.ordinal);
        let unkeyed_error = distinct.unkeyed_error;
        let entry_count = distinct.entries.len();
        let mut previous_ordinal = None;
        for entry in distinct.entries.values() {
            if previous_ordinal == Some(entry.ordinal) {
                return Err(OperatorError::Execution(
                    "duplicate RDF DISTINCT encounter ordinal".to_string(),
                ));
            }
            previous_ordinal = Some(entry.ordinal);
        }
        self.state = empty_rdf_set_function_state(self.function, separator)?;
        for index in 0..entry_count {
            let (ordinal, operand) = self
                .distinct
                .as_ref()
                .and_then(|distinct| distinct.entries.get_index(index))
                .map(|(_, entry)| (entry.ordinal, entry.operand.clone()))
                .ok_or_else(|| {
                    OperatorError::Execution("RDF DISTINCT replay entry disappeared".to_string())
                })?;
            self.apply_value_at(operand, ordinal)?;
        }
        if unkeyed_error {
            self.apply_value_at(Value::Null, u64::MAX)?;
        }
        Ok(())
    }

    #[cfg(feature = "spill")]
    fn merge_from(&mut self, other: Self) -> std::io::Result<()> {
        if self.function != other.function {
            return Err(invalid_spill(
                "cannot merge different RDF aggregate functions",
            ));
        }
        let other_state = other.state;
        match (self.distinct.take(), other.distinct) {
            (Some(mut left), Some(right)) => {
                let left_separator = validate_rdf_set_function_state(self.function, &self.state)?;
                let right_separator = validate_rdf_set_function_state(self.function, &other_state)?;
                if left_separator != right_separator {
                    return Err(invalid_spill(
                        "cannot merge RDF DISTINCT states with different configuration",
                    ));
                }
                left.entries
                    .try_reserve(right.entries.len())
                    .map_err(|error| {
                        std::io::Error::other(format!(
                            "cannot reserve merged RDF DISTINCT entries: {error}"
                        ))
                    })?;
                for (identity, candidate) in right.entries {
                    if let Some(existing) = left.entries.get_mut(&identity) {
                        if candidate.ordinal < existing.ordinal {
                            *existing = candidate;
                        } else if candidate.ordinal == existing.ordinal
                            && (candidate.key != existing.key
                                || candidate.operand != existing.operand)
                        {
                            return Err(invalid_spill(
                                "conflicting RDF DISTINCT entries share an encounter ordinal",
                            ));
                        }
                    } else {
                        left.entries.insert(identity, candidate);
                    }
                }
                left.unkeyed_error |= right.unkeyed_error;
                self.distinct = Some(left);
                return self
                    .rebuild_distinct_state()
                    .map_err(|error| invalid_spill(error.to_string()));
            }
            (None, None) => {}
            (left, right) => {
                self.distinct = left;
                let _ = right;
                return Err(invalid_spill(
                    "cannot merge DISTINCT and non-DISTINCT RDF aggregate states",
                ));
            }
        }
        match (&mut self.state, other_state) {
            (RdfSetFunctionState::Count(left), RdfSetFunctionState::Count(right)) => {
                *left = left
                    .checked_add(right)
                    .ok_or_else(|| invalid_spill("RDF COUNT merge overflow"))?;
                Ok(())
            }
            (RdfSetFunctionState::Sum(left), RdfSetFunctionState::Sum(right))
            | (RdfSetFunctionState::Average(left), RdfSetFunctionState::Average(right)) => {
                left.merge_from(right)
            }
            (
                RdfSetFunctionState::Minimum {
                    selected: left,
                    saw_error: left_error,
                },
                RdfSetFunctionState::Minimum {
                    selected: right,
                    saw_error: right_error,
                },
            ) => {
                *left_error |= right_error;
                if let Some(right) = right
                    && left.as_ref().is_none_or(|current| {
                        match rdf_aggregate_compare(&right.value, &current.value) {
                            Some(Ordering::Less) => true,
                            Some(Ordering::Equal) | None => right.ordinal < current.ordinal,
                            Some(Ordering::Greater) => false,
                        }
                    })
                {
                    *left = Some(right);
                }
                Ok(())
            }
            (RdfSetFunctionState::Maximum(left), RdfSetFunctionState::Maximum(right)) => {
                if let Some(right) = right
                    && left.as_ref().is_none_or(|current| {
                        match rdf_aggregate_compare(&right.value, &current.value) {
                            Some(Ordering::Greater) => true,
                            Some(Ordering::Equal) | None => right.ordinal < current.ordinal,
                            Some(Ordering::Less) => false,
                        }
                    })
                {
                    *left = Some(right);
                }
                Ok(())
            }
            (RdfSetFunctionState::Sample(left), RdfSetFunctionState::Sample(right)) => {
                if let Some(right) = right
                    && left
                        .as_ref()
                        .is_none_or(|current| right.ordinal < current.ordinal)
                {
                    *left = Some(right);
                }
                Ok(())
            }
            (
                RdfSetFunctionState::GroupConcat {
                    values: left,
                    separator: left_separator,
                    error: left_error,
                },
                RdfSetFunctionState::GroupConcat {
                    values: mut right,
                    separator: right_separator,
                    error: right_error,
                },
            ) => {
                if *left_separator != right_separator {
                    return Err(invalid_spill(
                        "cannot merge RDF GROUP_CONCAT states with different separators",
                    ));
                }
                left.try_reserve(right.len()).map_err(|error| {
                    std::io::Error::other(format!(
                        "cannot reserve merged RDF GROUP_CONCAT values: {error}"
                    ))
                })?;
                left.append(&mut right);
                left.sort_unstable_by_key(|value| value.ordinal);
                if left
                    .windows(2)
                    .any(|values| values[0].ordinal == values[1].ordinal)
                {
                    return Err(invalid_spill(
                        "duplicate RDF GROUP_CONCAT ordinal during state merge",
                    ));
                }
                *left_error |= right_error;
                Ok(())
            }
            (RdfSetFunctionState::Generic(_), RdfSetFunctionState::Generic(_)) => Err(
                invalid_spill("unsupported non-SPARQL aggregate state cannot be merged"),
            ),
            _ => Err(invalid_spill(
                "RDF aggregate function has mismatched merge state",
            )),
        }
    }

    fn finalize(&self) -> Value {
        match &self.state {
            RdfSetFunctionState::Count(count) => Value::Int64(*count),
            RdfSetFunctionState::Sum(state) => state.finalize(false),
            RdfSetFunctionState::Average(state) => state.finalize(true),
            RdfSetFunctionState::Minimum {
                selected,
                saw_error,
            } => {
                if *saw_error {
                    Value::Null
                } else {
                    selected
                        .as_ref()
                        .map_or(Value::Null, |selected| selected.value.clone())
                }
            }
            RdfSetFunctionState::Maximum(value) => value
                .as_ref()
                .map_or(Value::Null, |selected| selected.value.clone()),
            RdfSetFunctionState::Sample(value) => value
                .as_ref()
                .map_or(Value::Null, |selected| selected.value.clone()),
            RdfSetFunctionState::GroupConcat {
                values,
                separator,
                error,
            } => {
                if *error {
                    Value::Null
                } else {
                    Value::String(
                        values
                            .iter()
                            .map(|value| value.value.as_str())
                            .collect::<Vec<_>>()
                            .join(separator)
                            .into(),
                    )
                }
            }
            RdfSetFunctionState::Generic(state) => state.finalize(),
        }
    }

    #[cfg(feature = "spill")]
    fn write_spill(
        &self,
        writer: &mut impl Write,
        budget: &mut SpillCodecEncodeBudget,
    ) -> std::io::Result<()> {
        let function = aggregate_function_tag(self.function)?;
        writer.write_all(&[function, u8::from(self.distinct.is_some())])?;
        if let Some(distinct) = &self.distinct {
            let separator = validate_rdf_set_function_state(self.function, &self.state)?;
            if let Some(separator) = separator {
                write_string(writer, separator)?;
            }
            writer.write_all(&[u8::from(distinct.unkeyed_error)])?;
            budget
                .charge_items::<RdfDistinctEntry>(distinct.entries.len(), "RDF DISTINCT entry")?;
            match self.function {
                AggregateFunction::Sum | AggregateFunction::Avg => budget
                    .charge_items::<RdfNumericInput>(
                        distinct.entries.len(),
                        "RDF DISTINCT numeric replay value",
                    )?,
                AggregateFunction::GroupConcat => budget.charge_items::<RdfOrderedString>(
                    distinct.entries.len(),
                    "RDF DISTINCT string replay value",
                )?,
                _ => {}
            }
            let mut entries = Vec::new();
            entries
                .try_reserve_exact(distinct.entries.len())
                .map_err(|error| {
                    std::io::Error::other(format!(
                        "cannot reserve ordered RDF DISTINCT entries: {error}"
                    ))
                })?;
            entries.extend(distinct.entries.iter());
            entries.sort_unstable_by_key(|(_, entry)| entry.ordinal);
            if entries
                .windows(2)
                .any(|entries| entries[0].1.ordinal == entries[1].1.ordinal)
            {
                return Err(invalid_spill("duplicate RDF DISTINCT encounter ordinal"));
            }
            write_count(writer, entries.len(), "RDF DISTINCT entries")?;
            for (identity, entry) in entries {
                if *identity != HashableValue::from(entry.key.clone()) {
                    return Err(invalid_spill(
                        "RDF DISTINCT identity does not match its encoded key",
                    ));
                }
                writer.write_all(&entry.ordinal.to_le_bytes())?;
                budget.encode_value(&entry.key, writer)?;
                budget.encode_value(&entry.operand, writer)?;
            }
            return Ok(());
        }
        match (self.function, &self.state) {
            (
                AggregateFunction::Count | AggregateFunction::CountNonNull,
                RdfSetFunctionState::Count(count),
            ) => {
                if *count < 0 {
                    return Err(invalid_spill("RDF COUNT state cannot be negative"));
                }
                writer.write_all(&count.to_le_bytes())?;
            }
            (AggregateFunction::Sum, RdfSetFunctionState::Sum(state))
            | (AggregateFunction::Avg, RdfSetFunctionState::Average(state)) => {
                state.write_spill(writer, budget)?;
            }
            (
                AggregateFunction::Min,
                RdfSetFunctionState::Minimum {
                    selected,
                    saw_error,
                },
            ) => {
                write_optional_ordered_value(writer, selected.as_ref(), budget)?;
                writer.write_all(&[u8::from(*saw_error)])?;
            }
            (AggregateFunction::Max, RdfSetFunctionState::Maximum(value)) => {
                write_optional_ordered_value(writer, value.as_ref(), budget)?;
            }
            (AggregateFunction::Sample, RdfSetFunctionState::Sample(value)) => {
                write_optional_ordered_value(writer, value.as_ref(), budget)?;
            }
            (
                AggregateFunction::GroupConcat,
                RdfSetFunctionState::GroupConcat {
                    values,
                    separator,
                    error,
                },
            ) => {
                if values
                    .windows(2)
                    .any(|values| values[0].ordinal >= values[1].ordinal)
                {
                    return Err(invalid_spill(
                        "RDF GROUP_CONCAT ordinals are not strictly increasing",
                    ));
                }
                write_string(writer, separator)?;
                writer.write_all(&[u8::from(*error)])?;
                budget.charge_items::<RdfOrderedString>(values.len(), "RDF GROUP_CONCAT value")?;
                write_count(writer, values.len(), "RDF GROUP_CONCAT values")?;
                for value in values {
                    writer.write_all(&value.ordinal.to_le_bytes())?;
                    write_string(writer, &value.value)?;
                }
            }
            (_, RdfSetFunctionState::Generic(_)) => {
                return Err(invalid_spill(
                    "unsupported non-SPARQL aggregate state cannot be spilled",
                ));
            }
            _ => {
                return Err(invalid_spill(format!(
                    "RDF aggregate function tag {function} has mismatched live state"
                )));
            }
        }
        Ok(())
    }

    #[cfg(feature = "spill")]
    fn read_spill(
        reader: &mut Cursor<&[u8]>,
        budget: &mut SpillCodecDecodeBudget,
    ) -> std::io::Result<Self> {
        let function_tag = read_byte(reader)?;
        let function = aggregate_function_from_tag(function_tag)?;
        let distinct = read_bool(reader, "RDF DISTINCT")?;
        if distinct {
            let separator = if function == AggregateFunction::GroupConcat {
                Some(read_string(reader, "RDF GROUP_CONCAT separator")?)
            } else {
                None
            };
            let unkeyed_error = read_bool(reader, "RDF DISTINCT unkeyed error")?;
            let count = read_collection_count::<RdfDistinctEntry>(
                reader,
                budget,
                10,
                "RDF DISTINCT entries",
            )?;
            match function {
                AggregateFunction::Sum | AggregateFunction::Avg => budget
                    .charge_items::<RdfNumericInput>(count, "RDF DISTINCT numeric replay value")?,
                AggregateFunction::GroupConcat => budget
                    .charge_items::<RdfOrderedString>(count, "RDF DISTINCT string replay value")?,
                _ => {}
            }
            let mut entries = IndexMap::new();
            entries.try_reserve(count).map_err(|error| {
                std::io::Error::other(format!("cannot reserve RDF DISTINCT entries: {error}"))
            })?;
            let mut previous_ordinal = None;
            for _ in 0..count {
                let ordinal = read_u64(reader)?;
                if previous_ordinal.is_some_and(|previous| previous >= ordinal) {
                    return Err(invalid_spill(
                        "RDF DISTINCT ordinals are not strictly increasing",
                    ));
                }
                previous_ordinal = Some(ordinal);
                let key = budget.decode_value(reader)?;
                let operand = budget.decode_value(reader)?;
                let identity = HashableValue::from(key.clone());
                if entries.contains_key(&identity) {
                    return Err(invalid_spill(
                        "duplicate RDF DISTINCT identity in spill state",
                    ));
                }
                entries.insert(
                    identity,
                    RdfDistinctEntry {
                        key,
                        operand,
                        ordinal,
                    },
                );
            }
            let state = empty_rdf_set_function_state(function, separator)
                .map_err(|error| invalid_spill(error.to_string()))?;
            let mut accumulator = Self {
                function,
                distinct: Some(RdfDistinctState {
                    entries,
                    unkeyed_error,
                }),
                state,
            };
            accumulator
                .rebuild_distinct_state()
                .map_err(|error| invalid_spill(error.to_string()))?;
            return Ok(accumulator);
        }
        let state = match function {
            AggregateFunction::Count | AggregateFunction::CountNonNull => {
                let mut count = [0u8; 8];
                reader.read_exact(&mut count)?;
                let count = i64::from_le_bytes(count);
                if count < 0 {
                    return Err(invalid_spill("RDF COUNT state cannot be negative"));
                }
                RdfSetFunctionState::Count(count)
            }
            AggregateFunction::Sum | AggregateFunction::Avg => {
                let state = RdfNumericState::read_spill(reader, budget)?;
                if function == AggregateFunction::Sum {
                    RdfSetFunctionState::Sum(state)
                } else {
                    RdfSetFunctionState::Average(state)
                }
            }
            AggregateFunction::Min => RdfSetFunctionState::Minimum {
                selected: read_optional_ordered_value(reader, budget)?,
                saw_error: read_bool(reader, "RDF MIN error")?,
            },
            AggregateFunction::Max => {
                RdfSetFunctionState::Maximum(read_optional_ordered_value(reader, budget)?)
            }
            AggregateFunction::Sample => {
                RdfSetFunctionState::Sample(read_optional_ordered_value(reader, budget)?)
            }
            AggregateFunction::GroupConcat => {
                let separator = read_string(reader, "RDF GROUP_CONCAT separator")?;
                let error = read_bool(reader, "RDF GROUP_CONCAT error")?;
                let count = read_collection_count::<RdfOrderedString>(
                    reader,
                    budget,
                    12,
                    "RDF GROUP_CONCAT values",
                )?;
                let mut values = Vec::new();
                values.try_reserve_exact(count).map_err(|error| {
                    std::io::Error::other(format!(
                        "cannot reserve RDF GROUP_CONCAT spill values: {error}"
                    ))
                })?;
                for _ in 0..count {
                    let ordinal = read_u64(reader)?;
                    if values
                        .last()
                        .is_some_and(|value: &RdfOrderedString| value.ordinal >= ordinal)
                    {
                        return Err(invalid_spill(
                            "RDF GROUP_CONCAT ordinals are not strictly increasing",
                        ));
                    }
                    values.push(RdfOrderedString {
                        ordinal,
                        value: read_string(reader, "RDF GROUP_CONCAT value")?,
                    });
                }
                RdfSetFunctionState::GroupConcat {
                    values,
                    separator,
                    error,
                }
            }
            _ => {
                return Err(invalid_spill(format!(
                    "unsupported RDF aggregate function tag {function_tag}"
                )));
            }
        };
        Ok(Self {
            function,
            distinct: None,
            state,
        })
    }
}

fn empty_rdf_set_function_state(
    function: AggregateFunction,
    separator: Option<String>,
) -> Result<RdfSetFunctionState, OperatorError> {
    match function {
        AggregateFunction::Count | AggregateFunction::CountNonNull => {
            Ok(RdfSetFunctionState::Count(0))
        }
        AggregateFunction::Sum => Ok(RdfSetFunctionState::Sum(RdfNumericState::default())),
        AggregateFunction::Avg => Ok(RdfSetFunctionState::Average(RdfNumericState::default())),
        AggregateFunction::Min => Ok(RdfSetFunctionState::Minimum {
            selected: None,
            saw_error: false,
        }),
        AggregateFunction::Max => Ok(RdfSetFunctionState::Maximum(None)),
        AggregateFunction::Sample => Ok(RdfSetFunctionState::Sample(None)),
        AggregateFunction::GroupConcat => Ok(RdfSetFunctionState::GroupConcat {
            values: Vec::new(),
            separator: separator.unwrap_or_else(|| " ".to_string()),
            error: false,
        }),
        _ => Err(OperatorError::Execution(
            "unsupported non-SPARQL aggregate state cannot be resumed".to_string(),
        )),
    }
}

#[cfg(feature = "spill")]
fn validate_rdf_set_function_state(
    function: AggregateFunction,
    state: &RdfSetFunctionState,
) -> std::io::Result<Option<&str>> {
    match (function, state) {
        (
            AggregateFunction::Count | AggregateFunction::CountNonNull,
            RdfSetFunctionState::Count(_),
        )
        | (AggregateFunction::Sum, RdfSetFunctionState::Sum(_))
        | (AggregateFunction::Avg, RdfSetFunctionState::Average(_))
        | (AggregateFunction::Min, RdfSetFunctionState::Minimum { .. })
        | (AggregateFunction::Max, RdfSetFunctionState::Maximum(_))
        | (AggregateFunction::Sample, RdfSetFunctionState::Sample(_)) => Ok(None),
        (AggregateFunction::GroupConcat, RdfSetFunctionState::GroupConcat { separator, .. }) => {
            Ok(Some(separator))
        }
        (_, RdfSetFunctionState::Generic(_)) => Err(invalid_spill(
            "unsupported non-SPARQL aggregate state cannot be spilled",
        )),
        _ => Err(invalid_spill(
            "RDF aggregate function has mismatched live state",
        )),
    }
}

fn rdf_aggregate_compare(left: &Value, right: &Value) -> Option<Ordering> {
    if let (Some((_, left_term)), Some((_, right_term))) = (
        decode_tagged_rdf_filter_term(left),
        decode_tagged_rdf_filter_term(right),
    ) {
        return Some(rdf_compare_terms(&left_term, &right_term));
    }

    compare_values(left, right)
}

#[derive(Default)]
struct RdfNumericState {
    // P2 deliberately retains every ordered operand, including exact values.
    // Algebraically compacting an exact-only partial can change a later
    // float/double merge. The bounded aggregate packet must externalize these
    // records; it must not redefine the delivered sequential arithmetic.
    inputs: Vec<RdfNumericInput>,
    error: bool,
}

struct RdfNumericInput {
    ordinal: u64,
    value: RdfNumeric,
}

impl RdfNumericState {
    #[cfg(test)]
    fn update(&mut self, tagged: &Value) -> Result<(), OperatorError> {
        let ordinal = self.inputs.last().map_or(Ok(0), |input| {
            input.ordinal.checked_add(1).ok_or_else(|| {
                OperatorError::Execution("RDF numeric input ordinal overflow".to_string())
            })
        })?;
        self.update_at(tagged, ordinal)
    }

    fn update_at(&mut self, tagged: &Value, ordinal: u64) -> Result<(), OperatorError> {
        let Some((_, Term::Literal(literal))) = decode_tagged_rdf_filter_term(tagged) else {
            self.error = true;
            return Ok(());
        };
        let Some(input) = RdfNumeric::from_literal(&literal) else {
            self.error = true;
            return Ok(());
        };
        if self
            .inputs
            .last()
            .is_some_and(|previous| previous.ordinal >= ordinal)
        {
            return Err(OperatorError::Execution(
                "RDF numeric input ordinals are not strictly increasing".to_string(),
            ));
        }
        self.inputs.try_reserve(1).map_err(|error| {
            OperatorError::ResidentAllocation(format!(
                "cannot reserve RDF numeric aggregate input: {error}"
            ))
        })?;
        self.inputs.push(RdfNumericInput {
            ordinal,
            value: input,
        });
        Ok(())
    }

    fn finalize(&self, average: bool) -> Value {
        if self.error {
            return Value::Null;
        }
        let Some(first) = self.inputs.first() else {
            return Value::Int64(0);
        };
        let Some(sum) = self.inputs[1..]
            .iter()
            .try_fold(first.value.clone(), |sum, input| {
                sum.checked_add(input.value.clone())
            })
        else {
            return Value::Null;
        };
        if average {
            u64::try_from(self.inputs.len())
                .ok()
                .and_then(|count| sum.average(count))
                .map_or(Value::Null, RdfNumeric::into_value)
        } else {
            sum.into_value()
        }
    }

    #[cfg(feature = "spill")]
    fn merge_from(&mut self, mut other: Self) -> std::io::Result<()> {
        self.inputs
            .try_reserve(other.inputs.len())
            .map_err(|error| {
                std::io::Error::other(format!("cannot reserve merged RDF numeric inputs: {error}"))
            })?;
        self.inputs.append(&mut other.inputs);
        self.inputs.sort_unstable_by_key(|input| input.ordinal);
        if self
            .inputs
            .windows(2)
            .any(|inputs| inputs[0].ordinal == inputs[1].ordinal)
        {
            return Err(invalid_spill(
                "duplicate RDF numeric input ordinal during state merge",
            ));
        }
        self.error |= other.error;
        Ok(())
    }

    #[cfg(feature = "spill")]
    fn write_spill(
        &self,
        writer: &mut impl Write,
        budget: &mut SpillCodecEncodeBudget,
    ) -> std::io::Result<()> {
        if self
            .inputs
            .windows(2)
            .any(|inputs| inputs[0].ordinal >= inputs[1].ordinal)
        {
            return Err(invalid_spill(
                "RDF numeric input ordinals are not strictly increasing",
            ));
        }
        writer.write_all(&[u8::from(self.error)])?;
        budget.charge_items::<RdfNumericInput>(self.inputs.len(), "RDF numeric input")?;
        write_count(writer, self.inputs.len(), "RDF numeric inputs")?;
        for input in &self.inputs {
            writer.write_all(&input.ordinal.to_le_bytes())?;
            input.value.write_spill(writer)?;
        }
        Ok(())
    }

    #[cfg(feature = "spill")]
    fn read_spill(
        reader: &mut Cursor<&[u8]>,
        budget: &mut SpillCodecDecodeBudget,
    ) -> std::io::Result<Self> {
        let error = match read_byte(reader)? {
            0 => false,
            1 => true,
            value => {
                return Err(invalid_spill(format!(
                    "invalid RDF numeric error flag {value}"
                )));
            }
        };
        let count =
            read_collection_count::<RdfNumericInput>(reader, budget, 13, "RDF numeric inputs")?;
        let mut inputs = Vec::new();
        inputs.try_reserve_exact(count).map_err(|error| {
            std::io::Error::other(format!("cannot reserve RDF numeric inputs: {error}"))
        })?;
        for _ in 0..count {
            let mut ordinal = [0u8; 8];
            reader.read_exact(&mut ordinal)?;
            let ordinal = u64::from_le_bytes(ordinal);
            if inputs
                .last()
                .is_some_and(|input: &RdfNumericInput| input.ordinal >= ordinal)
            {
                return Err(invalid_spill(
                    "RDF numeric input ordinals are not strictly increasing",
                ));
            }
            inputs.push(RdfNumericInput {
                ordinal,
                value: RdfNumeric::read_spill(reader)?,
            });
        }
        Ok(Self { inputs, error })
    }
}

fn rdf_aggregate_string(tagged: &Value) -> Option<String> {
    let (_, term) = decode_tagged_rdf_filter_term(tagged)?;
    match term {
        Term::Literal(literal)
            if literal.language().is_some() || literal.datatype() == Literal::XSD_STRING =>
        {
            Some(literal.value().to_string())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_core::execution::DataChunk;
    use std::collections::VecDeque;

    struct MockOperator {
        chunks: VecDeque<DataChunk>,
    }

    impl Operator for MockOperator {
        fn next(&mut self) -> OperatorResult {
            Ok(self.chunks.pop_front())
        }

        fn reset(&mut self) {}

        fn name(&self) -> &'static str {
            "RdfAggregateMock"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    fn numeric(lexical: &str, datatype: &str) -> Value {
        super::super::tagged_rdf_term(
            Value::RdfLiteral {
                lexical: lexical.into(),
                language: None,
                datatype: Some(datatype.into()),
            },
            Term::typed_literal(lexical, datatype),
        )
    }

    #[cfg(feature = "spill")]
    fn reseal_aggregate_record(encoded: &mut [u8]) {
        let checksum = rdf_aggregate_checksum(&encoded[RDF_AGGREGATE_STATE_HEADER_BYTES..]);
        let checksum_offset = 4 + 1 + 8;
        encoded[checksum_offset..checksum_offset + 4].copy_from_slice(&checksum.to_le_bytes());
    }

    #[test]
    fn exact_numeric_state_promotes_integer_and_decimal_without_losing_precision() {
        let mut state = RdfNumericState::default();
        state
            .update(&numeric("9007199254740993", Literal::XSD_INTEGER))
            .unwrap();
        state
            .update(&numeric("0.25", Literal::XSD_DECIMAL))
            .unwrap();

        assert!(matches!(
            state.finalize(false),
            Value::RdfLiteral { lexical, datatype: Some(datatype), .. }
                if lexical.as_str() == "9007199254740993.25"
                    && datatype.as_str() == Literal::XSD_DECIMAL
        ));
    }

    #[test]
    fn exact_numeric_state_distinguishes_empty_from_error() {
        let empty = RdfNumericState::default();
        assert_eq!(empty.finalize(false), Value::Int64(0));
        assert_eq!(empty.finalize(true), Value::Int64(0));

        let mut error = RdfNumericState::default();
        error.update(&Value::Null).unwrap();
        assert_eq!(error.finalize(false), Value::Null);
        assert_eq!(error.finalize(true), Value::Null);
    }

    #[cfg(feature = "spill")]
    #[test]
    fn numeric_codec_preserves_float_and_double_bits_and_rejects_unknown_tags() {
        const XSD_FLOAT: &str = "http://www.w3.org/2001/XMLSchema#float";
        let cases = [
            numeric("-0", XSD_FLOAT),
            numeric("NaN", XSD_FLOAT),
            numeric("-0", Literal::XSD_DOUBLE),
            numeric("NaN", Literal::XSD_DOUBLE),
        ];

        for tagged in cases {
            let (_, Term::Literal(literal)) = decode_tagged_rdf_filter_term(&tagged).unwrap()
            else {
                unreachable!("the fixture is a typed numeric literal")
            };
            let value = RdfNumeric::from_literal(&literal).unwrap();
            let mut encoded = Vec::new();
            value.write_spill(&mut encoded).unwrap();
            let decoded = RdfNumeric::read_spill(&mut Cursor::new(encoded.as_slice())).unwrap();

            match literal.datatype() {
                XSD_FLOAT => assert_eq!(
                    decoded.as_f32().unwrap().to_bits(),
                    value.as_f32().unwrap().to_bits()
                ),
                Literal::XSD_DOUBLE => {
                    assert_eq!(decoded.as_f64().to_bits(), value.as_f64().to_bits());
                }
                _ => unreachable!("the fixture is float or double"),
            }
        }

        let Err(error) = RdfNumeric::read_spill(&mut Cursor::new([0xff].as_slice())) else {
            panic!("an unknown RDF numeric spill tag was accepted");
        };
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("unknown RDF numeric spill tag"));
    }

    #[cfg(feature = "spill")]
    #[test]
    fn resumable_group_state_round_trips_then_continues_exact_numeric_updates() {
        use grafeo_core::execution::spill::CodecLimits;

        let expression = AggregateExpr::sum(0);
        let mut group = RdfGroupState {
            key_values: vec![Value::String("group-a".into())],
            accumulators: vec![RdfAccumulator::new(&expression)],
        };
        group.accumulators[0]
            .update_at(
                Some(numeric("9007199254740993", Literal::XSD_INTEGER)),
                None,
                None,
                0,
            )
            .unwrap();

        let encoded = group.encode_spill(CodecLimits::default()).unwrap();
        let mut reopened = RdfGroupState::decode_spill(&encoded, CodecLimits::default()).unwrap();
        reopened.accumulators[0]
            .update_at(Some(numeric("0.25", Literal::XSD_DECIMAL)), None, None, 1)
            .unwrap();

        assert_eq!(reopened.key_values, vec![Value::String("group-a".into())]);
        assert!(matches!(
            reopened.accumulators[0].finalize(),
            Value::RdfLiteral { lexical, datatype: Some(datatype), .. }
                if lexical.as_str() == "9007199254740993.25"
                    && datatype.as_str() == Literal::XSD_DECIMAL
        ));
    }

    #[cfg(feature = "spill")]
    #[test]
    fn group_merge_uses_the_same_hash_identity_as_in_memory_grouping() {
        let key = super::super::tagged_rdf_term(
            Value::Float64(f64::NAN),
            Term::typed_literal("NaN", Literal::XSD_DOUBLE),
        );
        let mut left = RdfGroupState {
            key_values: vec![key.clone()],
            accumulators: Vec::new(),
        };
        let right = RdfGroupState {
            key_values: vec![key],
            accumulators: Vec::new(),
        };

        left.merge_from(right)
            .expect("matching NaN-valued RDF group keys must merge");
    }

    #[cfg(feature = "spill")]
    #[test]
    fn aggregate_state_checksum_rejects_a_well_formed_payload_mutation() {
        use grafeo_core::execution::spill::CodecLimits;

        let expression = AggregateExpr::sum(0);
        let group = RdfGroupState {
            key_values: vec![Value::String("group-a".into())],
            accumulators: vec![RdfAccumulator::new(&expression)],
        };
        let mut encoded = group.encode_spill(CodecLimits::default()).unwrap();
        let key_byte = encoded
            .windows(b"group-a".len())
            .position(|window| window == b"group-a")
            .expect("the lossless Value codec retains the group-key bytes")
            + b"group-".len();
        encoded[key_byte] = b'b';

        let Err(error) = RdfGroupState::decode_spill(&encoded, CodecLimits::default()) else {
            panic!("a checksum-protected aggregate state accepted modified bytes");
        };
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("checksum"));
    }

    #[cfg(feature = "spill")]
    #[test]
    fn aggregate_codec_rejects_every_truncated_record_prefix() {
        use grafeo_core::execution::spill::CodecLimits;

        let encoded = nested_value_budget_group_with_items(8)
            .encode_spill(CodecLimits::default())
            .unwrap();
        for truncated_length in 0..encoded.len() {
            let result =
                RdfGroupState::decode_spill(&encoded[..truncated_length], CodecLimits::default());
            assert!(
                result.is_err(),
                "accepted aggregate record truncated to {truncated_length} bytes"
            );
        }
    }

    #[cfg(feature = "spill")]
    #[test]
    fn aggregate_codec_enforces_total_byte_and_item_limits() {
        use grafeo_core::execution::spill::CodecLimits;

        let empty = RdfGroupState {
            key_values: Vec::new(),
            accumulators: Vec::new(),
        };
        let tight_bytes = CodecLimits::new(8, 16, 4, 8);
        let encode_error = empty.encode_spill(tight_bytes).unwrap_err();
        assert_eq!(encode_error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(encode_error.to_string().contains("codec byte limit"));

        let encoded = empty.encode_spill(CodecLimits::default()).unwrap();
        let Err(decode_error) = RdfGroupState::decode_spill(&encoded, tight_bytes) else {
            panic!("an aggregate payload above the codec byte limit was accepted");
        };
        assert_eq!(decode_error.kind(), std::io::ErrorKind::InvalidData);
        assert!(decode_error.to_string().contains("codec byte limit"));

        let two_keys = RdfGroupState {
            key_values: vec![Value::Int64(1), Value::Int64(2)],
            accumulators: Vec::new(),
        };
        let one_item = CodecLimits::new(1024, 1, 4, 8);
        let encode_error = two_keys.encode_spill(one_item).unwrap_err();
        assert_eq!(encode_error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(encode_error.to_string().contains("codec item limit"));

        let encoded = two_keys.encode_spill(CodecLimits::default()).unwrap();
        let Err(decode_error) = RdfGroupState::decode_spill(&encoded, one_item) else {
            panic!("aggregate collection counts exceeded the codec item limit");
        };
        assert_eq!(decode_error.kind(), std::io::ErrorKind::InvalidData);
        assert!(decode_error.to_string().contains("codec item limit"));
    }

    #[cfg(feature = "spill")]
    fn nested_value_budget_group() -> RdfGroupState {
        nested_value_budget_group_with_items(2)
    }

    #[cfg(feature = "spill")]
    fn nested_value_budget_group_with_items(item_count: usize) -> RdfGroupState {
        let nested_value = || Value::List(vec![Value::Null; item_count].into());
        let mut accumulator = RdfAccumulator::new(&AggregateExpr::min(0));
        accumulator
            .update_at(Some(nested_value()), None, None, 0)
            .unwrap();
        RdfGroupState {
            key_values: vec![nested_value()],
            accumulators: vec![accumulator],
        }
    }

    #[cfg(feature = "spill")]
    #[test]
    fn aggregate_encode_shares_nested_value_item_budget_across_keys_and_operands() {
        use grafeo_core::execution::spill::CodecLimits;

        // One key slot + one accumulator slot + two nested list items in each
        // Value is six items. Each Value independently fits; the complete
        // aggregate state does not fit a five-item operation budget.
        let limits = CodecLimits::new(4096, 5, 16, 8);
        let error = nested_value_budget_group()
            .encode_spill(limits)
            .expect_err("fresh per-Value budgets accepted six aggregate items as five");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("cumulative 5-item codec budget"));
    }

    #[cfg(feature = "spill")]
    #[test]
    fn aggregate_decode_shares_nested_value_item_budget_across_keys_and_operands() {
        use grafeo_core::execution::spill::CodecLimits;

        let encoded = nested_value_budget_group()
            .encode_spill(CodecLimits::default())
            .unwrap();
        let limits = CodecLimits::new(4096, 5, 16, 8);
        let Err(error) = RdfGroupState::decode_spill(&encoded, limits) else {
            panic!("fresh per-Value budgets amplified a five-item decode grant");
        };

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("cumulative 5-item codec budget"));
    }

    #[cfg(feature = "spill")]
    #[test]
    fn aggregate_encode_shares_nested_value_byte_budget_across_keys_and_operands() {
        use grafeo_core::execution::spill::CodecLimits;

        const NESTED_ITEMS: usize = 32;
        let one_nested_value_bytes = NESTED_ITEMS * std::mem::size_of::<Value>();
        let outer_slot_bytes = std::mem::size_of::<Value>() + std::mem::size_of::<RdfAccumulator>();
        let max_bytes = one_nested_value_bytes + outer_slot_bytes;
        let limits = CodecLimits::new(max_bytes, 128, 16, 8);
        let error = nested_value_budget_group_with_items(NESTED_ITEMS)
            .encode_spill(limits)
            .expect_err("fresh per-Value budgets accepted cumulative resident bytes");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(
            error
                .to_string()
                .contains(&format!("cumulative {max_bytes}-byte codec budget"))
        );
    }

    #[cfg(feature = "spill")]
    #[test]
    fn aggregate_decode_shares_nested_value_byte_budget_across_keys_and_operands() {
        use grafeo_core::execution::spill::CodecLimits;

        const NESTED_ITEMS: usize = 32;
        let encoded = nested_value_budget_group_with_items(NESTED_ITEMS)
            .encode_spill(CodecLimits::default())
            .unwrap();
        let one_nested_value_bytes = NESTED_ITEMS * std::mem::size_of::<Value>();
        let outer_slot_bytes = std::mem::size_of::<Value>() + std::mem::size_of::<RdfAccumulator>();
        let max_bytes = one_nested_value_bytes + outer_slot_bytes;
        let limits = CodecLimits::new(max_bytes, 128, 16, 8);
        let Err(error) = RdfGroupState::decode_spill(&encoded, limits) else {
            panic!("fresh per-Value budgets amplified the aggregate byte grant");
        };

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            error
                .to_string()
                .contains(&format!("cumulative {max_bytes}-byte codec budget"))
        );
    }

    #[cfg(feature = "spill")]
    #[test]
    fn aggregate_codec_rejects_count_and_length_bombs_before_reserve() {
        use grafeo_core::execution::spill::CodecLimits;

        let empty = RdfGroupState {
            key_values: Vec::new(),
            accumulators: Vec::new(),
        };
        let mut count_bomb = empty.encode_spill(CodecLimits::default()).unwrap();
        let key_count = RDF_AGGREGATE_STATE_HEADER_BYTES;
        count_bomb[key_count..key_count + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        reseal_aggregate_record(&mut count_bomb);
        let Err(error) = RdfGroupState::decode_spill(&count_bomb, CodecLimits::default()) else {
            panic!("an impossible RDF aggregate collection count was accepted");
        };
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("remaining spill record"));

        let expression = AggregateExpr {
            function: AggregateFunction::GroupConcat,
            column: Some(0),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: Some("unique-separator".to_string()),
        };
        let group = RdfGroupState {
            key_values: Vec::new(),
            accumulators: vec![RdfAccumulator::new(&expression)],
        };
        let mut length_bomb = group.encode_spill(CodecLimits::default()).unwrap();
        let separator = length_bomb
            .windows(b"unique-separator".len())
            .position(|window| window == b"unique-separator")
            .expect("the separator is encoded verbatim");
        let length = separator.checked_sub(4).unwrap();
        length_bomb[length..separator].copy_from_slice(&u32::MAX.to_le_bytes());
        reseal_aggregate_record(&mut length_bomb);
        let Err(error) = RdfGroupState::decode_spill(&length_bomb, CodecLimits::default()) else {
            panic!("an impossible RDF aggregate string length was accepted");
        };
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("remaining spill record"));
    }

    #[cfg(feature = "spill")]
    #[test]
    fn aggregate_codec_rejects_unknown_versions_functions_and_mismatched_live_state() {
        use grafeo_core::execution::spill::CodecLimits;

        let expression = AggregateExpr::sum(0);
        let group = RdfGroupState {
            key_values: Vec::new(),
            accumulators: vec![RdfAccumulator::new(&expression)],
        };
        let encoded = group.encode_spill(CodecLimits::default()).unwrap();

        let mut unknown_version = encoded.clone();
        unknown_version[4] = RDF_AGGREGATE_STATE_VERSION + 1;
        let Err(error) = RdfGroupState::decode_spill(&unknown_version, CodecLimits::default())
        else {
            panic!("an unknown RDF aggregate state version was accepted");
        };
        assert!(
            error
                .to_string()
                .contains("unsupported RDF aggregate state version")
        );

        let mut unknown_function = encoded;
        let function = RDF_AGGREGATE_STATE_HEADER_BYTES + 4 + 4;
        unknown_function[function] = 0xff;
        reseal_aggregate_record(&mut unknown_function);
        let Err(error) = RdfGroupState::decode_spill(&unknown_function, CodecLimits::default())
        else {
            panic!("an unknown RDF aggregate function tag was accepted");
        };
        assert!(
            error
                .to_string()
                .contains("unknown RDF aggregate function tag")
        );

        let mismatched = RdfGroupState {
            key_values: Vec::new(),
            accumulators: vec![RdfAccumulator {
                function: AggregateFunction::Sum,
                distinct: None,
                state: RdfSetFunctionState::Count(0),
            }],
        };
        let error = mismatched.encode_spill(CodecLimits::default()).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("mismatched live state"));
    }

    #[cfg(feature = "spill")]
    #[test]
    fn every_standard_rdf_set_function_reopens_as_live_state() {
        use grafeo_core::execution::spill::CodecLimits;

        fn expression(function: AggregateFunction) -> AggregateExpr {
            AggregateExpr {
                function,
                column: (!matches!(function, AggregateFunction::Count)).then_some(0),
                column2: None,
                distinct_key_column: None,
                distinct: false,
                alias: None,
                percentile: None,
                separator: (function == AggregateFunction::GroupConcat).then(|| "|".to_string()),
            }
        }

        let expressions = [
            expression(AggregateFunction::Count),
            expression(AggregateFunction::CountNonNull),
            expression(AggregateFunction::Sum),
            expression(AggregateFunction::Avg),
            expression(AggregateFunction::Min),
            expression(AggregateFunction::Max),
            expression(AggregateFunction::Sample),
            expression(AggregateFunction::GroupConcat),
        ];
        let mut group = RdfGroupState {
            key_values: Vec::new(),
            accumulators: expressions.iter().map(RdfAccumulator::new).collect(),
        };
        let first = [
            None,
            Some(Value::Int64(7)),
            Some(numeric("9007199254740993", Literal::XSD_INTEGER)),
            Some(numeric("1", Literal::XSD_INTEGER)),
            Some(Value::Int64(5)),
            Some(Value::Int64(5)),
            Some(Value::String("first".into())),
            Some(super::super::tagged_rdf_term(
                Value::String("a".into()),
                Term::literal("a"),
            )),
        ];
        for (accumulator, value) in group.accumulators.iter_mut().zip(first) {
            accumulator.update_at(value, None, None, 0).unwrap();
        }

        let encoded = group.encode_spill(CodecLimits::default()).unwrap();
        let mut reopened = RdfGroupState::decode_spill(&encoded, CodecLimits::default()).unwrap();
        let second = [
            None,
            Some(Value::Int64(9)),
            Some(numeric("0.25", Literal::XSD_DECIMAL)),
            Some(numeric("3", Literal::XSD_INTEGER)),
            Some(Value::Int64(3)),
            Some(Value::Int64(8)),
            Some(Value::String("second".into())),
            Some(super::super::tagged_rdf_term(
                Value::String("b".into()),
                Term::literal("b"),
            )),
        ];
        for (accumulator, value) in reopened.accumulators.iter_mut().zip(second) {
            accumulator.update_at(value, None, None, 1).unwrap();
        }

        let values = reopened
            .accumulators
            .iter()
            .map(RdfAccumulator::finalize)
            .collect::<Vec<_>>();
        assert_eq!(values[0], Value::Int64(2));
        assert_eq!(values[1], Value::Int64(2));
        assert!(matches!(
            &values[2],
            Value::RdfLiteral { lexical, datatype: Some(datatype), .. }
                if lexical.as_str() == "9007199254740993.25"
                    && datatype.as_str() == Literal::XSD_DECIMAL
        ));
        assert!(matches!(
            &values[3],
            Value::RdfLiteral { lexical, datatype: Some(datatype), .. }
                if lexical.as_str() == "2.0" && datatype.as_str() == Literal::XSD_DECIMAL
        ));
        assert_eq!(values[4], Value::Int64(3));
        assert_eq!(values[5], Value::Int64(8));
        assert_eq!(values[6], Value::String("first".into()));
        assert_eq!(values[7], Value::String("a|b".into()));
    }

    #[cfg(feature = "spill")]
    #[test]
    fn aggregate_error_and_unbound_state_survives_reopen_and_continuation() {
        use grafeo_core::execution::spill::CodecLimits;

        let expressions = [
            AggregateExpr::sum(0),
            AggregateExpr::avg(0),
            AggregateExpr::min(0),
            AggregateExpr {
                function: AggregateFunction::GroupConcat,
                column: Some(0),
                column2: None,
                distinct_key_column: None,
                distinct: false,
                alias: None,
                percentile: None,
                separator: Some("|".to_string()),
            },
        ];
        let mut group = RdfGroupState {
            key_values: Vec::new(),
            accumulators: expressions.iter().map(RdfAccumulator::new).collect(),
        };
        for accumulator in &mut group.accumulators {
            accumulator.update_at(None, None, None, 0).unwrap();
        }

        let mut reopened = RdfGroupState::decode_spill(
            &group.encode_spill(CodecLimits::default()).unwrap(),
            CodecLimits::default(),
        )
        .unwrap();
        let valid = [
            numeric("2", Literal::XSD_INTEGER),
            numeric("2", Literal::XSD_INTEGER),
            Value::Int64(2),
            super::super::tagged_rdf_term(Value::String("ok".into()), Term::literal("ok")),
        ];
        for (accumulator, value) in reopened.accumulators.iter_mut().zip(valid) {
            accumulator.update_at(Some(value), None, None, 1).unwrap();
        }

        assert!(
            reopened
                .accumulators
                .iter()
                .all(|accumulator| accumulator.finalize().is_null())
        );
    }

    #[cfg(feature = "spill")]
    #[test]
    fn numeric_state_merge_replays_cancellation_sensitive_inputs_by_ordinal() {
        let expression = AggregateExpr::sum(0);
        let mut even_rows = RdfAccumulator::new(&expression);
        even_rows
            .update_at(Some(numeric("1e16", Literal::XSD_DOUBLE)), None, None, 0)
            .unwrap();
        even_rows
            .update_at(Some(numeric("-1e16", Literal::XSD_DOUBLE)), None, None, 2)
            .unwrap();
        let mut odd_rows = RdfAccumulator::new(&expression);
        odd_rows
            .update_at(Some(numeric("1", Literal::XSD_DOUBLE)), None, None, 1)
            .unwrap();

        even_rows.merge_from(odd_rows).unwrap();

        assert_eq!(even_rows.finalize(), Value::Float64(0.0));
    }

    #[cfg(feature = "spill")]
    #[test]
    fn exact_operands_after_a_double_remain_ordered_across_partial_state_merge() {
        // Compacting the exact-only right partial to `2` would change the
        // delivered sequential result to 10_000_000_000_000_002. P2 keeps
        // exact operands ordered; the bounded aggregate packet must externalize
        // these records rather than algebraically rewriting them.
        let expression = AggregateExpr::sum(0);
        let mut floating = RdfAccumulator::new(&expression);
        floating
            .update_at(Some(numeric("1e16", Literal::XSD_DOUBLE)), None, None, 0)
            .unwrap();
        let mut exact = RdfAccumulator::new(&expression);
        exact
            .update_at(Some(numeric("1", Literal::XSD_INTEGER)), None, None, 1)
            .unwrap();
        exact
            .update_at(Some(numeric("1", Literal::XSD_INTEGER)), None, None, 2)
            .unwrap();

        floating.merge_from(exact).unwrap();

        assert_eq!(floating.finalize(), Value::Float64(1e16));
    }

    #[test]
    fn count_overflow_is_structured_instead_of_saturating() {
        let mut accumulator = RdfAccumulator::new(&AggregateExpr::count_star());
        accumulator.state = RdfSetFunctionState::Count(i64::MAX);

        let error = accumulator.update_at(None, None, None, 0).unwrap_err();

        assert!(matches!(
            error,
            OperatorError::Execution(message) if message.contains("COUNT") && message.contains("overflow")
        ));
    }

    #[cfg(feature = "spill")]
    #[test]
    fn count_merge_overflow_is_rejected_instead_of_saturating() {
        let mut left = RdfAccumulator::new(&AggregateExpr::count_star());
        left.state = RdfSetFunctionState::Count(i64::MAX);
        let mut right = RdfAccumulator::new(&AggregateExpr::count_star());
        right.state = RdfSetFunctionState::Count(1);

        let error = left.merge_from(right).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("COUNT merge overflow"));
    }

    #[test]
    fn numeric_helper_rejects_encounter_ordinal_overflow() {
        let mut state = RdfNumericState::default();
        state
            .update_at(&numeric("1", Literal::XSD_INTEGER), u64::MAX)
            .unwrap();

        let error = state
            .update(&numeric("1", Literal::XSD_INTEGER))
            .unwrap_err();

        assert!(matches!(
            error,
            OperatorError::Execution(message)
                if message.contains("numeric input ordinal overflow")
        ));
    }

    #[cfg(feature = "spill")]
    #[test]
    fn reopened_group_merge_uses_sample_and_concat_encounter_ordinals() {
        use grafeo_core::execution::spill::CodecLimits;

        let sample = AggregateExpr {
            function: AggregateFunction::Sample,
            column: Some(0),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        };
        let concat = AggregateExpr {
            function: AggregateFunction::GroupConcat,
            column: Some(0),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: Some("|".to_string()),
        };
        let concat_value = |value: &'static str| {
            super::super::tagged_rdf_term(Value::String(value.into()), Term::literal(value))
        };
        let mut left = RdfGroupState {
            key_values: vec![Value::String("g".into())],
            accumulators: vec![RdfAccumulator::new(&sample), RdfAccumulator::new(&concat)],
        };
        left.accumulators[0]
            .update_at(Some(Value::String("late".into())), None, None, 10)
            .unwrap();
        left.accumulators[1]
            .update_at(Some(concat_value("a")), None, None, 0)
            .unwrap();
        left.accumulators[1]
            .update_at(Some(concat_value("c")), None, None, 2)
            .unwrap();
        let mut right = RdfGroupState {
            key_values: vec![Value::String("g".into())],
            accumulators: vec![RdfAccumulator::new(&sample), RdfAccumulator::new(&concat)],
        };
        right.accumulators[0]
            .update_at(Some(Value::String("early".into())), None, None, 1)
            .unwrap();
        right.accumulators[1]
            .update_at(Some(concat_value("b")), None, None, 1)
            .unwrap();

        let mut left = RdfGroupState::decode_spill(
            &left.encode_spill(CodecLimits::default()).unwrap(),
            CodecLimits::default(),
        )
        .unwrap();
        let right = RdfGroupState::decode_spill(
            &right.encode_spill(CodecLimits::default()).unwrap(),
            CodecLimits::default(),
        )
        .unwrap();
        left.merge_from(right).unwrap();

        assert_eq!(
            left.accumulators[0].finalize(),
            Value::String("early".into())
        );
        assert_eq!(
            left.accumulators[1].finalize(),
            Value::String("a|b|c".into())
        );
    }

    #[cfg(feature = "spill")]
    #[test]
    fn distinct_merge_keeps_earliest_operand_and_replays_floats_by_ordinal() {
        use grafeo_core::execution::spill::CodecLimits;

        let expression = AggregateExpr {
            distinct: true,
            distinct_key_column: Some(1),
            ..AggregateExpr::sum(0)
        };
        let mut left = RdfGroupState {
            key_values: vec![Value::String("g".into())],
            accumulators: vec![RdfAccumulator::new(&expression)],
        };
        for (ordinal, lexical, key) in [
            (0, "1e16", "first"),
            (2, "-1e16", "last"),
            (3, "100", "duplicate"),
        ] {
            left.accumulators[0]
                .update_at(
                    Some(numeric(lexical, Literal::XSD_DOUBLE)),
                    Some(Value::String(key.into())),
                    None,
                    ordinal,
                )
                .unwrap();
        }
        let mut right = RdfGroupState {
            key_values: vec![Value::String("g".into())],
            accumulators: vec![RdfAccumulator::new(&expression)],
        };
        right.accumulators[0]
            .update_at(
                Some(numeric("1", Literal::XSD_DOUBLE)),
                Some(Value::String("duplicate".into())),
                None,
                1,
            )
            .unwrap();

        let mut left = RdfGroupState::decode_spill(
            &left.encode_spill(CodecLimits::default()).unwrap(),
            CodecLimits::default(),
        )
        .unwrap();
        let right = RdfGroupState::decode_spill(
            &right.encode_spill(CodecLimits::default()).unwrap(),
            CodecLimits::default(),
        )
        .unwrap();
        left.merge_from(right).unwrap();
        left.accumulators[0]
            .update_at(
                Some(numeric("2", Literal::XSD_DOUBLE)),
                Some(Value::String("continued".into())),
                None,
                4,
            )
            .unwrap();

        assert_eq!(left.accumulators[0].finalize(), Value::Float64(2.0));
    }

    #[test]
    fn rdf_aggregate_streams_grouped_results_across_output_chunks() {
        let mut input =
            DataChunkBuilder::with_capacity(&[LogicalType::Any, LogicalType::Any], 2050);
        for index in 0..2050 {
            input.column_mut(0).unwrap().push_value(Value::Int64(index));
            input
                .column_mut(1)
                .unwrap()
                .push_value(numeric("1", Literal::XSD_INTEGER));
            input.advance_row();
        }
        let aggregate = AggregateExpr {
            function: AggregateFunction::Sum,
            column: Some(1),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: Some("sum".to_string()),
            percentile: None,
            separator: None,
        };
        let child = Box::new(MockOperator {
            chunks: VecDeque::from([input.finish()]),
        });
        let mut operator = RdfAggregateOperator::new(
            child,
            vec![0],
            vec![aggregate],
            vec![LogicalType::Any, LogicalType::Any],
            None,
        );

        let first = operator.next().unwrap().unwrap();
        let second = operator.next().unwrap().unwrap();
        assert_eq!(first.row_count(), OUTPUT_CHUNK_SIZE);
        assert_eq!(second.row_count(), 2);
        assert!(operator.next().unwrap().is_none());
        assert_eq!(first.column(1).unwrap().get_value(0), Some(Value::Int64(1)));
        assert_eq!(
            second.column(0).unwrap().get_value(1),
            Some(Value::Int64(2049))
        );
    }

    #[test]
    fn rdf_aggregate_combines_multiple_input_chunks_into_one_global_group() {
        let chunks = (0..2)
            .map(|_| {
                let mut input =
                    DataChunkBuilder::with_capacity(&[LogicalType::Any], OUTPUT_CHUNK_SIZE);
                for _ in 0..OUTPUT_CHUNK_SIZE {
                    input
                        .column_mut(0)
                        .unwrap()
                        .push_value(numeric("1", Literal::XSD_INTEGER));
                    input.advance_row();
                }
                input.finish()
            })
            .collect::<VecDeque<_>>();
        let aggregate = AggregateExpr {
            function: AggregateFunction::Sum,
            column: Some(0),
            column2: None,
            distinct_key_column: None,
            distinct: false,
            alias: Some("sum".to_string()),
            percentile: None,
            separator: None,
        };
        let mut operator = RdfAggregateOperator::new(
            Box::new(MockOperator { chunks }),
            Vec::new(),
            vec![aggregate],
            vec![LogicalType::Any],
            None,
        );

        let result = operator.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(
            result.column(0).unwrap().get_value(0),
            Some(Value::Int64(
                i64::try_from(OUTPUT_CHUNK_SIZE * 2).expect("test chunk count fits i64")
            ))
        );
    }

    #[test]
    fn count_distinct_star_falls_back_to_native_identity_per_row() {
        let mut input = DataChunkBuilder::with_capacity(&[LogicalType::Any, LogicalType::Any], 4);
        for (visible, canonical_rdf) in [
            (
                Value::String("urn:x".into()),
                Value::String("<urn:x>".into()),
            ),
            (Value::Vector(vec![1.0].into()), Value::Null),
            (Value::Vector(vec![2.0].into()), Value::Null),
            (Value::Vector(vec![1.0].into()), Value::Null),
        ] {
            input.column_mut(0).unwrap().push_value(visible);
            input.column_mut(1).unwrap().push_value(canonical_rdf);
            input.advance_row();
        }
        let aggregate = AggregateExpr {
            function: AggregateFunction::Count,
            column: None,
            column2: None,
            distinct_key_column: None,
            distinct: true,
            alias: Some("count".to_string()),
            percentile: None,
            separator: None,
        };
        let mut operator = RdfAggregateOperator::new(
            Box::new(MockOperator {
                chunks: VecDeque::from([input.finish()]),
            }),
            Vec::new(),
            vec![aggregate],
            vec![LogicalType::Int64],
            Some(vec![RdfRowIdentityColumn {
                visible: 0,
                canonical_rdf: Some(1),
            }]),
        );

        let result = operator.next().unwrap().unwrap();
        assert_eq!(
            result.column(0).unwrap().get_value(0),
            Some(Value::Int64(3)),
            "native rows must not collapse merely because an RDF companion exists in the schema"
        );
    }
}
