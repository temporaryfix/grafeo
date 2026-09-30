//! Ordered group partitions: RAGS partials, exact membership, then ordinal replay.

use super::{
    RdfAccumulator, RdfAggregateOperator, RdfDistinctEntry, RdfGroupState, RdfNumeric,
    RdfNumericInput, RdfNumericState, RdfOrderedString, RdfRowIdentityColumn, RdfSetFunctionState,
    bounded,
};
use grafeo_common::memory::buffer::MemoryGrant;
use grafeo_common::types::{HashableValue, LogicalType, Value};
use grafeo_core::execution::{
    DataChunk, QueryResourceContext, QueryResourceContextError,
    chunk::DataChunkBuilder,
    operators::{
        AggregateExpr, AggregateFunction, Operator, OperatorError, OperatorResult, SortKey,
        SortOperator,
    },
};
use indexmap::IndexMap;
use std::cmp::Ordering;
use std::sync::Arc;

const PACKET_COLUMNS: usize = 7;
// canonical group, aggregate index, canonical member, operand ordinal,
// group encounter ordinal, original keys, RAGS accumulator fragment.
const GROUP: usize = 0;
const AGGREGATE: usize = 1;
const MEMBER: usize = 2;
const ORDINAL: usize = 3;
const ENCOUNTER: usize = 4;
const KEYS: usize = 5;
const STATE: usize = 6;

fn io(error: std::io::Error) -> OperatorError {
    match error.kind() {
        std::io::ErrorKind::QuotaExceeded | std::io::ErrorKind::StorageFull => {
            OperatorError::StorageFull(error.to_string())
        }
        std::io::ErrorKind::OutOfMemory => OperatorError::ResidentAllocation(error.to_string()),
        _ => OperatorError::Execution(error.to_string()),
    }
}
fn invariant(message: &'static str) -> OperatorError {
    OperatorError::ResidentContainerInvariant {
        container: "RDF aggregate packets",
        message,
    }
}
fn numeric_overflow() -> OperatorError {
    grafeo_common::memory::buffer::MemoryGrantError::ArithmeticOverflow {
        current_bytes: usize::MAX,
        additional_bytes: 1,
    }
    .into()
}
fn context(error: QueryResourceContextError) -> OperatorError {
    match error {
        QueryResourceContextError::Memory(error) => error.into(),
        error => OperatorError::Execution(error.to_string()),
    }
}
fn ordinal(value: u64) -> Value {
    Value::Bytes(value.to_be_bytes().to_vec().into())
}
fn read_ordinal(value: &Value) -> Result<u64, OperatorError> {
    let Value::Bytes(bytes) = value else {
        return Err(invariant("invalid ordinal type"));
    };
    let bytes: [u8; 8] = bytes
        .as_ref()
        .try_into()
        .map_err(|_| invariant("invalid ordinal width"))?;
    Ok(u64::from_be_bytes(bytes))
}
fn limits(bytes: usize) -> grafeo_core::execution::spill::CodecLimits {
    grafeo_core::execution::spill::CodecLimits::new(
        bytes,
        bytes / std::mem::size_of::<Value>(),
        PACKET_COLUMNS,
        128,
    )
}
fn build(values: Vec<Value>) -> DataChunk {
    let mut builder = DataChunkBuilder::with_capacity(&vec![LogicalType::Any; values.len()], 1);
    for (index, value) in values.into_iter().enumerate() {
        if let Some(column) = builder.column_mut(index) {
            column.push_value(value);
        }
    }
    builder.advance_row();
    builder.finish()
}

struct Rows {
    child: Box<dyn Operator>,
    chunk: Option<DataChunk>,
    position: usize,
}
impl Rows {
    fn next(&mut self, grant: &mut MemoryGrant) -> Result<Option<Vec<Value>>, OperatorError> {
        loop {
            if let Some(chunk) = &self.chunk
                && self.position < chunk.row_count()
            {
                let logical = self.position;
                self.position += 1;
                let physical = chunk.selection().map_or(logical, |selection| {
                    usize::from(selection.as_slice()[logical])
                });
                let mut retained = 0;
                for column in chunk.columns() {
                    if let Some(value) = column.get_value(physical) {
                        retained = bounded::add(retained, bounded::value_bytes(&value)?)?;
                    }
                }
                let required = bounded::add(1024, bounded::mul(retained, 2)?)?;
                if required > grant.size() {
                    grant.try_resize(required)?;
                }
                return Ok(Some(
                    chunk
                        .columns()
                        .iter()
                        .map(|column| column.get_value(physical).unwrap_or(Value::Null))
                        .collect(),
                ));
            }
            self.chunk = self.child.next()?;
            self.position = 0;
            if self.chunk.is_none() {
                return Ok(None);
            }
        }
    }
}

struct Fragment {
    accumulator: RdfAccumulator,
    member: Option<Value>,
    ordinal: u64,
}

enum Fragments {
    One(Option<RdfAccumulator>, u64),
    Distinct {
        entries: indexmap::map::IntoIter<HashableValue, RdfDistinctEntry>,
        error: bool,
        expression: AggregateExpr,
        ordinal: u64,
        emitted: bool,
    },
    Numeric {
        inputs: std::vec::IntoIter<RdfNumericInput>,
        error: bool,
        function: AggregateFunction,
        ordinal: u64,
    },
    Concat {
        values: std::vec::IntoIter<RdfOrderedString>,
        separator: String,
        error: bool,
        ordinal: u64,
    },
}
impl Fragments {
    fn new(mut accumulator: RdfAccumulator, expression: AggregateExpr, ordinal: u64) -> Self {
        if let Some(distinct) = accumulator.distinct.take() {
            return Self::Distinct {
                entries: distinct.entries.into_iter(),
                error: distinct.unkeyed_error,
                expression,
                ordinal,
                emitted: false,
            };
        }
        match accumulator.state {
            RdfSetFunctionState::Sum(state) | RdfSetFunctionState::Average(state) => {
                Self::Numeric {
                    inputs: state.inputs.into_iter(),
                    error: state.error,
                    function: accumulator.function,
                    ordinal,
                }
            }
            RdfSetFunctionState::GroupConcat {
                values,
                separator,
                error,
            } => Self::Concat {
                values: values.into_iter(),
                separator,
                error,
                ordinal,
            },
            state => Self::One(
                Some(RdfAccumulator {
                    function: accumulator.function,
                    distinct: None,
                    state,
                }),
                ordinal,
            ),
        }
    }
    fn next(&mut self) -> Result<Option<Fragment>, OperatorError> {
        match self {
            Self::One(accumulator, ordinal) => Ok(accumulator.take().map(|accumulator| Fragment {
                accumulator,
                member: None,
                ordinal: *ordinal,
            })),
            Self::Distinct {
                entries,
                error,
                expression,
                ordinal,
                emitted,
            } => {
                if let Some((_, entry)) = entries.next() {
                    *emitted = true;
                    let member = entry.key;
                    let mut accumulator = RdfAccumulator::new(expression);
                    accumulator.update_at(
                        Some(entry.operand),
                        Some(member.clone()),
                        expression.column.is_none().then(|| member.clone()),
                        entry.ordinal,
                    )?;
                    Ok(Some(Fragment {
                        accumulator,
                        member: Some(member),
                        ordinal: entry.ordinal,
                    }))
                } else if std::mem::take(error) {
                    *emitted = true;
                    let mut accumulator = RdfAccumulator::new(expression);
                    accumulator.update_at(None, None, None, *ordinal)?;
                    Ok(Some(Fragment {
                        accumulator,
                        member: None,
                        ordinal: *ordinal,
                    }))
                } else if !std::mem::replace(emitted, true) {
                    Ok(Some(Fragment {
                        accumulator: RdfAccumulator::new(expression),
                        member: None,
                        ordinal: *ordinal,
                    }))
                } else {
                    Ok(None)
                }
            }
            Self::Numeric {
                inputs,
                error,
                function,
                ordinal,
            } => {
                let input = inputs.next();
                if input.is_none() && !*error {
                    return Ok(None);
                }
                let row_ordinal = input.as_ref().map_or(*ordinal, |input| input.ordinal);
                let state = RdfNumericState {
                    inputs: input.into_iter().collect(),
                    error: std::mem::take(error),
                };
                let state = if *function == AggregateFunction::Sum {
                    RdfSetFunctionState::Sum(state)
                } else {
                    RdfSetFunctionState::Average(state)
                };
                Ok(Some(Fragment {
                    accumulator: RdfAccumulator {
                        function: *function,
                        distinct: None,
                        state,
                    },
                    member: None,
                    ordinal: row_ordinal,
                }))
            }
            Self::Concat {
                values,
                separator,
                error,
                ordinal,
            } => {
                let value = values.next();
                if value.is_none() && !*error {
                    return Ok(None);
                }
                let row_ordinal = value.as_ref().map_or(*ordinal, |value| value.ordinal);
                Ok(Some(Fragment {
                    accumulator: RdfAccumulator {
                        function: AggregateFunction::GroupConcat,
                        distinct: None,
                        state: RdfSetFunctionState::GroupConcat {
                            values: value.into_iter().collect(),
                            separator: separator.clone(),
                            error: std::mem::take(error),
                        },
                    },
                    member: None,
                    ordinal: row_ordinal,
                }))
            }
        }
    }
}

struct GroupFragments {
    key: Value,
    keys: Value,
    _key_grant: MemoryGrant,
    encounter: u64,
    accumulators: std::vec::IntoIter<RdfAccumulator>,
    fragments: Option<Fragments>,
    index: usize,
}

pub(super) struct Source {
    child: Rows,
    groups: indexmap::map::IntoIter<Vec<HashableValue>, RdfGroupState>,
    group_index: u64,
    current: Option<GroupFragments>,
    group_columns: Vec<usize>,
    aggregates: Vec<AggregateExpr>,
    identities: Option<Vec<RdfRowIdentityColumn>>,
    next_ordinal: u64,
    resources: QueryResourceContext,
    old_admission: Option<bounded::Admission>,
    packet_grant: MemoryGrant,
    member_grant: Option<MemoryGrant>,
    packet_bytes: usize,
}

impl Source {
    fn next_group(&mut self) -> Result<Option<(RdfGroupState, u64)>, OperatorError> {
        if let Some((_, group)) = self.groups.next() {
            let encounter = self.group_index;
            self.group_index = self
                .group_index
                .checked_add(1)
                .ok_or_else(|| invariant("group ordinal overflow"))?;
            return Ok(Some((group, encounter)));
        }
        self.groups = IndexMap::new().into_iter();
        self.old_admission = None;
        let Some(row) = self.child.next(&mut self.packet_grant)? else {
            return Ok(None);
        };
        let mut input_bytes = 0;
        for value in &row {
            input_bytes = bounded::add(input_bytes, bounded::value_bytes(value)?)?;
        }
        let peak = bounded::add(
            64 * 1024,
            bounded::add(
                bounded::mul(input_bytes, 256)?,
                bounded::mul(self.aggregates.len(), 512)?,
            )?,
        )?;
        if peak > self.packet_grant.size() {
            self.packet_grant.try_resize(peak)?;
        }
        let ordinal = self.next_ordinal;
        self.next_ordinal = self
            .next_ordinal
            .checked_add(1)
            .ok_or_else(|| invariant("input ordinal overflow"))?;
        let key_values = self
            .group_columns
            .iter()
            .map(|index| row.get(*index).cloned().unwrap_or(Value::Null))
            .collect();
        let identity = self.identities.as_ref().map(|columns| {
            Value::List(
                columns
                    .iter()
                    .map(|column| {
                        let visible = row.get(column.visible).cloned().unwrap_or(Value::Null);
                        let canonical = column
                            .canonical_rdf
                            .and_then(|index| row.get(index))
                            .filter(|value| !value.is_null())
                            .cloned();
                        let (canonical, value) =
                            canonical.map_or((false, visible), |value| (true, value));
                        Value::List(vec![Value::Bool(canonical), value].into())
                    })
                    .collect::<Vec<_>>()
                    .into(),
            )
        });
        let mut accumulators: Vec<_> = self.aggregates.iter().map(RdfAccumulator::new).collect();
        for (accumulator, expression) in accumulators.iter_mut().zip(&self.aggregates) {
            accumulator.update_at(
                expression.column.and_then(|index| row.get(index)).cloned(),
                expression
                    .distinct_key_column
                    .and_then(|index| row.get(index))
                    .cloned(),
                expression
                    .column
                    .is_none()
                    .then(|| identity.clone())
                    .flatten(),
                ordinal,
            )?;
        }
        Ok(Some((
            RdfGroupState {
                key_values,
                accumulators,
            },
            ordinal,
        )))
    }
}

impl Operator for Source {
    fn next(&mut self) -> OperatorResult {
        self.resources.check_cancelled()?;
        self.member_grant = None;
        loop {
            if self.current.is_none() {
                let Some((group, encounter)) = self.next_group()? else {
                    self.aggregates = Vec::new();
                    self.group_columns = Vec::new();
                    self.identities = None;
                    self.packet_grant.try_resize(0)?;
                    return Ok(None);
                };
                let canonical = grafeo_core::execution::operators::encode_accounted_semantic_key(
                    &self.resources,
                    &group.key_values,
                )?;
                let (bytes, key_grant) = canonical.into_parts();
                let key = Value::Bytes(bytes.into());
                self.current = Some(GroupFragments {
                    key,
                    _key_grant: key_grant,
                    keys: Value::List(group.key_values.into()),
                    encounter,
                    accumulators: group.accumulators.into_iter(),
                    fragments: None,
                    index: 0,
                });
            }
            let current = self
                .current
                .as_mut()
                .ok_or_else(|| invariant("missing fragment group"))?;
            if current.fragments.is_none() {
                if let Some(accumulator) = current.accumulators.next() {
                    let expression = self
                        .aggregates
                        .get(current.index)
                        .ok_or_else(|| invariant("aggregate width mismatch"))?
                        .clone();
                    current.fragments =
                        Some(Fragments::new(accumulator, expression, current.encounter));
                } else if self.aggregates.is_empty() && current.index == 0 {
                    current.fragments = Some(Fragments::One(
                        Some(RdfAccumulator {
                            function: AggregateFunction::Count,
                            distinct: None,
                            state: RdfSetFunctionState::Count(0),
                        }),
                        current.encounter,
                    ));
                } else {
                    self.current = None;
                    continue;
                }
            }
            let fragment = current
                .fragments
                .as_mut()
                .ok_or_else(|| invariant("missing accumulator fragment"))?
                .next()?;
            let Some(fragment) = fragment else {
                current.fragments = None;
                current.index += 1;
                continue;
            };
            let member = if let Some(member) = &fragment.member {
                let canonical = grafeo_core::execution::operators::encode_accounted_semantic_key(
                    &self.resources,
                    std::slice::from_ref(member),
                )?;
                let (bytes, key_grant) = canonical.into_parts();
                let member = Value::Bytes(bytes.into());
                self.member_grant = Some(key_grant);
                member
            } else {
                Value::Null
            };
            let state = RdfGroupState {
                key_values: Vec::new(),
                accumulators: vec![fragment.accumulator],
            };
            let encoded = state
                .encode_spill(limits(self.packet_bytes / 8))
                .map_err(io)?;
            let packet = vec![
                current.key.clone(),
                Value::Int64(
                    i64::try_from(current.index)
                        .map_err(|_| invariant("aggregate index overflow"))?,
                ),
                member,
                ordinal(fragment.ordinal),
                ordinal(current.encounter),
                current.keys.clone(),
                Value::Bytes(encoded.into()),
            ];
            return Ok(Some(build(packet)));
        }
    }
    fn reset(&mut self) {}
    fn name(&self) -> &'static str {
        "RdfAggregatePartialStates"
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

struct Membership {
    rows: Rows,
    previous: Option<(Value, Value, Value)>,
    resources: QueryResourceContext,
    _grant: MemoryGrant,
}
impl Operator for Membership {
    fn next(&mut self) -> OperatorResult {
        loop {
            self.resources.check_cancelled()?;
            let Some(packet) = self.rows.next(&mut self._grant)? else {
                self.previous = None;
                self._grant.try_resize(0)?;
                return Ok(None);
            };
            if packet.len() != PACKET_COLUMNS {
                return Err(invariant("membership packet width"));
            }
            if !packet[MEMBER].is_null() {
                let duplicate = self
                    .previous
                    .as_ref()
                    .is_some_and(|(group, aggregate, member)| {
                        group == &packet[GROUP]
                            && aggregate == &packet[AGGREGATE]
                            && member == &packet[MEMBER]
                    });
                if duplicate {
                    continue;
                }
                self.previous = Some((
                    packet[GROUP].clone(),
                    packet[AGGREGATE].clone(),
                    packet[MEMBER].clone(),
                ));
            }
            return Ok(Some(build(packet)));
        }
    }
    fn reset(&mut self) {}
    fn name(&self) -> &'static str {
        "RdfAggregateExactMembership"
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

struct FoldAccumulator {
    accumulator: RdfAccumulator,
    sum: Option<RdfNumeric>,
    count: u64,
    numeric_error: bool,
    // GROUP_CONCAT operands arrive in replay order. Retain their payload
    // capacity total so admission never rescans previously folded operands.
    concat_payload_bytes: usize,
}
impl FoldAccumulator {
    fn new(expression: &AggregateExpr) -> Self {
        let mut accumulator = RdfAccumulator::new(expression);
        accumulator.distinct = None;
        Self {
            accumulator,
            sum: None,
            count: 0,
            numeric_error: false,
            concat_payload_bytes: 0,
        }
    }
    fn merge(
        &mut self,
        mut other: RdfAccumulator,
        resources: &QueryResourceContext,
        state_grant: &mut MemoryGrant,
        packet_ordinal: u64,
    ) -> Result<(), OperatorError> {
        if other.function != self.accumulator.function {
            return Err(invariant("RAGS aggregate function mismatch"));
        }
        other.distinct = None;
        if matches!(other.state, RdfSetFunctionState::GroupConcat { .. }) {
            return self.merge_concat(other, resources, state_grant, packet_ordinal);
        }
        let old = bounded::accumulator_retained_bytes(&self.accumulator)?;
        let incoming = bounded::accumulator_retained_bytes(&other)?;
        let peak = bounded::add(1024, bounded::mul(bounded::add(old, incoming)?, 3)?)?;
        if peak > state_grant.size() {
            state_grant.try_resize(peak)?;
        }
        match other.state {
            RdfSetFunctionState::Sum(state) | RdfSetFunctionState::Average(state) => {
                self.numeric_error |= state.error;
                for input in state.inputs {
                    resources.check_cancelled()?;
                    let scratch = self
                        .sum
                        .as_ref()
                        .map_or_else(
                            || input.value.retained_bytes(),
                            |sum| sum.add_scratch_bytes(&input.value),
                        )
                        .ok_or_else(numeric_overflow)?;
                    if scratch > state_grant.size() {
                        state_grant.try_resize(scratch)?;
                    }
                    self.sum = match self.sum.take() {
                        None => Some(input.value),
                        Some(sum) => sum.checked_add(input.value),
                    };
                    if self.sum.is_none() {
                        self.numeric_error = true;
                    }
                    self.count = self
                        .count
                        .checked_add(1)
                        .ok_or_else(|| invariant("numeric count overflow"))?;
                }
                Ok(())
            }
            state => self
                .accumulator
                .merge_from(RdfAccumulator {
                    function: other.function,
                    distinct: None,
                    state,
                })
                .map_err(io),
        }
    }

    fn merge_concat(
        &mut self,
        other: RdfAccumulator,
        resources: &QueryResourceContext,
        state_grant: &mut MemoryGrant,
        packet_ordinal: u64,
    ) -> Result<(), OperatorError> {
        let incoming = bounded::accumulator_retained_bytes(&other)?;
        let RdfSetFunctionState::GroupConcat {
            values,
            separator,
            error,
        } = &mut self.accumulator.state
        else {
            return Err(invariant("GROUP_CONCAT fold state mismatch"));
        };
        let RdfSetFunctionState::GroupConcat {
            values: mut incoming_values,
            separator: incoming_separator,
            error: incoming_error,
        } = other.state
        else {
            return Err(invariant("GROUP_CONCAT fragment state mismatch"));
        };
        if *separator != incoming_separator {
            return Err(invariant("GROUP_CONCAT fragment separator mismatch"));
        }
        // The source currently emits singleton fragments, but validate the
        // decoded packet contract and every operand, including multi-operand
        // fragments. Generic RAGS merging remains free to reorder partials;
        // this consumer receives an explicitly ordinal-sorted replay.
        if incoming_values
            .first()
            .is_some_and(|value| value.ordinal != packet_ordinal)
        {
            return Err(invariant("GROUP_CONCAT packet ordinal mismatch"));
        }
        let mut previous = values.last().map(|value| value.ordinal);
        let mut payload_bytes = self.concat_payload_bytes;
        for value in &incoming_values {
            resources.check_cancelled()?;
            #[cfg(test)]
            bounded::record_concat_operand_visit();
            if previous.is_some_and(|ordinal| ordinal >= value.ordinal) {
                return Err(invariant(
                    "GROUP_CONCAT replay ordinals are not strictly increasing",
                ));
            }
            previous = Some(value.ordinal);
            payload_bytes = bounded::add(payload_bytes, value.value.capacity())?;
        }
        let old = bounded::add(
            std::mem::size_of::<RdfAccumulator>(),
            bounded::add(
                bounded::mul(values.capacity(), std::mem::size_of::<RdfOrderedString>())?,
                bounded::add(separator.capacity(), self.concat_payload_bytes)?,
            )?,
        )?;
        // The same envelope covers old/new vector allocations, both fragments
        // and payload owners while capacity grows geometrically.
        let peak = bounded::add(1024, bounded::mul(bounded::add(old, incoming)?, 3)?)?;
        resources.check_cancelled()?;
        if peak > state_grant.size() {
            state_grant.try_resize(peak)?;
        }
        let required = bounded::add(values.len(), incoming_values.len())?;
        if required > values.capacity() {
            let capacity = bounded::mul(values.capacity().max(4), 2)?.max(required);
            values
                .try_reserve_exact(capacity - values.len())
                .map_err(|error| {
                    OperatorError::ResidentAllocation(format!(
                        "cannot reserve ordered RDF GROUP_CONCAT values: {error}"
                    ))
                })?;
            if values.capacity() > capacity {
                return Err(invariant("GROUP_CONCAT vector exceeded admitted capacity"));
            }
        }
        values.append(&mut incoming_values);
        self.concat_payload_bytes = payload_bytes;
        *error |= incoming_error;
        Ok(())
    }
    fn finish(
        self,
        resources: &QueryResourceContext,
        state_grant: &mut MemoryGrant,
    ) -> Result<Value, OperatorError> {
        if matches!(
            self.accumulator.function,
            AggregateFunction::Sum | AggregateFunction::Avg
        ) {
            if self.numeric_error {
                return Ok(Value::Null);
            }
            let Some(sum) = self.sum else {
                return Ok(Value::Int64(0));
            };
            let average =
                (self.accumulator.function == AggregateFunction::Avg).then_some(self.count);
            let scratch = sum
                .finalize_scratch_bytes(average)
                .ok_or_else(numeric_overflow)?;
            resources.check_cancelled()?;
            if scratch > state_grant.size() {
                state_grant.try_resize(scratch)?;
            }
            return Ok(if average.is_some() {
                sum.average(self.count)
                    .map_or(Value::Null, RdfNumeric::into_value)
            } else {
                sum.into_value()
            });
        }
        let final_peak = bounded::add(
            bounded::accumulator_retained_bytes(&self.accumulator)?,
            bounded::accumulator_finalize_peak(&self.accumulator)?,
        )?;
        if final_peak > state_grant.size() {
            state_grant.try_resize(final_peak)?;
        }
        Ok(self.accumulator.finalize())
    }
}

fn push_output(
    output: &mut Vec<Value>,
    value: Value,
    grant: &mut MemoryGrant,
) -> Result<(), OperatorError> {
    let mut bytes = bounded::value_bytes(&value)?;
    for previous in output.iter() {
        bytes = bounded::add(bytes, bounded::value_bytes(previous)?)?;
    }
    bytes = bounded::add(
        bounded::mul(bytes, 2)?,
        bounded::mul(output.len() + 1, std::mem::size_of::<Value>() * 4)?,
    )?;
    if bytes > grant.size() {
        grant.try_resize(bytes)?;
    }
    output.push(value);
    Ok(())
}

struct Fold {
    rows: Rows,
    pending: Option<Vec<Value>>,
    expressions: Vec<AggregateExpr>,
    resources: QueryResourceContext,
    _grant: MemoryGrant,
    state_grant: MemoryGrant,
    packet_bytes: usize,
}
impl Operator for Fold {
    fn next(&mut self) -> OperatorResult {
        self.resources.check_cancelled()?;
        let first = match self.pending.take() {
            Some(packet) => packet,
            None => match self.rows.next(&mut self._grant)? {
                Some(packet) => packet,
                None => {
                    self.expressions = Vec::new();
                    self.state_grant.try_resize(0)?;
                    self._grant.try_resize(0)?;
                    return Ok(None);
                }
            },
        };
        if first.len() != PACKET_COLUMNS {
            return Err(invariant("fold packet width"));
        }
        let group = first[GROUP].clone();
        let Value::List(keys) = &first[KEYS] else {
            return Err(invariant("group witness type"));
        };
        let mut output = keys.to_vec();
        let mut encounter = read_ordinal(&first[ENCOUNTER])?;
        let mut packet = Some(first);
        let mut index = 0;
        let mut accumulator = self.expressions.first().map(FoldAccumulator::new);
        loop {
            let Some(current) = packet.take() else {
                break;
            };
            if current.len() != PACKET_COLUMNS {
                return Err(invariant("fold packet width"));
            }
            if current[GROUP] != group {
                self.pending = Some(current);
                break;
            }
            encounter = encounter.min(read_ordinal(&current[ENCOUNTER])?);
            let Value::Int64(aggregate) = current[AGGREGATE] else {
                return Err(invariant("aggregate index type"));
            };
            let aggregate =
                usize::try_from(aggregate).map_err(|_| invariant("negative aggregate index"))?;
            while index < aggregate {
                let finished = accumulator
                    .take()
                    .ok_or_else(|| invariant("missing fold accumulator"))?
                    .finish(&self.resources, &mut self.state_grant)?;
                push_output(&mut output, finished, &mut self._grant)?;
                index += 1;
                accumulator = self.expressions.get(index).map(FoldAccumulator::new);
            }
            let Value::Bytes(encoded) = &current[STATE] else {
                return Err(invariant("RAGS packet type"));
            };
            let decode_peak = bounded::add(64 * 1024, bounded::mul(encoded.len(), 512)?)?;
            let _decode_grant = self.resources.try_allocate(decode_peak).map_err(context)?;
            let mut decoded =
                RdfGroupState::decode_spill(encoded, limits(self.packet_bytes / 8)).map_err(io)?;
            if !decoded.key_values.is_empty() || decoded.accumulators.len() != 1 {
                return Err(invariant("RAGS fragment shape"));
            }
            let state = decoded
                .accumulators
                .pop()
                .ok_or_else(|| invariant("missing RAGS accumulator"))?;
            if self.expressions.is_empty() {
                if aggregate != 0
                    || state.function != AggregateFunction::Count
                    || state.distinct.is_some()
                    || !matches!(state.state, RdfSetFunctionState::Count(0))
                {
                    return Err(invariant("invalid group-only sentinel"));
                }
            } else {
                accumulator
                    .as_mut()
                    .ok_or_else(|| invariant("aggregate index exceeds expressions"))?
                    .merge(
                        state,
                        &self.resources,
                        &mut self.state_grant,
                        read_ordinal(&current[ORDINAL])?,
                    )?;
            }
            packet = self.rows.next(&mut self._grant)?;
        }
        while index < self.expressions.len() {
            let value = accumulator
                .take()
                .ok_or_else(|| invariant("missing final fold"))?
                .finish(&self.resources, &mut self.state_grant)?;
            push_output(&mut output, value, &mut self._grant)?;
            index += 1;
            accumulator = self.expressions.get(index).map(FoldAccumulator::new);
        }
        Ok(Some(build(vec![
            ordinal(encounter),
            Value::List(output.into()),
        ])))
    }
    fn reset(&mut self) {}
    fn name(&self) -> &'static str {
        "RdfAggregateOrdinalFold"
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

struct Output {
    rows: Rows,
    resources: QueryResourceContext,
    _grant: MemoryGrant,
}
impl Operator for Output {
    fn next(&mut self) -> OperatorResult {
        self.resources.check_cancelled()?;
        let Some(row) = self.rows.next(&mut self._grant)? else {
            self._grant.try_resize(0)?;
            return Ok(None);
        };
        let Some(Value::List(values)) = row.get(1) else {
            return Err(invariant("aggregate output witness type"));
        };
        Ok(Some(build(values.to_vec())))
    }
    fn reset(&mut self) {}
    fn name(&self) -> &'static str {
        "RdfAggregateSpilledOutput"
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

struct PacketComparator;
impl grafeo_core::execution::operators::AccountedValueComparator for PacketComparator {
    fn scratch_bytes(
        &self,
        _: Option<&Value>,
        _: Option<&Value>,
    ) -> Result<usize, grafeo_core::execution::operators::SemanticComparisonError> {
        Ok(0)
    }
    fn compare(
        &self,
        left: Option<&Value>,
        right: Option<&Value>,
    ) -> Result<Ordering, grafeo_core::execution::operators::SemanticComparisonError> {
        use grafeo_core::execution::operators::SemanticComparisonError;
        match (left, right) {
            (Some(Value::Bytes(left)), Some(Value::Bytes(right))) => {
                Ok(left.as_ref().cmp(right.as_ref()))
            }
            (Some(Value::Int64(left)), Some(Value::Int64(right))) => Ok(left.cmp(right)),
            (None | Some(Value::Null), None | Some(Value::Null)) => Ok(Ordering::Equal),
            (None | Some(Value::Null), _) => Ok(Ordering::Less),
            (_, None | Some(Value::Null)) => Ok(Ordering::Greater),
            _ => Err(SemanticComparisonError::Invalid(
                "invalid aggregate sort key",
            )),
        }
    }
}

fn sorted(
    child: Box<dyn Operator>,
    columns: &[usize],
    width: usize,
    resources: &QueryResourceContext,
) -> Result<Box<dyn Operator>, OperatorError> {
    let mut sort = SortOperator::new(
        child,
        columns
            .iter()
            .map(|column| SortKey::ascending(*column))
            .collect(),
        vec![LogicalType::Any; width],
    )
    .with_semantic_comparator(Arc::new(PacketComparator));
    sort.install_resource_context(resources).map_err(context)?;
    Ok(Box::new(sort))
}

struct SourceChild(Arc<parking_lot::Mutex<Box<dyn Operator>>>);
impl Operator for SourceChild {
    fn next(&mut self) -> OperatorResult {
        self.0.lock().next()
    }
    fn reset(&mut self) {
        self.0.lock().reset();
    }
    fn install_resource_context(
        &mut self,
        resources: &QueryResourceContext,
    ) -> Result<(), QueryResourceContextError> {
        self.0.lock().install_resource_context(resources)
    }
    fn name(&self) -> &'static str {
        "RdfAggregateSharedInput"
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}
struct Empty;
impl Operator for Empty {
    fn next(&mut self) -> OperatorResult {
        Ok(None)
    }
    fn reset(&mut self) {}
    fn name(&self) -> &'static str {
        "RdfAggregateInputTransfer"
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

pub(super) fn start(
    parent: &mut RdfAggregateOperator,
    chunk: DataChunk,
    position: usize,
    ordinal: u64,
) -> Result<Box<dyn Operator>, OperatorError> {
    let resources = parent
        .admission
        .as_ref()
        .ok_or_else(|| invariant("spill without resources"))?
        .resources
        .clone();
    let packet_bytes = (resources.query_stats().growth_limit_bytes / 32).max(4096);
    // Reserve all pipeline-owned packet/frontier metadata before constructing
    // the wrappers. Native sort operators own their separate resident/run grants.
    let source_grant = resources.try_allocate(packet_bytes).map_err(context)?;
    let membership_grant = if parent
        .aggregates
        .iter()
        .any(|expression| expression.distinct)
    {
        Some(resources.try_allocate(packet_bytes).map_err(context)?)
    } else {
        None
    };
    let fold_grant = resources.try_allocate(packet_bytes).map_err(context)?;
    let output_grant = resources.try_allocate(packet_bytes).map_err(context)?;
    let expressions = parent.aggregates.clone();
    let shared = if let Some(shared) = &parent.shared_input {
        Arc::clone(shared)
    } else {
        let original = std::mem::replace(&mut parent.child, Box::new(Empty));
        let shared = Arc::new(parking_lot::Mutex::new(original));
        parent.child = Box::new(SourceChild(Arc::clone(&shared)));
        parent.shared_input = Some(Arc::clone(&shared));
        shared
    };
    let source = Source {
        child: Rows {
            child: Box::new(SourceChild(shared)),
            chunk: Some(chunk),
            position,
        },
        groups: std::mem::take(&mut parent.groups).into_iter(),
        group_index: 0,
        current: None,
        group_columns: parent.group_columns.clone(),
        aggregates: expressions.clone(),
        identities: parent.row_identity_columns.clone(),
        next_ordinal: ordinal,
        resources: resources.clone(),
        old_admission: parent.admission.take(),
        packet_grant: source_grant,
        member_grant: None,
        packet_bytes,
    };
    // Without DISTINCT every MEMBER is NULL: membership filtering is an
    // identity operation and its sort has the same order as ordinal replay.
    // Mixed queries retain the full membership path for exact deduplication.
    let replay_input: Box<dyn Operator> = if let Some(grant) = membership_grant {
        let ordered = sorted(
            Box::new(source),
            &[GROUP, AGGREGATE, MEMBER, ORDINAL],
            PACKET_COLUMNS,
            &resources,
        )?;
        Box::new(Membership {
            rows: Rows {
                child: ordered,
                chunk: None,
                position: 0,
            },
            previous: None,
            resources: resources.clone(),
            _grant: grant,
        })
    } else {
        Box::new(source)
    };
    let replay = sorted(
        replay_input,
        &[GROUP, AGGREGATE, ORDINAL],
        PACKET_COLUMNS,
        &resources,
    )?;
    let fold = Fold {
        rows: Rows {
            child: replay,
            chunk: None,
            position: 0,
        },
        pending: None,
        expressions,
        resources: resources.clone(),
        _grant: fold_grant,
        state_grant: resources.try_allocate(0).map_err(context)?,
        packet_bytes,
    };
    let encounter = sorted(Box::new(fold), &[0], 2, &resources)?;
    Ok(Box::new(Output {
        rows: Rows {
            child: encounter,
            chunk: None,
            position: 0,
        },
        resources,
        _grant: output_grant,
    }))
}

#[cfg(test)]
mod concat_fold_tests {
    use super::*;
    use grafeo_common::memory::buffer::BufferManager;
    use grafeo_core::execution::QueryExecutionControl;

    fn expression() -> AggregateExpr {
        let mut expression = AggregateExpr::count(0);
        expression.function = AggregateFunction::GroupConcat;
        expression.separator = Some("|".to_string());
        expression
    }

    fn fragment(operands: &[(u64, &str)], error: bool) -> RdfAccumulator {
        RdfAccumulator {
            function: AggregateFunction::GroupConcat,
            distinct: None,
            state: RdfSetFunctionState::GroupConcat {
                values: operands
                    .iter()
                    .map(|(ordinal, value)| RdfOrderedString {
                        ordinal: *ordinal,
                        value: (*value).to_string(),
                    })
                    .collect(),
                separator: "|".to_string(),
                error,
            },
        }
    }

    #[test]
    fn ordered_concat_fold_accepts_compound_and_empty_fragments() {
        let memory = BufferManager::with_budget(1 << 20);
        let resources = QueryResourceContext::new(memory.clone()).unwrap();
        for error_fragment in [false, true] {
            let mut fold = FoldAccumulator::new(&expression());
            let mut grant = resources.try_allocate(0).unwrap();
            // Round-trip a compound RAGS state: replay admission must validate
            // decoded operands, not depend on the source emitting singletons.
            let encoded = RdfGroupState {
                key_values: Vec::new(),
                accumulators: vec![fragment(&[(2, "a"), (3, "b")], false)],
            }
            .encode_spill(limits(64 << 10))
            .unwrap();
            let mut decoded = RdfGroupState::decode_spill(&encoded, limits(64 << 10)).unwrap();
            fold.merge(
                decoded.accumulators.pop().unwrap(),
                &resources,
                &mut grant,
                2,
            )
            .unwrap();
            fold.merge(fragment(&[(4, "c")], false), &resources, &mut grant, 4)
                .unwrap();
            // Empty/error fragments have no operand ordinal to compare with
            // prior output, and must retain their exact error semantics.
            fold.merge(fragment(&[], error_fragment), &resources, &mut grant, 0)
                .unwrap();
            let RdfSetFunctionState::GroupConcat { values, .. } = &fold.accumulator.state else {
                panic!("concat state");
            };
            assert_eq!(
                fold.concat_payload_bytes,
                values
                    .iter()
                    .map(|value| value.value.capacity())
                    .sum::<usize>()
            );
            assert!(
                grant.size() >= bounded::accumulator_retained_bytes(&fold.accumulator).unwrap()
            );
            assert_eq!(
                fold.finish(&resources, &mut grant).unwrap(),
                if error_fragment {
                    Value::Null
                } else {
                    Value::from("a|b|c")
                }
            );
            drop(grant);
            assert_eq!(memory.allocated(), 0);
        }
        let fold = FoldAccumulator::new(&expression());
        let mut grant = resources.try_allocate(0).unwrap();
        assert_eq!(
            fold.finish(&resources, &mut grant).unwrap(),
            Value::from("")
        );
        drop(grant);
        assert_eq!(memory.allocated(), 0);
    }

    #[test]
    fn ordered_concat_fold_rejects_invalid_boundaries_before_mutation() {
        let memory = BufferManager::with_budget(1 << 20);
        let resources = QueryResourceContext::new(memory.clone()).unwrap();
        let mut wrong_separator = fragment(&[(11, "b")], false);
        if let RdfSetFunctionState::GroupConcat { separator, .. } = &mut wrong_separator.state {
            *separator = ",".to_string();
        }
        for (incoming, packet_ordinal) in [
            (fragment(&[(10, "duplicate")], false), 10),
            (fragment(&[(9, "backwards")], false), 9),
            (fragment(&[(11, "b"), (11, "duplicate")], false), 11),
            (fragment(&[(12, "c"), (11, "backwards")], false), 12),
            (fragment(&[(11, "wrong packet")], false), 99),
            (wrong_separator, 11),
        ] {
            let mut fold = FoldAccumulator::new(&expression());
            let mut grant = resources.try_allocate(0).unwrap();
            fold.merge(fragment(&[(10, "a")], false), &resources, &mut grant, 10)
                .unwrap();
            let granted_before = grant.size();
            let payload_before = fold.concat_payload_bytes;
            let error = fold
                .merge(incoming, &resources, &mut grant, packet_ordinal)
                .unwrap_err();
            assert!(matches!(
                error,
                OperatorError::ResidentContainerInvariant { .. }
            ));
            assert_eq!(grant.size(), granted_before);
            assert_eq!(fold.concat_payload_bytes, payload_before);
            assert_eq!(
                fold.finish(&resources, &mut grant).unwrap(),
                Value::from("a")
            );
            drop(grant);
            assert_eq!(memory.allocated(), 0);
        }
    }

    #[test]
    fn ordered_concat_fold_denial_and_cancellation_release_grants() {
        for cancelled in [false, true] {
            let memory = BufferManager::with_budget(if cancelled { 1 << 20 } else { 256 });
            let control = QueryExecutionControl::new();
            let resources =
                QueryResourceContext::new_with_cancellation(memory.clone(), control.token())
                    .unwrap();
            let mut grant = resources.try_allocate(0).unwrap();
            let mut fold = FoldAccumulator::new(&expression());
            if cancelled {
                control.cancellation_handle().cancel();
            }
            let error = fold
                .merge(
                    fragment(&[(0, "a"), (1, "b")], false),
                    &resources,
                    &mut grant,
                    0,
                )
                .unwrap_err();
            if cancelled {
                assert!(matches!(error, OperatorError::QueryCancelled(_)), "{error}");
            } else {
                assert!(matches!(error, OperatorError::ResidentMemory(_)), "{error}");
            }
            assert_eq!(fold.concat_payload_bytes, 0);
            drop(fold);
            drop(grant);
            assert_eq!(memory.allocated(), 0);
        }
    }
}
