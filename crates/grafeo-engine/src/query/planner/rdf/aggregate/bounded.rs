//! Query-owned allocation envelopes for RDF aggregate state and output.

use super::{
    AggregateExpr, OperatorError, QueryResourceContext, QueryResourceContextError, RdfAccumulator,
    RdfDistinctEntry, RdfGroupState, RdfNumericInput, RdfOrderedString, RdfSetFunctionState, Value,
};
use grafeo_common::memory::buffer::{MemoryGrant, MemoryGrantError};
use grafeo_common::types::HashableValue;
use grafeo_core::execution::DataChunk;
use std::mem::size_of;

#[cfg(test)]
thread_local! {
    static CONCAT_OPERAND_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn record_concat_operand_visit() {
    CONCAT_OPERAND_VISITS.with(|visits| visits.set(visits.get() + 1));
}

#[cfg(all(test, feature = "spill"))]
pub(super) fn take_concat_operand_visits() -> usize {
    CONCAT_OPERAND_VISITS.with(|visits| visits.replace(0))
}

pub(super) struct Admission {
    pub resources: QueryResourceContext,
    pub state: MemoryGrant,
    scratch: MemoryGrant,
    output: MemoryGrant,
    rejected_transfer: Option<MemoryGrant>,
}

pub(super) fn add(left: usize, right: usize) -> Result<usize, OperatorError> {
    left.checked_add(right).ok_or_else(|| {
        MemoryGrantError::ArithmeticOverflow {
            current_bytes: left,
            additional_bytes: right,
        }
        .into()
    })
}

pub(super) fn mul(left: usize, right: usize) -> Result<usize, OperatorError> {
    left.checked_mul(right).ok_or_else(|| {
        MemoryGrantError::ArithmeticOverflow {
            current_bytes: left,
            additional_bytes: right,
        }
        .into()
    })
}

pub(super) fn value_bytes(value: &Value) -> Result<usize, OperatorError> {
    value.retained_size_bytes().ok_or_else(|| {
        MemoryGrantError::ArithmeticOverflow {
            current_bytes: usize::MAX,
            additional_bytes: 1,
        }
        .into()
    })
}

impl Admission {
    pub fn new(resources: &QueryResourceContext) -> Result<Self, QueryResourceContextError> {
        Ok(Self {
            resources: resources.clone(),
            state: resources.try_allocate(0)?,
            scratch: resources.try_allocate(0)?,
            output: resources.try_allocate(0)?,
            rejected_transfer: None,
        })
    }

    pub fn check(&self) -> Result<(), OperatorError> {
        self.resources
            .cancellation_token()
            .check()
            .map_err(Into::into)
    }

    pub fn prepare_row(&mut self, chunk: &DataChunk, row: usize) -> Result<usize, OperatorError> {
        self.check()?;
        let mut bytes = 0;
        for column in chunk.columns() {
            if let Some(value) = column.get_value(row) {
                bytes = add(bytes, value_bytes(&value)?)?;
            }
        }
        // Tagged-term decoding, exact numeric parsing, hash-key companions,
        // and simultaneous old/new allocations fit under the same envelope.
        let scratch = add(64 * 1024, mul(bytes, 256)?)?;
        self.scratch.try_resize(scratch)?;
        Ok(bytes)
    }

    #[cfg(test)]
    pub fn resize_state(&mut self, bytes: usize) -> Result<(), OperatorError> {
        self.state.try_resize(bytes)?;
        Ok(())
    }

    /// Transfers already admitted update storage without charging it twice.
    /// On any invariant failure all grants remain owned until state cleanup.
    pub fn adopt_updated_state(&mut self, total_bytes: usize) -> Result<(), OperatorError> {
        if self.rejected_transfer.is_some() {
            return Err(OperatorError::ResidentContainerInvariant {
                container: "RDF aggregate admission",
                message: "a failed grant transfer is still awaiting cleanup",
            });
        }
        let current = self.state.size();
        if total_bytes <= current {
            self.state.try_resize(total_bytes)?;
            return Ok(());
        }
        let transfer = self.scratch.split(total_bytes - current).ok_or(
            OperatorError::ResidentContainerInvariant {
                container: "RDF aggregate admission",
                message: "updated state exceeds its preadmitted update storage",
            },
        )?;
        if let Err(transfer) = self.state.try_merge(transfer) {
            self.rejected_transfer = Some(transfer);
            return Err(OperatorError::ResidentContainerInvariant {
                container: "RDF aggregate admission",
                message: "update storage belongs to a different allocation account",
            });
        }
        Ok(())
    }

    pub fn set_scratch(&mut self, bytes: usize) -> Result<(), OperatorError> {
        self.check()?;
        self.scratch.try_resize(bytes)?;
        Ok(())
    }

    pub fn grow(&mut self, bytes: usize) -> Result<(), OperatorError> {
        let next = add(self.state.size(), bytes)?;
        self.state.try_resize(next)?;
        Ok(())
    }

    pub fn release_scratch(&mut self) -> Result<(), OperatorError> {
        self.scratch.try_resize(0)?;
        Ok(())
    }

    pub fn output(&mut self, bytes: usize) -> Result<(), OperatorError> {
        self.check()?;
        self.output.try_resize(bytes)?;
        Ok(())
    }
}

fn bound(bytes: Option<usize>) -> Result<usize, OperatorError> {
    bytes.ok_or_else(|| {
        MemoryGrantError::ArithmeticOverflow {
            current_bytes: usize::MAX,
            additional_bytes: 1,
        }
        .into()
    })
}

fn unsupported_state() -> OperatorError {
    OperatorError::ResidentContainerInvariant {
        container: "RDF aggregate state",
        message: "non-SPARQL accumulator has no qualified retained-storage bound",
    }
}

/// Includes every physically retained vector slot and owned pointee. Shared
/// Value payloads are deliberately counted for each owner. The caller retains
/// this grant through iterator/fragment transfer, including partially consumed
/// vector capacity, rather than reducing it to the number of remaining items.
#[cfg(test)]
pub(super) fn group_retained_bytes(group: &RdfGroupState) -> Result<usize, OperatorError> {
    let mut bytes = add(
        size_of::<RdfGroupState>(),
        mul(group.key_values.capacity(), size_of::<Value>())?,
    )?;
    for value in &group.key_values {
        bytes = add(bytes, value_bytes(value)?)?;
    }
    bytes = add(
        bytes,
        mul(group.accumulators.capacity(), size_of::<RdfAccumulator>())?,
    )?;
    for accumulator in &group.accumulators {
        let heap = accumulator_retained_bytes(accumulator)?
            .checked_sub(size_of::<RdfAccumulator>())
            .ok_or_else(unsupported_state)?;
        bytes = add(bytes, heap)?;
    }
    Ok(bytes)
}

#[cfg(any(test, feature = "spill"))]
pub(super) fn accumulator_retained_bytes(
    accumulator: &RdfAccumulator,
) -> Result<usize, OperatorError> {
    let mut bytes = size_of::<RdfAccumulator>();
    if let Some(distinct) = &accumulator.distinct {
        // IndexMap's entry vector and hash index grow independently. The
        // second capacity covers rounding of either backing allocation; hash
        // slots include their control byte and alignment allowance.
        let slot = add(size_of::<(HashableValue, RdfDistinctEntry)>(), 32)?;
        bytes = add(bytes, mul(distinct.entries.capacity(), mul(slot, 2)?)?)?;
        for (key, entry) in &distinct.entries {
            bytes = add(bytes, value_bytes(&key.0)?)?;
            bytes = add(bytes, value_bytes(&entry.key)?)?;
            bytes = add(bytes, value_bytes(&entry.operand)?)?;
        }
    }
    let state = match &accumulator.state {
        RdfSetFunctionState::Count(_) => 0,
        RdfSetFunctionState::Sum(state) | RdfSetFunctionState::Average(state) => {
            let mut bytes = mul(state.inputs.capacity(), size_of::<RdfNumericInput>())?;
            for input in &state.inputs {
                bytes = add(bytes, bound(input.value.retained_bytes())?)?;
            }
            bytes
        }
        RdfSetFunctionState::Minimum { selected, .. }
        | RdfSetFunctionState::Maximum(selected)
        | RdfSetFunctionState::Sample(selected) => selected
            .as_ref()
            .map_or(Ok(0), |selected| value_bytes(&selected.value))?,
        RdfSetFunctionState::GroupConcat {
            values, separator, ..
        } => {
            let mut bytes = add(
                mul(values.capacity(), size_of::<RdfOrderedString>())?,
                separator.capacity(),
            )?;
            for value in values {
                #[cfg(test)]
                record_concat_operand_visit();
                bytes = add(bytes, value.value.capacity())?;
            }
            bytes
        }
        RdfSetFunctionState::Generic(_) => return Err(unsupported_state()),
    };
    add(bytes, state)
}

#[derive(Clone, Copy)]
struct AccumulatorSnapshot {
    capacity_bytes: usize,
    selected_bytes: usize,
    numeric_len: usize,
    concat_len: usize,
    distinct_len: usize,
}

pub(super) struct GroupSnapshot {
    accumulators: Vec<AccumulatorSnapshot>,
}

fn accumulator_snapshot(
    accumulator: &RdfAccumulator,
) -> Result<AccumulatorSnapshot, OperatorError> {
    let mut result = AccumulatorSnapshot {
        capacity_bytes: 0,
        selected_bytes: 0,
        numeric_len: 0,
        concat_len: 0,
        distinct_len: 0,
    };
    if let Some(distinct) = &accumulator.distinct {
        let slot = add(size_of::<(HashableValue, RdfDistinctEntry)>(), 32)?;
        result.capacity_bytes = mul(distinct.entries.capacity(), mul(slot, 2)?)?;
        result.distinct_len = distinct.entries.len();
    }
    match &accumulator.state {
        RdfSetFunctionState::Sum(state) | RdfSetFunctionState::Average(state) => {
            result.capacity_bytes = add(
                result.capacity_bytes,
                mul(state.inputs.capacity(), size_of::<RdfNumericInput>())?,
            )?;
            result.numeric_len = state.inputs.len();
        }
        RdfSetFunctionState::GroupConcat { values, .. } => {
            result.capacity_bytes = add(
                result.capacity_bytes,
                mul(values.capacity(), size_of::<RdfOrderedString>())?,
            )?;
            result.concat_len = values.len();
        }
        RdfSetFunctionState::Minimum { selected, .. }
        | RdfSetFunctionState::Maximum(selected)
        | RdfSetFunctionState::Sample(selected) => {
            result.selected_bytes = selected.as_ref().map_or(Ok(0), |v| value_bytes(&v.value))?;
        }
        RdfSetFunctionState::Count(_) => {}
        RdfSetFunctionState::Generic(_) => return Err(unsupported_state()),
    }
    Ok(result)
}

/// Snapshot storage is included in group_update_peak and must be admitted
/// before this call. Work is proportional to the number of aggregate columns,
/// never to the number of previously ingested operands or distinct members.
pub(super) fn snapshot_group(group: &RdfGroupState) -> Result<GroupSnapshot, OperatorError> {
    let mut accumulators = Vec::new();
    accumulators
        .try_reserve_exact(group.accumulators.len())
        .map_err(|error| {
            OperatorError::ResidentAllocation(format!(
                "cannot reserve RDF aggregate accounting snapshot: {error}"
            ))
        })?;
    if accumulators.capacity() > group.accumulators.len() {
        return Err(OperatorError::ResidentContainerInvariant {
            container: "RDF aggregate snapshot",
            message: "allocator exceeded admitted snapshot capacity",
        });
    }
    for accumulator in &group.accumulators {
        accumulators.push(accumulator_snapshot(accumulator)?);
    }
    Ok(GroupSnapshot { accumulators })
}

fn appended(previous: usize, current: usize) -> Result<bool, OperatorError> {
    match current.checked_sub(previous) {
        Some(0) => Ok(false),
        Some(1) => Ok(true),
        _ => Err(OperatorError::ResidentContainerInvariant {
            container: "RDF aggregate accounting",
            message: "resident update must append at most one ordered operand",
        }),
    }
}

/// Returns old/new changed storage only, for `total - old + new`. The resident
/// caller supplies strictly increasing ordinals; DISTINCT therefore never
/// replaces an earlier witness or rebuilds the already accumulated state.
pub(super) fn updated_group_delta(
    group: &RdfGroupState,
    snapshot: GroupSnapshot,
) -> Result<(usize, usize), OperatorError> {
    if group.accumulators.len() != snapshot.accumulators.len() {
        return Err(unsupported_state());
    }
    let mut old_bytes = 0;
    let mut new_bytes = 0;
    for (accumulator, before) in group.accumulators.iter().zip(snapshot.accumulators) {
        let after = accumulator_snapshot(accumulator)?;
        old_bytes = add(
            old_bytes,
            add(before.capacity_bytes, before.selected_bytes)?,
        )?;
        new_bytes = add(new_bytes, add(after.capacity_bytes, after.selected_bytes)?)?;
        if appended(before.distinct_len, after.distinct_len)? {
            let (key, entry) = accumulator
                .distinct
                .as_ref()
                .and_then(|state| state.entries.last())
                .ok_or_else(unsupported_state)?;
            new_bytes = add(new_bytes, value_bytes(&key.0)?)?;
            new_bytes = add(new_bytes, value_bytes(&entry.key)?)?;
            new_bytes = add(new_bytes, value_bytes(&entry.operand)?)?;
        }
        if appended(before.numeric_len, after.numeric_len)? {
            let (RdfSetFunctionState::Sum(state) | RdfSetFunctionState::Average(state)) =
                &accumulator.state
            else {
                return Err(unsupported_state());
            };
            let input = state.inputs.last().ok_or_else(unsupported_state)?;
            new_bytes = add(new_bytes, bound(input.value.retained_bytes())?)?;
        }
        if appended(before.concat_len, after.concat_len)? {
            let RdfSetFunctionState::GroupConcat { values, .. } = &accumulator.state else {
                return Err(unsupported_state());
            };
            new_bytes = add(
                new_bytes,
                values
                    .last()
                    .ok_or_else(unsupported_state)?
                    .value
                    .capacity(),
            )?;
        }
    }
    Ok((old_bytes, new_bytes))
}

/// Additional simultaneous storage for one update, including a replacement
/// container while its old allocation remains live. The caller reserves this
/// before mutation, then transfers the resulting retained increment to state
/// before releasing the temporary reservation.
pub(super) fn group_update_peak(
    group: &RdfGroupState,
    input_bytes: usize,
) -> Result<usize, OperatorError> {
    let mut bytes = add(64 * 1024, mul(input_bytes, 256)?)?;
    bytes = add(
        bytes,
        mul(group.accumulators.len(), size_of::<AccumulatorSnapshot>())?,
    )?;
    for accumulator in &group.accumulators {
        // Newly parsed coefficient, decoded operand, and immutable clone
        // owners survive together for this accumulator's update.
        bytes = add(bytes, add(mul(input_bytes, 8)?, 256)?)?;
        if let Some(distinct) = &accumulator.distinct {
            let slot = add(size_of::<(HashableValue, RdfDistinctEntry)>(), 32)?;
            if distinct.entries.len() == distinct.entries.capacity() {
                bytes = add(
                    bytes,
                    mul(distinct.entries.capacity().max(4), mul(slot, 4)?)?,
                )?;
            }
            bytes = add(bytes, mul(input_bytes, 3)?)?;
        }
        match &accumulator.state {
            RdfSetFunctionState::Sum(state) | RdfSetFunctionState::Average(state) => {
                if state.inputs.len() == state.inputs.capacity() {
                    bytes = add(
                        bytes,
                        mul(
                            state.inputs.capacity().max(4),
                            size_of::<RdfNumericInput>() * 2,
                        )?,
                    )?;
                }
            }
            RdfSetFunctionState::GroupConcat { values, .. } => {
                if values.len() == values.capacity() {
                    bytes = add(
                        bytes,
                        mul(values.capacity().max(4), size_of::<RdfOrderedString>() * 2)?,
                    )?;
                }
                bytes = add(bytes, mul(input_bytes, 2)?)?;
            }
            RdfSetFunctionState::Minimum { selected, .. }
            | RdfSetFunctionState::Maximum(selected)
            | RdfSetFunctionState::Sample(selected) => {
                if let Some(selected) = selected {
                    bytes = add(bytes, mul(value_bytes(&selected.value)?, 256)?)?;
                }
            }
            RdfSetFunctionState::Count(_) => {}
            RdfSetFunctionState::Generic(_) => return Err(unsupported_state()),
        }
    }
    Ok(bytes)
}

pub(super) fn group_finalize_peak(group: &RdfGroupState) -> Result<usize, OperatorError> {
    let mut bytes = mul(
        add(group.key_values.len(), group.accumulators.len())?,
        size_of::<Value>(),
    )?;
    for key in &group.key_values {
        bytes = add(bytes, value_bytes(key)?)?;
    }
    for accumulator in &group.accumulators {
        bytes = add(bytes, accumulator_finalize_peak(accumulator)?)?;
    }
    Ok(bytes)
}

pub(super) fn accumulator_finalize_peak(
    accumulator: &RdfAccumulator,
) -> Result<usize, OperatorError> {
    match &accumulator.state {
        RdfSetFunctionState::Count(_) => Ok(size_of::<Value>()),
        RdfSetFunctionState::Sum(state) | RdfSetFunctionState::Average(state) => {
            let count = matches!(&accumulator.state, RdfSetFunctionState::Average(_))
                .then(|| u64::try_from(state.inputs.len()).ok())
                .flatten();
            let mut peak = 0;
            for input in &state.inputs {
                peak = peak.max(bound(input.value.finalize_scratch_bytes(count))?);
            }
            // Scale alignment spans at most both extreme input scales; sum
            // carry adds at most usize::BITS decimal digits. This bounds the
            // sequential fold without retaining or allocating a second replay.
            add(mul(peak, 2)?, mul(usize::BITS as usize, 256)?)
        }
        RdfSetFunctionState::GroupConcat {
            values,
            separator,
            error,
        } => {
            if *error {
                return Ok(size_of::<Value>());
            }
            let mut length = mul(values.len().saturating_sub(1), separator.len())?;
            for value in values {
                length = add(length, value.value.len())?;
            }
            // join's reference vector, output String and ArcStr conversion
            // overlap; the result also retains its small value/header storage.
            add(
                add(mul(values.len(), size_of::<&str>())?, mul(length, 3)?)?,
                128,
            )
        }
        RdfSetFunctionState::Minimum { selected, .. }
        | RdfSetFunctionState::Maximum(selected)
        | RdfSetFunctionState::Sample(selected) => selected
            .as_ref()
            .map_or(Ok(size_of::<Value>()), |selected| {
                value_bytes(&selected.value)
            }),
        RdfSetFunctionState::Generic(_) => Err(unsupported_state()),
    }
}

// Initial group allocation only. Subsequent updates use the actual current
// capacities above; whole input parsing envelopes must never become a
// permanently retained per-row multiplier.
pub(super) fn group_growth(
    new_group: bool,
    key_bytes: usize,
    _input_bytes: usize,
    aggregates: &[AggregateExpr],
) -> Result<usize, OperatorError> {
    if !new_group {
        return Ok(0);
    }
    let mut bytes = add(
        512,
        add(
            mul(key_bytes, 3)?,
            mul(aggregates.len(), size_of::<RdfAccumulator>())?,
        )?,
    )?;
    for expression in aggregates {
        if expression.function == super::AggregateFunction::GroupConcat
            && expression.separator.is_none()
        {
            bytes = add(bytes, 1)?;
        }
        if let Some(separator) = &expression.separator {
            bytes = add(bytes, separator.len())?;
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::{accumulator_finalize_peak, group_retained_bytes, group_update_peak};
    use crate::query::planner::rdf::aggregate::{
        RdfAccumulator, RdfGroupState, RdfOrderedString, RdfSetFunctionState,
    };
    use grafeo_common::types::Value;
    use grafeo_core::execution::operators::{AggregateExpr, AggregateFunction};

    #[test]
    fn append_deltas_match_full_storage_across_capacity_and_distinct_boundaries() {
        use super::{group_growth, snapshot_group, updated_group_delta};
        use crate::query::planner::rdf::tagged_rdf_term;
        use grafeo_core::graph::rdf::{Literal, Term};
        let mut expressions = Vec::new();
        for function in [
            AggregateFunction::Count,
            AggregateFunction::Sum,
            AggregateFunction::Avg,
            AggregateFunction::Min,
            AggregateFunction::Max,
            AggregateFunction::Sample,
            AggregateFunction::GroupConcat,
        ] {
            let mut expression = AggregateExpr::sum(0);
            expression.function = function;
            expressions.push(expression.clone());
            expression.distinct = true;
            expressions.push(expression);
        }
        let mut group = RdfGroupState {
            key_values: Vec::new(),
            accumulators: expressions.iter().map(RdfAccumulator::new).collect(),
        };
        assert!(
            group_growth(true, 0, 0, &expressions).unwrap()
                >= group_retained_bytes(&group).unwrap()
        );
        let mut retained = group_retained_bytes(&group).unwrap();
        for ordinal in 0..64 {
            let lexical = (ordinal % 7).to_string();
            let value = tagged_rdf_term(
                Value::RdfLiteral {
                    lexical: lexical.clone().into(),
                    language: None,
                    datatype: Some(Literal::XSD_INTEGER.into()),
                },
                Term::typed_literal(lexical.as_str(), Literal::XSD_INTEGER),
            );
            let snapshot = snapshot_group(&group).unwrap();
            for accumulator in &mut group.accumulators {
                accumulator
                    .update_at(Some(value.clone()), None, None, ordinal)
                    .unwrap();
            }
            let (old, new) = updated_group_delta(&group, snapshot).unwrap();
            retained = retained.checked_sub(old).unwrap().checked_add(new).unwrap();
            assert_eq!(retained, group_retained_bytes(&group).unwrap());
        }
    }

    #[test]
    fn updated_state_adopts_existing_scratch_without_double_admission() {
        use super::{Admission, QueryResourceContext};
        use grafeo_common::memory::buffer::BufferManager;
        let manager = BufferManager::with_budget(51);
        let resources = QueryResourceContext::new(manager.clone()).unwrap();
        let mut admission = Admission::new(&resources).unwrap();
        admission.resize_state(16).unwrap();
        admission.set_scratch(32).unwrap();
        admission.adopt_updated_state(48).unwrap();
        assert_eq!(manager.allocated(), 48);
        assert!(admission.adopt_updated_state(49).is_err());
        assert_eq!(manager.allocated(), 48);
        admission.release_scratch().unwrap();
        drop(admission);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn retained_bound_includes_spare_capacity_and_concat_payloads() {
        let mut values = Vec::with_capacity(128);
        values.push(RdfOrderedString {
            ordinal: 0,
            value: "x".repeat(4096),
        });
        let mut separator = String::with_capacity(1024);
        separator.push(',');
        let accumulator = RdfAccumulator {
            function: AggregateFunction::GroupConcat,
            distinct: None,
            state: RdfSetFunctionState::GroupConcat {
                values,
                separator,
                error: false,
            },
        };
        let mut accumulators = Vec::with_capacity(32);
        accumulators.push(accumulator);
        let group = RdfGroupState {
            key_values: vec![Value::Int64(1)],
            accumulators,
        };
        assert!(
            group_retained_bytes(&group).unwrap()
                >= 128 * std::mem::size_of::<RdfOrderedString>()
                    + 32 * std::mem::size_of::<RdfAccumulator>()
                    + 4096
                    + 1024
        );
        let output = group.accumulators[0].finalize();
        assert!(
            accumulator_finalize_peak(&group.accumulators[0]).unwrap()
                >= output.retained_size_bytes().unwrap()
        );
    }

    #[test]
    fn update_peak_detects_overflow_before_mutation() {
        let group = RdfGroupState {
            key_values: Vec::new(),
            accumulators: vec![RdfAccumulator::new(&AggregateExpr::sum(0))],
        };
        assert!(group_update_peak(&group, usize::MAX).is_err());
        assert!(group_retained_bytes(&group).unwrap() < 4096);
    }
}
